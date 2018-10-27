#include <stdio.h>
#include <ctime>
#include <cilk/cilk.h>
#include <cilk/reducer_opadd.h>
#include <chrono>

using namespace std::chrono;

// количество строк в исходной квадратной матрице
const int MATRIX_SIZE = 1500;

double _time_serial = 0.0;
double _time_parallel = 0.0;

/// Функция InitMatrix() заполняет переданную в качестве 
/// параметра квадратную матрицу случайными значениями
/// matrix - исходная матрица СЛАУ
void InitMatrix(double** matrix)
{
	for (int i = 0; i < MATRIX_SIZE; ++i)
	{
		matrix[i] = new double[MATRIX_SIZE + 1];
	}

	for (int i = 0; i < MATRIX_SIZE; ++i)
	{
		for (int j = 0; j <= MATRIX_SIZE; ++j)
		{
			matrix[i][j] = rand() % 2500 + 1;
		}
	}
}

/// Функция SerialGaussMethod() решает СЛАУ методом Гаусса 
/// matrix - исходная матрица коэффиициентов уравнений, входящих в СЛАУ,
/// последний столбец матрицы - значения правых частей уравнений
/// rows - количество строк в исходной матрице
/// result - массив ответов СЛАУ
void SerialGaussMethod(double **matrix, const int rows, double* result)
{
	int k;
	double koef;

	high_resolution_clock::time_point t1 = high_resolution_clock::now();
	for (k = 0; k < rows; ++k)
	{
		//
		for (int i = k + 1; i < rows; ++i)
		{
			koef = -matrix[i][k] / matrix[k][k];

			for (int j = k; j <= rows; ++j)
			{
				matrix[i][j] += koef * matrix[k][j];
			}
		}
	}

	high_resolution_clock::time_point t2 = high_resolution_clock::now();
	duration<double> duration = (t2 - t1);
	_time_serial = duration.count();
	printf("Calculation time of the direct method pass is %f sec.\n", duration.count());

	// обратный ход метода Гаусса
	result[rows - 1] = matrix[rows - 1][rows] / matrix[rows - 1][rows - 1];

	for (k = rows - 2; k >= 0; --k)
	{
		result[k] = matrix[k][rows];

		//
		for (int j = k + 1; j < rows; ++j)
		{
			result[k] -= matrix[k][j] * result[j];
		}

		result[k] /= matrix[k][k];
	}
}

void ParallelSerialGaussMethod(double **matrix, const int rows, double* result)
{
	int k;
	// double koef;

	// прямой ход метода Гаусса
	//измерение времени прямого хода
	high_resolution_clock::time_point t1 = high_resolution_clock::now();
	for (k = 0; k < rows; ++k) {
		//использование cilk_for во внутреннем цикле прямого хода
		cilk_for(int i = k + 1; i < rows; ++i) {
			double koef = -matrix[i][k] / matrix[k][k];
			for (int j = k; j <= rows; ++j) {
				matrix[i][j] += koef * matrix[k][j];
			}
		}
	}

	high_resolution_clock::time_point t2 = high_resolution_clock::now();
	duration<double> duration = (t2 - t1);
	_time_parallel = duration.count();
	printf("Calculation time of the direct method pass use cilk_for is %f sec.\n", duration.count());

	// обратный ход метода Гаусса
	result[rows - 1] = matrix[rows - 1][rows] / matrix[rows - 1][rows - 1];

	for (k = rows - 2; k >= 0; --k)
	{
		// result[k] = matrix[k][rows];
		cilk::reducer_opadd <double> temp(matrix[k][rows]);
		//использование cilk_for во внутреннем цикле обратного хода
		cilk_for(int j = k + 1; j < rows; ++j) {
			temp -= matrix[k][j] * result[j];
		}
		result[k] = temp.get_value();
		result[k] /= matrix[k][k];
	}
}

int main()
{
	srand((unsigned)time(0));

	int i;

	// кол-во строк в матрице, приводимой в качестве примера
	const int test_matrix_lines = 4;

	double **test_matrix = new double*[test_matrix_lines];

	// цикл по строкам
	for (i = 0; i < test_matrix_lines; ++i)
	{
		// (test_matrix_lines + 1)- количество столбцов в тестовой матрице,
		// последний столбец матрицы отведен под правые части уравнений, входящих в СЛАУ
		test_matrix[i] = new double[test_matrix_lines + 1];
	}

	// массив решений СЛАУ
	double *result = new double[test_matrix_lines];

	// инициализация тестовой матрицы
	test_matrix[0][0] = 2; test_matrix[0][1] = 5;  test_matrix[0][2] = 4;  test_matrix[0][3] = 1;  test_matrix[0][4] = 20;
	test_matrix[1][0] = 1; test_matrix[1][1] = 3;  test_matrix[1][2] = 2;  test_matrix[1][3] = 1;  test_matrix[1][4] = 11;
	test_matrix[2][0] = 2; test_matrix[2][1] = 10; test_matrix[2][2] = 9;  test_matrix[2][3] = 7;  test_matrix[2][4] = 40;
	test_matrix[3][0] = 3; test_matrix[3][1] = 8;  test_matrix[3][2] = 9;  test_matrix[3][3] = 2;  test_matrix[3][4] = 37;

	printf("System of linear algebraic equations:\n");
	for (i = 0; i < test_matrix_lines; i++) {
		printf("\t");
		for (int j = 0; j < test_matrix_lines + 1; j++) {
			if (j == test_matrix_lines)
				printf("= %4.1f", test_matrix[i][4]);
			else
				printf("%4.1f ", test_matrix[i][j]);
		}
		printf("\n");
	}

	SerialGaussMethod(test_matrix, test_matrix_lines, result);

	printf("Solution:\n");
	for (i = 0; i < test_matrix_lines; ++i)
		printf("\t x(%d) = %4.1f\n", i, result[i]);
		
	for (i = 0; i < test_matrix_lines; ++i)
		delete[]test_matrix[i];
	delete[] result;
	

	double **matrix = new double*[MATRIX_SIZE];

	for (i = 0; i < MATRIX_SIZE; ++i) {
		// (test_matrix_lines + 1)- количество столбцов в тестовой матрице,
		// последний столбец матрицы отведен под правые части уравнений, входящих в СЛАУ
		matrix[i] = new double[MATRIX_SIZE + 1];
	}

	// массив решений СЛАУ
	double *matrix_result = new double[MATRIX_SIZE];
	SerialGaussMethod(matrix, MATRIX_SIZE, matrix_result);
	ParallelSerialGaussMethod(matrix, MATRIX_SIZE, matrix_result);

	printf("Grown up is: %f\n", _time_serial / _time_parallel);

	for (i = 0; i < MATRIX_SIZE; ++i)
		delete[]matrix[i];
	delete[] matrix_result;

	return 0;
}