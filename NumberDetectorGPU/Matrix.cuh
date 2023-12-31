#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <math.h>

#include "DeviceFunctions.cuh"
#include "CublasHandler.cuh"

#include <cublas_v2.h>
#include <cuda_fp16.h>

enum class DATA_TYPE
{
	Float,
	Double,
	Int,
	Half,
	Unknown
};

enum class INIT_TYPE
{
	Zero,
	Xavier,
	Xavier_Normal,
	He,
	He_Normal,
	Unknown
};

template <class T>
class Matrix
{
public:
	// Create Matrix with random values
	Matrix(size_t column, size_t row, INIT_TYPE init_type = INIT_TYPE::Zero)
		: m_row(row)
		, m_column(column)
		, m_init_type(init_type)
	{
		InitVariables();
		Init();
	}

	// Initialize transposed weights
	// add other initialization methods
	void Init()
	{
		std::random_device rd;
		std::mt19937 gen(rd());

		m_matrix = new T[m_row * m_column];

		if (m_data_type == DATA_TYPE::Float)
		{
			if (m_init_type != INIT_TYPE::Zero) {
				std::normal_distribution<float> dist;

				switch (m_init_type)
				{
				case INIT_TYPE::Xavier:
				{
					float limit = std::sqrt(6.0f / (m_row + m_column));
					dist = std::normal_distribution<float>(-limit, limit);

					break;
				}
				case INIT_TYPE::Xavier_Normal:
				{
					float standardDeviation = std::sqrt(2.0f / (m_row + m_column));
					dist = std::normal_distribution<float>(0.0f, standardDeviation);

					break;
				}
				case INIT_TYPE::He:
				{
					float limit = std::sqrt(6.0f / m_row);
					dist = std::normal_distribution<float>(-limit, limit);

					break;
				}
				case INIT_TYPE::He_Normal:
				{
					float standardDeviation = std::sqrt(2.0f / m_row);
					dist = std::normal_distribution<float>(0.0f, standardDeviation);

					break;
				}
				default:
					break;
				}

				for (size_t i = 0; i < m_column; i++) {
					for (size_t j = 0; j < m_row; j++) {
						m_matrix[i * m_row + j] = dist(gen) * 0.01;
					}
				}

				cudaMemcpy(d_matrix, m_matrix, d_size, cudaMemcpyHostToDevice);

				//for (size_t i = 0; i < m_column; i++) {
				//	for (size_t j = 0; j < m_row; j++) {
				//		m_matrix[i][j] = i * m_column + j;
				//	}
				//}
			}
		}
		else if (m_data_type == DATA_TYPE::Half)
		{
			if (m_init_type != INIT_TYPE::Zero) {
				std::normal_distribution<float> dist;

				switch (m_init_type)
				{
				case INIT_TYPE::Xavier:
				{
					float limit = std::sqrt(6.0f / (m_row + m_column));
					dist = std::normal_distribution<float>(-limit, limit);

					break;
				}
				case INIT_TYPE::Xavier_Normal:
				{
					float standardDeviation = std::sqrt(2.0f / (m_row + m_column));
					dist = std::normal_distribution<float>(0.0f, standardDeviation);

					break;
				}
				case INIT_TYPE::He:
				{
					float limit = std::sqrt(6.0f / m_row);
					dist = std::normal_distribution<float>(-limit, limit);

					break;
				}
				case INIT_TYPE::He_Normal:
				{
					float standardDeviation = std::sqrt(2.0f / m_row);
					dist = std::normal_distribution<float>(0.0f, standardDeviation);

					break;
				}
				default:
					break;
				}

				for (size_t i = 0; i < m_column; i++) {
					for (size_t j = 0; j < m_row; j++) {
						m_matrix[i * m_row + j] = __float2half_rn(dist(gen) * 0.01);
					}
				}

				cudaMemcpy(d_matrix, m_matrix, d_size, cudaMemcpyHostToDevice);

				//for (size_t i = 0; i < m_column; i++) {
				//	for (size_t j = 0; j < m_row; j++) {
				//		m_matrix[i][j] = i * m_column + j;
				//	}
				//}
			}
		}

		delete[] m_matrix;
	}

	// Create Matrix with set values
	Matrix(std::vector<std::vector<T>> matrix)
	{
		m_row = matrix[0].size();
		m_column = matrix.size();
		InitVariables();

		if (m_data_type == DATA_TYPE::Float)
		{
			float* pointer_matrix = new float[m_row * m_column];

			for (int i = 0; i < m_column; i++)
			{
				for (int j = 0; j < m_row; j++)
				{
					pointer_matrix[i * m_row + j] = matrix[i][j];
				}
			}

			cudaMemcpy(d_matrix, pointer_matrix, d_size, cudaMemcpyHostToDevice);

			delete[] pointer_matrix;
		}
		else if (m_data_type == DATA_TYPE::Half)
		{
			half* pointer_matrix = new half[m_row * m_column];

			for (int i = 0; i < m_column; i++)
			{
				for (int j = 0; j < m_row; j++)
				{
					pointer_matrix[i * m_row + j] = matrix[i][j];
				}
			}

			cudaMemcpy(d_matrix, pointer_matrix, d_size, cudaMemcpyHostToDevice);

			delete[] pointer_matrix;
		}
	}

	Matrix(std::vector<std::vector<float>>* matrix)
	{
		m_row = matrix[0][0].size();
		m_column = matrix[0].size();
		InitVariables();

		if (m_data_type == DATA_TYPE::Float)
		{
			float* pointer_matrix = new float[m_row * m_column];

			for (int i = 0; i < m_column; i++)
			{
				for (int j = 0; j < m_row; j++)
				{
					pointer_matrix[i * m_row + j] = matrix[0][i][j];
				}
			}

			cudaMemcpy(d_matrix, pointer_matrix, d_size, cudaMemcpyHostToDevice);

			delete[] pointer_matrix;
		}
		else if (m_data_type == DATA_TYPE::Half)
		{
			half* pointer_matrix = new half[m_row * m_column];

			for (int i = 0; i < m_column; i++)
			{
				for (int j = 0; j < m_row; j++)
				{
					pointer_matrix[i * m_row + j] = __float2half(matrix[0][i][j]);
				}
			}

			cudaMemcpy(d_matrix, pointer_matrix, d_size, cudaMemcpyHostToDevice);

			delete[] pointer_matrix;
		}
	}

	Matrix(std::vector<T> matrix)
	{
		m_row = matrix.size();
		m_column = 1;
		InitVariables();

		if (m_data_type == DATA_TYPE::Float)
		{
			float* pointer_matrix = new float[m_row];

			for (int i = 0; i < m_row; i++)
			{
				pointer_matrix[i] = matrix[i];
			}

			cudaMemcpy(d_matrix, pointer_matrix, d_size, cudaMemcpyHostToDevice);

			delete[] pointer_matrix;
		}
		else if (m_data_type == DATA_TYPE::Half)
		{
			half* pointer_matrix = new half[m_row];

			for (int i = 0; i < m_row; i++)
			{
				pointer_matrix[i] = matrix[i];
			}

			cudaMemcpy(d_matrix, pointer_matrix, d_size, cudaMemcpyHostToDevice);

			delete[] pointer_matrix;
		}
	}

	Matrix(float matrix)
	{
		m_row = 1;
		m_column = 1;
		InitVariables();

		if (m_data_type == DATA_TYPE::Float)
		{
			float* pointer_matrix = new float[1];

			pointer_matrix[0] = matrix;

			cudaMemcpy(d_matrix, pointer_matrix, d_size, cudaMemcpyHostToDevice);

			delete[] pointer_matrix;
		}
		else if (m_data_type == DATA_TYPE::Half)
		{
			half* pointer_matrix = new half[1];

			pointer_matrix[0] = matrix;

			cudaMemcpy(d_matrix, pointer_matrix, d_size, cudaMemcpyHostToDevice);

			delete[] pointer_matrix;
		}
	}

	Matrix(float* matrix)
	{

	}

	Matrix(Matrix<T>* matrix)
		: m_row(matrix->GetRow())
		, m_column(matrix->GetCol())
	{
		InitVariables();

		//for (size_t i = 0; i < m_column; i++)
		//{
		//	for (size_t j = 0; j < m_row; j++)
		//	{
		//		m_matrix[i][j] = matrix->m_matrix[i][j];
		//	}
		//}

		GPUMatrix << < GRID_SIZE, BLOCK_SIZE >> > (d_matrix, matrix->d_matrix, m_row, m_column);
	}

	Matrix()
	{
		m_cleared = true;
	}

	~Matrix()
	{
		delete h_sum_result;

		cudaFree(d_matrix);
		cudaFree(sum_result);
	}

	static void SetCublasHandler(CuBLASHandler* handler)
	{
		m_handler = handler;
	}

	void InitMatrix(size_t column, size_t row, INIT_TYPE init_type = INIT_TYPE::Zero)
	{
		m_row = row;
		m_column = column;
		m_init_type = init_type;

		InitVariables();
	}

	void InitVariables()
	{
		InitDataType();

		d_matrix = new T[m_row * m_column];

		BLOCK_SIZE_X = 32;
		BLOCK_SIZE.x = BLOCK_SIZE_X;
		BLOCK_SIZE.y = BLOCK_SIZE_X;

		GRID_SIZE_X = (m_row + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X;
		GRID_SIZE_Y = (m_column + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X;
		GRID_SIZE.x = GRID_SIZE_X;
		GRID_SIZE.y = GRID_SIZE_Y;

		d_size = m_row * m_column * sizeof(T);
		cudaMalloc(&d_matrix, d_size);

		h_sum_result = new float;
		cudaMalloc(&sum_result, sizeof(float));

		m_cleared = false;
	}

	void InitDataType()
	{
		const std::type_info& type = typeid(T);

		if (type == typeid(float))
		{
			m_data_type = DATA_TYPE::Float;
		}
		else if (type == typeid(half))
		{
			m_data_type = DATA_TYPE::Half;
		}
		else
		{
			m_data_type = DATA_TYPE::Int;
		}
	}

	void Clear()
	{
		m_row = 0;
		m_column = 0;

		BLOCK_SIZE_X = 0;
		BLOCK_SIZE.x = 0;
		BLOCK_SIZE.y = 0;

		GRID_SIZE_X = 0;
		GRID_SIZE_Y = 0;
		GRID_SIZE.x = 0;
		GRID_SIZE.y = 0;

		d_size = 0;
		cudaFree(d_matrix);

		delete[] h_sum_result;
		cudaFree(sum_result);

		m_cleared = true;
	}

	void SetMatrix(Matrix<T>* matrix)
	{
		cudaMemcpy(d_matrix, matrix->d_matrix, d_size, cudaMemcpyDeviceToDevice);

		//for (size_t i = 0; i < m_column; i++)
		//{
		//	for (size_t j = 0; j < m_row; j++)
		//	{
		//		m_matrix[i][j] = matrix->m_matrix[i][j];
		//	}
		//}
	}

	void SetTransposedMatrix(Matrix<T>* matrix)
	{
		GPUTransposedMatrix << < GRID_SIZE, BLOCK_SIZE >> > (d_matrix, matrix->d_matrix, m_row, m_column);

		//for (size_t i = 0; i < m_column; i++)
		//{
		//	for (size_t j = 0; j < m_row; j++)
		//	{
		//		m_matrix[i][j] = matrix->m_matrix[j][i];
		//	}
		//}
	}

	void SetRowMatrixToRow(Matrix<T>* matrix, size_t row_to_get, size_t row_to_set)
	{
		GPUSetRowMatrixToRow << < (matrix->GetRow() + 1024 - 1) / 1024, 1024 >> > (d_matrix, matrix->d_matrix, m_row, m_column, row_to_get, row_to_set);

		//for (size_t i = 0; i < matrix->GetRow(); i++)
		//{
		//	m_matrix[row_to_set][i] = matrix->m_matrix[row_to_get][i];
		//}
	}

	void SetRowMatrixToColumn(Matrix<T>* matrix, size_t row_to_get, size_t col_to_set)
	{
		GPUSetRowMatrixToColumn << < (matrix->GetRow() + 1024 - 1) / 1024, 1024 >> > (d_matrix, matrix->d_matrix, m_row, m_column, row_to_get, col_to_set);

		//for (size_t i = 0; i < matrix->GetRow(); i++)
		//{
		//	m_matrix[i][col_to_set] = matrix->m_matrix[row_to_get][i];
		//}
	}

	void SetColMatrixToRow(Matrix<T>* matrix, size_t col_to_get, size_t row_to_set)
	{
		GPUSetColMatrixToRow << < (matrix->GetCol() + 1024 - 1) / 1024, 1024 >> > (d_matrix, matrix->d_matrix, m_row, m_column, col_to_get, row_to_set);

		//for (size_t i = 0; i < matrix->GetCol(); i++)
		//{
		//	m_matrix[row_to_set][i] = matrix->m_matrix[i][col_to_get];
		//}
	}

	void Dot(Matrix<T>* a, Matrix<T>* b, std::string op = "N")
	{
		//GPUDot <<< GRID_SIZE, BLOCK_SIZE >>> (d_matrix, a->d_matrix, b->d_matrix, a->GetRow(), a->GetCol(), b->GetRow());

		if (op == "N")
		{
			cublasSgemm(
				m_handler->GetHandle(), CUBLAS_OP_N, CUBLAS_OP_N,
				b->GetRow(), a->GetCol(), b->GetCol(),
				&alpha,
				b->d_matrix, b->GetRow(),
				a->d_matrix, a->GetRow(),
				&beta,
				d_matrix, b->GetRow()
			);
		}
		else if (op == "T")
		{
			cublasSgemm(
				m_handler->GetHandle(), CUBLAS_OP_N, CUBLAS_OP_T,
				b->GetRow(), a->GetRow(), b->GetCol(),
				&alpha,
				b->d_matrix, b->GetRow(),
				a->d_matrix, a->GetRow(),
				&beta,
				d_matrix, b->GetRow()
			);
		}
		else
		{
			cublasSgemm(
				m_handler->GetHandle(), CUBLAS_OP_T, CUBLAS_OP_N,
				b->GetCol(), a->GetCol(), b->GetRow(),
				&alpha,
				b->d_matrix, b->GetRow(),
				a->d_matrix, a->GetRow(),
				&beta,
				d_matrix, b->GetCol()
			);
		}

		//for (std::size_t i = 0; i < a->m_matrix.size(); ++i)
		//{
		//	for (std::size_t j = 0; j < b->m_matrix[0].size(); ++j)
		//	{
		//		m_matrix[i][j] = 0;
		//		for (std::size_t k = 0; k < a->m_matrix[0].size(); ++k)
		//		{
		//			m_matrix[i][j] += a->m_matrix[i][k] * b->m_matrix[k][j];
		//		}
		//	}
		//}
	}

	void Max(float min, Matrix<T>* matrix)
	{
		GPUMax << < GRID_SIZE, BLOCK_SIZE >> > (d_matrix, matrix->d_matrix, min, m_row, m_column);

		//for (size_t i = 0; i < m_column; i++)
		//{
		//	for (size_t j = 0; j < m_row; j++)
		//	{
		//		m_matrix[i][j] = std::max(min, matrix->m_matrix[i][j]);
		//	}
		//}
	}

	void RowMax(Matrix<T>* matrix)
	{
		GPURowMax << < (m_column + 1024 - 1) / 1024, 1024 >> > (d_matrix, matrix->d_matrix, matrix->m_row, matrix->m_column);

		//float max_num;
		//for (size_t i = 0; i < m_column; i++)
		//{
		//	max_num = matrix->m_matrix[i][0];
		//
		//	for (size_t j = 1; j < matrix->m_row; j++)
		//	{
		//		max_num = std::max(max_num, matrix->m_matrix[i][j]);
		//	}
		//
		//	m_matrix[i][0] = max_num;
		//}
	}

	void RowArgmax(Matrix<T>* matrix)
	{
		GPURowArgmax << < (m_column + 1024 - 1) / 1024, 1024 >> > (d_matrix, matrix->d_matrix, matrix->m_row, matrix->m_column);

		//float max_num;
		//for (size_t i = 0; i < matrix->m_column; i++)
		//{
		//	max_num = matrix->m_matrix[i][0];
		//
		//	for (size_t j = 1; j < matrix->m_row; j++)
		//	{
		//		max_num = std::max(max_num, matrix->m_matrix[i][j]);
		//
		//	}
		//
		//	// it is a pointer where searcheable value is located
		//	std::vector<float>::iterator it = std::find(matrix->m_matrix[i].begin(), matrix->m_matrix[i].end(), max_num);
		//	// substract pointer to searcheable value from the beggining of vector to get index of searcheable value
		//	m_matrix[i][0] = it - matrix->m_matrix[i].begin();
		//}
	}

	float Sum(Matrix<T>* matrix)
	{
		GPUMatrixSum << < 1, 1 >> > (matrix->d_matrix, sum_result, matrix->m_row, matrix->m_column);
		cudaMemcpy(h_sum_result, sum_result, sizeof(float), cudaMemcpyDeviceToHost);

		//float result = 0.0f;
		//
		//for (size_t i = 0; i < m_column; i++)
		//{
		//	for (size_t j = 0; j < m_row; j++)
		//	{
		//		result += matrix->m_matrix[i][j];
		//	}
		//}		

		return h_sum_result[0];
	}

	void RowSum(Matrix<T>* matrix)
	{
		GPURowSum << < (matrix->GetCol() + 1024 - 1) / 1024, 1024 >> > (d_matrix, matrix->d_matrix, matrix->m_row, matrix->m_column);

		//for (size_t i = 0; i < matrix->m_column; i++)
		//{
		//	m_matrix[i][0] = 0;
		//	for (size_t j = 0; j < matrix->m_row; j++)
		//	{
		//		m_matrix[i][0] += matrix->m_matrix[i][j];
		//	}
		//}
	}

	void ColSum(Matrix<T>* matrix)
	{
		GPUColSum << < (matrix->GetRow() + 1024 - 1) / 1024, 1024 >> > (d_matrix, matrix->d_matrix, matrix->m_row, matrix->m_column);

		//for (size_t i = 0; i < matrix->m_row; i++)
		//{
		//	m_matrix[0][i] = 0;
		//	for (size_t j = 0; j < matrix->m_column; j++)
		//	{
		//		m_matrix[0][i] += matrix->m_matrix[j][i];
		//	}
		//}
	}

	void AddMatrix(Matrix<T>* matrix)
	{
		GPUAddMatrix << < GRID_SIZE, BLOCK_SIZE >> > (d_matrix, matrix->d_matrix, m_row, m_column);

		//for (size_t i = 0; i < m_column; i++)
		//{
		//	for (size_t j = 0; j < m_row; j++)
		//	{
		//		m_matrix[i][j] = m_matrix[i][j] + matrix->m_matrix[i][j];
		//	}
		//}
	}

	void AddMatricies(Matrix<T>* matrix1, Matrix<T>* matrix2)
	{
		GPUAddMatricies << < GRID_SIZE, BLOCK_SIZE >> > (d_matrix, matrix1->d_matrix, matrix2->d_matrix, m_row, m_column);

		//for (size_t i = 0; i < m_column; i++)
		//{
		//	for (size_t j = 0; j < m_row; j++)
		//	{
		//		m_matrix[i][j] = matrix1->m_matrix[i][j] + matrix2->m_matrix[i][j];
		//	}
		//}
	}

	void AddValue(Matrix<T>* matrix, float value)
	{
		GPUAddValue << < GRID_SIZE, BLOCK_SIZE >> > (d_matrix, matrix->d_matrix, value, m_row, m_column);

		//for (size_t i = 0; i < m_column; i++)
		//{
		//	for (size_t j = 0; j < m_row; j++)
		//	{
		//		m_matrix[i][j] = matrix->m_matrix[i][j] + value;
		//	}
		//}
	}

	void AddSingleRow(Matrix<T>* matrix)
	{
		GPUAddSingleRow << < GRID_SIZE, BLOCK_SIZE >> > (d_matrix, matrix->d_matrix, m_row, m_column);

		//for (size_t i = 0; i < m_column; i++)
		//{
		//	for (size_t j = 0; j < m_row; j++)
		//	{
		//		m_matrix[i][j] = m_matrix[i][j] + matrix->m_matrix[0][j];
		//	}
		//}
	}

	void SubstractMatricies(Matrix<T>* minuend_matrix, Matrix<T>* subtrachend_matrix)
	{
		GPUSubstractMatricies << < GRID_SIZE, BLOCK_SIZE >> > (d_matrix, minuend_matrix->d_matrix, subtrachend_matrix->d_matrix, m_row, m_column);

		//for (size_t i = 0; i < m_column; i++)
		//{
		//	for (size_t j = 0; j < m_row; j++)
		//	{
		//		m_matrix[i][j] = minuend_matrix->m_matrix[i][j] - subtrachend_matrix->m_matrix[i][j];
		//	}
		//}
	}

	// Decreases all values in the first matrix' row by value in the second matrix
	void SubstractMatrixFromRowValues(Matrix<T>* minuend_matrix, Matrix<T>* subtrachend_matrix)
	{
		GPUSubstractMatrixFromRowValues << < GRID_SIZE, BLOCK_SIZE >> > (d_matrix, minuend_matrix->d_matrix, subtrachend_matrix->d_matrix, m_row, m_column);

		//for (size_t i = 0; i < m_column; i++)
		//{
		//	for (size_t j = 0; j < m_row; j++)
		//	{
		//		m_matrix[i][j] = minuend_matrix->m_matrix[i][j] - subtrachend_matrix->m_matrix[i][0];
		//	}
		//}
	}

	void SubstractValueFromMatrix(Matrix<T>* matrix, float value)
	{
		GPUSubstractValueFromMatrix << < GRID_SIZE, BLOCK_SIZE >> > (d_matrix, matrix->d_matrix, value, m_row, m_column);

		//for (size_t i = 0; i < m_column; i++)
		//{
		//	for (size_t j = 0; j < m_row; j++)
		//	{
		//		m_matrix[i][j] = value - matrix->m_matrix[i][j];
		//	}
		//}
	}

	// Substracts value from rows of first matrix at index set by second matrix
	void SubstractMatrixFromValueAtMatrixIdx(Matrix<T>* matrix, float value)
	{
		GPUSubstractMatrixFromValueAtMatrixIdx << < (m_column + 1024 - 1) / 1024, 1024 >> > (d_matrix, matrix->d_matrix, value, m_row, m_column);

		//for (size_t i = 0; i < m_column; i++)
		//{
		//	m_matrix[i][idx_matrix->m_matrix[0][i]] = m_matrix[i][idx_matrix->m_matrix[0][i]] - value;
		//}
	}

	void MatriciesMult(Matrix<T>* multiplier_matrix, Matrix<T>* multiplicand_matrix)
	{
		GPUMatriciesMult << < GRID_SIZE, BLOCK_SIZE >> > (d_matrix, multiplier_matrix->d_matrix, multiplicand_matrix->d_matrix, m_row, m_column);

		//for (size_t i = 0; i < m_column; i++)
		//{
		//	for (size_t j = 0; j < m_row; j++)
		//	{
		//		m_matrix[i][j] = multiplier_matrix->m_matrix[i][j] * multiplier_matrix->m_matrix[i][j];
		//	}
		//}
	}

	void MultByValue(Matrix<T>* matrix, float value)
	{
		GPUMultByValue << < GRID_SIZE, BLOCK_SIZE >> > (d_matrix, matrix->d_matrix, value, m_row, m_column);

		//for (size_t i = 0; i < m_column; i++)
		//{
		//	for (size_t j = 0; j < m_row; j++)
		//	{
		//		m_matrix[i][j] = matrix->m_matrix[i][j] * value;
		//	}
		//}
	}

	void DivideMatrixByValue(Matrix<T>* matrix, float value)
	{
		GPUDivideMatrixByValue << < GRID_SIZE, BLOCK_SIZE >> > (d_matrix, matrix->d_matrix, value, m_row, m_column);

		//for (size_t i = 0; i < m_column; i++)
		//{
		//	for (size_t j = 0; j < m_row; j++)
		//	{
		//		m_matrix[i][j] = dividend_matrix->m_matrix[i][j] / value;
		//	}
		//}
	}

	// Divide all values in the first matrix' row by value in the second matrix
	void DivideMatrixByRow(Matrix<T>* dividend_matrix, Matrix<T>* divisor_matrix)
	{
		GPUDivideMatrixByRow << < GRID_SIZE, BLOCK_SIZE >> > (d_matrix, dividend_matrix->d_matrix, divisor_matrix->d_matrix, m_row, m_column);

		//for (size_t i = 0; i < m_column; i++)
		//{
		//	for (size_t j = 0; j < m_row; j++)
		//	{
		//		m_matrix[i][j] = dividend_matrix->m_matrix[i][j] / divisor_matrix->m_matrix[i][0];
		//	}
		//}
	}

	void DivideMatrices(Matrix<T>* dividend_matrix, Matrix<T>* divisor_matrix)
	{
		GPUDivideMatrices << < GRID_SIZE, BLOCK_SIZE >> > (d_matrix, dividend_matrix->d_matrix, divisor_matrix->d_matrix, m_row, m_column);

		//for (size_t i = 0; i < m_column; i++)
		//{
		//	for (size_t j = 0; j < m_row; j++)
		//	{
		//		m_matrix[i][j] = dividend_matrix->m_matrix[i][j] / divisor_matrix->m_matrix[i][j];
		//	}
		//}
	}

	void SqrtMatrix(Matrix<T>* matrix)
	{
		GPUSqrtMatrix << < GRID_SIZE, BLOCK_SIZE >> > (d_matrix, matrix->d_matrix, m_row, m_column);

		//for (size_t i = 0; i < m_column; i++)
		//{
		//	for (size_t j = 0; j < m_row; j++)
		//	{
		//		m_matrix[i][j] = std::sqrt(matrix->m_matrix[i][j]);
		//	}
		//}
	}

	void PowerMatrix(Matrix<T>* matrix, float exponent)
	{
		GPUPowerMatrix << < GRID_SIZE, BLOCK_SIZE >> > (d_matrix, matrix->d_matrix, exponent, m_row, m_column);

		//for (size_t i = 0; i < m_column; i++)
		//{
		//	for (size_t j = 0; j < m_row; j++)
		//	{
		//		m_matrix[i][j] = std::pow(matrix->m_matrix[i][j], exponent);
		//	}
		//}
	}

	void SetZeroIfMatrixValueIsNegative(Matrix<T>* matrix)
	{
		GPUSetZeroIfMatrixValueIsNegative << < GRID_SIZE, BLOCK_SIZE >> > (d_matrix, matrix->d_matrix, m_row, m_column);

		//for (size_t i = 0; i < m_column; i++)
		//{
		//	for (size_t j = 0; j < m_row; j++)
		//	{
		//		if (matrix->m_matrix[i][j] < 0)
		//		{
		//			m_matrix[i][j] = 0;
		//		}
		//	}
		//}
	}

	void Exp(Matrix<T>* matrix)
	{
		GPUExp << < GRID_SIZE, BLOCK_SIZE >> > (d_matrix, matrix->d_matrix, m_row, m_column);

		//for (size_t i = 0; i < m_column; i++)
		//{
		//	for (size_t j = 0; j < m_row; j++)
		//	{
		//		m_matrix[i][j] = std::pow(2.71828182845904523536, matrix->m_matrix[i][j]);
		//	}
		//}
	}

	void Clip(Matrix<T>* matrix, float min, float max)
	{
		GPUClip << < GRID_SIZE, BLOCK_SIZE >> > (d_matrix, matrix->d_matrix, min, max, m_row, m_column);

		//for (size_t i = 0; i < m_column; i++)
		//{
		//	for (size_t j = 0; j < m_row; j++)
		//	{
		//		m_matrix[i][j] = std::max(min, std::min(max, matrix->m_matrix[i][j]));
		//	}
		//}
	}

	void GetValuesAccordingToMatrices(Matrix<T>* values_matrix, Matrix<T>* idxs_matrix)
	{
		GPUGetValuesAccordingToMatrices << < (m_row + 1024 - 1) / 1024, 1024 >> > (d_matrix, values_matrix->d_matrix, idxs_matrix->d_matrix, m_row, values_matrix->GetRow());

		//if (idxs_matrix->GetCol() == 1)
		//{
		//	for (size_t i = 0; i < m_row; i++)
		//	{
		//		m_matrix[0][i] = values_matrix->m_matrix[i][idxs_matrix->m_matrix[0][i]];
		//	}
		//}
	}

	void NegativeLog(Matrix<T>* matrix)
	{
		GPUNegativeLog << < GRID_SIZE, BLOCK_SIZE >> > (d_matrix, matrix->d_matrix, m_row, m_column);

		//for (size_t i = 0; i < m_column; i++)
		//{
		//	for (size_t j = 0; j < m_row; j++)
		//	{
		//		m_matrix[i][j] = -std::log(matrix->m_matrix[i][j]);
		//	}
		//}
	}

	void Abs(Matrix<T>* matrix)
	{
		GPUAbs << < GRID_SIZE, BLOCK_SIZE >> > (d_matrix, matrix->d_matrix, m_row, m_column);

		//for (size_t i = 0; i < m_column; i++)
		//{
		//	for (size_t j = 0; j < m_row; j++)
		//	{
		//		m_matrix[i][j] = std::abs(matrix->m_matrix[i][j]);
		//	}
		//}
	}

	// Returns array filled with 1 if values of matrices are same and 0 if not
	void CompareMatrixAndVector(Matrix<T>* matrix, Matrix<T>* compare_matrix)
	{
		GPUCompareMatrixAndVector << < (m_column + 1024 - 1) / 1024, 1024 >> > (d_matrix, matrix->d_matrix, compare_matrix->d_matrix, m_row, m_column);

		//for (size_t i = 0; i < m_column; i++)
		//{
		//	if (matrix->m_matrix[i][0] == vector->m_matrix[0][i])
		//	{
		//		m_matrix[i][0] = 1;
		//	}
		//	else
		//	{
		//		m_matrix[i][0] = 0;
		//	}
		//}
	}

	void OneHotEncode(Matrix<T>* matrix)
	{
		GPUOneHotEncode << < (m_column + 1024 - 1) / 1024, 1024 >> > (d_matrix, matrix->d_matrix, m_row, m_column);

		//for (size_t i = 0; i < m_column; i++)
		//{
		//	m_matrix[i][matrix->m_matrix[0][i]] = 1;
		//}
	}

	// Gets vector of size n
	// Creates matrix of size n * n with input vector variables on the diagonal
	void EyeVector(Matrix<T>* matrix)
	{
		GPUEyeVector << < (matrix->GetRow() + 1024 - 1) / 1024, 1024 >> > (d_matrix, matrix->d_matrix, matrix->GetRow(), m_column);

		//for (size_t i = 0; i < matrix->GetRow(); i++)
		//{
		//	m_matrix[i][i] = matrix->m_matrix[0][i];
		//}
	}

	float Mean()
	{
		GPUMean << < 1, 1 >> > (d_matrix, sum_result, m_row);
		cudaMemcpy(h_sum_result, sum_result, sizeof(T), cudaMemcpyDeviceToHost);

		//float sum = 0;
		//for (size_t i = 0; i < m_row; i++)
		//{
		//	sum += m_matrix[0][i];
		//}

		//return *sum_result
		return h_sum_result[0];
	}

	float ColMean()
	{
		GPUColMean << < 1, 1 >> > (d_matrix, sum_result, m_column);
		cudaMemcpy(h_sum_result, sum_result, sizeof(T), cudaMemcpyDeviceToHost);

		//float sum = 0;
		//for (size_t i = 0; i < m_column; i++)
		//{
		//	sum += m_matrix[i][0];
		//}

		//return *sum_result;
		return h_sum_result[0];
	}

	size_t GetRow()
	{
		return m_row;
	}

	size_t GetCol()
	{
		return m_column;
	}

	bool Cleared()
	{
		return m_cleared;
	}

	T* GetMatrix()
	{
		return d_matrix;
	}

	std::vector<std::vector<float>> GetVectorMatrix()
	{
		if (m_data_type == DATA_TYPE::Float)
		{
			float* float_m_matrix = new float[m_row * m_column];
			cudaMemcpy(float_m_matrix, d_matrix, m_row * m_column * sizeof(float), cudaMemcpyDeviceToHost);

			std::vector<std::vector<float>> vector_m_matrix;

			for (int i = 0; i < m_column; i++)
			{
				std::vector<float> tmp_vector;
				for (int j = 0; j < m_row; j++)
				{
					tmp_vector.push_back(float_m_matrix[i * m_row + j]);
				}
				vector_m_matrix.push_back(tmp_vector);
			}

			delete[] float_m_matrix;

			return vector_m_matrix;
		}
		else
		{
			//float* tmp_float_d_matrix;
			//cudaMalloc(&tmp_float_d_matrix, m_row * m_column * sizeof(float));
			//
			//halftofloat << < GRID_SIZE, BLOCK_SIZE >> > (tmp_float_d_matrix, d_matrix, m_row, m_column);
			//
			//float* float_m_matrix = new float[m_row * m_column];
			//cudaMemcpy(float_m_matrix, tmp_float_d_matrix, m_row * m_column * sizeof(float), cudaMemcpyDeviceToHost);
			//
			//std::vector<std::vector<float>> vector_m_matrix;
			//
			//for (int i = 0; i < m_column; i++)
			//{
			//	std::vector<float> tmp_vector;
			//	for (int j = 0; j < m_row; j++)
			//	{
			//		tmp_vector.push_back(float_m_matrix[i * m_row + j]);
			//	}
			//	vector_m_matrix.push_back(tmp_vector);
			//}
			//
			//delete[] float_m_matrix;
			//cudaFree(tmp_float_d_matrix);
			//
			//return vector_m_matrix;
		}
	}

private:
	size_t m_row;
	size_t m_column;

	T* d_matrix;
	T* m_matrix;

	int BLOCK_SIZE_X;
	int GRID_SIZE_X;
	int GRID_SIZE_Y;

	dim3 BLOCK_SIZE;
	dim3 GRID_SIZE;

	size_t d_size;

	// Use for SUM
	float* sum_result;
	float* h_sum_result;

	INIT_TYPE m_init_type;
	DATA_TYPE m_data_type;

	bool m_cleared;

	// Cublas
	static CuBLASHandler* m_handler;

	const T alpha = 1.0f;
	const T beta = 0.0f;
};