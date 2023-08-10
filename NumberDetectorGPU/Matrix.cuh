#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <math.h>

#include "DeviceFunctions.cuh"

enum class DATA_TYPE
{
	Float,
	Double,
	Int,
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
	{
		InitVariables();

		const std::type_info& type = typeid(T);

		if (type == typeid(float))
		{
			m_data_type = DATA_TYPE::Float;
		}
		else
		{
			m_data_type = DATA_TYPE::Int;
		}

		Init(init_type);
	}

	// Initialize transposed weights
	// add other initialization methods
	void Init(INIT_TYPE init_type = INIT_TYPE::Zero)
	{
		std::random_device rd;
		std::mt19937 gen(rd());

		if (m_data_type == DATA_TYPE::Float)
		{
			if (init_type != INIT_TYPE::Zero) {
				std::normal_distribution<float> dist;

				switch (init_type)
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
		else if (m_data_type == DATA_TYPE::Double)
		{
		}
	}

	// Create Matrix with set values
	Matrix(std::vector<std::vector<float>> matrix)
		: m_row(matrix[0].size())
		, m_column(matrix.size())
	{
		InitVariables();

		float* pointer_matrix = new float[m_row * m_column];

		for (int i = 0; i < m_column; i++)
		{
			for (int j = 0; j < m_row; j++)
			{
				pointer_matrix[i * m_row + j] = matrix[i][j];
			}
		}

		cudaMemcpy(d_matrix, pointer_matrix, d_size, cudaMemcpyHostToDevice);
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

		GPUMatrix <<< GRID_SIZE, BLOCK_SIZE >>> (d_matrix, matrix->d_matrix, m_row, m_column);
	}

	~Matrix()
	{

	}

	void InitVariables()
	{
		m_matrix = new float[m_row * m_column];
		d_matrix = new float[m_row * m_column];
		d_size = m_row * m_column * sizeof(float);
		BLOCK_SIZE_X = 32;
		BLOCK_SIZE.x = BLOCK_SIZE_X;
		BLOCK_SIZE.y = BLOCK_SIZE_X;
		GRID_SIZE_X = (m_row + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X;
		GRID_SIZE_Y = (m_column + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X;
		GRID_SIZE.x = GRID_SIZE_X;
		GRID_SIZE.y = GRID_SIZE_Y;

		cudaMalloc(&d_matrix, d_size);
		h_sum_result = new float;
		cudaMalloc(&sum_result, sizeof(float));
	}

	void SetMatrix(Matrix<T>* matrix)
	{
		cudaMemcpy(d_matrix, matrix->d_matrix, m_row * m_column * sizeof(float), cudaMemcpyDeviceToDevice);
		//GPUMatrix <<< GRID_SIZE, BLOCK_SIZE >>> (d_matrix, matrix->d_matrix, m_row, m_column);

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
		//float* tmp = new float[m_row * m_column];
		//cudaMemcpy(tmp, matrix->d_matrix, d_size, cudaMemcpyDeviceToHost);
		GPUTransposedMatrix <<< GRID_SIZE, BLOCK_SIZE >>> (d_matrix, matrix->d_matrix, m_row, m_column);
		//cudaMemcpy(tmp, matrix->d_matrix, d_size, cudaMemcpyDeviceToHost);

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
		GPUSetRowMatrixToRow <<< GRID_SIZE, BLOCK_SIZE >>> (d_matrix, matrix->d_matrix, m_row, m_column, row_to_get, row_to_set);

		//for (size_t i = 0; i < matrix->GetRow(); i++)
		//{
		//	m_matrix[row_to_set][i] = matrix->m_matrix[row_to_get][i];
		//}
	}

	void SetRowMatrixToColumn(Matrix<T>* matrix, size_t row_to_get, size_t col_to_set)
	{
		GPUSetRowMatrixToColumn <<< GRID_SIZE, BLOCK_SIZE >>> (d_matrix, matrix->d_matrix, m_row, m_column, row_to_get, col_to_set);

		//for (size_t i = 0; i < matrix->GetRow(); i++)
		//{
		//	m_matrix[i][col_to_set] = matrix->m_matrix[row_to_get][i];
		//}
	}

	void SetColMatrixToRow(Matrix<T>* matrix, size_t col_to_get, size_t row_to_set)
	{
		GPUSetColMatrixToRow <<< GRID_SIZE, BLOCK_SIZE >>> (d_matrix, matrix->d_matrix, m_row, m_column, col_to_get, row_to_set);

		//for (size_t i = 0; i < matrix->GetCol(); i++)
		//{
		//	m_matrix[row_to_set][i] = matrix->m_matrix[i][col_to_get];
		//}
	}

	void Dot(Matrix<T>* a, Matrix<T>* b)
	{
		GPUDot <<< GRID_SIZE, BLOCK_SIZE >>> (d_matrix, a->d_matrix, b->d_matrix, a->GetRow(), a->GetCol(), b->GetRow());

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
		GPUMax <<< GRID_SIZE, BLOCK_SIZE >>> (d_matrix, matrix->d_matrix, min, m_row, m_column);

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
		//float* result = new float[12000];
		//cudaMemcpy(result, d_matrix, 12000, cudaMemcpyDeviceToHost);
		//
		//float* result1 = new float[36000];
		//cudaMemcpy(result1, matrix->d_matrix, 36000, cudaMemcpyDeviceToHost);

		GPURowMax <<< GRID_SIZE, BLOCK_SIZE >>> (d_matrix, matrix->d_matrix, matrix->m_row, matrix->m_column);

		
		//cudaMemcpy(result, d_matrix, 12000, cudaMemcpyDeviceToHost);
		//
		//cudaError_t cudaStatus;
		//cudaStatus = cudaGetLastError();
		//if (cudaStatus != cudaSuccess) {
		//	fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		//}

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
		GPURowArgmax <<< GRID_SIZE, BLOCK_SIZE >>> (d_matrix, matrix->d_matrix, matrix->m_row, matrix->m_column);

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
		//float* result = new float[512];
		//
		//cudaMemcpy(result, matrix->d_matrix, 512, cudaMemcpyDeviceToHost);

		GPUMatrixSum <<< 1, 1 >>> (matrix->d_matrix, sum_result, matrix->m_row, matrix->m_column);

		//float* result1 = new float[512];
		//
		//cudaMemcpy(result1, matrix->d_matrix, 512, cudaMemcpyDeviceToHost);

		cudaMemcpy(h_sum_result, sum_result, 4, cudaMemcpyDeviceToHost);

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
		GPURowSum << < GRID_SIZE, BLOCK_SIZE >> > (d_matrix, matrix->d_matrix, matrix->m_row, matrix->m_column);

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
		GPUColSum << < GRID_SIZE, BLOCK_SIZE >> > (d_matrix, matrix->d_matrix, matrix->m_row, matrix->m_column);

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
		//float* result4 = new float[256];
		//
		//cudaMemcpy(result4, minuend_matrix->d_matrix, 256, cudaMemcpyDeviceToHost);
		//
		//float* result5 = new float[256];
		//
		//cudaMemcpy(result5, subtrachend_matrix->d_matrix, 256, cudaMemcpyDeviceToHost);

		GPUSubstractMatricies << < GRID_SIZE, BLOCK_SIZE >> > (d_matrix, minuend_matrix->d_matrix, subtrachend_matrix->d_matrix, m_row, m_column);

		//float* result6 = new float[256];
		//
		//cudaMemcpy(result6, d_matrix, 256, cudaMemcpyDeviceToHost);

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
	void SubstractMatrixFromValueAtMatrixIdx(Matrix<T>* matrix, Matrix<T>* idx_matrix, float value)
	{
		GPUSubstractMatrixFromValueAtMatrixIdx << < GRID_SIZE, BLOCK_SIZE >> > (d_matrix, idx_matrix->d_matrix, value, m_row, m_column);

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

	void DivideMatrixByValue(Matrix<T>* dividend_matrix, float value)
	{
		GPUDivideMatrixByValue << < GRID_SIZE, BLOCK_SIZE >> > (d_matrix, dividend_matrix->d_matrix, value, m_row, m_column);

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
		GPUGetValuesAccordingToMatrices << < GRID_SIZE, BLOCK_SIZE >> > (d_matrix, values_matrix->d_matrix, idxs_matrix->d_matrix, m_row, values_matrix->GetRow());

		//cudaError_t cudaStatus;
		//cudaStatus = cudaGetLastError();
		//if (cudaStatus != cudaSuccess) {
		//	fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		//}
		//
		//float* result = new float[36000];
		//
		//cudaMemcpy(result, values_matrix->d_matrix, 36000, cudaMemcpyDeviceToHost);
		//
		//float* result1 = new float[12000];
		//
		//cudaMemcpy(result1, idxs_matrix->d_matrix, 12000, cudaMemcpyDeviceToHost);
		//
		//float* result2 = new float[12000];
		//
		//cudaMemcpy(result2, d_matrix, 12000, cudaMemcpyDeviceToHost);

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
		GPUCompareMatrixAndVector << < GRID_SIZE, BLOCK_SIZE >> > (d_matrix, matrix->d_matrix, compare_matrix->d_matrix, m_row, m_column);

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
		GPUOneHotEncode << < GRID_SIZE, BLOCK_SIZE >> > (d_matrix, matrix->d_matrix, m_row, m_column);

		//for (size_t i = 0; i < m_column; i++)
		//{
		//	m_matrix[i][matrix->m_matrix[0][i]] = 1;
		//}
	}

	// Gets vector of size n
	// Creates matrix of size n * n with input vector variables on the diagonal
	void EyeVector(Matrix<T>* matrix)
	{
		GPUEyeVector << < GRID_SIZE, BLOCK_SIZE >> > (d_matrix, matrix->d_matrix, matrix->GetRow(), m_column);

		//for (size_t i = 0; i < matrix->GetRow(); i++)
		//{
		//	m_matrix[i][i] = matrix->m_matrix[0][i];
		//}
	}

	float Mean()
	{
		GPUMean << < 1, 1 >> > (d_matrix, sum_result, m_row);
		cudaMemcpy(h_sum_result, sum_result, 4, cudaMemcpyDeviceToHost);

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
		cudaMemcpy(h_sum_result, sum_result, 4, cudaMemcpyDeviceToHost);

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

	//float* get_dmatrix()
	//{
	//	return d_matrix;
	//}

	float* d_matrix;

private:
	size_t m_row;
	size_t m_column;

	float* m_matrix;

	int BLOCK_SIZE_X;
	int GRID_SIZE_X;
	int GRID_SIZE_Y;

	dim3 BLOCK_SIZE;
	dim3 GRID_SIZE;

	size_t d_size;

	// Use for SUM
	float* sum_result;
	float* h_sum_result;

	// Use for MAX and CLIP functions
	float m_tmp_max_result;

	// Use for MEAN function
	float m_mean;

	DATA_TYPE m_data_type;
};