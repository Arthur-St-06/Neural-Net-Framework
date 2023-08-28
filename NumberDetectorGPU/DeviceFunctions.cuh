#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <math.h>
#include <stdint.h>

__global__ void GPUMatrix(float* d_matrix, float* matrix, int X_dim, int Y_dim)
{
	int X = blockIdx.x * blockDim.x + threadIdx.x;
	int Y = blockIdx.y * blockDim.y + threadIdx.y;

	if (X < X_dim && Y < Y_dim)
	{
		d_matrix[Y * X_dim + X] = matrix[Y * X_dim + X];
	}
}

__global__ void GPUTransposedMatrix(float* d_matrix, float* matrix, int X_dim, int Y_dim)
{
	int X = blockIdx.x * blockDim.x + threadIdx.x;
	int Y = blockIdx.y * blockDim.y + threadIdx.y;

	if (X < X_dim && Y < Y_dim)
	{
		d_matrix[Y * X_dim + X] = matrix[X * Y_dim + Y];
	}
}

__global__ void GPUSetRowMatrixToRow(float* d_matrix, float* matrix, int X_dim, int Y_dim, size_t row_to_get, size_t row_to_set)
{
	int X = blockIdx.x * blockDim.x + threadIdx.x;

	if (X < Y_dim)
	{
		d_matrix[row_to_set * X_dim + X] = matrix[row_to_get * X_dim + X];
	}
}

__global__ void GPUSetRowMatrixToColumn(float* d_matrix, float* matrix, int X_dim, int Y_dim, size_t row_to_get, size_t col_to_set)
{
	int X = blockIdx.x * blockDim.x + threadIdx.x;

	if (X < Y_dim)
	{
		d_matrix[X * X_dim + col_to_set] = matrix[row_to_get * X_dim + X];
	}
}

__global__ void GPUSetColMatrixToRow(float* d_matrix, float* matrix, int X_dim, int Y_dim, size_t col_to_get, size_t row_to_set)
{
	int X = blockIdx.x * blockDim.x + threadIdx.x;

	if (X < Y_dim)
	{
		d_matrix[row_to_set * X_dim + X] = matrix[X * X_dim + col_to_get];
	}
}

__global__ void GPUDot(float* d_matrix, float* a, float* b, int A_x, int A_y, int B_x)
{
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	if ((row < A_y) && (col < B_x))
	{
		float tmp = 0.0f;
		for (int k = 0; k < A_x; k++)
		{
			tmp += a[row * A_x + k] * b[k * B_x + col];
		}

		d_matrix[row * B_x + col] = tmp;
	}
}

__device__ float DeviceMax(float a, float b)
{
	return a > b ? a : b;
}

__device__ float DeviceMin(float a, float b)
{
	return a < b ? a : b;
}

__global__ void GPUMax(float* d_matrix, float* matrix, float min, int X_dim, int Y_dim)
{
	int X = blockIdx.x * blockDim.x + threadIdx.x;
	int Y = blockIdx.y * blockDim.y + threadIdx.y;

	if (X < X_dim && Y < Y_dim)
	{
		d_matrix[Y * X_dim + X] = DeviceMax(min, matrix[Y * X_dim + X]);
	}
}

__global__ void GPURowMax(float* d_matrix, float* matrix, int X_dim, int Y_dim)
{
	int X = blockIdx.x * blockDim.x + threadIdx.x;

	if (X < Y_dim)
	{
		float max_num = matrix[X * X_dim];

		for (int i = 1; i < X_dim; i++)
		{
			max_num = DeviceMax(max_num, matrix[X * X_dim + i]);
		}

		d_matrix[X] = max_num;
	}
}

__global__ void GPURowArgmax(float* d_matrix, float* matrix, int X_dim, int Y_dim)
{
	int X = blockIdx.x * blockDim.x + threadIdx.x;

	if (X < Y_dim)
	{
		float max_num = matrix[X * X_dim];
		int max_num_idx = 0;

		for (int i = 1; i < X_dim; i++)
		{
			if (matrix[X * X_dim + i] > max_num)
			{
				max_num_idx = i;
				max_num = matrix[X * X_dim + i];
			}
		}

		d_matrix[X] = max_num_idx;
	}
}

__global__ void GPUMatrixSum(float* matrix, float* result, int X_dim, int Y_dim)
{
	result[0] = 0.0f;

	for (size_t i = 0; i < Y_dim; i++)
	{
		for (size_t j = 0; j < X_dim; j++)
		{
			result[0] += matrix[i * X_dim + j];
		}
	}
}

__global__ void GPURowSum(float* d_matrix, float* matrix, int X_dim, int Y_dim)
{
	int X = blockIdx.x * blockDim.x + threadIdx.x;

	if (X < Y_dim)
	{
		d_matrix[X] = 0;

		for (int i = 0; i < X_dim; i++)
		{
			d_matrix[X] += matrix[X * X_dim + i];
		}
	}
}

__global__ void GPUColSum(float* d_matrix, float* matrix, int X_dim, int Y_dim)
{
	int X = blockIdx.x * blockDim.x + threadIdx.x;

	if (X < X_dim)
	{
		d_matrix[X] = 0;

		for (int i = 1; i < Y_dim; i++)
		{
			d_matrix[X] += matrix[i * X_dim + X];
		}
	}
}

__global__ void GPUAddMatrix(float* d_matrix, float* matrix, int X_dim, int Y_dim)
{
	int X = blockIdx.x * blockDim.x + threadIdx.x;
	int Y = blockIdx.y * blockDim.y + threadIdx.y;

	if (X < X_dim && Y < Y_dim)
	{
		d_matrix[Y * X_dim + X] += matrix[Y * X_dim + X];
	}
}

__global__ void GPUAddMatricies(float* d_matrix, float* matrix1, float* matrix2, int X_dim, int Y_dim)
{
	int X = blockIdx.x * blockDim.x + threadIdx.x;
	int Y = blockIdx.y * blockDim.y + threadIdx.y;

	if (X < X_dim && Y < Y_dim)
	{
		d_matrix[Y * X_dim + X] = matrix1[Y * X_dim + X] + matrix2[Y * X_dim + X];
	}
}

__global__ void GPUAddValue(float* d_matrix, float* matrix, float value, int X_dim, int Y_dim)
{
	int X = blockIdx.x * blockDim.x + threadIdx.x;
	int Y = blockIdx.y * blockDim.y + threadIdx.y;

	if (X < X_dim && Y < Y_dim)
	{
		d_matrix[Y * X_dim + X] = matrix[Y * X_dim + X] + value;
	}
}

__global__ void GPUAddSingleRow(float* d_matrix, float* matrix, int X_dim, int Y_dim)
{
	int X = blockIdx.x * blockDim.x + threadIdx.x;
	int Y = blockIdx.y * blockDim.y + threadIdx.y;

	if (X < X_dim && Y < Y_dim)
	{
		d_matrix[Y * X_dim + X] += matrix[X];
	}
}

__global__ void GPUSubstractMatricies(float* d_matrix, float* minuend_matrix, float* subtrachend_matrix, int X_dim, int Y_dim)
{
	int X = blockIdx.x * blockDim.x + threadIdx.x;
	int Y = blockIdx.y * blockDim.y + threadIdx.y;

	if (X < X_dim && Y < Y_dim)
	{
		d_matrix[Y * X_dim + X] = minuend_matrix[Y * X_dim + X] - subtrachend_matrix[Y * X_dim + X];
	}
}

__global__ void GPUSubstractMatrixFromRowValues(float* d_matrix, float* minuend_matrix, float* subtrachend_matrix, int X_dim, int Y_dim)
{
	int X = blockIdx.x * blockDim.x + threadIdx.x;
	int Y = blockIdx.y * blockDim.y + threadIdx.y;

	if (X < X_dim && Y < Y_dim)
	{
		d_matrix[Y * X_dim + X] = minuend_matrix[Y * X_dim + X] - subtrachend_matrix[Y];
	}
}

__global__ void GPUSubstractValueFromMatrix(float* d_matrix, float* matrix, float value, int X_dim, int Y_dim)
{
	int X = blockIdx.x * blockDim.x + threadIdx.x;
	int Y = blockIdx.y * blockDim.y + threadIdx.y;

	if (X < X_dim && Y < Y_dim)
	{
		d_matrix[Y * X_dim + X] = value - matrix[Y * X_dim + X];
	}
}

__global__ void GPUSubstractMatrixFromValueAtMatrixIdx(float* d_matrix, float* idx_matrix, float value, int X_dim, int Y_dim)
{
	int X = blockIdx.x * blockDim.x + threadIdx.x;

	if (X < Y_dim)
	{
		d_matrix[X * X_dim + (int)idx_matrix[X]] -= value;
	}
}

__global__ void GPUMatriciesMult(float* d_matrix, float* multiplier_matrix, float* multiplicand_matrix, int X_dim, int Y_dim)
{
	int X = blockIdx.x * blockDim.x + threadIdx.x;
	int Y = blockIdx.y * blockDim.y + threadIdx.y;

	if (X < X_dim && Y < Y_dim)
	{
		d_matrix[Y * X_dim + X] = multiplier_matrix[Y * X_dim + X] * multiplicand_matrix[Y * X_dim + X];
	}
}

__global__ void GPUMultByValue(float* d_matrix, float* matrix, float value, int X_dim, int Y_dim)
{
	int X = blockIdx.x * blockDim.x + threadIdx.x;
	int Y = blockIdx.y * blockDim.y + threadIdx.y;

	if (X < X_dim && Y < Y_dim)
	{
		d_matrix[Y * X_dim + X] = matrix[Y * X_dim + X] * value;
	}
}

__global__ void GPUDivideMatrixByValue(float* d_matrix, float* dividend_matrix, float value, int X_dim, int Y_dim)
{
	int X = blockIdx.x * blockDim.x + threadIdx.x;
	int Y = blockIdx.y * blockDim.y + threadIdx.y;

	if (X < X_dim && Y < Y_dim)
	{
		d_matrix[Y * X_dim + X] = dividend_matrix[Y * X_dim + X] / value;
	}
}

__global__ void GPUDivideMatrixByRow(float* d_matrix, float* dividend_matrix, float* divisor_matrix, int X_dim, int Y_dim)
{
	int X = blockIdx.x * blockDim.x + threadIdx.x;
	int Y = blockIdx.y * blockDim.y + threadIdx.y;

	if (X < X_dim && Y < Y_dim)
	{
		d_matrix[Y * X_dim + X] = dividend_matrix[Y * X_dim + X] / divisor_matrix[Y];
	}
}

__global__ void GPUDivideMatrices(float* d_matrix, float* dividend_matrix, float* divisor_matrix, int X_dim, int Y_dim)
{
	int X = blockIdx.x * blockDim.x + threadIdx.x;
	int Y = blockIdx.y * blockDim.y + threadIdx.y;

	if (X < X_dim && Y < Y_dim)
	{
		d_matrix[Y * X_dim + X] = dividend_matrix[Y * X_dim + X] / divisor_matrix[Y * X_dim + X];
	}
}

__global__ void GPUSqrtMatrix(float* d_matrix, float* matrix, int X_dim, int Y_dim)
{
	int X = blockIdx.x * blockDim.x + threadIdx.x;
	int Y = blockIdx.y * blockDim.y + threadIdx.y;

	if (X < X_dim && Y < Y_dim)
	{
		d_matrix[Y * X_dim + X] = std::sqrt(matrix[Y * X_dim + X]);
	}
}

__global__ void GPUPowerMatrix(float* d_matrix, float* matrix, float exponent, int X_dim, int Y_dim)
{
	int X = blockIdx.x * blockDim.x + threadIdx.x;
	int Y = blockIdx.y * blockDim.y + threadIdx.y;

	if (X < X_dim && Y < Y_dim)
	{
		d_matrix[Y * X_dim + X] = std::pow(matrix[Y * X_dim + X], exponent);
	}
}

__global__ void GPUSetZeroIfMatrixValueIsNegative(float* d_matrix, float* matrix, int X_dim, int Y_dim)
{
	int X = blockIdx.x * blockDim.x + threadIdx.x;
	int Y = blockIdx.y * blockDim.y + threadIdx.y;

	if (X < X_dim && Y < Y_dim && matrix[Y * X_dim + X] < 0)
	{
		d_matrix[Y * X_dim + X] = 0;
	}
}

__global__ void GPUExp(float* d_matrix, float* matrix, int X_dim, int Y_dim)
{
	int X = blockIdx.x * blockDim.x + threadIdx.x;
	int Y = blockIdx.y * blockDim.y + threadIdx.y;

	if (X < X_dim && Y < Y_dim)
	{
		d_matrix[Y * X_dim + X] = std::pow(2.71828182845904523536, matrix[Y * X_dim + X]);
	}
}

__global__ void GPUClip(float* d_matrix, float* matrix, float min, float max, int X_dim, int Y_dim)
{
	int X = blockIdx.x * blockDim.x + threadIdx.x;
	int Y = blockIdx.y * blockDim.y + threadIdx.y;

	if (X < X_dim && Y < Y_dim)
	{
		d_matrix[Y * X_dim + X] = DeviceMax(min, DeviceMin(max, matrix[Y * X_dim + X]));
	}
}

__global__ void GPUGetValuesAccordingToMatrices(float* d_matrix, float* values_matrix, float* idxs_matrix, int X_dim, int Y_dim)
{
	int X = blockIdx.x * blockDim.x + threadIdx.x;

	if (X < X_dim)
	{
		d_matrix[X] = values_matrix[X * Y_dim + __float2int_rz(idxs_matrix[X])];
	}
}

__global__ void GPUNegativeLog(float* d_matrix, float* matrix, int X_dim, int Y_dim)
{
	int X = blockIdx.x * blockDim.x + threadIdx.x;
	int Y = blockIdx.y * blockDim.y + threadIdx.y;

	if (X < X_dim && Y < Y_dim)
	{
		d_matrix[Y * X_dim + X] = -std::log(matrix[Y * X_dim + X]);
	}
}

__global__ void GPUAbs(float* d_matrix, float* matrix, int X_dim, int Y_dim)
{
	int X = blockIdx.x * blockDim.x + threadIdx.x;
	int Y = blockIdx.y * blockDim.y + threadIdx.y;

	if (X < X_dim && Y < Y_dim)
	{
		d_matrix[Y * X_dim + X] = std::abs(matrix[Y * X_dim + X]);
	}
}

__global__ void GPUCompareMatrixAndVector(float* d_matrix, float* matrix, float* compare_matrix, int X_dim, int Y_dim)
{
	int X = blockIdx.x * blockDim.x + threadIdx.x;

	if (X < Y_dim)
	{
		if (matrix[X * X_dim] == compare_matrix[X])
		{
			d_matrix[X * X_dim] = 1;
		}
		else
		{
			d_matrix[X * X_dim] = 0;
		}
	}
}

__global__ void GPUOneHotEncode(float* d_matrix, float* matrix, int X_dim, int Y_dim)
{
	int X = blockIdx.x * blockDim.x + threadIdx.x;

	if (X < Y_dim)
	{
		d_matrix[X * X_dim + (int)matrix[X]] = 1;
	}
}

__global__ void GPUEyeVector(float* d_matrix, float* matrix, int X_dim, int Y_dim)
{
	int X = blockIdx.x * blockDim.x + threadIdx.x;

	if (X < Y_dim)
	{
		d_matrix[X * X_dim + X] = matrix[X];
	}
}

__global__ void GPUMean(float* d_matrix, float* mean_result, int X_dim)
{
	mean_result[0] = 0;

	for (size_t i = 0; i < X_dim; i++)
	{
		mean_result[0] += d_matrix[i];
	}

	mean_result[0] /= X_dim;
}

__global__ void GPUColMean(float* d_matrix, float* mean_result, int Y_dim)
{
	mean_result[0] = 0;

	for (size_t i = 0; i < Y_dim; i++)
	{
		mean_result[0] += d_matrix[i];
	}

	mean_result[0] /= Y_dim;
}