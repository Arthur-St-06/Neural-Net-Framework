#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cublas_v2.h>

#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <math.h>
#include <stdint.h>

__global__ void floattohalf(const float* src, __half* dest, int X_dim) {
	int X = blockIdx.x * blockDim.x + threadIdx.x;
	//int Y = blockIdx.y * blockDim.y + threadIdx.y;

	if (X < X_dim) {
		dest[X] = __float2half(src[X]);
	}
}

__global__ void halftofloat(float* dest, __half* src, int X_dim, int Y_dim) {
	int X = blockIdx.x * blockDim.x + threadIdx.x;
	int Y = blockIdx.y * blockDim.y + threadIdx.y;

	if (X < X_dim && Y < Y_dim) {
		dest[Y * X_dim + X] = __half2float(src[Y * X_dim + X]);
	}
}

__global__ void GPUMatrix(half* d_matrix, half* matrix, int X_dim, int Y_dim)
{
	int X = blockIdx.x * blockDim.x + threadIdx.x;
	int Y = blockIdx.y * blockDim.y + threadIdx.y;

	if (X < X_dim && Y < Y_dim)
	{
		d_matrix[Y * X_dim + X] = matrix[Y * X_dim + X];
	}
}

__global__ void GPUTransposedMatrix(half* d_matrix, half* matrix, int X_dim, int Y_dim)
{
	int X = blockIdx.x * blockDim.x + threadIdx.x;
	int Y = blockIdx.y * blockDim.y + threadIdx.y;

	if (X < X_dim && Y < Y_dim)
	{
		d_matrix[Y * X_dim + X] = matrix[X * Y_dim + Y];
	}
}

__global__ void GPUSetRowMatrixToRow(half* d_matrix, half* matrix, int X_dim, int Y_dim, size_t row_to_get, size_t row_to_set)
{
	int Y = blockIdx.y * blockDim.y + threadIdx.y;

	if (Y < Y_dim)
	{
		d_matrix[row_to_set * X_dim + Y] = matrix[row_to_get * X_dim + Y];
	}
}

__global__ void GPUSetRowMatrixToColumn(half* d_matrix, half* matrix, int X_dim, int Y_dim, size_t row_to_get, size_t col_to_set)
{
	int Y = blockIdx.y * blockDim.y + threadIdx.y;

	if (Y < Y_dim)
	{
		d_matrix[Y * X_dim + col_to_set] = matrix[row_to_get * X_dim + Y];
	}
}

__global__ void GPUSetColMatrixToRow(half* d_matrix, half* matrix, int X_dim, int Y_dim, size_t col_to_get, size_t row_to_set)
{
	int Y = blockIdx.y * blockDim.y + threadIdx.y;

	if (Y < Y_dim)
	{
		d_matrix[row_to_set * X_dim + Y] = matrix[Y * X_dim + col_to_get];
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

__global__ void GPUMax(half* d_matrix, half* matrix, float min, int X_dim, int Y_dim)
{
	int X = blockIdx.x * blockDim.x + threadIdx.x;
	int Y = blockIdx.y * blockDim.y + threadIdx.y;

	if (X < X_dim && Y < Y_dim)
	{
		d_matrix[Y * X_dim + X] = DeviceMax(min, matrix[Y * X_dim + X]);
	}
}

__global__ void GPURowMax(half* d_matrix, half* matrix, int X_dim, int Y_dim)
{
	int Y = blockIdx.y * blockDim.y + threadIdx.y;

	if (Y < Y_dim)
	{
		float max_num = matrix[Y * X_dim];

		for (int i = 1; i < X_dim; i++)
		{
			max_num = DeviceMax(max_num, matrix[Y * X_dim + i]);
		}

		d_matrix[Y] = max_num;
	}
}

__global__ void GPURowArgmax(half* d_matrix, half* matrix, int X_dim, int Y_dim)
{
	int Y = blockIdx.y * blockDim.y + threadIdx.y;

	if (Y < Y_dim)
	{
		half max_num = matrix[Y * X_dim];
		int max_num_idx = 0;

		for (int i = 1; i < X_dim; i++)
		{
			if (matrix[Y * X_dim + i] > max_num)
			{
				max_num_idx = i;
				max_num = matrix[Y * X_dim + i];
			}
		}

		d_matrix[Y] = max_num_idx;
	}
}

__global__ void GPUMatrixSum(half* matrix, float* result, int X_dim, int Y_dim)
{
	result[0] = 0.0f;

	for (size_t i = 0; i < Y_dim; i++)
	{
		for (size_t j = 0; j < X_dim; j++)
		{
			result[0] += __half2float(matrix[i * X_dim + j]);
		}
	}
}

__global__ void GPURowSum(half* d_matrix, half* matrix, int X_dim, int Y_dim)
{
	int Y = blockIdx.y * blockDim.y + threadIdx.y;

	if (Y < Y_dim)
	{
		d_matrix[Y] = 0;

		for (int i = 0; i < X_dim; i++)
		{
			d_matrix[Y] += matrix[Y * X_dim + i];
		}
	}
}

__global__ void GPUColSum(half* d_matrix, half* matrix, int X_dim, int Y_dim)
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

__global__ void GPUAddMatrix(half* d_matrix, half* matrix, int X_dim, int Y_dim)
{
	int X = blockIdx.x * blockDim.x + threadIdx.x;
	int Y = blockIdx.y * blockDim.y + threadIdx.y;

	if (X < X_dim && Y < Y_dim)
	{
		d_matrix[Y * X_dim + X] += matrix[Y * X_dim + X];
	}
}

__global__ void GPUAddMatricies(half* d_matrix, half* matrix1, half* matrix2, int X_dim, int Y_dim)
{
	int X = blockIdx.x * blockDim.x + threadIdx.x;
	int Y = blockIdx.y * blockDim.y + threadIdx.y;

	if (X < X_dim && Y < Y_dim)
	{
		d_matrix[Y * X_dim + X] = matrix1[Y * X_dim + X] + matrix2[Y * X_dim + X];
	}
}

__global__ void GPUAddValue(half* d_matrix, half* matrix, float value, int X_dim, int Y_dim)
{
	int X = blockIdx.x * blockDim.x + threadIdx.x;
	int Y = blockIdx.y * blockDim.y + threadIdx.y;

	if (X < X_dim && Y < Y_dim)
	{
		d_matrix[Y * X_dim + X] = matrix[Y * X_dim + X] + __float2half(value);
	}
}

__global__ void GPUAddSingleRow(half* d_matrix, half* matrix, int X_dim, int Y_dim)
{
	int X = blockIdx.x * blockDim.x + threadIdx.x;
	int Y = blockIdx.y * blockDim.y + threadIdx.y;

	if (X < X_dim && Y < Y_dim)
	{
		d_matrix[Y * X_dim + X] += matrix[X];
	}
}

__global__ void GPUSubstractMatricies(half* d_matrix, half* minuend_matrix, half* subtrachend_matrix, int X_dim, int Y_dim)
{
	int X = blockIdx.x * blockDim.x + threadIdx.x;
	int Y = blockIdx.y * blockDim.y + threadIdx.y;

	if (X < X_dim && Y < Y_dim)
	{
		d_matrix[Y * X_dim + X] = minuend_matrix[Y * X_dim + X] - subtrachend_matrix[Y * X_dim + X];
	}
}

__global__ void GPUSubstractMatrixFromRowValues(half* d_matrix, half* minuend_matrix, half* subtrachend_matrix, int X_dim, int Y_dim)
{
	int X = blockIdx.x * blockDim.x + threadIdx.x;
	int Y = blockIdx.y * blockDim.y + threadIdx.y;

	if (X < X_dim && Y < Y_dim)
	{
		d_matrix[Y * X_dim + X] = minuend_matrix[Y * X_dim + X] - subtrachend_matrix[Y];
	}
}

__global__ void GPUSubstractValueFromMatrix(half* d_matrix, half* matrix, float value, int X_dim, int Y_dim)
{
	int X = blockIdx.x * blockDim.x + threadIdx.x;
	int Y = blockIdx.y * blockDim.y + threadIdx.y;

	if (X < X_dim && Y < Y_dim)
	{
		d_matrix[Y * X_dim + X] = __float2half(value) - matrix[Y * X_dim + X];
	}
}

__global__ void GPUSubstractMatrixFromValueAtMatrixIdx(half* d_matrix, half* idx_matrix, float value, int X_dim, int Y_dim)
{
	int Y = blockIdx.y * blockDim.y + threadIdx.y;

	if (Y < Y_dim)
	{
		d_matrix[Y * X_dim + (int)idx_matrix[Y]] -= value;
	}
}

__global__ void GPUMatriciesMult(half* d_matrix, half* multiplier_matrix, half* multiplicand_matrix, int X_dim, int Y_dim)
{
	int X = blockIdx.x * blockDim.x + threadIdx.x;
	int Y = blockIdx.y * blockDim.y + threadIdx.y;

	if (X < X_dim && Y < Y_dim)
	{
		d_matrix[Y * X_dim + X] = multiplier_matrix[Y * X_dim + X] * multiplicand_matrix[Y * X_dim + X];
	}
}

__global__ void GPUMultByValue(half* d_matrix, half* matrix, float value, int X_dim, int Y_dim)
{
	int X = blockIdx.x * blockDim.x + threadIdx.x;
	int Y = blockIdx.y * blockDim.y + threadIdx.y;

	if (X < X_dim && Y < Y_dim)
	{
		d_matrix[Y * X_dim + X] = matrix[Y * X_dim + X] * __float2half(value);
	}
}

__global__ void GPUDivideMatrixByValue(half* d_matrix, half* dividend_matrix, float value, int X_dim, int Y_dim)
{
	int X = blockIdx.x * blockDim.x + threadIdx.x;
	int Y = blockIdx.y * blockDim.y + threadIdx.y;

	if (X < X_dim && Y < Y_dim)
	{
		d_matrix[Y * X_dim + X] = dividend_matrix[Y * X_dim + X] / __float2half(value);
	}
}

__global__ void GPUDivideMatrixByRow(half* d_matrix, half* dividend_matrix, half* divisor_matrix, int X_dim, int Y_dim)
{
	int X = blockIdx.x * blockDim.x + threadIdx.x;
	int Y = blockIdx.y * blockDim.y + threadIdx.y;

	if (X < X_dim && Y < Y_dim)
	{
		d_matrix[Y * X_dim + X] = dividend_matrix[Y * X_dim + X] / divisor_matrix[Y];
	}
}

__global__ void GPUDivideMatrices(half* d_matrix, half* dividend_matrix, half* divisor_matrix, int X_dim, int Y_dim)
{
	int X = blockIdx.x * blockDim.x + threadIdx.x;
	int Y = blockIdx.y * blockDim.y + threadIdx.y;

	if (X < X_dim && Y < Y_dim)
	{
		d_matrix[Y * X_dim + X] = dividend_matrix[Y * X_dim + X] / divisor_matrix[Y * X_dim + X];
	}
}

__global__ void GPUSqrtMatrix(half* d_matrix, half* matrix, int X_dim, int Y_dim)
{
	int X = blockIdx.x * blockDim.x + threadIdx.x;
	int Y = blockIdx.y * blockDim.y + threadIdx.y;

	if (X < X_dim && Y < Y_dim)
	{
		d_matrix[Y * X_dim + X] = sqrtf(matrix[Y * X_dim + X]) - 0.0001f;
	}
}

__global__ void GPUPowerMatrix(half* d_matrix, half* matrix, float exponent, int X_dim, int Y_dim)
{
	int X = blockIdx.x * blockDim.x + threadIdx.x;
	int Y = blockIdx.y * blockDim.y + threadIdx.y;

	if (X < X_dim && Y < Y_dim)
	{
		d_matrix[Y * X_dim + X] = powf(matrix[Y * X_dim + X], exponent);
	}
}

__global__ void GPUSetZeroIfMatrixValueIsNegative(half* d_matrix, half* matrix, int X_dim, int Y_dim)
{
	int X = blockIdx.x * blockDim.x + threadIdx.x;
	int Y = blockIdx.y * blockDim.y + threadIdx.y;

	if (X < X_dim && Y < Y_dim && __hlt(matrix[Y * X_dim + X], __float2half(0.0f)))
	{
		d_matrix[Y * X_dim + X] = 0.0f;
	}
}

__global__ void GPUExp(half* d_matrix, half* matrix, int X_dim, int Y_dim)
{
	int X = blockIdx.x * blockDim.x + threadIdx.x;
	int Y = blockIdx.y * blockDim.y + threadIdx.y;

	if (X < X_dim && Y < Y_dim)
	{
		d_matrix[Y * X_dim + X] = powf(2.71828182845904523536, matrix[Y * X_dim + X]);
	}
}

__global__ void GPUClip(half* d_matrix, half* matrix, float min, float max, int X_dim, int Y_dim)
{
	int X = blockIdx.x * blockDim.x + threadIdx.x;
	int Y = blockIdx.y * blockDim.y + threadIdx.y;

	if (X < X_dim && Y < Y_dim)
	{
		d_matrix[Y * X_dim + X] = DeviceMax(min, DeviceMin(max, matrix[Y * X_dim + X]));
	}
}

__global__ void GPUGetValuesAccordingToMatrices(half* d_matrix, half* values_matrix, half* idxs_matrix, int X_dim, int Y_dim)
{
	int X = blockIdx.x * blockDim.x + threadIdx.x;

	if (X < X_dim)
	{
		d_matrix[X] = values_matrix[X * Y_dim + (int)(idxs_matrix[X])];
	}
}

__global__ void GPUNegativeLog(half* d_matrix, half* matrix, int X_dim, int Y_dim)
{
	int X = blockIdx.x * blockDim.x + threadIdx.x;
	int Y = blockIdx.y * blockDim.y + threadIdx.y;

	if (X < X_dim && Y < Y_dim)
	{
		d_matrix[Y * X_dim + X] = -std::log(matrix[Y * X_dim + X]);
	}
}

__global__ void GPUAbs(half* d_matrix, half* matrix, int X_dim, int Y_dim)
{
	int X = blockIdx.x * blockDim.x + threadIdx.x;
	int Y = blockIdx.y * blockDim.y + threadIdx.y;

	if (X < X_dim && Y < Y_dim)
	{
		d_matrix[Y * X_dim + X] = fabs(matrix[Y * X_dim + X]);
	}
}

__global__ void GPUCompareMatrixAndVector(half* d_matrix, half* matrix, half* compare_matrix, int X_dim, int Y_dim)
{
	int Y = blockIdx.y * blockDim.y + threadIdx.y;

	if (Y < Y_dim)
	{
		if (matrix[Y * X_dim] == compare_matrix[Y])
		{
			d_matrix[Y * X_dim] = 1;
		}
		else
		{
			d_matrix[Y * X_dim] = 0;
		}
	}
}

__global__ void GPUOneHotEncode(half* d_matrix, half* matrix, int X_dim, int Y_dim)
{
	int Y = blockIdx.y * blockDim.y + threadIdx.y;

	if (Y < Y_dim)
	{
		d_matrix[Y * X_dim + (int)matrix[Y]] = 1;
	}
}

__global__ void GPUEyeVector(half* d_matrix, half* matrix, int X_dim, int Y_dim)
{
	int Y = blockIdx.y * blockDim.y + threadIdx.y;

	if (Y < Y_dim)
	{
		d_matrix[Y * X_dim + Y] = matrix[Y];
	}
}

__global__ void GPUMean(half* d_matrix, float* mean_result, int X_dim)
{
	mean_result[0] = 0;

	for (size_t i = 0; i < X_dim; i++)
	{
		mean_result[0] += __half2float(d_matrix[i]);
	}

	mean_result[0] /= X_dim;
}

__global__ void GPUColMean(half* d_matrix, float* mean_result, int Y_dim)
{
	mean_result[0] = 0;

	for (size_t i = 0; i < Y_dim; i++)
	{
		mean_result[0] += __half2float(d_matrix[i]);
	}

	mean_result[0] /= Y_dim;
}