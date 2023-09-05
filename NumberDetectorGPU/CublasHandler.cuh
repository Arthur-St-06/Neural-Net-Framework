#include <iostream>
#include <cuda_runtime.h>
#include <cublas_v2.h>

class CuBLASHandler {
public:
    CuBLASHandler()
    {
        // Create the cuBLAS handle in the constructor
        cublasCreate(&handle);
    }

    ~CuBLASHandler()
    {
        // Destroy the cuBLAS handle in the destructor
        cublasDestroy(handle);
    }

    cublasHandle_t GetHandle()
    {
        return handle;
    }

private:
    cublasHandle_t handle;
};