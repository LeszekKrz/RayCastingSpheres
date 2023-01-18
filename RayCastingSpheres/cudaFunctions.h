#include "cuda_runtime.h"
#include "device_launch_parameters.h"

cudaError_t fillWithCuda(char* table, int SCR_WIDTH, int SCR_HEIGHT);

__global__ void fillKernel(char* table, int width)
{
    char* m_table = table + blockIdx.x * width * 3 + blockIdx.y * 3;
    *m_table = 0;
    *(m_table + 1) = 255;
    *(m_table + 2) = 0;
}