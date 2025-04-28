#include <stdio.h>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#include <string>
#include <iostream>
using namespace std;
// 1
__global__ void getXY(int width)
{

    int l_x = threadIdx.x + blockIdx.x * blockDim.x;
    int l_y = threadIdx.y + blockIdx.y * blockDim.y;

    int index = l_y * width + l_x;
    printf("thread (%d, %d) -> index %d\n", l_x, l_y, index);
}

void cuShowXY(dim3 t_grid_size, dim3 t_block_size, int width)
{
    cudaError_t l_cerr;

    getXY<<<t_grid_size, t_block_size>>>(width);

    l_cerr = cudaGetLastError();
    if (l_cerr != cudaSuccess)
    {
        printf("CUDA Error [%d] - '%s'\n", __LINE__, cudaGetErrorString(l_cerr));
    }
    cudaDeviceSynchronize();
}
// 2
__global__ void getOneBinary(int *arr, int count)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= count) return;

    int value = arr[idx];
    char bin[9]; // 8 bitů + '\0'
    
    for (int i = 0; i < 8; ++i) {
        int bit = (value >> (7 - i)) & 1;
        bin[i] = (bit == 1 ? '1' : '0');
    }
    bin[8] = '\0'; // důležité!!

    printf("%3d -> %s\n", value, bin);
}
void cuShowBinary(int *arr, int N)
{
    cudaError_t l_cerr;
    dim3 blockSize(64);//64 vlaken 
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x);


    getOneBinary<<<gridSize, blockSize>>>(arr, N);
    l_cerr = cudaGetLastError();
    if (l_cerr != cudaSuccess)
    {
        printf("CUDA Error [%d] - '%s'\n", __LINE__, cudaGetErrorString(l_cerr));
    }

    cudaDeviceSynchronize();
}
