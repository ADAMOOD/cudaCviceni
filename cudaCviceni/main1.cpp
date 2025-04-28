
#include <stdio.h>
#include <cuda_runtime.h>
#include <time.h>
#include <cstdlib>//rand()

// Prototype of function from .cu file
void cuShowXY(dim3 t_grid_size, dim3 t_block_size, int width);
void cuShowBinary(int* arr,int N);

int main()
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0)
    {
        printf("No CUDA devices found\n");
        return -1; // Konec programu
    }
    printf("you have %d devices\n", deviceCount);
//1

    dim3 l_grid_size(3, 2), l_block_size(2, 3);
    int width=10;
    cuShowXY(l_grid_size, l_block_size,width);
//2
    srand(time(NULL));
    int count =10;
    int *arr = new int[count];
    if ( cudaMallocManaged( &arr, count * sizeof( *arr ) ) != cudaSuccess )
    {
        printf( "Unable to allocate Unified memory!\n" );
        return 1;
    }
    for(int i=0;i<count;i++)
    {
        arr[i]=rand()%255;
    }
    cuShowBinary(arr,count);
    return 0;
}
