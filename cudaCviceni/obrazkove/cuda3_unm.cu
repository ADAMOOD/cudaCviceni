
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include "cuda_img.h"

// Every threads identifies its position in grid and in block and modify image
__global__ void kernel_animation( CudaImg t_cuda_img )
{
    // X,Y coordinates 
    int l_y = blockDim.y * blockIdx.y + threadIdx.y;
    int l_x = blockDim.x * blockIdx.x + threadIdx.x;
    if ( l_x >= t_cuda_img.m_size.x ) return;
    if ( l_y >= t_cuda_img.m_size.y ) return;

    // Point [l_x,l_y] selection from image
    uchar3 l_bgr, l_tmp = t_cuda_img.m_p_uchar3[ l_y * t_cuda_img.m_size.x + l_x ];//to same co ve cviceni 1 je to proste index toho threadu - pixelu

    // color rotation
    l_bgr.x = l_tmp.y;
    l_bgr.y = l_tmp.z;
    l_bgr.z = l_tmp.x;

    // Store point [l_x,l_y] back to image
    t_cuda_img.m_p_uchar3[ l_y * t_cuda_img.m_size.x + l_x ] = l_bgr;
}

void cu_run_animation( CudaImg t_cuda_img, uint2 t_block_size )
{
    cudaError_t l_cerr;

    // Grid creation with computed organization
    dim3 l_grid( ( t_cuda_img.m_size.x + t_block_size.x - 1 ) / t_block_size.x,
                 ( t_cuda_img.m_size.y + t_block_size.y - 1 ) / t_block_size.y );//automaticka generace gridu podle obrazku

    kernel_animation<<< l_grid, dim3( t_block_size.x, t_block_size.y ) >>>( t_cuda_img );

    if ( ( l_cerr = cudaGetLastError() ) != cudaSuccess )
        printf( "CUDA Error [%d] - '%s'\n", __LINE__, cudaGetErrorString( l_cerr ) );

    cudaDeviceSynchronize();

}
//4
__global__ void setBrightness( CudaImg t_cuda_img,int percentage)
{
    // X,Y coordinates 
    int l_y = blockDim.y * blockIdx.y + threadIdx.y;
    int l_x = blockDim.x * blockIdx.x + threadIdx.x;
    if ( l_x >= t_cuda_img.m_size.x ) return;
    if ( l_y >= t_cuda_img.m_size.y ) return;

    // Point [l_x,l_y] selection from image
    uchar3 l_bgr, l_tmp = t_cuda_img.m_p_uchar3[ l_y * t_cuda_img.m_size.x + l_x ];//to same co ve cviceni 1 je to proste index toho threadu - pixelu
    
    l_bgr.x = min(255, (l_tmp.x * percentage) / 100);
    l_bgr.y = min(255, (l_tmp.y * percentage) / 100);
    l_bgr.z = min(255, (l_tmp.z * percentage) / 100);

    // Store point [l_x,l_y] back to image
    t_cuda_img.m_p_uchar3[ l_y * t_cuda_img.m_size.x + l_x ] =l_bgr;
}
void cuSetBrightness(CudaImg t_cuda_img, uint2 t_block_size,int percentage)
{
    cudaError_t l_cerr;

    // Grid creation with computed organization
    dim3 l_grid( ( t_cuda_img.m_size.x + t_block_size.x - 1 ) / t_block_size.x,
                 ( t_cuda_img.m_size.y + t_block_size.y - 1 ) / t_block_size.y );//automaticka generace gridu podle obrazku

    setBrightness<<< l_grid, dim3( t_block_size.x, t_block_size.y ) >>>( t_cuda_img,percentage);

    if ( ( l_cerr = cudaGetLastError() ) != cudaSuccess )
        printf( "CUDA Error [%d] - '%s'\n", __LINE__, cudaGetErrorString( l_cerr ) );

    cudaDeviceSynchronize(); 
}
//3
__global__ void rotate90(CudaImg t_cuda_img, CudaImg rot)
{
    int l_y = blockDim.y * blockIdx.y + threadIdx.y;
    int l_x = blockDim.x * blockIdx.x + threadIdx.x;
    
    if (l_x >= t_cuda_img.m_size.x || l_y >= t_cuda_img.m_size.y)
        return;
    
    uchar3 l_tmp = t_cuda_img.m_p_uchar3[l_y * t_cuda_img.m_size.x + l_x];

    // POZOR! Přepočítáme souřadnice (rotace o 90° doprava):
    int new_x = t_cuda_img.m_size.y - 1 - l_y;
    int new_y = l_x;

    rot.m_p_uchar3[new_y * rot.m_size.x + new_x] = l_tmp;
}


void cuRotate90(CudaImg t_cuda_img, cv::Mat& output_mat, uint2 t_block_size)
{
    cudaError_t l_cerr;

    // Připravíme výstupní CudaImg
    CudaImg rot;
    rot.m_size.x = t_cuda_img.m_size.y;  // Rozměry se prohodí!
    rot.m_size.y = t_cuda_img.m_size.x;
    rot.m_p_uchar3 = (uchar3*)output_mat.ptr<uchar3>();

    // Vytvoření gridu
    dim3 l_grid(
        (t_cuda_img.m_size.x + t_block_size.x - 1) / t_block_size.x,
        (t_cuda_img.m_size.y + t_block_size.y - 1) / t_block_size.y
    );

    // Spuštění kernelu
    rotate90<<<l_grid, dim3(t_block_size.x, t_block_size.y)>>>(t_cuda_img, rot);

    // Kontrola chyb
    if ((l_cerr = cudaGetLastError()) != cudaSuccess)
        printf("CUDA Error [%d] - '%s'\n", __LINE__, cudaGetErrorString(l_cerr));

    cudaDeviceSynchronize();
}

__global__ void mirror(CudaImg t_src_img, CudaImg t_dst_img)
{
    int l_y = blockDim.y * blockIdx.y + threadIdx.y;
    int l_x = blockDim.x * blockIdx.x + threadIdx.x;
    if (l_x >= t_src_img.m_size.x) return;
    if (l_y >= t_src_img.m_size.y) return;

    int mirrored_x = t_src_img.m_size.x - 1 - l_x;

    // načítám z originálního obrázku
    uchar3 l_tmp = t_src_img.m_p_uchar3[l_y * t_src_img.m_size.x + mirrored_x];

    // ukládám do nového obrázku
    t_dst_img.m_p_uchar3[l_y * t_dst_img.m_size.x + l_x] = l_tmp;
}

void cuMirror(CudaImg t_src_img, CudaImg t_dst_img, uint2 t_block_size)
{
    cudaError_t l_cerr;

    dim3 l_grid((t_src_img.m_size.x + t_block_size.x - 1) / t_block_size.x,
                (t_src_img.m_size.y + t_block_size.y - 1) / t_block_size.y);

    mirror<<<l_grid, dim3(t_block_size.x, t_block_size.y)>>>(t_src_img, t_dst_img);

    if ((l_cerr = cudaGetLastError()) != cudaSuccess)
        printf("CUDA Error [%d] - '%s'\n", __LINE__, cudaGetErrorString(l_cerr));

    cudaDeviceSynchronize();
}

