
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include "cuda_img.h"
#include "animation.h"

__global__ void kernel_insertimage(CudaImg dst_img, CudaImg src_img, int2 pos) {
    int src_x = blockIdx.x * blockDim.x + threadIdx.x;
    int src_y = blockIdx.y * blockDim.y + threadIdx.y;

    if (src_x >= src_img.m_size.x || src_y >= src_img.m_size.y)
        return;

    int dst_x = src_x + pos.x;
    int dst_y = src_y + pos.y;

    if (dst_x < 0 || dst_x >= dst_img.m_size.x || dst_y < 0 || dst_y >= dst_img.m_size.y)
        return;

    uchar4 src_pixel = src_img.m_p_uchar4[src_y * src_img.m_size.x + src_x];
    uchar3 dst_pixel = dst_img.m_p_uchar3[dst_y * dst_img.m_size.x + dst_x];

    float alpha = src_pixel.w / 255.0f;
    uchar3 result;
    result.x = static_cast<unsigned char>(src_pixel.x * alpha + dst_pixel.x * (1.0f - alpha));
    result.y = static_cast<unsigned char>(src_pixel.y * alpha + dst_pixel.y * (1.0f - alpha));
    result.z = static_cast<unsigned char>(src_pixel.z * alpha + dst_pixel.z * (1.0f - alpha));

    dst_img.m_p_uchar3[dst_y * dst_img.m_size.x + dst_x] = result;
}


void cu_insertimage(CudaImg t_big_cuda_pic, CudaImg t_small_cuda_pic, int2 t_position)
{
    cudaError_t l_cerr;

    // Grid creation, size of grid must be equal or greater than images
    int l_block_size = 32;
    dim3 l_blocks((t_small_cuda_pic.m_size.x + l_block_size - 1) / l_block_size,
                  (t_small_cuda_pic.m_size.y + l_block_size - 1) / l_block_size);
    dim3 l_threads(l_block_size, l_block_size);
    kernel_insertimage<<<l_blocks, l_threads>>>(t_big_cuda_pic, t_small_cuda_pic, t_position);

    if ((l_cerr = cudaGetLastError()) != cudaSuccess)
        printf("CUDA Error [%d] - '%s'\n", __LINE__, cudaGetErrorString(l_cerr));

    cudaDeviceSynchronize();
}
void Animation::start( CudaImg t_bg_cuda_img, CudaImg t_ins_cuda_img )
{
	if ( m_initialized ) return;
	cudaError_t l_cerr;

	m_bg_cuda_img = t_bg_cuda_img;
	m_res_cuda_img = t_bg_cuda_img;
	m_ins_cuda_img = t_ins_cuda_img;

	// Memory allocation in GPU device
	// Memory for background
	l_cerr = cudaMalloc( &m_bg_cuda_img.m_p_void, m_bg_cuda_img.m_size.x * m_bg_cuda_img.m_size.y * sizeof( uchar3 ) );
	if ( l_cerr != cudaSuccess )
		printf( "CUDA Error [%d] - '%s'\n", __LINE__, cudaGetErrorString( l_cerr ) );

	// Creation of background gradient
	int l_block_size = 32;
	dim3 l_blocks( ( m_bg_cuda_img.m_size.x + l_block_size - 1 ) / l_block_size,
			       ( m_bg_cuda_img.m_size.y + l_block_size - 1 ) / l_block_size );
	dim3 l_threads( l_block_size, l_block_size );

	m_initialized = 1;
}

void Animation::next( CudaImg t_res_cuda_img, int2 t_position )
{
	if ( !m_initialized ) return;

	cudaError_t cerr;

	// Copy data internally GPU from background into result
	cerr = cudaMemcpy( m_res_cuda_img.m_p_void, m_bg_cuda_img.m_p_void, m_bg_cuda_img.m_size.x * m_bg_cuda_img.m_size.y * sizeof( uchar3 ), cudaMemcpyDeviceToDevice );
	if ( cerr != cudaSuccess )
		printf( "CUDA Error [%d] - '%s'\n", __LINE__, cudaGetErrorString( cerr ) );

	// insert picture
	int l_block_size = 32;
	dim3 l_blocks( ( m_ins_cuda_img.m_size.x + l_block_size - 1 ) / l_block_size,
			       ( m_ins_cuda_img.m_size.y + l_block_size - 1 ) / l_block_size );
	dim3 l_threads( l_block_size, l_block_size );
	kernel_insertimage<<< l_blocks, l_threads >>>( m_res_cuda_img, m_ins_cuda_img, t_position );

	// Copy data to GPU device
	cerr = cudaMemcpy( t_res_cuda_img.m_p_void, m_res_cuda_img.m_p_void, m_res_cuda_img.m_size.x * m_res_cuda_img.m_size.y * sizeof( uchar3 ), cudaMemcpyDeviceToHost );
	if ( cerr != cudaSuccess )
		printf( "CUDA Error [%d] - '%s'\n", __LINE__, cudaGetErrorString( cerr ) );

}

void Animation::stop()
{
	if ( !m_initialized ) return;

	cudaFree( m_bg_cuda_img.m_p_void );
	cudaFree( m_res_cuda_img.m_p_void );
	cudaFree( m_ins_cuda_img.m_p_void );

	m_initialized = 0;
}
