
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include "cuda_img.h"
#include "animation.h"
#include "FishAnim.h"


__global__ void kernel_insertimage(CudaImg t_big_cuda_pic, CudaImg t_small_cuda_pic, int2 t_position)
{
    int l_y = blockIdx.y * blockDim.y + threadIdx.y;
    int l_x = blockIdx.x * blockDim.x + threadIdx.x;

    if (l_x >= t_small_cuda_pic.m_size.x || l_y >= t_small_cuda_pic.m_size.y)
        return;

    int l_bx = l_x + t_position.x;
    int l_by = l_y + t_position.y;

    if (l_bx < 0 || l_bx >= t_big_cuda_pic.m_size.x || l_by < 0 || l_by >= t_big_cuda_pic.m_size.y)
        return;

    uchar4 pixel_small = t_small_cuda_pic.m_p_uchar4[l_y * t_small_cuda_pic.m_size.x + l_x];
    uchar3 &pixel_big = t_big_cuda_pic.m_p_uchar3[l_by * t_big_cuda_pic.m_size.x + l_bx];

    float alpha = pixel_small.w / 255.0f;

    pixel_big.x = (uchar)(alpha * pixel_small.x + (1.0f - alpha) * pixel_big.x);
    pixel_big.y = (uchar)(alpha * pixel_small.y + (1.0f - alpha) * pixel_big.y);
    pixel_big.z = (uchar)(alpha * pixel_small.z + (1.0f - alpha) * pixel_big.z);
}


void cu_insertimage(CudaImg t_big_cuda_img, CudaImg t_small_cuda_img, int2 t_position)
{
    dim3 blockSize(32, 32);
    dim3 gridSize(
        (t_small_cuda_img.m_size.x + blockSize.x - 1) / blockSize.x,
        (t_small_cuda_img.m_size.y + blockSize.y - 1) / blockSize.y);

    kernel_insertimage<<<gridSize, blockSize>>>(t_big_cuda_img, t_small_cuda_img, t_position);
    cudaDeviceSynchronize(); // synchronizace pro jistotu
}


void Animation::start(CudaImg t_bg_cuda_img, CudaImg t_ins_cuda_img)
{
	if (m_initialized) return;
	cudaError_t l_cerr;

	m_bg_cuda_img = t_bg_cuda_img;
	m_res_cuda_img = t_bg_cuda_img;
	m_ins_cuda_img = t_ins_cuda_img;

	// Memory allocation in GPU device
	l_cerr = cudaMalloc(&m_bg_cuda_img.m_p_void, m_bg_cuda_img.m_size.x * m_bg_cuda_img.m_size.y * sizeof(uchar3));
	if (l_cerr != cudaSuccess)
		printf("CUDA Error [%d] - '%s'\n", __LINE__, cudaGetErrorString(l_cerr));

	// místo gradientu vložíme reálný obrázek
	l_cerr = cudaMemcpy(m_bg_cuda_img.m_p_void, t_bg_cuda_img.m_p_void,
						m_bg_cuda_img.m_size.x * m_bg_cuda_img.m_size.y * sizeof(uchar3),
						cudaMemcpyDeviceToDevice);
	if (l_cerr != cudaSuccess)
		printf("CUDA Error [%d] - '%s'\n", __LINE__, cudaGetErrorString(l_cerr));

	m_initialized = 1;
}
//tohle jsou jen pastenute funkce od pana ucitele takze je budu muset zmenit
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
//tohle jsou jen pastenute funkce od pana ucitele takze je budu muset zmenit
void Animation::stop()
{
	if ( !m_initialized ) return;

	cudaFree( m_bg_cuda_img.m_p_void );
	cudaFree( m_res_cuda_img.m_p_void );
	cudaFree( m_ins_cuda_img.m_p_void );

	m_initialized = 0;
}



void FishAnim::start(CudaImg bg_img, CudaImg *fishes_arr, int fish_count)
{
    if (m_initialized) return;
    cudaError_t l_cerr;

    m_bg_cuda_img = bg_img;
    m_fish_count = fish_count;
    fishes = fishes_arr;

    // Alokace výstupního obrázku (stejné rozměry jako pozadí)
    m_res_cuda_img.m_size = bg_img.m_size;
    l_cerr = cudaMalloc(&m_res_cuda_img.m_p_uchar3, sizeof(uchar3) * bg_img.m_size.x * bg_img.m_size.y);
    if (l_cerr != cudaSuccess)
        printf("CUDA Error [%d] - '%s'\n", __LINE__, cudaGetErrorString(l_cerr));

    // Zkopíruj pozadí do výstupního bufferu
    l_cerr = cudaMemcpy(m_res_cuda_img.m_p_uchar3, bg_img.m_p_uchar3,
                        sizeof(uchar3) * bg_img.m_size.x * bg_img.m_size.y,
                        cudaMemcpyDeviceToDevice);
    if (l_cerr != cudaSuccess)
        printf("CUDA Error [%d] - '%s'\n", __LINE__, cudaGetErrorString(l_cerr));

    m_initialized = true;
}



void FishAnim::next(int2 *positions)
{
    if (!m_initialized) return;

    // Zkopíruj pozadí do výstupního obrázku
    cudaMemcpy(m_res_cuda_img.m_p_uchar3, m_bg_cuda_img.m_p_uchar3,
               sizeof(uchar3) * m_bg_cuda_img.m_size.x * m_bg_cuda_img.m_size.y,
               cudaMemcpyDeviceToDevice);

    // Překresli ryby
    for (int i = 0; i < m_fish_count; ++i)
    {
        cu_insertimage(m_res_cuda_img, fishes[i], positions[i]);
    }

    cudaDeviceSynchronize(); // Pro jistotu
}
void FishAnim::get_result(CudaImg &dest)
{
    // Zkopíruj výstupní buffer do externího cíle
    cudaMemcpy(dest.m_p_uchar3, m_res_cuda_img.m_p_uchar3,
               sizeof(uchar3) * m_res_cuda_img.m_size.x * m_res_cuda_img.m_size.y,
               cudaMemcpyDeviceToDevice);
}
void FishAnim::stop()
{
    if (m_res_cuda_img.m_p_uchar3)
    {
        cudaFree(m_res_cuda_img.m_p_uchar3);
        m_res_cuda_img.m_p_uchar3 = nullptr;
    }

    m_initialized = false;
    fishes = nullptr;
    m_fish_count = 0;
}