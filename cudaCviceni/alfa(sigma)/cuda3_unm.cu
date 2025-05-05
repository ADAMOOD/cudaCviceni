
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include "cuda_img.h"

// Demo kernel to create picture with alpha channel gradient
__global__ void kernel_insertimage(CudaImg t_big_cuda_pic, CudaImg t_small_cuda_pic, int2 t_position)
{
    // Výpočty souřadnic
    int l_y = blockIdx.y * blockDim.y + threadIdx.y;
    int l_x = blockIdx.x * blockDim.x + threadIdx.x;

    // Podmínky pro kontrolu, zda jsou souřadnice uvnitř rozsahu obrázku
    if (l_x >= t_small_cuda_pic.m_size.x || l_y >= t_small_cuda_pic.m_size.y)
        return;

    int l_by = l_y + t_position.y;
    int l_bx = l_x + t_position.x;

    // Kontrola, zda jsou souřadnice v rámci většího obrázku
    if (l_bx >= t_big_cuda_pic.m_size.x || l_by >= t_big_cuda_pic.m_size.y)
        return;

    // Získání pixelu z malého obrázku
    uchar4 l_fg_bgra = t_small_cuda_pic.m_p_uchar4[l_y * t_small_cuda_pic.m_size.x + l_x];
    uchar3 l_bg_bgr = t_big_cuda_pic.m_p_uchar3[l_by * t_big_cuda_pic.m_size.x + l_bx];

    // Výpočet složek BGR na základě alfa kanálu
    uchar3 l_bgr = {
        (unsigned char)(l_fg_bgra.x * l_fg_bgra.w / 255 + l_bg_bgr.x * (255 - l_fg_bgra.w) / 255),
        (unsigned char)(l_fg_bgra.y * l_fg_bgra.w / 255 + l_bg_bgr.y * (255 - l_fg_bgra.w) / 255),
        (unsigned char)(l_fg_bgra.z * l_fg_bgra.w / 255 + l_bg_bgr.z * (255 - l_fg_bgra.w) / 255)};

    // Uložení výsledného pixelu do většího obrázku
    t_big_cuda_pic.m_p_uchar3[l_by * t_big_cuda_pic.m_size.x + l_bx] = l_bgr;
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

__global__ void kernel_RoundTransparency(CudaImg img, int lowerBound)
{
    int l_y = blockIdx.y * blockDim.y + threadIdx.y;
    int l_x = blockIdx.x * blockDim.x + threadIdx.x;

    if (l_x < img.m_size.x && l_y < img.m_size.y)
    {
        uchar4 img_bgra = img.m_p_uchar4[l_y * img.m_size.x + l_x];

        // Pokud hodnota alfa kanálu (A) je menší než lowerBound, nastavíme alfa kanál na 0
        if (img_bgra.w < lowerBound)
        {
            img_bgra.w = 0;
        }

        // Aktualizujeme hodnoty zpět do obrázku
        img.m_p_uchar4[l_y * img.m_size.x + l_x] = img_bgra;
    }
}
void cu_insertimageRoundTransparency(CudaImg t_big_cuda_pic, CudaImg t_small_cuda_pic, int2 t_position, int lowerBound)
{
    cudaError_t l_cerr;

    // Grid creation, size of grid must be equal or greater than images
    int l_block_size = 32;
    dim3 l_blocks((t_small_cuda_pic.m_size.x + l_block_size - 1) / l_block_size,
                  (t_small_cuda_pic.m_size.y + l_block_size - 1) / l_block_size);
    dim3 l_threads(l_block_size, l_block_size);

    kernel_RoundTransparency<<<l_blocks, l_threads>>>(t_small_cuda_pic, lowerBound);
    if ((l_cerr = cudaGetLastError()) != cudaSuccess)
        printf("CUDA Error [%d] - '%s'\n", __LINE__, cudaGetErrorString(l_cerr));
    // cudaDeviceSynchronize();

    kernel_insertimage<<<l_blocks, l_threads>>>(t_big_cuda_pic, t_small_cuda_pic, t_position);
    if ((l_cerr = cudaGetLastError()) != cudaSuccess)
        printf("CUDA Error [%d] - '%s'\n", __LINE__, cudaGetErrorString(l_cerr));

    cudaDeviceSynchronize();
}
__global__ void kernel_merge(CudaImg img1, CudaImg img2, CudaImg result)
{
    int l_y = blockIdx.y * blockDim.y + threadIdx.y;
    int l_x = blockIdx.x * blockDim.x + threadIdx.x;

    result.m_size = img1.m_size;

    if (l_x < result.m_size.x && l_y < result.m_size.y)
    {
        uchar4 img1_bgra = img1.m_p_uchar4[l_y * img1.m_size.x + l_x];
        uchar4 img2_bgra = img2.m_p_uchar4[l_y * img2.m_size.x + l_x];
        uchar4 result_bgra = result.m_p_uchar4[l_y * result.m_size.x + l_x];

        // Pokud hodnota alfa kanálu (A) je menší než lowerBound, nastavíme alfa kanál na 0
        result_bgra = (img1_bgra.w >= img2_bgra.w) ? img1_bgra : img2_bgra;

        // Aktualizujeme hodnoty zpět do obrázku
        result.m_p_uchar4[l_y * result.m_size.x + l_x] = result_bgra;
    }
}

void cu_mergeByAlpha(CudaImg img1, CudaImg img2, CudaImg background, int2 t_position)
{
    cudaError_t l_cerr;

    int l_block_size = 32;
    dim3 l_blocks((img1.m_size.x + l_block_size - 1) / l_block_size,
                  (img1.m_size.y + l_block_size - 1) / l_block_size);
    dim3 l_threads(l_block_size, l_block_size);

    // Alokace GPU paměti pro výsledek
    CudaImg mergedImages;
    mergedImages.m_size = img1.m_size;
    cudaMalloc(&mergedImages.m_p_uchar4, sizeof(uchar4) * mergedImages.m_size.x * mergedImages.m_size.y);

    kernel_merge<<<l_blocks, l_threads>>>(img1, img2, mergedImages);
    if ((l_cerr = cudaGetLastError()) != cudaSuccess)
        printf("CUDA Error [%d] - '%s'\n", __LINE__, cudaGetErrorString(l_cerr));
    cudaDeviceSynchronize();

    kernel_insertimage<<<l_blocks, l_threads>>>(background, mergedImages, t_position);
    if ((l_cerr = cudaGetLastError()) != cudaSuccess)
        printf("CUDA Error [%d] - '%s'\n", __LINE__, cudaGetErrorString(l_cerr));
    cudaDeviceSynchronize();

    cudaFree(mergedImages.m_p_uchar4); // uvolnění paměti
}
__global__ void kernel_add_shadow(CudaImg input_img, CudaImg output_img, int2 shadow_offset) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < input_img.m_size.x && y < input_img.m_size.y) {
        uchar4 pixel = input_img.m_p_uchar4[y * input_img.m_size.x + x];

        // Pokud má pixel neprůhlednost, připravíme stín
        if (pixel.w > 0) {
            int shadow_x = x + shadow_offset.x;
            int shadow_y = y + shadow_offset.y;

            // Pokud stín padá do výstupního obrázku
            if (shadow_x < output_img.m_size.x && shadow_y < output_img.m_size.y) {
                // Zjisti, zda v místě stínu není zároveň obraz
                // Stín vložíme JEN pokud na tom místě není obraz
                int output_idx = shadow_y * output_img.m_size.x + shadow_x;
                uchar4 existing_pixel = output_img.m_p_uchar4[output_idx];

                // Pokud tam ještě není plně neprůhledný pixel (A < 255), vložíme stín
                if (existing_pixel.w == 0) {
                    uchar4 shadow_pixel = make_uchar4(0, 0, 0, 128); // černý stín
                    output_img.m_p_uchar4[output_idx] = shadow_pixel;
                }
            }

            // Nyní vykreslíme původní obraz přesně na jeho souřadnicích
            output_img.m_p_uchar4[y * output_img.m_size.x + x] = pixel;
        }
    }
}



void cu_addShadow(CudaImg img1, CudaImg background, int2 t_position)
{
    cudaError_t l_cerr;

    int2 shadow_offset = {1, 1}; // nebo jiný posun dle potřeby

    // Výstupní obrázek musí být větší, aby se tam stín vešel
    CudaImg outputImg;
    outputImg.m_size.x = img1.m_size.x + shadow_offset.x;
    outputImg.m_size.y = img1.m_size.y + shadow_offset.y;

    cudaMalloc(&outputImg.m_p_uchar4, sizeof(uchar4) * outputImg.m_size.x * outputImg.m_size.y);
    cudaMemset(outputImg.m_p_uchar4, 0, sizeof(uchar4) * outputImg.m_size.x * outputImg.m_size.y); // průhledné pozadí

    int l_block_size = 32;
    dim3 l_blocks((img1.m_size.x + l_block_size - 1) / l_block_size,
                  (img1.m_size.y + l_block_size - 1) / l_block_size);
    dim3 l_threads(l_block_size, l_block_size);

    kernel_add_shadow<<<l_blocks, l_threads>>>(img1, outputImg, shadow_offset);
    if ((l_cerr = cudaGetLastError()) != cudaSuccess)
        printf("CUDA Error [%d] - '%s'\n", __LINE__, cudaGetErrorString(l_cerr));
    cudaDeviceSynchronize();

    // Vlož výsledek (větší obraz) na pozici
    kernel_insertimage<<<l_blocks, l_threads>>>(background, outputImg, t_position);
    if ((l_cerr = cudaGetLastError()) != cudaSuccess)
        printf("CUDA Error [%d] - '%s'\n", __LINE__, cudaGetErrorString(l_cerr));
    cudaDeviceSynchronize();

    cudaFree(outputImg.m_p_uchar4);
}

__global__ void kernel_antialias(CudaImg input_img, CudaImg output_img) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= input_img.m_size.x || y >= input_img.m_size.y)
        return;

    int idx = y * input_img.m_size.x + x;
    uchar4 pixel = input_img.m_p_uchar4[idx];

    // Pokud je pixel zcela průhledný, jen ho zkopíruj
    if (pixel.w == 0) {
        output_img.m_p_uchar4[idx] = pixel;
        return;
    }

    // Zjisti, kolik sousedů je průhledných
    int transparent_neighbors = 0;
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            if (dx == 0 && dy == 0) continue;

            int nx = x + dx;
            int ny = y + dy;

            if (nx >= 0 && ny >= 0 && nx < input_img.m_size.x && ny < input_img.m_size.y) {
                uchar4 neighbor = input_img.m_p_uchar4[ny * input_img.m_size.x + nx];
                if (neighbor.w == 0) transparent_neighbors++;
            }
        }
    }

    // Pokud má alespoň jeden soused průhlednost, upravíme alfa kanál
    uchar4 result_pixel = pixel;
    if (transparent_neighbors > 0) {
        // Oslabíme alfa kanál o 10% za každý průhledný soused (max 8)
        float alpha_factor = 1.0f - 0.1f * transparent_neighbors;
        alpha_factor = max(alpha_factor, 0.0f); // nesmí být záporné
        result_pixel.w = static_cast<unsigned char>(pixel.w * alpha_factor);
    }

    output_img.m_p_uchar4[idx] = result_pixel;
}
void cu_antialias(CudaImg input_img, CudaImg background, int2 t_position) {
    cudaError_t l_cerr;

    int l_block_size = 32;
    dim3 l_blocks((input_img.m_size.x + l_block_size - 1) / l_block_size,
                  (input_img.m_size.y + l_block_size - 1) / l_block_size);
    dim3 l_threads(l_block_size, l_block_size);

    // Alokace GPU paměti pro výsledek
    CudaImg outputImg;
    outputImg.m_size = input_img.m_size;
    cudaMalloc(&outputImg.m_p_uchar4, sizeof(uchar4) * outputImg.m_size.x * outputImg.m_size.y);

    kernel_antialias<<<l_blocks, l_threads>>>(input_img, outputImg);
    if ((l_cerr = cudaGetLastError()) != cudaSuccess)
        printf("CUDA Error [%d] - '%s'\n", __LINE__, cudaGetErrorString(l_cerr));
    cudaDeviceSynchronize();

        // Vlož výsledek (větší obraz) na pozici
        kernel_insertimage<<<l_blocks, l_threads>>>(background, outputImg, t_position);
        if ((l_cerr = cudaGetLastError()) != cudaSuccess)
            printf("CUDA Error [%d] - '%s'\n", __LINE__, cudaGetErrorString(l_cerr));
        cudaDeviceSynchronize();
    
        cudaFree(outputImg.m_p_uchar4);
}
