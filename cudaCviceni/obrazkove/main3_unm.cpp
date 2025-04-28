// ***********************************************************************
//
// Demo program for education in subject
// Computer Architectures and Parallel Systems.
// Petr Olivka, dep. of Computer Science, FEI, VSB-TU Ostrava, 2020/11
// email:petr.olivka@vsb.cz
//
// Example of CUDA Technology Usage without unified memory.
//
// Image creation and its modification using CUDA.
// Image manipulation is performed by OpenCV library.
//
// ***********************************************************************

#include <stdio.h>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

#include "uni_mem_allocator.h"
#include "cuda_img.h"

// Prototype of function in .cu file
void cu_run_animation(CudaImg t_pic, uint2 t_block_size);
void cuSetBrightness(CudaImg t_cuda_img, uint2 t_block_size, int percentage);
void cuRotate90(CudaImg t_cuda_img, cv::Mat& output_mat, uint2 t_block_size);
void cuMirror(CudaImg t_src_img, CudaImg t_dst_img, uint2 t_block_size);

// Image size
#define SIZEX 432 // Width of image
#define SIZEY 321 // Height of image
// Block size for threads
#define BLOCKX 40 // block width
#define BLOCKY 25 // block height

int main()
{
    // Uniform Memory allocator for Mat
    UniformAllocator allocator;
    cv::Mat::setDefaultAllocator(&allocator);

    while (true)
    {
        // Creation of empty image.
        // Image is stored line by line.
        cv::Mat l_cv_img(SIZEY, SIZEX, CV_8UC3); // kazdy kanal barvy ma 8 bitu a 3 kanaly - rgb

        // Image filling by color gradient blue-green-red
        for (int y = 0; y < l_cv_img.rows; y++)
            for (int x = 0; x < l_cv_img.cols; x++)
            {
                int l_dx = x - l_cv_img.cols / 2;

                unsigned char l_grad = 255 * abs(l_dx) / (l_cv_img.cols / 2);
                unsigned char l_inv_grad = 255 - l_grad;

                // left or right half of gradient
                uchar3 l_bgr = (l_dx < 0) ? (uchar3){l_grad, l_inv_grad, 0} : (uchar3){0, l_inv_grad, l_grad};

                // put pixel into image
                cv::Vec3b l_v3bgr(l_bgr.x, l_bgr.y, l_bgr.z);
                l_cv_img.at<cv::Vec3b>(y, x) = l_v3bgr;
                // also possible: cv_img.at<uchar3>( y, x ) = bgr;
            }

        CudaImg l_cuda_img;
        l_cuda_img.m_size.x = l_cv_img.size().width;  // equivalent to cv_img.cols
        l_cuda_img.m_size.y = l_cv_img.size().height; // equivalent to cv_img.rows
        l_cuda_img.m_p_uchar3 = (uchar3 *)l_cv_img.data;

        // Show image before modification
        cv::imshow("B-G-R Gradient", l_cv_img);

        // Function calling from .cu file
        uint2 l_block_size = {BLOCKX, BLOCKY};
        int whatTodo;
        printf("Give me a number of app\n");
        scanf("%d", &whatTodo);
        switch (whatTodo)
        {
        case 1:
        {
            cu_run_animation(l_cuda_img, l_block_size);
                // Show modified image
        cv::imshow("B-G-R Gradient & Color Rotation", l_cv_img);
            break;
        }
        case 2:
        {
            int percentage;
            printf("Give me a percentage \n");
            scanf("%d", &percentage);
            cuSetBrightness(l_cuda_img, l_block_size, percentage);
                    // Show modified image
        cv::imshow("B-G-R Gradient & Color Rotation", l_cv_img);
            break;
        }
        case 3:
        {
            // Připravíme výstupní obrázek s UniformAllocator
            cv::Mat rotated_img;
            rotated_img.allocator = new UniformAllocator();
            rotated_img.create(l_cuda_img.m_size.x, l_cuda_img.m_size.y, CV_8UC3); // pozor! výměna šířky a výšky

            // Bloková velikost (třeba 16x16)
            uint2 block_size = {16, 16};

            // Zavoláme rotaci
            cuRotate90(l_cuda_img, rotated_img, block_size);
            cv::imshow("ROTATED PICUS", rotated_img);
            break;
        }
        case 4:
        {
            cv::Mat balerina = cv::imread("balerina.jpeg", cv::IMREAD_COLOR);

            if (!balerina.data)
            {
                printf("Unable to read file '%s'\n", "balerina.jpeg");
                return 1;
            }
            
            // vytvoříme kopii obrázku
            cv::Mat mirrored_balerina = balerina.clone();
            
            CudaImg culerina_src;
            culerina_src.m_size.x = balerina.size().width;
            culerina_src.m_size.y = balerina.size().height;
            culerina_src.m_p_uchar3 = (uchar3 *)balerina.data;
            
            CudaImg culerina_dst;
            culerina_dst.m_size.x = mirrored_balerina.size().width;
            culerina_dst.m_size.y = mirrored_balerina.size().height;
            culerina_dst.m_p_uchar3 = (uchar3 *)mirrored_balerina.data;
            
            // velikost bloku
            uint2 l_block_size = {16, 16};
            
            cv::imshow("BALERIONA CAPUCHINA", balerina);
            
            // zavoláme mirror
            cuMirror(culerina_src, culerina_dst, l_block_size);
            
            cv::imshow("ANIHCUPAC ANOIRELAB", mirrored_balerina);
            
            break;
        }

        default:
            break;
        }
        cv::waitKey(0);
    }
}
