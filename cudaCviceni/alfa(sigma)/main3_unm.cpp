
#include <stdio.h>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

#include "uni_mem_allocator.h"
#include "cuda_img.h"

// Image size
#define SIZEX 432 // Width of image
#define SIZEY 321 // Height of image
// Block size for threads
#define BLOCKX 40 // block width
#define BLOCKY 25 // block height

void resize_image_if_needed(cv::Mat &input_img, int max_width, int max_height);

void cu_insertimage(CudaImg t_big_cuda_img, CudaImg t_small_cuda_img, int2 t_position);
void cu_insertimageRoundTransparency( CudaImg t_big_cuda_pic, CudaImg t_small_cuda_pic, int2 t_position, int lowerBound);
void cu_mergeByAlpha(CudaImg img1, CudaImg img2, CudaImg background, int2 t_position);
void cu_addShadow(CudaImg img1, CudaImg background, int2 t_position);
void cu_antialias(CudaImg input_img, CudaImg background, int2 t_position);


int main()
{
    // Uniform Memory allocator for Mat
    UniformAllocator allocator;
    cv::Mat::setDefaultAllocator(&allocator);
    bool contin = true;
    while (contin)
    {
        // Creation of empty image.
        // Image is stored line by line.
        cv::Mat CVGradient(SIZEY, SIZEX, CV_8UC3); // kazdy kanal barvy ma 8 bitu a 3 kanaly - rgb

        // Image filling by color gradient blue-green-red
        for (int y = 0; y < CVGradient.rows; y++)
            for (int x = 0; x < CVGradient.cols; x++)
            {
                int l_dx = x - CVGradient.cols / 2;

                unsigned char l_grad = 255 * abs(l_dx) / (CVGradient.cols / 2);
                unsigned char l_inv_grad = 255 - l_grad;

                // left or right half of gradient
                uchar3 l_bgr = (l_dx < 0) ? (uchar3){l_grad, l_inv_grad, 0} : (uchar3){0, l_inv_grad, l_grad};

                // put pixel into image
                cv::Vec3b l_v3bgr(l_bgr.x, l_bgr.y, l_bgr.z);
                CVGradient.at<cv::Vec3b>(y, x) = l_v3bgr;
                // also possible: cv_img.at<uchar3>( y, x ) = bgr;
            }

        CudaImg cudaGradient;
        cudaGradient.m_size.x = CVGradient.size().width;  // equivalent to cv_img.cols
        cudaGradient.m_size.y = CVGradient.size().height; // equivalent to cv_img.rows
        cudaGradient.m_p_uchar3 = (uchar3 *)CVGradient.data;

        char file[255];
        bool valid = false;
        cv::Mat CV_Input;
        while (!valid)
        {
            printf("give me a PNG file\n");
            scanf("%s", file);
            // Load image
            CV_Input = cv::imread(file, cv::IMREAD_UNCHANGED); // CV_LOAD_IMAGE_UNCHANGED );

            if (!CV_Input.data)
            {
                printf("Unable to read file '%s'\n", file);
                continue;
            }
            else if (CV_Input.channels() != 4)
            {
                printf("Image does not contain alpha channel!\n");
                continue;
            }
            valid = true;
        }
        resize_image_if_needed(CV_Input, SIZEX / 2, SIZEY / 2);// funkce ktera zmensi obrazek pokud je vetsi nez x a y

        // insert loaded image
        CudaImg cudaInput;
        cudaInput.m_size.x = CV_Input.cols;
        cudaInput.m_size.y = CV_Input.rows;
        cudaInput.m_p_uchar4 = (uchar4 *)CV_Input.data;

        // Function calling from .cu file
        uint2 l_block_size = {BLOCKX, BLOCKY};
        int whatTodo;
        printf("Select an option:\n");
        printf("1 - insert image onto gradient\n");
        printf("2 - insert rounded png\n");
        printf("3 - merge by sigma\n");
        printf("4 - anti aliasing\n");
        printf("5 - add shadow\n");
        printf("0 - Exit\n");
        scanf("%d", &whatTodo);
        switch (whatTodo)
        {
        case 1:
        {

            cu_insertimage(cudaGradient, cudaInput, {(int)cudaGradient.m_size.x / 2, (int)cudaGradient.m_size.y / 2});
            cv::imshow("input inserted on a gradient", CVGradient);
            cv::waitKey(0);
            break;
        }
        case 2:
        {
            int lower=0;
            printf("give me a lower bound \n");
            scanf("%d", &lower);
            cu_insertimageRoundTransparency(cudaGradient, cudaInput, {(int)cudaGradient.m_size.x/2 , (int)cudaGradient.m_size.y/2},lower);
            cv::imshow("input inserted on a gradient with rounded alpha", CVGradient);
            cv::waitKey(0);
            break;
        }
        case 3:
        {
            char file2[255];
            bool valid = false;
            cv::Mat CV_Input2;
            while (!valid)
            {
                printf("give me a PNG file\n");
                scanf("%s", file2);
                // Load image
                CV_Input2 = cv::imread(file2, cv::IMREAD_UNCHANGED); // CV_LOAD_IMAGE_UNCHANGED );
    
                if (!CV_Input2.data)
                {
                    printf("Unable to read file '%s'\n", file2);
                    continue;
                }
                else if (CV_Input2.channels() != 4)
                {
                    printf("Image does not contain alpha channel!\n");
                    continue;
                }
                valid = true;
            }
            resize_image_if_needed(CV_Input2, CV_Input2.cols, CV_Input2.rows);// funkce ktera zmensi obrazek pokud je vetsi nez x a y
            // insert loaded image
            CudaImg cudaInput2;
            cudaInput2.m_size.x = CV_Input2.cols;
            cudaInput2.m_size.y = CV_Input2.rows;
            cudaInput2.m_p_uchar4 = (uchar4 *)CV_Input2.data;

            cu_mergeByAlpha(cudaInput, cudaInput2, cudaGradient, {(int)cudaGradient.m_size.x / 2, (int)cudaGradient.m_size.y / 2});
            cv::imshow("Merged inputs", CVGradient);
            cv::waitKey(0);
        }
        case 4:
        {
            cu_antialias(cudaInput,cudaGradient,{(int)cudaGradient.m_size.x / 2, (int)cudaGradient.m_size.y / 2});
            cv::imshow("Anti-aliased", CVGradient);
            cv::waitKey(0);
            break;
        }
        case 5:
        {

            cu_addShadow(cudaInput,cudaGradient,  {(int)cudaGradient.m_size.x/2 , (int)cudaGradient.m_size.y/2});
            cv::imshow("input with shadow", CVGradient);
            cv::waitKey(0);
            break;
        }
        case 0:
        {
            printf("Exiting...\n");
            contin = false;
            break;
        }

        default:
            break;
        }
    }
}
void resize_image_if_needed(cv::Mat &input_img, int max_width, int max_height)
{
    // Zkontrolujeme, zda obrázek přesahuje maximální povolené rozměry
    if (input_img.cols > max_width || input_img.rows > max_height)
    {
        // Vypočteme nový rozměr pro zmenšení obrázku
        double scale_x = (double)max_width / input_img.cols;
        double scale_y = (double)max_height / input_img.rows;
        double scale = std::min(scale_x, scale_y);

        // Změníme velikost obrázku
        cv::resize(input_img, input_img, cv::Size(), scale, scale, cv::INTER_LINEAR);
    }
}
