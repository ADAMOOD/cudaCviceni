
#include <stdio.h>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

#include "uni_mem_allocator.h"
#include "cuda_img.h"
#include <cstdlib> // rand, srand
#include <ctime>   // time
#include "animation.h"

// Block size for threads
#define BLOCKX 40 // block width
#define BLOCKY 25 // block height

void resize_image_if_needed(cv::Mat &input_img, int max_width, int max_height);

void cu_insertimage(CudaImg t_big_cuda_img, CudaImg t_small_cuda_img, int2 t_position);

int main()
{

    srand(time(NULL));
    // Uniform Memory allocator for Mat
    UniformAllocator allocator;
    cv::Mat::setDefaultAllocator(&allocator);

    cv::Mat CV_Background_Original;
    CV_Background_Original = cv::imread("aquarium.jpg", cv::IMREAD_UNCHANGED); // CV_LOAD_IMAGE_UNCHANGED );

    if (!CV_Background_Original.data)
    {
        printf("Unable to read file '%s'\n", "aquarium.png");
        return -1;
    }
    CudaImg cudaBackground;
    cudaBackground.m_size.x = CV_Background_Original.size().width;  // equivalent to cv_img.cols
    cudaBackground.m_size.y = CV_Background_Original.size().height; // equivalent to cv_img.rows
    cudaBackground.m_p_uchar3 = (uchar3 *)CV_Background_Original.data;

    int SIZEX = CV_Background_Original.size().width;
    int SIZEY = CV_Background_Original.size().height;
    int choice;
    printf("Select an option:\n");
    printf("1 - aquarium\n");
    printf("0 - Exit\n");
    scanf("%d", &choice);
    switch (choice)
    {
    case 1:
    {
        Animation animation;
        cv::Mat CV_fish1, CV_fish2, CV_fish3;
        CV_fish1 = cv::imread("fish1.webp", cv::IMREAD_UNCHANGED); // CV_LOAD_IMAGE_UNCHANGED );
        if (!CV_fish1.data)
        {
            printf("Unable to read file '%s'\n", "fish1.webp");
            return -1;
        }
        else if (CV_fish1.channels() != 4)
        {
            printf("Image does not contain alpha channel!\n");
            return -1;
        }
        resize_image_if_needed(CV_fish1, SIZEX / 5, SIZEY / 5);
        CudaImg cudaFish1;
        cudaFish1.m_size.x = CV_fish1.size().width;
        cudaFish1.m_size.y = CV_fish1.size().height;
        cudaFish1.m_p_uchar4 = (uchar4 *)CV_fish1.data;
        /*
                CV_fish2 = cv::imread("fish2.png", cv::IMREAD_UNCHANGED); // CV_LOAD_IMAGE_UNCHANGED );
                if (!CV_fish2.data)
                {
                    printf("Unable to read file '%s'\n", "fish2.png");
                    return -1;
                }
                else if (CV_fish2.channels() != 4)
                {
                    printf("Image does not contain alpha channel!\n");
                    return -1;
                }
                resize_image_if_needed(CV_fish2, SIZEX / 5, SIZEY / 5);
                CudaImg cudaFish2;
                cudaFish2.m_size.x = CV_fish2.size().width;  // equivalent to cv_img.cols
                cudaFish2.m_size.y = CV_fish2.size().height; // equivalent to cv_img.rows
                cudaFish2.m_p_uchar4 = (uchar4 *)CV_fish2.data;

                CV_fish3 = cv::imread("fish3.webp", cv::IMREAD_UNCHANGED); // CV_LOAD_IMAGE_UNCHANGED );
                if (!CV_fish3.data)
                {
                    printf("Unable to read file '%s'\n", "fish3.webp");
                    return -1;
                }
                else if (CV_fish3.channels() != 4)
                {
                    printf("Image does not contain alpha channel!\n");
                    return -1;
                }
                resize_image_if_needed(CV_fish3, SIZEX / 5, SIZEY / 5);
                CudaImg cudaFish3;
                cudaFish3.m_size.x = CV_fish3.size().width;  // equivalent to cv_img.cols
                cudaFish3.m_size.y = CV_fish3.size().height; // equivalent to cv_img.rows
                cudaFish3.m_p_uchar4 = (uchar4 *)CV_fish3.data;
        */
        int x1 = rand() % (SIZEX - cudaFish1.m_size.x);
        int y1 = rand() % (SIZEY - cudaFish1.m_size.y);
        /* int x2 = rand() % (SIZEX - cudaFish2.m_size.x);
         int y2 = rand() % (SIZEY - cudaFish2.m_size.y);
         int x3 = rand() % (SIZEX - cudaFish3.m_size.x);
         int y3 = rand() % (SIZEY - cudaFish3.m_size.y);*/

        // náhodné směry pohybu
        int dx1 = (rand() % 3) - 1, dy1 = (rand() % 3) - 1;
        int dx2 = (rand() % 3) - 1, dy2 = (rand() % 3) - 1;
        int dx3 = (rand() % 3) - 1, dy3 = (rand() % 3) - 1;
        animation.start(cudaBackground, cudaFish1);
        while (true)
        {
            // aktualizuj pozice
            x1 += dx1;
            y1 += dy1;
            /* x2 += dx2;
             y2 += dy2;
             x3 += dx3;
             y3 += dy3;*/

            // kontrola hran
            if (x1 < 0 || x1 > SIZEX - cudaFish1.m_size.x)
                dx1 = -dx1;
            if (y1 < 0 || y1 > SIZEY - cudaFish1.m_size.y)
                dy1 = -dy1;
            /*  if (x2 < 0 || x2 > SIZEX - cudaFish2.m_size.x)
                  dx2 = -dx2;
              if (y2 < 0 || y2 > SIZEY - cudaFish2.m_size.y)
                  dy2 = -dy2;
              if (x3 < 0 || x3 > SIZEX - cudaFish3.m_size.x)
                  dx3 = -dx3;
              if (y3 < 0 || y3 > SIZEY - cudaFish3.m_size.y)
                  dy3 = -dy3;*/

            // vykresli ryby na CUDA obrázek
            // cu_insertimage(cudaBackground, cudaFish1, make_int2(x1, y1));
            // cu_insertimage(cudaBackground, cudaFish2, make_int2(x2, y2));
            // cu_insertimage(cudaBackground, cudaFish3, make_int2(x3, y3));
            animation.next(cudaBackground, make_int2(x1, y1));
            // **Přidej synchronizaci před zobrazením**
            cudaDeviceSynchronize();

            // Vytvoření nové Matice pro zobrazení
            cv::Mat CV_Background_Updated(CV_Background_Original.size(), CV_Background_Original.type());

            // Kopírování dat z GPU do CPU
            cudaMemcpy(CV_Background_Updated.data, cudaBackground.m_p_uchar3,
                       sizeof(uchar3) * SIZEX * SIZEY, cudaMemcpyDeviceToHost);

            // Zobrazení aktualizovaného obrázku
            cv::imshow("Akvarko", CV_Background_Updated);
            if (cv::waitKey(30) >= 0)
                break;
        }
        animation.stop();
        break;
    }

    default:
        break;
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
