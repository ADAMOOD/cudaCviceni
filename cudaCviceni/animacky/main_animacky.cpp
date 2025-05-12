
#include <stdio.h>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

#include "uni_mem_allocator.h"
#include "cuda_img.h"
#include <cstdlib> // rand, srand
#include <ctime>   // time
#include "animation.h"
#include "FishAnim.h"

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
    printf("1 - aquarium 1 ryba\n");
    printf("0 - Exit\n");
    scanf("%d", &choice);
    switch (choice)
    {
    case 1:
    {
        Animation animation;
        cv::Mat CV_fish1;
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

        int x1 = rand() % (SIZEX - cudaFish1.m_size.x);
        int y1 = rand() % (SIZEY - cudaFish1.m_size.y);

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

            // kontrola hran
            if (x1 < 0 || x1 > SIZEX - cudaFish1.m_size.x)
                dx1 = -dx1;
            if (y1 < 0 || y1 > SIZEY - cudaFish1.m_size.y)
                dy1 = -dy1;

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

    case 2:
    {
        cv::Mat CV_fish1 = cv::imread("fish1.webp", cv::IMREAD_UNCHANGED);
        cv::Mat CV_fish2 = cv::imread("fish2.png", cv::IMREAD_UNCHANGED);
        cv::Mat CV_fish3 = cv::imread("fish3.webp", cv::IMREAD_UNCHANGED);

        if (!CV_fish1.data || !CV_fish2.data || !CV_fish3.data)
        {
            printf("Unable to read one or more fish images.\n");
            return -1;
        }
        if (CV_fish1.channels() != 4 || CV_fish2.channels() != 4 || CV_fish3.channels() != 4)
        {
            printf("One or more fish images do not contain an alpha channel.\n");
            return -1;
        }

        resize_image_if_needed(CV_fish1, SIZEX / 5, SIZEY / 5);
        resize_image_if_needed(CV_fish2, SIZEX / 5, SIZEY / 5);
        resize_image_if_needed(CV_fish3, SIZEX / 5, SIZEY / 5);

        CudaImg cudaFish1 = {{CV_fish1.cols, CV_fish1.rows}, (uchar4 *)CV_fish1.data};
        CudaImg cudaFish2 = {{CV_fish2.cols, CV_fish2.rows}, (uchar4 *)CV_fish2.data};
        CudaImg cudaFish3 = {{CV_fish3.cols, CV_fish3.rows}, (uchar4 *)CV_fish3.data};

        int x[3], y[3], dx[3], dy[3];
        CudaImg fishes_array[3] = {cudaFish1, cudaFish2, cudaFish3};

        for (int i = 0; i < 3; ++i)
        {
            x[i] = rand() % (SIZEX - fishes_array[i].m_size.x);
            y[i] = rand() % (SIZEY - fishes_array[i].m_size.y);
            dx[i] = (rand() % 3) - 1;
            dy[i] = (rand() % 3) - 1;
        }

        FishAnim anim;
        anim.start(cudaBackground, fishes_array, 3);

        while (true)
        {
            // aktualizuj pozice všech ryb
            for (int i = 0; i < 3; ++i)
            {
                x[i] += dx[i];
                y[i] += dy[i];

                if (x[i] < 0 || x[i] > SIZEX - fishes_array[i].m_size.x)
                    dx[i] = -dx[i];
                if (y[i] < 0 || y[i] > SIZEY - fishes_array[i].m_size.y)
                    dy[i] = -dy[i];
            }

            int2 positions[3] = {make_int2(x[0], y[0]), make_int2(x[1], y[1]), make_int2(x[2], y[2])};
                       anim.next(positions); // vykreslí ryby do vnitřního bufferu

            CudaImg frame = {{SIZEX, SIZEY}, (uchar4 *)cudaBackground.m_p_uchar3}; // použijeme pozici, kam se to má zkopírovat
            anim.get_result(frame); // výstupní frame z animace

            cudaDeviceSynchronize();

            cv::Mat CV_Background_Updated(CV_Background_Original.size(), CV_Background_Original.type());
            cudaMemcpy(CV_Background_Updated.data, cudaBackground.m_p_uchar3,
                       sizeof(uchar3) * SIZEX * SIZEY, cudaMemcpyDeviceToHost);

            cv::imshow("Akvarko", CV_Background_Updated);
            if (cv::waitKey(30) >= 0)
                break;
        }

        anim.stop();
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
