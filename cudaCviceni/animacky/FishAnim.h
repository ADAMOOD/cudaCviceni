#include "cuda_img.h"
class FishAnim {
    CudaImg m_bg_cuda_img;
    CudaImg m_res_cuda_img;
    CudaImg *fishes; //pointer na pole ryb
    bool m_initialized = false;

public:
    void start(CudaImg bg_img, CudaImg *fishes);
    void next(CudaImg res_img, int2 *positions);
    void stop();
};