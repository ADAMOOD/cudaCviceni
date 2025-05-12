#include "cuda_img.h"
#include "cuda_runtime.h"
#include <cstring> // pro memcpy, pokud bude potřeba
#include <vector>
class FishAnim {
    CudaImg m_bg_cuda_img;      // původní pozadí
    CudaImg m_res_cuda_img;     // buffer pro výstupní obrázek
    CudaImg *fishes = nullptr;  // pole ryb
    int m_fish_count = 0;
    bool m_initialized = false;

public:
    void start(CudaImg bg_img, CudaImg *fishes_arr, int fish_count);
    void next(int2 *positions);
    void get_result(CudaImg &dest);
    void stop();
};
