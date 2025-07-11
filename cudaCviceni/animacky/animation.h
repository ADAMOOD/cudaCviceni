
// Simple animation.
//
// ***********************************************************************

#include "cuda_img.h"

class Animation
{
public:
    CudaImg m_bg_cuda_img, m_ins_cuda_img, m_res_cuda_img;
    int m_initialized;

    Animation() : m_initialized( 0 ) {}

    void start( CudaImg t_bg_pic, CudaImg t_ins_pic );

    void next( CudaImg t_res_pic, int2 t_position );

    void stop();

};
