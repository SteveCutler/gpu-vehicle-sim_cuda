#include "environment.hpp"
#include <cuda_runtime.h>

environment::environment(std::size_t w, std::size_t h): 
m_width(w), 
m_height(h),
elapsed(0.0f),
velFieldx(nullptr), 
velFieldy(nullptr){

    //wind vel field  zero across the board for first implementation
    //in later implementation create curl noise field

    //allocating velfields as CUDA buffers
    const std::size_t bytes = m_width * m_height * sizeof(float);
    
    //introduce more robust error handling
    cudaMalloc(&velFieldx, bytes);
    cudaMalloc(&velFieldy, bytes);

    cudaMemset(velFieldx, 0.0, bytes);
    cudaMemset(velFieldy, 0.0, bytes);
    
}

__device__ float2 environment::getDisturbance(std::size_t x, std::size_t y) const{
    {


    // bounds check
    if(x < 0 || x >= m_width || y < 0 || y >= m_height){
        return make_float2(0.0f, 0.0f);
    }

    std::size_t pos = y * m_width + x;
    //update velocity fields with dt
    return make_float2(velFieldx[pos], velFieldy[pos]);

}

void updateTime(float dt){
    //track elapsed time for random velocity generation later
    elapsed += dt;
}

}