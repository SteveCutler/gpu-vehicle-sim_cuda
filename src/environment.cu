#include "environment.hpp"
#include <cuda_runtime.h>

environment::environment(std::size_t w, std::size_t h): 
m_width(w), 
m_height(h),
elapsed(0.0f)
{

    //wind vel field  zero across the board for first implementation
    //in later implementation create curl noise field

    
}

__device__ float2 environment::getDisturbance(float x, float y, float time) const{
    
    // bounds check
    if(x < 0.0f || x >= static_cast<float>(m_width) || 
    y < 0.0f || y >= static_cast<float>(m_height)){
        return make_float2(0.0f, 0.0f);
    }

    //cyclical wind field parameters
    constexpr float amplitude = 5.5f;
    constexpr float wavelength = 1000.f;
    constexpr float timeMult = 10.f;
    constexpr float twopi = 6.283185f;

    //time mult factor
    float x_evolve = x - timeMult * time;
    float y_evolve = y - timeMult*3.231 * time;

    //creating gust strength based on cycle point
    float x_angle = (x_evolve / wavelength) * twopi;
    float x_gust = sinf(x_angle);

    float y_angle = (y_evolve / wavelength) * twopi;
    float y_gust = cosf(y_angle);

    float windX = amplitude * x_gust;
    float windY = amplitude * y_gust;

    return {windX, windY};

};

void environment::updateTime(float dt){
    //track elapsed time for random velocity generation later
    elapsed += dt;
}

