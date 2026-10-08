#pragma once
#include <vector>
#include <cuda_runtime.h>

class environment
{
public: 

std::size_t m_width;
std::size_t m_height;
float elapsed;

private:
float* velFieldx;
float* velFieldy;

public:
environment(std::size_t w, std::size_t h);

__device__ float2 getDisturbance(std::size_t x, std::size_t y) const;

void updateTime(float dt);

};

