#pragma once
#include <vector>
#include <cuda_runtime.h>

class environment
{
public: 

std::size_t m_width;
std::size_t m_height;


private:
float* velFieldx;
float* velFieldy;

public:
environment(std::size_t w, std::size_t h);

__device__ float2 getDisturbance(float x, float y, float time) const;



};

