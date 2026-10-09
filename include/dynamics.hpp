#pragma once
#include "vehicleTypes.hpp"
#include "environment.hpp"
#include <cuda_runtime.h>


class Dynamics
{

public:
    __device__ Dynamics() = default;

   __device__ VehicleState step_update(const VehicleState& vs, const Action& action, const environment& env, const float dt, std::size_t step);
};

