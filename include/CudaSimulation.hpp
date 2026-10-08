#pragma once
#include <cstddef>
#include "vehicleBatch.hpp"
#include "environment.hpp"

void runCudaSimulation(
    vehicleBatch& vehicles,
    const environment& env,
    std::size_t N,
    float dt,
    std::size_t steps);