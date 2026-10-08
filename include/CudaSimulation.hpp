#pragma once
#include <cstddef>
#include "vehicleBatch.hpp"
#include "environment.hpp"

void runCudaSimulation_soa(
    vehicleBatch& vehicles,
    environment& env,
    std::size_t N,
    float dt,
    std::size_t steps
);
void runCudaSimulation_aos(
    vehicleBatch& vehicles,
    environment& env,
    std::size_t N,
    float dt,
    std::size_t steps
);