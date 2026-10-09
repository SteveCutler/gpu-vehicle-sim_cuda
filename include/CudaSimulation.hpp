#pragma once
#include <cstddef>
#include "vehicleBatch.hpp"
#include "environment.hpp"

void runCudaSimulationSoA(
    vehicleBatch& vehicles,
    environment& env,
    std::size_t N,
    float dt,
    std::size_t steps
);
void runCudaSimulationSoA_batched(
    vehicleBatch& vehicles,
    environment& env,
    std::size_t N,
    float dt,
    std::size_t steps
);
void runCudaSimulationAoS(
    vehicleBatch& vehicles,
    environment& env,
    std::size_t N,
    float dt,
    std::size_t steps
);