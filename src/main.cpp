#include <cstddef>
#include "environment.hpp"
#include "vehicleBatch.hpp"
#include "controller.hpp"
#include "dynamics.hpp"
#include <chrono>
#include <iostream>
#include "CudaSimulation.cu"
#include <string>


using Clock = std::chrono::steady_clock;

int main(int argNum, char* argVals[]){


    std::cout << "starting up..." << std::endl;
    //master variables
    constexpr std::size_t width = 500;
    constexpr std::size_t height = 500;
    constexpr float dt = 0.02f;

    //default values
    std::size_t N = 1000;
    std::size_t steps = 1000;

    //take in input variables
    if (argNum > 1) N = std::stoi(argVals[1]);
    if (argNum > 2) steps = std::stoi(argVals[2]);

    std::cout << "Vehicles: " << N << "\nSteps: " << steps << '\n';

    //initialize vehicleState data
    vehicleBatch vehicles(N, width, height);

    //start wallclock timer for performance measurement
    const auto start = Clock::now();

    //launch Cuda Kernel operations
    try{
        runCudaSimulation(vehicles, N, dt, steps);
    }
    catch(const std::exception& error){
        throw std::runtime_error(cudaGetErrorString(error));
        return 1;
    }

    //end clock
    const auto stop = Clock::now();

    //calculate elapsed time
    const double seconds = 
        std::chrono::duration<double>(stop - start).count();

    //multiply steps by vehicles to get total updates
    const double updates = static_cast<double>(N) * curr_step;

    std::cout << "Vehicles: " << N << '\n';
    std::cout << "Completed steps: " << curr_step << '\n';
    std::cout << "Execution time: " << seconds << " seconds\n";
    std::cout << "Vehicle updates/sec: " << updates / seconds << '\n';



    return 0;
}