#include <cstddef>
#include "environment.hpp"
#include "vehicleBatch.hpp"
#include "controller.hpp"
#include "dynamics.hpp"
#include <chrono>
#include <iostream>
#include <string>
#include "CudaSimulation.hpp"
#include "stateExport.hpp"


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
    int type;

    //take in input variables

    //set batch size
    if (argNum > 1) N = std::stoi(argVals[1]);
    //set step size
    if (argNum > 2) steps = std::stoi(argVals[2]);
    //set AoS vs SoA version
    if (argNum > 3) type = std::stoi(argVals[3]);
    

    std::cout << "Vehicles: " << N << "\nSteps: " << steps << '\n';

    //initialize vehicleState data
    vehicleBatch vehicles(N, width, height);

    //init environment
    environment env(width, height);

    //start wallclock timer for performance measurement
    const auto start = Clock::now();

    //launch Cuda Kernel operations
    try{
        if(type == 1){
            std::cout << "SoA" << std::endl;
            runCudaSimulationSoA(vehicles, env, N, dt, steps);
        }
        else if(type == 2){
            std::cout << "SoA Batched" << std::endl;
            runCudaSimulationSoA_batched(vehicles, env, N, dt, steps);
        }
        else{
            std::cout << "AoS" << std::endl;
            runCudaSimulationAoS(vehicles, env, N, dt, steps);
        }
    }
    catch (const std::exception& error) {
        std::cerr << "Simulation failed: " << error.what() << '\n';
        return 1;
    }

    //end clock
    const auto stop = Clock::now();

    //calculate elapsed time
    const double seconds = 
        std::chrono::duration<double>(stop - start).count();

    //multiply steps by vehicles to get total updates
    const double updates = static_cast<double>(N) * steps;

    std::cout << "Vehicles: " << N << '\n';
    std::cout << "Completed steps: " << steps << '\n';
    std::cout << "Execution time: " << seconds << " seconds\n";
    std::cout << "Vehicle updates/sec: " << updates / seconds << '\n';

    //file name for data harness
    if (argNum > 4) {
        exportStates(vehicles, argVals[4]);
    }


    return 0;
}