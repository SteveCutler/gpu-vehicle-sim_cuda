#include <iostream>
#include "environment.hpp"
#include "vehicleBatch.hpp"
#include "dynamics.hpp"
#include "controller.hpp"
#include <cuda_runtime.h>
#include "vehicleTypes.hpp"
#include <stdexcept>
#include "CudaSimulation.hpp"



__global__ void updateAoS(VehicleState* vehicles, std::size_t N, environment env, float dt, std::size_t step){

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if(i >= N){
        return;
    }

    //create controller object
    Controller controller;

    //create dynamics updater
    Dynamics dynamics;

    //allocating variables outside hot loop
    VehicleState vs;
    Action action;
    VehicleState newState;
    
    //create vehicleState object with thread vehicle
    vs = vehicles[i];

    //pass to controller 
    action = controller.steer_controller(vs);

    //calculate new state
    newState = dynamics.step_update(vs, action, env, dt, step);

    //update old state
    vehicles[i] = newState;

    return;
}

void runCudaSimulationAoS(vehicleBatch& vehicles, environment& env, std::size_t N, float dt, std::size_t steps){

    std::size_t curr_step = 0;

    const std::size_t floatBytes = N * sizeof(VehicleState);

    //create VehicleState vector

    std::vector<VehicleState>states(N);

    for(int i = 0; i < N; i++){
        states[i] = vehicles.load(i);
    }
    
    //creating devices for vehicle batch data, allocating memory and copying data over
    VehicleState* d_vehicles = nullptr;
    cudaMalloc(&d_vehicles, floatBytes);
    cudaMemcpy(d_vehicles, states.data(), floatBytes, cudaMemcpyHostToDevice);
    

    //check for allocation errors
    cudaError_t error = cudaGetLastError();

    if (error != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(error));
    }

    //threading setup
    std::size_t threads = 256;
    std::size_t blocks = (N + threads-1)/threads;

        //run sim for steps amount of steps
    while(curr_step < steps){

        
        //launch cuda kernel
        updateAoS<<<blocks,threads>>>(d_vehicles, N, env, dt, curr_step);
        
        //check for kernel launch errors
        cudaError_t error = cudaGetLastError();

        if (error != cudaSuccess) {
            throw std::runtime_error(cudaGetErrorString(error));
        }

 
        curr_step++;
    }

    //error check

    error = cudaDeviceSynchronize();

    //Handle failure.
    if (error != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(error));
    }


    //copy data back over

    cudaMemcpy(states.data(), d_vehicles, floatBytes, cudaMemcpyDeviceToHost);
    cudaFree(d_vehicles);

    //set vehicleBatch values from computed data
    for(int i = 0; i < N; i++){
        vehicles.set(i,states[i]);
    }

    return;

};