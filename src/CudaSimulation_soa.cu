#include <iostream>
#include "environment.hpp"
#include "vehicleBatch.hpp"
#include "dynamics.hpp"
#include "controller.hpp"
#include <cuda_runtime.h>
#include "vehicleTypes.hpp"
#include <stdexcept>
#include "CudaSimulation.hpp"



__global__ void update(float* vs_x, float* vs_y, float* vs_vx, float* vs_vy, float* vs_heading, float* vs_turnRate, const float* vs_goalx, const float* vs_goaly, std::size_t N, environment env, float dt){

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
    vs = {
        vs_x[i],
        vs_y[i],
        vs_vx[i],
        vs_vy[i],
        vs_heading[i],
        vs_turnRate[i],
        vs_goalx[i],
        vs_goaly[i]
    };


    //pass to controller 
    action = controller.steer_controller(vs);

    //calculate new state
    newState = dynamics.step_update(vs, action, env, dt);

    //update old state
        vs_x[i] = newState.x;
        vs_y[i] = newState.y;
        vs_vx[i] = newState.vx;
        vs_vy[i] = newState.vy;
        vs_heading[i] = newState.heading;
        vs_turnRate[i] = newState.turnRate;

    return;
}

void runCudaSimulationSoA(vehicleBatch& vehicles, environment& env, std::size_t N, float dt, std::size_t steps){

    std::size_t curr_step = 0;

    const std::size_t floatBytes = N * sizeof(float);

    //create GPU environment struct
    
    //creating devices for vehicle batch data, allocating memory and copying data over
    float* vs_x = nullptr;
    cudaMalloc(&vs_x, floatBytes);
    cudaMemcpy(vs_x, vehicles.m_x.data(), floatBytes, cudaMemcpyHostToDevice);
    
    float* vs_y = nullptr;
    cudaMalloc(&vs_y, floatBytes);
    cudaMemcpy(vs_y, vehicles.m_y.data(), floatBytes, cudaMemcpyHostToDevice);
    
    float* vs_vx = nullptr;
    cudaMalloc(&vs_vx, floatBytes);
    cudaMemcpy(vs_vx, vehicles.m_vx.data(), floatBytes, cudaMemcpyHostToDevice);
    float* vs_vy = nullptr;
    cudaMalloc(&vs_vy, floatBytes);
    cudaMemcpy(vs_vy, vehicles.m_vy.data(), floatBytes, cudaMemcpyHostToDevice);
    
    float* vs_heading = nullptr;
    cudaMalloc(&vs_heading, floatBytes);
    cudaMemcpy(vs_heading, vehicles.m_heading.data(), floatBytes, cudaMemcpyHostToDevice);
    
    float* vs_turnRate = nullptr;
    cudaMalloc(&vs_turnRate, floatBytes);
    cudaMemcpy(vs_turnRate, vehicles.m_turnRate.data(), floatBytes, cudaMemcpyHostToDevice);
    
    // read only
    float* vs_goalx = nullptr;
    cudaMalloc(&vs_goalx, floatBytes);
    cudaMemcpy(vs_goalx, vehicles.m_goalx.data(), floatBytes, cudaMemcpyHostToDevice);

    float* vs_goaly = nullptr;
    cudaMalloc(&vs_goaly, floatBytes);
    cudaMemcpy(vs_goaly, vehicles.m_goaly.data(), floatBytes, cudaMemcpyHostToDevice);

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
        update<<<blocks,threads>>>(vs_x, vs_y, vs_vx, vs_vy, vs_heading, vs_turnRate, vs_goalx, vs_goaly, N, env, dt);
        
        //check for kernel launch errors
        cudaError_t error = cudaGetLastError();

        if (error != cudaSuccess) {
            throw std::runtime_error(cudaGetErrorString(error));
        }

        //increment step counter
        env.updateTime(dt);
        curr_step++;
    }

    //error check

    error = cudaDeviceSynchronize();

    //Handle failure.
    if (error != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(error));
    }


    //copy data back over

    cudaMemcpy(vehicles.m_x.data(), vs_x, floatBytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(vehicles.m_y.data(), vs_y, floatBytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(vehicles.m_vx.data(), vs_vx, floatBytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(vehicles.m_vy.data(), vs_vy, floatBytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(vehicles.m_heading.data(), vs_heading, floatBytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(vehicles.m_turnRate.data(), vs_turnRate, floatBytes, cudaMemcpyDeviceToHost);

    //free allocated memory
    float* allocations[] = {
        vs_x, vs_y, vs_vx, vs_vy, vs_heading, vs_turnRate, vs_goalx, vs_goaly
    };

    for (float* ptr : allocations){
        cudaFree(ptr);
    }

    return;

};