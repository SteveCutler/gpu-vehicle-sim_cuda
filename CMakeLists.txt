cmake_minimum_required(VERSION 3.18)
project(gpu_vehicle_sim LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)


option(ENABLE_RENDERER "Build the SFML renderer" OFF)

add_executable(vehicle_sim
    src/main.cpp
    src/vehicleBatch.cpp
    src/environment.cu
    src/dynamics.cu
    src/controller.cu
    src/CudaSimulation.cu

)

target_include_directories(vehicle_sim PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_features(vehicle_sim PRIVATE cxx_std_17)

target_link_libraries(vehicle_sim PRIVATE CUDA::cudart)

set_target_properties(vehicle_sim PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    CUDA_SEPARABLE_COMPILATION ON
)

if(ENABLE_RENDERER)
    find_package(SFML 3 REQUIRED COMPONENTS Graphics Window System)
    target_sources(vehicle_sim PRIVATE src/renderer.cpp)
    target_link_libraries(vehicle_sim PRIVATE SFML::Graphics SFML::Window SFML::System)
    target_compile_definitions(vehicle_sim PRIVATE ENABLE_RENDERER)
endif()