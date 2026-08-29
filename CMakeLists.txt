cmake_minimum_required(VERSION 3.28)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 75 CACHE STRING "CUDA architectures to build")
endif()

project(cuda_gpu_performance LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

add_executable(reduction_benchmark
    src/reduction/reduction.cu
    src/common/cuda_utils.cu
)

target_include_directories(reduction_benchmark PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

add_executable(transpose_benchmark
    src/transpose/transpose.cu
    src/common/cuda_utils.cu
)

target_include_directories(transpose_benchmark PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

add_executable(convolution_benchmark
    src/convolution/convolution.cu
    src/common/cuda_utils.cu
)

target_include_directories(convolution_benchmark PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
