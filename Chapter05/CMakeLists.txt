cmake_minimum_required(VERSION 3.16)
project(Chapter5 CUDA)

set(CMAKE_CUDA_STANDARD 11)

add_executable(sumReduction sumReduction.cu)
add_executable(matrixMultiplication matrixMultiplication.cu)

set_target_properties(sumReduction PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(matrixMultiplication PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
