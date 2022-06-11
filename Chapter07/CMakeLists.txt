cmake_minimum_required(VERSION 3.21)
project(Chapter7 CUDA)

set(CMAKE_CUDA_STANDARD 11)

add_executable(2DConvAdvanced 2DConvAdvanced.cu)
add_executable(2DConvBasic 2DConvBasic.cu)

set_target_properties(2DConvAdvanced PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(2DConvBasic PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
