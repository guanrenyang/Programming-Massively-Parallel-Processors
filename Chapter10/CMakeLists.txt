cmake_minimum_required(VERSION 3.22)
project(Chapter10 CUDA)

set(CMAKE_CUDA_STANDARD 11)

add_executable(Chapter10 JDS-ELL.cu)

set_target_properties(Chapter10 PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
