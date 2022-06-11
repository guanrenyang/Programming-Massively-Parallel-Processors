cmake_minimum_required(VERSION 3.21)
project(Chapter8)

set(CMAKE_CUDA_STANDARD 14)

add_executable(KoggeStoneExclusiveScan
        KoggeStoneExclusiveScan.cu)
set_target_properties(KoggeStoneExclusiveScan PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)