//
// Created by guanrenyang on 2022/4/3.
//
#define SECTION_SIZE 16
# include <iostream>
# include <bits/stdc++.h>
__global__ void Kogge_Stone_exclusive_scan_kernel(float *X, float *Y, int InputSize) {
    // `InputSize` must be <= `blockDim.x`(1D array)

    __shared__ float XY[SECTION_SIZE];

    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if(i<InputSize&&threadIdx.x!=0) {
        XY[threadIdx.x] = X[i-1];
    } else {
        XY[threadIdx.x] = 0;
    }

    // the code below performs iterative scan on XY
    for (unsigned int stride=1; stride<blockDim.x; stride*=2) {
        __syncthreads();
        if (threadIdx.x>=stride)
            XY[threadIdx.x] += XY[threadIdx.x-stride];
    }
    Y[i] = XY[threadIdx.x];
}
int main(){
    float * X_h = new float [SECTION_SIZE];
    float * Y_h = new float [SECTION_SIZE];
    // initialize X_h
    for (int i=0;i<SECTION_SIZE;i++) {
        X_h[i]= i + 1;
    }

    float *X_d;
    float *Y_d;
    cudaMalloc(&X_d, SECTION_SIZE*sizeof(float));
    cudaMalloc(&Y_d, SECTION_SIZE*sizeof(float));
    cudaMemcpy(X_d, X_h, SECTION_SIZE*sizeof(float), cudaMemcpyHostToDevice);

    dim3 dimBlock(SECTION_SIZE);
    dim3 dimGrid(1);
    Kogge_Stone_exclusive_scan_kernel<<<dimGrid, dimBlock>>>(X_d, Y_d, SECTION_SIZE);

    cudaMemcpy(Y_h, Y_d, SECTION_SIZE*sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(X_d);
    cudaFree(Y_d);

    
    // test
    float trueSum = 0;
    for (int i = 0; i < SECTION_SIZE; i++)
    {
        if(i!=0)
            trueSum+=X_h[i-1];
        std::cout<<i<<":("<<trueSum<<","<<Y_h[i]<<")\n";
    }

    
    return 0;
}
