//
// Created by guanrenyang on 2022/4/4.
//
#define ARRAY_SIZE 5000
#define SECTION_SIZE 64 // SECTION_SIZE must be a power of 2
#define NUM_SECTION (ARRAY_SIZE+SECTION_SIZE-1)/SECTION_SIZE
#define BLOCK_SIZE  (SECTION_SIZE+1)/2
# include <iostream>
# include <bits/stdc++.h>
__global__ void step_1_Brent_Kung_scan_kernel(float *X, float *Y, int InputSize){
    __shared__ float XY[SECTION_SIZE];
    int i=2*blockIdx.x*blockDim.x+threadIdx.x;
    if (i<InputSize)
        XY[threadIdx.x] = X[i];
    if (i+blockDim.x<InputSize)
        XY[threadIdx.x+blockDim.x] = X[i+blockDim.x];
    // phase 1
    for (unsigned int stride = 1; stride <= blockDim.x; stride*=2 ) {
        __syncthreads();
        int index = (threadIdx.x+1)*2*stride-1;
        if (index<SECTION_SIZE)
            XY[index]+=XY[index-stride];
    }

    // phase 2
    for (int stride = SECTION_SIZE/4; stride>0; stride/=2) {
        __syncthreads();
        int index = (threadIdx.x+1)*2*stride-1;
        if (index+stride<SECTION_SIZE) {
            XY[index+stride] += XY[index];
        }
    }

    __syncthreads();
    if (i<InputSize)
        Y[i] = XY[threadIdx.x];
    if (i+blockDim.x<InputSize){
        Y[i+blockDim.x] = XY[threadIdx.x+blockDim.x];
    }
}

// fetch elements from Y to S, and perform a Kogge Stone scan on it
__global__ void step_2_Kogge_Stone_scan_kernel(float *Y, int InputSize) {

    __shared__ float Sds[NUM_SECTION];
    int index = threadIdx.x * SECTION_SIZE + SECTION_SIZE - 1;
    if (index<InputSize)
        Sds[threadIdx.x] = Y[index];
    for (unsigned int stride = 1; stride < blockDim.x; stride*=2) {
        __syncthreads();
        if (threadIdx.x>=stride)
            Sds[threadIdx.x]+=Sds[threadIdx.x-stride];
    }
    Y[index] = Sds[threadIdx.x];
}
__global__ void step_3_distributes(float *Y, int InputSize) {
    __shared__ float preSum;
    int preSumIndex = (blockIdx.x-1)*SECTION_SIZE+SECTION_SIZE-1;
    if (threadIdx.x==0){
        preSum=0;
        if (preSumIndex>=0&&preSumIndex<InputSize)
            preSum =Y[preSumIndex];
    }
    __syncthreads();

    int i = 2*blockIdx.x*blockDim.x+threadIdx.x;
    if (i<InputSize)
        Y[i]+=preSum;
    if (i+blockDim.x<InputSize&&i+blockDim.x!=preSumIndex+SECTION_SIZE)
        Y[i+blockDim.x]+=preSum;
}
int main(){
    float * X_h = new float [ARRAY_SIZE];
    float * Y_h = new float [ARRAY_SIZE];
    // initialize X_h
    for (int i=0;i<ARRAY_SIZE;i++) {
        X_h[i]= i + 1;
    }

    float *X_d;
    float *Y_d;

    cudaMalloc(&X_d, ARRAY_SIZE*sizeof(float));
    cudaMalloc(&Y_d, ARRAY_SIZE*sizeof(float));

    cudaMemcpy(X_d, X_h, ARRAY_SIZE*sizeof(float), cudaMemcpyHostToDevice);

    // Step 1
    dim3 dimBlock_1(BLOCK_SIZE);
    dim3 dimGrid_1(NUM_SECTION);
    step_1_Brent_Kung_scan_kernel<<<dimGrid_1, dimBlock_1>>>(X_d, Y_d, ARRAY_SIZE);

    // Step 2
    dim3 dimBlock_2(NUM_SECTION); // number of sections must <= 1024
    dim3 dimGrid_2(1);
    step_2_Kogge_Stone_scan_kernel<<<dimGrid_2, dimBlock_2>>>(Y_d, ARRAY_SIZE);

    // Step 3
    dim3 dimBlock_3(BLOCK_SIZE);
    dim3 dimGrid_3(NUM_SECTION);
    step_3_distributes<<<dimGrid_3, dimBlock_3>>>(Y_d, ARRAY_SIZE);

    cudaMemcpy(Y_h, Y_d, ARRAY_SIZE*sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(X_d);
    cudaFree(Y_d);

    // test for Y
    float trueSum = 0;
    for (int i = 0; i < ARRAY_SIZE; i++)
    {
        trueSum+=X_h[i];
        std::cout<<i<<":("<<trueSum<<","<<Y_h[i]<<")\n";
    }


    return 0;
}

