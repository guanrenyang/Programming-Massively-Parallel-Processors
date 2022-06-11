#include <iostream>
#include <cuda.h>

const int originalDimBlock = 512;
const int n = 2 * originalDimBlock; // number of elements of X
// simple sum reduction kernel, FIGURE 5.13
__global__ void simpleSumReduction_Kernel(const float *X, float *res, const int size){
    __shared__ float partialSum[n/2];

    // modification 2-2
    // compute pairwise summation while fetching
    // they can't be in one line because of global memory bandwidth
    partialSum[threadIdx.x] = X[(2*blockDim.x) * blockIdx.x+threadIdx.x];
    // the pre-adding works in the `improved` version because it reduces thread divergence
    // it couldn't shorten execution time in the `simple` version due to the same reason
    // but it has the potential to reduce the usage of shared memory.
    partialSum[threadIdx.x] += X[(2*blockDim.x) * blockIdx.x+threadIdx.x+blockDim.x];

    // modification 2-1: `threadIdx.x`->`2*threadIdx.x`
    unsigned int t = threadIdx.x<<1;
    for (unsigned int stride = 1; stride<blockDim.x<<1;stride<<=1){
        __syncthreads();
        if(t%(2*stride)==0)
            partialSum[t]+=partialSum[t+stride];
    }
    __syncthreads();
    res[blockIdx.x] = partialSum[0];
}

// a kernel with fewer thread divergence, FIGURE 5.15
__global__ void improvedSumReduction_Kernel(const float *X, float *res, const int size){
    __shared__ float partialSum[n/2];

    // modification 1-2
    // compute pairwise summation while fetching
    // they can't be in one line because of global memory bandwidth
    partialSum[threadIdx.x] = X[(2*blockDim.x) * blockIdx.x+threadIdx.x];
    partialSum[threadIdx.x] += X[(2*blockDim.x) * blockIdx.x+threadIdx.x+blockDim.x];

    unsigned int t = threadIdx.x;

    // modification 1-1: `blockDim.x/2` -> `blockDim.x`
    // so that the block size is half of what it was originally
    // this is the critical modification
    for (unsigned int stride = blockDim.x>>1; stride>=1;  stride = stride>>1){
        __syncthreads();
        if(t<stride)
            partialSum[t] += partialSum[t+stride];
    }
    __syncthreads();
    res[blockIdx.x] = partialSum[0];
}
int main() {
    float *X;
    int size_X = n * sizeof(float);

    X = new float [n];
    float *cuda_X;
    float * res;
    float * res_X;


    for (int i=0;i<n;i++)
        X[i]=1.0;

    cudaMalloc(&cuda_X, size_X);
    cudaMemcpy(cuda_X, X, size_X, cudaMemcpyHostToDevice);


    int dimBlock = originalDimBlock;
    int dimGrid = (1024+dimBlock-1)/dimBlock;

    res = new float [dimGrid];

    cudaMalloc(&res_X, dimGrid*sizeof(float ));

    // modification 2-1
    // just half blocks are needed
    simpleSumReduction_Kernel<<<dimGrid, dimBlock/2>>>(cuda_X, res_X , 1024);

    // modification 1-3
    // just half blocks are needed
    improvedSumReduction_Kernel<<<dimGrid, dimBlock/2>>>(cuda_X, res_X , 1024);

    cudaMemcpy(res, res_X,dimGrid*sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(cuda_X);
    cudaFree(res_X);

    std::cout<<res[0]+res[1]<<std::endl;

    return 0;

}
