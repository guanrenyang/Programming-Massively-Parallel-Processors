//
// Created by guanrenyang on 2022/3/19.
//
#include <iostream>
#include <cuda.h>

#define HEIGHT_M 2048
#define WIDTH_M_HEIGHT_N 4096
#define WIDTH_N 2048
#define TILE_WIDTH 32
// the official solution
// but I think it is incorrect and not understandable
__global__ void matrixMul_kernel_official(float *Pd, float *Md, float *Nd)
{
    unsigned int bx = blockIdx.x;
    unsigned int by = blockIdx.y;
    unsigned int tx = threadIdx.x;
    unsigned int ty = threadIdx.y;

    __shared__ float Ms[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Ns[TILE_WIDTH][TILE_WIDTH*2];

    // Index of the first sub-matrix of M processed by the block
    int mBegin = WIDTH_M_HEIGHT_N * TILE_WIDTH * by;

    // Index of the last sub-matrix of M processed by the block
    int mEnd = mBegin + WIDTH_M_HEIGHT_N -1; //`mEnd` is in the same row as mBegin

    // Step size used to iterate through the sub-matrices of M
    int mStep = TILE_WIDTH;

    // Index of the first sub-matrix of N processed by the block
    int nBegin = TILE_WIDTH * bx; // nBegin is in the first row of N

    // Step size used to iterate through the sub-matrices of N
    int nStep = TILE_WIDTH * WIDTH_N; // nBegin and nStep traverse the column

    /*
     * The calculations of `mBegin` and `nBegin`, `mStep` and `nStep` are
     * different, because the array in C++ is stored in row-major manner.
     */

    // Psub is used to iterate through the sub-matrices of N
    // that is computed by the thread
    float Psub1 = 0.0f;
    float Psub2 = 0.0f;

    // loop over all the sub-matrices of A and B
    // required to compute the block sub-matrix
    for(int m=mBegin, n=nBegin; m<=mEnd; m+=mStep, n+=nStep)
    {
        // Load the matrices from device memory to shared memory;
        // each thread loads one element of each matrix
        Ms[ty][tx] = Md[m+WIDTH_M_HEIGHT_N*ty+tx];
        Ns[ty][tx] = Nd[n+WIDTH_N*ty+tx];
        Ns[ty][tx+blockDim.x] = Nd[n+WIDTH_N*ty+tx+blockDim.x];
        __syncthreads();
        // Multiply the two matrices together
        // each thread computes two elements
        for(int k=0;k<TILE_WIDTH;++k)
        {
            Psub1 += Ms[ty][k] * Ns[k][tx];
            Psub2 += Ms[ty][k] * Ns[k][tx+blockDim.x];
        }

        // synchronize to make sure that computation is done before
        // threads  load new sub-matrices in the next iteration
        __syncthreads();
    }
    // Write the block sub-matrix to device memory;
    // each thread writes one element
    int p = WIDTH_N * TILE_WIDTH * by + TILE_WIDTH * bx;
    p += WIDTH_N * ty + tx;
    Pd[p] = Psub1;
    Pd[p+blockDim.x] = Psub2;

}
// kernel 1
// a block calculate its tile and the tile half-width right to it
__global__ void matrixMul_Kernel_1(float *Pd, float *Md, float *Nd)
{
    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH*2];

    int tx = threadIdx.x, bx = blockIdx.x;
    int ty = threadIdx.y, by = blockIdx.y;

    // identify row and column of the d_P element to work on in a tile
    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;

    if ( Row < HEIGHT_M && Col < WIDTH_N ) {

        float pValue_1 = 0;
        float pValue_2 = 0;

        // Loop over the d_M and d_N tiles required to compute the d_P element
        for (int ph = 0; ph < WIDTH_M_HEIGHT_N / TILE_WIDTH; ph++) {

            // Collaborative loading of d_M and d_N tiles n to the shared memory
            Mds[ty][tx] = Md[Row * WIDTH_M_HEIGHT_N + ph * TILE_WIDTH + tx];
            Nds[ty][tx] = Nd[(ph * TILE_WIDTH + ty) * WIDTH_N + Col];
            Nds[ty][tx+TILE_WIDTH] = Nd[(ph * TILE_WIDTH + ty) * WIDTH_N + Col + (WIDTH_N/2)];

            // printf("ph = %d; block[%d,%d]; thread[%d,%d] --> Nds[0][%d] = %2.2f\n", ph, blockIdx.y, blockIdx.x, threadIdx.y, threadIdx.x, tx, Nds[0][tx]);
            __syncthreads();


            for(int k = 0; k < TILE_WIDTH; k++){
                //printf("ph = %d; block[%d,%d]; thread[%d,%d] --> %2.2f + %2.2f * %2.2f\n", ph, blockIdx.y, blockIdx.x, threadIdx.y, threadIdx.x, pValue, Mds[ty][k], Nds[k][tx]);
                pValue_1 += Mds[ty][k] * Nds[k][tx];
                pValue_2 += Mds[ty][k] * Nds[k][tx+TILE_WIDTH];

            }
            __syncthreads();
        }
        Pd[Row*WIDTH_N+Col] = pValue_1;
        Pd[Row*WIDTH_N+Col + (WIDTH_N/2)] = pValue_2;
    }
}

/*
 * The performance of kernel 2 is the same of kernel 1
 * my intuition of kernel 2 is that a block calculates two adjacent tiles,
 * while would reduce the global bandwidth usage.
 * However, kernel 2 makes no difference of kernel 1
 * because fetching number from one block and fetching from the other are
 * separated into two sentences. The numbers of two tiles could not be
 * fetched by one global memory access.
 */
// kernel 2
// a block calculate its tile and the tile adjacent right to it
// which mean the block with blockIdx.y==0 with calculate blockIdx.y==0 and 1.
__global__ void matrixMul_Kernel_2(float *Pd, float *Md, float *Nd)
{
    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH*2];

    int tx = threadIdx.x, bx = blockIdx.x*2; // different from kernel 1
    int ty = threadIdx.y, by = blockIdx.y;

    // identify row and column of the d_P element to work on in a tile
    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;

    if ( Row < HEIGHT_M && Col < WIDTH_N ) {

        float pValue_1 = 0;
        float pValue_2 = 0;

        // Loop over the d_M and d_N tiles required to compute the d_P element
        for (int ph = 0; ph < WIDTH_M_HEIGHT_N / TILE_WIDTH; ph++) {

            // Collaborative loading of d_M and d_N tiles n to the shared memory
            Mds[ty][tx] = Md[Row * WIDTH_M_HEIGHT_N + ph * TILE_WIDTH + tx];
            Nds[ty][tx] = Nd[(ph * TILE_WIDTH + ty) * WIDTH_N + Col];
            Nds[ty][tx+TILE_WIDTH] = Nd[(ph * TILE_WIDTH + ty) * WIDTH_N + Col + TILE_WIDTH]; // different from kernel 1

            // printf("ph = %d; block[%d,%d]; thread[%d,%d] --> Nds[0][%d] = %2.2f\n", ph, blockIdx.y, blockIdx.x, threadIdx.y, threadIdx.x, tx, Nds[0][tx]);
            __syncthreads();


            for(int k = 0; k < TILE_WIDTH; k++){
                //printf("ph = %d; block[%d,%d]; thread[%d,%d] --> %2.2f + %2.2f * %2.2f\n", ph, blockIdx.y, blockIdx.x, threadIdx.y, threadIdx.x, pValue, Mds[ty][k], Nds[k][tx]);
                pValue_1 += Mds[ty][k] * Nds[k][tx];
                pValue_2 += Mds[ty][k] * Nds[k][tx+TILE_WIDTH];

            }
            __syncthreads();
        }
        Pd[Row*WIDTH_N+Col] = pValue_1;
        Pd[Row*WIDTH_N+Col + TILE_WIDTH] = pValue_2; // different from kernel 1
    }
}

int main() {
    int nM = WIDTH_M_HEIGHT_N * HEIGHT_M;
    int nN = WIDTH_N * WIDTH_M_HEIGHT_N;
    int nP = WIDTH_N * HEIGHT_M;

    int sizeM = nM*sizeof(float );
    int sizeN = nN*sizeof(float );
    int sizeP = nP*sizeof(float );

    float *M, *N, *P;
    float *dM, *dN, *dP;

    M = new float [nM];
    N = new float [nN];
    P = new float [nP];
    for (int i=0;i<nM;i++){
        M[i]=1;
    }
    for (int i=0;i<nN;i++){
        N[i]=1;
    }

    cudaMalloc(&dM, sizeM);
    cudaMalloc(&dN, sizeN);
    cudaMalloc(&dP, sizeP);

    cudaMemcpy(dM, M, sizeM, cudaMemcpyHostToDevice);
    cudaMemcpy(dN, N, sizeN, cudaMemcpyHostToDevice);

    dim3 dimGrid(WIDTH_N/TILE_WIDTH/2, HEIGHT_M/TILE_WIDTH);
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);

    // modify here, to change a kernel function
    matrixMul_Kernel_1<<<dimGrid, dimBlock>>>(dP, dM, dN);

    cudaMemcpy(P, dP, sizeP, cudaMemcpyDeviceToHost);

    for(int i=0;i<nP;i++){
        std::cout<<P[i]<<' ';
        if((i+1)%WIDTH_N==0)
            std::cout<<std::endl;
    }
    return 0;
}