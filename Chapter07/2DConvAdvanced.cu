//
// Created by guanrenyang on 2022/3/28.
//
# include <iostream>
# include <cuda.h>
# include <bits/stdc++.h>
# define MAX_MASK_WIDTH 4
# define MAX_MASK_HEIGHT 4
__constant__ float M[MAX_MASK_WIDTH*MAX_MASK_HEIGHT]; // continuous in memory.

# define TILE_SIZE 16

__global__ void convolution_2D_const_memory_kernel(float *N, float *P, size_t N_pitch, size_t P_pitch, int Mask_Width, int Mask_Height, int Width, int Height) {

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float PValue = 0;
    int N_start_row = row-Mask_Height/2;
    int N_start_col = col-Mask_Width/2;
    for (int i=0; i<Mask_Height; i++) {
        float *r = (float *)((char*)N + (N_start_row+i)*N_pitch);
        for (int j=0;j<Mask_Width;j++) {
            if ((N_start_row+i)>=0&&(N_start_row+i)<Height&&(N_start_col+j)>=0&&(N_start_col+j)<Width)
                PValue += r[(N_start_col + j)] * M[i * Mask_Width + j];
        }
    }

    if (row<Height&&col<Width) // When the block size doesn't match the N matrix, superfluous threads couldn't be executed.
    {
        float *r = (float *)((char *)P + row * P_pitch);
        r[col] = PValue;
    }
}
__global__ void convolution_2D_shared_memory_kernel(float *N, float *P, size_t N_pitch, size_t P_pitch, int Mask_Width, int Mask_Height, int Width, int Height) {


    __shared__ float N_ds[TILE_SIZE + MAX_MASK_HEIGHT - 1][TILE_SIZE + MAX_MASK_WIDTH - 1];
    int Half_Mask_Width = Mask_Width / 2;
    int Half_Mask_Height = Mask_Height / 2;

    int center_row = blockIdx.y * blockDim.y + threadIdx.y;
    int center_col = blockIdx.x * blockDim.x + threadIdx.x;

    int halo_index_last_row = (blockIdx.y - 1) * blockDim.y + threadIdx.y;
    int halo_index_last_col = (blockIdx.x - 1) * blockDim.x + threadIdx.x;
    int halo_index_next_row = (blockIdx.y + 1) * blockDim.y + threadIdx.y;
    int halo_index_next_col = (blockIdx.x + 1) * blockDim.x + threadIdx.x;

    /* copy center and halo cells to Shared Memory */
    // left up corner
    if (threadIdx.x >= blockDim.x - Half_Mask_Width && threadIdx.y >= blockDim.y - Half_Mask_Height)
        N_ds[threadIdx.y - (blockDim.y - Half_Mask_Height)][threadIdx.x - (blockDim.x - Half_Mask_Width)]
                = (halo_index_last_row < 0 || halo_index_last_col < 0) ? 0 : ((float *)((char*)N + halo_index_last_row*N_pitch))[halo_index_last_col];

    // right down corner
    if (threadIdx.x < Half_Mask_Width && threadIdx.y < Half_Mask_Height)
        N_ds[threadIdx.y + blockDim.y + Half_Mask_Height][threadIdx.x + blockDim.x + Half_Mask_Width]
                = (halo_index_next_row >= Height || halo_index_next_col >= Width) ? 0 :((float *)((char*)N + halo_index_next_row*N_pitch))[halo_index_next_col];

    // left down corner
    if (threadIdx.x >= blockDim.x - Half_Mask_Width && threadIdx.y < Half_Mask_Height)
        N_ds[threadIdx.y + blockDim.y + Half_Mask_Height][threadIdx.x - (blockDim.x - Half_Mask_Width)]
                = (halo_index_last_row >= Height || halo_index_last_col < 0) ? 0 :((float *)((char*)N + halo_index_next_row*N_pitch))[halo_index_last_col];
    // right up corner
    if (threadIdx.x < Half_Mask_Width && threadIdx.y >= blockDim.y - Half_Mask_Height)
        N_ds[threadIdx.y - (blockDim.y - Half_Mask_Height)][threadIdx.x + blockDim.x + Half_Mask_Width]
                = (halo_index_last_row < 0 || halo_index_next_col >= Width) ? 0 :((float *)((char*)N + halo_index_last_row*N_pitch))[halo_index_next_col];

    // left edge
    if (threadIdx.x >= blockDim.x - Half_Mask_Width)
        N_ds[threadIdx.y + Half_Mask_Height][threadIdx.x - (blockDim.x - Half_Mask_Width)]
                = (halo_index_last_col < 0) ? 0 : ((float *)((char*)N + center_row*N_pitch))[halo_index_last_col];
    // right edge
    if (threadIdx.x < Half_Mask_Width)
        N_ds[threadIdx.y + Half_Mask_Height][threadIdx.x + blockDim.x + Half_Mask_Width]
                = (halo_index_next_col >= Width) ? 0 :((float *)((char*)N + center_row*N_pitch))[halo_index_next_col];

    // up edge
    if (threadIdx.y >= blockDim.y - Half_Mask_Height) {
        N_ds[threadIdx.y - (blockDim.y - Half_Mask_Height)][threadIdx.x + Half_Mask_Width]
                = (halo_index_last_row < 0) ? 0 : ((float *)((char*)N + halo_index_last_row*N_pitch))[center_col];
    }
    // down edge
    if (threadIdx.y < Half_Mask_Height)
        N_ds[threadIdx.y + blockDim.y + Half_Mask_Height][threadIdx.x + Half_Mask_Width]
                = (halo_index_next_row >= Height) ? 0 : ((float *)((char*)N + halo_index_next_row*N_pitch))[center_col];

    // center cells
    N_ds[threadIdx.y+Half_Mask_Height][threadIdx.x+Half_Mask_Width] = (center_row<0||center_row>=Height||center_col<0||center_col>=Width) ? 0 : ((float *)((char*)N + center_row*N_pitch))[center_col];

    __syncthreads();

    float PValue = 0;
    int N_start_row = threadIdx.y;
    int N_start_col = threadIdx.x;
    for (int i = 0; i < Mask_Height; i++) {
        for (int j = 0; j < Mask_Width; j++) {
                PValue += N_ds[N_start_row + i][N_start_col + j] * M[i * Mask_Width + j];
        }
    }
    if (center_row>=0&&center_row<Height&&center_col>=0&&center_col<Width)
        ((float *)((char*)P + center_row*N_pitch))[center_col] = PValue;
}
int main() {

    // Define Sizes
    int Width = 16, Height=16;
    int Mask_Width = 3, Mask_Height = 3;

    int size_M = Mask_Width*Mask_Height*sizeof(float);

    // Host Pointers
    float *N_h = new float [Width * Height];
    float *M_h = new float [Mask_Width * Mask_Height];
    float *P_h = new float [Width * Height];

    // Initialize M and N on host
    for (int i=0;i<Width*Height;++i)
        N_h[i]=1;
    for (int i=0;i<Mask_Width*Mask_Height;++i)
        M_h[i]=1;

    // Host Pointers
    float *N_d;
    float *P_d;

    // Pitches of N and P
    size_t N_pitch_d;// pitch is the number of Bytes of a row in memory.
                     // On GTX1650 and CUDA 11.4, the minimum pitch is 512
    size_t P_pitch_d;

    // Allocate 2D array on Device
    cudaMallocPitch(&N_d, &N_pitch_d, Width*sizeof(float ), Height);
    cudaMallocPitch(&P_d, &P_pitch_d, Width*sizeof(float ), Height);

    // Copy N from Host to Device
    cudaMemcpy2D(N_d, N_pitch_d, N_h, Width*sizeof(float ), Width*sizeof(float ), Height, cudaMemcpyHostToDevice);

    // Copy M from Host to Constant Memory
    cudaMemcpyToSymbol(M, M_h, size_M, 0, cudaMemcpyHostToDevice);

    dim3 dimGrid((Width + TILE_SIZE - 1) / TILE_SIZE, (Height + TILE_SIZE - 1) / TILE_SIZE);
    dim3 dimBlock(TILE_SIZE, TILE_SIZE);
    /* modify here, to change a kernel function */
    convolution_2D_shared_memory_kernel<<<dimGrid, dimBlock>>>(N_d, P_d, N_pitch_d, P_pitch_d, Mask_Width, Mask_Height, Width, Height);

    /* At first I set TILE_SIZE=1024. This means that the number of threads per block is 1024**2, which is not supported.*/

    // Copy P from Device to Host
    cudaMemcpy2D(P_h, Width*sizeof(float ), P_d, P_pitch_d, Width*sizeof(float ), Height, cudaMemcpyDeviceToHost);

    // Display P
    for(int i=0;i<Height;++i){
        for(int j=0;j<Width;++j){
            std::cout<<P_h[i*Width+j]<<' ';
        }
        std::cout<<std::endl;
    }

    return 0;
}