//
// Created by guanrenyang on 2022/3/27.
//
# include <iostream>
# include <cuda.h>
# include <bits/stdc++.h>
# define MAX_MASK_WIDTH 10
//__constant__ float M[MAX_MASK_WIDTH];

# define TILE_WIDTH 32

__global__ void convolution_2D_basic_kernel(float *N, float *M, float *P, int Mask_Width, int Mask_Height, int Width, int Height) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float PValue = 0;
    int N_start_row = row-Mask_Height/2;
    int N_start_col = col-Mask_Width/2;
    for (int i=0; i<Mask_Height; i++) {
        for (int j=0;j<Mask_Width;j++) {
            if ((N_start_row+i)>=0&&(N_start_row+i)<Height&&(N_start_col+j)>=0&&(N_start_col+j)<Width)
                PValue += N[(N_start_row + i) * Width + (N_start_col + j)] * M[i * Mask_Width + j];
        }
    }
    if (row<Height&&col<Width) // When the block size doesn't match the N matrix, superfluous threads couldn't be executed.
        P[row*Width+col] = PValue;
}
__global__ void convolution_2D_shared_memory_kernel(float *N, float *P, size_t N_pitch, size_t P_pitch, int Mask_Width, int Mask_Height, int Width, int Height) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ float N_ds[TILE_SIZE + MAX_MASK_HEIGHT - 1][TILE_SIZE + MAX_MASK_WIDTH - 1];
    int Half_Mask_Width = Mask_Width / 2;
    int Half_Mask_Height = Mask_Height / 2;

    int center_row = blockIdx.y * blockDim.y + threadIdx.y;
    int center_col = blockIdx.x * blockDim.x + threadIdx.x;

    int halo_index_left_row = (blockIdx.y - 1) * blockDim.y + threadIdx.y;
    int halo_index_left_col = (blockIdx.x - 1) * blockDim.x + threadIdx.x;
    int halo_index_right_row = (blockIdx.y + 1) * blockDim.y + threadIdx.y;
    int halo_index_right_col = (blockIdx.x + 1) * blockDim.x + threadIdx.x;

    /* copy center and halo cells to Shared Memory */
    // left up corner
    if (threadIdx.x >= blockDim.x - Half_Mask_Width && threadIdx.y >= blockDim.y - Half_Mask_Height)
        N_ds[threadIdx.y - (blockDim.y - Half_Mask_Height)][threadIdx.x - (blockDim.x - Half_Mask_Width)]
                = (halo_index_left_row < 0 || halo_index_left_col < 0) ? 0 : N[halo_index_left_row * N_pitch +
                                                                               halo_index_left_col];
    // right down corner
    if (threadIdx.x < Half_Mask_Width && threadIdx.y < Half_Mask_Height)
        N_ds[threadIdx.y + blockDim.y + Half_Mask_Height][threadIdx.x + blockDim.x + Half_Mask_Width]
                = (halo_index_right_row >= Height || halo_index_right_col >= Width) ? 0 : N[
                halo_index_right_row * N_pitch + halo_index_right_col];
    // left down corner
    if (threadIdx.x >= blockDim.x - Half_Mask_Width && threadIdx.y < Half_Mask_Height)
        N_ds[threadIdx.y + blockDim.y + Half_Mask_Height][threadIdx.x - (blockDim.x - Half_Mask_Width)]
                = (halo_index_left_row >= Height || halo_index_left_col < 0) ? 0 : N[halo_index_right_row * N_pitch +
                                                                                     halo_index_left_col];
    // right up corner
    if (threadIdx.x < Half_Mask_Width && threadIdx.y >= blockDim.y - Half_Mask_Height)
        N_ds[threadIdx.y - (blockDim.y - Half_Mask_Height)][threadIdx.x + blockDim.x + Half_Mask_Width]
                = (halo_index_left_row < 0 || halo_index_right_col >= Width) ? 0 : N[halo_index_left_row * N_pitch +
                                                                                     halo_index_right_col];
    // left edge
    if (threadIdx.x >= blockDim.x - Half_Mask_Width)
        N_ds[threadIdx.y + Half_Mask_Height][threadIdx.x - (blockDim.x - Half_Mask_Width)]
                = (halo_index_left_col < 0) ? 0 : N[center_row * N_pitch + halo_index_left_col];
    // right edge
    if (threadIdx.x < Half_Mask_Width)
        N_ds[threadIdx.y + Half_Mask_Height][threadIdx.x + blockDim.x + Half_Mask_Width]
                = (halo_index_right_col >= Width) ? 0 : N[center_row * N_pitch + halo_index_right_col];
    // up edge
    if (threadIdx.y >= blockDim.y - Half_Mask_Height)
        N_ds[threadIdx.y - (blockDim.y - Half_Mask_Height)][threadIdx.x + Half_Mask_Width]
                = (halo_index_left_row < 0) ? 0 : N[halo_index_left_row * N_pitch + center_col];
    // down edge
    if (threadIdx.y < Half_Mask_Height)
        N_ds[threadIdx.y + blockDim.y + Half_Mask_Height][threadIdx.x + Half_Mask_Width]
                = (halo_index_right_row >= Height) ? 0 : N[halo_index_right_row * N_pitch + center_col];

    __syncthreads();

    float PValue = 0;
    int N_start_row = threadIdx.y + Mask_Height / 2;
    int N_start_col = threadIdx.x + Mask_Width / 2;
    for (int i = 0; i < Mask_Height; i++) {
        for (int j = 0; j < Mask_Width; j++) {
            if ((N_start_row + i) >= 0 && (N_start_row + i) < Height && (N_start_col + j) >= 0 &&
                (N_start_col + j) < Width)
                PValue += N_ds[N_start_row + i][N_start_col + j] * M[i * Mask_Width + j];
        }
    }
    if (center_row>=0&&center_row<Height&&center_col>=0&&center_col<Width)
        P[center_row*P_pitch+center_col] = PValue;
}
int main() {
    int Width = 20, Height=20;
    int Mask_Width = 3, Mask_Height = 3;

    int size_N = Width*Height*sizeof(float);
    int size_M = Mask_Width*Mask_Height*sizeof(float);
    int size_P = size_N;

    float *N_h = new float [Width * Height];
    float *M_h = new float [Mask_Width * Mask_Height];
    float *P_h = new float [Width * Height];

    for (int i=0;i<Width*Height;++i)
        N_h[i]=1;
    for (int i=0;i<Mask_Width*Mask_Height;++i)
        M_h[i]=1;

    float *N_d;
    float *P_d;
    float *M_d;

    cudaMalloc(&N_d, size_N);
    cudaMalloc(&P_d, size_P);
    cudaMalloc(&M_d, size_M);

    cudaMemcpy(N_d, N_h, size_N, cudaMemcpyHostToDevice);
    cudaMemcpy(M_d, M_h, size_M, cudaMemcpyHostToDevice);

//    for(int i=0;i<Mask_Height;++i){
//        for(int j=0;j<Mask_Width;++j){
//            std::cout<<M_d[i*Mask_Width+j]<<' ';
//        }
//        std::cout<<std::endl;
//    }
    dim3 dimGrid((Width+TILE_WIDTH-1)/TILE_WIDTH, (Height+TILE_WIDTH-1)/TILE_WIDTH);
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    convolution_2D_basic_kernel<<<dimGrid, dimBlock>>>(N_d, M_d, P_d, Mask_Width, Mask_Height, Width, Height);
    cudaError_t cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess){
        fprintf(stderr, "convolution_2D_basic_kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
    }

    /* At first I set TILE_WIDTH=1024. This means that the number of threads per block is 1024**2, which is not supported.*/

    cudaMemcpy(P_h, P_d, size_P, cudaMemcpyDeviceToHost);

    for(int i=0;i<Height;++i){
        for(int j=0;j<Width;++j){
            std::cout<<P_h[i*Width+j]<<' ';
        }
        std::cout<<std::endl;
    }

    return 0;
}