//
// Created by guanrenyang on 22-5-3.
//
#include <iostream>
#include <vector>
#define MATRIX_SIZE 4
#define SECTION1_THRESHOLD 3
#define SECTION2_THRESHOLD 2
struct RowInfo{
    int row_idx;
    int num_nonzero_elem;
};
__global__ void SpMV_ELL(int num_rows, const float *data, const int *col_index, int num_elem, float *x, float *y){

    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < num_rows){
        float dot = 0;
        for (int i = 0; i<num_elem;i++){
            dot += data[row+i*num_rows] * x[col_index[row+i*num_rows]];
        }
        y[row] = dot;
    }
}
int main(){
/* Initialize the sparse matrix M on host */
    auto *M = new float [MATRIX_SIZE * MATRIX_SIZE];

    for (int i = 0; i < MATRIX_SIZE; i++){
        for (int j=0; j < MATRIX_SIZE; j++){
            M[i * MATRIX_SIZE + j]=0;
        }
    }
    M[0 * MATRIX_SIZE + 0] = 3;
    M[0 * MATRIX_SIZE + 2] = 1;
    M[2 * MATRIX_SIZE + 1] = 2;
    M[2 * MATRIX_SIZE + 2] = 4;
    M[2 * MATRIX_SIZE + 3] = 1;
    M[3 * MATRIX_SIZE + 0] = 1;
    M[3 * MATRIX_SIZE + 3] = 1;
    /* Debug: show M of host */
    std::cout<<"Original Matrix"<<std::endl;
    for (int i = 0;i<MATRIX_SIZE;i++){
        for (int j=0;j<MATRIX_SIZE;j++){
            std::cout<<M[i*MATRIX_SIZE+j]<<' ';
        }
        std::cout<<std::endl;
    }
    /* Debug Done */
    /* Initialization of matrix Done */

    /* Initialize x and y on host*/
    auto *x_h = new float [MATRIX_SIZE];
    auto *y_h = new float [MATRIX_SIZE];
    for (int i=0; i < MATRIX_SIZE; i++)
        x_h[i] = 1;

    /* Debug: show x */
    std::cout<<'x'<<std::endl;
    for (int i=0;i<MATRIX_SIZE;i++)
        std::cout<<x_h[i]<<' ';
    std::cout<<std::endl;
    /* Debug Done */
    /* Initializatin of x and y on host Done */

    /* Initialize x and y on device*/
    float *x_d;
    float *y_d;
    cudaMalloc(&x_d, MATRIX_SIZE * sizeof (float ));
    cudaMalloc(&y_d, MATRIX_SIZE * sizeof (float ));

    cudaMemcpy(x_d, x_h, MATRIX_SIZE * sizeof (float ), cudaMemcpyHostToDevice);
    /* Initialization of x and y on device done */

    /* Prepare for JDS transformation */
    std::vector<RowInfo> rowInfo(MATRIX_SIZE);

    // Key variables needed for SpMV


    /* Preparation Done */

    /* Section 1 */

    /* Section 1 Done */

}