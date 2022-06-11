#include <iostream>
#include <iomanip>

#define ELL_MAXIMUM 2 // The max non-zeros elements of a row in ELL representation
#define MATRIX_SIZE 4 // The width and height of the matrix

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
void SpMV_COO(const float *data, const int *row_index, const int *col_index, int num_elem, const float *x, float *y){
    for (int i=0;i<num_elem;i++){
        y[row_index[i]] += x[col_index[i]] * data[i];
    }
}
int main() {
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



    /* Compute the number of non-zero elements in a row having the maximum of them */
    int max_row_length=0;                           // the maximum of number of non-zero elements of each row

    for (int i=0; i < MATRIX_SIZE; i++){
        int curr_row_length=0;
        for(int j=0; j < MATRIX_SIZE; j++){
            if (M[i * MATRIX_SIZE + j] != 0) {
                curr_row_length++;
            }
        }
        if(curr_row_length > max_row_length){
            max_row_length = curr_row_length;
        }
    }
    /* Done */

    /* Extract the ELL and COO part */
    // ELL
    int ELL_row_length = std::min(ELL_MAXIMUM, max_row_length);
    auto *ELL_data_h = new float [MATRIX_SIZE * ELL_row_length];
    int *ELL_index_h = new int [MATRIX_SIZE * ELL_row_length];
    float *ELL_data_d;
    int *ELL_index_d;

    for (int i=0;i<MATRIX_SIZE*ELL_row_length;i++){
        ELL_data_h[i] = 0;
        ELL_index_h[i] = 0;
    }

    // COO (SpMV/COO is executed on host sequentially)
    auto *COO_data_h = new float [(max_row_length-ELL_row_length) * MATRIX_SIZE];
    int *COO_col_index_h = new int [(max_row_length-ELL_row_length) * MATRIX_SIZE];
    int *COO_row_index_h = new int [(max_row_length-ELL_row_length) * MATRIX_SIZE];
    int COO_num = 0;

    // temporary variables
    int ELL_avail[MATRIX_SIZE] = {0};
    int COO_avail = 0;
    bool COO_needed = true;
    if(max_row_length<=ELL_MAXIMUM)
        COO_needed= false;

    // Extraction
    for (int i=0; i < MATRIX_SIZE; i++){
        int num_nonzero_elem = 0;
        for (int j=0; j < MATRIX_SIZE; j++){
            if (M[i * MATRIX_SIZE + j] == 0)
                continue;
            if (num_nonzero_elem < ELL_row_length) {
                ELL_data_h[ELL_avail[i] * MATRIX_SIZE + i] = M[i * MATRIX_SIZE + j];
                ELL_index_h[ELL_avail[i] * MATRIX_SIZE + i] = j;
                ELL_avail[i]++;
            } else if (COO_needed) {
                COO_data_h[COO_avail] = M[i * MATRIX_SIZE + j];
                COO_row_index_h[COO_avail] = i;
                COO_col_index_h[COO_avail] = j;
                COO_num++;
                COO_avail++;
            }
            num_nonzero_elem++;
        }
    }
    /* Debug : show ELL_data, ELL_index, COO_data, COO_col_index, COO_row_index */

    // ELL_data
    std::cout<<"ELL_data"<<std::endl;
    for (int i=0;i<ELL_row_length;i++){
        for(int j=0; j < MATRIX_SIZE; j++){
            std::cout << ELL_data_h[i * MATRIX_SIZE + j] << ' ';
        }
        std::cout<<std::endl;
    }

    // ELL_Index
    std::cout<<std::endl<<"ELL_Index"<<std::endl;
    for (int i=0;i<ELL_row_length;i++){
        for(int j=0; j < MATRIX_SIZE; j++){
            std::cout << ELL_index_h[i * MATRIX_SIZE + j] << ' ';
        }
        std::cout<<std::endl;
    }

    // COO data and index
    std::cout.setf (std::ios::left); // 设置左对齐，setw设置输出长度
    std::cout<<"COO_data_h[i] "<<"COO_row_index_h[i] "<<"COO_col_index_h[i] "<<std::endl;
    for (int i=0;i<COO_num;i++){
        std::cout<<' '<<COO_data_h[i]<<' '<<COO_row_index_h[i]<<' '<<COO_col_index_h[i]<<std::endl;
    }
    /* Debug Done */

    /* Transfer data from host to device */
    cudaMalloc(&ELL_data_d, MATRIX_SIZE * ELL_row_length * sizeof(float ));
    cudaMalloc(&ELL_index_d, MATRIX_SIZE * ELL_row_length * sizeof(int ));

    cudaMemcpy(ELL_data_d, ELL_data_h, MATRIX_SIZE * ELL_row_length * sizeof(float ), cudaMemcpyHostToDevice);
    cudaMemcpy(ELL_index_d, ELL_index_h, MATRIX_SIZE * ELL_row_length * sizeof(int ), cudaMemcpyHostToDevice);
    /* Transfer Done */

    /* Launch ELL kernel */
    dim3 dimBlock(MATRIX_SIZE, MATRIX_SIZE);
    dim3 dimGrid(1, 1);
    SpMV_ELL<<<dimGrid, dimBlock>>>(MATRIX_SIZE, ELL_data_d, ELL_index_d, MATRIX_SIZE*ELL_row_length, x_d, y_d);
    /* Launch Done */

    // Transfer result y from device to host
    cudaMemcpy(y_h, y_d, MATRIX_SIZE*sizeof (float ), cudaMemcpyDeviceToHost);

    // Show result of SpMV/ELL
    std::cout<<"Intermediate result after ELL kernel"<<std::endl;
    for (int i=0;i<MATRIX_SIZE;i++){
        std::cout<<y_h[i]<<' ';
    }
    std::cout<<std::endl;

    /* Launch COO on host */
    SpMV_COO(COO_data_h, COO_row_index_h, COO_col_index_h,COO_num, x_h, y_h);
    /* Launch Done */

    // Show the final result
    std::cout<<"The final result is"<<std::endl;
    for (int i=0;i<MATRIX_SIZE;i++){
        std::cout<<y_h[i]<<' ';
    }

    return 0;
}
