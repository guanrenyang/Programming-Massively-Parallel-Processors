## Chapter 3: Scalable parallel execution

## Exercise 3.1

> A matrix addition takes two input matrices A and B and produces one output matrix C. Each element of the output matrix C is the sum of the corresponding elements of the input matrices A and B, i.e., C[i][j] = A[i][j] + B[i][j]. For simplicity, we will only handle square matrices whose elements are single precision floating-point numbers. Write a matrix addition kernel and the host stub function that can be called with four parameters: pointer- to-the-output matrix, pointer-to-the-first-input matrix, pointer-to-the-second-input matrix, and the number of elements in each dimension. Follow the instructions below:
> A. Write the host stub function by allocating memory for the input and output matrices, transferring input data to device; launch the kernel, transferring the output data to host and freeing the device memory for the input and output data. Leave the execution configuration parameters open for this step.
>B. Write a kernel that has each thread to produce one output matrix element. Fill in the execution configuration parameters for this design.
>C. Write a kernel that has each thread to produce one output matrix row. Fill in the execution configuration parameters for the design. 
>D. Write a kernel that has each thread to produce one output matrix column. Fill in the execution configuration parameters for the design.
>E. Analyze the pros and cons of each kernel design above.

There is a implementation of [vector addition](https://github.com/guanrenyang/AI3615-AI-Chip-Design/tree/main/homework1/3.%20cuda%20programming/1.%20vector%20add), but it may differ slightly from what is required by the exercise.

## Exercise 3.2

> A matrix–vector multiplication takes an input matrix B and a vector C and produces one output vector A. Each element of the output vector A is the dot product of one row of the input matrix B and C, i.e., A[i] = ∑ j B[i][j] + C[j]. For simplicity, we will only handle square matrices whose elements are single- precision floating-point numbers. Write a matrix–vector multiplication kernel and a host stub function that can be called with four parameters: pointer-to-the-output matrix, pointer-to-the-input matrix, pointer-to-the-input vector, and the number of elements in each dimension. Use one thread to calculate an output vector element.

There is a implementation of [matrix multiplication](https://github.com/guanrenyang/AI3615-AI-Chip-Design/tree/main/homework1/3.%20cuda%20programming/2.%20matrix%20multiplication), but it may differ slightly from what is required by the exercise.

## Exercise 3.3

> If the SM of a CUDA device can take up to 1536 threads and up to 4 thread blocks. Which of the following block configuration would result in the largest number of threads in the SM?
> A. 128 threads per block
> B. 256 threads per block
> C. 512 threads per block
> D. 1024 threads per block

**C**. A SM could take up 3 thread blocks which fully use the 1536 threads.

## Exercise 3.4

> For a vector addition, assume that the vector length is 2000, each thread calculates one output element, and the thread block size is 512 threads. How many threads will be in the grid?
> A. 2000
> B. 2024
> D. 2048
> D. 2096

**D**. At least 4 blocks needed, the number of threads in a grid is 4\*512=2048 threads.

## Exercise 3.5

> With reference to the previous question, how many warps do you expect to have divergence due to the boundary check on vector length?
> A. 1
> B. 2
> C. 3
> D. 6

**A**. Only the last but one warp has control divergence. The last warp will never be scheduled on any SM because none element is computed by it.

## Exercise 3.6

> You need to write a kernel that operates on an image of size 400 × 900 pixels. You would like to assign one thread to each pixel. You would like your thread blocks to be square and to use the maximum number of threads per block possible on the device (your device has compute capability 3.0). How would you select the grid dimensions and block dimensions of your kernel?

Block Size: (25, 25)

Grid Size: (16, 36)

Setting block size to (32, 32) could take full use of available threads, but setting it to (25, 25) could eliminate control divergence. (Not empirical test yet).

## Exercise 3.7

> With reference to the previous question, how many idle threads do you expect to have?

No idle thread.

## Exercise 3.8

Hint: Each thread has to wait for the third thread which executes for 3.0 microseconds.

## Exercise 3.9

> Consider a hypothetical block with 8 threads executing a section of code before reaching a barrier. The threads require the following amount of time (in microseconds) to execute the sections: 2.0, 2.3, 3.0, 2.8, 2.4, 1.9, 2.6, and 2.9 and to spend the rest of their time waiting for the barrier. What percentage of the total execution time of the thread is spent waiting for the barrier?

Only discuss capability 3.0

C: **Yes, it is possible**

D: **No, it is impossible**. The maximum thread in a block is 1024.

## Exercise 3.10

> A CUDA programmer says that if they launch a kernel with only 32 threads in each block, they can leave out the __syncthreads() instruction wherever barrier synchronization is needed. Do you think this is a good idea? Explain.

***I am not sure about the answer**.*

**Yes**. If you trust the implementation of GPU schedule, the answer is yes. However, new architecture may differ from current contents of the book. 

## Exercise 3.11

> A student mentioned that he was able to multiply two 1024 × 1024 matrices by using a tiled matrix multiplication code with 32 × 32 thread blocks. He is using a CUDA device that allows up to 512 threads per block and up to 8 blocks per SM. He further mentioned that each thread in a thread block calculates one element of the result matrix. What would be your reaction and why?

I think it is not a good idea. Since a block could not contain a 32\*32 threads but only 512 threads. 
