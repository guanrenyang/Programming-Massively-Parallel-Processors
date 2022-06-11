# Chapter 5: Performance Considerations

## Exercise 5.1: 

Full codes are in the comments of `sumReduction.cu`

Figure 5.13

```c
__shared__ float partialSum[];
unsigned int t = 2*threadIdx.x;
for (unsigned int stride = 1;
stride < (blockDim.x << 1); stride <<= 1)
{
__syncthreads();
if (t % (2*stride) == 0)
partialSum[t] += partialSum[t+stride];
}
```

Figure 5.15

```c
__shared__ float partialSum[];
unsigned int t = threadIdx.x;
for (unsigned int stride = blockDim.x;
stride > 0; stride >>=1 )
{
__syncthreads();
if (t < stride)
partialSum[t] += partialSum[t+stride];
}
```

The configuration parameters should be modified such that the block size is half of what it was
originally in both cases. In the first case, two extra operations were added: doubling both the base
index for each thread and the blockDim.x terminating condition. In the second case, one operation was
added, and another removed. Note that slight variations of equivalent implementations are possible.

## Exercise 5.2

The 'improved` version (the latter one). 

The second modification actually had no net effect on the number of operations,
as one was added, and another removed. This is less than the two operations added in the first case.


## Exercise 5.3:

The codes are in `sumReduction.cu`.

I fix the number of elements of array `X` as 1024, the number of blocks as 2, the orginally number of thread(dimBlock) as 512, the size of shared memory as 512,  **so I didn't check the bounds of the array, while it's actually important.**

Some other improvements are made. Details are in the comments of `sumReduction.cu`

***The official solution is shown below***:
```c
#define TILE_SIZE 512
__global__ void reduce_subarrays(float* input, float* output,
unsigned int numElements)
{
__shared__ float partialSum[TILE_SIZE];
unsigned int t = threadIdx.x;
unsigned int index = t + TILE_SIZE*blockIdx.x;
// Keep in mind that the last block might not have exactly the
// right number of elements to read from input.
if(index < numElements)
partialSum[t] = input[index];
else
partialSum[t] = 0.0f;
if(index + blockDim.x < numElements)
partialSum[t+blockDim.x] = input[index + blockDim.x];
else
partialSum[t+blockDim.x] = 0.0f;
}
```
*Actually I use `blockDim.x` instead of `TILE_SIZE`. `TILE_SIZE` is used in case that the shared memory can't hold the whole array.

## Exercise 5.4

My solution is in `main()` function of `sumReduction.cpp`. The official solution use a *TILE\_WISE* technique but I didn't.

***The official solution is shown below***:

```c
// Recall TILE_SIZE was defined earlier
__host__ float reduceArray(float* array, unsigned int size)
{
float* array_d, temp_d;
float result;
unsigned int array_bytes = size*sizeof(float);
//Allocate and initialize device memory
cudaMalloc((void**)& array_d, array_bytes);
cudaMalloc((void**)& temp_d, ((size/TILE_SIZE)+1)*sizeof(float));
cudaMemcpy(array_d, array, array_bytes, cudaMemcpyHostToDevice);
}
unsigned int numBlocks;
unsigned int temp_size = size;
while(temp_size > 1)
{
//Compute the number of thread blocks to execute, keeping in mind integer
// division always rounds towards zero
numBlocks = (temp_size + TILE_SIZE – 1) / TILE_SIZE;
// Reduce sub arrays with kernel
reduce_subarrays<<<numBlocks, TILE_SIZE / 2>>>(array_d, temp_d, temp_size);
//Swap input and output array pointers
float* swap = array_d;
array_d = temp_d;
temp_d = swap;
//Update the number of array elements
temp_size = numBlocks;
}
//Copy the single value back from the device, and clean up device memory
cudaMemcpy(&result, array_d, sizeof(float), cudaMemcpyDeviceToHost);
cudaFree(array_d);
cudaFree(temp_d);
return result;

```


### Exercise 5.5

The global memory access pattern is shown blow 

![image-20220319190454709](https://gitee.com/guanrenyang/picbed/raw/master/img/image-20220319190454709.png)


## Exercise 5.6

B

## Exercise 5.7 

C  

Accesses are coalesced when reading from global memory. After that, only reading *N* from shared memory is coalesced.

读取的时候都是对齐的，读到shared memory中以后仍然是只有N对齐.

## Exercise 5.8

D

Every warp in the `simple` version has the problem of thread divergence, because each of them has to do the `if` statement.

## Exercise 5.9

B

In contrary to Exercise 5.8

## Exercise 5.10

All the codes are in `matrixMultipulation.cu`, with understanble comments.

It includes the **official solution**, while I think it is **incorrect and difficult to understand**

*Kernel 1* and *Kernel 2* share the same logic. I though kernel 2 could improve global bandwidth usage, but they turn out to be equivalent.  

**More details are in the code comments**.

## Exercise 5.11

`MATRIX_SIZE` needs to be divisible by `BLOCK_SIZE` .

`MATRIX_SIZE` 需要可以被 `BLOCK_SIZE` 整除。

## Exercise 5.12
 
**No promotion**. This is essentially a *loop* expansion, and there is still waste after the fifth iteration. Moreover, the modification is **wrong**, reading and writing order error of shared data will occur in the last `if`.

**没有提升**。这个写法本质上是循环展开，第5轮以后依然有线程浪费。这个写法中还有**错误**，最后一个if内会发生共享数据的读写顺序错误。
