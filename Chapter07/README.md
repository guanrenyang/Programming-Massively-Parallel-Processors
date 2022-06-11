# Chapter 7: Parallel patterns: convolution (An introduction to stencil computation)


## Exercise 7.1: 

> Calculate the P[0] value in Fig. 7.3.

p[0] = 3\*0+4\*0+5\*1+4\*2+3\*3=22

## Exercise 7.4

> Consider performing a 1D convolution on an array of size n with a mask of
> size m:
> a. How many halo cells are there in total?
> b. How many multiplications are performed if halo cells are treated as
> multiplications (by 0)?
> c. How many multiplications are performed if halo cells are not treated as
> multiplications?

For the entire array, halo cells are ghost cells.

 / is a C operand.

1. m-1
2. n\*m
3. n\*m-(1+2+...+m/2)*2(Not being treated as multiplications means using a if clause to skip multiplication)

## Exercise 7.5

> Consider performing a 2D convolution on a square matrix of size nxn with a
> square mask of size mxm:
> a. How many halo cells are there in total?
> b. How many multiplications are performed if halo cells are treated as
> multiplications (by 0)?
> c. How many multiplications are performed if halo cells are not treated as
> multiplications?

1. (n+m-1)(n+m-1)-n*n
2. m\*m\*n\*n
3. Use algebra to remove corners and edges…    (too time consuming)

Element (i,j), i in [1,n-1], j in [1,n-1]

**Corner Case**: For *(i in [1,m/2], j in [1,m/2])* or *(i in [1,m/2], j in [n-m/2+1,n])* or *(i in [n-m/2+1,n], j in [1,m/2])* or *(i in [n-m/2+1,n], j in [n-m/2+1,n])*, number of halo elements' multiplications is **(m\*m-i\*j)**

**Edge Case**: While ***not in Corner Case***, for *i in [1,m/2]*, number of halo elements' multiplications is **(m\*m-m\*j)**;  for *j in [1,m/2]*, number of halo elements' multiplications is **(m\*m-m\*i)**

The answer is to sum the i,j. (Too time consuming...)

**A trick for this kind of questions are shown below**.

## Exercise 7.6

> Consider performing a 2D convolution on a rectangular matrix of size n1xn2
> with a rectangular mask of size m1xm2:
> a. How many halo cells are there in total?
> b. How many multiplications are performed if halo cells are treated as
> multiplications (by 0)?
> c. How many multiplications are performed if halo cells are not treated as
> multiplications?

Similar to 7.5. **The trick is to think each element in the output matrix separately**. For example, when solving 7.5.2, first think that every element need m\*m multiplications, and there are n\*n elements in total in the output array, so the answer is  m\*m\*n\*n. When think 7.5.3, **just minus the center cells from  the m\*m multiplications, instead of counting how many multiplications a halo cell involves in.**

## Exercise 7.7

> Consider performing a 1D tiled convolution with the kernel shown in Fig. 7.11
> on an array of size n with a mask of size m using a tiles of size t:
> a. How many blocks are needed?
> b. How many threads per block are needed?
> c. How much shared memory is needed in total?
> d. Repeat the same questions if you were using the kernel in Fig. 7.13.

Suppose BLOCK\_SIZE==TILE\_SIZE

Figure 7.11

1. Number of blocks is n/t+1
2. Number of threads per block is t
3. (t+ m - 1)\*sizeof(float) shared memory is used in a block

Figure 7.13

1. number of blocks is n/t+1, too.
2. Number of threads per block is t, too.
3. (t \* sizeof(float)) shared memory is used in a block. 

## Exercise 7.8

> Revise the 1D kernel in Fig. 7.6 to perform 2D convolution. Add more width
> parameters to the kernel declaration as needed.

Code is in `2DConvBasic.cu`, function `convolution_2D_basic_kernel`. ***This kernel has no shared memory or constant memory used.***

All the 2D arrays in logic are accessed as 1D array in C.

## Exercise 7.9

> Revise the tiled 1D kernel in Fig. 7.8 to perform 2D convolution. Keep in
> mind that the host code also needs to be changed to declare a 2D M array in
> the constant memory. Pay special attention to the increased usage of shared
> memory. Also, the N_ds needs to be declared as a 2D shared memory array.

Code is in `2DConvAdvanced.cu`, function `convolution_2D_const_memory_kernel`. 

Shared memory is not used in this answer. 2D array is a 1D array in C. For example, `M[i][j]` is `M[i*Width+j]`.    The reasons are as follows:

1.  Two-dimensional arrays in the C language are not very easy to copy to in CUDA。
2. The size of the pitch fetched at one time is much larger than the size of the entire mask, so the entire mask can be fetched at one time.

## Exercise 7.10

> Revise the tiled 1D kernel in Fig. 7.11 to perform 2D convolution. Keep in
> mind that the host code also needs to be changed to declare a 2D M array in
> the constant memory. Pay special attention to the increased usage of shared
> memory. Also, the N_ds needs to be declared as a 2D shared memory array.

Code is in `2DConvAdvanced.cu`, function `convolution_2D_shared_memory_kernel`. The `N_ds` in shared memory is defined as an 2D array in C for easier access.



