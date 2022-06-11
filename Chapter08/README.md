# Chapter 8 Parallel patterns: prefix sum (An introduction to work efficiency in parallel algorithms)

## Exercise 8.1

> Analyze the parallel scan kernel in Fig. 8.2. Show that control divergence only occurs in the first warp of each block for stride values up to half the warp size; i.e., for warp size 32, control divergence will occur to iterations for stride values 1, 2, 4, 8, and 16.

值得注意的是，一个block只能处理图 8.2 中的 `blockDim.x`个元素。 `blockDim.x`（一维网格）最大可达 1024，warp size 为 32。**当 stride 等于或大于 32 时，空闲线程的数量是 32 的整数倍。换句话说，一个块内部的线程要么全部工作要么全部空闲，并且不会出现块中的一些线程工作而一些不工作的情况。这意味着没有control divergence。**

It is worthing noting that a block could only handle `blockDim.x` elements in Figure 8.2. `blockDim.x` (1D grid) could be up to 1024 and warp size is 32. **When stride is equal or greater than 32, the number of idle threads is an integer multiple of 32. In other words, the threads within a block are either all working or all idle , and there will not be a situation where some threads in a block work and some do not. This means no control divergence.**

## Exercise 8.2

> For the Brent–Kung scan kernel, assume that we have 2048 elements. How many additions will be performed in both the reduction tree phase and the inverse reduction tree phase?
> a. (2048−1)\*2
> b. (1024−1)\*2
> c. 1024\*1024
> d. 10\*1024

**Reduction tree phase**: N-1 = 2047

**Inverse reduction tree**: (2 ‒ 1) + (4 ‒ 1) + ... + (N/4 ‒ 1) + (N/2 ‒ 1) = N-1-log2(N) = 2047-11=2036

The sum is 4083. The total number of additions of N elements is ![](https://latex.codecogs.com/svg.image?2N-2-\log_2(N)), so **a** is a good **approximation**.

总和是4083。N个元素相加的总数是![](https://latex.codecogs.com/svg.image?2N-2-\log_2(N))，所以**a**是 一个很好的**近似**。

## Exercise 8.3

>For the Kogge–Stone scan kernel based on reduction trees, assume that we have 2048 elements. Which of the following gives the closest approximation of the number of additions that will be performed?
>a. (2048−1)\*2
>b. (1024−1)\*2
>c. 1024\*1024
>d. 10\*1024

The sum is ![](https://latex.codecogs.com/svg.image?N\times\log_2(N)-(N-1)&space;), which is **2048×11−(2048−1)=20481**

a. (2048−1)\*2 = 4094
b. (1024−1)\*2 = 2046
c. 1024\*1024 = 1048576
d. 10\*1024 = 10240

Though **d** is a good **approximation** with respect to magnitude, it is half the true value.

尽管 **d** 在数量级上是一个很好的**近似值**，但它只有真实值的一半。

## Exercise 8.4

>  Use the algorithm in Fig. 8.3 to complete an exclusive scan kernel.

Code is in `KoggeStoneExclusiveScan.cu`, kernel function `Kogge_Stone_exclusive_scan_kernel`. In this kernel, `blockDim.x=InputSize=SECTION_SIZE` could be any number less than or equal to 1024(upper bound of threads).

代码在`KoggeStoneExclusiveScan.cu`，内核函数`Kogge_Stone_exclusive_scan_kernel`。 在这个内核中，`blockDim.x=InputSize=SECTION_SIZE` 可以是任何小于或等于 1024（线程上限）的数字。

## Exercise 8.5

> Complete the host code and all three kernels for the hierarchical parallel scan
> algorithm in Fig. 8.9.

Code is in `HierarchicalParallelScan.cu`.

## Exercise 8.6

*I am especially not sure about the answer to this question.*

Suppose the total number of elements is `N` and the section size is `S`. The number of additions in a section is `2*S-2-logS` for one section in phase one when using Brent Kung kernel, is `2*N/S-2-log(N/S)` in phase 2, is `N-S` in phase 3.

The total number of additions is `(2*S-2-logS)*N/S+2*N/S-2-log(N/S)+N-S<4N-3`

*我特别不确定这个问题的答案。*

假设元素总数为“N”，SECTION大小为“S”。 当使用 Brent Kung Kernel时，第1阶段加法数是 `2*S-2-logS`，在阶段 2 是 `2*N/S-2-log(N/S)`， 在第 3 阶段是`NS`。

加法总数是`(2*S-2-logS)*N/S+2*N/S-2-log(N/S)+N-S<4N-3`

## Exercise 8.7 8.8

These two exercises are skipped, because they are not difficult to answer after implementing Kogge-Stone kernel and Brent-Kung Kernel in previous exercises.

这两个问题不作回答。因为在前面的练习中实现了 Kogge-Stone 内核和 Brent-Kung Kernel 之后，这两个算法都很容易理解。
