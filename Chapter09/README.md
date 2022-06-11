# Chapter 9 Parallel pattern: parallel histogram computation (An introduction to atomic operations and privatization)
## Exercise 9.1

> Assume that each atomic operation in a DRAM system has a total latency of 100 ns. What is the maximal throughput we can get for atomic operations on the same global memory variable?
> a. 100 G atomic operations per second
> b. 1 G atomic operations per second
> c. 0.01 G atomic operations per second
> d. 0.0001 G atomic operations per second

An atomic add operation includes 1 DRAM read and 1 DRAM write. The total latency of both read and write is 100 ns.

The total throughput is 

![](https://latex.codecogs.com/svg.image?Max\&space;ThroughPut&space;=&space;\frac{1}{100ns}&space;=&space;1\times&space;10^7&space;=&space;0.01&space;G/s&space;)

Answer: **c**

## Exercise 9.2

> For a processor that supports atomic operations in L2 cache, assume that each atomic operation takes 4 ns to complete in L2 cache and 100 ns to complete in DRAM. Assume that 90% of the atomic operations hit in L2 cache. What is the approximate throughput for atomic operations on the same global memory variable?
> a. 0.225 G atomic operations per second
> b. 2.75 G atomic operations per second
> c. 0.0735 G atomic operations per second
> d. 100 G atomic operations per second

![](https://latex.codecogs.com/svg.image?Expected\&space;Latency&space;=&space;90%&space;\times&space;L_2&space;Cache&space;Latency&space;&plus;&space;10%&space;DRAMLatency=0.9\times&space;4ns&plus;0.1*100ns&space;=&space;13.6&space;ns)

![](https://latex.codecogs.com/svg.image?ThrouchPut&space;=&space;\frac{1}{13.6ns}&space;\approx&space;0.0735&space;G/s)

Answer: **c**

## Exercise 9.3

> In question 1, assume that a kernel performs 5 floating-point operations per atomic operation. What is the maximal floating-point throughput of the kernel execution as limited by the throughput of the atomic operations?
> a. 500 GFLOPS
> b. 5 GFLOPS
> c. 0.05 GFLOPS
> d. 0.0005 GFLOPS

A floating-point operation takes 100/5 = 20ns. The *ThroughPut* is 0.05GFLOPS.

Answer: **c**

## Exercise 9.4

> In Question 1, assume that we privatize the global memory variable into shared memory variables in the kernel and the shared memory access latency is 1 ns. All original global memory atomic operations are converted into shared memory atomic operation. For simplicity, assume that the additional global memory atomic operations for accumulating privatized variable into the global variable adds 10% to the total execution time. Assume that a kernel performs 5 floating-point operations per atomic operation. What is the maximal floating-point throughput of the kernel execution as limited by the throughput of the atomic operations?
> a. 4500 GFLOPS
> b. 45 GFLOPS
> c. 4.5 GFLOPS
> d. 0.45 GFLOPS

A float-point operation takes 0.2 ns in shared memory. The total latency is 0.2(1+10%) = 0.22 ns. The *ThroughPut* is 1/0.22ns = 4.5 GFLOPS

Answer: **c**

## Exercise 9.5

> To perform an atomic add operation to add the value of an integer variable Partial to a global memory integer variable Total, which one of the following statements should be used?
> a. atomicAdd(Total, 1)
> b. atomicAdd(&Total, &Partial)
> c. atomicAdd(Total, &Partial)
> d. atomicAdd(&Total, Partial)

Answer: **d**