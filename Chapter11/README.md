# Chapter11: Parallel patterns: merge sort(An introduction to tiling with dynamic input data identification)

## Exercise 11.1

> Assume that we need to merge lists A and B. A = (1, 7, 8, 9, 10) and B = (7, 10, 10, 12). What are the co-rank values for C[8]?

We need to find i, j such that i+j=k, and A[i-1]<=B[j], A[j-1]<B[i].

需要找到i, j满足 i+j=k, 和 A[i-1]<=B[j], A[j-1]<B[i].

**i=5, j=3**

## Exercise 11.2

> Complete the calculation of co-rank functions for Thread 1 and Thread 2 in the example shown in Fig. 11.7 through Fig. 11.9.