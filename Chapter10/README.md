# Chapter 10: Parallel patterns: sparse matrix computation

## Exercise 1

> Complete the host code to produce the hybrid ELL–COO format, launch the ELL kernel on the device, and complete the contributions of the COO elements.

The code is in `Hybird_ELL-COO.cu`.

## Exercise 2

> Complete the host code for creating JDS–ELL and launch one kernel for each section of the representation.

The code in `JDS-ELL.cu` is incomplete. Implementing a JDS-ELL host code for the toy 4\*4 matrix of the textbook is easy, but I fail to figure out a *generic programming* way to implement a JDS-ELL host code for matrices of arbitrary size. 

I may finish the answer after learning more about sparse matrix. As well, any feasible implement of you is welcomed, and you could just pull request to contribute to this solution. 
## Exercise 3

> Consider the following sparse matrix:
> 1 0 7 0
> 0 0 8 0
> 0 4 3 0
> 2 0 0 1
> Represent the matrix in each of the following formats: (a) COO, (b) CSR, and (c) ELL.

**COO**

`data[7]`= {1, 7, 8, 4, 3, 2, 1}

`row_index[7]` = {0, 0, 1, 2, 2, 3, 3}

`col_index[7]` = {0, 2, 2, 1, 2, 0, 3}

**ELL**

`data[8]` = {1, 8, 4, 2, 7, *, 3, 1}

`col_index[8]` = {0, 2, 1, 0, 0, *, 2, 3}

**CSR**

`row_ptr[5]` = {0, 2, 3, 5, 7}

`data[7]` = {1, 7, 8, 4, 3, 2, 1}

`col_index[7]` = {0, 2, 2, 1, 2, 0, 3}

## Exercise 4

> Given a sparse matrix of integers with m rows, n columns, and z nonzeros, how many integers are needed to represent the matrix in (a) COO, (b) CSR, and (c) ELL. If the information provided is insufficient, indicate the missing information.

**COO**

For each non-zero element, you need to store its **value** and **the coordinates of the two dimensions**. The total space needed is **3z**.

对于每一个非0元素，都需要存储它的值和两个维度的坐标。总数为 **3z**。

**CSR**:

Store the starting index of each row, including rows with no non-zero elements, the value and column\_index of non-zero elements. The total space is **2z+m**. 

需要存储每一行的起始索引，（包括没有非零元素的行），非零元素的值和列索引。 总空间为**2z+m**。

**ELL**:

Suppose that the number of non-zero elements in a row is at most `s`.

ELL stores the value and column\_index of each non-zero element, with paddings. The total space is **2ms**.
