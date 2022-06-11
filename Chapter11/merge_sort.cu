#include <iostream>

void merge_sequential(int *A, int m, int *B, int n, int *C) {
    int i = 0; // index into A
    int j = 0; // index into B
    int k = 0; // index into C
    
    //handle the start of A[] and B[]
    while ((i<m)&&(j<n))
    {
        if (A[i]<=B[j])
        {
            C[k++] = A[i++];
        } 
        else
        {
            C[k++] = B[j++];
        }

        if (i==m)
        {
            // done with A[] handle remaining B[]
            for (; j < n; j++)
            {
                C[k++] = B[j];
            }
        }
        else
        {
            // done with B[], handle remaining A[]
            for (; i<m; i++){
                C[k++] = A[i];
            }
        }
    }
    
}

int co_rank(int k, int* A, int m, int* B, int n){
    int i = k<m?k:m; // i=min(k,m)
    int j = k-i;
    int i_low = 0>(k-n)?0:k-n; // i_low = max(0, k-n)
    int j_low = 0>(k-m)?0:k-m; // j_low = max(0, k-m)
    int delta;
    bool active = true;
    while (active)
    {
        if(i>0 && j<n && A[i-1]>B[j])
        {
            delta = ((i-i_low+1)>>1); // ceil(i-i_low)/2
            j_low = j;
            j = j+delta;
            i = i-delta; 
        }
        else if(j>0 && i<m && B[j-1]>=A[i])
        {
            delta = ((j-j_low+1)>>1);
            i_low = i;
            i = i+delta;
            j = j-delta;
        } 
        else 
            active = false;
    }
    return i;
}