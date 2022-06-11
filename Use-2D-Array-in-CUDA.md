CUDA使用二维数组的方法

```C++
// CPU分配--使用一维数组
float *N_h = new float [Width * Height];//pitch == width
// GPU内存分配
float *N_d; //二维分配实际上是在给一维数加pitch，而不是二维数组(指针的数组)
// 此处并不确定二维数组会不会对齐
size_t N_pitch_d;
cudaMallocPitch(&N_d, &N_pitch_d, Width*sizeof(float ), Height);
// 将数据从CPU迁移到GPU
cudaMemcpy2D(N_d, N_pitch_d, N_h, Width*sizeof(float ), Width*sizeof(float ), Height, cudaMemcpyHostToDevice);
// 二维数组的访问
for (int i=0;i<Height;i++){
    float *row = (float *)((char*)N+i*N_pitch_d);
    for (int j=0;j<Width;j++){
        row[j]++;
    }
}
/* 不可以使用以下语句 */
for (int i=0;i<Height;i++){
    for (int j=0;j<Width;j++){
        N[i*N_pitch_d+j];
    }
}
```


