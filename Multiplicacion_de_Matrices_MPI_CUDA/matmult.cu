#include <cuda.h>
#include <stdio.h>

// Multiplicacion de Fila - Matriz

__global__ void multMatCUDA(float *d_a,float *d_b,float *d_c, int H, int W){
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  
  if(row < H && col < W){
    float result = 0;
    for(int k = 0; k < W; k++){
      result += d_a[k] * d_b[k * W + col];
    }
    d_c[col] = result;
  }
}

void Mult_RowMat(float *R_a, float *M_b, float *R_c, int H, int W){
  float blockSize = 32;
  float *d_a, *d_b, *d_c;

  printf("ROW\n");
  for (int i = 0; i < H; i++) {
    printf("%f ", R_a[i]);
  }
  printf("\n");

  printf("MAT\n");
  for (int i = 0; i < H; i++) {
        for (int j = 0; j < W; j++){
          printf("%f ", M_b[i * W + j]);
        }
        printf("\n");
      }
      
  //Asignacion de memoria en el device
  cudaMalloc(&d_a, sizeof(float) * H);
  cudaMalloc(&d_b, sizeof(float) * H * W);
  cudaMalloc(&d_c, sizeof(float) * W);  

  cudaMemcpy(d_a, R_a, H * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, M_b, H * W * sizeof(float), cudaMemcpyHostToDevice);
   
  dim3 dimBlock(blockSize, blockSize, 1);
  dim3 dimGrid(ceil(W/blockSize),ceil(H/blockSize),1);
  
  multMatCUDA<<< dimGrid, dimBlock >>>(d_a, d_b, d_c, H, W);
  cudaMemcpy(R_c, d_c, W * sizeof(float), cudaMemcpyDeviceToHost);

  printf("OUT\n");
  for (int i = 0; i < W; i++) {
    printf("%f ", R_c[i]);
  }
  printf("\n");
  
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

}