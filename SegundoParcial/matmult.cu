#include <cuda.h>
#include <stdio.h>

// Multiplicacion de Fila - Matriz

__global__ void multMatCUDA(float *d_a,float *d_b,float *d_c, int NRA, int NCA, int NCB){
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  
  if(row < NRA && col < NCB){
    float result = 0;
    for(int k = 0; k < NCA; k++){
      result += d_a[row * NCA + k] * d_b[k * NCB + col];
    }
    d_c[row * NCB + col] = result;
  }
}

void Mult_RowMat(float *M_a, float *M_b, float *R_c, int NRA, int NCA, int NCB) {
  float blockSize = 32;
  float *d_a, *d_b, *d_c;
      
  //Asignacion de memoria en el device
  cudaMalloc(&d_a, sizeof(float) * NRA * NCA);
  cudaMalloc(&d_b, sizeof(float) * NCA * NCB);
  cudaMalloc(&d_c, sizeof(float) * NRA * NCB);  

  cudaMemcpy(d_a, M_a, NRA * NCA * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, M_b, NCA * NCB * sizeof(float), cudaMemcpyHostToDevice);
   
  dim3 dimBlock(blockSize, blockSize, 1);
  dim3 dimGrid(ceil(NRA/blockSize),ceil(NCB/blockSize),1);
  
  multMatCUDA<<< dimGrid, dimBlock >>>(d_a, d_b, d_c, NRA, NCA, NCB);
  cudaMemcpy(R_c, d_c, NRA * NCB * sizeof(float), cudaMemcpyDeviceToHost);
  
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

}