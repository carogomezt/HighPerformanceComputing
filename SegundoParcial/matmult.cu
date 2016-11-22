#include <cuda.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Multiplicacion de Fila - Matriz

__global__ void multMatCUDA(double *d_a, double *d_b, double *d_c, int NRA,
                            int NCA, int NCB) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < NRA && col < NCB) {
    double result = 0;
    for (int j = 0; j < NCA; j++) {
      result += d_a[row * NCA + j] * d_b[j * NCB + col];
    }
    d_c[row * NCB + col] = result;
  }
}

void multMat(double *M_a, double *M_b, double *R_c, int NRA, int NCA, int NCB) {
  int blockSize = 32;
  double *d_a, *d_b, *d_c;

  printf("MAT A\n");
  for (int i = 0; i < NRA; i++) {
    for (int j = 0; j < NCA; j++) {
      printf("%f ", M_a[i * NCA + j]);
    }
    printf("\n");
  }

  printf("MAT B\n");
  for (int i = 0; i < NCA; i++) {
    for (int j = 0; j < NCB; j++) {
      printf("%f ", M_b[i * NCB + j]);
    }
    printf("\n");
  }

  // Asignacion de memoria en el device
  cudaMalloc(&d_a, sizeof(double) * NRA * NCA);
  cudaMalloc(&d_b, sizeof(double) * NCA * NCB);
  cudaMalloc(&d_c, sizeof(double) * NRA * NCB);

  cudaMemcpy(d_a, M_a, NRA * NCA * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, M_b, NCA * NCB * sizeof(double), cudaMemcpyHostToDevice);

  dim3 dimBlock(blockSize, blockSize, 1);
  dim3 dimGrid(ceil(NCB / blockSize), ceil(NRA / blockSize), 1);

  multMatCUDA<<<dimGrid, dimBlock>>>(d_a, d_b, d_c, NRA, NCA, NCB);
  cudaMemcpy(R_c, d_c, NRA * NCB * sizeof(double), cudaMemcpyDeviceToHost);

  printf("MAT CUDA\n");
  for (size_t i = 0; i < NRA; i++) {
    for (size_t j = 0; j < NCB; j++) {
      printf("%f ", R_c[i * NCB + j]);
    }
    printf("\n");
  }

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
}
