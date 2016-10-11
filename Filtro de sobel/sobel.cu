#include <cmath>
#include <cuda.h>
#include <cv.h>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <stdio.h>
#include <time.h>

using namespace cv;
using namespace std;
#define CHANNELS 3

__global__ void sobel(unsigned char *in, char *mask1, char *mask2,
                      unsigned char *out, int maskwidth, int w, int h) {
  int Col = blockIdx.x * blockDim.x + threadIdx.x;
  int Row = blockIdx.y * blockDim.y + threadIdx.y;

  if (Col < w && Row < h) {
    int Gx = 0, Gy = 0;
    int N_start_col = Col - (maskwidth / 2);
    int N_start_row = Row - (maskwidth / 2);

    for (int j = 0; j < maskwidth; j++) {
      for (int k = 0; k < maskwidth; k++) {
        int curRow = N_start_row + j;
        int curCol = N_start_col + k;

        if (curRow > -1 && curRow < h && curCol > -1 && curCol < w) {
          Gx += in[curRow * w + curCol] * mask1[j * maskwidth + k];
          Gy += in[curRow * w + curCol] * mask2[j * maskwidth + k];
        }
      }
    }
    if (Gx < 0)
      Gx = 0;
    else {
      if (Gx > 255)
        Gx = 255;
    }
    if (Gy < 0)
      Gy = 0;
    else {
      if (Gy > 255)
        Gy = 255;
    }

    out[Row * w + Col] = (unsigned char)sqrtf((Gx * Gx) + (Gy * Gy));
  }
}

int main(int argc, char **argv) {
  // Lectura de la imagen con openCV
  Mat image;
  image = imread("./inputs/img4.jpg", CV_LOAD_IMAGE_COLOR); // Read the file
  Size s = image.size();
  int width = s.width;
  int height = s.height;

  // Definicion de mascaras
  int maskwidth = 3;
  char h_mask1[] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
  char h_mask2[] = {-1, -2, -1, 0, 0, 0, 1, 2, 1};

  // Definicion de variables que se manejaran en el device
  unsigned char *d_image_Gray, *h_imageOutput, *out;
  char *mask1, *mask2;

  // Reserva de memora para variales en host
  h_imageOutput =
      (unsigned char *)malloc(sizeof(unsigned char) * width * height);

  // Reserva de memoria para variables en device
  cudaMalloc((void **)&d_image_Gray, sizeof(unsigned char) * width * height);
  cudaMalloc((void **)&out, sizeof(unsigned char) * width * height);
  cudaMalloc((void **)&mask1, sizeof(char) * maskwidth * maskwidth);
  cudaMalloc((void **)&mask2, sizeof(char) * maskwidth * maskwidth);

  // Definicion de los bloques e hilos por bloques
  int blockSize = 32;
  dim3 dimBlock(blockSize, blockSize, 1);
  dim3 dimGrid(ceil(width / float(blockSize)), ceil(height / float(blockSize)),
               1);

  // Copiando los datos del host al device
  cudaMemcpy(mask1, h_mask1, maskwidth * maskwidth * sizeof(char),
             cudaMemcpyHostToDevice);
  cudaMemcpy(mask2, h_mask2, maskwidth * maskwidth * sizeof(char),
             cudaMemcpyHostToDevice);

  // Convirtiendo imagen en escala de grises con openCV
  Mat grayImg;
  cvtColor(image, grayImg, CV_BGR2GRAY);

  // Copiando la imagen del host al device
  cudaMemcpy(d_image_Gray, grayImg.data, width * height * sizeof(unsigned char),
             cudaMemcpyHostToDevice);

  // Lanzando el kernel
  sobel<<<dimGrid, dimBlock>>>(d_image_Gray, mask1, mask2, out, maskwidth,
                               width, height);

  // Copiando el resultado del device al host
  cudaMemcpy(h_imageOutput, out, width * height * sizeof(unsigned char),
             cudaMemcpyDeviceToHost);

  // Generando la imagen de salida
  Mat grayImgCuda;
  grayImgCuda.create(s.height, s.width, CV_8UC1);
  grayImgCuda.data = h_imageOutput;

  // Filtro de Sobel con openCV
  Mat gray_image_opencv, grad_x, abs_grad_x;
  cvtColor(image, gray_image_opencv, CV_BGR2GRAY);
  Sobel(gray_image_opencv, grad_x, CV_8UC1, 1, 0, 3, 1, 0, BORDER_DEFAULT);
  convertScaleAbs(grad_x, abs_grad_x);

  if (!image.data) // Check for invalid input
  {
    cout << "Could not open or find the image" << endl;
    return -1;
  }

  // Guardando la imagen generada por CUDA
  imwrite("./outputs/1088331150.png", grayImgCuda);

  // Guardando la imagen generada por openCV
  // imwrite("./outputs/1088331150.png", abs_grad_x);
  cout << "La imagen esta lista." << std::endl;
  return 0;
}
