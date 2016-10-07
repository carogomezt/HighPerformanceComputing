#include <cuda.h>
#include <iostream>
#include <stdio.h>
#include <time.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <cv.h>

using namespace cv;
using namespace std;
#define CHANNELS 3

__global__ void sobel(unsigned char *in, unsigned char *mask1, 
unsigned char *mask2, unsigned char *out, int maskwidth, int w, int h) {
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
    out[Row * w + Col] = sqrtf((Gx*Gx) + (Gy*Gy));
  }
}

int main(int argc, char **argv) {
  Mat image;
  image = imread("./inputs/img2.jpg", CV_LOAD_IMAGE_COLOR); // Read the file
  Size s = image.size();
  unsigned char *image_Gray = new unsigned char[s.height * s.width];
  int width = s.width;
  int height = s.height;
  int maskwidth = 3;
  unsigned char h_mask1[] = {-1, 0, 1,
                    -2, 0, 2,
                    -1, 0, 1};
  unsigned char h_mask2[] = {-1, -2, -1,
                    0, 0, 0,
                    1, 2, 1};


  unsigned char *d_image, *d_image_Gray, *h_imageOutput, *out, *mask1, *mask2;
  int size = sizeof(unsigned char) * width * height * image.channels();
  int sizeGray = sizeof(unsigned char) * width * height;
  h_imageOutput = (unsigned char *)malloc(sizeGray);
  
  
  cudaMalloc((void **)&d_image, size);
  cudaMalloc((void **)&d_image_Gray, sizeGray);
  cudaMalloc((void **)&out, sizeGray);
  cudaMalloc((void **)&mask1, maskwidth * maskwidth);
  cudaMalloc((void **)&mask2, maskwidth * maskwidth);


  Mat grayImg;
  grayImg.create(s.height, s.width, CV_8UC1);
  grayImg.data = image_Gray;

  int blockSize = 32;
  dim3 dimBlock(blockSize, blockSize, 1);
  dim3 dimGrid(ceil(width / float(blockSize)), ceil(height / float(blockSize)),
               1);

  // cudaMemcpy(d_image_Gray, image_Gray, size, cudaMemcpyHostToDevice);
  
  cudaMemcpy(mask1, h_mask1, maskwidth * maskwidth * sizeof(unsigned char), cudaMemcpyHostToDevice);
  cudaMemcpy(mask2, h_mask2, maskwidth * maskwidth * sizeof(unsigned char), cudaMemcpyHostToDevice);

  //colorConvert<<<dimGrid, dimBlock>>>(d_image_Gray, d_image, width, height);
  
  
  Mat grad_x, abs_grad_x;
  cvtColor(image, grayImg, CV_BGR2GRAY);
  Sobel(grayImg, grad_x, CV_8UC1, 1, 0, 3, 1, 0, BORDER_DEFAULT);
  convertScaleAbs(grad_x, abs_grad_x);
  
  cudaMemcpy(d_image_Gray, grayImg.data, size, cudaMemcpyHostToDevice);
  
  //sobel<<<dimGrid, dimBlock>>>(d_image_Gray, mask1, mask2, out, maskwidth, width, height);
  //cudaMemcpy(h_imageOutput, out, sizeGray, cudaMemcpyDeviceToHost);
  
  Mat grayImgCuda;
  grayImgCuda.create(s.height, s.width, CV_8UC1);
  grayImgCuda.data = h_imageOutput;

  if (!image.data) // Check for invalid input
  {
    cout << "Could not open or find the image" << endl;
    return -1;
  }

  imwrite("./outputs/1088331150.png", abs_grad_x);
  return 0;
}