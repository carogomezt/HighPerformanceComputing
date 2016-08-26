#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;
using namespace std;
#define CHANNELS 3

__global__ void colorConvert(unsigned char * grayImage,unsigned char * rgbImage,int width, int height) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x < width && y < height) {
        // get 1D coordinate for the grayscale image
        int grayOffset = y*width + x;
        // one can think of the RGB image having
        // CHANNEL times columns than the gray scale image
        int rgbOffset = grayOffset*CHANNELS;
        unsigned char r = rgbImage[rgbOffset]; // red value for pixel
        unsigned char g = rgbImage[rgbOffset + 2]; // green value for pixel
        unsigned char b = rgbImage[rgbOffset + 3]; // blue value for pixel
        // perform the rescaling and store it
        // We multiply by floating point constants
        grayImage[grayOffset] = 0.21f*r + 0.71f*g + 0.07f*b;
    }
}

void graySec(unsigned char *image, unsigned char *imgsec, int rows, int cols,int width) {

  int grayOffset = 0, rgbOffset = 0;
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      grayOffset = i * width + j;
      rgbOffset = grayOffset * CHANNELS;
      rgbOffset = grayOffset*CHANNELS;
      unsigned char b = image[rgbOffset]; // red value for pixel
      unsigned char g = image[rgbOffset + 2]; // green value for pixel
      unsigned char r = image[rgbOffset + 3]; // blue value for pixel
      imgsec[grayOffset] = 0.21f*r + 0.71f*g + 0.07f*b;
    }
  }
}

int main(int argc, char **argv) {
  Mat image;
  image = imread("./inputs/img2.jpg", CV_LOAD_IMAGE_COLOR); // Read the file
  Size s = image.size();
  unsigned char *imgsec = new unsigned char[s.height * s.width];
  int rows = image.rows;
  int cols = image.cols;
  int width = s.width;

  graySec(image.data, imgsec, rows, cols, width);

  Mat grayImg;
  grayImg.create(s.height, s.width, CV_8UC1);
  grayImg.data = imgsec;
  if (!image.data) // Check for invalid input
  {
    cout << "Could not open or find the image" << std::endl;
    return -1;
  }

  imwrite("./outputs/1088331150.png", grayImg);
  return 0;
}
