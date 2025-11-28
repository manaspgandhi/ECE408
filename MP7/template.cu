// Histogram Equalization

#include <wb.h>

#define HISTOGRAM_LENGTH 256
#define uchar unsigned char

//@@ insert code here

__global__ void float_to_char(float* input, unsigned char* output, int height, int width) {
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int tz = threadIdx.z;
  
  int row = blockDim.y * blockIdx.y + ty;
  int col = blockDim.x * blockIdx.x + tx;
  
  if (row < height && col < width) {
    output[((row * width + col) * blockDim.z) + tz] = (unsigned char) (255 * input[((row * width + col) * blockDim.z) + tz]);
  }
}

__global__ void rgb_to_grayscale(unsigned char* input_img, unsigned char* grayscale_img, int height, int width) {
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int tz = threadIdx.z;
  
  int row = blockDim.y * blockIdx.y + ty;
  int col = blockDim.x * blockIdx.x + tx;
  int gray_index;

  if (row < height && col < width) {
    gray_index = (row * width) + col;

    unsigned char red = input_img[3 * gray_index];
    unsigned char green = input_img[(3 * gray_index) + 1];
    unsigned char blue = input_img[(3 * gray_index) + 2];
    unsigned char grayed_pixel = (unsigned char) (0.21*red + 0.71*green + 0.07*blue);
    grayscale_img[gray_index] = grayed_pixel;
  }

}

__global__ void grayscale_to_histo(unsigned char* grayscale_img, uint32_t* histogram, int height, int width) {

  __shared__ uint32_t histo_tiled[HISTOGRAM_LENGTH];

  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int tz = threadIdx.z;

  int row = blockDim.y * blockIdx.y + ty;
  int col = blockDim.x * blockIdx.x + tx;
  int hist_idx = blockDim.x * ty + tx;
  
  __syncthreads();
  histo_tiled[hist_idx] = 0;
  
  if (row < height && col < width) {
    uint32_t gray_img_elem = grayscale_img[row * width + col];
    atomicAdd(&(histo_tiled[gray_img_elem]), 1);
  }
  __syncthreads();

  if (hist_idx < HISTOGRAM_LENGTH) {
    atomicAdd(&(histogram[hist_idx]), histo_tiled[hist_idx]);
  }


}

__global__ void compute_cdf(uint32_t* histogram, float* cdf, int height, int width) {
  __shared__ uint32_t cdf_tiled[HISTOGRAM_LENGTH];

  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int tz = threadIdx.z;

  if (tx < HISTOGRAM_LENGTH) {
    cdf_tiled[tx] = (float) histogram[tx];
  }

  for (int stride = 1; stride < HISTOGRAM_LENGTH; stride *= 2) {
    __syncthreads();
    int cdf_index = (((tx + 1) * stride * 2) - 1);

    if ((cdf_index < HISTOGRAM_LENGTH) && (cdf_index - stride >= 0)) {
      cdf_tiled[cdf_index] = cdf_tiled[cdf_index - stride] + cdf_tiled[cdf_index];
    }
  }

  for (int stride = HISTOGRAM_LENGTH/4; stride > 0; stride /= 2) {
    __syncthreads();
    int cdf_index = ((threadIdx.x + 1) * stride * 2) - 1; 

    if (cdf_index + stride < HISTOGRAM_LENGTH) {
      cdf_tiled[(cdf_index) + stride] = cdf_tiled[(cdf_index)] + cdf_tiled[(cdf_index) + stride];
    }
  }
  __syncthreads();
  
  cdf[threadIdx.x] = (float)(cdf_tiled[threadIdx.x] / ((float)(height * width)));
}


__device__ float min_max(float x, float begin, float end) {
  return min(max(x, begin), end);
}


__global__ void equalize_histogram(unsigned char* input_img, float* cdf, int height, int width) {
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int tz = threadIdx.z;
  
  int row = blockDim.y * blockIdx.y + ty;
  int col = blockDim.x * blockIdx.x + tx;

  if (row < height && col < width) {
    float x = 255 * (cdf[input_img[(((row * width) + col) * blockDim.z) + tz]] - cdf[0])/((float)1 - cdf[0]);
    float start = 0.0f;
    float end = 255.0f; 
    input_img[(((row * width) + col) * blockDim.z) + tz] = min_max(x, start, end);
  }
}

__global__ void char_to_float(unsigned char* input, float* output, int height, int width) {
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int tz = threadIdx.z;

  int row = blockDim.y * blockIdx.y + ty;
  int col = blockDim.x * blockIdx.x + tx;
  
  if (row < height && col < width) {
    output[((row * width + col) * blockDim.z) + tz] = ((float) input[((row * width + col) * blockDim.z) + tz] / ((float) HISTOGRAM_LENGTH - 1));
  }

}

int main(int argc, char **argv) {
  wbArg_t args;
  int imageWidth;
  int imageHeight;
  int imageChannels;
  wbImage_t inputImage;
  wbImage_t outputImage;
  float *hostInputImageData;
  float *hostOutputImageData;
  const char *inputImageFile;

  //@@ Insert more code here
  float* deviceInput;
  float* deviceOutput;
  unsigned char* deviceUchar;
  unsigned char* deviceGray;
  uint32_t* deviceHistogram;
  float* cdf;


  args = wbArg_read(argc, argv); /* parse the input arguments */

  inputImageFile = wbArg_getInputFile(args, 0);

  wbTime_start(Generic, "Importing data and creating memory on host");
  inputImage = wbImport(inputImageFile);
  imageWidth = wbImage_getWidth(inputImage);
  imageHeight = wbImage_getHeight(inputImage);
  imageChannels = wbImage_getChannels(inputImage);
  outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);
  hostInputImageData = wbImage_getData(inputImage);
  hostOutputImageData = wbImage_getData(outputImage);
  wbTime_stop(Generic, "Importing data and creating memory on host");

  //@@ insert code here
  cudaMalloc((void**) &deviceInput, imageChannels * imageHeight * imageWidth * sizeof(float));
  cudaMalloc((void**) &deviceUchar, imageChannels * imageHeight * imageWidth * sizeof(unsigned char));
  cudaMalloc((void**) &deviceGray, imageHeight * imageWidth * sizeof(unsigned char));
  cudaMalloc((void**) &deviceHistogram, HISTOGRAM_LENGTH * sizeof(uint32_t));
  cudaMalloc((void**) &cdf, HISTOGRAM_LENGTH * sizeof(float));
  cudaMalloc((void**) &deviceOutput, imageChannels * imageHeight * imageWidth * sizeof(float));

  cudaMemcpy(deviceInput, hostInputImageData, imageChannels * imageHeight * imageWidth * sizeof(float), cudaMemcpyHostToDevice);

  dim3 DimBlock_float_to_char(16, 16, imageChannels);
  dim3 DimGrid(ceil(((float) imageWidth)/16), ceil(((float) imageHeight)/16), 1);
  float_to_char<<<DimGrid, DimBlock_float_to_char>>>(deviceInput, deviceUchar, imageHeight, imageWidth);

  dim3 DimBlock_rgb_to_grayscale = dim3(16, 16, 1);
  rgb_to_grayscale<<<DimGrid, DimBlock_rgb_to_grayscale>>>(deviceUchar, deviceGray, imageHeight, imageWidth);

  dim3 DimBlock_grayscale_to_histo = dim3(16, 16, 1);
  grayscale_to_histo<<<DimGrid, DimBlock_grayscale_to_histo>>>(deviceGray, deviceHistogram, imageHeight, imageWidth);

  dim3 DimGrid_compute_cdf = dim3(1, 1, 1);
  dim3 DimBlock_compute_cdf = dim3(HISTOGRAM_LENGTH, 1, 1);
  compute_cdf<<<DimGrid_compute_cdf, DimBlock_compute_cdf>>>(deviceHistogram, cdf, imageHeight, imageWidth);

  dim3 DimGrid_equalize_histogram = dim3(ceil(((float) imageWidth)/16), ceil(((float) imageHeight)/16), 1);
  dim3 DimBlock_equalize_histogram = dim3(16, 16, imageChannels);
  equalize_histogram<<<DimGrid_equalize_histogram, DimBlock_equalize_histogram>>>(deviceUchar, cdf, imageHeight, imageWidth);
  
  dim3 DimGrid_char_to_float = dim3(ceil(((float) imageWidth)/16), ceil(((float) imageHeight)/16), 1);
  dim3 DimBlock_char_to_float = dim3(16, 16, imageChannels);  
  char_to_float<<<DimGrid_char_to_float, DimBlock_char_to_float>>>(deviceUchar, deviceOutput, imageHeight, imageWidth);


  cudaDeviceSynchronize();

  cudaMemcpy(hostOutputImageData, deviceOutput, imageWidth * imageHeight * imageChannels * sizeof(float), cudaMemcpyDeviceToHost);


  wbSolution(args, outputImage);

  //@@ insert code here
  cudaFree(deviceInput);
  cudaFree(deviceUchar);
  cudaFree(deviceGray);
  cudaFree(deviceHistogram);
  cudaFree(cdf);
  cudaFree(deviceOutput);

  return 0;
}