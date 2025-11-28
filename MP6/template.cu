// MP Scan
// Given a list (lst) of length n
// Output its prefix sum = {lst[0], lst[0] + lst[1], lst[0] + lst[1] + ...
// +
// lst[n-1]}

#include <wb.h>

#define BLOCK_SIZE 512 //@@ You can change this

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

__global__ void scan(float *input, float *output, int len, float *Sum) {
  //@@ Modify the body of this function to complete the functionality of
  //@@ the scan on the device
  //@@ You may need multiple kernel calls; write your kernels before this
  //@@ function and call them from the host
  __shared__ float T[2*BLOCK_SIZE];
  int tx = threadIdx.x;
  int i = 2 * blockIdx.x * blockDim.x + tx;
  
  if (i < len) T[tx] = input[i];
  if (i+blockDim.x < len) T[tx+blockDim.x] = input[i+blockDim.x];

  for(int stride = 1; stride < 2 * BLOCK_SIZE; stride = stride * 2) {
    __syncthreads();
    int index = (threadIdx.x + 1) * stride * 2 - 1;
    if(index < 2 * BLOCK_SIZE && (index - stride) >= 0)
      T[index] += T[index - stride];
  }
  
  for(stride = BLOCK_SIZE/2; stride > 0; stride = stride/2) {
    __syncthreads();
    int index = (threadIdx.x + 1) * stride * 2 - 1;
    if ((index + stride) < 2 * BLOCK_SIZE)
      T[index + stride] += T[index]; 
  }
  __syncthreads();
  if(i < len){
    output[i] = T[threadIdx.x];
  }

  if(i + blockDim.x < len){
    output[i + blockDim.x] = T[threadIdx.x + blockDim.x];
  }

  __syncthreads();
  if(threadIdx.x == blockDim.x - 1){
    Sum[blockIdx.x] = T[2 * BLOCK_SIZE - 1];
  }
}

__global__ void scan2(float *input, float *output, int len) {
  __shared__ float T[2 * BLOCK_SIZE];

  int tx = threadIdx.x;
  int i = 2 * blockIdx.x * blockDim.x + tx;

  // Load data to shared memory with boundary checks
  T[tx] = (i < len) ? input[i] : 0;
  T[tx + blockDim.x] = (i + blockDim.x < len) ? input[i + blockDim.x] : 0;

  // Inclusive Hillis-Steele scan
  // Upsweep phase: Build the sum in a tree-like reduction
  for (int stride = 1; stride <= BLOCK_SIZE; stride *= 2) {
    __syncthreads();
    int index = (tx + 1) * stride * 2 - 1;
    if (index < 2 * BLOCK_SIZE) {
      T[index] += T[index - stride];
    }
  }

  // Downsweep phase: Build the scan using the sum from the reduction
  for (int stride = BLOCK_SIZE / 2; stride > 0; stride /= 2) {
    __syncthreads();
    int index = (tx + 1) * stride * 2 - 1;
    if (index + stride < 2 * BLOCK_SIZE) {
      T[index + stride] += T[index];
    }
  }
  __syncthreads();

  // Write back results to global memory with boundary checks
  if (i < len) {
    output[i] = T[tx];
  }
  if (i + blockDim.x < len) {
    output[i + blockDim.x] = T[tx + blockDim.x];
  }
}


__global__ void sumUp(float *S, float *Y, int len) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  float baseValue = (blockIdx.x > 0) ? S[blockIdx.x - 1] : 0;    if (i < len) {
      Y[i] += baseValue;
  }
}

int main(int argc, char **argv) {
  wbArg_t args;
  float *hostInput;  // The input 1D list
  float *hostOutput; // The output list
  float *deviceInput;
  float *deviceOutput;
  int numElements; // number of elements in the list

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &numElements);
  hostOutput = (float *)malloc(numElements * sizeof(float));
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The number of input elements in the input is ",
        numElements);

  wbTime_start(GPU, "Allocating GPU memory.");
  wbCheck(cudaMalloc((void **)&deviceInput, numElements * sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceOutput, numElements * sizeof(float)));
  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Clearing output memory.");
  wbCheck(cudaMemset(deviceOutput, 0, numElements * sizeof(float)));
  wbTime_stop(GPU, "Clearing output memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  wbCheck(cudaMemcpy(deviceInput, hostInput, numElements * sizeof(float),
                     cudaMemcpyHostToDevice));
  wbTime_stop(GPU, "Copying input memory to the GPU.");

  //@@ Initialize the grid and block dimensions here
  float *sums; 
  cudaMalloc((void **) &sums, ceil(numElements/(BLOCK_SIZE)*2.0) * sizeof(float));

  dim3 DimBlock(BLOCK_SIZE, 1, 1);
  dim3 DimGrid(ceil(numElements/(BLOCK_SIZE*2.0)), 1, 1);
  scan<<<DimGrid, DimBlock>>>(deviceInput, deviceOutput, numElements, sums);
  cudaDeviceSynchronize();
  
  dim3 DimGrid2(1, 1, 1);
  scan2<<<DimGrid2, DimBlock>>>(sums, sums, ceil(numElements/(BLOCK_SIZE * 2)*1.0));
  cudaDeviceSynchronize();

  dim3 DimBlock2(BLOCK_SIZE * 2, 1, 1);
  dim3 DimGrid3(ceil(numElements/(BLOCK_SIZE*2.0)), 1, 1);
  sumUp<<<DimGrid3, DimBlock2>>>(sums, deviceOutput, numElements);
  cudaDeviceSynchronize();

  wbTime_start(Compute, "Performing CUDA computation");
  //@@ Modify this to complete the functionality of the scan
  //@@ on the deivce

  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  wbCheck(cudaMemcpy(hostOutput, deviceOutput, numElements * sizeof(float),
                     cudaMemcpyDeviceToHost));
  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbTime_start(GPU, "Freeing GPU Memory");
  cudaFree(deviceInput);
  cudaFree(deviceOutput);
  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, hostOutput, numElements);

  free(hostInput);
  free(hostOutput);

  return 0;
}
