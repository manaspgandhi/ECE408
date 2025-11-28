#include <wb.h>

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "CUDA error: ", cudaGetErrorString(err));              \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      return -1;                                                          \
    }                                                                     \
  } while (0)

//@@ Define any useful program-wide constants here
#define MASK_WIDTH 3
#define MASK_RADIUS 1
#define TILE_WIDTH 3
#define BLOCK_WIDTH (TILE_WIDTH + MASK_WIDTH - 1)

//@@ Define constant memory for device kernel here
__constant__ float M[MASK_WIDTH][MASK_WIDTH][MASK_WIDTH];

__global__ void conv3d(float *input, float *output, const int z_size,
                       const int y_size, const int x_size) {
  //@@ Insert kernel code here
    float Pvalue = 0;

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tz = threadIdx.z;

    int x_i = blockIdx.x*TILE_WIDTH + tx;
    int y_i = blockIdx.y*TILE_WIDTH + ty;
    int z_i = blockIdx.z*TILE_WIDTH + tz;

    int x_N_start_point = x_i - MASK_RADIUS;
    int y_N_start_point = y_i - MASK_RADIUS;
    int z_N_start_point = z_i - MASK_RADIUS;

    __shared__ float convolve_input[BLOCK_WIDTH][BLOCK_WIDTH][BLOCK_WIDTH];

    // copy data from global memory to shared memory
    if ((x_N_start_point >= 0) && (x_N_start_point < x_size) && (y_N_start_point >= 0) && (y_N_start_point < y_size) && (z_N_start_point >= 0) && (z_N_start_point < z_size)) {
        convolve_input[tz][ty][tx] = input[(x_N_start_point) + (y_N_start_point * (x_size)) + (z_N_start_point * (y_size * x_size))];
    } else {
        convolve_input[tz][ty][tx] = 0.0;
    }

    __syncthreads();

    //convolve second part
    if (tx < TILE_WIDTH && ty < TILE_WIDTH && tz < TILE_WIDTH && x_i < x_size && y_i < y_size && z_i < z_size)
    {
        Pvalue = 0; // reset p val

        for (int i = 0; i < MASK_WIDTH; i++)
            for (int j = 0; j < MASK_WIDTH; j++)
                for (int k = 0; k < MASK_WIDTH; k++)
                    Pvalue += M[i][j][k] * convolve_input[tz + i][ty + j][tx + k];

        output[(x_i) + (y_i * (x_size)) + (z_i * (y_size * x_size))] = Pvalue;
    }
}

int main(int argc, char *argv[]) {
  wbArg_t args;
  int z_size;
  int y_size;
  int x_size;
  int inputLength, kernelLength;
  float *hostInput;
  float *hostKernel;
  float *hostOutput;
  float *deviceInput;
  float *deviceOutput;

  args = wbArg_read(argc, argv);

  // Import data
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &inputLength);
  hostKernel =
      (float *)wbImport(wbArg_getInputFile(args, 1), &kernelLength);
  hostOutput = (float *)malloc(inputLength * sizeof(float));

  // First three elements are the input dimensions
  z_size = hostInput[0];
  y_size = hostInput[1];
  x_size = hostInput[2];
  wbLog(TRACE, "The input size is ", z_size, "x", y_size, "x", x_size);
  assert(z_size * y_size * x_size == inputLength - 3);
  assert(kernelLength == 27);

  wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

  wbTime_start(GPU, "Doing GPU memory allocation");
  //@@ Allocate GPU memory here
  // Recall that inputLength is 3 elements longer than the input data
  // because the first  three elements were the dimensions
  cudaMalloc((void **)&deviceInput, (inputLength - 3) * sizeof(float));
  cudaMalloc((void **)&deviceOutput, (inputLength - 3) * sizeof(float));

  wbTime_stop(GPU, "Doing GPU memory allocation");

  wbTime_start(Copy, "Copying data to the GPU");
  //@@ Copy input and kernel to GPU here
  // Recall that the first three elements of hostInput are dimensions and
  // do
  // not need to be copied to the gpu
  cudaMemcpyToSymbol(M, hostKernel, MASK_WIDTH * MASK_WIDTH * MASK_WIDTH * sizeof(float));
  cudaMemcpy(deviceInput, &hostInput[3], (inputLength - 3) * sizeof(float), cudaMemcpyHostToDevice);

  wbTime_stop(Copy, "Copying data to the GPU");

  wbTime_start(Compute, "Doing the computation on the GPU");
  //@@ Initialize grid and block dimensions here
  dim3 DimGrid(ceil(((float)x_size) / TILE_WIDTH), ceil(((float)y_size) / TILE_WIDTH), ceil(((float)z_size) / TILE_WIDTH));
  dim3 DimBlock(BLOCK_WIDTH, BLOCK_WIDTH, BLOCK_WIDTH);


  //@@ Launch the GPU kernel here
  conv3d<<<DimGrid, DimBlock>>>(deviceInput, deviceOutput, z_size, y_size, x_size);
  
  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Doing the computation on the GPU");

  wbTime_start(Copy, "Copying data from the GPU");
  //@@ Copy the device memory back to the host here
  // Recall that the first three elements of the output are the dimensions
  // and should not be set here (they are set below)
  cudaMemcpy(&hostOutput[3], deviceOutput, (inputLength - 3) * sizeof(float), cudaMemcpyDeviceToHost);

  wbTime_stop(Copy, "Copying data from the GPU");

  wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

  // Set the output dimensions for correctness checking
  hostOutput[0] = z_size;
  hostOutput[1] = y_size;
  hostOutput[2] = x_size;
  wbSolution(args, hostOutput, inputLength);

  // Free device memory
  cudaFree(deviceInput);
  cudaFree(deviceOutput);

  // Free host memory
  free(hostInput);
  free(hostOutput);
  return 0;
}
