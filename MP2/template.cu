
#include <wb.h>

const int TILE_WIDTH = 16;

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

// Compute C = A * B
__global__ void matrixMultiply(float *A, float *B, float *C, int numARows,
                               int numAColumns, int numBRows,
                               int numBColumns, int numCRows,
                               int numCColumns) {
  //@@ Insert code to implement matrix multiplication here
  //after performing multipliaction, we need to call sync threads because all threads go at their own pace
  
  __shared__ float global_A[16][16];
  __shared__ float global_B[16][16];
  
  
  int row = blockIdx.y * TILE_WIDTH + threadIdx.y;

  int col = blockIdx.x * TILE_WIDTH + threadIdx.x;

  float sum = 0.0;

  for(int m = 0; m < (numAColumns - 1)/ TILE_WIDTH + 1; m++){
    if((row < numARows) && ((m * TILE_WIDTH + threadIdx.x) < numAColumns)){
      global_A[threadIdx.y][threadIdx.x] = A[row * numAColumns + m * TILE_WIDTH + threadIdx.x];
    }
    else {
      global_A[threadIdx.y][threadIdx.x] = 0.0;
    }

    if((col < numBColumns) && ((m * TILE_WIDTH + threadIdx.y) < numBRows)){
      global_B[threadIdx.y][threadIdx.x] = B[(m * TILE_WIDTH + threadIdx.y) * (numBColumns) + col];
    }
    else {
      global_B[threadIdx.y][threadIdx.x] = 0.0;
    }

    __syncthreads();

    for(int k = 0; k < TILE_WIDTH; k++){
      sum += global_A[threadIdx.y][k] * global_B[k][threadIdx.x];
    }

    __syncthreads();

    if((row < numCRows) && (col < numCColumns)){
      C[row * numCColumns + col] = sum;
    }
  }

}

int main(int argc, char **argv) {
  wbArg_t args;
  float *hostA; // The A matrix
  float *hostB; // The B matrix
  float *hostC; // The output C matrix
  float *deviceA;
  float *deviceB;
  float *deviceC;
  int numARows;    // number of rows in the matrix A
  int numAColumns; // number of columns in the matrix A
  int numBRows;    // number of rows in the matrix B
  int numBColumns; // number of columns in the matrix B
  int numCRows;    // number of rows in the matrix C (you have to set this)
  int numCColumns; // number of columns in the matrix C (you have to set
                   // this)

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostA = (float *)wbImport(wbArg_getInputFile(args, 0), &numARows,
                            &numAColumns);
  hostB = (float *)wbImport(wbArg_getInputFile(args, 1), &numBRows,
                            &numBColumns);
  //@@ Set numCRows and numCColumns
  numCRows = numARows;
  numCColumns = numBColumns;
  //@@ Allocate the hostC matrix
  hostC = (float*) malloc(sizeof(float) * numCRows * numCColumns);

  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The dimensions of A are ", numARows, " x ", numAColumns);
  wbLog(TRACE, "The dimensions of B are ", numBRows, " x ", numBColumns);

  wbTime_start(GPU, "Allocating GPU memory.");
  //@@ Allocate GPU memory here
  cudaMalloc((void **) &deviceA, sizeof(float) * numARows * numAColumns);
  cudaMalloc((void **) &deviceB, sizeof(float) * numBRows * numBColumns);
  cudaMalloc((void **) &deviceC, sizeof(float) * numCRows * numCColumns);

  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  //@@ Copy memory to the GPU here
  cudaMemcpy(deviceA, hostA, sizeof(float) * numARows * numAColumns, cudaMemcpyHostToDevice);
  cudaMemcpy(deviceB, hostB, sizeof(float) * numBRows * numBColumns, cudaMemcpyHostToDevice);

  wbTime_stop(GPU, "Copying input memory to the GPU.");

  //@@ Initialize the grid and block dimensions here
  dim3 block(TILE_WIDTH, TILE_WIDTH);
  dim3 grid(ceil(numCColumns/((float) TILE_WIDTH)), ceil(numCRows/((float) TILE_WIDTH)));

  wbTime_start(Compute, "Performing CUDA computation");
  //@@ Launch the GPU Kernel here
  matrixMultiply<<<grid, block>>>(deviceA, deviceB, deviceC, numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns);

  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  //@@ Copy the GPU memory back to the CPU here
  cudaMemcpy(hostC, deviceC, sizeof(float) * numCRows * numCColumns, cudaMemcpyDeviceToHost);

  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbTime_start(GPU, "Freeing GPU Memory");
  //@@ Free the GPU memory here
  cudaFree(deviceA);
  cudaFree(deviceB);
  cudaFree(deviceC);

  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, hostC, numCRows, numCColumns);

  free(hostA);
  free(hostB);
  free(hostC);

  return 0;
}
