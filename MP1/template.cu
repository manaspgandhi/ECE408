// MP 1
#include <wb.h>

//global = initialize kernel, GPU kernel code
//out = output vector, len = length of vector
__global__ void vecAdd(float *in1, float *in2, float *out, int len) {
  //@@ Insert code to implement vector addition here
  int curr_thread = blockIdx.x * blockDim.x + threadIdx.x;
  if(curr_thread < len){
    out[curr_thread] = in1[curr_thread] + in2[curr_thread];
  }
}

int main(int argc, char **argv) {
  wbArg_t args;
  int inputLength;
  float *hostInput1;
  float *hostInput2;
  float *hostOutput;
  float *deviceInput1;
  float *deviceInput2;
  float *deviceOutput;

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostInput1 =
      (float *)wbImport(wbArg_getInputFile(args, 0), &inputLength);
  hostInput2 =
      (float *)wbImport(wbArg_getInputFile(args, 1), &inputLength);
  hostOutput = (float *)malloc(inputLength * sizeof(float));


 // variables
  int size = inputLength * sizeof(float);
  
  /* Transfer A and B to device memory */

  // what is happening here is that host (user) provides some inputs,
  cudaMalloc((void **) &deviceInput1, size);
  cudaMemcpy(deviceInput1, hostInput1, size, cudaMemcpyHostToDevice);
  cudaMalloc((void **) &deviceInput2, size);
  cudaMemcpy(deviceInput2, hostInput2, size, cudaMemcpyHostToDevice);

  // Allocate device memory for
  cudaMalloc((void **) &deviceOutput, size);
  

  /* Kernel Invocation Code */
  dim3 DimGrid(ceil(inputLength/256.0), 1, 1);
  dim3 DimBlock(256, 1, 1);

  vecAdd<<<DimGrid,DimBlock>>>(deviceInput1, deviceInput2, deviceOutput, inputLength);

  /* Transfer C from device to host */
  // we now calculated it in the device, transfer from device to host
  cudaMemcpy(hostOutput, deviceOutput, size, cudaMemcpyDeviceToHost);

  // Free device memory for A, B, C
  cudaFree(deviceInput1); cudaFree(deviceInput2); cudaFree (deviceOutput);



  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The input length is ", inputLength);

  wbTime_start(GPU, "Allocating GPU memory.");
  //@@ Allocate GPU memory here

  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  //@@ Copy memory to the GPU here

  wbTime_stop(GPU, "Copying input memory to the GPU.");

  //@@ Initialize the grid and block dimensions here

  wbTime_start(Compute, "Performing CUDA computation");
  //@@ Launch the GPU Kernel here

  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  //@@ Copy the GPU memory back to the CPU here

  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbTime_start(GPU, "Freeing GPU Memory");
  //@@ Free the GPU memory here

  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, hostOutput, inputLength);

  free(hostInput1);
  free(hostInput2);
  free(hostOutput);

  return 0;
}
