// OPTIMIZATION 5: Using Streams to overlap computation with data transfer (4 points)

#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"

#define TILE_WIDTH 16

__global__ void conv_forward_kernel(float *output, const float *input, const float *mask, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
{
    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.

    Function paramter definitions:
    output - output
    input - input
    mask - convolution kernel
    B - batch_size (number of images in x)
    M - number of output feature maps
    C - number of input feature maps
    H - input height dimension
    W - input width dimension
    K - kernel height and width (K x K)
    S - stride step length
    */

    const int H_out = (H - K)/S + 1;
    const int W_out = (W - K)/S + 1;

    #define out_4d(i3, i2, i1, i0) output[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
    #define in_4d(i3, i2, i1, i0) input[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
    #define mask_4d(i3, i2, i1, i0) mask[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    // Insert your GPU convolution kernel code here

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int bz = blockIdx.z;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
        
    int W_size = ceil((float)(W)/((1.0) * TILE_WIDTH)); 
    int h = (by / W_size) * TILE_WIDTH + ty;
    int w = (by % W_size) * TILE_WIDTH + tx;
    
    int b = blockIdx.x;
    int m = blockIdx.y;
    
    float conv_result = 0.0f;
    
    for(int c = 0; c < C; c++){
        for(int p = 0; p < K; p++){
            for(int q = 0; q < K; q++) {
                if (((h * S + p < H) && (w * S + q < W))){
                    conv_result += in_4d(bz, c, h * S + p, w * S + q) * mask_4d(b, c, p, q);
                }
            }
        }
    }

    if(h < H_out && w < W_out)
        out_4d(bz, b, h, w) = conv_result;


    #undef out_4d
    #undef in_4d
    #undef mask_4d
}
	
__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
{
//     // Allocate memory and copy over the relevant data structures to the GPU

//     // We pass double pointers for you to initialize the relevant device pointers,
//     //  which are passed to the other two functions.

//     // Useful snippet for error checking
//     // cudaError_t error = cudaGetLastError();
//     // if(error != cudaSuccess)
//     // {
//     //     std::cout<<"CUDA error: "<<cudaGetErrorString(error)<<std::endl;
//     //     exit(-1);
//     // }

    const int W_out = (W-K)/S + 1;
    const int H_out = (H-K)/S + 1;
    int inputSize = B * C * H * W/B;
    int outputSize = B*M*((H - K)/S + 1)*((W - K)/S + 1)/B; 
    
    cudaMalloc((void **) device_mask_ptr, M * C * K * K * sizeof(float));
    cudaMalloc((void **) device_input_ptr, B * C * H * W * sizeof(float));
    cudaMalloc((void **) device_output_ptr, B * M * H_out * W_out * sizeof(float));

    cudaMemcpy(*device_input_ptr, host_input, B * C * H * W * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(*device_mask_ptr, host_mask, M * C * K * K * sizeof(float), cudaMemcpyHostToDevice);

    dim3 dimGrid(M, ceil(H/((1.0) * TILE_WIDTH)) * ceil(W/((1.0) * TILE_WIDTH)), 1); //z is just B/B
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);

    // OPTIMIZATION 5: Everything below is the optimization

    // 5000 streams
    cudaStream_t streams[5000];

    // First create all the streams
    for(int i = 0; i < B; i++){ //number of streams = B
        // This creates and allocates the memory for the streams
        cudaStreamCreate(&streams[i]);

        // Asynchronously memcopy
        cudaMemcpyAsync((void*)(*device_input_ptr + (i * inputSize)), (void*)(host_input + i * inputSize), (B*C*H*W/B) * sizeof(float), cudaMemcpyHostToDevice, streams[i]);
    }

    // Here, we declare all the kernels
    for(int j = 0; j < B; j++){
        conv_forward_kernel<<<dimGrid, dimBlock, 0, streams[j]>>>((*device_output_ptr + j * outputSize), (*device_input_ptr + j * inputSize), *device_mask_ptr, B, M, C, H, W, K, S);
    }

    // Here,we allocate memory for the output (should be done asynchronously)
    for(int k = 0; k < B; k++){
        // Copy the output back to host
        //cudaMemcpy(host_output, *device_output_ptr, B*M*((H - K)/S + 1)*((W - K)/S + 1)*sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpyAsync((void *)(host_output + k * outputSize), (void*)(*device_output_ptr + k * outputSize), (B*M*((H - K)/S + 1)*((W - K)/S + 1)/B) * sizeof(float), cudaMemcpyDeviceToHost, streams[k]);
    }

    // Here, we destroy all the streams because we are done queuing
    for(int l = 0; l < B; l++){
        cudaStreamDestroy(streams[l]);
    }
}


__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
{
    // Just need to return, we did the allocation earlier
    return;
}


__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
{
    // Just need to free
    cudaFree(device_output);
    cudaFree(device_input);
    cudaFree(device_mask);
}


__host__ void GPUInterface::get_device_properties()
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for(int dev = 0; dev < deviceCount; dev++)
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        std::cout<<"Device "<<dev<<" name: "<<deviceProp.name<<std::endl;
        std::cout<<"Computational capabilities: "<<deviceProp.major<<"."<<deviceProp.minor<<std::endl;
        std::cout<<"Max Global memory size: "<<deviceProp.totalGlobalMem<<std::endl;
        std::cout<<"Max Constant memory size: "<<deviceProp.totalConstMem<<std::endl;
        std::cout<<"Max Shared memory size per block: "<<deviceProp.sharedMemPerBlock<<std::endl;
        std::cout<<"Max threads per block: "<<deviceProp.maxThreadsPerBlock<<std::endl;
        std::cout<<"Max block dimensions: "<<deviceProp.maxThreadsDim[0]<<" x, "<<deviceProp.maxThreadsDim[1]<<" y, "<<deviceProp.maxThreadsDim[2]<<" z"<<std::endl;
        std::cout<<"Max grid dimensions: "<<deviceProp.maxGridSize[0]<<" x, "<<deviceProp.maxGridSize[1]<<" y, "<<deviceProp.maxGridSize[2]<<" z"<<std::endl;
        std::cout<<"Warp Size: "<<deviceProp.warpSize<<std::endl;
    }
}
