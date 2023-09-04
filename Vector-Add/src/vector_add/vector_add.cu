#include <assert.h>
#include <cuda_runtime.h>
#include <inttypes.h>
#include <math.h>
#include <stdio.h>

#include <iostream>

#include "../alloc_vector_mem/alloc_vector_mem.hpp"
#include "vector_add.cuh"

__global__ void vector_add(float* vector_a, float* vector_b, float* vector_out, uint32_t vector_size) {
    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < vector_size) {
        vector_out[tid] = vector_a[tid] + vector_b[tid] + 0.0f;
        printf("%d %.2f\n",tid, vector_out[tid]);
    }
}

void run_vector_add() {
    uint32_t vector_size = 100000;
    int threadsPerBlock = 256;
    int blocksPerGrid = (vector_size + threadsPerBlock - 1) / threadsPerBlock;
    dim3 grid(blocksPerGrid);
    dim3 block(threadsPerBlock);

    float* vector_a = alloc_vector(vector_size);
    float* vector_b = alloc_vector(vector_size);
    float* vector_c = alloc_vector(vector_size);

    float* vector_a_device;
    float* vector_b_device;
    float* vector_c_device;

    cudaMalloc(&vector_a_device, sizeof(float) * vector_size);
    cudaMemcpy(vector_a_device, vector_a, sizeof(float) * vector_size, cudaMemcpyHostToDevice);
    cudaMalloc(&vector_b_device, sizeof(float) * vector_size);
    cudaMemcpy(vector_b_device, vector_b, sizeof(float) * vector_size, cudaMemcpyHostToDevice);
    cudaMalloc(&vector_c_device, sizeof(float) * vector_size);

    vector_add<<<grid, block>>>(vector_a_device, vector_b_device, vector_c_device, vector_size);
    cudaError_t err = cudaGetLastError();
    cudaDeviceSynchronize();

    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    err = cudaMemcpy(vector_c, vector_c_device, sizeof(float) * vector_size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy vector C from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    for (size_t i = 0; i < vector_size; i++) {
        // printf("%.2f %.2f %.2f %.2f %.2f\n", vector_a[i], vector_b[i], vector_c[i], vector_a[i] + vector_b[i],
        //        vector_c[i] - vector_a[i] - vector_b[i]);
        assert(fabs(vector_c[i] - vector_a[i] - vector_b[i]) < 1e-5);
    }
}
