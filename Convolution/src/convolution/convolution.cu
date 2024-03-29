#include <assert.h>
#include <cuda_runtime.h>
#include <inttypes.h>
#include <math.h>
#include <stdio.h>

#include <iostream>

#include "../alloc_image_mem/alloc_image_mem.hpp"
#include "convolution.cuh"
#include "convolution_commom.hpp"
#include "cuda_common.cuh"

__device__ void convolute(float* image, float* kernel, float* sum_out, uint32_t image_size, uint32_t kernel_size, uint32_t x, uint32_t y,
                          uint32_t padding, uint32_t line_size) {
    *sum_out = 0;
    for (uint32_t desloc_i = 0; desloc_i < kernel_size; desloc_i++) {
        for (uint32_t desloc_j = 0; desloc_j < kernel_size; desloc_j++) {
            uint32_t image_desloc = (x - padding + desloc_i) + (line_size * (y - padding + desloc_j));
            float pixel = image[image_desloc];
            float kernel_pixel = kernel[desloc_j + kernel_size * desloc_i];
            *sum_out += pixel * kernel_pixel;
        }
    }
}

__global__ void par_convolution(float* image, float* kernel, float* out, uint32_t image_size, uint32_t kernel_size) {
    extern __shared__ float shared_kernel[];
    uint32_t shared_kernel_size = kernel_size * kernel_size;

    uint32_t tid = threadIdx.x * blockDim.x + threadIdx.y;

    if (tid < shared_kernel_size) {
        shared_kernel[tid] = kernel[tid];
    }

    __syncthreads();

    uint32_t padding = (kernel_size - 1) / 2;

    uint32_t x = threadIdx.x + blockDim.x * blockIdx.x + padding;
    uint32_t y = threadIdx.y + blockDim.y * blockIdx.y + padding;

    uint32_t line_size = blockDim.x * gridDim.x + 2 * padding;

    uint32_t gid = x + (y * line_size);

    // printf("%d %d %d\n", tid, x, y);

    float sum = 0;
    convolute(image, shared_kernel, &sum, image_size, kernel_size, x, y, padding, line_size);
    // assert(sum > 0);
    // printf("%f\n", sum);
    out[gid] = sum / (float)(kernel_size * kernel_size);
}

__global__ void checker(float* image, float* image_out, uint32_t image_size) {
    for (size_t i = 1; i < image_size - 1; i++) {
        for (size_t j = 1; j < image_size - 1; j++) {
            assert(image[j + i * image_size] != image_out[j + i * image_size]);
            // printf("%f %f\n", image[i + j * image_size], image_out[i + j * image_size]);
        }
    }
}

void convolution_loop(float* image, uint32_t image_size, float* kernel, uint32_t kernel_size, float* out, uint32_t qtd_loops) {
    uint32_t padding = (kernel_size - 1) / 2;
    // TODO(Otavio): Create a better logic for grid and block dims size
    // Make it in a way that (image_size - 2 * padding) is ways divisible
    // Aka, all convuluted pixels should be processed, no more no less
    dim3 block(32, 32);
    dim3 grid((image_size - 2 * padding) / 32, (image_size - 2 * padding) / 32);

    float *image_in_device, *image_out_device, *kernel_device;

    gpuErrchk(cudaMalloc(&image_in_device, sizeof(float) * image_size * image_size));
    gpuErrchk(cudaMemcpy(image_in_device, image, image_size * image_size * sizeof(float), cudaMemcpyHostToDevice));

    gpuErrchk(cudaMalloc(&image_out_device, sizeof(float) * image_size * image_size));
    cudaMemcpy(image_out_device, image_in_device, image_size * image_size * sizeof(float), cudaMemcpyDeviceToDevice);

    gpuErrchk(cudaMalloc(&kernel_device, sizeof(float) * kernel_size * kernel_size));
    gpuErrchk(cudaMemcpy(kernel_device, kernel, kernel_size * kernel_size * sizeof(float), cudaMemcpyHostToDevice));

    float* aux_image = image_in_device;

    for (size_t i = 0; i < qtd_loops; i++) {
        par_convolution<<<grid, block, sizeof(float) * kernel_size * kernel_size>>>(aux_image, kernel_device, image_out_device, image_size,
                                                                                    kernel_size);
        gpuErrchk(cudaGetLastError());
        aux_image = image_out_device;
    }

    gpuErrchk(cudaDeviceSynchronize());

    // checker<<<1, 1>>>(image_in_device, image_out_device, image_size);
    // gpuErrchk(cudaGetLastError());
    // gpuErrchk(cudaDeviceSynchronize());

    gpuErrchk(cudaMemcpy(out, image_out_device, image_size * image_size * sizeof(float), cudaMemcpyDeviceToHost));

    gpuErrchk(cudaFree(image_in_device));
    gpuErrchk(cudaFree(image_out_device));
    gpuErrchk(cudaFree(kernel_device));

    gpuErrchk(cudaDeviceSynchronize());
    gpuErrchk(cudaDeviceReset());
}

float* run_convolution(uint32_t n, uint32_t qtd_loops, uint32_t padding) {
    // uint32_t padding = (kernel_size - 1) / 2;
    uint32_t kernel_size = padding * 2 + 1;
    // uint32_t n = 32;
    uint32_t image_size = 32 * n + 2 * padding;

    srand(345);
    // printf("STARTING CREATION\n");

    float* image = alloc_image(image_size);
    float* image_out = alloc_image_out(image_size);
    memcpy(image_out, image, image_size * image_size * sizeof(float));

    float* kernel = alloc_kernel(kernel_size);

    // save_matrix(image, image_size, image_size, "data/convolution_matrix.txt");
    // printf("STARTING CONV\n");
    convolution_loop(image, image_size, kernel, kernel_size, image_out, qtd_loops);
    // printf("END CONV\n");

    // show_matrix(image, image_size, 15);
    // printf("----------------------------------------------\n");
    // show_matrix(image_out, image_size, 15);

    // save_matrix(image_out, image_size, 15, "data/convolution_matrix_out_gpu.txt");

    free(image);
    free(kernel);

    return image_out;
}
