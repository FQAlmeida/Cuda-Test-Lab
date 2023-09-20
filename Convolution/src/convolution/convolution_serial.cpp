#include "convolution_serial.hpp"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../alloc_image_mem/alloc_image_mem.hpp"
#include "convolution_commom.hpp"

void serial_convolution(float* image, float* kernel, float* out, uint32_t image_size, uint32_t kernel_size) {
    uint32_t padding = (kernel_size - 1) / 2;
    for (size_t i = padding; i < image_size - padding; i++) {
        for (size_t j = padding; j < image_size - padding; j++) {
            float sum = 0;
            for (size_t i_desloc = 0; i_desloc < kernel_size; i_desloc++) {
                for (size_t j_desloc = 0; j_desloc < kernel_size; j_desloc++) {
                    sum += image[(j - padding + j_desloc) + (i - padding + i_desloc) * image_size] *
                           kernel[i_desloc * kernel_size + j_desloc];
                }
            }
            out[j + i * image_size] = sum / (kernel_size * kernel_size);
        }
    }
}

void serial_convolution_loop(float* image, uint32_t image_size, float* kernel, uint32_t kernel_size, float* out, uint32_t qtd_loops) {
    uint32_t padding = (kernel_size - 1) / 2;

    for (size_t i = 0; i < qtd_loops; i++) {
        for (size_t j = 0; j < 1; j++) {
            serial_convolution(image, kernel, out, image_size, kernel_size);
            memcpy(image, out, sizeof(float) * image_size * image_size);
        }
    }

    // checker<<<1, 1>>>(image_in_device, image_out_device, image_size);
    // gpuErrchk(cudaGetLastError());
    // gpuErrchk(cudaDeviceSynchronize());
}

float* run_convolution_serial(uint32_t n, uint32_t qtd_loops) {
    uint32_t kernel_size = 3;
    uint32_t padding = (kernel_size - 1) / 2;
    // uint32_t n = 32;
    uint32_t image_size = 32 * n + 2 * padding;

    srand(345);

    float* image = alloc_image(image_size);
    float* image_out = alloc_image_out(image_size);
    float* kernel = alloc_kernel(kernel_size);
    // show_matrix(image, image_size, 15);

    // save_matrix(image, image_size, image_size, "data/convolution_matrix.txt");
    serial_convolution_loop(image, image_size, kernel, kernel_size, image_out, qtd_loops);

    // show_matrix(image_out, image_size, 15);

    // save_matrix(image_out, image_size, 15, "data/convolution_matrix_out_cpu.txt");

    free(image);
    free(kernel);

    return image_out;
}