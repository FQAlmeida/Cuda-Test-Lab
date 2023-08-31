#include <assert.h>
#include <cuda_runtime.h>
#include <inttypes.h>
#include <stdio.h>

#include <iostream>

__device__ void convolute(uint16_t* image, uint16_t* kernel, uint16_t* sum_out, uint32_t image_size,
                          uint32_t kernel_size, uint32_t x, uint32_t y, uint32_t padding, uint32_t line_size) {
    *sum_out = 0;
    for (uint32_t desloc_i = 0; desloc_i < kernel_size; desloc_i++) {
        for (uint32_t desloc_j = 0; desloc_j < kernel_size; desloc_j++) {
            uint32_t image_desloc = (x - padding + desloc_i) + ((y - padding + desloc_j) * line_size);
            // printf("%u\n", image_desloc);
            uint16_t pixel = image[image_desloc];
            uint16_t kernel_pixel = kernel[desloc_i + kernel_size * desloc_j];
            if (x == 1 && y == 1) {
                // printf("%lf %lf\n", image[0], kernel[0]);
            }
            *sum_out += pixel * kernel_pixel;
        }
    }
}

__global__ void par_convolution(uint16_t* image, uint16_t* kernel, uint16_t* out, uint32_t image_size,
                                uint32_t kernel_size) {
    uint32_t padding = (kernel_size - 1) / 2;

    uint32_t x = threadIdx.x + blockDim.x * blockIdx.x + padding;
    uint32_t y = threadIdx.y + blockDim.y * blockIdx.y + padding;

    // printf("%u %u %u\n", x, y, padding);

    uint32_t line_size = blockDim.x * gridDim.x + 2 * padding;

    uint32_t tid = x + (y * line_size);

    uint16_t sum = 0;
    convolute(image, kernel, &sum, image_size, kernel_size, x, y, padding, line_size);
    assert(sum > 0);
    out[tid] = sum / (uint16_t)(kernel_size * kernel_size);
    assert(*(out + tid) == sum / (uint16_t)(kernel_size * kernel_size));
}

void convolution(uint16_t* image, uint32_t image_size, uint16_t* kernel, uint32_t kernel_size, uint16_t* out) {
    uint32_t padding = (kernel_size - 1) / 2;

    // TODO(Otavio): Create a better logic for grid and block dims size
    // Make it in a way that (image_size - 2 * padding) is ways divisible
    // Aka, all convuluted pixels should be processed, no more no less
    dim3 grid(4, 4);
    dim3 block((image_size - 2 * padding) / 4, (image_size - 2 * padding) / 4);

    uint16_t *image_in_device, *image_out_device, *kernel_device;

    cudaMalloc(&image_in_device, sizeof(uint16_t) * image_size * image_size);
    // uint16_t* cache = (uint16_t*)malloc(sizeof(uint16_t) * image_size * image_size);
    // for (uint32_t i = 0; i < image_size; i++) {
    //     memcpy(cache + (i * image_size), *(image + i), image_size * sizeof(uint16_t));
    // }
    cudaMemcpy(image_in_device, image, image_size * image_size * sizeof(uint16_t), cudaMemcpyHostToDevice);

    cudaMalloc(&image_out_device, sizeof(uint16_t) * image_size * image_size);
    cudaMemcpy(image_out_device, image_in_device, image_size * image_size * sizeof(uint16_t), cudaMemcpyDeviceToDevice);
    // for (uint32_t i = 0; i < image_size; i++) {
    //     cudaMemcpy(image_out_device + (i * image_size), *(image + i), image_size * sizeof(uint32_t),
    //                cudaMemcpyHostToDevice);
    // }

    // uint16_t* cache_kernel = (uint16_t*)malloc(sizeof(uint16_t) * kernel_size * kernel_size);
    cudaMalloc(&kernel_device, sizeof(uint16_t) * kernel_size * kernel_size);
    // for (uint32_t i = 0; i < kernel_size; i++) {
    //     memcpy(cache_kernel + (i * kernel_size), *(kernel + i), kernel_size * sizeof(uint16_t));
    // }
    // for (size_t i = 0; i < kernel_size; i++) {
    //     for (size_t j = 0; j < kernel_size; j++) {
    //         printf("%lf\n", *(cache_kernel + i + j));
    //     }
    // }
    cudaError err =
        cudaMemcpy(kernel_device, kernel, kernel_size * kernel_size * sizeof(uint16_t), cudaMemcpyHostToDevice);

    for (size_t i = 0; i < 10000000; i++) {
        par_convolution<<<grid, block>>>(image_in_device, kernel_device, image_out_device, image_size, kernel_size);
    }
    cudaDeviceSynchronize();

    cudaMemcpy(out, image_out_device, image_size * image_size * sizeof(uint16_t), cudaMemcpyDeviceToHost);

    // for (uint32_t i = 0; i < image_size; i++) {
    //     memcpy(*(out + i), cache + (i * image_size), image_size * sizeof(uint16_t));
    // }
    // free(cache_kernel);
    // free(cache);
    cudaDeviceReset();
}

void show_matrix(uint16_t** matrix, uint32_t size) {
    for (uint32_t i = 0; i < size; i++) {
        for (uint32_t j = 0; j < size; j++) {
            printf("%.2u\t|", matrix[i][j]);
        }
        putchar('\n');
    }
}

int main(int argc, char const* argv[]) {
    uint32_t kernel_size = 100;
    uint32_t padding = (kernel_size - 1) / 2;
    uint32_t n = 10000;
    uint32_t image_size = 4 * n + 2 * padding;

    srand(345);
    printf("STARTING CREATION\n");

    uint16_t* image = (uint16_t*)malloc(sizeof(uint16_t) * image_size * image_size);
    printf("END ALLOC\n");
    for (uint32_t i = 0; i < image_size; i++) {
        // image[i] = (uint16_t*)malloc(sizeof(uint16_t) * image_size);
        for (uint32_t j = 0; j < image_size; j++) {
            image[i + j * image_size] = (uint16_t)(i + 1);
        }
    }
    for (uint32_t i = 0; i < image_size; i++) {
        image[i] = 0;
        image[image_size * (image_size - 1) + i] = 0;
        image[image_size * i] = 0;
        image[image_size * i + (image_size - 1)] = 0;
    }
    printf("END CREATION\n");

    uint16_t* image_out = (uint16_t*)malloc(sizeof(uint16_t) * image_size * image_size);
    printf("END ALLOC\n");
    for (uint32_t i = 0; i < image_size * image_size; i++) {
        // image_out[i] = (uint16_t*)malloc(sizeof(uint16_t) * image_size);
        // for (uint32_t j = 0; j < image_size; j++) {
        image_out[i] = 0;
        // }
    }

    uint16_t* kernel = (uint16_t*)malloc(sizeof(uint16_t*) * kernel_size * kernel_size);
    for (uint32_t i = 0; i < kernel_size; i++) {
        // kernel[i] = (uint16_t*)malloc(sizeof(uint16_t) * kernel_size);
        for (uint32_t j = 0; j < kernel_size; j++) {
            kernel[i + j * kernel_size] = 1;
        }
    }
    // kernel[padding][padding] = 1;

    // show_matrix(image, image_size);
    // show_matrix(kernel, kernel_size);
    printf("STARTING CONV\n");
    convolution(image, image_size, kernel, kernel_size, image_out);
    printf("END CONV\n");
    // printf("\n");
    // show_matrix(image, image_size);
    // printf("\n");
    // show_matrix(image_out, image_size);
    // printf("%lf", *(*(image_out + 10) + 100));

    // for (uint32_t i = 0; i < image_size; i++) {
    //     free(image[i]);
    // }
    free(image);

    // for (uint32_t i = 0; i < image_size; i++) {
    //     free(image_out[i]);
    // }
    free(image_out);

    // for (uint32_t i = 0; i < kernel_size; i++) {
    //     free(kernel[i]);
    // }
    free(kernel);

    return 0;
}
