#include <cuda_runtime.h>
#include <inttypes.h>
#include <stdio.h>

#include <iostream>

__global__ void par_convolution(uint32_t* image, uint32_t* kernel,
                                uint32_t* out, uint32_t image_size,
                                uint32_t kernel_size) {
    // uint32_t tid = (blockDim.x * blockDim.y * gridDim.x * blockIdx.y) +
    //    (gridDim.x * blockDim.x * threadIdx.y) +
    //    (blockDim.x * blockIdx.x) + threadIdx.x;

    uint32_t padding = (kernel_size - 1) / 2;

    uint32_t x = threadIdx.x + blockDim.x * blockIdx.x + padding;
    uint32_t y = threadIdx.y + blockDim.y * blockIdx.y + padding;

    uint32_t line_size = blockDim.x * gridDim.x + 2 * padding;
    // uint32_t col_size = blockDim.y * gridDim.y;

    uint32_t tid = x + (y * line_size);

    // if (x <= 0 || x >= line_size - 1) {
    //     return;
    // } else if (y <= 0 || y >= col_size - 1) {
    //     return;
    // }
    // printf("(%u %u) %u\n", y, x, tid);
    uint32_t sum = 0;
    for (uint32_t desloc_i = 0; desloc_i < kernel_size; desloc_i++) {
        for (uint32_t desloc_j = 0; desloc_j < kernel_size; desloc_j++) {
            uint32_t image_desloc = (x - padding + desloc_i) +
                                    ((y - padding + desloc_j) * line_size);

            uint32_t pixel = image[image_desloc];
            uint32_t kernel_pixel = kernel[desloc_i + kernel_size * desloc_j];
            // if (pixel == 0 && kernel_pixel == 1) {
            //     printf("(%u, %u)\n", (x - padding + desloc_i),
            //            (y - padding + desloc_j));
            // }
            // printf("(%u %u) (%u %u)\t|\n", (x - padding + desloc_i),
            //    (y - padding + desloc_j), desloc_i, desloc_j);
            sum += pixel * kernel_pixel;
        }
    }
    // printf("%d\n", sum / (kernel_size * kernel_size));
    out[tid] = sum / (kernel_size * kernel_size);

    // printf("%d %u %u %u %u (%u %u)\n", tid, threadIdx.x, threadIdx.y,
    //        blockIdx.x, blockIdx.y, x, y);
}

void convolution(uint32_t** image, uint32_t image_size, uint32_t** kernel,
                 uint32_t kernel_size, uint32_t** out) {
    uint32_t padding = (kernel_size - 1) / 2;

    // TODO(Otavio): Create a better logic for grid and block dims size
    // Make it in a way that (image_size - 2 * padding) is ways divisible
    // Aka, all convuluted pixels should be processed, no more no less
    dim3 grid(4, 4);
    dim3 block((image_size - 2 * padding) / 4, (image_size - 2 * padding) / 4);

    uint32_t *image_in_device, *image_out_device, *kernel_device;

    cudaMalloc((void**)&image_in_device,
               sizeof(uint32_t) * image_size * image_size);
    for (uint32_t i = 0; i < image_size; i++) {
        cudaMemcpy(image_in_device + (i * image_size), *(image + i),
                   image_size * sizeof(uint32_t), cudaMemcpyHostToDevice);
    }

    cudaMalloc(&image_out_device, sizeof(uint32_t) * image_size * image_size);
    for (uint32_t i = 0; i < image_size; i++) {
        cudaMemcpy(image_out_device + (i * image_size), *(image + i),
                   image_size * sizeof(uint32_t), cudaMemcpyHostToDevice);
    }

    cudaMalloc(&kernel_device, sizeof(uint32_t) * kernel_size * kernel_size);
    for (uint32_t i = 0; i < kernel_size; i++) {
        cudaMemcpy(kernel_device + (i * kernel_size), *(kernel + i),
                   kernel_size * sizeof(uint32_t), cudaMemcpyHostToDevice);
    }

    par_convolution<<<grid, block>>>(image_in_device, kernel_device,
                                     image_out_device, image_size, kernel_size);
    cudaDeviceSynchronize();

    for (uint32_t i = 0; i < image_size; i++) {
        cudaMemcpy(*(out + i), image_out_device + (i * image_size),
                   image_size * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    }

    cudaDeviceReset();
}

void show_matrix(uint32_t** matrix, uint32_t size) {
    for (uint32_t i = 0; i < size; i++) {
        for (uint32_t j = 0; j < size; j++) {
            printf("%u\t|", matrix[i][j]);
        }
        putchar('\n');
    }
}

int main(int argc, char const* argv[]) {
    uint32_t kernel_size = 3;
    uint32_t padding = (kernel_size - 1) / 2;
    uint32_t n = 10;
    uint32_t image_size = 4 * n + 2 * padding;

    uint32_t** image = (uint32_t**)malloc(sizeof(uint32_t*) * image_size);
    for (uint32_t i = 0; i < image_size; i++) {
        image[i] = (uint32_t*)malloc(sizeof(uint32_t) * image_size);
        for (uint32_t j = 0; j < image_size; j++) {
            image[i][j] = i;
        }
    }
    for (uint32_t i = 0; i < image_size; i++) {
        image[i][0] = 0;
        image[i][image_size - 1] = 0;
        image[0][i] = 0;
        image[image_size - 1][i] = 0;
    }

    uint32_t** image_out = (uint32_t**)malloc(sizeof(uint32_t*) * image_size);
    for (uint32_t i = 0; i < image_size; i++) {
        image_out[i] = (uint32_t*)malloc(sizeof(uint32_t) * image_size);
        for (uint32_t j = 0; j < image_size; j++) {
            image_out[i][j] = 0;
        }
    }

    uint32_t** kernel = (uint32_t**)malloc(sizeof(uint32_t*) * kernel_size);
    for (uint32_t i = 0; i < kernel_size; i++) {
        kernel[i] = (uint32_t*)malloc(sizeof(uint32_t) * kernel_size);
        for (uint32_t j = 0; j < kernel_size; j++) {
            kernel[i][j] = 1;
        }
    }
    kernel[padding][padding] = 1;

    // show_matrix(image, image_size);
    // show_matrix(kernel, kernel_size);

    convolution(image, image_size, kernel, kernel_size, image_out);
    printf("\n");
    show_matrix(image_out, 15);
    // printf("%u", *(*(image_out + 10) + 100));

    for (uint32_t i = 0; i < image_size; i++) {
        free(image[i]);
    }
    free(image);

    for (uint32_t i = 0; i < image_size; i++) {
        free(image_out[i]);
    }
    free(image_out);

    for (uint32_t i = 0; i < kernel_size; i++) {
        free(kernel[i]);
    }
    free(kernel);

    return 0;
}
