#include <cuda_runtime.h>
#include <inttypes.h>
#include <stdio.h>

__global__ void par_convolution(uint32_t* image, uint32_t* kernel,
                                uint32_t** out, uint32_t image_size,
                                uint32_t kernel_size) {
    uint32_t tid = (blockDim.x * blockDim.y * gridDim.x * blockIdx.y) +
                   (gridDim.x * blockDim.x * threadIdx.y) +
                   (blockDim.x * blockIdx.x) + threadIdx.x;
    printf("%d %u %u %u %u\n", tid, threadIdx.x, threadIdx.y, blockIdx.x,
           blockIdx.y);
}

void convolution(uint8_t** image, uint32_t image_size, uint8_t** kernel,
                 uint32_t kernel_size, uint8_t*** out) {
    dim3 grid(4, 4);
    dim3 block(image_size / 4, image_size / 4);

    uint32_t *image_in_device, *image_out_device, *kernel_device;

    cudaMalloc((void**)&image_in_device,
               sizeof(uint8_t) * image_size * image_size);
    for (uint32_t i = 0; i < image_size; i++) {
        cudaMemcpy(image_in_device + (i * image_size), image + i, image_size,
                   cudaMemcpyHostToDevice);
    }

    cudaMalloc(&image_out_device, sizeof(uint8_t*) * image_size * image_size);
    for (uint32_t i = 0; i < image_size; i++) {
        cudaMemcpy(image_out_device + (i * image_size), image + i, image_size,
                   cudaMemcpyHostToDevice);
    }

    cudaMalloc(&kernel_device, sizeof(uint8_t*) * kernel_size * kernel_size);
    for (uint32_t i = 0; i < kernel_size; i++) {
        cudaMemcpy(kernel_device + (i * kernel_size), kernel + i, kernel_size,
                   cudaMemcpyHostToDevice);
    }

    par_convolution<<<grid, block>>>(image_in_device, kernel_device,
                                     &image_out_device, image_size,
                                     kernel_size);
    for (uint32_t i = 0; i < image_size; i++) {
        cudaMemcpy((*out) + i, image_out_device + (i * image_size), image_size,
                   cudaMemcpyDeviceToHost);
    }

    cudaDeviceSynchronize();
    cudaDeviceReset();
}

void show_matrix(uint8_t** matrix, uint32_t size) {
    for (uint32_t i = 0; i < size; i++) {
        for (uint32_t j = 0; j < size; j++) {
            printf("%u\t|", matrix[i][j]);
        }
        putchar('\n');
    }
}

int main(int argc, char const* argv[]) {
    uint32_t image_size = 16;
    uint8_t** image = (uint8_t**)malloc(sizeof(uint8_t*) * image_size);
    for (uint32_t i = 0; i < image_size; i++) {
        image[i] = (uint8_t*)malloc(sizeof(uint8_t) * image_size);
        for (uint32_t j = 0; j < image_size; j++) {
            image[i][j] = 0;
        }
    }

    uint8_t** image_out = (uint8_t**)malloc(sizeof(uint8_t*) * image_size);
    for (uint32_t i = 0; i < image_size; i++) {
        image_out[i] = (uint8_t*)malloc(sizeof(uint8_t) * image_size);
        for (uint32_t j = 0; j < image_size; j++) {
            image_out[i][j] = 0;
        }
    }

    uint32_t kernel_size = 3;
    uint8_t** kernel = (uint8_t**)malloc(sizeof(uint8_t*) * kernel_size);
    for (uint32_t i = 0; i < kernel_size; i++) {
        kernel[i] = (uint8_t*)malloc(sizeof(uint8_t) * kernel_size);
        for (uint32_t j = 0; j < kernel_size; j++) {
            kernel[i][j] = 1;
        }
    }

    show_matrix(image, image_size);
    show_matrix(kernel, kernel_size);

    convolution(image, image_size, kernel, kernel_size, &image_out);

    show_matrix(image_out, image_size);

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
