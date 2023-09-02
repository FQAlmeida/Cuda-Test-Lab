#include "alloc_image_mem.hpp"

#include "../../../Thread-Pool/src/thread-pool.hpp"

uint16_t* alloc_image(uint32_t image_size) {
    uint16_t* image = (uint16_t*)malloc(sizeof(uint16_t) * image_size * image_size);
    printf("END ALLOC\n");
    ThreadPool threads = ThreadPool{};
    for (uint32_t i = 0; i < image_size; i++) {
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
    return image;
}

uint16_t* alloc_image_out(uint32_t image_size) {
    uint16_t* image_out = (uint16_t*)malloc(sizeof(uint16_t) * image_size * image_size);
    printf("END ALLOC\n");
    for (uint32_t i = 0; i < image_size * image_size; i++) {
        image_out[i] = 0;
    }
    return image_out;
}

uint16_t* alloc_kernel(uint32_t kernel_size) {
    uint16_t* kernel = (uint16_t*)malloc(sizeof(uint16_t*) * kernel_size * kernel_size);
    for (uint32_t i = 0; i < kernel_size; i++) {
        // kernel[i] = (uint16_t*)malloc(sizeof(uint16_t) * kernel_size);
        for (uint32_t j = 0; j < kernel_size; j++) {
            kernel[i + j * kernel_size] = 1;
        }
    }
    return kernel;
}
