#include <cuda_runtime.h>
#include <stdio.h>

#include <iostream>

__global__ void hello_cuda() { printf("Hello Cuda\n"); }

int main(int argc, char const *argv[]) {
    dim3 block(2, 1, 1);
    dim3 grid(2, 1, 1);
    hello_cuda<<<grid, block>>>();

    std::cout << "Hello" << std::endl;

    cudaDeviceSynchronize();
    cudaDeviceReset();

    return 0;
}
