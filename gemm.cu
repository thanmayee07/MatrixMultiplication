%%cuda

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

#define N 256
#define BLOCK_SIZE 16

__global__ void matrixMul(float *a, float *b, float *c) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float sum = 0.0f;
        for (int i = 0; i < N; ++i) {
            sum += a[row * N + i] * b[i * N + col];
        }
        c[row * N + col] = sum;
    }
}

int main() {
    float *a, *b, *c;
    float *d_a, *d_b, *d_c;
    int size = N * N * sizeof(float);

    // Allocate host memory
    a = (float *)malloc(size);
    b = (float *)malloc(size);
    c = (float *)malloc(size);

    // Initialize host matrices a and b
    for(int i = 0; i < N; ++i) {
        for(int j = 0; j < N; ++j) {
            a[i * N + j] = i + j;
            b[i * N + j] = i - j;
        }
    }

    // Allocate device memory
    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c, size);

    // Copy host matrices to device
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    // Set grid and block dimensions
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 numBlocks((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Launch kernel
    matrixMul<<<numBlocks, threadsPerBlock>>>(d_a, d_b, d_c);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Calculate the total number of floating-point operations
    long long total_flops = 2 * (long long)N * (long long)N * (long long)N;

    // Convert time to seconds
    double seconds = milliseconds / 1000.0;

    // Calculate GFLOPS
    double gflops = total_flops / (seconds * 1e9); // Convert time to seconds and GFLOPS to 1e9 scale

    printf("Time taken: %f milliseconds\n", milliseconds);
    printf("GFLOPS: %f\n", gflops);

    // Copy result from device to host
    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    // Free host memory
    free(a);
    free(b);
    free(c);

    return 0;
}
