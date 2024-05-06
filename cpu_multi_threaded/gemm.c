#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

#define N 4096
#define BLOCK_SIZE 32

// Function to perform matrix multiplication using OpenMP for multi-threading
void matrixMulTiled(float *a, float *b, float *c) {
    #pragma omp parallel for collapse(2)
    for(int row = 0; row < N; ++row) {
        for(int col = 0; col < N; ++col) {
            // Accumulator for the dot product
            float sum = 0.0f;

            // Iterate over tiles
            for (int t = 0; t < N / BLOCK_SIZE; ++t) {
                // Compute dot product of tiles
                for (int k = 0; k < BLOCK_SIZE; ++k) {
                    sum += a[row * N + t * BLOCK_SIZE + k] * b[(t * BLOCK_SIZE + k) * N + col];
                }
            }

            // Write the result to global memory
            c[row * N + col] = sum;
        }
    }
}

int main() {
    float *a, *b, *c;
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

    double start_time = omp_get_wtime();

    // Perform matrix multiplication
    matrixMulTiled(a, b, c);

    double end_time = omp_get_wtime();

    // Calculate execution time
    double execution_time = end_time - start_time;

    // Calculate the total number of floating-point operations
    long long total_flops = 2 * (long long)N * (long long)N * (long long)N;

    // Calculate GFLOPS
    double gflops = total_flops / (execution_time * 1e9); // Convert time to seconds and GFLOPS to 1e9 scale

    printf("Time taken: %f seconds\n", execution_time);

    // Free host memory
    free(a);
    free(b);
    free(c);

    return 0;
}

