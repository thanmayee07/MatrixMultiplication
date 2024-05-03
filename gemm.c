#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 256 // Define the size of the square matrices

// Function to perform matrix multiplication on CPU
void matrixMul(float *a, float *b, float *c) {
    for (int row = 0; row < N; ++row) {
        for (int col = 0; col < N; ++col) {
            float sum = 0.0f;
            // Iterate over each element of the row in 'a' and column in 'b'
            for (int i = 0; i < N; ++i) {
                // Accumulate the product of corresponding elements
                sum += a[row * N + i] * b[i * N + col];
            }
            // Store the sum in the corresponding position of matrix 'c'
            c[row * N + col] = sum;
        }
    }
}

int main() {
    float *a, *b, *c;
    int size = N * N * sizeof(float);

    // Allocate memory for matrices 'a', 'b', and 'c'
    a = (float *)malloc(size);
    if (a == NULL) {
        fprintf(stderr, "Memory allocation failed for matrix 'a'\n");
        exit(EXIT_FAILURE);
    }
    b = (float *)malloc(size);
    if (b == NULL) {
        fprintf(stderr, "Memory allocation failed for matrix 'b'\n");
        free(a);
        exit(EXIT_FAILURE);
    }
    c = (float *)malloc(size);
    if (c == NULL) {
        fprintf(stderr, "Memory allocation failed for matrix 'c'\n");
        free(a);
        free(b);
        exit(EXIT_FAILURE);
    }

    // Initialize matrices 'a' and 'b'
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            // Assign values to elements of matrices 'a' and 'b'
            a[i * N + j] = i + j;
            b[i * N + j] = i - j;
        }
    }

    // Calculate total memory usage
    int total_memory_usage = 3 * size;
    printf("Total memory usage: %d bytes\n", total_memory_usage);

    clock_t start = clock(); // Record the starting time

    // Perform matrix multiplication on CPU
    matrixMul(a, b, c);

    clock_t stop = clock(); // Record the stopping time
    double cpu_time = ((double)(stop - start)) / CLOCKS_PER_SEC * 1000.0; // Calculate the elapsed time in milliseconds

    // Print the time taken by CPU for matrix multiplication
    printf("Time taken by CPU: %f milliseconds\n", cpu_time);

    // Free allocated memory for matrices
    free(a);
    free(b);
    free(c);

    return 0;
}
