%%cuda

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Structure for CSR matrix
typedef struct {
    int rows_count;
    int cols_count;
    int non_zero_count;
    float* values;
    int* col_indices;
    int* row_ptr;
} csr_matrix;

// Function to generate a random CSR matrix
csr_matrix generate_random_csr_matrix(int rows, int cols, float density) {
    csr_matrix matrix;
    matrix.rows_count = rows;
    matrix.cols_count = cols;
    matrix.row_ptr = (int*)malloc((rows + 1) * sizeof(int));

    // Generate random values for the matrix
    srand(time(NULL));
    int max_non_zeros = (int)(density * rows * cols);
    matrix.non_zero_count = max_non_zeros;
    matrix.values = (float*)malloc(max_non_zeros * sizeof(float));
    matrix.col_indices = (int*)malloc(max_non_zeros * sizeof(int));

    int current_index = 0;
    matrix.row_ptr[0] = 0;
    for (int i = 0; i < rows; ++i) {
        int non_zeros_in_row = (int)(density * cols); // Determine number of non-zeros in this row
        matrix.row_ptr[i + 1] = matrix.row_ptr[i] + non_zeros_in_row; // Update row_ptr
        for (int j = 0; j < non_zeros_in_row; ++j) {
            matrix.values[current_index] = (float)(rand() % 1000 + 1); // Random value between 1 and 1000
            matrix.col_indices[current_index] = rand() % cols; // Random column index
            ++current_index;
        }
    }

    return matrix;
}

// CUDA kernel for SpMV computation
__global__ void csr_spmv_kernel(int rows_count, int* row_ptr, int* col_indices, float* values, float* x, float* y) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows_count) {
        float dot = 0.0f;
        for (int j = row_ptr[row]; j < row_ptr[row + 1]; ++j) {
            dot += values[j] * x[col_indices[j]];
        }
        y[row] = dot;
    }
}

// Function to perform SpMV on GPU and measure performance
void gpu_csr_spmv_perf(const csr_matrix* matrix, const float* x, float* y) {
    // Allocate memory on GPU
    float *d_values, *d_x, *d_y;
    int *d_col_indices, *d_row_ptr;
    cudaMalloc((void**)&d_values, matrix->non_zero_count * sizeof(float));
    cudaMalloc((void**)&d_col_indices, matrix->non_zero_count * sizeof(int));
    cudaMalloc((void**)&d_row_ptr, (matrix->rows_count + 1) * sizeof(int));
    cudaMalloc((void**)&d_x, matrix->cols_count * sizeof(float));
    cudaMalloc((void**)&d_y, matrix->rows_count * sizeof(float));

    // Copy data to GPU
    cudaMemcpy(d_values, matrix->values, matrix->non_zero_count * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_indices, matrix->col_indices, matrix->non_zero_count * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_row_ptr, matrix->row_ptr, (matrix->rows_count + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, x, matrix->cols_count * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    int threads_per_block = 256;
    int blocks_per_grid = (matrix->rows_count + threads_per_block - 1) / threads_per_block;
    csr_spmv_kernel<<<blocks_per_grid, threads_per_block>>>(matrix->rows_count, d_row_ptr, d_col_indices, d_values, d_x, d_y);

    // Copy result back to CPU
    cudaMemcpy(y, d_y, matrix->rows_count * sizeof(float), cudaMemcpyDeviceToHost);

    // Free GPU memory
    cudaFree(d_values);
    cudaFree(d_col_indices);
    cudaFree(d_row_ptr);
    cudaFree(d_x);
    cudaFree(d_y);
}

int main() {
    // Define matrix dimensions and density
    int rows = 10000; // Large number of rows
    int cols = 10000; // Large number of columns
    float density = 0.1; // Density of non-zero elements in the matrix (1%)

    // Generate a random CSR matrix
    csr_matrix matrix = generate_random_csr_matrix(rows, cols, density);

    // Allocate memory for input and output vectors
    float* x = (float*)malloc(cols * sizeof(float));
    float* y = (float*)malloc(rows * sizeof(float));

    // Initialize input vector x 
    for (int i = 0; i < cols; ++i) {
        x[i] = 1.0f;
    }

    // Perform SpMV on GPU and measure performance
    clock_t start_time = clock();
    gpu_csr_spmv_perf(&matrix, x, y);
    clock_t end_time = clock();

    // Calculate elapsed time
    double elapsed_seconds = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;

    // Calculate total number of floating-point operations
    double total_operations = (double)matrix.non_zero_count * 2; // Assuming one multiplication and one addition per non-zero element

    // Calculate GFLOPS
    double gflops = (total_operations / elapsed_seconds) / 1e9; // Divide by elapsed time in seconds and 1 billion

    // Output performance metrics
    printf("Elapsed Time: %f seconds\n", elapsed_seconds);
    printf("GFLOPS: %f\n", gflops);

    // Free memory
    free(matrix.values);
    free(matrix.col_indices);
    free(matrix.row_ptr);
    free(x);
    free(y);

    return 0;
}
