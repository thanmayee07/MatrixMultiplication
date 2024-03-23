%%cuda

#include <stdio.h>

// Structure to represent a sparse matrix in CSR format
typedef struct {
    int num_rows;
    int num_cols;
    int nnz; // Number of non-zero elements
    float *values; // Array of non-zero values
    int *row_ptr; // Row pointers
    int *col_indices; // Column indices
} CSRMatrix;

// GPU kernel for SpMV using CSR format
__global__ void spmv_gpu(const CSRMatrix matrix, const float *vector, float *result) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < matrix.num_rows) {
        float sum = 0.0f;
        for (int j = matrix.row_ptr[row]; j < matrix.row_ptr[row + 1]; ++j) {
            sum += matrix.values[j] * vector[matrix.col_indices[j]];
        }
        result[row] = sum;
    }
}

int main() {
    // Example usage
    CSRMatrix matrix;
    matrix.num_rows = 3;
    matrix.num_cols = 3;
    matrix.nnz = 4;

    // Example CSR data (values, row_ptr, col_indices)
    float values[] = {1.0f, 2.0f, 3.0f, 4.0f};
    int row_ptr[] = {0, 2, 3, 4};
    int col_indices[] = {0, 1, 1, 2};

    cudaMallocManaged(&matrix.values, matrix.nnz * sizeof(float));
    cudaMallocManaged(&matrix.row_ptr, (matrix.num_rows + 1) * sizeof(int));
    cudaMallocManaged(&matrix.col_indices, matrix.nnz * sizeof(int));

    cudaMemcpy(matrix.values, values, matrix.nnz * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(matrix.row_ptr, row_ptr, (matrix.num_rows + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(matrix.col_indices, col_indices, matrix.nnz * sizeof(int), cudaMemcpyHostToDevice);

    float vector[] = {1.0f, 2.0f, 3.0f};
    float *result;
    cudaMallocManaged(&result, matrix.num_rows * sizeof(float));

    // Define block and grid dimensions
    int block_size = 16;
    int num_blocks = (matrix.num_rows + block_size - 1) / block_size;

    // Launch GPU kernel
    spmv_gpu<<<num_blocks, block_size>>>(matrix, vector, result);
    cudaDeviceSynchronize();

    // Output the result
    for (int i = 0; i < matrix.num_rows; ++i) {
        printf("%f\n", result[i]);
    }

    // Free allocated memory
    cudaFree(matrix.values);
    cudaFree(matrix.row_ptr);
    cudaFree(matrix.col_indices);
    cudaFree(result);

    return 0;
}
