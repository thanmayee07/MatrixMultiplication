#include <stdio.h>
#include <stdlib.h>

// Structure to represent a sparse matrix in CSR format
typedef struct {
    int num_rows;
    int num_cols;
    int nnz; // Number of non-zero elements
    float *values; // Array of non-zero values
    int *row_ptr; // Row pointers
    int *col_indices; // Column indices
} CSRMatrix;

// Function to perform SpMV using CSR format on CPU
void spmv_cpu(const CSRMatrix *matrix, const float *vector, float *result) {
    for (int i = 0; i < matrix->num_rows; ++i) {
        float sum = 0.0f;
        for (int j = matrix->row_ptr[i]; j < matrix->row_ptr[i + 1]; ++j) {
            sum += matrix->values[j] * vector[matrix->col_indices[j]];
        }
        result[i] = sum;
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

    matrix.values = values;
    matrix.row_ptr = row_ptr;
    matrix.col_indices = col_indices;

    float vector[] = {1.0f, 2.0f, 3.0f};
    float result[3];

    // Perform SpMV on CPU
    spmv_cpu(&matrix, vector, result);

    // Output the result
    for (int i = 0; i < matrix.num_rows; ++i) {
        printf("%f\n", result[i]);
    }

    return 0;
}
