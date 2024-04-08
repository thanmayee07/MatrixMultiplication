#include <stdio.h>
#include <stdlib.h>
#include <time.h>

typedef struct {
    int rows_count;
    int cols_count;
    int non_zero_count;
    float* values;
    int* col_indices;
    int* row_ptr;
} csr_matrix;

typedef struct {
    char* name;
    double elapsed_time;
} measurement_class;

measurement_class cpu_csr_spmv_single_thread_naive(const csr_matrix* matrix, float* x, float* y) {
    // Fill the input vector x with appropriate values (for demonstration, we fill it with 1.0)
    for (int i = 0; i < matrix->cols_count; ++i) {
        x[i] = 1.0f;
    }

    clock_t start_time = clock();

    for (int row = 0; row < matrix->rows_count; ++row) {
        float dot = 0.0f;
        for (int j = matrix->row_ptr[row]; j < matrix->row_ptr[row + 1]; ++j) {
            dot += matrix->values[j] * x[matrix->col_indices[j]];
        }
        y[row] = dot;
    }

    clock_t end_time = clock();
    double elapsed_seconds = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;

    measurement_class result;
    result.name = "Naive Single-Threaded CSR SpMV";
    result.elapsed_time = elapsed_seconds;

    return result;
}

int main() {
    // Define matrix dimensions and allocate memory
    csr_matrix matrix;
    matrix.rows_count = 3;
    matrix.cols_count = 3;
    matrix.non_zero_count = 4;
    matrix.values = (float*)malloc(matrix.non_zero_count * sizeof(float));
    matrix.col_indices = (int*)malloc(matrix.non_zero_count * sizeof(int));
    matrix.row_ptr = (int*)malloc((matrix.rows_count + 1) * sizeof(int));

    // Initialize matrix values (for demonstration)
    matrix.values[0] = 1.0f;
    matrix.values[1] = 2.0f;
    matrix.values[2] = 3.0f;
    matrix.values[3] = 4.0f;

    matrix.col_indices[0] = 0;
    matrix.col_indices[1] = 1;
    matrix.col_indices[2] = 0;
    matrix.col_indices[3] = 2;

    matrix.row_ptr[0] = 0;
    matrix.row_ptr[1] = 2;
    matrix.row_ptr[2] = 3;
    matrix.row_ptr[3] = 4;

    // Allocate memory for vectors
    float* x = (float*)malloc(matrix.cols_count * sizeof(float));
    float* y = (float*)malloc(matrix.rows_count * sizeof(float));

    // Perform SpMV operation
    measurement_class result = cpu_csr_spmv_single_thread_naive(&matrix, x, y);

    // Output the result
    printf("Elapsed Time: %f seconds\n", result.elapsed_time);

    // Clean up memory
    free(matrix.values);
    free(matrix.col_indices);
    free(matrix.row_ptr);
    free(x);
    free(y);

    return 0;
}
