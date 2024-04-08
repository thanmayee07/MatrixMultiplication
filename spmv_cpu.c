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

// Function to generate a random CSR matrix with large dimensions and increased density
csr_matrix generate_large_random_csr_matrix(int rows, int cols, float density) {
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

// Naive single-threaded CSR SpMV implementation
measurement_class cpu_csr_spmv_single_thread_naive(const csr_matrix* matrix, float* x, float* y) {
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
    // Define matrix dimensions and density
    int rows = 10000; // Larger number of rows
    int cols = 10000; // Larger number of columns
    float density = 0.1; // Density of non-zero elements in the matrix (10%)

    // Generate a large random CSR matrix with increased density
    csr_matrix matrix = generate_large_random_csr_matrix(rows, cols, density);

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
