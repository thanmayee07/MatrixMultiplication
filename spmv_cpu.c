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
            matrix.values[current_index] = (float)rand() / RAND_MAX * 1000.0f; // Random value between 0 and 1000
            matrix.col_indices[current_index] = rand() % cols; // Random column index
            ++current_index;
        }
    }

    return matrix;
}

// Function to generate a diagonal CSR matrix
csr_matrix generate_diagonal_csr_matrix(int size) {
    csr_matrix matrix;
    matrix.rows_count = size;
    matrix.cols_count = size;
    matrix.non_zero_count = size;
    matrix.values = (float*)malloc(size * sizeof(float));
    matrix.col_indices = (int*)malloc(size * sizeof(int));
    matrix.row_ptr = (int*)malloc((size + 1) * sizeof(int));

    // Generate diagonal matrix with random values
    srand(time(NULL));

    for (int i = 0; i < size; ++i) {
        matrix.values[i] = (float)rand() / RAND_MAX * 1000.0f; // Random value between 0 and 1000
        matrix.col_indices[i] = i; // Diagonal element
        matrix.row_ptr[i] = i; // Diagonal elements are always present
    }
    matrix.row_ptr[size] = size; // Last element in row_ptr

    return matrix;
}

// Function to perform SpMV on CPU (single-threaded)
void cpu_csr_spmv(const csr_matrix* matrix, const float* x, float* y) {
    for (int i = 0; i < matrix->rows_count; ++i) {
        y[i] = 0.0f; // Initialize the result for this row
        for (int j = matrix->row_ptr[i]; j < matrix->row_ptr[i + 1]; ++j) {
            y[i] += matrix->values[j] * x[matrix->col_indices[j]]; // Perform dot product
        }
    }
}

int main() {
    // Define matrix dimensions and density
    int rows = 10000; // Large number of rows
    int cols = 10000; // Large number of columns
    float density = 0.1; // Density of non-zero elements in the matrix (10%)

    // Generate a random CSR matrix
    csr_matrix random_matrix = generate_random_csr_matrix(rows, cols, density);

    // Allocate memory for input and output vectors
    float* x = (float*)malloc(cols * sizeof(float));
    float* y_cpu_random = (float*)malloc(rows * sizeof(float));

    // Check for memory allocation errors
    if (!x || !y_cpu_random) {
        fprintf(stderr, "Memory allocation error\n");
        exit(EXIT_FAILURE);
    }

    // Initialize input vector x (for demonstration, fill it with 1.0)
    for (int i = 0; i < cols; ++i) {
        x[i] = 1.0f;
    }

    // Perform SpMV on CPU for random matrix
    printf("Performance metrics for CPU (single-threaded) random CSR matrix:\n");
    clock_t start_time_random = clock();
    cpu_csr_spmv(&random_matrix, x, y_cpu_random);
    clock_t end_time_random = clock();
    double elapsed_time_random = ((double)(end_time_random - start_time_random)) / CLOCKS_PER_SEC * 1000.0; // Convert to milliseconds

    // Output performance metrics
    printf("Elapsed Time: %f milliseconds\n", elapsed_time_random);

    // Free memory for random matrix
    free(random_matrix.values);
    free(random_matrix.col_indices);
    free(random_matrix.row_ptr);
    free(x);
    free(y_cpu_random);

    // Generate a diagonal CSR matrix
    int size = 10000; // Size of the diagonal matrix
    csr_matrix diagonal_matrix = generate_diagonal_csr_matrix(size);

    // Allocate memory for input and output vectors
    float* x_diag = (float*)malloc(size * sizeof(float));
    float* y_cpu_diag = (float*)malloc(size * sizeof(float));

    // Check for memory allocation errors
    if (!x_diag || !y_cpu_diag) {
        fprintf(stderr, "Memory allocation error\n");
        exit(EXIT_FAILURE);
    }

    // Initialize input vector x for diagonal matrix (for demonstration, fill it with 1.0)
    for (int i = 0; i < size; ++i) {
        x_diag[i] = 1.0f;
    }

    // Perform SpMV on CPU for diagonal matrix
    printf("\nPerformance metrics for CPU (single-threaded) diagonal CSR matrix:\n");
    clock_t start_time_diag = clock();
    cpu_csr_spmv(&diagonal_matrix, x_diag, y_cpu_diag);
    clock_t end_time_diag = clock();
    double elapsed_time_diag = ((double)(end_time_diag - start_time_diag)) / CLOCKS_PER_SEC * 1000.0; // Convert to milliseconds

    // Output performance metrics
    printf("Elapsed Time: %f milliseconds\n", elapsed_time_diag);

    // Free memory for diagonal matrix
    free(diagonal_matrix.values);
    free(diagonal_matrix.col_indices);
    free(diagonal_matrix.row_ptr);
    free(x_diag);
    free(y_cpu_diag);

    return 0;
}
