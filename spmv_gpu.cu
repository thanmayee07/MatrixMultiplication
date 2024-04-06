%%cuda

#include <stdio.h>
#include <sys/time.h>

#define ITER 1000

// Sparse Matrix-Vector Multiplication kernel for CSR format
template <typename T>
__global__ void spmv_csr_kernel(T *values, int *rowPtr, int *colIndices, T *x, T *y, int numRows) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < numRows) {
        T dot = 0.0f;
        int row_start = rowPtr[row];
        int row_end = rowPtr[row + 1];
        for (int i = row_start; i < row_end; ++i) {
            dot += values[i] * x[colIndices[i]];
        }
        y[row] = dot;
    }
}

// Main SpMV function
template <typename T>
void spmv_csr(T *values, int *rowPtr, int *colIndices, T *x, T *y, int numRows, int numCols, int nnz) {
    int numThreadsPerBlock = 256;
    int numBlocks = (numRows + numThreadsPerBlock - 1) / numThreadsPerBlock;

    spmv_csr_kernel<<<numBlocks, numThreadsPerBlock>>>(values, rowPtr, colIndices, x, y, numRows);
}

int main() {
    // Define matrix dimensions and allocate memory
    int numRows = 2;
    int numCols = 2;
    int nnz = 3;

    // Allocate memory for matrix in host
    float *values_host = (float*)malloc(nnz * sizeof(float));
    int *rowPtr_host = (int*)malloc((numRows + 1) * sizeof(int));
    int *colIndices_host = (int*)malloc(nnz * sizeof(int));

    // Allocate memory for vectors in host
    float *x_host = (float*)malloc(numCols * sizeof(float));
    float *y_host = (float*)malloc(numRows * sizeof(float));

    // Initialize matrix and vectors (for demonstration purposes)
    values_host[0] = 1.0f;
    values_host[1] = 2.0f;
    values_host[2] = 3.0f;

    rowPtr_host[0] = 0;
    rowPtr_host[1] = 1;
    rowPtr_host[2] = 3;

    colIndices_host[0] = 0;
    colIndices_host[1] = 1;
    colIndices_host[2] = 0;

    x_host[0] = 1.0f;
    x_host[1] = 2.0f;

    // Allocate memory for matrix and vectors in device
    float *values_device;
    int *rowPtr_device;
    int *colIndices_device;
    float *x_device;
    float *y_device;
    cudaMalloc(&values_device, nnz * sizeof(float));
    cudaMalloc(&rowPtr_device, (numRows + 1) * sizeof(int));
    cudaMalloc(&colIndices_device, nnz * sizeof(int));
    cudaMalloc(&x_device, numCols * sizeof(float));
    cudaMalloc(&y_device, numRows * sizeof(float));

    // Copy matrix and vectors from host to device
    cudaMemcpy(values_device, values_host, nnz * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(rowPtr_device, rowPtr_host, (numRows + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(colIndices_device, colIndices_host, nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(x_device, x_host, numCols * sizeof(float), cudaMemcpyHostToDevice);

    // Run SpMV and measure time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds = 0;
    cudaEventRecord(start);
    for (int i = 0; i < ITER; i++)
        spmv_csr(values_device, rowPtr_device, colIndices_device, x_device, y_device, numRows, numCols, nnz);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Copy result vector from device to host
    cudaMemcpy(y_host, y_device, numRows * sizeof(float), cudaMemcpyDeviceToHost);

    // Check for CUDA errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(error));
    }

    // Calculate performance metrics
    double gflop = 2 * (double) nnz / 1e9;
    double gbs = ((nnz * sizeof(float)) + ((numRows + 1) * sizeof(int)) + (nnz * sizeof(int)) + (numCols * sizeof(float)) + (numRows * sizeof(float))) / (milliseconds / ITER) / 1e6;
    double time_taken = (milliseconds / ITER) / 1000.0;

    // Print result (for demonstration purposes)
    printf("Result of SpMV operation:\n");
    for (int i = 0; i < numRows; ++i) {
        printf("%f\n", y_host[i]);
    }

    // Print performance metrics
    printf("\nPerformance Metrics:\n");
    printf("Average time taken for SpMV is %f seconds\n", time_taken);
    printf("Average GFLOP/s is %lf\n", gflop / time_taken);
    printf("Average GB/s is %lf\n", gbs);

    // Free memory in device
    cudaFree(values_device);
    cudaFree(rowPtr_device);
    cudaFree(colIndices_device);
    cudaFree(x_device);
    cudaFree(y_device);

    // Free memory in host
    free(values_host);
    free(rowPtr_host);
    free(colIndices_host);
    free(x_host);
    free(y_host);

    return 0;
}
