# MatrixMultiplication
Optimizing GEMM and SpMV using Parallel Hardware

# Features
- Implementation of GEMM and SpMV algorithms for both CPU and GPU architectures.
- Performance measurement metrics such as execution time, GFLOPS, and throughput.
- Multi-threaded CPU implementations utilizing OpenMP for parallelization.
- GPU implementations utilizing CUDA and shared memory for accelerated computation.

# Requirements
- C/C++ compiler with support for OpenMP (for CPU implementations).
- NVIDIA GPU with CUDA support (for GPU implementations).
- Google Colab environment for running CUDA programs.

# CPU Implementations
### Naive GEMM (General Matrix Multiplication) on CPU
The naive GEMM implementation performs matrix multiplication on the CPU using a simple triple-loop approach. It multiplies two square matrices of size N and measures the execution time.

### Naive SpMV (Sparse Matrix-Vector Multiplication) on CPU
The naive SpMV implementation performs sparse matrix-vector multiplication on the CPU using a simple double-loop approach. It multiplies a randomly generated sparse matrix with a vector and measures the execution time.

### Multi-threaded GEMM (General Matrix Multiplication) on CPU
The multi-threaded GEMM implementation utilizes OpenMP for parallelization. It performs matrix multiplication on the CPU using a tiled approach and measures the execution time.

### Multi-threaded SpMV (Sparse Matrix-Vector Multiplication) on CPU
The multi-threaded SpMV implementation utilizes OpenMP for parallelization. It performs sparse matrix-vector multiplication on the CPU using memory coalescing and measures the execution time.

# GPU Implementations
### Matrix Multiplication (GEMM) on GPU
This CUDA implementation utilizes shared memory for tiled matrix multiplication.

### Sparse Matrix-Vector Multiplication (SpMV) on GPU
This CUDA implementation optimizes SpMV using memory coalescing and shared memory.


