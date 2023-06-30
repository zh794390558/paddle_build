#include <cassert>
#include <chrono>
#include <cstdint>
#include <functional>
#include <iomanip>
#include <iostream>
#include <tuple>
#include <utility>
#include <vector>
#include <algorithm> // std::min(initializer_list<T> il)

#include <cublas_v2.h>
#include <cuda_runtime.h>

// nvcc cublas_mm.cu -lcublas

#define CHECK_CUBLAS_ERROR(val) checkCuBlas((val), #val, __FILE__, __LINE__)
template <typename T>
void checkCuBlas(T err, const char* const func, const char* const file,
                 const int line)
{
    if (err != CUBLAS_STATUS_SUCCESS)
    {
        std::cerr << "cuBlas Runtime Error at: " << file << ":" << line
                  << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

#define CHECK_CUDA_ERROR(val) checkCuda((val), #val, __FILE__, __LINE__)
template <typename T>
void checkCuda(T err, const char* const func, const char* const file,
               const int line)
{
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                  << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

#define CHECK_LAST_CUDA_ERROR() checkCudaLast(__FILE__, __LINE__)
void checkCudaLast(const char* const file, const int line)
{
    cudaError_t err{cudaGetLastError()};
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                  << std::endl;
        std::cerr << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

float measure_cublas_performance(
    std::function<cublasStatus_t(void)> bound_cublas_function,
    cudaStream_t stream, int num_repeats = 100, int num_warmups = 100)
{
    cudaEvent_t start, stop;
    float time;

    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));

    for (int i{0}; i < num_warmups; ++i)
    {
        CHECK_CUBLAS_ERROR(bound_cublas_function());
    }

    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));

    CHECK_CUDA_ERROR(cudaEventRecord(start, stream));
    for (int i{0}; i < num_repeats; ++i)
    {
        CHECK_CUBLAS_ERROR(bound_cublas_function());
    }
    CHECK_CUDA_ERROR(cudaEventRecord(stop, stream));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    CHECK_LAST_CUDA_ERROR();
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&time, start, stop));
    CHECK_CUDA_ERROR(cudaEventDestroy(start));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop));

    float const latency{time / num_repeats};

    return latency;
}

void print_latency(float latency)
{
    std::cout << std::fixed << std::setprecision(3) << "Latency: " << latency
              << " ms" << std::endl;
}

int main()
{
    constexpr uint32_t num_repeats{100};
    constexpr uint32_t num_warmups{100};

    constexpr uint32_t M{256};
    constexpr uint32_t K{256};
    constexpr uint32_t N{256};

    float* A{nullptr};
    float* B{nullptr};
    float* C{nullptr};

    CHECK_CUDA_ERROR(cudaMalloc(&A, M * K * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&B, K * N * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&C, M * N * sizeof(float)));

    uint32_t const matrix_a_col_major_ld{M};
    uint32_t const matrix_a_row_major_ld{K};
    uint32_t const matrix_a_transpose_col_major_ld{matrix_a_row_major_ld};
    uint32_t const matrix_a_transpose_row_major_ld{matrix_a_col_major_ld};

    uint32_t const matrix_b_col_major_ld{K};
    uint32_t const matrix_b_row_major_ld{N};
    uint32_t const matrix_b_transpose_col_major_ld{matrix_b_row_major_ld};
    uint32_t const matrix_b_transpose_row_major_ld{matrix_b_col_major_ld};

    uint32_t const matrix_c_col_major_ld{M};
    uint32_t const matrix_c_row_major_ld{N};
    uint32_t const matrix_c_transpose_col_major_ld{matrix_c_row_major_ld};
    uint32_t const matrix_c_transpose_row_major_ld{matrix_c_col_major_ld};

    cublasHandle_t cublas_handle;
    cudaStream_t stream;

    CHECK_CUDA_ERROR(cudaStreamCreate(&stream));
    CHECK_CUBLAS_ERROR(cublasCreate(&cublas_handle));
    CHECK_CUBLAS_ERROR(cublasSetStream(cublas_handle, stream));

    float const alpha{1.0};
    float const beta{0.0};

    // cublasSgemm assumes column-major matrices.
    std::function<cublasStatus_t(void)> const mm_a_col_major_b_col_major_a_b{
        std::bind(cublasSgemm, cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K,
                  &alpha, A, matrix_a_col_major_ld, B, matrix_b_col_major_ld,
                  &beta, C, matrix_c_col_major_ld)};

    std::function<cublasStatus_t(void)> const
        mm_a_col_major_b_col_major_a_transpose_b{
            std::bind(cublasSgemm, cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, M,
                      N, K, &alpha, A, matrix_a_transpose_col_major_ld, B,
                      matrix_b_col_major_ld, &beta, C, matrix_c_col_major_ld)};

    std::function<cublasStatus_t(void)> const
        mm_a_col_major_b_col_major_a_transpose_b_transpose{std::bind(
            cublasSgemm, cublas_handle, CUBLAS_OP_T, CUBLAS_OP_T, M, N, K,
            &alpha, A, matrix_a_transpose_col_major_ld, B,
            matrix_b_transpose_col_major_ld, &beta, C, matrix_c_col_major_ld)};

    std::function<cublasStatus_t(void)> const
        mm_a_col_major_b_col_major_a_b_transpose{std::bind(
            cublasSgemm, cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T, M, N, K,
            &alpha, A, matrix_a_col_major_ld, B,
            matrix_b_transpose_col_major_ld, &beta, C, matrix_c_col_major_ld)};

    std::cout << "C = A * B" << std::endl;
    float const latency_a_b = measure_cublas_performance(
        mm_a_col_major_b_col_major_a_b, stream, num_repeats, num_warmups);
    print_latency(latency_a_b);

    std::cout << "C = A^T * B" << std::endl;
    float const latency_a_transpose_b =
        measure_cublas_performance(mm_a_col_major_b_col_major_a_transpose_b,
                                   stream, num_repeats, num_warmups);
    print_latency(latency_a_transpose_b);

    std::cout << "C = A * B^T" << std::endl;
    float const latency_a_b_transpose =
        measure_cublas_performance(mm_a_col_major_b_col_major_a_b_transpose,
                                   stream, num_repeats, num_warmups);
    print_latency(latency_a_b_transpose);

    std::cout << "C = A^T * B^T" << std::endl;
    float const latency_a_transpose_b_transpose = measure_cublas_performance(
        mm_a_col_major_b_col_major_a_transpose_b_transpose, stream, num_repeats,
        num_warmups);
    print_latency(latency_a_transpose_b_transpose);

    CHECK_CUDA_ERROR(cudaFree(A));
    CHECK_CUDA_ERROR(cudaFree(B));
    CHECK_CUDA_ERROR(cudaFree(C));
    CHECK_CUBLAS_ERROR(cublasDestroy(cublas_handle));
    CHECK_CUDA_ERROR(cudaStreamDestroy(stream));
}