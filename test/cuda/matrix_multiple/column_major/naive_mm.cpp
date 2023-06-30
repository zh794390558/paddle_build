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

template <class T>
float measure_performance(std::function<T(void)> bound_function,
                          int num_repeats = 100, int num_warmups = 100)
{
    for (int i{0}; i < num_warmups; ++i)
    {
        bound_function();
    }

    std::chrono::steady_clock::time_point time_start{
        std::chrono::steady_clock::now()};
    for (int i{0}; i < num_repeats; ++i)
    {
        bound_function();
    }
    std::chrono::steady_clock::time_point time_end{
        std::chrono::steady_clock::now()};

    auto time_elapsed{std::chrono::duration_cast<std::chrono::milliseconds>(
                          time_end - time_start)
                          .count()};
    float latency{time_elapsed / static_cast<float>(num_repeats)};

    return latency;
}

// A and B are column-major matrices.
template <typename T>
void mm_a_col_major_b_col_major(T const* A, T const* B, T* C, uint32_t m,
                                uint32_t n, uint32_t k, uint32_t lda,
                                uint32_t ldb, uint32_t ldc, bool is_A_transpose,
                                bool is_B_transpose)
{
    for (uint32_t ni{0}; ni < n; ++ni)
    {
        for (uint32_t mi{0}; mi < m; ++mi)
        {
            // Compute C[mi, ni]
            T accum{0};
            // A * B
            if ((!is_A_transpose) && (!is_B_transpose))
            {
                for (uint32_t ki{0}; ki < k; ++ki)
                {
                    // A[mi, ki] * B[ki, ni]
                    accum += A[ki * lda + mi] * B[ni * ldb + ki];
                }
            }
            // A^T * B
            else if ((is_A_transpose) && (!is_B_transpose))
            {
                for (uint32_t ki{0}; ki < k; ++ki)
                {
                    // A[ki, mi] * B[ki, ni]
                    accum += A[mi * lda + ki] * B[ni * ldb + ki];
                }
            }
            // A * B^T
            else if ((!is_A_transpose) && (is_B_transpose))
            {
                for (uint32_t ki{0}; ki < k; ++ki)
                {
                    // A[mi, ki] * B[ni, ki]
                    accum += A[ki * lda + mi] * B[ki * ldb + ni];
                }
            }
            // A^T * B^T
            else
            {
                for (uint32_t ki{0}; ki < k; ++ki)
                {
                    // A[ki, mi] * B[ni, ki]
                    accum += A[mi * lda + ki] * B[ki * ldb + ni];
                }
            }
            C[ni * ldc + mi] = accum;
        }
    }
}

void print_latency(float latency)
{
    std::cout << std::fixed << std::setprecision(3) << "Latency: " << latency
              << " ms" << std::endl;
}

int main()
{
    constexpr uint32_t num_repeats{10};
    constexpr uint32_t num_warmups{10};

    constexpr uint32_t M{256};
    constexpr uint32_t K{256};
    constexpr uint32_t N{256};

    std::vector<float> matrix_a(M * K);
    std::vector<float> matrix_b(K * N);
    std::vector<float> matrix_c(M * N);

    float const* A{matrix_a.data()};
    float const* B{matrix_b.data()};
    float* C{matrix_c.data()};

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

    std::function<void(void)> const mm_a_col_major_b_col_major_a_b{
        std::bind(mm_a_col_major_b_col_major<float>, A, B, C, M, N, K,
                  matrix_a_col_major_ld, matrix_b_col_major_ld,
                  matrix_c_col_major_ld, false, false)};

    std::function<void(void)> const mm_a_col_major_b_col_major_a_transpose_b{
        std::bind(mm_a_col_major_b_col_major<float>, A, B, C, M, N, K,
                  matrix_a_transpose_col_major_ld, matrix_b_col_major_ld,
                  matrix_c_col_major_ld, true, false)};

    std::function<void(void)> const
        mm_a_col_major_b_col_major_a_transpose_b_transpose{std::bind(
            mm_a_col_major_b_col_major<float>, A, B, C, M, N, K,
            matrix_a_transpose_col_major_ld, matrix_b_transpose_col_major_ld,
            matrix_c_col_major_ld, true, true)};

    std::function<void(void)> const mm_a_col_major_b_col_major_a_b_transpose{
        std::bind(mm_a_col_major_b_col_major<float>, A, B, C, M, N, K,
                  matrix_a_col_major_ld, matrix_b_transpose_col_major_ld,
                  matrix_c_col_major_ld, false, true)};

    std::cout << "C = A * B" << std::endl;
    float const latency_a_b = measure_performance(
        mm_a_col_major_b_col_major_a_b, num_repeats, num_warmups);
    print_latency(latency_a_b);

    std::cout << "C = A^T * B" << std::endl;
    float const latency_a_transpose_b = measure_performance(
        mm_a_col_major_b_col_major_a_transpose_b, num_repeats, num_warmups);
    print_latency(latency_a_transpose_b);

    std::cout << "C = A * B^T" << std::endl;
    float const latency_a_b_transpose = measure_performance(
        mm_a_col_major_b_col_major_a_b_transpose, num_repeats, num_warmups);
    print_latency(latency_a_b_transpose);

    std::cout << "C = A^T * B^T" << std::endl;
    float const latency_a_transpose_b_transpose =
        measure_performance(mm_a_col_major_b_col_major_a_transpose_b_transpose,
                            num_repeats, num_warmups);
    print_latency(latency_a_transpose_b_transpose);

    assert(latency_a_transpose_b ==
           std::min({latency_a_b, latency_a_transpose_b, latency_a_b_transpose,
                     latency_a_transpose_b_transpose}));
    assert(latency_a_b_transpose ==
           std::max({latency_a_b, latency_a_transpose_b, latency_a_b_transpose,
                     latency_a_transpose_b_transpose}));
}