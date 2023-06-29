#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <functional>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <tuple>
#include <utility>
#include <vector>

#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#define THREAD_ID_X threadIdx.x
#define THREAD_ID_Y threadIdx.y
#define THREAD_ID_Z threadIdx.z

#define BLOCK_NUM_X blockDim.x
#define BLOCK_NUM_Y blockDim.y
#define BLOCK_NUM_Z blockDim.z

#define BLOCK_ID_X blockIdx.x
#define BLOCK_ID_Y blockIdx.y
#define BLOCK_ID_Z blockIdx.z

#define GRID_NUM_X gridDim.x
#define GRID_NUM_Y gridDim.y
#define GRID_NUM_Z gridDim.z

#define VecSizeL 4
#define VecSizeM 2
#define VecSizeS 1

// CUDA performs better when `thread_per_block` is between [64, 512]
#define BLOCK_SIZE 512

#if (defined(__CUDACC__) || defined(__HIPCC__) || defined(__xpu__))
#define HOSTDEVICE __host__ __device__
#define DEVICE __device__
#define HOST __host__
#else
#define HOSTDEVICE
#define DEVICE
#define HOST
#endif


template <typename T = int64_t> inline T DivUp(const T &a, const T &b) {
  return (a + b - 1) / b;
}

// round integer value into next highest power of 2.
// https://graphics.stanford.edu/~seander/bithacks.html#RoundUpPowerOf2
inline int64_t RoundUpToPowOfTwo(int64_t n, int64_t min_val = 1) {
  n--;
  n |= (n >> 1);
  n |= (n >> 2);
  n |= (n >> 4);
  n |= (n >> 8);
  n |= (n >> 16);
  return std::max(min_val, (n + 1));
}

inline int64_t RoundUpPowerOfTwo(int64_t n) {
  constexpr int64_t min_val = 32;
  constexpr int64_t max_val = 1024;
  int64_t num = RoundUpToPowOfTwo(n, min_val);
  return std::min(num, max_val);
}


namespace kps {
namespace details {

template <typename T, int VecSize>
struct alignas(sizeof(T) * VecSize) VectorType {
  T val[VecSize];
};

template <typename T, int VecSize>
using AlignedVector = VectorType<T, VecSize>;

/*
 * Only the address of input data is the multiplier of 1,2,4, vectorized load
 * with corresponding multiplier-value is possible. Moreover, the maximum length
 * of vectorized load is `128 bits` (16 bytes, 4 float, 2 double) once.
 * Hence, valid length of vectorized load shall be determined under both former constraints.
 */
template <typename T>
int GetVectorizedSize(const T* pointer){
  constexpr int max_load_bits = 128;
  // valid elements num of T, e.g float=4, double=2, char=16
  constexpr int valid_vec_size = max_load_bits / CHAR_BIT / sizeof(T);

  uint64_t address = reinterpret_cast<int64_t>(pointer);

  // (float, 32), (double, 64)
  constexpr int vec8 = std::alignment_of<AlignedVector<T, 8>>::value;
  // (float, 16), (double, 32)
  constexpr int vec4 = std::alignment_of<AlignedVector<T, 4>>::value;
  // (float,  8), (double, 16)
  constexpr int vec2 = std::alignment_of<AlignedVector<T, 2>>::value;
  
  /*
  * Currently, decide to deal with no more than 4 data once while adopting
  * vectorization load/store, if performance test shows that dealing with
  * 8 data once in vectorization load/store does get optimized, code below
  * can begin with :
    if (address % vec8 == 0) {
      return std::min(8, valid_vec_size);
  */
  if (address % vec4 == 0){
    return std::min(4, valid_vec_size);
  } else if (address % vec2 == 0){
    return std::min(2, valid_vec_size);
  } else {
    return 1;
  }
}


/**
 * Fast division : Replace division in CUDA with multiplication to improve
 * kernel performance.
 * 1. Complete the division calculation on the CPU, and record the calculation
 * results by using the divider and shift_val.
 * 2. Set the divisor on the GPU through Div() to complete the calculation.
 */
#define INT_BITS 32
struct FastDivMod {

  // 1st value represents the result of input number divides by recorded divisor
  // 2nd value represents the result of input number modulo by recorded divisor
  using DivModT = VectorType<uint32_t, 2>;

  FastDivMod() {}

  HOSTDEVICE FastDivMod(uint32_t d) : divisor(d) {
    static_assert(sizeof(unsigned int) == 4,
                  "Only Support 32-bit unsigned int.");

    for (shift_val = 0; shift_val < INT_BITS; ++shift_val) {
      auto shift_limit = 1 << shift_val;
      if (shift_limit >= divisor)
        break;
    }
    uint64_t long_one = 1;
    uint64_t temp_div =
        ((long_one << INT_BITS) * ((long_one << shift_val) - divisor)) /
            divisor +
        1;
    multiplier = temp_div;
  }

  __device__ __forceinline__ uint32_t Div(uint32_t n) const {
    uint32_t t = __umulhi(n, multiplier);
    return (t + n) >> shift_val;
  }

  __device__ __forceinline__ DivModT Divmod(uint32_t n) const {
    uint32_t q = Div(n);
    DivModT result = {q, n - q * divisor};
    return result;
  }

  int32_t divisor;
  int32_t shift_val;
  uint32_t multiplier;
};
#undef INT_BITS

template <typename T>
__device__ __forceinline__ void WriteData(T *dst, const T *__restrict__ src,
                                          int num) {
  for (int i = 0; i < num; i++) {
    dst[i] = src[i];
  }
}

template <typename T>
__device__ __forceinline__ void ReadData(T *dst, const T *__restrict__ src,
                                         int num) {
  for (int i = 0; i < num; i++) {
    dst[i] = src[i];
  }
}
} // namespace details

/**
 * @brief Initialize register with init_data.
 *
 * @template paraments
 * T: Data type of register.
 * NX: Number of data to initialize. vector size. Elements processed by one thread.
 *
 * @param：
 * dst: The register pointer of the thread, the size is NX.
 * init_data: Initial value.
 * read_lens: not used.
 */
template <typename T, int NX>
__device__ __forceinline__ void Init(T *dst, T init_data) {
#pragma unroll
  for (int i = 0; i < NX; i++) {
    dst[i] = init_data;
  }
}

template <typename T, int NX>
__device__ __forceinline__ void Init(T *dst, T init_data, int read_lens) {
#pragma unroll
  for (int i = 0; i < NX; i++) {
    dst[i] = init_data;
  }
}

/**
 * @brief Read 1D data from global memory to register. When IsBoundary = true
 * and (NX % 4 == 0 or Nx % 2 == 0), vectorized load data will be used to
 * improve memory access efficiency.
 *
 * @template paraments
 * T: The type of data.
 * NX: Each thread load NX data from global memory continuously.
 * NY: Each thread need to load NY rows, only NY = 1 was supported.
 * threadIdx.x is used as the thread index. Currently only GPU was supported.
 * IsBoundary: Whether to make an out-of-bounds judgment on access to memory.
 * When the number of data processed by this block is less than
 * NX x NY x blockDim.x, boundary judgment is required to avoid memory access
 * crossing the boundary.
 *
 * @param：
 * dst: The register pointer of the thread, the size is NX * NY.
 * src: The data pointer of the current block.
 * size: The current block needs to load size data continuously.
 * read_lens: not used.
 */
template <typename T, int NX, int NY, bool IsBoundary = false>
__device__ __forceinline__ void ReadData(T *dst, const T *__restrict__ src,
                                         int num) {
  if (IsBoundary) { // blockDim.x * NX > num
    int thread_offset = threadIdx.x * NX;
#pragma unroll
    for (int idx = 0; idx < NX; ++idx) {
      if (idx + thread_offset < num) {
        dst[idx] = src[thread_offset + idx];
      }
    }
  } else { // blockDim,x * NX < num
    // kVectorSize in (4, 2, 1)
    constexpr int kVectorSize = (NX % 4 == 0) ? 4 : (NX % 2 == 0) ? 2 : 1;
    constexpr int kVectorsPerThread = NX / kVectorSize;
    int thread_offset = threadIdx.x * kVectorsPerThread;

    using VecType = details::VectorType<T, kVectorSize>;
    const VecType *vec_input = reinterpret_cast<const VecType *>(src);
    VecType vec_temp[kVectorsPerThread];

#pragma unroll
    for (int i = 0; i < kVectorsPerThread; ++i) {
      vec_temp[i] = vec_input[thread_offset + i];
#pragma unroll
      for (int idx = 0; idx < NX; ++idx) {
        dst[idx] = *(reinterpret_cast<T *>(vec_temp) + idx);
      }
    }
  }
}

template <typename T, int NX, int NY, bool IsBoundary = false>
__device__ __forceinline__ void ReadData(T *dst, const T *__restrict__ src,
                                         int num, int read_lens) {
  if (IsBoundary) { // blockDim.x * NX > num
    int thread_offset = threadIdx.x * NX;
#pragma unroll
    for (int idx = 0; idx < NX; ++idx) {
      if (idx + thread_offset < num) {
        dst[idx] = src[thread_offset + idx];
      }
    }
  } else { // blockDim,x * NX < num
    constexpr int kVectorSize = (NX % 4 == 0) ? 4 : (NX % 2 == 0) ? 2 : 1;
    constexpr int kVectorsPerThread = NX / kVectorSize;
    int thread_offset = threadIdx.x * kVectorsPerThread;

    using VecType = details::VectorType<T, kVectorSize>;
    const VecType *vec_input = reinterpret_cast<const VecType *>(src);
    VecType vec_temp[kVectorsPerThread];

#pragma unroll
    for (int i = 0; i < kVectorsPerThread; ++i) {
      vec_temp[i] = vec_input[thread_offset + i];
#pragma unroll
      for (int idx = 0; idx < NX; ++idx) {
        dst[idx] = *(reinterpret_cast<T *>(vec_temp) + idx);
      }
    }
  }
}

/**
 * @brief Write 2D data from registers to global memory. When IsBoundary = true
 * and (NX % 4 == 0 or Nx % 2 == 0), the data will be vectorized to improve the
 * data loading efficiency
 *
 * @template paraments
 * T: The type of data.
 * NX: The number of data continuously writed by each thread.
 * NY: The number of data rows loaded by each thread, only NY = 1 was supported.
 * threadIdx.x is used as the thread index. Currently only GPU was supported.
 * IsBoundary: Indicates whether to perform block access storage out-of-bounds
 * judgment. When the number of data processed by the block is less than
 * NX x NY x blockDim.x, boundary judgment is required to avoid memory access
 * crossing the boundary.
 *
 * @param：
 * dst: The data pointer of the current block.
 * src: The register pointer, the size is NX * NY.
 * size: The current block needs to load size elements continuously.
 * read_lens: not used
 */
template <typename T, int NX, int NY, bool IsBoundary = false>
__device__ __forceinline__ void WriteData(T *dst, T *__restrict__ src,
                                          int num) {
  if (IsBoundary) {
    int thread_offset = threadIdx.x * NX;
#pragma unroll
    for (int idx = 0; idx < NX; ++idx) {
      if ((thread_offset + idx) < num) {
        dst[thread_offset + idx] = src[idx];
      }
    }
  } else {
    // Vector type
    constexpr int kVectorSize = (NX % 4 == 0) ? 4 : (NX % 2 == 0) ? 2 : 1;
    constexpr int kVectorsPerThread = NX / kVectorSize;

    int thread_offset = threadIdx.x * kVectorsPerThread;
    using VecType = details::VectorType<T, kVectorSize>;
    VecType *vec_dst = reinterpret_cast<VecType *>(dst);
    VecType vec_temp[kVectorsPerThread];
#pragma unroll
    for (int idx = 0; idx < kVectorsPerThread; ++idx) {
      vec_temp[idx] = *(reinterpret_cast<VecType *>(src) + idx);
      vec_dst[thread_offset + idx] = vec_temp[idx];
    }
  }
}

template <typename T, int NX, int NY, bool IsBoundary = false>
__device__ __forceinline__ void WriteData(T *dst, T *__restrict__ src, int num,
                                          int read_lens) {
  if (IsBoundary) {
    int thread_offset = threadIdx.x * NX;
#pragma unroll
    for (int idx = 0; idx < NX; ++idx) {
      if ((thread_offset + idx) < num) {
        dst[thread_offset + idx] = src[idx];
      }
    }
  } else {
    // Vector type
    constexpr int kVectorSize = (NX % 4 == 0) ? 4 : (NX % 2 == 0) ? 2 : 1;
    constexpr int kVectorsPerThread = NX / kVectorSize;

    int thread_offset = threadIdx.x * kVectorsPerThread;
    using VecType = details::VectorType<T, kVectorSize>;
    VecType *vec_dst = reinterpret_cast<VecType *>(dst);
    VecType vec_temp[kVectorsPerThread];
#pragma unroll
    for (int idx = 0; idx < kVectorsPerThread; ++idx) {
      vec_temp[idx] = *(reinterpret_cast<VecType *>(src) + idx);
      vec_dst[thread_offset + idx] = vec_temp[idx];
    }
  }
}

} // namespace kps


template <typename T, int VecSize> struct Loader {

  static __device__ void Apply(T *args, const T *in, int64_t offset, int num,
                               int read_lens, bool is_boundary) {

    kps::Init<T, VecSize>(args, static_cast<T>(1.0f), read_lens);

    if (is_boundary) {
      kps::ReadData<T, VecSize, 1, true>(args, in + offset, num, read_lens);
    } else {
      kps::ReadData<T, VecSize, 1, false>(args, in + offset, num, read_lens);
    }
  }
};

template <typename T, int VecSize, typename Functor>
struct SameDimsElementwisePrimitiveCaller {

  __device__ inline void operator()(Functor func, const T *args, T *result,
                                    int read_lens) {
#pragma unroll
    for (int idx = 0; idx < VecSize; ++idx) {
      result[idx] = static_cast<T>(func(args[idx]));
    }
  }
};

template <typename OutT, int VecSize, bool IsBoundary>
struct ElementwiseWriteDataCallerBc {

  __device__ __forceinline__ void operator()(OutT *out, OutT src[VecSize],
                                             int64_t offset, int num,
                                             int read_lens) {
    kps::WriteData<OutT, VecSize, 1, IsBoundary>(out + offset, src, num,
                                                 read_lens);
  }
};

template <typename T, typename Functor, int VecSize, bool IsBoundary>
__device__ void VectorizedElementwiseKernelImpl(const T *in, T *out,
                                                int64_t offset, int num,
                                                int read_lens, Functor func) {

  T ins_reg[VecSize];
  T outs_reg[VecSize];

  // load to register
  Loader<T, VecSize>::Apply(ins_reg, in, offset, num, read_lens, IsBoundary);
  // compute in register
  SameDimsElementwisePrimitiveCaller<T, VecSize, Functor>()(func, ins_reg, outs_reg,
                                                            read_lens);
  // save back to global mem
  ElementwiseWriteDataCallerBc<T, VecSize, IsBoundary>()(out, outs_reg, offset,
                                                         num, read_lens);
}

template <typename T, typename Functor, int VecSize>
__global__ void VectorizedElementwiseKernel(const T *in, T *out, int64_t numel,
                                            int64_t main_offset,
                                            int read_lens, // vector size
                                            Functor func) {

  auto vec_size{read_lens};

  int64_t data_offset{blockIdx.x * blockDim.x * vec_size};
  int64_t stride{blockDim.x * gridDim.x * vec_size};

  for (; data_offset < main_offset; data_offset += stride) {
    VectorizedElementwiseKernelImpl<T, Functor, VecSize, false>(
        in, out, data_offset, blockDim.x * vec_size, vec_size, func);
  }

  int64_t remain = numel - data_offset;

  if (remain > 0) {
    VectorizedElementwiseKernelImpl<T, Functor, VecSize, true>(
        in, out, data_offset, static_cast<int>(remain), vec_size, func);
  }
}

// Functor must be Unary Operator, e.g log, exp, relu...
// https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/kernels/funcs/elementwise_base.h#L779
template <typename T, typename Functor, int VecSize = 4>
void LaunchElementwiseCudaKernel(T *out, const T *in, int64_t numel,
                                 Functor func, cudaStream_t &stream) {
  dim3 thread_per_block = dim3(1, 1, 1);
  dim3 block_per_grid = dim3(1, 1, 1);

  thread_per_block.x = BLOCK_SIZE; // 512

  auto elems_per_block = VecSize * thread_per_block.x;

  block_per_grid.x = DivUp<int64_t>(numel, elems_per_block);

  // printf("Grid : {%d, %d, %d} blocks. Blocks : {%d, %d, %d} threads.\n",
  // block_per_grid.x, block_per_grid.y, block_per_grid.z, thread_per_block.x,
  // thread_per_block.y, thread_per_block.z);

  int64_t main_offset = (numel / elems_per_block) * elems_per_block;

  VectorizedElementwiseKernel<T, Functor, VecSize>
      <<<block_per_grid, thread_per_block, 0, stream>>>(
          in, out, numel, main_offset, VecSize, func);
}

template <typename T, typename Functor, int VecSize = 4>
void ElementwiseKernel(T *out, const T *in, int64_t numel, Functor func,
                       cudaStream_t &stream) {
  switch (VecSize) {
  case VecSizeL:
    LaunchElementwiseCudaKernel<T, Functor, VecSizeL>(out, in, numel, func,
                                                      stream);
    break;
  case VecSizeM:
    LaunchElementwiseCudaKernel<T, Functor, VecSizeM>(out, in, numel, func,
                                                      stream);
    break;
  case VecSizeS:
    LaunchElementwiseCudaKernel<T, Functor, VecSizeS>(out, in, numel, func,
                                                      stream);
    break;
  default: {
    std::stringstream ss;
    ss << "Unsupported vectorized size:" << VecSize;
    throw std::invalid_argument(ss.str());
    break;
  }
  }
}




#define CHECK_CUDA_ERROR(val) checkCuda((val), #val, __FILE__, __LINE__)
template <typename T>
void checkCuda(T err, const char *const func, const char *const file,
               const int line) {
  if (err != cudaSuccess) {
    std::cerr << "CUDA Runtime Error at: " << file << ":" << line << std::endl;
    std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
    std::exit(EXIT_FAILURE);
  }
}

#define CHECK_LAST_CUDA_ERROR(val) checkCudaLast(__FILE__, __LINE__)
void checkCudaLast(const char *const file, const int line) {
  cudaError_t err{cudaGetLastError()};
  if (err != cudaSuccess) {
    std::cerr << "CUDA Runtime Error at: " << file << ":" << line << std::endl;
    std::cerr << cudaGetErrorString(err) << std::endl;
    std::exit(EXIT_FAILURE);
  }
}

#define CHECK_CUBLAS_ERROR(val) checkCuBlas((val), #val, __FILE__, __LINE__)
template <typename T>
void checkCuBlas(T err, const char *const func, const char *const file,
                 const int line) {
  if (err != CUBLAS_STATUS_SUCCESS) {
    std::cerr << "cuBlas Runtime Error at: " << file << ":" << line;
    std::exit(EXIT_FAILURE);
  }
}

// create random vec
template <typename T> std::vector<T> CreateRandomVector(size_t n) {
  std::random_device r;
  std::default_random_engine e(r());
  std::uniform_real_distribution<float> uniform_dist(-256, 256);

  std::vector<T> vec(n);
  for (size_t i{0}; i < n; ++i) {
    vec.at(i) = static_cast<T>(uniform_dist(e));
  }
  return vec;
}

// All Close assert
template <typename T>
bool AllClose(const std::vector<T> &vec_1, const std::vector<T> &vec_2,
              const T &abs_tol) {
  if (vec_1.size() != vec_2.size()) {
    return false;
  }

  for (size_t i{0}; i < vec_1.size(); ++i) {
    if (std::abs(vec_1[i] - vec_2[i]) > abs_tol) {
      std::cout << "AllClose diff at " << i << " : " << vec_1[i] << " "
                << vec_2[i] << std::endl;
      return false;
    }
  }
  return true;
}

// kernel preformance measure
template <typename T>
float MeasurePerformance(std::function<T(cudaStream_t&)> bound_function,
                         cudaStream_t& stream, int num_repeats = 100,
                         int num_warmups = 100) {
  cudaEvent_t start, stop;
  float time;

  // create 
  CHECK_CUDA_ERROR(cudaEventCreate(&start));
  CHECK_CUDA_ERROR(cudaEventCreate(&stop));

  // warmup
  for (int i{0}; i < num_warmups; ++i) {
    bound_function(stream);
  }
  CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));

  // do it
  CHECK_CUDA_ERROR(cudaEventRecord(start, stream));
  for (int i{0}; i < num_repeats; ++i) {
    bound_function(stream);
  }
  CHECK_CUDA_ERROR(cudaEventRecord(stop, stream));
  CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
  CHECK_LAST_CUDA_ERROR();

  // compute elapsed time
  CHECK_CUDA_ERROR(cudaEventElapsedTime(&time, start, stop));

  // destory
  CHECK_CUDA_ERROR(cudaEventDestroy(start));
  CHECK_CUDA_ERROR(cudaEventDestroy(stop));

  float const latency{time / num_repeats};
  return latency;
}

void PrintLatency(float latency) {
  std::cout << std::fixed << std::setprecision(3) << "Latency: " << latency
            << " ms" << std::endl;
}

// apply functor on host
template <typename T, typename Functor>
void ApplyFunctorHost(const std::vector<T> &in, std::vector<T> *out,
                      Functor func) {
#pragma unroll
  for (int i{0}; i < in.size(); ++i) {
    out->at(i) = func(in[i]);
  }
}

// functors
template <typename T> struct LogFunctor {
  HOSTDEVICE inline T operator()(const T &a) const { return ::log(a); }
};


int main(void) {
  // unary function
  using Functor = LogFunctor<float>;

  // problem size
  constexpr uint32_t n{100000000};

  // host
  const std::vector<float> vec_in{CreateRandomVector<float>(n)};

  std::vector<float> vec_out_host(n);
  ApplyFunctorHost<float, Functor>(vec_in, &vec_out_host, Functor());

  // cuda
  constexpr uint32_t num_repeats{100};
  constexpr uint32_t num_warmups{10};

  // output mem
  std::vector<float> vec_out(n);

  const float *h_input{vec_in.data()};
  float *h_output{vec_out.data()};

  float *d_input{nullptr};
  float *d_output{nullptr};

  // malloc device memory
  CHECK_CUDA_ERROR(cudaMalloc(&d_input, n * sizeof(float)));
  CHECK_CUDA_ERROR(cudaMalloc(&d_output, n * sizeof(float)));
  // copy memory to device
  CHECK_CUDA_ERROR(
      cudaMemcpy(d_input, h_input, sizeof(float) * n, cudaMemcpyHostToDevice));

  // create stream
  cudaStream_t stream;
  CHECK_CUDA_ERROR(cudaStreamCreate(&stream));

  // vector size for simd
  constexpr uint32_t vecsize = 4;
  std::function<void(cudaStream_t&)> const function{
      std::bind(ElementwiseKernel<float, Functor, vecsize>, d_output, d_input,
                n, Functor(), std::placeholders::_1)};
  // St8functionIFvRP11CUstream_stEE
  // std::cout << typeid(function).name() << std::endl;

  float const latency{
      MeasurePerformance(function, stream, num_repeats, num_warmups)};
  PrintLatency(latency);

  // copy result to cpu
  CHECK_CUDA_ERROR(cudaMemcpy(h_output, d_output, sizeof(float) * n,
                              cudaMemcpyDeviceToHost));

  // check result
  assert(AllClose<float>(vec_out_host, vec_out, 1e-4));

  // free mem and stream obj
  CHECK_CUDA_ERROR(cudaFree(d_input));
  CHECK_CUDA_ERROR(cudaFree(d_output));
  CHECK_CUDA_ERROR(cudaStreamDestroy(stream));
}