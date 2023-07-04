#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <functional>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <tuple>
#include <utility>
#include <vector>

#include <cublas_v2.h>
#include <cuda.h>
// https://docs.nvidia.com/cuda/cuda-math-api/modules.html#modules
// #include <cuda_fp8.h>
#include <cub/cub.cuh>
#include <cuda_bf16.h>
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

struct NameGuard {

  NameGuard(const std::string &name) : name_(name) {
    std::cout << "---" << name_ << "---" << std::endl;
  }

  ~NameGuard() {
    std::cout << "---" << name_ << "---" << std::endl;
    std::cout << std::endl;
  }

  std::string name_;
};

template <typename T>
inline std::ostream &operator<<(std::ostream &os, const std::vector<T> &vec) {
  os << "(";
  for (int i{0}; i < vec.size(); ++i) {
    os << vec.at(i) << ",";
  }
  os << ")" << std::endl;
  return os;
}

template <typename T> int NumElements(const std::vector<T> &dims) {
  int numel =
      std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<int>());
  return numel;
}

namespace kps {
namespace details {

constexpr int kReduceMaxThread = 128;
// https://stackoverflow.com/questions/1486904/how-do-i-best-silence-a-warning-about-unused-variables
__attribute__((unused)) constexpr int kWarpSize = 32;

// kGlobalMode: block reduce, each block gets an output;
// kLocalMode: thread reduce, each thread gets an output;
enum ReduceMode { kGlobalMode, kLocalMode };

// Aligned vector generates vectorized load/store on CUDA.
template <typename T, int VecSize>
struct alignas(sizeof(T) * VecSize) AlignedVector {
  T val[VecSize];

  HOSTDEVICE inline const T &operator[](int i) const { return val[i]; }
  HOSTDEVICE inline T &operator[](int i) { return val[i]; }
};

// vectoried load to vec from addr
template <typename T, int VecSize>
HOSTDEVICE inline void Load(const T *addr, AlignedVector<T, VecSize> *vec) {

  const AlignedVector<T, VecSize> *addr_vec =
      reinterpret_cast<const AlignedVector<T, VecSize> *>(addr);
  *vec = *addr_vec;
}

// vectoried save to addr from vec
template <typename T, int VecSize>
HOSTDEVICE inline void Store(const AlignedVector<T, VecSize> &vec, T *addr) {

  AlignedVector<T, VecSize> *addr_vec =
      reinterpret_cast<AlignedVector<T, VecSize> *>(addr);
  *addr_vec = vec;
}

template <typename T, int VecSize> using VectorType = AlignedVector<T, VecSize>;

/*
 * Only the address of input data is the multiplier of 1,2,4, vectorized load
 * with corresponding multiplier-value is possible. Moreover, the maximum length
 * of vectorized load is `128 bits` (16 bytes, 4 float, 2 double) once.
 * Hence, valid length of vectorized load shall be determined under both former
 * constraints.
 */
template <typename T> int GetVectorizedSize(const T *pointer) {
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
  if (address % vec4 == 0) {
    return std::min(4, valid_vec_size);
  } else if (address % vec2 == 0) {
    return std::min(2, valid_vec_size);
  } else {
    return 1;
  }
}

template <typename Ta, typename Tb = Ta, typename Ty=Ta> inline Ty DivUp(const Ta &a, const Tb &b) {
  return (a + b - 1) / b;
}

/**
 * @brief get the last pow of 2
 * 
 * RoundDownPowOfTwo(10) = 8
 */
__host__ __device__ inline int RoundDownPowOfTwo(int n) {
  n |= (n >> 1);
  n |= (n >> 2);
  n |= (n >> 4);
  n |= (n >> 8);
  n |= (n >> 16);
  // error: calling a constexpr __host__ function("max") from a __device__
  // function("RoundDownPowOfTwo") is not allowed. The experimental flag
  // '--expt-relaxed-constexpr' can be used to allow this.
  //  return std::max(1, n - (n >> 1));
  //  https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__INT.html
  return ::max(1, n - (n >> 1));
}

// round integer value into next highest power of 2.
// https://graphics.stanford.edu/~seander/bithacks.html#RoundUpPowerOf2
// RoundUpToPowOfTwo(10) = 16
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

/**
 * Fast division : Replace division in CUDA with multiplication to improve
 * kernel performance.
 * 1. Complete the division calculation on the CPU, and record the calculation
 * results by using the `divider` and `shift_val`.
 * 2. Set the `divisor` on the GPU through Div() to complete the calculation.
 *
 * 参看：   OffsetInfo：https://www.52coding.com.cn/2019/05/05/PyTorch2/
 * 推导见：
 * http://ridiculousfish.com/blog/posts/labor-of-division-episode-i.html
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

    // s = ceil(log_2(d)) , first num that satisfy 2^s > d
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
  int32_t shift_val;   // s
  uint32_t multiplier; // m
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

struct GpuLaunchConfig {
public:
  GpuLaunchConfig() {}

  size_t GetThreadNum() const { return GetGridSize() * GetBlockSize(); }

  size_t GetGridSize() const {
    return block_per_grid.x * block_per_grid.y * block_per_grid.z;
  }

  size_t GetBlockSize() const {
    return thread_per_block.x * thread_per_block.y * thread_per_block.z;
  }

  int compute_capability = 0;
  dim3 thread_per_block = dim3(1, 1, 1);
  dim3 block_per_grid = dim3(1, 1, 1);
};

std::ostream &operator<<(std::ostream &os, const GpuLaunchConfig &obj) {
  os << "GpuLaunchConfig: ";
  os << "\t grids(" << obj.block_per_grid.x << "," << obj.block_per_grid.y
     << "," << obj.block_per_grid.z << ")";
  os << "\t blocks(" << obj.thread_per_block.x << "," << obj.thread_per_block.y
     << "," << obj.thread_per_block.z << ")" << std::endl;
  return os;
}

/* According to NVIDIA, if number of threads per block is 64/128/256/512,
 * cuda performs better. And number of blocks should be greater (at least
 * 2x~4x) than number of SMs. Hence, SM count is took into account within
 * this function to determine the right number of threads per block.
 *
 * 1. 64 <= thread_per_block <= 512
 * 2. active_threads_num / (sm_count * 2) < limit_threads
 * 3. active_threads_num / (sm_count * 4) < limit_threads
 * 4. blocks <= limit_blocks
 * 5. 1 thread process vec_size T elements.
 */

inline GpuLaunchConfig GetGpuLaunchConfig1D(int devid, int64_t numel,
                                            int vec_size = 1) {
  assert(numel >= 0);
  assert(vec_size >= 1);

  // https://github.com/NVIDIA/cuda-samples/blob/master/Samples/1_Utilities/deviceQuery/deviceQuery.cpp
  // paddle/phi/backends/gpu/gpu_resources.cc
  int dev_cnt = 0;
  CHECK_CUDA_ERROR(cudaGetDeviceCount(&dev_cnt));
  assert(dev_cnt > 0 && devid < dev_cnt);
  CHECK_CUDA_ERROR(cudaSetDevice(devid));
  cudaDeviceProp dev_prop;
  CHECK_CUDA_ERROR(cudaGetDeviceProperties(&dev_prop, devid));
  // Device 0: "NVIDIA GeForce RTX 3090"
  //  CUDA Capability Major/Minor version number:    8.6
  //  (082) Multiprocessors, (128) CUDA Cores/MP:    10496 CUDA Cores
  //  Maximum number of threads per block:           1024
  //  Max dimension size of a thread block (x,y,z): (1024, 1024, 64)
  //  Max dimension size of a grid size    (x,y,z): (2147483647, 65535, 65535)
  const int capability = dev_prop.major * 10 + dev_prop.minor;
  // If thread number per block is 64/128/256/512, cuda performs better.
  int limit_threads = std::min(BLOCK_SIZE, dev_prop.maxThreadsPerBlock);
  int limit_blocks = dev_prop.maxGridSize[0];
  int sm_count = dev_prop.multiProcessorCount;

  // threads
  int threads = limit_threads;
  int64_t active_threads_num = DivUp<int64_t>(numel, vec_size);
  if (active_threads_num / (sm_count << 1) < limit_threads) {
    // Round up threads number into an exponential multiple of 2, while number
    // of acitve blocks is about twice of SM, to acquire better performance.
    threads = RoundUpPowerOfTwo(active_threads_num / (sm_count << 1));
  } else if (active_threads_num / (sm_count << 2) < limit_threads) {
    // Round up threads number into an exponential multiple of 2, while number
    // of acitve blocks is about 4 times of SM, to acquire better performance.
    threads = RoundUpPowerOfTwo(active_threads_num / (sm_count << 2));
  }
  // Number of threads per block shall be larger than 64.
  threads = std::max(threads, 64);

  // blocks
  int blocks = DivUp<int64_t>(active_threads_num, threads);
  blocks = std::min(blocks, limit_blocks);

  GpuLaunchConfig config;
  config.thread_per_block.x = threads;
  config.block_per_grid.x = blocks;
  config.compute_capability = capability;

  // std::cout << "Get 1-D launch config: numel=" << numel
  //         << ", vec_size=" << vec_size << ", block_size=" << threads
  //         << ", grid_size=" << blocks << ", limit_blocks=" << limit_blocks
  //         << ", limit_threads=" << limit_threads << std::endl;

  return config;
}

} // namespace details

/**
 * @brief Initialize register with init_data.
 *
 * @template paraments
 * T: Data type of register.
 * NX: Number of data to initialize. vector size. Elements processed by one
 * thread.
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

/**
 * @brief Default binary add functor
 */
template <typename T> struct AddFunctor {
  inline T initial() { return static_cast<T>(0.0f); }

  __device__ T operator()(const T a, const T b) const { return b + a; }
};

/**
 * @brief Default unary identity functor
 */
template <typename Tx, typename Ty = Tx> struct IdentityFunctor {
  HOSTDEVICE inline IdentityFunctor() {}

  HOSTDEVICE explicit inline IdentityFunctor(int n) {}

  HOSTDEVICE inline Ty operator()(const Tx x) const {
    return static_cast<Ty>(x);
  }
};

/**
 * @brief Default unary div functor. Divide by a constant
 */
template <typename Tx, typename Ty = Tx> struct DivideFunctor {
private:
  using MPType = float;

public:
  HOSTDEVICE inline DivideFunctor() { n_inv = static_cast<MPType>(1.0f); }

  HOSTDEVICE explicit inline DivideFunctor(int n) : n_inv((MPType)(1.0 / n)) {}

  HOSTDEVICE inline Ty operator()(const Tx x) const {
    return static_cast<Ty>(static_cast<MPType>(x) * n_inv);
  }

private:
  MPType n_inv;
};

} // namespace kps


// Reduce split or not, Whether to use ReduceHigherDim
#define REDUCE_SPLIT_BOUNDARY 512
#define REDUCE_VEC_SIZE 4

enum ReduceType {
  kReduceLastDim = 0x01,   // when reduce_dim[0] == x_dim.size() - 1;
  kReduceHigherDim = 0x02, // ReduceFirstDim or reduceSecondDim
  kReduceAny = 0x03,       // when reduce_dim.size() > 1
};


// Get strides of x_dim, reduce_dim and left_dim for reduceLastDim and reduceAny
static inline std::vector<int> GetDimStrides(const std::vector<int> &dims,
                                             const std::vector<int> &idx) {
  int n = static_cast<int>(idx.size());
  if (n == 0)
    return std::vector<int>();
  std::vector<int> strides(n);
  strides.back() = 1;
  for (int i = n - 2; i >= 0; --i) {
    strides[i] = strides[i + 1] * dims[idx[i + 1]];
  }
  return strides;
}

template <typename Ty, int dev_id = 0> struct ReduceConfig {
  ReduceConfig(const std::vector<int> &origin_reduce_dims,
               const std::vector<int> &origin_x_dim)
      : reduce_dims_origin(origin_reduce_dims), x_dim(origin_x_dim),
        devid(dev_id) {
    CHECK_CUDA_ERROR(cudaSetDevice(devid));
    CHECK_CUDA_ERROR(cudaGetDeviceProperties(&dev_prop, devid));
  }

  ~ReduceConfig() {
    if (output_data && owned_output) {
      cudaFree(output_data);
      output_data = nullptr;
    }
  }

  int devid;
  cudaDeviceProp dev_prop;

  std::vector<int> reduce_dims_origin;
  std::vector<int> reduce_dim, x_dim, left_dim;
  std::vector<int> reduce_strides, x_strides, left_strides;

  int reduce_type;

  int reduce_num;
  int left_num = 1;

  int blocking_size;

  bool should_reduce_again = false;
  bool reduce_last_dim = false;
  bool vectorize_input = false;

  Ty *output_data{nullptr};
  bool owned_output{false};

  dim3 block;
  dim3 grid;

  void LimitGridDim(dim3 *grid_dim) {
    auto max_grid_dim =
        std::vector<int>(dev_prop.maxGridSize, dev_prop.maxGridSize + 3);
    grid_dim->x = grid_dim->x < max_grid_dim[0] ? grid_dim->x : max_grid_dim[0];
    grid_dim->y = grid_dim->y < max_grid_dim[1] ? grid_dim->y : max_grid_dim[1];
    grid_dim->z = grid_dim->z < max_grid_dim[2] ? grid_dim->z : max_grid_dim[2];
  }

  // Get the parameters of reduceKernel
  void Run() {
    // step1: update the reduce_dim left_dim and x_dim
    SetReduceDim();

    // step2: get the strides of dim for reduceAny and reduceLastDim
    SetStrides();

    // step3: get the type of reduce
    SetReduceType();

    // step4: set the block and grid for launch kernel
    SetBlockDim();

    // step5: limit the grid to prevent thead overflow
    LimitGridDim(&grid);
  }

  // Get blockDim for reduceLastDim and reduceAny
  int GetBlockDim(int block_dim) {

    return block_dim >= kps::details::kReduceMaxThread
               ? kps::details::kReduceMaxThread
               : kps::details::RoundDownPowOfTwo(block_dim);
  }

  // If should_reduce_again, we need malloc temp space for temp data
  void SetOutputData(Ty *y_data) {
    NameGuard g("SetOutputData");
    if (should_reduce_again) {
      int64_t size{
          static_cast<int64_t>(left_num * grid.z * grid.y * sizeof(Ty))};
      cudaMalloc(&output_data, size);
      owned_output = true;
    } else {
      output_data = y_data;
    }
  }

private:
  // set reduce_dim, left_dim and update x_dim
  // eg: x_dim = [2, 4, 6] origin_reduce_dims = [0, 1]
  //     --SetReduceDim--> x_dim = [8,6], reduce_dim = [0], left_dim = [1]
  void SetReduceDim() {
    NameGuard g("SetReduceDim");

    std::set<int> reduce_set;
    for (auto e : reduce_dims_origin) {
      auto pos = e >= 0 ? e : e + x_dim.size();
      reduce_set.insert(pos);
    }
    std::vector<int> reduce_dim_temp(reduce_set.begin(), reduce_set.end());
    std::sort(reduce_dim_temp.begin(), reduce_dim_temp.end());

    std::cout << "0. input x_dim: " << x_dim;
    std::cout << "0. input reduce_dims: " << reduce_dims_origin;

    // update reduce_dim and x_dim
    std::vector<int> x_new_dim;

    reduce_dim.push_back(reduce_dim_temp[0]);
    x_new_dim.push_back(x_dim[0]);

    int idx_reduce = 1;
    int num = 0;
    if (reduce_dim_temp.size() > 1) {
      for (int i = 1; i < x_dim.size(); i++) {
        if ((idx_reduce < reduce_dim_temp.size()) &&
            (i == reduce_dim_temp[idx_reduce])) {
          int result =
              reduce_dim_temp[idx_reduce] - reduce_dim[reduce_dim.size() - 1];
          bool is_equal = ((result - num) == 1);
          if (is_equal) {
            x_new_dim[x_new_dim.size() - 1] *= x_dim[i];
            num++;
          } else {
            reduce_dim.push_back(reduce_dim_temp[idx_reduce] - num);
            x_new_dim.push_back(x_dim[i]);
          }
          idx_reduce++;
        } else {
          x_new_dim.push_back(x_dim[i]);
        }
      }
    } else {
      // reduce_dim only has one num, x_dim not changed.
      x_new_dim = x_dim;
    }

    // update x_dim
    x_dim = x_new_dim;
    std::vector<int>().swap(x_new_dim);
    std::cout << "1. x_dim: " << x_dim;
    std::cout << "1. reduce_dim: " << reduce_dim;


    std::vector<int> reduce_dim_new;
    int is_reduced = 0;
    for (auto e : reduce_dim) {
      is_reduced |= 1 << e;
    }
    std::vector<int>().swap(reduce_dim);

    for (int i = 0; i < x_dim.size(); i++) {
      if ((i == 0) || (((is_reduced >> i) ^ (is_reduced >> (i - 1))) & 1)) {
        x_new_dim.push_back(x_dim[i]);
        if ((is_reduced >> i) & 1)
          reduce_dim_new.push_back(x_new_dim.size() - 1);
      } else {
        x_new_dim[x_new_dim.size() - 1] *= x_dim[i];
      }
    }

    // new x_dim, reduce_dim left_dim
    // dims 分为 reduce_dim set 和 left_dim set.
    x_dim = x_new_dim;
    reduce_dim = reduce_dim_new;
    std::cout << "2. x_dim: " << x_dim;
    std::cout << "2. reduce_dim: " << reduce_dim;

    int x_rank = static_cast<int>(x_dim.size());
    std::set<int> left_set;
    for (int i = 0; i < x_rank; ++i) {
      left_set.insert(i);
    }
    for (auto e : reduce_dim) {
      left_set.erase(e);
    }
    left_dim.assign(left_set.begin(), left_set.end());
    std::cout << "3. left_dim: " << left_dim;

    // if last_dim in reduce_dim set?
    reduce_last_dim = (reduce_dim.back() == x_dim.size() - 1);
    std::cout << "4. reduce_last_dim: " << reduce_last_dim << std::endl;
  }

  // set x_strides, reduce_strides, left_strides for reduceLastDim and reduceAny
  // eg: x_dim = [8, 6], reduce_dim = [0], left_dim = [1]
  //     --SetStrides--> x_strides= [6,1], reduce_strides = [1],
  //     left_strides = [1]
  void SetStrides() {
    NameGuard g("SetStrides");

    std::vector<int> idx_dim;
    for (int i = 0; i < x_dim.size(); i++) {
      idx_dim.push_back(i);
    }

    std::cout << "x_dim: " << x_dim;
    std::cout << "reduce_dim: " << reduce_dim;
    std::cout << "left_dim: " << left_dim;

    x_strides = GetDimStrides(x_dim, idx_dim);
    reduce_strides = GetDimStrides(x_dim, reduce_dim);
    left_strides = GetDimStrides(x_dim, left_dim);
    std::cout << "x_strides: " << x_strides;
    std::cout << "reduce_strides: " << reduce_strides;
    std::cout << "left_strides: " << left_strides;

    reduce_num = reduce_strides[0] * x_dim[reduce_dim[0]];
    std::cout << "reduce_num: " << reduce_num << std::endl;

    if (left_dim.size()) {
      left_num = left_strides[0] * x_dim[left_dim[0]];
      std::cout << "left_num: " << left_num << std::endl;
    }
  }

  // get the reduceType
  // eg: x_dim = [8, 6] reduce_dim = [0] --> ReduceHigherDim -->reduceFirstDim
  //     x_dim = [8, 6] reduce_dim = [1] --> reduceLastDim
  //     x_dim = [8] reduce_dim = [0] --> reduceAll
  //     x_dim = [8, 6, 4, 2] reduce_dim = [0, 2] --> reduceAny
  void SetReduceType() {
    NameGuard g("SetReduceType");
    int rank = x_dim.size();
    int reduce_rank = reduce_dim.size();

    // int max_grid_z = phi::backends::gpu::GetGpuMaxGridDimSize(device_id)[2];
    int max_grid_z = dev_prop.maxGridSize[2];
    bool not_higher = x_dim[0] >= max_grid_z;

    reduce_type = static_cast<int>(ReduceType::kReduceAny);
    if (reduce_last_dim && (reduce_rank == 1)) {
      reduce_type = static_cast<int>(ReduceType::kReduceLastDim);
    } else if (reduce_rank == 1) {
      reduce_type = static_cast<int>(ReduceType::kReduceHigherDim);
      if (rank == 3 && not_higher) {
        reduce_type = static_cast<int>(ReduceType::kReduceAny);
      }
    }
  }

  void SetBlockDimForReduceAny(dim3 *block_dim, dim3 *grid_dim) {
    NameGuard g("SetBlockDimForReduceAny");

    constexpr int min_reduce_num_per_thread = 16;
    constexpr int max_reduce_num_per_thread = 256;
    constexpr int max_num_threads = kps::details::kReduceMaxThread;

    // Set block size.
    // 1. If reduce_last_dim == true, all the threads whose threadIdx.y are same
    //    will process the reduction for one output.
    //    The number of output for one block is blockDim.y;
    // 2. If reduce_last_dim == false, different threadIdx.x will process
    //    different reduction and gets the output separately. If it is
    //    necessary, it should reduce in block y.
    //    The number of output for one block is blockDim.x;
    int block_x, block_y;
    int grid_num, reduce_num_per_thread;
    if (reduce_last_dim) {
      block_x = GetBlockDim(reduce_num);
      block_y = GetBlockDim(left_num);
      block_dim->x = block_x;
      block_dim->y =
          std::min(block_y, static_cast<int>(max_num_threads / block_dim->x));
      grid_num = kps::details::DivUp(left_num, block_dim->y);
      reduce_num_per_thread = kps::details::DivUp(reduce_num, block_dim->x);
    } else {
      block_x = GetBlockDim(left_num);
      block_y = GetBlockDim(reduce_num);
      block_dim->x = std::min(block_x, 32);
      block_dim->y =
          std::min(block_y, static_cast<int>(max_num_threads / block_dim->x));
      block_dim->x =
          std::min(block_x, static_cast<int>(max_num_threads / block_dim->y));
      grid_num = kps::details::DivUp(left_num, block_dim->x);
      reduce_num_per_thread = kps::details::DivUp(reduce_num, block_dim->y);
    }
    // int device_id = phi::backends::gpu::GetCurrentDeviceId();
    // int max_mp = phi::backends::gpu::GetGPUMultiProcessors(device_id);
    // int max_threads_per_mp =
    // phi::backends::gpu::GetGPUMaxThreadsPerMultiProcessor(device_id);
    int max_mp = dev_prop.multiProcessorCount;
    int max_threads_per_mp = dev_prop.maxThreadsPerMultiProcessor;
    std::cout << "multiProcessorCount(sm count): " << max_mp
              << " maxThreadsPerMultiProcessor: " << max_threads_per_mp
              << std::endl;
    int max_threads = max_threads_per_mp * max_mp;
    int num_threads = block_dim->x * block_dim->y;
    int max_num_blocks = max_threads / num_threads;

    // Set grid size.
    // Whether to set grid.y larger than 1, there are 3 following rules:
    // 1. The number that each thread process should no less than
    //    min_reduce_num_per_threadbut no more than max_reduce_num_per_thread;
    // 2. It should maximize the utilization of SM.
    // So we choose the minimum between input_split_num_1 and input_split_num_3
    // to make each thread process as mush data as possible. Meanwhile,
    // the number cannot be larger than max_reduce_num_per_thread, so we
    // choose the maximum between the result above and input_split_num_2.
    int input_split_num_1 =
        kps::details::DivUp(reduce_num_per_thread, min_reduce_num_per_thread);
    int input_split_num_2 =
        kps::details::DivUp(reduce_num_per_thread, max_reduce_num_per_thread);
    int input_split_num_3 = kps::details::DivUp(max_num_blocks, grid_num);

    grid_dim->x = grid_num;
    grid_dim->y = std::max(std::min(input_split_num_1, input_split_num_3),
                           input_split_num_2);
    // if grid.y > 1, we need launch reduce kernel again.
    if (grid_dim->y > 1) {
      should_reduce_again = true;
      std::cout << "should_reduce_again " << should_reduce_again << std::endl;
    }
  }

  // Set block and grid for launch kernel
  // for ReduceHigherDim: if block is enough -> splite reduce_num
  //                     else init block(32, 1) grid(block_num, 1)
  // for others: block(block_num, 1) , grid(left_num, 1)
  void SetBlockDimForHigher(dim3 *block_dim, dim3 *grid_dim) {
    NameGuard g("SetBlockDimForHigher");
    int last_dim_num = x_dim.back();
    // Update left_num
    int grid_z = left_num / last_dim_num;
    left_num = last_dim_num;
    grid_dim->z = grid_z;
    // int device_id = phi::backends::gpu::GetCurrentDeviceId();
    // int max_mp = phi::backends::gpu::GetGPUMultiProcessors(device_id);
    // int max_threads_per_mp =
    //     phi::backends::gpu::GetGPUMaxThreadsPerMultiProcessor(device_id);
    int max_mp = dev_prop.multiProcessorCount;
    int max_threads_per_mp = dev_prop.maxThreadsPerMultiProcessor;
    std::cout << "multiProcessorCount(sm count): " << max_mp
              << " maxThreadsPerMultiProcessor: " << max_threads_per_mp
              << std::endl;
    int max_threads = max_threads_per_mp * max_mp;
    std::cout << "max_thread: " << max_threads << std::endl;

    // init
    int num_block = (max_threads / left_num);
    block_dim->x = GetBlockDim(left_num);
    grid_dim->x = kps::details::DivUp(left_num, block_dim->x);
    blocking_size = reduce_num;

    if (num_block > 1 && reduce_num >= REDUCE_SPLIT_BOUNDARY) {
      blocking_size = kps::details::RoundDownPowOfTwo(reduce_num / num_block);
      if (blocking_size <= 1) {
        blocking_size = kps::details::RoundDownPowOfTwo(sqrt(reduce_num));
      } else if (blocking_size * 2 < reduce_num) {
        blocking_size *= 2;
      }
      should_reduce_again = true;
      grid_dim->y = kps::details::DivUp(reduce_num, blocking_size);
    }
  }

  void SetBlockDim() {
    NameGuard g("SetBlockDim");
    dim3 block_dim(1, 1, 1);
    dim3 grid_dim(left_num, 1, 1);
    blocking_size = reduce_num;

    if (reduce_type == ReduceType::kReduceHigherDim) {
      SetBlockDimForHigher(&block_dim, &grid_dim);
    } else {
      SetBlockDimForReduceAny(&block_dim, &grid_dim);
    }

    block = block_dim;
    grid = grid_dim;
  }
};

template <typename Ty>
inline std::ostream &operator<<(std::ostream &os,
                                const ReduceConfig<Ty> &config) {
  os << "ReduceConfig: " << std::endl;
  os << "\t reduce_dims_origin: " << config.reduce_dims_origin << std::endl;
  os << "---------shape---------: " << std::endl;
  os << "\t x_dim: " << (config.x_dim) << std::endl;
  os << "\t reduce_dim: " << (config.reduce_dim) << std::endl;
  os << "\t left_dim: " << (config.left_dim) << std::endl;
  os << "\t x_strides: " << (config.x_strides) << std::endl;
  os << "\t reduce_strides: " << (config.reduce_strides) << std::endl;
  os << "\t left_strides: " << (config.left_strides) << std::endl;
  os << "---------val---------: " << std::endl;
  os << "\t reduce_type: " << config.reduce_type << std::endl;
  os << "\t reduce_num: " << config.reduce_num << std::endl;
  os << "\t left_num: " << config.left_num << std::endl;
  os << "\t blocking_size: " << config.blocking_size << std::endl;
  os << "\t should_reduce_again: " << config.should_reduce_again << std::endl;
  os << "\t reduce_last_dim: " << config.reduce_last_dim << std::endl;
  os << "\t vectorize_input: " << config.vectorize_input << std::endl;
  os << "------------------: " << std::endl;
  os << "\t block: " << config.block.x << "," << config.block.y << ","
     << config.block.z << std::endl;
  os << "\t grid: " << config.grid.x << "," << config.grid.y << ","
     << config.grid.z << std::endl;
  os << "ReduceConfig End " << std::endl;
  return os;
}

// https://github.com/NVIDIA/cub
// https://nvlabs.github.io/cub/index.html
template <typename Tx, typename Ty, template <typename> class ReduceOp,
          typename TransformOp>
static void CubTensorReduceImpl(const Tx *x_data, Ty *y_data,
                                const TransformOp &transform, int reduce_num,
                                cudaStream_t stream) {
  auto reducer = ReduceOp<Ty>();
  cub::TransformInputIterator<Ty, TransformOp, const Tx *> trans_x(x_data,
                                                                   transform);
  size_t temp_storage_bytes = 0;
  cub::DeviceReduce::Reduce(nullptr, temp_storage_bytes, trans_x, y_data,
                            reduce_num, reducer, reducer.initial(), stream);

  uint8_t *temp_storage;
  CHECK_CUDA_ERROR(cudaMalloc(&temp_storage, temp_storage_bytes));

  cub::DeviceReduce::Reduce(temp_storage, temp_storage_bytes, trans_x, y_data,
                            reduce_num, reducer, reducer.initial(), stream);

  CHECK_CUDA_ERROR(cudaFree(temp_storage));
}

template <typename Tx, typename Ty, template <typename> class ReduceOp,
          typename TransformOp>
void ReduceKernel(const cudaStream_t &stream, const Tx *x_data, Ty *y_data,
                  const std::vector<int> &x_dim,
                  const std::vector<int> &origin_reduce_dims,
                  const TransformOp &transform, bool is_mean = false) {

  auto config = ReduceConfig<Ty>(origin_reduce_dims, x_dim);
  config.Run();
  std::cout << config;

  int numel = NumElements(x_dim);
  bool use_cub_reduce{config.reduce_num == numel};
  assert(use_cub_reduce);

  if (use_cub_reduce) {
    if (is_mean) {
      using Div = kps::DivideFunctor<Tx>;
      CubTensorReduceImpl<Tx, Ty, ReduceOp, Div>(
          x_data, y_data, Div(config.reduce_num), config.reduce_num, stream);
    } else {
      CubTensorReduceImpl<Tx, Ty, ReduceOp, TransformOp>(
          x_data, y_data, transform, config.reduce_num, stream);
    }
    return;
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
float MeasurePerformance(std::function<T(cudaStream_t &)> bound_function,
                         cudaStream_t &stream, int num_repeats = 100,
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

int main(void) {
  constexpr uint32_t n{100};

  constexpr uint32_t num_repeats{100};
  constexpr uint32_t num_warmups{10};

  std::vector<float> vec_in(n, 1.0);
  std::vector<float> vec_out(1);

  const float *h_input{vec_in.data()};
  float *h_output{vec_out.data()};

  float *d_input{nullptr};
  float *d_output{nullptr};

  CHECK_CUDA_ERROR(cudaMalloc(&d_input, n * sizeof(float)));
  CHECK_CUDA_ERROR(cudaMalloc(&d_output, 1 * sizeof(float)));

  CHECK_CUDA_ERROR(
      cudaMemcpy(d_input, h_input, sizeof(float) * n, cudaMemcpyHostToDevice));

  cudaStream_t stream;
  CHECK_CUDA_ERROR(cudaStreamCreate(&stream));

  // cast1. SumReduce, x = ones([10,]), reduce_dims = (0,)
  {
    using T = float;
    std::vector<int> dims{n};
    std::vector<int> reduce_dims{0};
    std::function<void(cudaStream_t &)> const function{
        std::bind(ReduceKernel<T, T, kps::AddFunctor, kps::IdentityFunctor<T>>,
                  std::placeholders::_1, d_input, d_output, dims, reduce_dims,
                  kps::IdentityFunctor<T>(), false)};

    float const latency{
        MeasurePerformance(function, stream, num_repeats, num_warmups)};
    PrintLatency(latency);

    CHECK_CUDA_ERROR(cudaMemcpy(h_output, d_output, sizeof(float) * 1,
                                cudaMemcpyDeviceToHost));

    assert(static_cast<uint32_t>(vec_out[0]) == n);
  }

  // cast2. SumReduce, x = ones([4, 25]), reduce_dims = (0,1)
  {
    using T = float;
    std::vector<int> dims{4, 25};
    assert(NumElements(dims) == n);
    std::vector<int> reduce_dims{0, 1};
    std::function<void(cudaStream_t &)> const function{
        std::bind(ReduceKernel<T, T, kps::AddFunctor, kps::IdentityFunctor<T>>,
                  std::placeholders::_1, d_input, d_output, dims, reduce_dims,
                  kps::IdentityFunctor<T>(), false)};

    float const latency{
        MeasurePerformance(function, stream, num_repeats, num_warmups)};
    PrintLatency(latency);

    CHECK_CUDA_ERROR(cudaMemcpy(h_output, d_output, sizeof(float) * 1,
                                cudaMemcpyDeviceToHost));

    assert(static_cast<uint32_t>(vec_out[0]) == n);
  }

  CHECK_CUDA_ERROR(cudaFree(d_input));
  CHECK_CUDA_ERROR(cudaFree(d_output));
  CHECK_CUDA_ERROR(cudaStreamDestroy(stream));
}