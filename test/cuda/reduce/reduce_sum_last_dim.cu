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

#if defined(_WIN32)
#define UNUSED
#define __builtin_expect(EXP, C) (EXP)
#else
#define UNUSED __attribute__((unused))
#endif

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

#define FULL_WARP_MASK 0xFFFFFFFF

#define CREATE_SHFL_MASK(mask, predicate)                                      \
  mask = __ballot_sync(FULL_WARP_MASK, (predicate))

template <typename T>
__forceinline__ __device__ T CudaShuffleDownSync(unsigned mask, T val,
                                                 int delta,
                                                 int width = warpSize) {
  return __shfl_down_sync(mask, val, static_cast<unsigned>(delta), width);
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

template <typename Ta, typename Tb = Ta, typename Ty = Ta>
inline Ty DivUp(const Ta &a, const Tb &b) {
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

template <typename T, typename ReduceOp>
__device__ __forceinline__ T WarpReduce(T val, ReduceOp reducer) {
  unsigned mask = 0u;
  CREATE_SHFL_MASK(mask, true);
  for (int stride = details::kWarpSize / 2; stride > 0; stride >>= 1) {
    T temp = CudaShuffleDownSync(mask, val, stride);
    val = reducer(val, temp);
  }
  return val;
}

/* e.g.
 * |---------block---------|
 * |warp0|warp1|warp2|warp3|
 * |0~31|32~63|64~95|96~127|  ---->blockDim.x = 128
 *  \|/  \|/   \|/    \|/     ---->1. First WarpReduce in each warp
 * res0  res1  res2  res3     ---->2. Store result of each warp to shared memory
 *   \    \    /     /        ---->3. Load the result above from shared memory
 *        res                         to warp0 and process the second WarpReduce
 */

/**
 * @brief BlockXReduce reduce along blockDim.x.
 */
template <typename T, typename ReduceOp>
__device__ __forceinline__ T BlockXReduce(T val, ReduceOp reducer) {
  __syncthreads();

  using details::kWarpSize;
  // WarpReduce result buffer
  __shared__ T shared[2 * kWarpSize];

  int block_dim_x = blockDim.x;

  if (blockDim.x > kWarpSize) {
    // Bit operation can be used when kWarpSize is 32 or 64 now
    // WarpSize
    constexpr int rshift_val =
        (kWarpSize != 32) ? ((kWarpSize == 64) ? 6 : 5) : 5;
    // NumWarps = blockDim.x / WarpSize
    block_dim_x = blockDim.x >> rshift_val;
    printf("BlockXReduce: NumWarps %d, blockDim.x %d, warpSize %d\n",
           block_dim_x, blockDim.x, kWarpSize);
    // lane in Warp
    int lane = threadIdx.x & (kWarpSize - 1);
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    // WarpIndex
    int wid = tid >> rshift_val;
    printf("BlockXReduce: tid %d, wid %d, lane %d\n", tid, wid, lane);
    val = WarpReduce(val, reducer);
    if (lane == 0) {
      shared[wid] = val;
    }
    __syncthreads();

    // block Index
    int bid = threadIdx.y;
    // load val for last WarpReduce
    // block_dim_x 可能 < lane，会读到无用的值。
    // 但在下面的for循环warp规约时，由block_dim_x限制规约次数，所以 thread 0
    // 不会计算错误，且是最终期望结果。
    val = shared[bid * block_dim_x + lane];
    printf("BlockXReduce: bid %d, block_dim_x %d, lane %d\n", bid, block_dim_x,
           lane);
  }

  // 根据 block_dim_x 判断规约几次
  unsigned mask = 0u;
  CREATE_SHFL_MASK(mask, true);
  for (int stride = 1; stride < block_dim_x; stride <<= 1) {
    T temp = CudaShuffleDownSync(mask, val, stride);
    val = reducer(val, temp);
  }
  __syncthreads();

  // 每行的 threadIdx.x == 0 保存改行的规约值。
  if (threadIdx.x == 0) {
    shared[threadIdx.y] = val;
  }
  __syncthreads();

  //返回当前行的结果。
  return shared[threadIdx.y];
}

/**
 * @brief Will be used in BlockYReduce, get the index of reduce_num in shared
 * memory.
 */
__device__ __forceinline__ int SharedMemoryIndex(int index) {
  return (threadIdx.y + index) * blockDim.x + threadIdx.x;
}

/**
 * @brief BlockYReduce reduce along blockDim.y.
 */
template <typename T, typename ReduceOp>
__device__ __forceinline__ T BlockYReduce(T val, ReduceOp reducer) {
  __shared__ T shared_memory[1024];
  shared_memory[SharedMemoryIndex(0)] = val;
  for (int stride = blockDim.y / 2; stride > 0; stride >>= 1) {
    __syncthreads();
    if (threadIdx.y < stride && threadIdx.y + stride < blockDim.y) {
      T temp = shared_memory[SharedMemoryIndex(stride)];
      val = reducer(val, temp);
    }
    shared_memory[SharedMemoryIndex(0)] = val;
  }
  __syncthreads();
  return shared_memory[threadIdx.x];
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
  (void)read_lens;
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
 */
template <typename T, int NX, int NY, bool IsBoundary = false>
__device__ __forceinline__ void ReadData(T *dst, const T *__restrict__ src,
                                         int num) {
  (void)NY;
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

template <typename T, int NX, int NY, bool IsBoundary = false>
__device__ __forceinline__ void ReadData(T *dst, const T *__restrict__ src,
                                         int num, int read_lens) {
  (void)read_lens;
  (void)NY;

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
 * @brief Read 1D data from global memory to register. The difference
 * from the above function is that it supports different data types of inputs.
 *
 * @template paraments
 * T: The type of data.
 * NX: Each thread load NX data from global memory continuously.
 * NY: Each thread need to load NY rows, only NY = 1 was supported.
 * ArgsT: The Type if dst, ArgsT can be std::tuple<T> or std::tuple<Args>
 * Index: The index of data stored in dst.
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
 */
template <typename T, int NX, int NY, typename ArgsT, int Index,
          bool IsBoundary = false>
__device__ __forceinline__ void ReadData(ArgsT *dst, const T *__restrict__ src,
                                         int num, int read_lens) {
  (void)NY;
  (void)read_lens;
  if (IsBoundary) { // blockDim.x * NX > num
    int thread_offset = threadIdx.x * NX;
#pragma unroll
    for (int idx = 0; idx < NX; ++idx) {
      if (idx + thread_offset < num) {
        std::get<Index>(dst[idx]) = src[thread_offset + idx];
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
        std::get<Index>(dst[idx]) = *(reinterpret_cast<T *>(vec_temp) + idx);
      }
    }
  }
}

/**
 * @brief Read 2D data from global memory to register according to Tx type, and
 * store it as Ty type into register.
 *
 * threadIdx.x is used as the thread index. Currently only GPU was supported.
 *
 * @template paraments
 * Tx: The type of data stored in the global memory.
 * Ty: The type of data that needs to be stored in registers.
 * NX: The number of data columns loaded by each thread.
 * NY: The number of data rows loaded by each thread.
 * IsBoundary: Indicates whether to perform block access storage out-of-bounds
 * judgment. When the number of data processed by the block is less than
 * NX x NY x blockDim, boundary judgment is required to avoid memory access
 * crossing the boundary.
 *
 * @param：
 * dst: The register pointer of the thread, the size is NX * NY.
 * src: The data pointer of the current block.
 * size_nx: The maximum offset of the current block is size_nx elements in the
 * lowest dimension. The parameters are only calculated when isboundary = true.
 * size_ny: The maximum offset of the current block is size_ny elements in the
 * first dimension. The parameters are only calculated when isboundary = true.
 * stride_nx: Each read one element stride stride_nx elements in the last dim.
 * stride_ny: Each read one element stride stride_ny elements in the first dim.
 */
template <typename Tx, typename Ty, int NX, int NY, bool IsBoundary = false>
__device__ __forceinline__ void ReadData(Ty *dst, const Tx *__restrict__ src,
                                         int size_nx, int size_ny,
                                         int stride_nx, int stride_ny) {
  // launch 1D block
  int thread_offset = threadIdx.x;
  // remdiner elements.
  int left_size_nx = size_nx - thread_offset;

  // Each branch is added for better performance
  if (NX == 1 && NY == 1) { // for NX == 1 and NY == 1
    // dst is one element registor
    // read one element from threadIdx.x
    if (IsBoundary) {
      if (left_size_nx > 0) {
        dst[0] = static_cast<Ty>(src[thread_offset]);
      }
    } else {
      dst[0] = static_cast<Ty>(src[thread_offset]);
    }
  } else if (NX == 1) { // for NX == 1 and NY != 1
                        // dst is NY elements registor
    // read NY elements from threadIdx.x, which is stride in y dim
#pragma unroll
    for (int idy = 0; idy < NY; ++idy) {
      if (IsBoundary) {
        if (idy * stride_ny >= size_ny) { // ?
          break;
        }
      }
      dst[idy] = static_cast<Ty>(src[thread_offset + idy * stride_ny]);
    }
  } else if (NY == 1) { // for NY == 1 and NX != 1
                        // dst is NX elements registor
    // read NX elements from threadIdx.x, which is stride in x dim
#pragma unroll
    for (int idx = 0; idx < NX; ++idx) {
      if (IsBoundary) {
        if (idx * stride_nx >= left_size_nx) {
          break;
        }
      }
      dst[idx] = static_cast<Ty>(src[thread_offset + idx * stride_nx]);
    }
  } else { // for NX != 1 and NY != 1
#pragma unroll
    for (int idx = 0; idx < NX; ++idx) {
      if (IsBoundary) {
        if (idx * stride_nx >= left_size_nx) {
          break;
        }
      }
#pragma unroll
      for (int idy = 0; idy < NY; ++idy) {
        if (IsBoundary) {
          if (idy * stride_ny >= size_ny) {
            break;
          }
        }
        dst[idy * NX + idx] = static_cast<Ty>(
            src[thread_offset + idx * stride_nx + idy * stride_ny]);
      }
    }
  }
}

/**
 * @brief Read 2D data from global memory to register with reduce form.
 *
 * @template paraments
 * T: The type of data.
 * NX: The number of data columns loaded by each thread.
 * NY: The number of data rows loaded by each thread.
 * threadIdx.x is used as the thread index. Currently only GPU was supported.
 * Rank: The shape size of out. eg in[1, 35], out[32, 35] then shape size is 2.
 * IsBoundary: Indicates whether to perform block access storage out-of-bounds
 * judgment. When the number of data processed by the block is less than
 * NX x NY x blockDim.x, boundary judgment is required to avoid memory access
 * crossing the boundary.
 *
 * @param：
 * dst: The register pointer of the thread, the size is NX * NY.
 * src: The input data pointer of this block.
 * block_offset: The data offset of this block, blockDim.x * blockIdx.x * NX.
 * index_cal: Calculation configuration of Reduce. It is used to calculate the
 * coordinate mapping relationship between output data and input data.
 * size_nx: The current block needs to load size_nx columns of data, this
 * parameter will participate in the calculation when isboundary = true.
 * size_ny: The current block needs to load size_ny rows of data, this parameter
 * will participate in the calculation when isboundary = true.
 * will be used when IsBoundary = true.
 * stride_nx: Each read one element stride stride_nx columns.
 * stride_ny: Each read one element stride stride_ny raws.
 * reduce_last_dim: Used to indicate whether the dimension of reduce contains
 * the lowest dimension.
 */
template <typename Tx, typename Ty, int NX, int NY, int Rank /*UNUSED*/,
          typename IndexCal, typename Functor, bool IsBoundary = false>
__device__ __forceinline__ void
ReadDataReduce(Ty *dst, const Tx *__restrict__ src, int block_offset,
               const IndexCal &index_cal, int size_nx, int size_ny,
               int stride_nx, int stride_ny, Functor func,
               bool reduce_last_dim) {
  (void)Rank;

  printf("ReadDataReduce: IsBoundary %d, reduce_last_dim %d, NX %d, NY %d,  "
         "size_nx %d, size_ny %d, stride_nx %d, stride_ny %d\n",
         IsBoundary, reduce_last_dim, NX, NY, size_nx, size_ny, stride_nx,
         stride_ny);

  int thread_offset = 0;
  int left_idx = 0;
  if (reduce_last_dim) {
    // (left, reduce)
    thread_offset = threadIdx.x;
    left_idx = threadIdx.y;
  } else {
    // thread_offset is reduce dim
    thread_offset = threadIdx.y;
    left_idx = threadIdx.x;
  }

  if (NX == 1) {
    // 1D kernel
#pragma unroll
    for (int ny = 0; ny < NY; ++ny) {
      if (IsBoundary) {
        if (thread_offset >= size_ny /*size*/) {
          break;
        }
      }
      uint32_t index_src = index_cal(thread_offset + block_offset);
      dst[ny] = static_cast<Ty>(func(src[index_src]));
      thread_offset += stride_ny /*stride*/;
    }
  } else {
#pragma unroll
    for (int nx = 0; nx < NX; ++nx) {
#pragma unroll
      for (int ny = 0; ny < NY; ++ny) {
        if (IsBoundary) {
          if ((thread_offset >= size_ny) ||
              (left_idx + nx * stride_nx >= size_nx)) {
            break;
          }
        }
        uint32_t index_src = index_cal(thread_offset + block_offset);
        dst[nx + ny * NX] = static_cast<Ty>(func(src[index_src]));
        thread_offset += stride_ny;
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
  (void)NY;
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
  (void)NY;
  (void)read_lens;
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
 * @brief Perform unary calculation according to OpFunc. Shape of input and
 * output are the same.
 *
 * @template paraments
 * InT: The data type of in.
 * OutT: The data type of out.
 * NX: The number of data columns loaded by each thread.
 * NY: The number of data rows loaded by each thread.
 * threadIdx.x is used as the thread index. Currently only GPU was supported.
 * OpFunc: Compute functor which has an operator() as following:
 *     template <typename InT, typename OutT>
 *     struct XxxFunctor {
 *       HOSTDEVICE OutT operator()(const InT& a) const {
 *         return ...;
 *       }
 *     };
 *
 * @param：
 * out: The register pointer of out, the size is NX * NY.
 * in: The register pointer of in, the size is NX * NY.
 * compute: Compute function which was declared like OpFunc<InT, OutT>().
 */
template <typename InT, typename OutT, int NX, int NY, class OpFunc>
__device__ __forceinline__ void ElementwiseUnary(OutT *out, const InT *in,
                                                 OpFunc compute) {
#pragma unroll
  for (int idx = 0; idx < NX * NY; idx++) {
    out[idx] = static_cast<OutT>(compute(in[idx]));
  }
}

/**
 * @brief The Reduce provides collective methods for computing a parallel
 * reduction of items partitioned across a CUDA block and intra thread. When
 * ReduceMode == kLocalMode, use shared memory to reduce between threads.When
 * ReduceMode == kGlobalMode, thread reduce along nx.
 *
 * @template paraments
 * T: The type of data.
 * NX: The number of data continuously loaded by each thread.
 * NY: The number of data rows loaded by each thread, only NY = 1 was supported.
 * threadIdx.x is used as the thread index. Currently only GPU was supported.
 * ReduceFunctor: Compute functor which has an operator() as following
 *     template <typename InT>
 *     struct ReduceFunctor {
 *       HOSTDEVICE InT operator()(const InT& a, const InT& b) const {
 *         return ...;
 *       }
 *     };
 * ReduceMode: Reduce mode, can be kLocalMode, kGlobalMode.
 *
 * @param
 * out: The register pointer of out, the size is NX * NY.
 * in: The register pointer of in, the size is NX * NY.
 * reducer: Compute function which was declared like ReduceFunctor<InT>().
 * reduce_last_dim: if the last dim gets involved in reduction.
 */
template <typename T, int NX, int NY, class ReduceFunctor,
          details::ReduceMode Mode>
__device__ __forceinline__ void
Reduce(T *out, const T *in, ReduceFunctor reducer, bool reduce_last_dim) {

  if (Mode == details::ReduceMode::kGlobalMode) {
    // globalMode, reduce in block
    (void)in;

    // reduce_last_dim 和 block_reduce_y 互斥
    bool block_reduce_y = (!reduce_last_dim) && (blockDim.y > 1);

    // when reduce is not required for the last dim, and reduce num has been
    // split into multiple threads
    if (block_reduce_y) {
#pragma unroll
      for (int i = 0; i < NY * NX; i++) { // reduce along blockdim.y
        out[i] = details::BlockYReduce<T, ReduceFunctor>(out[i], reducer);
      }
    }

    // when last dimension need to be reduced
    if (reduce_last_dim) {
#pragma unroll
      for (int i = 0; i < NY * NX; i++) { // reduce along blockDim.x
        out[i] = details::BlockXReduce<T, ReduceFunctor>(out[i], reducer);
      }
    }

  } else {
    // kLocalMode, reduce in registor
    // in -> out
    // in NY x NX, reduce along x dim
    // out is NY elements
#pragma unroll
    for (int i = 0; i < NY; ++i) {
#pragma unroll
      for (int j = 0; j < NX; ++j) {
        out[i] = reducer(out[i], in[i * NX + j]);
      }
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

struct DimConfig {
  int split_num_x; // gridDim.x
  int split_num_y; // gridDim.y
  int split_num_z; // gridDim.z

  int deal_size_x; // blockDim.x
  int deal_size_y; // blockDim.y
  int deal_size_z; // blockDim.z

  int rem_x; // remainder.x
  int rem_y; // remainder.y
  int rem_z; // remainder.z

  HOSTDEVICE explicit inline DimConfig(int split_x, int split_y, int split_z,
                                       int size_x, int size_y, int size_z) {
    split_num_x = split_x;
    split_num_y = split_y;
    split_num_z = split_z;
    deal_size_x = size_x;
    deal_size_y = size_y;
    deal_size_z = size_z;
  }

  HOSTDEVICE explicit inline DimConfig() {}

  HOSTDEVICE void SetRem(int rem_nx, int rem_ny, int rem_nz) {
    rem_x = rem_nx;
    rem_y = rem_ny;
    rem_z = rem_nz;
  }
};

inline std::ostream &operator<<(std::ostream &os, const DimConfig &dim) {
  os << "DimConfig: ";
  os << "block (" << dim.deal_size_x << ", " << dim.deal_size_y << ", "
     << dim.deal_size_z << ") ";
  os << "grid (" << dim.split_num_x << ", " << dim.split_num_y << ", "
     << dim.split_num_z << ") ";
  os << "remainder (" << dim.rem_x << ", " << dim.rem_y << ", " << dim.rem_z
     << ") ";
  os << std::endl;
  return os;
}

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

namespace detail {

template <typename T, size_t kStart, size_t kEnd, bool kStop>
struct UnrollVarArgsAssignImpl {
  template <typename... Args>
  HOSTDEVICE inline static void Run(T *d, T val, Args... args) {
    static_assert(sizeof...(args) + 1 == kEnd - kStart, "Wrong argument");
    d[kStart] = val;
    UnrollVarArgsAssignImpl<T, kStart + 1, kEnd, kStart + 1 == kEnd>::Run(
        d, args...);
  }
};

template <typename T, size_t kStart, size_t kEnd>
struct UnrollVarArgsAssignImpl<T, kStart, kEnd, true> {
  HOSTDEVICE inline static void Run(T *d) {}
};

template <typename T> struct UnrollVarArgsAssign {
  template <typename... Args>
  HOSTDEVICE inline static void Run(T *d, Args... args) {
    UnrollVarArgsAssignImpl<T, 0, sizeof...(Args), sizeof...(Args) == 0>::Run(
        d, args...);
  }
};

template <size_t kStart, size_t kEnd, bool kStop> struct UnrollCompare {
  template <typename T>
  HOSTDEVICE inline static bool Run(const T *d1, const T *d2) {
    return d1[kStart] == d2[kStart] &&
           UnrollCompare<kStart + 1, kEnd, kStart + 1 == kEnd>::Run(d1, d2);
  }
};

template <size_t kStart, size_t kEnd> struct UnrollCompare<kStart, kEnd, true> {
  template <typename T>
  HOSTDEVICE inline constexpr static bool Run(const T *d1 UNUSED,
                                              const T *d2 UNUSED) {
    return true;
  }
};

template <size_t kStart, size_t kEnd, bool kStop> struct UnrollFillConstant {
  template <typename T> HOSTDEVICE inline static void Run(T *data, T val) {
    data[kStart] = val;
    UnrollFillConstant<kStart + 1, kEnd, kStart + 1 == kEnd>::Run(data, val);
  }
};

template <size_t kStart, size_t kEnd>
struct UnrollFillConstant<kStart, kEnd, true> {
  template <typename T>
  HOSTDEVICE inline static void Run(T *data UNUSED, T val UNUSED) {}
};

} // namespace detail

template <typename T>
using UnrollVarArgsAssign = detail::UnrollVarArgsAssign<T>;

template <size_t N> using UnrollCompare = detail::UnrollCompare<0, N, N == 0>;

template <size_t N>
using UnrollFillConstant = detail::UnrollFillConstant<0, N, N == 0>;

template <typename T, size_t N> class Array {
public:
  static constexpr size_t kSize = N;

  HOSTDEVICE inline Array() {}

  template <typename... Args>
  HOSTDEVICE inline explicit Array(const T &val, Args... args) {
    static_assert(N == sizeof...(Args) + 1, "Invalid argument");
    UnrollVarArgsAssign<T>::Run(data_, val, args...);
  }

  HOSTDEVICE inline void Fill(const T &val) {
    UnrollFillConstant<N>::Run(data_, val);
  }

  HOSTDEVICE inline const T *Get() const { return data_; }

  HOSTDEVICE inline T *GetMutable() { return data_; }

  HOSTDEVICE inline T &operator[](size_t i) { return *advance(data_, i); }

  // Writing "return data_[i]" would cause compilation warning/error:
  // "array subscript is above array bound" in Python 35 CI.
  // It seems that it is a false warning of GCC if we do not check the bounds
  // of array index. But for better performance, we do not check in operator[]
  // like what is in STL. If users want to check the bounds, use at() instead
  HOSTDEVICE inline const T &operator[](size_t i) const {
    return *advance(data_, i);
  }

  HOSTDEVICE inline T &at(size_t i) {
#if !defined(__CUDA_ARCH__) && !defined(__HIPCC__)
    assert(i < N);
#endif
    return (*this)[i];
  }

  HOSTDEVICE inline const T &at(size_t i) const {
#if !defined(__CUDA_ARCH__) && !defined(__HIPCC__)
    assert(i < N);
#endif
    return (*this)[i];
  }

  HOSTDEVICE constexpr size_t size() const { return N; }

  HOSTDEVICE inline bool operator==(const Array<T, N> &other) const {
    return UnrollCompare<N>::Run(data_, other.data_);
  }

  HOSTDEVICE inline bool operator!=(const Array<T, N> &other) const {
    return !(*this == other);
  }

private:
  template <typename U> HOSTDEVICE static inline U *advance(U *ptr, size_t i) {
    return ptr + i;
  }

  T data_[N] = {};
};

constexpr int kMaxRank = 9;

// Convert dims from vector to array
template <typename T, size_t ElementCount, typename VectorLikeType>
static inline Array<T, ElementCount> VectorToArray(const VectorLikeType &vec) {
  assert(vec.size() <= ElementCount);

  size_t n = static_cast<size_t>(vec.size());
  Array<T, ElementCount> ret;
  for (size_t i = 0; i < n; ++i) {
    ret[i] = vec[i];
  }
  return ret;
}

struct IndexCalculator {
  IndexCalculator(int cal_rank, const std::vector<int> &cal_dims,
                  const std::vector<int> &cal_strides,
                  const std::vector<int> &full_strides)
      : rank(cal_rank) {

    strides = VectorToArray<int, kMaxRank>(full_strides);

    dims = VectorToArray<int, kMaxRank>(cal_dims);

    std::vector<kps::details::FastDivMod> cal_divmoders;
    // fast divmod
    for (auto i : cal_strides) {
      cal_divmoders.push_back(kps::details::FastDivMod(i));
    }
    divmoders =
        VectorToArray<kps::details::FastDivMod, kMaxRank>(cal_divmoders);
  }

  __device__ inline int operator()(int offset) const {
    int index = 0;
#pragma unroll
    for (int i = 0; i < kMaxRank; ++i) {
      if (i == rank) {
        break;
      }
      auto divmod = divmoders[i].Divmod(offset);
      index += (divmod.val[0] * strides[dims[i]]);
      offset = divmod.val[1]; // remainder
    }
    return index;
  }

  int rank;
  Array<int, kMaxRank> dims;

  Array<int, kMaxRank> strides;

  Array<kps::details::FastDivMod, kMaxRank> divmoders;
};

template <bool ReduceLastDim = false> struct ReduceIndexMapping {

  const kps::DimConfig dim;
  int loop_size;

  HOSTDEVICE ReduceIndexMapping(const kps::DimConfig &dims, int max_loop = 1)
      : dim(dims), loop_size(max_loop) {}

  HOSTDEVICE ReduceIndexMapping() {}

  __device__ __forceinline__ int BlockIdX() { return blockIdx.x; }

  __device__ __forceinline__ int BlockIdY() { return blockIdx.y; }

  __device__ __forceinline__ int BlockDimX() { return blockDim.x; }

  __device__ __forceinline__ int BlockDimY() { return blockDim.y; }

  __device__ __forceinline__ int GridDimX() { return gridDim.x; }

  __device__ __forceinline__ int GridDimY() { return gridDim.y; }

  __device__ int GetLoopSize() { return 1; }
};

template <typename Ty, int dev_id = 0> struct ReduceConfig {
  ReduceConfig(const std::vector<int> &origin_reduce_dims,
               const std::vector<int> &origin_x_dim)
      : reduce_dims_origin(origin_reduce_dims), x_dim(origin_x_dim),
        devid(dev_id) {
    std::cout << "0. input x_dim: " << x_dim;
    std::cout << "0. input reduce_dims: " << reduce_dims_origin;
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
      std::cout << "should_reduce_again: " << size << std::endl;
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
    constexpr int max_num_threads = kps::details::kReduceMaxThread; // 128

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
      // (left_num, reduce_num)
      block_x = GetBlockDim(reduce_num);
      block_y = GetBlockDim(left_num);
      block_dim->x = block_x;
      block_dim->y =
          std::min(block_y, static_cast<int>(max_num_threads / block_dim->x));
      // num of blockDim.y
      grid_num = kps::details::DivUp(left_num, block_dim->y);
      // VecSize, num of blockDim.x
      reduce_num_per_thread = kps::details::DivUp(reduce_num, block_dim->x);
    } else {
      // (reduce_num, left_num)
      block_x = GetBlockDim(left_num);
      block_y = GetBlockDim(reduce_num);
      block_dim->x = std::min(block_x, 32);
      block_dim->y =
          std::min(block_y, static_cast<int>(max_num_threads / block_dim->x));
      block_dim->x =
          std::min(block_x, static_cast<int>(max_num_threads / block_dim->y));
      // num of BlockDim.x
      grid_num = kps::details::DivUp(left_num, block_dim->x);
      // num of blockDim.y, VecSize
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

    grid_dim->x = grid_num; // left
    grid_dim->y = std::max(std::min(input_split_num_1, input_split_num_3),
                           input_split_num_2); // reduce

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
    // grid_dim (left_num, 1, 1)

    int last_dim_num = x_dim.back();
    // Update left_num
    int grid_z = left_num / last_dim_num; // always 1?
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

    // init X dim
    block_dim->x = GetBlockDim(left_num); // <= left_num, or == 128
    // num of blockDim.x
    grid_dim->x = kps::details::DivUp(left_num, block_dim->x);

    // fix blocking_size, compute gradDim.y
    int num_block = (max_threads / left_num);
    blocking_size = reduce_num;
    if (num_block > 1 && reduce_num >= REDUCE_SPLIT_BOUNDARY) {
      blocking_size = kps::details::RoundDownPowOfTwo(reduce_num / num_block);

      if (blocking_size <= 1) {
        blocking_size = kps::details::RoundDownPowOfTwo(sqrt(reduce_num));
      } else if (blocking_size * 2 < reduce_num) {
        blocking_size *= 2;
      }

      should_reduce_again = true;
      std::cout << "should_reduce_again: " << should_reduce_again << std::endl;

      // reduce on Y dim, num of blocking_size
      grid_dim->y = kps::details::DivUp(reduce_num, blocking_size);
    }
  }

  void SetBlockDim() {
    // 次函数最早为 ReduceHighDim 设计。 (reduce_num, left_num)
    // 后魔改支持RedceLastDim. (left_num, reduce_num)
    NameGuard g("SetBlockDim");
    dim3 block_dim(1, 1, 1);
    dim3 grid_dim(left_num, 1, 1); // when reduceHigh, left_num for X dim
    blocking_size = reduce_num;    // blockDim

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

// when reduce_dim.size() == 1 and reduce_dim[0] == x_dim.size() - 1, or
// when reduce_dim.size() != 1 and reduce_dim.size() != x_dim.size(), this
// function will be used
template <typename Tx, typename Ty, typename MPType, typename ReduceOp,
          typename TransformOp, typename Calculator>
__global__ void
ReduceAnyKernel(const Tx *x, Ty *y, ReduceOp reducer, TransformOp transformer,
                MPType init, int reduce_num, int left_num, bool reduce_last_dim,
                const Calculator reduce_index_calculator,
                const Calculator left_index_calculator,
                const kps::DimConfig dim, bool is_mean) {
  int input_idx, left_idx, stride;
  int block_size = 0;
  bool need_store = true;
  int loop_left = 0;
  int tid = 0;
  // the last dim gets involved in reduction
  int store_offset = 0;
  int stride_left = 0;

  if (reduce_last_dim) {
    // (left_num, reduce_num)
    auto block = ReduceIndexMapping<true>();
    // x 轴遍历
    tid = THREAD_ID_X;

    // blockDim.x
    block_size = block.BlockDimX();
    printf("reduce last dim: block_size %d\n", block_size);

    // left offset and stride in Y dim
    // block.BlockIdX() == index num of blockDim.y
    left_idx = block.BlockIdX() * block.BlockDimY() + THREAD_ID_Y;
    printf("reduce last dim: left_idx %d, blockIdx.x %d, blockDim.y %d, "
           "threadIdx.y %d\n",
           left_idx, block.BlockIdX(), block.BlockDimY(), THREAD_ID_Y);
    stride_left = 1;

    // CUDA 时，一次处理所有的Y，不需要在Y上循环
    // left_num - left_idx > 0, loop_left = 1
    // left_num - left_idx == 0, loop_left = 0
    // left_num - left_idx < 0, loop_left < 0
    loop_left = min(block.GetLoopSize(), left_num - left_idx);
    assert(loop_left <= 1);
    printf("reduce last dim: left_num %d, left_idx %d, loop_left %d, loop size "
           "%d\n",
           left_num, left_idx, loop_left, block.GetLoopSize());

    // block offset and stride in X dim
    // block.BlockIdY() == index num of blockDim.x
    input_idx = block.BlockIdY() * block.BlockDimX();
    printf("reduce last dim: input_idx %d, blockIdx.y %d, blockDim.x %d\n",
           input_idx, block.BlockIdY(), block.BlockDimX());
    //  block.GridDimY() == num of blockDim.x
    stride = block.GridDimY() * block.BlockDimX();
    printf("reduce last dim: stride %d, blockDim.y %d, blockDim.x %d\n", stride,
           block.GridDimY(), block.BlockDimX());

    // store reduce X result in (left_num, block_num_x)
    // offset for out, flag for left_index < left_num
    need_store = (THREAD_ID_X == 0) && (left_idx < left_num);
    printf("reduce last dim: need_store %d, threadIdx.x %d, left_idx %d, "
           "left_num %d\n",
           need_store, THREAD_ID_X, left_idx, left_num);
    // block.BlockIdY() = index num of block in X dim. Colomn Major?
    store_offset = block.BlockIdY() * left_num + left_idx;
    printf("reduce last dim: store_offset %d, blockIdx.y %d, left_idx %d, "
           "left_num %d\n",
           store_offset, block.BlockIdY(), left_idx, left_num);

  } else {
    // 这个分支是正常的，上面那个应该是后修改的。
    // (reduce_num, left_num)
    // Reduce Y dim
    auto block = ReduceIndexMapping<false>();
    tid = THREAD_ID_Y;

    // blockDim.y
    block_size = block.BlockDimY();

    input_idx = block.BlockIdY() * block.BlockDimY();
    stride = block.GridDimY() * block.BlockDimY();

    left_idx = block.BlockIdX() * block.BlockDimX() + THREAD_ID_X;
    stride_left = block.BlockDimX() * block.GridDimX();

    // loop_left <= 1
    loop_left = min(block.GetLoopSize(), left_num - left_idx);
    assert(loop_left <= 1);

    need_store = (THREAD_ID_Y == 0) && (left_idx < left_num);
    store_offset = block.BlockIdY() * left_num + left_idx;
  }

  // calculate the offset, means the addr where each thread really start.
  // 1. reduce for each thread
  MPType input_compute[REDUCE_VEC_SIZE]; // vecsize=4

  int input_idx_tmp = input_idx;

  // 1. loop on left dim
  // for reduce_last_dim, if valid, always i == 0
  for (int i = 0; i < loop_left; i += stride_left) {
    // in offset
    int input_offset = left_index_calculator(left_idx + i);
    // in + offset ptr
    const Tx *input = x + input_offset;

    MPType reduce_var = init;

    // load REDUCE_VEC_SIZE data once, and then compute
    int bound = reduce_num - (REDUCE_VEC_SIZE - 1) * stride;
    printf("ReduceAnyKernel: bound %d, reduce_num %d, stride %d\n", bound,
           reduce_num, stride);

    input_idx = input_idx_tmp;

    printf("ReduceAnyKernel: input_idx %d, block_size %d,  input_idx + "
           "block_size < bound %d, REDUCE_VEC_SIZE * stride %d\n",
           input_idx, block_size, input_idx + block_size < bound,
           REDUCE_VEC_SIZE * stride);
    // 1.0, evenly divisible reduce. Reduce gloabl problem into grid problem
    for (; input_idx + block_size < bound;
         input_idx += REDUCE_VEC_SIZE * stride) {
      printf("ReduceAnyKernel: in bound\n");
      // (VecSize, 1)， interleave read, not mem friend
      kps::ReadDataReduce<Tx, Tx, 1, REDUCE_VEC_SIZE, 1, Calculator,
                          TransformOp, false /*IsBoundary*/>(
          &input_compute[0], input, input_idx, reduce_index_calculator,
          1 /*size_nx*/, reduce_num /*size_ny*/, 1 /*stride_x*/,
          stride /*stride_y*/, transformer, reduce_last_dim);

      // (NY,NX) -> (NY,1), e.g. (1, VecSize) -> (1,1)， local reduce, mem
      // friend
      kps::Reduce<MPType, REDUCE_VEC_SIZE, 1, ReduceOp,
                  kps::details::ReduceMode::kLocalMode>(
          &reduce_var, &input_compute[0], reducer, reduce_last_dim);
    }

    // 1.1 remainder global reduce
    // reg init. not needed, since ReadDataReduce using assignment, not depend
    // on old val.
    kps::Init<MPType, REDUCE_VEC_SIZE>(&input_compute[0], init);

    // read REDUCE_VEC_SIZE into col vector (VEC_SIZE, 1)
    kps::ReadDataReduce<Tx, MPType, 1 /*NX*/, REDUCE_VEC_SIZE /*NY*/,
                        1 /*Rank*/, Calculator, TransformOp, true>(
        &input_compute[0], input, input_idx /*block_offset*/,
        reduce_index_calculator, 1 /*size_nx*/,
        reduce_num - input_idx /*size_ny*/, 1 /*stride_nx*/,
        stride /*stride_ny*/, transformer, reduce_last_dim);

    // LocalReduce, given (NY, NX)，reduce on x axis, has NY result
    kps::Reduce<MPType, REDUCE_VEC_SIZE /*NX*/, 1 /*NY*/, ReduceOp,
                kps::details::ReduceMode::kLocalMode>(
        &reduce_var, &input_compute[0], reducer, reduce_last_dim);

    // Block Reduce
    kps::Reduce<MPType, 1 /*NX*/, 1 /*NY*/, ReduceOp,
                kps::details::kGlobalMode>(&reduce_var, &reduce_var, reducer,
                                           reduce_last_dim);

    if (is_mean) {
      // ReduceMean
      reduce_var = reduce_var / static_cast<MPType>(reduce_num);
    }

    // write result
    Ty result = static_cast<Ty>(reduce_var);
    kps::details::WriteData<Ty>(y + store_offset + i, &result,
                                static_cast<int>(need_store));
  }
}

template <typename Tx, typename Ty, typename MPType, typename ReduceOp,
          typename TransformOp>
__global__ void
ReduceHigherDimKernel(const Tx *x, Ty *y, ReduceOp reducer,
                      TransformOp transformer, MPType init, int reduce_num,
                      int left_num, int blocking_size, const kps::DimConfig dim,
                      int mean_div, bool is_mean) {
  // when reduce_dim.size() == 1 and reduce_dim[0] != x_dim.size() - 1, this
  // function will be used
  auto block = ReduceIndexMapping<false>(dim);
  int idy = block.BlockIdY() * blocking_size;
  int idx = block.BlockIdX() * block.BlockDimX();
  int idz = BLOCK_ID_Z * left_num;
  int stride = dim.split_num_x * dim.deal_size_x;
  int size = left_num - dim.rem_x;
  int loop_size = min(reduce_num - idy, blocking_size);
  int store_offset = block.BlockIdY() * left_num + idz * block.GridDimY();
  int block_offset = idy * left_num + idz * reduce_num;
  const Tx *input = x + block_offset;
  Tx reduce_input;
  for (; idx < size; idx += stride) {
    MPType reduce_var = init;
    MPType reduce_compute = init;
    for (int loop_idx = 0; loop_idx < loop_size; ++loop_idx) {
      kps::ReadData<Tx, Tx, 1, 1, false>(&reduce_input,
                                         input + loop_idx * left_num + idx,
                                         block.BlockDimX(), 1, 1, left_num);
      kps::ElementwiseUnary<Tx, MPType, 1, 1, TransformOp>(
          &reduce_compute, &reduce_input, transformer);
      kps::Reduce<MPType, 1, 1, ReduceOp, kps::details::ReduceMode::kLocalMode>(
          &reduce_var, &reduce_compute, reducer, false);
    }
    if (is_mean) {
      reduce_var = reduce_var / static_cast<MPType>(mean_div);
    }
    Ty result = static_cast<Ty>(reduce_var);
    kps::WriteData<Ty, 1, 1, false>(y + store_offset + idx, &result,
                                    block.BlockDimX());
  }

  if (idx < left_num) {
    MPType reduce_var = init;
    MPType reduce_compute = init;
    for (int loop_idx = 0; loop_idx < loop_size; ++loop_idx) {
      kps::ReadData<Tx, Tx, 1, 1, true>(&reduce_input,
                                        input + loop_idx * left_num + idx,
                                        dim.rem_x, 1, 1, left_num);
      kps::ElementwiseUnary<Tx, MPType, 1, 1, TransformOp>(
          &reduce_compute, &reduce_input, transformer);
      kps::Reduce<MPType, 1, 1, ReduceOp, kps::details::ReduceMode::kLocalMode>(
          &reduce_var, &reduce_compute, reducer, false);
    }

    if (is_mean) {
      reduce_var = reduce_var / static_cast<MPType>(mean_div);
    }
    Ty result = static_cast<Ty>(reduce_var);
    kps::WriteData<Ty, 1, 1, true>(y + store_offset + idx, &result, dim.rem_x);
  }
}

// when reduce_type == kReduceLastDim this struct will be used
// for higher performance
struct OneDimIndexCal {
  int stride;

  explicit OneDimIndexCal(int stride) : stride(stride) {}

  __device__ inline int operator()(int index) const { return index * stride; }
};

template <typename Tx, typename Ty, typename MPType, typename ReduceOp,
          typename TransformOp>
static void LaunchReduceKernel(const Tx *x_data, Ty *y_data,
                               const ReduceOp &reducer,
                               const TransformOp &transform, MPType init,
                               const cudaStream_t &stream,
                               ReduceConfig<Ty> config, bool is_mean = false) {

  // 1. when reduce_dim.size() == 1 and reduce_dim[0] == x_dim.size() - 1, or
  // 2. when reduce_dim.size() != 1 and reduce_dim.size() != x_dim.size(), this
  // function will be used

  NameGuard g("LaunchReduceKernel");
  if (config.reduce_type == kReduceLastDim) {
    NameGuard g("kReduceLastDim");
    // dims (Y, X), strides (X, 1) X is reduce dim, Y is left dim

    // rank 2, reduce last dim
    // rank 3, reduce last dim

    // dims (4, 25), reduce_dims (1);
    // dims (4, 5, 5), reduce dims (2)
    // reduce_lst_dim = 1, should_reduce_agin =  0

    // dims (Y, X), strides (X, 1) X is reduce dim, Y is left dim
    int stride_reduce = 1;
    int stride_left = config.reduce_num; // reduce_num == X

    // for higher performance
    // from index in src ptr, to (y,x) index in matrix
    // reduce_index_calculator for x, left_index_calculator for y
    auto reduce_index_calculator = OneDimIndexCal(stride_reduce);
    auto left_index_calculator = OneDimIndexCal(stride_left);

    kps::DimConfig dim =
        kps::DimConfig(config.grid.x, config.grid.y, config.grid.z,
                       config.block.x, config.block.y, 0);
    // reducing along X axis, compute remaindor when blockDim.x
    dim.SetRem(config.reduce_num % config.block.x, 0, 0);
    std::cout << dim;

    auto grids = config.grid;
    auto blocks = config.block;

    // only OneDimIndexCal diff
    ReduceAnyKernel<Tx, Ty, MPType, ReduceOp, TransformOp, OneDimIndexCal>
        <<<grids, blocks, 0, stream>>>(
            x_data, config.output_data, reducer, transform, init,
            config.reduce_num, config.left_num, config.reduce_last_dim,
            reduce_index_calculator, left_index_calculator, dim,
            is_mean && (!config.should_reduce_again));
  } else {
    NameGuard g("kReduceFirstDim");
    // dims (Y, X), strides (X, 1), (Y) is reduce dim, X is left dim

    // 多少个reduce dim
    int reduce_rank = config.reduce_strides.size();
    // 剩余多少个dim不变
    int left_rank = config.left_strides.size();

    // from index in src ptr, to (y,x) index in matrix
    // reduce_index_calculator for y, left_index_calculator for x
    auto reduce_index_calculator =
        IndexCalculator(reduce_rank, config.reduce_dim, config.reduce_strides,
                        config.x_strides);
    auto left_index_calculator = IndexCalculator(
        left_rank, config.left_dim, config.left_strides, config.x_strides);

    kps::DimConfig dim =
        kps::DimConfig(config.grid.x, config.grid.y, config.grid.z,
                       config.block.x, config.block.y, 0);
    // reducing along X axis, compute remaindor when blockDim.x
    dim.SetRem(config.reduce_num % config.block.x, 0, 0);
    std::cout << dim;

    auto grids = config.grid;
    auto blocks = config.block;

    // only IndexCalculator diff
    ReduceAnyKernel<Tx, Ty, MPType, ReduceOp, TransformOp, IndexCalculator>
        <<<grids, blocks, 0, stream>>>(
            x_data, config.output_data, reducer, transform, init,
            config.reduce_num, config.left_num, config.reduce_last_dim,
            reduce_index_calculator, left_index_calculator, dim,
            is_mean && (!config.should_reduce_again));
  }

  if (config.should_reduce_again) {
    NameGuard g("should_reduce_again");
    dim3 block;
    dim3 grid;
    if (config.reduce_last_dim) {
      block = dim3(32, 1, 1);
      grid = dim3(kps::details::DivUp(config.left_num, 32), 1, 1);
    } else {
      block = dim3(config.block.x, 1, 1);
      grid = dim3(config.grid.x, 1, config.grid.z);
    }

    kps::DimConfig dim =
        kps::DimConfig(grid.x, grid.y, grid.z, block.x, config.grid.y, 0);
    dim.SetRem(config.left_num % block.x, 0, 0);

    auto grid_size = grid;
    auto block_size = block;

    // tmp data to out
    ReduceHigherDimKernel<Ty, Ty, MPType, ReduceOp,
                          kps::IdentityFunctor<Ty, MPType>>
        <<<grid_size, block_size, 0, stream>>>(
            config.output_data, y_data, reducer,
            kps::IdentityFunctor<Ty, MPType>(), init, config.grid.y,
            config.left_num, config.grid.y, dim, config.reduce_num, is_mean);
  }
}

template <typename Tx, typename Ty, template <typename> class ReduceOp,
          typename TransformOp>
void ReduceKernel(const cudaStream_t &stream, const Tx *x_data, Ty *y_data,
                  const std::vector<int> &x_dim,
                  const std::vector<int> &origin_reduce_dims,
                  const TransformOp &transform, bool is_mean = false) {

  auto config = ReduceConfig<Ty>(origin_reduce_dims, x_dim);
  config.Run();
  config.SetOutputData(y_data);
  std::cout << config;

  using MPType = Ty;
  auto reducer = ReduceOp<MPType>();

  // when reduce_dim.size() == 1 and reduce_dim[0] == x_dim.size() - 1, or
  // when reduce_dim.size() != 1 and reduce_dim.size() != x_dim.size(), this
  // function will be used
  LaunchReduceKernel<Tx, Ty, MPType, ReduceOp<MPType>, TransformOp>(
      x_data, y_data, reducer, transform, reducer.initial(), stream, config,
      is_mean);
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

int case1(void) {
  std::cout << "case1" << std::endl;
  constexpr uint32_t i{64}, j{1000};
  constexpr uint32_t n{i * j};

  constexpr uint32_t num_repeats{1};
  constexpr uint32_t num_warmups{0};

  std::vector<float> vec_in(n, 1.0);
  std::vector<float> vec_out(i);

  const float *h_input{vec_in.data()};
  float *h_output{vec_out.data()};

  float *d_input{nullptr};
  float *d_output{nullptr};

  CHECK_CUDA_ERROR(cudaMalloc(&d_input, vec_in.size() * sizeof(float)));
  CHECK_CUDA_ERROR(cudaMalloc(&d_output, vec_out.size() * sizeof(float)));

  CHECK_CUDA_ERROR(cudaMemcpy(d_input, h_input, sizeof(float) * vec_in.size(),
                              cudaMemcpyHostToDevice));

  cudaStream_t stream;
  CHECK_CUDA_ERROR(cudaStreamCreate(&stream));

  // cast. SumReduce, x = ones([64, 4096]), reduce_dims = (1)
  {
    using T = float;
    std::vector<int> dims{i, j};
    assert(NumElements(dims) == n);
    std::vector<int> reduce_dims{1};

    std::function<void(cudaStream_t &)> const function{
        std::bind(ReduceKernel<T, T, kps::AddFunctor, kps::IdentityFunctor<T>>,
                  std::placeholders::_1, d_input, d_output, dims, reduce_dims,
                  kps::IdentityFunctor<T>(), false)};

    float const latency{
        MeasurePerformance(function, stream, num_repeats, num_warmups)};
    PrintLatency(latency);

    CHECK_CUDA_ERROR(cudaMemcpy(h_output, d_output,
                                sizeof(float) * vec_out.size(),
                                cudaMemcpyDeviceToHost));

    for (int i{0}; i < vec_out.size(); ++i) {
      assert(static_cast<uint32_t>(vec_out[i]) == j);
    }
  }

  CHECK_CUDA_ERROR(cudaFree(d_input));
  CHECK_CUDA_ERROR(cudaFree(d_output));
  CHECK_CUDA_ERROR(cudaStreamDestroy(stream));
  return 0;
}

int case2(void) {
  std::cout << "case2" << std::endl;
  constexpr uint32_t i{64}, j{10}, k{100};
  constexpr uint32_t n{i * j * k};

  constexpr uint32_t num_repeats{1};
  constexpr uint32_t num_warmups{0};

  std::vector<float> vec_in(n, 1.0);
  std::vector<float> vec_out(i * j);

  const float *h_input{vec_in.data()};
  float *h_output{vec_out.data()};

  float *d_input{nullptr};
  float *d_output{nullptr};

  CHECK_CUDA_ERROR(cudaMalloc(&d_input, vec_in.size() * sizeof(float)));
  CHECK_CUDA_ERROR(cudaMalloc(&d_output, vec_out.size() * sizeof(float)));

  CHECK_CUDA_ERROR(cudaMemcpy(d_input, h_input, sizeof(float) * vec_in.size(),
                              cudaMemcpyHostToDevice));

  cudaStream_t stream;
  CHECK_CUDA_ERROR(cudaStreamCreate(&stream));

  // cast. SumReduce, x = ones([64, 64, 64]), reduce_dims = (2)
  {
    using T = float;
    std::vector<int> dims{i, j, k};
    assert(NumElements(dims) == n);
    std::vector<int> reduce_dims{2};

    std::function<void(cudaStream_t &)> const function{
        std::bind(ReduceKernel<T, T, kps::AddFunctor, kps::IdentityFunctor<T>>,
                  std::placeholders::_1, d_input, d_output, dims, reduce_dims,
                  kps::IdentityFunctor<T>(), false)};

    float const latency{
        MeasurePerformance(function, stream, num_repeats, num_warmups)};
    PrintLatency(latency);

    CHECK_CUDA_ERROR(cudaMemcpy(h_output, d_output,
                                sizeof(float) * vec_out.size(),
                                cudaMemcpyDeviceToHost));

    for (int i{0}; i < vec_out.size(); ++i) {
      assert(static_cast<uint32_t>(vec_out[i]) == k);
    }
  }

  CHECK_CUDA_ERROR(cudaFree(d_input));
  CHECK_CUDA_ERROR(cudaFree(d_output));
  CHECK_CUDA_ERROR(cudaStreamDestroy(stream));
  return 0;
}

int main(void) {
  case1();
  // cudaDeviceSynchronize();
  case2();
  return 0;
}