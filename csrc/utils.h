#pragma once

#include <cstddef>
#include <cstdint>
#include <torch/extension.h>

// NOTE:tensor malloc as device before we call
// e.g. data.to("cuda") in python
#define CHECK_CUDA(x)                                                          \
  TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x)                                                    \
  TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x)                                                         \
  CHECK_CUDA(x);                                                               \
  CHECK_CONTIGUOUS(x)

#define CUDA_ERROR_CHECK(condition)                                            \
  do {                                                                         \
    cudaError_t error = condition;                                             \
    if (error != cudaSuccess) {                                                \
      printf("CUDA_CHECK error in line %d of file %s \
              : %s \n",                                                        \
             __LINE__, __FILE__, cudaGetErrorString(error));                   \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)


// Inspired by
// https://github.com/NVIDIA/DALI/blob/main/include/dali/core/static_switch.h
// and https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/Dispatch.h
/// @param COND       - a boolean expression to switch by
/// @param CONST_NAME - a name given for the constexpr bool variable.
/// @param ...       - code to execute for true and false
///
/// Usage:
/// ```
/// BOOL_SWITCH(flag, BoolConst, [&] {
///     some_function<BoolConst>(...);
/// });
/// ```
#define BOOL_SWITCH(COND, CONST_NAME, ...)      \
  [&] {                                         \
    if (COND) {                                 \
      constexpr static bool CONST_NAME = true;  \
      return __VA_ARGS__();                     \
    } else {                                    \
      constexpr static bool CONST_NAME = false; \
      return __VA_ARGS__();                     \
    }                                           \
  }()

#define FP16_SWITCH(COND, ...)               \
  [&] {                                      \
    if (COND) {                              \
      using elem_type = cutlass::half_t;     \
      return __VA_ARGS__();                  \
    } else {                                 \
      using elem_type = cutlass::bfloat16_t; \
      return __VA_ARGS__();                  \
    }                                        \
  }()

#define FWD_HEADDIM_SWITCH(HEADDIM, ...)   \
  [&] {                                    \
    if (HEADDIM <= 32) {                   \
      constexpr static int kHeadDim = 32;  \
      return __VA_ARGS__();                \
    } else if (HEADDIM <= 64) {            \
      constexpr static int kHeadDim = 64;  \
      return __VA_ARGS__();                \
    } else if (HEADDIM <= 96) {            \
      constexpr static int kHeadDim = 96;  \
      return __VA_ARGS__();                \
    } else if (HEADDIM <= 128) {           \
      constexpr static int kHeadDim = 128; \
      return __VA_ARGS__();                \
    } else if (HEADDIM <= 160) {           \
      constexpr static int kHeadDim = 160; \
      return __VA_ARGS__();                \
    } else if (HEADDIM <= 192) {           \
      constexpr static int kHeadDim = 192; \
      return __VA_ARGS__();                \
    } else if (HEADDIM <= 224) {           \
      constexpr static int kHeadDim = 224; \
      return __VA_ARGS__();                \
    } else if (HEADDIM <= 256) {           \
      constexpr static int kHeadDim = 256; \
      return __VA_ARGS__();                \
    }                                      \
  }()

#define WARP_SWITCH(COND, CONST_NAME, ...)      \
  [&] {                                         \
    if (COND == 4) {                            \
      constexpr static int CONST_NAME = 4;      \
      return __VA_ARGS__();                     \
    } else if (COND == 8) {                     \
      constexpr static int CONST_NAME = 8;      \
      return __VA_ARGS__();                     \
    } else {                                    \
      constexpr static int CONST_NAME = 2;      \
      return __VA_ARGS__();                     \
    }                                           \
  }()

#define BLOCKM_SWITCH(COND, CONST_NAME, ...)     \
  [&] {                                         \
    if (COND == 64) {                    \
      constexpr static int CONST_NAME = 64;     \
      return __VA_ARGS__();                     \
    } else if (COND == 128) {                   \
      constexpr static int CONST_NAME = 128;    \
      return __VA_ARGS__();                     \
    } else if (COND == 256) {                   \
      constexpr static int CONST_NAME = 256;    \
      return __VA_ARGS__();                     \
    } else {                                    \
      constexpr static int CONST_NAME = 64;     \
      return __VA_ARGS__();                     \
    }                                           \
  }()

#define BLOCKN_SWITCH(COND, CONST_NAME, ...)     \
  [&] {                                         \
    if (COND == 32) {                           \
      constexpr static int CONST_NAME = 32;     \
      return __VA_ARGS__();                     \
    } else if (COND == 64) {                    \
      constexpr static int CONST_NAME = 64;     \
      return __VA_ARGS__();                     \
    } else if (COND == 128) {                   \
      constexpr static int CONST_NAME = 128;    \
      return __VA_ARGS__();                     \
    } else if (COND == 256) {                   \
      constexpr static int CONST_NAME = 256;    \
      return __VA_ARGS__();                     \
    } else {                                    \
      constexpr static int CONST_NAME = 64;     \
      return __VA_ARGS__();                     \
    }                                           \
  }()

#define STAGE_SWITCH(COND, CONST_NAME, ...)     \
  [&] {                                         \
    if (COND == 2) {                            \
      constexpr static int CONST_NAME = 2;      \
      return __VA_ARGS__();                     \
    } else if (COND == 3) {                     \
      constexpr static int CONST_NAME = 3;      \
      return __VA_ARGS__();                     \
    } else if (COND == 4) {                     \
      constexpr static int CONST_NAME = 4;      \
      return __VA_ARGS__();                     \
    } else if (COND == 5) {                     \
      constexpr static int CONST_NAME = 5;      \
      return __VA_ARGS__();                     \
    } else {                                    \
      constexpr static int CONST_NAME = 2;      \
      return __VA_ARGS__();                     \
    }                                           \
  }()


template<typename T>
struct MaxOp {
__device__ inline T operator()(T const & x, T const & y) { return x > y ? x : y; }
};

template <>
struct MaxOp<float> {
// This is slightly faster
__device__ inline float operator()(float const &x, float const &y) { return max(x, y); }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
struct SumOp {
__device__ inline T operator()(T const & x, T const & y) { return x + y; }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<int THREADS>
struct Allreduce {
    static_assert(THREADS == 32 || THREADS == 16 || THREADS == 8 || THREADS == 4);
    template<typename T, typename Operator>
    static __device__ inline T run(T x, Operator &op) {
        constexpr int OFFSET = THREADS / 2;
        x = op(x, __shfl_xor_sync(uint32_t(-1), x, OFFSET));
        return Allreduce<OFFSET>::run(x, op);
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
struct Allreduce<2> {
template<typename T, typename Operator> 
static __device__ inline T run(T x, Operator &op) {
    x = op(x, __shfl_xor_sync(uint32_t(-1), x, 1));
    return x;
}
};

// tensor:((2, MMA_M),(2, MMA_N))
// summary:(2 * MMA_N)
template<bool zero_init=true, typename Engine0, typename Layout0, typename Engine1, typename Layout1, typename Operator>
__device__ inline void thread_reduce_(Tensor<Engine0, Layout0> const &tensor, Tensor<Engine1, Layout1> &summary, Operator &op) {
    static_assert(Layout0::rank == 2, "Only support 2D Tensor");
    static_assert(Layout1::rank == 1, "Only support 1D Tensor");
    CUTE_STATIC_ASSERT_V(size<0>(summary) == size<0>(tensor));
    #pragma unroll
    for (int mi = 0; mi < size<0>(tensor); mi++) {
        summary(mi) = zero_init ? tensor(mi, 0) : op(summary(mi), tensor(mi, 0));
        #pragma unroll
        for (int ni = 1; ni < size<1>(tensor); ni++) {
            summary(mi) = op(summary(mi), tensor(mi, ni));
        }
    }
}

// summary:(2 * MMA_N)
// summary:(2 * MMA_N)
template<typename Engine0, typename Layout0, typename Engine1, typename Layout1, typename Operator>
__device__ inline void quad_allreduce_(Tensor<Engine0, Layout0> &dst, Tensor<Engine1, Layout1> &src, Operator &op) {
    CUTE_STATIC_ASSERT_V(size(dst) == size(src));
    #pragma unroll
    for (int i = 0; i < size(dst); i++){
        // NOTE: 4表示4个线程, 因为在SM80_16x8x16_F32F16F16F32_TN中,
        // 每组每行就是4个线程处理8个value的, 每个线程处理2个value,
        dst(i) = Allreduce<4>::run(src(i), op);
    }
}

// tensor:((2, MMA_M),(2, MMA_N))
// summary:(2 * MMA_N)
template<bool zero_init=true, typename Engine0, typename Layout0, typename Engine1, typename Layout1, typename Operator>
__device__ inline void reduce_(Tensor<Engine0, Layout0> const& tensor, Tensor<Engine1, Layout1> &summary, Operator &op) {
    // NOTE: 遍历tensor每行, 记录到summary中
    // reduce 当前thread的max
    thread_reduce_<zero_init>(tensor, summary, op);
    // NOTE: 二分法对summary[]进行reduce
    // reduce thread间的max
    quad_allreduce_(summary, summary, op);
}

// scores:((2, MMA_M),(2, MMA_N))
// scores_max:(2 * MMA_N)
template<bool zero_init=true, typename Engine0, typename Layout0, typename Engine1, typename Layout1>
__device__ inline void reduce_max(Tensor<Engine0, Layout0> const& tensor, Tensor<Engine1, Layout1> &max){
    MaxOp<float> max_op;
    reduce_<zero_init>(tensor, max, max_op);
}

template<typename Engine0, typename Layout0, typename Engine1, typename Layout1>
__device__ inline void reduce_sum(Tensor<Engine0, Layout0> const& tensor, Tensor<Engine1, Layout1> &sum){
    SumOp<float> sum_op;
    reduce_(tensor, sum, sum_op);
}
