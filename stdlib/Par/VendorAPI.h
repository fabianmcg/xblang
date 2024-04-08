#ifndef XBLIB_PAR_VENDORAPI_H
#define XBLIB_PAR_VENDORAPI_H

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <utility>

#include "Runtime.h"

#if defined(PAR_CUDA_BACKEND)
#include "cuda_runtime.h"

#define REPORT_IF_ERROR(expr)                                                  \
  [](cudaError_t result) {                                                     \
    if (!result)                                                               \
      return;                                                                  \
    const char *name = cudaGetErrorName(result);                               \
    if (!name)                                                                 \
      name = "<unknown>";                                                      \
    fprintf(stderr, "'%s' failed with '%s'\n", #expr, name);                   \
  }(expr)

#elif defined(PAR_HIP_BACKEND)
#include "hip/hip_runtime.h"

#define REPORT_IF_ERROR(expr)                                                  \
  [](hipError_t result) {                                                      \
    if (!result)                                                               \
      return;                                                                  \
    const char *name = hipGetErrorName(result);                                \
    if (!name)                                                                 \
      name = "<unknown>";                                                      \
    fprintf(stderr, "'%s' failed with '%s'\n", #expr, name);                   \
  }(expr)
#endif

#define PAR_RT_ASSERT_MESSAGE                                                  \
  assert("Function called without valid back-end." && false)

namespace xb {
namespace par {
namespace rt {
using kernel_t = void *;
#if defined(PAR_CUDA_BACKEND)
using stream_t = cudaStream_t;
using event_t = cudaEvent_t;
#elif defined(PAR_HIP_BACKEND)
using stream_t = hipStream_t;
using event_t = hipEvent_t;
#else
using stream_t = void *;
using event_t = void *;
#endif

inline void launchKernel(kernel_t function, intptr_t gridX, intptr_t gridY,
                         intptr_t gridZ, intptr_t blockX, intptr_t blockY,
                         intptr_t blockZ, int32_t smem, stream_t stream,
                         void **params, void **extra) {
#if defined(PAR_CUDA_BACKEND)
  REPORT_IF_ERROR(cudaLaunchKernel(function, dim3(gridX, gridY, gridZ),
                                   dim3(blockX, blockY, blockZ), params, smem,
                                   stream));
#elif defined(PAR_HIP_BACKEND)
  REPORT_IF_ERROR(hipLaunchKernel(function, dim3(gridX, gridY, gridZ),
                                  dim3(blockX, blockY, blockZ), params, smem,
                                  stream));
#else
  PAR_RT_ASSERT_MESSAGE;
#endif
}

inline stream_t streamCreate() {
  stream_t stream{};
#if defined(PAR_CUDA_BACKEND)
  REPORT_IF_ERROR(cudaStreamCreate(&stream));
#elif defined(PAR_HIP_BACKEND)
  HIP_REPORT_IF_ERROR(hipStreamCreate(&stream));
#else
  PAR_RT_ASSERT_MESSAGE;
#endif
  return stream;
}

inline void streamDestroy(stream_t stream) {
#if defined(PAR_CUDA_BACKEND)
  REPORT_IF_ERROR(cudaStreamDestroy(stream));
#elif defined(PAR_HIP_BACKEND)
  REPORT_IF_ERROR(hipStreamDestroy(stream));
#else
  PAR_RT_ASSERT_MESSAGE;
#endif
}

inline void streamSynchronize(stream_t stream) {
#if defined(PAR_CUDA_BACKEND)
  if (!stream)
    REPORT_IF_ERROR(cudaDeviceSynchronize());
  REPORT_IF_ERROR(cudaStreamSynchronize(stream));

#elif defined(PAR_HIP_BACKEND)
  if (!stream)
    REPORT_IF_ERROR(hipDeviceSynchronize());
  REPORT_IF_ERROR(hipStreamSynchronize(stream));
#else
  PAR_RT_ASSERT_MESSAGE;
#endif
}

inline void streamWaitEvent(stream_t stream, event_t event) {
#if defined(PAR_CUDA_BACKEND)
  REPORT_IF_ERROR(cudaStreamWaitEvent(stream, event, /*flags=*/0));
#elif defined(PAR_HIP_BACKEND)
  REPORT_IF_ERROR(hipStreamWaitEvent(stream, event, /*flags=*/0));
#else
  PAR_RT_ASSERT_MESSAGE;
#endif
}

inline event_t eventCreate() {
  event_t event{};
#if defined(PAR_CUDA_BACKEND)
  REPORT_IF_ERROR(cudaEventCreate(&event));
#elif defined(PAR_HIP_BACKEND)
  REPORT_IF_ERROR(hipEventCreateWithFlags(&event, hipEventDisableTiming));
#else
  PAR_RT_ASSERT_MESSAGE;
#endif
  return event;
}

inline void eventDestroy(event_t event) {
#if defined(PAR_CUDA_BACKEND)
  REPORT_IF_ERROR(cudaEventDestroy(event));
#elif defined(PAR_HIP_BACKEND)
  REPORT_IF_ERROR(hipEventDestroy(event));
#else
  PAR_RT_ASSERT_MESSAGE;
#endif
}

inline void eventSynchronize(event_t event) {
#if defined(PAR_CUDA_BACKEND)
  REPORT_IF_ERROR(cudaEventSynchronize(event));
#elif defined(PAR_HIP_BACKEND)
  REPORT_IF_ERROR(hipEventSynchronize(event));
#else
  PAR_RT_ASSERT_MESSAGE;
#endif
}

inline void eventRecord(event_t event, stream_t stream) {
#if defined(PAR_CUDA_BACKEND)
  REPORT_IF_ERROR(cudaEventRecord(event, stream));
#elif defined(PAR_HIP_BACKEND)
  REPORT_IF_ERROR(hipEventRecord(event, stream));
#else
  PAR_RT_ASSERT_MESSAGE;
#endif
}

inline Address devAlloc(uint64_t sizeBytes) {
  Address ptr{};
#if defined(PAR_CUDA_BACKEND)
  REPORT_IF_ERROR(cudaMalloc((void **)&ptr, sizeBytes));
#elif defined(PAR_HIP_BACKEND)
  REPORT_IF_ERROR(hipMalloc((void **)&ptr, sizeBytes));
#else
  PAR_RT_ASSERT_MESSAGE;
#endif
  return ptr;
}

inline void devFree(void *ptr) {
#if defined(PAR_CUDA_BACKEND)
  REPORT_IF_ERROR(cudaFree(ptr));
#elif defined(PAR_HIP_BACKEND)
  REPORT_IF_ERROR(hipFree(ptr));
#else
  PAR_RT_ASSERT_MESSAGE;
#endif
}

inline Address hostAlloc(uint64_t sizeBytes, uint32_t flags = 0) {
  Address ptr{};
#if defined(PAR_CUDA_BACKEND)
  REPORT_IF_ERROR(cudaHostAlloc((void **)&ptr, sizeBytes, flags));
#elif defined(PAR_HIP_BACKEND)
  REPORT_IF_ERROR(hipHostMalloc((void **)&ptr, sizeBytes, flags));
#else
  PAR_RT_ASSERT_MESSAGE;
#endif
  return ptr;
}

inline void hostFree(void *ptr) {
#if defined(PAR_CUDA_BACKEND)
  REPORT_IF_ERROR(cudaFreeHost(ptr));
#elif defined(PAR_HIP_BACKEND)
  REPORT_IF_ERROR(hipHostFree(ptr));
#else
  PAR_RT_ASSERT_MESSAGE;
#endif
}

inline void memCpy(void *dst, void *src, size_t sizeBytes, stream_t stream) {
#if defined(PAR_CUDA_BACKEND)
  REPORT_IF_ERROR(
      cudaMemcpyAsync(dst, src, sizeBytes, cudaMemcpyDefault, stream));
#elif defined(PAR_HIP_BACKEND)
  REPORT_IF_ERROR(
      hipMemcpyAsync(dst, src, sizeBytes, hipMemcpyDefault, stream));
#else
  PAR_RT_ASSERT_MESSAGE;
#endif
}

inline Address hostRegister(void *ptr, uint64_t sizeBytes, uint32_t flags = 0) {
#if defined(PAR_CUDA_BACKEND)
  REPORT_IF_ERROR(cudaHostRegister(ptr, sizeBytes, /*flags=*/flags));
#elif defined(PAR_HIP_BACKEND)
  REPORT_IF_ERROR(hipHostRegister(ptr, sizeBytes, /*flags=*/flags));
#else
  PAR_RT_ASSERT_MESSAGE;
#endif
  return static_cast<Address>(ptr);
}

inline void hostUnregister(void *ptr) {
#if defined(PAR_CUDA_BACKEND)
  REPORT_IF_ERROR(cudaHostUnregister(ptr));
#elif defined(PAR_HIP_BACKEND)
  REPORT_IF_ERROR(hipHostUnregister(ptr));
#else
  PAR_RT_ASSERT_MESSAGE;
#endif
}

inline int32_t getDeviceCount() {
  int devices = 0;
#if defined(PAR_CUDA_BACKEND)
  REPORT_IF_ERROR(cudaGetDeviceCount(&devices));
#elif defined(PAR_HIP_BACKEND)
  REPORT_IF_ERROR(hipGetDeviceCount(&devices));
#else
  PAR_RT_ASSERT_MESSAGE;
#endif
  return devices;
}

inline int32_t getDevice() {
  int device = 0;
#if defined(PAR_CUDA_BACKEND)
  REPORT_IF_ERROR(cudaGetDevice(&device));
#elif defined(PAR_HIP_BACKEND)
  REPORT_IF_ERROR(hipGetDevice(&device));
#else
  PAR_RT_ASSERT_MESSAGE;
#endif
  return device;
}

inline void setDevice(int32_t device) {
#if defined(PAR_CUDA_BACKEND)
  REPORT_IF_ERROR(cudaSetDevice(device));
#elif defined(PAR_HIP_BACKEND)
  REPORT_IF_ERROR(hipSetDevice(device));
#else
  PAR_RT_ASSERT_MESSAGE;
#endif
}

inline int maxActiveBlocks(const kernel_t func, int bsz, size_t smem = 0) {
  int blocks = 0;
#if defined(PAR_CUDA_BACKEND)
  REPORT_IF_ERROR(
      cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, func, bsz, smem));
#elif defined(PAR_HIP_BACKEND)
  REPORT_IF_ERROR(
      hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, func, bsz, smem));
#else
  PAR_RT_ASSERT_MESSAGE;
#endif
  return blocks;
}
} // namespace rt
} // namespace par
} // namespace xb

#undef PAR_RT_ASSERT_MESSAGE
#undef REPORT_IF_ERROR
#endif /* XBLIB_PAR_VENDORAPI_H */
