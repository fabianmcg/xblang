#include <cstdlib>

#include "ParRuntime.h"
#include <cstdlib>

using namespace xb::par::rt;

namespace {
template <typename T>
class ManagedStatic {
public:
  ManagedStatic() = default;

  T &operator*() {
    maybeInit();
    return *data;
  }

  void release() {
    if (data)
      data.release();
  }

private:
  void maybeInit() {
    if (!data)
      data = std::unique_ptr<T>(new T());
  }

  std::unique_ptr<T> data{};
};

ManagedStatic<xb::par::rt::ParRuntime> runtime;

void destroyRuntime() { runtime.release(); }

int runtimeReleaser = atexit(destroyRuntime);
} // namespace

extern "C" {
Address __xblangMapData(uint32_t kind, Address source, size_t sizeOfType,
                        size_t numElements, stream_t stream) {
  return (*runtime).map(kind, source, sizeOfType, numElements, stream);
}

Address __xblangAlloca(Address source, size_t sizeOfType, size_t numElements) {
  return (*runtime).getAlloca(source, sizeOfType, numElements);
}

void __xblangDealloca(Address source, Address stackSource) {
  if (source == stackSource) {
    __xblangMapData(Deallocate, source, 0, 0, nullptr);
    return;
  }
  (*runtime).restoreAlloca(stackSource);
}

void __xblangGpuWait(stream_t stream, bool destroy) {
  streamSynchronize(stream);
  if (stream && destroy)
    streamDestroy(stream);
}

uint64_t __xblangGetMatrixDim(kernel_t kernel) {
  return (*runtime).getBlockSize(kernel);
}

void __xblangLaunch(kernel_t function, intptr_t gridX, intptr_t gridY,
                    intptr_t gridZ, intptr_t blockX, intptr_t blockY,
                    intptr_t blockZ, int32_t smem, stream_t stream,
                    void **params, void **extra) {
  (*runtime).launchKernel(function, gridX, gridY, gridZ, blockX, blockY, blockZ,
                          smem, stream, params, extra);
}

stream_t mgpuStreamCreate() { return streamCreate(); }
}
