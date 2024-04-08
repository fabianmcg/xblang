#ifndef XBLIB_PAR_PARRUNTIME_H
#define XBLIB_PAR_PARRUNTIME_H

#include <mutex>
#include <thread>

#include "Cache.h"
#include <iostream>

#ifndef PAR_MAP_HOST
#define PAR_MAP_HOST 0
#endif

namespace xb {
namespace par {
namespace rt {
class ParRuntime {
public:
  static constexpr bool mapHost = PAR_MAP_HOST;
  static constexpr int defaultBlockSize = 128;

private:
  MemoryCache cache;
  std::mutex mutex{};
  Map<Address, Resource> mappings{};
  Map<Address, UniqueAddress> allocations{};
  Map<kernel_t, int> blockSizes;
  int blockSize = defaultBlockSize;
  static bool debug;

public:
  ParRuntime() : cache() { init(); }

  ~ParRuntime() = default;

  void init();

  Resource getMapping(Address source) const {
    auto it = mappings.find(source);
    if (it != mappings.end())
      return it->second.getResource();
    return {};
  }

  Resource allocOrGetMapping(Address source, size_t size, int64_t flags = 0);

  Address getAlloca(Address source, size_t sizeOfType, size_t numElements);

  void restoreAlloca(Address source);

  void removeMapping(Address source, bool destroy = false);

  Address map(uint32_t kind, Address source, size_t sizeOfType,
              size_t numElements, stream_t stream);

  int getBlockSize(kernel_t kernel);

  void launchKernel(kernel_t function, intptr_t gridX, intptr_t gridY,
                    intptr_t gridZ, intptr_t blockX, intptr_t blockY,
                    intptr_t blockZ, int32_t smem, stream_t stream,
                    void **params, void **extra) {
    if (blockX == 0)
      blockX = getBlockSize(function);
    if (gridY == 0) {
      gridX = (gridX + blockX - 1) / blockX;
      gridY = 1;
    }
    if (debug) {
      std::cerr << "(" << gridX << ", " << gridY << ", " << gridZ << ") - "
                << "(" << blockX << ", " << blockY << ", " << blockZ << ")"
                << std::endl;
    }
    xb::par::rt::launchKernel(function, gridX, gridY, gridZ, blockX, blockY,
                              blockZ, smem, stream, params, extra);
  }
};
} // namespace rt
} // namespace par
} // namespace xb

#endif /* XBLIB_PAR_PARRUNTIME_H */
