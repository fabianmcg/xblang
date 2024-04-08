#include "ParRuntime.h"
#include <iostream>

using namespace xb::par::rt;

namespace {
bool getDebugFlag() {
  if (auto val = getenv("PAR_DEBUG")) {
    char *end = val;
    return strtol(val, &end, 10) == 1;
  }
  return false;
}
} // namespace

bool ParRuntime::debug = getDebugFlag();

void ParRuntime::init() {
  if (auto val = getenv("PAR_MSZ")) {
    char *ptr{};
    int bsz = strtol(val, &ptr, 10);
    if (bsz > 0 && bsz <= 1024)
      blockSize = bsz;
  }
}

Resource ParRuntime::allocOrGetMapping(Address source, size_t size,
                                       int64_t flags) {
  if (auto mapping = getMapping(source)) {
    if (mapping.getSize() >= size)
      return mapping;
    else
      removeMapping(source, true);
  }
  if (size == 0)
    return {};
  Resource mapping{};
  if (size < cache.localCachePageSize && cache.hasPages())
    mapping = cache.getPage(flags);
  else {
    auto addr = mapHost ? UniqueAddress::makeDevRegistered(size, source)
                        : UniqueAddress::makeDev(size, source);
    mapping = addr.getResource(flags);
    allocations[source] = std::move(addr);
  }
  if (mapping)
    mappings[source] = mapping;
  return mapping;
}

void ParRuntime::removeMapping(Address source, bool destroy) {
  auto it = mappings.find(source);
  if (it == mappings.end())
    return;
  if (it->second.hasFlag(Resource::CachePage)) {
    cache.restorePage(std::move(it->second));
    mappings.erase(it);
    return;
  }
  if (destroy)
    if (auto it = allocations.find(source); it != allocations.end())
      allocations.erase(it);
  mappings.erase(it);
}

Address ParRuntime::getAlloca(Address source, size_t sizeOfType,
                              size_t numElements) {
  if (sizeOfType * numElements <= MemoryCache::localCachePageSize) {
    std::lock_guard<std::mutex> guard(mutex);
    auto mapping = cache.getPage(Resource::StackAlloca);
    Address allocaAddr{};
    if (mapping) {
      allocaAddr = mapping.getHostAddress();
      mappings[allocaAddr] = std::move(mapping);
      return allocaAddr;
    }
  }
  map(Allocate, source, sizeOfType, numElements, nullptr);
  return source;
}

void ParRuntime::restoreAlloca(Address source) {
  std::lock_guard<std::mutex> guard(mutex);
  removeMapping(source, true);
}

Address ParRuntime::map(uint32_t kind, Address source, size_t sizeOfType,
                        size_t numElements, stream_t stream) {
  size_t sizeInBytes = sizeOfType * numElements;
  if (!source)
    return nullptr;
  if (kind == Present) {
    std::lock_guard<std::mutex> guard(mutex);
    Resource mapping = getMapping(source);
    if (!mapping)
      std::cerr << "Mapping: " << mapping << " isn't present." << std::endl;
    assert(mapping && "Mapping isn't present.");
    if (mapping.getSize() < sizeInBytes) {
      assert(false && "Mapped memory has a smaller size than expected.");
      return nullptr;
    }
    return mapping.getAddress();
  }
  if (kind == Destroy || kind == Deallocate) {
    std::lock_guard<std::mutex> guard(mutex);
    removeMapping(source, kind == Destroy);
    return nullptr;
  }
  Resource mapping{};
  {
    std::lock_guard<std::mutex> guard(mutex);
    mapping = allocOrGetMapping(source, sizeInBytes);
  }
  if (sizeInBytes == 0 && !mapping)
    std::cerr << "Mapping isn't present and the size parameter is 0"
              << std::endl;
  assert(mapping);
  if (sizeInBytes == 0)
    sizeInBytes = mapping.getSize();
  assert(sizeInBytes);

  if (kind == To) {
    auto hostPtr = mapping.getHostAddress();
    if (hostPtr != source)
      memcpy(hostPtr, source, sizeInBytes);
    memCpy(mapping.getAddress(), hostPtr, sizeInBytes, stream);
  } else if (kind == From) {
    auto hostPtr = mapping.getHostAddress();
    assert(hostPtr && "invalid mapped host ptr");
    memCpy(hostPtr, mapping.getAddress(), sizeInBytes, stream);
    if (hostPtr != source) {
      streamSynchronize(stream);
      memcpy(source, hostPtr, sizeInBytes);
    }
  }
  return mapping.getAddress();
}

int ParRuntime::getBlockSize(kernel_t kernel) {
  std::lock_guard<std::mutex> guard(mutex);
  auto &sz = blockSizes[kernel];
  if (sz == 0) {
    if (kernel) {
      int maxBsz = 32;
      int maxActive = maxActiveBlocks(kernel, maxBsz) * maxBsz;
      for (int bsz = 64; bsz <= 512; bsz <<= 1) {
        int tmp = maxActiveBlocks(kernel, bsz) * bsz;
        if (tmp >= maxActive) {
          maxActive = tmp;
          maxBsz = bsz;
          continue;
        }
        break;
      }
      sz = maxBsz;
    } else
      sz = blockSize;
  }
  return sz;
}
