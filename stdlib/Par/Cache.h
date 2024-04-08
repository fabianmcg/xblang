#ifndef XBLIB_PAR_CACHE_H
#define XBLIB_PAR_CACHE_H

#include "MemUtil.h"

namespace xb {
namespace par {
namespace rt {
class MemoryCache {
public:
  static constexpr size_t localCachePageSize = 256; // Size in bytes
  static constexpr size_t localCacheDefaultNumPages = 4 * 4 * 1024;
  static constexpr size_t localCacheDefaultSize =
      localCacheDefaultNumPages * localCachePageSize;

  MemoryCache(uint32_t flags = 0) { init(flags); }

  inline Resource getPage(int64_t flags = 0);

  inline void restorePage(Resource &&resource);

  bool hasPages() const { return freePages.size(); }

protected:
  inline void init(uint32_t flags);

private:
  Map<Address, Resource> freePages{};
  UniqueAddress memory;
};

void MemoryCache::init(uint32_t flags) {
  size_t cacheSize = localCacheDefaultSize + localCachePageSize;
  memory = UniqueAddress::makeDevHost(cacheSize, flags);
  Address devMem = memory.getDev(), hostMem = memory.getHost();

  devMem = memory.getDev();
  assert(devMem && "couldn't allocate device memory");
  devMem = getAligned(devMem, localCacheDefaultSize, localCachePageSize);
  assert(devMem && "device memory alignment failed");

  hostMem = memory.getHost();
  assert(hostMem && "couldn't allocate host memory");
  hostMem = getAligned(hostMem, localCacheDefaultSize, localCachePageSize);
  assert(hostMem && "host memory alignment failed");

  Address end = devMem + memory.getSize();
  while (devMem < end) {
    freePages.insert({devMem, Resource(devMem, localCachePageSize, hostMem,
                                       Resource::CachePage)});
    devMem += localCachePageSize;
    hostMem += localCachePageSize;
  }
}

Resource MemoryCache::getPage(int64_t flags) {
  if (freePages.empty())
    return {};
  auto it = freePages.begin();
  Resource mapping = std::move(it->second);
  mapping.addFlags(flags);
  freePages.erase(it);
  return mapping;
}

void MemoryCache::restorePage(Resource &&resource) {
  if (resource.hasFlag(Resource::CachePage))
    freePages.insert({resource.getAddress(), std::move(resource)});
  else
    assert(false);
}
} // namespace rt
} // namespace par
} // namespace xb

#endif
