#ifndef XBLIB_PAR_MEMUTIL_H
#define XBLIB_PAR_MEMUTIL_H

#include <cstring>
#include <memory>

#include "Resource.h"
#include "VendorAPI.h"

namespace xb {
namespace par {
namespace rt {
inline Address getAligned(void *addr, size_t sz, size_t alignment) {
  return reinterpret_cast<Address>(std::align(alignment, sz, addr, sz));
}

class UniqueAddress {
public:
  enum class DevKind : int32_t {
    None,
    Device,
  };
  enum class HostKind : int32_t {
    None,
    Host,
    Registered,
  };
  UniqueAddress() = default;

  ~UniqueAddress() { release(); }

  inline UniqueAddress(UniqueAddress &&);
  inline UniqueAddress &operator=(UniqueAddress &&);

  UniqueAddress(const UniqueAddress &) = delete;
  UniqueAddress &operator=(const UniqueAddress &) = delete;

  operator Resource() const { return getResource(0); }

  static inline UniqueAddress makeDev(size_t sz, Address host = nullptr);
  static inline UniqueAddress makeHost(size_t sz, uint32_t flags = 0,
                                       Address dev = nullptr);
  static inline UniqueAddress makeRegistered(Address host, size_t sz = 0,
                                             uint32_t flags = 0);
  static inline UniqueAddress makeDevHost(size_t sz, uint32_t flags = 0);
  static inline UniqueAddress makeDevRegistered(size_t sz, Address host,
                                                uint32_t flags = 0);

  void inline release();

  Resource getResource(int64_t flags) const {
    return Resource(devPtr, size, hostPtr, flags);
  }

  size_t getSize() const { return size; }

  Address getDev() const { return devPtr; }

  Address getHost() const { return hostPtr; }

private:
  inline UniqueAddress(Address dev, Address host, size_t sz, DevKind dk,
                       HostKind hk);
  void inline releaseDev();
  void inline releaseHost();

  Address devPtr{};
  Address hostPtr{};
  size_t size{};
  DevKind devKind{};
  HostKind hostKind{};
};

UniqueAddress::UniqueAddress(Address dev, Address host, size_t sz, DevKind dk,
                             HostKind hk)
    : devPtr(dev), hostPtr(host), size(sz), devKind(dk), hostKind(hk) {}

UniqueAddress::UniqueAddress(UniqueAddress &&other)
    : devPtr(std::exchange(other.devPtr, nullptr)),
      hostPtr(std::exchange(other.hostPtr, nullptr)),
      size(std::exchange(other.size, 0)),
      devKind(std::exchange(other.devKind, DevKind::None)),
      hostKind(std::exchange(other.hostKind, HostKind::None)) {}

UniqueAddress &UniqueAddress::operator=(UniqueAddress &&other) {
  release();
  devPtr = std::exchange(other.devPtr, nullptr);
  hostPtr = std::exchange(other.hostPtr, nullptr);
  size = std::exchange(other.size, 0);
  devKind = std::exchange(other.devKind, DevKind::None);
  hostKind = std::exchange(other.hostKind, HostKind::None);
  return *this;
}

void UniqueAddress::release() {
  releaseDev();
  releaseHost();
  size = 0;
}

void UniqueAddress::releaseDev() {
  if (!devPtr)
    return;
  if (devKind == DevKind::Device)
    devFree(devPtr);
  devPtr = nullptr;
}

void UniqueAddress::releaseHost() {
  if (!hostPtr)
    return;
  if (hostKind == HostKind::Host)
    hostFree(hostPtr);
  else if (hostKind == HostKind::Registered)
    hostUnregister(hostPtr);
  hostPtr = nullptr;
}

UniqueAddress UniqueAddress::makeDev(size_t sz, Address host) {
  if (sz == 0)
    return {};
  return UniqueAddress(devAlloc(sz), host, sz, DevKind::Device, HostKind::None);
}

UniqueAddress UniqueAddress::makeHost(size_t sz, uint32_t flags, Address dev) {
  if (sz == 0)
    return {};
  return UniqueAddress(dev, hostAlloc(sz, flags), sz, DevKind::None,
                       HostKind::Host);
}

UniqueAddress UniqueAddress::makeRegistered(Address host, size_t sz,
                                            uint32_t flags) {
  if (!host)
    return {};
  return UniqueAddress(nullptr, hostRegister(host, flags), sz, DevKind::None,
                       HostKind::Registered);
}

UniqueAddress UniqueAddress::makeDevHost(size_t sz, uint32_t flags) {
  if (sz == 0)
    return {};
  return UniqueAddress(devAlloc(sz), hostAlloc(sz, flags), sz, DevKind::Device,
                       HostKind::Host);
}

UniqueAddress UniqueAddress::makeDevRegistered(size_t sz, Address host,
                                               uint32_t flags) {
  if (sz == 0 || !host)
    return {};
  return UniqueAddress(devAlloc(sz), hostRegister(host, sz, flags), sz,
                       DevKind::Device, HostKind::Registered);
}
} // namespace rt
} // namespace par
} // namespace xb

#endif /* XBLIB_PAR_MEMUTIL_H_ */
