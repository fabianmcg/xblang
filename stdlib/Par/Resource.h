#ifndef XBLIB_PAR_RESOURCE_H
#define XBLIB_PAR_RESOURCE_H

#include <cstddef>
#include <cstdint>
#include <utility>

#include "Runtime.h"

namespace xb {
namespace par {
namespace rt {
class Resource {
protected:
  Address address{};
  Address hostAddress{};
  size_t size{};
  int64_t flags{};

public:
  typedef enum : int64_t { StackAlloca = 1, CachePage = 2 } Flags;

  Resource() = default;

  Resource(Address address, size_t size, Address hostAddress = nullptr,
           int64_t flags = 0)
      : address(address), hostAddress(hostAddress), size(size), flags(flags) {}

  Resource(Resource &&other) {
    address = std::exchange(other.address, nullptr);
    hostAddress = std::exchange(other.hostAddress, nullptr);
    size = std::exchange(other.size, 0);
    flags = std::exchange(other.flags, 0);
  }

  Resource(const Resource &other) {
    address = other.address;
    hostAddress = other.hostAddress;
    size = other.size;
    flags = other.flags;
  }

  ~Resource() { invalidate(); }

  Resource &operator=(Resource &&other) {
    address = std::exchange(other.address, nullptr);
    hostAddress = std::exchange(other.hostAddress, nullptr);
    size = std::exchange(other.size, 0);
    flags = std::exchange(other.flags, 0);
    return *this;
  }

  Resource &operator=(const Resource &other) {
    address = other.address;
    hostAddress = other.hostAddress;
    size = other.size;
    flags = other.flags;
    return *this;
  }

  operator bool() const { return isValid(); }

  bool isValid() const { return address && size; }

  Resource getResource() const { return *this; }

  Address getAddress() const { return address; }

  Address getHostAddress() const { return hostAddress; }

  size_t getSize() const { return size; }

  int64_t getFlags() const { return flags; }

  void setFlags(int64_t f) { flags = f; }

  void addFlags(int64_t f) { flags |= f; }

  bool hasFlag(int64_t flags) const { return (this->flags & flags) == flags; }

  bool hasNoFlag(int64_t flags) const { return (this->flags & flags) != flags; }

  void invalidate() {
    address = nullptr;
    hostAddress = nullptr;
    size = 0;
  }
};
} // namespace rt
} // namespace par
} // namespace xb

#endif /* XBLIB_PAR_RESOURCE_H */
