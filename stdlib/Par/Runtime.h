#ifndef XBLIB_PAR_RUNTIME_H
#define XBLIB_PAR_RUNTIME_H

#include <cstdint>
#include <unordered_map>

namespace xb {
namespace par {
namespace rt {
using Address = int8_t *;
template <typename K, typename V>
using Map = std::unordered_map<K, V>;

typedef enum : uint32_t {
  Present = 0,
  To = 1,
  From = 2,
  ToFrom = 3,
  Allocate = 4,
  Destroy = 5,
  Deallocate = 6
} MappingKind;
} // namespace rt
} // namespace par
} // namespace xb

#endif /* XBLIB_PAR_RUNTIME_H */
