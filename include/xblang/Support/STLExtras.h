#ifndef XBLANG_SUPPORT_STLEXTRAS_H
#define XBLANG_SUPPORT_STLEXTRAS_H

#include <limits>
#include <utility>

namespace xblang {
template <class T, class... Args>
constexpr T *construct_at(T *p, Args &&...args) {
  return ::new (static_cast<void *>(p)) T(std::forward<Args>(args)...);
}
} // namespace xblang

#endif
