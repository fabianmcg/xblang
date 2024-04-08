#ifndef XBLANG_SUPPORT_FORMAT_H
#define XBLANG_SUPPORT_FORMAT_H

#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/FormatVariadic.h"

namespace xblang {
template <typename... Args>
std::string fmt(const char *str, Args &&...args) {
  return llvm::formatv(str, std::forward<Args>(args)...).str();
}

inline std::string utfToString(uint32_t c) {
  if (c < 128)
    return std::string(1, static_cast<char>(c));
  return fmt("\\u+{0}", llvm::utohexstr(c, false, 0));
}
} // namespace xblang

#endif
