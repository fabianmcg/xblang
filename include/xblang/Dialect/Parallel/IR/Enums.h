#ifndef XBLANG_DIALECT_PARALLEL_IR_ENUMS_H
#define XBLANG_DIALECT_PARALLEL_IR_ENUMS_H

#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/StringRef.h"
#include <optional>

#include "xblang/Dialect/Parallel/IR/ParallelEnums.h.inc"

namespace mlir {
namespace par {
inline constexpr ParallelHierarchy operator|(ParallelHierarchy lhs,
                                             ParallelHierarchy rhs) {
  return static_cast<ParallelHierarchy>(static_cast<uint32_t>(lhs) |
                                        static_cast<uint32_t>(rhs));
}

inline constexpr ParallelHierarchy operator&(ParallelHierarchy lhs,
                                             ParallelHierarchy rhs) {
  return static_cast<ParallelHierarchy>(static_cast<uint32_t>(lhs) &
                                        static_cast<uint32_t>(rhs));
}
} // namespace par
} // namespace mlir

namespace xblang {
using mlir::par::AtomicOps;
using mlir::par::DataSharingKind;
using mlir::par::MapKind;
using mlir::par::ParallelHierarchy;
using mlir::par::ReduceOps;
using mlir::par::operator&;
using mlir::par::operator|;
} // namespace xblang

#endif
