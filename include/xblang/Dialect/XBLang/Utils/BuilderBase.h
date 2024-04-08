#ifndef XBLANG_DIALECT_XBLANG_UTILS_BUILDERBASE_H
#define XBLANG_DIALECT_XBLANG_UTILS_BUILDERBASE_H

#include "mlir/IR/Builders.h"

namespace xblang {
namespace xb {
struct BuilderBase {
  using Builder = OpBuilder;
  using Guard = Builder::InsertionGuard;

  static Guard guard(Builder &builder) { return Guard(builder); }

  static Guard guard(Builder &builder, Operation *op) {
    Guard guard(builder);
    builder.setInsertionPoint(op);
    return guard;
  }

  static Guard guard(Builder &builder, Block *block, Block::iterator point) {
    Guard guard(builder);
    builder.setInsertionPoint(block, point);
    return guard;
  }

  static Guard guardAfter(Builder &builder, Operation *op) {
    Guard guard(builder);
    builder.setInsertionPointAfter(op);
    return guard;
  }

  static Guard guardAfter(Builder &builder, Value value) {
    Guard guard(builder);
    builder.setInsertionPointAfterValue(value);
    return guard;
  }

  template <typename CastOp, typename... Args>
  static CastOp createCast(Builder &builder, Type targetType, Value source,
                           Args &&...args) {
    return builder.create<CastOp>(source.getLoc(), targetType, source,
                                  std::forward<Args>(args)...);
  }
};
} // namespace xb
} // namespace xblang

#endif
