#ifndef XBLANG_DIALECT_XBLANG_LOWERING_COMMON_H
#define XBLANG_DIALECT_XBLANG_LOWERING_COMMON_H

#include "xblang/Dialect/XBLang/IR/XBLang.h"
#include "xblang/Dialect/XBLang/Transforms/PatternBase.h"
#include "xblang/Dialect/XBLang/Utils/BuilderBase.h"
#include "xblang/Sema/XBLangTypeSystemMixin.h"

namespace xblang {
namespace xb {
struct LoweringBuilderBase : public BuilderBase {
  /// Lowers a value from a ptr | ref to a memref.
  static Value lowerValue(Builder &builder, Value value);

  /// Returns a lowered value to memref, asserts and returns null on failure.
  static mlir::TypedValue<MemRefType> toMemref(Builder &builder, Value value);

  static std::optional<Value>
  nativeCast(Builder &builder, Type targetType, Type sourceType, Value source,
             const TypeConverter *typeConverter = nullptr);

  static std::optional<Value>
  nativeCast(Builder &builder, Type targetType, Value source,
             const TypeConverter *typeConverter = nullptr) {
    return nativeCast(builder, targetType, source.getType(), source,
                      typeConverter);
  }

  static Value trivialLoad(Builder &builder, Value memRef);

  static bool trivialStore(Builder &builder, Value memRef, Value value);
};
} // namespace xb
} // namespace xblang

#endif
