#include "xblang/Dialect/XBLang/Lowering/Common.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Index/IR/IndexAttrs.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Transforms/DialectConversion.h"
#include "xblang/ADT/DoubleTypeSwitch.h"

using namespace mlir;
using namespace xblang::xb;

Value LoweringBuilderBase::lowerValue(Builder &builder, Value value) {
  assert(value && "Value cannot be null.");
  auto type = value.getType();
  // If `value` is not a ptr or ref, return the value unchanged.
  if (isa<PointerType>(type) || isa<ReferenceType>(type)) {
    // If `value` is the result of a cast to ptr or ref from a memref, return
    // the original memref.
    if (CastOp castOp = dyn_cast_or_null<CastOp>(value.getDefiningOp()))
      if (isa<MemRefType>(castOp.getValue().getType()))
        return castOp.getValue();
    if (UnrealizedConversionCastOp castOp =
            dyn_cast_or_null<UnrealizedConversionCastOp>(value.getDefiningOp()))
      if (isa<MemRefType>(castOp.getInputs()[0].getType()))
        return castOp.getInputs()[0];

    // Create a cast to memref.
    if (auto ptr = dyn_cast<PointerType>(type)) {
      return builder.create<CastOp>(
          value.getLoc(),
          MemRefType::get({std::numeric_limits<int64_t>::max()},
                          ptr.getPointee(),
                          StridedLayoutAttr::get(builder.getContext(), 0, {1}),
                          ptr.getMemorySpace()),
          value);
    } else if (auto ref = dyn_cast<ReferenceType>(type)) {
      return builder.create<CastOp>(value.getLoc(),
                                    MemRefType::get({}, ref.getPointee(),
                                                    MemRefLayoutAttrInterface{},
                                                    ref.getMemorySpace()),
                                    value);
    }
  }
  return value;
}

TypedValue<MemRefType> LoweringBuilderBase::toMemref(Builder &builder,
                                                     Value value) {
  Value result = lowerValue(builder, value);
  if (isa<MemRefType>(result.getType()))
    return result.getImpl();
  assert(false && "The provided value couldn't be lowered to a memref.");
  return nullptr;
}

std::optional<Value>
LoweringBuilderBase::nativeCast(Builder &builder, Type targetType,
                                Type sourceType, Value source,
                                const TypeConverter *typeConverter) {
  assert(targetType && sourceType && source &&
         "Target type, source type and source value shouldn't be null.");
  auto cast = [&builder, &source, typeConverter](auto op, Type trgt) -> Value {
    using Op = llvm::remove_cvref_t<decltype(op)>;
    if (typeConverter)
      trgt = typeConverter->convertType(trgt);
    return createCast<Op>(builder, trgt, source).getResult();
  };
  std::optional<Value> result;
  using Switch = ::xblang::DoubleTypeSwitch<Type, Type, bool>;
  Switch::Switch(targetType, sourceType)
      .Case<IntegerType, IntegerType>(
          [&](IntegerType target, IntegerType source) {
            if (target.getWidth() > source.getWidth())
              result = source.isSigned() ? cast(arith::ExtSIOp{}, target)
                                         : cast(arith::ExtUIOp{}, target);
            else if (target.getWidth() < source.getWidth())
              result = cast(arith::TruncIOp{}, target);
            else
              result = nullptr;
            return true;
          })
      .Case<FloatType, FloatType>([&](FloatType target, FloatType source) {
        if (target.getWidth() > source.getWidth())
          result = cast(arith::ExtFOp{}, target);
        else if (target.getWidth() < source.getWidth())
          result = cast(arith::TruncFOp{}, target);
        else
          result = nullptr;
        return true;
      })
      .Case<IntegerType, FloatType>([&](IntegerType target, FloatType source) {
        result = target.isSignedInteger() ? cast(arith::FPToSIOp{}, target)
                                          : cast(arith::FPToUIOp{}, target);
        return true;
      })
      .Case<FloatType, IntegerType>([&](FloatType target, IntegerType source) {
        result = source.isSignedInteger() ? cast(arith::SIToFPOp{}, target)
                                          : cast(arith::UIToFPOp{}, target);
        return true;
      })
      .Case<IntegerType, IndexType, true>(
          [&](IntegerType lhs, IndexType rhs, bool reverseCase) {
            Type type = lhs;
            if (reverseCase)
              type = rhs;
            result = lhs.isSignedInteger() ? cast(index::CastSOp{}, type)
                                           : cast(index::CastUOp{}, type);
            return true;
          })
      .Case<PointerType, IndexType, true>(
          [&](Type lhs, Type rhs, bool reverseCase) {
            Type type = reverseCase ? rhs : lhs;
            result = createCast<CastOp>(builder, type, source, true);
            return true;
          })
      .Case<AddressType, IndexType, true>(
          [&](Type lhs, Type rhs, bool reverseCase) {
            Type type = reverseCase ? rhs : lhs;
            result = createCast<CastOp>(builder, type, source, true);
            return true;
          })
      .Case<AddressType, PointerType, true>(
          [&](Type lhs, Type rhs, bool reverseCase) {
            Type type = reverseCase ? rhs : lhs;
            result = createCast<CastOp>(builder, type, source, true);
            return true;
          })
      .DefaultValue(false);
  return result;
}

Value LoweringBuilderBase::trivialLoad(Builder &builder, Value value) {
  assert(value && "Value cannot be null.");
  if (!value)
    return nullptr;
  auto memRef = toMemref(builder, value);
  if (!memRef)
    return nullptr;
  MemRefType memRefTy = memRef.getType();
  if (!memRefTy.hasRank()) {
    assert(value && "The memref must be ranked.");
    return nullptr;
  }
  SmallVector<Value> indixes(memRefTy.getRank());
  if (memRefTy.getRank()) {
    auto zero = builder.create<index::ConstantOp>(value.getLoc(), 0);
    for (auto i : llvm::iota_range<int>(0, memRefTy.getRank(), false))
      indixes[i] = zero;
  }
  return builder.create<memref::LoadOp>(value.getLoc(), memRef, indixes);
}

bool LoweringBuilderBase::trivialStore(Builder &builder, Value mem,
                                       Value value) {
  assert(value && "Value cannot be null.");
  if (!value)
    return false;
  auto memRef = toMemref(builder, mem);
  if (!memRef)
    return false;
  MemRefType memRefTy = memRef.getType();
  if (!memRefTy.hasRank()) {
    assert(value && "The memref must be ranked.");
    return false;
  }
  SmallVector<Value> indixes(memRefTy.getRank());
  if (memRefTy.getRank()) {
    auto zero = builder.create<index::ConstantOp>(value.getLoc(), 0);
    for (auto i : llvm::iota_range<int>(0, memRefTy.getRank(), false))
      indixes[i] = zero;
  }
  builder.create<memref::StoreOp>(memRef.getLoc(), value, memRef, indixes);
  return true;
}
