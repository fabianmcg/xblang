#ifndef XBLANG_SEMA_XBLANGTYPESYSTEMMIXIN_H
#define XBLANG_SEMA_XBLANGTYPESYSTEMMIXIN_H

#include "mlir/IR/BuiltinTypes.h"
#include "xblang/Dialect/XBLang/IR/Type.h"
#include "xblang/Support/LLVM.h"

namespace xblang {
namespace xb {
class TypeSystemBase {
public:
  static bool isAlgebraic(Type type) {
    assert(type && "Type must be non null.");
    return isScalar(type) || type.hasTrait<AlgebraicTypeTrait>();
  }

  static bool isArithmetic(Type type) {
    assert(type && "Type must be non null.");
    return type.isIntOrIndexOrFloat() || type.hasTrait<ArithmeticTypeTrait>();
  }

  static inline bool isCompound(Type type) { return !isFundamental(type); }

  static bool isFundamental(Type type) {
    assert(type && "Type must be non null.");
    return type.isIntOrIndexOrFloat() || type.hasTrait<FundamentalTypeTrait>();
  }

  static bool isScalar(Type type) {
    assert(type && "Type must be non null.");
    return isArithmetic(type) || type.hasTrait<ScalarTypeTrait>();
  }

  static bool isAddress(Type type) { return isa<AddressType>(type); }

  static bool isBool(Type type) {
    return isa<IntegerType>(type) && type.getIntOrFloatBitWidth() == 1;
  }

  static bool isFloat(Type type) { return isa<FloatType>(type); }

  static bool isIndex(Type type) { return type.isIndex(); }

  static bool isInt(Type type) {
    return isa<IntegerType>(type) && type.getIntOrFloatBitWidth() > 1;
  }

  static bool isLiteralStruct(Type type) { return isa<StructType>(type); }

  static bool isMemRef(Type type) { return isa<MemRefType>(type); }

  static bool isNamed(Type type) { return isa<NamedType>(type); }

  static bool isNamedStruct(Type type) {
    auto named = dyn_cast<NamedType>(type);
    return named && named.getType() && isLiteralStruct(named.getType());
  }

  static bool isPtr(Type type) { return isa<PointerType>(type); }

  static bool isRange(Type type) { return isa<RangeType>(type); }

  static bool isRef(Type type) { return isa<ReferenceType>(type); }

  static bool isStruct(Type type) {
    return isLiteralStruct(type) || isNamedStruct(type);
  }

  static bool isTensor(Type type) { return isa<TensorType>(type); }

  static bool isAddressLike(Type type) {
    return isPtr(type) || isAddress(type) || type.isIndex();
  }

  static bool isLoadCast(Type target, Type source) {
    assert(target && source && "Both types must be valid.");
    if (auto refType = dyn_cast<ReferenceType>(source))
      return refType.getPointee() == target;
    return false;
  }

  static Type removeReference(Type type) {
    if (!type)
      return nullptr;
    if (auto refType = dyn_cast<ReferenceType>(type))
      return refType.getPointee();
    return type;
  }

  static Type removePtr(Type type) {
    if (!type)
      return nullptr;
    if (auto refType = dyn_cast<PointerType>(type))
      return refType.getPointee();
    return type;
  }

  static Type removeName(Type type) {
    if (!type)
      return nullptr;
    if (auto namedType = dyn_cast<NamedType>(type))
      return namedType.getType();
    return type;
  }
};

template <typename Derived>
class XBLangTypeSystemMixin : public TypeSystemBase {
private:
  Derived &getDerived() { return static_cast<Derived &>(*this); }

  const Derived &getDerived() const {
    return static_cast<const Derived &>(*this);
  }

  MLIRContext *context(MLIRContext *context) const { return context; }

  MLIRContext *context(MLIRContext &context) const { return &context; }

  MLIRContext *context() const { return context(getDerived().getContext()); }

public:
  using SignednessSemantics = IntegerType::SignednessSemantics;

  IntegerType
  Int(int width,
      SignednessSemantics semantics = SignednessSemantics::Signless) const {
    return IntegerType::get(context(), width, semantics);
  }

  IntegerType Bool() const { return Int(1, SignednessSemantics::Signless); }

  IntegerType
  I1(SignednessSemantics semantics = SignednessSemantics::Signless) const {
    return Int(1, semantics);
  }

  IntegerType
  I8(SignednessSemantics semantics = SignednessSemantics::Signless) const {
    return Int(8, semantics);
  }

  IntegerType
  I16(SignednessSemantics semantics = SignednessSemantics::Signless) const {
    return Int(16, semantics);
  }

  IntegerType
  I32(SignednessSemantics semantics = SignednessSemantics::Signless) const {
    return Int(32, semantics);
  }

  IntegerType
  I64(SignednessSemantics semantics = SignednessSemantics::Signless) const {
    return Int(64, semantics);
  }

  FloatType Float(int width) const {
    switch (width) {
    case 32:
      return FloatType::getF32(context());
    case 64:
      return FloatType::getF64(context());
    default:
      assert(false && "Invalid floating point width.");
      return nullptr;
    }
  }

  FloatType F32() const { return Float(32); }

  FloatType F64() const { return Float(64); }

  ComplexType Complex(Type base) const {
    assert(base && "The element type must be valid.");
    return ComplexType::get(base);
  }

  AddressType Address() const { return AddressType::get(context()); }

  PointerType Ptr(Type base, Attribute memorySpace = {}) const {
    assert(base && "The element type must be valid.");
    return PointerType::get(base, memorySpace);
  }

  ReferenceType Ref(Type base, Attribute memorySpace = {}) const {
    assert(base && "The element type must be valid.");
    if (ReferenceType type = dyn_cast<ReferenceType>(base))
      return type;
    return ReferenceType::get(base, memorySpace);
  }

  RangeType Range(Type base) const {
    assert(base && "The element type must be valid.");
    return RangeType::get(context(), base);
  }

  IndexType Index() const { return IndexType::get(context()); }

  template <typename... Args>
  MemRefType MemRef(Args &&...args) const {
    return MemRefType::get(std::forward<Args>(args)...);
  }

  template <typename... Args>
  RankedTensorType Tensor(Args &&...args) const {
    return RankedTensorType::get(std::forward<Args>(args)...);
  }
};
} // namespace xb
} // namespace xblang

#endif
