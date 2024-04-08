#ifndef XBLANG_SEMA_TYPESYSTEMBASE_H
#define XBLANG_SEMA_TYPESYSTEMBASE_H

#include "mlir/IR/BuiltinTypes.h"
#include "xblang/Support/LLVM.h"

namespace xblang {
namespace xb {
struct TypeSystemMixinBase {
  typedef enum {
    UnknownCast = -1,
    InvalidCast = 0,
    SourceToTarget = 1,
    TargetToSource = 2,
    Roundtrip = SourceToTarget | TargetToSource
  } CastValidityKind;
  enum class RankValidity { Unknown = -1, Invalid = 0, Valid = 1 };
};

template <typename Derived>
class TypeSystemMixin : public TypeSystemMixinBase {
private:
  Derived &getDerived() { return static_cast<Derived &>(*this); }

  const Derived &getDerived() const {
    return static_cast<const Derived &>(*this);
  }

public:
  CastValidityKind castKind(Type target, Type source) const {
    assert(target && source && "Both types must be valid.");
    return getDerived().castKindImpl(target, source);
  }

  bool isValidCast(Type target, Type source) const {
    auto kind = castKind(target, source);
    return kind != UnknownCast ? (kind & SourceToTarget) == SourceToTarget
                               : InvalidCast;
  }

  std::pair<RankValidity, Type> rankTypes(Type lhs, Type rhs) const {
    assert(lhs && rhs && "Both types must be valid.");
    return getDerived().rankTypesImpl(lhs, rhs);
  }

  CastValidityKind castSequence(Type target, Type source,
                                SmallVector<Type, 2> &sequence) const {
    assert(target && source && "Both types must be valid.");
    return getDerived().castSequenceImpl(target, source, sequence);
  }

protected:
  CastValidityKind castKindImpl(Type target, Type source) const {
    return UnknownCast;
  }

  std::pair<RankValidity, Type> rankTypesImpl(Type lhs, Type rhs) const {
    return {RankValidity::Unknown, nullptr};
  }

  CastValidityKind castSequenceImpl(Type target, Type source,
                                    SmallVector<Type, 2> &sequence) const {
    return castKindImpl(target, source);
  }
};

template <typename Derived, typename TypeSystem>
class TypeSystemMixinAdaptor : public TypeSystemMixinBase {
private:
  Derived &getDerived() { return static_cast<Derived &>(*this); }

  const Derived &getDerived() const {
    return static_cast<const Derived &>(*this);
  }

  const TypeSystem &typeSystem() const { return getDerived().getTypeSystem(); }

public:
  CastValidityKind castKind(Type target, Type source) const {
    return typeSystem().castKind(target, source);
  }

  bool isValidCast(Type target, Type source) const {
    return typeSystem().isValidCast(target, source);
  }

  bool isValidCast(CastValidityKind kind) const {
    return kind != CastValidityKind::UnknownCast
               ? (kind & SourceToTarget) == SourceToTarget
               : InvalidCast;
  }

  std::pair<RankValidity, Type> rankTypes(Type lhs, Type rhs) const {
    return typeSystem().rankTypes(lhs, rhs);
  }

  CastValidityKind castSequence(Type target, Type source,
                                SmallVector<Type, 2> &sequence) const {
    return typeSystem().castSequence(target, source, sequence);
  }
};
} // namespace xb
} // namespace xblang

#endif
