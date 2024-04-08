#ifndef XBLANG_DIALECT_XBLANG_CONCRETIZATION_COMMON_H
#define XBLANG_DIALECT_XBLANG_CONCRETIZATION_COMMON_H

#include "xblang/Dialect/XBLang/IR/XBLang.h"
#include "xblang/Dialect/XBLang/Transforms/PatternBase.h"
#include "xblang/Dialect/XBLang/Utils/BuilderBase.h"
#include "xblang/Sema/TypeSystem.h"
#include "xblang/Sema/XBLangTypeSystemMixin.h"

namespace xblang {
namespace xb {
template <typename Derived>
class ConcretizationBase : public BuilderBase, public TypeSystemMixinBase {
private:
  Derived &getDerived() { return static_cast<Derived &>(*this); }

  const Derived &getDerived() const {
    return static_cast<const Derived &>(*this);
  }

public:
  Value castValue(Builder &builder, Type target, Value source) const {
    if (target == source.getType())
      return source;
    return createCast<CastOp>(builder, target, source);
  }

  Value castValueSequence(Builder &builder, Type target, Value source) const {
    SmallVector<Type, 2> sequence;
    Value result = source;
    auto castResult =
        getDerived().castSequence(target, source.getType(), sequence);
    if (castResult != InvalidCast) {
      for (auto type : sequence) {
        auto cast = createCast<CastOp>(builder, type, result);
        assert(cast);
        result = cast.getResult();
      }
      return result;
    }
    return nullptr;
  }
};

template <typename Derived, typename Target, int Options = 0,
          typename... Parents>
class ConcretizationPattern
    : public GenericOpPattern<
          PatternInfo::Rewriter, Target, Options, ConcretizationBase<Derived>,
          XBLangTypeSystemMixin<Derived>,
          TypeSystemMixinAdaptor<Derived, XBLangTypeSystem>, Parents...> {
  using ThisBase = GenericOpPattern<
      PatternInfo::Rewriter, Target, Options, ConcretizationBase<Derived>,
      XBLangTypeSystemMixin<Derived>,
      TypeSystemMixinAdaptor<Derived, XBLangTypeSystem>, Parents...>;

public:
  using Base = ConcretizationPattern;

  ConcretizationPattern(XBLangTypeSystem &typeSystem, MLIRContext *context,
                        mlir::PatternBenefit benefit = 1,
                        ArrayRef<StringRef> generatedNames = {})
      : ThisBase(context, benefit, generatedNames), typeSystem(&typeSystem) {}

  const XBLangTypeSystem &getTypeSystem() const {
    assert(typeSystem);
    return *typeSystem;
  }

protected:
  XBLangTypeSystem *typeSystem{};
};
} // namespace xb
} // namespace xblang

#endif
