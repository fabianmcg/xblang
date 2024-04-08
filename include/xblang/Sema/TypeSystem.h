#ifndef XBLANG_SEMA_TYPESYSTEM_H
#define XBLANG_SEMA_TYPESYSTEM_H

#include "mlir/IR/OpDefinition.h"
#include "xblang/Sema/TypeSystemBase.h"
#include "xblang/Sema/XBLangTypeSystemMixin.h"

namespace xblang {
namespace xb {
class XBLangTypeSystem : public TypeSystemMixin<XBLangTypeSystem>,
                         public XBLangTypeSystemMixin<XBLangTypeSystem> {
public:
  using SignednessSemantics = IntegerType::SignednessSemantics;
  XBLangTypeSystem(MLIRContext &context);

  MLIRContext *getContext() const { return context; }

protected:
  friend class TypeSystemMixin<XBLangTypeSystem>;
  CastValidityKind castKindImpl(Type target, Type source) const;
  std::pair<RankValidity, Type> rankTypesImpl(Type lhs, Type rhs) const;
  CastValidityKind castSequenceImpl(Type target, Type source,
                                    SmallVector<Type, 2> &sequence) const;

private:
  mlir::MLIRContext *context{};
};
} // namespace xb
}

#endif
