#ifndef XBLANG_DIALECT_XBLANG_IR_TYPE_H
#define XBLANG_DIALECT_XBLANG_IR_TYPE_H

#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpDefinition.h"
#include "xblang/Support/LLVM.h"
#include "llvm/ADT/TypeSwitch.h"

namespace xblang {
namespace xb {
template <typename ConcreteType>
class FundamentalTypeTrait
    : public mlir::TypeTrait::TraitBase<ConcreteType, FundamentalTypeTrait> {};

template <typename ConcreteType>
class ArithmeticTypeTrait
    : public mlir::TypeTrait::TraitBase<ConcreteType, ArithmeticTypeTrait> {};

template <typename ConcreteType>
class ScalarTypeTrait
    : public mlir::TypeTrait::TraitBase<ConcreteType, ScalarTypeTrait> {};

template <typename ConcreteType>
class AlgebraicTypeTrait
    : public mlir::TypeTrait::TraitBase<ConcreteType, AlgebraicTypeTrait> {};
} // namespace xb
} // namespace xblang

#define GET_TYPEDEF_CLASSES
#include "xblang/Dialect/XBLang/IR/XBLangTypes.h.inc"

namespace xblang {
namespace xb {
namespace detail {
class NamedTypeStorage;
}

class NamedType
    : public Type::TypeBase<NamedType, Type, detail::NamedTypeStorage,
                            mlir::TypeTrait::IsMutable,
                            ::mlir::MemRefElementTypeInterface::Trait> {
public:
  using Base::Base;

  static constexpr ::llvm::StringLiteral name = "xb.unq";

  static constexpr ::llvm::StringLiteral getMnemonic() { return {"unq"}; }

  static NamedType get(MLIRContext *context, StringRef name, Type type = {});

  StringRef getName() const;

  Type getType() const;
  LogicalResult setType(Type type);

  bool isOpaque() const;

  static ::mlir::Type parse(::mlir::AsmParser &odsParser);
  void print(::mlir::AsmPrinter &odsPrinter) const;
};
} // namespace xb
} // namespace xblang

namespace xblang {
inline mlir::Type removeReference(mlir::Type type) {
  if (!type)
    return nullptr;
  if (auto refType = type.dyn_cast<xblang::xb::ReferenceType>())
    return refType.getPointee();
  return type;
}

std::pair<mlir::Type, int> arithmeticTypePromotion(mlir::Type, mlir::Type);
} // namespace xblang

#endif
