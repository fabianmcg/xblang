#include "xblang/Dialect/XBLang/IR/Dialect.h"
#include "xblang/Dialect/XBLang/IR/Enums.h"
#include "xblang/Dialect/XBLang/IR/Type.h"
#include "xblang/Dialect/XBLang/IR/XBLang.h"
#include "xblang/Support/CompareExtras.h"

#include "mlir/Interfaces/SideEffectInterfaces.h"

using namespace mlir;
using namespace xblang::xb;

//===----------------------------------------------------------------------===//
// XB constant Op
//===----------------------------------------------------------------------===//

OpFoldResult ConstantOp::fold(FoldAdaptor adaptor) { return getValue(); }

//===----------------------------------------------------------------------===//
// XB call Op
//===----------------------------------------------------------------------===//

CallOp::operand_range CallOp::getArgOperands() {
  return {operand_begin(), operand_end()};
}

MutableOperandRange CallOp::getArgOperandsMutable() {
  return getOperandsMutable();
}

CallInterfaceCallable CallOp::getCallableForCallee() {
  return (*this)->getAttrOfType<SymbolRefAttr>("callee");
}

FunctionType CallOp::getCalleeType() {
  return FunctionType::get(getContext(), getOperandTypes(), getResultTypes());
}

void CallOp::setCalleeFromCallable(CallInterfaceCallable callee) {
  (*this)->setAttr("callee", callee.get<SymbolRefAttr>());
}

//===----------------------------------------------------------------------===//
// XB cast Op
//===----------------------------------------------------------------------===//

OpFoldResult CastOp::fold(FoldAdaptor adaptor) {
  return getType() == getValue().getType() ? getValue() : nullptr;
}

void CastOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  if (isa<ReferenceType>(getType()) &&
      removeReference(getType()) == getValue().getType())
    effects.push_back(
        MemoryEffects::EffectInstance(MemoryEffects::Read::get(), getValue()));
}

Speculation::Speculatability CastOp::getSpeculatability() {
  if (isa<ReferenceType>(getType()) &&
      removeReference(getType()) == getValue().getType())
    return Speculation::Speculatability::NotSpeculatable;
  return Speculation::Speculatability::Speculatable;
}

//===----------------------------------------------------------------------===//
// XB nullptr Op
//===----------------------------------------------------------------------===//

OpFoldResult NullPtrOp::fold(FoldAdaptor adaptor) {
  return TypeAttr::get(getType());
}

//===----------------------------------------------------------------------===//
// XB range Op
//===----------------------------------------------------------------------===//

Type RangeOp::getImplicitCast(unsigned arg) {
  auto type = getOperand(arg).getType();
  if (type.isa<ReferenceType>() ||
      removeReference(type) != getType().getIteratorType())
    return getType().getIteratorType();
  return nullptr;
}

//===----------------------------------------------------------------------===//
// XB size_of Op
//===----------------------------------------------------------------------===//

OpFoldResult SizeOfOp::fold(FoldAdaptor adaptor) {
  return adaptor.getTypeAttr();
}
