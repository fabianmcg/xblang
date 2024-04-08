#include "xblang/Dialect/XBLang/IR/Type.h"
#include "xblang/Dialect/XBLang/IR/XBLang.h"

#include "mlir/Interfaces/FunctionImplementation.h"

using namespace mlir;
using namespace xblang::xb;

LogicalResult FunctionOp::verifyType() {
  auto type = getFunctionType();
  if (!type.isa<FunctionType>())
    return emitOpError("requires '" + getFunctionTypeAttrName().str() +
                       "' attribute of function type");
  if (getFunctionType().getNumResults() > 1)
    return emitOpError("cannot have more than one result");
  return success();
}

LogicalResult FunctionOp::verify() { return verifyType(); }

void FunctionOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                       llvm::StringRef name, mlir::FunctionType type,
                       llvm::ArrayRef<mlir::NamedAttribute> attrs,
                       llvm::ArrayRef<DictionaryAttr> argAttrs) {
  buildWithEntryBlock(builder, state, name, type, attrs, type.getInputs());

  if (argAttrs.empty())
    return;
  assert(type.getNumInputs() == argAttrs.size());
  function_interface_impl::addArgAndResultAttrs(
      builder, state, argAttrs, /*resultAttrs=*/std::nullopt,
      getArgAttrsAttrName(state.name), getResAttrsAttrName(state.name));
}

mlir::ParseResult FunctionOp::parse(mlir::OpAsmParser &parser,
                                    mlir::OperationState &result) {
  auto fnBuilder =
      [](mlir::Builder &builder, llvm::ArrayRef<mlir::Type> argTypes,
         llvm::ArrayRef<mlir::Type> results,
         mlir::function_interface_impl::VariadicFlag,
         std::string &) { return builder.getFunctionType(argTypes, results); };
  return mlir::function_interface_impl::parseFunctionOp(
      parser, result, false, getFunctionTypeAttrName(result.name), fnBuilder,
      getArgAttrsAttrName(result.name), getResAttrsAttrName(result.name));
}

void FunctionOp::print(mlir::OpAsmPrinter &p) {
  mlir::function_interface_impl::printFunctionOp(
      p, *this, false, getFunctionTypeAttrName(), getArgAttrsAttrName(),
      getResAttrsAttrName());
}

void VarOp::getAsmResultNames(function_ref<void(Value, StringRef)> setNameFn) {
  setNameFn(getResult(), getSymName());
}

Type VarOp::getImplicitCast(unsigned arg) {
  auto init = getInit();
  if (!init)
    return nullptr;
  auto type = getType();
  auto initType = init.getType();
  if (type.isa<ReferenceType>() && type != initType) {
    emitError("Is not possible to assign a reference between different types.");
    return nullptr;
  }
  if (type == initType)
    return nullptr;
  return type;
}
