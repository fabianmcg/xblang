#include "xblang/Dialect/XBLang/Lowering/Type.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "xblang/Dialect/XBLang/IR/XBLang.h"
#include "llvm/ADT/ScopeExit.h"

namespace xblang {
namespace xb {
namespace {
Type convertNamedType(XBLangTypeConverter &converter, NamedType type) {
  thread_local mlir::SetVector<StringRef> knownNames;
  unsigned stackSize = knownNames.size();
  (void)stackSize;
  auto guard = llvm::make_scope_exit([&]() {
    assert(knownNames.size() == stackSize &&
           "malformed identified stack when converting recursive types");
  });
  auto name = type.getName();
  if (knownNames.count(name))
    return type;
  if (type.getType()) {
    knownNames.insert(type.getName());
    auto result = type.setType(converter.convertType(type.getType()));
    knownNames.pop_back();
    if (failed(result))
      return {};
  }
  return type;
}
} // namespace

XBLangTypeConverter::XBLangTypeConverter(MLIRContext &ctx) : context(&ctx) {
  // If there's no conversion return the type as legal.
  addConversion([&](Type type) { return type; });
  // Convert tensors to memrefs.
  addConversion([&](TensorType type) {
    return mlir::MemRefType::get(type.getShape(),
                                 convertType(type.getElementType()));
  });
  // Convert signed & unsigned ints to sign-less ints.
  addConversion([&](IntegerType type) -> Type {
    return mlir::IntegerType::get(context, type.getWidth(),
                                  IntegerType::Signless);
  });
  // Update the inner type.
  addConversion([&](ReferenceType type) -> Type {
    return ReferenceType::get(context, convertType(type.getPointee()),
                              type.getMemorySpace());
  });
  // Update the inner type.
  addConversion([&](PointerType type) -> Type {
    return PointerType::get(context, convertType(type.getPointee()),
                            type.getMemorySpace());
  });
  // Update the members type.
  addConversion([&](StructType type) -> Type {
    SmallVector<Type> members;
    if (failed(convertTypes(type.getMembers(), members)))
      return Type();
    return StructType::get(context, members);
  });
  // Update the signature.
  addConversion([&](FunctionType type) -> Type {
    TypeConverter::SignatureConversion insTy(type.getNumInputs());
    SmallVector<Type, 1> outsTy;
    if (failed(convertSignatureArgs(type.getInputs(), insTy)) ||
        failed(convertTypes(type.getResults(), outsTy)))
      return Type();
    // Create the new funcOp.
    return FunctionType::get(getContext(), insTy.getConvertedTypes(), outsTy);
  });
  // Update the inner type.
  addConversion(
      [&](NamedType type) -> Type { return convertNamedType(*this, type); });
  // Materialize sources to unrealized casts.
  addSourceMaterialization([&](OpBuilder &builder, Type resultType,
                               ValueRange inputs,
                               Location loc) -> std::optional<Value> {
    if (inputs.size() != 1)
      return std::nullopt;
    return builder
        .create<mlir::UnrealizedConversionCastOp>(loc, resultType, inputs)
        .getResult(0);
  });
  // Materialize targets to unrealized casts.
  addTargetMaterialization([&](OpBuilder &builder, Type resultType,
                               ValueRange inputs,
                               Location loc) -> std::optional<Value> {
    if (inputs.size() != 1)
      return std::nullopt;
    return builder
        .create<mlir::UnrealizedConversionCastOp>(loc, resultType, inputs)
        .getResult(0);
  });
}

XBLangToLLVMTypeConverter::XBLangToLLVMTypeConverter(
    MLIRContext *ctx, const mlir::DataLayoutAnalysis *analysis)
    : mlir::LLVMTypeConverter(ctx, analysis) {
  init();
}

XBLangToLLVMTypeConverter::XBLangToLLVMTypeConverter(
    MLIRContext *ctx, const mlir::LowerToLLVMOptions &options,
    const mlir::DataLayoutAnalysis *analysis)
    : mlir::LLVMTypeConverter(ctx, options, analysis) {
  init();
}

void XBLangToLLVMTypeConverter::init() {
  addConversion([&](AddressType type) { return convertAddressType(type); });
  addConversion(
      [&](ReferenceType type) -> Type { return convertReferenceType(type); });
  addConversion(
      [&](PointerType type) -> Type { return convertPointerType(type); });
  addConversion(
      [&](StructType type) -> Type { return convertStructType(type); });
  addConversion([&](NamedType type) -> Type { return convertNamedType(type); });
}

Type XBLangToLLVMTypeConverter::convertAddressType(AddressType type) const {
  using namespace mlir::LLVM;
  auto result = convertTypeAttribute(type, type.getMemorySpace());
  unsigned addressSpace = 0;
  if (result)
    if (auto attr = dyn_cast<IntegerAttr>(*result))
      addressSpace = attr.getInt();
  return mlir::LLVM::LLVMPointerType::get(&getContext(), addressSpace);
}

Type XBLangToLLVMTypeConverter::convertReferenceType(ReferenceType type) const {
  using namespace mlir::LLVM;
  auto result = convertTypeAttribute(type, type.getMemorySpace());
  unsigned addressSpace = 0;
  if (result)
    if (auto attr = dyn_cast<IntegerAttr>(*result))
      addressSpace = attr.getInt();
  return mlir::LLVM::LLVMPointerType::get(&getContext(), addressSpace);
}

Type XBLangToLLVMTypeConverter::convertPointerType(PointerType type) const {
  using namespace mlir::LLVM;
  auto result = convertTypeAttribute(type, type.getMemorySpace());
  unsigned addressSpace = 0;
  if (result)
    if (auto attr = dyn_cast<IntegerAttr>(*result))
      addressSpace = attr.getInt();
  return mlir::LLVM::LLVMPointerType::get(&getContext(), addressSpace);
}

Type XBLangToLLVMTypeConverter::convertStructType(StructType type) const {
  SmallVector<Type> transformed(type.getMembers());
  for (auto &type : transformed)
    type = convertType(type);
  return mlir::LLVM::LLVMStructType::getLiteral(&getContext(), transformed,
                                                false);
}

Type XBLangToLLVMTypeConverter::convertNamedType(NamedType type) const {
  thread_local mlir::SetVector<StringRef> knownNames;
  unsigned stackSize = knownNames.size();
  (void)stackSize;
  auto guard = llvm::make_scope_exit([&]() {
    assert(knownNames.size() == stackSize &&
           "malformed identified stack when converting recursive types");
  });
  auto name = type.getName();
  if (knownNames.count(name)) {
    if (type.getType() && isa<StructType>(type.getType()))
      return mlir::LLVM::LLVMStructType::getIdentified(type.getContext(), name);
    return {};
  }
  if (type.getType()) {
    knownNames.insert(type.getName());
    auto converted = convertType(type.getType());
    knownNames.pop_back();
    if (auto st = dyn_cast<mlir::LLVM::LLVMStructType>(converted)) {
      if (st.isIdentified() && st.isInitialized())
        return st;
      auto identified =
          mlir::LLVM::LLVMStructType::getIdentified(type.getContext(), name);
      if (!identified.isInitialized() &&
          failed(identified.setBody(st.getBody(), false)))
        return {};
      return identified;
    }
    return {};
  }
  return mlir::LLVM::LLVMStructType::getOpaque(type.getName(), &getContext());
}
} // namespace xb
} // namespace xblang
