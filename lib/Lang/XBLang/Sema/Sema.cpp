//===- Sema.cpp - XBG semantic checker ---------------------------*- C++-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares function for registering semantic patterns.
//
//===----------------------------------------------------------------------===//

#include "xblang/Lang/XBLang/Sema/Sema.h"
#include "mlir/IR/DialectRegistry.h"
#include "xblang/Basic/TypeSystem.h"
#include "xblang/Dialect/XBLang/IR/Type.h"
#include "xblang/Interfaces/Sema.h"
#include "xblang/Lang/XBLang/Sema/Util.h"
#include "xblang/Lang/XBLang/XLG/XBGDialect.h"
#include "xblang/Lang/XBLang/XLG/XBGExpr.h"
#include "xblang/Sema/Sema.h"

using namespace xblang;
using namespace xblang::xbg;

namespace {
//===----------------------------------------------------------------------===//
// primitiveCast
//===----------------------------------------------------------------------===//
bool isPrimitiveCast(Type target, Type source) {
  return isPrimitiveType(target) && isPrimitiveType(source);
}

Value primitiveCast(Type target, Type source, Value val, OpBuilder &builder,
                    CastInfo *info) {
  XBContext *context =
      cast<xlg::ConceptType>(val.getType()).getConceptClass().getContext();
  assert(context && "invalid context");
  auto castType = builder.getType<xlg::ConceptType>(CastExprCep::get(context));
  bool srcIsIndex = isa<IndexType>(source);
  bool tgtIsIndex = isa<IndexType>(target);
  bool srcIsFloat = isa<FloatType>(source);
  bool tgtIsFloat = isa<FloatType>(target);
  if ((!srcIsIndex && !tgtIsIndex) || (!srcIsFloat && !tgtIsFloat))
    return builder.create<CastExpr>(val.getLoc(), castType,
                                    TypeAttr::get(target), val, nullptr);
  if ((tgtIsIndex && srcIsFloat) || (tgtIsFloat && srcIsIndex)) {
    val = builder.create<CastExpr>(
        val.getLoc(), castType,
        TypeAttr::get(builder.getIntegerType(64, tgtIsIndex && srcIsFloat)),
        val, nullptr);
    return builder.create<CastExpr>(val.getLoc(), castType,
                                    TypeAttr::get(target), val, nullptr);
  }
  return nullptr;
}

//===----------------------------------------------------------------------===//
// XGBSemaInterface
//===----------------------------------------------------------------------===//
class XGBSemaInterface : public xblang::SemaDialectInterface {
public:
  using xblang::SemaDialectInterface::SemaDialectInterface;

  void populateSemaPatterns(xblang::GenericPatternSet &set,
                            xblang::TypeSystem &typeSystem) const override {
    populateDeclSemaPatterns(set);
    populateExprSemaPatterns(set);
    populateStmtSemaPatterns(set);
    populateTypeSemaPatterns(set);
    typeSystem.setPrimitiveCast(isPrimitiveCast, primitiveCast);
  }
};
} // namespace

void xblang::xbg::registerXBGSemaInterface(mlir::DialectRegistry &registry) {
  registry.insert<XBGDialect>();
  registry.addExtension(+[](mlir::MLIRContext *ctx, XBGDialect *dialect) {
    dialect->addInterfaces<XGBSemaInterface>();
  });
}

void xblang::xbg::registerXBGSemaInterface(mlir::MLIRContext &context) {
  mlir::DialectRegistry registry;
  registerXBGSemaInterface(registry);
  context.appendDialectRegistry(registry);
}

//===----------------------------------------------------------------------===//
// Util functions
//===----------------------------------------------------------------------===//

Type xblang::xbg::getElementType(Type type) {
  mlir::Type pureType{};
  if (auto ty = dyn_cast<xb::ReferenceType>(type))
    pureType = ty.getPointee();
  else if (auto ty = dyn_cast<xb::PointerType>(type))
    pureType = ty.getPointee();
  else if (auto ty = dyn_cast<mlir::MemRefType>(type))
    pureType = ty.getElementType();
  return pureType;
}

Type xblang::xbg::getTypeOrElementType(Type type) {
  if (auto ty = getElementType(type))
    return ty;
  return type;
}

Operand xblang::xbg::getOrLoadValue(Operand value, sema::SemaDriver &driver) {
  mlir::Type pureType = getElementType(value);
  if (pureType)
    value = {driver.create<LoadExpr>(value.value.getLoc(),
                                     driver.getConceptClass<LoadExprCep>(),
                                     TypeAttr::get(pureType), value.value),
             pureType};
  return value;
}

Operation *xblang::xbg::storeValue(Operand address, Operand value,
                                   sema::SemaDriver &driver) {
  mlir::Type pureType = getElementType(address);
  if (pureType != value.type)
    return nullptr;
  return driver.create<StoreExpr>(value.value.getLoc(), TypeAttr::get(pureType),
                                  driver.getConceptClass<StoreExprCep>(),
                                  address.value, value.value);
}
