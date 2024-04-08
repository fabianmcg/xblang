//===- Type.cpp - XBG semantic checkers for type constructs -----*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements XBG semantic checkers for type constructs.
//
//===----------------------------------------------------------------------===//

#include "xblang/Lang/XBLang/Sema/Sema.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "xblang/Basic/Context.h"
#include "xblang/Dialect/XBLang/IR/Type.h"
#include "xblang/Lang/XBLang/XLG/XBGDialect.h"
#include "xblang/Lang/XBLang/XLG/XBGExpr.h"
#include "xblang/Lang/XBLang/XLG/XBGType.h"
#include "xblang/Sema/Sema.h"
#include "xblang/XLG/Concepts.h"
#include "xblang/XLG/Interfaces.h"

using namespace mlir;
using namespace xblang::xlg;
using namespace xblang::sema;
using namespace xblang::xbg;

//===----------------------------------------------------------------------===//
// XBG sema patterns
//===----------------------------------------------------------------------===//

namespace {
//===----------------------------------------------------------------------===//
// ArrayTypeVerifier
//===----------------------------------------------------------------------===//
struct ArrayTypeVerifier : public xblang::sema::SemaOpPattern<ArrayType> {
  using Base::Base;

  SemaResult check(Op op, Status status, xblang::SymbolTable *symTable,
                   SemaDriver &driver) const final;
  SemaResult checkOp(Op op, Status status, xblang::SymbolTable *symTable,
                     SemaDriver &driver) const final;
};

//===----------------------------------------------------------------------===//
// PointerTypeVerifier
//===----------------------------------------------------------------------===//
struct PointerTypeVerifier : public xblang::sema::SemaOpPattern<PointerType> {
  using Base::Base;

  SemaResult check(Op op, Status status, xblang::SymbolTable *symTable,
                   SemaDriver &driver) const final;
  SemaResult checkOp(Op op, Status status, xblang::SymbolTable *symTable,
                     SemaDriver &driver) const final;
};

//===----------------------------------------------------------------------===//
// ReferenceTypeVerifier
//===----------------------------------------------------------------------===//
struct ReferenceTypeVerifier
    : public xblang::sema::SemaOpPattern<ReferenceType> {
  using Base::Base;

  SemaResult check(Op op, Status status, xblang::SymbolTable *symTable,
                   SemaDriver &driver) const final;
  SemaResult checkOp(Op op, Status status, xblang::SymbolTable *symTable,
                     SemaDriver &driver) const final;
};

//===----------------------------------------------------------------------===//
// RefExprTypeVerifier
//===----------------------------------------------------------------------===//
struct RefExprTypeVerifier : public xblang::sema::SemaOpPattern<RefExprType> {
  using Base::Base;

  SemaResult check(Op op, Status status, xblang::SymbolTable *symTable,
                   SemaDriver &driver) const final;
};

//===----------------------------------------------------------------------===//
// RemoveReferenceTypeVerifier
//===----------------------------------------------------------------------===//
struct RemoveReferenceTypeVerifier
    : public xblang::sema::SemaOpPattern<xblang::xbg::RemoveReferenceType> {
  using Base::Base;

  SemaResult check(Op op, Status status, xblang::SymbolTable *symTable,
                   SemaDriver &driver) const final;
};

//===----------------------------------------------------------------------===//
// TemplateTypeVerifier
//===----------------------------------------------------------------------===//
struct TemplateTypeVerifier : public xblang::sema::SemaOpPattern<TemplateType> {
  using Base::Base;

  SemaResult check(Op op, Status status, xblang::SymbolTable *symTable,
                   SemaDriver &driver) const final;
};

//===----------------------------------------------------------------------===//
// TypeOfVerifier
//===----------------------------------------------------------------------===//
struct TypeOfVerifier : public xblang::sema::SemaOpPattern<TypeOf> {
  using Base::Base;

  SemaResult check(Op op, Status status, xblang::SymbolTable *symTable,
                   SemaDriver &driver) const final;
};
} // namespace

//===----------------------------------------------------------------------===//
// ArrayTypeVerifier
//===----------------------------------------------------------------------===//
SemaResult ArrayTypeVerifier::check(Op op, Status status,
                                    xblang::SymbolTable *symTable,
                                    SemaDriver &driver) const {
  if (SemaResult result = driver.checkOperands(op, symTable);
      !result.succeeded())
    return result;
  return checkOp(op, status, symTable, driver);
}

SemaResult ArrayTypeVerifier::checkOp(Op op, Status status,
                                      xblang::SymbolTable *symTable,
                                      SemaDriver &driver) const {
  auto typeIface =
      dyn_cast<xblang::TypeAttrInterface>(op.getBase().getDefiningOp());
  if (!typeIface)
    return op.emitError("the base cannot be checked");
  auto type = typeIface.getType();
  if (!type)
    return op.emitError("failed to infer the array type");
  // FIXME: Check constant expressions in general and verify they are positive
  // numbers.
  SmallVector<int64_t> dims;
  for (auto dim : op.getShape()) {
    if (Attribute constExpr = driver.eval(dim.getDefiningOp(), {})) {
      if (auto intAttr = dyn_cast<IntegerAttr>(constExpr)) {
        if (intAttr.getType().isSignedInteger())
          dims.push_back(intAttr.getSInt());
        else if (intAttr.getType().isUnsignedInteger())
          dims.push_back(intAttr.getUInt());
        else
          dims.push_back(intAttr.getInt());
        continue;
      }
    }
    dims.push_back(mlir::ShapedType::kDynamic);
  }
  if (dims.size() > 0)
    op.setTypeAttr(TypeAttr::get(MemRefType::get(dims, type)));
  else
    op.setTypeAttr(TypeAttr::get(xblang::xb::PointerType::get(type)));
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// PointerTypeVerifier
//===----------------------------------------------------------------------===//
SemaResult PointerTypeVerifier::check(Op op, Status status,
                                      xblang::SymbolTable *symTable,
                                      SemaDriver &driver) const {
  if (SemaResult result = driver.checkOperands(op, symTable);
      !result.succeeded())
    return result;
  return checkOp(op, status, symTable, driver);
}

SemaResult PointerTypeVerifier::checkOp(Op op, Status status,
                                        xblang::SymbolTable *symTable,
                                        SemaDriver &driver) const {
  auto typeIface = driver.getInterface<xblang::xlg::TypeInterface>(
      op.getBase().getDefiningOp());
  if (!typeIface)
    return op.emitError("the base cannot be checked");
  auto type = typeIface.getTypeAttr();
  if (!type)
    return op.emitError("failed to infer the pointer type");
  op.setTypeAttr(TypeAttr::get(xblang::xb::PointerType::get(type.getValue())));
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// ReferenceTypeVerifier
//===----------------------------------------------------------------------===//
SemaResult ReferenceTypeVerifier::check(Op op, Status status,
                                        xblang::SymbolTable *symTable,
                                        SemaDriver &driver) const {
  if (SemaResult result = driver.checkOperands(op, symTable);
      !result.succeeded())
    return result;
  return checkOp(op, status, symTable, driver);
}

SemaResult ReferenceTypeVerifier::checkOp(Op op, Status status,
                                          xblang::SymbolTable *symTable,
                                          SemaDriver &driver) const {
  auto typeIface = driver.getInterface<xblang::xlg::TypeInterface>(
      op.getBase().getDefiningOp());
  if (!typeIface)
    return op.emitError("the base cannot be checked");
  auto type = typeIface.getTypeAttr();
  if (!type)
    return op.emitError("failed to infer the pointer type");
  op.setTypeAttr(
      TypeAttr::get(xblang::xb::ReferenceType::get(type.getValue())));
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// RefExprTypeVerifier
//===----------------------------------------------------------------------===//

SemaResult RefExprTypeVerifier::check(Op op, Status status,
                                      xblang::SymbolTable *symTable,
                                      SemaDriver &driver) const {
  xblang::SymbolCollection symCollection = symTable->lookup(op.getSymName());
  if (symCollection.empty())
    return op.emitError("type couldn't be found");
  if (symCollection.size() > 1)
    return op.emitError("symbol is ambiguous");

  if (SemaResult result =
          driver.checkOp(symCollection[0].getSymbol().getOperation(), nullptr);
      !result.succeeded())
    return result;
  Operation *sym = symCollection[0].getSymbol().getOperation();

  auto decl = driver.getInterface<NamedDeclInterface>(sym);

  if (!decl)
    if (auto type = dyn_cast<xblang::TypeAttrInterface>(sym)) {
      if (!type.getType())
        return op.emitError("symbol doesn't have a valid type");
      driver.replaceOp(op, sym);
      return success();
    }

  if (!decl)
    return op.emitError("symbol doesn't refer to a declaration");
  assert(decl.getUsrAttr() && "invalid USR");
  op.setType(
      driver.getType<xblang::xb::NamedType>(decl.getUsrAttr().getValue()));
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// RemoveReferenceTypeVerifier
//===----------------------------------------------------------------------===//

SemaResult RemoveReferenceTypeVerifier::check(Op op, Status status,
                                              xblang::SymbolTable *symTable,
                                              SemaDriver &driver) const {
  if (SemaResult result = driver.checkOperands(op, symTable);
      !result.succeeded())
    return result;
  mlir::Type type =
      cast<xblang::TypeAttrInterface>(op.getBase().getDefiningOp()).getType();
  if (!type)
    return op.emitError("failed to infer the pointer type");
  op.setType(xblang::removeReference(type));
  return success();
}

//===----------------------------------------------------------------------===//
// TemplateTypeVerifier
//===----------------------------------------------------------------------===//

SemaResult TemplateTypeVerifier::check(Op op, Status status,
                                       xblang::SymbolTable *symTable,
                                       SemaDriver &driver) const {
  if (SemaResult result = driver.checkOperands(op, symTable);
      !result.succeeded())
    return result;
  xblang::TypeAttrInterface type = dyn_cast_or_null<xblang::TypeAttrInterface>(
      op.getParameter().getDefiningOp());
  if (!type || !type.getType())
    return op.emitError("invalid template type");
  op.setType(type.getType());
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// TypeOfVerifier
//===----------------------------------------------------------------------===//

SemaResult TypeOfVerifier::check(Op op, Status status,
                                 xblang::SymbolTable *symTable,
                                 SemaDriver &driver) const {
  if (SemaResult result = driver.checkOperands(op, symTable);
      !result.succeeded())
    return result;
  xblang::TypeAttrInterface type =
      dyn_cast_or_null<xblang::TypeAttrInterface>(op.getExpr().getDefiningOp());
  if (!type || !type.getType())
    return op.emitError("invalid type of expression");
  op.setType(type.getType());
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// XBG populate patterns
//===----------------------------------------------------------------------===//

void xblang::xbg::populateTypeSemaPatterns(GenericPatternSet &set) {
  set.add<ArrayTypeVerifier, PointerTypeVerifier, ReferenceTypeVerifier,
          RefExprTypeVerifier, RemoveReferenceTypeVerifier,
          TemplateTypeVerifier, TypeOfVerifier>(set.getMLIRContext());
}
