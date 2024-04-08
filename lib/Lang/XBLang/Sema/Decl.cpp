//===- Decl.cpp - XBG semantic checkers for decl constructs -----*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements XBG semantic checkers for decl constructs.
//
//===----------------------------------------------------------------------===//

#include "xblang/Lang/XBLang/Sema/Sema.h"

#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "xblang/Basic/Context.h"
#include "xblang/Dialect/XBLang/IR/Type.h"
#include "xblang/Lang/XBLang/Sema/Util.h"
#include "xblang/Lang/XBLang/XLG/XBGDecl.h"
#include "xblang/Lang/XBLang/XLG/XBGDialect.h"
#include "xblang/Lang/XBLang/XLG/XBGExpr.h"
#include "xblang/Lang/XBLang/XLG/XBGStmt.h"
#include "xblang/Lang/XBLang/XLG/XBGType.h"
#include "xblang/Sema/Sema.h"
#include "xblang/XLG/Concepts.h"
#include "xblang/XLG/IR/XLGDialect.h"
#include "xblang/XLG/Interfaces.h"

using namespace xblang;
using namespace xblang::sema;
using namespace xblang::xbg;

namespace {
//===----------------------------------------------------------------------===//
// FuncDeclVerifier
//===----------------------------------------------------------------------===//
struct FuncDeclVerifier : public xblang::sema::SemaOpPattern<FuncDecl> {
  using Base::Base;

  SemaResult check(Op op, Status status, xblang::SymbolTable *symTable,
                   SemaDriver &driver) const final;
  SemaResult checkOp(Op op, Status status, xblang::SymbolTable *symTable,
                     SemaDriver &driver) const final;
};

//===----------------------------------------------------------------------===//
// FuncDefVerifier
//===----------------------------------------------------------------------===//
struct FuncDefVerifier : public xblang::sema::SemaOpPattern<FuncDef> {
  using Base::Base;

  SemaResult check(Op op, Status status, xblang::SymbolTable *symTable,
                   SemaDriver &driver) const final;
  SemaResult checkOp(Op op, Status status, xblang::SymbolTable *symTable,
                     SemaDriver &driver) const final;
};

//===----------------------------------------------------------------------===//
// ObjectDeclVerifier
//===----------------------------------------------------------------------===//
struct ObjectDeclVerifier : public xblang::sema::SemaOpPattern<ObjectDecl> {
  using Base::Base;

  SemaResult check(Op op, Status status, xblang::SymbolTable *symTable,
                   SemaDriver &driver) const final;
  SemaResult checkOp(Op op, Status status, xblang::SymbolTable *symTable,
                     SemaDriver &driver) const final;
};

//===----------------------------------------------------------------------===//
// VarDeclVerifier
//===----------------------------------------------------------------------===//
struct VarDeclVerifier : public xblang::sema::SemaOpPattern<VarDecl> {
  using Base::Base;

  SemaResult check(Op op, Status status, xblang::SymbolTable *symTable,
                   SemaDriver &driver) const final;
  SemaResult checkOp(Op op, Status status, xblang::SymbolTable *symTable,
                     SemaDriver &driver) const final;
};
} // namespace

//===----------------------------------------------------------------------===//
// FuncDeclVerifier
//===----------------------------------------------------------------------===//

SemaResult FuncDeclVerifier::check(Op op, Status status,
                                   xblang::SymbolTable *symTable,
                                   SemaDriver &driver) const {
  if (status.getCount() == 0) {
    if (SemaResult result = driver.checkOperands(op, symTable);
        !result.succeeded())
      return result;
    if (SemaResult result = driver.checkRegions(op, symTable);
        !result.succeeded())
      return result;
    if (SemaResult result = checkOp(op, status, symTable, driver);
        !result.succeeded())
      return result;
    return SemaResult::successAndReschedule();
  } else if (status.getCount() == 1) {
    auto ret = cast<xlg::ReturnOp>(op.getBody(0)->getTerminator());
    // Re-check the body of the function.
    return driver.checkValue(ret.getExpr(), nullptr, true);
  } else
    return mlir::success();
}

SemaResult FuncDeclVerifier::checkOp(Op op, Status status,
                                     xblang::SymbolTable *symTable,
                                     SemaDriver &driver) const {
  auto ret = cast<xlg::ReturnOp>(op.getBody(0)->getTerminator());
  auto type = cast<TypeAttrInterface>(ret.getExpr().getDefiningOp()).getType();
  op.setTypeAttr(TypeAttr::get(type));
  StringAttr usr;
  if (op->getDiscardableAttr("extern") || op.getSymId() == "main")
    op.setUsrAttr(usr = op.getSymIdAttr());
  else
    op.setUsrAttr(usr = driver.getSymbolUSR(op.getSymIdAttr()));
  driver.setUSR(usr, op);
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// FuncDeclVerifier
//===----------------------------------------------------------------------===//

SemaResult FuncDefVerifier::check(Op op, Status status,
                                  xblang::SymbolTable *symTable,
                                  SemaDriver &driver) const {
  // On the first visit only check the signature.
  if (status.getCount() == 0) {
    if (SemaResult result = driver.checkOperands(op, symTable);
        !result.succeeded())
      return result;
    if (SemaResult result = checkOp(op, status, symTable, driver);
        !result.succeeded())
      return result;
    return SemaResult::success();
  } else if (status.getCount() == 1) {
    // On the second visit check the body of the function.
    return driver.checkRegions(op, symTable);
  } else
    return mlir::success();
}

SemaResult FuncDefVerifier::checkOp(Op op, Status status,
                                    xblang::SymbolTable *symTable,
                                    SemaDriver &driver) const {
  SmallVector<mlir::Type> ins;
  SmallVector<mlir::Type> outs;
  if (auto ret = op.getReturnType()) {
    auto typeIface =
        driver.getInterface<xblang::xlg::TypeInterface>(ret.getDefiningOp());
    outs.push_back(typeIface.getTypeAttr().getValue());
  }
  for (auto param : op.getArguments()) {
    auto varDecl = driver.getInterface<xblang::xlg::VarDeclInterface>(
        param.getDefiningOp());
    if (!varDecl)
      return op.emitError("parameter must be defined by an op");
    assert(varDecl.getValueType() && "invalid variable declaration");
    auto typeIface = driver.getInterface<xblang::xlg::TypeInterface>(
        varDecl.getValueType().getDefiningOp());
    assert(typeIface && "invalid type");
    ins.push_back(typeIface.getTypeAttr().getValue());
  }
  op.setTypeAttr(TypeAttr::get(driver.getFunctionType(ins, outs)));
  if (op.getRegion().empty())
    return mlir::success();
  Operation *lastOp{};
  Location loc = op.getLoc();
  if (!op.getBody()->empty()) {
    lastOp = &op.getBody()->back();
    loc = lastOp->getLoc();
  }
  if (lastOp && isa<xlg::ReturnStmtInterface>(lastOp))
    return mlir::success();
  {
    auto grd = lastOp ? driver.guardAfter(driver, lastOp)
                      : driver.guard(driver, op.getBody());
    if (outs.size() == 0) {
      driver.create<ReturnStmt>(loc, driver.getConceptClass<ReturnStmtCep>(),
                                nullptr);
      return success();
    }
    op.emitWarning("missing return statement in function returning non void");
    Value poisson = driver.create<mlir::ub::PoisonOp>(loc, outs[0]);
    poisson = driver.create<ToXLGExpr>(loc, driver.getConceptClass<xlg::Expr>(),
                                       poisson);
    driver.create<ReturnStmt>(loc, driver.getConceptClass<ReturnStmtCep>(),
                              poisson);
  }
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// ObjectDeclVerifier
//===----------------------------------------------------------------------===//
SemaResult ObjectDeclVerifier::check(Op op, Status status,
                                     xblang::SymbolTable *symTable,
                                     SemaDriver &driver) const {
  if (status.getCount() == 0) {
    op.setUsrAttr(driver.getSymbolUSR(op.getSymIdAttr()));
    if (SemaResult result = checkOp(op, status, symTable, driver);
        !result.succeeded())
      return result;
    return SemaResult::successAndReschedule();
  } else if (status.getCount() == 1) {
    return driver.checkRegions(op, symTable);
  } else
    return mlir::success();
}

SemaResult ObjectDeclVerifier::checkOp(Op op, Status status,
                                       xblang::SymbolTable *symTable,
                                       SemaDriver &driver) const {
  SmallVector<Type> members;
  if (!op.getDeclBody().empty())
    for (auto vd : op.getBodyRegion(0).getOps<VarDecl>()) {
      if (SemaResult result = driver.checkOp(vd, nullptr); !result.succeeded())
        return result;
      members.push_back(vd.getTypeAttr().getValue());
    }
  op.setType(driver.getType<xb::NamedType>(
      op.getUSR().getValue(), driver.getType<xb::StructType>(members)));
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// VarDeclVerifier
//===----------------------------------------------------------------------===//
SemaResult VarDeclVerifier::check(Op op, Status status,
                                  xblang::SymbolTable *symTable,
                                  SemaDriver &driver) const {
  if (SemaResult result = driver.checkOperands(op, symTable);
      !result.succeeded())
    return result;
  return checkOp(op, status, symTable, driver);
}

SemaResult VarDeclVerifier::checkOp(Op op, Status status,
                                    xblang::SymbolTable *symTable,
                                    SemaDriver &driver) const {
  if (!op.getValueType() && !op.getExpr())
    return op.emitError(
        "variable declarations require a type or an initializer");
  if (!op.getValueType() && op.getExpr()) {
    auto expr = driver.getInterface<xblang::xlg::ExprInterface>(
        op.getExpr().getDefiningOp());
    assert(expr && "invalid type");
    if (!expr.getTypeAttr())
      return op.emitError("the type couldn't be deduced from the initializer");
    op.getValueTypeMutable().assign(driver.create<BuiltinType>(
        op.getLoc(), driver.getConceptClass<BuiltinTypeCep>(),
        expr.getTypeAttr().getValue()));
  }
  TypeAttrInterface type =
      dyn_cast_or_null<TypeAttrInterface>(op.getValueType().getDefiningOp());
  if (!type || !type.getType())
    return op.emitError("the type couldn't be verified");
  auto cep = op.getOpConcept();
  if (isa<MemberDecl>(cep))
    op.setTypeAttr(TypeAttr::get(type.getType()));
  else if (auto mrTy = dyn_cast<MemRefType>(type.getType()))
    op.setType(mrTy);
  else
    op.setTypeAttr(TypeAttr::get(xb::ReferenceType::get(type.getType())));
  if (Value expr = op.getExpr()) {
    Type ty = type.getType();
    auto initType = cast<TypeAttrInterface>(expr.getDefiningOp()).getType();
    assert(initType && "invalid expression");
    Operand val = getOrLoadValue({expr, initType}, driver);
    expr = driver->makeCast(ty, val.type, val.value, driver);
    op.getExprMutable().assign(expr);
  }
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// XBG sema patterns
//===----------------------------------------------------------------------===//

void xblang::xbg::populateDeclSemaPatterns(GenericPatternSet &set) {
  set.add<FuncDeclVerifier, FuncDefVerifier, ObjectDeclVerifier,
          VarDeclVerifier>(set.getMLIRContext());
}
