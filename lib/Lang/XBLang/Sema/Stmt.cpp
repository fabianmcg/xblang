//===- Stmt.cpp - XBG semantic checkers for stmt constructs -----*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements XBG semantic checkers for stmt constructs.
//
//===----------------------------------------------------------------------===//

#include "xblang/Lang/XBLang/Sema/Sema.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "xblang/Basic/Context.h"
#include "xblang/Lang/XBLang/XLG/XBGDialect.h"
#include "xblang/Lang/XBLang/XLG/XBGStmt.h"
#include "xblang/Sema/Sema.h"
#include "xblang/XLG/Concepts.h"
#include "xblang/XLG/Interfaces.h"

using namespace xblang;
using namespace xblang::sema;
using namespace xblang::xbg;

//===----------------------------------------------------------------------===//
// XBG sema patterns
//===----------------------------------------------------------------------===//

namespace {
//===----------------------------------------------------------------------===//
// CompounStmtVerifier
//===----------------------------------------------------------------------===//
struct CompounStmtVerifier : public xblang::sema::SemaOpPattern<CompoundStmt> {
  using Base::Base;

  SemaResult check(Op op, Status status, xblang::SymbolTable *symTable,
                   SemaDriver &driver) const final;
};

//===----------------------------------------------------------------------===//
// IfStmtVerifier
//===----------------------------------------------------------------------===//
struct IfStmtVerifier : public xblang::sema::SemaOpPattern<IfStmt> {
  using Base::Base;

  SemaResult check(Op op, Status status, xblang::SymbolTable *symTable,
                   SemaDriver &driver) const final;
};

//===----------------------------------------------------------------------===//
// RangeForStmtVerifier
//===----------------------------------------------------------------------===//
struct RangeForStmtVerifier : public xblang::sema::SemaOpPattern<RangeForStmt> {
  using Base::Base;

  SemaResult check(Op op, Status status, xblang::SymbolTable *symTable,
                   SemaDriver &driver) const final;
};

//===----------------------------------------------------------------------===//
// ReturnStmtVerifier
//===----------------------------------------------------------------------===//
struct ReturnStmtVerifier : public xblang::sema::SemaOpPattern<ReturnStmt> {
  using Base::Base;

  SemaResult check(Op op, Status status, xblang::SymbolTable *symTable,
                   SemaDriver &driver) const final;
};

//===----------------------------------------------------------------------===//
// WhileStmtVerifier
//===----------------------------------------------------------------------===//
struct WhileStmtVerifier : public xblang::sema::SemaOpPattern<WhileStmt> {
  using Base::Base;

  SemaResult check(Op op, Status status, xblang::SymbolTable *symTable,
                   SemaDriver &driver) const final;
};

//===----------------------------------------------------------------------===//
// YieldStmtVerifier
//===----------------------------------------------------------------------===//
struct YieldStmtVerifier : public xblang::sema::SemaOpPattern<YieldStmt> {
  using Base::Base;

  SemaResult check(Op op, Status status, xblang::SymbolTable *symTable,
                   SemaDriver &driver) const final;
};
} // namespace

//===----------------------------------------------------------------------===//
// CompounStmtVerifier
//===----------------------------------------------------------------------===//

SemaResult CompounStmtVerifier::check(Op op, Status status,
                                      xblang::SymbolTable *symTable,
                                      SemaDriver &driver) const {
  return driver.checkRegions(op, symTable);
}

//===----------------------------------------------------------------------===//
// IfStmtVerifier
//===----------------------------------------------------------------------===//

SemaResult IfStmtVerifier::check(Op op, Status status,
                                 xblang::SymbolTable *symTable,
                                 SemaDriver &driver) const {
  if (auto result = driver.checkValue(op.getCondition(), symTable);
      !result.succeeded())
    return result;
  auto condType =
      cast<TypeAttrInterface>(op.getCondition().getDefiningOp()).getType();
  auto boolType = driver.getI1Type();
  if (!driver->isValidCast(boolType, condType)) {
    op.emitError("conditional is not convertible to bool");
    return SemaResult::failure();
  }
  if (boolType != condType) {
    Value cond =
        driver->makeCast(boolType, condType, op.getCondition(), driver);
    op.getConditionMutable().set(cond);
  }
  return driver.checkRegions(op, symTable);
}

//===----------------------------------------------------------------------===//
// RangeForStmtVerifier
//===----------------------------------------------------------------------===//

SemaResult RangeForStmtVerifier::check(Op op, Status status,
                                       xblang::SymbolTable *symTable,
                                       SemaDriver &driver) const {
  if (auto result = driver.checkOperands(op, symTable); !result.succeeded())
    return result;
  return driver.checkRegions(op, symTable);
}

//===----------------------------------------------------------------------===//
// ReturnStmtVerifier
//===----------------------------------------------------------------------===//

SemaResult ReturnStmtVerifier::check(Op op, Status status,
                                     xblang::SymbolTable *symTable,
                                     SemaDriver &driver) const {
  if (op.getExpr())
    if (SemaResult result = driver.checkValue(op.getExpr(), symTable);
        !result.succeeded())
      return result;
  // Get the parent function.
  auto fn = getParentOfConcept<xlg::FuncDecl>(op);
  if (!fn)
    return op.emitError("return is not inside a function");
  auto fnDecl = driver.getInterface<xlg::FuncDeclInterface>(fn.getOperation());
  assert(fnDecl && "invalid function");
  auto fnType = cast<mlir::FunctionType>(fnDecl.getTypeAttr().getValue());
  // Check the return type and the function type matches.
  if (op.getExpr() == nullptr && fnType.getResults().size() != 0)
    return op.emitError("function returns non void but `return` is void");
  if (fnType.getResults().size() > 1)
    return op.emitError("returning more than one value is unsupported");
  if (op.getExpr() == nullptr && fnType.getResults().size() == 0)
    return SemaResult::success();
  if (op.getExpr() != nullptr && fnType.getResults().size() == 0)
    return op.emitError("function returns void but `return` is non-void");
  // Return immediately if the types match.
  auto retTy = cast<TypeAttrInterface>(op.getExpr().getDefiningOp()).getType();
  if (retTy == fnType.getResult(0))
    return SemaResult::success();
  // Try casting the value if there's a type miss-match.
  if (auto expr =
          driver->makeCast(fnType.getResult(0), retTy, op.getExpr(), driver))
    op.getExprMutable().assign(expr);
  else
    return op.emitError("incompatible return with function type");
  return driver.checkRegions(op, symTable);
}

//===----------------------------------------------------------------------===//
// WhileStmtVerifier
//===----------------------------------------------------------------------===//

SemaResult WhileStmtVerifier::check(Op op, Status status,
                                    xblang::SymbolTable *symTable,
                                    SemaDriver &driver) const {
  if (auto result = driver.checkValue(op.getCondition(), symTable);
      !result.succeeded())
    return result;
  auto condType =
      cast<TypeAttrInterface>(op.getCondition().getDefiningOp()).getType();
  auto boolType = driver.getI1Type();
  if (!driver->isValidCast(boolType, condType)) {
    op.emitError("conditional is not convertible to bool");
    return SemaResult::failure();
  }
  if (boolType != condType) {
    Value cond =
        driver->makeCast(boolType, condType, op.getCondition(), driver);
    op.getConditionMutable().set(cond);
  }
  return driver.checkRegions(op, symTable);
}

//===----------------------------------------------------------------------===//
// YieldStmtVerifier
//===----------------------------------------------------------------------===//

SemaResult YieldStmtVerifier::check(Op op, Status status,
                                    xblang::SymbolTable *symTable,
                                    SemaDriver &driver) const {
  if (getParentOfConcept<xlg::LoopStmt>(op) == nullptr)
    return op.emitError("control-flow statement is not inside a loop");
  return SemaResult::success();
}

//===----------------------------------------------------------------------===//
// XBG populate patterns
//===----------------------------------------------------------------------===//

void xblang::xbg::populateStmtSemaPatterns(GenericPatternSet &set) {
  set.add<CompounStmtVerifier, IfStmtVerifier, RangeForStmtVerifier,
          ReturnStmtVerifier, WhileStmtVerifier, YieldStmtVerifier>(
      set.getMLIRContext());
}
