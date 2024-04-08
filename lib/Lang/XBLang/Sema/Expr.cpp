//===- Expr.cpp - XBG semantic checkers for expr constructs -----*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements XBG semantic checkers for expr constructs.
//
//===----------------------------------------------------------------------===//

#include "xblang/Lang/XBLang/Sema/Sema.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "xblang/Basic/Context.h"
#include "xblang/Dialect/XBLang/IR/Type.h"
#include "xblang/Lang/XBLang/Sema/Util.h"
#include "xblang/Lang/XBLang/XLG/XBGDialect.h"
#include "xblang/Lang/XBLang/XLG/XBGExpr.h"
#include "xblang/Sema/Sema.h"
#include "xblang/Sema/TypeUtil.h"
#include "xblang/Support/Format.h"
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
// ArrayExprVerifier
//===----------------------------------------------------------------------===//
struct ArrayExprVerifier : public xblang::sema::SemaOpPattern<ArrayExpr> {
  using Base::Base;

  SemaResult check(Op op, Status status, xblang::SymbolTable *symTable,
                   SemaDriver &driver) const final;
  SemaResult checkOp(Op op, Status status, xblang::SymbolTable *symTable,
                     SemaDriver &driver) const final;
};

//===----------------------------------------------------------------------===//
// BinOpExprVerifier
//===----------------------------------------------------------------------===//
struct BinOpExprVerifier : public xblang::sema::SemaOpPattern<BinOpExpr> {
  using Base::Base;

  SemaResult check(Op op, Status status, xblang::SymbolTable *symTable,
                   SemaDriver &driver) const final;
  SemaResult checkOp(Op op, Status status, xblang::SymbolTable *symTable,
                     SemaDriver &driver) const final;
  /// Handles the assignment operator.
  SemaResult handleAssign(Op op, Operand lhs, Operand rhs,
                          SemaDriver &driver) const;
};

//===----------------------------------------------------------------------===//
// CallExprVerifier
//===----------------------------------------------------------------------===//
struct CallExprVerifier
    : public xblang::sema::SemaOpPattern<xblang::xbg::CallExpr> {
  using Base::Base;

  /// Returns the types of the arguments of the call expression.
  SmallVector<mlir::Type> getArguments(Op op, SemaDriver &driver) const;
  /// Returns the types of the parameters of the function declaration.
  FunctionType getFuncType(xblang::xlg::FuncDeclInterface op,
                           SemaDriver &driver) const;
  /// Checks the case of an indirect call.
  SemaResult checkIndirectCall(Op op, SmallVector<mlir::Type> &arg,
                               SemaDriver &driver) const;
  /// Checks the case of an direct call.
  SemaResult checkDirectCall(Op op, SmallVector<mlir::Type> &arg,
                             xblang::SymbolTable *symTable,
                             SemaDriver &driver) const;
  SemaResult check(Op op, Status status, xblang::SymbolTable *symTable,
                   SemaDriver &driver) const final;
  SemaResult checkOp(Op op, Status status, xblang::SymbolTable *symTable,
                     SemaDriver &driver) const final;
};

//===----------------------------------------------------------------------===//
// CastExprVerifier
//===----------------------------------------------------------------------===//
struct CastExprVerifier : public xblang::sema::SemaOpPattern<CastExpr> {
  using Base::Base;

  SemaResult check(Op op, Status status, xblang::SymbolTable *symTable,
                   SemaDriver &driver) const final;
};

//===----------------------------------------------------------------------===//
// FromXLGExprVerifier
//===----------------------------------------------------------------------===//
struct FromXLGExprVerifier : public xblang::sema::SemaOpPattern<FromXLGExpr> {
  using Base::Base;

  SemaResult check(Op op, Status status, xblang::SymbolTable *symTable,
                   SemaDriver &driver) const final;
};

//===----------------------------------------------------------------------===//
// ListExprVerifier
//===----------------------------------------------------------------------===//
struct ListExprVerifier
    : public xblang::sema::SemaOpPattern<xblang::xbg::ListExpr> {
  using Base::Base;

  SemaResult check(Op op, Status status, xblang::SymbolTable *symTable,
                   SemaDriver &driver) const final;
  SemaResult checkOp(Op op, Status status, xblang::SymbolTable *symTable,
                     SemaDriver &driver) const final;
};

//===----------------------------------------------------------------------===//
// MemberRefExprVerifier
//===----------------------------------------------------------------------===//
struct MemberRefExprVerifier
    : public xblang::sema::SemaOpPattern<xblang::xbg::MemberRefExpr> {
  using Base::Base;

  SemaResult check(Op op, Status status, xblang::SymbolTable *symTable,
                   SemaDriver &driver) const final;
  SemaResult checkOp(Op op, Status status, xblang::SymbolTable *symTable,
                     SemaDriver &driver) const final;
};

//===----------------------------------------------------------------------===//
// RangeExprVerifier
//===----------------------------------------------------------------------===//
struct RangeExprVerifier
    : public xblang::sema::SemaOpPattern<xblang::xbg::RangeExpr> {
  using Base::Base;

  SemaResult check(Op op, Status status, xblang::SymbolTable *symTable,
                   SemaDriver &driver) const final;
  SemaResult checkOp(Op op, Status status, xblang::SymbolTable *symTable,
                     SemaDriver &driver) const final;
};

//===----------------------------------------------------------------------===//
// RefExprVerifier
//===----------------------------------------------------------------------===//
struct RefExprVerifier
    : public xblang::sema::SemaOpPattern<xblang::xbg::RefExpr> {
  using Base::Base;

  SemaResult check(Op op, Status status, xblang::SymbolTable *symTable,
                   SemaDriver &driver) const final;
  SemaResult checkOp(Op op, Status status, xblang::SymbolTable *symTable,
                     SemaDriver &driver) const final;
};

//===----------------------------------------------------------------------===//
// SelectExprVerifier
//===----------------------------------------------------------------------===//
struct SelectExprVerifier
    : public xblang::sema::SemaOpPattern<xblang::xbg::SelectExpr> {
  using Base::Base;

  SemaResult check(Op op, Status status, xblang::SymbolTable *symTable,
                   SemaDriver &driver) const final;
};

//===----------------------------------------------------------------------===//
// ValueRefExprVerifier
//===----------------------------------------------------------------------===//
struct ValueRefExprVerifier
    : public xblang::sema::SemaOpPattern<xblang::xbg::ValueRefExpr> {
  using Base::Base;

  SemaResult check(Op op, Status status, xblang::SymbolTable *symTable,
                   SemaDriver &driver) const final;
};

//===----------------------------------------------------------------------===//
// UnaryExprVerifier
//===----------------------------------------------------------------------===//
struct UnaryExprVerifier
    : public xblang::sema::SemaOpPattern<xblang::xbg::UnaryExpr> {
  using Base::Base;

  SemaResult check(Op op, Status status, xblang::SymbolTable *symTable,
                   SemaDriver &driver) const final;
};
} // namespace

//===----------------------------------------------------------------------===//
// ArrayExprVerifier
//===----------------------------------------------------------------------===//

SemaResult ArrayExprVerifier::check(Op op, Status status,
                                    xblang::SymbolTable *symTable,
                                    SemaDriver &driver) const {
  if (SemaResult result = driver.checkOperands(op, symTable);
      !result.succeeded())
    return result;
  return checkOp(op, status, symTable, driver);
}

SemaResult ArrayExprVerifier::checkOp(Op op, Status status,
                                      xblang::SymbolTable *symTable,
                                      SemaDriver &driver) const {
  TypeAttrInterface base =
      cast<TypeAttrInterface>(op.getBase().getDefiningOp());
  Type type = base.getType();
  assert(type && "missing base type");
  auto tmp = getOrLoadValue({op.getBase(), type}, driver);
  type = tmp.type;
  op.getBaseMutable().assign(tmp.value);
  Type elemType{};
  Attribute addrSpace{};
  bool isMemRef = false;
  size_t numElems = 1;
  if (auto ptrType = dyn_cast<xb::PointerType>(type)) {
    elemType = ptrType.getPointee();
    addrSpace = ptrType.getMemorySpace();
  } else if (auto mrTy = dyn_cast<mlir::MemRefType>(type)) {
    elemType = mrTy.getElementType();
    addrSpace = mrTy.getMemorySpace();
    isMemRef = true;
    numElems = mrTy.getShape().size();
  } else
    return op.emitError("base type is not sub-scriptable");
  auto indexTy = driver.getIndexType();
  for (auto &indx : op.getIndexesMutable()) {
    auto type = cast<TypeAttrInterface>(indx.get().getDefiningOp()).getType();
    assert(type && "invalid index type");
    if (indexTy == type || (!isMemRef && type.isIntOrIndex()))
      continue;
    // Try to load the index if it's a reference.
    Operand loadedVal = getOrLoadValue({indx.get(), type}, driver);
    Value val =
        driver->makeCast(indexTy, loadedVal.type, loadedVal.value, driver);
    if (!val)
      return op.emitError("index is not convertible to int");
    indx.set(val);
  }
  if (op.getIndexes().size() != numElems) {
    return op.emitError(
        fmt("array required `{0}` indexes, but only has `{1}` indexes",
            numElems, op.getIndexes().size()));
  }
  op.setType(xb::ReferenceType::get(elemType, addrSpace));
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// BinOpExprVerifier
//===----------------------------------------------------------------------===//

SemaResult BinOpExprVerifier::check(Op op, Status status,
                                    xblang::SymbolTable *symTable,
                                    SemaDriver &driver) const {
  if (SemaResult result = driver.checkOperands(op, symTable);
      !result.succeeded())
    return result;
  return checkOp(op, status, symTable, driver);
}

SemaResult BinOpExprVerifier::handleAssign(Op op, Operand lhs, Operand rhs,
                                           SemaDriver &driver) const {
  Operand loadedVal = getOrLoadValue(rhs, driver);
  mlir::Type underlyingType = getElementType(lhs.type);
  if (!underlyingType)
    return op.emitError("LHS is not a memory reference type");
  loadedVal = {driver->makeCast(underlyingType, loadedVal, loadedVal, driver),
               underlyingType};
  if (!loadedVal)
    return op.emitError("LHS and RHS types are incompatible");
  storeValue(lhs, loadedVal, driver);
  return mlir::success();
}

SemaResult BinOpExprVerifier::checkOp(Op op, Status status,
                                      xblang::SymbolTable *symTable,
                                      SemaDriver &driver) const {
  TypeAttrInterface lhsOp =
      dyn_cast_or_null<TypeAttrInterface>(op.getLhs().getDefiningOp());
  TypeAttrInterface rhsOp =
      dyn_cast_or_null<TypeAttrInterface>(op.getRhs().getDefiningOp());
  if (!lhsOp || !rhsOp)
    return op.emitError("expressions must have valid LHS and RHS exprs");
  Operand lhs{op.getLhs(), lhsOp.getType()},
      rhs = {op.getRhs(), rhsOp.getType()};
  Type lhsType = getTypeOrElementType(lhsOp.getType());
  Type rhsType = getTypeOrElementType(rhsOp.getType());
  if (!isPrimitiveType(lhsType) || !isPrimitiveType(rhsType))
    return op.emitError("operands types are not supported");
  // Check the assignment operator.
  if (op.getOp() == BinaryOperator::Assign) {
    if (handleAssign(op, lhs, rhs, driver).failed())
      return failure();
    driver.replaceOp(op, lhs.value);
    return success();
  }
  Operand lhsMod = getOrLoadValue(lhs, driver);
  rhs = getOrLoadValue(rhs, driver);
  Type type = xblang::promotePrimitiveTypes(lhsType, rhsType);
  if (!type)
    return op.emitError("operands are incompatible");
  if (lhsOp.getType() != type) {
    Value tmp = driver->makeCast(type, lhsMod.type, lhsMod.value, driver);
    if (!tmp)
      return op.emitError("failed to promote the LHS");
    op.getLhsMutable().set(tmp);
  }
  if (rhsOp.getType() != type) {
    Value tmp = driver->makeCast(type, rhs.type, rhs.value, driver);
    if (!tmp)
      return op.emitError("failed to promote the RHS");
    op.getRhsMutable().set(tmp);
  }
  op.setType(type);
  if (isCompoundOp(op.getOp())) {
    op.setOp(removeCompound(op.getOp()));
    if (handleAssign(op, lhs, rhs, driver).failed())
      return failure();
  }
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// CallExprVerifier
//===----------------------------------------------------------------------===//

SemaResult CallExprVerifier::check(Op op, Status status,
                                   xblang::SymbolTable *symTable,
                                   SemaDriver &driver) const {
  for (auto arg : op.getArguments())
    if (SemaResult result = driver.checkValue(arg, symTable);
        !result.succeeded())
      return result;
  if (!isa<xblang::xlg::RefExpr>(
          op.getCallee().getType().getConceptClass().getConcept()))
    if (SemaResult result = driver.checkValue(op.getCallee(), nullptr);
        !result.succeeded())
      return result;
  return checkOp(op, status, symTable, driver);
}

SmallVector<mlir::Type>
CallExprVerifier::getArguments(Op op, SemaDriver &driver) const {
  SmallVector<mlir::Type> types;
  for (auto arg : op.getArguments()) {
    auto expr = driver.getInterface<xlg::ExprInterface>(arg.getDefiningOp());
    assert(expr && expr.getTypeAttr() && "expr is invalid");
    types.push_back(expr.getTypeAttr().getValue());
  }
  return types;
}

FunctionType CallExprVerifier::getFuncType(xblang::xlg::FuncDeclInterface op,
                                           SemaDriver &driver) const {
  return cast<FunctionType>(op.getTypeAttr().getValue());
}

SemaResult CallExprVerifier::checkIndirectCall(Op op,
                                               SmallVector<mlir::Type> &args,
                                               SemaDriver &driver) const {
  auto expr =
      driver.getInterface<xlg::ExprInterface>(op.getCallee().getDefiningOp());
  assert(expr && "invalid expression");
  if (auto fnTy = dyn_cast<FunctionType>(expr.getTypeAttr().getValue())) {
    if (fnTy.getResults().size() == 1)
      op.setTypeAttr(TypeAttr::get(fnTy.getResults()[0]));
    else if (fnTy.getResults().size() > 0)
      op.setTypeAttr(TypeAttr::get(driver.getTupleType(fnTy.getResults())));
    // FIXME: verify argument types.
    return mlir::success();
  } else
    return op->emitError("invalid callee");
}

SemaResult CallExprVerifier::checkDirectCall(Op op,
                                             SmallVector<mlir::Type> &args,
                                             xblang::SymbolTable *symTable,
                                             SemaDriver &driver) const {
  auto callee = driver.getInterface<xlg::SymbolRefExprInterface>(
      op.getCallee().getDefiningOp());
  assert(symTable && "invalid symbol table");
  xblang::SymbolCollection collection =
      symTable->lookup(callee.getSymNameAttr());
  if (collection.empty())
    return op.emitError("symbol couldn't be found");
  FunctionType fnTy;
  xlg::FuncDeclInterface decl;
  for (auto sym : collection) {
    auto funcDecl =
        driver.getInterface<xlg::FuncDeclInterface>(sym.getSymbol());
    FunctionType ty = getFuncType(funcDecl, driver);
    if (ty.getInputs().size() != args.size())
      continue;
    if (args != ty.getInputs())
      continue;
    if (fnTy)
      return op.emitError("call is ambiguous");
    fnTy = ty;
    decl = funcDecl;
  }
  if (!fnTy)
    return op.emitError("callee couldn't be resolved");
  if (fnTy.getResults().size() == 0)
    op.setTypeAttr(
        TypeAttr::get(xblang::xb::VoidType::get(driver.getContext())));
  else if (fnTy.getResults().size() == 1)
    op.setTypeAttr(TypeAttr::get(fnTy.getResults()[0]));
  else if (fnTy.getResults().size() > 0)
    op.setTypeAttr(TypeAttr::get(driver.getTupleType(fnTy.getResults())));
  callee.setTypeAttr(TypeAttr::get(fnTy));
  callee.setSymNameAttr(mlir::FlatSymbolRefAttr::get(decl.getUsrAttr()));
  return mlir::success();
}

SemaResult CallExprVerifier::checkOp(Op op, Status status,
                                     xblang::SymbolTable *symTable,
                                     SemaDriver &driver) const {
  SmallVector<mlir::Type> args = getArguments(op, driver);
  // Check if symbol is not an expression.
  if (!isa<xblang::xlg::RefExpr>(
          op.getCallee().getType().getConceptClass().getConcept()))
    return checkIndirectCall(op, args, driver);
  return checkDirectCall(op, args, symTable, driver);
}

//===----------------------------------------------------------------------===//
// CastExprVerifier
//===----------------------------------------------------------------------===//

SemaResult CastExprVerifier::check(Op op, Status status,
                                   xblang::SymbolTable *symTable,
                                   SemaDriver &driver) const {
  if (SemaResult result = driver.checkOperands(op, symTable);
      !result.succeeded())
    return result;
  if (op.getTypeAttr())
    return success();
  if (op.getDstType() == nullptr)
    return op.emitError("invalid cast, it is missing the destination type");
  Type type =
      cast<TypeAttrInterface>(op.getDstType().getDefiningOp()).getType();
  Type exprType =
      cast<TypeAttrInterface>(op.getExpr().getDefiningOp()).getType();
  if (type == exprType) {
    driver.replaceOp(op, op.getExpr());
    return success();
  }
  // FIXME: Once there's inheritance, this should check for reference casting.
  Operand expr = getOrLoadValue({op.getExpr(), exprType}, driver);
  if (driver->isValidPrimitiveCast(type, expr.type)) {
    op.setType(type);
    op.getDstTypeMutable().clear();
    op.getExprMutable().assign(expr);
    return success();
  }
  if (auto val = driver->makeCast(type, expr.type, expr.value, driver)) {
    driver.replaceOp(op, val);
    return success();
  } else {
    return op.emitError("invalid cast, types are not castable");
  }
  return success();
}

//===----------------------------------------------------------------------===//
// FromXLGExprVerifier
//===----------------------------------------------------------------------===//

SemaResult FromXLGExprVerifier::check(Op op, Status status,
                                      xblang::SymbolTable *symTable,
                                      SemaDriver &driver) const {
  if (SemaResult result = driver.checkOperands(op, symTable);
      !result.succeeded())
    return result;
  Type type = cast<TypeAttrInterface>(op.getExpr().getDefiningOp()).getType();
  assert(type && "invalid expr type");
  Type valType = op.getUnderlyingType();
  Value expr = op.getExpr();
  if (isa<xb::ReferenceType>(type) && !isa<xb::ReferenceType>(valType)) {
    Operand operand = getOrLoadValue({expr, type}, driver);
    type = operand.type;
    expr = operand.value;
  }
  expr = driver->makeCast(valType, type, expr, driver);
  if (!expr) {
    return op.emitError(
        "the expression type is not compatible with the final value");
  }
  op.getExprMutable().set(expr);
  return success();
}

//===----------------------------------------------------------------------===//
// RangeExprVerifier
//===----------------------------------------------------------------------===//

SemaResult RangeExprVerifier::check(Op op, Status status,
                                    xblang::SymbolTable *symTable,
                                    SemaDriver &driver) const {
  if (SemaResult result = driver.checkOperands(op, symTable);
      !result.succeeded())
    return result;
  return checkOp(op, status, symTable, driver);
}

SemaResult RangeExprVerifier::checkOp(Op op, Status status,
                                      xblang::SymbolTable *symTable,
                                      SemaDriver &driver) const {
  Type beginTy =
      cast<TypeAttrInterface>(op.getBegin().getDefiningOp()).getType();
  Type endTy = cast<TypeAttrInterface>(op.getEnd().getDefiningOp()).getType();
  Type stepTy = getUnderlyingType(op.getStep());
  assert(beginTy && endTy && "invalid types");
  Operand begin = getOrLoadValue({op.getBegin(), beginTy}, driver);
  beginTy = begin.type;
  op.getBeginMutable().set(begin);
  Operand end = getOrLoadValue({op.getEnd(), endTy}, driver);
  endTy = end.type;
  op.getEndMutable().set(end);
  if (op.getStep()) {
    Operand step = getOrLoadValue({op.getStep(), stepTy}, driver);
    stepTy = step.type;
    op.getStepMutable().assign(step);
  }
  if (!stepTy) {
    if (beginTy != endTy)
      return op.emitError("begin and end values must have the same type");
  } else {
    if ((beginTy != endTy || stepTy != endTy))
      return op.emitError("begin, step and end values must have the same type");
  }
  if (!op.getComparatorAttr())
    op.setComparator(BinaryOperator::Less);
  if (!op.getStepOpAttr())
    op.setStepOp(BinaryOperator::Add);
  op.setType(xb::RangeType::get(op.getContext(), beginTy));
  return success();
}

//===----------------------------------------------------------------------===//
// RefExprVerifier
//===----------------------------------------------------------------------===//

SemaResult RefExprVerifier::check(Op op, Status status,
                                  xblang::SymbolTable *symTable,
                                  SemaDriver &driver) const {
  if (op.getDelayedResolution()) {
    if (status.getCount() == 0)
      return SemaResult::successAndReschedule();
    if (op.getTypeAttr() == nullptr)
      return op.emitError("the symbol couldn't be resolved");
    op.setDelayedResolution(false);
    return mlir::success();
  }
  return checkOp(op, status, symTable, driver);
}

SemaResult RefExprVerifier::checkOp(Op op, Status status,
                                    xblang::SymbolTable *symTable,
                                    SemaDriver &driver) const {
  assert(symTable && "invalid symbol table");
  xblang::SymbolCollection collection = symTable->lookup(op.getSymName());
  if (collection.empty())
    return op.emitError("symbol couldn't be found");
  // Reference expressions must refer to a single symbol.
  if (collection.size() != 1) {
    op.emitError("symbol is ambiguous");
    for (auto sym : collection)
      sym.getSymbol().emitRemark("defined here");
    return failure();
  }
  // Check the symbol.
  if (SemaResult result = driver.checkOp(collection[0].getSymbol(), symTable);
      !result.succeeded())
    return result;
  // Check that the symbol refers to a value declaration.
  auto valueDecl =
      dyn_cast<TypeAttrInterface>(collection[0].getSymbol().getOperation());
  if (!valueDecl)
    return op.emitError("symbol doesn't refer to a value");
  if (auto attr = collection[0].getSymbol().getUSR()) {
    op.setSymNameAttr(mlir::FlatSymbolRefAttr::get(attr));
    op.setType(valueDecl.getType());
  } else {
    driver.replaceOpWithNewOp<ValueRefExpr>(
        op, driver.getConceptClass<ValueRefExprCep>(),
        TypeAttr::get(valueDecl.getType()),
        collection[0].getSymbol().getOperation()->getResult(0));
  }
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// SelectExprVerifier
//===----------------------------------------------------------------------===//

SemaResult SelectExprVerifier::check(Op op, Status status,
                                     xblang::SymbolTable *symTable,
                                     SemaDriver &driver) const {
  if (SemaResult result = driver.checkOperands(op, symTable);
      !result.succeeded())
    return result;
  // Check the condition in the ternary operator.
  auto condType =
      cast<TypeAttrInterface>(op.getExpr().getDefiningOp()).getType();
  auto condValue = getOrLoadValue({op.getExpr(), condType}, driver);
  auto boolType = driver.getI1Type();
  if (!driver->isValidCast(boolType, condValue.type)) {
    op.emitError("conditional is not convertible to bool");
    return SemaResult::failure();
  }
  if (boolType != condType)
    op.getExprMutable().set(
        driver->makeCast(boolType, condValue, condValue, driver));
  auto trueType =
      cast<TypeAttrInterface>(op.getTrueValue().getDefiningOp()).getType();
  auto falseType =
      cast<TypeAttrInterface>(op.getFalseValue().getDefiningOp()).getType();
  if (trueType == falseType) {
    op.setType(trueType);
    return success();
  }
  // Try to promote the types
  auto trueValue = getOrLoadValue({op.getTrueValue(), trueType}, driver);
  auto falseValue = getOrLoadValue({op.getFalseValue(), falseType}, driver);
  if (auto type = promotePrimitiveTypes(trueValue, falseValue)) {
    if (type != falseType) {
      op.getFalseValueMutable().assign(
          driver->makeCast(type, falseValue, falseValue, driver));
    }
    if (type != trueType) {
      op.getTrueValueMutable().assign(
          driver->makeCast(type, trueValue, trueValue, driver));
    }
    op.setType(type);
  } else
    return op.emitError("the true and false values have incompatible types");
  return success();
}

//===----------------------------------------------------------------------===//
// ValueRefExprVerifier
//===----------------------------------------------------------------------===//

SemaResult ValueRefExprVerifier::check(Op op, Status status,
                                       xblang::SymbolTable *symTable,
                                       SemaDriver &driver) const {
  if (SemaResult result = driver.checkOperands(op, symTable);
      !result.succeeded())
    return result;
  auto ty = cast<TypeAttrInterface>(op.getValue().getDefiningOp()).getType();
  assert(ty && "invalid type");
  op.setType(ty);
  return success();
}

//===----------------------------------------------------------------------===//
// UnaryExprVerifier
//===----------------------------------------------------------------------===//

SemaResult UnaryExprVerifier::check(Op op, Status status,
                                    xblang::SymbolTable *symTable,
                                    SemaDriver &driver) const {
  if (SemaResult result = driver.checkOperands(op, symTable);
      !result.succeeded())
    return result;
  UnaryOperator uop = op.getOp();
  TypeAttrInterface exprOp =
      dyn_cast_or_null<TypeAttrInterface>(op.getExpr().getDefiningOp());
  Type exprTy = exprOp.getType();
  if (!exprTy)
    return op.emitError("invalid operand type");
  // Check the address operator: `&expr`.
  if (uop == UnaryOperator::Address) {
    auto refTy = dyn_cast<xb::ReferenceType>(exprTy);
    if (!refTy)
      return op.emitError("operand must be of reference type");
    op.setType(
        xb::PointerType::get(refTy.getPointee(), refTy.getMemorySpace()));
    return success();
  }
  // Check the dereference operator: `*expr`.
  if (uop == UnaryOperator::Dereference) {
    auto refTy = dyn_cast<xb::PointerType>(exprTy);
    if (!refTy)
      return op.emitError("operand must be of pointer type");
    op.setType(
        xb::ReferenceType::get(refTy.getPointee(), refTy.getMemorySpace()));
    return success();
  }
  // Load the value, as it will be needed.
  Operand expr = getOrLoadValue({op.getExpr(), exprTy}, driver);
  // Check the plus and minus operators: `+expr`, `-expr`.
  if (uop == UnaryOperator::Plus || uop == UnaryOperator::Minus) {
    op.getExprMutable().set(expr.value);
    op.setType(expr.type);
    return success();
  }
  // Check the negation operator: `!expr`.
  if (uop == UnaryOperator::Negation) {
    auto val =
        driver->makeCast(driver.getI1Type(), expr.type, expr.value, driver);
    op.getExprMutable().set(val);
    op.setType(driver.getI1Type());
    return success();
  }
  // Check the increment and decrement operators: `++expr`, `--expr`.
  if (uop == UnaryOperator::Plus || uop == UnaryOperator::Minus) {
    auto refTy = dyn_cast<xb::ReferenceType>(exprTy);
    if (!refTy)
      return op.emitError("operand must be of reference type");
    //    driver.replaceOpWithNewOp<BinOpExpr>(op,
    //    driver.getConceptClass<xbg::BinOpExprCep>(), refTy,
    //    BinaryOperator::Add, expr, );
    driver.replaceOp(op, op.getExpr());
    return success();
  }
  return op.emitError("invalid unary operator");
}

//===----------------------------------------------------------------------===//
// XBG populate patterns
//===----------------------------------------------------------------------===//

void xblang::xbg::populateExprSemaPatterns(GenericPatternSet &set) {
  set.add<ArrayExprVerifier, BinOpExprVerifier, CallExprVerifier,
          CastExprVerifier, FromXLGExprVerifier, RangeExprVerifier,
          RefExprVerifier, SelectExprVerifier, ValueRefExprVerifier,
          UnaryExprVerifier>(set.getMLIRContext());
}
