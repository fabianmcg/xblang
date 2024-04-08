//===- Expr.cpp - XBG code gen patterns for expr constructs -----*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements XBG code generation patterns for expr constructs.
//
//===----------------------------------------------------------------------===//

#include "xblang/Lang/XBLang/Codegen/Codegen.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "xblang/Basic/Context.h"
#include "xblang/Codegen/Codegen.h"
#include "xblang/Codegen/Utils.h"
#include "xblang/Lang/XBLang/XLG/XBGDialect.h"
#include "xblang/Lang/XBLang/XLG/XBGExpr.h"
#include "xblang/Lang/XBLang/XLG/XBGType.h"
#include "xblang/XLG/Concepts.h"
#include "xblang/XLG/Interfaces.h"

#include "xblang/Dialect/XBLang/IR/XBLang.h"

using namespace xblang;
using namespace xblang::codegen;
using namespace xblang::xbg;

namespace {
//===----------------------------------------------------------------------===//
// ArrayExprCG
//===----------------------------------------------------------------------===//
struct ArrayExprCG : public OpCGPattern<ArrayExpr> {
  using Base::Base;

  CGResult generate(Op op, CGDriver &driver) const final;
};

//===----------------------------------------------------------------------===//
// BinOpExprCG
//===----------------------------------------------------------------------===//
struct BinOpExprCG : public OpCGPattern<BinOpExpr> {
  using Base::Base;

  CGResult generate(Op op, CGDriver &driver) const final;
};

//===----------------------------------------------------------------------===//
// CastExprCG
//===----------------------------------------------------------------------===//
struct CastExprCG : public OpCGPattern<CastExpr> {
  using Base::Base;

  CGResult generate(Op op, CGDriver &driver) const final;
};

//===----------------------------------------------------------------------===//
// CallExprCG
//===----------------------------------------------------------------------===//
struct CallExprCG : public OpCGPattern<CallExpr> {
  using Base::Base;

  CGResult generate(Op op, CGDriver &driver) const final;
};

//===----------------------------------------------------------------------===//
// ConstExprCG
//===----------------------------------------------------------------------===//
struct ConstExprCG : public OpCGPattern<ConstExpr> {
  using Base::Base;

  CGResult generate(Op op, CGDriver &driver) const final;
};

//===----------------------------------------------------------------------===//
// FromXLGExprCG
//===----------------------------------------------------------------------===//
struct FromXLGExprCG : public OpCGPattern<FromXLGExpr> {
  using Base::Base;

  CGResult generate(Op op, CGDriver &driver) const final;
};

//===----------------------------------------------------------------------===//
// LoadExprCG
//===----------------------------------------------------------------------===//
struct LoadExprCG : public OpCGPattern<LoadExpr> {
  using Base::Base;

  CGResult generate(Op op, CGDriver &driver) const final;
};

//===----------------------------------------------------------------------===//
// RangeExprCG
//===----------------------------------------------------------------------===//
struct RangeExprCG : public OpCGPattern<RangeExpr> {
  using Base::Base;

  CGResult generate(Op op, CGDriver &driver) const final;
};

//===----------------------------------------------------------------------===//
// RefExprCG
//===----------------------------------------------------------------------===//
struct RefExprCG : public OpCGPattern<RefExpr> {
  using Base::Base;

  CGResult generate(Op op, CGDriver &driver) const final;
};

//===----------------------------------------------------------------------===//
// StoreExprCG
//===----------------------------------------------------------------------===//
struct StoreExprCG : public OpCGPattern<StoreExpr> {
  using Base::Base;

  CGResult generate(Op op, CGDriver &driver) const final;
};

//===----------------------------------------------------------------------===//
// ToXLGExprCG
//===----------------------------------------------------------------------===//
struct ToXLGExprCG : public OpCGPattern<ToXLGExpr> {
  using Base::Base;

  CGResult generate(Op op, CGDriver &driver) const final;
};

//===----------------------------------------------------------------------===//
// ValueRefExprCG
//===----------------------------------------------------------------------===//
struct ValueRefExprCG : public OpCGPattern<ValueRefExpr> {
  using Base::Base;

  CGResult generate(Op op, CGDriver &driver) const final;
};
} // namespace

//===----------------------------------------------------------------------===//
// ArrayExprCG
//===----------------------------------------------------------------------===//

CGResult ArrayExprCG::generate(Op op, CGDriver &driver) const {
  SmallVector<Value> indexes;
  Value array = driver.genValue(op.getBase()).dyn_cast<Value>();
  assert(array && "invalid array");
  for (Value indx : op.getIndexes()) {
    auto i = driver.genValue(indx).dyn_cast<Value>();
    assert(i && "invalid index");
    indexes.push_back(i);
  }
  return driver
      .replaceOpWithNewOp<xb::ArrayOp>(op, *op.getType(), array, indexes)
      .getResult();
}

//===----------------------------------------------------------------------===//
// BinOpExprCG
//===----------------------------------------------------------------------===//

CGResult CastExprCG::generate(Op op, CGDriver &driver) const {
  auto dstType = getUnderlyingType(op.getOperation());
  auto srcType = getUnderlyingType(op.getExpr());
  auto expr = driver.genValue(op.getExpr()).dyn_cast<Value>();
  assert(dstType && "invalid destination type");
  assert(expr && "invalid expr");
  Value value = driver.makeCast(dstType, srcType, expr);
  assert(value && "invalid cast");
  driver.replaceOp(op, value);
  return value;
}

//===----------------------------------------------------------------------===//
// BinOpExprCG
//===----------------------------------------------------------------------===//

CGResult BinOpExprCG::generate(Op op, CGDriver &driver) const {
  auto lhsType = getUnderlyingType(op.getLhs());
  auto rhsType = getUnderlyingType(op.getRhs());
  auto lhs = driver.genValue(op.getLhs()).dyn_cast<Value>();
  auto rhs = driver.genValue(op.getRhs()).dyn_cast<Value>();
  assert(lhs && lhsType && "invalid lhs");
  assert(rhs && rhsType && "invalid rhs");
  assert(lhsType == rhsType && "invalid operands");
  Value value;
  driver.replaceOp(
      op, value = createArithBinOp(driver, op.getOp(), lhs, rhs, lhsType));
  return value;
}

//===----------------------------------------------------------------------===//
// CallExprCG
//===----------------------------------------------------------------------===//

CGResult CallExprCG::generate(Op op, CGDriver &driver) const {
  mlir::SymbolRefAttr symName = dyn_cast_or_null<mlir::SymbolRefAttr>(
      driver.genValue(op.getCallee()).dyn_cast<Attribute>());
  // FIXME: support non symbol name callees.
  assert(symName && "invalid callee");
  SmallVector<mlir::Type> outs;
  SmallVector<Value> ins;
  for (auto arg : op.getArguments()) {
    Value val = driver.genValue(arg).dyn_cast<Value>();
    assert(val && "invalid value");
    ins.push_back(val);
  }
  if (auto ty = op.getTypeAttr())
    outs.push_back(convertType(ty.getValue()));
  auto callOp = driver.create<xb::CallOp>(
      op.getLoc(), outs, symName.getRootReference().getValue(), ins);
  driver.eraseOp(op);
  if (callOp.getNumResults() == 1)
    return callOp.getResult()[0];
  return callOp.getOperation();
}

//===----------------------------------------------------------------------===//
// ConstExprCG
//===----------------------------------------------------------------------===//

CGResult ConstExprCG::generate(Op op, CGDriver &driver) const {
  TypedAttr expr = dyn_cast_or_null<TypedAttr>(op.getExprAttr());
  auto type = op.getTypeAttr().getValue();
  assert(expr && "invalid expression");
  assert(type && "invalid type");
  if (auto attr = dyn_cast<IntegerAttr>(expr))
    expr = IntegerAttr::get(convertType(type), attr.getValue());
  return driver.replaceOpWithNewOp<xb::ConstantOp>(op, expr).getResult();
}

//===----------------------------------------------------------------------===//
// FromXLGExprCG
//===----------------------------------------------------------------------===//

CGResult FromXLGExprCG::generate(Op op, CGDriver &driver) const {
  Value expr = driver.genValue(op.getExpr()).dyn_cast<Value>();
  driver.replaceAllUsesWith(op, expr);
  driver.eraseOp(op);
  return expr;
}

//===----------------------------------------------------------------------===//
// LoadExprCG
//===----------------------------------------------------------------------===//

CGResult LoadExprCG::generate(Op op, CGDriver &driver) const {
  auto addr = driver.genValue(op.getAddress()).dyn_cast<Value>();
  assert(addr && "invalid address");
  return driver
      .replaceOpWithNewOp<xb::LoadOp>(op, removeReference(addr.getType()), addr)
      .getResult();
}

//===----------------------------------------------------------------------===//
// RangeExprCG
//===----------------------------------------------------------------------===//

CGResult RangeExprCG::generate(Op op, CGDriver &driver) const {
  Value begin = driver.genValue(op.getBegin()).dyn_cast<Value>();
  Value end = driver.genValue(op.getEnd()).dyn_cast<Value>();
  Value step{};
  if (Value tmp = op.getStep())
    step = driver.genValue(tmp).dyn_cast<Value>();
  return driver
      .replaceOpWithNewOp<xb::RangeOp>(op, convertType(*op.getType()),
                                       *op.getComparator(), begin, end,
                                       op.getStepOpAttr(), step)
      .getResult();
}

//===----------------------------------------------------------------------===//
// RefExprCG
//===----------------------------------------------------------------------===//

CGResult RefExprCG::generate(Op op, CGDriver &driver) const {
  Attribute value = op.getSymName();
  driver.eraseOp(op);
  return value;
}

//===----------------------------------------------------------------------===//
// StoreExprCG
//===----------------------------------------------------------------------===//

CGResult StoreExprCG::generate(Op op, CGDriver &driver) const {
  auto addr = driver.genValue(op.getAddress()).dyn_cast<Value>();
  assert(addr && "invalid address");
  auto value = driver.genValue(op.getValue()).dyn_cast<Value>();
  assert(value && "invalid value");
  return driver.replaceOpWithNewOp<xb::StoreOp>(op, addr, value).getOperation();
}

//===----------------------------------------------------------------------===//
// ToXLGExprCG
//===----------------------------------------------------------------------===//

CGResult ToXLGExprCG::generate(Op op, CGDriver &driver) const {
  Value value = op.getExpr();
  driver.replaceOp(op, value);
  return value;
}

//===----------------------------------------------------------------------===//
// ValueRefExprCG
//===----------------------------------------------------------------------===//

CGResult ValueRefExprCG::generate(Op op, CGDriver &driver) const {
  Value refExpr = driver.genValue(op.getValue()).dyn_cast<Value>();
  assert(refExpr && "invalid value");
  driver.replaceOp(op, refExpr);
  return refExpr;
}

//===----------------------------------------------------------------------===//
// XBG code generation patterns
//===----------------------------------------------------------------------===//

void xblang::xbg::populateExprCGPatterns(GenericPatternSet &patterns,
                                         const mlir::TypeConverter *converter) {
  // FIXME: The benefit is set to 10 to overcome the TypeAttrInterface pattern.
  patterns.add<ArrayExprCG, BinOpExprCG, CastExprCG, CallExprCG, ConstExprCG,
               FromXLGExprCG, LoadExprCG, RangeExprCG, RefExprCG, StoreExprCG,
               ToXLGExprCG, ValueRefExprCG>(patterns.getMLIRContext(),
                                            converter, 10);
}
