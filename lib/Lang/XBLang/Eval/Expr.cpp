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

#include "xblang/Lang/XBLang/Eval/Eval.h"

#include "mlir/IR/Builders.h"
#include "xblang/Basic/Context.h"
#include "xblang/Eval/Eval.h"
#include "xblang/Eval/Utils.h"
#include "xblang/Lang/XBLang/XLG/XBGDialect.h"
#include "xblang/Lang/XBLang/XLG/XBGExpr.h"
#include "xblang/Lang/XBLang/XLG/XBGType.h"
#include "xblang/XLG/Concepts.h"
#include "xblang/XLG/Interfaces.h"

using namespace xblang;
using namespace xblang::eval;
using namespace xblang::xbg;

namespace {
//===----------------------------------------------------------------------===//
// BinOpExprEval
//===----------------------------------------------------------------------===//
struct BinOpExprEval : public OpEvalPattern<BinOpExpr> {
  using Base::Base;

  Attribute eval(Op op, ArrayRef<Attribute> args,
                 EvalDriver &driver) const final;
};

//===----------------------------------------------------------------------===//
// CastExprEval
//===----------------------------------------------------------------------===//
struct CastExprEval : public OpEvalPattern<CastExpr> {
  using Base::Base;

  Attribute eval(Op op, ArrayRef<Attribute> args,
                 EvalDriver &driver) const final;
};

//===----------------------------------------------------------------------===//
// ConstExprEval
//===----------------------------------------------------------------------===//
struct ConstExprEval : public OpEvalPattern<ConstExpr> {
  using Base::Base;

  Attribute eval(Op op, ArrayRef<Attribute> args,
                 EvalDriver &driver) const final;
};
} // namespace

//===----------------------------------------------------------------------===//
// BinOpExprEval
//===----------------------------------------------------------------------===//

Attribute CastExprEval::eval(Op op, ArrayRef<Attribute> args,
                             EvalDriver &driver) const {
  return eval::evalCast(getUnderlyingType(op.getOperation()),
                        driver.eval(op.getExpr()), driver);
}

//===----------------------------------------------------------------------===//
// BinOpExprEval
//===----------------------------------------------------------------------===//

Attribute BinOpExprEval::eval(Op op, ArrayRef<Attribute> args,
                              EvalDriver &driver) const {
  return eval::evalBinOp(op.getOp(), driver.eval(op.getLhs()),
                         driver.eval(op.getRhs()), driver, op.getLoc());
}

//===----------------------------------------------------------------------===//
// ConstExprEval
//===----------------------------------------------------------------------===//

Attribute ConstExprEval::eval(Op op, ArrayRef<Attribute> args,
                              EvalDriver &driver) const {
  return op.getExprAttr();
}

//===----------------------------------------------------------------------===//
// XBG code generation patterns
//===----------------------------------------------------------------------===//

void xblang::xbg::populateExprEvalPatterns(GenericPatternSet &patterns) {
  patterns.add<BinOpExprEval, CastExprEval, ConstExprEval>(
      patterns.getMLIRContext());
}
