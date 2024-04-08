//===- Stmt.cpp - XBG code gen patterns for stmt constructs -----*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements XBG code generation patterns for stmt constructs.
//
//===----------------------------------------------------------------------===//

#include "xblang/Lang/XBLang/Codegen/Codegen.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "xblang/Basic/Context.h"
#include "xblang/Codegen/Codegen.h"
#include "xblang/Lang/XBLang/XLG/XBGDialect.h"
#include "xblang/Lang/XBLang/XLG/XBGStmt.h"
#include "xblang/Lang/XBLang/XLG/XBGType.h"
#include "xblang/XLG/Concepts.h"
#include "xblang/XLG/Interfaces.h"

#include "xblang/Dialect/XBLang/IR/XBLang.h"

using namespace xblang;
using namespace xblang::codegen;
using namespace xblang::xbg;

namespace {
//===----------------------------------------------------------------------===//
// CompoundStmtCG
//===----------------------------------------------------------------------===//
struct CompoundStmtCG : public OpCGPattern<CompoundStmt> {
  using Base::Base;

  CGResult generate(Op op, CGDriver &driver) const final;
};

//===----------------------------------------------------------------------===//
// IfStmtCG
//===----------------------------------------------------------------------===//
struct IfStmtCG : public OpCGPattern<IfStmt> {
  using Base::Base;

  CGResult generate(Op op, CGDriver &driver) const final;
};

//===----------------------------------------------------------------------===//
// RangeForStmtCG
//===----------------------------------------------------------------------===//
struct RangeForStmtCG : public OpCGPattern<RangeForStmt> {
  using Base::Base;

  CGResult generate(Op op, CGDriver &driver) const final;
};

//===----------------------------------------------------------------------===//
// ReturnStmtCG
//===----------------------------------------------------------------------===//
struct ReturnStmtCG : public OpCGPattern<ReturnStmt> {
  using Base::Base;

  CGResult generate(Op op, CGDriver &driver) const final;
};
} // namespace

//===----------------------------------------------------------------------===//
// CompoundStmtCG
//===----------------------------------------------------------------------===//

CGResult CompoundStmtCG::generate(Op op, CGDriver &driver) const {
  if (op.getRegion().empty()) {
    driver.eraseOp(op);
    return nullptr;
  }
  auto scopeOp = driver.create<xb::ScopeOp>(op.getLoc(), false);
  driver.inlineRegionBefore(op.getBodyRegion(), scopeOp.getBody(),
                            scopeOp.getBody().begin());
  driver.eraseOp(op);
  Operation *lastOp = scopeOp.getBackBlock()->empty()
                          ? nullptr
                          : &scopeOp.getBackBlock()->back();
  if (!lastOp) {
    driver.setInsertionPointToStart(scopeOp.getBackBlock());
    driver.create<xb::YieldOp>(op.getLoc(), xb::YieldKind::Fallthrough,
                               ValueRange());
    return scopeOp.getOperation();
  }
  driver.setInsertionPointAfter(lastOp);
  if (!isa<xlg::ControlFlowStmtInterface>(lastOp) &&
      !lastOp->hasTrait<mlir::OpTrait::IsTerminator>())
    driver.create<xb::YieldOp>(lastOp->getLoc(), xb::YieldKind::Fallthrough,
                               ValueRange());
  return scopeOp.getOperation();
}

//===----------------------------------------------------------------------===//
// IfStmtCG
//===----------------------------------------------------------------------===//

CGResult IfStmtCG::generate(Op op, CGDriver &driver) const {
  auto condition = driver.genValue(op.getCondition()).dyn_cast<Value>();
  assert(condition && "invalid condition");
  auto ifOp = driver.create<xb::IfOp>(op.getLoc(), condition);
  driver.inlineRegionBefore(op.getThenRegion(), ifOp.getThenRegion(),
                            ifOp.getThenRegion().begin());

  Operation *lastOp = &ifOp.getThenRegion().back().back();
  driver.setInsertionPointAfter(lastOp);
  if (!isa<xlg::ControlFlowStmtInterface>(lastOp) &&
      !lastOp->hasTrait<mlir::OpTrait::IsTerminator>())
    driver.create<xb::YieldOp>(lastOp->getLoc(), xb::YieldKind::Fallthrough,
                               ValueRange());

  if (!op.getElseRegion().empty()) {
    driver.inlineRegionBefore(op.getElseRegion(), ifOp.getElseRegion(),
                              ifOp.getElseRegion().begin());
    lastOp = &ifOp.getElseRegion().back().back();
    driver.setInsertionPointAfter(lastOp);
    if (!isa<xlg::ControlFlowStmtInterface>(lastOp) &&
        !lastOp->hasTrait<mlir::OpTrait::IsTerminator>())
      driver.create<xb::YieldOp>(lastOp->getLoc(), xb::YieldKind::Fallthrough,
                                 ValueRange());
  }
  driver.eraseOp(op);
  return ifOp.getOperation();
}

//===----------------------------------------------------------------------===//
// RangeForStmtCG
//===----------------------------------------------------------------------===//

CGResult RangeForStmtCG::generate(Op op, CGDriver &driver) const {
  SmallVector<Value> ranges, iterators;
  for (auto range : op.getRanges())
    ranges.push_back(driver.genValue(range).dyn_cast<Value>());
  for (auto it : op.getIterators())
    iterators.push_back(driver.genValue(it).dyn_cast<Value>());
  auto forOp = driver.create<xb::RangeForOp>(op.getLoc(), iterators, ranges);
  driver.inlineRegionBefore(op.getBodyRegion(), forOp.getBody(),
                            forOp.getBody().begin());
  {
    Operation *lastOp = &forOp.getBody().back().back();
    driver.setInsertionPointAfter(lastOp);
    if (!isa<xlg::ControlFlowStmtInterface>(lastOp) &&
        !lastOp->hasTrait<mlir::OpTrait::IsTerminator>())
      driver.create<xb::YieldOp>(lastOp->getLoc(), xb::YieldKind::Fallthrough,
                                 ValueRange());
  }
  driver.eraseOp(op);
  return forOp.getOperation();
}

//===----------------------------------------------------------------------===//
// ReturnStmtCG
//===----------------------------------------------------------------------===//

CGResult ReturnStmtCG::generate(Op op, CGDriver &driver) const {
  return driver.replaceOpWithNewOp<xb::ReturnOp>(op).getOperation();
}

//===----------------------------------------------------------------------===//
// XBG code generation patterns
//===----------------------------------------------------------------------===//

void xblang::xbg::populateStmtCGPatterns(GenericPatternSet &patterns,
                                         const mlir::TypeConverter *converter) {
  patterns.add<CompoundStmtCG, IfStmtCG, RangeForStmtCG, ReturnStmtCG>(
      patterns.getMLIRContext());
}
