//===- ProcessSyntax.cpp - Process syntax pass -------------------*- C++-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the syntax processing pass.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/CSE.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "xblang/Syntax/IR/SyntaxDialect.h"
#include "xblang/Syntax/Transforms/Passes.h"

namespace xblang {
namespace syntaxgen {
#define GEN_PASS_DEF_PROCESSSYNTAX
#include "xblang/Syntax/Transforms/Passes.h.inc"
} // namespace syntaxgen
} // namespace xblang

using namespace mlir;
using namespace xblang;
using namespace xblang::syntaxgen;

namespace {
struct ProcessSyntax
    : public xblang::syntaxgen::impl::ProcessSyntaxBase<ProcessSyntax> {
  using Base::Base;

  void runOnOperation() override;
};

struct RuleOpPattern : public OpRewritePattern<RuleOp> {
  using Base = OpRewritePattern<RuleOp>;
  using Base::OpRewritePattern;
  LogicalResult matchAndRewrite(RuleOp op,
                                PatternRewriter &rewriter) const final;
};

struct MacroOpPattern : public OpRewritePattern<MacroOp> {
  using Base = OpRewritePattern<MacroOp>;
  using Base::OpRewritePattern;
  LogicalResult matchAndRewrite(MacroOp op,
                                PatternRewriter &rewriter) const final;
};
} // namespace

LogicalResult RuleOpPattern::matchAndRewrite(RuleOp op,
                                             PatternRewriter &rewriter) const {
  auto ret = dyn_cast<ReturnOp>(op.getBody(0)->getTerminator());
  if (!ret)
    return failure();
  {
    PatternRewriter::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(ret);
    rewriter.replaceOpWithNewOp<LexRuleOp>(ret, ret.getExpr(), op.getSymName());
  }
  rewriter.inlineBlockBefore(op.getBody(0), op);
  rewriter.eraseOp(op);
  return success();
}

LogicalResult MacroOpPattern::matchAndRewrite(MacroOp op,
                                              PatternRewriter &rewriter) const {
  rewriter.eraseOp(op);
  return success();
}

void ProcessSyntax::runOnOperation() {
  {
    OpPassManager dynamicPM;
    dynamicPM.addPass(createInlinerPass());
    if (failed(runPipeline(dynamicPM, getOperation())))
      return signalPassFailure();
  }
  RewritePatternSet patterns(&getContext());
  patterns.add<MacroOpPattern>(&getContext());
  if (rulesToLex)
    patterns.add<RuleOpPattern, MacroOpPattern>(&getContext());
  if (failed(
          applyPatternsAndFoldGreedily(getOperation(), std::move(patterns)))) {
    return signalPassFailure();
  }
  {
    OpPassManager dynamicPM;
    dynamicPM.addPass(createCanonicalizerPass());
    dynamicPM.addPass(createCSEPass());
    dynamicPM.addPass(createCanonicalizerPass());
    if (failed(runPipeline(dynamicPM, getOperation())))
      return signalPassFailure();
  }
}
