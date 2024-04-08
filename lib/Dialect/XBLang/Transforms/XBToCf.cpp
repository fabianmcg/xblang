#include "xblang/Dialect/XBLang/Transforms/Passes.h"

#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/IR/BuiltinDialect.h"
#include "xblang/ADT/DoubleTypeSwitch.h"
#include "xblang/Dialect/XBLang/IR/XBLang.h"
#include "xblang/Dialect/XBLang/Lowering/Common.h"
#include "xblang/Dialect/XBLang/Lowering/Type.h"

#include <stack>

using namespace mlir;
using namespace xblang::xb;

namespace {
template <typename Target>
struct LoweringPattern : public OpConversionPattern<Target>,
                         BuilderBase,
                         TypeSystemBase {
  using Base = LoweringPattern;
  using Op = Target;
  using OpAdaptor = typename OpConversionPattern<Target>::OpAdaptor;

  LoweringPattern(const TypeConverter &typeConverter, MLIRContext *context,
                  PatternBenefit benefit = 1)
      : OpConversionPattern<Target>(typeConverter, context, benefit) {}

  Type convertType(Type type) const {
    if (auto converter = this->getTypeConverter())
      return converter->convertType(type);
    llvm_unreachable("The pattern should hold a valid type converter.");
    return nullptr;
  }
};

//===----------------------------------------------------------------------===//
// Conversion patterns.
//===----------------------------------------------------------------------===//
struct IfOpLowering : public LoweringPattern<IfOp> {
  using Base::Base;
  LogicalResult
  matchAndRewrite(Op op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final;
};

struct LoopOpLowering : public LoweringPattern<LoopOp> {
  using Base::Base;
  void replaceLoopControlOps(Op op, ConversionPatternRewriter &rewriter,
                             Block *iteration, Block *end) const;
  LogicalResult
  matchAndRewrite(Op op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final;
};

struct ScopeOpLowering : public LoweringPattern<ScopeOp> {
  using Base::Base;
  LogicalResult
  matchAndRewrite(Op op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final;
};
} // namespace

//===----------------------------------------------------------------------===//
// XB if Op conversion
//===----------------------------------------------------------------------===//

LogicalResult
IfOpLowering::matchAndRewrite(Op op, OpAdaptor adaptor,
                              ConversionPatternRewriter &rewriter) const {
  Region &thenRegion = op.getThenRegion();
  Region &elseRegion = op.getElseRegion();
  Block *parentBlock = rewriter.getInsertionBlock();
  Block *thenBlock = &thenRegion.front(), *elseBlock{};
  Block *endBlock =
      rewriter.splitBlock(parentBlock, rewriter.getInsertionPoint());
  bool hasElse = !elseRegion.empty();
  {
    auto block = &thenRegion.back();
    auto terminator = llvm::dyn_cast<YieldOp>(block->getTerminator());
    if (terminator && terminator.getKind() == YieldKind::Fallthrough) {
      auto grd = guard(rewriter, block, block->end());
      rewriter.create<cf::BranchOp>(terminator.getLoc(), endBlock);
      rewriter.eraseOp(terminator);
    }
    rewriter.inlineRegionBefore(thenRegion, endBlock);
  }
  if (hasElse) {
    elseBlock = &elseRegion.front();
    auto block = &elseRegion.back();
    auto terminator = llvm::dyn_cast<YieldOp>(block->getTerminator());
    if (terminator && terminator.getKind() == YieldKind::Fallthrough) {
      auto grd = guard(rewriter, block, block->end());
      rewriter.create<cf::BranchOp>(terminator.getLoc(), endBlock);
      rewriter.eraseOp(terminator);
    }
    rewriter.inlineRegionBefore(elseRegion, endBlock);
  }
  {
    auto grd = guard(rewriter, parentBlock, parentBlock->end());
    if (elseBlock)
      rewriter.create<cf::CondBranchOp>(op.getLoc(), op.getCondition(),
                                        thenBlock, elseBlock);
    else
      rewriter.create<cf::CondBranchOp>(op.getLoc(), op.getCondition(),
                                        thenBlock, endBlock);
  }
  rewriter.eraseOp(op);
  return success();
}

//===----------------------------------------------------------------------===//
// XB loop Op conversion
//===----------------------------------------------------------------------===//

void LoopOpLowering::replaceLoopControlOps(Op op,
                                           ConversionPatternRewriter &rewriter,
                                           Block *iteration, Block *end) const {
  Region *region = &op.getBody();
  std::stack<Region *> regions({region});
  while (regions.size()) {
    region = regions.top();
    regions.pop();
    for (Block &block : *region) {
      for (Operation &op : block) {
        if (isa<LoopOp>(&op))
          continue;
        if (auto yieldOp = dyn_cast<YieldOp>(&op)) {
          auto grd = guard(rewriter, yieldOp);
          if (yieldOp.getKind() == YieldKind::Continue) {
            rewriter.create<cf::BranchOp>(yieldOp.getLoc(), iteration);
            rewriter.eraseOp(yieldOp);
          } else if (yieldOp.getKind() == YieldKind::Break) {
            rewriter.create<cf::BranchOp>(yieldOp.getLoc(), end);
            rewriter.eraseOp(yieldOp);
          }
        }
        for (auto &region : op.getRegions())
          regions.push(&region);
      }
    }
  }
}

LogicalResult
LoopOpLowering::matchAndRewrite(Op op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const {
  Block *parentBlock = rewriter.getInsertionBlock();
  auto &conditionRegion = op.getCondition();
  auto conditionBlock = &conditionRegion.front();
  auto &bodyRegion = op.getBody();
  auto bodyBlock = &bodyRegion.front();
  auto &iterationRegion = op.getIteration();
  auto iterationBlock = &iterationRegion.front();
  Block *endBlock =
      rewriter.splitBlock(parentBlock, rewriter.getInsertionPoint());
  replaceLoopControlOps(op, rewriter, iterationBlock, endBlock);
  {
    auto block = &conditionRegion.back();
    auto terminator = llvm::dyn_cast<YieldOp>(block->getTerminator());
    if (terminator && terminator.getKind() == YieldKind::Fallthrough) {
      auto grd = guard(rewriter, block, block->end());
      assert(terminator.getArguments().size() == 1);
      Value condition = terminator.getArguments()[0];
      rewriter.create<cf::CondBranchOp>(terminator.getLoc(), condition,
                                        bodyBlock, endBlock);
      rewriter.eraseOp(terminator);
    }
    rewriter.inlineRegionBefore(conditionRegion, endBlock);
  }
  {
    auto block = &bodyRegion.back();
    auto terminator = llvm::dyn_cast<YieldOp>(block->getTerminator());
    if (terminator && terminator.getKind() == YieldKind::Fallthrough) {
      auto grd = guard(rewriter, block, block->end());
      rewriter.create<cf::BranchOp>(terminator.getLoc(), iterationBlock);
      rewriter.eraseOp(terminator);
    }
    rewriter.inlineRegionBefore(bodyRegion, endBlock);
  }
  {
    auto block = &iterationRegion.back();
    auto terminator = llvm::dyn_cast<YieldOp>(block->getTerminator());
    if (terminator && terminator.getKind() == YieldKind::Fallthrough) {
      auto grd = guard(rewriter, block, block->end());
      rewriter.create<cf::BranchOp>(terminator.getLoc(), conditionBlock);
      rewriter.eraseOp(terminator);
    }
    rewriter.inlineRegionBefore(iterationRegion, endBlock);
  }
  {
    auto grd = guard(rewriter, parentBlock, parentBlock->end());
    rewriter.create<cf::BranchOp>(op.getLoc(), conditionBlock);
  }

  rewriter.eraseOp(op);
  return success();
}

//===----------------------------------------------------------------------===//
// XB scope Op conversion
//===----------------------------------------------------------------------===//

LogicalResult
ScopeOpLowering::matchAndRewrite(Op op, OpAdaptor adaptor,
                                 ConversionPatternRewriter &rewriter) const {
  Region &body = op.getBody();
  Block *parentBlock = rewriter.getInsertionBlock();
  Block *endBlock =
      rewriter.splitBlock(parentBlock, rewriter.getInsertionPoint());
  {
    auto grd = guard(rewriter, parentBlock, parentBlock->end());
    rewriter.create<cf::BranchOp>(op.getLoc(), &body.front());
  }
  auto terminator = llvm::dyn_cast<YieldOp>(body.back().getTerminator());
  if (terminator && terminator.getKind() == YieldKind::Fallthrough) {
    auto block = &body.back();
    auto grd = guard(rewriter, block, block->end());
    rewriter.create<cf::BranchOp>(terminator.getLoc(), endBlock);
    rewriter.eraseOp(terminator);
  }
  rewriter.inlineRegionBefore(body, endBlock);
  rewriter.eraseOp(op);
  return success();
}

//===----------------------------------------------------------------------===//
// XB to CF patterns.
//===----------------------------------------------------------------------===//

void xblang::xb::populateXBLangToCF(TypeConverter &typeConverter,
                                    RewritePatternSet &patterns) {
  patterns.add<IfOpLowering, LoopOpLowering, ScopeOpLowering>(
      typeConverter, patterns.getContext());
}
