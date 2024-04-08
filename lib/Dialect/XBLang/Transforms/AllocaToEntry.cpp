#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "xblang/Dialect/XBLang/Transforms/Passes.h"

using namespace mlir;

namespace xblang {
namespace xb {
#define GEN_PASS_DEF_ALLOCATOENTRY
#include "xblang/Dialect/XBLang/Transforms/Passes.h.inc"
} // namespace xb
} // namespace xblang

using namespace xblang::xb;

namespace {
class AllocaToEntry
    : public xblang::xb::impl::AllocaToEntryBase<AllocaToEntry> {
public:
  using Base::Base;

  void runOnOperation() final;
};

struct AllocaToEntryPattern : public OpRewritePattern<memref::AllocaOp> {
  using OpRewritePattern<memref::AllocaOp>::OpRewritePattern;

  Block *getBlock(memref::AllocaOp op) const;

  LogicalResult matchAndRewrite(memref::AllocaOp op,
                                PatternRewriter &rewriter) const override;
};
} // namespace

Block *AllocaToEntryPattern::getBlock(memref::AllocaOp op) const {
  if (omp::ParallelOp parallel = op->getParentOfType<omp::ParallelOp>())
    return &parallel.getRegion().front();
  else if (gpu::LaunchOp launch = op->getParentOfType<gpu::LaunchOp>())
    return &launch.getRegion().front();
  else if (auto func = op->getParentOfType<FunctionOpInterface>())
    return &func.getFunctionBody().front();
  llvm_unreachable("Invalid parent ops.");
  return nullptr;
}

LogicalResult
AllocaToEntryPattern::matchAndRewrite(memref::AllocaOp op,
                                      PatternRewriter &rewriter) const {
  if (!op.getType().hasStaticShape())
    return failure();
  Block *entryBlock = getBlock(op);
  if (rewriter.getBlock() == entryBlock || !entryBlock)
    return failure();
  auto &firstOp = entryBlock->front();
  rewriter.modifyOpInPlace(op, [&]() { op->moveBefore(&firstOp); });
  return success();
}

void AllocaToEntry::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  patterns.add<AllocaToEntryPattern>(&getContext());
  FrozenRewritePatternSet patternSet(std::move(patterns));
  auto result = mlir::applyPatternsAndFoldGreedily(getOperation(), patternSet);
  if (result.failed())
    signalPassFailure();
}
