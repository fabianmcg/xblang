#include "Patterns.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/RegionUtils.h"
#include "xblang/Dialect/Parallel/IR/Parallel.h"
#include "xblang/Dialect/Parallel/Transforms/Passes.h"
#include "xblang/Dialect/XBLang/IR/XBLang.h"
#include <memory>

using namespace mlir;
using namespace mlir::par;
using namespace xblang;
using namespace xblang::xb;

namespace mlir {
namespace par {
#define GEN_PASS_DEF_PARALLELCONCRETIZER
#include "xblang/Dialect/Parallel/Transforms/Passes.h.inc"
} // namespace par
} // namespace mlir

namespace {
class DefaultQueueOpConcretization
    : public ParRewritePattern<DefaultQueueOp, PatternInfo::None, BuilderBase> {
public:
  using Base::ParRewritePattern;
  LogicalResult matchAndRewrite(Op op, PatternRewriter &rewriter) const final;
};

struct LoopOpConcretization
    : public ParRewritePattern<mlir::par::LoopOp, PatternInfo::None,
                               BuilderBase> {
public:
  using Base::ParRewritePattern;
  LogicalResult parSeq(Op op, PatternRewriter &rewriter) const final;
  LogicalResult parMp(Op op, PatternRewriter &rewriter) const final;
  LogicalResult parGPU(Op op, PatternRewriter &rewriter) const final;
  static void updateConf(Value &begin, Value &step, Value &end,
                         ParallelHierarchy rank, Location loc,
                         PatternRewriter &rewriter);

  static void buildRangeFor(PatternRewriter &builder, LoopDescriptor loop,
                            Region &body, Location loc);
  static BinaryOperator cmpToBinOp(LoopComparatorOp op);
  static BinaryOperator stepToBinOp(LoopStepOp op);
};

struct LoopOpCollapsingConcretization
    : public ParRewritePattern<mlir::par::LoopOp,
                               PatternInfo::HasBoundedRecursion, BuilderBase> {
public:
  using Base::Base;
  LogicalResult matchAndRewrite(Op op, PatternRewriter &rewriter) const final;
};

class MapOpConcretizer
    : public ParRewritePattern<MapOp, PatternInfo::None, BuilderBase> {
public:
  using Base::ParRewritePattern;
  LogicalResult parGPU(Op op, PatternRewriter &rewriter) const final;
};

class RegionOpConcretizer
    : public ParRewritePattern<RegionOp, PatternInfo::None, BuilderBase,
                               XBLangTypeSystemMixin<RegionOpConcretizer>> {
public:
  using Base::Base;
  LogicalResult matchAndRewrite(Op op, PatternRewriter &rewriter) const final;
};

class ParallelConcretizer
    : public mlir::par::impl::ParallelConcretizerBase<ParallelConcretizer> {
public:
  using Base::Base;
  void runOnOperation() final;
};
} // namespace

void ParallelConcretizer::runOnOperation() {
  OpBuilder builder(&getContext());
  RewritePatternSet patterns(&getContext());
  populateConcretizationPatterns(patterns);
  FrozenRewritePatternSet patternSet(std::move(patterns));
  if (failed(mlir::applyPatternsAndFoldGreedily(getOperation(), patternSet)))
    signalPassFailure();
}

LogicalResult
DefaultQueueOpConcretization::matchAndRewrite(Op op,
                                              PatternRewriter &rewriter) const {
  auto address = AddressType::get(rewriter.getContext());
  rewriter.replaceOpWithNewOp<NullPtrOp>(op, address);
  return success();
}

BinaryOperator LoopOpConcretization::cmpToBinOp(LoopComparatorOp op) {
  switch (op) {
  case LoopComparatorOp::Greater:
    return BinaryOperator::Greater;
  case LoopComparatorOp::LEQ:
    return BinaryOperator::LEQ;
  case LoopComparatorOp::GEQ:
    return BinaryOperator::GEQ;
  default:
    return BinaryOperator::Less;
  }
}

BinaryOperator LoopOpConcretization::stepToBinOp(LoopStepOp op) {
  switch (op) {
  case LoopStepOp::Sub:
    return BinaryOperator::Sub;
  case LoopStepOp::Mul:
    return BinaryOperator::Mul;
  case LoopStepOp::Div:
    return BinaryOperator::Div;
  case LoopStepOp::LShift:
    return BinaryOperator::LShift;
  case LoopStepOp::RShift:
    return BinaryOperator::RShift;
  default:
    return BinaryOperator::Add;
  }
}

void LoopOpConcretization::buildRangeFor(PatternRewriter &rewriter,
                                         LoopDescriptor loopInfo, Region &body,
                                         Location loc) {
  using namespace ::xblang::xb;
  Type varTy = loopInfo.var.getType();
  Value var = rewriter.create<VarOp>(
      loc, rewriter.getType<ReferenceType>(varTy),
      loopInfo.info.varName.getValue(), varTy, VarKind::local, nullptr);
  Value range = rewriter.create<RangeOp>(
      loc, rewriter.getType<RangeType>(varTy), cmpToBinOp(loopInfo.info.cmpOp),
      loopInfo.begin, loopInfo.end,
      rewriter.getAttr<::xblang::BinaryOperatorAttr>(
          stepToBinOp(loopInfo.info.stepOp)),
      loopInfo.step);
  auto loop = rewriter.create<RangeForOp>(loc, var, range);
  {

    // Block *block = rewriter.createBlock(&loop.getBody(),
    // loop.getBody().end()); rewriter.inlineBlockBefore(&body.front(), block,
    // block->begin());
    rewriter.inlineRegionBefore(body, loop.getBody(), loop.getBody().begin());
    auto block = &loop.getBody().front();
    auto grd = guard(rewriter, block, block->begin());
    auto itVar =
        rewriter.create<CastOp>(loc, removeReference(var.getType()), var);
    rewriter.replaceAllUsesWith(loop.getBody().getArgument(0), itVar);
    loop.getBody().eraseArgument(0);
    if (!block->getTerminator()) {
      auto grd = guard(rewriter, block, block->end());
      rewriter.create<YieldOp>(range.getLoc(), YieldKind::Fallthrough,
                               ValueRange());
    }
  }
}

void LoopOpConcretization::updateConf(Value &begin, Value &step, Value &end,
                                      ParallelHierarchy rank, Location loc,
                                      PatternRewriter &rewriter) {
  using namespace ::xblang::xb;
  auto itType = begin.getType();
  auto intType = rewriter.getIntegerType(32, true);
  auto cast = [itType, &rewriter](Value value) {
    CastOp result = itType != removeReference(value.getType())
                        ? createCast<CastOp>(rewriter, itType, value)
                        : nullptr;
    return result ? result : value;
  };
  bool stepIsC1 = false;
  if (step)
    if (auto stepOp = dyn_cast_or_null<ConstantOp>(step.getDefiningOp()))
      if (auto intAttr = dyn_cast_or_null<IntegerAttr>(stepOp.getValue()))
        if (intAttr.getValue().getZExtValue() == 1)
          stepIsC1 = true;
  auto set = [&](ParallelHierarchy h, ParallelHierarchy l) {
    Value id = rewriter.create<IdOp>(loc, intType, h | l, 0);
    Value dim = rewriter.create<DimOp>(loc, intType, h | l, 0);
    if (step && !stepIsC1) {
      id = rewriter.create<BinaryOp>(loc, itType, ::xblang::BinaryOperator::Mul,
                                     id, step);
      dim = rewriter.create<BinaryOp>(loc, itType,
                                      ::xblang::BinaryOperator::Mul, dim, step);
    }
    begin = rewriter.create<BinaryOp>(loc, itType,
                                      ::xblang::BinaryOperator::Add, id, begin);
    step = dim;
  };
  begin = cast(begin);
  end = cast(end);
  if (step)
    step = cast(step);
  if (rank != ParallelHierarchy::scalar) {
    switch (rank) {
    case ParallelHierarchy::vector:
    case ParallelHierarchy::v2s:
      set(ParallelHierarchy::vector, ParallelHierarchy::scalar);
      break;
    case ParallelHierarchy::matrix:
    case ParallelHierarchy::m2v:
      set(ParallelHierarchy::matrix, ParallelHierarchy::vector);
      break;
    case ParallelHierarchy::m2s:
      set(ParallelHierarchy::matrix, ParallelHierarchy::scalar);
      break;
    case ParallelHierarchy::tensor:
    case ParallelHierarchy::t2m:
      set(ParallelHierarchy::tensor, ParallelHierarchy::matrix);
      break;
    case ParallelHierarchy::t2v:
      set(ParallelHierarchy::tensor, ParallelHierarchy::vector);
      break;
    case ParallelHierarchy::t2s:
      set(ParallelHierarchy::tensor, ParallelHierarchy::scalar);
      break;
    default:
      break;
    }
  }
  if (!step)
    step = rewriter.create<ConstantOp>(loc, rewriter.getIntegerAttr(itType, 1))
               .getResult();
}

LogicalResult LoopOpConcretization::parSeq(Op op,
                                           PatternRewriter &rewriter) const {
  if (op.getBody().getNumArguments() > 1)
    return failure();
  LoopDescriptor loop = op.getLoop(0);
  buildRangeFor(rewriter, loop, op.getBody(), op.getLoc());
  rewriter.eraseOp(op);
  return success();
}

LogicalResult LoopOpConcretization::parMp(Op op,
                                          PatternRewriter &rewriter) const {
  if (op.getBody().getNumArguments() > 1)
    return failure();
  LoopDescriptor loop = op.getLoop(0);
  ParallelHierarchy hierarchy =
      op.getParallelization() == ParallelHierarchy::automatic
          ? ParallelHierarchy::t2s
          : op.getParallelization();
  if ((hierarchy & ParallelHierarchy::tensor) == ParallelHierarchy::tensor)
    return failure();
  buildRangeFor(rewriter, loop, op.getBody(), op.getLoc());
  rewriter.eraseOp(op);
  return success();
}

LogicalResult LoopOpConcretization::parGPU(Op op,
                                           PatternRewriter &rewriter) const {
  if (op.getBody().getNumArguments() > 1)
    return failure();
  LoopDescriptor loop = op.getLoop(0);
  ParallelHierarchy hierarchy =
      op.getParallelization() == ParallelHierarchy::automatic
          ? ParallelHierarchy::t2s
          : op.getParallelization();
  updateConf(loop.begin, loop.step, loop.end, hierarchy, op.getLoc(), rewriter);
  buildRangeFor(rewriter, loop, op.getBody(), op.getLoc());
  rewriter.eraseOp(op);
  return success();
}

LogicalResult LoopOpCollapsingConcretization::matchAndRewrite(
    Op op, PatternRewriter &rewriter) const {
  if (op.getBody().getNumArguments() <= 1)
    return failure();
  SmallVector<LoopDescriptor> loops = op.getLoops();

  // Create the type of the iteration variable.
  Type itTy;
  for (const LoopDescriptor &loop : loops) {
    Type varTy = loop.var.getType();
    itTy = itTy ? (itTy.getIntOrFloatBitWidth() > varTy.getIntOrFloatBitWidth()
                       ? itTy
                       : varTy)
                : varTy;
  }

  // Compute the loop bounds.
  Value step = rewriter.create<ConstantOp>(op.getLoc(),
                                           rewriter.getIntegerAttr(itTy, 1));
  Value begin = rewriter.create<ConstantOp>(op.getLoc(),
                                            rewriter.getIntegerAttr(itTy, 0));
  Value end{};
  for (LoopDescriptor &loop : loops) {
    Value tripCount = rewriter.create<BinaryOp>(
        op.getLoc(), itTy, ::xblang::BinaryOperator::Sub, loop.end, loop.begin);
    end = end ? rewriter.create<BinaryOp>(op.getLoc(), itTy,
                                          ::xblang::BinaryOperator::Mul,
                                          tripCount, end)
              : tripCount;
    loop.end = tripCount;
  }

  // Update the operation.
  rewriter.modifyOpInPlace(op, [&]() {
    Region &body = op.getBody();
    rewriter.setInsertionPointToStart(&body.front());
    Value it = body.insertArgument(0u, itTy, op.getLoc());
    for (size_t i = loops.size() - 1; i > 0; --i) {
      // loop[i].var = it % loop[i].tripCount + loop[i].begin
      Value mappedIndex = rewriter.create<BinaryOp>(
          op.getLoc(), itTy, ::xblang::BinaryOperator::Mod, it, loops[i].end);
      mappedIndex = rewriter.create<BinaryOp>(op.getLoc(), itTy,
                                              ::xblang::BinaryOperator::Add,
                                              loops[i].begin, mappedIndex);
      rewriter.replaceAllUsesWith(body.getArgument(i + 1), mappedIndex);

      // it = it / loop[i].tripCount
      it = rewriter.create<BinaryOp>(
          op.getLoc(), itTy, ::xblang::BinaryOperator::Div, it, loops[i].end);
    }
    it = rewriter.create<BinaryOp>(
        op.getLoc(), itTy, ::xblang::BinaryOperator::Add, loops[0].begin, it);
    rewriter.replaceAllUsesWith(body.getArgument(1), it);
    for (size_t i = 0; i < loops.size(); ++i)
      body.eraseArgument(1);
    op->eraseOperands(0, op->getNumOperands());
    op->insertOperands(0, ValueRange({begin, end, step}));
    op.getProperties().loopInfo.clear();
    op.getProperties().loopInfo.push_back(
        LoopInfo(rewriter.getStringAttr("__it")));
  });
  return success();
}

LogicalResult MapOpConcretizer::parGPU(Op op, PatternRewriter &rewriter) const {
  auto indexType = rewriter.getIndexType();
  auto addressType = AddressType::get(op.getContext());
  auto gpuAddressType = AddressType::get(
      op.getContext(),
      gpu::AddressSpaceAttr::get(op.getContext(), gpu::AddressSpace::Global));
  auto intType = rewriter.getIntegerType(32, false);
  auto kind = op.getKind();
  auto memRef = op.getMemReference();
  SmallVector<Value, 5> args;
  Value queue = op.getQueue();
  if (!queue)
    queue = rewriter.create<DefaultQueueOp>(op.getLoc(), addressType);
  args.push_back(
      rewriter
          .create<ConstantOp>(op.getLoc(), rewriter.getIntegerAttr(
                                               intType, static_cast<int>(kind)))
          .getResult());
  if (auto refType = dyn_cast<ReferenceType>(memRef.getType())) {
    auto baseType = refType.getPointee();
    auto ptr = PointerType::get(baseType);
    auto addr =
        rewriter
            .create<UnaryOp>(op.getLoc(), ptr, UnaryOperator::Address, memRef)
            .getResult();
    args.push_back(
        rewriter.create<CastOp>(op.getLoc(), addressType, addr).getResult());
    args.push_back(rewriter.create<SizeOfOp>(op.getLoc(), indexType, baseType)
                       .getResult());
    args.push_back(
        rewriter.create<ConstantOp>(op.getLoc(), rewriter.getIndexAttr(1))
            .getResult());
    args.push_back(queue);
    rewriter.replaceOpWithNewOp<CallOp>(op, gpuAddressType, "__xblangMapData",
                                        args);
  } else if (auto tensorType = dyn_cast<TensorType>(memRef.getType())) {
    auto baseType = tensorType.getElementType();
    auto arrayView = dyn_cast<ArrayViewOp>(memRef.getDefiningOp());
    assert(arrayView);
    auto base = arrayView.getBase();
    auto addr =
        rewriter.create<CastOp>(op.getLoc(), PointerType::get(baseType), base)
            .getResult();
    args.push_back(
        rewriter.create<CastOp>(op.getLoc(), addressType, addr).getResult());
    args.push_back(rewriter.create<SizeOfOp>(op.getLoc(), indexType, baseType)
                       .getResult());
    auto range = dyn_cast<RangeOp>(arrayView.getRanges()[0].getDefiningOp());
    auto size =
        rewriter
            .create<BinaryOp>(op.getLoc(), indexType, BinaryOperator::Sub,
                              range.getEnd(), range.getBegin())
            .getResult();
    args.push_back(size);
    args.push_back(queue);
    rewriter.replaceOpWithNewOp<CallOp>(op, gpuAddressType, "__xblangMapData",
                                        args);
    if (memRef.getUses().begin() == memRef.getUses().end()) {
      rewriter.eraseOp(arrayView);
      rewriter.eraseOp(range);
    }
  }
  return success();
}

LogicalResult
RegionOpConcretizer::matchAndRewrite(Op op, PatternRewriter &rewriter) const {
  using namespace xblang::xb;
  auto insert = [](auto &c1, auto c2) { c1.insert(c2.begin(), c2.end()); };

  // There's no work to do if there are no variables to handle.
  if (op.getFirstPrivateVars().empty() && op.getPrivateVars().empty() &&
      op.getSharedVars().empty() && op.getMappedVars().empty())
    return failure();

  // If the par=sequential just remove all variables.
  if (opts.isSequential()) {
    rewriter.modifyOpInPlace(op, [&]() {
      op.getFirstPrivateVarsMutable().clear();
      op.getPrivateVarsMutable().clear();
      op.getSharedVarsMutable().clear();
      op.getMappedVarsMutable().clear();
      op.getVarMappingsMutable().clear();
    });
    return success();
  }

  SmallVector<std::pair<VarOp, Value>> privatizationVars;
  // Create mutually exclusive set of variables.
  {
    SetVector<Value> fpSet, pSet, sSet, mSet;
    insert(fpSet, op.getFirstPrivateVars());
    insert(pSet, op.getPrivateVars());
    insert(sSet, op.getSharedVars());
    insert(mSet, op.getMappedVars());
    if (opts.isHost()) {
      fpSet.set_union(mSet);
      mSet.clear();
    } else
      fpSet.set_subtract(mSet);
    sSet.set_subtract(mSet);
    sSet.set_subtract(fpSet);
    pSet.set_subtract(mSet);
    pSet.set_subtract(fpSet);
    pSet.set_subtract(sSet);
    for (Value varVal : fpSet)
      privatizationVars.push_back(
          {dyn_cast<VarOp>(varVal.getDefiningOp()),
           rewriter.create<CastOp>(varVal.getLoc(),
                                   removeReference(varVal.getType()), varVal)});
    for (Value varVal : pSet)
      privatizationVars.push_back(
          {dyn_cast<VarOp>(varVal.getDefiningOp()), nullptr});
  }

  // Create the variable privatization.
  rewriter.modifyOpInPlace(op, [&]() {
    rewriter.setInsertionPointToStart(&op.getBody().front());

    // Map variables.
    for (auto [mappedVar, mapping] :
         llvm::zip(op.getMappedVars(), op.getVarMappings())) {
      VarOp var = dyn_cast<VarOp>(mappedVar.getDefiningOp());
      PointerType ptrTy = dyn_cast<PointerType>(mapping.getType());
      assert(ptrTy && var);
      Attribute addressSpace = ptrTy.getMemorySpace();
      Type elemTy = removeReference(var.getType());
      Value init{};
      if (isPtr(elemTy))
        init = mapping;
      else
        init = rewriter.create<UnaryOp>(
            mappedVar.getLoc(), ReferenceType::get(elemTy, addressSpace),
            UnaryOperator::Dereference, mapping);
      auto map = rewriter.create<VarOp>(mappedVar.getLoc(), Ref(init.getType()),
                                        var.getSymName(), init.getType(),
                                        VarKind::local, init);
      rewriter.replaceUsesWithIf(mappedVar, map, [&op](OpOperand &operand) {
        return op->isProperAncestor(operand.getOwner());
      });
    }
    for (auto [var, init] : privatizationVars) {
      auto map = rewriter.create<VarOp>(var.getLoc(), var.getDecl().getType(),
                                        var.getSymName(), var.getType(),
                                        VarKind::local, init);
      rewriter.replaceUsesWithIf(var.getDecl(), map, [&op](OpOperand &operand) {
        return op->isProperAncestor(operand.getOwner());
      });
    }
    // Update the operands.
    op.getFirstPrivateVarsMutable().clear();
    op.getPrivateVarsMutable().clear();
    op.getSharedVarsMutable().clear();
    op.getMappedVarsMutable().clear();
    op.getVarMappingsMutable().clear();
  });
  return success();
}

void mlir::par::populateConcretizationPatterns(RewritePatternSet &patterns) {
  patterns.add<DefaultQueueOpConcretization, LoopOpConcretization,
               MapOpConcretizer, RegionOpConcretizer>(patterns.getContext());
  patterns.add<LoopOpCollapsingConcretization>(patterns.getContext(),
                                               ParOptions(), 2);
}
