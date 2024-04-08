#include "Patterns.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinDialect.h"
#include "xblang/Dialect/Parallel/IR/Dialect.h"
#include "xblang/Dialect/Parallel/Transforms/Passes.h"
#include "xblang/Dialect/XBLang/IR/Dialect.h"
#include "xblang/Dialect/XBLang/IR/XBLang.h"
#include "xblang/Dialect/XBLang/Lowering/Common.h"
#include "xblang/Dialect/XBLang/Lowering/Type.h"
#include "xblang/Dialect/XBLang/Transforms/Passes.h"

#include "Patterns.h"

using namespace mlir;
using namespace mlir::par;

namespace {
class AtomicOpLowering
    : public ParConversionPattern<AtomicOp, PatternInfo::None, BuilderBase> {
public:
  using Base::Base;
  LogicalResult
  matchAndRewrite(Op op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final;
};

class DimOpLowering
    : public ParConversionPattern<DimOp, PatternInfo::None, BuilderBase> {
public:
  using Base::Base;
  LogicalResult parSeq(Op op, OpAdaptor adaptor,
                       ConversionPatternRewriter &rewriter) const final;
  LogicalResult parMp(Op op, OpAdaptor adaptor,
                      ConversionPatternRewriter &rewriter) const final;
  LogicalResult parGPU(Op op, OpAdaptor adaptor,
                       ConversionPatternRewriter &rewriter) const final;
};

class IdOpLowering
    : public ParConversionPattern<IdOp, PatternInfo::RequiresConverter,
                                  BuilderBase> {
public:
  using Base::Base;
  LogicalResult parSeq(Op op, OpAdaptor adaptor,
                       ConversionPatternRewriter &rewriter) const final;
  LogicalResult parMp(Op op, OpAdaptor adaptor,
                      ConversionPatternRewriter &rewriter) const final;
  LogicalResult parGPU(Op op, OpAdaptor adaptor,
                       ConversionPatternRewriter &rewriter) const final;
};

class LoopOpLowering
    : public ParConversionPattern<LoopOp, PatternInfo::RequiresConverter,
                                  BuilderBase> {
public:
  using Base::Base;
  LogicalResult parMp(Op op, OpAdaptor adaptor,
                      ConversionPatternRewriter &rewriter) const final;
};

class MakeQueueOpLowering
    : public ParConversionPattern<MakeQueueOp, PatternInfo::None, BuilderBase> {
public:
  using Base::Base;
  LogicalResult
  matchAndRewrite(Op op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final;
};

class ReduceOpLowering
    : public ParConversionPattern<ReduceOp, PatternInfo::RequiresConverter,
                                  BuilderBase> {
public:
  using Base::Base;
  LogicalResult parGPU(Op op, OpAdaptor adaptor,
                       ConversionPatternRewriter &rewriter) const final;
};

class RegionLowering
    : public ParConversionPattern<RegionOp, PatternInfo::RequiresConverter,
                                  BuilderBase> {
public:
  using Base::Base;
  LogicalResult parSeq(Op op, OpAdaptor adaptor,
                       ConversionPatternRewriter &rewriter) const final;
  LogicalResult parMp(Op op, OpAdaptor adaptor,
                      ConversionPatternRewriter &rewriter) const final;
  LogicalResult parGPU(Op op, OpAdaptor adaptor,
                       ConversionPatternRewriter &rewriter) const final;
};

class SyncOpLowering
    : public ParConversionPattern<SyncOp, PatternInfo::None, BuilderBase> {
public:
  using Base::Base;
  LogicalResult
  matchAndRewrite(Op op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final;
};

class WaitOpLowering
    : public ParConversionPattern<WaitOp, PatternInfo::None, BuilderBase> {
public:
  using Base::Base;
  LogicalResult
  matchAndRewrite(Op op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final;
};
} // namespace

void mlir::par::populateLoweringPatterns(ConversionTarget &target,
                                         RewritePatternSet &patterns,
                                         TypeConverter &converter) {
  xblang::xb::AddConversionPattern::add<
      AtomicOpLowering, ReduceOpLowering, IdOpLowering, RegionLowering,
      MakeQueueOpLowering, WaitOpLowering, SyncOpLowering, DimOpLowering,
      LoopOpLowering>(*patterns.getContext(), target, patterns, converter);
  target.addLegalDialect<omp::OpenMPDialect>();
  /* DefaultQueueOpConcretization */
}

namespace {
template <arith::AtomicRMWKind IK, arith::AtomicRMWKind UIK,
          arith::AtomicRMWKind FK>
arith::AtomicRMWKind getAtomicKind(Type type) {
  if (auto intType = dyn_cast<IntegerType>(type)) {
    if (intType.isSigned())
      return IK;
    return UIK;
  } else if (isa<FloatType>(type))
    return FK;
  assert(false);
  return IK;
}

ParallelHierarchy getHighHierarchy(ParallelHierarchy rank) {
  ParallelHierarchy h{};
  for (int i = 1; i < 16; i <<= 1) {
    auto tmp = static_cast<ParallelHierarchy>(i);
    if ((tmp & rank) == tmp)
      h = tmp;
  }
  return static_cast<ParallelHierarchy>(h);
}

ParallelHierarchy getLowHierarchy(ParallelHierarchy rank) {
  ParallelHierarchy h{};
  for (int i = 8; i >= 1; i >>= 1) {
    auto tmp = static_cast<ParallelHierarchy>(i);
    if ((tmp & rank) == tmp)
      h = tmp;
  }
  return static_cast<ParallelHierarchy>(h);
}
} // namespace

//===----------------------------------------------------------------------===//
// Par atomic Op
//===----------------------------------------------------------------------===//

LogicalResult
AtomicOpLowering::matchAndRewrite(Op op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const {
  using namespace arith;
  SmallVector<Value> indices;
  Value base = xblang::xb::LoweringBuilderBase::lowerValue(
      rewriter, adaptor.getDestination());
  auto destType = dyn_cast<MemRefType>(base.getType());
  if (!destType)
    return failure();

  if (destType.hasRank() && destType.getRank() > 0) {
    base = rewriter.create<xblang::xb::CollapseMemRefOp>(
        op.getLoc(),
        MemRefType::get({}, destType.getElementType(),
                        MemRefLayoutAttrInterface{}, destType.getMemorySpace()),
        base);
  }

  AtomicRMWKind kind;
  switch (op.getOp()) {
  case AtomicOps::Add:
    kind = getAtomicKind<AtomicRMWKind::addi, AtomicRMWKind::addi,
                         AtomicRMWKind::addf>(op.getType());
    break;
  case AtomicOps::Mul:
    kind = getAtomicKind<AtomicRMWKind::muli, AtomicRMWKind::muli,
                         AtomicRMWKind::mulf>(op.getType());
    break;
  case AtomicOps::Max:
    kind = getAtomicKind<AtomicRMWKind::maxs, AtomicRMWKind::maxu,
                         AtomicRMWKind::maximumf>(op.getType());
    break;
  case AtomicOps::Min:
    kind = getAtomicKind<AtomicRMWKind::mins, AtomicRMWKind::minu,
                         AtomicRMWKind::minimumf>(op.getType());
    break;
  default:
    break;
  }
  rewriter.replaceOpWithNewOp<memref::AtomicRMWOp>(op, kind, adaptor.getValue(),
                                                   base, indices);
  return success();
}

//===----------------------------------------------------------------------===//
// Par dim Op
//===----------------------------------------------------------------------===//

LogicalResult DimOpLowering::parSeq(Op op, OpAdaptor adaptor,
                                    ConversionPatternRewriter &rewriter) const {
  rewriter.replaceOpWithNewOp<xblang::xb::ConstantOp>(
      op, rewriter.getIntegerAttr(rewriter.getIntegerType(32, true), 1));
  return success();
}

LogicalResult DimOpLowering::parMp(Op op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter) const {
  auto rank = op.getRank();
  auto base = getLowHierarchy(rank);
  auto high = getHighHierarchy(rank);
  if (rank == ParallelHierarchy::automatic) {
    base = ParallelHierarchy::scalar;
    high = ParallelHierarchy::tensor;
  }
  rank = base | high;
  if (high < ParallelHierarchy::tensor)
    rewriter.replaceOpWithNewOp<xblang::xb::ConstantOp>(
        op, rewriter.getIntegerAttr(rewriter.getIntegerType(32, true), 1));
  else
    rewriter.replaceOpWithNewOp<xblang::xb::CallOp>(
        op, TypeRange(rewriter.getIntegerType(32, true)), "omp_get_num_threads",
        ValueRange());
  return success();
}

LogicalResult DimOpLowering::parGPU(Op op, OpAdaptor adaptor,
                                    ConversionPatternRewriter &rewriter) const {
  auto rank = op.getRank();
  auto base = getLowHierarchy(rank);
  auto high = getHighHierarchy(rank);
  if (rank == ParallelHierarchy::automatic) {
    base = ParallelHierarchy::scalar;
    high = ParallelHierarchy::tensor;
  }
  rank = base | high;
  gpu::Dimension dim = gpu::Dimension::x;
  if (op.getEntryAttr())
    dim = static_cast<gpu::Dimension>(op.getEntry());
  Value value{};
  switch (rank) {
  case ParallelHierarchy::scalar:
    value = rewriter.create<index::ConstantOp>(op.getLoc(), 1);
    break;
  case ParallelHierarchy::vector:
  case ParallelHierarchy::v2s:
    value = rewriter.create<gpu::SubgroupSizeOp>(op.getLoc());
    break;
  case ParallelHierarchy::matrix:
  case ParallelHierarchy::m2v:
    value = rewriter.create<gpu::NumSubgroupsOp>(op.getLoc());
    break;
  case ParallelHierarchy::m2s:
    value = rewriter.create<gpu::BlockDimOp>(op.getLoc(), dim);
    break;
  case ParallelHierarchy::tensor:
  case ParallelHierarchy::t2m:
    value = rewriter.create<gpu::GridDimOp>(op.getLoc(), dim);
    break;
  case ParallelHierarchy::t2v: {
    value = rewriter.create<gpu::GridDimOp>(op.getLoc(), dim);
    auto tmp = rewriter.create<gpu::NumSubgroupsOp>(op.getLoc());
    value = rewriter.create<index::MulOp>(op.getLoc(), value, tmp);
    break;
  }
  case ParallelHierarchy::t2s: {
    value = rewriter.create<gpu::GridDimOp>(op.getLoc(), dim);
    auto tmp = rewriter.create<gpu::BlockDimOp>(op.getLoc(), dim);
    value = rewriter.create<index::MulOp>(op.getLoc(), value, tmp);
    break;
  }
  default:
    break;
  }
  rewriter.replaceOpWithNewOp<index::CastSOp>(op, rewriter.getI32Type(), value);
  return success();
}

//===----------------------------------------------------------------------===//
// Par id Op
//===----------------------------------------------------------------------===//

LogicalResult IdOpLowering::parSeq(Op op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter) const {
  rewriter.replaceOpWithNewOp<xblang::xb::ConstantOp>(
      op, rewriter.getIntegerAttr(rewriter.getIntegerType(32, true), 0));
  return success();
}

LogicalResult IdOpLowering::parMp(Op op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const {
  auto rank = op.getRank();
  auto base = getLowHierarchy(rank);
  auto high = getHighHierarchy(rank);
  if (rank == ParallelHierarchy::automatic) {
    base = ParallelHierarchy::scalar;
    high = ParallelHierarchy::tensor;
  }
  rank = base | high;
  if (high < ParallelHierarchy::tensor)
    rewriter.replaceOpWithNewOp<xblang::xb::ConstantOp>(
        op, rewriter.getIntegerAttr(rewriter.getIntegerType(32, true), 0));
  else
    rewriter.replaceOpWithNewOp<xblang::xb::CallOp>(
        op, TypeRange(rewriter.getIntegerType(32, true)), "omp_get_thread_num",
        ValueRange());
  return success();
}

LogicalResult IdOpLowering::parGPU(Op op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter) const {
  auto rank = op.getRank();
  auto base = getLowHierarchy(rank);
  auto high = getHighHierarchy(rank);
  if (rank == ParallelHierarchy::automatic) {
    base = ParallelHierarchy::scalar;
    high = ParallelHierarchy::tensor;
  }
  rank = base | high;
  gpu::Dimension dim = gpu::Dimension::x;
  if (op.getEntryAttr())
    dim = static_cast<gpu::Dimension>(op.getEntry());
  Value value{};
  switch (rank) {
  case ParallelHierarchy::scalar:
  case ParallelHierarchy::v2s:
    value = rewriter.create<gpu::LaneIdOp>(op.getLoc());
    break;
  case ParallelHierarchy::vector:
  case ParallelHierarchy::m2v:
    value = rewriter.create<gpu::SubgroupIdOp>(op.getLoc());
    break;
  case ParallelHierarchy::m2s:
    value = rewriter.create<gpu::ThreadIdOp>(op.getLoc(), dim);
    break;
  case ParallelHierarchy::matrix:
  case ParallelHierarchy::t2m:
    value = rewriter.create<gpu::BlockIdOp>(op.getLoc(), dim);
    break;
  case ParallelHierarchy::t2v: {
    value = rewriter.create<gpu::BlockIdOp>(op.getLoc(), dim);
    Value tmp = rewriter.create<gpu::NumSubgroupsOp>(op.getLoc());
    value = rewriter.create<index::MulOp>(op.getLoc(), value, tmp);
    tmp = rewriter.create<gpu::SubgroupIdOp>(op.getLoc());
    value = rewriter.create<index::AddOp>(op.getLoc(), value, tmp);
  } break;
  case ParallelHierarchy::t2s:
    value = rewriter.create<gpu::GlobalIdOp>(op.getLoc(), dim);
    break;
  case ParallelHierarchy::tensor:
    value = rewriter.create<index::ConstantOp>(op.getLoc(), 0);
    break;
  default:
    break;
  }
  rewriter.replaceOpWithNewOp<index::CastSOp>(op, rewriter.getI32Type(), value);
  return success();
}

//===----------------------------------------------------------------------===//
// Par loop Op
//===----------------------------------------------------------------------===//

LogicalResult LoopOpLowering::parMp(Op op, OpAdaptor adaptor,
                                    ConversionPatternRewriter &rewriter) const {
  auto wsLoop = rewriter.create<omp::WsloopOp>(
      op.getLoc(), adaptor.getBegin(), adaptor.getEnd(), adaptor.getStep());
  if (op.getProperties().getLoopInfo().front().cmpOp == LoopComparatorOp::LEQ)
    wsLoop.setInclusive(true);
  wsLoop.setNowait(true);
  auto &wsBody = wsLoop.getRegion();
  rewriter.inlineRegionBefore(op.getBody(), wsBody, wsBody.begin());
  {
    auto grd = guard(rewriter, &wsBody.front().front());
    wsBody.insertArgument(0u, convertType(wsBody.getArgument(0).getType()),
                          wsBody.getArgument(0).getLoc());
    rewriter.replaceAllUsesWith(wsBody.getArgument(1), wsBody.getArgument(0));
    wsBody.eraseArgument(1);
  }
  {
    auto terminator = wsBody.back().getTerminator();
    auto grd = guard(rewriter, terminator);
    rewriter.create<omp::YieldOp>(terminator->getLoc());
    rewriter.eraseOp(terminator);
  }
  rewriter.eraseOp(op);
  return success();
}

//===----------------------------------------------------------------------===//
// Par make_queue Op
//===----------------------------------------------------------------------===//

LogicalResult MakeQueueOpLowering::matchAndRewrite(
    Op op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const {
  if (!opts.isOffload()) {
    rewriter.replaceOpWithNewOp<xblang::xb::NullPtrOp>(op, op.getType());
    return success();
  }
  Value val = rewriter
                  .create<gpu::WaitOp>(op.getLoc(),
                                       rewriter.getType<gpu::AsyncTokenType>(),
                                       ValueRange())
                  .getResult(0);
  rewriter.replaceOpWithNewOp<UnrealizedConversionCastOp>(
      op, TypeRange({rewriter.getType<xblang::xb::AddressType>()}),
      ValueRange({val}));
  return success();
}

//===----------------------------------------------------------------------===//
// Par reduce Op
//===----------------------------------------------------------------------===//

LogicalResult
ReduceOpLowering::parGPU(Op op, OpAdaptor adaptor,
                         ConversionPatternRewriter &rewriter) const {
  auto rank =
      op.getRank() ? op.getRank().value() : ParallelHierarchy::automatic;
  if (rank == ParallelHierarchy::automatic)
    rank = ParallelHierarchy::matrix;
  gpu::AllReduceOperation opKind;
  switch (op.getOp()) {
  case ReduceOps::Add:
    opKind = gpu::AllReduceOperation::ADD;
    break;
  case ReduceOps::Mul:
    opKind = gpu::AllReduceOperation::MUL;
    break;
    // FIXME: Handle UI
  case ReduceOps::Max:
    opKind = isa<FloatType>(op.getType()) ? gpu::AllReduceOperation::MAXNUMF
                                          : gpu::AllReduceOperation::MAXSI;
    break;
  case ReduceOps::Min:
    opKind = isa<FloatType>(op.getType()) ? gpu::AllReduceOperation::MINNUMF
                                          : gpu::AllReduceOperation::MINSI;
    break;
  default:
    assert(false);
    break;
  }
  switch (rank) {
  case ParallelHierarchy::scalar:
    rewriter.replaceOp(op, adaptor.getValue());
    break;
  case ParallelHierarchy::vector:
    rewriter.replaceOpWithNewOp<gpu::SubgroupReduceOp>(op, adaptor.getValue(),
                                                       opKind, true);
    break;
  case ParallelHierarchy::matrix:
    rewriter.replaceOpWithNewOp<gpu::AllReduceOp>(
        op, adaptor.getValue(),
        gpu::AllReduceOperationAttr::get(getContext(), opKind), true);
    break;
  default:
    assert(false);
    break;
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Par region Op
//===----------------------------------------------------------------------===//

LogicalResult
RegionLowering::parSeq(Op op, OpAdaptor adaptor,
                       ConversionPatternRewriter &rewriter) const {
  auto scope = rewriter.create<xblang::xb::ScopeOp>(op.getLoc(), false);
  rewriter.inlineRegionBefore(op.getBody(), scope.getBody(),
                              scope.getBody().end());
  rewriter.eraseOp(op);
  return success();
}

LogicalResult RegionLowering::parMp(Op op, OpAdaptor adaptor,
                                    ConversionPatternRewriter &rewriter) const {
  auto region = rewriter.create<omp::ParallelOp>(op.getLoc());
  rewriter.inlineRegionBefore(op.getBody(), region.getRegion(),
                              region.getRegion().begin());
  {
    auto terminator = region.getRegion().back().getTerminator();
    auto grd = guard(rewriter, terminator);
    rewriter.create<omp::TerminatorOp>(terminator->getLoc());
    rewriter.eraseOp(terminator);
  }
  rewriter.eraseOp(op);
  return success();
}

LogicalResult
RegionLowering::parGPU(Op op, OpAdaptor adaptor,
                       ConversionPatternRewriter &rewriter) const {
  auto grd = guard(rewriter, op);
  auto one = rewriter.create<index::ConstantOp>(op.getLoc(), 1);
  SmallVector<Value, 3> grid(3, one), block(3, one);
  auto tensorParams = op.getTensorDim();
  if (tensorParams.size()) {
    for (auto [i, p] : llvm::enumerate(tensorParams))
      if (i < 3)
        grid[i] = p;
  } else
    assert(false);
  auto matrixParams = op.getMatrixDim();
  if (matrixParams.size()) {
    for (auto [i, p] : llvm::enumerate(matrixParams))
      if (i < 3)
        block[i] = p;
  } else {
    auto c0 = rewriter.create<index::ConstantOp>(op.getLoc(), 0);
    block[0] = c0;
    if (tensorParams.size() == 1)
      grid[1] = c0;
  }
  SmallVector<Value, 1> queue;
  if (op.getQueue()) {
    Value value = op.getQueue();
    value = rewriter
                .create<UnrealizedConversionCastOp>(
                    op.getQueue().getLoc(),
                    TypeRange({rewriter.getType<gpu::AsyncTokenType>()}),
                    ValueRange({value}))
                .getResult(0);
    queue.push_back(value);
  }
  auto launch = rewriter.create<gpu::LaunchOp>(
      op.getLoc(), grid[0], grid[1], grid[2], block[0], block[1], block[2],
      nullptr, rewriter.getType<gpu::AsyncTokenType>(), queue);
  auto &body = launch.getBody();
  rewriter.inlineRegionBefore(op.getBody(), body, body.getBlocks().end());
  {
    auto grd = guard(rewriter, &body.front(), body.front().end());
    rewriter.create<cf::BranchOp>(body.getLoc(), &body.back());
  }
  {
    auto terminator = body.back().getTerminator();
    auto grd = guard(rewriter, terminator);
    rewriter.create<gpu::TerminatorOp>(terminator->getLoc());
    rewriter.eraseOp(terminator);
  }
  rewriter.eraseOp(op);
  return success();
}

//===----------------------------------------------------------------------===//
// Par sync Op
//===----------------------------------------------------------------------===//

LogicalResult
SyncOpLowering::matchAndRewrite(Op op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const {
  auto rank = op.getRank();
  if (opts.isOffload()) {
    if (rank == ParallelHierarchy::automatic ||
        rank == ParallelHierarchy::matrix) {
      rewriter.replaceOpWithNewOp<gpu::BarrierOp>(op);
      return success();
    }
  } else if (opts.isHost())
    if (rank == ParallelHierarchy::automatic ||
        rank == ParallelHierarchy::tensor) {
      rewriter.replaceOpWithNewOp<omp::BarrierOp>(op);
      return success();
    }
  rewriter.eraseOp(op);
  return success();
}

//===----------------------------------------------------------------------===//
// Par wait Op
//===----------------------------------------------------------------------===//

LogicalResult
WaitOpLowering::matchAndRewrite(Op op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const {
  if (!opts.isOffload()) {
    rewriter.eraseOp(op);
    return success();
  }
  auto operands = adaptor.getOperands();
  SmallVector<Value> queues(operands.begin(), operands.end());
  if (queues.empty())
    queues.push_back(rewriter.create<xblang::xb::NullPtrOp>(
        op.getLoc(), rewriter.getType<xblang::xb::AddressType>()));
  auto destroy = rewriter.create<arith::ConstantOp>(
      op.getLoc(),
      rewriter.getIntegerAttr(rewriter.getI1Type(), op.getDestroy()));
  for (auto queue : queues)
    rewriter.create<func::CallOp>(op.getLoc(), "__xblangGpuWait", TypeRange(),
                                  ValueRange({queue, destroy}));
  rewriter.eraseOp(op);
  return success();
}
