#include "Patterns.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "xblang/Dialect/Parallel/IR/Dialect.h"
#include "xblang/Dialect/Parallel/IR/Parallel.h"
#include "xblang/Dialect/Parallel/Transforms/Passes.h"
#include "xblang/Dialect/XBLang/IR/Dialect.h"
#include "xblang/Dialect/XBLang/Transforms/Passes.h"
#include "llvm/Support/TargetSelect.h"
#include <memory>

using namespace mlir;
using namespace mlir::par;

namespace mlir {
namespace par {
#define GEN_PASS_DEF_GPUTRANSFORMS
#include "xblang/Dialect/Parallel/Transforms/Passes.h.inc"
} // namespace par
} // namespace mlir

namespace {
struct ReduceFactory {
  using AccumalatorFn = Value (*)(OpBuilder &, Location, Value, Value);

  ReduceFactory(OpBuilder &builder, gpu::AllReduceOp reduceOp) {
    gpu::AllReduceOperation opKind = reduceOp.getOp()
                                         ? reduceOp.getOp().value()
                                         : gpu::AllReduceOperation::ADD;
    auto reductionType = reduceOp.getType();
    bool isFloatingPoint = isa<FloatType>(reductionType);
    switch (opKind) {
    case gpu::AllReduceOperation::ADD:
      accumulator = [](OpBuilder &builder, Location loc, Value lhs,
                       Value rhs) -> Value {
        return isa<FloatType>(lhs.getType())
                   ? builder.create<arith::AddFOp>(loc, lhs, rhs).getResult()
                   : builder.create<arith::AddIOp>(loc, lhs, rhs).getResult();
      };
      atomicKind = isFloatingPoint ? arith::AtomicRMWKind::addf
                                   : arith::AtomicRMWKind::addi;
      neutral =
          isFloatingPoint
              ? static_cast<TypedAttr>(builder.getFloatAttr(reductionType, 0.))
              : static_cast<TypedAttr>(
                    builder.getIntegerAttr(reductionType, 0));
      break;
    case gpu::AllReduceOperation::MUL:
      accumulator = [](OpBuilder &builder, Location loc, Value lhs,
                       Value rhs) -> Value {
        return isa<FloatType>(lhs.getType())
                   ? builder.create<arith::MulFOp>(loc, lhs, rhs).getResult()
                   : builder.create<arith::MulIOp>(loc, lhs, rhs).getResult();
      };
      atomicKind = isFloatingPoint ? arith::AtomicRMWKind::mulf
                                   : arith::AtomicRMWKind::muli;
      neutral =
          isFloatingPoint
              ? static_cast<TypedAttr>(builder.getFloatAttr(reductionType, 1.))
              : static_cast<TypedAttr>(
                    builder.getIntegerAttr(reductionType, 1));
      break;
    case gpu::AllReduceOperation::MAXNUMF:
    case gpu::AllReduceOperation::MAXUI:
    case gpu::AllReduceOperation::MAXSI: {
      accumulator = [](OpBuilder &builder, Location loc, Value lhs,
                       Value rhs) -> Value {
        return isa<FloatType>(lhs.getType())
                   ? builder.create<arith::MaxNumFOp>(loc, lhs, rhs).getResult()
                   : builder.create<arith::MaxSIOp>(loc, lhs, rhs).getResult();
      };
      // FIXME: there's no info to know if it's signed or unsigned.
      atomicKind = isFloatingPoint ? arith::AtomicRMWKind::maximumf
                                   : arith::AtomicRMWKind::maxs;
      auto width = reductionType.getIntOrFloatBitWidth();
      double minF = -1;
      int64_t minI = -1;
      if (width == 8) {
        minI = std::numeric_limits<int8_t>::min();
      } else if (width == 16) {
        minI = std::numeric_limits<int16_t>::min();
      } else if (width == 32) {
        minI = std::numeric_limits<int32_t>::min();
        minF = std::numeric_limits<float>::lowest();
      } else if (width == 64) {
        minI = std::numeric_limits<int64_t>::min();
        minF = std::numeric_limits<float>::lowest();
      }
      neutral = isFloatingPoint ? static_cast<TypedAttr>(
                                      builder.getFloatAttr(reductionType, minF))
                                : static_cast<TypedAttr>(builder.getIntegerAttr(
                                      reductionType, minI));
      break;
    }
    case gpu::AllReduceOperation::MINNUMF:
    case gpu::AllReduceOperation::MINUI:
    case gpu::AllReduceOperation::MINSI: {
      accumulator = [](OpBuilder &builder, Location loc, Value lhs,
                       Value rhs) -> Value {
        return isa<FloatType>(lhs.getType())
                   ? builder.create<arith::MinNumFOp>(loc, lhs, rhs).getResult()
                   : builder.create<arith::MaxSIOp>(loc, lhs, rhs).getResult();
      };
      atomicKind = isFloatingPoint ? arith::AtomicRMWKind::minimumf
                                   : arith::AtomicRMWKind::mins;
      auto width = reductionType.getIntOrFloatBitWidth();
      double maxF = -1;
      int64_t maxI = -1;
      if (width == 8) {
        maxI = std::numeric_limits<int8_t>::max();
      } else if (width == 16) {
        maxI = std::numeric_limits<int16_t>::max();
      } else if (width == 32) {
        maxI = std::numeric_limits<int32_t>::max();
        maxF = std::numeric_limits<float>::max();
      } else if (width == 64) {
        maxI = std::numeric_limits<int64_t>::max();
        maxF = std::numeric_limits<double>::max();
      }
      neutral = isFloatingPoint ? static_cast<TypedAttr>(
                                      builder.getFloatAttr(reductionType, maxF))
                                : static_cast<TypedAttr>(builder.getIntegerAttr(
                                      reductionType, maxI));
      break;
    }
    default:
      llvm_unreachable("unknown GPU AllReduceOperation");
    }
    sharedMemType = MemRefType::get(
        {}, reduceOp.getType(), MemRefLayoutAttrInterface{},
        gpu::AddressSpaceAttr::get(builder.getContext(),
                                   gpu::AddressSpace::Workgroup));
  }

  Value accumulate(OpBuilder &builder, Location loc, Value lhs, Value rhs) {
    assert(accumulator);
    if (accumulator)
      return accumulator(builder, loc, lhs, rhs);
    return {};
  }

  Value getNeutral(OpBuilder &builder, Location loc) {
    assert(neutral);
    if (neutral)
      return builder.create<arith::ConstantOp>(loc, neutral);
    return {};
  }

  AccumalatorFn accumulator{};
  TypedAttr neutral{};
  MemRefType sharedMemType;
  arith::AtomicRMWKind atomicKind{};
};

struct ReduceLoweringPatten : public ParRewritePattern<gpu::GPUFuncOp> {
  using Base::Base;

  void initialize() { setHasBoundedRewriteRecursion(true); }

  LogicalResult matchAndRewrite(gpu::GPUFuncOp op,
                                PatternRewriter &rewriter) const override {
    SmallVector<gpu::AllReduceOp> reduceOps;
    auto callback = [&](gpu::AllReduceOp reduceOp) -> WalkResult {
      reduceOps.emplace_back(reduceOp);
      return WalkResult::advance();
    };
    op.walk(callback);
    if (reduceOps.empty())
      return failure();

    auto guard = PatternRewriter::InsertionGuard(rewriter);
    if (reduceOps.size()) {
      DenseMap<Block *, gpu::BarrierOp> startSync;
      for (gpu::AllReduceOp reduceOp : reduceOps) {
        auto loc = reduceOp.getLoc();
        ReduceFactory reducer(rewriter, reduceOp);
        auto sharedMem = op.addWorkgroupAttribution(reducer.sharedMemType,
                                                    reduceOp.getLoc());
        auto parentBlock = reduceOp->getBlock();
        assert(parentBlock);
        auto &barrier = startSync[parentBlock];
        Value c0{};
        {
          if (!barrier) {
            rewriter.setInsertionPointToStart(parentBlock);
            barrier = rewriter.create<gpu::BarrierOp>(op.getLoc());
          }
          rewriter.setInsertionPointToStart(parentBlock);
          c0 =
              rewriter.create<index::ConstantOp>(loc, rewriter.getIndexAttr(0));
          auto neutral = reducer.getNeutral(rewriter, loc);
          rewriter.create<memref::StoreOp>(loc, neutral, sharedMem);
        }
        rewriter.setInsertionPoint(reduceOp);
        auto value = reduceOp.getValue();

        int vectorSize = 32;
        if (opts.isAMDGPU())
          vectorSize = 64;

        for (int offset = vectorSize / 2; offset > 0; offset /= 2) {
          Value shflVal =
              rewriter
                  .create<gpu::ShuffleOp>(op.getLoc(), value, offset,
                                          vectorSize, gpu::ShuffleMode::DOWN)
                  .getResult(0);
          value = reducer.accumulate(rewriter, loc, value, shflVal);
        }
        auto lane = rewriter.create<gpu::LaneIdOp>(loc);
        auto cond = rewriter.create<index::CmpOp>(
            loc, index::IndexCmpPredicate::EQ, lane, c0);
        auto ifOp = rewriter.create<scf::IfOp>(loc, cond, false);
        {
          auto guard = PatternRewriter::InsertionGuard(rewriter);
          auto &block = ifOp.getThenRegion().front();
          rewriter.setInsertionPoint(&block, block.begin());
          rewriter.create<memref::AtomicRMWOp>(loc, reducer.atomicKind, value,
                                               sharedMem, ValueRange());
        }
        rewriter.create<gpu::BarrierOp>(loc);
        rewriter.replaceOpWithNewOp<memref::LoadOp>(reduceOp, sharedMem,
                                                    ValueRange());
      }
    }
    return success();
  }
};

struct ShflLoweringPatten : public OpRewritePattern<gpu::ShuffleOp> {
  using OpRewritePattern<gpu::ShuffleOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(gpu::ShuffleOp op,
                                PatternRewriter &rewriter) const override {
    auto type = op.getValue().getType();
    if (type.getIntOrFloatBitWidth() != 32)
      return failure();
    std::string fn = "__shfl";
    switch (op.getMode()) {
    case gpu::ShuffleMode::DOWN:
      fn += "_down";
      break;
    case gpu::ShuffleMode::UP:
      fn += "_up";
      break;
    case gpu::ShuffleMode::XOR:
      fn += "_xor";
      break;
    default:
      assert(false);
      break;
    }
    auto i32 = rewriter.getI32Type();
    Value value = op.getValue();
    if (isa<FloatType>(type))
      value = rewriter.create<arith::BitcastOp>(op.getLoc(), i32, value);
    ValueRange operands({value, op.getOffset(), op.getWidth()});
    auto call = rewriter.create<func::CallOp>(op.getLoc(), fn, TypeRange({i32}),
                                              operands);
    value = call.getResult(0);
    if (isa<FloatType>(type))
      value = rewriter.create<arith::BitcastOp>(op.getLoc(), type, value);
    Value validity = rewriter.create<arith::ConstantOp>(
        op.getLoc(), rewriter.getIntegerAttr(rewriter.getI1Type(), 1));
    rewriter.replaceOp(op, ValueRange({value, validity}));
    return success();
  }
};

struct LaneIdLoweringPatten : public OpRewritePattern<gpu::LaneIdOp> {
  using OpRewritePattern<gpu::LaneIdOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(gpu::LaneIdOp op,
                                PatternRewriter &rewriter) const override {
    auto i32 = rewriter.getI32Type();
    auto call = rewriter.create<func::CallOp>(op.getLoc(), "__lane_id",
                                              TypeRange({i32}), ValueRange());
    Value value = call.getResult(0);
    rewriter.replaceOpWithNewOp<index::CastUOp>(op, rewriter.getIndexType(),
                                                value);
    return success();
  }
};

struct AMDUnsafeAtomicsPattern
    : public OpInterfaceRewritePattern<FunctionOpInterface> {
  using OpInterfaceRewritePattern<
      FunctionOpInterface>::OpInterfaceRewritePattern;

  LogicalResult matchAndRewrite(FunctionOpInterface op,
                                PatternRewriter &rewriter) const override {
    if (op.isExternal())
      return failure();
    Attribute rawPassthrough = op->getAttr("passthrough");
    SmallVector<Attribute> attributes;
    if (rawPassthrough) {
      if (auto passthrough = dyn_cast<ArrayAttr>(rawPassthrough)) {
        for (auto attr : passthrough)
          if (auto arrayAttr = dyn_cast<ArrayAttr>(attr))
            if (auto name = dyn_cast<StringAttr>(arrayAttr[0]))
              if (name.getValue() == "amdgpu-unsafe-fp-atomics")
                return failure();
        attributes = SmallVector<Attribute>(passthrough.getValue());
      } else {
        assert(false && "Attribute must be an ArrayAttr");
      }
    }
    attributes.push_back(
        rewriter.getStrArrayAttr({"amdgpu-unsafe-fp-atomics", "true"}));
    rewriter.modifyOpInPlace(op, [&op, &attributes, &rewriter]() {
      op->setAttr("passthrough", rewriter.getArrayAttr(attributes));
    });
    return success();
  }
};

struct AtomicRWPattern : public OpRewritePattern<memref::AtomicRMWOp> {
  using OpRewritePattern<memref::AtomicRMWOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::AtomicRMWOp op,
                                PatternRewriter &rewriter) const override {
    auto type = op.getValue().getType();
    if (!isa<FloatType>(type) || op.getKind() != arith::AtomicRMWKind::addf)
      return failure();
    Attribute addressSpace = op.getMemRefType().getMemorySpace();

    std::string name = type.isF32() ? "__atomic_add_f32" : "__atomic_add_f64";

    Value ptr =
        rewriter
            .create<xblang::xb::CastOp>(
                op.getLoc(), xblang::xb::PointerType::get(type, addressSpace),
                op.getMemref(), true)
            .getResult();
    ptr = rewriter
              .create<xblang::xb::CastOp>(
                  op.getLoc(), xblang::xb::PointerType::get(type), ptr, true)
              .getResult();
    rewriter.replaceOpWithNewOp<func::CallOp>(op, name, TypeRange({type}),
                                              ValueRange({ptr, op.getValue()}));
    return success();
  }
};

struct AllocaToEntryPattern : public OpRewritePattern<memref::AllocaOp> {
  using OpRewritePattern<memref::AllocaOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::AllocaOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.getType().hasStaticShape())
      return failure();
    auto func = op->getParentOfType<FunctionOpInterface>();
    Block *entryBlock = &func.getFunctionBody().front();
    if (rewriter.getBlock() == entryBlock)
      return failure();
    auto &firstOp = entryBlock->front();
    rewriter.modifyOpInPlace(op, [&]() { op->moveAfter(&firstOp); });
    return success();
  }
};

struct AMDPrivatePattern : public OpRewritePattern<memref::AllocaOp> {
  using OpRewritePattern<memref::AllocaOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::AllocaOp op,
                                PatternRewriter &rewriter) const override {
    auto type = op.getType();
    if (!op.getType().hasStaticShape())
      return failure();
    if (type.getRank() <= 0)
      return failure();
    if (type.getMemorySpace())
      return failure();
    auto as =
        gpu::AddressSpaceAttr::get(getContext(), gpu::AddressSpace::Private);
    auto alloca = rewriter.create<memref::AllocaOp>(
        op.getLoc(), MemRefType::get(type.getShape(), type.getElementType(),
                                     type.getLayout(), as));
    rewriter.replaceOpWithNewOp<memref::MemorySpaceCastOp>(op, type, alloca);
    return success();
  }
};

struct GPUTransforms
    : public mlir::par::impl::GPUTransformsBase<GPUTransforms> {
  using mlir::par::impl::GPUTransformsBase<GPUTransforms>::GPUTransformsBase;

  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    populateGpuRewritePatterns(patterns);
    patterns.add<ReduceLoweringPatten>(&getContext(), opts, 10);
    auto module = getOperation();
    if (opts.isAMDGPU()) {
      OpBuilder builder(module.getBody(0), module.getBody(0)->begin());
      auto i32 = builder.getI32Type();
      TypeRange results({i32});
      TypeRange operands({i32, i32, i32});
      builder
          .create<func::FuncOp>(module.getLoc(), "__shfl_down",
                                builder.getFunctionType(operands, results))
          .setVisibility(SymbolTable::Visibility::Private);
      builder
          .create<func::FuncOp>(module.getLoc(), "__shfl_up",
                                builder.getFunctionType(operands, results))
          .setVisibility(SymbolTable::Visibility::Private);
      builder
          .create<func::FuncOp>(module.getLoc(), "__shfl_xor",
                                builder.getFunctionType(operands, results))
          .setVisibility(SymbolTable::Visibility::Private);
      builder
          .create<func::FuncOp>(module.getLoc(), "__lane_id",
                                builder.getFunctionType(TypeRange(), results))
          .setVisibility(SymbolTable::Visibility::Private);

      auto f32 = builder.getF32Type();
      auto f64 = builder.getF64Type();
      results = TypeRange({f32});
      builder
          .create<func::FuncOp>(
              module.getLoc(), "__atomic_add_f32",
              builder.getFunctionType(
                  TypeRange({xblang::xb::PointerType::get(f32), f32}), results))
          .setVisibility(SymbolTable::Visibility::Private);
      results = TypeRange({f64});
      builder
          .create<func::FuncOp>(
              module.getLoc(), "__atomic_add_f64",
              builder.getFunctionType(
                  TypeRange({xblang::xb::PointerType::get(f64), f64}), results))
          .setVisibility(SymbolTable::Visibility::Private);
      patterns.add<ShflLoweringPatten>(&getContext());
      patterns.add<LaneIdLoweringPatten>(&getContext());
      patterns.add<AMDUnsafeAtomicsPattern>(&getContext());
      patterns.add<AtomicRWPattern>(&getContext());
      patterns.add<AMDPrivatePattern>(&getContext());
    }
    patterns.add<AllocaToEntryPattern>(&getContext(), 10);
    FrozenRewritePatternSet patternSet(std::move(patterns));
    auto result =
        mlir::applyPatternsAndFoldGreedily(getOperation(), patternSet);
    if (result.failed())
      signalPassFailure();
  }
};
} // namespace
