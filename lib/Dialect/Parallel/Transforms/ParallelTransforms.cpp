#include "xblang/Dialect/Parallel/Transforms/Passes.h"

#include "Patterns.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "xblang/Dialect/Parallel/IR/Parallel.h"
#include "xblang/Dialect/XBLang/Analysis/ScopeAnalysis.h"
#include "xblang/Dialect/XBLang/IR/XBLang.h"
#include "xblang/Support/CompareExtras.h"
#include <memory>

using namespace mlir;
using namespace mlir::par;

namespace mlir {
namespace par {
#define GEN_PASS_DEF_PARALLELRUNTIME
#include "xblang/Dialect/Parallel/Transforms/Passes.h.inc"

#define GEN_PASS_DEF_PARALLELTRANSFORMS
#include "xblang/Dialect/Parallel/Transforms/Passes.h.inc"
} // namespace par
} // namespace mlir

namespace {
using ScopeAnalysis = xblang::xb::SimpleScopedValueUseAnalysis;

class ParallelRuntime
    : public mlir::par::impl::ParallelRuntimeBase<ParallelRuntime> {
public:
  using Base::Base;

  void runOnOperation() final;

  void addGPURuntime(ModuleOp op) const;

  void addHostRuntime(ModuleOp op) const;
};

class ParallelTransforms
    : public mlir::par::impl::ParallelTransformsBase<ParallelTransforms> {
public:
  using Base::Base;

  void runOnOperation() final;
};

class AtomicTransforms
    : public ParRewritePattern<
          AtomicOp, PatternInfo::None, BuilderBase,
          xblang::xb::XBLangTypeSystemMixin<AtomicTransforms>> {
public:
  using Base::ParRewritePattern;
  LogicalResult matchAndRewrite(Op op, PatternRewriter &rewriter) const final;
};

class DataRegionTransform
    : public ParRewritePattern<
          DataRegionOp, PatternInfo::None, BuilderBase,
          xblang::xb::XBLangTypeSystemMixin<DataRegionTransform>> {
public:
  DataRegionTransform(MLIRContext *context, const ScopeAnalysis &analysis,
                      ParOptions opts = {}, PatternBenefit benefit = 1,
                      ArrayRef<StringRef> generatedNames = {})
      : Base(context, opts, benefit, generatedNames), analysis(analysis) {}

  LogicalResult matchAndRewrite(Op op, PatternRewriter &rewriter) const final;
  void transformMp(Op op, PatternRewriter &rewriter) const;
  void transformGPU(Op op, PatternRewriter &rewriter) const;
  Value createMapOp(Value value, Type type, MapKindAttr attr, Value queue,
                    Operation *terminator, PatternRewriter &rewriter) const;

protected:
  const ScopeAnalysis &analysis;
};

class ReduceOpTransforms
    : public ParRewritePattern<
          ReduceOp, PatternInfo::HasBoundedRecursion, BuilderBase,
          xblang::xb::XBLangTypeSystemMixin<ReduceOpTransforms>> {
public:
  using Base::ParRewritePattern;
  AtomicOps toAtomic(ReduceOps opKind) const;
  xblang::BinaryOperator toBinaryOp(ReduceOps opKind) const;
  LogicalResult parSeq(Op op, PatternRewriter &rewriter) const final;
  LogicalResult parMp(Op op, PatternRewriter &rewriter) const final;
  LogicalResult parGPU(Op op, PatternRewriter &rewriter) const final;
};

class RegionPrivatizationTransform
    : public ParRewritePattern<RegionOp, PatternInfo::None, BuilderBase,
                               xblang::xb::TypeSystemBase> {
public:
  RegionPrivatizationTransform(MLIRContext *context,
                               const ScopeAnalysis &analysis,
                               ParOptions opts = {}, PatternBenefit benefit = 1,
                               ArrayRef<StringRef> generatedNames = {})
      : Base(context, opts, benefit, generatedNames), analysis(analysis) {}

  LogicalResult matchAndRewrite(Op op, PatternRewriter &rewriter) const final;

protected:
  const ScopeAnalysis &analysis;
};

class LaunchPrivatizationTransform
    : public ParRewritePattern<gpu::LaunchOp, PatternInfo::None, BuilderBase,
                               xblang::xb::TypeSystemBase> {
public:
  LaunchPrivatizationTransform(MLIRContext *context,
                               const ScopeAnalysis &analysis,
                               ParOptions opts = {}, PatternBenefit benefit = 1,
                               ArrayRef<StringRef> generatedNames = {})
      : Base(context, opts, benefit, generatedNames), analysis(analysis) {}

  LogicalResult matchAndRewrite(Op op, PatternRewriter &rewriter) const final;

protected:
  const ScopeAnalysis &analysis;
};

class PrivatizationAnalysis : public ScopeAnalysis {
public:
  using Base = ScopeAnalysis;

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PrivatizationAnalysis);

  PrivatizationAnalysis(Operation *op);

  static Walk opFilter(Operation *op, bool isAnalysisRegion);

  static bool operandFilter(Operation *definingOp, OpOperand *operand);
};
} // namespace

PrivatizationAnalysis::PrivatizationAnalysis(Operation *op)
    : Base(op, opFilter, operandFilter) {}

PrivatizationAnalysis::Walk
PrivatizationAnalysis::opFilter(Operation *op, bool isAnalysisRegion) {
  return isa<RegionOp>(op) || isa<gpu::LaunchOp>(op) ? Walk::analyze
                                                     : Walk::visit;
}

bool PrivatizationAnalysis::operandFilter(Operation *definingOp,
                                          OpOperand *operand) {
  return definingOp ? isa<xblang::xb::VarOp>(definingOp) : false;
}

void ParallelRuntime::addGPURuntime(ModuleOp op) const {
  using namespace xblang::xb;
  OpBuilder builder(op.getBodyRegion());
  SmallVector<FunctionOp, 5> fns;
  auto intType = builder.getIntegerType(32, false);
  auto indexType = builder.getIndexType();
  auto addressType = AddressType::get(op.getContext());
  auto gpuAddressType = AddressType::get(
      op.getContext(),
      gpu::AddressSpaceAttr::get(op.getContext(), gpu::AddressSpace::Global));
  fns.push_back(builder.create<FunctionOp>(
      builder.getUnknownLoc(), "__xblangMapData",
      builder.getFunctionType(
          TypeRange({intType, addressType, indexType, indexType, addressType}),
          TypeRange({gpuAddressType}))));
  fns.push_back(builder.create<FunctionOp>(
      builder.getUnknownLoc(), "__xblangGpuWait",
      builder.getFunctionType(TypeRange({addressType, builder.getI1Type()}),
                              TypeRange({}))));
  fns.push_back(builder.create<FunctionOp>(
      builder.getUnknownLoc(), "__xblangGpuQueue",
      builder.getFunctionType(TypeRange(), TypeRange({addressType}))));
  fns.push_back(builder.create<FunctionOp>(
      builder.getUnknownLoc(), "__xblangGetMatrixDim",
      builder.getFunctionType(TypeRange({intType}), TypeRange({indexType}))));
  fns.push_back(builder.create<FunctionOp>(
      builder.getUnknownLoc(), "__xblangAlloca",
      builder.getFunctionType(TypeRange({addressType, indexType, indexType}),
                              TypeRange({addressType}))));
  fns.push_back(builder.create<FunctionOp>(
      builder.getUnknownLoc(), "__xblangDealloca",
      builder.getFunctionType(TypeRange({addressType, addressType}),
                              TypeRange({}))));
  for (auto fop : fns) {
    fop.setPrivate();
    fop.eraseBody();
  }
}

void ParallelRuntime::addHostRuntime(ModuleOp op) const {
  using namespace xblang::xb;
  OpBuilder builder(op.getBodyRegion());
  SmallVector<FunctionOp, 5> fns;
  auto intType = builder.getIntegerType(32, true);
  fns.push_back(builder.create<FunctionOp>(
      builder.getUnknownLoc(), "omp_get_thread_num",
      builder.getFunctionType(TypeRange(), TypeRange({intType}))));
  fns.push_back(builder.create<FunctionOp>(
      builder.getUnknownLoc(), "omp_get_num_threads",
      builder.getFunctionType(TypeRange(), TypeRange({intType}))));
  for (auto fop : fns) {
    fop.setPrivate();
    fop.eraseBody();
  }
}

void ParallelRuntime::runOnOperation() {
  if (opts.isHost())
    addHostRuntime(getOperation());
  if (opts.isOffload())
    addGPURuntime(getOperation());
}

void ParallelTransforms::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  populateTransformationPatterns(patterns);
  auto &analysis = getAnalysis<PrivatizationAnalysis>();
  patterns.add<DataRegionTransform, RegionPrivatizationTransform,
               LaunchPrivatizationTransform>(patterns.getContext(), analysis);
  FrozenRewritePatternSet patternSet(std::move(patterns));
  GreedyRewriteConfig config;
  config.useTopDownTraversal = true;
  auto result =
      mlir::applyPatternsAndFoldGreedily(getOperation(), patternSet, config);
  if (result.failed())
    signalPassFailure();
}

LogicalResult
AtomicTransforms::matchAndRewrite(Op op, PatternRewriter &rewriter) const {
  auto memType = removeReference(op.getDestination().getType());
  auto valueType = op.getValue().getType();
  if (isRef(valueType) || valueType != memType) {
    Value value = rewriter.create<xblang::xb::CastOp>(op.getLoc(), memType,
                                                      op.getValue());
    rewriter.modifyOpInPlace(op, [&]() { op.getValueMutable().assign(value); });
    return success();
  }
  return failure();
}

Value DataRegionTransform::createMapOp(Value value, Type type, MapKindAttr attr,
                                       Value queue, Operation *terminator,
                                       PatternRewriter &rewriter) const {
  using namespace xblang::xb;
  Value mapValue{};
  auto gpuAddressType = AddressType::get(
      value.getContext(), gpu::AddressSpaceAttr::get(
                              value.getContext(), gpu::AddressSpace::Global));
  MapKindAttr mapAttr = attr;
  switch (attr.getValue()) {
  case MapKind::present:
  case MapKind::to:
  case MapKind::toFrom:
  case MapKind::allocate:
  case MapKind::destroy:
    if (attr.getValue() == MapKind::toFrom)
      mapAttr = MapKindAttr::get(getContext(), MapKind::to);
    mapValue = rewriter.create<MapOp>(value.getLoc(), gpuAddressType, value,
                                      mapAttr, queue);
    mapValue = rewriter.create<CastOp>(value.getLoc(), type, mapValue);
    break;
  case MapKind::from:
    mapValue = rewriter.create<MapOp>(
        value.getLoc(), gpuAddressType, value,
        MapKindAttr::get(rewriter.getContext(), MapKind::allocate), queue);
    mapValue = rewriter.create<CastOp>(value.getLoc(), type, mapValue);
    break;
  }
  if (::xblang::isAnyOf(attr.getValue(), MapKind::toFrom, MapKind::from)) {
    auto grd = guard(rewriter, terminator);
    mapAttr = MapKindAttr::get(getContext(), MapKind::from);
    rewriter.create<MapOp>(value.getLoc(), value.getType(), value, mapAttr,
                           queue);
  }
  return mapValue;
}

void DataRegionTransform::transformGPU(Op op, PatternRewriter &rewriter) const {
  using namespace xblang::xb;
  auto addressSpace =
      gpu::AddressSpaceAttr::get(op.getContext(), gpu::AddressSpace::Global);
  auto ip = rewriter.saveInsertionPoint();
  auto grd = guard(rewriter);
  rewriter.setInsertionPointToStart(&op.getBody().front());
  for (auto [i, memMapping, mappedVar, attr, queue] :
       llvm::enumerate(op.getMemMappings(), op.getVariables(),
                       op.getMappings().value(), op.getQueues())) {
    VarOp var = dyn_cast<VarOp>(mappedVar.getDefiningOp());
    ReferenceType varTy = dyn_cast<ReferenceType>(mappedVar.getType());
    assert(var && varTy);
    auto elementTy = removePtr(varTy.getPointee());
    auto ptrTy = PointerType::get(op.getContext(), elementTy, addressSpace);
    Value mapping =
        createMapOp(memMapping, ptrTy, dyn_cast<MapKindAttr>(attr), queue,
                    op.getBody().back().getTerminator(), rewriter);
    const auto *users = analysis.getUsers(mappedVar);
    if (!users)
      continue;
    for (Operation *user : *users) {
      if (!op->isAncestor(user))
        continue;
      RegionOp region = dyn_cast<RegionOp>(user);
      assert(region);
      auto mappedVars = region.getMappedVarsMutable();
      auto it = std::find_if(mappedVars.begin(), mappedVars.end(),
                             [&](const auto &operand) {
                               return operand.get() == var.getResult();
                             });
      rewriter.modifyOpInPlace(region, [&]() {
        if (it == mappedVars.end()) {
          mappedVars.append(var.getResult());
          region.getVarMappingsMutable().append(mapping);
        } else {
          ptrdiff_t indx = it - mappedVars.begin();
          region.getVarMappingsMutable()[indx].set(mapping);
        }
      });
    }
  }
  rewriter.restoreInsertionPoint(ip);
  auto scope = rewriter.create<xblang::xb::ScopeOp>(op.getLoc(), false);
  rewriter.inlineRegionBefore(op.getBody(), scope.getBody(),
                              scope.getBody().end());
}

void DataRegionTransform::transformMp(Op op, PatternRewriter &rewriter) const {
  using namespace xblang::xb;
  DenseMap<Operation *, SetVector<Value>> regions2Vars;
  for (auto [i, mappedVar] : llvm::enumerate(op.getVariables())) {
    VarOp var = dyn_cast<VarOp>(mappedVar.getDefiningOp());
    assert(var);
    const auto *users = analysis.getUsers(mappedVar);
    if (!users)
      continue;
    for (Operation *user : *users) {
      RegionOp region = dyn_cast<RegionOp>(user);
      assert(region);
      regions2Vars[region].insert(mappedVar);
    }
  }
  for (auto &[op, vars] : regions2Vars) {
    RegionOp region = dyn_cast<RegionOp>(op);
    SetVector<Value> all;
    all.insert(region.getFirstPrivateVars().begin(),
               region.getFirstPrivateVars().end());
    all.insert(region.getSharedVars().begin(), region.getSharedVars().end());
    vars.set_subtract(all);
    if (vars.empty())
      continue;
    rewriter.modifyOpInPlace(region, [&]() {
      for (Value var : vars) {
        if (isPtr(removeReference(var.getType())))
          region.addFirstprivateVariable(var);
        else
          region.addSharedVariable(var);
      }
    });
  }
}

LogicalResult
DataRegionTransform::matchAndRewrite(Op op, PatternRewriter &rewriter) const {
  SmallVector<Operation *> ops{};
  if (!opts.isOffload()) {
    if (opts.isHost())
      transformMp(op, rewriter);
    auto scope = rewriter.create<xblang::xb::ScopeOp>(op.getLoc());
    rewriter.inlineRegionBefore(op.getBody(), scope.getFrontBlock());
    for (auto map : op.getMemMappings()) {
      if (auto mop = dyn_cast<xblang::xb::ArrayViewOp>(map.getDefiningOp()))
        ops.push_back(mop);
    }
  } else
    transformGPU(op, rewriter);
  op->eraseOperands(0, op.getNumOperands());
  rewriter.eraseOp(op);
  for (auto op : ops)
    rewriter.eraseOp(op);
  return success();
}

AtomicOps ReduceOpTransforms::toAtomic(ReduceOps op) const {
  AtomicOps opKind;
  switch (op) {
  case ReduceOps::Add:
    opKind = AtomicOps::Add;
    break;
  case ReduceOps::Mul:
    opKind = AtomicOps::Mul;
    break;
  case ReduceOps::Max:
    opKind = AtomicOps::Max;
    break;
  case ReduceOps::Min:
    opKind = AtomicOps::Min;
    break;
  default:
    assert(false);
    break;
  }
  return opKind;
}

xblang::BinaryOperator ReduceOpTransforms::toBinaryOp(ReduceOps op) const {
  using xblang::BinaryOperator;
  BinaryOperator opKind;
  switch (op) {
  case ReduceOps::Add:
    opKind = BinaryOperator::Add;
    break;
  case ReduceOps::Mul:
    opKind = BinaryOperator::Mul;
    break;
  default:
    assert(false);
    break;
  }
  return opKind;
}

LogicalResult ReduceOpTransforms::parSeq(Op op,
                                         PatternRewriter &rewriter) const {
  using namespace xblang;
  using namespace xblang::xb;
  auto red = rewriter.create<BinaryOp>(
      op.getLoc(), removeReference(op.getType()), toBinaryOp(op.getOp()),
      op.getInit(), op.getValue());
  auto ref =
      rewriter.create<BinaryOp>(op.getLoc(), Ref(op.getType()),
                                BinaryOperator::Assign, op.getValue(), red);
  rewriter.replaceOpWithNewOp<CastOp>(op, op.getType(), ref);
  return success();
}

LogicalResult ReduceOpTransforms::parMp(Op op,
                                        PatternRewriter &rewriter) const {
  using namespace xblang;
  using namespace xblang::xb;
  if (op.getRank()) {
    ParallelHierarchy level = op.getRank().value();
    Value init = op.getInit();
    assert(init);
    if (isRef(init.getType()))
      init = rewriter.create<xblang::xb::CastOp>(
          op.getLoc(), removeReference(init.getType()), init);
    if (level == ParallelHierarchy::tensor) {
      AtomicOps opKind = toAtomic(op.getOp());
      rewriter.replaceOpWithNewOp<AtomicOp>(op, op.getType(), op.getValue(),
                                            init, opKind);
    } else {
      auto red = rewriter.create<BinaryOp>(
          op.getLoc(), removeReference(op.getType()), toBinaryOp(op.getOp()),
          op.getInit(), op.getValue());
      auto ref =
          rewriter.create<BinaryOp>(op.getLoc(), Ref(op.getType()),
                                    BinaryOperator::Assign, op.getValue(), red);
      rewriter.replaceOpWithNewOp<CastOp>(op, op.getType(), ref);
    }
    return success();
  }
  return failure();
}

LogicalResult ReduceOpTransforms::parGPU(Op op,
                                         PatternRewriter &rewriter) const {
  Value value = op.getValue();
  auto valueType = value.getType();
  if (op.getRank() && op.getRank().value() == ParallelHierarchy::tensor) {
    AtomicOps opKind = toAtomic(op.getOp());
    Value init = op.getInit();
    assert(init);
    if (isRef(init.getType()))
      init = rewriter.create<xblang::xb::CastOp>(
          op.getLoc(), removeReference(init.getType()), init);
    init = rewriter.create<ReduceOp>(
        op.getLoc(), init.getType(), init, nullptr, op.getOp(),
        ParallelHierarchyAttr::get(getContext(), ParallelHierarchy::matrix));
    auto id = rewriter.create<IdOp>(
        op.getLoc(), rewriter.getIntegerType(32, true),
        ParallelHierarchy::matrix | ParallelHierarchy::scalar, 0);
    auto zero = rewriter.create<xblang::xb::ConstantOp>(
        op.getLoc(),
        rewriter.getIntegerAttr(rewriter.getIntegerType(32, true), 0));
    auto cmp = rewriter.create<xblang::xb::BinaryOp>(
        op.getLoc(), rewriter.getIntegerType(1), xblang::BinaryOperator::Equal,
        id, zero);
    if (init.getType() != removeReference(op.getValue().getType()))
      init = rewriter.create<xblang::xb::CastOp>(
          op.getLoc(), removeReference(op.getValue().getType()), init);
    auto ifOp = rewriter.create<xblang::xb::IfOp>(op.getLoc(), cmp);
    {
      auto &thenRegion = ifOp.getThenRegion();
      thenRegion.push_back(new mlir::Block());
      auto &thenBlock = thenRegion.front();
      auto grd = guard(rewriter, &thenBlock, thenBlock.begin());
      rewriter.create<AtomicOp>(op.getLoc(), op.getType(), op.getValue(), init,
                                opKind);
      rewriter.create<xblang::xb::YieldOp>(
          op.getLoc(), xblang::xb::YieldKind::Fallthrough, mlir::ValueRange());
    }
    rewriter.replaceOp(op, init);
    return success();
  }
  if (isRef(valueType)) {
    value = rewriter.create<xblang::xb::CastOp>(
        op.getLoc(), removeReference(valueType), value);
    rewriter.modifyOpInPlace(op, [&]() { op.getValueMutable().assign(value); });
    return success();
  }
  return failure();
}

LogicalResult
RegionPrivatizationTransform::matchAndRewrite(Op op,
                                              PatternRewriter &rewriter) const {
  auto insert = [](auto &c1, const auto &c2) {
    c1.insert(c2.begin(), c2.end());
  };
  if (opts.isSequential())
    return failure();
  const auto *values = analysis.getUses(op);
  if (!values)
    return failure();
  SetVector<Value> variables, specified;
  insert(variables, *values);
  insert(specified, op.getFirstPrivateVars());
  insert(specified, op.getPrivateVars());
  insert(specified, op.getSharedVars());
  insert(specified, op.getMappedVars());
  variables.set_subtract(specified);
  auto dataSharing = op.getDefaultDataSharing();
  if (variables.empty())
    return failure();
  rewriter.modifyOpInPlace(op, [&]() {
    switch (dataSharing) {
    case DataSharingKind::Firstprivate:
      for (auto var : variables)
        op.addFirstprivateVariable(var);
      break;
    case DataSharingKind::Private:
      for (auto var : variables)
        op.addPrivateVariable(var);
      break;
    case DataSharingKind::Shared:
      for (auto var : variables)
        op.addSharedVariable(var);
      break;
    }
  });
  return success();
}

LogicalResult
LaunchPrivatizationTransform::matchAndRewrite(Op op,
                                              PatternRewriter &rewriter) const {
  const auto *uses = analysis.getUses(op);
  if (!uses || op->getAttr("privatized"))
    return failure();
  llvm::SmallVector<Value> vars(uses->getArrayRef()), values;
  for (Value val : vars)
    values.push_back(rewriter.create<xblang::xb::CastOp>(
        val.getLoc(), removeReference(val.getType()), val));
  rewriter.modifyOpInPlace(op, [&]() {
    op->setAttr("privatized", rewriter.getUnitAttr());
    rewriter.setInsertionPointToStart(&op.getBody().front());
    for (size_t i = 0; i < vars.size(); ++i) {
      auto var = cast<xblang::xb::VarOp>(vars[i].getDefiningOp());
      auto val = values[i];
      auto newVar = rewriter.create<xblang::xb::VarOp>(
          val.getLoc(), var.getResult().getType(), var.getSymName(),
          var.getType(), xblang::xb::VarKind::local, val);
      rewriter.replaceUsesWithIf(var, newVar, [&](OpOperand &operand) -> bool {
        return op->isProperAncestor(operand.getOwner());
      });
    }
  });
  return success();
}

void mlir::par::populateTransformationPatterns(RewritePatternSet &patterns) {
  patterns.add<AtomicTransforms>(patterns.getContext());
  patterns.add<ReduceOpTransforms>(patterns.getContext());
}
