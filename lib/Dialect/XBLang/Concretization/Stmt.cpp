#include "xblang/Dialect/XBLang/Concretization/Stmt.h"

namespace xblang {
namespace xb {
LogicalResult RangeForCollapseOpConcretization::matchAndRewrite(
    Op op, PatternRewriter &rewriter) const {
  if (op.getRanges().size() <= 1)
    return failure();
  guard(rewriter);
  Type indxType;
  for (auto varRaw : op.getVariables()) {
    auto var = dyn_cast<VarOp>(varRaw.getDefiningOp());
    assert(var);
    if (!indxType)
      indxType = var.getType();
    else
      indxType = indxType.getIntOrFloatBitWidth() >
                         var.getType().getIntOrFloatBitWidth()
                     ? indxType
                     : var.getType();
  }
  auto c0 = rewriter.create<ConstantOp>(op.getLoc(),
                                        rewriter.getIntegerAttr(indxType, 0));
  auto index =
      rewriter.create<VarOp>(op.getLoc(), ReferenceType::get(indxType),
                             "__loopIndx", indxType, VarKind::local, c0);
  auto tmpRanges = op.getRanges();
  SmallVector<Value> ranges(tmpRanges.begin(), tmpRanges.end());
  Value end{};
  SmallVector<std::pair<Value, Value>> sizes;
  for (auto rangeRaw : ranges) {
    auto range = dyn_cast<RangeOp>(rangeRaw.getDefiningOp());
    assert(range);
    auto bop =
        rewriter.create<BinaryOp>(op.getLoc(), indxType, BinaryOperator::Sub,
                                  range.getEnd(), range.getBegin());
    sizes.push_back({bop, range.getBegin()});
    if (end)
      end = rewriter.create<BinaryOp>(op.getLoc(), indxType,
                                      BinaryOperator::Mul, bop, end);
    else
      end = bop;
  }
  auto range = rewriter.create<RangeOp>(
      op.getLoc(), RangeType::get(getContext(), indxType), BinaryOperator::Less,
      c0, end, nullptr, nullptr);
  auto loopOp =
      rewriter.create<RangeForOp>(op.getLoc(), index, range.getResult());
  {
    rewriter.setInsertionPointToStart(&op.getBody().front());
    Value value = index.getDecl();
    auto vars = op.getVariables();
    for (size_t i = sizes.size() - 1; i > 0; --i) {
      Value mappedIndex = rewriter.create<BinaryOp>(
          op.getLoc(), indxType, BinaryOperator::Mod, value, sizes[i].first);
      mappedIndex =
          rewriter.create<BinaryOp>(op.getLoc(), indxType, BinaryOperator::Add,
                                    sizes[i].second, mappedIndex);
      rewriter.create<BinaryOp>(op.getLoc(), vars[i].getType(),
                                BinaryOperator::Assign, vars[i], mappedIndex);
      value = rewriter.create<BinaryOp>(
          op.getLoc(), indxType, BinaryOperator::Div, value, sizes[i].first);
    }
    value = rewriter.create<BinaryOp>(
        op.getLoc(), indxType, BinaryOperator::Add, sizes[0].second, value);
    rewriter.create<BinaryOp>(op.getLoc(), vars[0].getType(),
                              BinaryOperator::Assign, vars[0], value);
  }
  rewriter.inlineRegionBefore(op.getBody(), loopOp.getBody(),
                              loopOp.getBody().begin());
  rewriter.eraseOp(op);
  for (auto range : ranges)
    rewriter.eraseOp(range.getDefiningOp());
  return success();
}

LogicalResult
RangeForOpConcretization::matchAndRewrite(Op op,
                                          PatternRewriter &rewriter) const {
  using namespace ::xblang;
  using BOp = BinaryOperator;
  assert(op.getRange() && op.getVariable());
  if (op.getRanges().size() > 1)
    return failure();
  auto iterator = op.getVariable();
  auto itType = removeReference(iterator.getType());
  auto range = llvm::dyn_cast<RangeOp>(op.getRange().getDefiningOp());
  assert(range);
  auto cast = [itType, &rewriter](Value value) {
    CastOp result = itType != removeReference(value.getType())
                        ? createCast<CastOp>(rewriter, itType, value)
                        : nullptr;
    return result ? result : value;
  };
  auto end = cast(range.getEnd());
  rewriter.create<BinaryOp>(range.getLoc(), iterator.getType(), BOp::Assign,
                            iterator, range.getBegin());
  Value step = range.getStep();
  if (!step)
    step = rewriter
               .create<ConstantOp>(range.getLoc(),
                                   rewriter.getIntegerAttr(itType, 1))
               .getResult();
  auto loop = rewriter.create<LoopOp>(op.getLoc());
  {
    auto &region = loop.getCondition();
    Block *block = rewriter.createBlock(&region, region.end());
    auto grd = guard(rewriter, block, block->end());
    auto condition =
        rewriter
            .create<BinaryOp>(range.getLoc(), rewriter.getI1Type(),
                              range.getComparator(), op.getVariable(), end)
            .getResult();
    rewriter.create<YieldOp>(range.getLoc(), YieldKind::Fallthrough,
                             ValueRange(condition));
  }
  {
    auto &region = loop.getBody();
    Block *block = rewriter.createBlock(&region, region.end());
    auto grd = guard(rewriter, block, block->end());
    rewriter.inlineBlockBefore(&op.getBody().front(), block, block->begin());
    if (!block->getTerminator())
      rewriter.create<YieldOp>(range.getLoc(), YieldKind::Fallthrough,
                               block->getArguments());
  }
  {
    auto &region = loop.getIteration();
    Block *block = rewriter.createBlock(&region, region.end());
    auto grd = guard(rewriter, block, block->end());
    BOp stepOp = addCompound(
        range.getStepOp().has_value() ? range.getStepOp().value() : BOp::Add);
    assert(isAlgebraicOp(removeCompound(stepOp)));
    rewriter.create<BinaryOp>(range.getLoc(), iterator.getType(), stepOp,
                              iterator, step);
    rewriter.create<YieldOp>(range.getLoc(), YieldKind::Fallthrough,
                             block->getArguments());
  }
  rewriter.eraseOp(op);
  rewriter.eraseOp(range);
  return success();
}
} // namespace xb
} // namespace xblang
