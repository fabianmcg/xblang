#include "mlir/IR/PatternMatch.h"
#include "xblang/Dialect/XBLang/IR/Type.h"
#include "xblang/Dialect/XBLang/IR/XBLang.h"
#include "xblang/Support/Format.h"

using namespace mlir;
using namespace xblang::xb;

// void IfOp::getSuccessorRegions(std::optional<unsigned> index,
//                                SmallVectorImpl<RegionSuccessor> &regions) {
//   if (index) {
//     regions.push_back(RegionSuccessor());
//     return;
//   }
//   Region *elseRegion = &this->getElseRegion();
//   if (elseRegion->empty())
//     elseRegion = nullptr;
//   regions.push_back(
//       RegionSuccessor(&getThenRegion(), getThenRegion().getArguments()));
//   if (elseRegion)
//     regions.push_back(
//         RegionSuccessor(elseRegion, getElseRegion().getArguments()));
//   return;
// }

Type IfOp::getImplicitCast(unsigned arg) {
  auto Bool = mlir::IntegerType::get(getContext(), 1);
  if (getOperand().getType() != Bool)
    return Bool;
  return nullptr;
}

LogicalResult IfOp::verify() { return success(); }

SmallVector<mlir::Region *> LoopOp::getLoopRegions() { return {&getBody()}; }

void LoopOp::getSuccessorRegions(RegionBranchPoint point,
                                 SmallVectorImpl<RegionSuccessor> &regions) {
  if (point.isParent()) {
    regions.push_back(
        RegionSuccessor(&getCondition(), getCondition().getArguments()));
    return;
  } else if (point == RegionBranchPoint(&getCondition())) {
    regions.push_back(RegionSuccessor(&getBody(), getBody().getArguments()));
    regions.push_back(RegionSuccessor());
  } else if (point == RegionBranchPoint(&getBody()))
    regions.push_back(
        RegionSuccessor(&getIteration(), getIteration().getArguments()));
  else
    regions.push_back(
        RegionSuccessor(&getCondition(), getCondition().getArguments()));
  return;
}

SmallVector<mlir::Region *> RangeForOp::getLoopRegions() {
  return {&getBody()};
}

void RangeForOp::getSuccessorRegions(
    RegionBranchPoint point, SmallVectorImpl<RegionSuccessor> &regions) {
  regions.push_back(RegionSuccessor(&getBody()));
  regions.push_back(RegionSuccessor());
}

// MutableOperandRange
// ReturnOp::getMutableSuccessorOperands(std::optional<unsigned> index) {
//   auto range = MutableOperandRange(*this, getNumOperands(), 0);
//   return range;
// }

Type ReturnOp::getImplicitCast(unsigned arg) {
  auto rt = getType();
  assert(rt);
  if (rt.has_value() && getOperand(arg).getType() != rt.value())
    return rt.value();
  return nullptr;
}

Block *ScopeOp::getFrontBlock() {
  if (getBody().begin() != getBody().end())
    return &getBody().front();
  return nullptr;
}

Block *ScopeOp::getBackBlock() {
  if (getBody().begin() != getBody().end())
    return &getBody().back();
  return nullptr;
}

LogicalResult ScopeOp::canonicalize(ScopeOp op, PatternRewriter &rewriter) {
  auto &body = op.getRegion();
  // TODO: Make it more general.
  if (body.hasOneBlock()) {
    Block &front = body.front();
    if (front.getOperations().size() == 2)
      if (auto childScope = dyn_cast<ScopeOp>(&front.getOperations().front())) {
        rewriter.modifyOpInPlace(op, [&]() {
          rewriter.inlineRegionBefore(childScope.getBody(), &front);
          rewriter.eraseOp(childScope);
          rewriter.eraseBlock(&front);
        });
        return success();
      }
  }
  return failure();
}

MutableOperandRange
YieldOp::getMutableSuccessorOperands(RegionBranchPoint point) {
  return MutableOperandRange(*this, getNumOperands(), 0);
}
