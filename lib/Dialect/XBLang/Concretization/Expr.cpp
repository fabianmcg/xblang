#include "xblang/Dialect/XBLang/Concretization/Expr.h"
#include "xblang/Sema/TypeSystem.h"
#include "xblang/Support/CompareExtras.h"

namespace xblang {
namespace xb {
LogicalResult ArrayOpConcretization::match(Op op) const {
  bool match = false;
  if (isRef(op.getBase().getType()))
    match = true;
  if (!match)
    for (auto indx : op.getIndex())
      if (isRef(indx.getType()) || !isa<IndexType>(indx.getType()))
        match = true;
  return match ? success() : failure();
}

void ArrayOpConcretization::rewrite(Op op, PatternRewriter &rewriter) const {
  Value base = op.getBase();
  auto baseType = base.getType();
  SmallVector<Value> indices{};
  if (isRef(baseType))
    base = castValue(rewriter, removeReference(baseType), base);
  assert(base && "The base pointer cannot be cast.");
  for (auto indx : op.getIndex()) {
    if (isRef(indx.getType()) || !isa<IndexType>(indx.getType())) {
      auto cast = castValue(rewriter, Index(), indx);
      assert(cast);
      indices.push_back(cast);
    } else
      indices.push_back(indx);
  }
  Type resultType = op.getType();
  if (auto ptr = dyn_cast<PointerType>(removeReference(baseType)))
    resultType = Ref(ptr.getPointee(), ptr.getMemorySpace());
  rewriter.replaceOpWithNewOp<ArrayOp>(op, resultType, base, indices);
}

LogicalResult ArrayViewOpConcretization::match(Op op) const {
  bool match = false;
  if (isRef(op.getBase().getType()))
    match = true;
  if (!match)
    for (auto indx : op.getRanges()) {
      if (isRef(indx.getType()))
        match = true;
      else if (auto rt = dyn_cast<RangeType>(indx.getType()))
        if (!isa<IndexType>(rt.getIteratorType()))
          match = true;
    }
  return match ? success() : failure();
}

void ArrayViewOpConcretization::rewrite(Op op,
                                        PatternRewriter &rewriter) const {
  Value base = op.getBase();
  auto baseType = base.getType();
  SmallVector<Value> indices{};
  if (isRef(baseType))
    base = castValue(rewriter, removeReference(baseType), base);
  assert(base && "The base pointer cannot be cast.");
  for (auto indx : op.getRanges()) {
    if (isRef(indx.getType())) {
      auto cast = castValue(rewriter, Index(), indx);
      assert(cast);
      indices.push_back(cast);
    } else if (auto rt = dyn_cast<RangeType>(indx.getType())) {
      if (!isa<IndexType>(rt.getIteratorType())) {
        auto cast = castValue(rewriter, Range(Index()), indx);
        assert(cast);
        indices.push_back(cast);
      }
    } else
      indices.push_back(indx);
  }
  rewriter.replaceOpWithNewOp<ArrayViewOp>(op, op.getType(), base, indices);
}

LogicalResult
BinaryOpConcretization::matchAndRewrite(Op op,
                                        PatternRewriter &rewriter) const {
  auto lhs = op.getLhs();
  auto rhs = op.getRhs();
  auto lt = lhs.getType();
  auto rt = rhs.getType();
  auto type = op.getType();
  auto ok = op.getOp();
  auto unqualLt = removeReference(lt);
  auto unqualRt = removeReference(rt);
  if (isScalar(unqualLt) && isScalar(unqualRt)) {
    if (isCompoundOp(ok)) {
      auto result = rewriter.create<Op>(op.getLoc(), removeReference(type),
                                        removeCompound(ok), lhs, rhs);
      rewriter.replaceOpWithNewOp<Op>(op, type, BinaryOperator::Assign, lhs,
                                      result);
      return success();
    }
    auto nlhs = lhs;
    auto nrhs = rhs;
    auto ntype = type;
    auto promotion = rankTypes(unqualLt, unqualRt);
    assert(promotion.first == RankValidity::Valid);
    if (rt != promotion.second) {
      nrhs = castValue(rewriter, promotion.second, nrhs);
      unqualRt = removeReference(nrhs.getType());
      assert(nrhs);
    }
    if (ok == BinaryOperator::Assign && unqualLt != unqualRt) {
      nrhs = castValue(rewriter, unqualLt, nrhs);
      assert(nrhs);
    }
    if (ok != BinaryOperator::Assign && lt != promotion.second) {
      nlhs = castValue(rewriter, promotion.second, lhs);
      assert(nlhs);
    }
    if (ok == BinaryOperator::Assign) {
      if (removeReference(ntype) == removeReference(nlhs.getType()) &&
          lt != unqualLt)
        ntype = nlhs.getType();
    }
    assert(removeReference(nlhs.getType()) == nrhs.getType());
    bool condition = type != nlhs.getType() && isAlgebraicOp(ok);
    if (nlhs != lhs || nrhs != rhs || ntype != type || condition) {
      if (type == nlhs.getType() || isComparisonOp(ok)) {
        rewriter.modifyOpInPlace(op, [&op, nlhs, nrhs, lhs, rhs]() {
          if (nlhs != lhs)
            op.setOperand(0, nlhs);
          if (nrhs != rhs)
            op.setOperand(1, nrhs);
        });
      } else if (type != ntype) {
        rewriter.replaceOpWithNewOp<Op>(op, ntype, ok, nlhs, nrhs);
      } else {
        assert(ok != BinaryOperator::Assign);
        auto result =
            rewriter.create<Op>(op.getLoc(), promotion.second, ok, nlhs, nrhs);
        auto cast = castValue(rewriter, type, result.getResult());
        assert(cast);
        rewriter.replaceOp(op, cast);
      }
      return success();
    }
  }
  if (ok == BinaryOperator::Assign && isRef(rt)) {
    auto nrhs = castValue(rewriter, unqualRt, rhs);
    rewriter.modifyOpInPlace(op, [&op, nrhs]() { op.setOperand(1, nrhs); });
    return success();
  }
  return failure();
}

LogicalResult
CastOpConcretization::matchAndRewrite(Op op, PatternRewriter &rewriter) const {
  if (op.getUnknown() || op.getLow())
    return failure();
  auto sourceType = op.getValue().getType();
  auto targetType = op.getType();
  auto unqualSourceType = removeReference(sourceType);
  auto unqualTargetType = removeReference(targetType);
  if (unqualSourceType != sourceType && targetType == unqualTargetType &&
      targetType != unqualSourceType) {
    auto cast =
        rewriter.create<CastOp>(op.getLoc(), unqualSourceType, op.getValue());
    rewriter.replaceOpWithNewOp<CastOp>(op, targetType, cast);
    return success();
  }
  if (isBool(unqualTargetType) && !isBool(unqualSourceType) &&
      isScalar(unqualSourceType)) {
    auto zero = rewriter.create<ConstantOp>(op.getLoc(),
                                            rewriter.getIntegerAttr(Bool(), 0));
    rewriter.replaceOpWithNewOp<BinaryOp>(op, Bool(), BinaryOperator::NEQ,
                                          op.getValue(), zero);
    return success();
  }
  if (isAddressLike(targetType) && isAddressLike(sourceType)) {
    rewriter.replaceOpWithNewOp<CastOp>(op, targetType, op.getValue(), false,
                                        true);
    return success();
  }
  if (isAddressLike(targetType) && isInt(sourceType) && !isIndex(targetType)) {
    auto value = castValueSequence(rewriter, targetType, op.getValue());
    assert(value);
    rewriter.replaceOp(op, value);
    return success();
  }
  if (isRange(targetType) && isRange(sourceType)) {
    auto tt = dyn_cast<RangeType>(targetType);
    auto rop = dyn_cast<RangeOp>(op.getValue().getDefiningOp());
    assert(rop);
    auto grd = guard(rewriter, op);
    rewriter.replaceOpWithNewOp<RangeOp>(op, tt, rop.getComparatorAttr(),
                                         rop.getBegin(), rop.getEnd(),
                                         rop.getStepOpAttr(), rop.getStep());
    rewriter.eraseOp(rop);
  }
  {
    Value value = op.getValue();
    SmallVector<Type, 2> sequence;
    auto castResult = castSequence(targetType, sourceType, sequence);
    if (castResult != InvalidCast && sequence.size() > 1 &&
        sequence[0] != targetType) {
      for (auto type : sequence) {
        auto cast = createCast<CastOp>(rewriter, type, value);
        assert(cast);
        value = cast.getResult();
      }
      rewriter.replaceOp(op, value);
      return success();
    }
  }
  if (isa<AnyType>(targetType)) {
    rewriter.replaceAllUsesWith(op.getResult(), op.getValue());
    rewriter.eraseOp(op);
    return success();
  }
  return failure();
}

LogicalResult
CallOpConcretization::matchAndRewrite(Op op, PatternRewriter &rewriter) const {
  if (op->getAttr("builtin")) {
    if (isa<AnyType>(op.getOperand(0).getType()))
      return failure();
    SmallVector<Value> operands(op.getOperands());
    Type resultType = op.getOperand(0).getType();
    if (!::xblang::isAnyOf(op.getCallee(), "max", "min", "abs", "floor",
                           "ceil")) {
      operands[0] =
          castValue(rewriter, rewriter.getF64Type(), op.getOperand(0));
      if (op.getNumOperands() > 1)
        operands[1] = op.getOperand(1);
      resultType = rewriter.getF64Type();
    }
    if (::xblang::isAnyOf(op.getCallee(), "max", "min")) {
      auto lhsType = op.getOperand(0).getType();
      auto rhsType = op.getOperand(1).getType();
      auto validity = rankTypes(lhsType, rhsType);
      if (validity.first != XBLangTypeSystem::RankValidity::Valid)
        assert(false);
      if (validity.second != lhsType)
        operands[0] = castValue(rewriter, validity.second, op.getOperand(0));
      if (validity.second != rhsType)
        operands[1] = castValue(rewriter, validity.second, op.getOperand(1));
      resultType = validity.second;
    }
    if (isa<AnyType>(op.getType(0))) {
      auto cop = rewriter.create<CallOp>(op.getLoc(), TypeRange({resultType}),
                                         op.getCallee(), operands);
      rewriter.replaceAllUsesWith(op.getResult(), cop.getResult()[0]);
      rewriter.eraseOp(op);
      cop->setAttr("builtin", rewriter.getUnitAttr());
      return success();
    }
    return failure();
  }
  return failure();
}

LogicalResult
SelectOpConcretization::matchAndRewrite(Op op,
                                        PatternRewriter &rewriter) const {
  auto cond = op.getCondition();
  auto lhs = op.getLhs();
  auto rhs = op.getRhs();
  auto lt = lhs.getType();
  auto rt = rhs.getType();
  auto type = op.getType();
  auto condType = cond.getType();
  bool modified = false;
  if (removeReference(lt) != lt) {
    lhs = castValue(rewriter, removeReference(lt), lhs);
    modified = true;
  }
  if (removeReference(rt) != rt) {
    rhs = castValue(rewriter, removeReference(rt), rhs);
    modified = true;
  }
  if (I1() != condType) {
    cond = castValue(rewriter, I1(), cond);
    modified = true;
  }
  if (modified) {
    rewriter.replaceOpWithNewOp<SelectOp>(op, type, cond, lhs, rhs);
    return success();
  }
  return failure();
}

LogicalResult
UnaryOponcretization::matchAndRewrite(Op op, PatternRewriter &rewriter) const {
  auto type = op.getType();
  auto expr = op.getExpr();
  if ((!isRef(type) && !isScalar(removeReference(type))))
    return failure();
  switch (op.getOp()) {
  case UnaryOperator::Address:
    rewriter.replaceOpWithNewOp<CastOp>(op, type, expr, false, true);
    return success();
  case UnaryOperator::Dereference: {
    Value base = expr;
    if (isRef(base.getType()))
      base = rewriter.create<CastOp>(op.getLoc(),
                                     removeReference(base.getType()), base);
    rewriter.replaceOpWithNewOp<CastOp>(op, type, base, false, true);
    return success();
  }
  case UnaryOperator::Plus:
    rewriter.replaceOp(op, castValue(rewriter, type, expr));
    return success();
  case UnaryOperator::Minus: {
    auto zero = rewriter.create<ConstantOp>(op.getLoc(),
                                            rewriter.getIntegerAttr(I1(), 0));
    rewriter.replaceOpWithNewOp<BinaryOp>(op, type, BinaryOperator::Sub, zero,
                                          castValue(rewriter, type, expr));
    return success();
  }
  case UnaryOperator::Negation: {
    auto zero = rewriter.create<ConstantOp>(op.getLoc(),
                                            rewriter.getIntegerAttr(Bool(), 0));
    rewriter.replaceOpWithNewOp<BinaryOp>(op, Bool(), BinaryOperator::Equal,
                                          expr, zero);
    return success();
  }
  case UnaryOperator::Increment:
  case UnaryOperator::Decrement: {
    assert(isRef(expr.getType()));
    auto one = rewriter.create<ConstantOp>(
        op.getLoc(), rewriter.getIntegerAttr(I8(IntegerType::Signed), 1));
    auto Operator = op.getOp() == UnaryOperator::Increment
                        ? BinaryOperator::Add
                        : BinaryOperator::Sub;
    auto bop = rewriter
                   .create<BinaryOp>(op.getLoc(), removeReference(type),
                                     Operator, expr, one)
                   .getResult();
    rewriter.replaceOpWithNewOp<BinaryOp>(op, type, BinaryOperator::Assign,
                                          expr, bop);
    return success();
  }
  case UnaryOperator::PostIncrement:
  case UnaryOperator::PostDecrement: {
    assert(isRef(expr.getType()));
    auto one = rewriter.create<ConstantOp>(op.getLoc(),
                                           rewriter.getIntegerAttr(I8(), 1));
    auto Operator = op.getOp() == UnaryOperator::PostIncrement
                        ? BinaryOperator::Add
                        : BinaryOperator::Sub;
    auto result = castValue(rewriter, removeReference(type), expr);
    auto bop = rewriter
                   .create<BinaryOp>(op.getLoc(), removeReference(type),
                                     Operator, result, one)
                   .getResult();
    rewriter.create<BinaryOp>(op.getLoc(), type, BinaryOperator::Assign, expr,
                              bop);
    rewriter.replaceOp(op, result);
    return success();
  }
  default:
    return failure();
  }
}
} // namespace xb
} // namespace xblang
