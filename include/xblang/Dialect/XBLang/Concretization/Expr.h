#ifndef XBLANG_DIALECT_XBLANG_CONCRETIZATION_EXPR_H
#define XBLANG_DIALECT_XBLANG_CONCRETIZATION_EXPR_H

#include "xblang/Dialect/XBLang/Concretization/Common.h"
#include "xblang/Dialect/XBLang/IR/XBLang.h"

namespace xblang {
namespace xb {
struct ArrayOpConcretization
    : public ConcretizationPattern<ArrayOpConcretization, ArrayOp> {
  using Base::ConcretizationPattern;
  LogicalResult match(Op op) const final;
  void rewrite(Op op, PatternRewriter &rewriter) const final;
};

struct ArrayViewOpConcretization
    : public ConcretizationPattern<ArrayViewOpConcretization, ArrayViewOp> {
  using Base::ConcretizationPattern;
  LogicalResult match(Op op) const final;
  void rewrite(Op op, PatternRewriter &rewriter) const final;
};

struct BinaryOpConcretization
    : public ConcretizationPattern<BinaryOpConcretization, BinaryOp,
                                   PatternInfo::HasBoundedRecursion> {
  using Base::ConcretizationPattern;
  LogicalResult matchAndRewrite(Op op, PatternRewriter &rewriter) const final;
};

struct CastOpConcretization
    : public ConcretizationPattern<CastOpConcretization, CastOp,
                                   PatternInfo::HasBoundedRecursion> {
  using Base::ConcretizationPattern;
  LogicalResult matchAndRewrite(Op op, PatternRewriter &rewriter) const final;
};

struct CallOpConcretization
    : public ConcretizationPattern<CallOpConcretization, CallOp,
                                   PatternInfo::HasBoundedRecursion> {
  using Base::ConcretizationPattern;
  LogicalResult matchAndRewrite(Op op, PatternRewriter &rewriter) const final;
};

struct SelectOpConcretization
    : public ConcretizationPattern<SelectOpConcretization, SelectOp,
                                   PatternInfo::HasBoundedRecursion> {
  using Base::ConcretizationPattern;
  LogicalResult matchAndRewrite(Op op, PatternRewriter &rewriter) const final;
};

struct UnaryOponcretization
    : public ConcretizationPattern<UnaryOponcretization, UnaryOp> {
  using Base::ConcretizationPattern;
  LogicalResult matchAndRewrite(Op op, PatternRewriter &rewriter) const final;
};
} // namespace xb
} // namespace xblang
#endif
