#ifndef XBLANG_DIALECT_XBLANG_CONCRETIZATION_STMT_H
#define XBLANG_DIALECT_XBLANG_CONCRETIZATION_STMT_H

#include "xblang/Dialect/XBLang/Concretization/Common.h"
#include "xblang/Dialect/XBLang/IR/XBLang.h"

namespace xblang {
namespace xb {
struct RangeForOpConcretization
    : public ConcretizationPattern<RangeForOpConcretization, RangeForOp> {
  using Base::ConcretizationPattern;
  LogicalResult matchAndRewrite(Op op, PatternRewriter &rewriter) const final;
};

struct RangeForCollapseOpConcretization
    : public ConcretizationPattern<RangeForOpConcretization, RangeForOp> {
  using Base::ConcretizationPattern;
  LogicalResult matchAndRewrite(Op op, PatternRewriter &rewriter) const final;
};
} // namespace xb
} // namespace xblang

#endif
