#ifndef XBLANG_DIALECT_XBLANG_TRANSFORMS_PATTERNS_H
#define XBLANG_DIALECT_XBLANG_TRANSFORMS_PATTERNS_H

#include "xblang/Dialect/XBLang/IR/Interfaces.h"
#include "xblang/Dialect/XBLang/Transforms/PatternBase.h"

namespace xblang {
namespace xb {
class ImplicitCastRewriter
    : public InterfaceRewritePattern<ImplicitCast,
                                     PatternInfo::HasBoundedRecursion> {
public:
  using Base::InterfaceRewritePattern;
  LogicalResult matchAndRewrite(Interface interface,
                                PatternRewriter &rewriter) const final;
};
} // namespace xb
} // namespace xblang

#endif
