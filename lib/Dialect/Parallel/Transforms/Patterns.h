#ifndef XBLANG_DIALECT_PARALLEL_TRANSFORMS_PATTERNS_H
#define XBLANG_DIALECT_PARALLEL_TRANSFORMS_PATTERNS_H

#include "xblang/Dialect/Parallel/IR/Parallel.h"
#include "xblang/Dialect/XBLang/Lowering/Common.h"
#include "xblang/Dialect/XBLang/Transforms/PatternBase.h"
#include "xblang/Dialect/XBLang/Utils/BuilderBase.h"
#include "xblang/Lang/Parallel/Frontend/Options.h"
#include "xblang/Sema/XBLangTypeSystemMixin.h"

namespace mlir {
namespace par {
using ::xblang::par::ParOptions;
using xblang::xb::BuilderBase;
using xblang::xb::PatternInfo;
using xblang::xb::PatternInformation;

template <typename Target, int Options = 0, typename... Parents>
class ParConversionPattern : public OpConversionPattern<Target>,
                             public Parents... {
public:
  using Info =
      PatternInformation<PatternInfo::Conversion, PatternInfo::Op, Options>;
  using PatternBase = OpConversionPattern<Target>;
  using Op = Target;
  using OpAdaptor = typename PatternBase::OpAdaptor;
  using Base = ParConversionPattern;

  ParConversionPattern(MLIRContext *context, ParOptions opts = {},
                       PatternBenefit benefit = 1)
      : PatternBase(context, benefit), opts(opts) {
    if constexpr (Info::hasBoundedRecursion)
      this->setHasBoundedRewriteRecursion(true);
  }

  ParConversionPattern(const TypeConverter &typeConverter, MLIRContext *context,
                       ParOptions opts = {}, PatternBenefit benefit = 1)
      : PatternBase(typeConverter, context, benefit), opts(opts) {
    if constexpr (Info::hasBoundedRecursion)
      this->setHasBoundedRewriteRecursion(true);
  }

  Type convertType(Type type) const {
    if (this->getTypeConverter())
      return this->getTypeConverter()->convertType(type);
    assert(false);
    return nullptr;
  }

protected:
  virtual LogicalResult parSeq(Op op, OpAdaptor adaptor,
                               ConversionPatternRewriter &rewriter) const {
    return failure();
  }

  virtual LogicalResult parMp(Op op, OpAdaptor adaptor,
                              ConversionPatternRewriter &rewriter) const {
    return failure();
  }

  virtual LogicalResult parGPU(Op op, OpAdaptor adaptor,
                               ConversionPatternRewriter &rewriter) const {
    return failure();
  }

  ParOptions opts;

public:
  LogicalResult
  matchAndRewrite(Op op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (opts.isSequential())
      return parSeq(op, adaptor, rewriter);
    else if (opts.isHost())
      return parMp(op, adaptor, rewriter);
    else if (opts.isOffload())
      return parGPU(op, adaptor, rewriter);
    return failure();
  }
};

template <typename Target, int Options = 0, typename... Parents>
class ParRewritePattern : public OpRewritePattern<Target>, public Parents... {
public:
  using Info =
      PatternInformation<PatternInfo::Rewriter, PatternInfo::Op, Options>;
  using Op = Target;
  using PatternBase = OpRewritePattern<Op>;
  using Base = ParRewritePattern;

  ParRewritePattern(MLIRContext *context, ParOptions opts = {},
                    PatternBenefit benefit = 1,
                    ArrayRef<StringRef> generatedNames = {})
      : PatternBase(context, benefit, generatedNames), opts(opts) {
    if constexpr (Info::hasBoundedRecursion)
      this->setHasBoundedRewriteRecursion(true);
  }

protected:
  virtual LogicalResult parSeq(Op op, PatternRewriter &rewriter) const {
    return failure();
  }

  virtual LogicalResult parMp(Op op, PatternRewriter &rewriter) const {
    return failure();
  }

  virtual LogicalResult parGPU(Op op, PatternRewriter &rewriter) const {
    return failure();
  }

  ParOptions opts{};

public:
  LogicalResult matchAndRewrite(Op op,
                                PatternRewriter &rewriter) const override {
    if (opts.isSequential())
      return parSeq(op, rewriter);
    else if (opts.isHost())
      return parMp(op, rewriter);
    else if (opts.isOffload())
      return parGPU(op, rewriter);
    return failure();
  }
};

} // namespace par
} // namespace mlir

#endif
