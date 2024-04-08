#ifndef XBLANG_DIALECT_XBLANG_LOWERING_PATTERNBASE_H
#define XBLANG_DIALECT_XBLANG_LOWERING_PATTERNBASE_H

#include "mlir/Transforms/DialectConversion.h"
#include <type_traits>

namespace xblang {
namespace xb {
struct PatternInfo {
  typedef enum {
    Conversion,
    Rewriter,
  } PatternKind;

  typedef enum { Any = 0, Op = 1, Interface = 2, Trait = 3 } PatternSource;

  typedef enum {
    None = 0,
    RequiresConverter = 1,
    HasBoundedRecursion = 2,
    _Legality = 63,
    IsLegal,
    IsDynamicallyLegal = IsLegal | (IsLegal << 1),
  } PatternOptions;
};

template <int Kind, int Source, int Options = 0>
class PatternInformation : public PatternInfo {
protected:
  static constexpr int kind = Kind;
  static constexpr int source = Source;
  static constexpr int options = Options;

  static constexpr bool checkFlag(int value, int flag) {
    return (value & flag) == flag;
  }

  static constexpr bool getLegality(int value, int flag) {
    return (value & ~_Legality) == flag;
  }

public:
  static constexpr bool isConversion = kind == Conversion;
  static constexpr bool isRewriter = kind == Rewriter;
  static constexpr bool matchesAny = source == Any;
  static constexpr bool matchesOp = source == Op;
  static constexpr bool matchesInterface = source == Interface;
  static constexpr bool matchesTrait = source == Trait;
  static constexpr bool requiresConverter =
      checkFlag(options, RequiresConverter);
  static constexpr bool hasBoundedRecursion =
      checkFlag(options, HasBoundedRecursion);
  static constexpr bool hasLegality = matchesOp;
  static constexpr int isLegal =
      hasLegality ? getLegality(options, IsLegal) : -1;
  static constexpr int isDynamicallyLegal =
      hasLegality ? getLegality(options, IsDynamicallyLegal) : -1;
  static constexpr int isIllegal =
      hasLegality ? (isLegal == isDynamicallyLegal) : -1;
};

template <typename Target, int Options = 0, typename... Parents>
struct ConversionPattern : public mlir::OpConversionPattern<Target>,
                           public Parents... {
  using Info =
      PatternInformation<PatternInfo::Conversion, PatternInfo::Op, Options>;
  using Op = Target;
  using PatternBase = mlir::OpConversionPattern<Op>;
  using Base = ConversionPattern;

  ConversionPattern(MLIRContext *context, PatternBenefit benefit = 1)
      : PatternBase(context, benefit) {
    if constexpr (Info::hasBoundedRecursion)
      this->setHasBoundedRewriteRecursion(true);
  }

  ConversionPattern(const TypeConverter &typeConverter, MLIRContext *context,
                    PatternBenefit benefit = 1)
      : PatternBase(typeConverter, context, benefit) {
    if constexpr (Info::hasBoundedRecursion)
      this->setHasBoundedRewriteRecursion(true);
  }

  Type convertType(Type type) const {
    if (this->getTypeConverter())
      return this->getTypeConverter()->convertType(type);
    assert(false);
    return nullptr;
  }
};

template <int Options, typename... Parents>
struct ConversionPattern<void, Options, Parents...>
    : public ::mlir::ConversionPattern, public Parents... {
  using Info =
      PatternInformation<PatternInfo::Conversion, PatternInfo::Any, Options>;
  using PatternBase = ::mlir::ConversionPattern;
  using Base = ConversionPattern;

  ConversionPattern(MLIRContext *context, PatternBenefit benefit = 1,
                    ArrayRef<StringRef> generatedNames = {})
      : PatternBase(MatchAnyOpTypeTag(), benefit, context, generatedNames) {
    if constexpr (Info::hasBoundedRecursion)
      this->setHasBoundedRewriteRecursion(true);
  }

  ConversionPattern(const TypeConverter &typeConverter, MLIRContext *context,
                    PatternBenefit benefit = 1,
                    ArrayRef<StringRef> generatedNames = {})
      : PatternBase(typeConverter, MatchAnyOpTypeTag(), benefit, context,
                    generatedNames) {
    if constexpr (Info::hasBoundedRecursion)
      this->setHasBoundedRewriteRecursion(true);
  }

  Type convertType(Type type) const {
    if (getTypeConverter())
      return getTypeConverter()->convertType(type);
    assert(false);
    return nullptr;
  }
};

template <typename Target, int Options = 0, typename... Parents>
struct RewritePattern : public mlir::OpRewritePattern<Target>,
                        public Parents... {
  using Info =
      PatternInformation<PatternInfo::Rewriter, PatternInfo::Op, Options>;
  using Op = Target;
  using PatternBase = mlir::OpRewritePattern<Op>;
  using Base = RewritePattern;

  RewritePattern(MLIRContext *context, PatternBenefit benefit = 1,
                 ArrayRef<StringRef> generatedNames = {})
      : PatternBase(context, benefit, generatedNames) {
    if constexpr (Info::hasBoundedRecursion)
      this->setHasBoundedRewriteRecursion(true);
  }
};

template <int Options, typename... Parents>
struct RewritePattern<void, Options, Parents...>
    : public ::mlir::RewritePattern, public Parents... {
  using Info =
      PatternInformation<PatternInfo::Rewriter, PatternInfo::Any, Options>;
  using PatternBase = ::mlir::RewritePattern;
  using Base = RewritePattern;

  RewritePattern(MLIRContext *context, PatternBenefit benefit = 1,
                 ArrayRef<StringRef> generatedNames = {})
      : PatternBase(MatchAnyOpTypeTag(), benefit, context, generatedNames) {
    if constexpr (Info::hasBoundedRecursion)
      this->setHasBoundedRewriteRecursion(true);
  }
};

template <typename Target, int Options = 0, typename... Parents>
struct InterfaceConversionPattern
    : public mlir::OpInterfaceConversionPattern<Target>,
      public Parents... {
  using Info = PatternInformation<PatternInfo::Conversion,
                                  PatternInfo::Interface, Options>;
  using Interface = Target;
  using PatternBase = mlir::OpInterfaceConversionPattern<Interface>;
  using Base = InterfaceConversionPattern;

  InterfaceConversionPattern(MLIRContext *context, PatternBenefit benefit = 1)
      : PatternBase(context, benefit) {
    if constexpr (Info::hasBoundedRecursion)
      this->setHasBoundedRewriteRecursion(true);
  }

  InterfaceConversionPattern(const TypeConverter &typeConverter,
                             MLIRContext *context, PatternBenefit benefit = 1)
      : PatternBase(typeConverter, context, benefit) {
    if constexpr (Info::hasBoundedRecursion)
      this->setHasBoundedRewriteRecursion(true);
  }

  Type convertType(Type type) const {
    if (this->getTypeConverter())
      return this->getTypeConverter()->convertType(type);
    assert(false);
    return nullptr;
  }
};

template <typename Target, int Options = 0, typename... Parents>
struct InterfaceRewritePattern : public mlir::OpInterfaceRewritePattern<Target>,
                                 public Parents... {
  using Info = PatternInformation<PatternInfo::Rewriter, PatternInfo::Interface,
                                  Options>;
  using Interface = Target;
  using PatternBase = mlir::OpInterfaceRewritePattern<Interface>;
  using Base = InterfaceRewritePattern;

  InterfaceRewritePattern(MLIRContext *context, PatternBenefit benefit = 1)
      : PatternBase(context, benefit) {
    if constexpr (Info::hasBoundedRecursion)
      this->setHasBoundedRewriteRecursion(true);
  }
};

template <PatternInfo::PatternKind Kind, typename Target, int Options = 0,
          typename... Parents>
using GenericOpPattern =
    std::conditional_t<Kind == PatternInfo::Conversion,
                       ConversionPattern<Target, Options, Parents...>,
                       RewritePattern<Target, Options, Parents...>>;

template <PatternInfo::PatternKind Kind, typename Target, int Options = 0,
          typename... Parents>
using GenericInterfacePattern =
    std::conditional_t<Kind == PatternInfo::Conversion,
                       InterfaceConversionPattern<Target, Options, Parents...>,
                       InterfaceRewritePattern<Target, Options, Parents...>>;

template <PatternInfo::PatternKind Kind, PatternInfo::PatternSource Source,
          typename Target, int Options = 0, typename... Parents>
struct GenericPatternImpl {
  using type = GenericOpPattern<Kind, Target, Options, Parents...>;
};

template <PatternInfo::PatternKind Kind, typename Target, int Options,
          typename... Parents>
struct GenericPatternImpl<Kind, PatternInfo::Interface, Target, Options,
                          Parents...> {
  using type = GenericInterfacePattern<Kind, Target, Options, Parents...>;
};

template <PatternInfo::PatternKind Kind, PatternInfo::PatternSource Source,
          typename Target, int Options = 0, typename... Parents>
using GenericPattern = typename GenericPatternImpl<Kind, Source, Target,
                                                   Options, Parents...>::type;

class AddConversionPattern {
public:
  template <typename Pattern>
  using info_t = typename Pattern::Info;
  template <typename Pattern>
  using operation_t = typename Pattern::Op;

  template <typename Pattern>
  static void setLegality(ConversionTarget &target) {
    using operation = operation_t<Pattern>;
    if constexpr (info_t<Pattern>::isIllegal == 1)
      target.addIllegalOp<operation>();
    else if constexpr (info_t<Pattern>::isLegal == 1)
      target.addLegalOp<operation>();
  }

  template <typename Pattern, typename Converter>
  static void addPattern(MLIRContext &context, mlir::ConversionTarget &target,
                         mlir::RewritePatternSet &patterns,
                         const Converter &converter) {
    using pattern = Pattern;
    if constexpr (info_t<pattern>::hasLegality)
      setLegality<pattern>(target);
    if constexpr (info_t<Pattern>::requiresConverter)
      patterns.add<pattern>(converter, &context);
    else
      patterns.add<pattern>(&context);
  }

  template <typename Converter>
  static void add(MLIRContext &, mlir::ConversionTarget &,
                  mlir::RewritePatternSet &, const Converter &) {}

  template <typename Pattern, typename... P, typename Converter>
  static void add(MLIRContext &context, mlir::ConversionTarget &target,
                  mlir::RewritePatternSet &patterns,
                  const Converter &converter) {
    using pattern = Pattern;
    addPattern<pattern>(context, target, patterns, converter);
    if constexpr (sizeof...(P) > 0)
      add<P...>(context, target, patterns, converter);
  }
};
} // namespace xb
} // namespace xblang

#endif
