#include "xblang/Dialect/XBLang/Transforms/Passes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "xblang/Dialect/Parallel/IR/Dialect.h"
#include "xblang/Dialect/Parallel/Transforms/Passes.h"
#include "xblang/Dialect/XBLang/IR/XBLang.h"
#include "xblang/Dialect/XBLang/Lowering/Common.h"
#include "xblang/Dialect/XBLang/Lowering/Type.h"
#include "xblang/Dialect/XBLang/Transforms/PatternBase.h"
#include "xblang/Sema/TypeSystem.h"

using namespace mlir;
using namespace xblang;
using namespace xblang::xb;

namespace xblang {
namespace xb {
#define GEN_PASS_DEF_XBLANGLOWERING
#include "xblang/Dialect/XBLang/Transforms/Passes.h.inc"
} // namespace xb
} // namespace xblang

namespace {
template <typename Target>
struct LoweringPattern : public OpConversionPattern<Target>,
                         LoweringBuilderBase,
                         TypeSystemBase {
  using Base = LoweringPattern;
  using Op = Target;
  using OpAdaptor = typename OpConversionPattern<Target>::OpAdaptor;

  LoweringPattern(const XBLangTypeConverter &typeConverter,
                  PatternBenefit benefit = 1)
      : OpConversionPattern<Target>(typeConverter, typeConverter.getContext(),
                                    benefit) {}

  Type convertType(Type type) const {
    if (auto converter = this->getTypeConverter())
      return converter->convertType(type);
    llvm_unreachable("The pattern should hold a valid type converter.");
    return nullptr;
  }
};

//===----------------------------------------------------------------------===//
// Patterns
//===----------------------------------------------------------------------===//

struct ArrayOpLowering : public LoweringPattern<ArrayOp> {
  using Base::Base;
  LogicalResult
  matchAndRewrite(Op op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final;
};

struct ArrayViewOpLowering : public LoweringPattern<ArrayViewOp> {
  using Base::Base;
  LogicalResult
  matchAndRewrite(Op op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final;
};

struct BinaryOpLowering : public LoweringPattern<BinaryOp> {
  using Base::Base;

  void initialize() { setHasBoundedRewriteRecursion(); }

  LogicalResult
  matchAndRewrite(Op op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final;
};

struct CallOpLowering : public LoweringPattern<CallOp> {
  using Base::Base;
  LogicalResult handleBuiltin(Op op, OpAdaptor adaptor,
                              ConversionPatternRewriter &rewriter) const;
  LogicalResult
  matchAndRewrite(Op op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final;
};

struct CastOpLowering : public LoweringPattern<CastOp> {
  using Base::Base;

  void initialize() { setHasBoundedRewriteRecursion(); }

  LogicalResult
  matchAndRewrite(Op op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final;
};

struct ConstantOpLowering : public LoweringPattern<ConstantOp> {
  using Base::Base;
  LogicalResult
  matchAndRewrite(Op op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final;
};

struct FuncOpLowering : public LoweringPattern<FunctionOp> {
  using Base::Base;
  LogicalResult
  matchAndRewrite(Op op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final;
};

struct FromElementsOpLowering : public LoweringPattern<FromElementsOp> {
  using Base::Base;
  LogicalResult
  matchAndRewrite(Op op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final;
};

struct GetElementOpLowering : public LoweringPattern<GetElementOp> {
  using Base::Base;
  LogicalResult
  matchAndRewrite(Op op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final;
};

struct LoadOpLowering : public LoweringPattern<LoadOp> {
  using Base::Base;
  LogicalResult
  matchAndRewrite(Op op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final;
};

struct NullptrOpLowering : public LoweringPattern<NullPtrOp> {
  using Base::Base;
  LogicalResult
  matchAndRewrite(Op op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final;
};

struct ReturnOpLowering : public LoweringPattern<ReturnOp> {
  using Base::Base;
  LogicalResult
  matchAndRewrite(Op op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final;
};

struct SelectOpLowering : public LoweringPattern<SelectOp> {
  using Base::Base;
  LogicalResult
  matchAndRewrite(Op op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final;
};

struct SizeOfOpLowering : public LoweringPattern<SizeOfOp> {
  using Base::Base;
  LogicalResult
  matchAndRewrite(Op op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final;
};

struct VarOpLowering : public LoweringPattern<VarOp> {
  using Base::Base;
  LogicalResult
  matchAndRewrite(Op op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final;
};

//===----------------------------------------------------------------------===//
// XB to Std Pass
//===----------------------------------------------------------------------===//
class XBToStd : public xblang::xb::impl::XBLangLoweringBase<XBToStd> {
public:
  using Base::Base;

  void runOnOperation() final;
};

} // namespace

//===----------------------------------------------------------------------===//
// XB array Op
//===----------------------------------------------------------------------===//

LogicalResult
ArrayOpLowering::matchAndRewrite(Op op, OpAdaptor adaptor,
                                 ConversionPatternRewriter &rewriter) const {
  auto array = toMemref(rewriter, adaptor.getBase());
  if (!array)
    return failure();
  auto arrayTy = array.getType();
  int64_t rank = arrayTy.getRank();
  llvm::SmallVector<int64_t, 5> offsets(rank, ShapedType::kDynamic),
      ones(rank, 1);
  auto subviewTy =
      memref::SubViewOp::inferResultType(arrayTy, offsets, ones, ones);
  auto subviewOp = rewriter.create<memref::SubViewOp>(
      op.getLoc(), subviewTy, array, adaptor.getIndex(), ValueRange(),
      ValueRange(), offsets, ones, ones);
  rewriter.replaceOpWithNewOp<CastOp>(op, convertType(op.getType()), subviewOp);
  return success();
}

//===----------------------------------------------------------------------===//
// XB array view Op
//===----------------------------------------------------------------------===//

LogicalResult ArrayViewOpLowering::matchAndRewrite(
    Op op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const {
  auto array = toMemref(rewriter, adaptor.getBase());
  if (!array)
    return failure();
  auto arrayTy = dyn_cast<MemRefType>(array.getType());
  int64_t rank = arrayTy.getRank();
  auto c1 = rewriter.create<index::ConstantOp>(op.getLoc(), 1);

  SmallVector<Value> offsetsDyn(rank), sizesDyn(rank);
  SmallVector<int64_t> offsets(rank, ShapedType::kDynamic),
      sizes(rank, ShapedType::kDynamic), ones(rank, 1);

  // Create the offsets and sizes of the subview.
  for (auto [i, range] : llvm::enumerate(adaptor.getRanges())) {
    // Case: [..., i ,...]
    if (isa<IndexType>(range.getType())) {
      sizesDyn[i] = c1;
      sizes[i] = 1;
      offsetsDyn[i] = range;
    }
    // Case: [..., b : e ,...]
    else if (isa<RangeType>(range.getType())) {
      auto rangeOp = dyn_cast<RangeOp>(range.getDefiningOp());
      assert(rangeOp && "A range must come from a RangeOp.");
      offsetsDyn[i] = rangeOp.getBegin();
      sizesDyn[i] = rewriter.create<index::SubOp>(
          rangeOp.getLoc(), rangeOp.getEnd(), offsetsDyn[i]);
    } else
      // TODO: Remove this with a verifier.
      assert(false && "Invalid index type.");
  }

  // Create the subview.
  auto subviewTy =
      memref::SubViewOp::inferResultType(arrayTy, offsets, sizes, ones);
  rewriter.replaceOpWithNewOp<mlir::memref::SubViewOp>(
      op, subviewTy, array, offsetsDyn, sizesDyn, ValueRange(), offsets, sizes,
      ones);
  return success();
}

//===----------------------------------------------------------------------===//
// XB binary Op
//===----------------------------------------------------------------------===//
namespace {
struct CmpInfo {
  CmpInfo(arith::CmpIPredicate siCmp, arith::CmpIPredicate uiCmp,
          arith::CmpFPredicate fpCmp, index::IndexCmpPredicate indxCmp)
      : siCmp(siCmp), uiCmp(uiCmp), fpCmp(fpCmp), indxCmp(indxCmp) {}

  CmpInfo(arith::CmpIPredicate siCmp, arith::CmpFPredicate fpCmp,
          index::IndexCmpPredicate indxCmp)
      : CmpInfo(siCmp, siCmp, fpCmp, indxCmp) {}

  arith::CmpIPredicate siCmp;
  arith::CmpIPredicate uiCmp;
  arith::CmpFPredicate fpCmp;
  index::IndexCmpPredicate indxCmp;

  arith::CmpIPredicate iCmp(Type type) const {
    if (type.isSignedInteger())
      return siCmp;
    return uiCmp;
  }
};

void replaceCmpOp(BinaryOp op, const CmpInfo &info, Type operandsType,
                  Value lhs, Value rhs, ConversionPatternRewriter &rewriter) {
  using namespace arith;
  auto type = operandsType;
  if (type.isa<IntegerType>())
    rewriter.replaceOpWithNewOp<CmpIOp>(op, info.iCmp(type), lhs, rhs);
  else if (type.isa<FloatType>())
    rewriter.replaceOpWithNewOp<CmpFOp>(op, info.fpCmp, lhs, rhs);
  else if (type.isa<IndexType>())
    rewriter.replaceOpWithNewOp<index::CmpOp>(op, info.indxCmp, lhs, rhs);
  else
    assert(false && "Invalid compare op.");
}

template <typename IO, typename FO, typename IndexOp, typename UO = IO>
void replaceOp(BinaryOp op, Type operandsType, Value lhs, Value rhs,
               ConversionPatternRewriter &rewriter) {
  auto type = operandsType;
  if (type.isa<IntegerType>()) {
    if (type.isUnsignedInteger())
      rewriter.replaceOpWithNewOp<UO>(op, lhs, rhs);
    else
      rewriter.replaceOpWithNewOp<IO>(op, lhs, rhs);
  } else if (type.isa<FloatType>())
    rewriter.replaceOpWithNewOp<FO>(op, lhs, rhs);
  else if (type.isa<IndexType>())
    rewriter.replaceOpWithNewOp<IndexOp>(op, lhs, rhs);
  else
    assert(false && "Invalid binary op.");
}
} // namespace

LogicalResult
BinaryOpLowering::matchAndRewrite(Op op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const {
  using namespace arith;
  using namespace index;
  using CI = CmpInfo;
  using CmpI = CmpIPredicate;
  using CmpF = CmpFPredicate;
  using CmpIn = IndexCmpPredicate;
  auto type = op.getRhs().getType();
  switch (op.getOp()) {
  case ::xblang::BinaryOperator::Add:
    replaceOp<AddIOp, AddFOp, AddOp>(op, type, adaptor.getLhs(),
                                     adaptor.getRhs(), rewriter);
    return success();
  case ::xblang::BinaryOperator::Sub:
    replaceOp<SubIOp, SubFOp, SubOp>(op, type, adaptor.getLhs(),
                                     adaptor.getRhs(), rewriter);
    return success();
  case ::xblang::BinaryOperator::Mul:
    replaceOp<MulIOp, MulFOp, MulOp>(op, type, adaptor.getLhs(),
                                     adaptor.getRhs(), rewriter);
    return success();
  case ::xblang::BinaryOperator::Div:
    replaceOp<DivSIOp, DivFOp, DivUOp, DivUIOp>(op, type, adaptor.getLhs(),
                                                adaptor.getRhs(), rewriter);
    return success();
  case ::xblang::BinaryOperator::Mod:
    replaceOp<RemSIOp, RemFOp, RemUOp, RemUIOp>(op, type, adaptor.getLhs(),
                                                adaptor.getRhs(), rewriter);
    return success();
  case ::xblang::BinaryOperator::Equal:
    replaceCmpOp(op, CI(CmpI::eq, CmpF::UEQ, CmpIn::EQ), type, adaptor.getLhs(),
                 adaptor.getRhs(), rewriter);
    return success();
  case ::xblang::BinaryOperator::NEQ:
    replaceCmpOp(op, CI(CmpI::ne, CmpF::UNE, CmpIn::NE), type, adaptor.getLhs(),
                 adaptor.getRhs(), rewriter);
    return success();
  case ::xblang::BinaryOperator::Less:
    replaceCmpOp(op, CI(CmpI::slt, CmpI::ult, CmpF::OLT, CmpIn::ULT), type,
                 adaptor.getLhs(), adaptor.getRhs(), rewriter);
    return success();
  case ::xblang::BinaryOperator::Greater:
    replaceCmpOp(op, CI(CmpI::sgt, CmpI::ugt, CmpF::OGT, CmpIn::UGT), type,
                 adaptor.getLhs(), adaptor.getRhs(), rewriter);
    return success();
  case ::xblang::BinaryOperator::LEQ:
    replaceCmpOp(op, CI(CmpI::sle, CmpI::ule, CmpF::OLE, CmpIn::ULE), type,
                 adaptor.getLhs(), adaptor.getRhs(), rewriter);
    return success();
  case ::xblang::BinaryOperator::GEQ:
    replaceCmpOp(op, CI(CmpI::sge, CmpI::uge, CmpF::OGE, CmpIn::UGE), type,
                 adaptor.getLhs(), adaptor.getRhs(), rewriter);
    return success();
  case ::xblang::BinaryOperator::BinaryAnd:
  case ::xblang::BinaryOperator::And:
    rewriter.replaceOpWithNewOp<arith::AndIOp>(op, adaptor.getLhs(),
                                               adaptor.getRhs());
    return success();
  case ::xblang::BinaryOperator::BinaryOr:
  case ::xblang::BinaryOperator::Or:
    rewriter.replaceOpWithNewOp<arith::OrIOp>(op, adaptor.getLhs(),
                                              adaptor.getRhs());
    return success();
  case ::xblang::BinaryOperator::BinaryXor:
    rewriter.replaceOpWithNewOp<arith::XOrIOp>(op, adaptor.getLhs(),
                                               adaptor.getRhs());
    return success();
  case ::xblang::BinaryOperator::LShift:
    replaceOp<arith::ShLIOp, arith::ShLIOp, index::ShlOp, arith::ShLIOp>(
        op, op.getType(), adaptor.getLhs(), adaptor.getRhs(), rewriter);
    return success();
  case ::xblang::BinaryOperator::RShift:
    replaceOp<arith::ShRSIOp, arith::ShRSIOp, index::ShrUOp, arith::ShRUIOp>(
        op, op.getType(), adaptor.getLhs(), adaptor.getRhs(), rewriter);
    return success();
  case ::xblang::BinaryOperator::Assign: {
    auto grd = guard(rewriter, op);
    trivialStore(rewriter, adaptor.getLhs(), adaptor.getRhs());
    rewriter.replaceOp(op, adaptor.getLhs());
    return success();
  }
  default:
    op.dump();
    op.getLoc().dump();
    assert(false);
    return failure();
  }
}

//===----------------------------------------------------------------------===//
// XB call Op
//===----------------------------------------------------------------------===//

namespace {
struct MathBuiltins : LoweringBuilderBase {
  MathBuiltins(CallOp op, CallOpAdaptor adaptor,
               ConversionPatternRewriter &rewriter,
               const TypeConverter *converter)
      : op(op), adaptor(adaptor), rewriter(rewriter), converter(converter) {}

  Value cast(Type target, Type source, Value value) {
    if (target == source)
      return value;
    std::optional<Value> res =
        nativeCast(rewriter, target, source, value, converter);
    return res ? res.value() : nullptr;
  }

  template <typename IOp, typename FOp>
  void emit() {
    auto first = adaptor.getOperands()[0];
    if (isa<IntegerType>(first.getType()))
      rewriter.replaceOpWithNewOp<IOp>(op, adaptor.getOperands());
    else if (isa<FloatType>(first.getType()))
      rewriter.replaceOpWithNewOp<FOp>(op, adaptor.getOperands());
  }

  template <typename Op>
  void emit() {
    Value value = adaptor.getOperands()[0];
    auto type = value.getType();
    if (isa<IntegerType>(type))
      value = cast(rewriter.getF64Type(), op.getOperand(0).getType(), value);
    value = rewriter.create<Op>(op.getLoc(), value,
                                ::mlir::arith::FastMathFlags::fast);
    if (isa<IntegerType>(type))
      value = rewriter.create<CastOp>(op.getLoc(), op.getType(0), value);
    rewriter.replaceOp(op, value);
  }

  template <typename Op>
  void emitRound() {
    auto first = adaptor.getOperands()[0];
    if (isa<IntegerType>(first.getType()))
      rewriter.replaceOp(op, first);
    else if (isa<FloatType>(first.getType()))
      rewriter.replaceOpWithNewOp<Op>(op, adaptor.getOperands());
  }

  void emitPow() {
    using namespace ::mlir::math;
    Value base = adaptor.getOperands()[0];
    auto baseType = base.getType();
    Value exponent = adaptor.getOperands()[1];
    auto exponentType = exponent.getType();
    Value result{};
    if (isa<IntegerType>(baseType))
      base = cast(rewriter.getF64Type(), op.getOperand(0).getType(), base);
    if (isa<IntegerType>(exponentType))
      result = rewriter.create<FPowIOp>(op.getLoc(), base, exponent);
    else if (isa<FloatType>(exponentType))
      result = rewriter.create<PowFOp>(op.getLoc(), base, exponent);
    if (isa<IntegerType>(baseType))
      result = rewriter.create<CastOp>(op.getLoc(), op.getType(0), result);
    rewriter.replaceOp(op, result);
  }

  template <typename IOp, typename UIOp, typename FOp>
  void emitMinMax() {
    XBLangTypeSystem typeSystem(*op.getContext());
    auto lhs = adaptor.getOperands()[0];
    auto rhs = adaptor.getOperands()[1];
    auto lhsType = op.getOperand(0).getType();
    auto rhsType = op.getOperand(1).getType();
    auto validity = typeSystem.rankTypes(lhsType, rhsType);
    Value result{};
    if (validity.first != XBLangTypeSystem::RankValidity::Valid)
      assert(false);
    if (validity.second != lhsType)
      lhs = cast(validity.second, op.getOperand(0).getType(), lhs);
    if (validity.second != rhsType)
      rhs = cast(validity.second, op.getOperand(1).getType(), rhs);
    if (auto intType = dyn_cast<IntegerType>(validity.second)) {
      if (intType.isSigned())
        result = rewriter.create<IOp>(op.getLoc(), lhs, rhs);
      else if (intType.isUnsigned())
        result = rewriter.create<UIOp>(op.getLoc(), lhs, rhs);
    } else if (isa<FloatType>(validity.second))
      result = rewriter.create<FOp>(op.getLoc(), lhs, rhs);
    assert(result);
    if (validity.second != lhsType)
      result = cast(lhsType, validity.second, result);
    rewriter.replaceOp(op, result);
  }

  CallOp op;
  CallOpAdaptor adaptor;
  ConversionPatternRewriter &rewriter;
  const TypeConverter *converter{};
};
} // namespace

LogicalResult
CallOpLowering::handleBuiltin(Op op, OpAdaptor adaptor,
                              ConversionPatternRewriter &rewriter) const {
  using namespace ::mlir::math;
  auto name = op.getCallee();
  MathBuiltins mathBultins(op, adaptor, rewriter, typeConverter);
  if (name == "abs")
    mathBultins.emit<AbsIOp, AbsFOp>();
  else if (name == "sqrt")
    mathBultins.emit<SqrtOp>();
  else if (name == "rsqrt")
    mathBultins.emit<RsqrtOp>();
  else if (name == "sin")
    mathBultins.emit<SinOp>();
  else if (name == "cos")
    mathBultins.emit<CosOp>();
  else if (name == "tan")
    mathBultins.emit<TanOp>();
  else if (name == "atan")
    mathBultins.emit<AtanOp>();
  else if (name == "exp")
    mathBultins.emit<ExpOp>();
  else if (name == "log")
    mathBultins.emit<LogOp>();
  else if (name == "floor")
    mathBultins.emitRound<FloorOp>();
  else if (name == "ceil")
    mathBultins.emitRound<CeilOp>();
  else if (name == "pow")
    mathBultins.emitPow();
  else if (name == "max")
    mathBultins.emitMinMax<arith::MaxSIOp, arith::MaxUIOp, arith::MaxNumFOp>();
  else if (name == "min")
    mathBultins.emitMinMax<arith::MinSIOp, arith::MinUIOp, arith::MinNumFOp>();
  return success();
}

LogicalResult
CallOpLowering::matchAndRewrite(Op op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const {
  if (op->getAttr("builtin"))
    return handleBuiltin(op, adaptor, rewriter);
  auto operands = adaptor.getOperands();
  SmallVector<Value> args(operands.begin(), operands.end());
  for (auto [i, value] : llvm::enumerate(op.getOperands())) {
    if (isRef(value.getType()))
      if (convertType(value.getType()) != args[i].getType())
        args[i] = rewriter.create<CastOp>(
            value.getLoc(), convertType(value.getType()), args[i], true);
  }
  auto types = op.getResultTypes();
  if (types.size() > 0)
    rewriter.replaceOpWithNewOp<func::CallOp>(
        op, op.getCallee(), TypeRange(convertType(types[0])), args);
  else
    rewriter.replaceOpWithNewOp<func::CallOp>(op, op.getCallee(), TypeRange(),
                                              args);
  return success();
}

//===----------------------------------------------------------------------===//
// XB cast Op
//===----------------------------------------------------------------------===//

LogicalResult
CastOpLowering::matchAndRewrite(Op op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const {
  auto sourceType = op.getValue().getType();
  auto targetType = op.getType();
  Value value = adaptor.getValue();
  if (isa<AnyType>(targetType)) {
    rewriter.replaceAllUsesWith(op.getResult(), value);
    rewriter.eraseOp(op);
    return success();
  }
  if (value.getType() == convertType(targetType)) {
    rewriter.replaceOp(op, value);
    return success();
  }
  if (isLoadCast(targetType, sourceType)) {
    if (auto result = trivialLoad(rewriter, value)) {
      rewriter.replaceOp(op, result);
      return success();
    }
    return failure();
  }
  if (op.getUnknown()) {
    rewriter.replaceOpWithNewOp<CastOp>(op, convertType(targetType), value);
    return success();
  }
  if (isa<TensorType>(sourceType) && isPtr(targetType)) {
    rewriter.replaceOpWithNewOp<CastOp>(op, convertType(targetType), value);
    return success();
  }
  if (auto result =
          nativeCast(rewriter, targetType, sourceType, value, typeConverter)) {
    if (result.value())
      rewriter.replaceOp(op, result.value());
    else
      rewriter.replaceOp(op, value);
    return success();
  }
  assert(false && "Invalid cast operation.");
  return failure();
}

//===----------------------------------------------------------------------===//
// XB constant Op
//===----------------------------------------------------------------------===//

LogicalResult
ConstantOpLowering::matchAndRewrite(Op op, OpAdaptor adaptor,
                                    ConversionPatternRewriter &rewriter) const {
  if (auto ic = dyn_cast<IntegerAttr>(op.getValue())) {
    rewriter.replaceOpWithNewOp<arith::ConstantOp>(
        op, IntegerAttr::get(convertType(op.getType()), ic.getValue()));
  } else if (auto fc = dyn_cast<FloatAttr>(op.getValue()))
    rewriter.replaceOpWithNewOp<arith::ConstantOp>(op, fc);
  else
    return failure();
  return success();
}

//===----------------------------------------------------------------------===//
// XB from_elements Op
//===----------------------------------------------------------------------===//

LogicalResult FromElementsOpLowering::matchAndRewrite(
    Op op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const {
  auto memrefType = dyn_cast<MemRefType>(convertType(op.getType()));
  assert(memrefType && "The list should have tensor type.");
  if (!memrefType.hasStaticShape())
    return failure();
  // Create a local array and push back the elements.
  Value mem =
      rewriter.create<memref::AllocaOp>(op.getLoc(), memrefType).getResult();
  auto elements = adaptor.getElements();
  auto shape = memrefType.getShape();
  if (shape.size() != 1)
    return failure();
  for (int64_t i = 0; i < shape[0]; ++i)
    rewriter.create<memref::StoreOp>(
        op.getLoc(), elements[i], mem,
        ValueRange({rewriter.create<index::ConstantOp>(op.getLoc(), i)}));
  rewriter.replaceOp(op, mem);
  return success();
}

//===----------------------------------------------------------------------===//
// XB func Op
//===----------------------------------------------------------------------===//

LogicalResult
FuncOpLowering::matchAndRewrite(Op op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const {
  // Delete built-in functions.
  if (op->getAttr("builtin")) {
    rewriter.eraseOp(op);
    return success();
  }
  // Create the new func Op.
  func::FuncOp funcOp = rewriter.create<func::FuncOp>(
      op.getLoc(), op.getName(),
      dyn_cast<FunctionType>(convertType(op.getFunctionType())));
  if (!op.isDeclaration()) {
    // Add attributes.
    if (op->getAttr("inline"))
      funcOp->setAttr("passthrough", rewriter.getArrayAttr(
                                         {rewriter.getStringAttr("inline")}));
    else if (op->getAttr("noinline"))
      funcOp->setAttr("passthrough", rewriter.getArrayAttr(
                                         {rewriter.getStringAttr("noinline")}));
    if (op->getAttr("par"))
      funcOp->setAttr("par", rewriter.getUnitAttr());
    if (op->getAttr("internal"))
      funcOp->setAttr("internal", rewriter.getUnitAttr());
    rewriter.inlineRegionBefore(op.getRegion(), funcOp.getBody(), funcOp.end());
    if (failed(rewriter.convertRegionTypes(&funcOp.getFunctionBody(),
                                           *typeConverter)))
      return failure();
  } else
    funcOp.setVisibility(SymbolTable::Visibility::Private);
  rewriter.eraseOp(op);
  return success();
}

//===----------------------------------------------------------------------===//
// XB get_element Op
//===----------------------------------------------------------------------===//

LogicalResult GetElementOpLowering::matchAndRewrite(
    Op op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const {
  Value base = adaptor.getBase();
  if (isStruct(base.getType()) && isRef(base.getType())) {
    assert(false && "The base should be a pure struct or a reference to one.");
    return failure();
  }
  rewriter.replaceOpWithNewOp<GetElementOp>(op, convertType(op.getType()), base,
                                            adaptor.getIndex());
  return success();
}

//===----------------------------------------------------------------------===//
// XB load Op
//===----------------------------------------------------------------------===//

LogicalResult
LoadOpLowering::matchAndRewrite(Op op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const {
  rewriter.replaceOpWithNewOp<xb::CastOp>(op, op.getResult().getType(),
                                          op.getValue());
  return success();
}

//===----------------------------------------------------------------------===//
// XB nullptr Op
//===----------------------------------------------------------------------===//

LogicalResult
NullptrOpLowering::matchAndRewrite(Op op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter) const {
  rewriter.replaceOpWithNewOp<NullPtrOp>(op, convertType(op.getType()));
  return success();
}

//===----------------------------------------------------------------------===//
// XB return Op
//===----------------------------------------------------------------------===//

LogicalResult
ReturnOpLowering::matchAndRewrite(Op op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const {
  if (op.hasOperands())
    rewriter.replaceOpWithNewOp<ReturnOp>(op, adaptor.getInput());
  else
    return failure();
  return success();
}

//===----------------------------------------------------------------------===//
// XB select Op
//===----------------------------------------------------------------------===//

LogicalResult
SelectOpLowering::matchAndRewrite(Op op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const {
  auto convertedType = convertType(op.getType());
  if (!isArithmetic(convertedType))
    return failure();
  rewriter.replaceOpWithNewOp<arith::SelectOp>(
      op, convertedType, adaptor.getCondition(), adaptor.getLhs(),
      adaptor.getRhs());
  return success();
}

//===----------------------------------------------------------------------===//
// XB sizeof Op
//===----------------------------------------------------------------------===//

LogicalResult
SizeOfOpLowering::matchAndRewrite(Op op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const {
  rewriter.modifyOpInPlace(op,
                           [&]() { op.setType(convertType(op.getType())); });
  return success();
}

//===----------------------------------------------------------------------===//
// XB var Op
//===----------------------------------------------------------------------===//

LogicalResult
VarOpLowering::matchAndRewrite(Op op, OpAdaptor adaptor,
                               ConversionPatternRewriter &rewriter) const {
  Value init = adaptor.getInit();
  // If it's a reference, just replace the variable with the reference.
  if (isRef(op.getType())) {
    assert(init && "References must have a valid source.");
    rewriter.replaceOp(op, init);
    return success();
  }

  // Create an Alloca Op. Tensors are converted to directly to memrefs.
  bool isTensorTy = isTensor(op.getType());
  Type convertedTy = convertType(op.getType());
  MemRefType memrefTy = isTensorTy ? dyn_cast<MemRefType>(convertedTy)
                                   : mlir::MemRefType::get({}, convertedTy);
  memref::AllocaOp allocaOp =
      rewriter.create<memref::AllocaOp>(op.getLoc(), memrefTy);
  if (init) {
    if (isTensor(op.getInit().getType()))
      rewriter.create<memref::CopyOp>(op.getLoc(), adaptor.getInit(), allocaOp);
    else
      rewriter.create<memref::StoreOp>(op.getLoc(), init, allocaOp);
  }
  if (!isTensorTy)
    rewriter.replaceOpWithNewOp<CastOp>(
        op, convertType(op.getResult().getType()), allocaOp.getResult());
  else
    rewriter.replaceOp(op, allocaOp.getResult());
  return success();
}

//===----------------------------------------------------------------------===//
// XBLang to standard MLIR pass
//===----------------------------------------------------------------------===//

namespace {
bool isAddressCast(CastOp op) {
  Type destTy = op.getType();
  Type srcTy = op.getValue().getType();
  Type strippedSrc = removeReference(srcTy);
  if (strippedSrc != srcTy && strippedSrc == destTy)
    return false;
  bool isAddrDest =
      TypeSystemBase::isAddressLike(destTy) || TypeSystemBase::isRef(destTy);
  bool isAddrSrc =
      TypeSystemBase::isAddressLike(srcTy) || TypeSystemBase::isRef(srcTy);
  return (isAddrSrc && isAddrSrc) ||
         (TypeSystemBase::isMemRef(destTy) && isAddrSrc) ||
         (isAddrDest && TypeSystemBase::isMemRef(srcTy));
}
} // namespace

void xblang::xb::populateXblangToStd(ConversionTarget &conversionTarget,
                                     RewritePatternSet &patterns,
                                     const XBLangTypeConverter &typeConverter) {
  patterns.add<ArrayOpLowering, ArrayViewOpLowering, BinaryOpLowering,
               CallOpLowering, CastOpLowering, ConstantOpLowering,
               FromElementsOpLowering, FuncOpLowering, LoadOpLowering,
               GetElementOpLowering, NullptrOpLowering, ReturnOpLowering,
               SelectOpLowering, SizeOfOpLowering, VarOpLowering>(
      typeConverter);
  conversionTarget
      .addIllegalOp<ArrayOp, ArrayViewOp, BinaryOp, CallOp, LoadOp, ConstantOp,
                    FromElementsOp, FunctionOp, SelectOp, VarOp>();

  conversionTarget.addLegalDialect<
      ::mlir::affine::AffineDialect, ::mlir::BuiltinDialect,
      ::mlir::arith::ArithDialect, ::mlir::func::FuncDialect,
      ::mlir::gpu::GPUDialect, ::mlir::memref::MemRefDialect,
      ::mlir::math::MathDialect, ::mlir::index::IndexDialect,
      ::mlir::scf::SCFDialect, ::mlir::cf::ControlFlowDialect,
      ::mlir::LLVM::LLVMDialect, ::mlir::par::ParDialect, XBLangDialect>();

  // Determine dynamic legality
  conversionTarget.addDynamicallyLegalOp<CastOp>([&](CastOp op) {
    return isAddressCast(op) && typeConverter.isLegal(op);
  });
  conversionTarget.addDynamicallyLegalOp<GetElementOp>(
      [&typeConverter](GetElementOp op) -> bool {
        return typeConverter.isLegal(op);
      });
  conversionTarget.addDynamicallyLegalOp<NullPtrOp>(
      [&typeConverter](NullPtrOp op) { return typeConverter.isLegal(op); });
  conversionTarget.addDynamicallyLegalOp<SizeOfOp>(
      [&typeConverter](SizeOfOp op) {
        auto type = TypeSystemBase::removeName(op.getType());
        return typeConverter.convertType(type) == type;
      });
  conversionTarget.addDynamicallyLegalOp<ReturnOp>(
      [&typeConverter](ReturnOp op) { return typeConverter.isLegal(op); });
}

void XBToStd::runOnOperation() {
  ConversionTarget target(getContext());
  RewritePatternSet patterns(&getContext());
  XBLangTypeConverter converter(getContext());
  populateXblangToStd(target, patterns, converter);
  mlir::par::populateLoweringPatterns(target, patterns, converter);
  if (mlir::failed(
          applyPartialConversion(getOperation(), target, std::move(patterns))))
    signalPassFailure();
}
