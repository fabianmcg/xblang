#include "xblang/Dialect/XBLang/Transforms/Passes.h"

#include "mlir/Conversion/GPUCommon/GPUCommonPass.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/Pass/Pass.h"
#include "xblang/ADT/DoubleTypeSwitch.h"
#include "xblang/Dialect/XBLang/IR/XBLang.h"
#include "xblang/Dialect/XBLang/Lowering/Common.h"
#include "xblang/Dialect/XBLang/Lowering/Type.h"
#include "xblang/Lang/Parallel/Frontend/Options.h"

#include <stack>

using namespace mlir;
using namespace xblang::xb;

namespace xblang {
namespace xb {
#define GEN_PASS_DEF_XBLANGTOLLVM
#include "xblang/Dialect/XBLang/Transforms/Passes.h.inc"
} // namespace xb
} // namespace xblang

namespace {
template <typename Target>
struct LoweringPattern : public OpConversionPattern<Target>,
                         BuilderBase,
                         TypeSystemBase {
  using Base = LoweringPattern;
  using Op = Target;
  using OpAdaptor = typename OpConversionPattern<Target>::OpAdaptor;

  LoweringPattern(const TypeConverter &typeConverter, MLIRContext *context,
                  PatternBenefit benefit = 1)
      : OpConversionPattern<Target>(typeConverter, context, benefit) {}

  Type convertType(Type type) const {
    if (auto converter = this->getTypeConverter())
      return converter->convertType(type);
    llvm_unreachable("The pattern should hold a valid type converter.");
    return nullptr;
  }
};

//===----------------------------------------------------------------------===//
// Conversion patterns.
//===----------------------------------------------------------------------===//

// struct CastOpToLLVM : public LoweringPattern<CastOp> {
//   using Base::Base;
//
//   void initialize() { setHasBoundedRewriteRecursion(); }
//
//   bool ptr2Int(ConversionPatternRewriter &rewriter, Op op, Value value,
//                Type lhs, Type rhs, bool reversed) const;
//
//   bool ptr2Memref(ConversionPatternRewriter &rewriter, Op op, Value ptr,
//                   MemRefType target, PointerType rhs) const;
//
//   bool ref2Memref(ConversionPatternRewriter &rewriter, Op op, Value value,
//                   MemRefType target, ReferenceType rhs) const;
//
//   bool memref2Ptr(ConversionPatternRewriter &rewriter, Op op, Value value,
//                   PointerType target, MemRefType source) const;
//
//   bool memref2Ref(ConversionPatternRewriter &rewriter, Op op, Value value,
//                   ReferenceType target, MemRefType source) const;
//
//   bool ptr2ptr(ConversionPatternRewriter &rewriter, Op op, Value value,
//                PointerType target, PointerType source) const;
//
//   LogicalResult
//   matchAndRewrite(Op op, OpAdaptor adaptor,
//                   ConversionPatternRewriter &rewriter) const final;
// };

// struct CollapseMemRefOpLLVMLowering : public
// LoweringPattern<CollapseMemRefOp> {
//   using Base::Base;
//   LogicalResult
//   matchAndRewrite(Op op, OpAdaptor adaptor,
//                   ConversionPatternRewriter &rewriter) const final;
// };
//
// struct GetElementOpLLVMLowering : public LoweringPattern<GetElementOp> {
//   using Base::Base;
//   LogicalResult
//   matchAndRewrite(Op op, OpAdaptor adaptor,
//                   ConversionPatternRewriter &rewriter) const final;
// };

struct NullptrOpSPIRVLowering : public LoweringPattern<NullPtrOp> {
  using Base::Base;
  LogicalResult
  matchAndRewrite(Op op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final;
};

struct ReturnOpGPUFuncLowering : public LoweringPattern<ReturnOp> {
  using Base::Base;
  LogicalResult
  matchAndRewrite(Op op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final;
};

// struct SizeOfOpLLVMLowering : public LoweringPattern<SizeOfOp> {
//   using Base::Base;
//   LogicalResult
//   matchAndRewrite(Op op, OpAdaptor adaptor,
//                   ConversionPatternRewriter &rewriter) const final;
// };

} // namespace

//===----------------------------------------------------------------------===//
// XB cast Op conversion
//===----------------------------------------------------------------------===//

// bool CastOpToLLVM::ptr2Int(ConversionPatternRewriter &rewriter, Op op,
//                            Value value, Type lhs, Type rhs,
//                            bool reversed) const {
//   if (reversed)
//     rewriter.replaceOpWithNewOp<LLVM::PtrToIntOp>(op, convertType(rhs),
//     value);
//   else
//     rewriter.replaceOpWithNewOp<LLVM::IntToPtrOp>(op, convertType(lhs),
//     value);
//   return true;
// }
//
// bool CastOpToLLVM::ptr2Memref(ConversionPatternRewriter &rewriter, Op op,
//                               Value ptr, MemRefType target,
//                               PointerType rhs) const {
//   auto typeConverter =
//       static_cast<const LLVMTypeConverter *>(getTypeConverter());
//   auto memrefTy = dyn_cast<MemRefType>(target);
//   MemRefDescriptor descriptor = MemRefDescriptor::fromStaticShape(
//       rewriter, op.getLoc(), *typeConverter, memrefTy, ptr);
//   rewriter.replaceOpWithNewOp<CastOp>(op, target, descriptor);
//   return true;
// }
//
// bool CastOpToLLVM::ref2Memref(ConversionPatternRewriter &rewriter, Op op,
//                               Value value, MemRefType target,
//                               ReferenceType rhs) const {
//   auto structType = dyn_cast<LLVM::LLVMStructType>(convertType(target));
//   assert(structType);
//   Value descriptor = rewriter.create<LLVM::UndefOp>(op.getLoc(), structType);
//   descriptor = rewriter.create<LLVM::InsertValueOp>(
//       descriptor.getLoc(), descriptor, value, llvm::ArrayRef<int64_t>{0});
//   descriptor = rewriter.create<LLVM::InsertValueOp>(
//       descriptor.getLoc(), descriptor, value, llvm::ArrayRef<int64_t>{1});
//   auto offset = rewriter.create<LLVM::ConstantOp>(
//       descriptor.getLoc(), structType.getBody().back(), 0);
//   descriptor = rewriter.create<LLVM::InsertValueOp>(
//       op.getLoc(), descriptor, offset, llvm::ArrayRef<int64_t>{2});
//   rewriter.replaceOpWithNewOp<CastOp>(op, target, descriptor);
//   return true;
// }
//
// bool CastOpToLLVM::memref2Ptr(ConversionPatternRewriter &rewriter, Op op,
//                               Value value, PointerType target,
//                               MemRefType source) const {
//   auto structType = dyn_cast<LLVM::LLVMStructType>(value.getType());
//   assert(structType);
//   Value ptr = rewriter.create<LLVM::ExtractValueOp>(op.getLoc(), value,
//                                                     ArrayRef<int64_t>({1}));
//   Value offset = rewriter.create<LLVM::ExtractValueOp>(op.getLoc(), value,
//                                                        ArrayRef<int64_t>({2}));
//   int64_t addressSpace = 0;
//   if (source.getMemorySpace()) {
//     auto addrSpace = getTypeConverter()->convertTypeAttribute(
//         source, source.getMemorySpace());
//     if (addrSpace)
//       if (auto as = dyn_cast<IntegerAttr>(*addrSpace))
//         addressSpace = as.getInt();
//   }
//   rewriter.replaceOpWithNewOp<LLVM::GEPOp>(
//       op, LLVM::LLVMPointerType::get(getContext(), addressSpace),
//       convertType(target.getPointee()), ptr, offset);
//   return true;
// }
//
// bool CastOpToLLVM::memref2Ref(ConversionPatternRewriter &rewriter, Op op,
//                               Value value, ReferenceType target,
//                               MemRefType source) const {
//   auto structType = dyn_cast<LLVM::LLVMStructType>(value.getType());
//   assert(structType);
//   Value ptr = rewriter.create<LLVM::ExtractValueOp>(op.getLoc(), value,
//                                                     ArrayRef<int64_t>({1}));
//   Value offset = rewriter.create<LLVM::ExtractValueOp>(op.getLoc(), value,
//                                                        ArrayRef<int64_t>({2}));
//   int64_t addressSpace = 0;
//   if (source.getMemorySpace())
//     if (auto as = dyn_cast<IntegerAttr>(source.getMemorySpace()))
//       addressSpace = as.getInt();
//   rewriter.replaceOpWithNewOp<LLVM::GEPOp>(
//       op, LLVM::LLVMPointerType::get(getContext(), addressSpace),
//       convertType(target.getPointee()), ptr, offset);
//   return true;
// }
//
// bool CastOpToLLVM::ptr2ptr(ConversionPatternRewriter &rewriter, Op op,
//                            Value value, PointerType target,
//                            PointerType source) const {
//   if (target.getPointee() == source.getPointee() &&
//       target.getMemorySpace() != source.getMemorySpace()) {
//     rewriter.replaceOpWithNewOp<LLVM::AddrSpaceCastOp>(op,
//     convertType(target),
//                                                        value);
//   }
//   return true;
// }
//
// LogicalResult
// CastOpToLLVM::matchAndRewrite(Op op, OpAdaptor adaptor,
//                               ConversionPatternRewriter &rewriter) const {
//   auto sourceType = op.getValue().getType();
//   auto targetType = op.getType();
//   Value value = adaptor.getValue();
//   using Switch = ::xblang::DoubleTypeSwitch<Type, Type, bool>;
//   bool result =
//       Switch::Switch(targetType, sourceType)
//           .Case<PointerType, IndexType, true>(
//               [&](PointerType lhs, IndexType rhs, bool reversed) {
//                 return ptr2Int(rewriter, op, value, lhs, rhs, reversed);
//               })
//           .Case<PointerType, IntegerType, true>(
//               [&](PointerType lhs, IntegerType rhs, bool reversed) {
//                 return ptr2Int(rewriter, op, value, lhs, rhs, reversed);
//               })
//           .Case<AddressType, IndexType, true>(
//               [&](AddressType lhs, IndexType rhs, bool reversed) {
//                 return ptr2Int(rewriter, op, value, lhs, rhs, reversed);
//               })
//           .Case<AddressType, IntegerType, true>(
//               [&](AddressType lhs, IntegerType rhs, bool reversed) {
//                 return ptr2Int(rewriter, op, value, lhs, rhs, reversed);
//               })
//           .Case<ReferenceType, PointerType, true>(
//               [&](ReferenceType target, PointerType source, bool reversed) {
//                 rewriter.replaceOp(op, value);
//                 return true;
//               })
//           .Case<AddressType, PointerType, true>(
//               [&](AddressType target, PointerType source, bool reversed) {
//                 rewriter.replaceOp(op, value);
//                 return true;
//               })
//           .Case<MemRefType, PointerType>(
//               [&](MemRefType target, PointerType source) {
//                 return ptr2Memref(rewriter, op, value, target, source);
//               })
//           .Case<MemRefType, ReferenceType>(
//               [&](MemRefType target, ReferenceType source) {
//                 return ref2Memref(rewriter, op, value, target, source);
//               })
//           .Case<PointerType, MemRefType>(
//               [&](PointerType target, MemRefType source) {
//                 return memref2Ptr(rewriter, op, value, target, source);
//               })
//           .Case<ReferenceType, MemRefType>(
//               [&](ReferenceType target, MemRefType source) {
//                 return memref2Ref(rewriter, op, value, target, source);
//               })
//           .Case<MemRefType, LLVM::LLVMStructType>(
//               [&](MemRefType target, LLVM::LLVMStructType source) {
//                 rewriter.replaceOp(op, value);
//                 return true;
//               })
//           .Case<PointerType, PointerType>(
//               [&](PointerType target, PointerType source) {
//                 return ptr2ptr(rewriter, op, value, target, source);
//               })
//           .DefaultValue(false);
//   if (result)
//     return success();
//   return failure();
// }
//
////===----------------------------------------------------------------------===//
//// XB collapse Op conversion
////===----------------------------------------------------------------------===//
//
// LogicalResult CollapseMemRefOpLLVMLowering::matchAndRewrite(
//    Op op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const {
//  auto value = adaptor.getInput();
//  auto target = op.getResult().getType();
//  auto sourceType = dyn_cast<LLVM::LLVMStructType>(value.getType());
//  assert(sourceType);
//  auto structType = dyn_cast<LLVM::LLVMStructType>(convertType(target));
//  assert(structType);
//  auto ptr = rewriter.create<LLVM::ExtractValueOp>(op.getLoc(), value,
//                                                   ArrayRef<int64_t>({0}));
//  auto alignedPtr = rewriter.create<LLVM::ExtractValueOp>(
//      op.getLoc(), value, ArrayRef<int64_t>({1}));
//  auto offset = rewriter.create<LLVM::ExtractValueOp>(op.getLoc(), value,
//                                                      ArrayRef<int64_t>({2}));
//  Value descriptor = rewriter.create<LLVM::UndefOp>(op.getLoc(), structType);
//  Value offPtr = rewriter.create<LLVM::GEPOp>(
//      op.getLoc(), ptr.getType(), op.getType().getElementType(), alignedPtr,
//      ArrayRef<LLVM::GEPArg>{LLVM::GEPArg(offset)});
//  descriptor = rewriter.create<LLVM::InsertValueOp>(
//      descriptor.getLoc(), descriptor, ptr, llvm::ArrayRef<int64_t>{0});
//  descriptor = rewriter.create<LLVM::InsertValueOp>(
//      descriptor.getLoc(), descriptor, offPtr, llvm::ArrayRef<int64_t>{1});
//  // TODO: set this to 0.
//  descriptor = rewriter.create<LLVM::InsertValueOp>(
//      op.getLoc(), descriptor, offset, llvm::ArrayRef<int64_t>{2});
//  rewriter.replaceOpWithNewOp<CastOp>(op, target, descriptor);
//  return success();
//}
//
////===----------------------------------------------------------------------===//
//// XB GEP Op conversion
////===----------------------------------------------------------------------===//
//
// LogicalResult GetElementOpLLVMLowering::matchAndRewrite(
//    Op op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const {
//  auto indx = dyn_cast<IntegerAttr>(op.getIndex());
//  assert(indx);
//  Value base = adaptor.getBase();
//  if (isa<LLVM::LLVMStructType>(base.getType())) {
//    rewriter.replaceOpWithNewOp<LLVM::ExtractValueOp>(op, base,
//    indx.getInt()); return success();
//  }
//  rewriter.replaceOpWithNewOp<LLVM::GEPOp>(
//      op, convertType(op.getType()),
//      convertType(removeReference(op.getBase().getType())), base,
//      ArrayRef<LLVM::GEPArg>({LLVM::GEPArg(0), LLVM::GEPArg(indx.getInt())}));
//  return success();
//}
//
////===----------------------------------------------------------------------===//
//// XB nullptr Op conversion
////===----------------------------------------------------------------------===//

LogicalResult NullptrOpSPIRVLowering::matchAndRewrite(
    Op op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const {
  //  rewriter.replaceOpWithNewOp<spirv::ConstantOp>(op,
  //  convertType(op.getType()));
  return success();
}

////===----------------------------------------------------------------------===//
//// XB return Op conversion
////===----------------------------------------------------------------------===//

LogicalResult ReturnOpGPUFuncLowering::matchAndRewrite(
    Op op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const {
  if (op.hasOperands())
    rewriter.replaceOpWithNewOp<gpu::ReturnOp>(op, adaptor.getInput());
  else
    rewriter.replaceOpWithNewOp<gpu::ReturnOp>(op);
  return success();
}

////===----------------------------------------------------------------------===//
//// XB sizeof Op conversion
////===----------------------------------------------------------------------===//
//
// LogicalResult SizeOfOpLLVMLowering::matchAndRewrite(
//    Op op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const {
//  auto opaque = LLVM::LLVMPointerType::get(getContext());
//  auto null = rewriter.create<LLVM::ZeroOp>(op.getLoc(), opaque).getResult();
//  auto offset = rewriter.create<LLVM::GEPOp>(
//      op.getLoc(), opaque, convertType(op.getType()), null,
//      ArrayRef<LLVM::GEPArg>({LLVM::GEPArg(1)}));
//  rewriter.replaceOpWithNewOp<LLVM::PtrToIntOp>(
//      op, convertType(rewriter.getIndexType()), offset.getRes());
//  return success();
//}

void xblang::xb::populateXBLangGPUToSPIRVConversionPatterns(
    TypeConverter &typeConverter, RewritePatternSet &patterns) {
  //  populateXBLangToCF(typeConverter, patterns);
  //  patterns.add<CastOpToLLVM, CollapseMemRefOpLLVMLowering,
  //               GetElementOpLLVMLowering, NullptrOpSPIRVLowering,
  //               ReturnOpGPUFuncLowering, SizeOfOpLLVMLowering>(
  //      typeConverter, patterns.getContext());
}
