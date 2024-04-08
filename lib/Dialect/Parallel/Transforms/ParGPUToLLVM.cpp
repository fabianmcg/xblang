#include "mlir/Conversion/AMDGPUToROCDL/AMDGPUToROCDL.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/GPUToNVVM/GPUToNVVMPass.h"
#include "mlir/Conversion/GPUToROCDL/GPUToROCDLPass.h"
#include "mlir/Conversion/GPUToROCDL/Runtimes.h"
#include "mlir/Conversion/IndexToLLVM/IndexToLLVM.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "xblang/ADT/DoubleTypeSwitch.h"
#include "xblang/Dialect/Parallel/IR/Parallel.h"
#include "xblang/Dialect/Parallel/Transforms/Passes.h"
#include "xblang/Dialect/XBLang/IR/XBLang.h"
#include "xblang/Dialect/XBLang/Lowering/Type.h"
#include "xblang/Dialect/XBLang/Transforms/Passes.h"

using namespace mlir;
using namespace mlir::par;
using namespace xblang::xb;

namespace mlir {
namespace par {
#define GEN_PASS_DEF_PARGPUTOLLVM
#include "xblang/Dialect/Parallel/Transforms/Passes.h.inc"
} // namespace par
} // namespace mlir

namespace {
IntegerAttr wrapNumericMemorySpace(MLIRContext *ctx, unsigned space) {
  return IntegerAttr::get(IntegerType::get(ctx, 64), space);
}

using MemorySpaceMapping =
    std::function<unsigned(gpu::AddressSpace gpuAddressSpace)>;

void populateGpuMemorySpaceAttributeConversions(
    TypeConverter &typeConverter, const MemorySpaceMapping &mapping) {
  typeConverter.addTypeAttributeConversion(
      [mapping](BaseMemRefType type, gpu::AddressSpaceAttr memorySpaceAttr) {
        gpu::AddressSpace memorySpace = memorySpaceAttr.getValue();
        unsigned addressSpace = mapping(memorySpace);
        return wrapNumericMemorySpace(memorySpaceAttr.getContext(),
                                      addressSpace);
      });
  typeConverter.addTypeAttributeConversion(
      [mapping](xblang::xb::PointerType type,
                gpu::AddressSpaceAttr memorySpaceAttr) {
        gpu::AddressSpace memorySpace = memorySpaceAttr.getValue();
        unsigned addressSpace = mapping(memorySpace);
        return wrapNumericMemorySpace(memorySpaceAttr.getContext(),
                                      addressSpace);
      });
  typeConverter.addTypeAttributeConversion(
      [mapping](xblang::xb::ReferenceType type,
                gpu::AddressSpaceAttr memorySpaceAttr) {
        gpu::AddressSpace memorySpace = memorySpaceAttr.getValue();
        unsigned addressSpace = mapping(memorySpace);
        return wrapNumericMemorySpace(memorySpaceAttr.getContext(),
                                      addressSpace);
      });
  typeConverter.addTypeAttributeConversion(
      [mapping](xblang::xb::AddressType type,
                gpu::AddressSpaceAttr memorySpaceAttr) {
        gpu::AddressSpace memorySpace = memorySpaceAttr.getValue();
        unsigned addressSpace = mapping(memorySpace);
        return wrapNumericMemorySpace(memorySpaceAttr.getContext(),
                                      addressSpace);
      });
}

class ParGPUToLLVM : public mlir::par::impl::ParGPUToLLVMBase<ParGPUToLLVM> {
public:
  using Base::Base;
  void nvvm(LLVMConversionTarget &target, XBLangToLLVMTypeConverter &converter,
            RewritePatternSet &llvmPatterns);
  void rocdl(LLVMConversionTarget &target, XBLangToLLVMTypeConverter &converter,
             RewritePatternSet &llvmPatterns);
  void afterRocdl();
  void runOnOperation() final;
};
} // namespace

void ParGPUToLLVM::nvvm(LLVMConversionTarget &target,
                        XBLangToLLVMTypeConverter &converter,
                        RewritePatternSet &llvmPatterns) {
  populateGpuMemorySpaceAttributeConversions(
      converter, [](gpu::AddressSpace space) -> unsigned {
        switch (space) {
        case gpu::AddressSpace::Global:
          return static_cast<unsigned>(
              NVVM::NVVMMemorySpace::kGlobalMemorySpace);
        case gpu::AddressSpace::Workgroup:
          return static_cast<unsigned>(
              NVVM::NVVMMemorySpace::kSharedMemorySpace);
        case gpu::AddressSpace::Private:
          return 0;
        }
        llvm_unreachable("unknown address space enum value");
        return 0;
      });
  // Lowering for MMAMatrixType.
  converter.addConversion([&](gpu::MMAMatrixType type) -> Type {
    return convertMMAToLLVMType(type);
  });
  populateGpuToNVVMConversionPatterns(converter, llvmPatterns);
  populateGpuWMMAToNVVMConversionPatterns(converter, llvmPatterns);
  if (this->hasRedux)
    populateGpuSubgroupReduceOpLoweringPattern(converter, llvmPatterns);
  configureGpuToNVVMConversionLegality(target);
}

void ParGPUToLLVM::rocdl(LLVMConversionTarget &target,
                         XBLangToLLVMTypeConverter &converter,
                         RewritePatternSet &llvmPatterns) {
  populateGpuMemorySpaceAttributeConversions(
      converter, [](gpu::AddressSpace space) {
        switch (space) {
        case gpu::AddressSpace::Global:
          return 1;
        case gpu::AddressSpace::Workgroup:
          return 3;
        case gpu::AddressSpace::Private:
          return 5;
        }
        llvm_unreachable("unknown address space enum value");
        return 0;
      });

  populateVectorToLLVMConversionPatterns(converter, llvmPatterns);
  cf::populateControlFlowToLLVMConversionPatterns(converter, llvmPatterns);
  populateFuncToLLVMConversionPatterns(converter, llvmPatterns);
  populateFinalizeMemRefToLLVMConversionPatterns(converter, llvmPatterns);
  populateGpuToROCDLConversionPatterns(converter, llvmPatterns, gpu::amd::HIP);
  configureGpuToROCDLConversionLegality(target);
}

void ParGPUToLLVM::afterRocdl() {
  gpu::GPUModuleOp m = getOperation();
  MLIRContext *ctx = m.getContext();
  auto *rocdlDialect = getContext().getLoadedDialect<ROCDL::ROCDLDialect>();
  auto reqdWorkGroupSizeAttrHelper =
      rocdlDialect->getReqdWorkGroupSizeAttrHelper();
  auto flatWorkGroupSizeAttrHelper =
      rocdlDialect->getFlatWorkGroupSizeAttrHelper();
  m.walk([&](LLVM::LLVMFuncOp op) {
    if (auto blockSizes =
            op->removeAttr(gpu::GPUFuncOp::getKnownBlockSizeAttrName())
                .dyn_cast_or_null<DenseI32ArrayAttr>()) {
      reqdWorkGroupSizeAttrHelper.setAttr(op, blockSizes);
      // Also set up the rocdl.flat_work_group_size attribute to prevent
      // conflicting metadata.
      uint32_t flatSize = 1;
      for (uint32_t size : blockSizes.asArrayRef()) {
        flatSize *= size;
      }
      StringAttr flatSizeAttr =
          StringAttr::get(ctx, Twine(flatSize) + "," + Twine(flatSize));
      flatWorkGroupSizeAttrHelper.setAttr(op, flatSizeAttr);
    }
  });
}

void ParGPUToLLVM::runOnOperation() {
  gpu::GPUModuleOp m = getOperation();
  LowerToLLVMOptions options(
      m.getContext(),
      DataLayout(cast<DataLayoutOpInterface>(m.getOperation())));
  if (indexBitwidth != kDeriveIndexBitwidthFromDataLayout)
    options.overrideIndexBitwidth(indexBitwidth);
  {
    RewritePatternSet llvmPatterns(m.getContext());
    populateGpuRewritePatterns(llvmPatterns);
    if (failed(mlir::applyPatternsAndFoldGreedily(m, std::move(llvmPatterns))))
      return signalPassFailure();
  }
  xblang::xb::XBLangToLLVMTypeConverter converter(m.getContext(), options);
  RewritePatternSet llvmPatterns(m.getContext());
  arith::populateArithToLLVMConversionPatterns(converter, llvmPatterns);
  cf::populateControlFlowToLLVMConversionPatterns(converter, llvmPatterns);
  populateFuncToLLVMConversionPatterns(converter, llvmPatterns);
  populateFinalizeMemRefToLLVMConversionPatterns(converter, llvmPatterns);
  populateXBLangToLLVMConversionPatterns(converter, llvmPatterns);
  populateAffineToStdConversionPatterns(llvmPatterns);
  populateSCFToControlFlowConversionPatterns(llvmPatterns);
  index::populateIndexToLLVMConversionPatterns(converter, llvmPatterns);
  memref::populateExpandStridedMetadataPatterns(llvmPatterns);
  LLVMConversionTarget target(getContext());
  Attribute gpuTarget;
  if (opts.isNVPTX()) {
    nvvm(target, converter, llvmPatterns);
    gpuTarget = NVVM::NVVMTargetAttr::get(&getContext(), opts.getOptLevel(),
                                          opts.getTriple(), opts.getChip(),
                                          opts.getTargetFeatures(), nullptr);
  } else if (opts.isAMDGPU()) {
    rocdl(target, converter, llvmPatterns);
    gpuTarget = ROCDL::ROCDLTargetAttr::get(
        &getContext(), opts.getOptLevel(), opts.getTriple(), opts.getChip(),
        opts.getTargetFeatures(), "400", nullptr);
  }
  m.setTargetsAttr(
      ArrayAttr::get(&getContext(), ArrayRef<Attribute>(gpuTarget)));

  if (failed(applyPartialConversion(m, target, std::move(llvmPatterns))))
    signalPassFailure();
  if (opts.isAMDGPU())
    afterRocdl();
}
