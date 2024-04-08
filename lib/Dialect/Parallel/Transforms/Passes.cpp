#include "xblang/Dialect/Parallel/Transforms/Passes.h"

#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVM/NVVM/Target.h"
#include "mlir/Target/LLVM/ROCDL/Target.h"
#include "mlir/Target/LLVMIR/Dialect/GPU/GPUToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/NVVM/NVVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/OpenMP/OpenMPToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/ROCDL/ROCDLToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/SPIRV/SPIRVToLLVMIRTranslation.h"
#include "mlir/Target/SPIRV/Target.h"
#include "xblang/Dialect/Parallel/IR/Parallel.h"
#include "xblang/Dialect/XBLang/IR/XBLang.h"

void mlir::par::populateConcretizationPasses(mlir::PassManager &pm,
                                             ParOptions options) {
  pm.addPass(createParallelConcretizer({options}));
}

void mlir::par::populateLoweringTransformsPasses(mlir::PassManager &pm,
                                                 ParOptions options) {
  if (options.isOffload()) {
    MLIRContext *ctx = pm.getContext();
    mlir::NVVM::registerNVVMTargetInterfaceExternalModels(*ctx);
    mlir::ROCDL::registerROCDLTargetInterfaceExternalModels(*ctx);
    mlir::spirv::registerSPIRVTargetInterfaceExternalModels(*ctx);
    pm.addPass(createGpuLauchSinkIndexComputationsPass());
    pm.addPass(createGpuKernelOutliningPass());
    pm.addPass(mlir::par::createGpuNameMangling());
    mlir::OpPassManager &optPM = pm.nest<mlir::gpu::GPUModuleOp>();
    optPM.addPass(createGPUTransforms({options}));
    optPM.addPass(par::createParGPUToLLVM({options}));
  }
}

void mlir::par::populateTransformationPasses(mlir::PassManager &pm,
                                             ParOptions options) {
  //  if (!options.isSequential())
  //    pm.addPass(createParallelRuntime({options}));
  mlir::OpPassManager &optPM = pm.nest<xblang::xb::FunctionOp>();
  //  if (options.isOffload())
  //    optPM.addPass(createPromoteStackToMem({options}));
  optPM.addPass(createParallelTransforms({options}));
}

void mlir::par::populateLLVMLoweringPasses(mlir::PassManager &pm,
                                           ParOptions options) {
  MLIRContext *ctx = pm.getContext();
  if (options.isOffload()) {
    mlir::registerNVVMDialectTranslation(*ctx);
    mlir::registerROCDLDialectTranslation(*ctx);
    mlir::registerSPIRVDialectTranslation(*ctx);
    mlir::registerGPUDialectTranslation(*ctx);
    DialectRegistry registry;
    mlir::gpu::registerOffloadingLLVMTranslationInterfaceExternalModels(
        registry);
    ctx->appendDialectRegistry(registry);
    StringRef offKind = options.emitLLVM() ? "llvm" : "bin";

    pm.addPass(
        mlir::createGpuModuleToBinaryPass(mlir::GpuModuleToBinaryPassOptions{
            gpu::SelectObjectAttr::get(ctx, nullptr),
            options.getToolkitPath().str(), options.getLinkLibs(),
            options.getToolOpts().str(), offKind.str()}));
    pm.addPass(par::createOffloadMergeBinaries());
  } else if (options.isHost())
    mlir::registerOpenMPDialectTranslation(*ctx);
}
