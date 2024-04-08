#include "xblang/Frontend/MLIRPipeline.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/OpenMP/OpenMPToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Transforms/Passes.h"
#include "xblang/Codegen/Passes.h"
#include "xblang/Dialect/Parallel/Transforms/Passes.h"
#include "xblang/Dialect/XBLang/Concretization/Concretization.h"
#include "xblang/Dialect/XBLang/IR/XBLang.h"
#include "xblang/Dialect/XBLang/Lowering/Type.h"
#include "xblang/Dialect/XBLang/Transforms/Passes.h"
#include "xblang/Frontend/CompilerInstance.h"
#include "xblang/Frontend/CompilerInvocation.h"
#include "xblang/Sema/Passes.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/TargetSelect.h"

using namespace xblang;

MLIRPipeline::MLIRPipeline(CompilerInstance &ci) : ci(ci) {
  pm = std::unique_ptr<mlir::PassManager>(
      new mlir::PassManager(&ci.getMLIRContext()));
  ci.getMLIRContext().disableMultithreading(true);
}

MLIRPipeline::~MLIRPipeline() = default;

int MLIRPipeline::generateLLVM() {
  auto &invocation = ci.getInvocation();
  llvm::LLVMContext llvmContext;
  auto llvmModule = mlir::translateModuleToLLVMIR(ci.getModule(), llvmContext);
  if (!llvmModule) {
    llvm::errs() << "Failed to emit LLVM IR\n";
    return -1;
  }
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();
  auto tmBuilderOrError = llvm::orc::JITTargetMachineBuilder::detectHost();
  if (!tmBuilderOrError) {
    llvm::errs() << "Could not create JITTargetMachineBuilder\n";
    return -1;
  }

  auto tmOrError = tmBuilderOrError->createTargetMachine();
  if (!tmOrError) {
    llvm::errs() << "Could not create the target machine\n";
    return -1;
  }
  mlir::ExecutionEngine::setupTargetTripleAndDataLayout(llvmModule.get(),
                                                        tmOrError.get().get());
  if (invocation.getOptLevel() != CompilerInvocation::O0) {
    auto optPipeline = mlir::makeOptimizingTransformer(
        invocation.getOptLevel(), /*sizeLevel=*/0,
        /*targetMachine=*/nullptr);
    if (auto err = optPipeline(llvmModule.get())) {
      llvm::errs() << "Failed to optimize LLVM IR " << err << "\n";
      return -1;
    }
  }
  if (invocation.dump())
    llvm::errs() << *llvmModule << "\n";
  else if (invocation.getOutputFile().size()) {
    auto file = ci.getOutput(invocation.getOutputFile());
    if (file)
      llvmModule->print(*file, nullptr);
  }
  return 0;
}

int MLIRPipeline::run() {
  auto invocation = ci.getInvocation().getCLOpts();
  int error = 0;
  if (invocation.runStage(CompilerInvocation::Sema)) {
    pm->addPass(xblang::sema::createSemaChecker());
  }
  if (invocation.runStage(CompilerInvocation::CodeGen)) {
    pm->addPass(xblang::codegen::createCodegen());
  }
  //  pm->addPass(mlir::createCanonicalizerPass());
  if (invocation.runStage(CompilerInvocation::HighIRTransforms)) {
    mlir::par::populateTransformationPasses(*pm);
  }
  if (invocation.runStage(CompilerInvocation::HighIRConcretization)) {
    //     mlir::par::populateConcretizationPasses(*pm);
    xblang::xb::populateConcretizationPasses(ci.getTypeContext(), *pm);
    if (invocation.runOptStage(CompilerInvocation::OptHighIR)) {
      mlir::OpPassManager &optPM = pm->nest<xblang::xb::FunctionOp>();
      optPM.addPass(mlir::createCanonicalizerPass());
    }
  }
  if (invocation.runStage(CompilerInvocation::HighIRLowering)) {
    //     if (xblang::par::ParOptions().isHost())
    //       ci.getMLIRContext().getOrLoadDialect<mlir::omp::OpenMPDialect>();
    pm->addPass(xblang::xb::createXBLangLowering());
    //     pm->addPass(mlir::memref::createNormalizeMemRefsPass());
    if (invocation.runOptStage(CompilerInvocation::OptLowIR)) {
      mlir::OpPassManager &optPM = pm->nest<mlir::func::FuncOp>();
      optPM.addPass(mlir::createCanonicalizerPass());
      optPM.addPass(mlir::createCSEPass());
      //       optPM.addPass(mlir::affine::createLoopFusionPass());
      //       optPM.addPass(mlir::affine::createAffineScalarReplacementPass());
    }
  }
  if (invocation.runStage(CompilerInvocation::LowIRTransforms)) {
    mlir::registerBuiltinDialectTranslation(ci.getMLIRContext());
    mlir::registerLLVMDialectTranslation(ci.getMLIRContext());
    mlir::registerOpenMPDialectTranslation(ci.getMLIRContext());
    mlir::par::populateLoweringTransformsPasses(*pm);
    mlir::OpPassManager &optPM = pm->nest<mlir::func::FuncOp>();
    optPM.addPass(xblang::xb::createAllocaToEntry());
    if (invocation.runOptStage(CompilerInvocation::OptLLVM)) {
      pm->addPass(mlir::createCanonicalizerPass());
      pm->addPass(mlir::createCSEPass());
    }
  }
  if (invocation.runStage(CompilerInvocation::LowIRLowering)) {
    pm->addPass(xblang::xb::createXBLangToLLVM());
    mlir::par::populateLLVMLoweringPasses(*pm);
    pm->addPass(mlir::createCanonicalizerPass());
    if (invocation.runOptStage(CompilerInvocation::OptLLVM)) {
      pm->addPass(mlir::createCSEPass());
    }
  }
  if (mlir::failed(pm->run(ci.getModule()))) {
    ci.getModule()->dump();
    return 2;
  }
  if (mlir::failed(mlir::verify(ci.getModule(), true))) {
    llvm::errs() << "Verification failed.";
    return 3;
  }
  if (invocation.isBetweenStages(CompilerInvocation::Sema,
                                 CompilerInvocation::LowIRLowering)) {
    if (invocation.dump())
      ci.getModule()->dump();
    else if (invocation.getOutputFile().size()) {
      auto file = ci.getOutput(invocation.getOutputFile());
      if (file)
        ci.getModule().print(*file);
    }
  }
  if (invocation.runStage(CompilerInvocation::LLVMIR))
    if ((error = generateLLVM()))
      return error;
  return error;
}
