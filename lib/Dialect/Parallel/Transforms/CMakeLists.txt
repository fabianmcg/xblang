add_xblang_library(
  ParallelTransforms
  Concretizer.cpp
  LowerToLLVM.cpp
  LowerToStd.cpp
  NameMangling.cpp
  OffloadMergeBinaries.cpp
  Passes.cpp
  ParallelTransforms.cpp
  ParGPUToLLVM.cpp
  PromoteStackToMem.cpp
  DEPENDS
  MLIRParallelTransforms
  LINK_LIBS
  PUBLIC
  MLIRAnalysis
  MLIRArithDialect
  MLIRBufferizationDialect
  MLIRBufferizationTransforms
  MLIRFuncDialect
  MLIRFuncTransforms
  MLIRInferIntRangeInterface
  MLIRIR
  MLIRMemRefDialect
  MLIRPass
  MLIRTransforms
  MLIRTransformUtils
  MLIRVectorDialect
  MLIRParallel
  MLIRXBAnalysis
  MLIRSPIRVToLLVMIRTranslation
  MLIRFuncDialect
  MLIRMathDialect
  MLIRArithToLLVM
  MLIRMathToLLVM
  MLIRIndexToLLVM
  MLIRGPUToGPURuntimeTransforms
  MLIRMemRefToLLVM
  MLIRSCFToControlFlow
  MLIRIndexDialect
  MLIROpenMPDialect
  MLIRGPUToROCDLTransforms
  MLIROpenMPToLLVMIRTranslation
)

llvm_update_compile_flags(ParallelTransforms)
