add_xblang_library(
  XBLangTransforms
  AllocaToEntry.cpp
  Common.cpp
  ImplicitCastConcretizer.cpp
  Type.cpp
  XBToLLVM.cpp
  XBToStd.cpp
  XBToCf.cpp
  XBToSPIRV.cpp
  DEPENDS
  MLIRXBLangTransforms
  LINK_LIBS
  PUBLIC
  MLIRAnalysis
  MLIRArithDialect
  MLIRBufferizationDialect
  MLIRBufferizationTransforms
  MLIRFuncDialect
  MLIRMathDialect
  MLIRArithToLLVM
  MLIRMathToLLVM
  MLIRIndexToLLVM
  MLIRGPUToGPURuntimeTransforms
  MLIRMemRefToLLVM
  MLIRSCFToControlFlow
  MLIRIndexDialect
  MLIRLLVMCommonConversion
  MLIRFuncTransforms
  MLIRInferIntRangeInterface
  MLIRIR
  MLIRMemRefDialect
  MLIRPass
  MLIRTransforms
  MLIRTransformUtils
  MLIRVectorDialect
  MLIRXBLang
  MLIRParallel
  ParallelTransforms
  XBLangSema
)
