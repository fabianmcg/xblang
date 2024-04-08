#ifndef XBLANG_DIALECT_PARALLEL_TRANSFORMS_PASSES_TD
#define XBLANG_DIALECT_PARALLEL_TRANSFORMS_PASSES_TD

include "mlir/Pass/PassBase.td"

def ParallelRuntime: Pass<"parallel-runtime",
                                 "::mlir::ModuleOp"> {
  let options = [
    Option<"opts", "opts", "::xblang::par::ParOptions",
           /*default=*/ "::xblang::par::ParOptions()",
           "Parallel front-end options">
  ];
}

def ParallelTransforms :
    Pass<"parallel-transforms", ""> {
  let options = [
    Option<"opts", "opts", "::xblang::par::ParOptions",
           /*default=*/ "::xblang::par::ParOptions()",
           "Parallel front-end options">
  ];
}

def ParallelConcretizer: Pass<"parallel-concretizer",
                                 "::mlir::ModuleOp"> {
  let options = [
    Option<"opts", "opts", "::xblang::par::ParOptions",
           /*default=*/ "::xblang::par::ParOptions()",
           "Parallel front-end options">
  ];
}

def GPUTransforms: Pass<"gpu-transforms",
                                 "::mlir::gpu::GPUModuleOp"> {
  let dependentDialects = [
    "::mlir::func::FuncDialect",
    "::mlir::memref::MemRefDialect",
    "::mlir::index::IndexDialect",
    "::mlir::arith::ArithDialect",
    "::mlir::scf::SCFDialect",
    "::mlir::LLVM::LLVMDialect",
  ];
  let options = [
    Option<"opts", "opts", "::xblang::par::ParOptions",
           /*default=*/ "::xblang::par::ParOptions()",
           "Parallel front-end options">
  ];
}

def PromoteStackToMem: Pass<"promote-stack-to-mem",
                                 "::xblang::xb::FunctionOp"> {
  let options = [
    Option<"opts", "opts", "::xblang::par::ParOptions",
           /*default=*/ "::xblang::par::ParOptions()",
           "Parallel front-end options">
  ];
}

def ParGPUToLLVM: Pass<"par-to-llvm", "::mlir::gpu::GPUModuleOp"> {
  let dependentDialects = [
    "::mlir::NVVM::NVVMDialect",
    "::mlir::ROCDL::ROCDLDialect",
    "::mlir::LLVM::LLVMDialect"
  ];
  let options = [
    Option<"opts", "opts", "::xblang::par::ParOptions",
           /*default=*/ "::xblang::par::ParOptions()",
           "Parallel front-end options">,
    Option<"indexBitwidth", "index-bitwidth", "unsigned",
           /*default=kDeriveIndexBitwidthFromDataLayout*/"0",
           "Bitwidth of the index type, 0 to use size of machine word">,
    Option<"hasRedux", "has-redux", "bool", /*default=*/"false",
           "Target gpu supports redux">,
    Option<"useOpaquePointers", "use-opaque-pointers", "bool",
                   /*default=*/"true", "Generate LLVM IR using opaque pointers "
                   "instead of typed pointers">,
  ];
}

def GpuNameMangling: Pass<"gpu-name-mangling",
                          "::mlir::ModuleOp"> {
  let summary = "Mangle the names of all the top symbols inside a gpu.module.";
  let description = [{
    Mangle the names of all the top level definitions inside a `gpu.module`
    for all the `gpu.module`s inside a `module`, from: `<symbol>` to:
    `__G<gpu module name>_S<symbol>`, and updates all the symbol references.
  }];
}

def OffloadMergeBinaries: Pass<"offload-merge-binaries",
                          "::mlir::ModuleOp"> {
  let summary = "Merge all binaries into a single binary.";
}

#endif
