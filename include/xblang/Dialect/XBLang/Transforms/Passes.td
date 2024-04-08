#ifndef XBLANG_DIALECT_XBLANG_TRANSFORMS_PASSES_TD
#define XBLANG_DIALECT_XBLANG_TRANSFORMS_PASSES_TD

include "mlir/Pass/PassBase.td"

def XBLangConcretizer: Pass<"xblang-concretizer",
                                 "::mlir::ModuleOp"> {
}

def ImplicitCastConcretizer: Pass<"implicit-cast-concretizer",
                                 "::mlir::ModuleOp"> {
}

def XBLangLowering: Pass<"xblang-lowering",
                                 "::mlir::ModuleOp"> {
  let dependentDialects = [
    "::mlir::affine::AffineDialect",
    "::mlir::func::FuncDialect",
    "::mlir::memref::MemRefDialect",
    "::mlir::index::IndexDialect",
    "::mlir::arith::ArithDialect",
    "::mlir::scf::SCFDialect",
    "::mlir::cf::ControlFlowDialect",
    "::mlir::gpu::GPUDialect",
    "::mlir::math::MathDialect",
    "::mlir::LLVM::LLVMDialect"
  ];
}

def XBLangToLLVM: Pass<"xblang-to-llvm",
                                 "::mlir::ModuleOp"> {
  let dependentDialects = [
    "::mlir::LLVM::LLVMDialect"
  ];
}

def AllocaToEntry: Pass<"alloca-to-entry",
                                 "::mlir::func::FuncOp"> {
}

#endif
