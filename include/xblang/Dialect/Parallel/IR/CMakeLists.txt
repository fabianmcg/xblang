add_mlir_dialect(Parallel par)
add_mlir_doc(Parallel Parallel Parallel/ -gen-dialect-doc)

set(LLVM_TARGET_DEFINITIONS Enums.td)
mlir_tablegen(ParallelEnums.h.inc -gen-enum-decls --dialect=par)
mlir_tablegen(ParallelEnums.cpp.inc -gen-enum-defs --dialect=par)
add_public_tablegen_target(MLIRParallelEnums)

set(LLVM_TARGET_DEFINITIONS Parallel.td)
mlir_tablegen(ParallelAttributes.h.inc -gen-attrdef-decls -attrdefs-dialect=par)
mlir_tablegen(
  ParallelAttributes.cpp.inc -gen-attrdef-defs -attrdefs-dialect=par
)
add_public_tablegen_target(MLIRParallelAttributes)
