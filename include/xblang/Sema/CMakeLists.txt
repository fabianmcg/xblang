set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name Sema)
add_public_tablegen_target(XBLangSemaPassesIncGen)

add_mlir_doc(Passes SemaPasses ./ -gen-pass-doc)
