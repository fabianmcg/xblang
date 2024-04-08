set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name Codegen)
add_public_tablegen_target(XBLangCodegenPassesIncGen)

add_mlir_doc(Passes CodegenPasses ./ -gen-pass-doc)
