set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name Syntax)
add_public_tablegen_target(XBSyntaxPassesIncGen)

add_mlir_doc(Passes SyntaxPasses ./ -gen-pass-doc)
