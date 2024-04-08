add_mlir_dialect(SyntaxOps syntax)
add_mlir_doc(SyntaxOps SyntaxOps Dialects/ -gen-op-doc)

set(LLVM_TARGET_DEFINITIONS SyntaxOps.td)
mlir_tablegen(SyntaxOpsEnums.h.inc -gen-enum-decls)
mlir_tablegen(SyntaxOpsEnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(XBSyntaxOpsEnumsGen)

set(LLVM_TARGET_DEFINITIONS SyntaxOps.td)
mlir_tablegen(
  SyntaxOpsAttributes.h.inc -gen-attrdef-decls -attrdefs-dialect=syntax
)
mlir_tablegen(
  SyntaxOpsAttributes.cpp.inc -gen-attrdef-defs -attrdefs-dialect=syntax
)
add_public_tablegen_target(XBSyntaxOpsAttributesIncGen)
