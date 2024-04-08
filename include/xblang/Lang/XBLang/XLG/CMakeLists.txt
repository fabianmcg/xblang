set(LLVM_TARGET_DEFINITIONS XBGDialect.td)
mlir_tablegen(XBGTypes.h.inc -gen-typedef-decls -typedefs-dialect=xbg)
mlir_tablegen(XBGTypes.cpp.inc -gen-typedef-defs -typedefs-dialect=xbg)
mlir_tablegen(XBGDialect.h.inc -gen-dialect-decls -dialect=xbg)
mlir_tablegen(XBGDialect.cpp.inc -gen-dialect-defs -dialect=xbg)
add_public_tablegen_target(XBXBGIncGen)

set(LLVM_TARGET_DEFINITIONS XBGDecl.td)
mlir_tablegen(XBGDecl.h.inc -gen-op-decls)
mlir_tablegen(XBGDecl.cpp.inc -gen-op-defs)
xblang_tablegen_new(
  XBGConceptDecl.h.inc -gen-concept-decls -concept-class=XBGDecl -format
)
xblang_tablegen_new(
  XBGConceptDecl.cpp.inc -gen-concept-defs -concept-class=XBGDecl -format
)
add_public_tablegen_target(XBXBGDeclIncGen)

set(LLVM_TARGET_DEFINITIONS XBGType.td)
mlir_tablegen(XBGType.h.inc -gen-op-decls)
mlir_tablegen(XBGType.cpp.inc -gen-op-defs)
xblang_tablegen_new(
  XBGConceptType.h.inc -gen-concept-decls -concept-class=XBGType -format
)
xblang_tablegen_new(
  XBGConceptType.cpp.inc -gen-concept-defs -concept-class=XBGType -format
)
add_public_tablegen_target(XBXBGTypeIncGen)

set(LLVM_TARGET_DEFINITIONS XBGStmt.td)
mlir_tablegen(XBGStmt.h.inc -gen-op-decls)
mlir_tablegen(XBGStmt.cpp.inc -gen-op-defs)
xblang_tablegen_new(
  XBGConceptStmt.h.inc -gen-concept-decls -concept-class=XBGStmt -format
)
xblang_tablegen_new(
  XBGConceptStmt.cpp.inc -gen-concept-defs -concept-class=XBGStmt -format
)
add_public_tablegen_target(XBXBGStmtIncGen)

set(LLVM_TARGET_DEFINITIONS XBGExpr.td)
mlir_tablegen(XBGExpr.h.inc -gen-op-decls --op-include-regex="xbg.*expr")
mlir_tablegen(XBGExpr.cpp.inc -gen-op-defs --op-include-regex="xbg.*expr")
xblang_tablegen_new(
  XBGConceptExpr.h.inc -gen-concept-decls -concept-class=XBGExpr -format
)
xblang_tablegen_new(
  XBGConceptExpr.cpp.inc -gen-concept-defs -concept-class=XBGExpr -format
)
add_public_tablegen_target(XBXBGExprIncGen)

add_mlir_doc(XBGOps XBGOps Dialects/ -gen-op-doc)
