add_mlir_dialect(XLGOps xlg)
add_mlir_doc(XLGOps XLGOps Dialects/ -gen-op-doc)

set(LLVM_TARGET_DEFINITIONS XLGOps.td)
mlir_tablegen(XLGOpsEnums.h.inc -gen-enum-decls)
mlir_tablegen(XLGOpsEnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(XBXLGOpsEnumsGen)

set(LLVM_TARGET_DEFINITIONS XLGOps.td)
mlir_tablegen(XLGOpsAttributes.h.inc -gen-attrdef-decls -attrdefs-dialect=xlg)
mlir_tablegen(XLGOpsAttributes.cpp.inc -gen-attrdef-defs -attrdefs-dialect=xlg)
add_public_tablegen_target(XBXLGOpsAttributesIncGen)
