add_mlir_dialect(XBLang xb)
add_mlir_doc(XBLang XBLang XBLang/ -gen-dialect-doc)

set(LLVM_TARGET_DEFINITIONS Enums.td)
mlir_tablegen(XBLangEnums.h.inc -gen-enum-decls)
mlir_tablegen(XBLangEnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(MLIRXBLANGEnums)

set(LLVM_TARGET_DEFINITIONS XBLang.td)
mlir_tablegen(XBLangAttributes.h.inc -gen-attrdef-decls)
mlir_tablegen(XBLangAttributes.cpp.inc -gen-attrdef-defs)
add_public_tablegen_target(MLIRXBLANGAttributes)

set(LLVM_TARGET_DEFINITIONS Interfaces.td)
mlir_tablegen(XBLangInterfaces.h.inc -gen-op-interface-decls)
mlir_tablegen(XBLangInterfaces.cpp.inc -gen-op-interface-defs)
add_public_tablegen_target(MLIRXBLANGInterfaces)
