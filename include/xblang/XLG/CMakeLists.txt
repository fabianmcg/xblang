add_subdirectory(IR)

set(LLVM_TARGET_DEFINITIONS Interfaces.td)
mlir_tablegen(XLGInterfaces.h.inc -gen-op-interface-decls)
mlir_tablegen(XLGInterfaces.cpp.inc -gen-op-interface-defs)
add_public_tablegen_target(XBXLGInterfacesIncGen)

set(LLVM_TARGET_DEFINITIONS Concepts.td)
xblang_tablegen_new(XLGConcepts.h.inc -gen-concept-decls --format)
xblang_tablegen_new(XLGConcepts.cpp.inc -gen-concept-defs --format)
add_public_tablegen_target(XBXLGConceptsIncGen)
