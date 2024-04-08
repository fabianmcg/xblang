get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LLVM_LINK_COMPONENTS Support)

set(LIBS XBSyntaxDialect XBSyntaxTransforms MLIROptLib)

add_llvm_executable(xb-syntax-opt syntax-opt.cpp)

target_link_libraries(xb-syntax-opt PRIVATE ${LIBS})

llvm_update_compile_flags(xb-syntax-opt)
