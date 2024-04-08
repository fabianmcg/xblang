set(LLVM_OPTIONAL_SOURCES null.cpp)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LLVM_LINK_COMPONENTS Core Support AsmParser)

set(LIBS
    MLIRAnalysis
    MLIRDialect
    MLIROptLib
    MLIRParser
    MLIRPass
    MLIRTransforms
    MLIRTransformUtils
    MLIRSupport
    MLIRIR
    MLIRXBLang
    MLIRParallel
    ParallelFrontend
    XBLangSema
    XBXLGConcepts
    XBXLGDialect
    XBXBGDialect
)

# Exclude from libMLIR.so because this has static options intended for opt-like
# tools only.
add_mlir_library(
  XBLangMlirOptMain mlir-opt.cpp EXCLUDE_FROM_LIBMLIR LINK_LIBS PUBLIC ${LIBS}
)

add_llvm_executable(xblang-opt mlir-opt.cpp DEPENDS XBLangMlirOptMain ${LIBS})

target_link_libraries(xblang-opt PRIVATE ${LIBS})
llvm_update_compile_flags(xblang-opt)
