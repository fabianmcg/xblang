set(LLVM_TARGET_DEFINITIONS Syntax.td)
xblang_tablegen_new(
  GPUParser.h.inc --gen-parser-decls --format --parser-name=GPUParser
)
xblang_tablegen_new(
  GPUParser.cpp.inc --gen-parser-defs --format --parser-name=GPUParser
)
add_public_tablegen_target(GPUParserInc)

add_dependencies(GPUParserInc XBLangLexerInc)

set(LLVM_OPTIONAL_SOURCES Syntax.cpp Plugin.cpp)

add_xblang_library(
  GPUExtension
  Extension.cpp
  Syntax.cpp
  DEPENDS
  GPUParserInc
  LINK_LIBS
  PUBLIC
  XBLangSyntax
  MLIRIR
  MLIRGPUDialect
  MLIRIndexDialect
)

add_llvm_library(
  GPUPlugin
  MODULE
  BUILDTREE_ONLY
  Plugin.cpp
  DEPENDS
  GPUParserInc
  PLUGIN_TOOL
  xblang
  LINK_LIBS
  GPUExtension
)
