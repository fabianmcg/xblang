set(LLVM_TARGET_DEFINITIONS Syntax.td)
xblang_tablegen_new(
  OMPParser.h.inc --gen-parser-decls --format --parser-name=OMPParser
)
xblang_tablegen_new(
  OMPParser.cpp.inc --gen-parser-defs --format --parser-name=OMPParser
)
add_public_tablegen_target(OMPParserInc)

add_dependencies(OMPParserInc XBLangLexerInc)

set(LLVM_OPTIONAL_SOURCES Syntax.cpp Plugin.cpp)

add_xblang_library(
  OMPExtension
  Extension.cpp
  Syntax.cpp
  DEPENDS
  OMPParserInc
)

add_llvm_library(
  OMPPlugin
  MODULE
  BUILDTREE_ONLY
  Plugin.cpp
  DEPENDS
  OMPParserInc
  PLUGIN_TOOL
  xblang
  LINK_LIBS
  OMPExtension
)
