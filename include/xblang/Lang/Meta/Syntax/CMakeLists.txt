set(LLVM_TARGET_DEFINITIONS Syntax.td)
xblang_tablegen_new(
  MetaParser.h.inc --gen-parser-decls --format --parser-name=MetaParser
)
xblang_tablegen_new(
  MetaParser.cpp.inc --gen-parser-defs --format --parser-name=MetaParser
)
add_public_tablegen_target(MetaParserInc)

add_dependencies(MetaParserInc XBLangLexerInc)
