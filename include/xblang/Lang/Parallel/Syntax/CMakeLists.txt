set(LLVM_TARGET_DEFINITIONS Syntax.td)
xblang_tablegen_new(
  ParParser.h.inc --gen-parser-decls --format --parser-name=ParParser
)
xblang_tablegen_new(
  ParParser.cpp.inc --gen-parser-defs --format --parser-name=ParParser
)
add_public_tablegen_target(ParParserInc)

add_dependencies(ParParserInc XBLangLexerInc)
