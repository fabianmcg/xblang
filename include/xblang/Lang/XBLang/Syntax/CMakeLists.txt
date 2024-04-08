set(LLVM_TARGET_DEFINITIONS Syntax.td)
xblang_tablegen_new(XBLangLexer.h.inc --gen-lexer-decls --format)
xblang_tablegen_new(XBLangLexer.cpp.inc --gen-lexer-defs --format)
add_public_tablegen_target(XBLangLexerInc)

set(LLVM_TARGET_DEFINITIONS Syntax.td)
xblang_tablegen_new(XBLangParser.h.inc --gen-parser-decls --format)
xblang_tablegen_new(XBLangParser.cpp.inc --gen-parser-defs --format)
add_public_tablegen_target(XBLangParserInc)

add_dependencies(XBLangLexerInc MLIRXBLANGEnums)
add_dependencies(XBLangParserInc XBLangLexerInc)
