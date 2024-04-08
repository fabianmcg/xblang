add_xb_unittest(XBLangLexTests
  Lex.cpp
)

target_link_libraries(XBLangLexTests
  PRIVATE
  XBSyntaxDialect
  XBLangBasic
  XBLangSyntax
  MLIRIR
  MLIRInferTypeOpInterface
  XBSyntaxLexGen
  XBSyntaxParserGen
)
