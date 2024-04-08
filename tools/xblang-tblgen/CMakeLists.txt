set(LLVM_LINK_COMPONENTS Support TableGen)

message(STATUS "Using CURRENT_SOURCE_DIR: ${CMAKE_CURRENT_SOURCE_DIR}")

add_tablegen(
  xblang-tblgen-new
  XBLANG_NEW
  DESTINATION
  "${XBLANG_TOOLS_INSTALL_DIR}"
  EXPORT
  XBLANG
  xblang-tblgen.cpp
  LexerGen.cpp
  ParserGen.cpp
  ConceptGen.cpp
  TemplateEngine.cpp
)

target_link_libraries(
  xblang-tblgen-new
  PRIVATE MLIRTableGen
          XBSyntaxDialect
          XBSyntaxLexGen
          XBSyntaxParserGen
          XBSyntaxTransforms
          XBLangBasic
          XBLangSyntax
          XBLangSupport
          MLIRIR
          MLIRInferTypeOpInterface
          MLIRPass
          MLIRRewrite
          MLIRTransforms
)

set_target_properties(xblang-tblgen-new PROPERTIES FOLDER "Tablegenning")
