add_xblang_library(XBLangSyntax LexerBase.cpp Syntax.cpp)

add_subdirectory(IR)
add_subdirectory(LexGen)
add_subdirectory(ParserGen)
add_subdirectory(Transforms)
add_subdirectory(Utils)

target_compile_options(
  XBLangSyntax PRIVATE -Wno-unused-local-typedef -Wno-unused-variable
                       -Wno-unused-label
)
