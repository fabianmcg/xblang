add_xblang_library(
  XBLangXBGSema
  Decl.cpp
  Expr.cpp
  Stmt.cpp
  Type.cpp
  Sema.cpp
  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/xblang/Lang/XBLang/Sema
  DEPENDS
  LINK_LIBS
  PUBLIC
  XBXBGDialect
  XBLangSema
  MLIRUBDialect
)
