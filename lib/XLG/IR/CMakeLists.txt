add_xblang_library(
  XBXLGDialect
  XLG.cpp
  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/xblang/XLG/IR
  DEPENDS
  MLIRXLGOpsIncGen
  XBXLGOpsEnumsGen
  XBXLGOpsAttributesIncGen
  LINK_LIBS
  PUBLIC
  MLIRIR
  MLIRControlFlowInterfaces
  XBXLGConcepts
  XBLangSymbol
)
