add_subdirectory(IR)
add_subdirectory(Codegen)
add_subdirectory(Sema)

add_xblang_library(
  XBXLGConcepts
  Concepts.cpp
  Interfaces.cpp
  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/xblang/XLG
  DEPENDS
  XBInterfacesConceptIncGen
  XBXLGInterfacesIncGen
  XBXLGConceptsIncGen
  MLIRXLGOpsIncGen
  LINK_LIBS
  PUBLIC
  XBLangBasic
  XBLangConceptInterface
)

add_subdirectory(Utils)
