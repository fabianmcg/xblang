set(LLVM_OPTIONAL_SOURCES Symbol.cpp Concept.cpp)

add_xblang_library(
  XBLangSymbol
  SymbolTable.cpp
  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/xblang/Interfaces
  DEPENDS
  XBInterfacesSymbolIncGen
  LINK_LIBS
  PUBLIC
  MLIRIR
)

add_xblang_library(
  XBLangConceptInterface
  Concept.cpp
  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/xblang/Interfaces
  DEPENDS
  XBInterfacesConceptIncGen
  LINK_LIBS
  PUBLIC
  MLIRIR
)
