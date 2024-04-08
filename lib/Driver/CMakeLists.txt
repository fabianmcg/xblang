set(LLVM_OPTIONAL_SOURCES main.cpp Driver.cpp)

set(LLVM_LINK_COMPONENTS Support Option)

add_xblang_library(XBLangDriver Driver.cpp LINK_LIBS PUBLIC XBLangFrontend)

add_llvm_executable(xblang main.cpp)

if(XBLANG_ENABLE_PLUGINS)
  export_executable_symbols_for_plugins(xblang)
endif()

llvm_update_compile_flags(xblang)

target_link_libraries(xblang PRIVATE XBLangDriver)

install(TARGETS xblang DESTINATION bin)
