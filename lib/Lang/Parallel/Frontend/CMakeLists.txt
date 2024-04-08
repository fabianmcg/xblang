set(LLVM_LINK_COMPONENTS Support Option)

add_xblang_library(ParallelFrontend Options.cpp)

llvm_update_compile_flags(ParallelFrontend)
