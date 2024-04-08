include_directories(${XBLANG_SOURCE_DIR}/extensions)
include_directories(${XBLANG_BINARY_DIR}/extensions)

add_subdirectory(gpu)
add_subdirectory(omp)
