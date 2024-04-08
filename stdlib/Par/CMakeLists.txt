option(PAR_CUDA_RUNTIME "Enable CUDA runtime" OFF)
option(PAR_ROCM_RUNTIME "Enable ROCM runtime" OFF)
option(PAR_MAP_HOST "Map memory" ON)

if(PAR_CUDA_RUNTIME)
  find_package(CUDAToolkit REQUIRED)

  add_library(
    par_runtime
    STATIC
    Runtime.cpp
    ParRuntime.cpp)
  target_link_libraries(par_runtime PRIVATE CUDA::cudart)
  if (PAR_MAP_HOST)
    target_compile_definitions(par_runtime PRIVATE PAR_CUDA_BACKEND PAR_MAP_HOST=1)
  else()
    target_compile_definitions(par_runtime PRIVATE PAR_CUDA_BACKEND PAR_MAP_HOST=0)
  endif()
  install(TARGETS par_runtime LIBRARY DESTINATION lib)
  set_target_properties(par_runtime
    PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
  )

elseif(PAR_ROCM_RUNTIME)
  find_package(hip REQUIRED)

  add_library(
    par_runtime
    STATIC
    Parallel.cpp
    ParRuntime.cpp)
  target_link_libraries(par_runtime PRIVATE hip::amdhip64)
  if (PAR_MAP_HOST)
    target_compile_definitions(par_runtime PRIVATE PAR_HIP_BACKEND PAR_MAP_HOST=1)
  else()
    target_compile_definitions(par_runtime PRIVATE PAR_HIP_BACKEND PAR_MAP_HOST=0)
  endif()
  install(TARGETS par_runtime LIBRARY DESTINATION lib)
  set_target_properties(par_runtime
    PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
    LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib
  )

endif()
