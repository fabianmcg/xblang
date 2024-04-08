set_target_properties(XBLangUnitTests PROPERTIES FOLDER "XBLang Tests")

function(add_xblang_unittest test_suite test_name)
  if (NOT DEFINED LLVM_REQUIRES_RTTI)
    set(LLVM_REQUIRES_RTTI OFF)
  endif()

  list(APPEND LLVM_LINK_COMPONENTS Support) # gtest needs it for raw_ostream
  add_llvm_executable(${test_name} IGNORE_EXTERNALIZE_DEBUGINFO NO_INSTALL_RPATH ${ARGN})

  target_link_options(${test_name} PRIVATE "${LLVM_UNITTEST_LINK_FLAGS}")

  set(outdir ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_CFG_INTDIR})
  set_output_directory(${test_name} BINARY_DIR ${outdir} LIBRARY_DIR ${outdir})
  target_link_libraries(${test_name} PRIVATE  GTest::gtest_main ${LLVM_PTHREAD_LIB})

  add_dependencies(${test_suite} ${test_name})
  get_target_property(test_suite_folder ${test_suite} FOLDER)
  if (test_suite_folder)
    set_property(TARGET ${test_name} PROPERTY FOLDER "${test_suite_folder}")
  endif ()
endfunction()

function(add_xb_unittest test_dirname)
  add_xblang_unittest(XBLangUnitTests ${test_dirname} ${ARGN})
endfunction()

add_subdirectory(Basic)
add_subdirectory(SyntaxGen)
