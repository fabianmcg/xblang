option(XBLANG_ENABLE_LOCAL_LIT "Enable local LIT" ON)
option(XBLANG_LIT_EXTERNAL_UTILS "Use LLVM utils in path for LIT" ON)

if(XBLANG_ENABLE_TEST)
  if(XBLANG_ENABLE_LOCAL_LIT)
    set(XBLANG_LOCAL_LIT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/.lit)
    configure_file(
      ${CMAKE_CURRENT_SOURCE_DIR}/llvm-lit.in
      ${CMAKE_CURRENT_SOURCE_DIR}/llvm-lit USE_SOURCE_PERMISSIONS @ONLY)
    set(LLVM_EXTERNAL_LIT ${CMAKE_CURRENT_SOURCE_DIR}/llvm-lit)
    set(LLVM_DEFAULT_EXTERNAL_LIT ${LLVM_EXTERNAL_LIT})
    add_custom_command(
      OUTPUT ${XBLANG_LOCAL_LIT_PATH}
      COMMAND pip3 install lit --target=${XBLANG_LOCAL_LIT_PATH}
              --ignore-installed --upgrade
      VERBATIM)
    add_custom_target(testing-local-lit ALL DEPENDS ${XBLANG_LOCAL_LIT_PATH}
                                                    ${LLVM_EXTERNAL_LIT})
  endif()
  configure_lit_site_cfg(
    ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
    ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py MAIN_CONFIG
    ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py)
  configure_lit_site_cfg(
    ${CMAKE_CURRENT_SOURCE_DIR}/Unit/lit.site.cfg.py.in
    ${CMAKE_CURRENT_BINARY_DIR}/Unit/lit.site.cfg.py MAIN_CONFIG
    ${CMAKE_CURRENT_SOURCE_DIR}/Unit/lit.cfg.py)
  if(XBLANG_LIT_EXTERNAL_UTILS)
    add_lit_testsuite(
      check-xblang
      "Verify dev passes perform correctly"
      ${CMAKE_CURRENT_BINARY_DIR}
      DEPENDS
      xb-syntax-opt
      ARGS
      -v
      -DFileCheck=FileCheck
      -Dnot=not
      -Dcount=count)
  else()
    add_lit_testsuite(
      check-xblang
      "Verify dev passes perform correctly"
      ${CMAKE_CURRENT_BINARY_DIR}
      DEPENDS
      xb-syntax-opt
      ARGS
      -v)
  endif()

  if(XBLANG_ENABLE_LOCAL_LIT)
    add_dependencies(check-xblang testing-local-lit)
  endif()

  set_target_properties(check-xblang PROPERTIES FOLDER "Tests")
endif()
