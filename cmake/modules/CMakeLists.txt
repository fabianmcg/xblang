include(GNUInstallPackageDir)
include(ExtendPath)
include(LLVMDistributionSupport)
include(FindPrefixFromConfig)

# Generate a list of CMake library targets so that other CMake projects can link
# against them. LLVM calls its version of this file LLVMExports.cmake, but the
# usual CMake convention seems to be ${Project}Targets.cmake.
set(XBLANG_INSTALL_PACKAGE_DIR
    "${CMAKE_INSTALL_PACKAGEDIR}/xblang"
    CACHE
      STRING
      "Path for CMake subdirectory for Polly (defaults to '${CMAKE_INSTALL_PACKAGEDIR}/polly')"
)
# CMAKE_INSTALL_PACKAGEDIR might be absolute, so don't reuse below.
set(xblang_cmake_builddir
    "${CMAKE_BINARY_DIR}/lib${LLVM_LIBDIR_SUFFIX}/cmake/xblang"
)

# Keep this in sync with llvm/cmake/CMakeLists.txt!
set(LLVM_INSTALL_PACKAGE_DIR
    "${CMAKE_INSTALL_PACKAGEDIR}/llvm"
    CACHE
      STRING
      "Path for CMake subdirectory for LLVM (defaults to '${CMAKE_INSTALL_PACKAGEDIR}/llvm')"
)
# CMAKE_INSTALL_PACKAGEDIR might be absolute, so don't reuse below.
string(REPLACE "${CMAKE_CFG_INTDIR}" "." llvm_cmake_builddir
               "${LLVM_LIBRARY_DIR}"
)
set(llvm_cmake_builddir "${llvm_cmake_builddir}/cmake/llvm")

get_property(XBLANG_EXPORTS GLOBAL PROPERTY XBLANG_EXPORTS)
export(TARGETS ${XBLANG_EXPORTS}
       FILE ${xblang_cmake_builddir}/XBLANGTargets.cmake
)

get_property(XBLANG_ALL_LIBS GLOBAL PROPERTY XBLANG_ALL_LIBS)

# Generate MlirConfig.cmake for the build tree.
set(XBLANG_CONFIG_CMAKE_DIR "${xblang_cmake_builddir}")
set(XBLANG_CONFIG_LLVM_CMAKE_DIR "${llvm_cmake_builddir}")
set(XBLANG_CONFIG_INCLUDE_EXPORTS
    "include(\"\${XBLANG_CMAKE_DIR}/XBLANGTargets.cmake\")"
)
set(XBLANG_CONFIG_INCLUDE_DIRS "${XBLANG_SOURCE_DIR}/include"
                               "${XBLANG_BINARY_DIR}/include"
)
# Refer to the best host xblang-tbgen, which might be a host-optimized version
set(XBLANG_CONFIG_TABLEGEN_EXE "${XBLANG_TABLEGEN_EXE}")

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/XBLANGConfig.cmake.in
  ${xblang_cmake_builddir}/XBLANGConfig.cmake @ONLY
)
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/XBLANGConfigVersion.cmake.in
  ${xblang_cmake_builddir}/XBLANGConfigVersion.cmake @ONLY
)
set(XBLANG_CONFIG_CMAKE_DIR)
set(XBLANG_CONFIG_LLVM_CMAKE_DIR)
set(XBLANG_CONFIG_INCLUDE_DIRS)

# For compatibility with projects that include(XBLANGConfig) via
# CMAKE_MODULE_PATH, place API modules next to it. Copy without source
# permissions because the source could be read-only, but we need to write into
# the copied folder. This should be removed in the future.
file(
  COPY .
  DESTINATION ${xblang_cmake_builddir}
  NO_SOURCE_PERMISSIONS FILES_MATCHING
  PATTERN *.cmake
  PATTERN CMakeFiles EXCLUDE
)

# Generate XBLANGConfig.cmake for the install tree.
find_prefix_from_config(
  XBLANG_CONFIG_CODE XBLANG_INSTALL_PREFIX "${XBLANG_INSTALL_PACKAGE_DIR}"
)
extend_path(
  XBLANG_CONFIG_CMAKE_DIR "\${XBLANG_INSTALL_PREFIX}"
  "${XBLANG_INSTALL_PACKAGE_DIR}"
)
extend_path(
  XBLANG_CONFIG_LLVM_CMAKE_DIR "\${XBLANG_INSTALL_PREFIX}"
  "${LLVM_INSTALL_PACKAGE_DIR}"
)
get_config_exports_includes(XBLANG XBLANG_CONFIG_INCLUDE_EXPORTS)
extend_path(
  base_includedir "\${XBLANG_INSTALL_PREFIX}" "${CMAKE_INSTALL_INCLUDEDIR}"
)
set(XBLANG_CONFIG_INCLUDE_DIRS "${base_includedir}")
# Ensure that we are using the installed xblang-tblgen. This might not be
# XBLANG_TABLEGEN_EXE if we're building with a host-optimized xblang-tblgen
# (with LLVM_OPTIMIZED_TABLEGEN).
set(XBLANG_CONFIG_TABLEGEN_EXE xblang-tblgen)

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/XBLANGConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/XBLANGConfig.cmake @ONLY
)
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/XBLANGConfigVersion.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/XBLANGConfigVersion.cmake @ONLY
)
set(XBLANG_CONFIG_CODE)
set(XBLANG_CONFIG_CMAKE_DIR)
set(XBLANG_CONFIG_LLVM_CMAKE_DIR)
set(XBLANG_CONFIG_INCLUDE_DIRS)

if(NOT LLVM_INSTALL_TOOLCHAIN_ONLY)
  # Not TOOLCHAIN ONLY, so install the XBLANG parts as well Include the cmake
  # files so other tools can use xblang-tblgen, etc.
  install_distribution_exports(XBLANG)

  install(
    FILES ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/XBLANGConfig.cmake
          ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/XBLANGConfigVersion.cmake
          ${CMAKE_CURRENT_SOURCE_DIR}/AddXBLang.cmake
    DESTINATION ${XBLANG_INSTALL_PACKAGE_DIR}
    COMPONENT xblang-cmake-exports
  )

  if(NOT LLVM_ENABLE_IDE)
    # Add a dummy target so this can be used with LLVM_DISTRIBUTION_COMPONENTS
    add_custom_target(xblang-cmake-exports)
    add_llvm_install_targets(
      install-xblang-cmake-exports COMPONENT xblang-cmake-exports
    )
  endif()
endif()
