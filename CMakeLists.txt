cmake_minimum_required(VERSION 3.21)

set(CMAKE_BUILD_TYPE DEBUG)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(INCLUDE_MLIR_DILAECTS True)

set(XBLANG_ENABLE_PLUGINS ON)
set(BUILD_SHARED_LIBS OFF)

project(
  xblang
  VERSION 0.42
  LANGUAGES CXX C
)

# Make sure that our source directory is on the current cmake module path so
# that we can include cmake files from this directory.
list(INSERT CMAKE_MODULE_PATH 0 "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules")

include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules/CMakePolicy.cmake
        NO_POLICY_SCOPE
)

# ##############################################################################
# Find and configure LLVM & MLIR
# ##############################################################################

find_package(MLIR REQUIRED CONFIG)

message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

# ##############################################################################
# XBLang options
# ##############################################################################

option(XBLANG_ENABLE_TEST "Enable XBLang tests" OFF)

# ##############################################################################
# Configure XBLang
# ##############################################################################
set(XBLANG_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(XBLANG_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})
set(XBLANG_INCLUDE_DIR ${CMAKE_CURRENT_BINARY_DIR}/include)
set(XBLANG_TOOLS_DIR ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})

include_directories(${XBLANG_SOURCE_DIR}/include)
include_directories(${XBLANG_INCLUDE_DIR})

set(XBLANG_TOOLS_INSTALL_DIR
    "${CMAKE_INSTALL_BINDIR}"
    CACHE PATH
          "Path for binary subdirectory (defaults to '${CMAKE_INSTALL_BINDIR}')"
)
mark_as_advanced(XBLANG_TOOLS_INSTALL_DIR)

# ##############################################################################
# -fno-rtti needed because llvm is usually compiled with rtti it. -gdwarf-4
# needed because:
# https://discourse.llvm.org/t/gdb-10-1-cant-read-clangs-dwarf-v5/6035
# ##############################################################################
add_compile_options(
  -fno-rtti -gdwarf-4 -Wno-deprecated-this-capture
  -Wno-ambiguous-reversed-operator -Wno-unused-label
)

add_subdirectory(tools/xblang-tblgen)
set(XBLANG_NEW_TABLEGEN_EXE "${XBLANG_NEW_TABLEGEN_EXE}" CACHE INTERNAL "")

include(AddXBLang)

set(XBLANG_TABLEGEN_EXE
    "${XBLANG_TABLEGEN_EXE}"
    CACHE INTERNAL ""
)

function(xblang_tablegen ofn)
  tablegen(XBLANG ${ARGV})
  set(TABLEGEN_OUTPUT
      ${TABLEGEN_OUTPUT} ${CMAKE_CURRENT_BINARY_DIR}/${ofn}
      PARENT_SCOPE
  )

  # Get the current set of include paths for this td file.
  cmake_parse_arguments(ARG "" "" "DEPENDS;EXTRA_INCLUDES" ${ARGN})
  get_directory_property(tblgen_includes INCLUDE_DIRECTORIES)
  list(APPEND tblgen_includes ${ARG_EXTRA_INCLUDES})
  # Filter out any empty include items.
  list(REMOVE_ITEM tblgen_includes "")

  # Build the absolute path for the current input file.
  if(IS_ABSOLUTE ${LLVM_TARGET_DEFINITIONS})
    set(LLVM_TARGET_DEFINITIONS_ABSOLUTE ${LLVM_TARGET_DEFINITIONS})
  else()
    set(LLVM_TARGET_DEFINITIONS_ABSOLUTE
        ${CMAKE_CURRENT_SOURCE_DIR}/${LLVM_TARGET_DEFINITIONS}
    )
  endif()

  # Append the includes used for this file to the tablegen_compile_commands
  # file.
  file(
    APPEND ${CMAKE_BINARY_DIR}/tablegen_compile_commands.yml
    "--- !FileInfo:\n" "  filepath: \"${LLVM_TARGET_DEFINITIONS_ABSOLUTE}\"\n"
    "  includes: \"${CMAKE_CURRENT_SOURCE_DIR};${tblgen_includes}\"\n"
  )
endfunction()

if (INCLUDE_MLIR_DILAECTS)
  add_definitions(-DXBC_INCLUDE_MLIR_DIALECTS=1)
else()
  add_definitions(-DXBC_INCLUDE_MLIR_DIALECTS=0)
endif()

add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(stdlib)
add_subdirectory(tools)

set(LLVM_BUILD_TESTS ON)
if(XBLANG_ENABLE_TEST)
  add_definitions(-DXBLANG_INCLUDE_TESTS)
  add_custom_target(XBLangUnitTests)
  if(EXISTS ${LLVM_THIRD_PARTY_DIR}/unittest/googletest/include/gtest/gtest.h)

  else()
    enable_testing()
    include(FetchContent)
    FetchContent_Declare(
      googletest
      URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
    )
    # For Windows: Prevent overriding the parent project's compiler/linker
    # settings
    set(gtest_force_shared_crt
        ON
        CACHE BOOL "" FORCE
    )
    FetchContent_MakeAvailable(googletest)
  endif()
  add_subdirectory(unittests)
  add_subdirectory(test)
endif()

add_subdirectory(extensions)
