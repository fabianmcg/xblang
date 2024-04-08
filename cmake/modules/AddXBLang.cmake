include(GNUInstallDirs)
include(LLVMDistributionSupport)

function(xblang_tablegen_new ofn)
  tablegen(XBLANG_NEW ${ARGV})
  set(TABLEGEN_OUTPUT
      ${TABLEGEN_OUTPUT} ${CMAKE_CURRENT_BINARY_DIR}/${ofn}
      PARENT_SCOPE
  )
endfunction()

# Sets ${srcs} to contain the list of additional headers for the target. Extra
# arguments are included into the list of additional headers.
function(_set_xblang_additional_headers_as_srcs)
  set(srcs)
  if(MSVC_IDE OR XCODE)
    # Add public headers
    file(RELATIVE_PATH lib_path ${XBLANG_SOURCE_DIR}/lib/
         ${CMAKE_CURRENT_SOURCE_DIR}
    )
    if(NOT lib_path MATCHES "^[.][.]")
      file(GLOB_RECURSE headers
           ${XBLANG_SOURCE_DIR}/include/xblang/${lib_path}/*.h
           ${XBLANG_SOURCE_DIR}/include/xblang/${lib_path}/*.def
      )
      set_source_files_properties(${headers} PROPERTIES HEADER_FILE_ONLY ON)

      file(GLOB_RECURSE tds
           ${XBLANG_SOURCE_DIR}/include/xblang/${lib_path}/*.td
      )
      source_group("TableGen descriptions" FILES ${tds})
      set_source_files_properties(${tds}} PROPERTIES HEADER_FILE_ONLY ON)

      if(headers OR tds)
        set(srcs ${headers} ${tds})
      endif()
    endif()
  endif(MSVC_IDE OR XCODE)
  if(srcs OR ARGN)
    set(srcs
        ADDITIONAL_HEADERS ${srcs} ${ARGN} # It may contain unparsed unknown
                                           # args.
        PARENT_SCOPE
    )
  endif()
endfunction()

# Checks that the LLVM components are not listed in the extra arguments, assumed
# to be coming from the LINK_LIBS variable.
function(_check_llvm_components_usage name)
  # LINK_COMPONENTS is necessary to allow libLLVM.so to be properly substituted
  # for individual library dependencies if LLVM_LINK_LLVM_DYLIB Perhaps this
  # should be in llvm_add_library instead?  However, it fails on libclang-cpp.so
  get_property(llvm_component_libs GLOBAL PROPERTY LLVM_COMPONENT_LIBS)
  foreach(lib ${ARGN})
    if(${lib} IN_LIST llvm_component_libs)
      message(
        SEND_ERROR
          "${name} specifies LINK_LIBS ${lib}, but LINK_LIBS cannot be used for LLVM libraries.  Please use LINK_COMPONENTS instead."
      )
    endif()
  endforeach()
endfunction()

# Declare an xblang library which can be compiled in libXBLANG.so In addition to
# everything that llvm_add_library accepts, this also has the following option:
# EXCLUDE_FROM_LIBXBLANG Don't include this library in libXBLANG.so.  This
# option should be used for test libraries, executable-specific libraries, or
# rarely used libraries with large dependencies. ENABLE_AGGREGATION Forces
# generation of an OBJECT library, exports additional metadata, and installs
# additional object files needed to include this as part of an aggregate shared
# library. TODO: Make this the default for all XBLANG libraries once all
# libraries are compatible with building an object library.
function(add_xblang_library name)
  cmake_parse_arguments(
    ARG
    "SHARED;INSTALL_WITH_TOOLCHAIN;EXCLUDE_FROM_LIBXBLANG;DISABLE_INSTALL;ENABLE_AGGREGATION"
    ""
    "ADDITIONAL_HEADERS;DEPENDS;LINK_COMPONENTS;LINK_LIBS"
    ${ARGN}
  )
  _set_xblang_additional_headers_as_srcs(${ARG_ADDITIONAL_HEADERS})

  # Is an object library needed.
  set(NEEDS_OBJECT_LIB OFF)
  if(ARG_ENABLE_AGGREGATION)
    set(NEEDS_OBJECT_LIB ON)
  endif()

  # Determine type of library.
  if(ARG_SHARED)
    set(LIBTYPE SHARED)
  else()
    # llvm_add_library ignores BUILD_SHARED_LIBS if STATIC is explicitly set, so
    # we need to handle it here.
    if(BUILD_SHARED_LIBS)
      set(LIBTYPE SHARED)
    else()
      set(LIBTYPE STATIC)
    endif()
    # Test libraries and such shouldn't be include in libXBLANG.so
    if(NOT ARG_EXCLUDE_FROM_LIBXBLANG)
      set(NEEDS_OBJECT_LIB ON)
      set_property(GLOBAL APPEND PROPERTY XBLANG_STATIC_LIBS ${name})
      set_property(
        GLOBAL APPEND PROPERTY XBLANG_LLVM_LINK_COMPONENTS
                               ${ARG_LINK_COMPONENTS}
      )
      set_property(
        GLOBAL APPEND PROPERTY XBLANG_LLVM_LINK_COMPONENTS
                               ${LLVM_LINK_COMPONENTS}
      )
    endif()
  endif()

  if(NEEDS_OBJECT_LIB AND NOT XCODE)
    # The Xcode generator doesn't handle object libraries correctly. We special
    # case xcode when building aggregates.
    list(APPEND LIBTYPE OBJECT)
  endif()

  # XBLANG libraries uniformly depend on LLVMSupport.  Just specify it once
  # here.
  list(APPEND ARG_LINK_COMPONENTS Support)
  _check_llvm_components_usage(${name} ${ARG_LINK_LIBS})

  # list(APPEND ARG_DEPENDS xblang-generic-headers)
  llvm_add_library(
    ${name}
    ${LIBTYPE}
    ${ARG_UNPARSED_ARGUMENTS}
    ${srcs}
    DEPENDS
    ${ARG_DEPENDS}
    LINK_COMPONENTS
    ${ARG_LINK_COMPONENTS}
    LINK_LIBS
    ${ARG_LINK_LIBS}
  )

  if(TARGET ${name})
    target_link_libraries(${name} INTERFACE ${LLVM_COMMON_LIBS})
    if(NOT ARG_DISABLE_INSTALL)
      add_xblang_library_install(${name})
    endif()
  else()
    # Add empty "phony" target
    add_custom_target(${name})
  endif()
  set_target_properties(${name} PROPERTIES FOLDER "XBLANG libraries")

  # Setup aggregate.
  if(ARG_ENABLE_AGGREGATION)
    # Compute and store the properties needed to build aggregates.
    set(AGGREGATE_OBJECTS)
    set(AGGREGATE_OBJECT_LIB)
    set(AGGREGATE_DEPS)
    if(XCODE)
      # XCode has limited support for object libraries. Instead, add dep flags
      # that force the entire library to be embedded.
      list(APPEND AGGREGATE_DEPS "-force_load" "${name}")
    else()
      list(APPEND AGGREGATE_OBJECTS "$<TARGET_OBJECTS:obj.${name}>")
      list(APPEND AGGREGATE_OBJECT_LIB "obj.${name}")
    endif()

    # For each declared dependency, transform it into a generator expression
    # which excludes it if the ultimate link target is excluding the library.
    set(NEW_LINK_LIBRARIES)
    get_target_property(CURRENT_LINK_LIBRARIES ${name} LINK_LIBRARIES)
    get_xblang_filtered_link_libraries(
      NEW_LINK_LIBRARIES ${CURRENT_LINK_LIBRARIES}
    )
    set_target_properties(
      ${name} PROPERTIES LINK_LIBRARIES "${NEW_LINK_LIBRARIES}"
    )
    list(APPEND AGGREGATE_DEPS ${NEW_LINK_LIBRARIES})
    set_target_properties(
      ${name}
      PROPERTIES
        EXPORT_PROPERTIES
        "XBLANG_AGGREGATE_OBJECT_LIB_IMPORTED;XBLANG_AGGREGATE_DEP_LIBS_IMPORTED"
        XBLANG_AGGREGATE_OBJECTS "${AGGREGATE_OBJECTS}"
        XBLANG_AGGREGATE_DEPS "${AGGREGATE_DEPS}"
        XBLANG_AGGREGATE_OBJECT_LIB_IMPORTED "${AGGREGATE_OBJECT_LIB}"
        XBLANG_AGGREGATE_DEP_LIBS_IMPORTED "${CURRENT_LINK_LIBRARIES}"
    )

    # In order for out-of-tree projects to build aggregates of this library, we
    # need to install the OBJECT library.
    if(XBLANG_INSTALL_AGGREGATE_OBJECTS AND NOT ARG_DISABLE_INSTALL)
      add_xblang_library_install(obj.${name})
    endif()
  endif()
endfunction(add_xblang_library)

macro(add_xblang_tool name)
  llvm_add_tool(XBLANG ${ARGV})
endmacro()

# Sets a variable with a transformed list of link libraries such individual
# libraries will be dynamically excluded when evaluated on a final library which
# defines an XBLANG_AGGREGATE_EXCLUDE_LIBS which contains any of the libraries.
# Each link library can be a generator expression but must not resolve to an
# arity > 1 (i.e. it can be optional).
function(get_xblang_filtered_link_libraries output)
  set(_results)
  foreach(linklib ${ARGN})
    # In English, what this expression does: For each link library, resolve the
    # property XBLANG_AGGREGATE_EXCLUDE_LIBS on the context target (i.e. the
    # executable or shared library being linked) and, if it is not in that list,
    # emit the library name. Otherwise, empty.
    list(
      APPEND
      _results
      "$<$<NOT:$<IN_LIST:${linklib},$<GENEX_EVAL:$<TARGET_PROPERTY:XBLANG_AGGREGATE_EXCLUDE_LIBS>>>>:${linklib}>"
    )
  endforeach()
  set(${output}
      "${_results}"
      PARENT_SCOPE
  )
endfunction(get_xblang_filtered_link_libraries)

# Adds an XBLANG library target for installation. This is usually done as part
# of add_xblang_library but is broken out for cases where non-standard library
# builds can be installed.
function(add_xblang_library_install name)
  if(NOT LLVM_INSTALL_TOOLCHAIN_ONLY)
    get_target_export_arg(
      ${name} XBLANG export_to_xblangtargets UMBRELLA xblang-libraries
    )
    install(
      TARGETS ${name}
      COMPONENT ${name}
      ${export_to_xblangtargets}
      LIBRARY DESTINATION lib${LLVM_LIBDIR_SUFFIX}
      ARCHIVE DESTINATION lib${LLVM_LIBDIR_SUFFIX}
      RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
              # Note that CMake will create a directory like:
              # objects-${CMAKE_BUILD_TYPE}/obj.LibName
              # and put object files there.
      OBJECTS DESTINATION lib${LLVM_LIBDIR_SUFFIX}
    )

    if(NOT LLVM_ENABLE_IDE)
      add_llvm_install_targets(
        install-${name} DEPENDS ${name} COMPONENT ${name}
      )
    endif()
    set_property(GLOBAL APPEND PROPERTY XBLANG_ALL_LIBS ${name})
  endif()
  set_property(GLOBAL APPEND PROPERTY XBLANG_EXPORTS ${name})
endfunction()

# Declare the library associated with a dialect.
function(add_xblang_dialect_library name)
  set_property(GLOBAL APPEND PROPERTY XBLANG_DIALECT_LIBS ${name})
  add_xblang_library(${ARGV} DEPENDS xblang-headers)
endfunction(add_xblang_dialect_library)
