# CMake policy settings shared between LLVM projects CMP0116: Ninja generators
# transform `DEPFILE`s from `add_custom_command()` New in CMake 3.20.
# https://cmake.org/cmake/help/latest/policy/CMP0116.html
if(POLICY CMP0116)
  cmake_policy(SET CMP0116 OLD)
endif()
