cmake_minimum_required(VERSION 3.20.0)
project(stdlib LANGUAGES CXX C)

set(CMAKE_BUILD_WITH_INSTALL_NAME_DIR ON)

set(CMAKE_CXX_STANDARD
    17
    CACHE STRING "C++ standard to conform to")

set(STDLIB_SOURCE_DIR ${PROJECT_SOURCE_DIR})
set(STDLIB_BINARY_DIR ${PROJECT_BINARY_DIR})

add_subdirectory(Par)
