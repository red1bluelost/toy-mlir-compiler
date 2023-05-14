cmake_minimum_required(VERSION 3.20)
project(toy-mlir LANGUAGES CXX C)

include(FetchContent)

find_package(MLIR REQUIRED CONFIG)
message(STATUS "Found MLIR ${MLIR_PACKAGE_VERSION}")
message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
separate_arguments(LLVM_DEFINITIONS_LIST NATIVE_COMMAND ${LLVM_DEFINITIONS})
add_definitions(${LLVM_DEFINITIONS})
llvm_map_components_to_libnames(llvm_libs support)

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

FetchContent_Declare(fmt
  GIT_REPOSITORY https://github.com/fmtlib/fmt.git
  GIT_TAG master)
FetchContent_MakeAvailable(fmt)
FetchContent_Declare(ctl
  GIT_REPOSITORY https://github.com/red1bluelost/ctl.git
  GIT_TAG main)
FetchContent_MakeAvailable(ctl)

include(CheckIPOSupported)
check_ipo_supported(RESULT supported OUTPUT error)
if (CMAKE_BUILD_TYPE STREQUAL Release)
  set_property(GLOBAL PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
endif ()

if (CMAKE_CXX_COMPILER_ID MATCHES "(GNU|Clang)")
  add_compile_options(
    -Wall
    -Wextra
    -Wpedantic
    -Wconversion
    -march=native
    -fno-rtti
    $<$<CONFIG:DEBUG>:-fsanitize=address>
    $<$<CONFIG:DEBUG>:-g3>
    $<$<CONFIG:RELEASE>:-g0>
  )
  add_link_options(
    -rdynamic
    $<$<CONFIG:DEBUG>:-fsanitize=address>
  )
elseif (CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
  add_compile_options(
    /W4
    /GR-
  )
else ()
  message(FATAL_ERROR "Unknown compiler '${CMAKE_CXX_COMPILER_ID}'")
endif ()

set(TOY_MLIR_SOURCES
  lib/ast/ast.cpp
  lib/lexer/lexer.cpp
  lib/parser/parser.cpp)
set(TOY_MLIR_HEADERS
  include/toy/ast/ast.hpp
  include/toy/core/types.hpp
  include/toy/lexer/lexer.hpp
  include/toy/lexer/location.hpp
  include/toy/parser/parser.hpp
  include/toy/util/line_reader.hpp
  include/toy/util/string_view_line_reader.hpp)

add_library(
  toy_mlir_library
  ${TOY_MLIR_SOURCES}
  ${TOY_MLIR_HEADERS}
)
target_include_directories(
  toy_mlir_library
  PUBLIC
  ${PROJECT_SOURCE_DIR}/include
  SYSTEM ${LLVM_INCLUDE_DIRS}
  SYSTEM ${MLIR_INCLUDE_DIRS}
)
target_link_libraries(
  toy_mlir_library
  fmt::fmt
  ctl::core
  ctl::object
  ${llvm_libs}
  MLIRSupport
)

add_subdirectory(tools)

enable_testing()
add_subdirectory(unittests)