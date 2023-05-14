FetchContent_Declare(googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG release-1.12.1)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)


add_executable(
  unittests
  parser_test.cpp
)
target_link_libraries(
  unittests
  gtest_main
  gmock
  toy_mlir_library
)

include(GoogleTest)
gtest_discover_tests(unittests)