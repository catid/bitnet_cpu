cmake_minimum_required(VERSION 3.10)
project(tests)

# GoogleTest
include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG release-1.11.0
)
FetchContent_MakeAvailable(googletest)

add_library(common_tests STATIC common_tests.cpp common_tests.h)

add_executable(masked_subtraction_test masked_subtraction_test.cpp)
target_link_libraries(masked_subtraction_test
    bitnet_cpu
    common_tests
    gtest_main
)

target_compile_options(masked_subtraction_test PRIVATE -O3)

install(TARGETS masked_subtraction_test DESTINATION bin)
