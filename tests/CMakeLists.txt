cmake_minimum_required(VERSION 3.10)
project(tests)

add_executable(math_test math_test.cpp)
target_link_libraries(math_test
    bitnet_cpu
)
#target_compile_options(math_test PRIVATE -O3)
install(TARGETS math_test DESTINATION bin)
