cmake_minimum_required(VERSION 3.10)
project(MaskedSubtraction)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2")

add_library(bitnet_cpu STATIC
    math_functions.cpp
    math_functions.h
    tools.cpp
    tools.h
)
target_include_directories(bitnet_cpu PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
install(TARGETS bitnet_cpu DESTINATION bin)

add_executable(benchmark_model
    benchmark_model.cpp
)
target_link_libraries(benchmark_model
    bitnet_cpu
)
target_compile_options(benchmark_model PRIVATE -O3)
install(TARGETS benchmark_model DESTINATION bin)

add_subdirectory(tests)
