cmake_minimum_required(VERSION 3.10)
project(bitnet_cpu_project)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Enable OpenMP
find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

if(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /arch:AVX512 /Ox /fp:fast")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -O3 -funroll-loops")
endif()

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
install(TARGETS benchmark_model DESTINATION bin)

add_subdirectory(tests)
