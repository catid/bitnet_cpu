#pragma once

#include <cstdlib>

constexpr size_t BITNET_CPU_ALIGNMENT = 32;

template <typename T>
T* allocate_aligned_buffer(size_t size);

void free_aligned_buffer(void* ptr);