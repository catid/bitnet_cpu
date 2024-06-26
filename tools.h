#pragma once

#include <cstdlib>

constexpr size_t BITNET_CPU_ALIGNMENT = 64;

template <typename T>
T* allocate_aligned_buffer(size_t size);

void free_aligned_buffer(void* ptr);

#if defined(__AVX512BW__) && defined(__AVX512VL__)
#define ENABLE_AVX512_BUILD
#endif

// Returns true if the CPU supports the extensions we need for the fast version
bool CpuSupportsAVX512();
