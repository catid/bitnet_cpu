#pragma once

#include <cstdint>
#include <cstddef>

#include "tools.h"

// You can use CpuSupportsAVX512() to check for AVX-512 support

// Masks defined with 1 byte per mask (16 bits per parameter):

void bitnet_vmul_ref(const int8_t* x, const int8_t* mask_add, const int8_t* mask_sub,
                     const float* scale_x, const float* scale_y,
                     size_t input_size, size_t output_size, float* output);

void bitnet_vmul_simd(const int8_t* x, const int8_t* mask_add, const int8_t* mask_sub,
                      const float* scale_x, const float* scale_y,
                      size_t input_size, size_t output_size, float* output);

void bitnet_vmul_simd_unrolled(const int8_t* x, const int8_t* mask_add, const int8_t* mask_sub,
                               const float* scale_x, const float* scale_y,
                               size_t input_size, size_t output_size, float* output);

#ifdef ENABLE_AVX512_BUILD

// x[] 1D array: One value for each input (input_size of these)
// Masks 2D array: One 64-bit mask word for every 64 inputs,
//        and rows=output_size (2 bits per model parameter)
// scale_x[] 1D array: One value for every 64 inputs (quantization scale factor)
// scale_y[] 2D array: One value for every 64 inputs, and rows=output_size
// output[] 1D array: `output_size` floats
void bitnet_vmul_avx512(const int8_t* x, const uint64_t* mask_add, const uint64_t* mask_sub,
                               const float* scale_x, const float* scale_y,
                               size_t input_size, size_t output_size, float* output);

void bitnet_vmul_avx512_ref(const int8_t* x, const uint64_t* mask_add, const uint64_t* mask_sub,
                               const float* scale_x, const float* scale_y,
                               size_t input_size, size_t output_size, float* output);

#endif // ENABLE_AVX512_BUILD
