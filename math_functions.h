#pragma once

#include <cstdint>
#include <cstddef>

void bitnet_vmul_ref(const int8_t* x, const int8_t* mask_add, const int8_t* mask_sub,
                     const float* scale_x, const float* scale_y,
                     size_t input_size, size_t output_size, float* output);

void bitnet_vmul_simd(const int8_t* x, const int8_t* mask_add, const int8_t* mask_sub,
                      const float* scale_x, const float* scale_y,
                      size_t input_size, size_t output_size, float* output);

void bitnet_vmul_simd_unrolled(const int8_t* x, const int8_t* mask_add, const int8_t* mask_sub,
                               const float* scale_x, const float* scale_y,
                               size_t input_size, size_t output_size, float* output);
