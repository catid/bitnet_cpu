#pragma once

#include <cstdint>
#include <cstddef>

void subtract_masked_arrays(const int8_t* x, const int8_t* mask_a_, const int8_t* mask_b_, int8_t* out, size_t size);
void subtract_masked_arrays_2d(const int8_t* x, const int8_t* mask_a_, const int8_t* mask_b_, int8_t* out, size_t input_size, size_t output_size);
