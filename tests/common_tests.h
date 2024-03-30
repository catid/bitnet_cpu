#pragma once

#include <cstdint>
#include <cstddef>

bool validate_result(const int8_t* x, const int8_t* mask_a, const int8_t* mask_b, const int8_t* out, size_t size);
bool validate_result_2d(const int8_t* x, const int8_t* mask_a, const int8_t* mask_b, const int8_t* out, size_t input_size, size_t output_size);
