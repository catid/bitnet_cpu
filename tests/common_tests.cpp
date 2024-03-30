#include "common_tests.h"

bool validate_result(const int8_t* x, const int8_t* mask_a, const int8_t* mask_b, const int8_t* out, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        int8_t expected = (x[i] & mask_a[i]) - (x[i] & mask_b[i]);
        if (out[i] != expected) {
            return false;
        }
    }
    return true;
}

bool validate_result_2d(const int8_t* x, const int8_t* mask_a, const int8_t* mask_b, const int8_t* out, size_t input_size, size_t output_size) {
    for (size_t i = 0; i < output_size; ++i) {
        int32_t expected = 0;
        for (size_t j = 0; j < input_size; ++j) {
            expected += (x[j] & mask_a[i * input_size + j]) - (x[j] & mask_b[i * input_size + j]);
        }
        if (out[i] != static_cast<int8_t>(expected)) {
            return false;
        }
    }
    return true;
}
