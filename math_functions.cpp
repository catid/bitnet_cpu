#include "math_functions.h"
#include "tools.h"

#include <immintrin.h>

void subtract_masked_arrays_2d(const int8_t* x, const int8_t* mask_a_, const int8_t* mask_b_, int8_t* out, size_t input_size, size_t output_size) {
    __m256i* mask_a = (__m256i*)mask_a_;
    __m256i* mask_b = (__m256i*)mask_b_;

    const size_t simd_width = 32;
    size_t i = 0;

    static_assert(BITNET_CPU_ALIGNMENT % sizeof(__m256i) == 0, "BITNET_CPU_ALIGNMENT must be a multiple of the size of __m256i");
    static_assert(simd_width == sizeof(__m256i) / sizeof(int8_t), "simd_width must be equal to the size of __m256i");

    for (; i + simd_width * 4 <= output_size; i += simd_width * 4) {
        __m256i result_vec_0 = _mm256_setzero_si256();
        __m256i result_vec_1 = _mm256_setzero_si256();
        __m256i result_vec_2 = _mm256_setzero_si256();
        __m256i result_vec_3 = _mm256_setzero_si256();

        for (size_t j = 0; j < input_size; j += simd_width) {
            __m256i x_vec = _mm256_load_si256((__m256i*)(x + j));
            __m256i mask_a_vec_0 = _mm256_load_si256(mask_a + (i + 0 * simd_width) * input_size / simd_width + j / simd_width);
            __m256i mask_a_vec_1 = _mm256_load_si256(mask_a + (i + 1 * simd_width) * input_size / simd_width + j / simd_width);
            __m256i mask_a_vec_2 = _mm256_load_si256(mask_a + (i + 2 * simd_width) * input_size / simd_width + j / simd_width);
            __m256i mask_a_vec_3 = _mm256_load_si256(mask_a + (i + 3 * simd_width) * input_size / simd_width + j / simd_width);
            __m256i mask_b_vec_0 = _mm256_load_si256(mask_b + (i + 0 * simd_width) * input_size / simd_width + j / simd_width);
            __m256i mask_b_vec_1 = _mm256_load_si256(mask_b + (i + 1 * simd_width) * input_size / simd_width + j / simd_width);
            __m256i mask_b_vec_2 = _mm256_load_si256(mask_b + (i + 2 * simd_width) * input_size / simd_width + j / simd_width);
            __m256i mask_b_vec_3 = _mm256_load_si256(mask_b + (i + 3 * simd_width) * input_size / simd_width + j / simd_width);

            __m256i masked_a_vec_0 = _mm256_and_si256(x_vec, mask_a_vec_0);
            __m256i masked_a_vec_1 = _mm256_and_si256(x_vec, mask_a_vec_1);
            __m256i masked_a_vec_2 = _mm256_and_si256(x_vec, mask_a_vec_2);
            __m256i masked_a_vec_3 = _mm256_and_si256(x_vec, mask_a_vec_3);
            __m256i masked_b_vec_0 = _mm256_and_si256(x_vec, mask_b_vec_0);
            __m256i masked_b_vec_1 = _mm256_and_si256(x_vec, mask_b_vec_1);
            __m256i masked_b_vec_2 = _mm256_and_si256(x_vec, mask_b_vec_2);
            __m256i masked_b_vec_3 = _mm256_and_si256(x_vec, mask_b_vec_3);

            result_vec_0 = _mm256_sub_epi8(result_vec_0, _mm256_sub_epi8(masked_a_vec_0, masked_b_vec_0));
            result_vec_1 = _mm256_sub_epi8(result_vec_1, _mm256_sub_epi8(masked_a_vec_1, masked_b_vec_1));
            result_vec_2 = _mm256_sub_epi8(result_vec_2, _mm256_sub_epi8(masked_a_vec_2, masked_b_vec_2));
            result_vec_3 = _mm256_sub_epi8(result_vec_3, _mm256_sub_epi8(masked_a_vec_3, masked_b_vec_3));
        }

        _mm256_stream_si256((__m256i*)(out + i + 0 * simd_width), result_vec_0);
        _mm256_stream_si256((__m256i*)(out + i + 1 * simd_width), result_vec_1);
        _mm256_stream_si256((__m256i*)(out + i + 2 * simd_width), result_vec_2);
        _mm256_stream_si256((__m256i*)(out + i + 3 * simd_width), result_vec_3);
    }

    for (; i + simd_width <= output_size; i += simd_width) {
        __m256i result_vec = _mm256_setzero_si256();

        for (size_t j = 0; j < input_size; j += simd_width) {
            __m256i x_vec = _mm256_load_si256((__m256i*)(x + j));
            __m256i mask_a_vec = _mm256_load_si256(mask_a + i * input_size / simd_width + j / simd_width);
            __m256i mask_b_vec = _mm256_load_si256(mask_b + i * input_size / simd_width + j / simd_width);

            __m256i masked_a_vec = _mm256_and_si256(x_vec, mask_a_vec);
            __m256i masked_b_vec = _mm256_and_si256(x_vec, mask_b_vec);

            result_vec = _mm256_sub_epi8(result_vec, _mm256_sub_epi8(masked_a_vec, masked_b_vec));
        }

        _mm256_store_si256((__m256i*)(out + i), result_vec);
    }

    for (; i < output_size; ++i) {
        int32_t result = 0;
        for (size_t j = 0; j < input_size; ++j) {
            result += (x[j] & mask_a_[i * input_size + j]) - (x[j] & mask_b_[i * input_size + j]);
        }
        out[i] = static_cast<int8_t>(result);
    }
}

void subtract_masked_arrays(const int8_t* x, const int8_t* mask_a_, const int8_t* mask_b_, int8_t* out, size_t size) {
    __m256i* mask_a = (__m256i*)mask_a_;
    __m256i* mask_b = (__m256i*)mask_b_;

    // Process the input array in chunks of 32 elements
    const size_t simd_width = 32;  // AVX2 processes 32 int8_t elements at a time
    size_t i = 0;

    static_assert(BITNET_CPU_ALIGNMENT % sizeof(__m256i) == 0, "BITNET_CPU_ALIGNMENT must be a multiple of the size of __m256i");
    static_assert(simd_width == sizeof(__m256i) / sizeof(int8_t), "simd_width must be equal to the size of __m256i");

    for (; i + simd_width * 4 <= size; i += simd_width * 4) {
        // Process 4 AVX2 vectors per iteration
        for (int j = 0; j < 4; ++j) {
            __m256i x_vec = _mm256_load_si256((__m256i*)(x + i + j * simd_width));
            __m256i mask_a_vec = _mm256_load_si256(mask_a + (i + j * simd_width) / simd_width);
            __m256i mask_b_vec = _mm256_load_si256(mask_b + (i + j * simd_width) / simd_width);
            __m256i masked_a_vec = _mm256_and_si256(x_vec, mask_a_vec);
            __m256i masked_b_vec = _mm256_and_si256(x_vec, mask_b_vec);
            __m256i result_vec = _mm256_sub_epi8(masked_a_vec, masked_b_vec);
            _mm256_stream_si256((__m256i*)(out + i + j * simd_width), result_vec);
        }
    }

    for (; i + simd_width <= size; i += simd_width) {
        __m256i x_vec = _mm256_load_si256((__m256i*)(x + i));
        __m256i mask_a_vec = _mm256_load_si256(mask_a + i / simd_width);
        __m256i mask_b_vec = _mm256_load_si256(mask_b + i / simd_width);
        __m256i masked_a_vec = _mm256_and_si256(x_vec, mask_a_vec);
        __m256i masked_b_vec = _mm256_and_si256(x_vec, mask_b_vec);
        __m256i result_vec = _mm256_sub_epi8(masked_a_vec, masked_b_vec);
        _mm256_store_si256((__m256i*)(out + i), result_vec);
    }

    // Process the remaining elements (if any) using scalar operations
    for (; i < size; ++i) {
        out[i] = (x[i] & (mask_a_)[i]) - (x[i] & (mask_b_)[i]);
    }
}
