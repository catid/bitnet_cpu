#include "math_functions.h"
#include "tools.h"

#include <immintrin.h>
#include <cassert>

void bitnet_vmul_simd_unrolled(const int8_t* x, const int8_t* mask_add, const int8_t* mask_sub,
                               const float* scale_x, const float* scale_y,
                               size_t input_size, size_t output_size, float* output) {
    assert(input_size % 32 == 0);
    assert(output_size % 2 == 0);  // Ensure output_size is even

    for (size_t i = 0; i < output_size; i += 2) {
        float out_row1 = 0.0f;
        float out_row2 = 0.0f;
        const float* row_scale1 = scale_y + i * (input_size / 32);
        const float* row_scale2 = scale_y + (i + 1) * (input_size / 32);
        const int8_t* row_add1 = mask_add + i * input_size;
        const int8_t* row_add2 = mask_add + (i + 1) * input_size;
        const int8_t* row_sub1 = mask_sub + i * input_size;
        const int8_t* row_sub2 = mask_sub + (i + 1) * input_size;

        for (size_t j = 0; j < input_size; j += 32) {
            int block_index = j / 32;
            __m256i mask_add_block1 = _mm256_loadu_si256((__m256i*)(row_add1 + j));
            __m256i mask_add_block2 = _mm256_loadu_si256((__m256i*)(row_add2 + j));
            __m256i mask_sub_block1 = _mm256_loadu_si256((__m256i*)(row_sub1 + j));
            __m256i mask_sub_block2 = _mm256_loadu_si256((__m256i*)(row_sub2 + j));

            __m256i x_block = _mm256_loadu_si256((__m256i*)(x + j));

            __m256i temp_add1 = _mm256_and_si256(x_block, mask_add_block1);
            __m256i temp_add2 = _mm256_and_si256(x_block, mask_add_block2);
            __m256i temp_sub1 = _mm256_and_si256(x_block, mask_sub_block1);
            __m256i temp_sub2 = _mm256_and_si256(x_block, mask_sub_block2);

            __m256i add_lo1 = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(temp_add1));
            __m256i add_hi1 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(temp_add1, 1));
            __m256i add_lo2 = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(temp_add2));
            __m256i add_hi2 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(temp_add2, 1));

            __m256i sub_lo1 = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(temp_sub1));
            __m256i sub_hi1 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(temp_sub1, 1));
            __m256i sub_lo2 = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(temp_sub2));
            __m256i sub_hi2 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(temp_sub2, 1));

            __m256i add_sum1 = _mm256_add_epi16(add_lo1, add_hi1);
            __m256i add_sum2 = _mm256_add_epi16(add_lo2, add_hi2);
            __m256i sub_sum1 = _mm256_add_epi16(sub_lo1, sub_hi1);
            __m256i sub_sum2 = _mm256_add_epi16(sub_lo2, sub_hi2);

            __m256i sum1 = _mm256_sub_epi16(add_sum1, sub_sum1);
            __m256i sum2 = _mm256_sub_epi16(add_sum2, sub_sum2);

            sum1 = _mm256_hadd_epi16(sum1, sum1);
            sum1 = _mm256_hadd_epi16(sum1, sum1);
            sum1 = _mm256_hadd_epi16(sum1, sum1);

            sum2 = _mm256_hadd_epi16(sum2, sum2);
            sum2 = _mm256_hadd_epi16(sum2, sum2);
            sum2 = _mm256_hadd_epi16(sum2, sum2);

            int y1 = static_cast<int16_t>(_mm256_extract_epi16(sum1, 0)) +
                     static_cast<int16_t>(_mm256_extract_epi16(sum1, 8));
            int y2 = static_cast<int16_t>(_mm256_extract_epi16(sum2, 0)) +
                     static_cast<int16_t>(_mm256_extract_epi16(sum2, 8));

            out_row1 += scale_x[block_index] * row_scale1[block_index] * y1;
            out_row2 += scale_x[block_index] * row_scale2[block_index] * y2;
        }

        output[i] = out_row1;
        output[i + 1] = out_row2;
    }
}

void bitnet_vmul_simd(const int8_t* x, const int8_t* mask_add, const int8_t* mask_sub,
                      const float* scale_x, const float* scale_y,
                      size_t input_size, size_t output_size, float* output) {
    assert(input_size % 32 == 0);

    for (size_t i = 0; i < output_size; ++i) {
        float out_row = 0.0f;
        const float* row_scale = scale_y + i * (input_size / 32);
        const int8_t* row_add = mask_add + i * input_size;
        const int8_t* row_sub = mask_sub + i * input_size;

        for (size_t j = 0; j < input_size; j += 32) {
            int block_index = j / 32;
            __m256i mask_add_block = _mm256_loadu_si256((__m256i*)(row_add + j));
            __m256i mask_sub_block = _mm256_loadu_si256((__m256i*)(row_sub + j));

            __m256i x_block = _mm256_loadu_si256((__m256i*)(x + j));

            __m256i temp_add = _mm256_and_si256(x_block, mask_add_block);
            __m256i temp_sub = _mm256_and_si256(x_block, mask_sub_block);

            __m256i add_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(temp_add));
            __m256i add_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(temp_add, 1));

            __m256i sub_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(temp_sub));
            __m256i sub_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(temp_sub, 1));

            __m256i add_sum = _mm256_add_epi16(add_lo, add_hi);
            __m256i sub_sum = _mm256_add_epi16(sub_lo, sub_hi);

            __m256i sum = _mm256_sub_epi16(add_sum, sub_sum); 

            sum = _mm256_hadd_epi16(sum, sum);
            sum = _mm256_hadd_epi16(sum, sum);
            sum = _mm256_hadd_epi16(sum, sum);
            int y = static_cast<int16_t>( _mm256_extract_epi16(sum, 0) ) + static_cast<int16_t>( _mm256_extract_epi16(sum, 8) );

            out_row += scale_x[block_index] * row_scale[block_index] * y;
        }

        output[i] = out_row;
    }
}

void bitnet_vmul_ref(const int8_t* x, const int8_t* mask_add, const int8_t* mask_sub,
                     const float* scale_x, const float* scale_y,
                     size_t input_size, size_t output_size, float* output) {
    assert(input_size % 32 == 0);

    for (size_t i = 0; i < output_size; ++i) {
        float out_row = 0.0f;
        const float* row_scale = scale_y + i * (input_size / 32);
        const int8_t* row_add = mask_add + i * input_size;
        const int8_t* row_sub = mask_sub + i * input_size;

        for (size_t j = 0; j < input_size; j += 32) {
            int block_index = j / 32;

            int32_t temp = 0;
            for (size_t k = 0; k < 32; ++k) {
                size_t index = j + k;
                int8_t x_val = x[index];

                int y = static_cast<int16_t>(x_val & row_add[index]) - 
                        static_cast<int16_t>(x_val & row_sub[index]);

                temp += y;
            }

            out_row += scale_x[block_index] * row_scale[block_index] * temp;
        }

        output[i] = out_row;
    }
}
