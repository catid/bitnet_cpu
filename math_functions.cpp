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

#ifdef ENABLE_AVX512_BUILD

static int16_t _mm256_reduce_add_epi16(__m256i ymm0) {
    __m128i xmm0 = _mm256_extracti128_si256(ymm0, 0);
    __m128i xmm1 = _mm256_extracti128_si256(ymm0, 1);
    
    xmm0 = _mm_add_epi16(xmm0, xmm1);
    
    xmm1 = _mm_shuffle_epi32(xmm0, 0x4E); // 0x4E = 0b01001110
    xmm0 = _mm_add_epi16(xmm0, xmm1);
    
    xmm1 = _mm_shuffle_epi32(xmm0, 0xB1); // 0xB1 = 0b10110001
    xmm0 = _mm_add_epi16(xmm0, xmm1);
    
    xmm1 = _mm_shufflelo_epi16(xmm0, 0xB1); // 0xB1 = 0b10110001
    xmm0 = _mm_add_epi16(xmm0, xmm1);
    
    return static_cast<int16_t>(_mm_cvtsi128_si32(xmm0));
}

void bitnet_vmul_avx512(const int8_t* x, const uint64_t* mask_add, const uint64_t* mask_sub, const float* scale_x, const float* scale_y, size_t input_size, size_t output_size, float* output) {
    assert(input_size % 64 == 0);
    for (size_t i = 0; i < output_size; ++i) {
        float out_row = 0.0f;
        const float* row_scale = scale_y + i * (input_size / 64);
        const uint64_t* row_add = mask_add + i * (input_size / 64);
        const uint64_t* row_sub = mask_sub + i * (input_size / 64);
        for (size_t j = 0; j < input_size; j += 64) {
            int block_index = j / 64;
            const uint64_t ma = row_add[block_index];
            const uint64_t ms = row_sub[block_index];

            __m512i x_block = _mm512_loadu_epi8(x + j);
            __m512i x_lo = _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(x_block, 0));
            __m512i x_hi = _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(x_block, 1));

            __m512i sum = _mm512_setzero_si512();
            sum = _mm512_mask_add_epi16(sum, (__mmask32)ma, sum, x_lo);
            sum = _mm512_mask_sub_epi16(sum, (__mmask32)ms, sum, x_lo);
            sum = _mm512_mask_add_epi16(sum, (__mmask32)(ma >> 32), sum, x_hi);
            sum = _mm512_mask_sub_epi16(sum, (__mmask32)(ms >> 32), sum, x_hi);
#if 0
            __m256i reduce = _mm256_add_epi16(_mm512_extracti64x4_epi64(sum, 0), _mm512_extracti64x4_epi64(sum, 1));

            int y0 = _mm256_reduce_add_epi16(_mm512_extracti64x4_epi64(sum, 0))
            int y1 = _mm256_reduce_add_epi16(_mm512_extracti64x4_epi64(sum, 1))
#endif           
            int y = 0; //y0 + y1;

            out_row += scale_x[block_index] * row_scale[block_index] * y;
        }
        output[i] = out_row;
    }
}

void bitnet_vmul_avx512_ref(const int8_t* x, const uint64_t* mask_add, const uint64_t* mask_sub,
                               const float* scale_x, const float* scale_y,
                               size_t input_size, size_t output_size, float* output)
{
    assert(input_size % 64 == 0);

    for (size_t i = 0; i < output_size; ++i) {
        float out_row = 0.0f;
        const float* row_scale = scale_y + i * (input_size / 64);
        const uint64_t* row_add = mask_add + i * (input_size / 64);
        const uint64_t* row_sub = mask_sub + i * (input_size / 64);

        for (size_t j = 0; j < input_size; j += 64) {
            int block_index = j / 64;
            const uint64_t ma = row_add[block_index];
            const uint64_t ms = row_sub[block_index];

            int32_t temp = 0;
            for (size_t k = 0; k < 64; ++k) {
                size_t index = j + k;
                int8_t x_val = x[index];

                int y = static_cast<int16_t>(x_val & -(int8_t)((ma >> k) & 1)) - 
                        static_cast<int16_t>(x_val & -(int8_t)((ms >> k) & 1));

                temp += y;
            }

            out_row += scale_x[block_index] * row_scale[block_index] * temp;
        }

        output[i] = out_row;
    }
}

#endif // ENABLE_AVX512_BUILD
