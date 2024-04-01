#include "math_functions.h"
#include "tools.h"

#include <immintrin.h>
#include <cassert>

void bitnet_vmul_ref(const int8_t* x, const int8_t* mask_add, const int8_t* mask_sub,
                     const float* scale_x, float out_scale,
                     size_t input_size, size_t output_size, float* output) {
    assert(input_size % 32 == 0);

    for (size_t i = 0; i < output_size; ++i) {
        double out_row = 0.0;
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

            out_row += scale_x[block_index] * out_scale * temp;
        }

        output[i] = out_row;
    }
}

void bitnet_vmul_simd_unrolled(const int8_t* x, const int8_t* mask_add, const int8_t* mask_sub,
                               const float* scale_x, float out_scale,
                               size_t input_size, size_t output_size, float* output) {
    assert(input_size % 32 == 0);
    assert(output_size % 2 == 0);  // Ensure output_size is even

    for (size_t i = 0; i < output_size; i += 2) {
        double out_row1 = 0.0;
        double out_row2 = 0.0;
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

            out_row1 += scale_x[block_index] * out_scale * y1;
            out_row2 += scale_x[block_index] * out_scale * y2;
        }

        output[i] = out_row1;
        output[i + 1] = out_row2;
    }
}

void bitnet_vmul_simd(const int8_t* x, const int8_t* mask_add, const int8_t* mask_sub,
                      const float* scale_x, float out_scale,
                      size_t input_size, size_t output_size, float* output) {
    assert(input_size % 32 == 0);

    for (size_t i = 0; i < output_size; ++i) {
        double out_row = 0.0;
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

            out_row += scale_x[block_index] * out_scale * y;
        }

        output[i] = out_row;
    }
}

#ifdef ENABLE_AVX512_BUILD

void bitnet_vmul_avx512_ref(const int8_t* x, const uint64_t* mask_add, const uint64_t* mask_sub,
                               const float* scale_x, float out_scale,
                               size_t input_size, size_t output_size, float* output)
{
    assert(input_size % 64 == 0);

    for (size_t i = 0; i < output_size; ++i) {
        double out_row = 0.0;

        for (size_t j = 0; j < input_size; j += 64) {
            const uint64_t ma = mask_add[j / 64];
            const uint64_t ms = mask_sub[j / 64];

            int32_t temp = 0;
            for (size_t k = 0; k < 64; ++k) {
                size_t index = j + k;
                int8_t x_val = x[index];

                int y = static_cast<int16_t>(x_val & -(int8_t)((ma >> k) & 1)) - 
                        static_cast<int16_t>(x_val & -(int8_t)((ms >> k) & 1));

                temp += y;
            }

            out_row += (double)scale_x[j / 256] * temp;
        }

        output[i] = out_row * out_scale;

        mask_add += input_size / 64;
        mask_sub += input_size / 64;
    }
}

static int16_t mm256_reduce_add_epi16(__m256i ymm0) {
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

void bitnet_vmul_avx512(const int8_t* x, const uint64_t* mask_add, const uint64_t* mask_sub,
                               const float* scale_x, float out_scale,
                               size_t input_size, size_t output_size, float* output)
{
    assert(input_size % 64 == 0);

    for (size_t i = 0; i < output_size; ++i) {
        double out_row = 0.0;

        for (size_t j = 0; j < input_size; j += 64) {
            const uint64_t ma = mask_add[j / 64];
            const uint64_t ms = mask_sub[j / 64];

            __m512i x_block = _mm512_loadu_epi8(x + j);
            __m512i x_lo = _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(x_block, 0));
            __m512i x_hi = _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(x_block, 1));

            __m512i sum = _mm512_setzero_si512();
            sum = _mm512_mask_add_epi16(sum, (__mmask32)ma, sum, x_lo);
            sum = _mm512_mask_sub_epi16(sum, (__mmask32)ms, sum, x_lo);
            sum = _mm512_mask_add_epi16(sum, (__mmask32)(ma >> 32), sum, x_hi);
            sum = _mm512_mask_sub_epi16(sum, (__mmask32)(ms >> 32), sum, x_hi);

            __m256i reduce = _mm256_add_epi16(_mm512_extracti64x4_epi64(sum, 0), _mm512_extracti64x4_epi64(sum, 1));
            int y = mm256_reduce_add_epi16(reduce);

            out_row += (double)scale_x[j / 256] * y;
        }

        output[i] = out_row * out_scale;

        mask_add += input_size / 64;
        mask_sub += input_size / 64;
    }
}

void bitnet_vmul_avx512_unroll(const int8_t* x, const uint64_t* mask_add, const uint64_t* mask_sub,
                               const float* scale_x, float out_scale,
                               size_t input_size, size_t output_size, float* output)
{
    assert(input_size % 64 == 0);

    for (int i = 0; i < output_size; ++i) {
        double row_sum = 0.0;

        const int blocks = input_size / 256;
        for (int j = 0; j < blocks; ++j) {
            const uint64_t ma0 = mask_add[j * 4 + 0];
            const uint64_t ma1 = mask_add[j * 4 + 1];
            const uint64_t ma2 = mask_add[j * 4 + 2];
            const uint64_t ma3 = mask_add[j * 4 + 3];

            const uint64_t ms0 = mask_sub[j * 4 + 0];
            const uint64_t ms1 = mask_sub[j * 4 + 1];
            const uint64_t ms2 = mask_sub[j * 4 + 2];
            const uint64_t ms3 = mask_sub[j * 4 + 3];

            __m512i x0 = _mm512_loadu_epi8(x + j * 256);
            __m512i x1 = _mm512_loadu_epi8(x + j * 256 + 64);
            __m512i x2 = _mm512_loadu_epi8(x + j * 256 + 128);
            __m512i x3 = _mm512_loadu_epi8(x + j * 256 + 192);

            __m512i x0lo = _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(x0, 0));
            __m512i x1lo = _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(x1, 0));
            __m512i x2lo = _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(x2, 0));
            __m512i x3lo = _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(x3, 0));

            x0 = _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(x0, 1));
            x1 = _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(x1, 1));
            x2 = _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(x2, 1));
            x3 = _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(x3, 1));

            __m512i s0 = _mm512_setzero_si512();
            __m512i s1 = _mm512_setzero_si512();
            __m512i s2 = _mm512_setzero_si512();
            __m512i s3 = _mm512_setzero_si512();

            s0 = _mm512_mask_add_epi16(s0, (__mmask32)ma0, s0, x0lo);
            s1 = _mm512_mask_add_epi16(s1, (__mmask32)ma1, s1, x1lo);
            s2 = _mm512_mask_add_epi16(s2, (__mmask32)ma2, s2, x2lo);
            s3 = _mm512_mask_add_epi16(s3, (__mmask32)ma3, s3, x3lo);

            s0 = _mm512_mask_add_epi16(s0, (__mmask32)(ma0 >> 32), s0, x0);
            s1 = _mm512_mask_add_epi16(s1, (__mmask32)(ma1 >> 32), s1, x1);
            s2 = _mm512_mask_add_epi16(s2, (__mmask32)(ma2 >> 32), s2, x2);
            s3 = _mm512_mask_add_epi16(s3, (__mmask32)(ma3 >> 32), s3, x3);

            s0 = _mm512_mask_sub_epi16(s0, (__mmask32)ms0, s0, x0lo);
            s1 = _mm512_mask_sub_epi16(s1, (__mmask32)ms1, s1, x1lo);
            s2 = _mm512_mask_sub_epi16(s2, (__mmask32)ms2, s2, x2lo);
            s3 = _mm512_mask_sub_epi16(s3, (__mmask32)ms3, s3, x3lo);

            s0 = _mm512_mask_sub_epi16(s0, (__mmask32)(ms0 >> 32), s0, x0);
            s1 = _mm512_mask_sub_epi16(s1, (__mmask32)(ms1 >> 32), s1, x1);
            s2 = _mm512_mask_sub_epi16(s2, (__mmask32)(ms2 >> 32), s2, x2);
            s3 = _mm512_mask_sub_epi16(s3, (__mmask32)(ms3 >> 32), s3, x3);

            s0 = _mm512_add_epi16(s0, s1);
            s2 = _mm512_add_epi16(s2, s3);
            s0 = _mm512_add_epi16(s0, s2);

            __m256i reduce = _mm256_add_epi16(_mm512_extracti64x4_epi64(s0, 0), _mm512_extracti64x4_epi64(s0, 1));
            int y = mm256_reduce_add_epi16(reduce);
            row_sum += (double)scale_x[j] * y;
        }

        for (size_t j = blocks * 256; j < input_size; j += 64) {
            const uint64_t ma = mask_add[j / 64];
            const uint64_t ms = mask_sub[j / 64];

            __m512i x_block = _mm512_loadu_epi8(x + j);
            __m512i x_lo = _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(x_block, 0));
            __m512i x_hi = _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(x_block, 1));

            __m512i sum = _mm512_setzero_si512();
            sum = _mm512_mask_add_epi16(sum, (__mmask32)ma, sum, x_lo);
            sum = _mm512_mask_sub_epi16(sum, (__mmask32)ms, sum, x_lo);
            sum = _mm512_mask_add_epi16(sum, (__mmask32)(ma >> 32), sum, x_hi);
            sum = _mm512_mask_sub_epi16(sum, (__mmask32)(ms >> 32), sum, x_hi);

            __m256i reduce = _mm256_add_epi16(_mm512_extracti64x4_epi64(sum, 0), _mm512_extracti64x4_epi64(sum, 1));
            int y = mm256_reduce_add_epi16(reduce);
            row_sum += (double)scale_x[j / 256] * y;
        }
        output[i] = row_sum * out_scale;

        mask_add += input_size / 64;
        mask_sub += input_size / 64;
    }
}

#endif // ENABLE_AVX512_BUILD
