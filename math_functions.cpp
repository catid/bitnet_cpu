#include "math_functions.h"
#include "tools.h"

#include <immintrin.h>
#include <cassert>
#include <omp.h>

// Sum all 16-bit values in a 256-bit vector
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

void bitnet_vmul_ref(const int8_t* x, const int8_t* mask_opcode,
                     const float* scale_x, float out_scale,
                     size_t input_size, size_t output_size, float* output) {
    assert(input_size % 32 == 0);

    for (size_t i = 0; i < output_size; ++i) {
        double out_row = 0.0;

        for (size_t j = 0; j < input_size; j += 32) {
            int32_t y = 0;

            for (size_t k = 0; k < 32; ++k) {
                size_t index = j + k;
                int32_t x_val = static_cast<int32_t>(x[index]);

                if (mask_opcode[index] < 0) {
                    x_val = -x_val;
                }
                else if (mask_opcode[index] > 0) {
                    // Do nothing
                } else {
                    x_val = 0;
                }

                y += x_val;
            }

            out_row += (double)scale_x[j / 256] * y;
        }

        output[i] = out_row * out_scale;

        mask_opcode += input_size;
    }
}

void bitnet_vmul_simd(const int8_t* x, const int8_t* mask_opcode,
                      const float* scale_x, float out_scale,
                      size_t input_size, size_t output_size, float* output) {
    assert(input_size % 32 == 0);

    for (size_t i = 0; i < output_size; ++i) {
        double out_row = 0.0;

        for (size_t j = 0; j < input_size; j += 32) {
            __m256i mask = _mm256_loadu_si256((__m256i*)(mask_opcode + j));
            __m256i x_block = _mm256_loadu_si256((__m256i*)(x + j));

            x_block = _mm256_sign_epi8(x_block, mask);

            __m256i x_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(x_block));
            __m256i x_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(x_block, 1));
            __m256i sum = _mm256_add_epi16(x_lo, x_hi); 

            int y = mm256_reduce_add_epi16(sum);
            out_row += (double)scale_x[j / 256] * y;
        }

        output[i] = out_row * out_scale;

        mask_opcode += input_size;
    }
}

void bitnet_vmul_simd_unroll(const int8_t* x, const int8_t* mask_opcode,
                      const float* scale_x, float out_scale,
                      size_t input_size, size_t output_size, float* output) {
    assert(input_size % 32 == 0);

    #pragma omp parallel for num_threads(20)
    for (size_t i = 0; i < output_size; ++i) {
        double out_row = 0.0;

        const int8_t* mask_row = mask_opcode + i * input_size;

        size_t j = 0;
        for (; j + 256 <= input_size; j += 256) {
            __m256i mask0 = _mm256_loadu_si256((__m256i*)(mask_row + j));
            __m256i mask1 = _mm256_loadu_si256((__m256i*)(mask_row + j + 32));
            __m256i mask2 = _mm256_loadu_si256((__m256i*)(mask_row + j + 64));
            __m256i mask3 = _mm256_loadu_si256((__m256i*)(mask_row + j + 96));
            __m256i mask4 = _mm256_loadu_si256((__m256i*)(mask_row + j + 128));
            __m256i mask5 = _mm256_loadu_si256((__m256i*)(mask_row + j + 160));
            __m256i mask6 = _mm256_loadu_si256((__m256i*)(mask_row + j + 192));
            __m256i mask7 = _mm256_loadu_si256((__m256i*)(mask_row + j + 224));

            __m256i x0 = _mm256_loadu_si256((__m256i*)(x + j));
            __m256i x1 = _mm256_loadu_si256((__m256i*)(x + j + 32));
            __m256i x2 = _mm256_loadu_si256((__m256i*)(x + j + 64));
            __m256i x3 = _mm256_loadu_si256((__m256i*)(x + j + 96));
            __m256i x4 = _mm256_loadu_si256((__m256i*)(x + j + 128));
            __m256i x5 = _mm256_loadu_si256((__m256i*)(x + j + 160));
            __m256i x6 = _mm256_loadu_si256((__m256i*)(x + j + 192));
            __m256i x7 = _mm256_loadu_si256((__m256i*)(x + j + 224));

            // Apply mask opcode to x[]
            x0 = _mm256_sign_epi8(x0, mask0);
            x1 = _mm256_sign_epi8(x1, mask1);
            x2 = _mm256_sign_epi8(x2, mask2);
            x3 = _mm256_sign_epi8(x3, mask3);
            x4 = _mm256_sign_epi8(x4, mask4);
            x5 = _mm256_sign_epi8(x5, mask5);
            x6 = _mm256_sign_epi8(x6, mask6);
            x7 = _mm256_sign_epi8(x7, mask7);

            // Sum within each 256 byte chunk
            __m256i x0_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(x0, 1));
            __m256i x1_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(x1, 1));
            __m256i x2_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(x2, 1));
            __m256i x3_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(x3, 1));
            __m256i x4_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(x4, 1));
            __m256i x5_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(x5, 1));
            __m256i x6_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(x6, 1));
            __m256i x7_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(x7, 1));

            x0 = _mm256_add_epi16(_mm256_cvtepi8_epi16(_mm256_castsi256_si128(x0)), x0_hi);
            x1 = _mm256_add_epi16(_mm256_cvtepi8_epi16(_mm256_castsi256_si128(x1)), x1_hi);
            x2 = _mm256_add_epi16(_mm256_cvtepi8_epi16(_mm256_castsi256_si128(x2)), x2_hi);
            x3 = _mm256_add_epi16(_mm256_cvtepi8_epi16(_mm256_castsi256_si128(x3)), x3_hi);
            x4 = _mm256_add_epi16(_mm256_cvtepi8_epi16(_mm256_castsi256_si128(x4)), x4_hi);
            x5 = _mm256_add_epi16(_mm256_cvtepi8_epi16(_mm256_castsi256_si128(x5)), x5_hi);
            x6 = _mm256_add_epi16(_mm256_cvtepi8_epi16(_mm256_castsi256_si128(x6)), x6_hi);
            x7 = _mm256_add_epi16(_mm256_cvtepi8_epi16(_mm256_castsi256_si128(x7)), x7_hi);

            // Reduce!
            x0 = _mm256_add_epi16(x0, x1);
            x2 = _mm256_add_epi16(x2, x3);
            x4 = _mm256_add_epi16(x4, x5);
            x6 = _mm256_add_epi16(x6, x7);

            x0 = _mm256_add_epi16(x0, x2);
            x4 = _mm256_add_epi16(x4, x6);

            x0 = _mm256_add_epi16(x0, x4);

            int y = mm256_reduce_add_epi16(x0);
            out_row += (double)scale_x[j / 256] * y;
        }

        for (; j < input_size; j += 32) {
            __m256i mask = _mm256_loadu_si256((__m256i*)(mask_row + j));
            __m256i x_block = _mm256_loadu_si256((__m256i*)(x + j));

            x_block = _mm256_sign_epi8(x_block, mask);

            __m256i x_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(x_block));
            __m256i x_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(x_block, 1));
            __m256i sum = _mm256_add_epi16(x_lo, x_hi); 

            int y = mm256_reduce_add_epi16(sum);
            out_row += (double)scale_x[j / 256] * y;
        }

        output[i] = out_row * out_scale;
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

    #pragma omp parallel for num_threads(20)
    for (int i = 0; i < output_size; ++i) {
        double row_sum = 0.0;

        const uint64_t* add_row = mask_add + i * input_size / 64;
        const uint64_t* sub_row = mask_sub + i * input_size / 64;

        const int blocks = input_size / 256;
        for (int j = 0; j < blocks; ++j) {
            const uint64_t ma0 = add_row[j * 4 + 0];
            const uint64_t ma1 = add_row[j * 4 + 1];
            const uint64_t ma2 = add_row[j * 4 + 2];
            const uint64_t ma3 = add_row[j * 4 + 3];

            const uint64_t ms0 = sub_row[j * 4 + 0];
            const uint64_t ms1 = sub_row[j * 4 + 1];
            const uint64_t ms2 = sub_row[j * 4 + 2];
            const uint64_t ms3 = sub_row[j * 4 + 3];

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
    }
}

#endif // ENABLE_AVX512_BUILD
