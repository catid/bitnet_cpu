#include <random>
#include <iostream>

#include "math_functions.h"
#include "tools.h"

#include <immintrin.h>

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

bool test_mm256_reduce_add_epi16() {
    // Test case 1: AllOnes
    __m256i ymm0 = _mm256_set1_epi16(1);
    int16_t expected = 16;
    int16_t result = mm256_reduce_add_epi16(ymm0);
    if (result == expected) {
        std::cout << "Test case 1 passed!" << std::endl;
    } else {
        std::cout << "Test case 1 failed. Expected: " << expected << ", Got: " << result << std::endl;
        return false;
    }

    // Test case 2: Incremental
    ymm0 = _mm256_set_epi16(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
    expected = 120;
    result = mm256_reduce_add_epi16(ymm0);
    if (result == expected) {
        std::cout << "Test case 2 passed!" << std::endl;
    } else {
        std::cout << "Test case 2 failed. Expected: " << expected << ", Got: " << result << std::endl;
        return false;
    }

    // Test case 3: Negative
    ymm0 = _mm256_set1_epi16(-1);
    expected = -16;
    result = mm256_reduce_add_epi16(ymm0);
    if (result == expected) {
        std::cout << "Test case 3 passed!" << std::endl;
    } else {
        std::cout << "Test case 3 failed. Expected: " << expected << ", Got: " << result << std::endl;
        return false;
    }

    // Test case 4: Mixed
    ymm0 = _mm256_set_epi16(-1, 2, -3, 4, -5, 6, -7, 8, -9, 10, -11, 12, -13, 14, -15, 16);
    expected = 8;
    result = mm256_reduce_add_epi16(ymm0);
    if (result == expected) {
        std::cout << "Test case 4 passed!" << std::endl;
    } else {
        std::cout << "Test case 4 failed. Expected: " << expected << ", Got: " << result << std::endl;
        return false;
    }

    return true;
}

bool random_unit_test() {
    const size_t input_size = 3200;
    const size_t output_size = 8640;

    std::random_device rd;
    std::mt19937 gen(rd());
    //std::mt19937 gen(1234);
    std::uniform_int_distribution<int> int8_dist(-128, 127);
    std::uniform_real_distribution<float> float_dist(0.1f, 20.0f);

    std::vector<int8_t> x(input_size);

    // Note for AVX-512 version we ignore most of these because we process g256 at a time
    std::vector<float> scale_x(input_size / 32);
    const float out_scale = 3.5f;

    std::vector<float> out_simd(output_size);
    std::vector<float> out_ref(output_size);

    for (size_t i = 0; i < x.size(); ++i) {
        x[i] = int8_dist(gen);

        // Note: The kernel does not work with -128 as the input
        if (x[i] == -128) {
            x[i] = -127;
        }
    }

    for (size_t i = 0; i < scale_x.size(); ++i) {
        scale_x[i] = float_dist(gen);
    }

    std::cout << "Running non-AVX-512 version" << std::endl;

    std::vector<int8_t> mask_opcode(output_size * input_size);

    for (size_t i = 0; i < mask_opcode.size(); ++i) {
        int opcode = int8_dist(gen);
        if (opcode < -42) {
            mask_opcode[i] = -1;
        } else if (opcode > 42) {
            mask_opcode[i] = 1;
        } else {
            mask_opcode[i] = 0;
        }
    }

    bitnet_vmul_simd_opt(x.data(), mask_opcode.data(),
                scale_x.data(), out_scale,
                input_size, output_size, out_simd.data());

    bitnet_vmul_ref(x.data(), mask_opcode.data(),
                    scale_x.data(), out_scale,
                    input_size, output_size, out_ref.data());

    int sum = 0;
    for (size_t i = 0; i < output_size; ++i) {
        sum |= out_ref[i] != 0.f;
        double delta = std::abs(out_simd[i] - out_ref[i]);
        if (delta > 1e-5) {
            std::cout << "Error at output index " << i << ": " << out_simd[i] << " vs " << out_ref[i] << std::endl;
            return false;
        }
    }
    if (sum == 0) {
        std::cout << "Error: output is 0" << std::endl;
        return false;
    }
    return true;
}

#ifdef ENABLE_AVX512_BUILD

bool random_unit_test_avx512() {
    if (!CpuSupportsAVX512()) {
        std::cout << "AVX-512 not supported so not running that test" << std::endl;
        return true;
    }

    const size_t input_size = 3200;
    const size_t output_size = 8640;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> int8_dist(-128, 127);
    std::uniform_real_distribution<float> float_dist(0.1f, 20.0f);

    std::vector<int8_t> x(input_size);

    std::vector<float> scale_x(input_size / 256);
    const float out_scale = 3.5f;

    std::vector<float> out_simd(output_size);
    std::vector<float> out_ref(output_size);

    for (size_t i = 0; i < x.size(); ++i) {
        x[i] = int8_dist(gen);
    }

    for (size_t i = 0; i < scale_x.size(); ++i) {
        scale_x[i] = float_dist(gen);
    }

    std::cout << "Running AVX-512 version" << std::endl;

    std::vector<uint64_t> mask_add(output_size * input_size / 64);
    std::vector<uint64_t> mask_sub(output_size * input_size / 64);

    for (size_t i = 0; i < mask_add.size(); ++i) {
        mask_add[i] = gen();
        mask_sub[i] = gen();
    }

    bitnet_vmul_avx512_ref(x.data(), mask_add.data(), mask_sub.data(),
                scale_x.data(), out_scale,
                input_size, output_size, out_simd.data());

    bitnet_vmul_avx512_opt(x.data(), mask_add.data(), mask_sub.data(),
                    scale_x.data(), out_scale,
                    input_size, output_size, out_ref.data());

    for (size_t i = 0; i < output_size; ++i) {
        double delta = std::abs(out_simd[i] - out_ref[i]);
        if (delta > 1e-5) {
            std::cout << "Error at output index " << i << ": " << out_simd[i] << " vs " << out_ref[i] << ": delta=" << delta << std::endl;
            return false;
        }
    }
    return true;
}

#endif // ENABLE_AVX512_BUILD

int main() {
    if (!random_unit_test()) {
        std::cout << "random_unit_test failed!" << std::endl;
        return -1;
    }
    if (!test_mm256_reduce_add_epi16()) {
        std::cout << "test_mm256_reduce_add_epi16 failed!" << std::endl;
        return -1;
    }
#ifdef ENABLE_AVX512_BUILD
    if (!random_unit_test_avx512()) {
        std::cout << "random_unit_test_avx512 failed!" << std::endl;
        return -1;
    }
#endif
    std::cout << "Tests passed!" << std::endl;
    return 0;
}
