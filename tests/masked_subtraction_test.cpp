#include <gtest/gtest.h>
#include "math_functions.h"
#include "tools.h"
#include "common_tests.h"

#include <random>

class MaskedSubtractionTest : public ::testing::Test {
protected:
    const size_t input_size_1d = 3200;
    const size_t input_size_2d = 3200;
    const size_t output_size_2d = 8640;

    int8_t* x_1d;
    int8_t* mask_a_1d;
    int8_t* mask_b_1d;
    int8_t* out_1d;

    int8_t* x_2d;
    int8_t* mask_a_2d;
    int8_t* mask_b_2d;
    int8_t* out_2d;

    void SetUp() override {
        x_1d = allocate_aligned_buffer<int8_t>(input_size_1d);
        mask_a_1d = allocate_aligned_buffer<int8_t>(input_size_1d);
        mask_b_1d = allocate_aligned_buffer<int8_t>(input_size_1d);
        out_1d = allocate_aligned_buffer<int8_t>(input_size_1d);

        x_2d = allocate_aligned_buffer<int8_t>(input_size_2d);
        mask_a_2d = allocate_aligned_buffer<int8_t>(output_size_2d * input_size_2d);
        mask_b_2d = allocate_aligned_buffer<int8_t>(output_size_2d * input_size_2d);
        out_2d = allocate_aligned_buffer<int8_t>(output_size_2d);

        // Fill input arrays and masks with random values
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<int> dist(-128, 127);

        for (size_t i = 0; i < input_size_1d; ++i) {
            x_1d[i] = static_cast<int8_t>(dist(gen));
            mask_a_1d[i] = static_cast<int8_t>(dist(gen));
            mask_b_1d[i] = static_cast<int8_t>(dist(gen));
        }

        for (size_t i = 0; i < input_size_2d; ++i) {
            x_2d[i] = static_cast<int8_t>(dist(gen));
        }

        for (size_t i = 0; i < output_size_2d * input_size_2d; ++i) {
            mask_a_2d[i] = static_cast<int8_t>(dist(gen));
            mask_b_2d[i] = static_cast<int8_t>(dist(gen));
        }
    }

    void TearDown() override {
        free_aligned_buffer(x_1d);
        free_aligned_buffer(mask_a_1d);
        free_aligned_buffer(mask_b_1d);
        free_aligned_buffer(out_1d);

        free_aligned_buffer(x_2d);
        free_aligned_buffer(mask_a_2d);
        free_aligned_buffer(mask_b_2d);
        free_aligned_buffer(out_2d);
    }
};

TEST_F(MaskedSubtractionTest, OneDimensional) {
    subtract_masked_arrays(x_1d, mask_a_1d, mask_b_1d, out_1d, input_size_1d);
    EXPECT_TRUE(validate_result(x_1d, mask_a_1d, mask_b_1d, out_1d, input_size_1d));
}

TEST_F(MaskedSubtractionTest, TwoDimensional) {
    subtract_masked_arrays_2d(x_2d, mask_a_2d, mask_b_2d, out_2d, input_size_2d, output_size_2d);
    EXPECT_TRUE(validate_result_2d(x_2d, mask_a_2d, mask_b_2d, out_2d, input_size_2d, output_size_2d));
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
