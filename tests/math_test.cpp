#include <random>
#include <iostream>

#include "math_functions.h"
#include "tools.h"

bool random_unit_test() {
    const size_t input_size = 3200;
    const size_t output_size = 8640;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int8_t> int8_dist(-128, 127);
    std::uniform_real_distribution<float> float_dist(0.1f, 20.0f);

    std::vector<int8_t> x(input_size);
    std::vector<int8_t> mask_add(output_size * input_size);
    std::vector<int8_t> mask_sub(output_size * input_size);
    std::vector<float> scale_x(input_size / 32);
    std::vector<float> scale_y(input_size * output_size / 32);
    std::vector<float> out_simd(output_size);
    std::vector<float> out_ref(output_size);

    for (size_t i = 0; i < x.size(); ++i) {
        x[i] = int8_dist(gen);
    }

    for (size_t i = 0; i < mask_add.size(); ++i) {
        mask_add[i] = int8_dist(gen) >= 0 ? -1 : 0;
        mask_sub[i] = int8_dist(gen) >= 0 ? -1 : 0;
    }

    for (size_t i = 0; i < scale_x.size(); ++i) {
        scale_x[i] = float_dist(gen);
    }

    for (size_t i = 0; i < scale_y.size(); ++i) {
        scale_y[i] = float_dist(gen);
    }

    bitnet_vmul_simd_unrolled(x.data(), mask_add.data(), mask_sub.data(),
                scale_x.data(), scale_y.data(),
                input_size, output_size, out_simd.data());

    bitnet_vmul_ref(x.data(), mask_add.data(), mask_sub.data(),
                    scale_x.data(), scale_y.data(),
                    input_size, output_size, out_ref.data());

    for (size_t i = 0; i < output_size; ++i) {
        double delta = std::abs(out_simd[i] - out_ref[i]);
        if (delta > 1e-5) {
            std::cout << "Error at output index " << i << ": " << out_simd[i] << " vs " << out_ref[i] << std::endl;
            return false;
        }
    }
    return true;
}

int main() {
    bool success = random_unit_test();
    if (!success) {
        std::cout << "Random unit test failed!" << std::endl;
        return -1;
    }
    std::cout << "Tests passed!" << std::endl;
    return 0;
}
