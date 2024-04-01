#include <iostream>
#include <chrono>
#include <random>
#include <vector>
#include <utility>

#include <omp.h>

#include "math_functions.h"
#include "tools.h"

// Generated by generate_sizes.py
std::vector<std::pair<size_t, size_t>> ModelWeightSizes = {
    { 3200, 3200 },
    { 3200, 3200 },
    { 3200, 3200 },
    { 3200, 3200 },
    { 3200, 8640 },
    { 3200, 3200 },
    { 3200, 3200 },
    { 3200, 3200 },
    { 3200, 3200 },
    { 3200, 3200 },
    { 3200, 3200 },
    { 3200, 3200 },
    { 3200, 8640 },
    { 3200, 8640 },
    { 3200, 3200 },
    { 3200, 8640 },
    { 3200, 8640 },
    { 3200, 3200 },
    { 8640, 3200 },
    { 3200, 8640 },
    { 3200, 3200 },
    { 3200, 3200 },
    { 8640, 3200 },
    { 3200, 3200 },
    { 8640, 3200 },
    { 3200, 8640 },
    { 3200, 8640 },
    { 3200, 3200 },
    { 3200, 3200 },
    { 8640, 3200 },
    { 3200, 3200 },
    { 3200, 3200 },
    { 3200, 3200 },
    { 3200, 3200 },
    { 3200, 3200 },
    { 8640, 3200 },
    { 3200, 3200 },
    { 3200, 8640 },
    { 3200, 8640 },
    { 3200, 3200 },
    { 3200, 8640 },
    { 3200, 3200 },
    { 3200, 3200 },
    { 3200, 3200 },
    { 3200, 8640 },
    { 8640, 3200 },
    { 8640, 3200 },
    { 8640, 3200 },
    { 3200, 3200 },
    { 3200, 8640 },
    { 3200, 3200 },
    { 3200, 3200 },
    { 3200, 3200 },
    { 3200, 3200 },
    { 3200, 8640 },
    { 3200, 3200 },
    { 3200, 3200 },
    { 3200, 8640 },
    { 3200, 3200 },
    { 3200, 8640 },
    { 3200, 8640 },
    { 3200, 3200 },
    { 3200, 3200 },
    { 3200, 8640 },
    { 3200, 3200 },
    { 8640, 3200 },
    { 3200, 3200 },
    { 3200, 3200 },
    { 3200, 8640 },
    { 3200, 8640 },
    { 3200, 8640 },
    { 3200, 3200 },
    { 3200, 3200 },
    { 3200, 8640 },
    { 3200, 3200 },
    { 3200, 3200 },
    { 3200, 3200 },
    { 3200, 3200 },
    { 3200, 3200 },
    { 3200, 8640 },
    { 3200, 3200 },
    { 3200, 3200 },
    { 8640, 3200 },
    { 3200, 8640 },
    { 3200, 8640 },
    { 3200, 3200 },
    { 3200, 3200 },
    { 3200, 3200 },
    { 3200, 8640 },
    { 3200, 8640 },
    { 3200, 8640 },
    { 3200, 3200 },
    { 3200, 3200 },
    { 3200, 3200 },
    { 3200, 3200 },
    { 8640, 3200 },
    { 3200, 8640 },
    { 3200, 3200 },
    { 3200, 8640 },
    { 8640, 3200 },
    { 8640, 3200 },
    { 3200, 3200 },
    { 3200, 3200 },
    { 3200, 3200 },
    { 3200, 8640 },
    { 3200, 3200 },
    { 3200, 3200 },
    { 3200, 3200 },
    { 3200, 8640 },
    { 8640, 3200 },
    { 3200, 3200 },
    { 3200, 3200 },
    { 3200, 8640 },
    { 3200, 8640 },
    { 3200, 3200 },
    { 8640, 3200 },
    { 3200, 3200 },
    { 8640, 3200 },
    { 3200, 3200 },
    { 3200, 3200 },
    { 3200, 8640 },
    { 8640, 3200 },
    { 3200, 8640 },
    { 8640, 3200 },
    { 3200, 3200 },
    { 3200, 3200 },
    { 3200, 3200 },
    { 3200, 8640 },
    { 3200, 3200 },
    { 3200, 3200 },
    { 3200, 3200 },
    { 8640, 3200 },
    { 3200, 3200 },
    { 3200, 3200 },
    { 3200, 3200 },
    { 3200, 3200 },
    { 3200, 8640 },
    { 3200, 3200 },
    { 3200, 8640 },
    { 3200, 3200 },
    { 3200, 3200 },
    { 3200, 3200 },
    { 3200, 8640 },
    { 3200, 8640 },
    { 8640, 3200 },
    { 8640, 3200 },
    { 8640, 3200 },
    { 3200, 3200 },
    { 3200, 8640 },
    { 8640, 3200 },
    { 3200, 3200 },
    { 3200, 8640 },
    { 3200, 3200 },
    { 3200, 3200 },
    { 8640, 3200 },
    { 8640, 3200 },
    { 3200, 3200 },
    { 3200, 8640 },
    { 3200, 8640 },
    { 3200, 3200 },
    { 3200, 3200 },
    { 3200, 8640 },
    { 3200, 8640 },
    { 3200, 8640 },
    { 3200, 3200 },
    { 3200, 8640 },
    { 3200, 8640 },
    { 3200, 3200 },
    { 3200, 3200 },
    { 3200, 3200 },
    { 3200, 3200 },
    { 3200, 3200 },
    { 3200, 3200 },
    { 3200, 3200 },
    { 3200, 8640 },
    { 3200, 3200 },
    { 3200, 3200 },
    { 8640, 3200 },
    { 3200, 3200 },
    { 3200, 8640 },
    { 3200, 3200 },
    { 3200, 3200 },
};

const size_t NUM_BENCHMARK_ITERATIONS = 1;

int main() {

    std::cout << "Preallocating buffers..." << std::endl;

    // Preallocate input vectors, mask vectors, scale vectors, and output buffers
    std::vector<int8_t*> input_vectors;
    std::vector<int8_t*> mask_add_vectors;
    std::vector<int8_t*> mask_sub_vectors;
    std::vector<uint64_t*> mask64_add_vectors;
    std::vector<uint64_t*> mask64_sub_vectors;
    std::vector<float*> scale_x_vectors;
    const float out_scale = 3.25f;
    std::vector<float*> output_buffers;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> int8_dist(-128, 127);
    std::uniform_real_distribution<float> float_dist(0.1f, 20.0f);

    for (const auto& tensor_size : ModelWeightSizes) {
        size_t input_size = tensor_size.first;
        size_t output_size = tensor_size.second;

        int8_t* input_vector = allocate_aligned_buffer<int8_t>(input_size);
        float* scale_x_vector = allocate_aligned_buffer<float>(input_size / 32);
        float* output_buffer = allocate_aligned_buffer<float>(output_size);

        for (size_t i = 0; i < input_size; ++i) {
            input_vector[i] = static_cast<int8_t>(int8_dist(gen));
        }

        if (CpuSupportsAVX512()) {
            uint64_t* mask64_add_vector = allocate_aligned_buffer<uint64_t>(output_size * input_size / 64);
            uint64_t* mask64_sub_vector = allocate_aligned_buffer<uint64_t>(output_size * input_size / 64);
            for (size_t i = 0; i < output_size * input_size / 64; ++i) {
                mask64_add_vector[i] = gen();
                mask64_sub_vector[i] = gen();
            }
            mask64_add_vectors.push_back(mask64_add_vector);
            mask64_sub_vectors.push_back(mask64_sub_vector);
        } else {
            int8_t* mask_add_vector = allocate_aligned_buffer<int8_t>(output_size * input_size);
            int8_t* mask_sub_vector = allocate_aligned_buffer<int8_t>(output_size * input_size);
            for (size_t i = 0; i < output_size * input_size; ++i) {
                mask_add_vector[i] = static_cast<int8_t>(int8_dist(gen) >= 0 ? -1 : 0);
                mask_sub_vector[i] = static_cast<int8_t>(int8_dist(gen) >= 0 ? -1 : 0);
            }
            mask_add_vectors.push_back(mask_add_vector);
            mask_sub_vectors.push_back(mask_sub_vector);
        }

        for (size_t i = 0; i < input_size / 32; ++i) {
            scale_x_vector[i] = float_dist(gen);
        }

        input_vectors.push_back(input_vector);
        scale_x_vectors.push_back(scale_x_vector);
        output_buffers.push_back(output_buffer);
    }

    std::cout << "Benchmarking model..." << std::endl;

    double avg_time = 0.0;

#ifdef ENABLE_AVX512_BUILD

    if (CpuSupportsAVX512())
    {
        {
            auto start_time = std::chrono::high_resolution_clock::now();
            for (size_t j = 0; j < ModelWeightSizes.size(); ++j) {
                size_t input_size = ModelWeightSizes[j].first;
                size_t output_size = ModelWeightSizes[j].second;
                bitnet_vmul_avx512_opt(input_vectors[j], mask64_add_vectors[j], mask64_sub_vectors[j],
                                        scale_x_vectors[j], out_scale, input_size, output_size,
                                        output_buffers[j]);
            }
            auto end_time = std::chrono::high_resolution_clock::now();
            std::cout << "Warmup took " << std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count() << " milliseconds" << std::endl;
        }

        // Benchmark iterations
        auto start_time = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < NUM_BENCHMARK_ITERATIONS; ++i) {
            #pragma omp parallel for
            for (size_t j = 0; j < ModelWeightSizes.size(); ++j) {
                size_t input_size = ModelWeightSizes[j].first;
                size_t output_size = ModelWeightSizes[j].second;
                bitnet_vmul_avx512_opt(input_vectors[j], mask64_add_vectors[j], mask64_sub_vectors[j],
                                        scale_x_vectors[j], out_scale, input_size, output_size,
                                        output_buffers[j]);
            }
        }
        auto end_time = std::chrono::high_resolution_clock::now();

        // Calculate average time per iteration
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        avg_time = static_cast<double>(duration.count()) / NUM_BENCHMARK_ITERATIONS;
    }
    else
#endif // ENABLE_AVX512_BUILD
    {
        {
            auto start_time = std::chrono::high_resolution_clock::now();
            for (size_t j = 0; j < ModelWeightSizes.size(); ++j) {
                size_t input_size = ModelWeightSizes[j].first;
                size_t output_size = ModelWeightSizes[j].second;
                bitnet_vmul_simd_unrolled(input_vectors[j], mask_add_vectors[j], mask_sub_vectors[j],
                                        scale_x_vectors[j], out_scale, input_size, output_size,
                                        output_buffers[j]);
            }
            auto end_time = std::chrono::high_resolution_clock::now();
            std::cout << "Warmup took " << std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count() << " milliseconds" << std::endl;
        }

        // Benchmark iterations
        auto start_time = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < NUM_BENCHMARK_ITERATIONS; ++i) {
            for (size_t j = 0; j < ModelWeightSizes.size(); ++j) {
                size_t input_size = ModelWeightSizes[j].first;
                size_t output_size = ModelWeightSizes[j].second;
                bitnet_vmul_simd_unrolled(input_vectors[j], mask_add_vectors[j], mask_sub_vectors[j],
                                        scale_x_vectors[j], out_scale, input_size, output_size,
                                        output_buffers[j]);
            }
        }
        auto end_time = std::chrono::high_resolution_clock::now();

        // Calculate average time per iteration
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        avg_time = static_cast<double>(duration.count()) / NUM_BENCHMARK_ITERATIONS;
    }

    // Print benchmark results
    std::cout << "Benchmark Results:" << std::endl;
    std::cout << "Number of Layers: " << ModelWeightSizes.size() << std::endl;
    std::cout << "Number of Benchmark Iterations: " << NUM_BENCHMARK_ITERATIONS << std::endl;
    std::cout << "Average Time per Iteration: " << avg_time << " milliseconds" << std::endl;

    // Free allocated buffers
    for (size_t i = 0; i < ModelWeightSizes.size(); ++i) {
        free_aligned_buffer(input_vectors[i]);
        if (CpuSupportsAVX512()) {
            free_aligned_buffer(mask64_add_vectors[i]);
            free_aligned_buffer(mask64_sub_vectors[i]);
        } else {
            free_aligned_buffer(mask_add_vectors[i]);
            free_aligned_buffer(mask_sub_vectors[i]);
        }
        free_aligned_buffer(scale_x_vectors[i]);
        free_aligned_buffer(output_buffers[i]);
    }

    return 0;
}