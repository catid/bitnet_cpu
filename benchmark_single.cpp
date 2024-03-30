#include <iostream>
#include <chrono>
#include <random>
#include <immintrin.h>

#include "math_functions.h"
#include "tools.h"

const size_t INPUT_SIZE = 3200;
const size_t OUTPUT_SIZE = 8640;
const size_t NUM_WARMUP_ITERATIONS = 10;
const size_t NUM_BENCHMARK_ITERATIONS = 1000;

int main() {
    // Allocate aligned buffers
    int8_t* x = allocate_aligned_buffer<int8_t>(INPUT_SIZE);
    int8_t* mask_a = allocate_aligned_buffer<int8_t>(OUTPUT_SIZE * INPUT_SIZE);
    int8_t* mask_b = allocate_aligned_buffer<int8_t>(OUTPUT_SIZE * INPUT_SIZE);
    int8_t* out = allocate_aligned_buffer<int8_t>(OUTPUT_SIZE);

    // Fill input array and masks with random values
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dist(-128, 127);

    for (size_t i = 0; i < INPUT_SIZE; ++i) {
        x[i] = static_cast<int8_t>(dist(gen));
    }

    for (size_t i = 0; i < OUTPUT_SIZE * INPUT_SIZE; ++i) {
        mask_a[i] = static_cast<int8_t>(dist(gen));
        mask_b[i] = static_cast<int8_t>(dist(gen));
    }

    // Warmup iterations
    for (size_t i = 0; i < NUM_WARMUP_ITERATIONS; ++i) {
        subtract_masked_arrays_2d(x, mask_a, mask_b, out, INPUT_SIZE, OUTPUT_SIZE);
    }

    // Benchmark iterations
    auto start_time = std::chrono::high_resolution_clock::now();
    uint64_t start = __rdtsc();
    for (size_t i = 0; i < NUM_BENCHMARK_ITERATIONS; ++i) {
        subtract_masked_arrays_2d(x, mask_a, mask_b, out, INPUT_SIZE, OUTPUT_SIZE);
    }
    uint64_t end = __rdtsc();
    auto end_time = std::chrono::high_resolution_clock::now();

    // Calculate average CPU cycles and microseconds per iteration
    uint64_t total_cycles = end - start;
    double avg_cycles = static_cast<double>(total_cycles) / NUM_BENCHMARK_ITERATIONS;

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    double avg_time = static_cast<double>(duration.count()) / NUM_BENCHMARK_ITERATIONS;

    // Print benchmark results
    std::cout << "Benchmark Results:" << std::endl;
    std::cout << "Input Size: " << INPUT_SIZE << std::endl;
    std::cout << "Output Size: " << OUTPUT_SIZE << std::endl;
    std::cout << "Number of Warmup Iterations: " << NUM_WARMUP_ITERATIONS << std::endl;
    std::cout << "Number of Benchmark Iterations: " << NUM_BENCHMARK_ITERATIONS << std::endl;
    std::cout << "Average CPU Cycles per Iteration: " << avg_cycles << std::endl;
    std::cout << "Average Time per Iteration: " << avg_time << " microseconds" << std::endl;

    // Free allocated buffers
    free_aligned_buffer(x);
    free_aligned_buffer(mask_a);
    free_aligned_buffer(mask_b);
    free_aligned_buffer(out);

    return 0;
}
