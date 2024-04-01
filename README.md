# BitNet CPU

This repo provides some tests to see how well the BitNet inference idea works on a CPU.  This is based on the open-source reproduction here: https://huggingface.co/1bitLLM/bitnet_b1_58-3B/ which is reproducing the results from this series of Microsoft papers:

* https://github.com/microsoft/unilm/blob/master/bitnet/The-Era-of-1-bit-LLMs__Training_Tips_Code_FAQ.pdf
* "The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits" https://arxiv.org/abs/2402.17764
* "BitNet: Scaling 1-bit Transformers for Large Language Models" https://arxiv.org/abs/2310.11453
* Maybe more here? https://github.com/microsoft/unilm/tree/master/bitnet

## Setup

```bash
git clone https://github.com/catid/bitnet_cpu.git
cd bitnet_cpu
mkdir build
cd build
cmake ..
make -j
```

## CPU Approach 1: AVX-512

The best SIMD intrinsics for this do not exist on most Intel computers, but you can use AVX512 _mm256_mask_add_epi16() and _mm256_mask_sub_epi16() on Intel servers to speed this up a lot.  With those intrinsics, the weights can be 2 bits per parameter, and the model might be competitive for memory/compute efficiency.  I optimized the reference inference kernel using AVX-512 on a rented Intel Xeon W-2295.  This is using OMP for multi-thread optimization, and some careful unrolling for pipeline parallelism.

```bash
catid@project512:~/sources/bitnet_cpu/build$ ./tests/math_test 
Running non-AVX-512 version
Test case 1 passed!
Test case 2 passed!
Test case 3 passed!
Test case 4 passed!
Running AVX-512 version
Tests passed!

catid@project512:~/sources/bitnet_cpu/build$ ./benchmark_model 
Preallocating buffers...
Benchmarking model...
Warmup took 131 milliseconds
Benchmark Results:
Number of Layers: 182
Number of Benchmark Iterations: 1
Average Time per Iteration: 25 milliseconds
```

This would be about 1000/25 = ~40 tokens per second on CPUs with AVX-512.

Compare this to AMD Ryzen 5 7535HS CPU achieves about 7.4 tokens/second for Gemma 3B, and it's clear that the BitNet inference kernel is competitive with 8-bit inference on CPU in some cases.

## CPU Approach 2: AVX2

Using the _mm256_sign_epi8() intrinsic, we can use 1 byte per parameter (8-bit quant) to implement an efficient BitNet kernel that is portable to most Intel CPUs using AVX2.  This is using the same unrolling/multi-threading optimizations as the AVX-512 version.

Running on an 12th Gen Intel(R) Core(TM) i9-12900K with 20 threads:

```
(base) ➜  build git:(master) ✗ ./tests/math_test
Running non-AVX-512 version
Tests passed!

(base) ➜  build git:(master) ✗ ./benchmark_model
Preallocating buffers...
Benchmarking model...
Warmup took 35 milliseconds
Benchmark Results:
Number of Layers: 182
Number of Benchmark Iterations: 100
Average Time per Iteration: 34.85 milliseconds
```

We are getting about 1000/35 = ~28 tokens per second on most Intel CPUs with AVX2.

## CPU Approach 3: LOL

I also tried converting the model weights to C++ code, which was kind of funny.  I put the results in the `lol_bitnet_to_cpp` folder.  This experiment obviously failed: The C++ files are 200 MB and the compiler takes more than 10 hours to compile each one.
