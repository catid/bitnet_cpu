# BitNet CPU

Some tests to see if the BitNet idea works on a CPU:

The best SIMD intrinsics for this do not exist on most Intel computers, but you can use AVX512 _mm256_mask_add_epi16() and _mm256_mask_sub_epi16() on Intel servers to speed this up a lot.  With those intrinsics, the weights can be 2 bits per parameter, and the model might competitive for memory/compute efficiency.

Without those intrinsics, it takes about ~1.5 seconds to run through all the heavy layers of the model.  If this was properly parallelized, you might be able to achieve a speedup of 16x or so.  So let's say the best we could do is about 8 tokens per second with this type of model.

I optimized and tested an AVX-512 version, results below.

Also to achieve this performance you have to use 16 bits per weight, so it does not have size advantages over normal models either despite being ternary.

```
(base) ➜  build git:(master) ./benchmark_model
Preallocating buffers...
Bencharking model...
Benchmark Results:
Number of Layers: 182
Number of Benchmark Iterations: 100
Average Time per Iteration: 1264.97 milliseconds
```

I also tried converting the model weights to C++ code, which was kind of funny.  I put the results in the `lol_bitnet_to_cpp` folder.  This experiment obviously failed: The C++ files are 200 MB and the compiler takes more than 10 hours to compile each one.

## Setup

```bash
git clone https://github.com/catid/bitnet_cpu.git
cd bitnet_cpu
mkdir build
cd build
cmake ..
make -j
```

## Results for AVX-512

I optimized the reference inference kernel using AVX-512 on a rented Intel Xeon W-2295:

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

This would be about ~40 tokens per second on CPUs with AVX-512.

AMD Ryzen 5 7535HS CPU achieves about 7.4 tokens/second for Gemma 3B.
