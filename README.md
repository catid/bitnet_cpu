# BitNet CPU

Some tests to see if the BitNet idea works on a CPU:

The best SIMD intrinsics for this do not exist on most Intel computers, but you can use _mm256_mask_add_epi16() and _mm256_mask_sub_epi16() to speed this up a lot.  With those intrinsics, the weights can be 2 bits per parameter, and the model might competitive for memory/compute efficiency.

Without those intrinsics, it takes about 1.5 seconds to run through all the heavy layers of the model.  If this was properly parallelized, you might be able to achieve a speedup of 16x or so.  So let's say the best we could do is about 8 tokens per second with this type of model.

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
