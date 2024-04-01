#include "tools.h"

#include <array>
#include <cstdint>

#ifdef _MSC_VER
#include <malloc.h>
#include <intrin.h>
#else
#include <cstdlib>
#endif

template <typename T>
T* allocate_aligned_buffer(size_t size) {
#ifdef _MSC_VER
    void* ptr = _aligned_malloc(size * sizeof(T), BITNET_CPU_ALIGNMENT);
#else
    void* ptr = std::aligned_alloc(BITNET_CPU_ALIGNMENT, size * sizeof(T));
#endif
    if (ptr == nullptr) {
        return nullptr;
    }
    return static_cast<T*>(ptr);
}

void free_aligned_buffer(void* ptr) {
#ifdef _MSC_VER
    _aligned_free(ptr);
#else
    std::free(ptr);
#endif
}

// Explicit instantiation of the template function
template int8_t* allocate_aligned_buffer<int8_t>(size_t size);
template float* allocate_aligned_buffer<float>(size_t size);
template uint64_t* allocate_aligned_buffer<uint64_t>(size_t size);
// Add more here...


#ifdef ENABLE_AVX512_BUILD

bool CpuSupportsAVX512()
{
    std::array<int, 4> cpuidResult{};
    bool avx512FSupported = false, avx512BWSupported = false, avx512VLSupported = false;

#ifdef _MSC_VER
    __cpuid(cpuidResult.data(), 7);
#else
    // Invoke CPUID with eax=7, ecx=0 to check extended features
    __asm__(
        "cpuid"
        : "=a"(cpuidResult[0]), "=b"(cpuidResult[1]), "=c"(cpuidResult[2]), "=d"(cpuidResult[3])
        : "a"(7), "c"(0)
    );
#endif

    // Check AVX512F - Bit 16 of EBX
    avx512FSupported = (cpuidResult[1] & (1 << 16)) != 0;

    // Check AVX512BW - Bit 30 of EBX
    avx512BWSupported = (cpuidResult[1] & (1 << 30)) != 0;

    // Check AVX512VL - Bit 31 of EBX
    avx512VLSupported = (cpuidResult[1] & (1U << 31)) != 0;

    return avx512FSupported && avx512BWSupported && avx512VLSupported;
}

#else // ENABLE_AVX512_BUILD

bool CpuSupportsAVX512()
{
    return false;
}

#endif // ENABLE_AVX512_BUILD
