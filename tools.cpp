#include "tools.h"

#include <array>

template <typename T>
T* allocate_aligned_buffer(size_t size) {
    void* ptr = std::aligned_alloc(BITNET_CPU_ALIGNMENT, size * sizeof(T));
    if (ptr == nullptr) {
        return nullptr;
    }
    return static_cast<T*>(ptr);
}

void free_aligned_buffer(void* ptr) {
    std::free(ptr);
}

// Explicit instantiation of the template function
template int8_t* allocate_aligned_buffer<int8_t>(size_t size);
template float* allocate_aligned_buffer<float>(size_t size);

#ifdef ENABLE_AVX512_BUILD

bool CpuSupportsAVX512()
{
    std::array<int, 4> cpuidResult{};
    bool avx512FSupported = false, avx512BWSupported = false, avx512VLSupported = false;

    // Invoke CPUID with eax=7, ecx=0 to check extended features
    __asm__(
        "cpuid"
        : "=a"(cpuidResult[0]), "=b"(cpuidResult[1]), "=c"(cpuidResult[2]), "=d"(cpuidResult[3])
        : "a"(7), "c"(0)
    );

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
