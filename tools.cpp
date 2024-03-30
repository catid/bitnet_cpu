#include "tools.h"

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
