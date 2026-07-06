#pragma once
#include <cuda_runtime.h>
#include <iostream>

#define CHECK_CUDA(val) check_cuda_error((val), #val, __FILE__, __LINE__)

inline void check_cuda_error(cudaError_t result, char const *const func, const char *const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
            file << ":" << line << " '" << func << "' \n";
        cudaDeviceReset();
        exit(99);
    }
}

// Memory alignment and mathematical utility constants
#define BLOCK_SIZE 256
#define TILE_WIDTH 16
#define TILE_HEIGHT 16
