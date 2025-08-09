#include <cstdio>
#include <vector>
#include <cstdint>
#include <cstring>
#include <cuda_runtime.h>
#include "cpu_xof.cpp"
#include "sanity_vector.cpp"

// GPU helper we just added:
extern "C" void blake3_xof_cuda(const void*, size_t, void*, size_t, int, cudaStream_t);

static void hexdump64(const uint8_t* p) {
    for (int i = 0; i < 64; ++i) {
        std::printf("%02X%s", p[i], ((i & 15)==15 ? "\n" : " "));
    }
}

int main() {
    const int BATCH = 1;
    // Seed = "a" × 240
    std::vector<uint8_t> h_seeds(BATCH * 240, 'a');

    // --- CPU XOF ---
    std::vector<uint8_t> xof_cpu;
    blake3_xof_cpu(h_seeds.data(), xof_cpu);

    // Sanity: check the first 64 bytes vs Elixir result
    if (std::memcmp(xof_cpu.data(), ELIXIR_Ax240_FIRST64, 64) != 0) {
        std::puts("❌ CPU XOF[0..63] != Elixir expected");
        std::puts("--- CPU ---"); hexdump64(xof_cpu.data());
        std::puts("--- EXP ---"); hexdump64(ELIXIR_Ax240_FIRST64);
        return 1;
    } else {
        std::puts("✅ CPU XOF[0..63] matches Elixir.");
    }

    // --- GPU XOF ---
    uint8_t *d_seeds = nullptr, *d_xof = nullptr;
    cudaMalloc(&d_seeds, h_seeds.size());
    cudaMemcpy(d_seeds, h_seeds.data(), h_seeds.size(), cudaMemcpyHostToDevice);

    cudaMalloc(&d_xof, MAT);
    blake3_xof_cuda(d_seeds, 240, d_xof, MAT, BATCH, 0);
    cudaDeviceSynchronize();

    std::vector<uint8_t> xof_gpu(MAT);
    cudaMemcpy(xof_gpu.data(), d_xof, MAT, cudaMemcpyDeviceToHost);

    // --- Compare all blocks (25,120 × 64 B = 1,607,680 B) ---
    const size_t BLOCKS = MAT / 64;
    size_t first_mismatch = BLOCKS; // sentinel
    for (size_t b = 0; b < BLOCKS; ++b) {
        if (std::memcmp(&xof_cpu[b*64], &xof_gpu[b*64], 64) != 0) {
            first_mismatch = b;
            break;
        }
    }

    if (first_mismatch == BLOCKS) {
        std::puts("✅ GPU XOF matches CPU XOF for all 25,120 blocks.");
    } else {
        std::printf("❌ First mismatch at block %zu (offset %zu bytes)\n",
                    first_mismatch, first_mismatch*64);
        std::puts("--- CPU ---"); hexdump64(&xof_cpu[first_mismatch*64]);
        std::puts("--- GPU ---"); hexdump64(&xof_gpu[first_mismatch*64]);

        // Optional: also show blk0 to confirm both sides agree at the start
        std::puts("\n[blk0 sanity]");
        std::puts("--- CPU ---"); hexdump64(&xof_cpu[0]);
        std::puts("--- GPU ---"); hexdump64(&xof_gpu[0]);
        return 2;
    }

    cudaFree(d_seeds);
    cudaFree(d_xof);
    return 0;
}