#include <cstdio>
#include <vector>
#include <cstdint>
#include <cstring>
#include <string>
#include <cstdlib>   // strtoul
#include <cuda_runtime.h>
#include "cpu_xof.cpp"
#include "sanity_vector.cpp"

// GPU helpers
extern "C" void blake3_xof_cuda(const void*, size_t, void*, size_t, int, cudaStream_t);
extern "C" void blake3_xof_layout(int counter_offset, int b0_mode);

static void hexdump64(const uint8_t* p) {
    for (int i = 0; i < 64; ++i) {
        std::printf("%02X%s", p[i], ((i & 15)==15 ? "\n" : " "));
    }
}

static void hexdump_block(const char* tag, size_t idx, const uint8_t* p) {
    std::printf("[%s] blk %zu\n", tag, idx);
    hexdump64(p);
}

int main(int argc, char** argv) {
    // CLI: --dump N, --dump-all, --gpu-counter-offset N, --gpu-b0-half <mode>
    size_t dump_blocks = 0;
    bool dump_all = false;
    int gpu_counter_offset = 0;
    int gpu_b0_mode = 0; // 0=lower,1=upper,2=upper_xor_root,3=lower_xor_root

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--dump" && i+1 < argc) {
            dump_blocks = std::strtoul(argv[++i], nullptr, 10);
        } else if (a == "--dump-all") {
            dump_all = true;
        } else if (a == "--gpu-counter-offset" && i+1 < argc) {
            gpu_counter_offset = int(std::strtol(argv[++i], nullptr, 10));
        } else if (a == "--gpu-b0-half" && i+1 < argc) {
            std::string m = argv[++i];
            if      (m == "lower")          gpu_b0_mode = 0;
            else if (m == "upper")          gpu_b0_mode = 1;
            else if (m == "upper_xor_root") gpu_b0_mode = 2;
            else if (m == "lower_xor_root") gpu_b0_mode = 3;
            else {
                std::fprintf(stderr, "Unknown --gpu-b0-half mode: %s\n", m.c_str());
                return 2;
            }
        } else {
            std::fprintf(stderr,
                "Usage: %s [--dump N|--dump-all] [--gpu-counter-offset N] "
                "[--gpu-b0-half lower|upper|upper_xor_root|lower_xor_root]\n",
                argv[0]);
            return 2;
        }
    }

    // Program the GPU layout (counter offset + blk0 second half wiring)
    blake3_xof_layout(gpu_counter_offset, gpu_b0_mode);

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
    constexpr size_t MAT = 1'607'680;
    uint8_t *d_seeds = nullptr, *d_xof = nullptr;
    cudaMalloc(&d_seeds, h_seeds.size());
    cudaMemcpy(d_seeds, h_seeds.data(), h_seeds.size(), cudaMemcpyHostToDevice);

    cudaMalloc(&d_xof, MAT);
    blake3_xof_cuda(d_seeds, 240, d_xof, MAT, BATCH, 0);
    cudaDeviceSynchronize();

    std::vector<uint8_t> xof_gpu(MAT);
    cudaMemcpy(xof_gpu.data(), d_xof, MAT, cudaMemcpyDeviceToHost);

    // Optional dumps:
    const size_t BLOCKS = MAT / 64;
    if (dump_all) {
        for (size_t b = 0; b < BLOCKS; ++b) {
            hexdump_block("CPU", b, &xof_cpu[b*64]);
            hexdump_block("GPU", b, &xof_gpu[b*64]);
        }
    } else if (dump_blocks) {
        for (size_t b = 0; b < dump_blocks && b < BLOCKS; ++b) {
            hexdump_block("CPU", b, &xof_cpu[b*64]);
            hexdump_block("GPU", b, &xof_gpu[b*64]);
        }
    }

    // --- Compare all blocks (25,120 × 64 B) ---
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

        // Also show blk0 to compare starting point unconditionally
        std::puts("\n[blk0 sanity]");
        std::puts("--- CPU ---"); hexdump64(&xof_cpu[0]);
        std::puts("--- GPU ---"); hexdump64(&xof_gpu[0]);
        cudaFree(d_seeds);
        cudaFree(d_xof);
        return 2;
    }

    cudaFree(d_seeds);
    cudaFree(d_xof);
    return 0;
}
