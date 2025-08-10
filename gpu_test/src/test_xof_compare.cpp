#include <cstdio>
#include <vector>
#include <cstdint>
#include <cstring>
#include <cstdlib>      // strtoul
#include <cuda_runtime.h>

#include "cpu_xof.cpp"
#include "sanity_vector.cpp"

// GPU helpers
extern "C" void blake3_xof_cuda(const void*, size_t, void*, size_t, int, cudaStream_t);
extern "C" void blake3_xof_layout(int counter_offset, int b0_mode, int xof_flag_mode);

static void hexdump64(const uint8_t* p) {
    for (int i = 0; i < 64; ++i) {
        std::printf("%02X%s", p[i], ((i & 15)==15 ? "\n" : " "));
    }
}

int main(int argc, char** argv) {
    int dump_blocks = 0;     // how many 64-B blocks to print
    int counter_off = 0;
    int b0_mode     = 0;     // 0=spec hi^precv (default)
    int flag_mode   = 0;     // 0=ROOT (spec), 1=DERIVE (debug)
    enum class SeedKind { A, Zero } seed_kind = SeedKind::A;

    // CLI
    for (int i=1; i<argc; ++i) {
        if (!std::strcmp(argv[i], "--dump") && i+1 < argc) {
            dump_blocks = (int)std::strtoul(argv[++i], nullptr, 10);
        } else if (!std::strcmp(argv[i], "--gpu-counter-offset") && i+1 < argc) {
            counter_off = (int)std::strtoul(argv[++i], nullptr, 10);
        } else if (!std::strcmp(argv[i], "--gpu-b0-half") && i+1 < argc) {
            const char* s = argv[++i];
            if      (!std::strcmp(s,"spec"))             b0_mode = 0;
            else if (!std::strcmp(s,"tmpL"))             b0_mode = 1;
            else if (!std::strcmp(s,"tmpH"))             b0_mode = 2;
            else if (!std::strcmp(s,"rootcmpL"))         b0_mode = 3;
            else if (!std::strcmp(s,"rootcmpH"))         b0_mode = 4;
            else if (!std::strcmp(s,"rootcmpH_xor_cv"))  b0_mode = 5;
            else b0_mode = 0;
        } else if (!std::strcmp(argv[i], "--gpu-xof-flags") && i+1 < argc) {
            const char* s = argv[++i];
            flag_mode = (!std::strcmp(s,"derive") ? 1 : 0);
        } else if (!std::strcmp(argv[i], "--seed") && i+1 < argc) {
            const char* s = argv[++i];
            if (!std::strcmp(s,"zero") || !std::strcmp(s,"zeros"))
                seed_kind = SeedKind::Zero;
            else
                seed_kind = SeedKind::A; // default
        }
    }
    blake3_xof_layout(counter_off, b0_mode, flag_mode);

    const int BATCH = 1;
    const size_t SEEDLEN = 240;
    const size_t MAT = 1'607'680; // 25,120 * 64

    // Seed buffer
    std::vector<uint8_t> h_seeds(BATCH * SEEDLEN);
    if (seed_kind == SeedKind::A) {
        std::fill(h_seeds.begin(), h_seeds.end(), (uint8_t)'a');
        std::puts("[seed] using 'a' × 240");
    } else {
        std::fill(h_seeds.begin(), h_seeds.end(), (uint8_t)0);
        std::puts("[seed] using 0x00 × 240");
    }

    // --- CPU XOF ---
    std::vector<uint8_t> xof_cpu;
    blake3_xof_cpu(h_seeds.data(), xof_cpu);

    // Optional Elixir sanity only for 'a'×240 (we have a known first64 vector)
    if (seed_kind == SeedKind::A) {
        if (std::memcmp(xof_cpu.data(), ELIXIR_Ax240_FIRST64, 64) != 0) {
            std::puts("❌ CPU XOF[0..63] != Elixir expected");
            std::puts("--- CPU ---"); hexdump64(xof_cpu.data());
            std::puts("--- EXP ---"); hexdump64(ELIXIR_Ax240_FIRST64);
            return 1;
        } else {
            std::puts("✅ CPU XOF[0..63] matches Elixir.");
        }
    } else {
        // For zeros there’s no Elixir vector baked in; just show the CPU first64
        std::puts("[info] CPU first 64 bytes (zeros seed):");
        hexdump64(xof_cpu.data());
    }

    // --- GPU XOF ---
    uint8_t *d_seeds = nullptr, *d_xof = nullptr;
    cudaMalloc(&d_seeds, h_seeds.size());
    cudaMemcpy(d_seeds, h_seeds.data(), h_seeds.size(), cudaMemcpyHostToDevice);

    cudaMalloc(&d_xof, MAT);
    blake3_xof_cuda(d_seeds, SEEDLEN, d_xof, MAT, BATCH, 0);
    cudaDeviceSynchronize();

    std::vector<uint8_t> xof_gpu(MAT);
    cudaMemcpy(xof_gpu.data(), d_xof, MAT, cudaMemcpyDeviceToHost);

    // Print blk0 from both sides
    if (dump_blocks > 0) {
        std::puts("[blk0 sanity]");
        std::puts("--- CPU ---"); hexdump64(&xof_cpu[0]);
        std::puts("--- GPU ---"); hexdump64(&xof_gpu[0]);
    }

    // Compare all blocks
    const size_t BLOCKS = MAT / 64;
    size_t first_mismatch = BLOCKS;
    for (size_t b = 0; b < BLOCKS; ++b) {
        if (std::memcmp(&xof_cpu[b*64], &xof_gpu[b*64], 64) != 0) {
            first_mismatch = b;
            break;
        }
        if (dump_blocks > (int)b) {
            std::printf("[CPU] blk %zu\n", b); hexdump64(&xof_cpu[b*64]);
            std::printf("[GPU] blk %zu\n", b); hexdump64(&xof_gpu[b*64]);
        }
    }

    if (first_mismatch == BLOCKS) {
        std::puts("✅ GPU XOF matches CPU XOF for all 25,120 blocks.");
    } else {
        std::printf("❌ First mismatch at block %zu (offset %zu bytes)\n",
                    first_mismatch, first_mismatch*64);
        std::puts("--- CPU ---"); hexdump64(&xof_cpu[first_mismatch*64]);
        std::puts("--- GPU ---"); hexdump64(&xof_gpu[first_mismatch*64]);
    }

    cudaFree(d_seeds);
    cudaFree(d_xof);
    return 0;
}
