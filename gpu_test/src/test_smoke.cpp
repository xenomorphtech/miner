#include <cstdio>
#include <cstdint>
#include <unistd.h>
#include <vector>
#include <random>
#include <cuda_runtime.h>
#include <cstring>
#include "../blake3/c/blake3.h"

static void print_hex(const uint8_t* p, size_t n, const char* label) {
    if (label) puts(label);
    for (size_t i = 0; i < n; ++i) {
        printf("%02X%s", p[i], ((i + 1) % 16 == 0) ? "\n" : " ");
    }
    if (n % 16) puts("");
}

static bool check_blake3_kat_a240() {
    // Seed = 240 bytes of ASCII 'a' (0x61)
    uint8_t seed[240]; memset(seed, 'a', sizeof(seed));

    // Compute first 64 bytes of XOF
    uint8_t out[64];
    blake3_hasher h; blake3_hasher_init(&h);
    blake3_hasher_update(&h, seed, sizeof(seed));
    blake3_hasher_finalize_seek(&h, 0, out, sizeof(out));

    // Expected from your Elixir snippet (same bytes, decimal -> hex is fine)
    static const uint8_t expected[64] = {
        228,239,214, 50, 69, 78,178, 40, 61, 73, 87,120,128, 98, 51, 97,
        100,156, 74,210,197,164, 88,192,230, 93,238, 52,215,107,109,135,
        100,184,115,103, 12, 29,220,175, 84, 29, 47,237,246,230, 39,208,
         89,212,247, 80, 96, 78,243, 43,149,250, 74,166, 83, 33, 62, 69
    };

    if (memcmp(out, expected, 64) != 0) {
        puts("❌ BLAKE3 KAT (a×240, first 64 bytes) mismatch.");
        print_hex(expected, 64, "Expected:");
        print_hex(out, 64, "Got     :");
        return false;
    }
    puts("✅ BLAKE3 KAT passes (a×240 → first 64 bytes).");
    return true;
}

extern "C" void blake3_matmul_cuda(const void*, size_t, void*, size_t, int, cudaStream_t);
void cpu_reference(const uint8_t*, std::vector<int32_t>&);


static bool check_cpu_known_vector() {
    // 1) Prepare seed: 240 x 'a'
    std::vector<uint8_t> seed(240, 'a');

    // 2) Run CPU reference
    std::vector<int32_t> ref;
    cpu_reference(seed.data(), ref);

    const size_t ref_bytes = ref.size() * sizeof(ref[0]);
    if (ref_bytes < 64) {
        fprintf(stderr, "cpu_reference output too short: %zu bytes (need >= 64)\n", ref_bytes);
        return false;
    }

    // 3) Expected first 64 bytes from Elixir (finalize_xof 64)
    static const uint8_t expected[64] = {
        228,239,214, 50, 69, 78,178, 40, 61, 73, 87,120,128, 98, 51, 97,
        100,156, 74,210,197,164, 88,192,230, 93,238, 52,215,107,109,135,
        100,184,115,103, 12, 29,220,175, 84, 29, 47,237,246,230, 39,208,
         89,212,247, 80, 96, 78,243, 43,149,250, 74,166, 83, 33, 62, 69
    };

    const uint8_t* ref_bytes_ptr = reinterpret_cast<const uint8_t*>(ref.data());

    // 4) Compare and report
    if (std::memcmp(ref_bytes_ptr, expected, 64) != 0) {
        puts("❌ CPU known-vector mismatch (first 64 bytes).");
        print_hex(expected, 64, "Expected (64 bytes):");
        print_hex(ref_bytes_ptr, 64, "CPU got (64 bytes):");
        return false;
    }

    puts("✅ CPU known-vector matches (first 64 bytes).");
    return true;
}

int main() {
    if (!check_blake3_kat_a240()) return 1;

    // Step 2: Proceed with your existing GPU vs CPU smoke test
    const int BATCH = 1; // small for first run
    std::vector<uint8_t> h_seeds(BATCH * 240);

    std::mt19937_64 rng(123);
    for (auto &b : h_seeds) b = rng() & 0xFF;

    uint8_t *d_seeds = nullptr;
    int32_t *d_prod = nullptr;

    cudaError_t err;
    err = cudaMalloc(&d_seeds, h_seeds.size());
    if (err != cudaSuccess) { fprintf(stderr, "cudaMalloc d_seeds failed: %s\n", cudaGetErrorString(err)); return 1; }
    err = cudaMalloc(&d_prod, BATCH * 256 * 4);
    if (err != cudaSuccess) { fprintf(stderr, "cudaMalloc d_prod failed: %s\n", cudaGetErrorString(err)); cudaFree(d_seeds); return 1; }

    err = cudaMemcpy(d_seeds, h_seeds.data(), h_seeds.size(), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { fprintf(stderr, "cudaMemcpy H2D failed: %s\n", cudaGetErrorString(err)); cudaFree(d_seeds); cudaFree(d_prod); return 1; }

    blake3_matmul_cuda(d_seeds, 240, d_prod, 256 * 4, BATCH, 0);
    cudaDeviceSynchronize();

    sleep(1); // optional

    std::vector<int32_t> h_gpu(BATCH * 256);
    err = cudaMemcpy(h_gpu.data(), d_prod, h_gpu.size() * 4, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) { fprintf(stderr, "cudaMemcpy D2H failed: %s\n", cudaGetErrorString(err)); cudaFree(d_seeds); cudaFree(d_prod); return 1; }

    for (int s = 0; s < BATCH; ++s) {
        std::vector<int32_t> ref;
        cpu_reference(&h_seeds[s * 240], ref);

        if (std::memcmp(ref.data(), h_gpu.data() + s * 256, 256 * 4) != 0) {
            puts("❌ GPU vs CPU mismatch. Showing first 64 bytes of each:");
            // Print CPU (ref) first 64 bytes
            print_hex(reinterpret_cast<uint8_t*>(ref.data()), 64, "CPU (first 64):");
            // Print GPU first 64 bytes
            print_hex(reinterpret_cast<uint8_t*>(h_gpu.data() + s * 256), 64, "GPU (first 64):");

            cudaFree(d_seeds);
            cudaFree(d_prod);
            return 1;
        }
    }

    puts("✅ GPU matches CPU on all samples.");
    cudaFree(d_seeds);
    cudaFree(d_prod);
    return 0;
}