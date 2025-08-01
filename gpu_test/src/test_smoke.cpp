// smoke_one.cpp
#include <cstdio>
#include <vector>
#include <random>
#include <cuda_runtime.h>
#include <cstring>

extern "C" void blake3_matmul_cuda(const void*,size_t,void*,size_t,int,cudaStream_t);
void cpu_reference(const uint8_t*, std::vector<int32_t>&);

static constexpr size_t SEED_LEN   = 240;
static constexpr size_t PROD_BYTES = 256*4;   // 16×16 int32


int main(int argc,char** argv) {
        bool use_zero = false, use_a = false;
        if (argc>1) {
          if (std::strcmp(argv[1],"zero")==0) {
            use_zero = true;
          } else if (std::strcmp(argv[1],"a")==0) {
            use_a = true;
          }
        }

    //------------------------------------ prepare one 240‑byte seed
    std::vector<uint8_t> seed(SEED_LEN);
        if (use_zero) {
            std::fill(seed.begin(), seed.end(), 0);
        } else if (use_a) {
            std::fill(seed.begin(), seed.end(), 0x61);  // all ‘a’
        } else {
            std::mt19937_64 rng(123);
            for (auto &b:seed) b = rng() & 0xFF;
        }
    //------------------------------------ CPU reference
    std::vector<int32_t> ref;
    cpu_reference(seed.data(), ref);

        if (use_a) {
          static constexpr uint8_t EXPECTED[64] = {
            228,239,214, 50, 69, 78,178, 40,  61, 73, 87,120,128, 98, 51, 97,
            100,156, 74,210,197,164, 88,192, 230, 93,238, 52,215,107,109,135,
            100,184,115,103, 12, 29,220,175,  84, 29, 47,237,246,230, 39,208,
             89,212,247, 80, 96, 78,243, 43, 149,250, 74,166, 83, 33, 62, 69
          };

          auto *bytes = reinterpret_cast<const uint8_t*>(ref.data());
          printf("CPU-first-64-bytes: ");
          for (int i = 0; i < 64; i++) printf("%02X ", bytes[i]);
          puts("");

          printf("EXPECTED-64-bytes:  ");
          for (int i = 0; i < 64; i++) printf("%02X ", EXPECTED[i]);
          puts("");
        }

    //------------------------------------ upload to GPU and run
    uint8_t  *d_seed;
    int32_t  *d_prod;
    cudaMalloc(&d_seed, SEED_LEN);
    cudaMalloc(&d_prod, PROD_BYTES);
    cudaMemcpy(d_seed, seed.data(), SEED_LEN, cudaMemcpyHostToDevice);

    blake3_matmul_cuda(d_seed, SEED_LEN, d_prod, PROD_BYTES, /*batch=*/1, 0);
    cudaDeviceSynchronize();

    std::vector<int32_t> gpu(256);
    cudaMemcpy(gpu.data(), d_prod, PROD_BYTES, cudaMemcpyDeviceToHost);

    //------------------------------------ compare
    if (memcmp(ref.data(), gpu.data(), PROD_BYTES)!=0) {
        puts("❌ mismatch!");
        printf("CPU first 32 bytes : ");
        for (int i=0;i<32;++i) printf("%02X ",
                 reinterpret_cast<uint8_t*>(ref.data())[i]); puts("");
        printf("GPU first 32 bytes : ");
        for (int i=0;i<32;++i) printf("%02X ",
                 reinterpret_cast<uint8_t*>(gpu.data())[i]); puts("");
        return 1;
    }
    puts("✅ GPU matches CPU for this seed.");
    return 0;
}

//<<228, 239, 214, 50, 69, 78, 178, 40, 61, 73, 87, 120, 128, 98, 51, 97, 100,
//156, 74, 210, 197, 164, 88, 192, 230, 93, 238, 52, 215, 107, 109, 135, 100,
//184, 115, 103, 12, 29, 220, 175, 84, 29, 47, 237, 246, 230, 39, 208, 89, 212,
//247, 80, 96, 78, 243, 43, 149, 250, 74, 166, 83, 33, 62, 69>>
