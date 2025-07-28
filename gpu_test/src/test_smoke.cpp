#include <cstdio>
#include <unistd.h>
#include <vector>
#include <random>
#include <cuda_runtime.h>
#include <cstring>

extern "C" void blake3_matmul_cuda(const void*,size_t,void*,size_t,int,cudaStream_t);
void cpu_reference(const uint8_t*, std::vector<int32_t>&);

int main(){
    const int BATCH=8;                // small for first run
    std::vector<uint8_t> h_seeds(BATCH*240);
    std::mt19937_64 rng(123);
    for(auto &b:h_seeds) b=rng()&0xFF;

    uint8_t *d_seeds; int32_t *d_prod;
    cudaMalloc(&d_seeds,h_seeds.size());
    cudaMalloc(&d_prod,BATCH*256*4);
    cudaMemcpy(d_seeds,h_seeds.data(),h_seeds.size(),cudaMemcpyHostToDevice);

    blake3_matmul_cuda(d_seeds,240,d_prod,256*4,BATCH,0);
    cudaDeviceSynchronize();

    sleep(1);

    std::vector<int32_t> h_gpu(BATCH*256);
    cudaMemcpy(h_gpu.data(),d_prod,h_gpu.size()*4,cudaMemcpyDeviceToHost);

    for(int s=0;s<BATCH;++s){
        std::vector<int32_t> ref;
        cpu_reference(&h_seeds[s*240],ref);
        if(memcmp(ref.data(),h_gpu.data()+s*256,256*4)!=0){
            for (int i = 0; i < 64; ++i) {
                printf("%02X ", reinterpret_cast<uint8_t*>(ref.data())[i]);
            }
            puts("\n--- GPU ---");
            for (int i = 0; i < 64; ++i) {
                printf("%02X ", reinterpret_cast<uint8_t*>(h_gpu.data()+s*256)[i]);
            }
            puts("");
            return 1;
        }
    }
    puts("✅ GPU matches CPU on all samples.");
    return 0;
}
