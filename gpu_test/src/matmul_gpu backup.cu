
#include <stdint.h>
#include <cuda_runtime.h>


#ifdef __CUDA_ARCH__
#define cudaDeviceSynchronize()  /* no‑op in device code */
#endif


#define BLAZE3_DISABLE_RECURSIVE 1
#include "../include/blaze3.cuh"
#include <stdio.h>            /* printf inside device code */

#include <stdint.h>
#include <cuda_runtime.h>


__device__ __constant__ uint32_t g_IV[8] = {
    0x6A09E667u, 0xBB67AE85u, 0x3C6EF372u, 0xA54FF53Au,
    0x510E527Fu, 0x9B05688Cu, 0x1F83D9ABu, 0x5BE0CD19u
};

using u32 = uint32_t;

__global__ void root_hash8(const uint8_t* __restrict__ d_seeds,
                           u32*         __restrict__ d_roots,
                           u32*         __restrict__ d_last_words,
                           uint8_t*     __restrict__ d_last_len,
                           int seeds);

__global__ void xof_expand(const u32* d_roots,
                           uint8_t*   d_mat,
                           size_t     per_mat,
                           int        batch);


extern "C"
void blake3_xof_cuda(const void *d_seeds, size_t seed_len,
                     void *d_xof,   size_t xof_len,
                     int batch, cudaStream_t s=0)
{
    constexpr size_t SEED = 240;
    constexpr size_t MAT_BYTES = 1'607'680;     // must match kernel’s 25,120 blocks × 64
    if (seed_len != SEED || xof_len != MAT_BYTES) return;

    static u32    *d_roots      = nullptr;
    static u32    *d_last_words = nullptr;
    static uint8_t*d_last_len   = nullptr;
    static int cap = 0;
    if (batch > cap) {
        if (d_roots)      cudaFree(d_roots);
        if (d_last_words) cudaFree(d_last_words);
        if (d_last_len)   cudaFree(d_last_len);
        cudaMalloc(&d_roots,      batch*32);
        cudaMalloc(&d_last_words, batch*16*sizeof(u32));
        cudaMalloc(&d_last_len,   batch);
        cap = batch;
    }

    int packs = (batch + 7) / 8;
    root_hash8<<<packs, 32, 0, s>>>(static_cast<const uint8_t*>(d_seeds),
                                    d_roots, d_last_words, d_last_len, batch);

    // 25,120 blocks per matrix, as already hardcoded inside xof_expand
    uint64_t threads = (uint64_t)batch * 25120;
    dim3 blk(256), grd((threads + 255) / 256);

    // write XOF into d_xof provided by caller
    xof_expand<<<grd, blk, 0, s>>>(d_roots,
                                   static_cast<uint8_t*>(d_xof),
                                   MAT_BYTES,
                                   batch);
}

/*-------------------- kernel 1 : 8 seeds / warp --------------------------*/
__global__ void root_hash8(const uint8_t * __restrict__ d_seeds,
                           u32          * __restrict__ d_roots,
                           /* new */      u32          * __restrict__ d_last_words,
                           /* new */      uint8_t      * __restrict__ d_last_len,
                           int seeds)
{
    #ifdef DEBUG_TRACE
        if (threadIdx.x == 0 && blockIdx.x == 0) {
            printf("DEVICE args  d_seeds=%p  d_roots=%p  seeds=%d\n",
                   d_seeds, d_roots, seeds);
        }
    #endif

    const int pack = blockIdx.x;                 // one warp hashes 8 seeds
    const int lane = threadIdx.x;                // 0 … 31

    const int idx0  = pack * 8;
    if (idx0 >= seeds) return;

    const int seeds_in_pack = min(8, seeds - idx0);

    //printf("---- %d, %d, %d, %d \n", pack, lane, idx0, seeds);


    /* ---- 1. load 8 seeds (240 B each) into shared memory -------------- */
    __shared__ uint8_t sm[8][240];
    for (int i = lane; i < 240; i += 32) {
        #pragma unroll
        for (int s = 0; s < seeds_in_pack; ++s)
            sm[s][i] = d_seeds[(idx0 + s) * 240 + i];
    }



    __syncthreads();



    /* ---- 2. hash each seed, one lane per seed ------------------------- */

    if (lane < seeds_in_pack) {

        const uint8_t *msg = sm[lane];
        uint32_t cv[8];
        #pragma unroll
        for (int i = 0; i < 8; ++i) cv[i] = g_IV[i];



        printf("DBG cv after IV copy = %08X %08X %08X %08X idx = %d \n",
                           cv[0], cv[1], cv[2], cv[3], idx0);


        uint32_t m[16];
        uint32_t tmp_state[16];

        /* four 64‑byte blocks (last one is 48 B + 16 × 0 padding) */
        #pragma unroll


        for (int blk = 0; blk < 4; ++blk) {

            const uint32_t block_len = (blk == 3 ? 48 : 64);
            const uint8_t *src       = msg + blk * 64;

            /* --- build message words with zero-padding -------------------- */
            #pragma unroll
            for (int w = 0; w < 16; ++w) m[w] = 0;
            #pragma unroll
            for (int i = 0; i < block_len; ++i)
                reinterpret_cast<uint8_t*>(m)[i] = src[i];


            uint32_t flags = (blk == 0 ? CHUNK_START : 0) |
                           (blk == 3 ? (CHUNK_END | ROOT) : 0);

                /* Inside one chunk the counter is always 0 */
            const uint64_t counter = 0ULL;

                        /* ---- DEBUG: print the inputs fed to g_compress -------- */
            #ifdef DEBUG_TRACE
                        if (idx0 == 0 && lane == 0 && blk == 0) {
                            printf("DBG pre‑cmp cv[0..3]=%08X %08X %08X %08X\n",
                                   cv[0], cv[1], cv[2], cv[3]);
                            printf("DBG pre‑cmp m[0..3] =%08X %08X %08X %08X  "
                                   "len=%u flags=%02X\n",
                                   m[0], m[1], m[2], m[3], block_len, flags);
                        }
            #endif

            #ifdef DEBUG_TRACE
                        if (idx0 == 0 && lane == 0) {          /* first seed, lane 0 */
                            /* first 4 bytes of the seed we are hashing */
                            uint32_t first_word;
                            memcpy(&first_word, msg, 4);
                            printf("DBG seed0 word0 = %08X\n", first_word);

                            /* chaining value BEFORE g_compress ------------------ */
                            printf("DBG cv in  [%d] = %08X %08X %08X %08X\n",
                                   blk, cv[0], cv[1], cv[2], cv[3]);
                        }
            #endif

            g_compress(cv, m, counter, block_len, flags, tmp_state);

                        if (blk == 3 && lane < seeds_in_pack) {
                            u32 *dst = d_last_words + (idx0 + lane) * 16;
                            #pragma unroll
                            for (int w = 0; w < 16; ++w) dst[w] = m[w];
                            d_last_len[idx0 + lane] = (uint8_t)block_len;   /* 48 */
                        }


            /* XOR low and high halves to build the next chaining value */
            #pragma unroll
            for (int w = 0; w < 8; ++w)
                cv[w] = tmp_state[w];    // already XORed inside g_compress

        }

        /* ---- 3. write the 32‑byte root out --------------------------- */


        #ifdef DEBUG_TRACE
            if (idx0 == 0 && lane == 0) {          // first seed only
                printf("GPU‑kern cv[0..3] = %08X %08X %08X %08X\n",
                       cv[0], cv[1], cv[2], cv[3]);
            }
        #endif

        #pragma unroll
        for (int w = 0; w < 8; ++w) {
            d_roots[(idx0 + lane) * 8 + w] = cv[w];
        }
    }
}

/*-------------------- kernel 2 : XOF expansion ---------------------------*/
__device__ inline void expand_one(const u32  root[8],
                                  uint64_t   ctr,
                                  u32        out[16])
{
     u32 m[16] = {0};
     m[15] = DERIVE_KEY_MATERIAL;


    u32 cv[8];
    #pragma unroll
    for (int i = 0; i < 8; ++i) cv[i] = root[i];

    const u32 xof_flags = DERIVE_KEY_MATERIAL;  /* 0x40, upstream */
    g_compress(cv, m, ctr, 64, xof_flags, out);

    /* return the full 64-byte output block */
    #pragma unroll
    for (int w = 0; w < 8; ++w) out[w + 8] ^= root[w];
}

__global__ void xof_expand(const u32 *d_roots,
                            uint8_t  *d_mat,
                            size_t    per_mat,
                            int       batch)
{
    const uint64_t tid   = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t blocks = 25120;
    const int      mat_id = tid / blocks;
    if (mat_id >= batch) return;

    const uint32_t blk = tid % blocks;

    u32 block[16];
           expand_one(d_roots + mat_id * 8,
               blk,
               block);

    #ifdef DEBUG_TRACE
        if (mat_id == 0 && blk == 0 && threadIdx.x == 0) {
            printf("GPU XOF blk0[0..3] = %08X %08X %08X %08X\n",
                   block[0], block[1], block[2], block[3]);
        }
    #endif

    uint8_t *dst = d_mat + (size_t)mat_id * per_mat + blk * 64;
    #pragma unroll
    for (int w = 0; w < 16; ++w)
        reinterpret_cast<u32 *>(dst)[w] = block[w];
}

/*-------------------- public facade --------------------------------------*/
extern "C"
void blake3_matmul_cuda(const void *d_seeds,size_t seed_len,
                        void *d_out,size_t prod_len,int batch,
                        cudaStream_t s=0)
{
    constexpr size_t SEED=240, MAT=1'607'680, PROD=16*16*4;
    if(seed_len!=SEED||prod_len!=PROD) return;

        static u32    *d_roots      = nullptr;
        static u32    *d_last_words = nullptr;
        static uint8_t*d_last_len   = nullptr;
        static int cap = 0;
        if(batch>cap){
            if(d_roots)      cudaFree(d_roots);
            if(d_last_words) cudaFree(d_last_words);
            if(d_last_len)   cudaFree(d_last_len);
            cudaMalloc(&d_roots,      batch*32);
            cudaMalloc(&d_last_words, batch*16*sizeof(u32));
            cudaMalloc(&d_last_len,   batch);
            cap=batch;
        }

    int packs=(batch+7)/8;

    assert(batch > 0);
    printf("HOST launching root_hash8 batch=%d packs=%d\n", batch, packs);


        root_hash8<<<packs,32,0,s>>>(static_cast<const uint8_t*>(d_seeds),
                                     d_roots,
                                     d_last_words,
                                     d_last_len,
                                     batch);

    #ifdef DEBUG_TRACE
    {
        uint32_t h_roots[8];
        cudaMemcpy(h_roots, d_roots, sizeof(h_roots), cudaMemcpyDeviceToHost);
        printf("GPU‑host d_roots[0..7] = %08X %08X %08X %08X %08X %08X %08X %08X\n",
               h_roots[0], h_roots[1], h_roots[2], h_roots[3], h_roots[4], h_roots[5], h_roots[6], h_roots[7]);

        /* --- read the IV directly from constant memory ---------------- */
        uint32_t iv_host[8];
        cudaMemcpyFromSymbol(iv_host, g_IV, sizeof(iv_host), 0,
                                 cudaMemcpyDeviceToHost);
        printf("GPU‑host g_IV[0..3]    = %08X %08X %08X %08X\n",
                   iv_host[0], iv_host[1], iv_host[2], iv_host[3]);
    }
    #endif

    uint64_t threads=(uint64_t)batch*25120;
    dim3 blk(256), grd((threads+255)/256);
    static uint8_t *d_mat=nullptr; static int matcap=0;
    if(batch>matcap){ if(d_mat)cudaFree(d_mat);
                      cudaMalloc(&d_mat,batch*MAT); matcap=batch;}
            xof_expand<<<grd,blk,0,s>>>(d_roots,
                                d_mat,
                                MAT,
                                batch);

    /* GEMM on device – naïve 16×16×k kernel (k=50240) */
    const int ROWS=16,COLS=16,K=50240;
    int32_t *prod = static_cast<int32_t*>(d_out);

    // 1 thread per element
    __global__ void gemm(int32_t*,const uint8_t*,const int8_t*);
    gemm<<<batch,dim3(ROWS,COLS,1),0,s>>>(
            prod,
            d_mat,                                         // A  uint8_t*
            reinterpret_cast<const int8_t*>(d_mat+ROWS*K));// B  int8_t*
}

/* naive GEMM kernel (will be inlined above) */
__global__ void gemm(int32_t *C,const uint8_t *A,const int8_t *B)
{
    const int batch = gridDim.x;
    const int bid   = blockIdx.x;
    const int i = threadIdx.x, j = threadIdx.y;
    if(i>=16||j>=16) return;

    const uint8_t* a = A + (size_t)bid*16*50240;
    const int8_t * b = B + (size_t)bid*50240*16;

    int32_t sum=0;
    for(int k=0;k<50240;++k)
        sum+=int32_t(a[i*50240+k])*int32_t(b[k*16+j]);

    C[bid*256 + i*16 + j]=sum;
}
