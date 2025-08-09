#include <stdint.h>
#include <cuda_runtime.h>
#include <assert.h>
#include <stdio.h>

#ifdef __CUDA_ARCH__
#define cudaDeviceSynchronize()  /* no-op in device code */
#endif

#define BLAZE3_DISABLE_RECURSIVE 1
#include "../include/blaze3.cuh"

/* IV in constant memory (device) */
__device__ __constant__ uint32_t g_IV[8] = {
    0x6A09E667u, 0xBB67AE85u, 0x3C6EF372u, 0xA54FF53Au,
    0x510E527Fu, 0x9B05688Cu, 0x1F83D9ABu, 0x5BE0CD19u
};

using u32 = uint32_t;

/* -----------------------------------------------------------------------
   XOF layout control (for debugging blk0 wiring and counter start)
   ----------------------------------------------------------------------- */
struct XofLayout {
    int counter_offset;  // add to block counter
    int b0_mode;         // 0=lower, 1=upper, 2=upper^root, 3=lower^root
};
__device__ __constant__ XofLayout g_xof_layout = {0, 0};

extern "C"
void blake3_xof_layout(int counter_offset, int b0_mode) {
    XofLayout h{counter_offset, b0_mode};
    cudaMemcpyToSymbol(g_xof_layout, &h, sizeof(h), 0, cudaMemcpyHostToDevice);
}

/* Kernels (decls) */
__global__ void root_hash8(const uint8_t*, u32*, u32*, uint8_t*, int);
__global__ void xof_expand(const u32*, uint8_t*, size_t, int);

/* -------------------- kernel 1 : 8 seeds / warp ----------------------- */
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

    __shared__ uint8_t sm[8][240];
    for (int i = lane; i < 240; i += 32) {
#pragma unroll
        for (int s = 0; s < seeds_in_pack; ++s)
            sm[s][i] = d_seeds[(idx0 + s) * 240 + i];
    }
    __syncthreads();

    if (lane < seeds_in_pack) {
        const uint8_t *msg = sm[lane];
        u32 cv[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) cv[i] = g_IV[i];

#ifdef DEBUG_TRACE
        if (idx0 == 0 && lane == 0) {
            printf("DBG cv after IV copy = %08X %08X %08X %08X idx = %d \n",
                   cv[0], cv[1], cv[2], cv[3], idx0);
        }
#endif

        u32 m[16];
        u32 tmp_state[16];

        /* four 64-byte blocks (last one is 48 B + 16×0 padding) */
#pragma unroll
        for (int blk = 0; blk < 4; ++blk) {
            const u32 block_len = (blk == 3 ? 48u : 64u);
            const uint8_t *src  = msg + blk * 64;

#pragma unroll
            for (int w = 0; w < 16; ++w) m[w] = 0;
#pragma unroll
            for (int i = 0; i < (int)block_len; ++i)
                reinterpret_cast<uint8_t*>(m)[i] = src[i];

            u32 flags = (blk == 0 ? CHUNK_START : 0) |
                        (blk == 3 ? (CHUNK_END | ROOT) : 0);
            const uint64_t counter = 0ULL;  // within chunk, always 0

#ifdef DEBUG_TRACE
            if (idx0 == 0 && lane == 0 && blk == 0) {
                printf("DBG pre-cmp cv[0..3]=%08X %08X %08X %08X\n",
                       cv[0], cv[1], cv[2], cv[3]);
                printf("DBG pre-cmp m[0..3] =%08X %08X %08X %08X  len=%u flags=%02X\n",
                       m[0], m[1], m[2], m[3], block_len, flags);
            }
            if (idx0 == 0 && lane == 0) {
                u32 first_word; memcpy(&first_word, msg, 4);
                printf("DBG seed0 word0 = %08X\n", first_word);
                printf("DBG cv in  [%d] = %08X %08X %08X %08X\n",
                       blk, cv[0], cv[1], cv[2], cv[3]);
            }
#endif
            g_compress(cv, m, counter, block_len, flags, tmp_state);

            if (blk == 3) {
                u32 *dst = d_last_words + (idx0 + lane) * 16;
#pragma unroll
                for (int w = 0; w < 16; ++w) dst[w] = m[w];
                d_last_len[idx0 + lane] = (uint8_t)block_len;   /* 48 */
            }

#pragma unroll
            for (int w = 0; w < 8; ++w) cv[w] = tmp_state[w]; // already XORed in g_compress
        }

#ifdef DEBUG_TRACE
        if (idx0 == 0 && lane == 0) {
            printf("GPU-kern cv[0..3] = %08X %08X %08X %08X\n",
                   cv[0], cv[1], cv[2], cv[3]);
        }
#endif
#pragma unroll
        for (int w = 0; w < 8; ++w)
            d_roots[(idx0 + lane) * 8 + w] = cv[w];
    }
}

/* -------------------- kernel 2 : XOF expansion ------------------------ */
/* Helper: produce full 16 words (tmp) without final XOR mixing. */
__device__ inline void xof_block_raw(const u32  root[8],
                                     uint64_t   ctr,
                                     u32        tmp[16])
{
    u32 m[16] = {0};
    m[15] = DERIVE_KEY_MATERIAL;        // upstream mode for XOF/derive
    u32 cv[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) cv[i] = root[i];
    const u32 xof_flags = DERIVE_KEY_MATERIAL;  /* 0x40 */
    g_compress(cv, m, ctr, 64, xof_flags, tmp); // tmp[0..15] = g_compress state
}

/* blk0 layout:
   - First 32B always from tmpL (tmp[0..7]) for consistency with CPU logs.
   - Second 32B depends on g_xof_layout.b0_mode:
       0: tmpL
       1: tmpH
       2: tmpH ^ root
       3: tmpL ^ root
   Other blocks:
   - Standard BLAKE3 stream: tmpL || (tmpH ^ root)
*/
__global__ void xof_expand(const u32 *d_roots,
                           uint8_t  *d_mat,
                           size_t    per_mat,
                           int       batch)
{
    const uint64_t tid    = blockIdx.x * blockDim.x + threadIdx.x;
    const u32 blocks      = 25120;
    const int mat_id      = tid / blocks;
    if (mat_id >= batch) return;

    const u32 blk = tid % blocks;

    const u32 *root = d_roots + mat_id * 8;

    u32 tmp[16];
    // Allow shifting the starting counter (debug)
    const uint64_t ctr = (uint64_t)blk + (uint64_t)g_xof_layout.counter_offset;
    xof_block_raw(root, ctr, tmp);

    u32 out[16];
    // First 32 bytes: tmp lower half (common across all modes for visibility)
#pragma unroll
    for (int w = 0; w < 8; ++w) out[w] = tmp[w];

    if (blk == 0) {
        switch (g_xof_layout.b0_mode) {
            case 0: // lower
#pragma unroll
                for (int w = 0; w < 8; ++w) out[8 + w] = tmp[w];
                break;
            case 1: // upper
#pragma unroll
                for (int w = 0; w < 8; ++w) out[8 + w] = tmp[8 + w];
                break;
            case 2: // upper ^ root
#pragma unroll
                for (int w = 0; w < 8; ++w) out[8 + w] = tmp[8 + w] ^ root[w];
                break;
            case 3: // lower ^ root
#pragma unroll
                for (int w = 0; w < 8; ++w) out[8 + w] = tmp[w] ^ root[w];
                break;
            default:
#pragma unroll
                for (int w = 0; w < 8; ++w) out[8 + w] = tmp[8 + w] ^ root[w];
                break;
        }
#ifdef DEBUG_TRACE
        if (mat_id == 0 && threadIdx.x == 0) {
            printf("[GPU dbg] blk0 cfg: counter_offset=%d b0_mode=%d\n",
                   g_xof_layout.counter_offset, g_xof_layout.b0_mode);
            printf("[GPU dbg] tmpL[0..3]=%08X %08X %08X %08X\n",
                   tmp[0], tmp[1], tmp[2], tmp[3]);
            printf("[GPU dbg] tmpH[0..3]=%08X %08X %08X %08X\n",
                   tmp[8], tmp[9], tmp[10], tmp[11]);
            printf("[GPU dbg] root[0..3]=%08X %08X %08X %08X\n",
                   root[0], root[1], root[2], root[3]);
            printf("[GPU] blk 0\n");
            for (int i = 0; i < 64; ++i) {
                const uint8_t *p = reinterpret_cast<const uint8_t*>(out);
                printf("%02X%s", p[i], ((i & 15)==15 ? "\n" : " "));
            }
        }
#endif
    } else {
        // Standard stream for blk>0: lower || (upper ^ root)
#pragma unroll
        for (int w = 0; w < 8; ++w) out[8 + w] = tmp[8 + w] ^ root[w];
    }

    uint8_t *dst = d_mat + (size_t)mat_id * per_mat + blk * 64;
#pragma unroll
    for (int w = 0; w < 16; ++w)
        reinterpret_cast<u32 *>(dst)[w] = out[w];
}

/* -------------------- public facades ---------------------------------- */

extern "C"
void blake3_xof_cuda(const void *d_seeds, size_t seed_len,
                     void *d_xof,   size_t xof_len,
                     int batch, cudaStream_t s)
{
    constexpr size_t SEED = 240;
    constexpr size_t MAT_BYTES = 1'607'680;     // 25,120 * 64
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

    uint64_t threads = (uint64_t)batch * 25120;
    dim3 blk(256), grd((threads + 255) / 256);

    xof_expand<<<grd, blk, 0, s>>>(d_roots,
                                   static_cast<uint8_t*>(d_xof),
                                   MAT_BYTES,
                                   batch);
}

extern "C"
void blake3_matmul_cuda(const void *d_seeds,size_t seed_len,
                        void *d_out,size_t prod_len,int batch,
                        cudaStream_t s)
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
    root_hash8<<<packs,32,0,s>>>(static_cast<const uint8_t*>(d_seeds),
                                 d_roots,
                                 d_last_words,
                                 d_last_len,
                                 batch);

    uint64_t threads=(uint64_t)batch*25120;
    dim3 blk(256), grd((threads+255)/256);
    static uint8_t *d_mat=nullptr; static int matcap=0;
    if(batch>matcap){ if(d_mat)cudaFree(d_mat);
                      cudaMalloc(&d_mat,batch*MAT); matcap=batch;}
    xof_expand<<<grd,blk,0,s>>>(d_roots, d_mat, MAT, batch);

    /* Naive GEMM kernel (16x16) */
    const int ROWS=16,COLS=16,K=50240;
    int32_t *prod = static_cast<int32_t*>(d_out);

    __global__ void gemm(int32_t*,const uint8_t*,const int8_t*);
    gemm<<<batch,dim3(ROWS,COLS,1),0,s>>>(
        prod,
        d_mat,                                         // A  uint8_t*
        reinterpret_cast<const int8_t*>(d_mat+ROWS*K));// B  int8_t*
}

/* naive GEMM kernel */
__global__ void gemm(int32_t *C,const uint8_t *A,const int8_t *B)
{
    const int bid = blockIdx.x;
    const int i = threadIdx.x, j = threadIdx.y;
    if(i>=16||j>=16) return;

    const uint8_t* a = A + (size_t)bid*16*50240;
    const int8_t * b = B + (size_t)bid*50240*16;

    int32_t sum=0;
    for(int k=0;k<50240;++k)
        sum+=int32_t(a[i*50240+k])*int32_t(b[k*16+j]);

    C[bid*256 + i*16 + j]=sum;
}
