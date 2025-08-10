#include <stdint.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <stdio.h>

/* Neutralize cudaDeviceSynchronize() when seen from device code inside
   the blaze3 kernel header (which calls it from a __global__ function). */
#ifdef __CUDA_ARCH__
  #ifndef cudaDeviceSynchronize
    #define cudaDeviceSynchronize() ((void)0)
  #endif
#endif

/* Use the existing blaze3 device kernel utilities (g_compress, flags, etc.) */
#define BLAZE3_DISABLE_RECURSIVE 1
#include "../include/blaze3.cuh"   // does NOT get edited

/* Device IV – provide the storage that blaze3.cuh declares as extern */
__device__ __constant__ uint32_t g_IV[8] = {
    0x6A09E667u, 0xBB67AE85u, 0x3C6EF372u, 0xA54FF53Au,
    0x510E527Fu, 0x9B05688Cu, 0x1F83D9ABu, 0x5BE0CD19u
};

using u32 = uint32_t;

/* ---------------- debug layout control (optional) ---------------- */
struct XofLayout {
  int counter_offset;  // add to t
  int b0_mode;         // 0=spec (hi^precv), 1=tmpL, 2=tmpH, 3=rootcmpL, 4=rootcmpH, 5=rootcmpH^precv
  int xof_flag_mode;   // 0=CHUNK_END|ROOT (spec), 1=DERIVE_KEY_MATERIAL (debug)
};
__device__ __constant__ XofLayout g_xof_layout = {0, 0, 0};

extern "C" void blake3_xof_layout(int counter_offset, int b0_mode, int xof_flag_mode) {
  XofLayout h{counter_offset, b0_mode, xof_flag_mode};
  cudaMemcpyToSymbol(g_xof_layout, &h, sizeof(h));
}

/* ---------------- kernels we own ---------------- */
__global__ void root_hash8(const uint8_t* __restrict__ d_seeds,
                           u32*         __restrict__ d_roots,
                           u32*         __restrict__ d_precv,      // pre-final CV
                           u32*         __restrict__ d_last_words,  // last 64B words
                           uint8_t*     __restrict__ d_last_len,    // last block length
                           int seeds);

__device__ inline void recompress_final_leaf(const u32  precv[8],
                                             const u32  last_words[16],
                                             u32        last_len,
                                             uint64_t   t,
                                             /*out*/ u32 state[16])
{
    u32 cv[8];
    #pragma unroll
    for (int i = 0; i < 8; ++i) cv[i] = precv[i];

    const u32 flags = (CHUNK_END | ROOT);
    // g_compress writes 16 words to state; its low half is already lo^hi (root CV),
    // high half is the raw hi (no feed-forward).
    g_compress(cv, const_cast<u32*>(last_words), t, last_len, flags, state);
}

__global__ void xof_expand(const u32* __restrict__ d_roots,
                           const u32* __restrict__ d_precv,
                           const u32* __restrict__ d_last_words,
                           const uint8_t* __restrict__ d_last_len,
                           uint8_t* __restrict__ d_mat,
                           size_t per_mat,
                           int batch)
{
    const uint64_t tid    = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t blocks = 25120u;
    const int      mid    = tid / blocks;
    if (mid >= batch) return;

    const uint32_t blk = tid % blocks;

    const u32* root   = d_roots      + mid*8;
    const u32* precv  = d_precv      + mid*8;
    const u32* lwords = d_last_words + mid*16;
    const u32  llen   = d_last_len[mid];

    u32 out[16];
    u32* dstw = reinterpret_cast<u32*>(d_mat + (size_t)mid*per_mat + (size_t)blk*64);

    if (g_xof_layout.xof_flag_mode == 1) {
        // --- DEBUG: legacy DERIVE_KEY_MATERIAL path ---
        u32 cv[8]; for (int i=0;i<8;++i) cv[i]=root[i];
        u32 m[16] = {0}; m[15] = DERIVE_KEY_MATERIAL;
        g_compress(cv, m,
                   (uint64_t)blk + (uint64_t)g_xof_layout.counter_offset,
                   64, DERIVE_KEY_MATERIAL, out);
        #pragma unroll
        for (int w=0; w<16; ++w) dstw[w] = out[w];
        return;
    }

    // --- SPEC: re-compress the final leaf with CHUNK_END|ROOT, t = blk+offset ---
    recompress_final_leaf(precv, lwords, llen,
                          (uint64_t)blk + (uint64_t)g_xof_layout.counter_offset,
                          out);

    // Low 32B = out[0..7] (root CV for this t)
    #pragma unroll
    for (int w=0; w<8; ++w) dstw[w] = out[w];

    // Upper 32B selection (debug modes for blk0; otherwise spec)
    if (blk == 0) {
        if      (g_xof_layout.b0_mode == 1) { // tmpL
            #pragma unroll
            for (int w=0; w<8; ++w) dstw[w+8] = out[w];
            return;
        } else if (g_xof_layout.b0_mode == 2) { // tmpH
            #pragma unroll
            for (int w=0; w<8; ++w) dstw[w+8] = out[w+8];
            return;
        } else if (g_xof_layout.b0_mode == 3) { // rootcmpL
            #pragma unroll
            for (int w=0; w<8; ++w) dstw[w+8] = out[w];
            return;
        } else if (g_xof_layout.b0_mode == 4) { // rootcmpH
            #pragma unroll
            for (int w=0; w<8; ++w) dstw[w+8] = out[w+8];
            return;
        } else if (g_xof_layout.b0_mode == 5) { // rootcmpH ^ precv  (spec upper)
            #pragma unroll
            for (int w=0; w<8; ++w) dstw[w+8] = out[w+8] ^ precv[w];
            return;
        }
    }

    // Default/spec upper: hi ^ precv
    #pragma unroll
    for (int w=0; w<8; ++w) dstw[w+8] = out[w+8] ^ precv[w];
}

/* ---------------- root_hash8: compute root + save preCV/last block -------- */
__global__ void root_hash8(const uint8_t * __restrict__ d_seeds,
                           u32          * __restrict__ d_roots,
                           u32          * __restrict__ d_precv,
                           u32          * __restrict__ d_last_words,
                           uint8_t      * __restrict__ d_last_len,
                           int seeds)
{
    const int pack = blockIdx.x;   // 1 warp handles up to 8 seeds
    const int lane = threadIdx.x;  // 0..31
    const int idx0 = pack * 8;
    if (idx0 >= seeds) return;
    const int in_pack = min(8, seeds - idx0);

    __shared__ uint8_t sm[8][240];
    for (int i = lane; i < 240; i += 32) {
        #pragma unroll
        for (int s = 0; s < in_pack; ++s)
            sm[s][i] = d_seeds[(idx0 + s)*240 + i];
    }
    __syncthreads();

    if (lane >= in_pack) return;

    const uint8_t *msg = sm[lane];
    u32 cv[8];
    #pragma unroll
    for (int i = 0; i < 8; ++i) cv[i] = g_IV[i];

    u32 m[16];
    u32 tmp_state[16];

    for (int blk = 0; blk < 4; ++blk) {
        const u32 block_len = (blk == 3 ? 48u : 64u);
        const uint8_t *src  = msg + blk * 64;

        #pragma unroll
        for (int w = 0; w < 16; ++w) m[w] = 0u;
        #pragma unroll
        for (u32 i = 0; i < block_len; ++i)
            reinterpret_cast<uint8_t*>(m)[i] = src[i];

        u32 flags = 0;
        if (blk == 0) flags |= CHUNK_START;
        if (blk == 3) flags |= (CHUNK_END | ROOT);

        if (blk == 3) {
            // pre-final CV and final block for XOF recompress
            #pragma unroll
            for (int w=0; w<8; ++w)  d_precv[(idx0+lane)*8  + w]  = cv[w];
            #pragma unroll
            for (int w=0; w<16; ++w) d_last_words[(idx0+lane)*16 + w] = m[w];
            d_last_len[idx0+lane] = (uint8_t)block_len; // 48
        }

        g_compress(cv, m, 0ULL, block_len, flags, tmp_state);

        #pragma unroll
        for (int w=0; w<8; ++w) cv[w] = tmp_state[w];
    }

    #pragma unroll
    for (int w=0; w<8; ++w)
        d_roots[(idx0 + lane)*8 + w] = cv[w];
}

/* ---------------- public facades ---------------- */

extern "C"
void blake3_xof_cuda(const void *d_seeds, size_t seed_len,
                     void *d_xof,   size_t xof_len,
                     int batch, cudaStream_t s)
{
    constexpr size_t SEED      = 240;
    constexpr size_t MAT_BYTES = 1'607'680; // 25,120 * 64
    if (seed_len != SEED || xof_len != MAT_BYTES || batch <= 0) return;

    static u32     *d_roots      = nullptr;
    static u32     *d_precv      = nullptr;
    static u32     *d_last_words = nullptr;
    static uint8_t *d_last_len   = nullptr;
    static int cap = 0;

    if (batch > cap) {
        if (d_roots)      cudaFree(d_roots);
        if (d_precv)      cudaFree(d_precv);
        if (d_last_words) cudaFree(d_last_words);
        if (d_last_len)   cudaFree(d_last_len);
        cudaMalloc(&d_roots,      batch*8*sizeof(u32));
        cudaMalloc(&d_precv,      batch*8*sizeof(u32));
        cudaMalloc(&d_last_words, batch*16*sizeof(u32));
        cudaMalloc(&d_last_len,   batch);
        cap = batch;
    }

    int packs = (batch + 7) / 8;
    root_hash8<<<packs, 32, 0, s>>>(
        static_cast<const uint8_t*>(d_seeds),
        d_roots, d_precv, d_last_words, d_last_len, batch);

    const uint64_t blocks  = 25120ULL;
    const uint64_t threads = (uint64_t)batch * blocks;
    dim3 blk(256), grd((threads + 255) / 256);

    xof_expand<<<grd, blk, 0, s>>>(
        d_roots, d_precv, d_last_words, d_last_len,
        static_cast<uint8_t*>(d_xof),
        MAT_BYTES, batch);
}

/* keep a stub so test_smoke links, but do nothing here */
extern "C"
void blake3_matmul_cuda(const void *d_seeds,size_t seed_len,
                        void *d_out,size_t prod_len,int batch,
                        cudaStream_t s)
{
    (void)d_seeds; (void)seed_len; (void)d_out; (void)prod_len; (void)batch; (void)s;
}
