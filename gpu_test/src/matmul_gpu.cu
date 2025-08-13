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

// --- GEMM: C16x16 = A(16x50240, u8) * B(50240x16, i8) -----------------
__global__ void matmul16x50240_u8_i8(
    const uint8_t* __restrict__ d_ab,  // A||B per seed (1,607,680 bytes)
    size_t per_ab,
    int32_t* __restrict__ d_C,         // 16*16 i32 per seed (256 i32)
    int batch)
{
    // 2D thread: (i,j) in 16x16 tile, one block per seed
    const int i = threadIdx.y; // 0..15
    const int j = threadIdx.x; // 0..15
    const int seed = blockIdx.x;
    if (seed >= batch || i >= 16 || j >= 16) return;

    const size_t A_BYTES = 16ull * 50240ull; // 803,840
    const size_t B_BYTES = 16ull * 50240ull; // 803,840
    const uint8_t* base = d_ab + (size_t)seed * per_ab;

    const uint8_t* A = base;                 // row-major u8 [16 x 50240]
    const uint8_t* B = base + A_BYTES;       // row-major u8 [50240 x 16] (to be read as i8)

    int32_t acc = 0;
    const int K = 50240;

    // Row i of A, Column j of B (note B is row-major: B[k*16 + j])
    const size_t rowA = (size_t)i * K;

    #pragma unroll 4
    for (int k = 0; k < K; ++k) {
        const int32_t a = (int32_t)A[rowA + k];              // u8 -> i32
        const int32_t b = (int32_t)((int8_t)B[(size_t)k*16 + j]); // u8 reinterpreted as i8 -> i32
        acc += a * b;
    }

    d_C[(size_t)seed * 256 + (size_t)i * 16 + j] = acc;
}

// --- BLAKE3 hash (single chunk, 1024B) of C (16x16 i32) ----------------
__global__ void blake3_hash_c1024(
    const int32_t* __restrict__ d_C,  // 256 i32 per seed
    uint8_t* __restrict__ d_out32,    // 32B per seed
    int batch)
{
    const int seed = blockIdx.x * blockDim.x + threadIdx.x;
    if (seed >= batch) return;

    // CV starts from IV
    u32 cv[8];
    #pragma unroll
    for (int w = 0; w < 8; ++w) cv[w] = g_IV[w];

    // Process exactly 16 blocks (16 * 64B = 1024B)
    const u32* data = reinterpret_cast<const u32*>(d_C + (size_t)seed * 256);
    u32 tmp_state[16];

    for (int blk = 0; blk < 16; ++blk) {
        u32 m[16];
        #pragma unroll
        for (int w = 0; w < 16; ++w) m[w] = data[blk*16 + w];

        u32 flags = 0;
        if (blk == 0)  flags |= CHUNK_START;
        if (blk == 15) flags |= (CHUNK_END | ROOT);

        g_compress(cv, m, 0ULL, 64u, flags, tmp_state);

        // Next CV = low half (already lo^hi)
        #pragma unroll
        for (int w = 0; w < 8; ++w) cv[w] = tmp_state[w];
    }

    // Write 32-byte hash (8 u32 words, little-endian)
    u32* dst = reinterpret_cast<u32*>(d_out32 + (size_t)seed * 32);
    #pragma unroll
    for (int w = 0; w < 8; ++w) dst[w] = cv[w];
}


// --- BLAKE3 hash of seed||C (240B + 1024B = 1264B) -> 32B ----------------
__global__ void blake3_hash_seed_plus_c(
    const uint8_t* __restrict__ d_seeds,   // 240B per item
    const int32_t* __restrict__ d_C,       // 256 i32 (1024B) per item
    uint8_t* __restrict__ d_out32,         // 32B per item
    int batch)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch) return;

    const uint8_t* seed = d_seeds + (size_t)idx * 240;
    const uint8_t* Cb   = reinterpret_cast<const uint8_t*>(d_C + (size_t)idx * 256);

    // ---- chunk 0 (1024B): seed[0..239] ++ Cb[0..783] ----
    u32 cv0[8];
    #pragma unroll
    for (int w=0; w<8; ++w) cv0[w] = g_IV[w];
    {
        u32 state[16];
        for (int blk=0; blk<16; ++blk) {
            u32 m[16] = {0};
            // fill 64B block from concatenated stream
            #pragma unroll
            for (int i=0; i<64; ++i) {
                const int off = blk*64 + i;      // 0..1023 within chunk 0
                uint8_t b;
                if (off < 240)  b = seed[off];
                else            b = Cb[off - 240]; // from C
                reinterpret_cast<uint8_t*>(m)[i] = b;
            }
            u32 flags = 0;
            if (blk == 0)  flags |= CHUNK_START;
            if (blk == 15) flags |= CHUNK_END;
            g_compress(cv0, m, /*t=*/0ULL, /*block_len=*/64u, flags, state);
            #pragma unroll
            for (int w=0; w<8; ++w) cv0[w] = state[w]; // next CV = low half
        }
    }

    // ---- chunk 1 (240B): Cb[784..1023] ----
    u32 cv1[8];
    #pragma unroll
    for (int w=0; w<8; ++w) cv1[w] = g_IV[w];
    {
        u32 state[16];
        for (int blk=0; blk<4; ++blk) {
            const u32 blen = (blk == 3) ? 48u : 64u; // 240B = 3*64 + 48
            u32 m[16] = {0};
            // fill from C tail (offset 784)
            #pragma unroll
            for (u32 i=0; i<blen; ++i) {
                reinterpret_cast<uint8_t*>(m)[i] = Cb[784 + blk*64 + i];
            }
            u32 flags = 0;
            if (blk == 0) flags |= CHUNK_START;
            if (blk == 3) flags |= CHUNK_END;
            g_compress(cv1, m, /*t=*/1ULL, blen, flags, state);
            #pragma unroll
            for (int w=0; w<8; ++w) cv1[w] = state[w];
        }
    }

    // ---- parent combine (ROOT) of the two leaf CVs ----
    {
        u32 parent_in[16];
        #pragma unroll
        for (int w=0; w<8; ++w) parent_in[w]   = cv0[w];
        #pragma unroll
        for (int w=0; w<8; ++w) parent_in[8+w] = cv1[w];

        u32 cv[8];  // parent uses IV as CV input
        #pragma unroll
        for (int w=0; w<8; ++w) cv[w] = g_IV[w];

        u32 st[16];
        g_compress(cv, parent_in, /*t=*/0ULL, /*block_len=*/64u,
                   /*flags=*/PARENT | ROOT, st);

        // write 32B root (low 8 words already lo^hi)
        u32* dst = reinterpret_cast<u32*>(d_out32 + (size_t)idx * 32);
        #pragma unroll
        for (int w=0; w<8; ++w) dst[w] = st[w];
    }
}


extern "C"
void blake3_matmul_cuda(const void *d_seeds, size_t seed_len,
                        void *d_out, size_t prod_len, int batch,
                        cudaStream_t s)
{
    // Expect 240-byte seeds, and either 1024B (C) or 32B (hash) per item.
    constexpr size_t SEED = 240;
    constexpr size_t A_BYTES = 16ull * 50240ull;      // 803,840
    constexpr size_t B_BYTES = 16ull * 50240ull;      // 803,840
    constexpr size_t AB_BYTES = A_BYTES + B_BYTES;    // 1,607,680
    constexpr size_t C_BYTES = 16ull * 16ull * 4ull;  // 1,024
    if (seed_len != SEED || batch <= 0) return;
    if (prod_len != 32 && prod_len != C_BYTES) return;

    // Scratch buffers (grow-once semantics like above)
    static uint8_t* d_ab = nullptr;   // A||B per seed
    static int32_t* d_C  = nullptr;   // 16x16 i32 per seed
    static int cap = 0;

    if (batch > cap) {
        if (d_ab) cudaFree(d_ab);
        if (d_C)  cudaFree(d_C);
        cudaMalloc(&d_ab, (size_t)batch * AB_BYTES);
        cudaMalloc(&d_C,  (size_t)batch * (C_BYTES));
        cap = batch;
    }

    // 1) Build roots & final-leaf materials for XOF
    static u32     *d_roots      = nullptr;
    static u32     *d_precv      = nullptr;
    static u32     *d_last_words = nullptr;
    static uint8_t *d_last_len   = nullptr;
    static int cap2 = 0;
    if (batch > cap2) {
        if (d_roots)      cudaFree(d_roots);
        if (d_precv)      cudaFree(d_precv);
        if (d_last_words) cudaFree(d_last_words);
        if (d_last_len)   cudaFree(d_last_len);
        cudaMalloc(&d_roots,      (size_t)batch * 8 * sizeof(u32));
        cudaMalloc(&d_precv,      (size_t)batch * 8 * sizeof(u32));
        cudaMalloc(&d_last_words, (size_t)batch * 16 * sizeof(u32));
        cudaMalloc(&d_last_len,   (size_t)batch * sizeof(uint8_t));
        cap2 = batch;
    }

    // One warp per pack of 8 seeds
    int packs = (batch + 7) / 8;
    root_hash8<<<packs, 32, 0, s>>>(
        static_cast<const uint8_t*>(d_seeds),
        d_roots, d_precv, d_last_words, d_last_len, batch);

    // 2) XOF expand to A||B (exactly 1,607,680 bytes per seed)
    {
        const uint64_t blocks  = 25120ULL;           // 25,120 * 64B = 1,607,680B
        const uint64_t threads = (uint64_t)batch * blocks;
        dim3 blk(256), grd((threads + 255) / 256);
        xof_expand<<<grd, blk, 0, s>>>(
            d_roots, d_precv, d_last_words, d_last_len,
            d_ab, AB_BYTES, batch);
    }

    // 3) Matmul C = A * B (u8 x i8 -> i32), one block (16x16) per seed
    {
        dim3 blk(16, 16, 1);
        dim3 grd(batch, 1, 1);
        matmul16x50240_u8_i8<<<grd, blk, 0, s>>>(d_ab, AB_BYTES, d_C, batch);
    }

//    // 4) Output selection: 1024B matrix or 32B BLAKE3(C) hash
//    if (prod_len == C_BYTES) {
//        // Raw 16x16 i32 matrix (row-major, little-endian)
//        cudaMemcpyAsync(d_out, d_C, (size_t)batch * C_BYTES,
//                        cudaMemcpyDeviceToDevice, s);
//    } else {
//        // BLAKE3 hash of exactly 1024 bytes (C) per seed (single chunk)
//        int threads = 256;
//        int blocks  = (batch + threads - 1) / threads;
//        blake3_hash_c1024<<<blocks, threads, 0, s>>>(
//            d_C, static_cast<uint8_t*>(d_out), batch);
//    }
    int threads = 256;
    int blocks  = (batch + threads - 1) / threads;
    blake3_hash_seed_plus_c<<<blocks, threads, 0, s>>>(
        static_cast<const uint8_t*>(d_seeds),
        d_C,
        static_cast<uint8_t*>(d_out),
        batch);
}

// --- helpers: write per-seed nonces into the last 8B (LE) of each seed ----
__device__ inline void store_u64_le(uint8_t* p, uint64_t x) {
    #pragma unroll
    for (int i=0; i<8; ++i) p[i] = (uint8_t)((x >> (8*i)) & 0xFF);
}

__global__ void write_nonces_tail_le(
    uint8_t* __restrict__ d_seeds,  // 240B per item (will be modified in-place)
    int batch,
    uint64_t start_nonce,
    uint64_t round,                  // round index
    uint64_t batch_stride            // usually == batch
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch) return;
    const uint64_t nonce = start_nonce + round*batch_stride + (uint64_t)idx;

    uint8_t* seed_tail = d_seeds + (size_t)idx*240 + 232; // last 8 bytes
    store_u64_le(seed_tail, nonce);
}

// --- check: does hash start with two 0x00 bytes? set the first global winner ---
__global__ void find_first_two_zeroes(
    const uint8_t* __restrict__ d_hash32, // 32B per item
    const uint8_t* __restrict__ d_seeds,  // to re-read fields for the winner
    int batch,
    int*       __restrict__ d_found_idx,        // -1 if none yet
    uint64_t*  __restrict__ d_found_nonce,      // still reporting nonce
    uint32_t*  __restrict__ d_found_u32_at_228  // NEW: the u32 at offset 228
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch) return;

    const uint8_t* h = d_hash32 + (size_t)idx * 32;
    if ((h[0] == 0x00) && (h[1] == 0x00) && (h[2] == 0x00)) {
        // claim "first"
        int old = atomicCAS(d_found_idx, -1, idx);
        if (old == -1) {
            // read nonce (LE) from seed tail [232..239]
            const uint8_t* tail = d_seeds + (size_t)idx*240 + 232;
            uint64_t n = 0;
            #pragma unroll
            for (int i=0;i<8;++i) n |= ((uint64_t)tail[i]) << (8*i);
            *d_found_nonce = n;

            // read u32 at byte offset 228 (aligned)
            const uint32_t* p228 =
                reinterpret_cast<const uint32_t*>(d_seeds + (size_t)idx*240 + 228);
            *d_found_u32_at_228 = *p228;
        }
    }
}
// Searches for the first hash (of seed||C) that starts with two 0x00 bytes.
// Assumptions:
// - Nonce is stored LE in seed bytes [232..239] (last 8 bytes).
// - We iterate nonces as: start_nonce + round*batch + idx.
// Returns true if found; sets *h_found_idx (0..batch-1) and *h_found_nonce.
extern "C"
bool blake3_matmul_cuda_find2zero(
    void*       d_seeds,       size_t seed_len,   // 240
    void*       d_out_hashes,  size_t out_len,    // must be 32*batch
    int         batch,
    uint64_t    start_nonce,
    int         max_rounds,    // safety cap
    int*        h_found_idx,
    uint64_t*   h_found_nonce,
    uint32_t*   h_found_u32_at_228,
    cudaStream_t s)
{
    constexpr size_t SEED = 240;
    if (seed_len != SEED || batch <= 0) return false;
    if (out_len != (size_t)batch * 32) return false;

    // device-side found flags
    int *d_found_idx = nullptr;
    uint64_t *d_found_nonce = nullptr;
    uint32_t *d_found_u32_at_228 = nullptr;
                                            
    cudaMalloc(&d_found_idx, sizeof(int));
    cudaMalloc(&d_found_nonce, sizeof(uint64_t));
    cudaMalloc(&d_found_u32_at_228, sizeof(uint32_t));

    auto reset_found = [&]() {
        int init = -1;
        uint64_t z = 0;
        cudaMemcpyAsync(d_found_idx, &init, sizeof(int), cudaMemcpyHostToDevice, s);
        cudaMemcpyAsync(d_found_nonce, &z, sizeof(uint64_t), cudaMemcpyHostToDevice, s);
     };

    // Scratch hash buffer is the user's d_out_hashes (32B * batch)
    // We reuse your existing internal scratch from blake3_matmul_cuda path:
    // d_ab, d_C, d_roots, d_precv, d_last_words, d_last_len etc.
    // We'll replicate the pipeline with nonce write + compute + check in a loop.

    // Internal constants matching your existing code:
    constexpr size_t A_BYTES  = 16ull * 50240ull;
    constexpr size_t B_BYTES  = 16ull * 50240ull;
    constexpr size_t AB_BYTES = A_BYTES + B_BYTES;
    constexpr size_t C_BYTES  = 16ull * 16ull * 4ull;

    // ---- grow-once scratch (mirroring your function) ----
    static uint8_t* d_ab = nullptr;   // A||B per seed
    static int32_t* d_C  = nullptr;   // 16x16 i32 per seed
    static int cap = 0;

    if (batch > cap) {
        if (d_ab) cudaFree(d_ab);
        if (d_C)  cudaFree(d_C);
        cudaMalloc(&d_ab, (size_t)batch * AB_BYTES);
        cudaMalloc(&d_C,  (size_t)batch * C_BYTES);
        cap = batch;
    }

    static u32     *d_roots      = nullptr;
    static u32     *d_precv      = nullptr;
    static u32     *d_last_words = nullptr;
    static uint8_t *d_last_len   = nullptr;
    static int cap2 = 0;
    if (batch > cap2) {
        if (d_roots)      cudaFree(d_roots);
        if (d_precv)      cudaFree(d_precv);
        if (d_last_words) cudaFree(d_last_words);
        if (d_last_len)   cudaFree(d_last_len);
        cudaMalloc(&d_roots,      (size_t)batch * 8 * sizeof(u32));
        cudaMalloc(&d_precv,      (size_t)batch * 8 * sizeof(u32));
        cudaMalloc(&d_last_words, (size_t)batch * 16 * sizeof(u32));
        cudaMalloc(&d_last_len,   (size_t)batch * sizeof(uint8_t));
        cap2 = batch;
    }

    // Kernel launch shapes
    const int threads = 256;
    const int blocksB = (batch + threads - 1) / threads;

    // XOF expansion grid (same as your code)
    const uint64_t blocks  = 25120ULL;
    const uint64_t threads_total = (uint64_t)batch * blocks;
    dim3 xof_blk(256), xof_grd((threads_total + 255) / 256);

    // GEMM launch
    dim3 gemm_blk(16, 16, 1);
    dim3 gemm_grd(batch, 1, 1);

    // ---- rounds loop ----
    bool found = false;
    *h_found_idx   = -1;
    *h_found_nonce = 0;
    *h_found_u32_at_228  = 0;

    reset_found();

cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

    for (int round = 0; round < max_rounds; ++round) {

        cudaEventRecord(start, s);

        // 0) Write nonces for this round into seeds tail (in-place)
        write_nonces_tail_le<<<blocksB, threads, 0, s>>>(
            static_cast<uint8_t*>(d_seeds),
            batch,
            start_nonce,
            (uint64_t)round,
            (uint64_t)batch
        );

        // 1) Build roots/final-leaf for XOF
        {
            int packs = (batch + 7) / 8;
            root_hash8<<<packs, 32, 0, s>>>(
                static_cast<const uint8_t*>(d_seeds),
                d_roots, d_precv, d_last_words, d_last_len, batch);
        }

        // 2) XOF expand -> A||B
        xof_expand<<<xof_grd, xof_blk, 0, s>>>(
            d_roots, d_precv, d_last_words, d_last_len,
            d_ab, AB_BYTES, batch);

        // 3) GEMM C = A * B
        matmul16x50240_u8_i8<<<gemm_grd, gemm_blk, 0, s>>>(d_ab, AB_BYTES, d_C, batch);

        // 4) Hash seed||C -> 32B
        blake3_hash_seed_plus_c<<<blocksB, threads, 0, s>>>(
            static_cast<const uint8_t*>(d_seeds),
            d_C,
            static_cast<uint8_t*>(d_out_hashes),
            batch);

        // 5) Check for two leading zero bytes, claim first
        find_first_two_zeroes<<<blocksB, threads, 0, s>>>(
            static_cast<const uint8_t*>(d_out_hashes),
            static_cast<const uint8_t*>(d_seeds),
            batch,
            d_found_idx,
            d_found_nonce,
            d_found_u32_at_228
        );


        cudaEventRecord(stop, s);
        cudaEventSynchronize(stop);

        float ms_round = 0.0f;
        cudaEventElapsedTime(&ms_round, start, stop);

        double hps = (double)batch / (ms_round / 1000.0);
        printf("Round %d: %.2f MH/s\n", round, hps / 1e6);

        // Pull flag to host
        int h_idx = -1;
        cudaMemcpyAsync(&h_idx, d_found_idx, sizeof(int), cudaMemcpyDeviceToHost, s);
        cudaStreamSynchronize(s);


        if (h_idx != -1) {
            uint64_t h_nonce = 0;
            uint32_t h_u32_228 = 0;
            cudaMemcpy(&h_nonce, d_found_nonce, sizeof(uint64_t), cudaMemcpyDeviceToHost);

            cudaMemcpy(&h_u32_228, d_found_u32_at_228, sizeof(uint32_t), cudaMemcpyDeviceToHost);
            *h_found_idx   = h_idx;
            *h_found_nonce = h_nonce;
            *h_found_u32_at_228 = h_u32_228;
            found = true;
            break;
        }
        // else continue next round with new nonces
    }

    cudaFree(d_found_idx);
    cudaFree(d_found_nonce);
    return found;
}
