#include "blaze3_cpu.cuh"

// Number of threads per thread block
__constant__ const int NUM_THREADS = 16;

// redefine functions, but for the GPU
// all of them are the same but with g_ prefixed
__constant__ const u32 g_IV[8] = {
    0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A, 
    0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19,
};

__constant__ const int g_MSG_PERMUTATION[] = {
    2, 6, 3, 10, 7, 0, 4, 13, 
    1, 11, 12, 5, 9, 14, 15, 8
};

__device__ u32 g_rotr(u32 value, int shift) {
    return (value >> shift)|(value << (usize - shift));
}

__device__ void g_g(u32 state[16], u32 a, u32 b, u32 c, u32 d, u32 mx, u32 my) {
    state[a] = state[a] + state[b] + mx;
    state[d] = g_rotr((state[d] ^ state[a]), 16);
    state[c] = state[c] + state[d];

    state[b] = g_rotr((state[b] ^ state[c]), 12);
    state[a] = state[a] + state[b] + my;
    state[d] = g_rotr((state[d] ^ state[a]), 8);

    state[c] = state[c] + state[d];
    state[b] = g_rotr((state[b] ^ state[c]), 7);
}

__device__ void g_round(u32 state[16], u32 m[16]) {
    // Mix the columns.
    g_g(state, 0, 4, 8, 12, m[0], m[1]);
    g_g(state, 1, 5, 9, 13, m[2], m[3]);
    g_g(state, 2, 6, 10, 14, m[4], m[5]);
    g_g(state, 3, 7, 11, 15, m[6], m[7]);
    // Mix the diagonals.
    g_g(state, 0, 5, 10, 15, m[8], m[9]);
    g_g(state, 1, 6, 11, 12, m[10], m[11]);
    g_g(state, 2, 7, 8, 13, m[12], m[13]);
    g_g(state, 3, 4, 9, 14, m[14], m[15]);
}

__device__ void g_permute(u32 m[16]) {
    u32 permuted[16];
    for(int i=0; i<16; i++)
        permuted[i] = m[g_MSG_PERMUTATION[i]];
    for(int i=0; i<16; i++)
        m[i] = permuted[i];
}

// custom memcpy, apparently cuda's memcpy is slow 
// when called within a kernel
__device__ void g_memcpy(u32 *lhs, const u32 *rhs, int size) {
    // assuming u32 is 4 bytes
    int len = size / 4;
    for(int i=0; i<len; i++)
        lhs[i] = rhs[i];
}

// custom memset
template<typename T, typename ptr_t>
__device__ void g_memset(ptr_t dest, T val, int count) {
    for(int i=0; i<count; i++)
        dest[i] = val;
}

__device__ void g_compress(
    u32 *chaining_value,
    u32 *block_words,
    u64 counter,
    u32 block_len,
    u32 flags,
    u32 *state
) {
    // Search for better alternative
    g_memcpy(state, chaining_value, 32);
    g_memcpy(state+8, g_IV, 16);
    state[12] = (u32)counter;
    state[13] = (u32)(counter >> 32);
    state[14] = block_len;
    state[15] = flags;

    u32 block[16];
    g_memcpy(block, block_words, 64);
    
    g_round(state, block); // round 1
    g_permute(block);
    g_round(state, block); // round 2
    g_permute(block);
    g_round(state, block); // round 3
    g_permute(block);
    g_round(state, block); // round 4
    g_permute(block);
    g_round(state, block); // round 5
    g_permute(block);
    g_round(state, block); // round 6
    g_permute(block);
    g_round(state, block); // round 7

    for(int i=0; i<8; i++){
        state[i] ^= state[i + 8];
        state[i + 8] ^= chaining_value[i];
    }
}

__forceinline__ __device__ u32 rotr(u32 x, int n) {
    return __funnelshift_r(x, x, n);
}

#define G(a,b,c,d, mx, my)        \
    do {                          \
        a = a + b + (mx);         \
        d = rotr(d ^ a, 16);      \
        c = c + d;                \
        b = rotr(b ^ c, 12);      \
        a = a + b + (my);         \
        d = rotr(d ^ a, 8);       \
        c = c + d;                \
        b = rotr(b ^ c, 7);       \
    } while(0)

__forceinline__ __device__
void g_compress_fast(const u32* __restrict cv,
                     const u32* __restrict block_words,
                     u64 counter, u32 block_len, u32 flags,
                     u32* __restrict out_state /* 16 */)
{
    // State in registers
    u32 v0, v1, v2, v3, v4, v5, v6, v7;
    u32 v8, v9, v10, v11, v12, v13, v14, v15;

    v0 = cv[0]; v1 = cv[1]; v2 = cv[2]; v3 = cv[3];
    v4 = cv[4]; v5 = cv[5]; v6 = cv[6]; v7 = cv[7];

    v8  = kIV[0]; v9  = kIV[1]; v10 = kIV[2]; v11 = kIV[3];
    v12 = (u32)counter;
    v13 = (u32)(counter >> 32);
    v14 = block_len;
    v15 = flags;

    // Load message to registers (assumes aligned, little-endian)
    u32 m0 = block_words[0];  u32 m1 = block_words[1];
    u32 m2 = block_words[2];  u32 m3 = block_words[3];
    u32 m4 = block_words[4];  u32 m5 = block_words[5];
    u32 m6 = block_words[6];  u32 m7 = block_words[7];
    u32 m8 = block_words[8];  u32 m9 = block_words[9];
    u32 m10= block_words[10]; u32 m11= block_words[11];
    u32 m12= block_words[12]; u32 m13= block_words[13];
    u32 m14= block_words[14]; u32 m15= block_words[15];

    // Helper to index m by compile-time schedule
    auto M = [&](int r, int i)->u32 {
        switch (kMsgIdx[r][i]) {
            case  0: return m0;  case  1: return m1;  case  2: return m2;  case  3: return m3;
            case  4: return m4;  case  5: return m5;  case  6: return m6;  case  7: return m7;
            case  8: return m8;  case  9: return m9;  case 10: return m10; case 11: return m11;
            case 12: return m12; case 13: return m13; case 14: return m14; case 15: return m15;
        }
        return 0u;
    };

    #pragma unroll
    for (int r = 0; r < 7; ++r) {
        // columns
        G(v0, v4, v8, v12,  M(r,0),  M(r,1));
        G(v1, v5, v9, v13,  M(r,2),  M(r,3));
        G(v2, v6, v10,v14,  M(r,4),  M(r,5));
        G(v3, v7, v11,v15,  M(r,6),  M(r,7));
        // diagonals
        G(v0, v5, v10,v15,  M(r,8),  M(r,9));
        G(v1, v6, v11,v12,  M(r,10), M(r,11));
        G(v2, v7, v8, v13,  M(r,12), M(r,13));
        G(v3, v4, v9, v14,  M(r,14), M(r,15));
    }

    // feedforward
    out_state[0]  = v0 ^ v8  ^ cv[0];
    out_state[1]  = v1 ^ v9  ^ cv[1];
    out_state[2]  = v2 ^ v10 ^ cv[2];
    out_state[3]  = v3 ^ v11 ^ cv[3];
    out_state[4]  = v4 ^ v12 ^ cv[4];
    out_state[5]  = v5 ^ v13 ^ cv[5];
    out_state[6]  = v6 ^ v14 ^ cv[6];
    out_state[7]  = v7 ^ v15 ^ cv[7];
    out_state[8]  = v8  ^ cv[0];
    out_state[9]  = v9  ^ cv[1];
    out_state[10] = v10 ^ cv[2];
    out_state[11] = v11 ^ cv[3];
    out_state[12] = v12 ^ cv[4];
    out_state[13] = v13 ^ cv[5];
    out_state[14] = v14 ^ cv[6];
    out_state[15] = v15 ^ cv[7];
}
__device__ void g_words_from_little_endian_bytes(
    u8 *bytes, u32 *words, u32 bytes_len
) {
    u32 tmp;
    for(u32 i=0; i<bytes_len; i+=4) {
        tmp = (bytes[i+3]<<24) | (bytes[i+2]<<16) | (bytes[i+1]<<8) | bytes[i];
        words[i/4] = tmp;
    }
}

__device__ void Chunk::g_compress_chunk(u32 out_flags) {
    if(flags&PARENT) {
        g_compress(
            key,
            data,
            0,  // counter is always zero for parent nodes
            BLOCK_LEN,
            flags | out_flags,
            raw_hash
        );
        return;
    }

    u32 chaining_value[8];
    u32 block_len = BLOCK_LEN, flagger;
    g_memcpy(chaining_value, key, 32);

    bool empty_input = (leaf_len==0);
    if(empty_input) {
        for(u32 i=0; i<BLOCK_LEN; i++)
            leaf_data[i] = 0U;
        leaf_len = BLOCK_LEN;
    }

    // move all mem allocs outside loop
    u32 block_words[16];
    u8 block_cast[BLOCK_LEN];

    for(u32 i=0; i<leaf_len; i+=BLOCK_LEN) {
        flagger = flags;
        // for the last message block
        if(i+BLOCK_LEN > leaf_len)
            block_len = leaf_len%BLOCK_LEN;
        else
            block_len = BLOCK_LEN;

        // special case
        if(empty_input)
            block_len = 0;
        
        // clear up block_words
        g_memset(block_words, 0, 16);

        u32 new_block_len(block_len);
        if(block_len%4)
            new_block_len += 4 - (block_len%4);
        
        // This memcpy is fine since data is a byte array
        memcpy(block_cast, leaf_data+i, new_block_len*sizeof(*block_cast));
        
        g_words_from_little_endian_bytes(leaf_data+i, block_words, new_block_len);

        if(i==0)
            flagger |= CHUNK_START;
        if(i+BLOCK_LEN >= leaf_len)
            flagger |= CHUNK_END | out_flags;

        // raw hash for root node
        g_compress(
            chaining_value,
            block_words,
            counter,
            block_len,
            flagger,
            raw_hash
        );

        g_memcpy(chaining_value, raw_hash, 32);
    }
}

__global__ void compute(Chunk *data, int l, int r) {
    // n is always a power of 2
    int n = r-l;
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if(tid >= n)
        return;
    
    if(n==1) {
        data[l].g_compress_chunk();
        // printf("Compressing : %d\n", l);
    }
    else {
        compute<<<n/2,16>>>(data, l, l+n/2);
        cudaDeviceSynchronize();
        compute<<<n/2,16>>>(data, l+n/2, r);
        cudaDeviceSynchronize();

        data[l].flags |= PARENT;

        memcpy(data[l].data, data[l].raw_hash, 32);
        memcpy(data[l].data+8, data[l+n/2].raw_hash, 32);
        data[l].g_compress_chunk();
        // printf("Compressing : %d to %d\n", l, r);
    }
}

void light_hash(Chunk *data, int N, Chunk *result, Chunk *memory_bar) {
    const int data_size = N*sizeof(Chunk);

    // Device settings
    // Allows DeviceSync to be called upto 16 levels of recursion
    cudaDeviceSetLimit(cudaLimitDevRuntimeSyncDepth, 16);

    // Device vector
    Chunk *g_data = memory_bar;
    cudaMemcpy(g_data, data, data_size, cudaMemcpyHostToDevice);

    // Actual computation of hash
    compute<<<N,32>>>(g_data, 0, N);

    cudaMemcpy(result, g_data, sizeof(Chunk), cudaMemcpyDeviceToHost);
}
