#pragma once
#include <vector>
#include <cstdint>
extern "C" {
#include "../blake3/c/blake3.h"
}

static constexpr int ROWS = 16, COLS = 16, K = 50240;
static constexpr size_t MAT = size_t(ROWS)*K + size_t(K)*COLS;     // 1,607,680 bytes

inline void blake3_xof_cpu(const uint8_t seed[240], std::vector<uint8_t>& out) {
    out.resize(MAT);
    blake3_hasher h;
    blake3_hasher_init(&h);
    blake3_hasher_update(&h, seed, 240);
    // XOF from offset 0, full length
    blake3_hasher_finalize(&h, out.data(), out.size());
}