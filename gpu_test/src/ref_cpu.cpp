#include <vector>
#include <cstdint>
#include <cstring>
#include "../blake3/c/blake3.h"

static constexpr int ROWS=16, COLS=16, K=50240;
static constexpr size_t MAT_BYTES = ROWS*K + K*COLS + ROWS*64;   // 1 607 680

void cpu_reference(const uint8_t seed[240], std::vector<int32_t>& C_out)
{
    static uint8_t whole[MAT_BYTES];
    blake3_hasher h; blake3_hasher_init(&h);
    blake3_hasher_update(&h, seed, 240);
    blake3_hasher_finalize_seek(&h, /*seek=*/0, whole, MAT_BYTES);

    const uint8_t *A = whole;
    const int8_t  *B = reinterpret_cast<int8_t*>(whole + ROWS*K);

    C_out.resize(ROWS*COLS);
    for(int i=0;i<ROWS;++i)
        for(int j=0;j<COLS;++j){
            int32_t sum=0;
            for(int k=0;k<K;++k)
                sum += int32_t(A[i*K+k]) * int32_t(B[k*COLS+j]);
            C_out[i*COLS+j] = sum;
        }
}
