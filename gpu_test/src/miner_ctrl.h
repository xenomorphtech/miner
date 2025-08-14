// miner_ctrl.h
#pragma once
#include <stdint.h>

struct __align__(64) MinerCtrl {
  volatile uint64_t epoch;        // host increments to force all blocks to exit
  volatile uint64_t start_nonce;  // base nonce
  volatile int      stop;         // set 1 to stop gracefully
  volatile uint32_t pad;          // padding
};

struct FoundResult {
  int       seed_idx;
  uint32_t  u32_at_228;
  uint64_t  nonce;
  uint8_t   hash32[32];
  uint8_t   seed240[240];
};

// device ring meta (in device mem or mapped pinned)
struct __align__(64) RingMeta {
  volatile uint32_t head;   // atomic producer index
  uint32_t          cap;    // capacity
};
