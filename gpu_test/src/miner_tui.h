// miner_tui.h - header-only API for the console TUI + JSON logging.
// Build: link with -lnvidia-ml (NVML). If NVML unavailable, compile with -DMINER_TUI_NO_NVML

#pragma once
#include <cstdint>
#include <cstddef>

#ifdef __cplusplus
extern "C" {
#endif

// Initialize the live TUI + logger.
//  - device_index: which CUDA/NVML device to watch (default 0)
//  - refresh_ms:   UI update interval (e.g. 250..1000)
//  - json_path:    NDJSON log path (e.g. "miner_bench.ndjson")
void miner_tui_start(int device_index, int refresh_ms, const char* json_path);

// Stop the UI thread and close the log.
void miner_tui_stop(void);

// Tell the TUI once at start what batch size you mine per round.
void miner_tui_set_batch(int64_t batch);

// Record one mining round timing (ms). The TUI derives hashes/s from batch + time.
//  - ms_round: wall time for the full “nonce write -> roots -> fused XOF+GEMM -> hash -> check”
void miner_tui_record_round_ms(double ms_round);

// Optional: mark if a winning hash was found in this round (appears in UI as a blip).
void miner_tui_mark_found(int found_idx);

// Optional: override a short status line (e.g., “warming up”, “profiling”, etc.)
void miner_tui_set_status(const char* status);

// Optional: set a tag that will be stored in JSON samples (e.g., git SHA, config name)
void miner_tui_set_tag(const char* tag);

// Convenience: suspend/resume screen painting (e.g., around heavy stderr prints)
void miner_tui_pause_paint(int pause_on);

#ifdef __cplusplus
}
#endif
