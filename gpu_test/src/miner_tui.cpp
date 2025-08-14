// miner_tui.cpp
#include "miner_tui.h"

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <mutex>
#include <atomic>
#include <vector>
#include <deque>
#include <chrono>
#include <algorithm>
#include <fstream>
#include <iomanip>

#ifndef MINER_TUI_NO_NVML
  #include <nvml.h>
  #define NVML_OKAY(expr) ((expr) == NVML_SUCCESS)
#else
  // Stubs if NVML is not present
  typedef int nvmlReturn_t;
  #define NVML_OKAY(x) false
#endif

// --------- ANSI colors / helpers ----------
#define C_RESET   "\x1b[0m"
#define C_DIM     "\x1b[2m"
#define C_BOLD    "\x1b[1m"
#define C_RED     "\x1b[31m"
#define C_GREEN   "\x1b[32m"
#define C_YELLOW  "\x1b[33m"
#define C_BLUE    "\x1b[34m"
#define C_MAGENTA "\x1b[35m"
#define C_CYAN    "\x1b[36m"
#define C_GREY    "\x1b[90m"

static inline void term_hide_cursor() { std::fputs("\x1b[?25l", stdout); }
static inline void term_show_cursor() { std::fputs("\x1b[?25h", stdout); }
static inline void term_home()        { std::fputs("\x1b[H", stdout); }
static inline void term_clear_all()   { std::fputs("\x1b[2J", stdout); }
static inline void term_clear_eol()   { std::fputs("\x1b[K", stdout); }

// --------- State ----------
namespace {
struct KernelWindow {
  std::vector<double> samples_ms;
  size_t cap = 512;
  void push(double ms) {
    if (samples_ms.size() == cap) samples_ms.erase(samples_ms.begin());
    samples_ms.push_back(ms);
  }
  static double pctile(std::vector<double> v, double p01) {
    if (v.empty()) return 0.0;
    if (p01 < 0) p01 = 0; if (p01 > 1) p01 = 1;
    size_t k = (size_t)((v.size()-1) * p01);
    std::nth_element(v.begin(), v.begin()+k, v.end());
    return v[k];
  }
  double p50() const { return pctile(samples_ms, 0.50); }
  double p95() const { return pctile(samples_ms, 0.95); }
  double p99() const { return pctile(samples_ms, 0.99); }
  double last()const { return samples_ms.empty()?0.0:samples_ms.back(); }
};

struct HsRing {
  // store (t_ms, hashes_done_since_last)
  std::deque<std::pair<uint64_t,int64_t>> q;
  int64_t total = 0;
  void add(uint64_t t_ms, int64_t h) {
    q.emplace_back(t_ms, h);
    total += h;
    // keep last 2 minutes (~120s)
    while (!q.empty() && (t_ms - q.front().first > 120000ULL)) {
      total -= q.front().second;
      q.pop_front();
    }
  }
  static double window_hps(const std::deque<std::pair<uint64_t,int64_t>>& q, uint64_t now_ms, uint64_t win_ms) {
    if (q.empty()) return 0.0;
    int64_t sum = 0;
    uint64_t t0  = now_ms - win_ms;
    for (auto it = q.rbegin(); it != q.rend(); ++it) {
      if (it->first < t0) break;
      sum += it->second;
    }
    double sec = win_ms / 1000.0;
    return sum / sec;
  }
  double hps_1s (uint64_t now_ms) const { return window_hps(q, now_ms, 1000); }
  double hps_10s(uint64_t now_ms) const { return window_hps(q, now_ms, 10000); }
  double hps_60s(uint64_t now_ms) const { return window_hps(q, now_ms, 60000); }
};

struct NvmlCtx {
#ifndef MINER_TUI_NO_NVML
  nvmlDevice_t h = nullptr;
  bool ok = false;
  int index = 0;
#endif
};

struct GpuProps {
  std::string name;
  int cc_major = 0, cc_minor = 0;
  int sm_count = 0;
  size_t vram_total = 0;
};

struct Telemetry {
  // NVML live
  int tempC = -1;
  int pstate = -1;
  int util_gpu = -1;
  int util_mem = -1;
  int sm_clock_MHz = -1;
  int mem_clock_MHz = -1;
  int pcie_gen_cur = -1, pcie_gen_max = -1, pcie_width = -1;
  uint64_t vram_used = 0, vram_total = 0;
  double power_W = 0.0;
};

struct Tui {
  std::mutex m;
  std::atomic<bool> running{false};
  std::atomic<bool> paint_paused{false};
  std::thread thr;

  // config
  int device_index = 0;
  int refresh_ms   = 500;
  std::string json_path = "miner_bench.ndjson";
  std::string tag;

  // mining stats
  std::chrono::steady_clock::time_point t0;
  int64_t batch = 0;
  int64_t rounds = 0;
  int64_t hashes_total = 0;
  int last_found = -1;
  KernelWindow win;
  HsRing ring;
  std::string status;

  // device
  GpuProps gpu;
  NvmlCtx  nvml;

  // file
  std::ofstream log;
} G;
} // anon

// --------- NVML helpers ----------
static void nvml_init_if_needed() {
#ifdef MINER_TUI_NO_NVML
  (void)G;
#else
  if (G.nvml.ok) return;
  if (!NVML_OKAY(nvmlInit_v2())) return;
  nvmlDevice_t h;
  if (NVML_OKAY(nvmlDeviceGetHandleByIndex_v2(G.device_index, &h))) {
    G.nvml.h = h;
    G.nvml.ok = true;
    G.nvml.index = G.device_index;
  }
#endif
}

static Telemetry poll_nvml() {
  Telemetry t{};
#ifndef MINER_TUI_NO_NVML
  if (!G.nvml.ok) { nvml_init_if_needed(); }
  if (!G.nvml.ok) return t;

  nvmlUtilization_t u{}; nvmlMemory_t mem{};
  unsigned int val = 0;

  if (NVML_OKAY(nvmlDeviceGetUtilizationRates(G.nvml.h, &u))) {
    t.util_gpu = (int)u.gpu;
    t.util_mem = (int)u.memory;
  }
  if (NVML_OKAY(nvmlDeviceGetTemperature(G.nvml.h, NVML_TEMPERATURE_GPU, &val)))
    t.tempC = (int)val;

  if (NVML_OKAY(nvmlDeviceGetClockInfo(G.nvml.h, NVML_CLOCK_SM, &val)))
    t.sm_clock_MHz = (int)val;
  if (NVML_OKAY(nvmlDeviceGetClockInfo(G.nvml.h, NVML_CLOCK_MEM, &val)))
    t.mem_clock_MHz = (int)val;

  if (NVML_OKAY(nvmlDeviceGetPerformanceState(G.nvml.h, (nvmlPstates_t*)&val)))
    t.pstate = (int)val;

  if (NVML_OKAY(nvmlDeviceGetPowerUsage(G.nvml.h, &val)))
    t.power_W = val / 1000.0;

  // *** PCIe (Curr/Max) ***
  if (NVML_OKAY(nvmlDeviceGetCurrPcieLinkGeneration(G.nvml.h, &val)))
    t.pcie_gen_cur = (int)val;
  if (NVML_OKAY(nvmlDeviceGetMaxPcieLinkGeneration(G.nvml.h, &val)))
    t.pcie_gen_max = (int)val;

  if (NVML_OKAY(nvmlDeviceGetCurrPcieLinkWidth(G.nvml.h, &val)))
    t.pcie_width = (int)val;
  // (Optional) if you add a max-width field to Telemetry, also call:
  // if (NVML_OKAY(nvmlDeviceGetMaxPcieLinkWidth(G.nvml.h, &val))) t.pcie_width_max = (int)val;

  if (NVML_OKAY(nvmlDeviceGetMemoryInfo(G.nvml.h, &mem))) {
    t.vram_used  = mem.used;
    t.vram_total = mem.total;
  }
#endif
  return t;
}

// --------- Device props ----------
static void fill_cuda_props() {
  cudaDeviceProp p{};
  int dev = 0; cudaGetDevice(&dev);
  cudaGetDeviceProperties(&p, dev);
  G.gpu.name = p.name ? p.name : "NVIDIA GPU";
  G.gpu.cc_major = p.major; G.gpu.cc_minor = p.minor;
  G.gpu.sm_count = p.multiProcessorCount;
  G.gpu.vram_total = p.totalGlobalMem;
}

// --------- JSON logging ----------
static void write_sample_json(const Telemetry& t, uint64_t now_ms,
                              double h1, double h10, double h60,
                              double p50, double p95, double p99, double last_ms) {
  if (!G.log.is_open()) return;
  double uptime_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - G.t0).count();
  double eff = (t.power_W > 0.1) ? (h10 / t.power_W) : 0.0;

  G.log << std::fixed << std::setprecision(3)
        << "{\"ts_ms\":" << now_ms
        << ",\"uptime_s\":" << uptime_s
        << ",\"tag\":\"" << G.tag << "\""
        << ",\"rounds\":" << G.rounds
        << ",\"hashes_total\":" << G.hashes_total
        << ",\"batch\":" << G.batch
        << ",\"hashrate_hps\":{\"h1\":" << h1 << ",\"h10\":" << h10 << ",\"h60\":" << h60 << "}"
        << ",\"kernel_ms\":{\"last\":" << last_ms << ",\"p50\":" << p50 << ",\"p95\":" << p95 << ",\"p99\":" << p99 << "}"
        << ",\"gpu\":{\"name\":\"" << G.gpu.name << "\",\"cc\":\"" << G.gpu.cc_major << "." << G.gpu.cc_minor
        << "\",\"sms\":" << G.gpu.sm_count
        << ",\"tempC\":" << t.tempC
        << ",\"power_W\":" << t.power_W
        << ",\"sm_clock_MHz\":" << t.sm_clock_MHz
        << ",\"mem_clock_MHz\":" << t.mem_clock_MHz
        << ",\"util_gpu\":" << t.util_gpu
        << ",\"util_mem\":" << t.util_mem
        << ",\"pstate\":" << t.pstate
        << ",\"pcie_gen_cur\":" << t.pcie_gen_cur
        << ",\"pcie_gen_max\":" << t.pcie_gen_max
        << ",\"pcie_width\":" << t.pcie_width
        << ",\"vram_used_MB\":" << (t.vram_used / (1024.0*1024.0))
        << ",\"vram_total_MB\":" << (t.vram_total / (1024.0*1024.0))
        << ",\"eff_hps_per_W\":" << eff
        << "}}\n";
  G.log.flush();
}

// --------- Pretty formatting ----------
static std::string human(double v) {
  char buf[64];
  if (v >= 1e9)      std::snprintf(buf, sizeof(buf), "%.2f G", v/1e9);
  else if (v >= 1e6) std::snprintf(buf, sizeof(buf), "%.2f M", v/1e6);
  else if (v >= 1e3) std::snprintf(buf, sizeof(buf), "%.2f K", v/1e3);
  else               std::snprintf(buf, sizeof(buf), "%.0f", v);
  return std::string(buf);
}

static std::string fmt_seconds(double s) {
  int64_t t = (int64_t)s;
  int h = (int)(t/3600); t %= 3600;
  int m = (int)(t/60); int sec=(int)(t%60);
  char buf[32]; std::snprintf(buf,sizeof(buf),"%02d:%02d:%02d",h,m,sec);
  return buf;
}

static void draw_once() {
  if (G.paint_paused.load()) return;
  term_home();

  const auto now = std::chrono::steady_clock::now();
  const double uptime_s = std::chrono::duration<double>(now - G.t0).count();
  const uint64_t now_ms =
      (uint64_t)std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::system_clock::now().time_since_epoch()).count();

  Telemetry t = poll_nvml();

  double h1  = G.ring.hps_1s (now_ms);
  double h10 = G.ring.hps_10s(now_ms);
  double h60 = G.ring.hps_60s(now_ms);

  double p50 = G.win.p50();
  double p95 = G.win.p95();
  double p99 = G.win.p99();
  double last_ms = G.win.last();
  double eff = (t.power_W > 0.1) ? (h10 / t.power_W) : 0.0;

  // HEADER
  std::printf(C_BOLD " BLAKE3 Miner Monitor " C_RESET C_DIM " | Uptime %s | Tag: %s\n" C_RESET,
              fmt_seconds(uptime_s).c_str(), G.tag.c_str());
  term_clear_eol();

  // DEVICE ROW
  std::printf(C_CYAN " GPU " C_RESET "%s  " C_DIM "(CC %d.%d, %d SMs)" C_RESET "\n",
              G.gpu.name.c_str(), G.gpu.cc_major, G.gpu.cc_minor, G.gpu.sm_count);
  term_clear_eol();

  // HASH RATE ROW
  std::printf(C_BOLD " Hashrate  " C_RESET
              "1s: " C_GREEN "%s" C_RESET " H/s   "
              "10s: " C_GREEN "%s" C_RESET " H/s   "
              "60s: " C_GREEN "%s" C_RESET " H/s   "
              "Total: " C_BLUE "%s" C_RESET " hashes   Batch: " C_BLUE "%ld" C_RESET "\n",
              human(h1).c_str(), human(h10).c_str(), human(h60).c_str(),
              human((double)G.hashes_total).c_str(), (long)G.batch);
  term_clear_eol();

  // KERNEL ROW
  std::printf(C_BOLD " Kernel ms " C_RESET
              "last: " C_YELLOW "%.3f" C_RESET
              "   p50: " C_YELLOW "%.3f" C_RESET
              "   p95: " C_YELLOW "%.3f" C_RESET
              "   p99: " C_YELLOW "%.3f" C_RESET
              "   rounds: " C_BLUE "%ld" C_RESET
              "   found: " C_GREEN "%s" C_RESET "\n",
              last_ms, p50, p95, p99,
              (long)G.rounds,
              (G.last_found >= 0 ? "yes" : "no"));
  term_clear_eol();

  // GPU ROW
  std::printf(C_BOLD " GPU stats " C_RESET
              "T: " C_YELLOW "%d°C" C_RESET "  "
              "P: " C_YELLOW "%.1fW" C_RESET "  "
              "Eff: " C_GREEN "%.2f H/s/W" C_RESET "  "
              "Clk: " C_YELLOW "%d" C_RESET "/" C_YELLOW "%d" C_RESET " MHz  "
              "Util: " C_YELLOW "%d%%" C_RESET "/" C_YELLOW "%d%%" C_RESET "  "
              "Pstate: " C_YELLOW "P%d" C_RESET "  "
              "PCIe: Gen%d(x%d)\n",
              t.tempC, t.power_W, eff,
              t.sm_clock_MHz, t.mem_clock_MHz,
              t.util_gpu, t.util_mem, t.pstate, t.pcie_gen_cur, t.pcie_width);
  term_clear_eol();

  // VRAM ROW
  double usedGB = t.vram_used / (1024.0*1024.0*1024.0);
  double totGB  = (t.vram_total? t.vram_total : G.gpu.vram_total) / (1024.0*1024.0*1024.0);
  std::printf(C_BOLD " Memory    " C_RESET
              "VRAM: " C_YELLOW "%.2f" C_RESET " / " C_YELLOW "%.2f GB" C_RESET
              "   JSON: %s   Status: " C_DIM "%s" C_RESET "\n",
              usedGB, totGB, G.json_path.c_str(), (G.status.empty()? "-" : G.status.c_str()));
  term_clear_eol();

  // HINTS / WARNINGS
  bool warn_idle = (t.util_gpu >= 0 && t.util_gpu < 20);
  bool warn_p8   = (t.pstate >= 5 || t.sm_clock_MHz > 0 && t.sm_clock_MHz < 600);
  bool warn_pcie = (t.pcie_gen_cur > 0 && t.pcie_gen_cur < 4);
  std::printf(C_DIM " Notes: " C_RESET);
  if (warn_idle) std::printf(C_RED  "low util  " C_RESET);
  if (warn_p8)   std::printf(C_RED  "low clocks/P-state  " C_RESET);
  if (warn_pcie) std::printf(C_RED  "low PCIe gen  " C_RESET);
  std::printf("\n");

  // spacer to keep screen height stable
  std::printf(C_DIM " ─────────────────────────────────────────────────────────────────────────────\n" C_RESET);

  std::fflush(stdout);

  // log sample
  write_sample_json(t, now_ms, h1, h10, h60, p50, p95, p99, last_ms);

  // reset last_found flag so it blips only once
  G.last_found = -1;
}

// --------- UI thread ----------
static void ui_thread() {
  term_hide_cursor();
  term_clear_all();
  term_home();
  fill_cuda_props();

  for (;;) {
    if (!G.running.load()) break;
    draw_once();
    std::this_thread::sleep_for(std::chrono::milliseconds(G.refresh_ms));
  }

#ifndef MINER_TUI_NO_NVML
  if (G.nvml.ok) nvmlShutdown();
#endif
  term_show_cursor();
}

// --------- Public API ----------
extern "C" void miner_tui_start(int device_index, int refresh_ms, const char* json_path) {
  std::lock_guard<std::mutex> lk(G.m);
  if (G.running.load()) return;
  G.device_index = device_index;
  G.refresh_ms   = (refresh_ms <= 0 ? 500 : refresh_ms);
  G.json_path    = (json_path && *json_path) ? json_path : "miner_bench.ndjson";
  G.t0           = std::chrono::steady_clock::now();
  G.rounds = 0; G.hashes_total = 0; G.last_found = -1;
  G.status.clear();

  fill_cuda_props();
  nvml_init_if_needed();

  G.log.open(G.json_path, std::ios::out | std::ios::app);
  G.running.store(true);
  G.thr = std::thread(ui_thread);
}

extern "C" void miner_tui_stop(void) {
  {
    std::lock_guard<std::mutex> lk(G.m);
    if (!G.running.load()) return;
    G.running.store(false);
  }
  if (G.thr.joinable()) G.thr.join();
  if (G.log.is_open()) G.log.close();
}

extern "C" void miner_tui_set_batch(int64_t batch) {
  std::lock_guard<std::mutex> lk(G.m);
  G.batch = batch;
}

extern "C" void miner_tui_record_round_ms(double ms_round) {
  const uint64_t now_ms =
      (uint64_t)std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
  std::lock_guard<std::mutex> lk(G.m);
  G.rounds += 1;
  G.win.push(ms_round);
  if (G.batch > 0) {
    G.hashes_total += G.batch;
    G.ring.add(now_ms, G.batch);
  }
}

extern "C" void miner_tui_mark_found(int found_idx) {
  std::lock_guard<std::mutex> lk(G.m);
  G.last_found = found_idx;
}

extern "C" void miner_tui_set_status(const char* s) {
  std::lock_guard<std::mutex> lk(G.m);
  G.status = (s ? s : "");
}

extern "C" void miner_tui_set_tag(const char* s) {
  std::lock_guard<std::mutex> lk(G.m);
  G.tag = (s ? s : "");
}

extern "C" void miner_tui_pause_paint(int pause_on) {
  G.paint_paused.store(!!pause_on);
}
