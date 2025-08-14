# ==================== Config ====================
CUDA_ARCH   := sm_89

NVCC        := nvcc
CUDA_PATH   ?= /usr/local/cuda

# --- NVCC / CXX flags
NVCCFLAGS   := -arch=$(CUDA_ARCH) -O3 -std=c++17 -rdc=true \
               -Iinclude -Iblake3 -Iblake3/c -Isrc

CXX         := g++
CXXFLAGS    := -O3 -std=c++17 \
               -Iinclude -Iblake3 -Isrc -I$(CUDA_PATH)/include \
               -DBLAKE3_NO_SSE2 -DBLAKE3_NO_SSE41 -DBLAKE3_NO_AVX512

CC          := gcc
CFLAGS      := -O3 -std=c99 \
               -Iblake3/c \
               -DBLAKE3_NO_SSE2 -DBLAKE3_NO_SSE41 \
               -DBLAKE3_NO_AVX2 -DBLAKE3_NO_AVX512

# --- Link
LDFLAGS     := -L$(CUDA_PATH)/lib64 -lcudart -lcudadevrt

# ==================== NVML autodetect ====================
# If nvml.h is found in a common location, link -lnvidia-ml; else compile without NVML.
NVML_HDR := $(firstword \
  $(wildcard $(CUDA_PATH)/include/nvml.h) \
  $(wildcard /usr/include/nvidia-ml/nvml.h) \
  $(wildcard /usr/include/nvml.h))

ifeq ($(NVML_HDR),)
  $(info [miner] NVML headers not found -> building miner_tui without NVML metrics)
  CXXFLAGS += -DMINER_TUI_NO_NVML
else
  $(info [miner] NVML headers found at $(NVML_HDR))
  LDFLAGS  += -lnvidia-ml
endif

# ==================== Files ====================
SRC_CUDA    := src/matmul_gpu.cu
SRC_CPP     := src/test_smoke.cpp src/ref_cpu.cpp src/test_xof_compare.cpp \
               src/miner_tui.cpp
SRC_C       := blake3/c/blake3.c blake3/c/blake3_dispatch.c blake3/c/blake3_portable.c

OBJ_CUDA    := $(SRC_CUDA:.cu=.o)
OBJ_CPP     := $(SRC_CPP:.cpp=.o)
OBJ_C       := $(SRC_C:.c=.o)

TARGETS     := smoke xof_test

# ==================== Default ====================
.PHONY: all
all: $(TARGETS)

# ==================== Link ====================
# Include miner_tui.o only where you need the live dashboard (smoke).
smoke: src/test_smoke.o src/ref_cpu.o src/miner_tui.o $(OBJ_CUDA) $(OBJ_C)
	$(NVCC) -arch=$(CUDA_ARCH) -O3 -std=c++17 -rdc=true $^ $(LDFLAGS) -o $@

# xof_test does not need the TUI; exclude miner_tui.o here.
xof_test: src/test_xof_compare.o src/miner_tui.o $(OBJ_CUDA) $(OBJ_C)
	$(NVCC) -arch=$(CUDA_ARCH) -O3 -std=c++17 -rdc=true $^ $(LDFLAGS) -o $@

# ==================== Compile rules ====================
%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

%.o: %.cpp
	$(CXX)  $(CXXFLAGS) -c $< -o $@

%.o: %.c
	$(CC)   $(CFLAGS)   -c $< -o $@

# ==================== Helpers ====================
.PHONY: clean print-nvml
clean:
	rm -f $(OBJ_CUDA) $(OBJ_CPP) $(OBJ_C) $(TARGETS)

print-nvml:
	@echo "NVML_HDR = $(NVML_HDR)"
	@echo "LDFLAGS  = $(LDFLAGS)"
