CUDA_ARCH   := sm_89

NVCC        := nvcc
NVCCFLAGS   := -arch=$(CUDA_ARCH) -O3 -std=c++17 -rdc=true \
               -Iinclude -Iblake3 -Iblake3/c -DDEBUG_TRACE

CXX         := g++
CUDA_PATH   ?= /usr/local/cuda
CXXFLAGS    := -O3 -std=c++17 \
               -Iinclude -Iblake3 -I$(CUDA_PATH)/include \
               -DDEBUG_CPU -DDEBUG_TRACE \
               -DBLAKE3_NO_SSE2 -DBLAKE3_NO_SSE41 -DBLAKE3_NO_AVX512

CC          := gcc
CFLAGS      := -O3 -std=c99 \
               -Iblake3/c \
               -DBLAKE3_NO_SSE2 -DBLAKE3_NO_SSE41 \
               -DBLAKE3_NO_AVX2 -DBLAKE3_NO_AVX512 \
               -DDEBUG_CPU -DDEBUG_TRACE

LDFLAGS     := -L/usr/local/cuda/lib64 -lcudart -lcudadevrt

# ---------- files ----------
SRC_CUDA    := src/matmul_gpu.cu
SRC_CPP     := src/test_smoke.cpp src/ref_cpu.cpp src/test_xof_compare.cpp
SRC_C       := blake3/c/blake3.c blake3/c/blake3_dispatch.c blake3/c/blake3_portable.c

OBJ_CUDA    := $(SRC_CUDA:.cu=.o)
OBJ_CPP     := $(SRC_CPP:.cpp=.o)
OBJ_C       := $(SRC_C:.c=.o)

TARGETS     := smoke xof_test

# ---------- default target ----------
.PHONY: all
all: $(TARGETS)

# ---------- link ----------
smoke: src/test_smoke.o src/ref_cpu.o $(OBJ_CUDA) $(OBJ_C)
	$(NVCC) -arch=$(CUDA_ARCH) -O3 -std=c++17 -rdc=true $^ $(LDFLAGS) -o $@

xof_test: src/test_xof_compare.o $(OBJ_CUDA) $(OBJ_C)
	$(NVCC) -arch=$(CUDA_ARCH) -O3 -std=c++17 -rdc=true $^ $(LDFLAGS) -o $@

# ---------- compile rules ----------
%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

%.o: %.cpp
	$(CXX)  $(CXXFLAGS) -c $< -o $@

%.o: %.c
	$(CC)   $(CFLAGS)   -c $< -o $@

# ---------- housekeeping ----------
.PHONY: clean
clean:
	rm -f $(OBJ_CUDA) $(OBJ_CPP) $(OBJ_C) $(TARGETS)
