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

# ---------- file lists ----------
SRC_CUDA        := src/matmul_gpu.cu
SRC_CPP_COMMON  := src/ref_cpu.cpp
SRC_CPP_SMOKE   := src/test_smoke.cpp
SRC_CPP_XOF     := src/test_xof_compare.cpp
SRC_C           := blake3/c/blake3.c blake3/c/blake3_dispatch.c blake3/c/blake3_portable.c

OBJ_CUDA        := $(SRC_CUDA:.cu=.o)
OBJ_CPP_COMMON  := $(SRC_CPP_COMMON:.cpp=.o)
OBJ_CPP_SMOKE   := $(SRC_CPP_SMOKE:.cpp=.o)
OBJ_CPP_XOF     := $(SRC_CPP_XOF:.cpp=.o)
OBJ_C           := $(SRC_C:.c=.o)

OBJS_SMOKE      := $(OBJ_CUDA) $(OBJ_CPP_COMMON) $(OBJ_CPP_SMOKE) $(OBJ_C)
OBJS_XOF        := $(OBJ_CUDA) $(OBJ_CPP_COMMON) $(OBJ_CPP_XOF)   $(OBJ_C)

TARGETS := smoke xof_test

.PHONY: all
all: $(TARGETS)

smoke: $(OBJS_SMOKE)
	$(NVCC) -arch=$(CUDA_ARCH) -O3 -std=c++17 -rdc=true $^ $(LDFLAGS) -o $@

xof_test: $(OBJS_XOF)
	$(NVCC) -arch=$(CUDA_ARCH) -O3 -std=c++17 -rdc=true $^ $(LDFLAGS) -o $@

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

%.o: %.cpp
	$(CXX)  $(CXXFLAGS) -c $< -o $@

%.o: %.c
	$(CC)   $(CFLAGS)   -c $< -o $@

.PHONY: clean
clean:
	rm -f $(OBJS_SMOKE) $(OBJS_XOF) $(TARGETS)
