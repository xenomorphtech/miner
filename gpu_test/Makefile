# ---------- toolchain & options ----------
CUDA_ARCH   := sm_89

NVCC        := nvcc
NVCCFLAGS   := -arch=$(CUDA_ARCH) -O3 -std=c++17 -rdc=true -Iinclude -Iblake3

CXX         := g++
CUDA_PATH   ?= /usr/local/cuda
CXXFLAGS    := -O3 -std=c++17 -Iinclude -Iblake3 -I$(CUDA_PATH)/include

CC          := gcc
CFLAGS      := -O3 -std=c99 \
               -Iblake3/c \
               -DBLAKE3_NO_SSE2 -DBLAKE3_NO_SSE41 \
               -DBLAKE3_NO_AVX2 -DBLAKE3_NO_AVX512

LDFLAGS     := -L/usr/local/cuda/lib64 -lcudart -lcudadevrt

# ---------- file lists ----------
SRC_CUDA    := src/matmul_gpu.cu
SRC_CPP     := src/test_smoke.cpp src/ref_cpu.cpp
SRC_C       := blake3/c/blake3.c blake3/c/blake3_dispatch.c blake3/c/blake3_portable.c

OBJ_CUDA    := $(SRC_CUDA:.cu=.o)
OBJ_CPP     := $(SRC_CPP:.cpp=.o)
OBJ_C       := $(SRC_C:.c=.o)

OBJS        := $(OBJ_CUDA) $(OBJ_CPP) $(OBJ_C)

TARGET      := smoke

# ---------- default target ----------
.PHONY: all
all: $(TARGET)

# ---------- link ----------
$(TARGET): $(OBJS)
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
	rm -f $(OBJS) $(TARGET)
