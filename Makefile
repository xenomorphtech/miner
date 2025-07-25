# Makefile

# Compiler settings
CC      := gcc
NVCC    := nvcc
CFLAGS  := -O3 -march=znver4 -msse4.1 -mavx2 -DBLAKE3_NO_AVX512 \
           -Ithird_party/blake3/c
NVCCFLAGS := -O3 -Ithird_party/blake3/c
LDFLAGS := -Xcompiler="$(CFLAGS)"

# Source directories
B3_SRC_DIR := third_party/blake3/c
SRC_DIR    := src
BUILD_DIR  := build

# Gather all .c files from the BLAKE3 dir and map to .o in build/
B3_SRCS  := $(wildcard $(B3_SRC_DIR)/*.c)
B3_OBJS  := $(patsubst $(B3_SRC_DIR)/%.c, $(BUILD_DIR)/%.o, $(B3_SRCS))

# Your CUDA object
SOLVER_OBJ := $(BUILD_DIR)/solver.o

# Final binary
TARGET := solver

.PHONY: all clean

all: $(TARGET)

# Rule to build the CUDA translation unit
$(SOLVER_OBJ): $(SRC_DIR)/solver.cu | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -dc $< -o $@

# Rule to compile each BLAKE3 .c into .o
$(BUILD_DIR)/%.o: $(B3_SRC_DIR)/%.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

# Link step
$(TARGET): $(SOLVER_OBJ) $(B3_OBJS)
	$(NVCC) $(NVCCFLAGS) $^ $(LDFLAGS) -o $@

# Ensure build directory exists
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

clean:
	rm -rf $(BUILD_DIR) $(TARGET)
