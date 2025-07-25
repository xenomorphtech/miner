# Miner Project

## Overview

This repository contains a CUDA-based miner that integrates the BLAKE3 hashing algorithm. The code is organized to keep third-party libraries isolated under `third_party/`, with all build artifacts placed in `build/`.

## Prerequisites

Ensure you have the following installed:

- **CUDA Toolkit** (compatible with your GPU)
- **NVIDIA NVCC** (comes with the CUDA Toolkit)
- **GCC** (for compiling the Makefile steps)
- **Make** (to drive the build via the provided Makefile)
- **Curl** (for fetching missing headers in the setup script)

## Directory Structure

```text
miner/
├── src/                   # Your CUDA source (solver.cu)
├── third_party/           # Vendor code for BLAKE3
│   └── blake3/c/          # Core BLAKE3 .c and header files
├── build/                 # All compiled object files (.o)
├── Makefile               # Top-level build script
└── README.md              # This documentation
```

## Setup

Before the first build (or if you ever remove the BLAKE3 headers), you need to fetch two header files (`blake3.h` and `blake3_impl.h`) into `third_party/blake3/c/` so that the Makefile can compile the C sources correctly.

You have two options:

1. **Run the helper script** (must be created in `scripts/setup_blake3_headers.sh`):
   ```bash
   bash scripts/setup_blake3_headers.sh
   ```
2. **Or run the commands manually**:
   ```bash
   mkdir -p third_party/blake3/c
   curl -fsSL https://raw.githubusercontent.com/BLAKE3-team/BLAKE3/master/c/blake3.h       -o third_party/blake3/c/blake3.h
   curl -fsSL https://raw.githubusercontent.com/BLAKE3-team/BLAKE3/master/c/blake3_impl.h  -o third_party/blake3/c/blake3_impl.h
   ```

*(If you choose the script option, create ****\`\`**** lines.)*

## Building the Binary

Simply run:

```bash
make
```

This will:

1. Compile all six core BLAKE3 C sources into object files under `build/` using GCC.
2. Compile your CUDA code (`src/solver.cu`) into `build/solver.o` using NVCC.
3. Link everything into the final `solver` binary in the project root.

## Cleaning Up

To remove all build artifacts and the binary:

```bash
make clean
```

This will delete the `build/` directory and the `solver` executable.

## Running the Miner

Once built, execute:

```bash
./solver [args]
```

Replace `[args]` with any command-line options as needed. To list all available options and view usage details, run:

```bash
./solver --help
```

---

*Generated on July 25, 2025*

