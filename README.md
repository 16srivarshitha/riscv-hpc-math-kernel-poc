# riscv-hpc-optimization-sandbox

This repository serves as a **Proof-of-Concept (PoC) framework** for exploring automated kernel optimization on the RISC-V architecture. 

The goal of this project is to provide a structured, reproducible environment to identify, optimize, and verify high-performance computational kernels (BLAS, FFT, Stencil) required for porting HPC and scientific applications to RISC-V.

## Core Philosophy
Instead of attempting a manual, code-by-code port of the target application list, this project focuses on **kernel-level acceleration**. By optimizing foundational math primitives—consistent with the standards in *Numerical Recipes: The Art of Scientific Computing*—we create a "performance layer" that can be linked into multiple applications simultaneously.

## Methodology
This project implements a 4-stage optimization pipeline:

1.  **Workload Classification:** A dependency analysis tool (`tools/analyzer.py`) scans codebases to map application dependencies to common HPC library primitives (e.g., CBLAS, Eigen, FFTW).
2.  **Kernel Decomposition:** Identified hotspots are decomposed into fundamental operations (GEMM, Sparse Mat-Vec, Vector-Add) and mapped to RISC-V Vector (RVV) ISA requirements.
3.  **Vectorized Implementation:** Implementation of target kernels using RVV intrinsics, prioritizing **FP64 (Double Precision)** for scientific accuracy.
4.  **Verification & Benchmarking:** A build system (`Makefile`) and harness (`tools/benchmark.py`) compare naive implementations against optimized kernels to calculate performance speedup and numerical divergence.

## Key Features
*   **`hal/dispatch.h`:** A runtime dispatch shim that allows the library to select the optimal kernel path (RVV-optimized vs. Scalar-fallback) based on hardware capabilities.
*   **Python Harness:** A `ctypes`-based benchmarking tool that allows for rapid performance evaluation without requiring a full re-build of target applications.
*   **Scalability:** A modular structure that allows for the addition of new kernel primitives (FFT, Stencil, etc.) as the project requirements grow.

## Repository Structure
```text
/riscv-hpc-optimization-sandbox
├── Makefile                # Build system for shared object library (.so)
├── hal/                    # Hardware abstraction and runtime dispatch
├── kernels/                # RVV-optimized kernel implementations (FP64)
├── tools/                  # Analysis and benchmarking utilities
├── docs/                   # Kernel taxonomy and methodology notes
└── tests/                  # Accuracy and precision validation suite
```

## Current Status
This repository is currently in the **prototyping phase**. It demonstrates the infrastructure for:
*   Building a drop-in shared library (`.so`).
*   Executing runtime dispatch between vectorized and scalar paths.
*   Validating kernel precision against baseline implementations.

## A Note on QEMU Benchmarks

RVV intrinsic kernels show lower GFLOPS under QEMU emulation than scalar 
baselines due to per-instruction emulation overhead. This is expected behavior.
The speedup ratio is meaningful only on physical RVV-capable hardware 
(e.g., SiFive P670, Sophgo CV1800B).

QEMU results validate **correctness**, not performance.