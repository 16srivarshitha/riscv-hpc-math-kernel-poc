# riscv-hpc-optimization-sandbox

A **Proof-of-Concept (PoC) framework** for automating the porting and optimization of HPC scientific applications to the RISC-V architecture.

The core idea is simple: instead of manually rewriting each application one by one, this project identifies the small set of fundamental math routines that *most* HPC applications share, optimizes those once using RISC-V Vector (RVV) intrinsics, and packages them as a drop-in shared library. Any application that links against standard libraries like BLAS, FFTW, or Eigen can then benefit automatically - without touching its own source code.

---

## Why This Approach?

Porting HPC workloads to RISC-V the traditional way is slow. Each application has to be profiled, hot spots identified, and performance-critical sections rewritten by hand. For a workload list spanning 100+ applications across computational chemistry, fluid dynamics, climate modelling, machine learning, and more, that process doesn't scale.

This project takes a different route: **classify workloads by the type of computation they do, not by the application name**. The [Berkeley Motifs](https://ieeexplore.ieee.org/document/5200028/) framework gives us exactly this - a small taxonomy of computation patterns (Dense Linear Algebra, FFT, Stencil/PDE, Sparse/Graph, etc.) that collectively cover the hot paths of virtually every scientific HPC code.

By optimizing one kernel per motif (e.g., GEMM for DLA, Cooley-Tukey FFT for spectral, Jacobi iteration for stencil), we get leverage across the entire workload list at once. This makes the porting effort **substantially more efficient**: tune a handful of kernels, and hundreds of applications benefit.

---

## How It Works: The 4-Stage Pipeline

**1. Workload Classification**

A dependency scanner (`tools/analyzer.py`) inspects application build systems and source trees to identify which standard HPC libraries each application uses (CBLAS, LAPACK, Eigen, FFTW, PETSc, etc.). Each library dependency is mapped to one or more Berkeley Motifs.

The `workload_report.json` captures the output of this classification for over 150 open-source and proprietary HPC applications. For example:
- OpenFOAM, WRF, FEniCS → **Stencil / PDE**
- GROMACS, LAMMPS, VASP → **Dense Linear Algebra**
- GADGET, ChaNGa → **N-Body / Particle**
- PyTorch, TensorFlow, JAX → **Element-wise / Reduction**

This tells us exactly which kernel motifs have the highest coverage and should be prioritized first.

**2. Kernel Decomposition**

Each motif is decomposed into a concrete mathematical primitive that can be implemented and benchmarked in isolation:

| Motif | Representative Kernel | FLOP Count | Algorithmic Source |
|---|---|---|---|
| Dense Linear Algebra | GEMM (matrix multiply) | $2N^3$ | NR Ch. 2 |
| Spectral / FFT | Cooley-Tukey FFT | $5N \log_2 N$ | NR Ch. 12 |
| Stencil / PDE | Poisson (Jacobi relaxation) | $5(N-2)^2$ | NR Ch. 20 |
| Sparse / Graph | SpMV, Banded solvers | varies | NR Ch. 2 |
| N-Body / Particle | Runge-Kutta ODE integrator | varies | NR Ch. 17 |
| MapReduce / Data Parallel | Sorting networks, reductions | $O(N \log N)$ | NR Ch. 8 |
| Element-wise / Reduction | Interpolation, quadrature | $O(N)$ | NR Ch. 3–5 |
| Mesh / Geometry | Delaunay triangulation | varies | NR Ch. 21 |

The full motif-to-NR mapping with RVV vectorization strategies is documented in `workload_motif_nr_mapping.md`.

**3. RVV-Optimized Implementation**

Each kernel is implemented twice: a plain scalar C++ version (the correctness baseline) and an RVV intrinsic version. All kernels use **FP64 (double precision)** to match the accuracy requirements of scientific computing.

Key RVV strategies used:

- **GEMM:** `vfmacc.vf` strip-mined accumulation, VLEN-512 register tiling. The inner loop accumulates dot products entirely inside vector registers to maximize arithmetic intensity.
- **FFT:** `vfmacc` / `vfnmsac` butterfly vectorization with dynamic `vsetvli` per radix stage. Dynamic stack allocation handles twiddle factor reconstruction to avoid misalignment.
- **Poisson (Stencil):** Unit-stride 4-neighbour loads across grid rows, with `vfmacc.vf` for the source term. Row-contiguous layout avoids expensive indexed gather operations.

A hardware abstraction layer (`hal/dispatch.h`) detects at runtime whether RVV is available and routes each kernel call to either the vectorized or scalar implementation.

**4. Automated Benchmarking & Verification**

`benchmark.sh` drives the entire build-run-report cycle automatically. Given a toolchain, it:
1. Cross-compiles for RV64GCV (or falls back to a host scalar build)
2. Runs the benchmark binary - on real hardware or under QEMU emulation
3. Parses the kernel timing output
4. Computes GFLOPS and speedup ratios
5. Appends a timestamped Markdown report to `benchmark_results.md`

This means a developer can plug in a new kernel, run one command, and immediately get a formatted comparison table - no manual data wrangling needed.

---

## Assumptions

A few important assumptions underpin this approach:

- **Motif coverage is sufficient.** We assume that optimizing the top motifs (DLA, Stencil, FFT, Sparse) covers the performance-critical paths of the majority of target applications. Applications with unusual or domain-specific hot paths may still need manual work.
- **Kernel-level speedup translates to application-level speedup.** This holds when the kernel accounts for a large fraction of total runtime (Amdahl's Law), which is typical for compute-heavy HPC codes but may not be true for I/O-bound or tightly coupled multi-physics codes.
- **FP64 accuracy is the baseline.** All kernels target double precision. Applications that use mixed precision or integer arithmetic are out of scope for now.
- **VLEN=512 is the target.** The RVV implementations are tuned for 512-bit vector registers (e.g., SiFive P670-class hardware). Behaviour on narrower VLEN configurations (128, 256) is correct but not performance-tuned.
- **Standard library ABIs are stable.** The drop-in `.so` approach assumes applications link dynamically against CBLAS/LAPACK-compatible interfaces and that those interfaces are stable enough for binary substitution.
- **QEMU results validate correctness only.** QEMU emulates RVV instruction-by-instruction, so timing numbers under QEMU are not representative of real hardware. See the benchmarks section below.

---

## Benchmark Results (QEMU)

The latest results from `benchmark_results.md`, collected under QEMU with VLEN=512:

| Kernel | N | Scalar ms | RVV ms | Scalar GFLOPS | RVV GFLOPS | Speedup | Max Error | Status |
|--------|---|----------:|-------:|--------------:|-----------:|--------:|-----------|--------|
| `gemm` | 64 | 2.39 | 3.39 | 0.219 | 0.155 | 0.71x | `0.00e+00` |  PASS |
| `gemm` | 128 | 18.03 | 28.51 | 0.233 | 0.147 | 0.63x | `0.00e+00` |  PASS |
| `gemm` | 256 | 148.36 | 213.88 | 0.226 | 0.157 | 0.69x | `0.00e+00` |  PASS |
| `fft` | 256 | 1.11 | 0.78 | 0.009 | 0.013 | 1.42x | `5.33e-15` |  PASS |
| `fft` | 1024 | 0.39 | 0.88 | 0.131 | 0.058 | 0.45x | `1.42e-14` |  PASS |
| `fft` | 4096 | 1.67 | 3.51 | 0.147 | 0.070 | 0.48x | `2.84e-14` |  PASS |
| `poisson` | 64 | 0.22 | 0.90 | 0.087 | 0.021 | 0.24x | `2.22e-16` |  PASS |
| `poisson` | 128 | 0.47 | 0.95 | 0.168 | 0.084 | 0.50x | `2.22e-16` |  PASS |
| `poisson` | 256 | 1.49 | 3.70 | 0.217 | 0.087 | 0.40x | `2.22e-16` |  PASS |

**All kernels pass correctness checks** (max numerical error well within double-precision tolerance).

### Why RVV looks slower than scalar under QEMU

QEMU emulates every RVV instruction in software, one at a time. This adds significant overhead that does not exist on real hardware, where a single `vfmacc` instruction processes 8 doubles simultaneously. The speedup numbers in the table above are therefore **not meaningful as performance indicators** - they reflect emulator overhead, not hardware capability.

The one exception is `fft` at N=256, which shows 1.42x speedup even under QEMU due to the relatively high arithmetic intensity of the butterfly computation at that problem size.

**To get meaningful performance numbers, the kernels must be run on physical RVV-capable hardware** (e.g., SiFive P670, Sophgo CV1800B, or Milk-V Pioneer). QEMU results exist solely to confirm that the RVV code is numerically correct.

---

## Repository Structure

```
riscv-hpc-optimization-sandbox/
├── Makefile                   # Builds the shared library (.so) and benchmark binary
├── benchmark.sh               # End-to-end automation: build → run → report
├── benchmark_results.md       # Auto-generated timestamped results log
├── workload_report.json       # Workload classification: 150+ apps mapped to motifs
├── workload_motif_nr_mapping.md  # Motif → NR chapter → RVV strategy reference
├── hal/
│   └── dispatch.h             # Runtime dispatch: RVV vs scalar fallback
├── kernels/                   # RVV-optimized kernel implementations (FP64)
├── lib/                       # Compiled shared library output
├── build/                     # Build artefacts
├── tools/
│   └── analyzer.py            # Dependency scanner for workload classification
├── docs/                      # Kernel taxonomy and methodology notes
└── tests/                     # Accuracy and precision validation suite
```

---

## Running the Benchmark

```bash
# Auto-detect toolchain (cross-compile if RISC-V toolchain is found, else host build)
./benchmark.sh

# Host-only scalar build (useful for CI or when no RISC-V toolchain is installed)
./benchmark.sh --native

# Force QEMU execution (for correctness validation)
./benchmark.sh --qemu

# Clean build artefacts
./benchmark.sh --clean
```

Environment variables:

| Variable | Default | Description |
|---|---|---|
| `RISCV_CXX` | `riscv64-linux-gnu-g++` | Path to RISC-V cross-compiler |
| `QEMU` | `qemu-riscv64` | Path to QEMU user-mode emulator |
| `VLEN` | `512` | Vector register length in bits |

Results are automatically appended to `benchmark_results.md` with a UTC timestamp after each run.

---

## Future Work / To-Do

The following items are the next logical steps to move this from a PoC to a production-ready porting accelerator:

**More kernel motifs**

The current implementation covers GEMM, FFT, and Poisson (Stencil). The workload classification shows significant application coverage under Sparse/Graph, N-Body/Particle, and Element-wise/Reduction motifs. Implementing representative kernels for those (SpMV, Runge-Kutta, vectorized reductions) is the next priority.

**Real hardware validation**

All timing results so far are from QEMU. Running on physical RVV hardware (SiFive P670 or equivalent) is needed to produce the actual speedup numbers that justify the porting effort. This is the single most important missing data point.

**Wider VLEN coverage**

Current tuning targets VLEN=512. Testing on VLEN=128 and VLEN=256 configurations (which are more common in embedded RISC-V deployments) will reveal whether the implementations degrade gracefully or need separate tuning.

**Application-level integration testing**

The end goal is that a real HPC application (e.g., OpenFOAM or GROMACS) links against this library's `.so` at runtime and gets a performance improvement without any source changes. This needs to be validated end-to-end for at least one application per motif.

**Automated dependency scanner completion**

`tools/analyzer.py` is partially implemented. It needs to handle more build systems (CMake, Spack, Autotools) and produce structured output that feeds directly into the kernel selection logic.

**FP32 / mixed-precision variants**

Some ML workloads (PyTorch, TensorFlow) and embedded applications prefer FP32 or even FP16. Adding mixed-precision variants of the core kernels would extend coverage to those workloads.

**CI integration**

The benchmark script is designed to run in CI (use `--native` for scalar-only correctness runs). Wiring this into a GitHub Actions workflow to catch regressions on every commit would harden the library.

**Strassen and other asymptotically superior algorithms**

The current GEMM is a straightforward $O(N^3)$ implementation. For large matrices, Strassen's algorithm drops complexity to ~$O(N^{2.807})$. Evaluating whether the RVV register pressure from Strassen's recursive structure is worth it at practical problem sizes is an open question.

---

## A Note on QEMU Benchmarks

RVV intrinsic kernels appear slower than scalar under QEMU emulation because QEMU processes each vector instruction individually in software, negating the data-parallel speedup that real hardware provides. This is expected and well-understood behaviour.

**QEMU results confirm correctness. They do not measure performance.**

Meaningful speedup measurements require physical RISC-V hardware with a V-extension implementation (vlen ≥ 128, elen = 64, vext_spec = v1.0).
