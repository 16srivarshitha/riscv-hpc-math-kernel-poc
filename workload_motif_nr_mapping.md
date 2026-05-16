# Workload Optimization Matrix: Berkeley Motifs to Numerical Recipes

## Tier 1: Matrix Index

| Kernel Motif | NR Location | Core Algorithmic Scope | Primary RVV Vectorization Focus |
| :--- | :--- | :--- | :--- |
| **Dense Linear Algebra (DLA)** | Ch 2, Ch 11 | Matrix factorization, inversions, and eigensystems | Fused Multiply-Accumulate (`vfmacc`), cache tiling |
| **Stencil / PDE** | Ch 20 | Finite-difference solvers and relaxation sweeps | Unit-stride sweeps, boundary condition masking |
| **Spectral / FFT** | Ch 12, Ch 13 | Discrete Fourier transforms and frequency domain filters | Strided loads (`vlse32`), radix-8 loop strip-mining |
| **Sparse / Graph** | Ch 2 | Banded solvers, iterative methods, CSR/CSC structures | Indexed gather-scatter (`vluxei32`), prefetching |
| **N-Body / Particle** | Ch 17 | ODE integrators and particle state progression | Pair-exclusion masking, register state caching |
| **MapReduce / Data Parallel** | Ch 8 | Sorting networks, selection, and index layout tracking | Branch elimination via masked vector comparisons |
| **Element-wise / Reduction** | Ch 3, Ch 4, Ch 5 | Interpolation, quadratures, and coordinate functions | Kernel fusion, vectorized reductions (`vredsum`) |
| **Mesh / Geometry** | Ch 21 | Graph-based spatial layouts and boundary tracing | Vectorized stack tracking, multi-coordinate loads |
| **General / Infrastructure** | Ch 7, Ch 22 | Hashing, pseudo-random bitstreams, utility pipelines | Bitmanip (`Zbb`) and performance counters (`Zicsr`) |

---

## Tier 2: Optimization Profiles

### 1. Dense Linear Algebra (DLA)
> **Hardware Strategy:** Structure loops into cache-blocked sub-matrices. Utilize the vector register file as an active accumulation tile via `vfmacc.vv` to achieve peak floating-point throughput.

*   **LU vs. Gauss-Jordan Decomposition:** While Gauss-Jordan computes a full matrix inversion, it requires all right-hand sides to be known upfront. LU decomposition is structurally preferred for general linear systems; its inner loop requires approximately $N^3 / 3$ executions, optimizing raw clock cycles by a factor of 3 over naive inversion methods.
*   **Cholesky Factorization:** Restricted to symmetric and positive-definite matrices ($A = L \cdot L^T$). It requires only $N^3 / 6$ inner loop executions—cutting the computational footprint of LU decomposition in half while maintaining extreme numerical stability without tracking pivot arrays.
*   **Singular Value Decomposition (SVD):** The optimal fallback for near-singular or degenerate systems. It constructs a Moore-Penrose pseudoinverse, allowing vector pipes to isolate zero or near-zero singular values and resolve minimum-residual solutions.
*   **QR Updating:** Bypasses expensive $O(N^3)$ complete recalculations when a matrix undergoes a rank-one alteration (such as adding an outer product). Modifying the factorization can be streamed in $O(N^2)$ operations.
*   **Strassen’s Fast Matrix Multiplication:** Minimizes raw scalar multiplications to drop the asymptotic complexity of matrix scaling and inversions from $O(N^3)$ down to $O(N^{\log_2 7}) \approx O(N^{2.807})$.

### 2. Stencil / PDE
> **Hardware Strategy:** Maximize spatial cache line reuse. Since these operations are strictly memory-bandwidth bound, map contiguous grid rows to unit-stride vector sweeps and use vector mask registers to handle boundary conditions.

*   **Cyclic Tridiagonal Pipelines:** Finite differencing configurations applied to partial differential equations with periodic boundary conditions are mathematically reduced to cyclic tridiagonal linear systems, allowing localized array operations.
*   **Multigrid & Relaxation Methods:** Prioritizes multigrid strategies to rapidly eliminate low-frequency error components across varied grid hierarchies, alongside localized relaxation updates.
*   **Spectral vs. Pseudospectral Sweeps:** Transitions localized spatial differences into global algebraic calculations by using orthogonal polynomial transforms to optimize evaluation accuracy.

### 3. Spectral / FFT
> **Hardware Strategy:** Address the power-of-2 non-unit memory strides that surface during butterfly stages. Dynamically alter the hardware vector length configuration (`vsetvli`) to match changing butterfly radix steps.

*   **Danielson-Lanczos Reductions:** Translates the $O(N^2)$ Discrete Fourier Transform (DFT) into a highly parallelizable $O(N \log_2 N)$ recursive tree. Uses bit-reversal indexing to reorder input arrays cleanly before executing combination steps.
*   **Real Data Packing Optimizations:** Eliminates wasted memory channels when processing purely real arrays by packing two distinct real functions into a single complex array's real and imaginary fields, or splitting a single function into alternating even/odd arrays to halve runtime.
*   **Zero-Padded Convolutions:** Frequency-domain filtering via multiplying transforms requires strict zero-padding configurations to insulate the target dataset from cyclic wraparound processing artifacts.
*   **Power Spectrum Windowing:** Minimizes spectral leakage at finite data boundaries by applying windowing algorithms (e.g., Bartlett, Hann, Welch, or Slepian multitaper functions) over data sequences, reducing variance through heavily overlapped segment averaging.

### 4. Sparse / Graph
> **Hardware Strategy:** Mitigate indirect addressing latency and random pointer chasing. Leverage hardware vector-indexed gather-scatter operations (`vluxei32.v` / `vsuxei32.v`) and issue software prefetches for index pointers.

*   **Banded Direct Solvers:** Simplifies localized sparse matrices, like tridiagonal configurations, down to $O(N)$ operations by using direct partitioning and elimination recursions instead of general solvers.
*   **Rank-One Corrections:** Uses Sherman-Morrison and Woodbury formulas to calculate updated matrix inverses in just $3N^2$ steps when a sparse framework changes by a single row, column, or block correction, dodging $O(N^3)$ rebuilds.
*   **Compressed Storage Management:** Employs explicit sparse layouts (such as Compressed Column Storage / Harwell-Boeing / CSC format) using three tightly packed 1D vectors (`val`, `row_ind`, `col_ptr`) to keep null values completely out of vector execution pipelines.
*   **Iterative Matrix-Vector Solvers:** Relies on the Preconditioned Biconjugate Gradient (PBCG) method for massive scale systems. The operation reduces down to consecutive matrix-vector products, accelerated by choosing an optimal preconditioning matrix.

### 5. N-Body / Particle
> **Hardware Strategy:** Keep inter-particle states inside the vector register file across consecutive steps. Isolate pair-exclusion rules (e.g., self-interaction checks) into vector mask registers to keep pipeline execution completely linear.

*   **Adaptive Step Runge-Kutta:** Employs fifth-order Dormand-Prince embedded formulas to compute local truncation errors dynamically, optimizing step sizes automatically for target accuracy constraints.
*   **Bulirsch-Stoer Extrapolation:** Tailored for highly smooth trajectories requiring strict accuracy tolerances. It takes large analytical steps and applies Richardson's deferred approach to the limit, leveraging rational or polynomial extrapolation to drive step errors to zero.
*   **Stoermer’s Conservative Integration:** Specifically targets second-order systems lacking explicit first derivatives ($y'' = f(x, y)$), halving total force evaluation calls compared to generic first-order ODE breakdowns.

### 6. MapReduce / Data Parallel
> **Hardware Strategy:** Eliminate deep pipeline flushes caused by unpredictable conditional branches. Use masked vector comparison operators (`vmslt.vv`, `vmseq.vv`) and population counts (`vpopc.m`) to implement sorting and selection networks.

*   **Quicksort Partitioning:** A partition-exchange engine operating at $O(N \log_2 N)$ average complexity. It recursively divides subarrays around a calculated median value, though it requires specific pivot protections to avoid degrading to $O(N^2)$ on pre-sorted sequences.
*   **In-Place Heapsort:** Guarantees a strict $O(N \log_2 N)$ execution bound without requiring auxiliary allocation arrays. It shapes data structures into an active binary heap, executing parent-child element shifts in $O(\log_2 N)$ cycles per step.
*   **Targeted Subtraction Selection:** Avoids sorting a whole dataset when hunting down the $M$-th largest entry. It uses localized in-place partitioning operating in linear $O(N)$ time, or uses live heaps to process streaming data in $O(M \log M)$ cycles.
*   **Index-Pointer Layout Sorting:** Prevents expensive data record shuffles in memory by sorting a lightweight array of pointer indices ($I_j$), leaving the underlying structured database layout completely untouched.

### 7. Element-wise / Reduction
> **Hardware Strategy:** Fuse consecutive operations (e.g., executing an activation directly inside a normalization pass) to keep intermediate values in vector registers, bypassing the main memory bus completely.

*   **Continuous Interpolation Networks:** Leverages Neville's algorithm to recursively evaluate polynomial interpolations. Employs cubic splines to generate simple tridiagonal arrays that map to linear $O(N)$ vector passes, while using barycentric rational equations to clear poles cleanly.
*   **Romberg Quadrature Acceleration:** Combines extended trapezoidal and midpoint numerical integration steps with polynomial extrapolation routines to systematically eliminate higher-order truncation errors, boosting convergence rates.
*   **Custom Gaussian Quadratures:** Resolves intricate integrals by matching specific weight functions to specialized orthogonal polynomials (Legendre, Laguerre, Hermite, Jacobi). Employs the Stieltjes procedure to numerically generate custom weights for singular functions.

### 8. Mesh / Geometry
> **Hardware Strategy:** Optimize vertex and coordinate storage using structural layout packing. Map multi-coordinate spatial tracking vectors to vector segment loads to stream geometric data fields smoothly.

*   **Unstructured Spatial Graphing:** Focuses heavily on managing spatial datasets using dynamic indexing patterns, explicitly relying on Delaunay Triangulation routines to build stable, geometric mesh frameworks.
*   **Hierarchical Space Partitioning:** Organizes unstructured spatial coordinates into deterministic multi-dimensional structures, utilizing KD-Trees, quad-trees, and octrees to accelerate range searching and neighbor identification.
*   **Intersection Geometry:** Minimizes geometric collision processing pipelines by vectorizing polygon and sphere intersection calculations, stripping away scalar conditional paths.

### 9. General / Infrastructure
> **Hardware Strategy:** Rely directly on scalar performance extensions. Leverage standard Control and Status Registers (`Zicsr`) for real-time profiling, and tap into Bitmanip (`Zbb`) or Crypto units to handle bit streams.

*   **Multi-Algorithmic Random Bitstreams:** Avoids mathematical bias by layering independent pseudorandom engines (e.g., Linear Congruential Generators blended with Xorshift shifts) to build massive $2^{64}$ sequence horizons that pass spectral verification.
*   **Poisson Distribution Hashing:** Models table distribution slots through Poisson statistical maps, setting up high-speed associative array lookups that convert arbitrary keys into direct integer index lanes without matching full array data footprints.
*   **Data Footprint Compacting:** Implements bit-level data compression architectures, leveraging variable-length Huffman coding loops, arithmetic coding logic, and cyclic redundancy checksum (CRC) error detection loops.
