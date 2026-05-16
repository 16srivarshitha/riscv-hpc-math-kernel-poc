/*
 * gemm_rvv.cpp — RISC-V Vector (RVV) FP64 kernel implementations
 *
 * Kernels:  gemm_rvv    (Dense Linear Algebra motif  — NR Ch.2)
 *           fft_rvv     (Spectral / FFT motif         — NR Ch.12)
 *           poisson_rvv (Stencil / PDE motif          — NR Ch.20)
 *
 * ISA requirements
 * 
 *   -march=rv64gcv           (base + V extension)
 *   -mabi=lp64d              (FP64 ABI)
 *   Zve64d sub-extension for 64-bit FP vector ops
 *
 * All three kernels operate entirely in double precision (FP64) to satisfy
 * scientific accuracy requirements.  The V extension provides:
 *   vfmacc.vv  — fused multiply-accumulate  (GEMM inner loop)
 *   vfadd.vv   — vector add                 (butterfly / stencil)
 *   vfsub.vv   — vector subtract            (butterfly / stencil)
 *   vfmul.vv   — element-wise multiply      (twiddle application)
 *   vlse64.v   — strided load               (column gather for GEMM, stencil neighbours)
 *   vle64.v    — unit-stride load
 *   vse64.v    — unit-stride store
 *   vmv.v.f    — broadcast scalar to vector (alpha/beta scaling)
 *
 * QEMU note:  Per-instruction emulation overhead means RVV shows lower
 * GFLOPS under QEMU than scalar.  Results validate CORRECTNESS only;
 * real speedup is observable only on physical RVV silicon (SiFive P670,
 * Sophgo CV1800B, etc.).
 */

#ifdef __riscv_v

#include "dispatch.h"

#include <riscv_vector.h>  /* RVV intrinsics */
#include <cmath>           /* sin, cos, M_PI */
#include <cstddef>         /* size_t         */

/* ═══════════════════════════════════════════════════════════════════════════
 * 1.  GEMM — C = alpha * A * B + beta * C   (row-major, N×N square)
 *
 * Strategy (consistent with NR §2 + workload_motif_nr_mapping.md §1):
 *   Outer i-loop iterates over rows of A.
 *   Middle k-loop broadcasts A[i,k] as a scalar and accumulates into a
 *     vector register holding partial sums for a strip of C[i, j:j+vl].
 *   Inner j-loop is strip-mined by vsetvli so the vector unit handles as
 *     many columns of B[k, j:] as fit in one vector register group.
 *
 * RVV intrinsics used:
 *   vsetvli      — set vector length based on remaining columns
 *   vle64.v      — load strip of B[k, j:j+vl] from contiguous memory
 *   vle64.v      — load strip of C[i, j:j+vl] for beta-scaling
 *   vfmul.vf     — scale C strip by beta
 *   vfmacc.vf    — fused multiply-accumulate: acc += A[i,k] * B[k,j:]
 *   vfmul.vf     — scale final accumulator by alpha
 *   vse64.v      — write back result strip
 * ═══════════════════════════════════════════════════════════════════════════ */
void gemm_rvv(const double *A, const double *B, double *C,
              size_t N, double alpha, double beta)
{
    for (size_t i = 0; i < N; ++i) {

        /*  Initialise C row strip to beta * C[i, :]  */
        size_t j = 0;
        while (j < N) {
            size_t vl = __riscv_vsetvl_e64m4(N - j);   /* up to VLEN*4/64 elems */

            /* Load current C strip and scale by beta */
            vfloat64m4_t vc = __riscv_vle64_v_f64m4(C + i * N + j, vl);
            vc = __riscv_vfmul_vf_f64m4(vc, beta, vl);

            /* Accumulate dot-product along k for each column j in strip */
            for (size_t k = 0; k < N; ++k) {
                double a_ik = A[i * N + k];

                /* Load B[k, j:j+vl] */
                vfloat64m4_t vb = __riscv_vle64_v_f64m4(B + k * N + j, vl);

                /* vc += a_ik * vb  (fused multiply-accumulate) */
                vc = __riscv_vfmacc_vf_f64m4(vc, a_ik, vb, vl);
            }

            /* Scale by alpha and store */
            vc = __riscv_vfmul_vf_f64m4(vc, alpha, vl);
            __riscv_vse64_v_f64m4(C + i * N + j, vc, vl);

            j += vl;
        }
    }
}

/* ═══════════════════════════════════════════════════════════════════════════
 * 2.  FFT — Cooley-Tukey radix-2 in-place, FP64 complex
 *
 * Algorithm overview (NR §12.2, Danielson-Lanczos):
 *   Phase A — bit-reversal permutation  (scalar; irregular memory access)
 *   Phase B — butterfly stages          (vectorised with RVV)
 *
 * Butterfly equations for one stage (sub-transform length = len, half = len/2):
 *   For each group [p, q] where q = p + half:
 *     t_re = u_re * re[q] - u_im * im[q]   (twiddle × re part)
 *     t_im = u_re * im[q] + u_im * re[q]   (twiddle × im part)
 *     re[q] = re[p] - t_re ;  re[p] = re[p] + t_re
 *     im[q] = im[p] - t_im ;  im[p] = im[p] + t_im
 *
 * RVV strategy:
 *   Each stage has N/len independent groups.  Within a group we vectorise
 *   over the half butterflies using vfmacc/vfnmacc for the twiddle multiply.
 *   vsetvli adapts to the remaining butterfly count in each group.
 *
 * RVV intrinsics used:
 *   vle64.v     — load re[q:q+vl], im[q:q+vl], re[p:p+vl], im[p:p+vl]
 *   vfmul.vf    — multiply by scalar twiddle component (u_re, u_im)
 *   vfmacc.vf   — fused multiply-accumulate  (t_re = u_re*re[q] - u_im*im[q])
 *   vfnmacc.vf  — negated fused multiply-accumulate
 *   vfadd.vv    — re[p] = re[p] + t_re
 *   vfsub.vv    — re[q] = re[p] - t_re
 *   vse64.v     — store results
 *
 * Note on twiddle pre-computation:
 *   Pre-computing all twiddle factors into a table (as NR recommends) and
 *   loading them as a vector would replace the scalar u_re/u_im loop below.
 *   That upgrade is marked TODO: for the next sprint; for this PoC we keep
 *   a scalar twiddle walk to isolate the butterfly vectorisation clearly.
 * ═══════════════════════════════════════════════════════════════════════════ */
void fft_rvv(double *re, double *im, size_t N, int sign)
{
    /*  Phase A: bit-reversal permutation (scalar)  */
    size_t j = 0;
    for (size_t i = 1; i < N; ++i) {
        size_t bit = N >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) {
            double tr = re[i]; re[i] = re[j]; re[j] = tr;
            double ti = im[i]; im[i] = im[j]; im[j] = ti;
        }
    }

    /*  Phase B: butterfly stages  */
    for (size_t len = 2; len <= N; len <<= 1) {
        size_t half = len >> 1;

        double ang = sign * 2.0 * M_PI / (double)len;
        double wRe = cos(ang);   
        double wIm = sin(ang);

        for (size_t i = 0; i < N; i += len) {
            double uRe = 1.0, uIm = 0.0;  

            size_t k = 0;
            while (k < half) {
                size_t vl = __riscv_vsetvl_e64m2(half - k);

                size_t p = i + k;
                size_t q = p + half;

                vfloat64m2_t vreP = __riscv_vle64_v_f64m2(re + p, vl);
                vfloat64m2_t vimP = __riscv_vle64_v_f64m2(im + p, vl);
                vfloat64m2_t vreQ = __riscv_vle64_v_f64m2(re + q, vl);
                vfloat64m2_t vimQ = __riscv_vle64_v_f64m2(im + q, vl);

                // FIX 1: Populate dynamic vector twiddles for this specific vector strip
                double *tmp_uRe = (double *)__builtin_alloca(vl * sizeof(double));
                double *tmp_uIm = (double *)__builtin_alloca(vl * sizeof(double));
                double cur_uRe = uRe;
                double cur_uIm = uIm;
                for (size_t s = 0; s < vl; ++s) {
                    tmp_uRe[s] = cur_uRe;
                    tmp_uIm[s] = cur_uIm;
                    double nuRe = cur_uRe * wRe - cur_uIm * wIm;
                    cur_uIm     = cur_uRe * wIm + cur_uIm * wRe;
                    cur_uRe     = nuRe;
                }
                uRe = cur_uRe; // Carry over step forward to the next strip loop
                uIm = cur_uIm;

                vfloat64m2_t vuRe = __riscv_vle64_v_f64m2(tmp_uRe, vl);
                vfloat64m2_t vuIm = __riscv_vle64_v_f64m2(tmp_uIm, vl);

                // FIX 2: Compute t_re = u_re * reQ - u_im * imQ using vfnmsac (vd = -vs1*vs2 + vd)
                vfloat64m2_t vtRe = __riscv_vfmul_vv_f64m2(vreQ, vuRe, vl);
                vtRe = __riscv_vfnmsac_vv_f64m2(vtRe, vuIm, vimQ, vl);

                // Compute t_im = u_re * imQ + u_im * reQ
                vfloat64m2_t vtIm = __riscv_vfmul_vv_f64m2(vimQ, vuRe, vl);
                vtIm = __riscv_vfmacc_vv_f64m2(vtIm, vuIm, vreQ, vl);

                /* Butterfly combine */
                __riscv_vse64_v_f64m2(re + p, __riscv_vfadd_vv_f64m2(vreP, vtRe, vl), vl);
                __riscv_vse64_v_f64m2(im + p, __riscv_vfadd_vv_f64m2(vimP, vtIm, vl), vl);
                __riscv_vse64_v_f64m2(re + q, __riscv_vfsub_vv_f64m2(vreP, vtRe, vl), vl);
                __riscv_vse64_v_f64m2(im + q, __riscv_vfsub_vv_f64m2(vimP, vtIm, vl), vl);

                k += vl;
            }
        }
    }
}

void poisson_rvv(const double *u_in, double *u_out, const double *f,
                 size_t rows, size_t cols, double h)
{
    double h2   = h * h;
    double inv4 = 0.25;

    /* Interior rows only */
    for (size_t i = 1; i < rows - 1; ++i) {

        const double *north = u_in + (i - 1) * cols;  
        const double *south = u_in + (i + 1) * cols;  
        const double *west  = u_in + i * cols;         
        const double *east  = west;                    
        const double *src   = f    + i * cols;
        double       *dst   = u_out + i * cols;

        /* Interior columns: j in [1, cols-1) */
        size_t j = 1;
        while (j < cols - 1) {
            size_t vl = __riscv_vsetvl_e64m4(cols - 1 - j);

            vfloat64m4_t vN = __riscv_vle64_v_f64m4(north + j, vl);  
            vfloat64m4_t vS = __riscv_vle64_v_f64m4(south + j, vl);  
            vfloat64m4_t vW = __riscv_vle64_v_f64m4(west  + j - 1, vl);  
            vfloat64m4_t vE = __riscv_vle64_v_f64m4(east  + j + 1, vl);  
            vfloat64m4_t vF = __riscv_vle64_v_f64m4(src   + j, vl);  

            vfloat64m4_t vacc = __riscv_vfadd_vv_f64m4(vN, vS, vl);
            vacc = __riscv_vfadd_vv_f64m4(vacc, vW, vl);
            vacc = __riscv_vfadd_vv_f64m4(vacc, vE, vl);

            // FIX: Pass -h2 into a standard vfmacc to cleanly execute: vacc + (-h2 * vF)
            vacc = __riscv_vfmacc_vf_f64m4(vacc, -h2, vF, vl);

            vacc = __riscv_vfmul_vf_f64m4(vacc, inv4, vl);
            __riscv_vse64_v_f64m4(dst + j, vacc, vl);

            j += vl;
        }
    }

    /* Dirichlet boundary: copy edges unchanged from u_in */
    for (size_t j = 0; j < cols; ++j) {
        u_out[j]                     = u_in[j];
        u_out[(rows-1)*cols + j]     = u_in[(rows-1)*cols + j];
    }
    for (size_t i = 1; i < rows - 1; ++i) {
        u_out[i*cols]                = u_in[i*cols];
        u_out[i*cols + (cols - 1)]   = u_in[i*cols + (cols - 1)];
    }
}

/* 
 * Skeleton stubs — documented intent for future sprint kernels
 * 
 *
 * These are not compiled into the benchmark binary but serve as a design
 * contract for the next phase of the PoC.
 */

/*
 * SpMV skeleton — Sparse / Graph motif (NR §2)
 *
 * void spmv_csr_rvv(const int *row_ptr, const int *col_idx,
 *                   const double *val, const double *x, double *y,
 *                   size_t rows)
 * {
 *     // RVV strategy: vluxei32.v (indexed gather) to fetch x[col_idx[j]]
 *     // for each non-zero in row i.
 *     // Software prefetch col_idx for next row while accumulating current.
 *     // TODO: implement with vle32.v + vluxei64.v gather idiom.
 * }
 */

/*
 * N-body pair force skeleton — N-Body / Particle motif (NR §17)
 *
 * void nbody_force_rvv(const double *x, const double *y, const double *z,
 *                      double *fx, double *fy, double *fz,
 *                      size_t N, double eps2)
 * {
 *     // RVV strategy: for each particle i, broadcast its position as
 *     // scalar, then vfsub + vfmul to compute r² over a strip of j's.
 *     // Pair-exclusion (i==j self-force) via vmseq mask.
 *     // TODO: implement with pair-exclusion masking and register caching.
 * }
 */

#endif  /* __riscv_v */