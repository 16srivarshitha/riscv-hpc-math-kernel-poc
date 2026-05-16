#pragma once
/*
 * dispatch.h — Hardware Abstraction Layer (HAL) for RISC-V HPC Kernel Library
 *
 * Provides a single unified API for GEMM, FFT, and Poisson kernels.
 * At runtime, queries the hardware for RVV support and routes to the
 * appropriate implementation (RVV-optimised or scalar fallback).
 *
 * "Transitional Library" pattern: legacy HPC callers link against this
 * header and never need to know which code path executed.
 *
 * Supported kernels
 * -----------------
 *  GEMM   : Dense matrix multiply  (Dense Linear Algebra motif)
 *  FFT    : Cooley-Tukey radix-2   (Spectral / FFT motif)
 *  Poisson: 5-point 2-D stencil    (Stencil / PDE motif)
 *
 * All kernels operate in FP64 (double precision) to meet scientific
 * accuracy requirements consistent with Numerical Recipes Ch. 2/12/20.
 */

#include <stddef.h>   /* size_t */
#include <stdbool.h>  /* bool   */

#ifdef __cplusplus
extern "C" {
#endif

/* 
 * Runtime capability detection
 *  */

/**
 * hal_has_rvv() — returns true if the executing hart supports the V extension.
 *
 * Detection strategy:
 *   1. On Linux: parse /proc/cpuinfo for "v" in the isa field.
 *   2. Compile-time guard: if __riscv_v is defined the toolchain already
 *      confirmed RVV support; we still do the runtime check so that a
 *      cross-compiled binary can fall back cleanly on a non-V core.
 */
bool hal_has_rvv(void);

/* 
 * Kernel function pointer typedefs
 *  */

/* GEMM: C = alpha*A*B + beta*C  (row-major, all matrices N×N for PoC) */
typedef void (*gemm_fn_t)(
    const double *A,   /* [N×N] left operand         */
    const double *B,   /* [N×N] right operand        */
    double       *C,   /* [N×N] accumulator (in/out) */
    size_t        N,   /* matrix dimension           */
    double        alpha,
    double        beta
);

/* FFT: in-place complex DFT of length N (N must be a power of 2).
 * re[] and im[] are the real and imaginary parts of the signal. */
typedef void (*fft_fn_t)(
    double *re,   /* real part  [N]  (in/out) */
    double *im,   /* imag part  [N]  (in/out) */
    size_t  N,    /* transform length, must be power of 2 */
    int     sign  /* -1 = forward DFT,  +1 = inverse DFT  */
);

/* Poisson: one Jacobi relaxation sweep on a (rows×cols) grid.
 * Solves ∇²u = f using the 5-point Laplacian stencil.
 * u_out[] <- one iteration from u_in[] with source term f[]. */
typedef void (*poisson_fn_t)(
    const double *u_in,   /* current solution  [rows×cols] */
    double       *u_out,  /* next iteration    [rows×cols] */
    const double *f,      /* RHS source term   [rows×cols] */
    size_t        rows,
    size_t        cols,
    double        h       /* uniform grid spacing          */
);

/* 
 * Scalar (fallback) implementations — always available
 *  */

void gemm_scalar    (const double *A, const double *B, double *C,
                     size_t N, double alpha, double beta);

void fft_scalar     (double *re, double *im, size_t N, int sign);

void poisson_scalar (const double *u_in, double *u_out, const double *f,
                     size_t rows, size_t cols, double h);

/* 
 * RVV-optimised implementations — compiled only when __riscv_v is defined
 *  */

#ifdef __riscv_v
void gemm_rvv    (const double *A, const double *B, double *C,
                  size_t N, double alpha, double beta);

void fft_rvv     (double *re, double *im, size_t N, int sign);

void poisson_rvv (const double *u_in, double *u_out, const double *f,
                  size_t rows, size_t cols, double h);
#endif  /* __riscv_v */

/* 
 * Unified HAL entry-points  (these are what application code calls)
 *  */

/**
 * hal_gemm — dispatch to RVV or scalar GEMM.
 * Equivalent to BLAS dgemm for square matrices (PoC scope).
 */
void hal_gemm (const double *A, const double *B, double *C,
               size_t N, double alpha, double beta);

/**
 * hal_fft — dispatch to RVV or scalar FFT.
 * Danielson-Lanczos Cooley-Tukey radix-2 in-place.
 * N must be a power of 2.  sign=-1 → forward, sign=+1 → inverse.
 */
void hal_fft  (double *re, double *im, size_t N, int sign);

/**
 * hal_poisson — dispatch to RVV or scalar 2-D Poisson relaxation sweep.
 * Caller is responsible for iterating until convergence.
 */
void hal_poisson (const double *u_in, double *u_out, const double *f,
                  size_t rows, size_t cols, double h);

#ifdef __cplusplus
}  /* extern "C" */
#endif