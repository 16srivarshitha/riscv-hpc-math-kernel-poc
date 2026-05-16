/*
 * gemm_scalar.cpp — Scalar (reference) implementations for all three kernels
 *
 * These are the ground-truth baselines used for:
 *   1. Correctness verification  (main.cpp compares RVV output against these)
 *   2. Fallback on non-V hardware
 *   3. QEMU benchmarking where RVV intrinsics are slower due to emulation overhead
 *
 * Numerical Recipes references:
 *   GEMM    — Ch. 2  (LU / matrix operations)
 *   FFT     — Ch. 12 (Danielson-Lanczos Cooley-Tukey)
 *   Poisson — Ch. 20 (Jacobi relaxation, 5-point Laplacian)
 */

#include "dispatch.h"

#include <cmath>    /* sin, cos, M_PI */
#include <cstring>  /* memcpy        */

/* 
 * 1. GEMM — C = alpha * A * B + beta * C  (row-major, N×N square)
 *
 * Triple-nested loop; inner accumulation uses a register variable to help
 * the compiler avoid repeated memory traffic on C[i*N+j].
 *  */
void gemm_scalar(const double *A, const double *B, double *C,
                 size_t N, double alpha, double beta)
{
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            double acc = 0.0;
            for (size_t k = 0; k < N; ++k) {
                acc += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = alpha * acc + beta * C[i * N + j];
        }
    }
}

/* 
 * 2. FFT — Cooley-Tukey radix-2, Danielson-Lanczos, in-place
 *
 * Algorithm (NR §12.2):
 *   a. Bit-reversal permutation.
 *   b. Butterfly stages: log2(N) passes, each doubling the sub-transform
 *      length and multiplying by the twiddle factor W = e^{±2πi/m}.
 *
 * sign = -1  →  forward DFT   (X[k] = Σ x[n] e^{-2πikn/N})
 * sign = +1  →  inverse DFT   (caller must divide by N afterwards)
 *  */
void fft_scalar(double *re, double *im, size_t N, int sign)
{
    /*  bit-reversal  */
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

    /*  butterfly stages  */
    for (size_t len = 2; len <= N; len <<= 1) {
        /* twiddle angle for this stage */
        double ang  = sign * 2.0 * M_PI / (double)len;
        double wRe  = cos(ang);
        double wIm  = sin(ang);

        for (size_t i = 0; i < N; i += len) {
            double uRe = 1.0, uIm = 0.0;  /* running twiddle factor */
            size_t half = len >> 1;
            for (size_t k = 0; k < half; ++k) {
                size_t p = i + k;
                size_t q = p + half;

                /* twiddle × lower half */
                double tRe = uRe * re[q] - uIm * im[q];
                double tIm = uRe * im[q] + uIm * re[q];

                re[q] = re[p] - tRe;
                im[q] = im[p] - tIm;
                re[p] = re[p] + tRe;
                im[p] = im[p] + tIm;

                /* advance twiddle: u *= w */
                double nuRe = uRe * wRe - uIm * wIm;
                uIm         = uRe * wIm + uIm * wRe;
                uRe         = nuRe;
            }
        }
    }
}

/* 
 * 3. Poisson — one Jacobi relaxation sweep (5-point 2-D stencil)
 *
 * Discretisation of ∇²u = f on a uniform grid with spacing h:
 *
 *   u_new[i,j] = (u[i-1,j] + u[i+1,j] + u[i,j-1] + u[i,j+1] - h²f[i,j]) / 4
 *
 * Boundary: Dirichlet — boundary nodes are copied unchanged from u_in.
 * Caller iterates this sweep until ||u_new − u|| < tolerance.
 *
 * Algorithm reference: NR §20.5 (relaxation methods for elliptic PDEs).
 *  */
void poisson_scalar(const double *u_in, double *u_out, const double *f,
                    size_t rows, size_t cols, double h)
{
    double h2 = h * h;
    double inv4 = 0.25;

    /* Interior nodes only; boundaries remain from u_in. */
    for (size_t i = 1; i < rows - 1; ++i) {
        for (size_t j = 1; j < cols - 1; ++j) {
            size_t idx = i * cols + j;
            u_out[idx] = inv4 * (
                u_in[(i - 1) * cols + j] +
                u_in[(i + 1) * cols + j] +
                u_in[i * cols + (j - 1)] +
                u_in[i * cols + (j + 1)] -
                h2 * f[idx]
            );
        }
    }

    /* Copy boundary rows/cols unchanged (Dirichlet BC). */
    for (size_t j = 0; j < cols; ++j) {
        u_out[j]                       = u_in[j];                        /* top */
        u_out[(rows - 1) * cols + j]   = u_in[(rows - 1) * cols + j];  /* bottom */
    }
    for (size_t i = 1; i < rows - 1; ++i) {
        u_out[i * cols]                = u_in[i * cols];                /* left */
        u_out[i * cols + (cols - 1)]   = u_in[i * cols + (cols - 1)];  /* right */
    }
}