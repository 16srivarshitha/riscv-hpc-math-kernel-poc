/*
 * main.cpp — Verification runner & benchmarking harness
 *
 * For each kernel (GEMM, FFT, Poisson):
 *   1. Run the scalar reference implementation.
 *   2. Run the RVV implementation (or scalar fallback if no V support).
 *   3. Verify correctness: compute max absolute error; fail if > threshold.
 *   4. Time both paths and emit a CSV row.
 *
 * Output format (stdout):
 *   kernel,impl,N,elapsed_ms,max_err
 *
 * The benchmark.sh script consumes this CSV and renders a Markdown table.
 */

#include "dispatch.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <ctime>

/* 
 * Helpers
 *  */

static double wall_ms()
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec * 1e-6;
}

static double max_abs_err(const double *a, const double *b, size_t n)
{
    double err = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double d = fabs(a[i] - b[i]);
        if (d > err) err = d;
    }
    return err;
}

static void fill_rand(double *buf, size_t n, double lo, double hi)
{
    double range = hi - lo;
    for (size_t i = 0; i < n; ++i)
        buf[i] = lo + range * ((double)rand() / RAND_MAX);
}

/* 
 * GEMM verification + timing
 *  */
static int bench_gemm(size_t N)
{
    size_t sz = N * N;
    double *A   = (double*)malloc(sz * sizeof(double));
    double *B   = (double*)malloc(sz * sizeof(double));
    double *Cs  = (double*)calloc(sz, sizeof(double));  /* scalar result */
    double *Cr  = (double*)calloc(sz, sizeof(double));  /* rvv result    */

    if (!A || !B || !Cs || !Cr) { fputs("OOM\n", stderr); return 1; }

    srand(42);
    fill_rand(A, sz, -1.0, 1.0);
    fill_rand(B, sz, -1.0, 1.0);

    /*  Scalar  */
    double t0 = wall_ms();
    gemm_scalar(A, B, Cs, N, 1.0, 0.0);
    double scalar_ms = wall_ms() - t0;

    /*  RVV (or fallback)  */
    t0 = wall_ms();
    hal_gemm(A, B, Cr, N, 1.0, 0.0);
    double rvv_ms = wall_ms() - t0;

    double err = max_abs_err(Cs, Cr, sz);
    const char *impl = hal_has_rvv() ? "rvv" : "scalar_fallback";

    printf("gemm,scalar,%zu,%.3f,0.0\n", N, scalar_ms);
    printf("gemm,%s,%zu,%.3f,%.2e\n", impl, N, rvv_ms, err);

    int ok = (err < 1e-9);
    if (!ok) fprintf(stderr, "[FAIL] gemm max_err=%.3e (threshold 1e-9)\n", err);

    free(A); free(B); free(Cs); free(Cr);
    return ok ? 0 : 1;
}

/* 
 * FFT verification + timing
 *
 * Ground truth: run scalar forward FFT, then scalar inverse FFT.
 * Verify that forward→inverse round-trip recovers original signal.
 * Compare scalar forward against RVV forward.
 *  */
static int bench_fft(size_t N)
{
    double *re_orig = (double*)malloc(N * sizeof(double));
    double *im_orig = (double*)calloc(N, sizeof(double));
    double *re_s    = (double*)malloc(N * sizeof(double));
    double *im_s    = (double*)calloc(N, sizeof(double));
    double *re_r    = (double*)malloc(N * sizeof(double));
    double *im_r    = (double*)calloc(N, sizeof(double));

    if (!re_orig || !im_orig || !re_s || !im_s || !re_r || !im_r) {
        fputs("OOM\n", stderr); return 1;
    }

    srand(7);
    fill_rand(re_orig, N, -1.0, 1.0);

    /* Copy to scalar and rvv buffers */
    memcpy(re_s, re_orig, N * sizeof(double));
    memcpy(re_r, re_orig, N * sizeof(double));

    /*  Scalar forward FFT  */
    double t0 = wall_ms();
    fft_scalar(re_s, im_s, N, -1);
    double scalar_ms = wall_ms() - t0;

    /*  RVV forward FFT  */
    t0 = wall_ms();
    hal_fft(re_r, im_r, N, -1);
    double rvv_ms = wall_ms() - t0;

    /* Compare real and imaginary parts */
    double err_re = max_abs_err(re_s, re_r, N);
    double err_im = max_abs_err(im_s, im_r, N);
    double err    = (err_re > err_im) ? err_re : err_im;

    /* Round-trip check on scalar (self-consistency) */
    double *re_rt = (double*)malloc(N * sizeof(double));
    double *im_rt = (double*)calloc(N, sizeof(double));
    memcpy(re_rt, re_s, N * sizeof(double));
    memcpy(im_rt, im_s, N * sizeof(double));
    fft_scalar(re_rt, im_rt, N, +1);
    for (size_t i = 0; i < N; ++i) { re_rt[i] /= N; im_rt[i] /= N; }
    double rt_err = max_abs_err(re_orig, re_rt, N);

    const char *impl = hal_has_rvv() ? "rvv" : "scalar_fallback";
    printf("fft,scalar,%zu,%.3f,0.0\n", N, scalar_ms);
    printf("fft,%s,%zu,%.3f,%.2e\n", impl, N, rvv_ms, err);
    printf("# fft round-trip error (scalar self-check): %.2e\n", rt_err);

    int ok = (err < 1e-10) && (rt_err < 1e-10);
    if (!ok) fprintf(stderr, "[FAIL] fft err=%.3e rt_err=%.3e\n", err, rt_err);

    free(re_orig); free(im_orig);
    free(re_s); free(im_s);
    free(re_r); free(im_r);
    free(re_rt); free(im_rt);
    return ok ? 0 : 1;
}

/* 
 * Poisson verification + timing
 *
 * Use a manufactured solution: u_exact = sin(πx)·sin(πy)
 * on the unit square.  Source term: f = −2π²·u_exact.
 * Run one Jacobi sweep with scalar and RVV, compare outputs.
 * Also report max error vs. exact for the scalar sweep (sanity check).
 *  */
static int bench_poisson(size_t rows, size_t cols)
{
    size_t n = rows * cols;
    double *u_in    = (double*)calloc(n, sizeof(double));
    double *f       = (double*)malloc(n * sizeof(double));
    double *u_s     = (double*)calloc(n, sizeof(double));  /* scalar out */
    double *u_r     = (double*)calloc(n, sizeof(double));  /* rvv out    */

    if (!u_in || !f || !u_s || !u_r) { fputs("OOM\n", stderr); return 1; }

    double h  = 1.0 / (double)(rows - 1);
    // double h2 = h * h; // used via poisson_scalar call

    /* Fill manufactured solution and source term */
    for (size_t i = 0; i < rows; ++i) {
        double y = i * h;
        for (size_t j = 0; j < cols; ++j) {
            double x = j * h;
            double u_ex = sin(M_PI * x) * sin(M_PI * y);
            u_in[i * cols + j] = u_ex;        /* initialise from exact solution */
            f   [i * cols + j] = -2.0 * M_PI * M_PI * u_ex;  /* ∇²u = f */
        }
    }

    /*  Scalar sweep  */
    double t0 = wall_ms();
    poisson_scalar(u_in, u_s, f, rows, cols, h);
    double scalar_ms = wall_ms() - t0;

    /*  RVV sweep  */
    t0 = wall_ms();
    hal_poisson(u_in, u_r, f, rows, cols, h);
    double rvv_ms = wall_ms() - t0;

    double err = max_abs_err(u_s, u_r, n);

    const char *impl = hal_has_rvv() ? "rvv" : "scalar_fallback";
    printf("poisson,scalar,%zu,%.3f,0.0\n", rows, scalar_ms);
    printf("poisson,%s,%zu,%.3f,%.2e\n", impl, rows, rvv_ms, err);

    int ok = (err < 1e-14);
    if (!ok) fprintf(stderr, "[FAIL] poisson max_err=%.3e (threshold 1e-14)\n", err);

    free(u_in); free(f); free(u_s); free(u_r);
    return ok ? 0 : 1;
}

/* 
 * Entry point
 *  */
int main(int argc, char **argv)
{
    (void)argc; (void)argv;

    printf("# RISC-V HPC Kernel Benchmark\n");
    printf("# RVV available: %s\n", hal_has_rvv() ? "YES" : "NO (scalar fallback)");
    printf("#\n");
    printf("kernel,impl,N,elapsed_ms,max_err\n");

    int rc = 0;

    /* GEMM: 64×64, 128×128, 256×256 */
    rc |= bench_gemm(64);
    rc |= bench_gemm(128);
    rc |= bench_gemm(256);

    /* FFT: N = 256, 1024, 4096 (must be power of 2) */
    rc |= bench_fft(256);
    rc |= bench_fft(1024);
    rc |= bench_fft(4096);

    /* Poisson: 64×64, 128×128, 256×256 */
    rc |= bench_poisson(64,  64);
    rc |= bench_poisson(128, 128);
    rc |= bench_poisson(256, 256);

    printf("#\n");
    printf("# Exit status: %s\n", rc == 0 ? "PASS" : "FAIL");
    return rc;
}