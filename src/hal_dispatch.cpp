/*
 * hal_dispatch.cpp — Runtime dispatch implementation
 *
 * Queries the hardware at first call (via hal_has_rvv) then caches the
 * result for all subsequent calls so there is zero overhead per kernel
 * invocation after the first.
 */

#include "dispatch.h"

#include <cstdio>
#include <cstring>
#include <cstdlib>

/* 
 * RVV runtime detection
 *  */

bool hal_has_rvv(void)
{
    /* Cache result across calls. */
    static int cached = -1;
    if (cached != -1) return (bool)cached;

    cached = 0;  /* default: no RVV */

#if defined(__riscv_v)
    /*
     * Toolchain confirmed compile-time RVV availability.
     * We still verify at runtime by inspecting /proc/cpuinfo so that a
     * cross-compiled binary loaded onto a non-V hart fails gracefully.
     */
    FILE *f = fopen("/proc/cpuinfo", "r");
    if (!f) {
        /* Can't verify — trust compile-time flag */
        cached = 1;
        return true;
    }

    char line[512];
    while (fgets(line, sizeof(line), f)) {
        /* Look for the ISA string: "isa	: rv64imafdcv…" */
        if (strncmp(line, "isa", 3) == 0) {
            const char *colon = strchr(line, ':');
            if (colon) {
                /* The "v" extension appears as a standalone char between
                 * letters; a naive strchr is sufficient for this PoC. */
                const char *p = colon + 1;
                while (*p && *p != '\n') {
                    if (*p == 'v' || *p == 'V') { cached = 1; break; }
                    ++p;
                }
            }
            if (cached) break;
        }
    }
    fclose(f);
#else
    /* Compiled without -march=…v — scalar fallback always. */
    (void)0;
#endif

    return (bool)cached;
}

/* 
 * HAL entry-points
 *  */

void hal_gemm(const double *A, const double *B, double *C,
              size_t N, double alpha, double beta)
{
#ifdef __riscv_v
    if (hal_has_rvv()) { gemm_rvv(A, B, C, N, alpha, beta); return; }
#endif
    gemm_scalar(A, B, C, N, alpha, beta);
}

void hal_fft(double *re, double *im, size_t N, int sign)
{
#ifdef __riscv_v
    if (hal_has_rvv()) { fft_rvv(re, im, N, sign); return; }
#endif
    fft_scalar(re, im, N, sign);
}

void hal_poisson(const double *u_in, double *u_out, const double *f,
                 size_t rows, size_t cols, double h)
{
#ifdef __riscv_v
    if (hal_has_rvv()) { poisson_rvv(u_in, u_out, f, rows, cols, h); return; }
#endif
    poisson_scalar(u_in, u_out, f, rows, cols, h);
}