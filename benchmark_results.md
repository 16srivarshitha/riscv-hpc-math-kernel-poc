
## Benchmark Results — 2026-05-16 05:49 UTC

> **RVV status:** `rvv`
>   QEMU emulation: RVV timing is NOT representative.  Run on physical hardware for speedup.

| Kernel | N | Scalar ms | RVV ms | Speedup | Max Error | Status |
|--------|---|----------:|-------:|-------:|-----------|--------|
| `fft` | 1024 | 0.408 | 0.890 | 0.46x | `1.42e-14` |  PASS |
| `poisson` | 256 | 1.515 | 3.844 | 0.39x | `2.22e-16` |  PASS |
| `fft` | 256 | 1.382 | 0.804 | 1.72x | `5.33e-15` |  PASS |
| `poisson` | 128 | 0.496 | 1.008 | 0.49x | `2.22e-16` |  PASS |
| `poisson` | 64 | 0.218 | 0.893 | 0.24x | `2.22e-16` |  PASS |
| `gemm` | 256 | 154.542 | 219.978 | 0.70x | `0.00e+00` |  PASS |
| `gemm` | 128 | 18.626 | 27.318 | 0.68x | `0.00e+00` |  PASS |
| `gemm` | 64 | 2.418 | 3.494 | 0.69x | `0.00e+00` |  PASS |
| `fft` | 4096 | 1.764 | 3.618 | 0.49x | `2.84e-14` |  PASS |


## Benchmark Results — 2026-05-16 05:58 UTC

> **RVV status:** `rvv`
>   QEMU emulation: RVV timing is NOT representative.  Run on physical hardware for speedup.

| Kernel | N | Scalar ms | RVV ms | Speedup | Max Error | Status |
|--------|---|----------:|-------:|-------:|-----------|--------|
| `fft` | 1024 | 0.392 | 0.880 | 0.45x | `1.42e-14` |  PASS |
| `poisson` | 256 | 1.485 | 3.698 | 0.40x | `2.22e-16` |  PASS |
| `fft` | 256 | 1.108 | 0.782 | 1.42x | `5.33e-15` |  PASS |
| `poisson` | 128 | 0.474 | 0.947 | 0.50x | `2.22e-16` |  PASS |
| `poisson` | 64 | 0.220 | 0.900 | 0.24x | `2.22e-16` |  PASS |
| `gemm` | 256 | 148.359 | 213.876 | 0.69x | `0.00e+00` |  PASS |
| `gemm` | 128 | 18.028 | 28.505 | 0.63x | `0.00e+00` |  PASS |
| `gemm` | 64 | 2.394 | 3.389 | 0.71x | `0.00e+00` |  PASS |
| `fft` | 4096 | 1.673 | 3.506 | 0.48x | `2.84e-14` |  PASS |


## Benchmark Results — 2026-05-16 05:58 UTC

> **RVV status:** `rvv`
>   QEMU emulation: RVV timing is NOT representative.  Run on physical hardware for speedup.

| Kernel | N | Scalar ms | RVV ms | Scalar GFLOPS | RVV GFLOPS | Speedup | Max Error | Status |
|--------|---|----------:|-------:|--------------:|-----------:|-------:|-----------|--------|
| `fft` | 1024 | 0.392 | 0.880 | 0.1306 | 0.0582 | 0.45x | `1.42e-14` |  PASS |
| `poisson` | 256 | 1.485 | 3.698 | 0.2172 | 0.0872 | 0.40x | `2.22e-16` |  PASS |
| `fft` | 256 | 1.108 | 0.782 | 0.0092 | 0.0131 | 1.42x | `5.33e-15` |  PASS |
| `poisson` | 128 | 0.474 | 0.947 | 0.1675 | 0.0838 | 0.50x | `2.22e-16` |  PASS |
| `poisson` | 64 | 0.220 | 0.900 | 0.0874 | 0.0214 | 0.24x | `2.22e-16` |  PASS |
| `gemm` | 256 | 148.359 | 213.876 | 0.2262 | 0.1569 | 0.69x | `0.00e+00` |  PASS |
| `gemm` | 128 | 18.028 | 28.505 | 0.2327 | 0.1471 | 0.63x | `0.00e+00` |  PASS |
| `gemm` | 64 | 2.394 | 3.389 | 0.2190 | 0.1547 | 0.71x | `0.00e+00` |  PASS |
| `fft` | 4096 | 1.673 | 3.506 | 0.1469 | 0.0701 | 0.48x | `2.84e-14` |  PASS |

