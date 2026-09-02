# 2D Convolution

## Implementation

The workload is a fixed `3x3` same-size float stencil with zero padding. Filter coefficients are used directly rather than flipped. The CPU reference is compared with two CUDA kernels:

- a naive kernel that reads each valid `3x3` neighborhood from global memory;
- a tiled kernel that stages an `18x18` input region for each `16x16` output block.

Both CUDA versions use one thread per output pixel and the same filter, input data, block size, and boundary handling.

## Correctness

| Implementation | Checks |
| --- | ---: |
| CPU reference | 15/15 |
| Naive CUDA | 15/15 |
| Shared-memory tiled CUDA | 15/15 |
| **Total** | **45/45** |

The deterministic cases include empty, small, rectangular, boundary, exact-block, and partial-block images. CUDA results were checked against the CPU reference with an absolute and relative tolerance of `1e-5`.

## Results

Each comparison used 5 warmup launches and 20 measured launches per implementation and size, with alternating execution order. CUDA events measured kernel execution only; allocation, transfers, CPU work, and validation were outside the timed region.

The two recorded runs are shown separately. Times are mean kernel times in milliseconds, and the change is the tiled time relative to naive.

| Size | Run 1 naive | Run 1 tiled | Run 1 change | Run 2 naive | Run 2 tiled | Run 2 change |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 32x32 | 0.009899 | 0.009400 | 5.04% faster | 0.024670 | 0.009861 | 60.03% faster |
| 256x256 | 0.009251 | 0.025352 | 174.04% slower | 0.008942 | 0.009658 | 8.00% slower |
| 1000x1003 | 0.038746 | 0.049400 | 27.50% slower | 0.041368 | 0.051902 | 25.47% slower |
| 2048x2048 | 0.143806 | 0.190342 | 32.36% slower | 0.144216 | 0.192830 | 33.71% slower |
| 4096x4096 | 0.757131 | 0.947651 | 25.16% slower | 0.720208 | 0.983429 | 36.55% slower |

## Notes

The smallest cases were noisy, but the medium and large results consistently favored the naive kernel. At 2048x2048, the tiled version was approximately 32–34% slower across the two recorded runs.

This negative result was kept because both implementations passed the same correctness checks and were benchmarked under the same conditions. [Nsight Compute profiling](profiling.md) found less L1 load work in the tiled kernel, but no corresponding reduction in downstream misses or DRAM reads. The added shared-memory and execution overhead was not offset by the reduced L1 work.
