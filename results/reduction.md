# Reduction

## Implementation

The CPU reference accumulates `uint32_t` inputs into a `uint64_t` result. The baseline CUDA kernel uses one global 64-bit atomic contribution per input element. The shared-memory version reduces 256 values within each block and makes one global atomic contribution per block.

## Correctness

| Implementation | Checks |
| --- | ---: |
| CPU reference | 17/17 |
| Global-atomic CUDA | 17/17 |
| Shared-memory CUDA | 17/17 |
| **Total** | **51/51** |

The deterministic tests include empty and small inputs, values near block boundaries, awkward sizes, randomized inputs, and 64-bit accumulation.

## Results

Both kernels used 256 threads per block and the same deterministic input at each size. Each comparison used 5 warmup launches and 20 measured launches, with alternating kernel order. CUDA events measured kernel execution only; allocation, transfers, accumulator resets, CPU work, and validation were outside the timed region.

The two recorded runs are shown separately. Times are mean kernel times in milliseconds.

| Elements | Run 1 atomic | Run 1 shared | Run 1 speedup | Run 2 atomic | Run 2 shared | Run 2 speedup |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1,024 | 0.007424 | 0.007475 | 0.993x | 0.007168 | 0.007373 | 0.972x |
| 10,000 | 0.015763 | 0.009523 | 1.655x | 0.014171 | 0.014634 | 0.968x |
| 100,000 | 0.064803 | 0.013598 | 4.766x | 0.065013 | 0.011816 | 5.502x |
| 1,000,000 | 0.589598 | 0.048230 | 12.225x | 0.603853 | 0.048477 | 12.457x |
| 4,000,000 | 2.418920 | 0.169618 | 14.261x | 2.380179 | 0.168890 | 14.093x |

## Notes

The small workloads were noisy and did not show a consistent advantage. At 4,000,000 elements, the shared-memory kernel was approximately 14.1–14.3x faster across the two runs.

The optimization replaces roughly one global atomic contribution per element with one per block. [Nsight Compute profiling](profiling.md) found unchanged input-load work and about 256x less relevant global-reduction activity. This supports lower global-reduction pressure, but it does not prove a single bottleneck caused the entire speedup.
