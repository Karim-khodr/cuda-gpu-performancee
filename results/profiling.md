# Nsight Compute Profiling

Nsight Compute was used on one representative launch for each workload: 4,000,000 elements for reduction, 4096x4096 for transpose, and 2048x2048 for convolution. Profiler measurements are used as diagnostic evidence; the CUDA-event benchmark results remain the source of timing and speedup claims.

## Reduction

| Metric | Global atomic | Shared block |
| --- | ---: | ---: |
| Input global-load instructions | 125,000 | 125,000 |
| Input global-load sectors | 500,000 | 500,000 |
| L1 global-reduction wavefronts | 4,000,000 | 15,625 |
| L1 global-reduction sectors | 4,000,000 | 15,625 |
| L2 reduction requests | 3,998,578 | 15,625 |
| L2 reduction sectors | 3,997,169 | 15,625 |

The input-read work was unchanged, while the relevant global-reduction counters were roughly 256x lower in the shared-memory kernel. This supports lower global-reduction pressure as an important part of the measured 14.1–14.3x speedup without assigning the full result to one proven bottleneck.

## Matrix Transpose

| Global-memory metric | Naive | Tiled 16x16 | Padded 16x17 |
| --- | ---: | ---: | ---: |
| Load sectors/request | 4 | 4 | 4 |
| Store sectors/request | 16 | 4 | 4 |
| Global-store sectors | 8,388,608 | 2,097,152 | 2,097,152 |

Tiling reorganized the transposed output writes and reduced global-store sectors/request from 16 to 4. Input-load organization was already the same across the three kernels.

| Shared-memory metric | Tiled 16x16 | Padded 16x17 |
| --- | ---: | ---: |
| Shared-load bank conflicts | 3,747,410 | 552,128 |
| Shared-store bank conflicts | 22,063 | 534,530 |
| Total reported conflicts | 3,766,817 | 1,082,611 |
| Total shared-memory wavefronts | 6,499,931 | 3,805,953 |

Padding reduced total reported conflicts by about 71% and shared-memory wavefronts by about 41%. Shared loads improved, but shared stores became less favorable. This supports better aggregate bank behavior, not a claim that padding must improve runtime; a later reproducibility run reversed the padded and unpadded timing order.

## 2D Convolution

| Cache and memory metric | Naive | Shared tiled |
| --- | ---: | ---: |
| Global-load requests | 2,359,296 | 1,376,128 |
| Global-load sectors | 7,459,126 | 2,520,031 |
| L1 load-sector misses | 1,157,494 | 1,165,281 |
| L2 read misses | 524,290 | 524,290 |
| DRAM bytes read | 16.78 MB | 17.98 MB |

Tiling reduced L1 global-load requests by about 41.7% and sectors by about 66.2%. L1 misses were effectively unchanged, L2 read misses were identical, and DRAM reads did not decrease.

| Execution metric | Naive | Shared tiled |
| --- | ---: | ---: |
| Shared-load instructions | 0 | 1,179,648 |
| Shared-store instructions | 0 | 196,608 |
| Shared-memory bank conflicts | 0 | 1,188,094 |
| Registers/thread | 27 | 46 |
| Theoretical occupancy | 100% | 83.33% |
| Achieved occupancy | 77.22% | 75.72% |
| Eligible warps/scheduler | 1.95 | 1.06 |
| Issued warps/scheduler | 0.65 | 0.51 |
| No eligible warp | 34.99% | 49.05% |
| Stall barrier | 0 | 4.74 |
| Stall MIO throttle | 0.38 | 1.83 |
| Stall long scoreboard | 6.21 | 4.34 |

The tiled kernel added shared-memory traffic and conflicts, synchronization, higher register use, lower theoretical occupancy, and poorer ready-warp availability. Together with the unchanged downstream traffic, these measurements help explain why tiled convolution was 32–34% slower. No single counter is treated as the sole cause.
