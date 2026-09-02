# Day 4 — Nsight Compute Profiling and Performance Analysis

## 1. Objective

Days 1–3 established correctness and kernel-only performance using CUDA events.
Day 4 used Nsight Compute on representative kernels to test the architectural
explanations behind those benchmark outcomes against GPU hardware counters.

```text
benchmarking → how fast?
profiling    → what was the GPU doing?
```

CUDA-event measurements remain authoritative for project speedups and timing.
Nsight Compute results are diagnostic architectural evidence.

## 2. Methodology

The profiling environment was:

```text
GPU: NVIDIA GeForce RTX 4060 Laptop GPU
Compute capability: 8.9
Nsight Compute: 2026.2.1.0
```

Representative workloads were selected from the completed benchmarks:

| Workload | Profiled size | Selection rationale |
| --- | ---: | --- |
| Reduction | 4,000,000 elements | Strongest stable Day 1 architecture difference |
| Transpose | 4096×4096 | Clear large-matrix tiling improvement and repeatable padding difference |
| Convolution | 2048×2048 | Most repeatable large negative tiling result |

Each report collected one verified target launch using kernel-name filtering,
launch skipping, and a launch count of one. The common section set was:

- `SpeedOfLight`
- `MemoryWorkloadAnalysis`
- `MemoryWorkloadAnalysis_Tables`
- `LaunchStats`
- `Occupancy`

Convolution additionally used `SchedulerStats` and `WarpStateStats` to examine
ready-warp behavior and synchronization-related stalls.

Nsight required multi-pass replay and instrumentation. Convolution used 38
passes, and the other workloads also required replay. Profiler duration is
therefore diagnostic only and is not used for the benchmark conclusions below.

Selected raw counters cited in this report are also available in the compact
[Day 4 profiling CSV](profiling_day4.csv).

## 3. Reduction

### Benchmark result

At 4,000,000 elements, the shared-memory block reduction measured approximately
14.1–14.3× lower mean kernel time than the global-atomic baseline across the two
authoritative CUDA-event runs.

### Measured profiler evidence

| Metric | Global-atomic baseline | Shared block reduction |
| --- | ---: | ---: |
| Input global-load instructions | 125,000 | 125,000 |
| Input global-load sectors | 500,000 | 500,000 |
| L1 global-reduction wavefronts | 4,000,000 | 15,625 |
| L1 global-reduction sectors | 4,000,000 | 15,625 |
| L2 reduction requests | 3,998,578 | 15,625 |
| L2 reduction sectors | 3,997,169 | 15,625 |
| Shared-load bank conflicts | 0 | 0 |
| Shared-store bank conflicts | 0 | 0 |
| Registers/thread | 16 | 16 |
| Theoretical occupancy | 100% | 100% |

The baseline used only 3.09% of peak DRAM throughput.

### Interpretation

The kernels performed the same input-read work, but block-local reduction
lowered the relevant global-reduction wavefront, sector, and L2-reduction
counters by approximately 256×. The optimized kernel introduced substantial
shared-memory work without reported shared-load or shared-store bank conflicts.
This strongly supports substantially lower global-reduction pressure as the
architectural explanation for the large benchmark improvement.

The low baseline DRAM utilization does not support raw DRAM-bandwidth saturation
as the primary explanation. Scheduler and warp-state sections were not collected
for reduction, so no exact scheduler-level stall cause is claimed.

## 4. Transpose

### Benchmark result

At 4096×4096, padded tiled transpose measured approximately 1.27–1.28× faster
than naive, corresponding to approximately 21–22% lower mean kernel time. The
padded tile measured approximately 2.7–3.4% faster than the unpadded tile across
the two authoritative CUDA-event runs.

### Measured global-access evidence

| Metric | Naive | Tiled 16×16 | Padded 16×17 |
| --- | ---: | ---: | ---: |
| Global-load requests | 524,288 | 524,288 | 524,288 |
| Global-load sectors | 2,097,152 | 2,097,152 | 2,097,152 |
| Load sectors/request | 4 | 4 | 4 |
| Global-store requests | 524,288 | 524,288 | 524,288 |
| Global-store sectors | 8,388,608 | 2,097,152 | 2,097,152 |
| Store sectors/request | 16 | 4 | 4 |

Input loads were already efficiently organized and unchanged by tiling. The
main measured global-memory difference was on output stores: tiling reduced
global-store sectors/request from 16 to 4. Physical DRAM byte counts remained
broadly similar, so the counters do not suggest that tiling eliminated the
matrix's DRAM traffic.

### Measured padding evidence

| Shared-memory metric | Tiled 16×16 | Padded 16×17 |
| --- | ---: | ---: |
| Shared-load conflicts | 3,747,410 | 552,128 |
| Shared-store conflicts | 22,063 | 534,530 |
| Total reported conflicts | 3,766,817 | 1,082,611 |
| Total shared-memory wavefronts | 6,499,931 | 3,805,953 |

Padding reported approximately 71% fewer total bank conflicts and 41% fewer
total shared-memory wavefronts. The change was mixed: shared loads improved
substantially, while shared stores became less favorable. All three transpose
kernels retained 100% theoretical occupancy.

### Interpretation

Nsight strongly supports improved global output-store transaction organization
as the main benefit of transpose tiling. It also provides evidence that the
16×17 layout improved aggregate shared-memory bank behavior, primarily through
the transposed shared loads. That improvement is a plausible contributor to the
measured 2.7–3.4% padding advantage, but the counters do not establish that bank
conflicts alone produced the complete timing difference.

## 5. Convolution

### Benchmark result

At 2048×2048, shared-memory tiled convolution consistently measured
approximately 32–34% higher mean kernel time than naive across the two
authoritative CUDA-event runs.

### Measured input and cache evidence

| Metric | Naive | Shared-memory tiled |
| --- | ---: | ---: |
| Global-load requests | 2,359,296 | 1,376,128 |
| Global-load sectors | 7,459,126 | 2,520,031 |
| L1 global-load hit rate | 84.48% | 53.76% |
| L1 load-sector misses | 1,157,494 | 1,165,281 |
| L2 read misses | 524,290 | 524,290 |
| DRAM bytes read | 16.78 MB | 17.98 MB |

Tiling reduced L1 global-load requests by approximately 41.7% and sectors by
approximately 66.2%. Despite that reduction, L1 misses were effectively
unchanged, L2 read misses were identical, and DRAM reads did not decrease.
Aggregate load counters include image and ordinary global-memory filter reads,
which this collection cannot separate.

### Measured tiled overhead

| Metric | Naive | Shared-memory tiled |
| --- | ---: | ---: |
| Shared-load instructions | 0 | 1,179,648 |
| Shared-store instructions | 0 | 196,608 |
| Total shared-memory wavefronts | Not used as a comparable baseline | 2,865,193 |
| Total shared-memory bank conflicts | 0 | 1,188,094 |
| Registers/thread | 27 | 46 |
| Theoretical occupancy | 100% | 83.33% |
| Achieved occupancy | 77.22% | 75.72% |
| Eligible warps/scheduler | 1.95 | 1.06 |
| Issued warps/scheduler | 0.65 | 0.51 |
| No eligible warp | 34.99% | 49.05% |
| Stall Barrier | 0 | 4.74 |
| Stall MIO Throttle | 0.38 | 1.83 |
| Stall Long Scoreboard | 6.21 | 4.34 |

The theoretical-occupancy reduction was driven primarily by the increase in
register use, not the small 1,296-byte shared-memory tile.

### Interpretation

Profiling strongly supports useful cache reuse in the naive kernel. The naive
kernel performed much more L1 load work, but that did not become additional
downstream miss or DRAM-read traffic. Explicit tiling therefore removed L1 work
without creating a comparable downstream traffic benefit.

The tiled kernel then added shared-memory traffic and bank conflicts,
synchronization pressure, higher register use, reduced theoretical occupancy,
and poorer ready-warp availability. The combined effects provide the safest
explanation for the negative optimization result. No one counter is identified
as the sole cause.

## 6. Cross-Workload Lessons

| Workload | Optimization | Benchmark outcome | Profiler evidence | Engineering lesson |
| --- | --- | --- | --- | --- |
| Reduction | Block-local shared-memory reduction | Large win: ~14.1–14.3× | ~256× lower relevant global-reduction wavefront/sector activity with unchanged input loads | The measured counters support replacing massive global-reduction activity with block-local work. |
| Transpose | Shared-memory tiling; then 16×17 padding | Moderate tiling win; small additional padding win | Store sectors/request fell from 16 to 4; padding improved aggregate bank behavior | Shared memory paid off by reorganizing inefficient output stores; padding refined local access behavior. |
| Convolution | Explicit 18×18 shared input tile | Loss: tiled ~32–34% slower | Less L1 load activity but unchanged downstream misses, plus staging, bank, register, barrier, and scheduler costs | The evidence is consistent with caches capturing much reuse while explicit staging added overhead. |

Shared memory is a tool, not an automatic optimization. Its value depends on
which expensive hardware behavior it replaces and which execution costs it
introduces.

## 7. Limitations

- Nsight replay and instrumentation alter execution conditions.
- Profiler duration is diagnostic only; CUDA events provide authoritative
  benchmark timing.
- Some memory metrics aggregate multiple source-level load types, such as image
  and filter reads in convolution.
- Cache hit rates and warp-stall categories support interpretation but do not
  establish sole causality.
- Separate profiler runs can show small physical-traffic variation.
- Representative sizes were profiled rather than every benchmark size.
- Laptop GPU clocks, power, thermal state, and background activity were not
  rigorously fixed for the profiling runs.

## 8. Conclusion

Day 4 connected the three CUDA-event benchmark outcomes to hardware evidence.
Reduction's large improvement coincided with block-local work replacing large
global-reduction activity. Transpose's improvement is consistent with tiling
reorganizing inefficient output stores and padding improving aggregate
shared-memory bank behavior. For convolution, the counters support the
hypothesis that reduced L1 load work did not offset the additional staging and
execution costs.

The results reinforce that an architectural optimization should be evaluated by
both the expensive behavior it removes and the overhead it adds.
