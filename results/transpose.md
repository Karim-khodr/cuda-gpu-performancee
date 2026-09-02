# Day 2 — Matrix Transpose

## Objective

Day 2 implements and evaluates a rectangular matrix transpose using a CPU
reference and three controlled CUDA architectures. The work progresses from a
direct global-memory implementation to shared-memory tiling, followed by a
single-variable padding experiment.

## Implementations

All matrices contain `float` elements. A `rows × cols` input becomes a
`cols × rows` output.

### CPU reference

The single-threaded CPU implementation is the correctness oracle for every CUDA
implementation. It copies each element from:

```text
input[row * cols + col]
```

to:

```text
output[col * rows + row]
```

### Naive CUDA

Each thread determines one `(row, col)` coordinate, reads
`input[row * cols + col]`, and writes `output[col * rows + row]`.
Neighboring threads along the x direction access nearby input elements in
row-major storage, while the direct transposed writes are strided with respect
to the output layout.

### Shared-memory tiled CUDA

The basic tiled kernel uses a static `16×16` shared-memory tile:

```text
global input
     ↓
16×16 shared-memory tile
     ↓
__syncthreads()
     ↓
shared tile read with transposed local indices
     ↓
global transposed output
```

Shared memory is a staging area that allows the block to reorganize a tile
before global output writes. The tile consumes 1,024 bytes of shared memory per
block.

### Padded shared-memory tiled CUDA

The padded kernel isolates one architectural change:

```text
unpadded: tile[16][16]
padded:   tile[16][17]
```

The padded tile consumes 1,088 bytes of shared memory per block. The additional
column changes the row stride used by transposed shared-memory accesses without
changing the number of threads, global coordinates, boundary conditions, or
logical transpose. Padding changes the shared-memory bank mapping of transposed
accesses and was tested as a controlled follow-up experiment.

## Correctness

The final deterministic correctness regression reports:

```text
CPU transpose tests: PASS (13 cases)
Naive CUDA transpose tests: PASS (12 cases)
Tiled CUDA transpose tests: PASS (12 cases)
Padded tiled CUDA transpose tests: PASS (12 cases)
```

The CPU transpose is the oracle for all GPU implementations. GPU output values
are compared exactly because transpose copies the finite test values without
performing floating-point arithmetic.

CUDA correctness coverage uses the following dimensions:

```text
0×0
1×1
1×7
7×1
2×3
4×4
17×31
31×32
32×32
33×31
255×257
1000×1003
```

The awkward rectangular dimensions deliberately exercise partially filled
input and output blocks or tiles. Random inputs use the deterministic seed
`0x5452414E`.

## Benchmark methodology

The final comparison uses:

- GPU: NVIDIA GeForce RTX 4060 Laptop GPU, compute capability 8.9
- Element type: `float`
- Block size: `16×16` threads, or 256 threads per block, for all kernels
- Architectures: naive, unpadded `16×16` tiled, and padded `16×17` tiled
- Dimensions: `32×32`, `256×256`, `1000×1003`, `2048×2048`, and
  `4096×4096`
- Warmups: 5 launches per implementation and matrix size
- Measurements: 20 timings per implementation and matrix size
- Timing: kernel-only CUDA events
- Ordering: deterministic three-way rotation
- Input seed: `0x5452414E`

For each matrix size, all three implementations use the same host input, device
input, CPU reference, block configuration, warmup count, and measurement count.
The three final device outputs are compared exactly against the CPU reference
before the benchmark row is accepted.

The timed region is only:

```text
START CUDA EVENT
       ↓
one transpose kernel
       ↓
STOP CUDA EVENT
       ↓
stop-event synchronization
```

Input generation, CPU reference work, `cudaMalloc`, `cudaFree`, host-to-device
and device-to-host transfers, correctness comparison, statistics, and console
output are outside the timed region. These results are not end-to-end
application timings.

## Effective bandwidth definition

For `N = rows × cols`, useful bytes are defined as:

```text
useful bytes = 2 × N × sizeof(float)
```

The factor of two represents one useful matrix read and one useful matrix
write. Effective bandwidth is:

```text
effective bandwidth = useful bytes / mean kernel time in seconds
```

The implementation uses milliseconds and decimal GB/s:

```text
GB/s = useful_bytes / (mean_ms × 1,000,000)
```

Effective bandwidth represents useful logical matrix bytes divided by measured
kernel time. It is not the same as physical DRAM traffic or profiler-measured
hardware bandwidth.

Mean-time speedups are:

```text
N/T = naive mean / unpadded tiled mean
N/P = naive mean / padded tiled mean
T/P = unpadded tiled mean / padded tiled mean
```

A ratio greater than 1 means the denominator architecture measured faster.

## Final benchmark results

The two authoritative runs are retained separately and are not averaged.
`N` is naive, `T` is unpadded tiled, and `P` is padded tiled.

### Run 1

| Rows × Cols | N mean (ms) | T mean (ms) | P mean (ms) | N effective GB/s | T effective GB/s | P effective GB/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 32×32 | 0.008850 | 0.026744 | 0.019397 | 0.926 | 0.306 | 0.422 |
| 256×256 | 0.013694 | 0.010474 | 0.011222 | 38.285 | 50.058 | 46.718 |
| 1000×1003 | 0.038890 | 0.023203 | 0.024410 | 206.328 | 345.814 | 328.723 |
| 2048×2048 | 0.123026 | 0.112274 | 0.109094 | 272.743 | 298.863 | 307.572 |
| 4096×4096 | 0.763699 | 0.621029 | 0.600842 | 175.747 | 216.122 | 223.383 |

| Rows × Cols | N/T | N/P | T/P |
| ---: | ---: | ---: | ---: |
| 32×32 | 0.331 | 0.456 | 1.379 |
| 256×256 | 1.308 | 1.220 | 0.933 |
| 1000×1003 | 1.676 | 1.593 | 0.951 |
| 2048×2048 | 1.096 | 1.128 | 1.029 |
| 4096×4096 | 1.230 | 1.271 | 1.034 |

### Run 2

| Rows × Cols | N mean (ms) | T mean (ms) | P mean (ms) | N effective GB/s | T effective GB/s | P effective GB/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 32×32 | 0.008027 | 0.007162 | 0.007974 | 1.021 | 1.144 | 1.027 |
| 256×256 | 0.009965 | 0.009206 | 0.009874 | 52.614 | 56.948 | 53.100 |
| 1000×1003 | 0.038747 | 0.023325 | 0.023018 | 207.086 | 344.012 | 348.603 |
| 2048×2048 | 0.124082 | 0.110320 | 0.110883 | 270.422 | 304.155 | 302.611 |
| 4096×4096 | 0.769010 | 0.616133 | 0.599752 | 174.533 | 217.839 | 223.789 |

| Rows × Cols | N/T | N/P | T/P |
| ---: | ---: | ---: | ---: |
| 32×32 | 1.121 | 1.007 | 0.898 |
| 256×256 | 1.082 | 1.009 | 0.932 |
| 1000×1003 | 1.661 | 1.683 | 1.013 |
| 2048×2048 | 1.125 | 1.119 | 0.995 |
| 4096×4096 | 1.248 | 1.282 | 1.027 |

## Interpretation

- `32×32` showed high proportional variability, so no strong optimization
  conclusion is drawn from this tiny workload.
- At `256×256`, the tiled variants provided little consistent advantage over
  naive across the final runs, and padding was slower than unpadded tiling.
- At `1000×1003`, both tiled versions substantially outperformed naive, while
  padding versus unpadded tiling was mixed.
- At `2048×2048`, both tiled versions were faster than naive, while padding
  was approximately neutral or slightly mixed.
- At `4096×4096`, padded tiling produced the strongest repeatable result. Run
  1 measured 0.763699 ms naive versus 0.600842 ms padded, or 1.271× N/P. Run 2
  measured 0.769010 ms naive versus 0.599752 ms padded, or 1.282× N/P.

At 4096×4096, the padded tiled transpose measured approximately 1.27–1.28×
faster mean kernel execution than the naive transpose across the two final
comparison runs. This corresponds to approximately 21–22% lower mean kernel
execution time.

## Padding experiment

Padding was a mixed, size-dependent optimization. Compared with the unpadded
tiled kernel, it was slower at `256×256`, mixed or approximately neutral at
`1000×1003` and `2048×2048`, and consistently about 2.7–3.4% faster in mean
kernel time at `4096×4096`.

The padded tiled version is retained as the final optimized architecture
because it preserved correctness and provided the strongest repeatable result
on the largest tested workload.

The padding experiment was motivated by the shared-memory access pattern of
the unpadded tiled transpose. The measured `4096×4096` improvement is
consistent with the hypothesis that changing the shared-memory row stride can
reduce conflict-related overhead. Nsight profiling has not yet been used to
establish the hardware cause.

## Limitations

- RTX 4060 Laptop GPU power and thermal state were not rigorously controlled.
- GPU dynamic clocks were not fixed.
- Background GPU activity was not rigorously controlled.
- Very small kernels show significant relative timing variability.
- The `16×16` block size was fixed rather than tuned.
- Timing is kernel-only, not end-to-end.
- Effective bandwidth is useful-data bandwidth, not physical DRAM traffic.
- No Nsight profiling was used on Day 2.
- The bank-conflict explanation is architecture-based and not
  profiler-confirmed.
- Padding benefit was size-dependent.
- Only `float` matrices were tested.
- The benchmark size set is representative, not exhaustive.

## Day 4 and Day 5 follow-up

The profiler limitations above describe the state at the end of Day 2.
[Nsight Compute profiling on Day 4](profiling_day4.md) later showed that global
store sectors/request fell from 16 in the naive kernel to 4 in both tiled
variants. For padding, total reported shared-memory conflicts fell by
approximately 71% and shared-memory wavefronts by approximately 41%, primarily
through improved transposed-load behavior, although shared-store conflicts
increased.

The 2.7–3.4% padding runtime advantage at `4096×4096` remains specific to the
two authoritative Day 2 runs. A single fresh-build Day 5 sanity run reversed
the padded/unpadded ordering while preserving the substantially stronger
padded-versus-naive improvement. The profiling evidence therefore supports
improved aggregate shared-memory behavior, not a claim that padding is
universally faster.

## Day 2 conclusion

Day 2 established a tested progression from a CPU oracle through naive and
shared-memory CUDA transpose implementations to a controlled padding
experiment. Both tiled kernels improved large-matrix performance relative to
the naive kernel, while padding itself produced mixed results across sizes.

The padded tiled implementation is retained as the final Day 2 optimized
architecture because it remained exactly correct and delivered the strongest
repeatable result at `4096×4096`. The measurements support a
workload-specific performance conclusion, not a claim that padding is
universally beneficial or that a particular hardware bottleneck has been
proven.
