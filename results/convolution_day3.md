# Day 3 — 2D Convolution

## Goal

Day 3 implements and evaluates a fixed 3×3 same-size image operation using a
CPU reference and two controlled CUDA architectures:

```text
CPU reference
      ↓
naive CUDA
      ↓
shared-memory tiled CUDA
```

The experiment asks whether explicitly staging overlapping input neighborhoods
in shared memory improves kernel execution time relative to direct global-memory
neighborhood reads on an NVIDIA GeForce RTX 4060 Laptop GPU.

## Workload definition

All input pixels, filter coefficients, and output pixels use `float`.

For an input with `height` rows and `width` columns, storage is row-major:

```text
flat index = row × width + column
```

The filter is fixed at 3×3 with radius 1. Output dimensions equal input
dimensions. Coordinates outside the input are treated as `0.0f`, so every
image boundary uses zero padding.

For output coordinate `(row, column)`:

```text
sum = 0

for filter_row = 0..2
    for filter_column = 0..2
        image_row    = row    + filter_row    - 1
        image_column = column + filter_column - 1

        if the image coordinate is valid:
            sum += input[image_row, image_column]
                   × filter[filter_row, filter_column]
```

Filter coefficients are used directly and are not flipped. More precisely, the
project uses the common unflipped 3×3 stencil/cross-correlation convention,
although the workload and executable retain the conventional “convolution”
name.

## CPU reference

The CPU reference is deliberately direct and readable. It checks every
candidate input coordinate before indexing the row-major input vector and
therefore handles empty, single-row, single-column, undersized, rectangular,
and awkward images without out-of-bounds access.

Small cases contain manually constructed expected outputs. These independently
verify corner, edge, interior, zero-padding, row/column indexing, and filter
orientation behavior rather than comparing two copies of the same algorithm.

## Naive CUDA architecture

The naive kernel uses:

- A 16×16 CUDA block, or 256 threads
- One thread per output pixel
- Direct global-memory reads for the valid 3×3 input neighborhood
- A 3×3 filter stored in ordinary device global memory
- Signed coordinate arithmetic for potentially negative filter offsets
- One guarded row-major output write per valid thread

Threads outside the output dimensions return without reading or writing image
memory.

## Shared-memory tiled architecture

The tiled kernel keeps the same:

- 16×16 block
- One-thread-per-output mapping
- 3×3 mathematical operation
- Zero-padding policy
- Global-memory filter
- Input, filter, and output data types

Only the input-neighborhood access strategy changes. Each block cooperatively
loads an 18×18 input region:

```text
16×16 output region
+ one-pixel halo on every side
= 18×18 shared input tile
```

The tile contains 324 floats but the block contains 256 threads. Threads use a
linear block index and advance through tile indices in strides of 256. The
first 68 threads therefore load a second tile element. Valid global coordinates
load an input pixel; invalid coordinates explicitly store `0.0f`.

All threads, including threads outside a partial output block, finish
cooperative loading and reach `__syncthreads()`. Only after the barrier do
invalid output threads return. Valid threads read their 3×3 neighborhood from
the staged tile.

## Halo and shared-memory design

Shared tile coordinates map to global input coordinates as:

```text
global row    = block output row    + tile row    - 1
global column = block output column + tile column - 1
```

The subtraction is performed in signed `std::ptrdiff_t` arithmetic. After
bounds checks, valid coordinates are explicitly converted to `std::size_t`
for flat indexing.

Static shared-memory usage is:

```text
18 × 18 × sizeof(float)
= 18 × 18 × 4
= 1296 bytes/block
```

Compiled kernel resource inspection also reported `SHARED:1296`. No dynamic
shared memory is used.

## Correctness

The final deterministic regression reports:

```text
CPU convolution tests: PASS (15/15 cases)
Naive CUDA convolution tests: PASS (15/15 cases)
Tiled CUDA convolution tests: PASS (15/15 cases)

Total: 45/45 PASS
```

The complete dimension coverage is:

```text
0×0
1×1
1×7
7×1
2×2
2×3
3×3
15×15
16×16
17×17
17×31
31×32
32×32
33×31
255×257
```

This covers an empty/no-work image, images smaller than a CUDA block, exact
block dimensions, dimensions immediately below and above block boundaries,
rectangular images, partial edge blocks, explicit boundary pixels,
non-symmetric coefficients, and deterministic random data.

The CPU expected-value seed is `0x434F4E56`. CUDA correctness uses the
deterministic seed `0x43554441`, and benchmark input generation uses
`0x42415345`.

The reusable floating-point policy is:

```text
abs(actual - expected)
<= 1e-5 + 1e-5 × abs(expected)
```

All listed naive and tiled CUDA correctness cases observed zero maximum
absolute error in the final regression. The policy remains tolerance-based;
exact equality is not required as a general property of convolution
arithmetic.

## Benchmark methodology

The authoritative comparison uses:

- GPU: NVIDIA GeForce RTX 4060 Laptop GPU, compute capability 8.9
- CUDA Toolkit/compiler: 13.3 / `nvcc V13.3.73`
- Dimensions: `32×32`, `256×256`, `1000×1003`, `2048×2048`, and
  `4096×4096`
- Block size: 16×16 for both architectures
- Filter: the same deterministic, nontrivial 3×3 global-memory filter
- Input: one deterministic input shared by both architectures at each size
- Warmups: 5 launches per architecture and size
- Measurements: 20 samples per architecture and size
- Timing: kernel-only CUDA events
- Streams: the default stream only, with no kernel overlap
- Ordering: deterministic paired alternation

The paired order is:

```text
even pair: naive → tiled
odd pair:  tiled → naive
```

For each size, both implementations share one host input, CPU reference, device
input, device filter, launch configuration, warmup count, and measured count.
They use separate device output buffers. Both outputs are copied back and
validated against the same CPU reference before timing results are accepted.

The following operations are outside the measured region:

- Allocation and deallocation
- Input generation
- CPU reference calculation
- Host-to-device input and filter transfers
- Device-to-host output transfers
- Correctness comparison
- Statistics and reporting

The results therefore describe kernel execution rather than end-to-end
application time.

## Metrics

Mean and minimum kernel times are computed from the same 20 measured samples.

Mean-time speedup is:

```text
speedup = naive mean kernel time / tiled mean kernel time
```

Therefore, a value greater than 1 means tiled measured faster; a value below 1
means tiled measured slower.

Output throughput is:

```text
MPixels/s =
    output pixels
    / mean elapsed seconds
    / 1,000,000
```

MPixels/s is output pixel throughput. It is not memory bandwidth, physical DRAM
traffic, or FLOP/s.

The “tiled time change” column is relative to naive mean time. “Lower” means
tiled improved mean time; “higher” means tiled regressed.

## Final Run 1

| Dimensions | Naive mean (ms) | Naive min (ms) | Naive MPixels/s | Tiled mean (ms) | Tiled min (ms) | Tiled MPixels/s | N/T speedup | Tiled time change |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 32×32 | 0.009899 | 0.004096 | 103.442703 | 0.009400 | 0.004928 | 108.936171 | 1.053106× | 5.04% lower |
| 256×256 | 0.009251 | 0.006144 | 7084.053916 | 0.025352 | 0.007168 | 2585.042647 | 0.364910× | 174.04% higher |
| 1000×1003 | 0.038746 | 0.036864 | 25886.810134 | 0.049400 | 0.048128 | 20303.643306 | 0.784324× | 27.50% higher |
| 2048×2048 | 0.143806 | 0.140192 | 29166.323885 | 0.190342 | 0.187392 | 22035.573847 | 0.755514× | 32.36% higher |
| 4096×4096 | 0.757131 | 0.696320 | 22158.928304 | 0.947651 | 0.908288 | 17703.999136 | 0.798956× | 25.16% higher |

## Final Run 2

| Dimensions | Naive mean (ms) | Naive min (ms) | Naive MPixels/s | Tiled mean (ms) | Tiled min (ms) | Tiled MPixels/s | N/T speedup | Tiled time change |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 32×32 | 0.024670 | 0.004096 | 41.507231 | 0.009861 | 0.005120 | 103.845530 | 2.501866× | 60.03% lower |
| 256×256 | 0.008942 | 0.006144 | 7328.681271 | 0.009658 | 0.007168 | 6785.950954 | 0.925944× | 8.00% higher |
| 1000×1003 | 0.041368 | 0.036864 | 24245.794383 | 0.051902 | 0.049088 | 19324.732799 | 0.797034× | 25.47% higher |
| 2048×2048 | 0.144216 | 0.139264 | 29083.485824 | 0.192830 | 0.187392 | 21751.259203 | 0.747890× | 33.71% higher |
| 4096×4096 | 0.720208 | 0.696320 | 23294.959101 | 0.983429 | 0.911360 | 17059.919437 | 0.732344× | 36.55% higher |

Both complete runs are retained. They are not averaged, and no unfavorable
result is discarded.

## Performance interpretation

- At `32×32`, launch-scale variability and unstable means prevent a strong
  architecture conclusion.
- At `256×256`, tiled was slower in both runs, but the magnitude varied
  substantially from approximately 8% to 174%.
- At `1000×1003`, tiled consistently measured approximately 25–28% higher
  mean kernel time.
- At `2048×2048`, tiled consistently measured approximately 32–34% higher
  mean kernel time.
- At `4096×4096`, tiled consistently measured approximately 25–37% higher
  mean kernel time.

The strongest repeatable Day 3 statement is:

> At 2048×2048, the shared-memory tiled convolution consistently measured
> approximately 32–34% higher mean kernel time than the naive CUDA
> implementation across the two final comparison runs.

The controlled shared-memory optimization did not outperform naive convolution
at the meaningful medium and large benchmark dimensions.

## Architectural interpretation

The shared-memory hypothesis was reasonable: neighboring output threads require
overlapping input pixels, so staging one 18×18 region could enable explicit
reuse across a 16×16 output block.

The tiled kernel also performs cooperative halo loading, additional
indexing/control work, shared-memory writes, a block-wide synchronization, and
shared-memory reads. In these measurements, the benefit of explicit staging was
insufficient to offset that work.

Modern GPU caches may already serve repeated naive input reads efficiently
enough that explicit staging adds little benefit for this small fixed stencil.
That is a possible architectural explanation, not profiler-established proof.
Day 3 did not use Nsight, so the experiment does not establish cache hit rates,
DRAM transaction counts, occupancy limitations, bank conflicts, or a specific
hardware bottleneck.

## Engineering lesson

A theoretically reasonable GPU optimization is not automatically faster.
Correctness, controlled benchmarking, and measured data determine whether an
optimization is worthwhile for a particular workload and GPU.

The tiled regression is therefore an informative performance-engineering result
rather than a project failure. The implementation was not retuned or replaced
to manufacture a positive speedup.

## Limitations

- Results were collected on one NVIDIA GeForce RTX 4060 Laptop GPU.
- Laptop power and thermal state were not rigorously fixed.
- Dynamic GPU clocks were not locked.
- Background GPU activity was not rigorously controlled.
- CUDA events measure kernel execution only.
- H2D and D2H transfers are excluded.
- The 16×16 block is a controlled design choice, not a tuned optimum.
- Only fixed 3×3 filters were tested.
- Input, filter, and output use only `float`.
- Boundaries use zero padding.
- The operation uses the unflipped stencil/cross-correlation convention.
- The filter remains in ordinary device global memory.
- No constant-memory experiment was performed.
- No texture-memory experiment was performed.
- No separable convolution was implemented.
- No block-size tuning was performed.
- No asynchronous copies, multiple streams, cuDNN, or Tensor Cores were used.
- No Nsight profiling was performed.
- Cache and DRAM explanations remain hypotheses.
- Small-kernel measurements are noisy.
- Benchmark dimensions are representative rather than exhaustive.

## Conclusion

Day 3 established a deterministic CPU oracle, a correct naive CUDA baseline,
and a correct shared-memory tiled implementation for a fixed 3×3 same-size
zero-padded float workload. The complete correctness regression passed 45/45
architecture-level checks.

Two fair, balanced final benchmark runs showed that shared-memory input tiling
did not improve medium or large workloads on the tested GPU. The result
demonstrates the central Day 3 lesson: GPU optimizations must be evaluated with
controlled experiments, including when the measured answer contradicts the
initial architectural hypothesis.
