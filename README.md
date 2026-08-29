# GPU Architecture & CUDA Performance Optimization

This project studies CUDA/GPU performance engineering through three completed
workloads:

- Day 1: parallel reduction
- Day 2: matrix transpose
- Day 3: fixed 3×3 2D convolution

Each workload progresses from a CPU correctness reference to a baseline CUDA
implementation and then to a controlled architecture-driven experiment.
Measurements are collected rather than assumed on an NVIDIA GeForce RTX 4060
Laptop GPU, then connected to hardware behavior with Nsight Compute.

## Environment

| Component | Authoritative environment |
| --- | --- |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU |
| Compute capability | 8.9 |
| GPU memory | Approximately 8 GiB |
| OS/runtime | WSL2 / Ubuntu Linux |
| CUDA Toolkit | 13.3 |
| `nvcc` | 13.3.73 |
| CMake | 3.28.3 |
| Nsight Compute | 2026.2.1.0 |
| Nsight Systems | 2026.1.3 |

## Validation and benchmark methodology

The established deterministic/reference correctness suites passed in full:

| Workload | Correctness result |
| --- | ---: |
| Reduction | 51/51 PASS |
| Transpose | 49/49 PASS |
| Convolution | 45/45 PASS |
| **Combined** | **145/145 PASS** |

Each benchmark uses a CPU reference for correctness, the same workload/input
when comparing CUDA architectures, 5 warmup launches, 20 measured launches,
controlled architecture measurement ordering, and kernel-only CUDA-event
timing. Allocation, host-device transfers, CPU reference work, and validation
are outside the authoritative timed region.

## Headline results

| Workload and representative size | Experimented architecture | Authoritative outcome |
| --- | --- | --- |
| Reduction — 4,000,000 elements | Shared block reduction vs global atomic | 14.1–14.3× faster |
| Transpose — 4096×4096 | Padded tiled vs naive | 1.27–1.28× faster |
| Convolution — 2048×2048 | Shared tiled vs naive | 32–34% higher mean kernel time |

![Optimized mean kernel time normalized independently to each workload's baseline](results/headline_performance.svg)

The figure uses the two authoritative runs and normalizes each workload
independently to a baseline of 1.0; it does not compare absolute runtime across
different workloads.

## Workloads

### Day 1 — Parallel reduction

- Single-threaded CPU reference
- Global-atomic CUDA baseline
- Shared-memory block reduction
- Deterministic correctness regression
- Kernel-only CUDA-event comparison

At 4,000,000 elements, the shared-memory implementation measured approximately
14.1–14.3× faster mean kernel execution than the global-atomic baseline across
the two final runs.

### Day 2 — Matrix transpose

- CPU correctness oracle
- Naive, shared-memory tiled, and padded tiled CUDA implementations
- Rectangular and awkward-dimension correctness coverage
- Controlled three-way kernel-only benchmark

At 4096×4096, padded tiled transpose measured approximately 1.27–1.28× faster
mean kernel execution than naive across the two final runs. Padding was not
beneficial at every tested size.

Across those two authoritative 4096×4096 runs, the padded 16×17 tile was
approximately 2.7–3.4% faster than the unpadded tile. The effect was small and
not universal: the Day 5 reproducibility sanity run reversed the
padded/unpadded ordering while preserving the much stronger tiled-versus-naive
result.

### Day 3 — 2D convolution

- CPU reference for a same-size, zero-padded 3×3 float stencil
- Naive one-thread-per-output CUDA implementation
- Shared-memory tiled CUDA implementation using a 16×16 output block and an
  18×18 input tile with a one-pixel halo
- 45/45 architecture-level correctness checks
- Fair paired benchmark using the same input, filter, block size, warmups, and
  measured-launch count for both CUDA architectures

The implementation uses the common unflipped 3×3 stencil/cross-correlation
convention: filter coefficients are not reversed.

The shared-memory optimization did not improve the meaningful medium and large
benchmark sizes. At 2048×2048, tiled convolution measured approximately 32–34%
higher mean kernel time than naive across the two final runs. This negative
result reinforces that architecture-driven optimizations must be validated by
controlled measurements rather than assumed to help.

### Day 4 — Nsight Compute profiling

Day 4 used Nsight Compute on representative kernels to test the architectural
explanations behind the Day 1–3 CUDA-event benchmark results. CUDA events remain
the source of authoritative timing; profiler counters provide diagnostic
hardware evidence.

- **Reduction:** the shared-memory implementation's approximately 14.1–14.3×
  CUDA-event speedup at 4,000,000 elements coincided with approximately 256×
  lower relevant global-reduction wavefront and L2-sector activity, while input
  global-load work remained unchanged.
- **Transpose:** tiling reduced global-store sectors per request from 16 to 4 at
  4096×4096. The 16×17 padded tile also reported approximately 71% fewer total
  shared-memory bank conflicts and 41% fewer shared-memory wavefronts than the
  16×16 tile. Transposed loads improved substantially while store conflicts
  increased; the aggregate improvement is a plausible contributor to the small
  measured advantage, not proof of a sole cause or universal speedup.
- **Convolution:** tiling reduced L1 global-load requests by approximately 41.7%
  and sectors by approximately 66.2% at 2048×2048, but downstream L1 misses, L2
  misses, and DRAM-read traffic did not improve. Added shared-memory activity,
  bank conflicts, synchronization pressure, register use, and poorer ready-warp
  availability help explain why tiled convolution remained approximately 32–34%
  slower in the authoritative CUDA-event runs.

## Repository structure

```text
include/          reusable CUDA declarations and timing/error utilities
src/common/       shared CUDA utility implementation
src/reduction/    parallel-reduction study
src/transpose/    matrix-transpose study
src/convolution/  fixed 3×3 convolution/stencil study
results/          benchmark reports, profiler evidence, data, and figure
scripts/          deterministic headline-figure generator
```

## Build and run

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/reduction_benchmark
./build/transpose_benchmark
./build/convolution_benchmark
```

The authoritative Day 1–4 experiments used CUDA architecture 75, which CUDA
13.3 selected automatically in this environment. A completely fresh Day 5
configure selected the same value. The tracked CMake default is therefore pinned
to 75 to preserve the recorded experimental build, while still honoring an
explicit user override.

For a native compute-capability 8.9 build on this RTX 4060, configure separately:

```bash
cmake \
  -S . \
  -B build-sm89 \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build-sm89 -j
```

This override changes generated device code. Its timings should not be treated
as directly interchangeable with the recorded architecture-75 measurements.

## Reproducibility and limitations

Results were collected on one RTX 4060 Laptop GPU; laptop power state, dynamic
clocks, thermal conditions, and background activity can introduce timing
variation. Authoritative performance numbers use kernel-only CUDA events, while
Nsight profiler timing is diagnostic and is not substituted for benchmark
timing. Architecture 75 is intentionally pinned as the recorded experiment
configuration; overrides can change code generation and performance. The main
transpose tiling benefit was reproducible, but the much smaller padding runtime
advantage was less stable and reversed in the Day 5 sanity run.

## Results

- [Day 1 Reduction Results](results/reduction_day1.md)
- [Day 2 Matrix Transpose Results](results/transpose_day2.md)
- [Day 3 2D Convolution Results](results/convolution_day3.md)
- [Day 4 Nsight Compute Profiling](results/profiling_day4.md)
- [Day 4 Selected Profiling Metrics CSV](results/profiling_day4.csv)
- [Headline Performance Figure Data](results/headline_performance.csv)

Each result document records correctness coverage, benchmark methodology,
measured results, interpretation, and experiment limitations.
