# GPU Architecture & CUDA Performance Optimization

This project studies CUDA/GPU performance engineering through three completed
workloads:

- Day 1: parallel reduction
- Day 2: matrix transpose
- Day 3: fixed 3×3 2D convolution

Each workload progresses from a CPU correctness reference to a baseline CUDA
implementation and then to a controlled architecture-driven experiment.
Measurements are collected rather than assumed on an NVIDIA GeForce RTX 4060
Laptop GPU.

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
  shared-memory bank conflicts than the 16×16 tile, primarily through improved
  transposed shared loads; this is a plausible contributor to its small measured
  advantage, not proof of a sole cause.
- **Convolution:** tiling reduced L1 global-load requests by approximately 41.7%
  and sectors by approximately 66.2% at 2048×2048, but downstream L1 misses, L2
  misses, and DRAM-read traffic did not improve. Added shared-memory activity,
  bank conflicts, synchronization pressure, register use, and poorer ready-warp
  availability help explain why tiled convolution remained approximately 32–34%
  slower in the authoritative CUDA-event runs.

## Build and run

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/reduction_benchmark
./build/transpose_benchmark
./build/convolution_benchmark
```

## Results

- [Day 1 Reduction Results](results/reduction_day1.md)
- [Day 2 Matrix Transpose Results](results/transpose_day2.md)
- [Day 3 2D Convolution Results](results/convolution_day3.md)
- [Day 4 Nsight Compute Profiling](results/profiling_day4.md)

Each result document records correctness coverage, benchmark methodology,
measured results, interpretation, and experiment limitations.
