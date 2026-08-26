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

Each result document records correctness coverage, benchmark methodology,
measured results, interpretation, and experiment limitations.
