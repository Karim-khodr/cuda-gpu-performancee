# GPU Architecture & CUDA Performance Optimization

This project studies CUDA/GPU performance engineering through three workloads:

- Parallel reduction
- Matrix transpose
- 2D convolution

Each workload will compare a CPU reference, a baseline CUDA implementation, and
a justified optimized CUDA implementation. Measurements are collected rather
than assumed on an NVIDIA GeForce RTX 4060 Laptop GPU.

## Current status

Day 1 parallel reduction and Day 2 matrix transpose are implemented.

Reduction includes:

- A single-threaded CPU reference
- A global-atomic CUDA baseline
- A shared-memory CUDA implementation
- Deterministic CPU and CUDA correctness testing
- Kernel-only CUDA-event timing and side-by-side comparison

Matrix transpose includes:

- A CPU reference used as the correctness oracle
- Naive, shared-memory tiled, and padded tiled CUDA implementations
- Deterministic correctness coverage including awkward rectangular dimensions
- A controlled three-way, kernel-only CUDA-event benchmark

On the RTX 4060 Laptop GPU, the padded tiled transpose measured approximately
1.27–1.28x faster mean kernel execution than the naive version at 4096×4096 in
the two final comparison runs. This is a workload-specific kernel-only result;
padding was not beneficial at every tested size.

2D convolution remains planned work.

## Build and run

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/reduction_benchmark
./build/transpose_benchmark
```

## Results

See [Day 1 Reduction Results](results/reduction_day1.md) for the verified
environment, correctness coverage, benchmark methodology, measured comparison,
and current limitations.

See [Day 2 Matrix Transpose Results](results/transpose_day2.md) for the three
CUDA architectures, correctness coverage, controlled benchmark methodology,
two final measured runs, interpretation, and limitations.
