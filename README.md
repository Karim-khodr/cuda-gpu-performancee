# GPU Architecture & CUDA Performance Optimization

This project studies CUDA/GPU performance engineering through three planned
workloads:

- Parallel reduction
- Matrix transpose
- 2D convolution

Each workload will compare a CPU reference, a baseline CUDA implementation, and
a justified optimized CUDA implementation. Measurements are collected rather
than assumed on an NVIDIA GeForce RTX 4060 Laptop GPU.

## Current status

Day 1 reduction is implemented. It currently includes:

- A single-threaded CPU reference
- A global-atomic CUDA baseline
- A shared-memory CUDA implementation
- Deterministic CPU and CUDA correctness testing
- Kernel-only CUDA-event timing and side-by-side comparison

Matrix transpose and 2D convolution remain planned work.

## Build and run

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/reduction_benchmark
```

## Day 1 results

See [Day 1 Reduction Results](results/reduction_day1.md) for the verified
environment, correctness coverage, benchmark methodology, measured comparison,
and current limitations.
