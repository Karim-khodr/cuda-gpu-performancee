# CUDA GPU Optimization

This project compares baseline and shared-memory CUDA implementations of parallel reduction, matrix transpose, and 2D convolution. Each CUDA implementation was checked against a CPU reference before benchmarking, and CUDA events were used for kernel timing. Nsight Compute was used to examine the main performance differences. Tests were run on an NVIDIA GeForce RTX 4060 Laptop GPU using CUDA 13.3 under WSL2.

## Implementations

- **Reduction:** global atomic baseline and shared-memory block reduction
- **Transpose:** naive, tiled, and padded tiled
- **Convolution:** naive and shared-memory tiled

## Results

| Workload | Comparison | Size | Result |
| --- | --- | --- | --- |
| Reduction | Global atomic vs shared | 4M elements | Shared ~14.1–14.3x faster |
| Transpose | Naive vs padded tiled | 4096x4096 | Padded tiled ~1.27–1.28x faster |
| Convolution | Naive vs tiled | 2048x2048 | Tiled ~32–34% slower |

All implementations passed their CPU-reference correctness tests (145/145 total checks).

## Profiling

- **Reduction:** Nsight showed about 256x less relevant global reduction activity in the shared-memory version.
- **Transpose:** tiling reduced global-store sectors/request from 16 to 4.
- **Convolution:** tiling reduced L1 load work, but downstream misses did not improve enough to offset the additional shared-memory, synchronization, and register overhead.

## Build and Run

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

```bash
./build/reduction_benchmark
./build/transpose_benchmark
./build/convolution_benchmark
```

The recorded project results used CUDA architecture 75, which is the default in the CMake configuration for reproducibility. A different target can be selected with `-DCMAKE_CUDA_ARCHITECTURES=89`, but timings from a different build should not be compared directly with the recorded results.

## Detailed Results

- [Reduction](results/reduction_day1.md)
- [Matrix transpose](results/transpose_day2.md)
- [2D convolution](results/convolution_day3.md)
- [Nsight Compute profiling](results/profiling_day4.md)
- [Selected profiler metrics](results/profiling_day4.csv)
