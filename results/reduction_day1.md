# Day 1 Reduction Results

## Hardware / environment

The Day 1 executable reported and verified the following environment:

- NVIDIA GeForce RTX 4060 Laptop GPU
- Compute capability 8.9
- Approximately 8187.5 MiB global memory reported by CUDA
- 24 multiprocessors reported by CUDA
- Maximum 1024 threads per block
- WSL2 / Linux development environment
- CUDA Toolkit 13.3
- CMake 3.28.3

These development measurements did not rigorously record or control every
laptop variable, including AC-power state, thermal state, dynamic clocks,
background GPU activity, and performance-mode settings. Final project
benchmarking should standardize and record these conditions.

## Correctness

The complete Day 1 regression reported:

```text
CPU reduction tests: PASS (17 cases)
Baseline CUDA reduction tests: PASS (17 cases)
Shared-memory CUDA reduction tests: PASS (17 cases)
```

Coverage includes:

- Empty input
- A single element
- Small arrays with known sums
- Inputs containing zeros
- 64-bit accumulation using two `UINT32_MAX` inputs
- Deterministic randomized inputs
- Sizes 31, 32, and 33
- Sizes 255, 256, and 257
- Sizes 1000, 1003, and 100000

The fixed random-number-generator seed is `0x00C0FFEE`, and randomized
values are in the range 0 through 1000. This coverage is deterministic and
useful for regression testing, but it is not exhaustive verification.

## Implementations

### CPU reference

The CPU reference is single-threaded. It consumes `uint32_t` elements and
accumulates into a `uint64_t` result.

### Baseline CUDA

- 256 threads per block
- One CUDA thread per input element
- One global unsigned 64-bit atomic contribution per valid input element

### Shared-memory CUDA

- 256 threads per block
- One unsigned 64-bit shared value per thread
- 2048 bytes of shared memory per block
- Tree-reduction strides: `128 → 64 → 32 → 16 → 8 → 4 → 2 → 1`
- Block synchronization between reduction stages
- One global unsigned 64-bit atomic contribution per launched block

The fixed 256-thread configuration is the current Day 1 design, not a claim of
universal optimality.

## Benchmark methodology

The final comparison uses:

- Kernel-only timing with CUDA events
- One deterministic input vector shared by both implementations at each size
- 256 threads per block for both kernels
- 5 warmup launches per implementation and size
- 20 measured launches per implementation and size
- Alternating measured order: baseline/shared, then shared/baseline
- Accumulator reset with `cudaMemset` before every launch and outside timing
- Device allocation outside timing
- Host-to-device and device-to-host transfers outside timing
- CPU reference calculation outside timing
- Exact validation of both CUDA results against the CPU reference
- Arithmetic mean of 20 measurements
- Minimum observed timing among the same 20 measurements
- Mean-time speedup calculated as baseline mean divided by shared mean

These measurements compare kernel execution only. They do not represent
end-to-end application time including allocation, transfers, input generation,
CPU reference work, validation, or reporting.

## Actual Day 1 comparison

Both authoritative final runs are retained independently. They are not averaged
together, and no result is selectively discarded.

### Run 1

| Elements | Baseline mean (ms) | Shared mean (ms) | Speedup | Baseline min (ms) | Shared min (ms) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1024 | 0.007424 | 0.007475 | 0.993151 | 0.006144 | 0.006144 |
| 10000 | 0.015763 | 0.009523 | 1.655242 | 0.009216 | 0.004096 |
| 100000 | 0.064803 | 0.013598 | 4.765502 | 0.061440 | 0.008192 |
| 1000000 | 0.589598 | 0.048230 | 12.224622 | 0.585728 | 0.045056 |
| 4000000 | 2.418920 | 0.169618 | 14.261020 | 2.331648 | 0.166912 |

### Run 2

| Elements | Baseline mean (ms) | Shared mean (ms) | Speedup | Baseline min (ms) | Shared min (ms) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1024 | 0.007168 | 0.007373 | 0.972222 | 0.004096 | 0.006144 |
| 10000 | 0.014171 | 0.014634 | 0.968401 | 0.010048 | 0.005088 |
| 100000 | 0.065013 | 0.011816 | 5.502099 | 0.061440 | 0.008192 |
| 1000000 | 0.603853 | 0.048477 | 12.456532 | 0.585728 | 0.045056 |
| 4000000 | 2.380179 | 0.168890 | 14.093107 | 2.331648 | 0.166912 |

## Observations

- The 1,024-element workload showed no shared-memory speed advantage in either
  run.
- The 10,000-element result varied substantially between the two runs, so no
  strong conclusion should be made for that size.
- The shared-memory implementation was consistently faster in both runs at
  100,000, 1,000,000, and 4,000,000 elements.
- At 4,000,000 elements, the measured mean kernel-time speedup was approximately
  14.1–14.3x across the two runs.
- Larger workloads produced substantially more repeatable speedup results than
  the smallest workloads.

The architectural difference is:

```text
baseline:
approximately one global atomic contribution per valid element

shared:
approximately one global atomic contribution per launched block
```

Reducing global atomic traffic was the intended optimization. These
measurements alone do not prove that atomic contention was the bottleneck.

## Limitations / next analysis

- Laptop GPU power and thermal dynamics were not rigorously controlled.
- Microsecond-scale kernels show noticeable timing variability.
- The fixed 256-thread block size has not been tuned.
- Only kernel execution is compared; end-to-end transfer and setup costs are
  excluded.
- Nsight profiling is still required before making strong bottleneck,
  occupancy, memory-bandwidth, cache, or warp-efficiency conclusions.
- Warp-shuffle reduction was intentionally outside the Day 1 MVP.
