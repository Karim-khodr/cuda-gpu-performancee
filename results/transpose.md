# Matrix Transpose

## Implementation

The project includes a single-threaded CPU reference and three CUDA kernels:

- naive one-thread-per-element transpose;
- a `16x16` shared-memory tile;
- a padded `16x17` shared-memory tile.

The padded version changes the shared-memory row stride without changing the logical transpose or the `16x16` thread block.

## Correctness

| Implementation | Checks |
| --- | ---: |
| CPU reference | 13/13 |
| Naive CUDA | 12/12 |
| Tiled CUDA | 12/12 |
| Padded tiled CUDA | 12/12 |
| **Total** | **49/49** |

The deterministic tests include empty, square, rectangular, and awkward dimensions that produce partially filled tiles. CUDA outputs were compared exactly with the CPU reference.

## Results

Each implementation used the same input and `16x16` block size. The benchmark used 5 warmup launches and 20 measured launches per implementation and size, with a rotating execution order. CUDA events measured kernel execution only; setup, transfers, CPU work, and validation were outside the timed region.

The two recorded runs are shown separately. Times are mean kernel times in milliseconds.

| Size | Run 1 naive | Run 1 tiled | Run 1 padded | Run 2 naive | Run 2 tiled | Run 2 padded |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 32x32 | 0.008850 | 0.026744 | 0.019397 | 0.008027 | 0.007162 | 0.007974 |
| 256x256 | 0.013694 | 0.010474 | 0.011222 | 0.009965 | 0.009206 | 0.009874 |
| 1000x1003 | 0.038890 | 0.023203 | 0.024410 | 0.038747 | 0.023325 | 0.023018 |
| 2048x2048 | 0.123026 | 0.112274 | 0.109094 | 0.124082 | 0.110320 | 0.110883 |
| 4096x4096 | 0.763699 | 0.621029 | 0.600842 | 0.769010 | 0.616133 | 0.599752 |

## Notes

At 4096x4096, padded tiling was 1.271x and 1.282x faster than naive in the two recorded runs, or approximately 1.27–1.28x overall. [Nsight Compute profiling](profiling.md) showed that tiling reduced global-store sectors/request from 16 to 4.

Padding itself provided only a small 2.7–3.4% improvement over the unpadded tile at 4096x4096 in these runs. Its effect was mixed at other sizes, and a later fresh-build reproducibility run reversed the padded and unpadded ordering while preserving the larger tiled-versus-naive improvement. Padding is therefore not treated as universally faster.
