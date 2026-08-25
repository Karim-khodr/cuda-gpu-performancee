#include "cuda_utils.cuh"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <string>
#include <vector>

// Fixed correctness-baseline configuration; 256 is not assumed to be optimal.
constexpr unsigned int baseline_block_size = 256;

// Fixed Day 1 shared-memory configuration; not assumed to be optimal.
constexpr unsigned int shared_block_size = 256;

// Current Day 1 methodology; these iteration counts are not assumed optimal.
constexpr std::size_t benchmark_warmup_iterations = 5;
constexpr std::size_t benchmark_measured_iterations = 20;
constexpr std::uint32_t benchmark_seed = 0x00C0FFEEU;

constexpr std::array<std::size_t, 5> benchmark_sizes{
    1024, 10000, 100000, 1000000, 4000000};

struct ComparisonBenchmarkResult {
    std::size_t element_count;
    double baseline_mean_ms;
    double shared_mean_ms;
    double speedup;
    double baseline_minimum_ms;
    double shared_minimum_ms;
};

__global__ void atomic_reduction_kernel(const std::uint32_t* input,
                                        std::size_t element_count,
                                        unsigned long long* result) {
    const std::size_t index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (index < element_count) {
        atomicAdd(result, static_cast<unsigned long long>(input[index]));
    }
}

__global__ void shared_reduction_kernel(const std::uint32_t* input,
                                        std::size_t element_count,
                                        unsigned long long* result) {
    __shared__ unsigned long long shared_values[shared_block_size];

    const unsigned int local_index = threadIdx.x;
    const std::size_t global_index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + local_index;

    shared_values[local_index] =
        global_index < element_count
            ? static_cast<unsigned long long>(input[global_index])
            : 0ULL;
    __syncthreads();

    for (unsigned int stride = shared_block_size / 2;
         stride > 0;
         stride /= 2) {
        if (local_index < stride) {
            shared_values[local_index] += shared_values[local_index + stride];
        }
        __syncthreads();
    }

    if (local_index == 0) {
        atomicAdd(result, shared_values[0]);
    }
}

std::uint64_t cpu_reduce(const std::vector<std::uint32_t>& input) {
    std::uint64_t sum = 0;

    for (const std::uint32_t value : input) {
        sum += value;
    }

    return sum;
}

std::uint64_t cuda_reduce_atomic(const std::vector<std::uint32_t>& input) {
    if (input.empty()) {
        return 0;
    }

    std::uint32_t* device_input = nullptr;
    unsigned long long* device_result = nullptr;

    try {
        const std::size_t input_bytes = input.size() * sizeof(std::uint32_t);
        CUDA_CHECK(cudaMalloc(
            reinterpret_cast<void**>(&device_input), input_bytes));
        CUDA_CHECK(cudaMalloc(
            reinterpret_cast<void**>(&device_result), sizeof(unsigned long long)));

        CUDA_CHECK(cudaMemset(device_result, 0, sizeof(unsigned long long)));
        CUDA_CHECK(cudaMemcpy(
            device_input, input.data(), input_bytes, cudaMemcpyHostToDevice));

        const std::size_t block_count =
            (input.size() + baseline_block_size - 1) / baseline_block_size;

        atomic_reduction_kernel<<<static_cast<unsigned int>(block_count),
                                  baseline_block_size>>>(
            device_input, input.size(), device_result);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        unsigned long long host_result = 0;
        CUDA_CHECK(cudaMemcpy(
            &host_result,
            device_result,
            sizeof(unsigned long long),
            cudaMemcpyDeviceToHost));

        const cudaError_t input_free_result = cudaFree(device_input);
        device_input = nullptr;
        CUDA_CHECK(input_free_result);

        const cudaError_t result_free_result = cudaFree(device_result);
        device_result = nullptr;
        CUDA_CHECK(result_free_result);

        return static_cast<std::uint64_t>(host_result);
    } catch (...) {
        if (device_input != nullptr) {
            cudaFree(device_input);
        }
        if (device_result != nullptr) {
            cudaFree(device_result);
        }
        throw;
    }
}

std::uint64_t cuda_reduce_shared(const std::vector<std::uint32_t>& input) {
    if (input.empty()) {
        return 0;
    }

    std::uint32_t* device_input = nullptr;
    unsigned long long* device_result = nullptr;

    try {
        const std::size_t input_bytes = input.size() * sizeof(std::uint32_t);
        CUDA_CHECK(cudaMalloc(
            reinterpret_cast<void**>(&device_input), input_bytes));
        CUDA_CHECK(cudaMalloc(
            reinterpret_cast<void**>(&device_result), sizeof(unsigned long long)));

        CUDA_CHECK(cudaMemset(device_result, 0, sizeof(unsigned long long)));
        CUDA_CHECK(cudaMemcpy(
            device_input, input.data(), input_bytes, cudaMemcpyHostToDevice));

        const std::size_t block_count =
            (input.size() + shared_block_size - 1) / shared_block_size;

        shared_reduction_kernel<<<static_cast<unsigned int>(block_count),
                                  shared_block_size>>>(
            device_input, input.size(), device_result);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        unsigned long long host_result = 0;
        CUDA_CHECK(cudaMemcpy(
            &host_result,
            device_result,
            sizeof(unsigned long long),
            cudaMemcpyDeviceToHost));

        const cudaError_t input_free_result = cudaFree(device_input);
        device_input = nullptr;
        CUDA_CHECK(input_free_result);

        const cudaError_t result_free_result = cudaFree(device_result);
        device_result = nullptr;
        CUDA_CHECK(result_free_result);

        return static_cast<std::uint64_t>(host_result);
    } catch (...) {
        if (device_input != nullptr) {
            cudaFree(device_input);
        }
        if (device_result != nullptr) {
            cudaFree(device_result);
        }
        throw;
    }
}

bool verify_cpu_reduction(const std::string& test_name,
                          const std::vector<std::uint32_t>& input,
                          std::uint64_t expected) {
    const std::uint64_t actual = cpu_reduce(input);
    if (actual != expected) {
        std::cerr << "CPU reduction test failed: " << test_name << '\n'
                  << "  Expected: " << expected << '\n'
                  << "  Actual: " << actual << '\n';
        return false;
    }

    return true;
}

bool run_cpu_reduction_tests() {
    std::size_t test_count = 0;

    const auto run_test = [&test_count](const std::string& test_name,
                                        const std::vector<std::uint32_t>& input,
                                        std::uint64_t expected) {
        ++test_count;
        return verify_cpu_reduction(test_name, input, expected);
    };

    if (!run_test("empty", {}, 0) ||
        !run_test("single element", {7}, 7) ||
        !run_test("small known array", {1, 2, 3, 4, 5}, 15) ||
        !run_test("zero values", {0, 5, 0, 10, 0}, 15)) {
        return false;
    }

    constexpr std::uint32_t max_value =
        std::numeric_limits<std::uint32_t>::max();
    if (!run_test("64-bit accumulator", {max_value, max_value}, 8589934590ULL)) {
        return false;
    }

    constexpr std::uint32_t rng_seed = 0x00C0FFEEU;
    std::mt19937 rng(rng_seed);
    std::uniform_int_distribution<std::uint32_t> distribution(0, 1000);

    constexpr std::array<std::size_t, 12> random_sizes{
        1, 2, 3, 31, 32, 33, 255, 256, 257, 1000, 1003, 100000};

    for (const std::size_t size : random_sizes) {
        std::vector<std::uint32_t> input(size);
        for (std::uint32_t& value : input) {
            value = distribution(rng);
        }

        const std::uint64_t expected =
            std::accumulate(input.begin(), input.end(), std::uint64_t{0});
        if (!run_test("random size " + std::to_string(size), input, expected)) {
            return false;
        }
    }

    std::cout << "CPU reduction tests: PASS (" << test_count << " cases)\n";
    return true;
}

bool verify_cuda_reduction(const std::string& test_name,
                           const std::vector<std::uint32_t>& input) {
    const std::uint64_t cpu_expected = cpu_reduce(input);
    const std::uint64_t cuda_actual = cuda_reduce_atomic(input);

    if (cuda_actual != cpu_expected) {
        std::cerr << "Baseline CUDA reduction test failed: " << test_name << '\n'
                  << "  CPU expected: " << cpu_expected << '\n'
                  << "  CUDA actual: " << cuda_actual << '\n';
        return false;
    }

    return true;
}

bool run_cuda_reduction_tests() {
    std::size_t test_count = 0;

    const auto run_test = [&test_count](const std::string& test_name,
                                        const std::vector<std::uint32_t>& input) {
        ++test_count;
        return verify_cuda_reduction(test_name, input);
    };

    if (!run_test("empty", {}) ||
        !run_test("single element", {7}) ||
        !run_test("small known array", {1, 2, 3, 4, 5}) ||
        !run_test("zero values", {0, 5, 0, 10, 0})) {
        return false;
    }

    constexpr std::uint32_t max_value =
        std::numeric_limits<std::uint32_t>::max();
    if (!run_test("64-bit accumulator", {max_value, max_value})) {
        return false;
    }

    constexpr std::uint32_t rng_seed = 0x00C0FFEEU;
    std::mt19937 rng(rng_seed);
    std::uniform_int_distribution<std::uint32_t> distribution(0, 1000);

    constexpr std::array<std::size_t, 12> random_sizes{
        1, 2, 3, 31, 32, 33, 255, 256, 257, 1000, 1003, 100000};

    for (const std::size_t size : random_sizes) {
        std::vector<std::uint32_t> input(size);
        for (std::uint32_t& value : input) {
            value = distribution(rng);
        }

        if (!run_test("random size " + std::to_string(size), input)) {
            return false;
        }
    }

    std::cout << "Baseline CUDA reduction tests: PASS ("
              << test_count << " cases)\n";
    return true;
}

bool verify_shared_reduction(const std::string& test_name,
                             const std::vector<std::uint32_t>& input) {
    const std::uint64_t cpu_expected = cpu_reduce(input);
    const std::uint64_t shared_actual = cuda_reduce_shared(input);

    if (shared_actual != cpu_expected) {
        std::cerr << "Shared-memory CUDA reduction test failed: "
                  << test_name << '\n'
                  << "  CPU expected: " << cpu_expected << '\n'
                  << "  Shared CUDA actual: " << shared_actual << '\n';
        return false;
    }

    return true;
}

bool run_shared_reduction_tests() {
    std::size_t test_count = 0;

    const auto run_test = [&test_count](const std::string& test_name,
                                        const std::vector<std::uint32_t>& input) {
        ++test_count;
        return verify_shared_reduction(test_name, input);
    };

    if (!run_test("empty", {}) ||
        !run_test("single element", {7}) ||
        !run_test("small known array", {1, 2, 3, 4, 5}) ||
        !run_test("zero values", {0, 5, 0, 10, 0})) {
        return false;
    }

    constexpr std::uint32_t max_value =
        std::numeric_limits<std::uint32_t>::max();
    if (!run_test("64-bit accumulator", {max_value, max_value})) {
        return false;
    }

    constexpr std::uint32_t rng_seed = 0x00C0FFEEU;
    std::mt19937 rng(rng_seed);
    std::uniform_int_distribution<std::uint32_t> distribution(0, 1000);

    constexpr std::array<std::size_t, 12> random_sizes{
        1, 2, 3, 31, 32, 33, 255, 256, 257, 1000, 1003, 100000};

    for (const std::size_t size : random_sizes) {
        std::vector<std::uint32_t> input(size);
        for (std::uint32_t& value : input) {
            value = distribution(rng);
        }

        if (!run_test("random size " + std::to_string(size), input)) {
            return false;
        }
    }

    std::cout << "Shared-memory CUDA reduction tests: PASS ("
              << test_count << " cases)\n";
    return true;
}

bool run_cuda_reduction_comparison(const char* gpu_name) {
    std::mt19937 rng(benchmark_seed);
    std::uniform_int_distribution<std::uint32_t> distribution(0, 1000);

    std::vector<ComparisonBenchmarkResult> results;
    results.reserve(benchmark_sizes.size());

    for (const std::size_t element_count : benchmark_sizes) {
        std::vector<std::uint32_t> input(element_count);
        for (std::uint32_t& value : input) {
            value = distribution(rng);
        }

        const std::uint64_t expected = cpu_reduce(input);
        const std::size_t input_bytes =
            input.size() * sizeof(std::uint32_t);
        const std::size_t block_count =
            (input.size() + baseline_block_size - 1) / baseline_block_size;

        std::uint32_t* device_input = nullptr;
        unsigned long long* baseline_result = nullptr;
        unsigned long long* shared_result = nullptr;

        try {
            CUDA_CHECK(cudaMalloc(
                reinterpret_cast<void**>(&device_input), input_bytes));
            CUDA_CHECK(cudaMalloc(
                reinterpret_cast<void**>(&baseline_result),
                sizeof(unsigned long long)));
            CUDA_CHECK(cudaMalloc(
                reinterpret_cast<void**>(&shared_result),
                sizeof(unsigned long long)));
            CUDA_CHECK(cudaMemcpy(
                device_input,
                input.data(),
                input_bytes,
                cudaMemcpyHostToDevice));

            for (std::size_t iteration = 0;
                 iteration < benchmark_warmup_iterations;
                 ++iteration) {
                CUDA_CHECK(cudaMemset(
                    baseline_result, 0, sizeof(unsigned long long)));
                atomic_reduction_kernel<<<static_cast<unsigned int>(block_count),
                                          baseline_block_size>>>(
                    device_input, input.size(), baseline_result);
                CUDA_CHECK(cudaGetLastError());
                CUDA_CHECK(cudaDeviceSynchronize());
            }

            for (std::size_t iteration = 0;
                 iteration < benchmark_warmup_iterations;
                 ++iteration) {
                CUDA_CHECK(cudaMemset(
                    shared_result, 0, sizeof(unsigned long long)));
                shared_reduction_kernel<<<static_cast<unsigned int>(block_count),
                                          shared_block_size>>>(
                    device_input, input.size(), shared_result);
                CUDA_CHECK(cudaGetLastError());
                CUDA_CHECK(cudaDeviceSynchronize());
            }

            std::vector<float> baseline_timings_ms;
            std::vector<float> shared_timings_ms;
            baseline_timings_ms.reserve(benchmark_measured_iterations);
            shared_timings_ms.reserve(benchmark_measured_iterations);

            CudaEventTimer baseline_timer;
            CudaEventTimer shared_timer;

            const auto measure_baseline = [&]() {
                CUDA_CHECK(cudaMemset(
                    baseline_result, 0, sizeof(unsigned long long)));

                baseline_timer.start();
                atomic_reduction_kernel<<<static_cast<unsigned int>(block_count),
                                          baseline_block_size>>>(
                    device_input, input.size(), baseline_result);
                const float elapsed_ms = baseline_timer.stop();
                CUDA_CHECK(cudaGetLastError());
                baseline_timings_ms.push_back(elapsed_ms);
            };

            const auto measure_shared = [&]() {
                CUDA_CHECK(cudaMemset(
                    shared_result, 0, sizeof(unsigned long long)));

                shared_timer.start();
                shared_reduction_kernel<<<static_cast<unsigned int>(block_count),
                                          shared_block_size>>>(
                    device_input, input.size(), shared_result);
                const float elapsed_ms = shared_timer.stop();
                CUDA_CHECK(cudaGetLastError());
                shared_timings_ms.push_back(elapsed_ms);
            };

            for (std::size_t iteration = 0;
                 iteration < benchmark_measured_iterations;
                 ++iteration) {
                if (iteration % 2 == 0) {
                    measure_baseline();
                    measure_shared();
                } else {
                    measure_shared();
                    measure_baseline();
                }
            }

            unsigned long long baseline_host_result = 0;
            unsigned long long shared_host_result = 0;
            CUDA_CHECK(cudaMemcpy(
                &baseline_host_result,
                baseline_result,
                sizeof(unsigned long long),
                cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(
                &shared_host_result,
                shared_result,
                sizeof(unsigned long long),
                cudaMemcpyDeviceToHost));

            const double baseline_mean_ms =
                std::accumulate(
                    baseline_timings_ms.begin(),
                    baseline_timings_ms.end(),
                    0.0) /
                static_cast<double>(baseline_timings_ms.size());
            const double shared_mean_ms =
                std::accumulate(
                    shared_timings_ms.begin(),
                    shared_timings_ms.end(),
                    0.0) /
                static_cast<double>(shared_timings_ms.size());

            const double baseline_minimum_ms =
                static_cast<double>(*std::min_element(
                    baseline_timings_ms.begin(),
                    baseline_timings_ms.end()));
            const double shared_minimum_ms =
                static_cast<double>(*std::min_element(
                    shared_timings_ms.begin(),
                    shared_timings_ms.end()));
            const double speedup = baseline_mean_ms / shared_mean_ms;

            const cudaError_t input_free_result = cudaFree(device_input);
            device_input = nullptr;
            CUDA_CHECK(input_free_result);

            const cudaError_t baseline_free_result = cudaFree(baseline_result);
            baseline_result = nullptr;
            CUDA_CHECK(baseline_free_result);

            const cudaError_t shared_free_result = cudaFree(shared_result);
            shared_result = nullptr;
            CUDA_CHECK(shared_free_result);

            const std::uint64_t baseline_actual =
                static_cast<std::uint64_t>(baseline_host_result);
            const std::uint64_t shared_actual =
                static_cast<std::uint64_t>(shared_host_result);

            if (baseline_actual != expected || shared_actual != expected) {
                std::cerr << "CUDA comparison validation failed for "
                          << element_count << " elements\n"
                          << "  CPU expected: " << expected << '\n'
                          << "  Baseline actual: " << baseline_actual << '\n'
                          << "  Shared actual: " << shared_actual << '\n';
                return false;
            }

            results.push_back({
                element_count,
                baseline_mean_ms,
                shared_mean_ms,
                speedup,
                baseline_minimum_ms,
                shared_minimum_ms});
        } catch (...) {
            if (device_input != nullptr) {
                cudaFree(device_input);
            }
            if (baseline_result != nullptr) {
                cudaFree(baseline_result);
            }
            if (shared_result != nullptr) {
                cudaFree(shared_result);
            }
            throw;
        }
    }

    std::cout
        << "\nCUDA reduction final comparison\n"
        << "GPU: " << gpu_name << '\n'
        << "Block size: " << baseline_block_size << " threads\n"
        << "Warmups: " << benchmark_warmup_iterations
        << " per implementation\n"
        << "Measured iterations: " << benchmark_measured_iterations
        << " per implementation\n"
        << "Timing: kernel-only CUDA events\n"
        << "Speedup baseline: global atomic reduction\n"
        << "Optimized: block shared-memory reduction + "
           "one global atomic per block\n\n"
        << std::left << std::setw(12) << "Elements"
        << " | " << std::setw(19) << "Baseline mean (ms)"
        << " | " << std::setw(16) << "Shared mean (ms)"
        << " | " << std::setw(17) << "Speedup (B/S)"
        << " | " << std::setw(18) << "Baseline min (ms)"
        << " | " << "Shared min (ms)\n"
        << std::string(119, '-') << '\n';

    std::cout << std::fixed << std::setprecision(6);
    for (const ComparisonBenchmarkResult& result : results) {
        std::cout << std::right << std::setw(12) << result.element_count
                  << " | " << std::setw(19) << result.baseline_mean_ms
                  << " | " << std::setw(16) << result.shared_mean_ms
                  << " | " << std::setw(17) << result.speedup
                  << " | " << std::setw(18) << result.baseline_minimum_ms
                  << " | " << std::setw(15) << result.shared_minimum_ms
                  << '\n';
    }

    std::cout
        << "\nBaseline performs one global atomic contribution per valid "
           "input element.\n"
        << "Shared-memory version reduces inside each block and performs one "
           "global atomic contribution per block.\n";

    return true;
}

int main() {
    try {
        int device_count = 0;
        CUDA_CHECK(cudaGetDeviceCount(&device_count));

        if (device_count < 1) {
            std::cerr << "No CUDA-capable device was found.\n";
            return EXIT_FAILURE;
        }

        CUDA_CHECK(cudaSetDevice(0));

        int device = -1;
        CUDA_CHECK(cudaGetDevice(&device));

        cudaDeviceProp properties{};
        CUDA_CHECK(cudaGetDeviceProperties(&properties, device));

        constexpr double bytes_per_mib = 1024.0 * 1024.0;

        std::cout << "GPU name: " << properties.name << '\n'
                  << "Compute capability: " << properties.major << '.'
                  << properties.minor << '\n'
                  << "Total global memory: "
                  << properties.totalGlobalMem / bytes_per_mib << " MiB\n"
                  << "Multiprocessor count: " << properties.multiProcessorCount << '\n'
                  << "Maximum threads per block: " << properties.maxThreadsPerBlock
                  << '\n';

        if (!run_cpu_reduction_tests()) {
            return EXIT_FAILURE;
        }

        if (!run_cuda_reduction_tests()) {
            return EXIT_FAILURE;
        }

        if (!run_shared_reduction_tests()) {
            return EXIT_FAILURE;
        }

        if (!run_cuda_reduction_comparison(properties.name)) {
            return EXIT_FAILURE;
        }
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
