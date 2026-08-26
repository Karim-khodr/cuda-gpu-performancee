#include "cuda_utils.cuh"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr std::uint32_t kTransposeRngSeed = 0x5452414EU;
constexpr unsigned int kBlockDimX = 16;
constexpr unsigned int kBlockDimY = 16;
constexpr std::size_t kBenchmarkWarmupIterations = 5;
constexpr std::size_t kBenchmarkMeasuredIterations = 20;
constexpr std::uint32_t kBenchmarkRngSeed = 0x5452414EU;

struct MatrixDimensions {
    std::size_t rows;
    std::size_t cols;
};

constexpr std::array<MatrixDimensions, 5> kBenchmarkDimensions{{
    {32, 32},
    {256, 256},
    {1000, 1003},
    {2048, 2048},
    {4096, 4096},
}};

struct TransposeLaunchConfig {
    dim3 block;
    dim3 grid;
};

struct ArchitectureTiming {
    double mean_ms;
    double minimum_ms;
    double effective_bandwidth_gbps;
};

struct ComparisonResult {
    std::size_t rows;
    std::size_t cols;
    ArchitectureTiming naive;
    ArchitectureTiming tiled;
    ArchitectureTiming padded;
    double speedup_naive_over_tiled;
    double speedup_naive_over_padded;
    double speedup_tiled_over_padded;
};

__global__ void transpose_naive_kernel(const float* input,
                                       float* output,
                                       std::size_t rows,
                                       std::size_t cols) {
    const std::size_t col =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t row =
        static_cast<std::size_t>(blockIdx.y) * blockDim.y + threadIdx.y;

    if (row < rows && col < cols) {
        const std::size_t input_index = row * cols + col;
        const std::size_t output_index = col * rows + row;
        output[output_index] = input[input_index];
    }
}

__global__ void transpose_tiled_kernel(const float* input,
                                       float* output,
                                       std::size_t rows,
                                       std::size_t cols) {
    __shared__ float tile[kBlockDimY][kBlockDimX];

    const std::size_t input_col =
        static_cast<std::size_t>(blockIdx.x) * kBlockDimX + threadIdx.x;
    const std::size_t input_row =
        static_cast<std::size_t>(blockIdx.y) * kBlockDimY + threadIdx.y;

    if (input_row < rows && input_col < cols) {
        tile[threadIdx.y][threadIdx.x] =
            input[input_row * cols + input_col];
    } else {
        tile[threadIdx.y][threadIdx.x] = 0.0F;
    }

    __syncthreads();

    const std::size_t output_col =
        static_cast<std::size_t>(blockIdx.y) * kBlockDimY + threadIdx.x;
    const std::size_t output_row =
        static_cast<std::size_t>(blockIdx.x) * kBlockDimX + threadIdx.y;

    if (output_row < cols && output_col < rows) {
        output[output_row * rows + output_col] =
            tile[threadIdx.x][threadIdx.y];
    }
}

__global__ void transpose_tiled_padded_kernel(const float* input,
                                              float* output,
                                              std::size_t rows,
                                              std::size_t cols) {
    __shared__ float tile[kBlockDimY][kBlockDimX + 1];

    const std::size_t input_col =
        static_cast<std::size_t>(blockIdx.x) * kBlockDimX + threadIdx.x;
    const std::size_t input_row =
        static_cast<std::size_t>(blockIdx.y) * kBlockDimY + threadIdx.y;

    if (input_row < rows && input_col < cols) {
        tile[threadIdx.y][threadIdx.x] =
            input[input_row * cols + input_col];
    } else {
        tile[threadIdx.y][threadIdx.x] = 0.0F;
    }

    __syncthreads();

    const std::size_t output_col =
        static_cast<std::size_t>(blockIdx.y) * kBlockDimY + threadIdx.x;
    const std::size_t output_row =
        static_cast<std::size_t>(blockIdx.x) * kBlockDimX + threadIdx.y;

    if (output_row < cols && output_col < rows) {
        output[output_row * rows + output_col] =
            tile[threadIdx.x][threadIdx.y];
    }
}

std::size_t checked_element_count(std::size_t rows, std::size_t cols) {
    if (rows != 0 && cols > std::numeric_limits<std::size_t>::max() / rows) {
        throw std::overflow_error("matrix dimensions overflow size_t");
    }
    return rows * cols;
}

std::vector<float> transpose_cpu(const std::vector<float>& input,
                                 std::size_t rows,
                                 std::size_t cols) {
    const std::size_t element_count = checked_element_count(rows, cols);
    if (input.size() != element_count) {
        throw std::invalid_argument("input size does not match matrix dimensions");
    }

    std::vector<float> output(element_count);
    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t col = 0; col < cols; ++col) {
            output[col * rows + row] = input[row * cols + col];
        }
    }
    return output;
}

std::size_t checked_byte_count(std::size_t element_count) {
    if (element_count > std::numeric_limits<std::size_t>::max() / sizeof(float)) {
        throw std::overflow_error("matrix byte count overflows size_t");
    }
    return element_count * sizeof(float);
}

TransposeLaunchConfig make_transpose_launch_config(std::size_t rows,
                                                   std::size_t cols) {
    if (rows == 0 || cols == 0) {
        throw std::invalid_argument(
            "CUDA launch dimensions must be non-zero");
    }

    const std::size_t grid_x_size = 1 + (cols - 1) / kBlockDimX;
    const std::size_t grid_y_size = 1 + (rows - 1) / kBlockDimY;
    if (grid_x_size > std::numeric_limits<unsigned int>::max() ||
        grid_y_size > std::numeric_limits<unsigned int>::max()) {
        throw std::overflow_error("CUDA grid dimensions exceed dim3 range");
    }

    return {
        dim3(kBlockDimX, kBlockDimY),
        dim3(static_cast<unsigned int>(grid_x_size),
             static_cast<unsigned int>(grid_y_size)),
    };
}

void launch_naive_transpose(const float* device_input,
                            float* device_output,
                            std::size_t rows,
                            std::size_t cols,
                            const TransposeLaunchConfig& config) {
    transpose_naive_kernel<<<config.grid, config.block>>>(
        device_input, device_output, rows, cols);
}

void launch_tiled_transpose(const float* device_input,
                            float* device_output,
                            std::size_t rows,
                            std::size_t cols,
                            const TransposeLaunchConfig& config) {
    transpose_tiled_kernel<<<config.grid, config.block>>>(
        device_input, device_output, rows, cols);
}

void launch_tiled_padded_transpose(const float* device_input,
                                   float* device_output,
                                   std::size_t rows,
                                   std::size_t cols,
                                   const TransposeLaunchConfig& config) {
    transpose_tiled_padded_kernel<<<config.grid, config.block>>>(
        device_input, device_output, rows, cols);
}

std::vector<float> transpose_cuda_naive(const std::vector<float>& input,
                                        std::size_t rows,
                                        std::size_t cols) {
    const std::size_t element_count = checked_element_count(rows, cols);
    if (input.size() != element_count) {
        throw std::invalid_argument("input size does not match matrix dimensions");
    }

    std::vector<float> output(element_count);
    if (element_count == 0) {
        return output;
    }

    const std::size_t byte_count = checked_byte_count(element_count);
    const TransposeLaunchConfig launch_config =
        make_transpose_launch_config(rows, cols);

    float* device_input = nullptr;
    float* device_output = nullptr;

    try {
        CUDA_CHECK(cudaMalloc(
            reinterpret_cast<void**>(&device_input), byte_count));
        CUDA_CHECK(cudaMalloc(
            reinterpret_cast<void**>(&device_output), byte_count));
        CUDA_CHECK(cudaMemcpy(
            device_input,
            input.data(),
            byte_count,
            cudaMemcpyHostToDevice));

        launch_naive_transpose(
            device_input, device_output, rows, cols, launch_config);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(
            output.data(),
            device_output,
            byte_count,
            cudaMemcpyDeviceToHost));

        const cudaError_t input_free_result = cudaFree(device_input);
        device_input = nullptr;
        CUDA_CHECK(input_free_result);

        const cudaError_t output_free_result = cudaFree(device_output);
        device_output = nullptr;
        CUDA_CHECK(output_free_result);

        return output;
    } catch (...) {
        if (device_input != nullptr) {
            cudaFree(device_input);
        }
        if (device_output != nullptr) {
            cudaFree(device_output);
        }
        throw;
    }
}

std::vector<float> transpose_cuda_tiled(const std::vector<float>& input,
                                        std::size_t rows,
                                        std::size_t cols) {
    const std::size_t element_count = checked_element_count(rows, cols);
    if (input.size() != element_count) {
        throw std::invalid_argument("input size does not match matrix dimensions");
    }

    std::vector<float> output(element_count);
    if (element_count == 0) {
        return output;
    }

    const std::size_t byte_count = checked_byte_count(element_count);
    const TransposeLaunchConfig launch_config =
        make_transpose_launch_config(rows, cols);

    float* device_input = nullptr;
    float* device_output = nullptr;

    try {
        CUDA_CHECK(cudaMalloc(
            reinterpret_cast<void**>(&device_input), byte_count));
        CUDA_CHECK(cudaMalloc(
            reinterpret_cast<void**>(&device_output), byte_count));
        CUDA_CHECK(cudaMemcpy(
            device_input,
            input.data(),
            byte_count,
            cudaMemcpyHostToDevice));

        launch_tiled_transpose(
            device_input, device_output, rows, cols, launch_config);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(
            output.data(),
            device_output,
            byte_count,
            cudaMemcpyDeviceToHost));

        const cudaError_t input_free_result = cudaFree(device_input);
        device_input = nullptr;
        CUDA_CHECK(input_free_result);

        const cudaError_t output_free_result = cudaFree(device_output);
        device_output = nullptr;
        CUDA_CHECK(output_free_result);

        return output;
    } catch (...) {
        if (device_input != nullptr) {
            cudaFree(device_input);
        }
        if (device_output != nullptr) {
            cudaFree(device_output);
        }
        throw;
    }
}

std::vector<float> transpose_cuda_tiled_padded(
    const std::vector<float>& input,
    std::size_t rows,
    std::size_t cols) {
    const std::size_t element_count = checked_element_count(rows, cols);
    if (input.size() != element_count) {
        throw std::invalid_argument("input size does not match matrix dimensions");
    }

    std::vector<float> output(element_count);
    if (element_count == 0) {
        return output;
    }

    const std::size_t byte_count = checked_byte_count(element_count);
    const TransposeLaunchConfig launch_config =
        make_transpose_launch_config(rows, cols);

    float* device_input = nullptr;
    float* device_output = nullptr;

    try {
        CUDA_CHECK(cudaMalloc(
            reinterpret_cast<void**>(&device_input), byte_count));
        CUDA_CHECK(cudaMalloc(
            reinterpret_cast<void**>(&device_output), byte_count));
        CUDA_CHECK(cudaMemcpy(
            device_input,
            input.data(),
            byte_count,
            cudaMemcpyHostToDevice));

        launch_tiled_padded_transpose(
            device_input, device_output, rows, cols, launch_config);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(
            output.data(),
            device_output,
            byte_count,
            cudaMemcpyDeviceToHost));

        const cudaError_t input_free_result = cudaFree(device_input);
        device_input = nullptr;
        CUDA_CHECK(input_free_result);

        const cudaError_t output_free_result = cudaFree(device_output);
        device_output = nullptr;
        CUDA_CHECK(output_free_result);

        return output;
    } catch (...) {
        if (device_input != nullptr) {
            cudaFree(device_input);
        }
        if (device_output != nullptr) {
            cudaFree(device_output);
        }
        throw;
    }
}

bool compare_with_expected(const std::string& name,
                           std::size_t rows,
                           std::size_t cols,
                           const std::vector<float>& actual,
                           const std::vector<float>& expected) {
    if (actual.size() != expected.size()) {
        std::cerr << "FAIL " << name << " (" << rows << "x" << cols
                  << "): expected " << expected.size() << " output elements, got "
                  << actual.size() << '\n';
        return false;
    }

    for (std::size_t index = 0; index < expected.size(); ++index) {
        if (actual[index] != expected[index]) {
            const std::size_t output_cols = rows;
            const std::size_t output_row = output_cols == 0 ? 0 : index / output_cols;
            const std::size_t output_col = output_cols == 0 ? 0 : index % output_cols;
            std::cerr << "FAIL " << name << " (" << rows << "x" << cols
                      << "): output index " << index << " (row " << output_row
                      << ", col " << output_col << "), expected " << expected[index]
                      << ", got " << actual[index] << '\n';
            return false;
        }
    }
    return true;
}

bool run_known_case(const std::string& name,
                    std::size_t rows,
                    std::size_t cols,
                    const std::vector<float>& input,
                    const std::vector<float>& expected) {
    try {
        return compare_with_expected(name, rows, cols,
                                     transpose_cpu(input, rows, cols), expected);
    } catch (const std::exception& error) {
        std::cerr << "FAIL " << name << " (" << rows << "x" << cols
                  << "): unexpected exception: " << error.what() << '\n';
        return false;
    }
}

std::vector<float> make_random_matrix(std::size_t rows,
                                      std::size_t cols,
                                      std::mt19937& rng) {
    std::uniform_int_distribution<int> distribution(-1000, 1000);
    std::vector<float> input(checked_element_count(rows, cols));
    for (float& value : input) {
        value = static_cast<float>(distribution(rng));
    }
    return input;
}

bool run_generated_case(std::size_t rows, std::size_t cols, std::mt19937& rng) {
    try {
        const std::vector<float> input = make_random_matrix(rows, cols, rng);
        const std::vector<float> actual = transpose_cpu(input, rows, cols);

        for (std::size_t row = 0; row < rows; ++row) {
            for (std::size_t col = 0; col < cols; ++col) {
                const std::size_t input_index = row * cols + col;
                const std::size_t output_index = col * rows + row;
                if (actual[output_index] != input[input_index]) {
                    std::cerr << "FAIL generated (" << rows << "x" << cols
                              << "): output index " << output_index << " (row " << col
                              << ", col " << row << "), expected " << input[input_index]
                              << ", got " << actual[output_index] << '\n';
                    return false;
                }
            }
        }
        return true;
    } catch (const std::exception& error) {
        std::cerr << "FAIL generated (" << rows << "x" << cols
                  << "): unexpected exception: " << error.what() << '\n';
        return false;
    }
}

bool run_invalid_size_case() {
    try {
        static_cast<void>(transpose_cpu({1.0F, 2.0F, 3.0F, 4.0F, 5.0F}, 2, 3));
    } catch (const std::invalid_argument&) {
        return true;
    } catch (const std::exception& error) {
        std::cerr << "FAIL invalid-size validation: wrong exception: " << error.what()
                  << '\n';
        return false;
    }

    std::cerr << "FAIL invalid-size validation: inconsistent input was accepted\n";
    return false;
}

struct TestSummary {
    std::size_t passed = 0;
    std::size_t total = 0;
};

bool run_cuda_case(const std::string& name,
                   std::size_t rows,
                   std::size_t cols,
                   const std::vector<float>& input) {
    try {
        const std::vector<float> expected = transpose_cpu(input, rows, cols);
        const std::vector<float> actual =
            transpose_cuda_naive(input, rows, cols);
        return compare_with_expected(
            "naive CUDA " + name, rows, cols, actual, expected);
    } catch (const std::exception& error) {
        std::cerr << "FAIL naive CUDA " << name << " (" << rows << "x" << cols
                  << "): unexpected exception: " << error.what() << '\n';
        return false;
    }
}

bool run_cuda_generated_case(std::size_t rows,
                             std::size_t cols,
                             std::mt19937& rng) {
    try {
        return run_cuda_case(
            "generated", rows, cols, make_random_matrix(rows, cols, rng));
    } catch (const std::exception& error) {
        std::cerr << "FAIL naive CUDA generated (" << rows << "x" << cols
                  << "): input generation failed: " << error.what() << '\n';
        return false;
    }
}

bool run_tiled_cuda_case(const std::string& name,
                         std::size_t rows,
                         std::size_t cols,
                         const std::vector<float>& input) {
    try {
        const std::vector<float> expected = transpose_cpu(input, rows, cols);
        const std::vector<float> actual =
            transpose_cuda_tiled(input, rows, cols);
        return compare_with_expected(
            "tiled CUDA " + name, rows, cols, actual, expected);
    } catch (const std::exception& error) {
        std::cerr << "FAIL tiled CUDA " << name << " (" << rows << "x" << cols
                  << "): unexpected exception: " << error.what() << '\n';
        return false;
    }
}

bool run_tiled_cuda_generated_case(std::size_t rows,
                                   std::size_t cols,
                                   std::mt19937& rng) {
    try {
        return run_tiled_cuda_case(
            "generated", rows, cols, make_random_matrix(rows, cols, rng));
    } catch (const std::exception& error) {
        std::cerr << "FAIL tiled CUDA generated (" << rows << "x" << cols
                  << "): input generation failed: " << error.what() << '\n';
        return false;
    }
}

bool run_padded_tiled_cuda_case(const std::string& name,
                                std::size_t rows,
                                std::size_t cols,
                                const std::vector<float>& input) {
    try {
        const std::vector<float> expected = transpose_cpu(input, rows, cols);
        const std::vector<float> actual =
            transpose_cuda_tiled_padded(input, rows, cols);
        return compare_with_expected(
            "padded tiled CUDA " + name, rows, cols, actual, expected);
    } catch (const std::exception& error) {
        std::cerr << "FAIL padded tiled CUDA " << name << " ("
                  << rows << "x" << cols
                  << "): unexpected exception: " << error.what() << '\n';
        return false;
    }
}

bool run_padded_tiled_cuda_generated_case(std::size_t rows,
                                          std::size_t cols,
                                          std::mt19937& rng) {
    try {
        return run_padded_tiled_cuda_case(
            "generated", rows, cols, make_random_matrix(rows, cols, rng));
    } catch (const std::exception& error) {
        std::cerr << "FAIL padded tiled CUDA generated ("
                  << rows << "x" << cols
                  << "): input generation failed: " << error.what() << '\n';
        return false;
    }
}

TestSummary run_cpu_transpose_tests() {
    TestSummary summary;
    const auto record = [&summary](bool result) {
        ++summary.total;
        if (result) {
            ++summary.passed;
        }
    };

    record(run_known_case("empty", 0, 0, {}, {}));
    record(run_known_case("single element", 1, 1, {42.0F}, {42.0F}));
    record(run_known_case("single row", 1, 7,
                          {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F},
                          {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F}));
    record(run_known_case("single column", 7, 1,
                          {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F},
                          {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F}));
    record(run_known_case("known rectangular", 2, 3,
                          {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F},
                          {1.0F, 4.0F, 2.0F, 5.0F, 3.0F, 6.0F}));

    std::mt19937 rng(kTransposeRngSeed);
    record(run_generated_case(4, 4, rng));
    record(run_generated_case(17, 31, rng));
    record(run_generated_case(31, 32, rng));
    record(run_generated_case(32, 32, rng));
    record(run_generated_case(33, 31, rng));
    record(run_generated_case(255, 257, rng));
    record(run_generated_case(1000, 1003, rng));
    record(run_invalid_size_case());

    return summary;
}

TestSummary run_naive_cuda_transpose_tests() {
    TestSummary summary;
    const auto record = [&summary](bool result) {
        ++summary.total;
        if (result) {
            ++summary.passed;
        }
    };

    record(run_cuda_case("empty", 0, 0, {}));
    record(run_cuda_case("single element", 1, 1, {42.0F}));
    record(run_cuda_case("single row", 1, 7,
                         {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F}));
    record(run_cuda_case("single column", 7, 1,
                         {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F}));
    record(run_cuda_case("known rectangular", 2, 3,
                         {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F}));

    std::mt19937 rng(kTransposeRngSeed);
    record(run_cuda_generated_case(4, 4, rng));
    record(run_cuda_generated_case(17, 31, rng));
    record(run_cuda_generated_case(31, 32, rng));
    record(run_cuda_generated_case(32, 32, rng));
    record(run_cuda_generated_case(33, 31, rng));
    record(run_cuda_generated_case(255, 257, rng));
    record(run_cuda_generated_case(1000, 1003, rng));

    return summary;
}

TestSummary run_tiled_cuda_transpose_tests() {
    TestSummary summary;
    const auto record = [&summary](bool result) {
        ++summary.total;
        if (result) {
            ++summary.passed;
        }
    };

    record(run_tiled_cuda_case("empty", 0, 0, {}));
    record(run_tiled_cuda_case("single element", 1, 1, {42.0F}));
    record(run_tiled_cuda_case("single row", 1, 7,
                               {1.0F, 2.0F, 3.0F, 4.0F,
                                5.0F, 6.0F, 7.0F}));
    record(run_tiled_cuda_case("single column", 7, 1,
                               {1.0F, 2.0F, 3.0F, 4.0F,
                                5.0F, 6.0F, 7.0F}));
    record(run_tiled_cuda_case("known rectangular", 2, 3,
                               {1.0F, 2.0F, 3.0F,
                                4.0F, 5.0F, 6.0F}));

    std::mt19937 rng(kTransposeRngSeed);
    record(run_tiled_cuda_generated_case(4, 4, rng));
    record(run_tiled_cuda_generated_case(17, 31, rng));
    record(run_tiled_cuda_generated_case(31, 32, rng));
    record(run_tiled_cuda_generated_case(32, 32, rng));
    record(run_tiled_cuda_generated_case(33, 31, rng));
    record(run_tiled_cuda_generated_case(255, 257, rng));
    record(run_tiled_cuda_generated_case(1000, 1003, rng));

    return summary;
}

TestSummary run_padded_tiled_cuda_transpose_tests() {
    TestSummary summary;
    const auto record = [&summary](bool result) {
        ++summary.total;
        if (result) {
            ++summary.passed;
        }
    };

    record(run_padded_tiled_cuda_case("empty", 0, 0, {}));
    record(run_padded_tiled_cuda_case("single element", 1, 1, {42.0F}));
    record(run_padded_tiled_cuda_case(
        "single row", 1, 7,
        {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F}));
    record(run_padded_tiled_cuda_case(
        "single column", 7, 1,
        {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F}));
    record(run_padded_tiled_cuda_case(
        "known rectangular", 2, 3,
        {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F}));

    std::mt19937 rng(kTransposeRngSeed);
    record(run_padded_tiled_cuda_generated_case(4, 4, rng));
    record(run_padded_tiled_cuda_generated_case(17, 31, rng));
    record(run_padded_tiled_cuda_generated_case(31, 32, rng));
    record(run_padded_tiled_cuda_generated_case(32, 32, rng));
    record(run_padded_tiled_cuda_generated_case(33, 31, rng));
    record(run_padded_tiled_cuda_generated_case(255, 257, rng));
    record(run_padded_tiled_cuda_generated_case(1000, 1003, rng));

    return summary;
}

ArchitectureTiming summarize_timings(
    const std::vector<float>& timings_ms,
    std::size_t element_count) {
    if (timings_ms.empty()) {
        throw std::invalid_argument("cannot summarize empty timing samples");
    }

    const double mean_ms =
        std::accumulate(timings_ms.begin(), timings_ms.end(), 0.0) /
        static_cast<double>(timings_ms.size());
    const double minimum_ms =
        static_cast<double>(
            *std::min_element(timings_ms.begin(), timings_ms.end()));
    if (mean_ms <= 0.0) {
        throw std::runtime_error("non-positive mean CUDA event time");
    }

    const double useful_bytes =
        2.0 * static_cast<double>(element_count) *
        static_cast<double>(sizeof(float));
    return {
        mean_ms,
        minimum_ms,
        useful_bytes / (mean_ms * 1000000.0),
    };
}

bool benchmark_comparison_case(const MatrixDimensions& dimensions,
                               std::mt19937& rng,
                               ComparisonResult& result) {
    const std::vector<float> input =
        make_random_matrix(dimensions.rows, dimensions.cols, rng);
    const std::vector<float> expected =
        transpose_cpu(input, dimensions.rows, dimensions.cols);
    std::vector<float> output_naive(input.size());
    std::vector<float> output_tiled(input.size());
    std::vector<float> output_padded(input.size());

    const std::size_t element_count =
        checked_element_count(dimensions.rows, dimensions.cols);
    const std::size_t byte_count = checked_byte_count(element_count);
    const TransposeLaunchConfig launch_config =
        make_transpose_launch_config(dimensions.rows, dimensions.cols);

    float* device_input = nullptr;
    float* device_output_naive = nullptr;
    float* device_output_tiled = nullptr;
    float* device_output_padded = nullptr;
    std::vector<float> naive_timings_ms;
    std::vector<float> tiled_timings_ms;
    std::vector<float> padded_timings_ms;
    naive_timings_ms.reserve(kBenchmarkMeasuredIterations);
    tiled_timings_ms.reserve(kBenchmarkMeasuredIterations);
    padded_timings_ms.reserve(kBenchmarkMeasuredIterations);

    try {
        CUDA_CHECK(cudaMalloc(
            reinterpret_cast<void**>(&device_input), byte_count));
        CUDA_CHECK(cudaMalloc(
            reinterpret_cast<void**>(&device_output_naive), byte_count));
        CUDA_CHECK(cudaMalloc(
            reinterpret_cast<void**>(&device_output_tiled), byte_count));
        CUDA_CHECK(cudaMalloc(
            reinterpret_cast<void**>(&device_output_padded), byte_count));
        CUDA_CHECK(cudaMemcpy(
            device_input,
            input.data(),
            byte_count,
            cudaMemcpyHostToDevice));

        const auto warmup_naive = [&]() {
            launch_naive_transpose(
                device_input,
                device_output_naive,
                dimensions.rows,
                dimensions.cols,
                launch_config);
            CUDA_CHECK(cudaGetLastError());
        };
        const auto warmup_tiled = [&]() {
            launch_tiled_transpose(
                device_input,
                device_output_tiled,
                dimensions.rows,
                dimensions.cols,
                launch_config);
            CUDA_CHECK(cudaGetLastError());
        };
        const auto warmup_padded = [&]() {
            launch_tiled_padded_transpose(
                device_input,
                device_output_padded,
                dimensions.rows,
                dimensions.cols,
                launch_config);
            CUDA_CHECK(cudaGetLastError());
        };

        for (std::size_t iteration = 0;
             iteration < kBenchmarkWarmupIterations;
             ++iteration) {
            switch (iteration % 3) {
                case 0:
                    warmup_naive();
                    warmup_tiled();
                    warmup_padded();
                    break;
                case 1:
                    warmup_tiled();
                    warmup_padded();
                    warmup_naive();
                    break;
                default:
                    warmup_padded();
                    warmup_naive();
                    warmup_tiled();
                    break;
            }
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        CudaEventTimer naive_timer;
        CudaEventTimer tiled_timer;
        CudaEventTimer padded_timer;

        const auto measure_naive = [&]() {
            naive_timer.start();
            launch_naive_transpose(
                device_input,
                device_output_naive,
                dimensions.rows,
                dimensions.cols,
                launch_config);
            const float elapsed_ms = naive_timer.stop();
            CUDA_CHECK(cudaGetLastError());
            naive_timings_ms.push_back(elapsed_ms);
        };

        const auto measure_tiled = [&]() {
            tiled_timer.start();
            launch_tiled_transpose(
                device_input,
                device_output_tiled,
                dimensions.rows,
                dimensions.cols,
                launch_config);
            const float elapsed_ms = tiled_timer.stop();
            CUDA_CHECK(cudaGetLastError());
            tiled_timings_ms.push_back(elapsed_ms);
        };

        const auto measure_padded = [&]() {
            padded_timer.start();
            launch_tiled_padded_transpose(
                device_input,
                device_output_padded,
                dimensions.rows,
                dimensions.cols,
                launch_config);
            const float elapsed_ms = padded_timer.stop();
            CUDA_CHECK(cudaGetLastError());
            padded_timings_ms.push_back(elapsed_ms);
        };

        for (std::size_t iteration = 0;
             iteration < kBenchmarkMeasuredIterations;
             ++iteration) {
            switch (iteration % 3) {
                case 0:
                    measure_naive();
                    measure_tiled();
                    measure_padded();
                    break;
                case 1:
                    measure_tiled();
                    measure_padded();
                    measure_naive();
                    break;
                default:
                    measure_padded();
                    measure_naive();
                    measure_tiled();
                    break;
            }
        }

        CUDA_CHECK(cudaMemcpy(
            output_naive.data(),
            device_output_naive,
            byte_count,
            cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(
            output_tiled.data(),
            device_output_tiled,
            byte_count,
            cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(
            output_padded.data(),
            device_output_padded,
            byte_count,
            cudaMemcpyDeviceToHost));

        const cudaError_t input_free_result = cudaFree(device_input);
        device_input = nullptr;
        CUDA_CHECK(input_free_result);

        const cudaError_t naive_free_result = cudaFree(device_output_naive);
        device_output_naive = nullptr;
        CUDA_CHECK(naive_free_result);

        const cudaError_t tiled_free_result = cudaFree(device_output_tiled);
        device_output_tiled = nullptr;
        CUDA_CHECK(tiled_free_result);

        const cudaError_t padded_free_result = cudaFree(device_output_padded);
        device_output_padded = nullptr;
        CUDA_CHECK(padded_free_result);
    } catch (...) {
        if (device_input != nullptr) {
            cudaFree(device_input);
        }
        if (device_output_naive != nullptr) {
            cudaFree(device_output_naive);
        }
        if (device_output_tiled != nullptr) {
            cudaFree(device_output_tiled);
        }
        if (device_output_padded != nullptr) {
            cudaFree(device_output_padded);
        }
        throw;
    }

    if (!compare_with_expected(
            "naive CUDA comparison benchmark",
            dimensions.rows,
            dimensions.cols,
            output_naive,
            expected)) {
        return false;
    }
    if (!compare_with_expected(
            "tiled CUDA comparison benchmark",
            dimensions.rows,
            dimensions.cols,
            output_tiled,
            expected)) {
        return false;
    }
    if (!compare_with_expected(
            "padded tiled CUDA comparison benchmark",
            dimensions.rows,
            dimensions.cols,
            output_padded,
            expected)) {
        return false;
    }

    const ArchitectureTiming naive =
        summarize_timings(naive_timings_ms, element_count);
    const ArchitectureTiming tiled =
        summarize_timings(tiled_timings_ms, element_count);
    const ArchitectureTiming padded =
        summarize_timings(padded_timings_ms, element_count);
    result = {
        dimensions.rows,
        dimensions.cols,
        naive,
        tiled,
        padded,
        naive.mean_ms / tiled.mean_ms,
        naive.mean_ms / padded.mean_ms,
        tiled.mean_ms / padded.mean_ms,
    };
    return true;
}

bool run_comparison_benchmark(std::vector<ComparisonResult>& results) {
    std::mt19937 rng(kBenchmarkRngSeed);
    results.clear();
    results.reserve(kBenchmarkDimensions.size());

    for (const MatrixDimensions& dimensions : kBenchmarkDimensions) {
        try {
            ComparisonResult result{};
            if (!benchmark_comparison_case(dimensions, rng, result)) {
                return false;
            }
            results.push_back(result);
        } catch (const std::exception& error) {
            std::cerr << "FAIL three-way CUDA transpose benchmark ("
                      << dimensions.rows << "x" << dimensions.cols
                      << "): " << error.what() << '\n';
            return false;
        }
    }
    return true;
}

void print_comparison_results(
    const std::vector<ComparisonResult>& results) {
    std::cout
        << "\nThree-way CUDA transpose benchmark\n"
        << "Block: " << kBlockDimX << "x" << kBlockDimY << " threads\n"
        << "Unpadded shared memory: " << kBlockDimY << "x" << kBlockDimX
        << " floats ("
        << kBlockDimY * kBlockDimX * sizeof(float) << " bytes/block)\n"
        << "Padded shared memory: " << kBlockDimY << "x"
        << kBlockDimX + 1 << " floats ("
        << kBlockDimY * (kBlockDimX + 1) * sizeof(float)
        << " bytes/block)\n"
        << "Warmups: " << kBenchmarkWarmupIterations
        << " per implementation\n"
        << "Measured iterations: " << kBenchmarkMeasuredIterations
        << " per implementation\n"
        << "Timing: kernel-only CUDA events\n"
        << "Order: deterministic three-way rotation\n"
        << "Input seed: 0x" << std::hex << std::uppercase
        << kBenchmarkRngSeed << std::dec
        << "\n\nLatency and effective bandwidth\n"
        << std::left << std::setw(13) << "Rows x Cols"
        << " | " << std::right << std::setw(10) << "N mean"
        << " | " << std::setw(10) << "T mean"
        << " | " << std::setw(10) << "P mean"
        << " | " << std::setw(10) << "N min"
        << " | " << std::setw(10) << "T min"
        << " | " << std::setw(10) << "P min"
        << " | " << std::setw(11) << "N GB/s"
        << " | " << std::setw(11) << "T GB/s"
        << " | " << std::setw(11) << "P GB/s"
        << '\n'
        << std::string(139, '-') << '\n';

    for (const ComparisonResult& result : results) {
        const std::string dimensions =
            std::to_string(result.rows) + "x" + std::to_string(result.cols);
        std::cout
            << std::left << std::setw(13) << dimensions
            << " | " << std::right << std::fixed << std::setprecision(6)
            << std::setw(10) << result.naive.mean_ms
            << " | " << std::setw(10) << result.tiled.mean_ms
            << " | " << std::setw(10) << result.padded.mean_ms
            << " | " << std::setw(10) << result.naive.minimum_ms
            << " | " << std::setw(10) << result.tiled.minimum_ms
            << " | " << std::setw(10) << result.padded.minimum_ms
            << " | " << std::setprecision(3)
            << std::setw(11) << result.naive.effective_bandwidth_gbps
            << " | " << std::setw(11) << result.tiled.effective_bandwidth_gbps
            << " | " << std::setw(11) << result.padded.effective_bandwidth_gbps
            << '\n';
    }

    std::cout
        << "\nSpeedups (ratio > 1 means denominator architecture is faster)\n"
        << std::left << std::setw(13) << "Rows x Cols"
        << " | " << std::right << std::setw(13) << "N/T"
        << " | " << std::setw(13) << "N/P"
        << " | " << std::setw(13) << "T/P"
        << '\n'
        << std::string(61, '-') << '\n';

    for (const ComparisonResult& result : results) {
        const std::string dimensions =
            std::to_string(result.rows) + "x" + std::to_string(result.cols);
        std::cout
            << std::left << std::setw(13) << dimensions
            << " | " << std::right << std::fixed << std::setprecision(3)
            << std::setw(13) << result.speedup_naive_over_tiled
            << " | " << std::setw(13) << result.speedup_naive_over_padded
            << " | " << std::setw(13) << result.speedup_tiled_over_padded
            << '\n';
    }
}

}  // namespace

int main() {
    const TestSummary cpu = run_cpu_transpose_tests();
    const TestSummary naive_cuda = run_naive_cuda_transpose_tests();
    const TestSummary tiled_cuda = run_tiled_cuda_transpose_tests();
    const TestSummary padded_tiled_cuda = run_padded_tiled_cuda_transpose_tests();

    std::cout << "Deterministic RNG seed: 0x" << std::hex << std::uppercase
              << kTransposeRngSeed << std::dec << '\n';

    if (cpu.passed == cpu.total) {
        std::cout << "CPU transpose tests: PASS (" << cpu.total << " cases)\n";
    } else {
        std::cout << "CPU transpose tests: FAIL (" << cpu.passed << "/"
                  << cpu.total << " cases passed)\n";
    }

    if (naive_cuda.passed == naive_cuda.total) {
        std::cout << "Naive CUDA transpose tests: PASS (" << naive_cuda.total
                  << " cases)\n";
    } else {
        std::cout << "Naive CUDA transpose tests: FAIL (" << naive_cuda.passed
                  << "/" << naive_cuda.total << " cases passed)\n";
    }

    if (tiled_cuda.passed == tiled_cuda.total) {
        std::cout << "Tiled CUDA transpose tests: PASS (" << tiled_cuda.total
                  << " cases)\n";
    } else {
        std::cout << "Tiled CUDA transpose tests: FAIL (" << tiled_cuda.passed
                  << "/" << tiled_cuda.total << " cases passed)\n";
    }

    if (padded_tiled_cuda.passed == padded_tiled_cuda.total) {
        std::cout << "Padded tiled CUDA transpose tests: PASS ("
                  << padded_tiled_cuda.total << " cases)\n";
    } else {
        std::cout << "Padded tiled CUDA transpose tests: FAIL ("
                  << padded_tiled_cuda.passed << "/"
                  << padded_tiled_cuda.total << " cases passed)\n";
    }

    if (cpu.passed != cpu.total ||
        naive_cuda.passed != naive_cuda.total ||
        tiled_cuda.passed != tiled_cuda.total ||
        padded_tiled_cuda.passed != padded_tiled_cuda.total) {
        return 1;
    }

    std::vector<ComparisonResult> benchmark_results;
    if (!run_comparison_benchmark(benchmark_results)) {
        return 1;
    }
    print_comparison_results(benchmark_results);
    return 0;
}
