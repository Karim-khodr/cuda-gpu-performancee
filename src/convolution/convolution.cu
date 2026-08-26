#include "cuda_utils.cuh"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr std::size_t kFilterWidth = 3;
constexpr std::size_t kFilterRadius = 1;
constexpr float kAbsoluteTolerance = 1.0e-5F;
constexpr float kRelativeTolerance = 1.0e-5F;
constexpr std::uint32_t kRandomSeed = 0x434F4E56U;
constexpr std::uint32_t kCudaCorrectnessSeed = 0x43554441U;
constexpr std::uint32_t kBenchmarkSeed = 0x42415345U;
constexpr unsigned int kBlockDimX = 16;
constexpr unsigned int kBlockDimY = 16;
constexpr std::size_t kSharedTileWidth =
    kBlockDimX + 2 * kFilterRadius;
constexpr std::size_t kSharedTileHeight =
    kBlockDimY + 2 * kFilterRadius;
constexpr std::size_t kThreadsPerBlock = kBlockDimX * kBlockDimY;
constexpr std::size_t kSharedTileElements =
    kSharedTileWidth * kSharedTileHeight;
constexpr std::size_t kBenchmarkWarmupIterations = 5;
constexpr std::size_t kBenchmarkMeasuredIterations = 20;

using Filter3x3 = std::array<float, kFilterWidth * kFilterWidth>;

constexpr Filter3x3 kIdentityFilter{
    0.0F, 0.0F, 0.0F,
    0.0F, 1.0F, 0.0F,
    0.0F, 0.0F, 0.0F,
};

constexpr Filter3x3 kCrossFilter{
    0.0F, 1.0F, 0.0F,
    1.0F, 1.0F, 1.0F,
    0.0F, 1.0F, 0.0F,
};

constexpr Filter3x3 kAsymmetricFilter{
    1.0F, 2.0F, 3.0F,
    4.0F, 5.0F, 6.0F,
    7.0F, 8.0F, 9.0F,
};
constexpr Filter3x3 kBenchmarkFilter{
    0.125F, -0.25F, 0.375F,
    -0.5F, 0.75F, 0.25F,
    -0.125F, 0.5F, -0.375F,
};

struct ImageDimensions {
    std::size_t height;
    std::size_t width;
};

constexpr std::array<ImageDimensions, 5> kBenchmarkDimensions{{
    {32, 32},
    {256, 256},
    {1000, 1003},
    {2048, 2048},
    {4096, 4096},
}};

struct ConvolutionLaunchConfig {
    dim3 block;
    dim3 grid;
};

struct ArchitectureTiming {
    double mean_ms;
    double minimum_ms;
    double output_throughput_mpixels_per_second;
};

struct BenchmarkResult {
    ImageDimensions dimensions;
    ArchitectureTiming naive;
    ArchitectureTiming tiled;
    double naive_over_tiled_speedup;
    double mean_time_reduction_percent;
    double naive_maximum_absolute_error;
    double tiled_maximum_absolute_error;
};

__global__ void convolution_naive_kernel(const float* input,
                                         const float* filter,
                                         float* output,
                                         std::size_t height,
                                         std::size_t width) {
    const std::size_t column =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t row =
        static_cast<std::size_t>(blockIdx.y) * blockDim.y + threadIdx.y;

    if (row >= height || column >= width) {
        return;
    }

    float sum = 0.0F;
    for (std::size_t filter_row = 0; filter_row < kFilterWidth; ++filter_row) {
        for (std::size_t filter_column = 0;
             filter_column < kFilterWidth;
             ++filter_column) {
            const std::ptrdiff_t image_row =
                static_cast<std::ptrdiff_t>(row) +
                static_cast<std::ptrdiff_t>(filter_row) -
                static_cast<std::ptrdiff_t>(kFilterRadius);
            const std::ptrdiff_t image_column =
                static_cast<std::ptrdiff_t>(column) +
                static_cast<std::ptrdiff_t>(filter_column) -
                static_cast<std::ptrdiff_t>(kFilterRadius);

            if (image_row < 0 || image_column < 0 ||
                image_row >= static_cast<std::ptrdiff_t>(height) ||
                image_column >= static_cast<std::ptrdiff_t>(width)) {
                continue;
            }

            const std::size_t image_index =
                static_cast<std::size_t>(image_row) * width +
                static_cast<std::size_t>(image_column);
            const std::size_t filter_index =
                filter_row * kFilterWidth + filter_column;
            sum += input[image_index] * filter[filter_index];
        }
    }

    output[row * width + column] = sum;
}
__global__ void convolution_tiled_kernel(const float* input,
                                         const float* filter,
                                         float* output,
                                         std::size_t height,
                                         std::size_t width) {
    __shared__ float tile[kSharedTileHeight][kSharedTileWidth];

    const std::size_t block_output_row =
        static_cast<std::size_t>(blockIdx.y) * kBlockDimY;
    const std::size_t block_output_column =
        static_cast<std::size_t>(blockIdx.x) * kBlockDimX;
    const std::size_t output_row = block_output_row + threadIdx.y;
    const std::size_t output_column = block_output_column + threadIdx.x;
    const std::size_t linear_thread =
        static_cast<std::size_t>(threadIdx.y) * blockDim.x + threadIdx.x;

    for (std::size_t tile_index = linear_thread;
         tile_index < kSharedTileElements;
         tile_index += kThreadsPerBlock) {
        const std::size_t tile_row = tile_index / kSharedTileWidth;
        const std::size_t tile_column = tile_index % kSharedTileWidth;
        const std::ptrdiff_t global_input_row =
            static_cast<std::ptrdiff_t>(block_output_row) +
            static_cast<std::ptrdiff_t>(tile_row) -
            static_cast<std::ptrdiff_t>(kFilterRadius);
        const std::ptrdiff_t global_input_column =
            static_cast<std::ptrdiff_t>(block_output_column) +
            static_cast<std::ptrdiff_t>(tile_column) -
            static_cast<std::ptrdiff_t>(kFilterRadius);

        if (global_input_row >= 0 && global_input_column >= 0 &&
            global_input_row < static_cast<std::ptrdiff_t>(height) &&
            global_input_column < static_cast<std::ptrdiff_t>(width)) {
            const std::size_t input_index =
                static_cast<std::size_t>(global_input_row) * width +
                static_cast<std::size_t>(global_input_column);
            tile[tile_row][tile_column] = input[input_index];
        } else {
            tile[tile_row][tile_column] = 0.0F;
        }
    }

    __syncthreads();

    if (output_row >= height || output_column >= width) {
        return;
    }

    float sum = 0.0F;
    for (std::size_t filter_row = 0; filter_row < kFilterWidth; ++filter_row) {
        for (std::size_t filter_column = 0;
             filter_column < kFilterWidth;
             ++filter_column) {
            const std::size_t filter_index =
                filter_row * kFilterWidth + filter_column;
            sum += tile[threadIdx.y + filter_row]
                       [threadIdx.x + filter_column] *
                   filter[filter_index];
        }
    }

    output[output_row * width + output_column] = sum;
}


std::size_t checked_element_count(std::size_t height, std::size_t width) {
    if (height != 0 && width > std::numeric_limits<std::size_t>::max() / height) {
        throw std::overflow_error("image dimensions overflow size_t");
    }
    return height * width;
}
std::size_t checked_byte_count(std::size_t element_count) {
    if (element_count >
        std::numeric_limits<std::size_t>::max() / sizeof(float)) {
        throw std::overflow_error("image byte count overflows size_t");
    }
    return element_count * sizeof(float);
}

ConvolutionLaunchConfig make_convolution_launch_config(std::size_t height,
                                                       std::size_t width) {
    if (height == 0 || width == 0) {
        throw std::invalid_argument("CUDA launch dimensions must be non-zero");
    }

    const std::size_t grid_x_size = 1 + (width - 1) / kBlockDimX;
    const std::size_t grid_y_size = 1 + (height - 1) / kBlockDimY;
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

void launch_naive_convolution(const float* device_input,
                              const float* device_filter,
                              float* device_output,
                              std::size_t height,
                              std::size_t width,
                              const ConvolutionLaunchConfig& config) {
    convolution_naive_kernel<<<config.grid, config.block>>>(
        device_input, device_filter, device_output, height, width);
}
void launch_tiled_convolution(const float* device_input,
                              const float* device_filter,
                              float* device_output,
                              std::size_t height,
                              std::size_t width,
                              const ConvolutionLaunchConfig& config) {
    convolution_tiled_kernel<<<config.grid, config.block>>>(
        device_input, device_filter, device_output, height, width);
}


class DeviceFloatBuffer {
public:
    explicit DeviceFloatBuffer(std::size_t element_count) {
        if (element_count != 0) {
            CUDA_CHECK(cudaMalloc(
                reinterpret_cast<void**>(&data_),
                checked_byte_count(element_count)));
        }
    }

    ~DeviceFloatBuffer() noexcept {
        if (data_ != nullptr) {
            cudaFree(data_);
        }
    }

    DeviceFloatBuffer(const DeviceFloatBuffer&) = delete;
    DeviceFloatBuffer& operator=(const DeviceFloatBuffer&) = delete;

    float* get() {
        return data_;
    }

    void release_checked() {
        if (data_ != nullptr) {
            const cudaError_t result = cudaFree(data_);
            data_ = nullptr;
            CUDA_CHECK(result);
        }
    }

private:
    float* data_ = nullptr;
};


std::vector<float> convolve_cpu(const std::vector<float>& input,
                                std::size_t height,
                                std::size_t width,
                                const Filter3x3& filter) {
    const std::size_t element_count = checked_element_count(height, width);
    if (input.size() != element_count) {
        throw std::invalid_argument("input size does not match height * width");
    }

    std::vector<float> output(element_count, 0.0F);

    for (std::size_t row = 0; row < height; ++row) {
        for (std::size_t column = 0; column < width; ++column) {
            float sum = 0.0F;

            for (std::size_t filter_row = 0; filter_row < kFilterWidth; ++filter_row) {
                for (std::size_t filter_column = 0;
                     filter_column < kFilterWidth;
                     ++filter_column) {
                    const std::ptrdiff_t image_row =
                        static_cast<std::ptrdiff_t>(row) +
                        static_cast<std::ptrdiff_t>(filter_row) -
                        static_cast<std::ptrdiff_t>(kFilterRadius);
                    const std::ptrdiff_t image_column =
                        static_cast<std::ptrdiff_t>(column) +
                        static_cast<std::ptrdiff_t>(filter_column) -
                        static_cast<std::ptrdiff_t>(kFilterRadius);

                    if (image_row < 0 || image_column < 0 ||
                        image_row >= static_cast<std::ptrdiff_t>(height) ||
                        image_column >= static_cast<std::ptrdiff_t>(width)) {
                        continue;
                    }

                    const std::size_t image_index =
                        static_cast<std::size_t>(image_row) * width +
                        static_cast<std::size_t>(image_column);
                    const std::size_t filter_index =
                        filter_row * kFilterWidth + filter_column;
                    sum += input[image_index] * filter[filter_index];
                }
            }

            output[row * width + column] = sum;
        }
    }

    return output;
}

bool nearly_equal(float actual, float expected) {
    return std::abs(actual - expected) <=
           kAbsoluteTolerance + kRelativeTolerance * std::abs(expected);
}

struct ComparisonResult {
    bool passed = true;
    double maximum_absolute_error = 0.0;
    std::size_t maximum_error_index = 0;
    std::string diagnostic;
};

ComparisonResult compare_images(const std::vector<float>& actual,
                                const std::vector<float>& expected,
                                std::size_t height,
                                std::size_t width) {
    if (actual.size() != expected.size()) {
        std::ostringstream message;
        message << "size mismatch: expected " << expected.size()
                << ", actual " << actual.size();
        return {false, 0.0, 0, message.str()};
    }

    ComparisonResult result;
    for (std::size_t index = 0; index < expected.size(); ++index) {
        const double absolute_error = std::abs(
            static_cast<double>(actual[index]) -
            static_cast<double>(expected[index]));
        if (!std::isfinite(absolute_error)) {
            result.maximum_absolute_error =
                std::numeric_limits<double>::infinity();
            result.maximum_error_index = index;
        } else if (absolute_error > result.maximum_absolute_error) {
            result.maximum_absolute_error = absolute_error;
            result.maximum_error_index = index;
        }

        if (!nearly_equal(actual[index], expected[index]) && result.passed) {
            const std::size_t row = width == 0 ? 0 : index / width;
            const std::size_t column = width == 0 ? 0 : index % width;
            std::ostringstream message;
            message << std::setprecision(9)
                    << "index=" << index
                    << ", row=" << row
                    << ", column=" << column
                    << ", expected=" << expected[index]
                    << ", actual=" << actual[index]
                    << ", absolute_error=" << absolute_error;
            result.passed = false;
            result.diagnostic = message.str();
        }
    }

    const std::size_t expected_count = checked_element_count(height, width);
    if (actual.size() != expected_count) {
        result.passed = false;
        result.diagnostic =
            "output dimensions do not match the requested image dimensions";
    }
    return result;
}

struct TestResult {
    std::string dimensions;
    std::string filter;
    std::string purpose;
    bool passed;
    std::string diagnostic;
};

TestResult run_expected_image_test(const std::string& dimensions,
                                   const std::string& filter_name,
                                   const std::string& purpose,
                                   const std::vector<float>& input,
                                   std::size_t height,
                                   std::size_t width,
                                   const Filter3x3& filter,
                                   const std::vector<float>& expected) {
    const std::vector<float> actual = convolve_cpu(input, height, width, filter);
    const ComparisonResult comparison = compare_images(actual, expected, height, width);
    return {dimensions, filter_name, purpose, comparison.passed, comparison.diagnostic};
}

std::vector<float> make_pattern(std::size_t height, std::size_t width) {
    std::vector<float> values(checked_element_count(height, width));
    for (std::size_t index = 0; index < values.size(); ++index) {
        const int value = static_cast<int>((index * 7U + 3U) % 19U) - 9;
        values[index] = static_cast<float>(value) * 0.25F;
    }
    return values;
}

TestResult run_identity_test(std::size_t height, std::size_t width) {
    const std::vector<float> input = make_pattern(height, width);
    return run_expected_image_test(std::to_string(height) + "x" + std::to_string(width),
                                   "identity",
                                   "identity invariant",
                                   input,
                                   height,
                                   width,
                                   kIdentityFilter,
                                   input);
}

TestResult run_constant_cross_test(std::size_t height, std::size_t width) {
    const std::vector<float> input(checked_element_count(height, width), 1.0F);
    std::vector<float> expected(input.size(), 0.0F);
    for (std::size_t row = 0; row < height; ++row) {
        for (std::size_t column = 0; column < width; ++column) {
            int contributors = 1;
            contributors += row > 0 ? 1 : 0;
            contributors += row + 1 < height ? 1 : 0;
            contributors += column > 0 ? 1 : 0;
            contributors += column + 1 < width ? 1 : 0;
            expected[row * width + column] = static_cast<float>(contributors);
        }
    }

    return run_expected_image_test(std::to_string(height) + "x" + std::to_string(width),
                                   "cross",
                                   "constant-image neighbor count",
                                   input,
                                   height,
                                   width,
                                   kCrossFilter,
                                   expected);
}

TestResult run_impulse_test(std::size_t height, std::size_t width) {
    std::vector<float> input(checked_element_count(height, width), 0.0F);
    std::vector<float> expected(input.size(), 0.0F);
    const std::size_t impulse_row = height / 2;
    const std::size_t impulse_column = width / 2;
    input[impulse_row * width + impulse_column] = 1.0F;

    // A unit impulse exposes the coefficient orientation directly. With the
    // authoritative no-flip indexing convention, the visible footprint is
    // the filter in reverse row/column order around the impulse.
    for (std::size_t row_offset = 0; row_offset < kFilterWidth; ++row_offset) {
        for (std::size_t column_offset = 0;
             column_offset < kFilterWidth;
             ++column_offset) {
            const std::size_t output_row = impulse_row + row_offset - kFilterRadius;
            const std::size_t output_column =
                impulse_column + column_offset - kFilterRadius;
            const std::size_t filter_row = kFilterWidth - 1 - row_offset;
            const std::size_t filter_column = kFilterWidth - 1 - column_offset;
            expected[output_row * width + output_column] =
                kAsymmetricFilter[filter_row * kFilterWidth + filter_column];
        }
    }

    return run_expected_image_test(std::to_string(height) + "x" + std::to_string(width),
                                   "non-symmetric",
                                   "unit-impulse coefficient orientation",
                                   input,
                                   height,
                                   width,
                                   kAsymmetricFilter,
                                   expected);
}

TestResult run_randomized_test() {
    constexpr std::size_t height = 255;
    constexpr std::size_t width = 257;
    std::mt19937 generator(kRandomSeed);
    std::vector<float> input(checked_element_count(height, width));
    for (float& value : input) {
        value = static_cast<float>(static_cast<int>(generator() % 17U) - 8) * 0.25F;
    }

    Filter3x3 filter{};
    for (float& coefficient : filter) {
        coefficient =
            static_cast<float>(static_cast<int>(generator() % 9U) - 4) * 0.125F;
    }

    const std::vector<float> actual = convolve_cpu(input, height, width, filter);

    // Fixed regression signatures for the seed above. All generated values
    // are binary fractions, so these sums are exactly representable here.
    constexpr double expected_sum = -424.0;
    constexpr double expected_weighted_sum = -21913.28125;
    double sum = 0.0;
    double weighted_sum = 0.0;
    for (std::size_t index = 0; index < actual.size(); ++index) {
        sum += actual[index];
        weighted_sum += actual[index] * static_cast<double>(index % 97U + 1U);
    }

    const bool passed = sum == expected_sum && weighted_sum == expected_weighted_sum;
    std::ostringstream diagnostic;
    if (!passed) {
        diagnostic << std::setprecision(17)
                   << "signature mismatch: sum=" << sum
                   << ", weighted_sum=" << weighted_sum;
    }
    return {"255x257",
            "seeded non-symmetric random",
            "fixed-seed whole-output signatures",
            passed,
            diagnostic.str()};
}

struct CudaTestResult {
    std::string dimensions;
    std::string filter;
    bool passed;
    double maximum_absolute_error;
    std::string diagnostic;
};

std::vector<float> convolve_cuda_naive(const std::vector<float>& input,
                                       std::size_t height,
                                       std::size_t width,
                                       const Filter3x3& filter) {
    const std::size_t element_count = checked_element_count(height, width);
    if (input.size() != element_count) {
        throw std::invalid_argument("input size does not match height * width");
    }

    std::vector<float> output(element_count, 0.0F);
    if (element_count == 0) {
        return output;
    }

    const std::size_t image_byte_count = checked_byte_count(element_count);
    const std::size_t filter_byte_count = checked_byte_count(filter.size());
    const ConvolutionLaunchConfig config =
        make_convolution_launch_config(height, width);

    DeviceFloatBuffer device_input(element_count);
    DeviceFloatBuffer device_filter(filter.size());
    DeviceFloatBuffer device_output(element_count);

    CUDA_CHECK(cudaMemcpy(device_input.get(),
                          input.data(),
                          image_byte_count,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_filter.get(),
                          filter.data(),
                          filter_byte_count,
                          cudaMemcpyHostToDevice));

    launch_naive_convolution(device_input.get(),
                             device_filter.get(),
                             device_output.get(),
                             height,
                             width,
                             config);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(output.data(),
                          device_output.get(),
                          image_byte_count,
                          cudaMemcpyDeviceToHost));

    device_output.release_checked();
    device_filter.release_checked();
    device_input.release_checked();
    return output;
}
std::vector<float> convolve_cuda_tiled(const std::vector<float>& input,
                                       std::size_t height,
                                       std::size_t width,
                                       const Filter3x3& filter) {
    const std::size_t element_count = checked_element_count(height, width);
    if (input.size() != element_count) {
        throw std::invalid_argument("input size does not match height * width");
    }

    std::vector<float> output(element_count, 0.0F);
    if (element_count == 0) {
        return output;
    }

    const std::size_t image_byte_count = checked_byte_count(element_count);
    const std::size_t filter_byte_count = checked_byte_count(filter.size());
    const ConvolutionLaunchConfig config =
        make_convolution_launch_config(height, width);

    DeviceFloatBuffer device_input(element_count);
    DeviceFloatBuffer device_filter(filter.size());
    DeviceFloatBuffer device_output(element_count);

    CUDA_CHECK(cudaMemcpy(device_input.get(),
                          input.data(),
                          image_byte_count,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_filter.get(),
                          filter.data(),
                          filter_byte_count,
                          cudaMemcpyHostToDevice));

    launch_tiled_convolution(device_input.get(),
                             device_filter.get(),
                             device_output.get(),
                             height,
                             width,
                             config);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(output.data(),
                          device_output.get(),
                          image_byte_count,
                          cudaMemcpyDeviceToHost));

    device_output.release_checked();
    device_filter.release_checked();
    device_input.release_checked();
    return output;
}


CudaTestResult run_cuda_case(std::size_t height,
                             std::size_t width,
                             const std::string& filter_name,
                             const std::vector<float>& input,
                             const Filter3x3& filter) {
    const std::vector<float> expected =
        convolve_cpu(input, height, width, filter);
    const std::vector<float> actual =
        convolve_cuda_naive(input, height, width, filter);
    const ComparisonResult comparison =
        compare_images(actual, expected, height, width);

    return {
        std::to_string(height) + "x" + std::to_string(width),
        filter_name,
        comparison.passed,
        comparison.maximum_absolute_error,
        comparison.diagnostic,
    };
}

std::vector<CudaTestResult> run_cuda_correctness_tests() {
    std::vector<CudaTestResult> results;
    results.push_back(run_cuda_case(0, 0, "identity", {}, kIdentityFilter));
    results.push_back(
        run_cuda_case(1, 1, "non-symmetric", make_pattern(1, 1), kAsymmetricFilter));
    results.push_back(
        run_cuda_case(1, 7, "cross", make_pattern(1, 7), kCrossFilter));
    results.push_back(
        run_cuda_case(7, 1, "cross", make_pattern(7, 1), kCrossFilter));
    results.push_back(
        run_cuda_case(2, 2, "non-symmetric", make_pattern(2, 2), kAsymmetricFilter));
    results.push_back(
        run_cuda_case(2, 3, "identity", make_pattern(2, 3), kIdentityFilter));
    results.push_back(
        run_cuda_case(3, 3, "non-symmetric", make_pattern(3, 3), kAsymmetricFilter));
    results.push_back(
        run_cuda_case(15, 15, "identity", make_pattern(15, 15), kIdentityFilter));
    results.push_back(
        run_cuda_case(16, 16, "cross", make_pattern(16, 16), kCrossFilter));
    results.push_back(
        run_cuda_case(17, 17, "non-symmetric", make_pattern(17, 17), kAsymmetricFilter));
    results.push_back(
        run_cuda_case(17, 31, "identity", make_pattern(17, 31), kIdentityFilter));
    results.push_back(
        run_cuda_case(31, 32, "cross", make_pattern(31, 32), kCrossFilter));
    results.push_back(
        run_cuda_case(32, 32, "non-symmetric", make_pattern(32, 32), kAsymmetricFilter));
    results.push_back(
        run_cuda_case(33, 31, "identity", make_pattern(33, 31), kIdentityFilter));

    constexpr std::size_t random_height = 255;
    constexpr std::size_t random_width = 257;
    std::mt19937 generator(kCudaCorrectnessSeed);
    std::vector<float> random_input(
        checked_element_count(random_height, random_width));
    for (float& value : random_input) {
        value =
            static_cast<float>(static_cast<int>(generator() % 33U) - 16) *
            0.125F;
    }
    Filter3x3 random_filter{};
    for (float& coefficient : random_filter) {
        coefficient =
            static_cast<float>(static_cast<int>(generator() % 17U) - 8) *
            0.0625F;
    }
    results.push_back(run_cuda_case(random_height,
                                    random_width,
                                    "seeded randomized",
                                    random_input,
                                    random_filter));
    return results;
}

CudaTestResult run_tiled_cuda_case(std::size_t height,
                                   std::size_t width,
                                   const std::string& filter_name,
                                   const std::vector<float>& input,
                                   const Filter3x3& filter) {
    const std::vector<float> expected =
        convolve_cpu(input, height, width, filter);
    const std::vector<float> actual =
        convolve_cuda_tiled(input, height, width, filter);
    const ComparisonResult comparison =
        compare_images(actual, expected, height, width);

    return {
        std::to_string(height) + "x" + std::to_string(width),
        filter_name,
        comparison.passed,
        comparison.maximum_absolute_error,
        comparison.diagnostic,
    };
}

std::vector<CudaTestResult> run_tiled_cuda_correctness_tests() {
    std::vector<CudaTestResult> results;
    results.push_back(
        run_tiled_cuda_case(0, 0, "identity", {}, kIdentityFilter));
    results.push_back(run_tiled_cuda_case(
        1, 1, "non-symmetric", make_pattern(1, 1), kAsymmetricFilter));
    results.push_back(
        run_tiled_cuda_case(1, 7, "cross", make_pattern(1, 7), kCrossFilter));
    results.push_back(
        run_tiled_cuda_case(7, 1, "cross", make_pattern(7, 1), kCrossFilter));
    results.push_back(run_tiled_cuda_case(
        2, 2, "non-symmetric", make_pattern(2, 2), kAsymmetricFilter));
    results.push_back(run_tiled_cuda_case(
        2, 3, "identity", make_pattern(2, 3), kIdentityFilter));
    results.push_back(run_tiled_cuda_case(
        3, 3, "non-symmetric", make_pattern(3, 3), kAsymmetricFilter));
    results.push_back(run_tiled_cuda_case(
        15, 15, "identity", make_pattern(15, 15), kIdentityFilter));
    results.push_back(run_tiled_cuda_case(
        16, 16, "cross", make_pattern(16, 16), kCrossFilter));
    results.push_back(run_tiled_cuda_case(
        17, 17, "non-symmetric", make_pattern(17, 17), kAsymmetricFilter));
    results.push_back(run_tiled_cuda_case(
        17, 31, "identity", make_pattern(17, 31), kIdentityFilter));
    results.push_back(run_tiled_cuda_case(
        31, 32, "cross", make_pattern(31, 32), kCrossFilter));
    results.push_back(run_tiled_cuda_case(
        32, 32, "non-symmetric", make_pattern(32, 32), kAsymmetricFilter));
    results.push_back(run_tiled_cuda_case(
        33, 31, "identity", make_pattern(33, 31), kIdentityFilter));

    constexpr std::size_t random_height = 255;
    constexpr std::size_t random_width = 257;
    std::mt19937 generator(kCudaCorrectnessSeed);
    std::vector<float> random_input(
        checked_element_count(random_height, random_width));
    for (float& value : random_input) {
        value =
            static_cast<float>(static_cast<int>(generator() % 33U) - 16) *
            0.125F;
    }
    Filter3x3 random_filter{};
    for (float& coefficient : random_filter) {
        coefficient =
            static_cast<float>(static_cast<int>(generator() % 17U) - 8) *
            0.0625F;
    }
    results.push_back(run_tiled_cuda_case(random_height,
                                          random_width,
                                          "seeded randomized",
                                          random_input,
                                          random_filter));
    return results;
}

std::vector<float> make_benchmark_input(const ImageDimensions& dimensions) {
    const std::uint32_t dimension_seed =
        static_cast<std::uint32_t>(
            dimensions.height * 131U + dimensions.width * 17U);
    std::mt19937 generator(kBenchmarkSeed ^ dimension_seed);
    std::vector<float> input(
        checked_element_count(dimensions.height, dimensions.width));
    for (float& value : input) {
        value =
            static_cast<float>(static_cast<int>(generator() % 33U) - 16) *
            0.0625F;
    }
    return input;
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
    const double minimum_ms = static_cast<double>(
        *std::min_element(timings_ms.begin(), timings_ms.end()));
    if (mean_ms <= 0.0) {
        throw std::runtime_error("non-positive mean CUDA event time");
    }

    return {
        mean_ms,
        minimum_ms,
        static_cast<double>(element_count) / (mean_ms * 1000.0),
    };
}

BenchmarkResult benchmark_comparison_case(
    const ImageDimensions& dimensions) {
    const std::vector<float> input = make_benchmark_input(dimensions);
    const std::vector<float> expected =
        convolve_cpu(input,
                     dimensions.height,
                     dimensions.width,
                     kBenchmarkFilter);
    std::vector<float> naive_output(input.size(), 0.0F);
    std::vector<float> tiled_output(input.size(), 0.0F);

    const std::size_t element_count =
        checked_element_count(dimensions.height, dimensions.width);
    const std::size_t image_byte_count = checked_byte_count(element_count);
    const std::size_t filter_byte_count =
        checked_byte_count(kBenchmarkFilter.size());
    const ConvolutionLaunchConfig config =
        make_convolution_launch_config(dimensions.height, dimensions.width);

    DeviceFloatBuffer device_input(element_count);
    DeviceFloatBuffer device_filter(kBenchmarkFilter.size());
    DeviceFloatBuffer device_naive_output(element_count);
    DeviceFloatBuffer device_tiled_output(element_count);

    CUDA_CHECK(cudaMemcpy(device_input.get(),
                          input.data(),
                          image_byte_count,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_filter.get(),
                          kBenchmarkFilter.data(),
                          filter_byte_count,
                          cudaMemcpyHostToDevice));

    launch_naive_convolution(device_input.get(),
                             device_filter.get(),
                             device_naive_output.get(),
                             dimensions.height,
                             dimensions.width,
                             config);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(naive_output.data(),
                          device_naive_output.get(),
                          image_byte_count,
                          cudaMemcpyDeviceToHost));

    launch_tiled_convolution(device_input.get(),
                             device_filter.get(),
                             device_tiled_output.get(),
                             dimensions.height,
                             dimensions.width,
                             config);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(tiled_output.data(),
                          device_tiled_output.get(),
                          image_byte_count,
                          cudaMemcpyDeviceToHost));

    const ComparisonResult naive_comparison =
        compare_images(naive_output,
                       expected,
                       dimensions.height,
                       dimensions.width);
    const ComparisonResult tiled_comparison =
        compare_images(tiled_output,
                       expected,
                       dimensions.height,
                       dimensions.width);
    if (!naive_comparison.passed || !tiled_comparison.passed) {
        std::ostringstream message;
        message << "benchmark correctness failed for "
                << dimensions.height << "x" << dimensions.width;
        if (!naive_comparison.passed) {
            message << "; naive: " << naive_comparison.diagnostic;
        }
        if (!tiled_comparison.passed) {
            message << "; tiled: " << tiled_comparison.diagnostic;
        }
        throw std::runtime_error(message.str());
    }

    const auto warmup_naive = [&]() {
        launch_naive_convolution(device_input.get(),
                                 device_filter.get(),
                                 device_naive_output.get(),
                                 dimensions.height,
                                 dimensions.width,
                                 config);
        CUDA_CHECK(cudaGetLastError());
    };
    const auto warmup_tiled = [&]() {
        launch_tiled_convolution(device_input.get(),
                                 device_filter.get(),
                                 device_tiled_output.get(),
                                 dimensions.height,
                                 dimensions.width,
                                 config);
        CUDA_CHECK(cudaGetLastError());
    };

    for (std::size_t iteration = 0;
         iteration < kBenchmarkWarmupIterations;
         ++iteration) {
        if (iteration % 2 == 0) {
            warmup_naive();
            warmup_tiled();
        } else {
            warmup_tiled();
            warmup_naive();
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> naive_timings_ms;
    std::vector<float> tiled_timings_ms;
    naive_timings_ms.reserve(kBenchmarkMeasuredIterations);
    tiled_timings_ms.reserve(kBenchmarkMeasuredIterations);
    CudaEventTimer naive_timer;
    CudaEventTimer tiled_timer;

    const auto measure_naive = [&]() {
        naive_timer.start();
        launch_naive_convolution(device_input.get(),
                                 device_filter.get(),
                                 device_naive_output.get(),
                                 dimensions.height,
                                 dimensions.width,
                                 config);
        CUDA_CHECK(cudaGetLastError());
        naive_timings_ms.push_back(naive_timer.stop());
    };
    const auto measure_tiled = [&]() {
        tiled_timer.start();
        launch_tiled_convolution(device_input.get(),
                                 device_filter.get(),
                                 device_tiled_output.get(),
                                 dimensions.height,
                                 dimensions.width,
                                 config);
        CUDA_CHECK(cudaGetLastError());
        tiled_timings_ms.push_back(tiled_timer.stop());
    };

    for (std::size_t iteration = 0;
         iteration < kBenchmarkMeasuredIterations;
         ++iteration) {
        if (iteration % 2 == 0) {
            measure_naive();
            measure_tiled();
        } else {
            measure_tiled();
            measure_naive();
        }
    }

    device_tiled_output.release_checked();
    device_naive_output.release_checked();
    device_filter.release_checked();
    device_input.release_checked();

    const ArchitectureTiming naive =
        summarize_timings(naive_timings_ms, element_count);
    const ArchitectureTiming tiled =
        summarize_timings(tiled_timings_ms, element_count);
    return {
        dimensions,
        naive,
        tiled,
        naive.mean_ms / tiled.mean_ms,
        (naive.mean_ms - tiled.mean_ms) / naive.mean_ms * 100.0,
        naive_comparison.maximum_absolute_error,
        tiled_comparison.maximum_absolute_error,
    };
}

std::vector<BenchmarkResult> run_comparison_benchmarks() {
    std::vector<BenchmarkResult> results;
    results.reserve(kBenchmarkDimensions.size());
    for (const ImageDimensions& dimensions : kBenchmarkDimensions) {
        results.push_back(benchmark_comparison_case(dimensions));
    }
    return results;
}

void print_cuda_results(const std::vector<CudaTestResult>& results,
                        const std::string& heading) {
    std::cout << "\n" << heading << "\n";
    std::cout << "CUDA RNG seed: 0x" << std::hex << std::uppercase
              << kCudaCorrectnessSeed << std::dec << std::nouppercase << "\n";
    std::cout << std::left << std::setw(12) << "Dimensions"
              << std::setw(24) << "Filter"
              << std::setw(22) << "Max absolute error"
              << "Result\n";
    std::cout << std::string(66, '-') << "\n";

    for (const CudaTestResult& result : results) {
        std::cout << std::left << std::setw(12) << result.dimensions
                  << std::setw(24) << result.filter
                  << std::setw(22) << std::scientific
                  << result.maximum_absolute_error
                  << (result.passed ? "PASS" : "FAIL") << "\n";
        if (!result.passed && !result.diagnostic.empty()) {
            std::cout << "  " << result.diagnostic << "\n";
        }
    }
    std::cout << std::defaultfloat;
}

void print_benchmark_results(const std::vector<BenchmarkResult>& results,
                             const std::string& run_label) {
    std::cout << "\n" << run_label
              << ": naive vs shared-memory tiled convolution\n";
    std::cout << "Block: " << kBlockDimX << "x" << kBlockDimY << " threads\n";
    std::cout << "Warmups: " << kBenchmarkWarmupIterations
              << " per architecture\n";
    std::cout << "Measured launches: " << kBenchmarkMeasuredIterations
              << " per architecture\n";
    std::cout << "Timing: kernel-only CUDA events\n";
    std::cout << "Order: even pairs naive->tiled; odd pairs tiled->naive\n";
    std::cout << "CPU, allocation, H2D, D2H, and correctness: outside timing\n";
    std::cout << "Speedup = naive mean / tiled mean; time change > 0 means "
                 "tiled reduced mean time\n\n";

    std::cout << std::left << std::setw(14) << "Dimensions"
              << std::right << std::setw(13) << "N mean"
              << std::setw(13) << "N min"
              << std::setw(13) << "T mean"
              << std::setw(13) << "T min"
              << std::setw(12) << "N/T"
              << std::setw(14) << "Time change %"
              << std::setw(17) << "N MPixels/s"
              << std::setw(17) << "T MPixels/s"
              << "\n";
    std::cout << std::string(126, '-') << "\n";

    std::cout << std::fixed << std::setprecision(6);
    for (const BenchmarkResult& result : results) {
        const std::string dimensions =
            std::to_string(result.dimensions.height) + "x" +
            std::to_string(result.dimensions.width);
        std::cout << std::left << std::setw(14) << dimensions
                  << std::right << std::setw(13) << result.naive.mean_ms
                  << std::setw(13) << result.naive.minimum_ms
                  << std::setw(13) << result.tiled.mean_ms
                  << std::setw(13) << result.tiled.minimum_ms
                  << std::setw(12) << result.naive_over_tiled_speedup
                  << std::setw(14) << result.mean_time_reduction_percent
                  << std::setw(17)
                  << result.naive.output_throughput_mpixels_per_second
                  << std::setw(17)
                  << result.tiled.output_throughput_mpixels_per_second
                  << "\n";
    }
    std::cout << std::defaultfloat;
}

void print_results(const std::vector<TestResult>& results) {
    std::cout << "CPU convolution correctness tests\n";
    std::cout << "Deterministic RNG seed: 0x" << std::hex << std::uppercase
              << kRandomSeed << std::dec << std::nouppercase << "\n";
    std::cout << "Comparison: abs(actual - expected) <= "
              << kAbsoluteTolerance << " + " << kRelativeTolerance
              << " * abs(expected)\n\n";
    std::cout << std::left << std::setw(12) << "Dimensions"
              << std::setw(31) << "Filter"
              << std::setw(40) << "Purpose"
              << "Result\n";
    std::cout << std::string(91, '-') << "\n";

    for (const TestResult& result : results) {
        std::cout << std::left << std::setw(12) << result.dimensions
                  << std::setw(31) << result.filter
                  << std::setw(40) << result.purpose
                  << (result.passed ? "PASS" : "FAIL") << "\n";
        if (!result.passed && !result.diagnostic.empty()) {
            std::cout << "  " << result.diagnostic << "\n";
        }
    }
}

}  // namespace

int main() {
    try {
        std::vector<TestResult> results;

        results.push_back(run_expected_image_test(
            "0x0", "identity", "empty image", {}, 0, 0, kIdentityFilter, {}));
        results.push_back(run_expected_image_test(
            "1x1", "non-symmetric", "manual center coefficient", {2.0F}, 1, 1,
            kAsymmetricFilter, {10.0F}));
        results.push_back(run_expected_image_test(
            "1x7", "cross", "manual top/bottom boundaries",
            {1, 2, 3, 4, 5, 6, 7}, 1, 7, kCrossFilter,
            {3, 6, 9, 12, 15, 18, 13}));
        results.push_back(run_expected_image_test(
            "7x1", "cross", "manual left/right boundaries",
            {1, 2, 3, 4, 5, 6, 7}, 7, 1, kCrossFilter,
            {3, 6, 9, 12, 15, 18, 13}));
        results.push_back(run_expected_image_test(
            "2x2", "non-symmetric", "manual four corners", {1, 2, 3, 4}, 2, 2,
            kAsymmetricFilter, {77, 67, 47, 37}));
        results.push_back(run_expected_image_test(
            "2x3", "identity", "smaller-than-neighborhood identity",
            {-2, -1, 0, 1, 2, 3}, 2, 3, kIdentityFilter,
            {-2, -1, 0, 1, 2, 3}));
        results.push_back(run_expected_image_test(
            "3x3", "non-symmetric", "manual impulse: edges and interior",
            {0, 0, 0, 0, 1, 0, 0, 0, 0}, 3, 3, kAsymmetricFilter,
            {9, 8, 7, 6, 5, 4, 3, 2, 1}));

        results.push_back(run_identity_test(15, 15));
        results.push_back(run_constant_cross_test(16, 16));
        results.push_back(run_impulse_test(17, 17));
        results.push_back(run_identity_test(17, 31));
        results.push_back(run_constant_cross_test(31, 32));
        results.push_back(run_impulse_test(32, 32));
        results.push_back(run_identity_test(33, 31));
        results.push_back(run_randomized_test());

        print_results(results);
        std::size_t passed_count = 0;
        for (const TestResult& result : results) {
            passed_count += result.passed ? 1U : 0U;
        }

        std::cout << "\nCPU convolution tests: "
                  << (passed_count == results.size() ? "PASS" : "FAIL")
                  << " (" << passed_count << "/" << results.size()
                  << " cases)\n";
        if (passed_count != results.size()) {
            return 1;
        }

        const std::vector<CudaTestResult> cuda_results =
            run_cuda_correctness_tests();
        print_cuda_results(
            cuda_results, "Naive CUDA convolution correctness tests");
        std::size_t cuda_passed_count = 0;
        for (const CudaTestResult& result : cuda_results) {
            cuda_passed_count += result.passed ? 1U : 0U;
        }
        std::cout << "\nNaive CUDA convolution tests: "
                  << (cuda_passed_count == cuda_results.size() ? "PASS" : "FAIL")
                  << " (" << cuda_passed_count << "/" << cuda_results.size()
                  << " cases)\n";
        if (cuda_passed_count != cuda_results.size()) {
            return 1;
        }

        const std::vector<CudaTestResult> tiled_results =
            run_tiled_cuda_correctness_tests();
        print_cuda_results(
            tiled_results,
            "Shared-memory tiled CUDA convolution correctness tests");
        std::size_t tiled_passed_count = 0;
        for (const CudaTestResult& result : tiled_results) {
            tiled_passed_count += result.passed ? 1U : 0U;
        }
        std::cout << "\nTiled CUDA convolution tests: "
                  << (tiled_passed_count == tiled_results.size()
                          ? "PASS"
                          : "FAIL")
                  << " (" << tiled_passed_count << "/"
                  << tiled_results.size() << " cases)\n";
        if (tiled_passed_count != tiled_results.size()) {
            return 1;
        }

        const std::vector<BenchmarkResult> final_run_1 =
            run_comparison_benchmarks();
        print_benchmark_results(final_run_1, "Final Run 1");

        const std::vector<BenchmarkResult> final_run_2 =
            run_comparison_benchmarks();
        print_benchmark_results(final_run_2, "Final Run 2");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Convolution benchmark aborted: " << error.what() << "\n";
        return 1;
    }
}
