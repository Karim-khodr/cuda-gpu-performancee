#include "cuda_utils.cuh"

#include <sstream>
#include <stdexcept>

void check_cuda(cudaError_t result, const char* file, int line) {
    if (result == cudaSuccess) {
        return;
    }

    std::ostringstream message;
    message << "CUDA error: " << cudaGetErrorString(result)
            << " (" << file << ':' << line << ')';
    throw std::runtime_error(message.str());
}

CudaEventTimer::CudaEventTimer() {
    CUDA_CHECK(cudaEventCreate(&start_event_));

    const cudaError_t result = cudaEventCreate(&stop_event_);
    if (result != cudaSuccess) {
        cudaEventDestroy(start_event_);
        start_event_ = nullptr;
        CUDA_CHECK(result);
    }
}

CudaEventTimer::~CudaEventTimer() noexcept {
    if (stop_event_ != nullptr) {
        cudaEventDestroy(stop_event_);
    }

    if (start_event_ != nullptr) {
        cudaEventDestroy(start_event_);
    }
}

void CudaEventTimer::start() {
    CUDA_CHECK(cudaEventRecord(start_event_));
}

float CudaEventTimer::stop() {
    CUDA_CHECK(cudaEventRecord(stop_event_));
    CUDA_CHECK(cudaEventSynchronize(stop_event_));

    float elapsed_ms = 0.0F;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start_event_, stop_event_));
    return elapsed_ms;
}
