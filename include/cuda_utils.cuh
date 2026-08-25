#pragma once

#include <cuda_runtime.h>

void check_cuda(cudaError_t result, const char* file, int line);

#define CUDA_CHECK(call) check_cuda((call), __FILE__, __LINE__)

class CudaEventTimer {
public:
    CudaEventTimer();
    ~CudaEventTimer() noexcept;

    CudaEventTimer(const CudaEventTimer&) = delete;
    CudaEventTimer& operator=(const CudaEventTimer&) = delete;

    void start();
    float stop();

private:
    cudaEvent_t start_event_ = nullptr;
    cudaEvent_t stop_event_ = nullptr;
};
