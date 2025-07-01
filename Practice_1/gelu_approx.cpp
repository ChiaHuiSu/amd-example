#include <hip/hip_runtime.h>
#include <iostream>
#include <cmath>

#define CHECK(cmd) \
{ \
    hipError_t error = cmd; \
    if (error != hipSuccess) { \
        std::cerr << "Error: " << hipGetErrorString(error) << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

// GELU approximation: 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x^3)))
__device__ float gelu_approx(float x) {
    float coeff = sqrtf(2.0f / M_PI);
    float x3 = x * x * x;
    return 0.5f * x * (1.0f + tanhf(coeff * (x + 0.044715f * x3)));
}

__global__ void gelu_kernel(const float* input, float* output, int size) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < size) {
        output[idx] = gelu_approx(input[idx]);
    }
}

int main() {
    const int size = 8;
    float h_input[size] = {-3.0f, -1.0f, 0.0f, 0.5f, 1.0f, 2.0f, 3.0f, 4.0f};
    float h_output[size];

    float *d_input, *d_output;
    CHECK(hipMalloc(&d_input, size * sizeof(float)));
    CHECK(hipMalloc(&d_output, size * sizeof(float)));

    CHECK(hipMemcpy(d_input, h_input, size * sizeof(float), hipMemcpyHostToDevice));

    dim3 blockSize(256);
    dim3 gridSize((size + blockSize.x - 1) / blockSize.x);
    hipLaunchKernelGGL(gelu_kernel, gridSize, blockSize, 0, 0, d_input, d_output, size);

    CHECK(hipMemcpy(h_output, d_output, size * sizeof(float), hipMemcpyDeviceToHost));

    std::cout << "Input\tGELU Output" << std::endl;
    for (int i = 0; i < size; ++i) {
        std::cout << h_input[i] << "\t" << h_output[i] << std::endl;
    }

    hipFree(d_input);
    hipFree(d_output);
    return 0;
}
