#include <hip/hip_runtime.h>
#include <iostream>
#include <cmath>

// Compute GELU using the exact formula: GELU(x) = x * Φ(x) = x * 0.5 * (1 + erf(x / sqrt(2)))
__device__ float gelu_exact(float x) {
    return 0.5f * x * (1.0f + erf(x / sqrtf(2.0f)));
}

// HIP kernel for GELU activation using float32 and exact formula
__global__ void gelu_kernel(const float* input, float* output, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N) {
        output[idx] = gelu_exact(input[idx]);
    }
}

int main() {
    const int N = 1024;
    const int size = N * sizeof(float);

    // Allocate host memory
    float* h_input = new float[N];
    float* h_output = new float[N];

    // Initialize input data: range [-5.0, 5.0]
    for (int i = 0; i < N; ++i) {
        h_input[i] = static_cast<float>(i) / 100.0f - 5.0f;
    }

    // Allocate device memory
    float *d_input, *d_output;
    hipMalloc(&d_input, size);
    hipMalloc(&d_output, size);

    // Copy input to device
    hipMemcpy(d_input, h_input, size, hipMemcpyHostToDevice);

    // Launch HIP kernel
    int threadsPerBlock = 256;
    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    hipLaunchKernelGGL(gelu_kernel, dim3(blocks), dim3(threadsPerBlock), 0, 0, d_input, d_output, N);

    // Copy result back to host
    hipMemcpy(h_output, d_output, size, hipMemcpyDeviceToHost);

    // Print a few output values
    std::cout << "GELU activation using true formula (float32):\n";
    for (int i = 500; i < 505; ++i) {
        std::cout << "x = " << h_input[i] << ", GELU(x) = " << h_output[i] << std::endl;
    }

    // Cleanup
    delete[] h_input;
    delete[] h_output;
    hipFree(d_input);
    hipFree(d_output);

    return 0;
}
