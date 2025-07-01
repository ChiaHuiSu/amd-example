#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <iostream>

// HIP kernel to perform vector addition using __half (FP16)
__global__ void vecAdd_half(const __half* A, const __half* B, __half* C, int N) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < N) {
        // Add two __half values element-wise
        C[idx] = __hadd(A[idx], B[idx]);
    }
}

int main() {
    const int N = 1024;
    const int size = N * sizeof(__half);

    // Host arrays
    __half* h_A = new __half[N];
    __half* h_B = new __half[N];
    __half* h_C = new __half[N];

    // Initialize input vectors with float values converted to __half
    for (int i = 0; i < N; ++i) {
        h_A[i] = __float2half(static_cast<float>(i));
        h_B[i] = __float2half(static_cast<float>(2 * i));
    }

    // Device arrays
    __half *d_A, *d_B, *d_C;
    hipMalloc(&d_A, size);
    hipMalloc(&d_B, size);
    hipMalloc(&d_C, size);

    // Copy input data to device
    hipMemcpy(d_A, h_A, size, hipMemcpyHostToDevice);
    hipMemcpy(d_B, h_B, size, hipMemcpyHostToDevice);

    // Launch HIP kernel with 256 threads per block
    int threadsPerBlock = 256;
    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    hipLaunchKernelGGL(vecAdd_half, dim3(blocks), dim3(threadsPerBlock), 0, 0, d_A, d_B, d_C, N);

    // Copy result back to host
    hipMemcpy(h_C, d_C, size, hipMemcpyDeviceToHost);

    // Print a few results
    std::cout << "Sample output (as float):" << std::endl;
    for (int i = 0; i < 5; ++i) {
        float a = __half2float(h_A[i]);
        float b = __half2float(h_B[i]);
        float c = __half2float(h_C[i]);
        std::cout << "A: " << a << ", B: " << b << ", C: " << c << std::endl;
    }

    // Clean up
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    hipFree(d_A);
    hipFree(d_B);
    hipFree(d_C);

    return 0;
}
