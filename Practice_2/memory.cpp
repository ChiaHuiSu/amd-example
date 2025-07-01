__global__ void copy(float* input, float* output) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    output[tid] = input[tid];
}