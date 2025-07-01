// Multiply two matrices: C = A x B
void matmul(const float* A, const float* B, float* C, int M, int K, int N) {
    // Iterate over rows of A
    for (int i = 0; i < M; ++i) {
        // Iterate over columns of B
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            // Dot product of row from A and column from B
            for (int k = 0; k < K; ++k) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}