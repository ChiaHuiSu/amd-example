#include <stdio.h>
#include <immintrin.h>  // AVX

int main() {
    float A[8] = {1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0};
    float B[8] = {5.0, 6.0, 7.0, 8.0, 1.0, 2.0, 3.0, 4.0};
    float C[8];

    // Load 256-bit (8 floats) vectors
    __m256 vecA = _mm256_loadu_ps(A);
    __m256 vecB = _mm256_loadu_ps(B);

    // Perform vector addition
    __m256 vecC = _mm256_add_ps(vecA, vecB);

    // Store result
    _mm256_storeu_ps(C, vecC);

    // Print result
    printf("Result: ");
    for (int i = 0; i < 8; i++) {
        printf("%.1f ", C[i]);
    }
    printf("\n");

    return 0;
}
