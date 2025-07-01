#include <stdio.h>
#include <xmmintrin.h>  // SSE

int main() {
    // Initialize two float arrays (each 4 floats)
    float A[4] = {1.0, 2.0, 3.0, 4.0};
    float B[4] = {5.0, 6.0, 7.0, 8.0};
    float C[4];

    // Load data into SSE 128-bit registers
    __m128 vecA = _mm_loadu_ps(A);  // Load unaligned
    __m128 vecB = _mm_loadu_ps(B);

    // Perform SIMD addition
    __m128 vecC = _mm_add_ps(vecA, vecB);

    // Store result back to array
    _mm_storeu_ps(C, vecC);

    // Print result
    printf("Result: ");
    for (int i = 0; i < 4; i++) {
        printf("%.1f ", C[i]);
    }
    printf("\n");

    return 0;
}