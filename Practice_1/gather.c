#include <immintrin.h>
#include <stdio.h>

int main() {
    // Source memory (data array)
    int data[16] = {
        10, 20, 30, 40,
        50, 60, 70, 80,
        90, 100, 110, 120,
        130, 140, 150, 160
    };

    // Index vector: which elements to gather
    __m256i index = _mm256_set_epi32(14, 12, 10, 8, 6, 4, 2, 0);

    // Perform gather: each index accesses data[index[i] * scale]
    // scale = 4 because int32_t is 4 bytes
    __m256i gathered = _mm256_i32gather_epi32(data, index, 4);

    // Store result to memory and print
    int result[8];
    _mm256_storeu_si256((__m256i*)result, gathered);

    printf("Gathered result:\n");
    for (int i = 0; i < 8; i++) {
        printf("result[%d] = %d\n", i, result[i]);
    }

    return 0;
}
