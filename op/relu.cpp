// Apply ReLU activation: output[i] = max(0, input[i])
void relu(float* input, float* output, int size) {
    for (int i = 0; i < size; ++i) {
        output[i] = input[i] > 0.0f ? input[i] : 0.0f;
    }
}
