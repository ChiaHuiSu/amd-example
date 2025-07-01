// Compute mean for each feature across the batch
void compute_mean(const float* input, float* mean, int batch_size, int num_features) {
    for (int f = 0; f < num_features; ++f) {
        mean[f] = 0.0f;
        for (int b = 0; b < batch_size; ++b) {
            mean[f] += input[b * num_features + f];
        }
        mean[f] /= batch_size;
    }
}

// Compute variance for each feature across the batch
void compute_variance(const float* input, const float* mean, float* variance, int batch_size, int num_features) {
    for (int f = 0; f < num_features; ++f) {
        variance[f] = 0.0f;
        for (int b = 0; b < batch_size; ++b) {
            float diff = input[b * num_features + f] - mean[f];
            variance[f] += diff * diff;
        }
        variance[f] /= batch_size;
    }
}

// Perform batch normalization
void batch_normalize(float* input, int batch_size, int num_features, float epsilon) {
    float mean[num_features];
    float variance[num_features];

    compute_mean(input, mean, batch_size, num_features);
    compute_variance(input, mean, variance, batch_size, num_features);

    for (int b = 0; b < batch_size; ++b) {
        for (int f = 0; f < num_features; ++f) {
            int idx = b * num_features + f;
            input[idx] = (input[idx] - mean[f]) / sqrtf(variance[f] + epsilon);
        }
    }
}