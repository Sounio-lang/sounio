extern "C" __global__ void attention(
    const float* q, const float* k, const float* v, float* out, int n, int d) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float score = 0.0f;
    for (int j = 0; j < d; ++j) score += q[i * d + j] * k[i * d + j];
    float w = score / (1.0f + fabsf(score));
    for (int j = 0; j < d; ++j) out[i * d + j] = w * v[i * d + j];
  }
}
