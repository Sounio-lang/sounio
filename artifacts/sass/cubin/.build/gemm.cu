extern "C" __global__ void gemm(
    const float* a, const float* b, float* c, int m, int n, int k) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < m && col < n) {
    float acc = 0.0f;
    for (int i = 0; i < k; ++i) acc += a[row * k + i] * b[i * n + col];
    c[row * n + col] = acc;
  }
}
