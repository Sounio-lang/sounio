extern "C" __global__ void monte_carlo(unsigned int seed, float* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    unsigned int x = seed ^ (0x9e3779b9u * (i + 1u));
    x = 1664525u * x + 1013904223u;
    float u = (x & 0x00ffffff) / 16777216.0f;
    out[i] = u * (1.0f - u);
  }
}
