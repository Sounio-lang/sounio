extern "C" __global__ void epistemic_elementwise(
    const float* x, const float* y, const float* eps, float* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = x[i] + y[i] * (1.0f + eps[i]);
}
