// Minimal HIP smoke test for AMD Radeon R7900 AI PRO (gfx1100).
// This test is independent of the Sounio compiler; it validates that the
// ROCm/HIP toolchain can produce and run a working HSACO on the target node.
//
// Build: hipcc -offload-arch=gfx1100 smoke_vec_add.hip.cpp -o hip_smoke
// Run:   ./hip_smoke

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

#define N 1024

__global__ void vec_add_f32(const float *a, const float *b, float *out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] + b[idx];
    }
}

int main() {
    float *h_a = (float *)malloc(N * sizeof(float));
    float *h_b = (float *)malloc(N * sizeof(float));
    float *h_out = (float *)malloc(N * sizeof(float));

    for (int i = 0; i < N; ++i) {
        h_a[i] = (float)i;
        h_b[i] = (float)(N - i);
    }

    float *d_a = nullptr, *d_b = nullptr, *d_out = nullptr;
    hipMalloc(&d_a, N * sizeof(float));
    hipMalloc(&d_b, N * sizeof(float));
    hipMalloc(&d_out, N * sizeof(float));

    hipMemcpy(d_a, h_a, N * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_b, h_b, N * sizeof(float), hipMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    hipLaunchKernelGGL(vec_add_f32, dim3(gridSize), dim3(blockSize), 0, 0,
                       d_a, d_b, d_out, N);

    hipMemcpy(h_out, d_out, N * sizeof(float), hipMemcpyDeviceToHost);

    int errors = 0;
    for (int i = 0; i < N; ++i) {
        float expected = h_a[i] + h_b[i];
        if (std::fabs(h_out[i] - expected) > 1e-5f) {
            if (errors < 5) {
                std::fprintf(stderr, "error at %d: got %f expected %f\n", i, h_out[i], expected);
            }
            ++errors;
        }
    }

    hipFree(d_a);
    hipFree(d_b);
    hipFree(d_out);
    free(h_a);
    free(h_b);
    free(h_out);

    if (errors == 0) {
        std::printf("sounio_hip_smoke status=PASS reason=vec_add_f32_oracle_match device=R7900AI_PRO\n");
        return 0;
    } else {
        std::printf("sounio_hip_smoke status=FAIL reason=oracle_mismatch errors=%d\n", errors);
        return 1;
    }
}
