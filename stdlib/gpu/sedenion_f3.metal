#include <metal_stdlib>
using namespace metal;

// sed_f3_l1_mass_kernel — sedenion F₃ L¹-associator mass
// Sounio source: stdlib/gpu/sedenion_kernels.sio
// Metal backend: self-hosted/gpu/metal.sio
//
// Note: emitted with float (f32) — sufficient for 1e-4 tolerance on F₃≈85.47
// Source-level f64 on Metal requires an explicit Sounio lowering policy.
kernel void sed_f3_l1_mass_kernel(
    device int*   n           [[buffer(0)]],
    device float* states_buf  [[buffer(1)]],
    device int*   sign_table  [[buffer(2)]],
    device int*   triples     [[buffer(3)]],
    device int*   n_triples   [[buffer(4)]],
    device float* out_buf     [[buffer(5)]],
    uint tid  [[thread_position_in_grid]],
    uint bid  [[threadgroup_position_in_grid]],
    uint ntid [[threads_per_threadgroup]]
) {
    if ((int)tid >= n[0]) return;

    int base = (int)tid * 16;

    // Load h (16 components)
    float h0  = states_buf[base+ 0], h1  = states_buf[base+ 1],
          h2  = states_buf[base+ 2], h3  = states_buf[base+ 3],
          h4  = states_buf[base+ 4], h5  = states_buf[base+ 5],
          h6  = states_buf[base+ 6], h7  = states_buf[base+ 7],
          h8  = states_buf[base+ 8], h9  = states_buf[base+ 9],
          h10 = states_buf[base+10], h11 = states_buf[base+11],
          h12 = states_buf[base+12], h13 = states_buf[base+13],
          h14 = states_buf[base+14], h15 = states_buf[base+15];

    float h[16];
    h[ 0]=h0;  h[ 1]=h1;  h[ 2]=h2;  h[ 3]=h3;
    h[ 4]=h4;  h[ 5]=h5;  h[ 6]=h6;  h[ 7]=h7;
    h[ 8]=h8;  h[ 9]=h9;  h[10]=h10; h[11]=h11;
    h[12]=h12; h[13]=h13; h[14]=h14; h[15]=h15;

    float total = 0.0f;
    int nt = n_triples[0];

    for (int ti = 0; ti < nt; ti++) {
        int I = triples[ti * 3 + 0];
        int J = triples[ti * 3 + 1];
        int K = triples[ti * 3 + 2];

        // hI[m] = h[m^I] * sign_table[(m^I)*16 + I]
        float hI[16], hJ[16], hK[16];
        for (int m = 0; m < 16; m++) {
            int mxI = m ^ I;  int mxJ = m ^ J;  int mxK = m ^ K;
            hI[m] = h[mxI] * (float)sign_table[mxI * 16 + I];
            hJ[m] = h[mxJ] * (float)sign_table[mxJ * 16 + J];
            hK[m] = h[mxK] * (float)sign_table[mxK * 16 + K];
        }

        // aIJ = hI * hJ,  bJK = hJ * hK
        float aIJ[16], bJK[16];
        for (int m = 0; m < 16; m++) {
            float saij = 0.0f, sbjk = 0.0f;
            for (int i = 0; i < 16; i++) {
                int mi = m ^ i;
                float s = (float)sign_table[i * 16 + mi];
                saij += hI[i] * hJ[mi] * s;
                sbjk += hJ[i] * hK[mi] * s;
            }
            aIJ[m] = saij;
            bJK[m] = sbjk;
        }

        // Accumulate L¹ norm of associator: |(aIJ*hK)[m] - (hI*bJK)[m]|
        for (int m = 0; m < 16; m++) {
            float lhs = 0.0f, rhs = 0.0f;
            for (int i = 0; i < 16; i++) {
                int mi = m ^ i;
                float s = (float)sign_table[i * 16 + mi];
                lhs += aIJ[i] * hK[mi] * s;
                rhs += hI[i]  * bJK[mi] * s;
            }
            float diff = lhs - rhs;
            total += diff < 0.0f ? -diff : diff;
        }
    }

    out_buf[(int)tid] = total / (float)nt;
}
