#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <sys/mman.h>
#include <assert.h>

static inline int tho_bit(int val, int n) { return (val >> n) & 1; }
static inline int tho_inv(int b) { return b == 0 ? 1 : 0; }
static inline int tho_lo3(int id) { return id & 7; }

static inline int tho_p1_fixed(int dst, int src2, int map) {
    int r_inv  = tho_inv(tho_bit(dst, 3));
    int x_inv  = tho_inv(tho_bit(src2, 4)); // bit 4 of src2 (r/m)
    int b_inv  = tho_inv(tho_bit(src2, 3));
    int rp_inv = tho_inv(tho_bit(dst, 4));
    return (r_inv << 7) | (x_inv << 6) | (b_inv << 5) | (rp_inv << 4) | (map & 0xF);
}

static inline int tho_p2_pd(int src1) {
    int vvvv_inv = (0 - src1 - 1) & 0xF;
    return 0x80 | (vvvv_inv << 3) | 0x4 | 0x1;
}

static inline int tho_p3_vl_full(int src1, int vl, int mask_k, int z_bit) {
    int vp_inv = tho_inv(tho_bit(src1, 4));
    return ((z_bit & 1) << 7) | ((vl & 3) << 5) | (vp_inv << 3) | (mask_k & 7);
}

static int emit_evex_pd_rr_full(uint8_t* buf, int pos, int map, int opcode, int dst, int src1, int src2, int vl, int mask_k, int z_bit) {
    buf[pos++] = 0x62;
    buf[pos++] = (uint8_t)tho_p1_fixed(dst, src2, map);
    buf[pos++] = (uint8_t)tho_p2_pd(src1);
    buf[pos++] = (uint8_t)tho_p3_vl_full(src1, vl, mask_k, z_bit);
    buf[pos++] = (uint8_t)opcode;
    buf[pos++] = (uint8_t)(0xC0 | (tho_lo3(dst) << 3) | tho_lo3(src2));
    return pos;
}

typedef void (*fano_exec_fn)(const double* a, const double* b, double* out, const void* ctrl_table);

int main() {
    size_t code_size = 4096;
    uint8_t* code = (uint8_t*)mmap(NULL, code_size, PROT_READ | PROT_WRITE | PROT_EXEC, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);

    int pos = 0;

    // Load a into zmm0: vmovupd zmm0, [rdi]
    code[pos++] = 0x62; code[pos++] = 0xF1; code[pos++] = 0xFD; code[pos++] = 0x48; code[pos++] = 0x10; code[pos++] = 0x07;

    // Load b into zmm1: vmovupd zmm1, [rsi]
    code[pos++] = 0x62; code[pos++] = 0xF1; code[pos++] = 0xFD; code[pos++] = 0x48; code[pos++] = 0x10; code[pos++] = 0x0E;

    int ctrl_base = 11;
    for (int k = 0; k < 21; k++) {
        int zmm_reg = ctrl_base + k;
        int disp = k * 64;
        code[pos++] = 0x62;
        int r_inv  = tho_inv(tho_bit(zmm_reg, 3));
        int x_inv  = 1;
        int b_inv  = tho_inv(tho_bit(1, 3));
        int rp_inv = tho_inv(tho_bit(zmm_reg, 4));
        code[pos++] = (r_inv << 7) | (x_inv << 6) | (b_inv << 5) | (rp_inv << 4) | 1;
        code[pos++] = 0xFD;
        code[pos++] = (uint8_t)tho_p3_vl_full(0, 2, 0, 0);
        code[pos++] = 0x10;
        code[pos++] = (uint8_t)(0x80 | (tho_lo3(zmm_reg) << 3) | 0x01);
        code[pos++] = (uint8_t)(disp & 0xFF);
        code[pos++] = (uint8_t)((disp >> 8) & 0xFF);
        code[pos++] = (uint8_t)((disp >> 16) & 0xFF);
        code[pos++] = (uint8_t)((disp >> 24) & 0xFF);
    }

    // --- EXACT 186-BYTE KERNEL FROM lower_ir.sio:1650 ---
    int kernel_start = pos;
    int za = 0;
    int zb_in = 1;
    int dst = 2;
    int t_bj = dst + 2; // 4
    int t_aj = dst + 4; // 6
    int accum = dst + 6; // 8
    int vl = 2;

    // Column 0:
    pos = emit_evex_pd_rr_full(code, pos, 2, 0x19, t_bj, 0, zb_in, vl, 0, 0);
    pos = emit_evex_pd_rr_full(code, pos, 1, 0x59, accum, za, t_bj, vl, 0, 0);

    // Columns 1-7:
    for (int j = 0; j < 7; j++) {
        pos = emit_evex_pd_rr_full(code, pos, 2, 0x16, t_bj, ctrl_base + j, zb_in, vl, 0, 0);
        pos = emit_evex_pd_rr_full(code, pos, 2, 0x16, t_aj, ctrl_base + 7 + j, za, vl, 0, 0);
        pos = emit_evex_pd_rr_full(code, pos, 1, 0x57, t_aj, t_aj, ctrl_base + 14 + j, vl, 0, 0);
        pos = emit_evex_pd_rr_full(code, pos, 2, 0xB8, accum, t_aj, t_bj, vl, 0, 0);
    }

    // Final store: VMOVAPD dst, accum
    pos = emit_evex_pd_rr_full(code, pos, 1, 0x28, dst, 0, accum, vl, 0, 0);

    int kernel_len = pos - kernel_start;
    assert(kernel_len == 186);

    // Store dst (zmm2) to [rdx]:
    code[pos++] = 0x62; code[pos++] = 0xF1; code[pos++] = 0xFD; code[pos++] = 0x48; code[pos++] = 0x11; code[pos++] = 0x12;

    code[pos++] = 0xC3;

    uint64_t ctrl_data[21][8];
    memset(ctrl_data, 0, sizeof(ctrl_data));

    for (int j = 1; j <= 7; j++) {
        for (int lane = 0; lane < 8; lane++) {
            ctrl_data[j - 1][lane] = j;
        }
    }

    static const uint64_t perms[7][8] = {
        {1, 0, 3, 2, 5, 4, 7, 6},
        {2, 3, 0, 1, 6, 7, 4, 5},
        {3, 2, 1, 0, 7, 6, 5, 4},
        {4, 5, 6, 7, 0, 1, 2, 3},
        {5, 4, 7, 6, 1, 0, 3, 2},
        {6, 7, 4, 5, 2, 3, 0, 1},
        {7, 6, 5, 4, 3, 2, 1, 0}
    };
    for (int j = 0; j < 7; j++) {
        for (int lane = 0; lane < 8; lane++) {
            ctrl_data[7 + j][lane] = perms[j][lane];
        }
    }

    static const int signs[7][8] = {
        {-1,  1,  1, -1,  1, -1, -1,  1},
        {-1, -1,  1,  1,  1,  1, -1, -1},
        {-1,  1, -1,  1,  1, -1,  1, -1},
        {-1, -1, -1, -1,  1,  1,  1,  1},
        {-1,  1, -1,  1, -1,  1, -1,  1},
        {-1,  1,  1, -1, -1,  1,  1, -1},
        {-1, -1,  1,  1, -1, -1,  1,  1}
    };
    for (int j = 0; j < 7; j++) {
        for (int lane = 0; lane < 8; lane++) {
            ctrl_data[14 + j][lane] = (signs[j][lane] < 0) ? 0x8000000000000000ULL : 0ULL;
        }
    }

    fano_exec_fn fn = (fano_exec_fn)code;

    printf("Kernel byte length: %d instructions: 31\n", kernel_len);

    // --- TEST 1: e6 * e4 on silicon ---
    double e6[8] = {0, 0, 0, 0, 0, 0, 1, 0};
    double e4[8] = {0, 0, 0, 0, 1, 0, 0, 0};
    double out1[8] = {0};
    fn(e6, e4, out1, ctrl_data);

    printf("TEST 1 (e6 * e4 on hardware AVX-512): [");
    for (int i = 0; i < 8; i++) printf("%.1f%s", out1[i], i < 7 ? ", " : "]\n");
    assert(out1[2] == -1.0);
    for (int i = 0; i < 8; i++) if (i != 2) assert(out1[i] == 0.0);

    // --- TEST 2: a * b match Lean 4 #eval ---
    double a[8] = {1, 2, -1, 3, 0, -2, 1, 4};
    double b[8] = {2, -1, 0, 1, 3, 2, -1, 1};
    double out2[8] = {0};
    fn(a, b, out2, ctrl_data);

    printf("TEST 2 (a * b on hardware AVX-512): [");
    for (int i = 0; i < 8; i++) printf("%.1f%s", out2[i], i < 7 ? ", " : "]\n");

    static const double exp2[8] = {2, 3, -20, -6, 1, 1, -4, 17};
    for (int i = 0; i < 8; i++) assert(out2[i] == exp2[i]);

    // --- TEST 3: Norm multiplicativity on silicon ---
    double na = 0, nb = 0, nab = 0;
    for (int i = 0; i < 8; i++) {
        na += a[i] * a[i];
        nb += b[i] * b[i];
        nab += out2[i] * out2[i];
    }
    printf("TEST 3 (|ab|^2 on silicon): %.1f == %.1f * %.1f (%.1f)\n", nab, na, nb, na * nb);
    assert(nab == na * nb);
    assert(nab == 756.0);

    // --- TEST 4: Left alternativity on silicon: a * (a * b) == (a * a) * b ---
    double aa[8] = {0};
    fn(a, a, aa, ctrl_data);

    double a_ab[8] = {0};
    fn(a, out2, a_ab, ctrl_data);

    double aa_b[8] = {0};
    fn(aa, b, aa_b, ctrl_data);

    printf("TEST 4 (Left alternativity on silicon): match = %d\n", memcmp(a_ab, aa_b, sizeof(a_ab)) == 0);
    assert(memcmp(a_ab, aa_b, sizeof(a_ab)) == 0);

    // --- TEST 5: Right alternativity on silicon: (b * a) * a == b * (a * a) ---
    double ba[8] = {0};
    fn(b, a, ba, ctrl_data);

    double ba_a[8] = {0};
    fn(ba, a, ba_a, ctrl_data);

    double b_aa[8] = {0};
    fn(b, aa, b_aa, ctrl_data);

    printf("TEST 5 (Right alternativity on silicon): match = %d\n", memcmp(ba_a, b_aa, sizeof(ba_a)) == 0);
    assert(memcmp(ba_a, b_aa, sizeof(ba_a)) == 0);

    printf("\nALL 5 HARDWARE SILICON TESTS PASSED WITH ZERO ULP DRIFT!\n");
    return 0;
}
