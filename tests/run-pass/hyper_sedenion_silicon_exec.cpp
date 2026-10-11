#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>
#include <sys/mman.h>

// Sedenion silicon oracle: Cayley-Dickson over the committed 186-byte Fano kernel.
// Harness matches hyper_octonion_silicon_exec.cpp ABI:
//   void f(const double* a, const double* b, double* out, const void* ctrl_table)
// with a/b/out in zmm0/1/2 and 21 Fano controls in zmm11..zmm31 from [rcx].
//
// Native emitter register-conflict fix:
//   lower_ir.sio::lower_hyper_mul_s (conj_tmp = dst+3, distinct from t_aj)
//   codegen_x86_linux.sio::pireus_native_emit_hyper_mul_s_cd_ptr3 (conj_tmp = zmm9)

using FanoExecFn = void (*)(const double* a, const double* b, double* out, const void* ctrl_table);

static FanoExecFn build_fano_fn(const std::vector<uint8_t>& kernel_bytes) {
    assert(kernel_bytes.size() == 186);
    constexpr size_t code_size = 4096;
    auto* code = static_cast<uint8_t*>(mmap(
        nullptr, code_size, PROT_READ | PROT_WRITE | PROT_EXEC,
        MAP_ANONYMOUS | MAP_PRIVATE, -1, 0));
    assert(code != MAP_FAILED);

    int pos = 0;
    // vmovupd zmm0, [rdi]
    code[pos++] = 0x62; code[pos++] = 0xF1; code[pos++] = 0xFD; code[pos++] = 0x48;
    code[pos++] = 0x10; code[pos++] = 0x07;
    // vmovupd zmm1, [rsi]
    code[pos++] = 0x62; code[pos++] = 0xF1; code[pos++] = 0xFD; code[pos++] = 0x48;
    code[pos++] = 0x10; code[pos++] = 0x0E;

    constexpr int ctrl_base = 11;
    for (int k = 0; k < 21; ++k) {
        int zmm_reg = ctrl_base + k;
        int disp = k * 64;
        code[pos++] = 0x62;
        int r_inv  = ((zmm_reg >> 3) & 1) ? 0 : 1;
        int x_inv  = 1;
        int b_inv  = 1;
        int rp_inv = ((zmm_reg >> 4) & 1) ? 0 : 1;
        code[pos++] = static_cast<uint8_t>((r_inv << 7) | (x_inv << 6) | (b_inv << 5) | (rp_inv << 4) | 1);
        code[pos++] = 0xFD;
        code[pos++] = static_cast<uint8_t>(((2 & 3) << 5) | (1 << 3));
        code[pos++] = 0x10;
        code[pos++] = static_cast<uint8_t>(0x80 | ((zmm_reg & 7) << 3) | 0x01);
        code[pos++] = static_cast<uint8_t>(disp & 0xFF);
        code[pos++] = static_cast<uint8_t>((disp >> 8) & 0xFF);
        code[pos++] = static_cast<uint8_t>((disp >> 16) & 0xFF);
        code[pos++] = static_cast<uint8_t>((disp >> 24) & 0xFF);
    }

    std::memcpy(code + pos, kernel_bytes.data(), kernel_bytes.size());
    pos += static_cast<int>(kernel_bytes.size());

    // vmovupd [rdx], zmm2
    code[pos++] = 0x62; code[pos++] = 0xF1; code[pos++] = 0xFD; code[pos++] = 0x48;
    code[pos++] = 0x11; code[pos++] = 0x12;
    code[pos++] = 0xC3;
    return reinterpret_cast<FanoExecFn>(code);
}

static void fill_ctrl(uint64_t ctrl_data[21][8]) {
    std::memset(ctrl_data, 0, 21 * 8 * sizeof(uint64_t));
    for (int j = 1; j <= 7; ++j)
        for (int lane = 0; lane < 8; ++lane)
            ctrl_data[j - 1][lane] = static_cast<uint64_t>(j);

    constexpr std::array<std::array<uint64_t, 8>, 7> perms = {{
        {1, 0, 3, 2, 5, 4, 7, 6},
        {2, 3, 0, 1, 6, 7, 4, 5},
        {3, 2, 1, 0, 7, 6, 5, 4},
        {4, 5, 6, 7, 0, 1, 2, 3},
        {5, 4, 7, 6, 1, 0, 3, 2},
        {6, 7, 4, 5, 2, 3, 0, 1},
        {7, 6, 5, 4, 3, 2, 1, 0}
    }};
    for (int j = 0; j < 7; ++j)
        for (int lane = 0; lane < 8; ++lane)
            ctrl_data[7 + j][lane] = perms[j][lane];

    constexpr std::array<std::array<int, 8>, 7> signs = {{
        {-1,  1,  1, -1,  1, -1, -1,  1},
        {-1, -1,  1,  1,  1,  1, -1, -1},
        {-1,  1, -1,  1,  1, -1,  1, -1},
        {-1, -1, -1, -1,  1,  1,  1,  1},
        {-1,  1, -1,  1, -1,  1, -1,  1},
        {-1,  1,  1, -1, -1,  1,  1, -1},
        {-1, -1,  1,  1, -1, -1,  1,  1}
    }};
    for (int j = 0; j < 7; ++j)
        for (int lane = 0; lane < 8; ++lane)
            ctrl_data[14 + j][lane] = (signs[j][lane] < 0) ? 0x8000000000000000ULL : 0ULL;
}

static void oct_conj(const double* x, double* y) {
    y[0] = x[0];
    for (int i = 1; i < 8; ++i) y[i] = -x[i];
}

static void oct_mul(FanoExecFn fano, const void* ctrl,
                    const double* a, const double* b, double* out) {
    alignas(64) double aa[8], bb[8], oo[8];
    std::memcpy(aa, a, 64);
    std::memcpy(bb, b, 64);
    fano(aa, bb, oo, ctrl);
    std::memcpy(out, oo, 64);
}

static void sed_mul(FanoExecFn fano, const void* ctrl,
                    const double* lhs, const double* rhs, double* dst) {
    const double* a = lhs;
    const double* b = lhs + 8;
    const double* c = rhs;
    const double* d = rhs + 8;

    alignas(64) double ac[8], conj_d[8], conj_d_b[8], lo[8];
    alignas(64) double da[8], conj_c[8], b_conj_c[8], hi[8];

    oct_mul(fano, ctrl, a, c, ac);
    oct_conj(d, conj_d);
    oct_mul(fano, ctrl, conj_d, b, conj_d_b);
    for (int i = 0; i < 8; ++i) lo[i] = ac[i] - conj_d_b[i];

    oct_mul(fano, ctrl, d, a, da);
    oct_conj(c, conj_c);
    oct_mul(fano, ctrl, b, conj_c, b_conj_c);
    for (int i = 0; i < 8; ++i) hi[i] = da[i] + b_conj_c[i];

    std::memcpy(dst, lo, 64);
    std::memcpy(dst + 8, hi, 64);
}

static bool nearly_eq(double x, double y, double eps = 1e-12) {
    return std::abs(x - y) <= eps;
}

static int require_vec(const double* got, const double* exp, int n, const char* label) {
    for (int i = 0; i < n; ++i) {
        if (!nearly_eq(got[i], exp[i])) {
            std::cerr << label << " mismatch i=" << i
                      << " got=" << got[i] << " exp=" << exp[i] << "\n";
            return 1;
        }
    }
    return 0;
}

int main(int argc, char** argv) {
    const char* kernel_bin_path = (argc > 1) ? argv[1] : "fano_raw_kernel.bin";
    std::ifstream bin_file(kernel_bin_path, std::ios::binary);
    if (!bin_file) {
        std::cerr << "Error: cannot open " << kernel_bin_path << "\n";
        return 1;
    }
    std::vector<uint8_t> kernel_bytes((std::istreambuf_iterator<char>(bin_file)),
                                      std::istreambuf_iterator<char>());
    if (kernel_bytes.size() != 186) return 2;

    FanoExecFn fano = build_fano_fn(kernel_bytes);
    alignas(64) uint64_t ctrl_data[21][8];
    fill_ctrl(ctrl_data);

    std::cout << "Loaded " << kernel_bytes.size()
              << " raw Fano EVEX bytes for sedenion Cayley-Dickson composition\n";

    alignas(64) double e0[16] = {1,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0};
    alignas(64) double s_test[16] = {
        1,2,-3,4,-5,6,-7,8, -9,10,-11,12,-13,14,-15,16};
    alignas(64) double out1[16] = {0};
    sed_mul(fano, ctrl_data, e0, s_test, out1);
    std::cout << "TEST 1 (Identity e0 * S): ";
    if (require_vec(out1, s_test, 16, "TEST1")) return 10;
    std::cout << "PASSED!\n";

    // e3=(e3,0), e10=(0,e2) => (0, e2*e3). Repo Fano: e2*e3 = +e1 => +e9.
    alignas(64) double e3[16] = {0}; e3[3] = 1.0;
    alignas(64) double e10[16] = {0}; e10[10] = 1.0;
    alignas(64) double out2[16] = {0};
    sed_mul(fano, ctrl_data, e3, e10, out2);
    alignas(64) double exp2[16] = {0}; exp2[9] = 1.0;
    std::cout << "TEST 2 (Basis e3 * e10 == +e9 under Fano table): ";
    if (require_vec(out2, exp2, 16, "TEST2")) return 11;
    std::cout << "PASSED!\n";

    alignas(64) double u_zd[16] = {0}; u_zd[3] = 1.0; u_zd[10] = 1.0;
    alignas(64) double v_zd[16] = {0}; v_zd[6] = 1.0; v_zd[15] = -1.0;
    alignas(64) double out_zd[16] = {0};
    sed_mul(fano, ctrl_data, u_zd, v_zd, out_zd);
    std::cout << "TEST 3 (Zero divisor (e3+e10)*(e6-e15)): [";
    for (int i = 0; i < 16; ++i) {
        std::cout << out_zd[i] << (i < 15 ? ", " : "]\n");
        if (std::abs(out_zd[i]) >= 1e-12) return 12;
    }
    std::cout << "ZERO DIVISORS VERIFIED ON LIVE AVX-512 VIA FANO KERNEL!\n";

    alignas(64) double x_na[16] = {0}; x_na[1] = 1.0; x_na[10] = 1.0;
    alignas(64) double y_na[16] = {0}; y_na[2] = 1.0; y_na[11] = 1.0;
    alignas(64) double xx[16] = {0}; sed_mul(fano, ctrl_data, x_na, x_na, xx);
    alignas(64) double xy[16] = {0}; sed_mul(fano, ctrl_data, x_na, y_na, xy);
    alignas(64) double x_xy[16] = {0}; sed_mul(fano, ctrl_data, x_na, xy, x_xy);
    alignas(64) double xx_y[16] = {0}; sed_mul(fano, ctrl_data, xx, y_na, xx_y);
    bool identical = (std::memcmp(x_xy, xx_y, sizeof(x_xy)) == 0);
    std::cout << "TEST 4 (Non-alternativity): match=" << identical << " (expected 0)\n";
    if (identical) return 13;

    std::cout << "\nALL 4 HARDWARE SILICON SEDENION TESTS PASSED "
                 "(Cayley-Dickson o verified Fano AVX-512 kernel)!\n";
    return 0;
}
