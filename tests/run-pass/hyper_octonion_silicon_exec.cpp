#include <iostream>
#include <fstream>
#include <vector>
#include <array>
#include <cstdint>
#include <cstring>
#include <sys/mman.h>

// Function signature: void f(const double* a, const double* b, double* out, const void* ctrl_table)
using FanoExecFn = void (*)(const double* a, const double* b, double* out, const void* ctrl_table);

int main(int argc, char** argv) {
    const char* kernel_bin_path = (argc > 1) ? argv[1] : "fano_raw_kernel.bin";
    std::ifstream bin_file(kernel_bin_path, std::ios::binary);
    if (!bin_file) {
        std::cerr << "Error: cannot open raw kernel binary file at " << kernel_bin_path << "\n";
        return 1;
    }

    std::vector<uint8_t> kernel_bytes((std::istreambuf_iterator<char>(bin_file)),
                                       std::istreambuf_iterator<char>());
    if (kernel_bytes.size() != 186) return 2;

    constexpr size_t code_size = 4096;
    auto* code = static_cast<uint8_t*>(mmap(nullptr, code_size, PROT_READ | PROT_WRITE | PROT_EXEC, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0));
    if (code == MAP_FAILED) return 3;

    int pos = 0;

    // Harness prolog:
    // Load a into zmm0: vmovupd zmm0, [rdi]
    code[pos++] = 0x62; code[pos++] = 0xF1; code[pos++] = 0xFD; code[pos++] = 0x48; code[pos++] = 0x10; code[pos++] = 0x07;

    // Load b into zmm1: vmovupd zmm1, [rsi]
    code[pos++] = 0x62; code[pos++] = 0xF1; code[pos++] = 0xFD; code[pos++] = 0x48; code[pos++] = 0x10; code[pos++] = 0x0E;

    // Load 21 ctrl vectors from [rcx] into zmm11..zmm31
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

    // --- INJECT THE EXACT 186-BYTE KERNEL READ FROM DISK ---
    std::memcpy(code + pos, kernel_bytes.data(), kernel_bytes.size());
    pos += kernel_bytes.size();

    // Store dst (zmm2) to [rdx]: vmovupd [rdx], zmm2
    code[pos++] = 0x62; code[pos++] = 0xF1; code[pos++] = 0xFD; code[pos++] = 0x48; code[pos++] = 0x11; code[pos++] = 0x12;

    // ret (0xC3)
    code[pos++] = 0xC3;

    alignas(64) uint64_t ctrl_data[21][8];
    std::memset(ctrl_data, 0, sizeof(ctrl_data));

    for (int j = 1; j <= 7; ++j) {
        for (int lane = 0; lane < 8; ++lane) {
            ctrl_data[j - 1][lane] = j;
        }
    }

    constexpr std::array<std::array<uint64_t, 8>, 7> perms = {{
        {1, 0, 3, 2, 5, 4, 7, 6},
        {2, 3, 0, 1, 6, 7, 4, 5},
        {3, 2, 1, 0, 7, 6, 5, 4},
        {4, 5, 6, 7, 0, 1, 2, 3},
        {5, 4, 7, 6, 1, 0, 3, 2},
        {6, 7, 4, 5, 2, 3, 0, 1},
        {7, 6, 5, 4, 3, 2, 1, 0}
    }};
    for (int j = 0; j < 7; ++j) {
        for (int lane = 0; lane < 8; ++lane) {
            ctrl_data[7 + j][lane] = perms[j][lane];
        }
    }

    constexpr std::array<std::array<int, 8>, 7> signs = {{
        {-1,  1,  1, -1,  1, -1, -1,  1},
        {-1, -1,  1,  1,  1,  1, -1, -1},
        {-1,  1, -1,  1,  1, -1,  1, -1},
        {-1, -1, -1, -1,  1,  1,  1,  1},
        {-1,  1, -1,  1, -1,  1, -1,  1},
        {-1,  1,  1, -1, -1,  1,  1, -1},
        {-1, -1,  1,  1, -1, -1,  1,  1}
    }};
    for (int j = 0; j < 7; ++j) {
        for (int lane = 0; lane < 8; ++lane) {
            ctrl_data[14 + j][lane] = (signs[j][lane] < 0) ? 0x8000000000000000ULL : 0ULL;
        }
    }

    auto fn = reinterpret_cast<FanoExecFn>(code);

    std::cout << "Loaded " << kernel_bytes.size() << " raw EVEX bytes from " << kernel_bin_path << "\n";

    // --- TEST 1: e6 * e4 on silicon ---
    alignas(64) const double e6[8] = {0, 0, 0, 0, 0, 0, 1, 0};
    alignas(64) const double e4[8] = {0, 0, 0, 0, 1, 0, 0, 0};
    alignas(64) double out1[8] = {0};
    fn(e6, e4, out1, ctrl_data);

    std::cout << "TEST 1 (e6 * e4 on hardware AVX-512): [";
    for (int i = 0; i < 8; ++i) std::cout << out1[i] << (i < 7 ? ", " : "]\n");
    if (out1[2] != -1.0) return 1;
    for (int i = 0; i < 8; ++i) if (i != 2 && out1[i] != 0.0) return 1;

    // --- TEST 2: a * b match Lean 4 #eval ---
    alignas(64) const double a[8] = {1, 2, -1, 3, 0, -2, 1, 4};
    alignas(64) const double b[8] = {2, -1, 0, 1, 3, 2, -1, 1};
    alignas(64) double out2[8] = {0};
    fn(a, b, out2, ctrl_data);

    std::cout << "TEST 2 (a * b on hardware AVX-512): [";
    for (int i = 0; i < 8; ++i) std::cout << out2[i] << (i < 7 ? ", " : "]\n");

    constexpr std::array<double, 8> exp2 = {2, 3, -20, -6, 1, 1, -4, 17};
    for (int i = 0; i < 8; ++i) if (out2[i] != exp2[i]) return 1;

    // --- TEST 3: Norm multiplicativity on silicon ---
    double na = 0, nb = 0, nab = 0;
    for (int i = 0; i < 8; ++i) {
        na += a[i] * a[i];
        nb += b[i] * b[i];
        nab += out2[i] * out2[i];
    }
    std::cout << "TEST 3 (|ab|^2 on silicon): " << nab << " == " << na << " * " << nb << " (" << na * nb << ")\n";
    if (nab != na * nb) return 1;
    if (nab != 756.0) return 1;

    // --- TEST 4: Left alternativity on silicon: a * (a * b) == (a * a) * b ---
    alignas(64) double aa[8] = {0};
    fn(a, a, aa, ctrl_data);

    alignas(64) double a_ab[8] = {0};
    fn(a, out2, a_ab, ctrl_data);

    alignas(64) double aa_b[8] = {0};
    fn(aa, b, aa_b, ctrl_data);

    std::cout << "TEST 4 (Left alternativity on silicon): match = " << (std::memcmp(a_ab, aa_b, sizeof(a_ab)) == 0) << "\n";
    if (std::memcmp(a_ab, aa_b, sizeof(a_ab)) != 0) return 1;

    // --- TEST 5: Right alternativity on silicon: (b * a) * a == b * (a * a) ---
    alignas(64) double ba[8] = {0};
    fn(b, a, ba, ctrl_data);

    alignas(64) double ba_a[8] = {0};
    fn(ba, a, ba_a, ctrl_data);

    alignas(64) double b_aa[8] = {0};
    fn(b, aa, b_aa, ctrl_data);

    std::cout << "TEST 5 (Right alternativity on silicon): match = " << (std::memcmp(ba_a, b_aa, sizeof(ba_a)) == 0) << "\n";
    if (std::memcmp(ba_a, b_aa, sizeof(ba_a)) != 0) return 1;

    std::cout << "\nALL 5 HARDWARE SILICON TESTS PASSED FROM FILE-LOADED BYTES IN C++23!\n";
    return 0;
}
