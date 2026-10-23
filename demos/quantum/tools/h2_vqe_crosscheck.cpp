// demos/quantum/tools/h2_vqe_crosscheck.cpp — independent C++23 cross-check of
// demos/quantum/h2_vqe_2q.sio.
//
// Written from the physics, not translated from the Sounio code:
//   * the Hamiltonian is assembled as a dense 4x4 complex matrix from Kronecker
//     products of Pauli matrices (Sounio evaluates closed-form expectation
//     values instead);
//   * the ground energy comes from a cyclic Jacobi eigenvalue iteration on the
//     real symmetric 4x4 (Sounio uses the closed-form 2x2 block);
//   * the ansatz is a product of 4x4 unitaries built with Kronecker products,
//     applied to |00> with std::complex (Sounio uses a butterfly update);
//   * the Monte Carlo uses std::mt19937_64 + std::normal_distribution with its
//     own seed (Sounio uses a L'Ecuyer combined LCG + Box-Muller);
//   * the second-order band uses a central finite-difference gradient and
//     Hessian (Sounio uses the parameter-shift rule).
//
// Conventions shared with the Sounio code (these are definitions, not code):
//   basis index i = 2*b1 + b0, b_k = state of qubit k, so an operator A on
//   qubit 1 and B on qubit 0 is kron(A, B);
//   H = c0 I + c1 Z0 + c2 Z1 + c3 Z0Z1 + c4 (X0X1 + Y0Y1),
//   (c0..c4) = (-0.4804, 0.3435, -0.4347, 0.5716, 0.0910), O'Malley et al.,
//   PRX 6, 031007 (2016); ansatz Ry(t0) q0, Ry(t1) q1, CNOT(q0 -> q1),
//   Ry(t2) q0, Ry(t3) q1; Ry(t) = exp(-i t Y / 2).
//
// Build and run:
//   g++ -std=c++23 -O2 -o /tmp/h2x demos/quantum/tools/h2_vqe_crosscheck.cpp
//   /tmp/h2x t0 t1 t2 t3 E_vqe sigma_mc(0.01) sigma_mc(0.05) sigma_mc(0.1)
// The numbers come from the Sounio demo's "theta*" / "E_vqe" / "MC sigma"
// lines. With no arguments it prints its own values only. With arguments it
// ends with H2_VQE_CROSSCHECK_OK if
//   |E_exact(C++) - E_exact(closed form)| < 1e-12,
//   |E(theta*)(C++) - E_vqe(Sounio)|     < 1e-10,
//   each Sounio MC sigma is within 5 % of the C++ MC sigma at N = 1e6
//   (the standard error of a sample sd at N = 2e4 is ~1-2 % here),
// and exits 1 otherwise.

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <numbers>
#include <random>

using cd = std::complex<double>;
using M2 = std::array<std::array<cd, 2>, 2>;
using M4 = std::array<std::array<cd, 4>, 4>;
using V4 = std::array<cd, 4>;

static M4 kron(const M2& a, const M2& b) {
    M4 r{};
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j)
            for (int k = 0; k < 2; ++k)
                for (int l = 0; l < 2; ++l) r[2 * i + k][2 * j + l] = a[i][j] * b[k][l];
    return r;
}
static M4 mul(const M4& a, const M4& b) {
    M4 r{};
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            for (int k = 0; k < 4; ++k) r[i][j] += a[i][k] * b[k][j];
    return r;
}
static M4 axpy(const M4& a, cd s, const M4& b) {  // a + s b
    M4 r = a;
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j) r[i][j] += s * b[i][j];
    return r;
}

static const cd I1{0.0, 1.0};
static const M2 PI2{{{1, 0}, {0, 1}}};
static const M2 PX{{{0, 1}, {1, 0}}};
static const M2 PY{{{0, -I1}, {I1, 0}}};
static const M2 PZ{{{1, 0}, {0, -1}}};

static M2 ry(double t) {
    // exp(-i t Y / 2) = cos(t/2) I - i sin(t/2) Y
    const double c = std::cos(t / 2), s = std::sin(t / 2);
    M2 r{};
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j) r[i][j] = c * PI2[i][j] - I1 * s * PY[i][j];
    return r;
}

static M4 hamiltonian() {
    const double c0 = -0.4804, c1 = 0.3435, c2 = -0.4347, c3 = 0.5716, c4 = 0.0910;
    M4 h{};
    h = axpy(h, c0, kron(PI2, PI2));
    h = axpy(h, c1, kron(PI2, PZ));  // Z on qubit 0
    h = axpy(h, c2, kron(PZ, PI2));  // Z on qubit 1
    h = axpy(h, c3, kron(PZ, PZ));
    h = axpy(h, c4, kron(PX, PX));
    h = axpy(h, c4, kron(PY, PY));
    return h;
}

// CNOT with control qubit 0 and target qubit 1, built from projectors:
// |0><0|_q0 (x) I_q1 + |1><1|_q0 (x) X_q1, i.e. kron(I, P0) + kron(X, P1).
static M4 cnot_q0_to_q1() {
    const M2 P0{{{1, 0}, {0, 0}}}, P1{{{0, 0}, {0, 1}}};
    return axpy(kron(PI2, P0), 1.0, kron(PX, P1));
}

static double energy(const M4& H, const std::array<double, 4>& t) {
    // U = (Ry(t3)_q1 (x) Ry(t2)_q0) CNOT (Ry(t1)_q1 (x) Ry(t0)_q0)
    const M4 U = mul(kron(ry(t[3]), ry(t[2])), mul(cnot_q0_to_q1(), kron(ry(t[1]), ry(t[0]))));
    V4 psi{};
    for (int i = 0; i < 4; ++i) psi[i] = U[i][0];
    cd e = 0;
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j) e += std::conj(psi[i]) * H[i][j] * psi[j];
    return e.real();
}

// Cyclic Jacobi on a real symmetric matrix; returns the smallest eigenvalue.
static double jacobi_min(const M4& Hc) {
    double a[4][4];
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j) {
            if (std::abs(Hc[i][j].imag()) > 1e-15) { std::fprintf(stderr, "H not real\n"); std::exit(2); }
            a[i][j] = Hc[i][j].real();
        }
    for (int sweep = 0; sweep < 100; ++sweep) {
        double off = 0;
        for (int p = 0; p < 4; ++p)
            for (int q = p + 1; q < 4; ++q) off += a[p][q] * a[p][q];
        if (off < 1e-30) break;
        for (int p = 0; p < 4; ++p)
            for (int q = p + 1; q < 4; ++q) {
                if (std::abs(a[p][q]) < 1e-300) continue;
                const double theta = (a[q][q] - a[p][p]) / (2 * a[p][q]);
                const double t = (theta >= 0 ? 1.0 : -1.0) / (std::abs(theta) + std::sqrt(theta * theta + 1));
                const double c = 1 / std::sqrt(t * t + 1), s = t * c;
                for (int k = 0; k < 4; ++k) {  // columns p, q
                    const double akp = a[k][p], akq = a[k][q];
                    a[k][p] = c * akp - s * akq;
                    a[k][q] = s * akp + c * akq;
                }
                for (int k = 0; k < 4; ++k) {  // rows p, q
                    const double apk = a[p][k], aqk = a[q][k];
                    a[p][k] = c * apk - s * aqk;
                    a[q][k] = s * apk + c * aqk;
                }
            }
    }
    double m = a[0][0];
    for (int i = 1; i < 4; ++i) m = std::min(m, a[i][i]);
    return m;
}

static double mc_sigma(const M4& H, const std::array<double, 4>& t0, double u, long n, std::uint64_t seed) {
    std::mt19937_64 gen(seed);
    std::normal_distribution<double> nd(0.0, 1.0);
    double mean = 0, m2 = 0;
    for (long k = 1; k <= n; ++k) {
        std::array<double, 4> t = t0;
        for (auto& x : t) x += u * nd(gen);
        const double e = energy(H, t), d = e - mean;
        mean += d / k;
        m2 += d * (e - mean);
    }
    return std::sqrt(m2 / (n - 1));
}

// Second-order band from central finite differences (step h).
static double gum2_fd(const M4& H, const std::array<double, 4>& t0, double u) {
    const double h = 1e-4;
    auto E = [&](int i, double di, int j, double dj) {
        auto t = t0;
        t[i] += di;
        t[j] += dj;
        return energy(H, t);
    };
    double g2 = 0, hh = 0;
    const double e0 = energy(H, t0);
    for (int i = 0; i < 4; ++i) {
        const double g = (E(i, h, i, 0) - E(i, -h, i, 0)) / (2 * h);
        g2 += g * g;
        for (int j = 0; j < 4; ++j) {
            const double hij = (i == j) ? (E(i, h, i, 0) - 2 * e0 + E(i, -h, i, 0)) / (h * h)
                                        : (E(i, h, j, h) - E(i, h, j, -h) - E(i, -h, j, h) + E(i, -h, j, -h)) / (4 * h * h);
            hh += hij * hij;
        }
    }
    return std::sqrt(u * u * g2 + 0.5 * u * u * u * u * hh);
}

int main(int argc, char** argv) {
    const M4 H = hamiltonian();
    const double e_jacobi = jacobi_min(H);
    // closed form of the {|01>,|10>} block, for the oracle check
    const double d01 = -0.4804 - 0.3435 - 0.4347 - 0.5716, d10 = -0.4804 + 0.3435 + 0.4347 - 0.5716;
    const double e_closed = 0.5 * (d01 + d10) - std::sqrt(0.25 * (d01 - d10) * (d01 - d10) + 4 * 0.0910 * 0.0910);
    std::printf("E_exact (Jacobi 4x4)      = %.15f\n", e_jacobi);
    std::printf("E_exact (2x2 closed form) = %.15f\n", e_closed);
    bool ok = std::abs(e_jacobi - e_closed) < 1e-12;

    if (argc < 2) return ok ? 0 : 1;
    if (argc != 9) {
        std::fprintf(stderr, "usage: %s t0 t1 t2 t3 E_vqe sigma001 sigma005 sigma010\n", argv[0]);
        return 2;
    }
    std::array<double, 4> t{};
    for (int i = 0; i < 4; ++i) t[i] = std::strtod(argv[1 + i], nullptr);
    const double e_vqe = std::strtod(argv[5], nullptr);
    const double e_cpp = energy(H, t);
    std::printf("E(theta*) C++             = %.15f\n", e_cpp);
    std::printf("E_vqe Sounio              = %.15f   |diff| = %.3e\n", e_vqe, std::abs(e_cpp - e_vqe));
    std::printf("E(theta*) - E_exact (C++) = %.3e\n", e_cpp - e_jacobi);
    ok = ok && std::abs(e_cpp - e_vqe) < 1e-10;

    const double us[3] = {0.01, 0.05, 0.1};
    for (int k = 0; k < 3; ++k) {
        const double s_sio = std::strtod(argv[6 + k], nullptr);
        const double s20k = mc_sigma(H, t, us[k], 20000, 0xC0FFEEULL + k);
        const double s1m = mc_sigma(H, t, us[k], 1000000, 0xBADC0DEULL + k);
        const double g2 = gum2_fd(H, t, us[k]);
        const double rel = std::abs(s_sio - s1m) / s1m;
        std::printf("u=%.2f  C++ MC sigma N=2e4 %.6e  N=1e6 %.6e  | Sounio %.6e (rel %.2f%%) | C++ GUM2(fd) %.6e  GUM2/MC(1e6) %.4f\n",
                    us[k], s20k, s1m, s_sio, 100 * rel, g2, g2 / s1m);
        ok = ok && rel < 0.05;
    }
    std::printf(ok ? "H2_VQE_CROSSCHECK_OK\n" : "H2_VQE_CROSSCHECK_FAILED\n");
    return ok ? 0 : 1;
}
