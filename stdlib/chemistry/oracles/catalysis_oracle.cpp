// Enzyme / surface-catalysis rate-law oracle — nine textbook rate laws
// (mass action, Michaelis-Menten in three variants, Hill cooperativity,
// Langmuir-Hinshelwood in two variants, and Monod dual-substrate growth).
//
// An INDEPENDENT check on the Sounio chemistry stdlib's catalysis-mechanism
// implementation. This program is not a translation of that Sounio code and
// was not written by reading it: it is a from-the-textbook-formula
// reimplementation of the same nine rate laws, in a different language and a
// different codebase, meant to be diffed against Sounio's own printed output
// for matching inputs. Agreement between the two corroborates both;
// disagreement is a finding either way. Neither is a replica of the other.
// (This mirrors the role the C++ oracles in the sibling sounio-uhs-coupled
// repo play against PHREEQC — see oracles/chabab_solubility.cpp there for the
// convention this file follows.)
//
// The nine kinds, indexed 0..8, and the exact formula each implements:
//
//   0  mass_action
//        rate = k_rate * c_sub * (has_sub2 ? c_sub2 : 1.0)
//
//   1  Michaelis-Menten, irreversible
//        rate = k_rate * c_cat * c_sub / (km1 + c_sub)
//
//   2  Michaelis-Menten, reversible (Haldane relationship)
//        rate = c_cat * (k_rate*c_sub/km1 - vmax*c_sub2/km2)
//               / (1.0 + c_sub/km1 + c_sub2/km2)
//
//   3  Hill cooperative binding
//        rate = k_rate * c_cat * c_sub^n / (km1^n + c_sub^n)
//
//   4  Langmuir-Hinshelwood, single adsorbed site
//        rate = k_rate * (km1*c_sub) / (1.0 + km1*c_sub)
//
//   5  Langmuir-Hinshelwood, two competing adsorbed sites
//        rate = k_rate * (km1*c_sub) * (km2*c_sub2)
//               / (1.0 + km1*c_sub + km2*c_sub2)^2
//
//   6  Michaelis-Menten, competitive inhibition
//        rate = k_rate * c_cat * c_sub / (km1*(1.0 + c_sub2/km2) + c_sub)
//
//   7  Michaelis-Menten, noncompetitive inhibition
//        rate = k_rate * c_cat * c_sub / ((km1 + c_sub) * (1.0 + c_sub2/km2))
//
//   8  Monod dual-substrate growth
//        rate = k_rate * c_cat * (c_sub/(km1+c_sub))
//               * (has_sub2 ? c_sub2/(km2+c_sub2) : 1.0)
//
// Every division that could hit a zero denominator (km1, km2, or a computed
// sum-of-terms denominator going to exactly zero) is guarded: the guarded
// term reports 0.0 rather than inf or nan. This matches the fail-safe
// behavior of the Sounio side, which is defensive in the same way for the
// same reason — a rate law is not defined at a singular point, and refusing
// to produce a number there is more honest than propagating inf/nan through
// a downstream calculation that will not check for it.
//
// --- CLI -------------------------------------------------------------------
//
//   --kind N          required, integer 0..8, selects the rate law above
//   --k-rate X         rate constant                       (default 0.0)
//   --vmax X           second-branch rate constant (kind 2) (default 0.0)
//   --km1 X            first Michaelis/adsorption constant  (default 0.0)
//   --km2 X            second Michaelis/adsorption constant (default 0.0)
//   --hill-n X         Hill coefficient (kind 3)            (default 1.0)
//   --c-sub X          primary substrate concentration      (default 0.0)
//   --c-sub2 X         secondary substrate/inhibitor/product concentration
//                       -- OMIT THIS FLAG ENTIRELY to mean "no second
//                          substrate" (has_sub2 = false) for kinds 0 and 8,
//                          which branch on its presence. Passing
//                          --c-sub2 0.0 explicitly is NOT the same as
//                          omitting it: it counts as "present, with value
//                          0.0" and, for kind 0, gives rate = 0.0 rather than
//                          rate = k_rate*c_sub. This mirrors the natural
//                          reading of an optional argument: presence and
//                          value are independent signals. (Kinds other than
//                          0 and 8 always use c_sub2 numerically and treat an
//                          omitted flag as 0.0, per the formulas above.)
//   --c-cat X          catalyst / enzyme concentration      (default 0.0)
//   --self-test        run the built-in sanity suite instead of a single
//                       evaluation; ignores all other flags; exit code is
//                       0 iff every case passes
//
// Prints exactly one value to stdout: the computed rate, at
// std::setprecision(17), with no other text, so the output is directly
// script-diffable against Sounio's own printed rate for the same case.
// (--self-test is the one exception: it prints a PASS/FAIL report instead.)
//
// Build: g++ -std=c++23 -O2 -o catalysis_oracle catalysis_oracle.cpp
// Usage: ./catalysis_oracle --kind 1 --k-rate 2.0 --km1 5.0 --c-sub 5.0 --c-cat 3.0
//        ./catalysis_oracle --self-test

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <string>

namespace {

// Guarded division: returns 0.0 instead of inf/nan when the denominator is
// (numerically) zero. Every formula below that divides by something that
// could vanish — km1, km2, or a computed sum of terms — routes through this
// rather than through raw '/'.
double safe_div(double num, double den) {
    if (den == 0.0) return 0.0;
    return num / den;
}

// Guarded power: std::pow(0.0, n) for n <= 0 is inf/nan/1 depending on n's
// sign and parity, which can leak into a Hill-law denominator that is
// otherwise well-guarded by safe_div. Clamp the pathological base==0 with
// non-positive-exponent case to 0.0 up front so it always resolves through
// the ordinary safe_div path below instead of injecting a nan/inf operand.
double safe_pow(double base, double exp) {
    if (base == 0.0 && exp <= 0.0) return 0.0;
    return std::pow(base, exp);
}

struct Params {
    int kind = -1;
    double k_rate = 0.0;
    double vmax = 0.0;
    double km1 = 0.0;
    double km2 = 0.0;
    double hill_n = 1.0;
    double c_sub = 0.0;
    double c_sub2 = 0.0;
    bool has_sub2 = false;
    double c_cat = 0.0;
};

// The nine rate laws. Each takes the full parameter set; each ignores the
// inputs its formula does not use. Guarded against divide-by-zero per the
// header comment above.
double rate_mass_action(const Params& p) {
    return p.k_rate * p.c_sub * (p.has_sub2 ? p.c_sub2 : 1.0);
}

double rate_mm_irreversible(const Params& p) {
    return p.k_rate * p.c_cat * safe_div(p.c_sub, p.km1 + p.c_sub);
}

double rate_mm_reversible(const Params& p) {
    const double denom = 1.0 + safe_div(p.c_sub, p.km1) + safe_div(p.c_sub2, p.km2);
    const double numer = safe_div(p.k_rate * p.c_sub, p.km1)
                        - safe_div(p.vmax * p.c_sub2, p.km2);
    return p.c_cat * safe_div(numer, denom);
}

double rate_hill(const Params& p) {
    const double sub_n = safe_pow(p.c_sub, p.hill_n);
    const double km1_n = safe_pow(p.km1, p.hill_n);
    return p.k_rate * p.c_cat * safe_div(sub_n, km1_n + sub_n);
}

double rate_lh_1site(const Params& p) {
    const double km1_sub = p.km1 * p.c_sub;
    return p.k_rate * safe_div(km1_sub, 1.0 + km1_sub);
}

double rate_lh_2site(const Params& p) {
    const double km1_sub = p.km1 * p.c_sub;
    const double km2_sub2 = p.km2 * p.c_sub2;
    const double denom = std::pow(1.0 + km1_sub + km2_sub2, 2.0);
    return p.k_rate * safe_div(km1_sub * km2_sub2, denom);
}

double rate_mm_competitive_inhib(const Params& p) {
    const double denom = p.km1 * (1.0 + safe_div(p.c_sub2, p.km2)) + p.c_sub;
    return p.k_rate * p.c_cat * safe_div(p.c_sub, denom);
}

double rate_mm_noncompetitive_inhib(const Params& p) {
    const double denom = (p.km1 + p.c_sub) * (1.0 + safe_div(p.c_sub2, p.km2));
    return p.k_rate * p.c_cat * safe_div(p.c_sub, denom);
}

double rate_monod_dual(const Params& p) {
    const double term1 = safe_div(p.c_sub, p.km1 + p.c_sub);
    const double term2 = p.has_sub2 ? safe_div(p.c_sub2, p.km2 + p.c_sub2) : 1.0;
    return p.k_rate * p.c_cat * term1 * term2;
}

double evaluate(const Params& p) {
    switch (p.kind) {
        case 0: return rate_mass_action(p);
        case 1: return rate_mm_irreversible(p);
        case 2: return rate_mm_reversible(p);
        case 3: return rate_hill(p);
        case 4: return rate_lh_1site(p);
        case 5: return rate_lh_2site(p);
        case 6: return rate_mm_competitive_inhib(p);
        case 7: return rate_mm_noncompetitive_inhib(p);
        case 8: return rate_monod_dual(p);
        default:
            std::cerr << "FAIL: --kind must be 0..8, got " << p.kind << "\n";
            std::exit(1);
    }
}

[[noreturn]] void die(const std::string& m) {
    std::cerr << "FAIL: " << m << "\n";
    std::exit(1);
}

double need(const char* s, const char* n) {
    if (!s) die(std::string("missing value for ") + n);
    char* e = nullptr;
    double v = std::strtod(s, &e);
    if (e == s || *e) die(std::string(n) + " is not a number: " + s);
    return v;
}

// Approximate equality for self-test comparisons: relative tolerance for
// magnitudes away from zero, absolute tolerance near zero.
bool nearly_equal(double a, double b, double tol = 1e-12) {
    return std::fabs(a - b) <= tol * std::max({1.0, std::fabs(a), std::fabs(b)});
}

bool run_case(const std::string& name, double got, double expect) {
    bool ok = nearly_equal(got, expect);
    std::cout << (ok ? "PASS  " : "FAIL  ") << name
               << "  got=" << std::setprecision(17) << got
               << "  expect=" << std::setprecision(17) << expect << "\n";
    return ok;
}

int self_test() {
    int failures = 0;
    auto check = [&](const std::string& name, double got, double expect) {
        if (!run_case(name, got, expect)) ++failures;
    };

    // kind 0: mass action, has_sub2=true, hand-computed 2*3*4 = 24.0
    {
        Params p; p.kind = 0; p.k_rate = 2.0; p.c_sub = 3.0;
        p.c_sub2 = 4.0; p.has_sub2 = true;
        check("kind0_mass_action_two_substrates", evaluate(p), 24.0);
    }
    // kind 0: mass action, has_sub2=false -> just k_rate*c_sub = 2*3 = 6.0
    {
        Params p; p.kind = 0; p.k_rate = 2.0; p.c_sub = 3.0; p.has_sub2 = false;
        check("kind0_mass_action_one_substrate", evaluate(p), 6.0);
    }
    // kind 1: MM irreversible at c_sub == km1 -> half-saturation, rate = k_rate*c_cat/2
    {
        Params p; p.kind = 1; p.k_rate = 4.0; p.km1 = 5.0; p.c_sub = 5.0; p.c_cat = 3.0;
        check("kind1_mm_irreversible_half_saturation", evaluate(p), 4.0 * 3.0 / 2.0);
    }
    // kind 1: MM irreversible with km1=0 guarded division -> denom c_sub, no zero-div, sanity check c_sub>0
    {
        Params p; p.kind = 1; p.k_rate = 4.0; p.km1 = 0.0; p.c_sub = 5.0; p.c_cat = 3.0;
        // denom = 0 + 5 = 5, rate = 4*3*5/5 = 12.0
        check("kind1_mm_irreversible_km1_zero", evaluate(p), 12.0);
    }
    // kind 1: MM irreversible with km1=0 AND c_sub=0 -> guarded 0/0 -> 0.0
    {
        Params p; p.kind = 1; p.k_rate = 4.0; p.km1 = 0.0; p.c_sub = 0.0; p.c_cat = 3.0;
        check("kind1_mm_irreversible_zero_over_zero_guarded", evaluate(p), 0.0);
    }
    // kind 2: Haldane reversible, forward-only case: c_sub2=0 collapses to a
    // plain MM-style forward term. km1=km2=2, c_sub=2 -> c_sub/km1=1, vmax
    // term vanishes (c_sub2=0), denom = 1 + 1 + 0 = 2.
    // numer = k_rate*1 - 0 = k_rate. rate = c_cat * k_rate / 2.
    {
        Params p; p.kind = 2; p.k_rate = 6.0; p.vmax = 9.0; p.km1 = 2.0; p.km2 = 2.0;
        p.c_sub = 2.0; p.c_sub2 = 0.0; p.c_cat = 5.0;
        check("kind2_mm_reversible_forward_only", evaluate(p), 5.0 * 6.0 / 2.0);
    }
    // kind 3: Hill law at c_sub == km1 gives exactly k_rate*c_cat/2 for ANY
    // hill_n, since sub^n == km1^n at that point regardless of n.
    {
        for (double n : {0.5, 1.0, 2.0, 4.0, 7.3}) {
            Params p; p.kind = 3; p.k_rate = 10.0; p.km1 = 3.0; p.c_sub = 3.0;
            p.c_cat = 2.0; p.hill_n = n;
            check("kind3_hill_half_saturation_n=" + std::to_string(n),
                  evaluate(p), 10.0 * 2.0 / 2.0);
        }
    }
    // kind 4: Langmuir-Hinshelwood 1-site at km1*c_sub == 1 -> half coverage,
    // rate = k_rate/2.
    {
        Params p; p.kind = 4; p.k_rate = 8.0; p.km1 = 0.5; p.c_sub = 2.0; // km1*c_sub = 1
        check("kind4_lh_1site_half_coverage", evaluate(p), 8.0 / 2.0);
    }
    // kind 4: LH 1-site with c_sub=0 -> coverage 0 -> rate 0.
    {
        Params p; p.kind = 4; p.k_rate = 8.0; p.km1 = 0.5; p.c_sub = 0.0;
        check("kind4_lh_1site_zero_substrate", evaluate(p), 0.0);
    }
    // kind 5: LH 2-site, symmetric case km1*c_sub = km2*c_sub2 = 1 ->
    // denom = (1+1+1)^2 = 9, numer = 1*1 = 1, rate = k_rate/9.
    {
        Params p; p.kind = 5; p.k_rate = 9.0; p.km1 = 0.5; p.c_sub = 2.0; // =1
        p.km2 = 0.25; p.c_sub2 = 4.0; // =1
        check("kind5_lh_2site_symmetric", evaluate(p), 9.0 / 9.0);
    }
    // kind 6: MM competitive inhibition with c_sub2=0 (no inhibitor) reduces
    // to plain MM irreversible at c_sub==km1 -> k_rate*c_cat/2.
    {
        Params p; p.kind = 6; p.k_rate = 3.0; p.km1 = 4.0; p.km2 = 1.0;
        p.c_sub = 4.0; p.c_sub2 = 0.0; p.c_cat = 6.0;
        check("kind6_competitive_inhib_no_inhibitor", evaluate(p), 3.0 * 6.0 / 2.0);
    }
    // kind 7: MM noncompetitive inhibition with c_sub2=0 (no inhibitor)
    // reduces to plain MM irreversible at c_sub==km1 -> k_rate*c_cat/2.
    {
        Params p; p.kind = 7; p.k_rate = 3.0; p.km1 = 4.0; p.km2 = 1.0;
        p.c_sub = 4.0; p.c_sub2 = 0.0; p.c_cat = 6.0;
        check("kind7_noncompetitive_inhib_no_inhibitor", evaluate(p), 3.0 * 6.0 / 2.0);
    }
    // kind 8: Monod dual-substrate, has_sub2=false -> single-Monod term
    // at c_sub==km1 -> k_rate*c_cat/2.
    {
        Params p; p.kind = 8; p.k_rate = 5.0; p.km1 = 2.0; p.c_sub = 2.0;
        p.c_cat = 4.0; p.has_sub2 = false;
        check("kind8_monod_single_substrate", evaluate(p), 5.0 * 4.0 / 2.0);
    }
    // kind 8: Monod dual-substrate, both at half-saturation -> (1/2)*(1/2) factor.
    {
        Params p; p.kind = 8; p.k_rate = 5.0; p.km1 = 2.0; p.c_sub = 2.0;
        p.km2 = 3.0; p.c_sub2 = 3.0; p.c_cat = 4.0; p.has_sub2 = true;
        check("kind8_monod_dual_both_half_saturation", evaluate(p), 5.0 * 4.0 * 0.5 * 0.5);
    }
    // Guard check: division by zero anywhere must yield 0.0, never inf/nan.
    {
        Params p; p.kind = 4; p.k_rate = 8.0; p.km1 = 0.0; p.c_sub = 0.0;
        // km1*c_sub = 0, denom = 1, rate = 0 -- not itself a zero-div, so
        // also directly probe safe_div/safe_pow guards:
        check("guard_safe_div_zero_over_zero", safe_div(0.0, 0.0), 0.0);
        check("guard_safe_div_nonzero_over_zero", safe_div(5.0, 0.0), 0.0);
        check("guard_safe_pow_zero_base_zero_exp", safe_pow(0.0, 0.0), 0.0);
        check("guard_safe_pow_zero_base_neg_exp", safe_pow(0.0, -2.0), 0.0);
    }

    std::cout << "\n" << (failures == 0 ? "ALL PASS" : "FAILURES: " + std::to_string(failures))
               << " (" << (failures == 0 ? "self-test complete" : "see FAIL lines above") << ")\n";
    return failures == 0 ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
    bool self_test_flag = false;
    Params p;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto nxt = [&]() -> const char* { return (i + 1 < argc) ? argv[++i] : nullptr; };
        if (a == "--self-test")        self_test_flag = true;
        else if (a == "--kind")        p.kind = static_cast<int>(need(nxt(), "--kind"));
        else if (a == "--k-rate")      p.k_rate = need(nxt(), "--k-rate");
        else if (a == "--vmax")        p.vmax = need(nxt(), "--vmax");
        else if (a == "--km1")         p.km1 = need(nxt(), "--km1");
        else if (a == "--km2")         p.km2 = need(nxt(), "--km2");
        else if (a == "--hill-n")      p.hill_n = need(nxt(), "--hill-n");
        else if (a == "--c-sub")       p.c_sub = need(nxt(), "--c-sub");
        else if (a == "--c-sub2")      { p.c_sub2 = need(nxt(), "--c-sub2"); p.has_sub2 = true; }
        else if (a == "--c-cat")       p.c_cat = need(nxt(), "--c-cat");
        else die("unknown argument: " + a);
    }

    if (self_test_flag) return self_test();

    if (p.kind < 0 || p.kind > 8)
        die("need --kind N with N in 0..8");

    const double rate = evaluate(p);
    std::cout << std::setprecision(17) << rate << "\n";
    return 0;
}
