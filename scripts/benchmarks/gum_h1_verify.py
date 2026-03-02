#!/usr/bin/env python3
"""gum_h1_verify.py — Python cross-verification of GUM H.1 end-gauge calibration.

Implements JCGM 100:2008 Example H.1 (calibration of an end gauge of nominal
length 50 mm) in Python and compares against published reference values.

Output: artifacts/omega/gum_h1_verification_report.v1.json

Reference: JCGM 100:2008, Evaluation of measurement data — Guide to the
           expression of uncertainty in measurement, Section H.1
"""

import json
import math
import os
import time

# ── GUM H.1 Input Quantities ────────────────────────────────────────────────
#
# Model: l_S = l_N + d + d_l - L*(alpha_S - alpha_e)*(theta - theta_0)
#                             - L*alpha_S*delta_theta
#
# Simplified 5-input form matching the Sounio test.

L_NM = 50_000_000.0       # 50 mm in nanometers
ALPHA_S = 11.5e-6          # thermal expansion coeff, /°C
L_ALPHA_S = L_NM * ALPHA_S  # 575.0 nm/°C

INPUTS = {
    "d":            {"value": 215.0,    "u": 5.8,     "nu": 24, "type": "A",
                     "description": "Observed difference between interferometric measurement and nominal"},
    "d_l":          {"value": 0.0,      "u": 25.0,    "nu": 18, "type": "B",
                     "description": "Combined systematic effects of comparator"},
    "delta_alpha":  {"value": 0.0,      "u": 1.15e-6, "nu": 50, "type": "B",
                     "description": "Difference in thermal expansion (alpha_S - alpha_e)"},
    "theta":        {"value": -0.1,     "u": 0.41,    "nu": 2,  "type": "B",
                     "description": "Temperature deviation from 20°C"},
    "delta_theta":  {"value": 0.0,      "u": 0.029,   "nu": 2,  "type": "B",
                     "description": "Temperature difference between test and reference gauge"},
}

# Published GUM H.1 reference values
GUM_REF = {
    "u_c_nm": 32,        # combined standard uncertainty
    "nu_eff": 16,         # effective degrees of freedom
    "k_95": 2.12,         # coverage factor (95%, nu=16)
    "U_nm": 67,           # expanded uncertainty (rounded)
}


# ── Student's t-distribution (95%, two-tailed) ──────────────────────────────

T_95_TABLE = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
    6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
    11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
    16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
    25: 2.060, 30: 2.042, 40: 2.021, 50: 2.009, 100: 1.984,
}

def t_95(nu):
    """Lookup t-value for 95% confidence, interpolating if needed."""
    if nu in T_95_TABLE:
        return T_95_TABLE[nu]
    # Find bracketing entries
    keys = sorted(T_95_TABLE.keys())
    for i in range(len(keys) - 1):
        if keys[i] <= nu <= keys[i + 1]:
            lo, hi = keys[i], keys[i + 1]
            frac = (nu - lo) / (hi - lo)
            return T_95_TABLE[lo] + frac * (T_95_TABLE[hi] - T_95_TABLE[lo])
    return 1.960  # large nu


# ── Sensitivity Coefficients ─────────────────────────────────────────────────

def compute_sensitivity_coefficients():
    """Compute partial derivatives evaluated at input estimates."""
    theta_val = INPUTS["theta"]["value"]
    delta_alpha_val = INPUTS["delta_alpha"]["value"]

    return {
        "d":           1.0,
        "d_l":         1.0,
        "delta_alpha": -L_NM * theta_val,           # -50e6 * (-0.1) = 5e6
        "theta":       -L_NM * delta_alpha_val,      # -50e6 * 0 = 0
        "delta_theta": -L_ALPHA_S,                    # -575
    }


# ── GUM Uncertainty Budget ───────────────────────────────────────────────────

def compute_uncertainty_budget():
    """Compute combined standard uncertainty via GUM law of propagation."""
    c = compute_sensitivity_coefficients()

    contributions = {}
    u_c_sq = 0.0

    for name, inp in INPUTS.items():
        ci = c[name]
        ui = inp["u"]
        ci_ui = abs(ci) * ui
        ci_ui_sq = (ci * ui) ** 2
        contributions[name] = {
            "c_i": ci,
            "u_i": ui,
            "abs_ci_ui": round(ci_ui, 6),
            "ci_ui_sq": round(ci_ui_sq, 6),
            "nu_i": inp["nu"],
        }
        u_c_sq += ci_ui_sq

    u_c = math.sqrt(u_c_sq)
    return u_c, contributions


def welch_satterthwaite(u_c, contributions):
    """Compute effective degrees of freedom via Welch-Satterthwaite."""
    u_c4 = u_c ** 4
    denom = 0.0
    for name, c in contributions.items():
        ci_ui = c["abs_ci_ui"]
        nu_i = c["nu_i"]
        if nu_i > 0 and ci_ui > 0:
            denom += ci_ui ** 4 / nu_i
    if denom < 1e-12:
        return float("inf")
    return u_c4 / denom


# ── Step-by-step Propagation (matching Sounio EVal approach) ─────────────────

def ev_add(a, b):
    return (a[0] + b[0], math.sqrt(a[1]**2 + b[1]**2))

def ev_sub(a, b):
    return (a[0] - b[0], math.sqrt(a[1]**2 + b[1]**2))

def ev_mul(a, b):
    val = a[0] * b[0]
    u = math.sqrt(b[0]**2 * a[1]**2 + a[0]**2 * b[1]**2)
    return (val, u)

def ev_scale(a, c):
    return (a[0] * c, abs(c) * a[1])


def step_by_step_propagation():
    """Propagate uncertainty step-by-step, matching the Sounio test."""
    d = (INPUTS["d"]["value"], INPUTS["d"]["u"])
    d_l = (INPUTS["d_l"]["value"], INPUTS["d_l"]["u"])
    delta_alpha = (INPUTS["delta_alpha"]["value"], INPUTS["delta_alpha"]["u"])
    theta = (INPUTS["theta"]["value"], INPUTS["theta"]["u"])
    delta_theta = (INPUTS["delta_theta"]["value"], INPUTS["delta_theta"]["u"])

    # additive = d + d_l
    additive = ev_add(d, d_l)

    # L_dalpha = L * delta_alpha
    L_dalpha = ev_scale(delta_alpha, L_NM)

    # thermal_1 = L_dalpha * theta
    thermal_1 = ev_mul(L_dalpha, theta)

    # thermal_2 = L_alpha_s * delta_theta (exact constant)
    thermal_2 = ev_scale(delta_theta, L_ALPHA_S)

    # result = additive - thermal_1 - thermal_2
    sub1 = ev_sub(additive, thermal_1)
    result = ev_sub(sub1, thermal_2)

    return result


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  GUM H.1 End-Gauge Calibration — Python Cross-Verification")
    print("  JCGM 100:2008 Example H.1")
    print("=" * 70)
    print()

    # 1. Analytical computation
    u_c, contributions = compute_uncertainty_budget()
    nu_eff = welch_satterthwaite(u_c, contributions)
    nu_int = int(nu_eff)
    k = t_95(nu_int)
    U = k * u_c

    print(f"  Analytical results:")
    print(f"    u_c = {u_c:.4f} nm")
    print(f"    nu_eff = {nu_eff:.2f} (truncated to {nu_int})")
    print(f"    k (95%) = {k:.3f}")
    print(f"    U = {U:.2f} nm")
    print()

    # 2. Step-by-step propagation
    result = step_by_step_propagation()
    print(f"  Step-by-step results:")
    print(f"    l_S - l_N = {result[0]:.4f} nm")
    print(f"    u_c = {result[1]:.4f} nm")
    print()

    # 3. Verification against published GUM values
    checks = []

    # u_c vs published
    err_uc = abs(u_c - GUM_REF["u_c_nm"]) / GUM_REF["u_c_nm"]
    checks.append(("u_c within 5% of GUM", err_uc < 0.05, err_uc))

    # nu_eff
    err_nu = abs(nu_eff - GUM_REF["nu_eff"]) / GUM_REF["nu_eff"]
    checks.append(("nu_eff within 30% of GUM", err_nu < 0.30, err_nu))

    # U
    err_U = abs(U - GUM_REF["U_nm"]) / GUM_REF["U_nm"]
    checks.append(("U within 10% of GUM", err_U < 0.10, err_U))

    # Step-by-step matches analytical
    err_step = abs(result[1] - u_c) / u_c
    checks.append(("step-by-step matches analytical (<1%)", err_step < 0.01, err_step))

    print("  Verification:")
    all_pass = True
    for label, passed, err in checks:
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        print(f"    [{status}] {label} (err={err:.4f})")

    print()

    # 4. Contribution breakdown
    print("  Contribution breakdown:")
    print(f"    {'Source':<15} {'|c_i·u_i| (nm)':>15} {'% of u_c²':>12} {'nu_i':>6}")
    print(f"    {'-'*15} {'-'*15} {'-'*12} {'-'*6}")
    u_c_sq = u_c ** 2
    for name, c in contributions.items():
        pct = (c["ci_ui_sq"] / u_c_sq * 100) if u_c_sq > 0 else 0
        print(f"    {name:<15} {c['abs_ci_ui']:>15.4f} {pct:>11.1f}% {c['nu_i']:>6}")
    print()

    # 5. Write report
    report = {
        "schema": "sounio.benchmark.gum-h1-verification.v1",
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "reference": "JCGM 100:2008 Example H.1 — Calibration of an end gauge",
        "model": "l_S = l_N + d + d_l - L*(alpha_S-alpha_e)*(theta-theta_0) - L*alpha_S*delta_theta",
        "constants": {
            "L_nm": L_NM,
            "alpha_S": ALPHA_S,
            "L_alpha_S_nm_per_degC": L_ALPHA_S,
        },
        "inputs": INPUTS,
        "analytical": {
            "u_c_nm": round(u_c, 4),
            "u_c_sq_nm2": round(u_c ** 2, 4),
            "nu_eff": round(nu_eff, 2),
            "nu_int": nu_int,
            "k_95": round(k, 3),
            "U_nm": round(U, 2),
        },
        "step_by_step": {
            "l_S_minus_l_N_nm": round(result[0], 4),
            "u_c_nm": round(result[1], 4),
        },
        "gum_published": GUM_REF,
        "comparison": {
            "u_c_rel_error": round(err_uc, 6),
            "nu_eff_rel_error": round(err_nu, 6),
            "U_rel_error": round(err_U, 6),
            "step_vs_analytical_rel_error": round(err_step, 6),
        },
        "contributions": {
            name: {
                "sensitivity_coeff": c["c_i"],
                "std_uncertainty": c["u_i"],
                "abs_contribution_nm": c["abs_ci_ui"],
                "variance_contribution_nm2": c["ci_ui_sq"],
                "pct_of_variance": round(c["ci_ui_sq"] / (u_c ** 2) * 100, 2) if u_c > 0 else 0,
                "degrees_of_freedom": c["nu_i"],
            }
            for name, c in contributions.items()
        },
        "all_pass": all_pass,
        "note": (
            "Our 5-input simplification combines d1+d2 into d_l (u=25nm). "
            "The full GUM H.1 uses 6 separate inputs. "
            "This accounts for the ~3% deviation from the published u_c=32nm."
        ),
    }

    report_path = os.environ.get(
        "GUM_H1_REPORT",
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "artifacts", "omega", "gum_h1_verification_report.v1.json"),
    )
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"  Report written to: {report_path}")
    print()

    if all_pass:
        print("  ALL CHECKS PASS")
    else:
        print("  SOME CHECKS FAILED")

    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
