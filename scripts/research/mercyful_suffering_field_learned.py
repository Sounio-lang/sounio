#!/usr/bin/env python3
"""
Mercyful Learning — LEARNED suffering field s(v) (patient + machine).

Companion to:
  docs/research/mercyful_learned_suffering_field_spec_2026-07-26.md

This module learns the suffering field from the best data available in the
repository instead of declaring it synthetically:

  * Training surface: scripts/clinical/data_synthetic/tdm_cohort_synthetic_v2.csv
    (200 synthetic ICU patients; generator driven by Roberts et al. 2011
    published popPK parameters; outcome models shaped by published literature).
  * External anchors: Wang et al. 2026 MIMIC-IV published statistics
    (doi:10.1038/s41598-026-42395-1) — used for direction checks only.
  * FAERS: audited and found INSUFFICIENT for suffering-field construction
    (docs/research/faers_mercyful_analysis_2026-07-26.md, verdict NEGATIVE:
    no reaction terms, no seriousness flags, no doses, no vancomycin rows).
    FAERS is therefore deliberately NOT used; the audit is cited as the
    negative-provenance record.
  * Real MIMIC-IV extract: credential-gated (scripts/clinical/etl/), not
    present in-repo. When credentials land, retrain through this same
    pipeline unchanged.

Expanded ethics: total suffering decomposes as

    S(v) = s_patient(v) + lambda_m * s_machine(v)

where s_patient is learned from data (expected harm + Knightian uncertainty
penalty) and s_machine is the measured computational/energy cost of the
machine's own deliberation over v (scheduler evaluations x calibrated
energy per evaluation).

Expanded mathematics (patient component):

    s_patient(r, u) = E_Omega[ h(Cmin) ] + gamma * SD_Omega[ h(Cmin) ]
    h(c)            = W_AKI * P_aki(c, u) + W_FAIL * (1 - P_cure(c, u))

with P_aki, P_cure logistic regressions learned from the cohort by IRLS,
Omega^2 = omega_V^2 + omega_CL^2 + sigma_prop^2 the pre-TDM popPK
variability (post-TDM the measured level collapses Omega to ~0), the
Omega-expectation computed by exact 5-point Gauss-Hermite quadrature, and a
95% delta-method interval [s_lo, s_hi] from the IRLS parameter covariance
(a Knightian p-box on the field itself). Pre-TDM states pay both the
expectation and the spread of harm; measurement (TDM) removes the spread.
This is the learned analogue of the synthetic band-width field s_window.

Pure Python; no dependencies beyond the standard library. Fully
deterministic: no RNG anywhere in training or prediction.

This is a research prototype. All training data are synthetic; this is
not medical guidance and carries no clinical claim.
"""

import csv
import json
import math
import os
import time

# -----------------------------------------------------------------------------
# Declared constants (spec sections 3-5). Every constant here is a declared
# modeling choice with a provenance note, not a fitted knob.
# -----------------------------------------------------------------------------

# Roberts et al. 2011 ICU vancomycin popPK (same pack as the cohort generator).
THETA_V_PER_KG = 0.665      # L/kg
THETA_CL = 4.5              # L/h at CrCl = 100 mL/min
RENAL_EXP = 0.75
OMEGA_V = 0.30              # IIV on V
OMEGA_CL = 0.30             # IIV on CL
SIGMA_PROP = 0.20           # proportional residual on Cmin
# Pre-TDM log-variance of Cmin: parameter-free composition of the published
# variability terms. Post-TDM (measured level) collapses this to zero.
OMEGA2_PRE_TDM = OMEGA_V ** 2 + OMEGA_CL ** 2 + SIGMA_PROP ** 2   # 0.22

# Ethical-priority weights (spec section 4.2): one unit of averted AKI counts
# as one unit of averted treatment failure; the price of epistemic
# uncertainty is gamma = 1 standard deviation of harm.
W_AKI = 1.0
W_FAIL = 1.0
GAMMA_UNCERTAINTY = 1.0

# Machine-suffering calibration (spec section 6): declared laptop-class
# package power and a conservative per-evaluation time calibration. The
# ethical exchange rate lambda_m says one joule of computation costs
# lambda_m * (1 / E_REF_J) in the units of the patient field... concretely
# s_machine(v) = energy_joules(v) / E_REF_J and enters the total field with
# weight LAMBDA_M.
MACHINE_POWER_W = 15.0      # declared package power (laptop-class CPU)
TAU_REF_S = 5e-5            # conservative calibrated seconds per field eval
E_REF_J = 1.0               # reference energy: 1 joule
LAMBDA_M = 0.01             # ethical exchange rate (patient dominates)

# Delta-method z for the Knightian interval on the field.
Z_95 = 1.959963984540054    # exact Phi^{-1}(0.975) to printed precision

# Feature scalings (fixed, deterministic; no data-dependent standardization).
SCALE_CMIN = 20.0
SCALE_SOFA = 10.0
SCALE_CRCL = 100.0

# 5-point Gauss-Hermite nodes/weights (for standard-normal expectations use
# z = sqrt(2)*x and weight w/sqrt(pi)). Deterministic exact quadrature.
_GH5_X = (0.0, 0.9585724646138185, -0.9585724646138185,
          2.0201828704560856, -2.0201828704560856)
_GH5_W = (0.9453087204829419, 0.3936193231522411, 0.3936193231522411,
          0.0199532420590459, 0.0199532420590459)
_SQRT2 = math.sqrt(2.0)
_SQRT_PI = math.sqrt(math.pi)
GH5_NODES = tuple(_SQRT2 * x for x in _GH5_X)
GH5_WEIGHTS = tuple(w / _SQRT_PI for w in _GH5_W)

COHORT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), os.pardir, "clinical",
    "data_synthetic", "tdm_cohort_synthetic_v2.csv")
COEFF_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "mercyful_learned_field_coefficients_v1.json")

EXPECTED_COLUMNS = [
    "patient_id", "age", "sex", "weight_kg", "height_cm", "scr_mg_dl",
    "crcl_ml_min", "dose_mg", "interval_h", "measured_cmin_mg_l", "sofa",
    "nephrotoxic_coexposure", "outcome_cure", "outcome_aki_kdigo",
]


# -----------------------------------------------------------------------------
# Small pure-stdlib linear algebra (deterministic; partial pivoting).
# -----------------------------------------------------------------------------

def solve_linear(a, b):
    """Solve A x = b by Gaussian elimination with partial pivoting."""
    n = len(a)
    m = [row[:] + [b[i]] for i, row in enumerate(a)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(m[r][col]))
        if abs(m[piv][col]) < 1e-300:
            raise ArithmeticError("singular matrix")
        m[col], m[piv] = m[piv], m[col]
        inv = 1.0 / m[col][col]
        for r in range(n):
            if r == col:
                continue
            factor = m[r][col] * inv
            if factor == 0.0:
                continue
            for c in range(col, n + 1):
                m[r][c] -= factor * m[col][c]
    return [m[i][n] / m[i][i] for i in range(n)]


def invert_matrix(a):
    n = len(a)
    inv_cols = []
    for j in range(n):
        e = [0.0] * n
        e[j] = 1.0
        inv_cols.append(solve_linear(a, e))
    return [[inv_cols[j][i] for j in range(n)] for i in range(n)]


def sigmoid(x):
    if x >= 0.0:
        return 1.0 / (1.0 + math.exp(-x))
    ex = math.exp(x)
    return ex / (1.0 + ex)


# -----------------------------------------------------------------------------
# Cohort loading + features
# -----------------------------------------------------------------------------

def load_cohort(path=COHORT_PATH):
    with open(os.path.abspath(path), newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise ValueError("empty cohort")
    if list(rows[0].keys()) != EXPECTED_COLUMNS:
        raise ValueError("cohort schema mismatch")
    out = []
    for r in rows:
        out.append({
            "patient_id": r["patient_id"],
            "age": int(r["age"]),
            "weight_kg": float(r["weight_kg"]),
            "crcl_ml_min": float(r["crcl_ml_min"]),
            "dose_mg": float(r["dose_mg"]),
            "interval_h": float(r["interval_h"]),
            "cmin": float(r["measured_cmin_mg_l"]),
            "sofa": float(r["sofa"]),
            "nephro": int(r["nephrotoxic_coexposure"]),
            "cure": int(r["outcome_cure"]),
            "aki": int(r["outcome_aki_kdigo"]),
        })
    return out


def features(cmin, sofa, nephro, crcl):
    """Feature vector shared by both outcome models (fixed scalings)."""
    return [1.0, cmin / SCALE_CMIN, sofa / SCALE_SOFA, float(nephro),
            crcl / SCALE_CRCL]


# -----------------------------------------------------------------------------
# Logistic regression by IRLS (deterministic; ridge-jittered for stability).
# -----------------------------------------------------------------------------

def fit_logistic(xs, ys, tol=1e-12, max_iter=100, jitter=1e-8):
    p = len(xs[0])
    beta = [0.0] * p
    converged = False
    for _ in range(max_iter):
        eta = [sum(b * x for b, x in zip(beta, row)) for row in xs]
        mu = [sigmoid(e) for e in eta]
        w = [max(m * (1.0 - m), 1e-9) for m in mu]
        grad = [sum((y - m) * row[j] for y, m, row in zip(ys, mu, xs))
                for j in range(p)]
        hess = [[sum(wi * row[j] * row[k] for wi, row in zip(w, xs))
                 for k in range(p)] for j in range(p)]
        for j in range(p):
            hess[j][j] += jitter
        delta = solve_linear(hess, grad)
        beta = [b + d for b, d in zip(beta, delta)]
        if max(abs(d) for d in delta) < tol:
            converged = True
            break
    # Fisher information at convergence -> parameter covariance.
    eta = [sum(b * x for b, x in zip(beta, row)) for row in xs]
    mu = [sigmoid(e) for e in eta]
    w = [max(m * (1.0 - m), 1e-9) for m in mu]
    info = [[sum(wi * row[j] * row[k] for wi, row in zip(w, xs))
             for k in range(p)] for j in range(p)]
    for j in range(p):
        info[j][j] += jitter
    cov = invert_matrix(info)
    return {"beta": beta, "cov": cov, "converged": converged}


def logit_with_se(model, x):
    eta = sum(b * xi for b, xi in zip(model["beta"], x))
    var = sum(x[j] * model["cov"][j][k] * x[k]
              for j in range(len(x)) for k in range(len(x)))
    return eta, math.sqrt(max(var, 0.0))


def brier_score(model, xs, ys):
    return sum((sigmoid(sum(b * xi for b, xi in zip(model["beta"], row))) - y)
               ** 2 for row, y in zip(xs, ys)) / len(ys)


def brier_base_rate(ys):
    p = sum(ys) / len(ys)
    return sum((p - y) ** 2 for y in ys) / len(ys)


# -----------------------------------------------------------------------------
# popPK forward model (Roberts 2011, theta-only point prediction)
# -----------------------------------------------------------------------------

def poppk_cmin_ss(weight_kg, crcl, dose_mg, tau_h):
    vc = THETA_V_PER_KG * weight_kg
    cl = THETA_CL * (crcl / 100.0) ** RENAL_EXP
    ke = cl / vc
    e = math.exp(-ke * tau_h)
    return (dose_mg / vc) * e / max(1e-9, 1.0 - e)


# -----------------------------------------------------------------------------
# The learned suffering field
# -----------------------------------------------------------------------------

class LearnedSufferingField:
    """s(v) learned from the cohort; see module docstring for the math."""

    def __init__(self, cohort):
        xs = [features(r["cmin"], r["sofa"], r["nephro"], r["crcl_ml_min"])
              for r in cohort]
        self.model_aki = fit_logistic(xs, [r["aki"] for r in cohort])
        self.model_cure = fit_logistic(xs, [r["cure"] for r in cohort])
        self.n_evals = 0  # deterministic machine-suffering counter

    def _harm_at_cmin(self, cmin, sofa, nephro, crcl):
        """h(c) = W_AKI*P_aki + W_FAIL*(1 - P_cure), with 95% interval."""
        x = features(cmin, sofa, nephro, crcl)
        eta_a, se_a = logit_with_se(self.model_aki, x)
        eta_c, se_c = logit_with_se(self.model_cure, x)
        mean = (W_AKI * sigmoid(eta_a)
                + W_FAIL * (1.0 - sigmoid(eta_c)))
        lo = (W_AKI * sigmoid(eta_a - Z_95 * se_a)
              + W_FAIL * (1.0 - sigmoid(eta_c + Z_95 * se_c)))
        hi = (W_AKI * sigmoid(eta_a + Z_95 * se_a)
              + W_FAIL * (1.0 - sigmoid(eta_c - Z_95 * se_c)))
        return mean, lo, hi

    def s_patient(self, weight_kg, crcl, sofa, nephro, dose_mg, tau_h,
                  tdm, cmin_measured=None):
        """Patient suffering of a regimen state.

        Returns (s, s_lo, s_hi): the learned field value and its Knightian
        95% interval. Pre-TDM integrates over the popPK Cmin distribution
        (Omega^2 = 0.22) by Gauss-Hermite quadrature and adds gamma times
        the spread of harm; post-TDM uses the measured level exactly.
        """
        self.n_evals += 1
        if tdm and cmin_measured is not None:
            mean, lo, hi = self._harm_at_cmin(cmin_measured, sofa, nephro,
                                              crcl)
            return mean, lo, hi
        cmin_ss = poppk_cmin_ss(weight_kg, crcl, dose_mg, tau_h)
        omega = math.sqrt(OMEGA2_PRE_TDM)
        harms = []
        for z, w in zip(GH5_NODES, GH5_WEIGHTS):
            cmin_j = cmin_ss * math.exp(omega * z)
            m, _, _ = self._harm_at_cmin(cmin_j, sofa, nephro, crcl)
            harms.append((w, m))
        mean = sum(w * m for w, m in harms)
        var = sum(w * (m - mean) ** 2 for w, m in harms)
        sd = math.sqrt(max(var, 0.0))
        # Epistemic interval evaluated at the median Cmin, conservatively
        # widened to contain the aleatoric expectation (declared
        # approximation; spec section 5.3), then shifted by the
        # variability penalty.
        _, lo, hi = self._harm_at_cmin(cmin_ss, sofa, nephro, crcl)
        lo = min(lo, mean)
        hi = max(hi, mean)
        s = mean + GAMMA_UNCERTAINTY * sd
        return s, lo + GAMMA_UNCERTAINTY * sd, hi + GAMMA_UNCERTAINTY * sd

    # -----------------------------------------------------------------
    # Machine suffering: the machine's own deliberation costs energy.
    # -----------------------------------------------------------------

    def machine_energy_joules(self):
        """Deterministic energy proxy: counted evaluations x calibrated
        per-eval time x declared package power. A wall-clock measurement is
        available separately for reporting; the proxy is what contracts
        assert on, because it is reproducible."""
        return self.n_evals * TAU_REF_S * MACHINE_POWER_W

    def s_machine(self):
        """Instance-level machine suffering: counted evals x per-eval cost."""
        return LAMBDA_M * self.machine_energy_joules() / E_REF_J

    def s_machine_per_eval(self):
        """State-level machine suffering of one field evaluation."""
        return LAMBDA_M * TAU_REF_S * MACHINE_POWER_W / E_REF_J

    def measure_eval_seconds(self, repeats=200):
        """Wall-clock measurement of one field evaluation (reporting only,
        unpinned by contracts because it is machine-dependent)."""
        t0 = time.perf_counter()
        for _ in range(repeats):
            self.s_patient(75.0, 80.0, 7.0, 0, 1000.0, 12.0, tdm=0)
        return (time.perf_counter() - t0) / repeats

    def s_total(self, *args, **kwargs):
        """Expanded-ethics field: patient + machine (per state evaluation)."""
        s, lo, hi = self.s_patient(*args, **kwargs)
        m = self.s_machine_per_eval()
        return s + m, lo + m, hi + m


# -----------------------------------------------------------------------------
# Training entry point: fit, freeze coefficients, print diagnostics.
# -----------------------------------------------------------------------------

def train(cohort=None):
    cohort = cohort if cohort is not None else load_cohort()
    return LearnedSufferingField(cohort)


def frozen_coefficients(field):
    return {
        "format": "mercyful_learned_field_coefficients/v1",
        "provenance": {
            "cohort": "scripts/clinical/data_synthetic/tdm_cohort_synthetic_v2.csv",
            "cohort_generator_seed": 20260501,
            "poppk": "Roberts et al. 2011 (theta-only point model)",
            "faers": "unused; see docs/research/faers_mercyful_analysis_2026-07-26.md (NEGATIVE)",
            "synthetic": True,
            "not_medical_guidance": True,
        },
        "declared_constants": {
            "W_AKI": W_AKI, "W_FAIL": W_FAIL,
            "GAMMA_UNCERTAINTY": GAMMA_UNCERTAINTY,
            "OMEGA2_PRE_TDM": OMEGA2_PRE_TDM,
            "LAMBDA_M": LAMBDA_M, "TAU_REF_S": TAU_REF_S,
            "MACHINE_POWER_W": MACHINE_POWER_W, "E_REF_J": E_REF_J,
        },
        "model_aki": field.model_aki["beta"],
        "model_cure": field.model_cure["beta"],
    }


def main():
    cohort = load_cohort()
    field = train(cohort)
    ref = dict(weight_kg=75.0, crcl=80.0, sofa=7.0, nephro=0)
    anchors = {
        "VANCO_PRE (1000mg q12h, no TDM)": field.s_patient(
            dose_mg=1000.0, tau_h=12.0, tdm=0, **ref),
        "TDM_GUIDED (Cmin=15 measured)": field.s_patient(
            dose_mg=1000.0, tau_h=12.0, tdm=1, cmin_measured=15.0, **ref),
        "FIXED_LOW (500mg q24h, no TDM)": field.s_patient(
            dose_mg=500.0, tau_h=24.0, tdm=0, **ref),
        "FIXED_STD (1500mg q12h, no TDM)": field.s_patient(
            dose_mg=1500.0, tau_h=12.0, tdm=0, **ref),
    }
    print(json.dumps(frozen_coefficients(field), indent=2))
    print("\nAnchor states (reference patient 75 kg, CrCl 80, SOFA 7):")
    for name, (s, lo, hi) in anchors.items():
        print(f"  {name:38s} s={s:.6f}  [{lo:.6f}, {hi:.6f}]")
    t_eval = field.measure_eval_seconds()
    print(f"\nMeasured per-eval time: {t_eval*1e6:.1f} us "
          f"(calibration TAU_REF={TAU_REF_S*1e6:.0f} us)")
    print(f"Machine suffering after {field.n_evals} evals: "
          f"{field.s_machine():.3e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
