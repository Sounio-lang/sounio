#!/usr/bin/env python3
"""generate_realistic_cohort.py

High-fidelity synthetic vancomycin TDM cohort generator.

Produces a CSV cohort that exhibits the joint distributions of:
  - age, sex, weight, height
  - serum creatinine and Cockcroft-Gault CrCl
  - vancomycin dose / interval (TDM-driven adjustments)
  - measured Cmin (sampled from a popPK forward model with realistic
    intra- and inter-subject variability)
  - SOFA score, nephrotoxic co-exposure, clinical outcomes
    (cure / AKI by KDIGO grade)

The forward model uses Roberts et al. (2011) ICU vancomycin popPK
parameters with proportional residual variability:

  Vc  = θ_V · (WT/70)^1.0  · exp(η_V),   η_V ~ N(0, ω_V²)
  CL  = θ_CL · (CrCl/100)^0.75 · exp(η_CL), η_CL ~ N(0, ω_CL²)

  θ_V    = 0.665 L/kg      ⇒ ~46.5 L for 70-kg patient
  θ_CL   = 4.5 L/h         ⇒ scaled by (CrCl/100)^0.75
  ω_V    = 0.30  (30% IIV)
  ω_CL   = 0.30  (30% IIV)
  σ_prop = 0.20  (20% proportional residual variability on Cmin)

For each patient we compute the steady-state trough (Cmin):

  ke    = CL / Vc
  Cmin  = (Dose / Vc) · exp(-ke · τ) / (1 - exp(-ke · τ))

then add proportional residual: Cmin_obs = Cmin · (1 + ε), ε ~ N(0, σ_prop²).

Outcome model (clinically-grounded, not real cohort):
  P(cure)        = sigmoid(  0.4·(Cmin_obs - 8) − 0.05·SOFA  )
  P(AKI grade≥1) = sigmoid( -2.5 + 0.18·(Cmin_obs - 15)
                             + 0.6·nephro + 0.05·SOFA )

This is **synthetic skeleton data** for pipeline-validation only.
DO NOT use for inference. Replace with institutional TDM cohort or
MIMIC-IV extract (see scripts/clinical/etl/mimic_iv_vancomycin.sql)
once credentials are in hand.

Usage:
  python3 generate_realistic_cohort.py [--n 200] [--seed 20260501]
                                       [--output cohort.csv]

Defaults to n=200, seed=20260501 (deterministic for CI reproducibility).
"""

from __future__ import annotations

import argparse
import csv
import math
import random
from pathlib import Path


# --- popPK parameter pack (Roberts 2011 ICU vancomycin) ----------------------

THETA_V_PER_KG = 0.665      # L/kg
THETA_CL = 4.5              # L/h at CrCl=100
ALLO_EXP = 1.0              # allometric exponent for V on weight
RENAL_EXP = 0.75            # exponent for CL on CrCl/100
OMEGA_V = 0.30              # 30% IIV on V
OMEGA_CL = 0.30             # 30% IIV on CL
SIGMA_PROP = 0.20           # 20% proportional residual on Cmin


def cockcroft_gault(age: int, sex: str, weight_kg: float, scr_mg_dl: float) -> float:
    """Cockcroft-Gault CrCl (mL/min)."""
    factor = 0.85 if sex.upper().startswith("F") else 1.0
    crcl = (140 - age) * weight_kg * factor / (72.0 * scr_mg_dl)
    return max(5.0, crcl)


def cmin_steady_state(dose_mg: float, interval_h: float, vc_l: float,
                      cl_l_h: float) -> float:
    """Single-compartment IV-bolus steady-state trough (mg/L)."""
    ke = cl_l_h / vc_l
    tau = interval_h
    e = math.exp(-ke * tau)
    return (dose_mg / vc_l) * e / max(1e-9, 1.0 - e)


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def sample_one_patient(rng: random.Random, pid: str) -> dict:
    """Draw one patient with joint distributions consistent with ICU TDM."""
    # Demographics: ICU-skewed.
    age = rng.randint(18, 88)
    sex = rng.choice(["M", "F"])
    if sex == "M":
        height = round(rng.gauss(175, 8), 0)
        weight = max(45, round(rng.gauss(80, 14), 1))
    else:
        height = round(rng.gauss(162, 7), 0)
        weight = max(40, round(rng.gauss(68, 13), 1))
    # SCr — log-normal around 1.0, ICU-skewed high.
    scr = max(0.4, round(math.exp(rng.gauss(0.05, 0.4)), 2))
    crcl = round(cockcroft_gault(age, sex, weight, scr), 0)

    # popPK draw.
    eta_v = rng.gauss(0, OMEGA_V)
    eta_cl = rng.gauss(0, OMEGA_CL)
    vc = THETA_V_PER_KG * (weight ** ALLO_EXP) * math.exp(eta_v)
    cl = THETA_CL * ((crcl / 100.0) ** RENAL_EXP) * math.exp(eta_cl)

    # Initial dose: weight-based 15 mg/kg, q12 if CrCl ≥ 50 else q24.
    dose_per_kg = rng.choice([15.0, 17.5, 20.0])
    raw_dose = round(dose_per_kg * weight / 250.0) * 250.0  # nearest 250 mg
    dose = max(500.0, min(2000.0, raw_dose))
    interval = 12 if crcl >= 50 else 24

    # Cmin from popPK + proportional residual.
    cmin_pred = cmin_steady_state(dose, interval, vc, cl)
    eps = rng.gauss(0, SIGMA_PROP)
    cmin_obs = max(0.5, round(cmin_pred * (1.0 + eps), 1))

    # SOFA score (0-24, ICU-realistic).
    sofa = max(0, min(24, round(rng.gauss(7, 3))))
    nephrotoxic = 1 if rng.random() < 0.35 else 0

    # Outcome models.
    p_cure = sigmoid(0.4 * (cmin_obs - 8) - 0.05 * sofa)
    p_aki = sigmoid(-2.5 + 0.18 * (cmin_obs - 15) + 0.6 * nephrotoxic
                    + 0.05 * sofa)
    cure = 1 if rng.random() < p_cure else 0
    aki = 1 if rng.random() < p_aki else 0

    return {
        "patient_id": pid,
        "age": age,
        "sex": sex,
        "weight_kg": weight,
        "height_cm": int(height),
        "scr_mg_dl": scr,
        "crcl_ml_min": int(crcl),
        "dose_mg": int(dose),
        "interval_h": interval,
        "measured_cmin_mg_l": cmin_obs,
        "sofa": sofa,
        "nephrotoxic_coexposure": nephrotoxic,
        "outcome_cure": cure,
        "outcome_aki_kdigo": aki,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate a high-fidelity synthetic vancomycin TDM cohort."
    )
    parser.add_argument("--n", type=int, default=200,
                        help="Number of patients to generate.")
    parser.add_argument("--seed", type=int, default=20260501,
                        help="Deterministic RNG seed.")
    parser.add_argument(
        "--output", type=Path,
        default=Path(__file__).parent / "tdm_cohort_synthetic_v2.csv",
        help="Output CSV path.")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    patients = [sample_one_patient(rng, f"S{i:04d}")
                for i in range(1, args.n + 1)]

    fieldnames = list(patients[0].keys())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for p in patients:
            w.writerow(p)

    # Quick summary to stdout.
    cmins = [p["measured_cmin_mg_l"] for p in patients]
    crcls = [p["crcl_ml_min"] for p in patients]
    therapeutic = sum(1 for c in cmins if 10 <= c <= 20)
    sub = sum(1 for c in cmins if c < 10)
    tox = sum(1 for c in cmins if c > 20)
    aki_n = sum(p["outcome_aki_kdigo"] for p in patients)
    cure_n = sum(p["outcome_cure"] for p in patients)
    print(f"Wrote {args.output} ({len(patients)} patients)")
    print(f"  Cmin mean        = {sum(cmins)/len(cmins):.1f} mg/L")
    print(f"  CrCl mean        = {sum(crcls)/len(crcls):.0f} mL/min")
    print(f"  Subtherapeutic   = {sub:>3} ({100*sub/len(cmins):.0f}%)")
    print(f"  Therapeutic      = {therapeutic:>3} "
          f"({100*therapeutic/len(cmins):.0f}%)")
    print(f"  Toxic            = {tox:>3} ({100*tox/len(cmins):.0f}%)")
    print(f"  AKI (KDIGO ≥ 1)  = {aki_n:>3} ({100*aki_n/len(patients):.0f}%)")
    print(f"  Clinical cure    = {cure_n:>3} "
          f"({100*cure_n/len(patients):.0f}%)")
    print()
    print("CAVEAT: this is synthetic data — DO NOT use for inference.")
    print("Replace with MIMIC-IV extract once credentials are in hand.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
