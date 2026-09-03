#!/usr/bin/env python3
"""Independent numeric oracle for EXP123 receipts (N5 forensic audit).

Science remains in Sounio; this script only *checks* printed keys against
closed-form BW deficit and the G_F Sirlin M_W construction used in
stdlib/particle_physics/ew_precision.sio.

Tolerances are derivation-based, not retrofitted to a single failing run.
"""
from __future__ import annotations

import math
import re
import sys
from pathlib import Path

# PDG-ish centrals matching sm_params.sio
M_Z = 91.1876
M_W_MEAS = 80.377
G_F = 1.1663787e-5
ALPHA = 0.0072973525693
M_T = 172.69
M_H = 125.20
ALPHA_S = 0.1179
SIN2_W = 0.23121
GAMMA_Z = 2.4952
DELTA_ALPHA_HAD = 0.02766
DELTA_ALPHA_TOP = -0.000072


def parse_kv(text: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for line in text.splitlines():
        m = re.match(r"^([A-Z0-9_]+)\s+([-+0-9.eE]+)\s*$", line.strip())
        if m:
            out[m.group(1)] = float(m.group(2))
    return out


def nu_deficit(s: float, mass: float, gamma: float) -> float:
    """deficit = (M Γ)² / [(s−M²)² + (M Γ)²]"""
    re_ = s - mass * mass
    im = mass * gamma
    d2 = re_ * re_ + im * im
    if d2 <= 0.0:
        return 1.0
    return (im * im) / d2


def delta_rho(mt: float, alpha_s: float) -> float:
    pi = math.pi
    delta_qcd = -2.0 * alpha_s / pi * (pi * pi / 9.0 - 1.0 / 6.0)
    return 3.0 * G_F * mt * mt / (8.0 * pi * pi * math.sqrt(2.0)) * (1.0 + delta_qcd)


def delta_alpha_lep(mz: float) -> float:
    pi = math.pi
    masses = (0.0005109989, 0.1056583745, 1.77686)
    s = 0.0
    for mf in masses:
        s += (ALPHA / pi) * ((1.0 / 3.0) * math.log(mz * mz / (mf * mf)) - 5.0 / 9.0)
    return s


def delta_r_rem_bos(mh: float) -> float:
    pi = math.pi
    return 0.0075 + (ALPHA / (4.0 * pi)) * math.log(mh / 100.0)


def m_w_tree() -> float:
    return M_Z * math.sqrt(1.0 - SIN2_W)


def m_w_rho() -> float:
    drho = delta_rho(M_T, ALPHA_S)
    return M_Z * math.sqrt((1.0 - SIN2_W) * (1.0 + drho))


def m_w_gf() -> float:
    pi = math.pi
    a0 = pi * ALPHA / (math.sqrt(2.0) * G_F)
    drho = delta_rho(M_T, ALPHA_S)
    dalpha = delta_alpha_lep(M_Z) + DELTA_ALPHA_HAD + DELTA_ALPHA_TOP
    drem = delta_r_rem_bos(M_H)
    mw = 80.35
    for _ in range(40):
        s2 = 1.0 - (mw / M_Z) ** 2
        c2 = 1.0 - s2
        dr = dalpha - (c2 / s2) * drho + drem
        x = 4.0 * a0 / (M_Z * M_Z * (1.0 - dr))
        mw = M_Z * math.sqrt(0.5 * (1.0 + math.sqrt(1.0 - x)))
    return mw


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: particle_exp123_oracle.py <exp123_stdout.txt>", file=sys.stderr)
        return 2
    text = Path(sys.argv[1]).read_text(encoding="utf-8", errors="replace")
    kv = parse_kv(text)
    fails = 0

    # Deficit at pole
    d_pole = nu_deficit(M_Z * M_Z, M_Z, GAMMA_Z)
    got = kv.get("EXP2_DEFICIT_POLE")
    if got is None:
        print("MISSING EXP2_DEFICIT_POLE")
        fails += 1
    else:
        err = abs(got - d_pole)
        ok = err < 1e-6
        print(f"oracle deficit_pole ref={d_pole:.9f} got={got:.9f} err={err:.3e} {'OK' if ok else 'FAIL'}")
        fails += 0 if ok else 1

    # M_W constructions (central only; pull needs full GUM σ)
    for key, ref_fn, tol in (
        ("EXP3_MW_PRED_TREE", m_w_tree, 5e-3),
        ("EXP3_MW_PRED_RAD", m_w_rho, 5e-3),
        ("EXP3_MW_PRED_GF", m_w_gf, 2e-2),
    ):
        got = kv.get(key)
        ref = ref_fn()
        if got is None:
            print(f"MISSING {key}")
            fails += 1
            continue
        err = abs(got - ref)
        ok = err < tol
        print(f"oracle {key} ref={ref:.6f} got={got:.6f} err={err:.3e} tol={tol} {'OK' if ok else 'FAIL'}")
        fails += 0 if ok else 1

    # Ladder honesty on printed pulls
    pt = kv.get("EXP3_MW_PULL_TREE")
    pr = kv.get("EXP3_MW_PULL_RAD")
    pg = kv.get("EXP3_MW_PULL_GF")
    if None in (pt, pr, pg):
        print("MISSING pull keys")
        fails += 1
    else:
        ok = abs(pr) < abs(pt) and abs(pg) < abs(pr)
        print(f"oracle pull_ladder |tree|={abs(pt):.3f} |rho|={abs(pr):.3f} |gf|={abs(pg):.3f} {'OK' if ok else 'FAIL'}")
        fails += 0 if ok else 1

    if fails:
        print(f"PARTICLE_EXP123_ORACLE_FAIL n={fails}")
        return 1
    print("PARTICLE_EXP123_ORACLE_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
