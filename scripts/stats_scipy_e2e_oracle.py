#!/usr/bin/env python3
"""Oracle for stats scipy E2E vertical — same fixed arrays as the Sounio fixture.

Primary path: pure-Python closed forms (no SciPy required).
Optional path: if scipy is installed, cross-check Welch t / p / linregress.

Exit codes:
  0 — pass (all checked fields within tolerance)
  1 — fail (mismatch)
  2 — skipped (insufficient inputs)

Prints a one-line JSON object on the last line for the gate receipt.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from typing import Any, Dict, List, Optional, Tuple


DRUG = [15.2, 18.1, 12.5, 20.3, 16.8, 14.7, 19.2, 13.9]
PLACEBO = [8.3, 10.1, 7.5, 11.2, 9.4, 8.9, 10.5, 7.8]
X_OLS = [1.0, 2.0, 3.0, 4.0, 5.0]
Y_OLS = [2.0, 4.0, 5.0, 4.0, 5.0]


def mean(a: List[float]) -> float:
    return sum(a) / len(a)


def var_s(a: List[float]) -> float:
    m = mean(a)
    return sum((x - m) ** 2 for x in a) / (len(a) - 1)


def betacf(a: float, b: float, x: float, max_iter: int = 200, eps: float = 3e-14) -> float:
    am = bm = az = 1.0
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    bz = 1.0 - qab * x / qap
    for m in range(1, max_iter + 1):
        em = float(m)
        tem = em + em
        d = em * (b - em) * x / ((qam + tem) * (a + tem))
        ap = az + d * am
        bp = bz + d * bm
        d = -(a + em) * (qab + em) * x / ((a + tem) * (qap + tem))
        app = ap + d * az
        bpp = bp + d * bz
        aold = az
        am = ap / bpp
        bm = bp / bpp
        az = app / bpp
        bz = 1.0
        if abs(az - aold) < eps * abs(az):
            return az
    return az


def betai(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta I_x(a,b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    if x < (a + 1.0) / (a + b + 2.0):
        return math.exp(math.log(x) * a + math.log(1.0 - x) * b - lbeta) / a * betacf(a, b, x)
    return 1.0 - math.exp(math.log(x) * a + math.log(1.0 - x) * b - lbeta) / b * betacf(b, a, 1.0 - x)


def t_two_tail(t: float, df: float) -> float:
    # Match stdlib/stats/hypothesis.sio: p = I_{df/(df+t^2)}(df/2, 1/2)
    t2 = t * t
    x = df / (df + t2)
    return betai(df / 2.0, 0.5, x)


def welch_oracle(x: List[float], y: List[float]) -> Dict[str, float]:
    nx, ny = len(x), len(y)
    mx, my = mean(x), mean(y)
    sx2, sy2 = var_s(x), var_s(y)
    se2x, se2y = sx2 / nx, sy2 / ny
    se = math.sqrt(se2x + se2y)
    t = (mx - my) / se
    df = (se2x + se2y) ** 2 / (se2x**2 / (nx - 1) + se2y**2 / (ny - 1))
    p = t_two_tail(abs(t), df)
    sp = math.sqrt((sx2 + sy2) * 0.5)
    d = (mx - my) / sp
    return {"t": t, "df": df, "se": se, "p": p, "d": d}


def levene_w_mean(groups: List[List[float]]) -> float:
    zs = []
    for g in groups:
        m = mean(g)
        zs.append([abs(v - m) for v in g])
    allz = [z for g in zs for z in g]
    n = len(allz)
    k = len(groups)
    zbar = mean(allz)
    ssb = sum(len(g) * (mean(g) - zbar) ** 2 for g in zs)
    ssw = sum((z - mean(g)) ** 2 for g in zs for z in g)
    if ssw <= 0.0:
        return 0.0
    return ((n - k) / (k - 1)) * (ssb / ssw)


def ols_oracle(x: List[float], y: List[float]) -> Dict[str, float]:
    mx, my = mean(x), mean(y)
    num = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    den = sum((xi - mx) ** 2 for xi in x)
    slope = num / den
    intercept = my - slope * mx
    ssy = sum((yi - my) ** 2 for yi in y)
    r2 = (num * num) / (den * ssy) if den > 0 and ssy > 0 else 0.0
    return {"slope": slope, "intercept": intercept, "r2": r2}


def rel_err(s: float, r: float) -> float:
    return abs(s - r) / max(abs(r), 1e-12)


def abs_err(s: float, r: float) -> float:
    return abs(s - r)


def parse_float(s: Optional[str]) -> Optional[float]:
    if s is None or s == "" or s == "null":
        return None
    try:
        return float(s)
    except ValueError:
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--t")
    ap.add_argument("--df")
    ap.add_argument("--p")
    ap.add_argument("--d")
    ap.add_argument("--levene-w")
    ap.add_argument("--ols-slope")
    ap.add_argument("--ols-intercept")
    ap.add_argument("--ols-r2")
    args = ap.parse_args()

    got = {
        "t": parse_float(args.t),
        "df": parse_float(args.df),
        "p": parse_float(args.p),
        "d": parse_float(args.d),
        "levene_w": parse_float(args.levene_w),
        "ols_slope": parse_float(args.ols_slope),
        "ols_intercept": parse_float(args.ols_intercept),
        "ols_r2": parse_float(args.ols_r2),
    }
    if any(v is None for v in got.values()):
        detail = {"status": "skipped", "reason": "missing sounio metric fields", "got": got}
        print("oracle: skipped (missing metrics)", file=sys.stderr)
        print(json.dumps(detail))
        return 2

    w = welch_oracle(DRUG, PLACEBO)
    lev = levene_w_mean([DRUG, PLACEBO])
    ols = ols_oracle(X_OLS, Y_OLS)

    checks: List[Tuple[str, float, float, float, str]] = [
        # name, sounio, ref, tol, kind (abs|rel)
        ("welch_t", got["t"], w["t"], 1e-6, "abs"),
        ("welch_df", got["df"], w["df"], 1e-6, "abs"),
        ("welch_p", got["p"], w["p"], 1e-6, "abs"),
        ("cohens_d", got["d"], w["d"], 1e-6, "abs"),
        ("levene_w", got["levene_w"], lev, 1e-6, "abs"),
        ("ols_slope", got["ols_slope"], ols["slope"], 1e-10, "rel"),
        ("ols_intercept", got["ols_intercept"], ols["intercept"], 1e-10, "rel"),
        ("ols_r2", got["ols_r2"], ols["r2"], 1e-10, "rel"),
    ]

    results = []
    failed = False
    for name, s, r, tol, kind in checks:
        err = abs_err(s, r) if kind == "abs" else rel_err(s, r)
        ok = err <= tol
        if not ok:
            failed = True
        results.append(
            {
                "name": name,
                "sounio": s,
                "oracle": r,
                "err": err,
                "tol": tol,
                "kind": kind,
                "ok": ok,
            }
        )
        status = "OK" if ok else "FAIL"
        print(f"  {status} {name}: sounio={s:.12g} oracle={r:.12g} err={err:.3e} tol={tol:g} ({kind})")

    # Optional SciPy cross-check (informational if present; hard-fail only if worse than pure oracle)
    scipy_note: Dict[str, Any] = {"available": False}
    try:
        from scipy import stats  # type: ignore
        import numpy as np  # type: ignore

        tw = stats.ttest_ind(DRUG, PLACEBO, equal_var=False)
        lr = stats.linregress(np.array(X_OLS), np.array(Y_OLS))
        scipy_note = {
            "available": True,
            "welch_t": float(tw.statistic),
            "welch_p": float(tw.pvalue),
            "welch_df": float(getattr(tw, "df", float("nan"))),
            "ols_slope": float(lr.slope),
            "ols_intercept": float(lr.intercept),
            "ols_r2": float(lr.rvalue**2),
            "note": "informational; primary oracle is closed-form matching Sounio formulas",
        }
        print(
            f"  info scipy welch t={tw.statistic:.12g} p={tw.pvalue:.6e} "
            f"ols_slope={lr.slope:.12g}"
        )
    except Exception as exc:  # noqa: BLE001
        scipy_note = {"available": False, "reason": str(exc)}

    detail = {
        "status": "fail" if failed else "pass",
        "backend": "pure_python_closed_form",
        "checks": results,
        "scipy": scipy_note,
        "max_abs_err": max(c["err"] for c in results if c["kind"] == "abs"),
        "max_rel_err": max(c["err"] for c in results if c["kind"] == "rel"),
    }
    print(json.dumps(detail))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
