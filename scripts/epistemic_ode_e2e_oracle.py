#!/usr/bin/env python3
"""Closed-form oracle for 1-cmt IV GUM budget (same formulas as the driver)."""
from __future__ import annotations

import argparse
import json
import math
import sys
from typing import Optional


def parse_f(s: Optional[str]) -> Optional[float]:
    if s is None or s in ("", "null"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--c1")
    ap.add_argument("--u1")
    ap.add_argument("--frac-cl")
    ap.add_argument("--frac-v")
    args = ap.parse_args()

    got = {
        "c": parse_f(args.c1),
        "u": parse_f(args.u1),
        "frac_cl": parse_f(args.frac_cl),
        "frac_v": parse_f(args.frac_v),
    }
    if any(v is None for v in got.values()):
        print(json.dumps({"status": "skipped", "reason": "missing metrics", "got": got}))
        return 2

    dose, cl, v, u_cl, u_v, t = 100.0, 5.0, 50.0, 0.5, 2.5, 4.0
    k = cl / v
    c0 = dose / v
    c = c0 * math.exp(-k * t)
    d_cl = -(t / v) * c
    d_v = (c / v) * (k * t - 1.0)
    term_cl = d_cl * u_cl
    term_v = d_v * u_v
    u_c = math.sqrt(term_cl**2 + term_v**2)
    contrib_cl = abs(term_cl)
    contrib_v = abs(term_v)
    s = contrib_cl + contrib_v
    frac_cl = contrib_cl / s
    frac_v = contrib_v / s

    checks = [
        ("c", got["c"], c, 1e-5, "abs"),
        ("u_c", got["u"], u_c, 1e-5, "abs"),
        ("frac_cl", got["frac_cl"], frac_cl, 1e-5, "abs"),
        ("frac_v", got["frac_v"], frac_v, 1e-5, "abs"),
    ]
    failed = False
    results = []
    for name, sounio, ref, tol, kind in checks:
        err = abs(sounio - ref)
        ok = err <= tol
        if not ok:
            failed = True
        results.append(
            {"name": name, "sounio": sounio, "oracle": ref, "err": err, "tol": tol, "ok": ok}
        )
        print(f"  {'OK' if ok else 'FAIL'} {name}: sounio={sounio} oracle={ref} abs_err={err:.3e}")

    detail = {
        "status": "fail" if failed else "pass",
        "backend": "closed_form_1cmt_gum",
        "checks": results,
    }
    print(json.dumps(detail))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
