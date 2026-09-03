#!/usr/bin/env python3
"""Is the surviving perturbation isolated, or a structured invariance?

The load-bearing battery found exactly one level-8 sign flip that leaves the
ZD-fiber spectrum unchanged: e_64 . e_192. Its arithmetic is not arbitrary --
192 = 128 + 64, so the second operand is the Cayley-Dickson DOUBLE of the first:
e_192 = (0, e_64) while e_64 = (e_64, 0).

Hypothesis, stated before running: sign flips on the "doubling diagonal"
{(k, 2^(bits-1) + k)} are invisible to the annihilation-graph spectrum, while
generic flips at the same level are not.

Design: family A is the diagonal; family B is a size-matched control of generic
pairs at the same level drawn the same way. A clean result needs A to survive
and B to kill -- if B also survives, the level simply is not sensitive and there
is nothing structured here.

NOTE ON INTERPRETATION, fixed in advance: invariance here is not automatically a
defect. If the flip yields an ISOMORPHIC annihilation graph then an unchanged
spectrum is CORRECT behaviour for a complete invariant, not blindness. This
probe establishes whether the invariance is structured; it does not by itself
say which of the two it is.
"""
from __future__ import annotations

import concurrent.futures as cf
import json, os, subprocess, sys, time

WORK, OUT = "/tmp/work", "/tmp/diag_out"
PROBE = f"{WORK}/inject_probe2.py"
CONTRACT = "cd_tower_zd_fiber_signed_localization_contract.py"
FN = "cd_sigma"
BITS = 8
HALF = 1 << (BITS - 1)          # 128
WORKERS = 12
TIMEOUT = 1800

_PRE = ("_orig_{n} = {n}\n"
        "def _bits_of_{n}(r, k):\n"
        "    if 'bits' in k:\n        return k['bits']\n"
        "    if r:\n        return r[0]\n"
        "    d = _orig_{n}.__defaults__\n"
        "    return d[-1] if d else None\n")
FLIP = _PRE + (
    "def {n}(a, b, *r, **k):\n"
    "    s = _orig_{n}(a, b, *r, **k)\n"
    "    if _bits_of_{n}(r, k) == {bits} and "
    "((a, b) == ({a}, {b}) or (a, b) == ({b}, {a})):\n"
    "        return -s\n    return s\n")

DIAG = [(k, HALF + k) for k in (1, 2, 3, 4, 8, 16, 32, 64, 96, 127)]
# control: same level, same index magnitudes, but NOT b == HALF + a
CTRL = [(1, 130), (2, 133), (3, 136), (4, 140), (8, 145), (16, 160),
        (32, 176), (64, 200), (96, 208), (127, 250)]


def run(job):
    label, a, b = job
    patch = "" if a is None else FLIP.format(n=FN, a=a, b=b, bits=BITS)
    out = f"{OUT}/{label}.json"
    env = dict(os.environ, PROBE_OUT=out, PROBE_PATCH=patch)
    t0 = time.time()
    try:
        subprocess.run([sys.executable, PROBE, f"{WORK}/{CONTRACT}", FN],
                       env=env, cwd=WORK, capture_output=True, timeout=TIMEOUT)
    except subprocess.TimeoutExpired:
        return label, {"verdict": None, "error": "TIMEOUT"}, time.time() - t0
    try:
        r = json.load(open(out))
    except Exception as exc:                                    # noqa: BLE001
        r = {"verdict": None, "error": f"nojson {exc}"}
    return label, r, time.time() - t0


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    jobs = [("baseline", None, None)]
    jobs += [(f"diag_{a}_{b}", a, b) for a, b in DIAG]
    jobs += [(f"ctrl_{a}_{b}", a, b) for a, b in CTRL]
    print(f"{len(jobs)} runs on {CONTRACT}, level {BITS}, {WORKERS} workers",
          flush=True)

    res = {}
    with cf.ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for label, r, dt in ex.map(run, jobs):
            res[label] = {"verdict": r.get("verdict"), "error": r.get("error"),
                          "secs": round(dt, 1)}
            print(f"  {label:<16} {dt:6.1f}s  "
                  f"{(r.get('verdict') or 'ERR ' + str(r.get('error'))[:30])[:52]}",
                  flush=True)

    base = res["baseline"]["verdict"]
    def surv(lbl):
        r = res[lbl]
        return (not r["error"]) and r["verdict"] == base
    d = [l for l in res if l.startswith("diag_")]
    c = [l for l in res if l.startswith("ctrl_")]
    ns, cs = sum(surv(x) for x in d), sum(surv(x) for x in c)
    print(f"\nbaseline {base}")
    print(f"  doubling diagonal : {ns}/{len(d)} survive")
    print(f"  generic control   : {cs}/{len(c)} survive")
    if ns == len(d) and cs == 0:
        print("  => STRUCTURED INVARIANCE on the doubling diagonal")
    elif ns == cs:
        print("  => no structure: the two families behave the same")
    else:
        print("  => partial / mixed; report the numbers, claim nothing more")
    json.dump(res, open(f"{OUT}/diag.json", "w"), indent=1)


if __name__ == "__main__":
    main()
