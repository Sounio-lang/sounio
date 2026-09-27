#!/usr/bin/env python3
"""Is (h/2, h + h/2) systematically not load-bearing, across levels?

Where h = 2^(bits-1) is the level's doubling unit. Two independent sightings
came out of the load-bearing battery, in different contracts at different
levels:

    level 4   (4, 12)    = (2^2, 2^3 + 2^2)   functor_f_ord3_module_decomp
    level 8   (64, 192)  = (2^6, 2^7 + 2^6)   both cd_tower spectral contracts

and the level-8 diagonal probe showed the neighbouring shapes (h/4, h+h/4),
(3h/4, h+3h/4) all DO kill, so the form is specific rather than a diagonal
effect.

This tests the form at every level a single contract spans, against two
size-matched controls per level, on a contract that was 8/8 load-bearing
everywhere -- so a survivor here cannot be explained by that contract being
insensitive.

Pre-registered reading: the claim is only that a flip on this pair does not
change the verdict. Whether that is a blind spot or a correct invariance (the
flip yielding an isomorphic structure) is NOT settled by this probe.
"""
from __future__ import annotations

import concurrent.futures as cf
import json, os, subprocess, sys, time

WORK, OUT = "/tmp/work", "/tmp/hh_out"
PROBE = f"{WORK}/inject_probe2.py"
CONTRACT = os.environ.get("HH_CONTRACT", "cd_tower_nullity_histogram_law_contract.py")
FN = os.environ.get("HH_FN", "cds")
WORKERS, TIMEOUT = 16, 1800

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


def jobs_for(bits):
    h = 1 << (bits - 1)
    out = [(f"L{bits}_TARGET_{h//2}_{h+h//2}", h // 2, h + h // 2, bits)]
    # controls: same level, neighbouring shapes that are NOT (h/2, h+h/2)
    for a, b in ((h // 2, h + h // 4), (h // 4, h + h // 4)):
        if a and b and a != b:
            out.append((f"L{bits}_ctrl_{a}_{b}", a, b, bits))
    return out


def run(job):
    label, a, b, bits = job
    patch = "" if a is None else FLIP.format(n=FN, a=a, b=b, bits=bits)
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
    levels = [int(x) for x in os.environ.get("HH_LEVELS", "4,5,6,7,8").split(",")]
    jobs = [("baseline", None, None, None)]
    for b in levels:
        jobs += jobs_for(b)
    print(f"{CONTRACT} / {FN}, levels {levels}, {len(jobs)} runs", flush=True)

    res = {}
    with cf.ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for label, r, dt in ex.map(run, jobs):
            res[label] = {"verdict": r.get("verdict"), "error": r.get("error")}
            print(f"  {label:<26} {dt:6.1f}s  "
                  f"{(r.get('verdict') or 'ERR:' + str(r.get('error'))[:26])[:46]}",
                  flush=True)

    base = res["baseline"]["verdict"]
    print(f"\nbaseline {base}\n")
    print(f"{'level':>6}  {'(h/2, h+h/2)':>14}   controls")
    for b in levels:
        h = 1 << (b - 1)
        t = res.get(f"L{b}_TARGET_{h//2}_{h+h//2}", {})
        tv = "SURVIVES" if (not t.get("error") and t.get("verdict") == base) else "kills"
        cs = [k for k in res if k.startswith(f"L{b}_ctrl_")]
        cv = ["SURVIVES" if (not res[k].get("error") and res[k].get("verdict") == base)
              else "kills" for k in cs]
        print(f"{b:>6}  {tv:>14}   {cv}")
    json.dump(res, open(f"{OUT}/hhalf.json", "w"), indent=1)


if __name__ == "__main__":
    main()
