#!/usr/bin/env python3
"""Escalation battery — separate 'unchecked level' from 'genuine invariance'.

The load-bearing battery flips ONE base pair at a time. A level where no single
flip moves the verdict has two possible causes, and they are opposite in
meaning:

  (a) the contract queries that level but its conclusion does not depend on the
      level's structure at all -- vacuity, a defect;
  (b) the quantity the contract checks is INVARIANT under a single sign flip --
      a mathematical fact about the claim, not a defect.

Escalating distinguishes them. For each (contract, level) that came back zero,
run progressively larger perturbations at that level:

  elem_e   flip every product involving basis element e   (a whole row/column)
  half     flip every product where both indices >= 2^(bits-1)
           (the units introduced AT that level -- the level's own content)
  all      flip every non-trivial product at that level

If a bigger perturbation moves the verdict, the level IS checked and the
single-flip zero was invariance (b). If nothing moves it, the level's structure
is not load-bearing for the stated conclusion (a).

Reads lb_analysis.json's escalate list. Child processes, JSON out, os._exit.
"""
from __future__ import annotations

import concurrent.futures as cf
import json, os, subprocess, sys, time

WORK, OUT = "/tmp/work", "/tmp/esc_out"
PROBE = f"{WORK}/inject_probe2.py"
WORKERS = int(os.environ.get("WORKERS", "24"))
TIMEOUT = 1800

_PRE = ("_orig_{n} = {n}\n"
        "def _bits_of_{n}(r, k):\n"
        "    if 'bits' in k:\n        return k['bits']\n"
        "    if r:\n        return r[0]\n"
        "    d = _orig_{n}.__defaults__\n"
        "    return d[-1] if d else None\n")

SHAPES = {
    "null": "",
    "elem": _PRE + (
        "def {n}(a, b, *r, **k):\n"
        "    s = _orig_{n}(a, b, *r, **k)\n"
        "    if _bits_of_{n}(r, k) == {bits} and ({e} in (a, b)):\n"
        "        return -s\n    return s\n"),
    "half": _PRE + (
        "_H_{n} = 1 << ({bits} - 1)\n"
        "def {n}(a, b, *r, **k):\n"
        "    s = _orig_{n}(a, b, *r, **k)\n"
        "    if _bits_of_{n}(r, k) == {bits} and a >= _H_{n} and b >= _H_{n} and a != b:\n"
        "        return -s\n    return s\n"),
    "all": _PRE + (
        "def {n}(a, b, *r, **k):\n"
        "    s = _orig_{n}(a, b, *r, **k)\n"
        "    if _bits_of_{n}(r, k) == {bits} and a and b and a != b:\n"
        "        return -s\n    return s\n"),
}


def run(contract, fn, patch, tag):
    out = f"{OUT}/{tag}.json"
    env = dict(os.environ, PROBE_OUT=out, PROBE_PATCH=patch)
    try:
        subprocess.run([sys.executable, PROBE, f"{WORK}/{contract}", fn],
                       env=env, cwd=WORK, capture_output=True, timeout=TIMEOUT)
    except subprocess.TimeoutExpired:
        return {"verdict": None, "error": "TIMEOUT"}
    try:
        return json.load(open(out))
    except Exception as exc:                                    # noqa: BLE001
        return {"verdict": None, "error": f"nojson {exc}"}


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    A = json.load(open("/tmp/lb_out/lb_analysis.json"))
    plan = json.load(open("/tmp/lb_out/loadbearing.json"))["plan"]
    jobs = []
    for c, lvl, _n, _q in A["escalate"]:
        fn = plan[c]["fn"]
        bits = int(lvl)
        half = 1 << (bits - 1)
        jobs.append((c, fn, lvl, "null", ""))
        for e in (half, half + 1, 1):
            jobs.append((c, fn, lvl, f"elem{e}",
                         SHAPES["elem"].format(n=fn, bits=bits, e=e)))
        jobs.append((c, fn, lvl, "half", SHAPES["half"].format(n=fn, bits=bits)))
        jobs.append((c, fn, lvl, "all", SHAPES["all"].format(n=fn, bits=bits)))
    # one baseline per contract, deduplicated
    for c in sorted({j[0] for j in jobs}):
        jobs.append((c, plan[c]["fn"], "-", "baseline", ""))

    print(f"{len(A['escalate'])} (contract,level) zeros -> {len(jobs)} runs, "
          f"{WORKERS} workers", flush=True)

    def one(j):
        c, fn, lvl, shape, patch = j
        r = run(c, fn, patch, f"{abs(hash((c, lvl, shape))) % 10**12}")
        return c, lvl, shape, r

    res = {}
    t0 = time.time()
    with cf.ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for i, (c, lvl, shape, r) in enumerate(ex.map(one, jobs), 1):
            res.setdefault(c, {}).setdefault(lvl, {})[shape] = {
                "verdict": r.get("verdict"), "error": r.get("error")}
            if i % 20 == 0:
                print(f"  {i}/{len(jobs)} ({time.time()-t0:.0f}s)", flush=True)

    json.dump(res, open(f"{OUT}/escalation.json", "w"), indent=1)
    print(f"\nwall {time.time()-t0:.0f}s -> {OUT}/escalation.json", flush=True)


if __name__ == "__main__":
    main()
