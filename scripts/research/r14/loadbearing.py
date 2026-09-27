#!/usr/bin/env python3
"""Load-bearing battery: of what a contract COMPUTES, how much does it CHECK?

R13 perturbed a fixed global list of base pairs. This perturbs, per contract,
the pairs that contract actually queries -- taken from the call trace -- and
stratifies the sample by Cayley-Dickson level.

The measure per (contract, level):

    load-bearing fraction = (# sampled perturbations at that level that change
                             the verdict) / (# sampled at that level)

A level with fraction 0 is queried but not checked: the contract computes that
structure and its stated conclusion does not depend on any of it. That is a
mechanical vacuity signal, and it is the positive use of the instrument R13
built (R13 detects shared fate; this detects absent fate).

PRE-REGISTERED, before results:
  * Sample is deterministic (sorted, evenly spaced) -- no seed to tune.
  * A level counts as NOT LOAD-BEARING only if ALL sampled perturbations at that
    level leave the verdict identical AND the null-wrap control is inert for
    that contract. Sampling means this is evidence of absence, not proof: with k
    samples from n queried pairs, a level could still be load-bearing on pairs
    not drawn. Reported as "no load-bearing pair found in k of n", never as
    "the level is vacuous".
  * Contracts with no baseline verdict are excluded and counted.
"""
from __future__ import annotations

import concurrent.futures as cf
import json, os, subprocess, sys, time

WORK, OUT = "/tmp/work", "/tmp/lb_out"
PROBE = f"{WORK}/inject_probe2.py"
WORKERS = int(os.environ.get("WORKERS", "96"))
TIMEOUT = 600
PER_LEVEL = int(os.environ.get("PER_LEVEL", "8"))

_PRELUDE = (
    "_orig_{n} = {n}\n"
    "def _bits_of_{n}(r, k):\n"
    "    if 'bits' in k:\n        return k['bits']\n"
    "    if r:\n        return r[0]\n"
    "    d = _orig_{n}.__defaults__\n"
    "    return d[-1] if d else None\n")

FLIP = _PRELUDE + (
    "def {n}(a, b, *r, **k):\n"
    "    s = _orig_{n}(a, b, *r, **k)\n"
    "    if _bits_of_{n}(r, k) == {bits} and "
    "((a, b) == ({a}, {b}) or (a, b) == ({b}, {a})):\n"
    "        return -s\n"
    "    return s\n")


def sample(pairs, per_level):
    """Deterministic, evenly spaced, stratified by level. No RNG to tune."""
    by = {}
    for a, b, bits in pairs:
        if bits is None or a == b or a == 0 or b == 0:
            continue                       # trivial products carry no sign info
        by.setdefault(bits, []).append((a, b))
    out = []
    for bits in sorted(by):
        v = sorted(set(by[bits]))
        step = max(1, len(v) // per_level)
        out += [(a, b, bits) for a, b in v[::step][:per_level]]
    return out, {str(k): len(set(v)) for k, v in by.items()}


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
    trace = json.load(open("/tmp/trace_out/trace.json"))
    jobs, plan = [], {}
    for t in trace:
        if not t.get("verdict") or not t.get("pairs"):
            continue
        s, counts = sample(t["pairs"], PER_LEVEL)
        plan[t["contract"]] = {"fn": t["fn"], "queried": counts,
                               "sampled": [list(x) for x in s]}
        jobs.append((t["contract"], t["fn"], "baseline", None))
        jobs.append((t["contract"], t["fn"], "null_wrap", (-1, -2, -3)))
        for a, b, bits in s:
            jobs.append((t["contract"], t["fn"], f"L{bits}_{a}_{b}", (a, b, bits)))

    print(f"{len(plan)} contracts, {len(jobs)} runs, {WORKERS} workers", flush=True)
    for c, p in sorted(plan.items()):
        print(f"  {c[:52]:<54} queried {p['queried']}", flush=True)
    print(flush=True)

    def one(job):
        c, fn, mid, tgt = job
        patch = "" if tgt is None else FLIP.format(n=fn, a=tgt[0], b=tgt[1], bits=tgt[2])
        r = run(c, fn, patch, f"{abs(hash((c, mid))) % 10**12}")
        return c, mid, r

    res = {}
    t0 = time.time()
    with cf.ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = [ex.submit(one, j) for j in jobs]
        for i, f in enumerate(cf.as_completed(futs), 1):
            c, mid, r = f.result()
            res.setdefault(c, {})[mid] = {"verdict": r.get("verdict"),
                                          "error": r.get("error")}
            if i % 100 == 0:
                print(f"  {i}/{len(jobs)} ({time.time()-t0:.0f}s)", flush=True)

    json.dump({"plan": plan, "results": res},
              open(f"{OUT}/loadbearing.json", "w"), indent=1)
    print(f"\nwall {time.time()-t0:.0f}s -> {OUT}/loadbearing.json", flush=True)


if __name__ == "__main__":
    main()
