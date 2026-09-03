#!/usr/bin/env python3
"""Route-independence battery.

For each contract that defines a CD-sign function (`cds` or `cd_sigma` -- two
structurally INDEPENDENT derivations, R6 similarity 0.507), apply a graduated
series of TARGETED VALUE PERTURBATIONS and record whether the contract's verdict
token changes.

The perturbation is expressed against the SHARED MATHEMATICAL OBJECT, not
against code: "flip the Cayley-Dickson sign on base pair (a,b) at level L".
Both families take (a, b, bits) and return +/-1, so the identical conceptual
perturbation crosses both derivations. That is what makes this non-tautological:
mutating the *source* of `cds` could only ever hit `cds` users, which would
re-derive R6's structural partition by construction.

Kill vector per contract -> co-sensitivity. The question: does the partition by
co-sensitivity differ from R6's partition by structural distance?

PRE-REGISTERED DISCRIMINATION FLOOR (fixed before running):
  A mutant is INFORMATIVE if it kills > 10% and < 90% of usable contracts.
  If fewer than 8 mutants are informative the battery is DEGENERATE and the
  result is "no discriminating perturbation found", NOT a partition.

Every probe runs in its own child process (R11 SS3: never run foreign code in
the process that has to report).
"""
from __future__ import annotations

import concurrent.futures as cf
import json, os, subprocess, sys, time

WORK = "/tmp/work"
OUT = "/tmp/battery_out"
PROBE = f"{WORK}/inject_probe2.py"
TIMEOUT = 600
WORKERS = int(os.environ.get("WORKERS", "96"))


def mutants() -> list[dict]:
    """Graduated: single base pair, then whole levels, then wider slices."""
    m = [{"id": "baseline", "kind": "identity"},
         # NEGATIVE CONTROL: identical wrapper machinery, condition never true.
         # Any contract that "dies" here is reacting to the instrument, not to a
         # perturbation. Added after the first battery reported two such deaths
         # (a wrapper that redeclared the target's `bits` default). A control
         # that costs one column per contract would have caught it immediately.
         {"id": "null_wrap", "kind": "flip_pair", "a": -1, "b": -2, "bits": -3}]
    # 1. single base-pair flips, level 4 (sedenions) and level 3 (octonions)
    for bits in (3, 4):
        for a, b in ((1, 2), (1, 3), (2, 4), (3, 5), (1, 8), (2, 8),
                     (5, 10), (7, 9), (6, 11), (4, 12)):
            m.append({"kind": "flip_pair", "a": a, "b": b, "bits": bits,
                      "id": f"pair_{a}_{b}_L{bits}"})
    # 2. flip every product involving one basis element
    for bits in (3, 4):
        for e in (1, 2, 4, 7, 8, 15):
            m.append({"kind": "flip_elem", "e": e, "bits": bits,
                      "id": f"elem_{e}_L{bits}"})
    # 3. flip an entire level
    for bits in (3, 4, 5):
        m.append({"kind": "flip_level", "bits": bits, "id": f"level_{bits}"})
    # 4. catastrophic -- sanity anchor, must kill nearly everything
    m.append({"kind": "constant", "id": "catastrophic"})
    return m


# NEVER redeclare the target's signature. The first version of this wrapper
# wrote `def cds(a, b, bits={bits}, ...)`, which OVERRODE the original default.
# Contracts declaring `def cds(a, b, bits=4)` and calling `cds(k ^ j, j)` were
# then silently switched from sedenion to octonion arithmetic by an L3 mutant --
# and duly "died", which read as a contract responding to a perturbation of a
# part of the table that does not exist at level 3. That was the instrument, not
# the corpus. The wrapper now forwards *args untouched and RECOVERS the
# effective `bits` from the original's own defaults.
_PRELUDE = (
    "_orig_{n} = {n}\n"
    "def _bits_of_{n}(r, k):\n"
    "    if 'bits' in k:\n"
    "        return k['bits']\n"
    "    if r:\n"
    "        return r[0]\n"
    "    d = _orig_{n}.__defaults__\n"
    "    return d[-1] if d else None\n")

WRAP = {
    "identity": "",
    "constant": "_orig_{n} = {n}\ndef {n}(*a, **k):\n    return 1\n",
    "flip_pair": _PRELUDE + (
        "def {n}(a, b, *r, **k):\n"
        "    s = _orig_{n}(a, b, *r, **k)\n"
        "    if _bits_of_{n}(r, k) == {bits} and "
        "((a, b) == ({a}, {b}) or (a, b) == ({b}, {a})):\n"
        "        return -s\n"
        "    return s\n"),
    "flip_elem": _PRELUDE + (
        "def {n}(a, b, *r, **k):\n"
        "    s = _orig_{n}(a, b, *r, **k)\n"
        "    if _bits_of_{n}(r, k) == {bits} and ({e} in (a, b)):\n"
        "        return -s\n"
        "    return s\n"),
    "flip_level": _PRELUDE + (
        "def {n}(a, b, *r, **k):\n"
        "    s = _orig_{n}(a, b, *r, **k)\n"
        "    if _bits_of_{n}(r, k) == {bits} and a and b and a != b:\n"
        "        return -s\n"
        "    return s\n"),
}


def run(contract: str, fn: str, mut: dict, tag: str):
    patch = WRAP[mut["kind"]].format(n=fn, **{k: v for k, v in mut.items()
                                              if k not in ("kind", "id")})
    out = f"{OUT}/{tag}.json"
    env = dict(os.environ, PROBE_OUT=out, PROBE_PATCH=patch)
    t0 = time.time()
    try:
        subprocess.run([sys.executable, PROBE, f"{WORK}/{contract}", fn],
                       env=env, timeout=TIMEOUT, capture_output=True, cwd=WORK)
    except subprocess.TimeoutExpired:
        return {"verdict": None, "error": "TIMEOUT"}, time.time() - t0
    try:
        return json.load(open(out)), time.time() - t0
    except Exception as exc:                                    # noqa: BLE001
        return {"verdict": None, "error": f"nojson {exc}"}, time.time() - t0


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    man = json.load(open("/tmp/manifest.json"))
    muts = mutants()
    jobs = [(c["file"], c["fn"], m) for c in man for m in muts]
    print(f"{len(man)} contracts x {len(muts)} mutants = {len(jobs)} runs, "
          f"{WORKERS} workers", flush=True)

    res: dict = {}
    t0 = time.time()
    with cf.ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {}
        for i, (c, fn, m) in enumerate(jobs):
            futs[ex.submit(run, c, fn, m, f"{i}")] = (c, fn, m["id"])
        done = 0
        for fut in cf.as_completed(futs):
            c, fn, mid = futs[fut]
            r, dt = fut.result()
            res.setdefault(c, {})[mid] = {"verdict": r.get("verdict"),
                                          "error": r.get("error"),
                                          "secs": round(dt, 1)}
            done += 1
            if done % 50 == 0:
                print(f"  {done}/{len(jobs)}  ({time.time()-t0:.0f}s)", flush=True)

    json.dump({"mutants": [m["id"] for m in muts], "results": res},
              open(f"{OUT}/battery.json", "w"), indent=1)
    print(f"\nwall {time.time()-t0:.0f}s -> {OUT}/battery.json", flush=True)


if __name__ == "__main__":
    main()
