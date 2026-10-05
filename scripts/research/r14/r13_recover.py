#!/usr/bin/env python3
"""Recover the two contracts R13's battery lost to a timeout.

R13 excluded three contracts for "no baseline verdict". The call trace shows two
of them DO emit verdicts -- `cd_tower_zd_fiber_spectral_classifier`
(ZD_FIBER_SPECTRUM_COMPLETE_INVARIANT_N_LE_8, 86 s) and
`..._spectral_forall_n_progress` (19 s). They did not lack a token; they hit the
600 s cap under 96-way contention.

Both are cd_sigma, the scarce derivation family: R13 ran with 2 of them and this
takes it to 4. Re-runs the identical 36-mutant battery for those two alone, with
low concurrency so the timeout cannot recur.
"""
from __future__ import annotations

import concurrent.futures as cf
import json, os, subprocess, sys, time

sys.path.insert(0, "/tmp")
from battery import mutants, WRAP                              # noqa: E402

WORK, OUT = "/tmp/work", "/tmp/rec_out"
PROBE = f"{WORK}/inject_probe2.py"
TIMEOUT = 2400                      # 4x the longest observed solo runtime
WORKERS = 6

TARGETS = [("cd_tower_zd_fiber_spectral_classifier_contract.py", "cd_sigma"),
           ("cd_tower_zd_fiber_spectral_forall_n_progress_contract.py", "cd_sigma")]


def one(job):
    contract, fn, m = job
    patch = WRAP[m["kind"]].format(
        n=fn, **{k: v for k, v in m.items() if k not in ("kind", "id")})
    out = f"{OUT}/{abs(hash((contract, m['id']))) % 10**12}.json"
    env = dict(os.environ, PROBE_OUT=out, PROBE_PATCH=patch)
    t0 = time.time()
    try:
        subprocess.run([sys.executable, PROBE, f"{WORK}/{contract}", fn],
                       env=env, cwd=WORK, capture_output=True, timeout=TIMEOUT)
    except subprocess.TimeoutExpired:
        return contract, m["id"], {"verdict": None, "error": "TIMEOUT"}, time.time() - t0
    try:
        r = json.load(open(out))
    except Exception as exc:                                    # noqa: BLE001
        r = {"verdict": None, "error": f"nojson {exc}"}
    return contract, m["id"], r, time.time() - t0


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    muts = mutants()
    jobs = [(c, fn, m) for c, fn in TARGETS for m in muts]
    print(f"{len(TARGETS)} contracts x {len(muts)} mutants = {len(jobs)} runs, "
          f"{WORKERS} workers, timeout {TIMEOUT}s", flush=True)
    res = {}
    t0 = time.time()
    with cf.ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for i, (c, mid, r, dt) in enumerate(ex.map(one, jobs), 1):
            res.setdefault(c, {})[mid] = {"verdict": r.get("verdict"),
                                          "error": r.get("error")}
            if i % 20 == 0:
                print(f"  {i}/{len(jobs)} ({time.time()-t0:.0f}s)", flush=True)
    json.dump({"mutants": [m["id"] for m in muts], "results": res},
              open(f"{OUT}/recovered.json", "w"), indent=1)
    for c in res:
        b = res[c].get("baseline", {}).get("verdict")
        nk = sum(1 for m, r in res[c].items()
                 if m not in ("baseline", "null_wrap")
                 and (r["verdict"] != b or r["error"]))
        ctrl = res[c].get("null_wrap", {})
        print(f"\n{c}\n  baseline {b}\n  kills {nk}/{len(muts)-2}"
              f"\n  null-wrap control {'INERT' if ctrl.get('verdict') == b else 'REACTS'}",
              flush=True)
    print(f"\nwall {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
