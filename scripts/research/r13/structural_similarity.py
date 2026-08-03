#!/usr/bin/env python3
"""R6 structural similarity for the battery's contracts, in parallel.

Reuses R6's own fingerprints()/_Canon/threshold by import -- copying them would
be the exact failure R6 measures, inside a rung that builds on R6.
"""
from __future__ import annotations

import concurrent.futures as cf
import difflib, importlib.util, itertools, json, os

SD = "/tmp/claude-1000/-workspace-sounio/1d762349-7a51-4c09-8c6a-44223c57352d/scratchpad"
REPO = "/workspace/sounio"
os.chdir(REPO)

spec = importlib.util.spec_from_file_location(
    "r6", "scripts/research/self_falsifying_compilation_line_r6_contract.py")
r6 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r6)

man = json.load(open(f"{SD}/manifest.json"))
fps = {c["file"]: r6.fingerprints(f"scripts/research/{c['file']}") for c in man}
pairs = list(itertools.combinations(sorted(fps), 2))


def sim(ab):
    a, b = ab
    w = 0.0
    for ad in fps[a].values():
        for bd in fps[b].values():
            r = difflib.SequenceMatcher(None, ad, bd).ratio()
            if r > w:
                w = r
    return f"{a}|{b}", round(w, 4)


with cf.ProcessPoolExecutor(max_workers=10) as ex:
    out = dict(ex.map(sim, pairs, chunksize=4))

json.dump(out, open(f"{SD}/struct_sim.json", "w"))
v = sorted(out.values())
print(f"{len(out)} pairs; R6 structural similarity")
print(f"  min {v[0]:.3f}  median {v[len(v)//2]:.3f}  max {v[-1]:.3f}")
print(f"  >= {r6.DUP_THRESHOLD} (R6: SHARED)      {sum(x >= r6.DUP_THRESHOLD for x in v)}")
print(f"  <  {r6.DUP_THRESHOLD} (R6: INDEPENDENT) {sum(x < r6.DUP_THRESHOLD for x in v)}")
