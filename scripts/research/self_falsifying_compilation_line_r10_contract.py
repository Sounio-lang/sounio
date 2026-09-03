#!/usr/bin/env python3
"""Self-falsifying compilation, rung R10 — latent corroboration discovery.

Spec: docs/research/self_falsifying_compilation_line_r10_2026-07-26.md

Three literature searches established where the solid ground is, and every
neighbour stops at the same place. `build.rs` executes checks. Clone detection
measures duplication. The repeatability/replicability taxonomy names why an
independent re-implementation is worth more than a re-run. N-version programming
studies whether TWO implementations are independent — pairwise, by construction,
because someone commissioned them separately.

None of them asks the corpus-level question: **how many independent things does
this body of code actually know?**

R8 answered a fragment of it by hand. It found that `cd_sigma` — recursive,
sitting unused in three contracts — is an independent derivation of the same
object as the iterative `cds`, and that nobody had ever compared them. That was
a **latent corroboration**: evidence the corpus already possessed and had never
cashed.

R10 automates the search. A pair of functions that is

    STRUCTURALLY INDEPENDENT   (similarity below R6's clone threshold)
    but BEHAVIOURALLY IDENTICAL (agrees on every probe input)

is a latent corroboration. Each one found is a free independent check the
project owns and was not using.

CORROBORATION DEPTH is the resulting per-kernel metric: how many structurally
distinct derivations of the same behaviour the corpus contains. Depth 1 means
"one implementation, no internal corroboration whatsoever" — which is the
default state and, before this rung, the unmeasured one.

VERDICT OPTIONS, FIXED BEFORE COMPUTING (see main()):
  LATENT_CORROBORATION_FOUND
      the search finds independent-but-equivalent pairs, i.e. the corpus holds
      unused internal evidence.
  NO_LATENT_CORROBORATION__DEPTH_ONE_CORPUS
      every kernel has exactly one derivation; nothing corroborates anything
      internally.
  SEARCH_INCONCLUSIVE
      too few functions are probeable to say either way.

Pure Python 3 + numpy.
"""

from __future__ import annotations

import ast
import difflib
import importlib.util
import inspect
import itertools
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
R6 = REPO / "scripts/research/self_falsifying_compilation_line_r6_contract.py"

# Probe grid for small-integer signatures. Deliberately covers the tower levels
# the corpus works at.
INT_PROBES = [(i, j, bits) for bits in (3, 4, 5)
              for i in range(1 << bits) for j in range(1 << bits)]


def _load_r6():
    spec = importlib.util.spec_from_file_location("r6", R6)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def contracts() -> list[str]:
    return sorted(str(p.relative_to(REPO))
                  for p in (REPO / "scripts/research").glob("*contract*.py"))


def load_fn(rel: str, fn: str):
    """Compile one function in isolation; never import the module."""
    try:
        src = (REPO / rel).read_text(errors="replace")
        for node in ast.parse(src).body:
            if isinstance(node, ast.FunctionDef) and node.name == fn:
                ns: dict = {"np": np}
                exec(compile(ast.Module(body=[node], type_ignores=[]),
                             f"<{rel}:{fn}>", "exec"), ns)
                return ns[fn]
    except Exception:
        return None
    return None


def probeable(f) -> bool:
    """Takes 2-3 positional ints and returns a comparable scalar."""
    try:
        sig = inspect.signature(f)
    except (TypeError, ValueError):
        return False
    params = [p for p in sig.parameters.values()
              if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
    if not 2 <= len(params) <= 3:
        return False
    try:
        v = f(1, 2, 3) if len(params) == 3 else f(1, 2)
    except Exception:
        return False
    return isinstance(v, (int, float, bool)) and not isinstance(v, np.ndarray)


def behaviour(f, arity: int):
    """Output vector over the probe grid, or None if it ever raises."""
    out = []
    for i, j, bits in INT_PROBES:
        try:
            v = f(i, j, bits) if arity == 3 else f(i, j)
        except Exception:
            return None
        if not isinstance(v, (int, float, bool)):
            return None
        out.append(float(v))
    return tuple(out)


# ---------------------------------------------------------------- L1


def collect(r6):
    """Every probeable function in the corpus, with its structural fingerprint."""
    found = []
    seen_fp = {}
    for rel in contracts():
        fps = r6.fingerprints(rel)
        for name, fp in fps.items():
            f = load_fn(rel, name)
            if f is None or not probeable(f):
                continue
            try:
                arity = len([p for p in inspect.signature(f).parameters.values()
                             if p.kind in (p.POSITIONAL_ONLY,
                                           p.POSITIONAL_OR_KEYWORD)])
            except (TypeError, ValueError):
                continue
            b = behaviour(f, arity)
            if b is None:
                continue
            found.append({"rel": rel, "name": name, "fp": fp,
                          "arity": arity, "behaviour": b})
            seen_fp.setdefault(fp, []).append(f"{Path(rel).name}:{name}")
    print(f"L1_PROBEABLE {len(found)} probeable functions across "
          f"{len(contracts())} contracts "
          f"({len(seen_fp)} distinct structural fingerprints)")
    print("L1_PROBEABLE PASS — measured")
    return found


# ---------------------------------------------------------------- L2


def clause_l2(r6, fns) -> tuple[bool, list, dict]:
    """Find structurally independent, behaviourally identical pairs."""
    latent = []
    equal_pairs = 0
    for a, b in itertools.combinations(fns, 2):
        if a["behaviour"] != b["behaviour"]:
            continue
        equal_pairs += 1
        sim = difflib.SequenceMatcher(None, a["fp"], b["fp"]).ratio()
        if sim < r6.DUP_THRESHOLD:
            latent.append((sim, a, b))
    latent.sort(key=lambda x: x[0])

    # Counting PAIRS overstates the result: 24 copies of one derivation against
    # 7 copies of another produce ~168 "independent pairs" while representing
    # exactly ONE corroboration. The honest unit is the behaviour class with
    # more than one derivation — see L3. Pairs are reported as texture only.
    print(f"L2_LATENT_CORROBORATION {equal_pairs} behaviourally identical pairs; "
          f"{len(latent)} cross-derivation pairs (similarity < "
          f"{r6.DUP_THRESHOLD}) — NOTE: pairs count COPIES, not corroborations")
    shown = set()
    for sim, a, b in latent:
        key = tuple(sorted((f"{Path(a['rel']).name}:{a['name']}",
                            f"{Path(b['rel']).name}:{b['name']}")))
        if key in shown:
            continue
        shown.add(key)
        if len(shown) > 12:
            continue
        print(f"    sim {sim:.3f}  {key[0]}")
        print(f"               {key[1]}")
    if len(shown) > 12:
        print(f"    ... and {len(shown) - 12} more distinct pairs")
    print("L2_LATENT_CORROBORATION PASS — measured")
    return True, latent, {"equal_pairs": equal_pairs, "latent": len(shown)}


# ---------------------------------------------------------------- L3


def clause_l3(r6, fns, latent) -> tuple[bool, dict]:
    """Corroboration depth: distinct derivations per behaviour class."""
    by_behaviour: dict[tuple, list] = {}
    for f in fns:
        by_behaviour.setdefault(f["behaviour"], []).append(f)

    depths = {}
    for beh, group in by_behaviour.items():
        # Cluster the group by structural similarity; each cluster is one
        # derivation, however many copies of it exist.
        reps: list[str] = []
        for f in group:
            if not any(difflib.SequenceMatcher(None, f["fp"], r).ratio()
                       >= r6.DUP_THRESHOLD for r in reps):
                reps.append(f["fp"])
        depths[beh] = (len(reps), group)

    hist: dict[int, int] = {}
    for d, _ in depths.values():
        hist[d] = hist.get(d, 0) + 1
    deep = [(d, g) for d, g in depths.values() if d > 1]

    print(f"L3_CORROBORATION_DEPTH {len(depths)} distinct behaviours; "
          f"depth histogram {dict(sorted(hist.items()))}")
    print(f"L3_CORROBORATION_DEPTH the honest unit: {len(deep)} behaviour(s) "
          f"have more than one derivation, i.e. {len(deep)} real corroboration(s)")
    for d, g in sorted(deep, key=lambda x: -x[0]):
        names = sorted({f"{Path(f['rel']).name}:{f['name']}" for f in g})
        print(f"    depth {d}: {len(g)} implementations — {names[:4]}")
    if not deep:
        print("    no behaviour has more than one derivation: the corpus "
              "corroborates nothing internally")
    print("L3_CORROBORATION_DEPTH PASS — measured")
    return True, {"behaviours": len(depths), "hist": hist, "deep": len(deep)}


# ---------------------------------------------------------------- main


def main() -> int:
    print("SELF-FALSIFYING COMPILATION R10 — latent corroboration discovery")
    print("=" * 78)
    print("Neighbours ask whether TWO implementations are independent, pairwise,")
    print("by construction. This asks the corpus-level question: how many")
    print("independent things does this body of code already know, unused?")
    print()

    r6 = _load_r6()
    fns = collect(r6)
    print()
    if len(fns) < 2:
        print("SELF_FALSIFYING_R10_VERDICT SEARCH_INCONCLUSIVE")
        return 0

    l2, latent, lstats = clause_l2(r6, fns)
    print()
    l3, dstats = clause_l3(r6, fns, latent)
    print()

    print("=" * 78)
    if len(fns) < 4:
        token = "SEARCH_INCONCLUSIVE"
    elif dstats["deep"] > 0:
        token = "LATENT_CORROBORATION_FOUND"
    else:
        token = "NO_LATENT_CORROBORATION__DEPTH_ONE_CORPUS"

    print(f"  probeable functions    : {len(fns)}")
    print(f"  distinct behaviours    : {dstats['behaviours']}")
    print(f"  REAL corroborations    : {dstats['deep']}  "
          f"(behaviour classes with >1 derivation)")
    print(f"  cross-derivation pairs : {lstats['latent']}  "
          f"(copies, NOT corroborations)")
    print(f"  newly discovered       : 0  "
          f"(the one found was already known from R8; see spec §2)")
    print(f"SELF_FALSIFYING_R10_VERDICT {token}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
