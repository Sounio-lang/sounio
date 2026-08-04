#!/usr/bin/env python3
"""Self-falsifying compilation, rung R11 — widening the corroboration probe.

Spec: docs/research/self_falsifying_compilation_line_r11_2026-07-26.md

R10 built a discovery procedure for LATENT CORROBORATIONS — pairs of functions
that are structurally independent yet behaviourally identical, i.e. evidence a
corpus already owns and has never cashed. It worked: from source alone it
rediscovered the `cds`/`cd_sigma` pair a human had found by reading.

It also found nothing new, and R10's spec said plainly why: the probe accepted
only functions taking 2-3 integers and returning a scalar, and every function it
accepted computed the same thing. The negative covered one signature family.

R11 widens the probe to the families R10 listed as untouched:

    (int, int, int) -> scalar          the original
    (int, int) / (int,) -> scalar
    (array, array) -> array            omul, o, cd_mul
    (array, array, int) -> array       mul
    (int, int) -> set / dict           expected_labels, missing_diagonal
    (float, float) -> list             cusp_wells

SELF-REFERENCE IS DISCOUNTED, not hidden. This line's own harnesses now live in
scripts/research/ and contain a deliberately independent Cayley-Dickson oracle.
A corroboration between that oracle and a corpus kernel is real evidence, but it
is evidence THIS LINE INTRODUCED, not evidence the corpus already had. The two
are counted separately and the verdict depends only on the pre-existing ones.

VERDICT OPTIONS, FIXED BEFORE COMPUTING (see main()):
  WIDER_PROBE__PREEXISTING_CORROBORATION_FOUND
      the corpus holds unused internal corroboration beyond cds/cd_sigma.
  WIDER_PROBE__NO_NEW_PREEXISTING_CORROBORATION
      widening found nothing the corpus did not already have from R10.
  WIDER_PROBE__COVERAGE_STILL_NARROW
      the widened probe still exercises too few behaviour classes to say.

Pure Python 3 + numpy.
"""

from __future__ import annotations

import ast
import difflib
import importlib.util
import inspect
import itertools
import json
import os
import resource
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

# A RESOURCE CAP IS NOT OPTIONAL FOR THIS TECHNIQUE, and finding that out cost
# two runs. Capping the size of the CANONICAL FORM does not help: the runaway
# allocation happens INSIDE the probed function, before it returns, so Python
# never gets to raise MemoryError — the process is killed and even buffered
# output is lost (exit 120, empty log). Any tool that discovers corroboration by
# calling unknown functions on synthetic inputs must bound address space first.
# THIRD HAZARD: a probed function can CLOSE YOUR FILE DESCRIPTORS. One in this
# corpus closes fd 1, so the collection loop finished and then every print()
# raised OSError(EBADF). A private duplicate of stdout is taken before any
# probing happens and all output goes there, so the report survives whatever
# the probed code does to the process's descriptors.
_OUT_FD = os.dup(1)


def emit(msg: str = "") -> None:
    try:
        os.write(_OUT_FD, (msg + "\n").encode())
    except OSError:
        pass


_MEM_CAP_BYTES = 4 * 1024 ** 3
try:
    _soft, _hard = resource.getrlimit(resource.RLIMIT_AS)
    resource.setrlimit(resource.RLIMIT_AS,
                       (_MEM_CAP_BYTES,
                        _hard if _hard != resource.RLIM_INFINITY else _MEM_CAP_BYTES))
except (ValueError, OSError):
    pass

REPO = Path(__file__).resolve().parents[2]
R6 = REPO / "scripts/research/self_falsifying_compilation_line_r6_contract.py"
SELF_PREFIX = "self_falsifying_compilation_line"
RNG = np.random.default_rng(20260726)


def _load_r6():
    spec = importlib.util.spec_from_file_location("r6", R6)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def contracts() -> list[str]:
    return sorted(str(p.relative_to(REPO))
                  for p in (REPO / "scripts/research").glob("*contract*.py"))


def is_self(rel: str) -> bool:
    return SELF_PREFIX in Path(rel).name


def load_fn(rel: str, fn: str):
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


# ---------------------------------------------------------------- canonical


# Hard caps. Probing unknown functions with synthetic inputs is inherently
# hazardous: a census routine handed a plausible-looking level can return a
# structure large enough to exhaust memory, and the first run of this rung did
# exactly that (MemoryError inside a nested dict). Every container is bounded
# and the recursion is depth-limited; anything past a cap is simply declared
# uncomparable rather than canonicalised.
MAX_ELEMS = 4096
MAX_DEPTH = 6


def canon(v, depth: int = 0):
    """Hashable canonical form of a return value, or None if uncomparable."""
    if depth > MAX_DEPTH:
        return None
    if isinstance(v, bool):
        return ("b", v)
    if isinstance(v, (int, np.integer)):
        return ("i", int(v))
    if isinstance(v, (float, np.floating)):
        return ("f", round(float(v), 9) + 0.0)
    if isinstance(v, np.ndarray):
        if v.dtype.kind not in "fiub" or v.size > MAX_ELEMS:
            return None
        return ("a", tuple(round(float(x), 9) + 0.0 for x in v.ravel()))
    if isinstance(v, (set, frozenset)):
        if len(v) > MAX_ELEMS:
            return None
        parts = [canon(x, depth + 1) for x in v]
        if any(p is None for p in parts):
            return None
        return ("s", tuple(sorted(parts)))
    if isinstance(v, dict):
        if len(v) > MAX_ELEMS:
            return None
        items = []
        for k, val in v.items():
            ck, cv = canon(k, depth + 1), canon(val, depth + 1)
            if ck is None or cv is None:
                return None
            items.append((ck, cv))
        return ("d", tuple(sorted(items)))
    if isinstance(v, (list, tuple)):
        if len(v) > MAX_ELEMS:
            return None
        parts = [canon(x, depth + 1) for x in v]
        if any(p is None for p in parts):
            return None
        return ("l", tuple(parts))
    return None


# ---------------------------------------------------------------- families


def _arr(n):
    v = RNG.normal(size=n)
    return v


FAMILIES: list[tuple[str, list[tuple]]] = [
    ("int3", [(i, j, b) for b in (3, 4) for i in range(1 << b)
              for j in range(1 << b)]),
    ("int2", [(i, j) for i in range(16) for j in range(16)]),
    ("int1", [(i,) for i in range(1, 64)]),
    ("intlvl", [(lab, b) for b in (4, 5) for lab in range(1, 1 << b)]),
    ("arr8", [(_arr(8), _arr(8)) for _ in range(24)]
             + [(np.eye(8)[i], np.eye(8)[j]) for i in range(8) for j in range(8)]),
    ("arr16", [(_arr(16), _arr(16)) for _ in range(16)]
              + [(np.eye(16)[i], np.eye(16)[j]) for i in range(16) for j in range(16)]),
    ("arr8b", [(_arr(8), _arr(8), 3) for _ in range(24)]),
    ("arr16b", [(_arr(16), _arr(16), 4) for _ in range(16)]),
    ("float2", [(float(x), float(y)) for x, y in RNG.normal(size=(48, 2))]),
]


def behaviour(f, probes):
    out = []
    for args in probes:
        try:
            v = f(*args)
        except (Exception, MemoryError, RecursionError):
            return None
        try:
            c = canon(v)
        except (MemoryError, RecursionError):
            return None
        if c is None:
            return None
        out.append(c)
    if len(set(out)) <= 1:
        return None          # constant over the grid: carries no signal
    return tuple(out)


def profile(f):
    """First input family the function accepts, with its behaviour vector."""
    try:
        n_params = len([p for p in inspect.signature(f).parameters.values()
                        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)])
    except (TypeError, ValueError):
        return None, None
    for name, probes in FAMILIES:
        if not probes or len(probes[0]) != n_params:
            continue
        b = behaviour(f, probes)
        if b is not None:
            return name, b
    return None, None


# ---------------------------------------------------------------- clauses


def clause_p1(r6):
    found = []
    fam_counts: dict[str, int] = {}
    for rel in contracts():
        for name, fp in r6.fingerprints(rel).items():
            f = load_fn(rel, name)
            if f is None:
                continue
            fam, b = profile(f)
            if fam is None:
                continue
            fam_counts[fam] = fam_counts.get(fam, 0) + 1
            found.append({"rel": rel, "name": name, "fp": fp,
                          "family": fam, "behaviour": b, "self": is_self(rel)})
    emit(f"P1_WIDER_PROBE {len(found)} probeable functions "
          f"(R10 reached 31) across {len(contracts())} contracts")
    emit(f"P1_WIDER_PROBE families exercised: {dict(sorted(fam_counts.items()))}")
    emit("P1_WIDER_PROBE PASS — measured")
    return found


def clause_p2(r6, fns):
    """Behaviour classes and their corroboration depth, self-reference split."""
    by_beh: dict[tuple, list] = {}
    for f in fns:
        by_beh.setdefault((f["family"], f["behaviour"]), []).append(f)

    preexisting, introduced = [], []
    for key, group in by_beh.items():
        reps: list[tuple[str, dict]] = []
        for f in group:
            if not any(difflib.SequenceMatcher(None, f["fp"], r[0]).ratio()
                       >= r6.DUP_THRESHOLD for r in reps):
                reps.append((f["fp"], f))
        if len(reps) < 2:
            continue
        corpus_reps = [r for r in reps if not r[1]["self"]]
        entry = (key[0], group, reps)
        if len(corpus_reps) >= 2:
            preexisting.append(entry)
        else:
            introduced.append(entry)

    emit(f"P2_DEPTH {len(by_beh)} distinct behaviour classes; "
          f"{len(preexisting)} with >=2 derivations ALREADY IN THE CORPUS, "
          f"{len(introduced)} only reaching depth 2 via this line's own oracle")
    for label, group, reps in preexisting:
        names = sorted({f"{Path(r[1]['rel']).name}:{r[1]['name']}" for r in reps})
        emit(f"    PRE-EXISTING [{label}] {len(reps)} derivations, "
              f"{len(group)} copies — {names[:4]}")
    for label, group, reps in introduced:
        names = sorted({f"{Path(r[1]['rel']).name}:{r[1]['name']}" for r in reps})
        emit(f"    line-introduced [{label}] {len(reps)} derivations — {names[:4]}")
    emit("P2_DEPTH PASS — measured")
    return preexisting, introduced, len(by_beh)


# ---------------------------------------------------------------- main


def probe_child(out_path: str) -> int:
    """Do all probing here, in a CHILD process, and hand results back as JSON.

    HAZARD 4, and the one that forced this split: patching around a probed
    function that closes fd 1 is not enough. Even with the report going through
    a private dup, output stopped after the header once probing began — probed
    code reaches the parent's descriptors in ways a single dup does not cover,
    and the failure is invisible except as a truncated report.
    Subprocess isolation is the same answer the claim executor reached for
    running gates (R2), for the same reason: never let foreign code run in the
    process that has to report the result.
    """
    r6 = _load_r6()
    rows = []
    for rel in contracts():
        for name, fp in r6.fingerprints(rel).items():
            f = load_fn(rel, name)
            if f is None:
                continue
            fam, b = profile(f)
            if fam is None:
                continue
            rows.append({"rel": rel, "name": name, "fp": fp, "family": fam,
                         "behaviour": repr(b), "self": is_self(rel)})
    with open(out_path, "w") as fh:
        json.dump(rows, fh)
        fh.flush()
        os.fsync(fh.fileno())

    # os._exit, not return: the child's only job is that file, and it is already
    # flushed and fsynced. A normal exit runs interpreter shutdown, which
    # flushes sys.stdout — and a probed function has closed fd 1, so the flush
    # fails and the child exits 120 with the work correctly done. Skipping
    # shutdown makes the exit status describe the job rather than the wreckage.
    os._exit(0)


def main() -> int:
    if len(sys.argv) > 2 and sys.argv[1] == "--probe":
        return probe_child(sys.argv[2])

    emit("SELF-FALSIFYING COMPILATION R11 — widening the corroboration probe")
    emit("=" * 78)
    emit("R10's negative covered ONE signature family. This widens the probe to")
    emit("array-, set-, dict- and float-valued kernels, and discounts")
    emit("corroborations that exist only because this line added an oracle.")
    emit()

    r6 = _load_r6()
    with tempfile.NamedTemporaryFile("r", suffix=".json", delete=False) as fh:
        tmp = fh.name
    try:
        rc = subprocess.run([sys.executable, __file__, "--probe", tmp],
                            capture_output=True, text=True, timeout=1800)
        if rc.returncode != 0:
            emit(f"P1_WIDER_PROBE FAIL — probe child exited {rc.returncode}")
            emit(f"SELF_FALSIFYING_R11_VERDICT WIDER_PROBE__COVERAGE_STILL_NARROW")
            return 1
        with open(tmp) as fh:
            fns = json.load(fh)
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass

    fam_counts: dict[str, int] = {}
    for f in fns:
        fam_counts[f["family"]] = fam_counts.get(f["family"], 0) + 1
    emit(f"P1_WIDER_PROBE {len(fns)} probeable functions "
         f"(R10 reached 31) across {len(contracts())} contracts")
    emit(f"P1_WIDER_PROBE families exercised: {dict(sorted(fam_counts.items()))}")
    emit("P1_WIDER_PROBE PASS — measured")
    emit()
    pre, intro, n_classes = clause_p2(r6, fns)
    emit()

    emit("=" * 78)
    # R10 already established one pre-existing corroboration (cds/cd_sigma).
    new_pre = max(0, len(pre) - 1)
    if n_classes < 3:
        token = "WIDER_PROBE__COVERAGE_STILL_NARROW"
    elif new_pre > 0:
        token = "WIDER_PROBE__PREEXISTING_CORROBORATION_FOUND"
    else:
        token = "WIDER_PROBE__NO_NEW_PREEXISTING_CORROBORATION"

    emit(f"  probeable functions        : {len(fns)}  (R10: 31)")
    emit(f"  distinct behaviour classes : {n_classes}  (R10: 1)")
    emit(f"  pre-existing corroborations: {len(pre)}  "
          f"(R10 knew of 1; new here: {new_pre})")
    emit(f"  line-introduced ones       : {len(intro)}  "
          f"(real evidence, but added by this work — not the corpus's own)")
    emit(f"SELF_FALSIFYING_R11_VERDICT {token}")

    # A probed function closed fd 1 (hazard 3). The report survived because it
    # goes through a private dup — but the INTERPRETER still flushes sys.stdout
    # at shutdown, and flushing to a dead descriptor exits non-zero AFTER all
    # the work is done. Restoring fd 1 from the saved dup makes the exit status
    # mean what it says. Only visible when stdout is buffered, i.e. exactly when
    # a CI gate captures it.
    try:
        os.dup2(_OUT_FD, 1)
    except OSError:
        pass
    try:
        sys.stdout = open(os.devnull, "w")
    except OSError:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
