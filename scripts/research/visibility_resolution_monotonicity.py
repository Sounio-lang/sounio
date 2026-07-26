#!/usr/bin/env python3
"""Detector for the visibility-widening non-monotonicity hazard.

CLAIM (proved experimentally on Madaros, witness below)
-------------------------------------------------------
When symbol resolution over a MERGED table consults visibility as a FILTER inside a
fallback chain — rather than resolving first and validating access afterwards — widening
a definition's visibility is NOT semantics-preserving. Adding `pub` to one definition can
change which definition an unrelated call site resolves to.

Madaros' chain is (self-hosted/check/defs.sio:1440-1473 fn_sig_table_find_prefer_module):
  T1  a free fn whose defining_module_id == the caller's module      (preferred)
  T2  ANY fn whose visibility_kind != VISIBILITY_PRIVATE             (visibility as FILTER)
  T3  fn_sig_table_find(t, name) — first match, private or not       (fallback)

Before widening: T1 misses, T2 misses (every candidate private), T3 returns first-in-table.
After  widening: T2 now matches the widened definition and is consulted BEFORE T3, so the
resolved target changes whenever the widened definition is not T3's pick.

EXPERIMENTAL WITNESS (reproduced in both directions)
  fixtures: <job tmp>/hijack2/{good,bad,caller,hjmain}.sio
    good::f  private, arity 2   bad::f  private->pub, arity 0   caller globs both, calls f(10,20)
  bad::f private -> error[E175]  (resolved good::f, arity 2, correctly typed)
  bad::f pub     -> error[E010]  (resolved bad::f,  arity 0, call now ill-typed)
  reverting restores E175. One `pub` changed the target of a call that `bad::f` never
  participated in.

WHY IT IS SILENT IN THE DANGEROUS CASE
  The flip is only loud when the candidates' signatures differ observably. When two
  definitions share an identical TYPE SEQUENCE the flipped call typechecks, and if their
  parameters are PERMUTED the emitted code is silently wrong. Live instance in this tree:
    self-hosted/compiler/main.sio  fn ir_call_sret(dst, dest_reg, fn_name, fn_id, arg0, n)
    self-hosted/ir/ir.sio      pub fn ir_call_sret(dst, fn_id, fn_name, dest_reg, first, n)
  Same sequence (i64, i64, Name, i64, i64, i64); positions 2 and 4 swapped. A flip there
  transposes a destination register with a function id in emitted IR, typechecking cleanly.

WHAT THIS SCRIPT DOES
  Computes the exact set of names for which ANY visibility change is potentially
  resolution-changing: names defined in more than one file whose definitions have unifiable
  (here: identical) parameter type sequences. That set bounds the blast radius of a
  `pub`-widening campaign — outside it, widening is safe; inside it, each edit needs review.

  It reports three classes, because they carry different risk:
    cross-module + identical types  -> SILENT flip risk (the dangerous class)
    same-file duplicates            -> "which definition wins" hygiene, not a flip
    differing type sequences        -> LOUD flip (a type error will surface it)

USAGE
  python3 scripts/research/visibility_resolution_monotonicity.py [entry.sio]
  default entry: self-hosted/compiler/main.sio  (its transitive `use` closure)

The generalisation, and the reason this is not a Sounio quirk: any self-hosted or
bootstrapped compiler that resolves names over one flat merged table with a
visibility-filtered fallback tier has this property. Resolving by module-qualified identity
and validating access afterwards removes it by construction.
"""
import os
import re
import sys
import collections

ROOT = "self-hosted"
USE = re.compile(r'^use\s+([A-Za-z0-9_:]+)::(?:\*|\{)')
FN = re.compile(r'^(pub\s+)?fn\s+([A-Za-z0-9_]+)\s*\(([^)]*)\)')


def closure(entry):
    """Transitive `use` closure, resolved the way the compiler resolves imports."""
    seen, work = set(), [entry]
    while work:
        f = work.pop()
        if f in seen or not os.path.exists(f):
            continue
        seen.add(f)
        for line in open(f, errors="ignore"):
            m = USE.match(line.strip())
            if not m:
                continue
            p = os.path.join(ROOT, m.group(1).replace("::", "/") + ".sio")
            if os.path.exists(p):
                work.append(p)
    return seen


def type_sequence(params):
    """Parameter TYPES only. Names are irrelevant to whether a flip typechecks — and
    stripping them is what exposes the permuted-parameter case as identical."""
    return tuple(t.split(":")[-1].strip() for t in params.split(",") if t.strip())


def definitions(files):
    defs = collections.defaultdict(list)
    for f in sorted(files):
        for i, line in enumerate(open(f, errors="ignore"), 1):
            m = FN.match(line)
            if m:
                defs[m.group(2)].append(
                    {"file": f, "line": i, "pub": bool(m.group(1)),
                     "types": type_sequence(m.group(3))}
                )
    return defs


def main(entry):
    files = closure(entry)
    defs = definitions(files)
    homonyms = {n: v for n, v in defs.items() if len(v) > 1}

    silent, hygiene, loud = [], [], []
    for n, v in homonyms.items():
        distinct_files = {d["file"] for d in v}
        distinct_types = {d["types"] for d in v}
        if len(distinct_types) > 1:
            loud.append((n, v))
        elif len(distinct_files) == 1:
            hygiene.append((n, v))
        else:
            silent.append((n, v))

    print(f"entry:            {entry}")
    print(f"closure files:    {len(files)}")
    print(f"distinct fn names:{len(defs)}")
    print(f"homonyms:         {len(homonyms)}")
    print()
    print(f"SILENT flip risk (cross-file, identical type sequence): {len(silent)}")
    for n, v in sorted(silent):
        print(f"  {n}  arity={len(v[0]['types'])}")
        for d in v:
            print(f"      {'pub ' if d['pub'] else '    '}{d['file']}:{d['line']}")
    print()
    print(f"LOUD flip (differing type sequences, a type error surfaces it): {len(loud)}")
    for n, v in sorted(loud):
        print(f"  {n}")
        for d in v:
            print(f"      {'pub ' if d['pub'] else '    '}{d['file']}:{d['line']}  {d['types']}")
    print()
    print(f"same-file duplicates ('which definition wins' hygiene): {len(hygiene)}")
    for n, v in sorted(hygiene):
        lines = ", ".join(str(d["line"]) for d in v)
        print(f"  {n}  {v[0]['file']}:{lines}")
    print()
    print("VERDICT: a `pub`-widening campaign is resolution-safe for every name NOT listed")
    print("under SILENT or LOUD above. Names listed there require per-edit review, because")
    print("widening one of them can retarget calls that do not mention it.")
    return 0 if not silent else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else f"{ROOT}/compiler/main.sio"))
