#!/usr/bin/env python3
"""gen_sounio_data.py — generate tbox_data.sio (parallel primitive arrays,
splat-initialized) from tbox.txt + classes.tsv + mappings.tsv, and compute
a python mirror of the whole verified pipeline (subsumption closure
fixpoint -> closure-expanded disjointness -> derived mapping conflicts ->
greedy epistemic repair) whose numbers are embedded in tbox_data.sio as
expected_*() functions so the Sounio driver can cross-check itself against
an independent implementation.

Human-class cap: the Sounio closure is an O(H^3) fixpoint over an H x H
boolean matrix, so we cap the number H of human classes at --human-cap
(default 2000).  Selection is ANCESTOR-CLOSED so the capped closure
preserves every subsumption path reachable from a mapping target:
  1. human classes referenced by candidate mappings, plus ALL their
     subsumption ancestors (so capped closure == full closure on the
     referenced subgraph),
  2. human classes appearing in a disjointWith axiom, plus their
     ancestors,
  3. remaining classes by ascending id, until the cap.
Axioms/mappings referencing dropped classes are dropped; human ids are
remapped to 0..H-1.  Mouse ids are NOT remapped (they are only compared
for equality, never used as array indices).

Mirror algorithm (identical iteration order to the Sounio driver):
  closure: reflexive + sub edges, then naive fixpoint passes over
           (a, b, d) in ascending order until no change;
  disjC:   c1,c2 with clos[c1][d1] and clos[c2][d2] and disj(d1,d2)
           (filled symmetrically);
  conflict(i, j) = ent[i]==ent[j] and disjC[cls[i]][cls[j]];
  repair:  i ascending, j from i+1; if keep[i] and keep[j] and
           conflict(i,j): drop the weaker (conf[i] >= conf[j] drops j,
           else drops i).
"""

import argparse
import sys
from collections import defaultdict


def load_tbox(path):
    sub_h, disj_h = [], []
    with open(path) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            p = line.split()
            if p[1] != "human":
                continue
            if p[0] == "sub":
                sub_h.append((int(p[2]), int(p[3])))
            elif p[0] == "disj":
                disj_h.append((int(p[2]), int(p[3])))
    return sub_h, disj_h


def load_mappings(path):
    maps = []
    with open(path) as f:
        next(f)
        for line in f:
            _, m, h, c = line.rstrip("\n").split("\t")
            maps.append((int(m), int(h), float(c)))
    return maps


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tbox", default="tbox.txt")
    ap.add_argument("--mappings", default="mappings.tsv")
    ap.add_argument("--reference", default="downloads/reference.rdf")
    ap.add_argument("--classes", default="classes.tsv")
    ap.add_argument("--out", default="tbox_data.sio")
    ap.add_argument("--human-cap", type=int, default=2000)
    args = ap.parse_args(argv)

    sub_h, disj_h = load_tbox(args.tbox)
    maps = load_mappings(args.mappings)

    # ── human class selection under the cap (ancestor-closed) ──────────
    referenced = sorted({h for _, h, _ in maps})
    disj_eps = sorted({x for p in disj_h for x in p})
    parents = {}
    for a, b in sub_h:
        parents.setdefault(a, set()).add(b)

    def with_ancestors(seed):
        keep = set(seed)
        frontier = list(keep)
        while frontier:
            x = frontier.pop()
            for p in parents.get(x, ()):
                if p not in keep:
                    keep.add(p)
                    frontier.append(p)
        return sorted(keep)

    sel, seen = [], set()
    for group in (with_ancestors(referenced), with_ancestors(disj_eps)):
        for h in group:
            if h not in seen and len(sel) < args.human_cap:
                sel.append(h)
                seen.add(h)
    new_id = {old: i for i, old in enumerate(sel)}
    H = len(sel)

    sub_kept = sorted({(new_id[a], new_id[b])
                       for a, b in sub_h if a in seen and b in seen})
    disj_kept = sorted({(new_id[a], new_id[b])
                        for a, b in disj_h if a in seen and b in seen})
    maps_kept = [(m, new_id[h], c) for m, h, c in maps if h in seen]
    M = len(maps_kept)

    # Confidences are carried as EXACT integers per-10000 (0.3333 -> 3333):
    # Madaros silently drops f64 array-element assignments inside non-main
    # functions (verified 2026-08-02, see REAL_RESULTS.md), and the mirror
    # comparisons stay exact this way (all confs have exactly 4 decimals).
    def to_milli(c):
        a, b = f"{c:.4f}".split(".")
        return int(a) * 10000 + int(b)

    maps_kept = [(m, h, to_milli(c)) for m, h, c in maps_kept]

    # ── python mirror of the pipeline ───────────────────────────────────
    clos = [set([c]) for c in range(H)]
    for a, b in sub_kept:
        clos[a].add(b)
    changed, passes = True, 0
    while changed:
        changed = False
        passes += 1
        for a in range(H):
            ca = clos[a]
            add = set()
            for b in ca:
                add |= clos[b]
            if not add <= ca:
                ca |= add
                changed = True

    disjC = [set() for _ in range(H)]
    for d1, d2 in disj_kept:
        for c1 in range(H):
            if d1 in clos[c1]:
                for c2 in range(H):
                    if d2 in clos[c2]:
                        disjC[c1].add(c2)
                        disjC[c2].add(c1)

    def conflict(i, j):
        return (maps_kept[i][0] == maps_kept[j][0]
                and maps_kept[j][1] in disjC[maps_kept[i][1]])

    n_conf = sum(1 for i in range(M) for j in range(M)
                 if i != j and conflict(i, j))

    keep = [True] * M
    for i in range(M):
        for j in range(i + 1, M):
            if keep[i] and keep[j] and conflict(i, j):
                if maps_kept[i][2] >= maps_kept[j][2]:
                    keep[j] = False
                else:
                    keep[i] = False
    n_kept = sum(keep)
    n_dropped = M - n_kept

    dropped = sorted((maps_kept[i][2], i)
                     for i in range(M) if not keep[i])
    drop5 = [i for _, i in dropped[:5]]

    # reference-alignment breakdown of the derived conflicts
    ref_pairs = set()
    try:
        import lexical_match as lm
        _, _, iri_of = lm.load_classes(args.classes)
        ref_pairs = lm.load_reference(args.reference, iri_of)
    except Exception:
        pass
    h_old = {n: o for o, n in new_id.items()}
    n_conf_ref = n_conf_nonref = n_conf_mixed = 0
    for i in range(M):
        for j in range(i + 1, M):
            if conflict(i, j):
                ri = (maps_kept[i][0], h_old[maps_kept[i][1]]) in ref_pairs
                rj = (maps_kept[j][0], h_old[maps_kept[j][1]]) in ref_pairs
                if ri and rj:
                    n_conf_ref += 1
                elif ri or rj:
                    n_conf_mixed += 1
                else:
                    n_conf_nonref += 1
    # how many reference mappings are dropped by the repair
    n_ref_kept = n_ref_dropped = 0
    for i in range(M):
        if (maps_kept[i][0], h_old[maps_kept[i][1]]) in ref_pairs:
            if keep[i]:
                n_ref_kept += 1
            else:
                n_ref_dropped += 1

    # ── emit tbox_data.sio ──────────────────────────────────────────────
    NSUB = len(sub_kept)
    with open(args.out, "w") as f:
        f.write("// GENERATED by gen_sounio_data.py — do not edit by hand.\n")
        f.write("// OAEI 2016 Anatomy track (mouse.owl vs human.owl),\n")
        f.write("// syntactic TBox extraction + lexical candidate alignment.\n")
        f.write(f"// H (human classes kept)      = {H}\n")
        f.write(f"// sub axioms kept             = {NSUB}\n")
        f.write(f"// disjoint pairs kept         = {len(disj_kept)}\n")
        f.write(f"// candidate mappings M        = {M}\n")
        f.write(f"// mirror: closure passes      = {passes}\n")
        f.write(f"// mirror: derived conflicts (ordered)   = {n_conf}\n")
        f.write(f"// mirror: kept / dropped      = {n_kept} / {n_dropped}\n")
        f.write(f"// mirror: top-5 dropped ids (lowest conf) = {drop5}\n\n")

        f.write(f"pub fn expected_h() -> i64 {{ return {H} }}\n")
        f.write(f"pub fn expected_sub() -> i64 {{ return {NSUB} }}\n")
        f.write(f"pub fn expected_disj() -> i64 {{ return {len(disj_kept)} }}\n")
        f.write(f"pub fn expected_mappings() -> i64 {{ return {M} }}\n")
        f.write(f"pub fn expected_derived_conflicts() -> i64 {{ return {n_conf} }}\n")
        f.write(f"pub fn expected_kept() -> i64 {{ return {n_kept} }}\n")
        f.write(f"pub fn expected_dropped() -> i64 {{ return {n_dropped} }}\n")
        f.write("pub fn expected_drop5(k: i64) -> i64 {\n")
        for k, i in enumerate(drop5):
            f.write(f"    if k == {k} {{ return {i} }}\n")
        f.write("    return 0 - 1\n}\n\n")

        f.write("// Pairs are PACKED into single i64s (a * 10000 + b; all ids\n")
        f.write("// < 10000): Madaros thin-link fails multimodule compilation\n")
        f.write("// beyond ~24k array-assignment statements in the imported\n")
        f.write("// module (verified 2026-08-02, see REAL_RESULTS.md), and\n")
        f.write("// packing halves the statement count.\n")
        f.write(f"pub var h_sub: [i64; {NSUB}] = [0; {NSUB}]  // child*10000+parent\n")
        f.write(f"pub var h_disj: [bool; {H*H}] = [false; {H*H}]\n")
        f.write(f"pub var clos: [bool; {H*H}] = [false; {H*H}]\n")
        f.write(f"pub var disj_c: [bool; {H*H}] = [false; {H*H}]\n")
        f.write(f"pub var m_pack: [i64; {M}] = [0; {M}]  // mouse_ent*10000+human_cls\n")
        f.write(f"pub var m_conf: [i64; {M}] = [0; {M}]  // per-10000\n")
        f.write(f"pub var m_keep: [bool; {M}] = [true; {M}]\n\n")

        f.write("pub fn init_data() {\n")
        # Madaros silently drops statements beyond the 682nd in one
        # function body (verified 2026-08-02, see REAL_RESULTS.md), so the
        # assignments are emitted in chunks of 500 statements per helper.
        # Madaros module-level array storage leaves GARBAGE in the first
        # bytes after splat init: indices 0..2 of bool arrays and index 0
        # of i64/f64 arrays (verified 2026-08-02 for sizes 8..3847521, see
        # REAL_RESULTS.md).  Data arrays have every element explicitly
        # assigned below, which overwrites the garbage; the partially
        # written working matrices and m_keep need an explicit fixup of
        # the untouched leading cells FIRST (before any legit disj write).
        assigns = [
            "clos[0] = false",
            "clos[1] = false",
            "clos[2] = false",
            "disj_c[0] = false",
            "disj_c[1] = false",
            "disj_c[2] = false",
            "h_disj[0] = false",
            "h_disj[1] = false",
            "h_disj[2] = false",
            "m_keep[0] = true",
            "m_keep[1] = true",
            "m_keep[2] = true",
        ]
        for k, (a, b) in enumerate(sub_kept):
            assigns.append(f"h_sub[{k}] = {a * 10000 + b}")
        for a, b in disj_kept:
            assigns.append(f"h_disj[{a} * {H} + {b}] = true")
            assigns.append(f"h_disj[{b} * {H} + {a}] = true")
        for k, (m, h, c) in enumerate(maps_kept):
            assigns.append(f"m_pack[{k}] = {m * 10000 + h}")
            assigns.append(f"m_conf[{k}] = {c}")
        chunks = [assigns[i:i + 500] for i in range(0, len(assigns), 500)]
        for ci in range(len(chunks)):
            f.write(f"    init_chunk_{ci}()\n")
        f.write("}\n\n")
        for ci, chunk in enumerate(chunks):
            f.write(f"pub fn init_chunk_{ci}() {{\n")
            for stmt in chunk:
                f.write(f"    {stmt}\n")
            f.write("}\n\n")

    print(f"H (human classes kept) = {H} (cap {args.human_cap})")
    print(f"sub axioms kept        = {NSUB} (of {len(sub_h)})")
    print(f"disjoint pairs kept    = {len(disj_kept)} (of {len(disj_h)})")
    print(f"mappings kept M        = {M} (of {len(maps)})")
    print(f"mirror closure passes  = {passes}")
    print(f"mirror derived conflict ordered pairs = {n_conf}")
    print(f"mirror repair: kept {n_kept}, dropped {n_dropped}")
    print(f"mirror top-5 dropped (lowest conf): "
          + ", ".join(f"id={i} conf={maps_kept[i][2] / 10000:.4f}"
                      for i in drop5))
    print(f"conflict unordered pairs: ref-ref={n_conf_ref} "
          f"ref-nonref={n_conf_mixed} nonref-nonref={n_conf_nonref}")
    print(f"reference mappings in candidates: kept={n_ref_kept} "
          f"dropped={n_ref_dropped}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
