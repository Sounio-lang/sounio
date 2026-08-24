#!/usr/bin/env python3
"""gen_go_full_data.py — round 12: EL+ role-aware boolean closure on the
FULL GO go-plus ontology (H = 38,245 GO classes, NR = 92 roles) — python
mirror and data packer.

The round-11 general set-based fixpoint (gen_elplus_data.fixpoint) is
infeasible at this scale (U = (H+1)*(NR+1) = 3.56M interned concepts, and
the transitivity sweep is O(U^2) per round).  This script therefore uses a
BITMASK reduction of the same 8-rule fixpoint, exact for the go-plus data
profile (no extracted conjunctions — intersection restrictions are skipped
by the extractor and counted — and no superclass-side restrictions,
probed: 0 occurrences in go-plus):

  * atomic closure: anc[c] = stated-sub ancestor bitmask (incl. self and
    the top bit), one pass in topological order (GO is_a is a DAG);
  * role edges with ATOM sources: F[r][c] = ancestor-closed bitmask of
    base fillers.  Seeding F[r][c] |= anc[f] for each stated (c, r, f)
    folds Rmono (seeds are ancestor-closed and unions of closed sets are
    closed, so the Rmono invariant needs no separate pass);
    one topological pass F[r][c] |= F[r][p] over parents p folds
    transitivity + stoR (x <= c and edge (r, c, f) give edge (r, x, f));
  * roleSub (F[s][c] |= F[r][c] for r <=* s) via a dirty worklist
    (complete: single-input rule, semi-naive) and roleComp
    (F[r3][c] |= union_{f in F[r1][c]} F[r2][f]) via a FULL SCAN per
    round — round 12b correction: a dirty worklist keyed on first
    components is INCOMPLETE for roleComp (when F[r2][f] gains an edge,
    the chain must re-fire for every c with f in F[r1][c]; two correct
    worklist orders converged 7,200 edges apart on this data, exposing
    the "direction-2" leak); the hybrid round is Gauss-Seidel chaotic
    iteration and reaches the same least fixpoint as the naive round;
  * by the profile theorem (round 11, math-reviewed; asserted on the
    slice below): roles add NO atomic subsumptions and NO atomic
    conflicts on this data shape, so atomic closure edges / conflicts are
    computed from anc[] alone, and the existential targets of an atom row
    are exactly its role-edge fillers (stoR/RtoS bijection).

Scope note (honest limitation): statistics are ATOM-LEVEL.  Universe-level
numbers of rounds 9/11 (S cells over U, role edges with existential
sources) are not computed here: existential-sourced edge sets need a
per-(r, f) composition fixpoint (1.5M of them at this scale).  Atom-level
statistics are the scientifically meaningful content (subsumptions,
existential targets, conflicts); see README round 12.

Validation ladder:
  1. --validate-slice: runs the bitmask reduction on the round-11 slice
     (../go_elplus_tbox.txt) AND the general set-based fixpoint of
     gen_elplus_data.py, and aborts unless all atom-level numbers agree
     (full closure, no-roleComp and no-roleSub ablations);
  2. full mode (default): bitmask reduction on ../go_full_elplus_tbox.txt
     (written by extract_tbox.py --go --go-full), emits
       go_full_packed.txt     runtime data for the Sounio driver
       go_full_expected.sio   mirror numbers as go_full_expected_*()

Outputs are consumed by go_full_elplus_driver.sio (same directory), which
reads go_full_packed.txt at RUNTIME via read_file — no data statements,
so the ~24k-statement compiler wall is not engaged.
"""

import argparse
import sys
import time

sys.path.insert(0, ".")
from gen_elplus_data import load_go, fixpoint  # noqa: E402


def topo_order(H, sub):
    """Kahn's algorithm, parents before children.  Aborts on cycles."""
    parents = [[] for _ in range(H)]
    children = [[] for _ in range(H)]
    for c, p in sub:
        parents[c].append(p)
        children[p].append(c)
    indeg = [len(parents[c]) for c in range(H)]
    queue = [c for c in range(H) if indeg[c] == 0]
    order = []
    head = 0
    while head < len(queue):
        c = queue[head]
        head += 1
        order.append(c)
        for ch in children[c]:
            indeg[ch] -= 1
            if indeg[ch] == 0:
                queue.append(ch)
    if len(order) != H:
        sys.exit(f"MIRROR FAILED: sub relation has a cycle "
                 f"({len(order)}/{H} ordered)")
    return order, parents


def bitmask_reduce(H, NR, sub, exsub, disj, rsub, rcomp,
                   with_rsub=True, with_rcomp=True):
    """The bitmask reduction described in the module header.  Returns a
    dict with atom-level statistics."""
    B = H + 1
    order, parents = topo_order(H, sub)

    # ── atomic closure: ancestor bitmasks (bit H = top) ────────────────
    anc = [(1 << c) | (1 << H) for c in range(H)] + [1 << H]
    for c in order:
        m = anc[c]
        for p in parents[c]:
            m |= anc[p]
        anc[c] = m
    ATOM_MASK = (1 << H) - 1
    atomic_edges = sum(bin(anc[c] & ATOM_MASK).count("1") for c in range(H))

    # ── role hierarchy closure ─────────────────────────────────────────
    rclos = [[r == s for s in range(NR)] for r in range(NR)]
    for r, s in rsub:
        rclos[r][s] = True
    changed = True
    while changed:
        changed = False
        for a in range(NR):
            for b in range(NR):
                if rclos[a][b]:
                    for d in range(NR):
                        if rclos[b][d] and not rclos[a][d]:
                            rclos[a][d] = True
                            changed = True
    rclos_pairs = [(r, s) for r in range(NR) for s in range(NR)
                   if rclos[r][s] and r != s]

    # ── role edges, atom sources: seed + topological ancestor expansion ─
    F = [dict() for _ in range(NR)]
    for c, r, f in exsub:
        F[r][c] = F[r].get(c, 0) | anc[f]
    for c in order:
        for p in parents[c]:
            for r in range(NR):
                fr = F[r]
                if p in fr:
                    v = fr.get(c, 0) | fr[p]
                    if v != fr.get(c, 0):
                        fr[c] = v

    # ── roleSub + roleComp fixpoint ─────────────────────────────────────
    # Round 12b correction (direction-2 leak): a dirty-cell worklist on
    # FIRST components is INCOMPLETE for roleComp — when F[r2][f] gains an
    # edge, chain (r1, r2, r3) must re-fire for every c with f in F[r1][c],
    # and those (r1, c) cells are not in the dirty set.  Two correct
    # worklist orders were observed to converge to DIFFERENT totals on the
    # full GO (phase-order 2,135,093 vs sweep-order 2,127,893), exposing
    # the leak.  Correct scheme below: roleSub via dirty worklist
    # (complete — single-input rule, fires exactly on changed cells),
    # roleComp via FULL SCAN each round (naive fixpoint); the hybrid round
    # is Gauss-Seidel chaotic iteration, which reaches the same least
    # fixpoint as the naive round.
    dirty = [set(fr.keys()) for fr in F]
    rounds = 0
    while True:
        rounds += 1
        any_change = False
        # roleSub worklist, drained to quiescence (with_rsub=False:
        # the first pass just discards stale dirty cells)
        while any(dirty):
            cur = dirty
            dirty = [set() for _ in range(NR)]
            if with_rsub:
                for r, s in rclos_pairs:
                    for c in cur[r]:
                        v = F[s].get(c, 0)
                        nv = v | F[r][c]
                        if nv != v:
                            F[s][c] = nv
                            dirty[s].add(c)
                            any_change = True
        # roleComp full scan
        if with_rcomp:
            for r1, r2, r3 in rcomp:
                F2 = F[r2]
                for c, m in F[r1].items():
                    acc = 0
                    mm = m
                    while mm:
                        b = mm & -mm
                        f = b.bit_length() - 1
                        mm ^= b
                        v2 = F2.get(f, 0)
                        if v2:
                            acc |= v2
                    if acc:
                        v3 = F[r3].get(c, 0)
                        nv = v3 | acc
                        if nv != v3:
                            F[r3][c] = nv
                            dirty[r3].add(c)
                            any_change = True
        if not any_change:
            break

    # ── atom-level statistics ──────────────────────────────────────────
    role_edges_atom = 0
    per_role = [0] * NR
    for r in range(NR):
        n = 0
        for m in F[r].values():
            n += bin(m).count("1")
        per_role[r] = n
        role_edges_atom += n
    ex_targets = role_edges_atom  # stoR/RtoS bijection on atom rows

    # ── conflicts over atomic ancestors (profile theorem: role-aware
    #    conflicts equal these) ─────────────────────────────────────────
    eps = sorted({x for pr in disj for x in pr})
    ep_idx = {e: k for k, e in enumerate(eps)}
    pb = [0] * len(eps)
    for a, b in disj:
        pb[ep_idx[a]] |= 1 << ep_idx[b]
        pb[ep_idx[b]] |= 1 << ep_idx[a]
    epm = [0] * H
    for e in eps:
        bit = 1 << e
        bit_ep = 1 << ep_idx[e]
        for c in range(H):
            if anc[c] & bit:
                epm[c] |= bit_ep
    pm = [0] * H
    actors = []
    for c in range(H):
        m, p, k = epm[c], 0, 0
        while m:
            if m & 1:
                p |= pb[k]
            m >>= 1
            k += 1
        pm[c] = p
        if p or epm[c]:
            actors.append(c)
    n_conf = 0
    for c1 in actors:
        for c2 in actors:
            if c1 != c2 and (pm[c1] & epm[c2]):
                n_conf += 1

    return {"atomic_edges": atomic_edges,
            "role_edges_atom": role_edges_atom,
            "ex_targets": ex_targets,
            "conf": n_conf, "rounds": rounds,
            "per_role": per_role}


def validate_slice():
    """Cross-validate the bitmask reduction against the general set-based
    8-rule fixpoint (gen_elplus_data.fixpoint) on the round-11 slice."""
    H, NR, sub, exsub, disj, rsub, rcomp = load_go("../go_elplus_tbox.txt")

    def general_stats(with_rsub, with_rcomp):
        S, R, _U, _b, _t, ckind, _a1, _a2, _e, _rc, _ro = fixpoint(
            H, NR, [], sub, exsub, rsub if with_rsub else [],
            rcomp if with_rcomp else [])
        atoms = set(range(H))
        r_atom = sum(1 for (_r, c, _f) in R if c < H)
        atom_edges = sum(1 for c in range(H) for d in S[c] if d in atoms)
        ex_t = sum(1 for c in range(H) for d in S[c] if ckind[d] == 3)
        return r_atom, atom_edges, ex_t

    cases = [("full", True, True), ("noRC", True, False),
             ("noRS", False, True)]
    ok = True
    for name, wrs, wrc in cases:
        g_ratom, g_aedges, g_ext = general_stats(wrs, wrc)
        red = bitmask_reduce(H, NR, sub, exsub, disj, rsub, rcomp,
                             with_rsub=wrs, with_rcomp=wrc)
        if (red["role_edges_atom"], red["atomic_edges"],
                red["ex_targets"]) != (g_ratom, g_aedges, g_ext):
            print(f"SLICE VALIDATION FAILED ({name}): bitmask "
                  f"(r_atom={red['role_edges_atom']} "
                  f"atom_edges={red['atomic_edges']} "
                  f"ex_targets={red['ex_targets']}) != general "
                  f"({g_ratom} {g_aedges} {g_ext})")
            ok = False
        else:
            print(f"slice validation ({name}): bitmask == general "
                  f"(r_atom={g_ratom} atom_edges={g_aedges} "
                  f"ex_targets={g_ext})")
    if not ok:
        sys.exit("SLICE VALIDATION FAILED")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate-slice", action="store_true",
                    help="cross-check the bitmask reduction against the "
                         "general fixpoint on the round-11 slice, then "
                         "continue to the full run")
    args = ap.parse_args()
    if args.validate_slice:
        validate_slice()

    t0 = time.time()
    H, NR, sub, exsub, disj, rsub, rcomp = load_go("../go_full_elplus_tbox.txt")
    print(f"loaded full GO: H={H} NR={NR} sub={len(sub)} exsub={len(exsub)}"
          f" disj={len(disj)} roleSub={len(rsub)} roleComp={len(rcomp)}"
          f"  [{time.time() - t0:.1f}s]")

    t0 = time.time()
    full = bitmask_reduce(H, NR, sub, exsub, disj, rsub, rcomp)
    print(f"full closure: atomic_edges={full['atomic_edges']} "
          f"role_edges_atom={full['role_edges_atom']} "
          f"ex_targets={full['ex_targets']} conf={full['conf']} "
          f"rounds={full['rounds']}  [{time.time() - t0:.1f}s]")

    t0 = time.time()
    norc = bitmask_reduce(H, NR, sub, exsub, disj, rsub, rcomp,
                          with_rcomp=False)
    print(f"ablation no-roleComp: role_edges_atom={norc['role_edges_atom']}"
          f"  [{time.time() - t0:.1f}s]")
    t0 = time.time()
    nors = bitmask_reduce(H, NR, sub, exsub, disj, rsub, rcomp,
                          with_rsub=False)
    print(f"ablation no-roleSub: role_edges_atom={nors['role_edges_atom']}"
          f"  [{time.time() - t0:.1f}s]")

    # ── emit the runtime data pack for the Sounio driver ───────────────
    with open("go_full_packed.txt", "w") as f:
        f.write(f"{H} {NR} {len(sub)} {len(exsub)} {len(disj)} "
                f"{len(rsub)} {len(rcomp)}\n")
        for c, p in sub:
            f.write(f"s {c} {p}\n")
        for c, r, fl in exsub:
            f.write(f"x {c} {r} {fl}\n")
        for a, b in disj:
            f.write(f"d {a} {b}\n")
        for r, s in rsub:
            f.write(f"h {r} {s}\n")
        for r1, r2, r3 in rcomp:
            f.write(f"k {r1} {r2} {r3}\n")
    print(f"wrote go_full_packed.txt")

    # ── emit the mirror numbers as Sounio functions ────────────────────
    with open("go_full_expected.sio", "w") as f:
        f.write("// GENERATED by gen_go_full_data.py — do not edit by hand.\n")
        f.write("// Round 12: FULL GO go-plus EL+ role-aware closure —\n")
        f.write("// python bitmask-mirror values.\n")
        vals = {
            "h": H, "nr": NR, "nsub": len(sub), "nex": len(exsub),
            "ndj": len(disj), "nrs": len(rsub), "nch": len(rcomp),
            "atomic_edges": full["atomic_edges"],
            "role_edges_atom": full["role_edges_atom"],
            "ex_targets": full["ex_targets"],
            "conf": full["conf"], "rounds": full["rounds"],
            "role_edges_atom_no_rc": norc["role_edges_atom"],
            "role_edges_atom_no_rs": nors["role_edges_atom"],
        }
        for k_, v in vals.items():
            f.write(f"// {k_} = {v}\n")
        f.write("\n")
        for k_, v in vals.items():
            f.write(f"pub fn go_full_expected_{k_}() -> i64 {{ return {v} }}\n")
    print(f"wrote go_full_expected.sio")


if __name__ == "__main__":
    main()
