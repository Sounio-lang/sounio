#!/usr/bin/env python3
"""gen_chebi_data.py — round 15: EL+ role-aware boolean closure on ChEBI
(H=218253 classes, rich role structure: has_part, is_conjugate_base_of, ...)
and PATO.  Python sparse/bitmask mirror + self-validating packed-data emitter.

Reuses the validated machinery unchanged:
  - extract_tbox.parse_go / parse_ro   (rounds 11-12 OWL extraction)
  - gen_go_full_data.bitmask_reduce    (round-12 bitmask fixpoint mirror,
                                        validated against the general
                                        set-based fixpoint on the round-11
                                        slice, incl. ablations)
  - gen_multi_data.extract_obo         (round-13 namespace-restricted OBO
                                        extraction, RO-closed role set)
  - gen_multi_data.run_mirror          (full mirror + ablations +
                                        grouped-conflict cross-check)
  - gen_multi_data.emit_packed         (round-13 self-validating packed
                                        format: 13-int header carrying the
                                        7 axiom counts AND the 6 mirror
                                        values, then axiom lines)

Targets:
  chebi    CHEBI-namespace classes of downloads/chebi.owl (+ RO axioms)
  pato     PATO-namespace classes of downloads/pato.owl (+ RO axioms)

Same namespace-only policy as rounds 12/13: classes/parents/fillers/
disjoint partners restricted to the ontology's own namespace,
owl:deprecated classes excluded, role axioms from ro.owl, role set
RO-closed (superproperties + composition targets, iterated).

Outputs (all in this directory):
  chebi_packed.txt, pato_packed.txt   runtime data for the Sounio driver
  <target>_classes.tsv, _roles.tsv, _elplus_tbox.txt
                                      documentation/extraction record

The Sounio driver (chebi_pato_elplus_driver.sio, hand-written, round-14
sparse sorted-list engine multi-target generalisation) reads the packed
files at RUNTIME; its compile-time capacities are sized from the numbers
printed by this script (it refuses the data and FAILs loudly if a
capacity is exceeded).

Run from this directory:
  python3 gen_chebi_data.py              # everything
  python3 gen_chebi_data.py --only chebi # one target
"""

import argparse
import sys
import time

sys.path.insert(0, "../real-data/scale")
sys.path.insert(0, "../real-data")
sys.path.insert(0, ".")

from gen_multi_data import extract_obo, run_mirror, emit_packed, \
    emit_tsvs, grouped_conf  # noqa: E402
from gen_go_full_data import topo_order  # noqa: E402

OBO_ONTS = [
    ("chebi", "CHEBI", "downloads/chebi.owl"),
    ("pato", "PATO", "downloads/pato.owl"),
]

RO = "downloads/ro.owl"

# Threshold: Python bigint bitmasks cost O(H²/64) words and OOM near H≈2e5.
# Sparse set fixpoint is bit-equivalent (validated on PATO) and scales.
SPARSE_H_THRESHOLD = 50000


def sparse_reduce(H, NR, sub, exsub, disj, rsub, rcomp,
                  with_rsub=True, with_rcomp=True):
    """Set-based mirror of gen_go_full_data.bitmask_reduce (same stats).

    Uses frozenset-like set unions instead of H-bit integers.  Required for
    ChEBI-scale H (≈2e5): bitmask_reduce MemoryError's on this machine.
    Cross-checked equal to bitmask_reduce on PATO (H=1887) for full +
    both ablations.
    """
    order, parents = topo_order(H, sub)
    TOP = H
    anc = [{c, TOP} for c in range(H)]
    for c in order:
        s = set(anc[c])
        for p in parents[c]:
            s |= anc[p]
        anc[c] = s
    atomic_edges = sum(len(anc[c]) - 1 for c in range(H))  # drop TOP

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

    F = [dict() for _ in range(NR)]
    for c, r, f in exsub:
        s = F[r].get(c)
        if s is None:
            F[r][c] = set(anc[f])
        else:
            s |= anc[f]
    for c in order:
        for p in parents[c]:
            for r in range(NR):
                fr = F[r]
                if p in fr:
                    v = fr.get(c)
                    if v is None:
                        fr[c] = set(fr[p])
                    else:
                        nv = v | fr[p]
                        if len(nv) != len(v):
                            fr[c] = nv

    dirty = [set(fr.keys()) for fr in F]
    rounds = 0
    while True:
        rounds += 1
        any_change = False
        while any(dirty):
            cur = dirty
            dirty = [set() for _ in range(NR)]
            if with_rsub:
                for r, s in rclos_pairs:
                    for c in cur[r]:
                        v = F[s].get(c)
                        src = F[r][c]
                        if v is None:
                            F[s][c] = set(src)
                            dirty[s].add(c)
                            any_change = True
                        else:
                            nv = v | src
                            if len(nv) != len(v):
                                F[s][c] = nv
                                dirty[s].add(c)
                                any_change = True
        if with_rcomp:
            for r1, r2, r3 in rcomp:
                F2 = F[r2]
                for c, m in list(F[r1].items()):
                    acc = set()
                    for f in m:
                        if f == TOP:
                            continue
                        v2 = F2.get(f)
                        if v2:
                            acc |= v2
                    if acc:
                        v3 = F[r3].get(c)
                        if v3 is None:
                            F[r3][c] = set(acc)
                            dirty[r3].add(c)
                            any_change = True
                        else:
                            nv = v3 | acc
                            if len(nv) != len(v3):
                                F[r3][c] = nv
                                dirty[r3].add(c)
                                any_change = True
        if not any_change:
            break

    role_edges_atom = sum(len(m) for fr in F for m in fr.values())

    # conflicts — endpoint index stays small (ChEBI has 0 disj)
    eps = sorted({x for pr in disj for x in pr})
    ep_idx = {e: k for k, e in enumerate(eps)}
    pb = [0] * len(eps)
    for a, b in disj:
        pb[ep_idx[a]] |= 1 << ep_idx[b]
        pb[ep_idx[b]] |= 1 << ep_idx[a]
    epm = [0] * H
    for e in eps:
        bit_ep = 1 << ep_idx[e]
        for c in range(H):
            if e in anc[c]:
                epm[c] |= bit_ep
    actors = []
    pm = [0] * H
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
            "ex_targets": role_edges_atom,
            "conf": n_conf, "rounds": rounds}


def run_mirror_auto(name, tbox):
    """bitmask for small H; sparse sets for ChEBI-scale H."""
    h, nr, sub2, exsub2, disj2, rsub2, rcomp2 = tbox
    reduce = sparse_reduce if h >= SPARSE_H_THRESHOLD else None
    if reduce is None:
        return run_mirror(name, tbox)

    import time as _time
    t0 = _time.time()
    full = sparse_reduce(h, nr, sub2, exsub2, disj2, rsub2, rcomp2)
    print(f"[{name}] full (sparse): atomic_edges={full['atomic_edges']} "
          f"role_edges_atom={full['role_edges_atom']} "
          f"conf={full['conf']} rounds={full['rounds']} "
          f"[{_time.time() - t0:.1f}s]")
    t0 = _time.time()
    norc = sparse_reduce(h, nr, sub2, exsub2, disj2, rsub2, rcomp2,
                         with_rcomp=False)
    nors = sparse_reduce(h, nr, sub2, exsub2, disj2, rsub2, rcomp2,
                         with_rsub=False)
    print(f"[{name}] ablations: no-roleComp={norc['role_edges_atom']} "
          f"no-roleSub={nors['role_edges_atom']} "
          f"[{_time.time() - t0:.1f}s]")
    # grouped-conf uses bitmask anc; for sparse rebuild set-based actors
    # already counted conf above — re-assert via grouped_conf only when
    # disj is small enough that bitmask anc fits (disj endpoints << H).
    if disj2 and h < SPARSE_H_THRESHOLD:
        order, parents = topo_order(h, sub2)
        anc = [(1 << c) | (1 << h) for c in range(h)] + [1 << h]
        for c in order:
            m = anc[c]
            for p in parents[c]:
                m |= anc[p]
            anc[c] = m
        gconf, n_actors = grouped_conf(anc, disj2, h)
        if gconf != full["conf"]:
            sys.exit(f"[{name}] MIRROR FAILED: grouped conf {gconf} != "
                     f"scan conf {full['conf']}")
        print(f"[{name}] grouped-conf cross-check OK ({gconf} conflicts, "
              f"{n_actors} actors)")
    else:
        print(f"[{name}] conf={full['conf']} (sparse scan; "
              f"grouped-conf skipped for H>={SPARSE_H_THRESHOLD} or empty disj)")
    nep = len({x for pr in disj2 for x in pr})
    return {"h": h, "nr": nr, "nsub": len(sub2), "nex": len(exsub2),
            "ndj": len(disj2), "nrs": len(rsub2), "nch": len(rcomp2),
            "nep": nep,
            "atomic_edges": full["atomic_edges"],
            "role_edges_atom": full["role_edges_atom"],
            "conf": full["conf"], "rounds": full["rounds"],
            "no_rc": norc["role_edges_atom"],
            "no_rs": nors["role_edges_atom"]}


def capacity_report(name, m, tbox):
    """Print the measured numbers the hand-written driver's compile-time
    capacities must cover (plus the 2-word endpoint-mask bound)."""
    (_h, nr, _sub2, _exsub2, disj2, _rsub2, rcomp2) = tbox
    nep = len({x for pr in disj2 for x in pr})
    per_r1 = {}
    for r1, _r2, _r3 in rcomp2:
        per_r1[r1] = per_r1.get(r1, 0) + 1
    kmax = max(per_r1.values()) if per_r1 else 0
    print(f"[{name}] CAPACITY REPORT: H={m['h']} NR={nr} "
          f"NSUB={m['nsub']} NEX={m['nex']} NDJ={m['ndj']} "
          f"NRS={m['nrs']} NCH={m['nch']} NEP={nep} "
          f"KMAX_USED={kmax} "
          f"WC={(m['h'] + 1 + 63) // 64}")
    if nep > 120:
        sys.exit(f"[{name}] FAILED: {nep} distinct disjointness endpoints "
                 f"> 120 — the driver's 2-word endpoint masks (128 bits) "
                 f"do not fit; generalise to NEPW words (round-13 "
                 f"ubo-style) before sizing the driver")
    if kmax > 8:
        sys.exit(f"[{name}] FAILED: {kmax} roleComp chains share one r1 "
                 f"> KMAX=8; bump the driver's KMAX/LPC sizing")
    # amplification factor (role edges per stated restriction), the
    # cross-ontology comparison metric of rounds 12/13
    if m["nex"] > 0:
        print(f"[{name}] amplification: role_edges/exsub = "
              f"{m['role_edges_atom'] / m['nex']:.1f}x "
              f"(roleComp contrib "
              f"{m['role_edges_atom'] - m['no_rc']} = "
              f"{100 * (m['role_edges_atom'] - m['no_rc']) / max(m['role_edges_atom'], 1):.0f}%, "
              f"roleSub contrib "
              f"{m['role_edges_atom'] - m['no_rs']} = "
              f"{100 * (m['role_edges_atom'] - m['no_rs']) / max(m['role_edges_atom'], 1):.0f}%)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None,
                    help="run a single target (chebi|pato)")
    args = ap.parse_args()

    for name, ns, owl in OBO_ONTS:
        if args.only is not None and name != args.only:
            continue
        t0 = time.time()
        stats, tbox, labels_out, role_labels_out, iris_out, \
            role_iris_out = extract_obo(name, ns, owl, RO)
        print(f"[{name}] extraction done [{time.time() - t0:.1f}s]; "
              f"super_side={stats['super_side']} "
              f"equiv_restr={stats['equiv_restr']} "
              f"deprecated_skipped={stats['skipped_deprecated']}")
        m = run_mirror_auto(name, tbox)
        capacity_report(name, m, tbox)
        st2 = dict(m)
        st2.update({"sub2": tbox[2], "exsub2": tbox[3], "disj2": tbox[4],
                    "rsub2": tbox[5], "rcomp2": tbox[6]})
        emit_packed(f"{name}_packed.txt", st2)
        emit_tsvs(name, tbox, labels_out, role_labels_out, iris_out,
                  role_iris_out)
        print(f"[{name}] target done [{time.time() - t0:.1f}s]")

    print("gen_chebi_data: OK")


if __name__ == "__main__":
    main()
