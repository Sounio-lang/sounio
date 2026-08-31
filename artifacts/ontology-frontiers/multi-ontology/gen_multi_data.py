#!/usr/bin/env python3
"""gen_multi_data.py — round 13: EL+ role-aware boolean closure on
MULTIPLE GO roots (the three top-level cones of GO) and on additional
OBO ontologies (CL, UBERON).  Python bitmask mirror + packed-data and
driver emitter.

Reuses the validated round 11/12 machinery unchanged:
  - extract_tbox.parse_go / parse_ro   (rounds 11-12 OWL extraction)
  - gen_elplus_data.load_go            (round-12 tbox text format)
  - gen_go_full_data.bitmask_reduce    (round-12 bitmask fixpoint mirror,
                                        validated against the general
                                        set-based fixpoint on the round-11
                                        slice, incl. ablations)

Targets:
  go_bp    descendant cone of GO:0008150 (biological_process)
  go_cc    descendant cone of GO:0005575 (cellular_component)
  go_mf    descendant cone of GO:0003674 (molecular_function)
  cl       CL-namespace classes of cl.owl (+ RO role axioms)
  uberon   UBERON-namespace classes of uberon.owl (+ RO role axioms)

GO-cone policy: cones are cut from the ROUND-12 full-GO extraction
(../real-data/go_full_elplus_tbox.txt), not from a re-parse of go-plus.
A cone keeps the sub/exsub/disj axioms with both endpoints in the cone;
the role set is the roles used by the kept restrictions, RO-closed
(superproperties + composition targets), exactly the round-11/12 policy.
The three cones are verified to PARTITION the 38,245 full-GO classes
(pairwise disjoint, union = all classes), so the round-12 full-GO numbers
must decompose exactly:
    sum(cone atomic_edges)    == 395,939      (round-12 full value)
    sum(cone role_edges_atom) == 2,135,207    (requires 0 cross-cone
                                               stated restrictions; the
                                               generator measures them)
    sum(cone conf) + cross    == 792,814,846  (cross = ordered conflict
                                               pairs across cones)
All three identities are ASSERTED (the full-GO side is recomputed here
from the cone ancestor masks with an independent grouped counter, not
taken on faith from the round-12 log).

OBO policy (cl / uberon): same shape as the round-12 GO-only policy —
classes/parents/fillers/disjoint partners restricted to the ontology's
own namespace (/CL_ / /UBERON_), owl:deprecated classes excluded, role
axioms from ro.owl, role set RO-closed.  Superclass-side restrictions
(ex r.F sq C) are PROBED and reported (go-plus had 0; CL and UBERON have
1 each — the same axiom, ex RO:0000053.PATO:0010006 sq CL:0000000).
CAUTION (math-review 2026-08-05): such axioms CAN induce atom-atom
subsumptions in general (transitivity through an existential node), so
the profile theorem of rounds 11/12 does NOT cover them; here they are
provably inert under the namespace-only policy (the PATO filler is never
interned; for UBERON the CL:0000000 target is out-of-namespace too), so
the reported numbers are exact for the extracted TBox.

Grouped conflict counter (independent cross-check): pm[c] is a FUNCTION
of epm[c] (pm = union of partner bits over the set bits of epm), so
classes with equal endpoint masks contribute uniformly:
    conf = sum_{v1} sum_{v2 : pm(v1)&v2} n(v1)*n(v2) - sum_{v : pm(v)&v} n(v)
over DISTINCT endpoint masks v.  This agrees with the O(nact^2) counter
inside bitmask_reduce on every target (asserted) and recomputes the
round-12 full-GO conflict total from the cone masks alone.

Outputs (all in this directory):
  <target>_packed.txt               runtime data for the Sounio drivers
                                    (13-int header: 7 axiom counts +
                                     6 mirror values; then axiom lines)
  <target>_classes.tsv, _roles.tsv, _elplus_tbox.txt
                                    documentation/extraction record
  go_roots_elplus_driver.sio        3 GO cones, one run
  obo_elplus_driver.sio             CL + UBERON, one run

Driver model: same bitmask reduction as go_full_elplus_driver.sio
(round 12), but the expected mirror values travel INSIDE the packed
file header, so the drivers need no generated expected_*.sio modules.
Ablations (no-roleComp / no-roleSub) are re-run in the drivers and
checked against the header values, as in round 12.

Run from this directory:
  python3 gen_multi_data.py              # everything
  python3 gen_multi_data.py --go-only    # GO cones only (no OWL parse)
"""

import argparse
import sys
import time

sys.path.insert(0, "../real-data/scale")
sys.path.insert(0, "../real-data")

from gen_go_full_data import bitmask_reduce  # noqa: E402
from gen_elplus_data import load_go  # noqa: E402
from extract_tbox import parse_go, parse_ro  # noqa: E402

FULL_GO = {"h": 38245, "atomic_edges": 395939, "role_edges_atom": 2135207,
           "conf": 792814846}

GO_ROOTS = [
    ("go_bp", "GO:0008150", "biological_process"),
    ("go_cc", "GO:0005575", "cellular_component"),
    ("go_mf", "GO:0003674", "molecular_function"),
]

OBO_ONTS = [
    ("cl", "CL", "downloads/cl.owl"),
    ("uberon", "UBERON", "downloads/uberon.owl"),
]


# ── grouped conflict counter (independent of bitmask_reduce's O(nact^2)
#    loop) ──────────────────────────────────────────────────────────────

def grouped_conf(anc, disj, H):
    """Ordered conflict pairs over atom ancestors, counted via distinct
    endpoint masks.  anc[c]: ancestor bitmask (bit H = top allowed);
    disj: [(a, b)] class-id pairs.  Returns (conf, n_actors)."""
    eps = sorted({x for pr in disj for x in pr})
    ep_idx = {e: k for k, e in enumerate(eps)}
    pb = [0] * len(eps)
    for a, b in disj:
        pb[ep_idx[a]] |= 1 << ep_idx[b]
        pb[ep_idx[b]] |= 1 << ep_idx[a]
    # endpoint mask per class, grouped
    groups = {}
    ep_bits = [(1 << e, 1 << ep_idx[e]) for e in eps]
    for c in range(H):
        a = anc[c]
        m = 0
        for bit, epbit in ep_bits:
            if a & bit:
                m |= epbit
        groups[m] = groups.get(m, 0) + 1
    # pm is a function of the endpoint mask
    def pm_of(m):
        p, k = 0, 0
        while m:
            if m & 1:
                p |= pb[k]
            m >>= 1
            k += 1
        return p
    items = [(m, pm_of(m), n) for m, n in groups.items()]
    conf = 0
    for m1, p1, n1 in items:
        for m2, _p2, n2 in items:
            if p1 & m2:
                conf += n1 * n2
        if p1 & m1:
            conf -= n1  # exclude the diagonal c1 == c2
    n_actors = sum(n for m, n in groups.items() if m or pm_of(m))
    return conf, n_actors


# ── GO cone slicing ──────────────────────────────────────────────────────

def find_go_root_ids():
    """Map the three root IRIs to class ids in the round-12 full-GO
    extraction."""
    want = {"GO_0008150": None, "GO_0005575": None, "GO_0003674": None}
    with open("../real-data/go_full_classes.tsv") as f:
        next(f)
        for line in f:
            p = line.rstrip("\n").split("\t")
            for suffix in want:
                if p[1].endswith("/" + suffix):
                    want[suffix] = int(p[0])
    if any(v is None for v in want.values()):
        sys.exit(f"GO ROOT LOOKUP FAILED: {want}")
    return want


def slice_cone(H, NR, sub, exsub, disj, rsub, rcomp, root):
    """Descendant cone of root (incl. root) with axioms restricted to the
    cone and the role set RO-closed.  Returns (stats, tbox) where tbox =
    (h, nr, sub2, exsub2, disj2, rsub2, rcomp2, id_map, role_map)."""
    children = {}
    for c, p in sub:
        children.setdefault(p, []).append(c)
    desc = set()
    stack = [root]
    while stack:
        c = stack.pop()
        if c in desc:
            continue
        desc.add(c)
        stack.extend(children.get(c, ()))
    # integrity measures (reported, and asserted by the caller)
    orphan_parents = sum(1 for c, p in sub if c in desc and p not in desc)
    id_map = {c: i for i, c in enumerate(sorted(desc))}
    h = len(desc)
    sub2 = sorted({(id_map[c], id_map[p]) for c, p in sub
                   if c in desc and p in desc})
    cross_ex = [1 for c, _r, f in exsub if c in desc and f not in desc]
    exsub_kept = [(id_map[c], r, id_map[f]) for c, r, f in exsub
                  if c in desc and f in desc]
    disj2 = sorted({(id_map[a], id_map[b]) for a, b in disj
                    if a in desc and b in desc})
    cross_disj = sum(1 for a, b in disj
                     if (a in desc) != (b in desc))
    # roles: used by kept restrictions, then RO-closed (round-11 policy)
    used = {r for _c, r, _f in exsub_kept}
    keep = set(used)
    changed = True
    while changed:
        changed = False
        for r, s in rsub:
            if r in keep and s not in keep:
                keep.add(s)
                changed = True
        for r1, r2, t in rcomp:
            if r1 in keep and r2 in keep and t not in keep:
                keep.add(t)
                changed = True
    role_map = {r: i for i, r in enumerate(sorted(keep))}
    nr = len(role_map)
    exsub2 = sorted({(c, role_map[r], f) for c, r, f in exsub_kept})
    rsub2 = sorted({(role_map[r], role_map[s]) for r, s in rsub
                    if r in keep and s in keep})
    rcomp2 = sorted({(role_map[r1], role_map[r2], role_map[t])
                     for r1, r2, t in rcomp
                     if r1 in keep and r2 in keep and t in keep})
    stats = {"h": h, "nr": nr, "orphan_parents": orphan_parents,
             "cross_exsub": len(cross_ex), "cross_disj": cross_disj,
             "desc": desc}
    return stats, (h, nr, sub2, exsub2, disj2, rsub2, rcomp2,
                   id_map, role_map)


# ── OBO extraction (round-12 policy, parameterised namespace) ────────────

def probe_obo_shape(path):
    """Count superclass-side restrictions and equivalentClass restrictions
    (both skipped by the syntactic extraction; reported for honesty)."""
    import xml.etree.ElementTree as ET
    OWL = "{http://www.w3.org/2002/07/owl#}"
    RDFS = "{http://www.w3.org/2000/01/rdf-schema#}"
    RDF = "{http://www.w3.org/1999/02/22-rdf-syntax-ns#}"
    n_super_side = 0
    n_equiv_restr = 0
    for _ev, el in ET.iterparse(path):
        if el.tag == OWL + "Restriction":
            for ch in el:
                if ch.tag == RDFS + "subClassOf" and \
                        ch.get(RDF + "resource") is not None:
                    n_super_side += 1
            el.clear()
        elif el.tag == OWL + "Class":
            has_equiv_restr = False
            for ch in el:
                if ch.tag == OWL + "equivalentClass":
                    for se in ch:
                        if se.tag == OWL + "Restriction":
                            has_equiv_restr = True
            if has_equiv_restr:
                n_equiv_restr += 1
            el.clear()
    return n_super_side, n_equiv_restr


def extract_obo(name, ns, owl_path, ro_path, policy="ns_only"):
    """OBO syntactic extraction with RO-closed roles.

    policy:
      ns_only       — round-12/13: classes/parents/fillers/disj partners
                      restricted to `/{ns}_` (default; prior receipts).
      open_fillers  — primary-NS classes PLUS any parent/filler/disj partner
                      of a primary-NS subject, closed under superclasses of
                      everything kept.  Measures the axioms namespace-only
                      dropped (cross-namespace fillers / parents).

    Returns (stats, tbox, labels_out, role_labels_out, iris_out, role_iris_out)
    in the same shape as slice_cone.
    """
    if policy not in ("ns_only", "open_fillers"):
        sys.exit(f"[{name}] unknown extract policy {policy!r}")
    print(f"[{name}] parsing {owl_path} (policy={policy}) ...")
    t0 = time.time()
    order, labels, sub, exsub, disj, onto_role_labels, gstats = \
        parse_go(owl_path, skip_deprecated=True)
    print(f"[{name}] declared classes: {len(order)} (deprecated skipped: "
          f"{gstats['skipped_deprecated_class']}), sub={len(sub)} "
          f"exsub={len(exsub)} disj={len(disj)} "
          f"(skipped restr_shape={gstats['skipped_restr_shape']}, "
          f"anon_subclassof={gstats['skipped_anon_subclassof']}) "
          f"[{time.time() - t0:.1f}s]")

    primary = {iri for iri in order if f"/{ns}_" in iri}
    n_primary = len(primary)
    # axioms with primary subject but foreign parent/filler (the ns_only
    # drop set — measured even when policy=ns_only, for honesty).
    foreign_parent = sum(1 for c, p in sub if c in primary and p not in primary)
    foreign_filler = sum(1 for c, _r, f in exsub
                         if c in primary and f not in primary)
    foreign_disj = sum(1 for a, b in disj
                       if (a in primary) ^ (b in primary))

    if policy == "ns_only":
        keep = set(primary)
    else:
        keep = set(primary)
        for c, p in sub:
            if c in primary:
                keep.add(p)
        for c, _r, f in exsub:
            if c in primary:
                keep.add(f)
        for a, b in disj:
            if a in primary or b in primary:
                keep.add(a)
                keep.add(b)
        # close under superclasses of kept classes (complete ancestor rows)
        changed = True
        while changed:
            changed = False
            for c, p in sub:
                if c in keep and p not in keep:
                    keep.add(p)
                    changed = True

    keep_iris = [iri for iri in order if iri in keep]
    inset = set(keep_iris)
    n_foreign = sum(1 for iri in keep_iris if f"/{ns}_" not in iri)
    id_map = {iri: i for i, iri in enumerate(keep_iris)}
    h = len(keep_iris)
    sub2 = sorted({(id_map[c], id_map[p]) for c, p in sub
                   if c in inset and p in inset})
    disj2 = sorted({(id_map[a], id_map[b]) for a, b in disj
                    if a in inset and b in inset})
    exsub_kept = [(id_map[c], r, id_map[f]) for c, r, f in exsub
                  if c in inset and f in inset]
    print(f"[{name}] {ns}/{policy}: H={h} (primary={n_primary}, "
          f"foreign_interned={n_foreign}), sub={len(sub2)}, "
          f"exsub={len(exsub_kept)}, disj={len(disj2)}; "
          f"ns_only would drop foreign_parent={foreign_parent} "
          f"foreign_filler={foreign_filler} foreign_disj_half={foreign_disj}")

    t0 = time.time()
    rsub, chains, ro_role_labels, rstats = parse_ro(ro_path)
    print(f"[{name}] ro: {len(rsub)} subPropertyOf, {len(chains)} chains "
          f"(incl. transitive) [{time.time() - t0:.1f}s]")

    used = {r for _c, r, _f in exsub_kept}
    keep = set(used)
    changed = True
    while changed:
        changed = False
        for r, s in rsub:
            if r in keep and s not in keep:
                keep.add(s)
                changed = True
        for r1, r2, t in chains:
            if r1 in keep and r2 in keep and t not in keep:
                keep.add(t)
                changed = True
    role_list = sorted(keep)
    role_map = {r: i for i, r in enumerate(role_list)}
    nr = len(role_map)
    if nr == 0:
        sys.exit(f"[{name}] FAILED: no roles used; extraction bug")
    exsub2 = sorted({(c, role_map[r], f) for c, r, f in exsub_kept})
    rsub2 = sorted({(role_map[r], role_map[s]) for r, s in rsub
                    if r in keep and s in keep})
    rcomp2 = sorted({(role_map[r1], role_map[r2], role_map[t])
                     for r1, r2, t in chains
                     if r1 in keep and r2 in keep and t in keep})
    print(f"[{name}] H={h} NR={nr} sub={len(sub2)} exsub={len(exsub2)} "
          f"disj={len(disj2)} roleSub={len(rsub2)} roleComp={len(rcomp2)}")

    n_super_side, n_equiv_restr = probe_obo_shape(owl_path)
    print(f"[{name}] probe: superclass-side restrictions="
          f"{n_super_side}, equivalentClass restrictions="
          f"{n_equiv_restr} (both skipped, reported for honesty)")

    stats = {"h": h, "nr": nr, "super_side": n_super_side,
             "equiv_restr": n_equiv_restr,
             "skipped_deprecated": gstats["skipped_deprecated_class"],
             "policy": policy, "n_primary": n_primary,
             "n_foreign": n_foreign,
             "foreign_parent": foreign_parent,
             "foreign_filler": foreign_filler,
             "foreign_disj": foreign_disj}
    labels_out = {id_map[iri]: labels.get(iri, "") for iri in keep_iris}
    role_labels_out = {role_map[r]: (ro_role_labels.get(r) or
                                     onto_role_labels.get(r, ""))
                       for r in role_list}
    iris_out = {id_map[iri]: iri for iri in keep_iris}
    role_iris_out = {role_map[r]: r for r in role_list}
    tbox = (h, nr, sub2, exsub2, disj2, rsub2, rcomp2)
    return stats, tbox, labels_out, role_labels_out, iris_out, role_iris_out


# ── mirror runner + emitters ─────────────────────────────────────────────

def run_mirror(name, tbox):
    """bitmask_reduce full + ablations + grouped-conf cross-check.
    Returns the stats dict."""
    h, nr, sub2, exsub2, disj2, rsub2, rcomp2 = tbox
    t0 = time.time()
    full = bitmask_reduce(h, nr, sub2, exsub2, disj2, rsub2, rcomp2)
    print(f"[{name}] full: atomic_edges={full['atomic_edges']} "
          f"role_edges_atom={full['role_edges_atom']} "
          f"conf={full['conf']} rounds={full['rounds']} "
          f"[{time.time() - t0:.1f}s]")
    t0 = time.time()
    norc = bitmask_reduce(h, nr, sub2, exsub2, disj2, rsub2, rcomp2,
                          with_rcomp=False)
    nors = bitmask_reduce(h, nr, sub2, exsub2, disj2, rsub2, rcomp2,
                          with_rsub=False)
    print(f"[{name}] ablations: no-roleComp={norc['role_edges_atom']} "
          f"no-roleSub={nors['role_edges_atom']} "
          f"[{time.time() - t0:.1f}s]")
    # independent grouped-conf cross-check (needs the ancestor masks;
    # rebuild them exactly as bitmask_reduce does)
    from gen_go_full_data import topo_order
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
    nep = len({x for pr in disj2 for x in pr})
    return {"h": h, "nr": nr, "nsub": len(sub2), "nex": len(exsub2),
            "ndj": len(disj2), "nrs": len(rsub2), "nch": len(rcomp2),
            "nep": nep,
            "atomic_edges": full["atomic_edges"],
            "role_edges_atom": full["role_edges_atom"],
            "conf": full["conf"], "rounds": full["rounds"],
            "no_rc": norc["role_edges_atom"],
            "no_rs": nors["role_edges_atom"]}


def emit_packed(path, st):
    """13-int header (7 axiom counts + 6 mirror values) + axiom lines."""
    with open(path, "w") as f:
        f.write(f"{st['h']} {st['nr']} {st['nsub']} {st['nex']} "
                f"{st['ndj']} {st['nrs']} {st['nch']} "
                f"{st['atomic_edges']} {st['role_edges_atom']} "
                f"{st['conf']} {st['rounds']} {st['no_rc']} {st['no_rs']}\n")
        for c, p in st["sub2"]:
            f.write(f"s {c} {p}\n")
        for c, r, fl in st["exsub2"]:
            f.write(f"x {c} {r} {fl}\n")
        for a, b in st["disj2"]:
            f.write(f"d {a} {b}\n")
        for r, s in st["rsub2"]:
            f.write(f"h {r} {s}\n")
        for r1, r2, r3 in st["rcomp2"]:
            f.write(f"k {r1} {r2} {r3}\n")
    print(f"  wrote {path}")


def emit_tsvs(name, tbox_extra, labels_out=None, role_labels_out=None,
              iris_out=None, role_iris_out=None):
    (h, nr, sub2, exsub2, disj2, rsub2, rcomp2) = tbox_extra
    with open(f"{name}_elplus_tbox.txt", "w") as f:
        f.write(f"# classes_go {h}\n# roles_go {nr}\n# sub_go {len(sub2)}\n"
                f"# exsub_go {len(exsub2)}\n# disj_go {len(disj2)}\n"
                f"# rolesub_go {len(rsub2)}\n# rolecomp_go {len(rcomp2)}\n")
        for c, p in sub2:
            f.write(f"sub go {c} {p}\n")
        for c, r, fl in exsub2:
            f.write(f"exsub go {c} {r} {fl}\n")
        for a, b in disj2:
            f.write(f"disj go {a} {b}\n")
        for r, s in rsub2:
            f.write(f"roleSub go {r} {s}\n")
        for r1, r2, r3 in rcomp2:
            f.write(f"roleComp go {r1} {r2} {r3}\n")
    if iris_out is not None:
        with open(f"{name}_classes.tsv", "w") as f:
            f.write("id\tiri\tlabel\n")
            for i in range(h):
                f.write(f"{i}\t{iris_out[i]}\t{labels_out.get(i, '')}\n")
        with open(f"{name}_roles.tsv", "w") as f:
            f.write("id\tiri\tlabel\n")
            for i in range(nr):
                f.write(f"{i}\t{role_iris_out[i]}\t"
                        f"{role_labels_out.get(i, '')}\n")


# ── Sounio driver template (round-12 bitmask driver, multi-target) ───────
# Tokens: @@TITLE@@, @@HC@@, @@NRC@@, @@WC@@, @@SUBC@@, @@EXC@@, @@DJC@@,
# @@RSC@@, @@RCC@@, @@EPC@@, @@FMN@@ (NRC*HC*WC), @@DCN@@ (NRC*HC),
# @@ANCN@@ (HC*WC), @@TARGETS@@ (fn definitions), @@CALLS@@ (main body).

DRIVER_TEMPLATE = r"""//@ run-pass
//@ expect-stdout: ALL PASS
// @@TITLE@@
//
// Round 13.  Executable mirror of the verified saturation engine of
// formal/OntologyELPlusClosureComplete.lean (crStep / closeSatF), same
// bitmask reduction as the round-12 full-GO driver
// (../real-data/scale/go_full_elplus_driver.sio), generalised to a
// MULTI-TARGET run: each target's TBox is loaded at RUNTIME from a
// packed text file whose 13-int header carries the 7 axiom counts AND
// the 6 python-mirror values (atomic edges, role edges, conflicts,
// rounds, 2 ablations) — the round-12 separate expected_*.sio module is
// replaced by self-validating data files.
//
// Per target: role-hierarchy closure, Kahn topological order, ancestor
// bitmasks (one topo pass), role-edge seeding + ancestor expansion
// (folds Rmono / transitivity+stoR), hybrid roleSub-worklist /
// roleComp-full-scan fixpoint (round 12b/c corrections), atom-level
// conflict count over endpoint masks, and the two ablation re-runs —
// every number checked against the mirror header.
//
// Data + mirror: gen_multi_data.py (this directory).  Run from the repo
// root (the gate does): the read_file paths below are repo-relative.

// ── capacities (compile-time; sized to the largest target + headroom) ──
const HC: i64 = @@HC@@
const NRC: i64 = @@NRC@@
const WC: i64 = @@WC@@
const SUBC: i64 = @@SUBC@@
const EXC: i64 = @@EXC@@
const DJC: i64 = @@DJC@@
const RSC: i64 = @@RSC@@
const RCC: i64 = @@RCC@@
const EPC: i64 = @@EPC@@
const NEPW: i64 = @@NEPW@@
const FMN: i64 = @@FMN@@
const DCN: i64 = @@DCN@@
const ANCN: i64 = @@ANCN@@

// ── module-level workspace (mutated by own-module fns only — the &!-
//    boundary miscompile of rounds 9-12) ─────────────────────────────────
pub var anc: [i64; @@ANCN@@] = [0; @@ANCN@@]
pub var fm: [i64; @@FMN@@] = [0; @@FMN@@]
pub var cur_dirty: [bool; @@DCN@@] = [false; @@DCN@@]
pub var nxt_dirty: [bool; @@DCN@@] = [false; @@DCN@@]
pub var f_nonempty: [bool; @@DCN@@] = [false; @@DCN@@]
pub var sub_c: [i64; @@SUBC@@] = [0; @@SUBC@@]
pub var sub_p: [i64; @@SUBC@@] = [0; @@SUBC@@]
pub var ex_c: [i64; @@EXC@@] = [0; @@EXC@@]
pub var ex_r: [i64; @@EXC@@] = [0; @@EXC@@]
pub var ex_f: [i64; @@EXC@@] = [0; @@EXC@@]
pub var dj_a: [i64; @@DJC@@] = [0; @@DJC@@]
pub var dj_b: [i64; @@DJC@@] = [0; @@DJC@@]
pub var rs_r: [i64; @@RSC@@] = [0; @@RSC@@]
pub var rs_s: [i64; @@RSC@@] = [0; @@RSC@@]
pub var ch1: [i64; @@RCC@@] = [0; @@RCC@@]
pub var ch2: [i64; @@RCC@@] = [0; @@RCC@@]
pub var ch3: [i64; @@RCC@@] = [0; @@RCC@@]
pub var rclos: [bool; @@RCN@@] = [false; @@RCN@@]
pub var rc_r: [i64; @@RCN@@] = [0; @@RCN@@]
pub var rc_s: [i64; @@RCN@@] = [0; @@RCN@@]
pub var indeg: [i64; @@HC@@] = [0; @@HC@@]
pub var order: [i64; @@HC@@] = [0; @@HC@@]
pub var queue: [i64; @@HC@@] = [0; @@HC@@]
pub var coff: [i64; @@HC1@@] = [0; @@HC1@@]
pub var clist: [i64; @@SUBC@@] = [0; @@SUBC@@]
pub var poff: [i64; @@HC1@@] = [0; @@HC1@@]
pub var plist: [i64; @@SUBC@@] = [0; @@SUBC@@]
pub var csr_cnt: [i64; @@HC@@] = [0; @@HC@@]
pub var epm: [i64; @@EPMN@@] = [0; @@EPMN@@]
pub var pm: [i64; @@EPMN@@] = [0; @@EPMN@@]
pub var actors: [i64; @@HC@@] = [0; @@HC@@]
pub var ep_id: [i64; @@HC@@] = [0; @@HC@@]
pub var acc: [i64; @@WC@@] = [0; @@WC@@]

pub var g_pos: i64 = 0
pub var g_len: i64 = 0

// ── bit helpers ──────────────────────────────────────────────────────────

fn ctz64(x: i64) -> i64 with Div {
    var n: i64 = 0
    var v = x
    if (v & 4294967295) == 0 { n = n + 32; v = v / 4294967296 }
    if (v & 65535) == 0 { n = n + 16; v = v / 65536 }
    if (v & 255) == 0 { n = n + 8; v = v / 256 }
    if (v & 15) == 0 { n = n + 4; v = v / 16 }
    if (v & 3) == 0 { n = n + 2; v = v / 4 }
    if (v & 1) == 0 { n = n + 1 }
    return n
}

fn row_popcount(base: i64) -> i64 with Mut, Div, Panic {
    var n: i64 = 0
    var w: i64 = 0
    while w < WC {
        var m = anc[base + w]
        while m != 0 {
            n = n + 1
            m = m & (m - 1)
        }
        w = w + 1
    }
    return n
}

// ── runtime text parsing (read_file result; str_char_at access only) ─────

fn skip_ws(text: string) with Mut, Div, Panic {
    while g_pos < g_len {
        let c = str_char_at(text, g_pos) as i64
        if c != 32 && c != 10 && c != 13 && c != 9 {
            return
        }
        g_pos = g_pos + 1
    }
}

fn read_int(text: string) -> i64 with Mut, Div, Panic {
    skip_ws(text)
    var v: i64 = 0
    while g_pos < g_len {
        let c = str_char_at(text, g_pos) as i64
        if c < 48 || c > 57 {
            return v
        }
        v = v * 10 + (c - 48)
        g_pos = g_pos + 1
    }
    return v
}

fn read_kind(text: string) -> i64 with Mut, Div, Panic {
    skip_ws(text)
    if g_pos >= g_len {
        return 0
    }
    let c = str_char_at(text, g_pos) as i64
    g_pos = g_pos + 1
    return c
}

// ── fixpoint pieces (identical semantics to the round-12 driver) ────────

fn or_anc_into_f(r: i64, c: i64, f: i64, mark: i64) with Mut, Div, Panic {
    let dst = (r * HC + c) * WC
    let src = f * WC
    var w: i64 = 0
    while w < WC {
        let sv = anc[src + w]
        if sv != 0 {
            let dv = fm[dst + w]
            let nv = dv | sv
            if nv != dv {
                fm[dst + w] = nv
                f_nonempty[r * HC + c] = true
                if mark == 1 {
                    cur_dirty[r * HC + c] = true
                }
            }
        }
        w = w + 1
    }
}

fn or_f_into_f(r: i64, c: i64, p: i64, mark: i64) with Mut, Div, Panic {
    let dst = (r * HC + c) * WC
    let src = (r * HC + p) * WC
    var w: i64 = 0
    while w < WC {
        let sv = fm[src + w]
        if sv != 0 {
            let dv = fm[dst + w]
            let nv = dv | sv
            if nv != dv {
                fm[dst + w] = nv
                f_nonempty[r * HC + c] = true
                if mark == 1 {
                    cur_dirty[r * HC + c] = true
                }
            }
        }
        w = w + 1
    }
}

fn seed_and_expand(nex: i64, h: i64, nr: i64) with Mut, Div, Panic {
    var k: i64 = 0
    while k < nex {
        or_anc_into_f(ex_r[k], ex_c[k], ex_f[k], 1)
        k = k + 1
    }
    var i: i64 = 0
    var c: i64 = 0
    var t: i64 = 0
    var p: i64 = 0
    var r: i64 = 0
    while i < h {
        c = order[i]
        t = poff[c]
        while t < poff[c + 1] {
            p = plist[t]
            r = 0
            while r < nr {
                or_f_into_f(r, c, p, 1)
                r = r + 1
            }
            t = t + 1
        }
        i = i + 1
    }
}

// Hybrid roleSub-worklist / roleComp-full-scan fixpoint (rounds 12b/12c).
fn role_fixpoint(with_rsub: i64, with_rcomp: i64, nrp: i64, nch: i64, h: i64, nr: i64) -> i64 with Mut, Div, Panic {
    var rounds: i64 = 0
    var any = true
    var rsub_any = true
    var r: i64 = 0
    var c: i64 = 0
    var k: i64 = 0
    var s: i64 = 0
    var r2: i64 = 0
    var r3: i64 = 0
    var w: i64 = 0
    var m: i64 = 0
    var low: i64 = 0
    var f: i64 = 0
    var dst: i64 = 0
    var src: i64 = 0
    var cell: i64 = 0
    var changed_cell = false
    var i: i64 = 0
    while any {
        any = false
        rounds = rounds + 1
        rsub_any = true
        while rsub_any {
            rsub_any = false
            r = 0
            while r < nr {
                c = 0
                while c < h {
                    cell = r * HC + c
                    if cur_dirty[cell] {
                        cur_dirty[cell] = false
                        if with_rsub == 1 {
                            k = 0
                            while k < nrp {
                                if rc_r[k] == r {
                                    s = rc_s[k]
                                    src = (r * HC + c) * WC
                                    dst = (s * HC + c) * WC
                                    changed_cell = false
                                    w = 0
                                    while w < WC {
                                        let sv = fm[src + w]
                                        if sv != 0 {
                                            let dv = fm[dst + w]
                                            let nv = dv | sv
                                            if nv != dv {
                                                fm[dst + w] = nv
                                                changed_cell = true
                                            }
                                        }
                                        w = w + 1
                                    }
                                    if changed_cell {
                                        f_nonempty[s * HC + c] = true
                                        nxt_dirty[s * HC + c] = true
                                        rsub_any = true
                                        any = true
                                    }
                                }
                                k = k + 1
                            }
                        }
                    }
                    c = c + 1
                }
                r = r + 1
            }
            i = 0
            while i < DCN {
                cur_dirty[i] = nxt_dirty[i]
                nxt_dirty[i] = false
                i = i + 1
            }
        }
        if with_rcomp == 1 {
            r = 0
            while r < nr {
                c = 0
                while c < h {
                    cell = r * HC + c
                    if f_nonempty[cell] {
                        k = 0
                        while k < nch {
                            if ch1[k] == r {
                                r2 = ch2[k]
                                r3 = ch3[k]
                                w = 0
                                while w < WC {
                                    acc[w] = 0
                                    w = w + 1
                                }
                                src = (r * HC + c) * WC
                                w = 0
                                while w < WC {
                                    m = fm[src + w]
                                    while m != 0 {
                                        low = m & (0 - m)
                                        f = w * 64 + ctz64(low)
                                        m = m & (m - 1)
                                        let src2 = (r2 * HC + f) * WC
                                        var w2: i64 = 0
                                        while w2 < WC {
                                            let sv2 = fm[src2 + w2]
                                            if sv2 != 0 {
                                                acc[w2] = acc[w2] | sv2
                                            }
                                            w2 = w2 + 1
                                        }
                                    }
                                    w = w + 1
                                }
                                dst = (r3 * HC + c) * WC
                                changed_cell = false
                                w = 0
                                while w < WC {
                                    let av = acc[w]
                                    if av != 0 {
                                        let dv = fm[dst + w]
                                        let nv = dv | av
                                        if nv != dv {
                                            fm[dst + w] = nv
                                            changed_cell = true
                                        }
                                    }
                                    w = w + 1
                                }
                                if changed_cell {
                                    f_nonempty[r3 * HC + c] = true
                                    nxt_dirty[r3 * HC + c] = true
                                    any = true
                                }
                            }
                            k = k + 1
                        }
                    }
                    c = c + 1
                }
                r = r + 1
            }
        }
        // end-of-round dirty swap (round 12c fix)
        i = 0
        while i < DCN {
            cur_dirty[i] = nxt_dirty[i]
            nxt_dirty[i] = false
            i = i + 1
        }
    }
    return rounds
}

fn count_role_edges(h: i64, nr: i64) -> i64 with Mut, Div, Panic {
    var n: i64 = 0
    var r: i64 = 0
    var c: i64 = 0
    var w: i64 = 0
    var m: i64 = 0
    while r < nr {
        c = 0
        while c < h {
            let src = (r * HC + c) * WC
            w = 0
            while w < WC {
                m = fm[src + w]
                while m != 0 {
                    n = n + 1
                    m = m & (m - 1)
                }
                w = w + 1
            }
            c = c + 1
        }
        r = r + 1
    }
    return n
}

// full workspace reset between targets
fn clear_all() with Mut, Div, Panic {
    var i: i64 = 0
    while i < ANCN {
        anc[i] = 0
        i = i + 1
    }
    i = 0
    while i < FMN {
        fm[i] = 0
        i = i + 1
    }
    i = 0
    while i < DCN {
        cur_dirty[i] = false
        nxt_dirty[i] = false
        f_nonempty[i] = false
        i = i + 1
    }
    i = 0
    while i < @@RCN@@ {
        rclos[i] = false
        i = i + 1
    }
    i = 0
    while i < @@EPMN@@ {
        epm[i] = 0
        pm[i] = 0
        i = i + 1
    }
    i = 0
    while i < HC {
        ep_id[i] = 0
        i = i + 1
    }
}

// fm-only reset for the ablation re-runs (anc and the role tables stay)
fn clear_f() with Mut, Div, Panic {
    var i: i64 = 0
    while i < FMN {
        fm[i] = 0
        i = i + 1
    }
    i = 0
    while i < DCN {
        cur_dirty[i] = false
        nxt_dirty[i] = false
        f_nonempty[i] = false
        i = i + 1
    }
}

@@TARGETS@@

fn main() -> i32 with IO, Mut, Div, Panic {
    // fixup writes for the garbage leading cells of module-level splat
    // arrays (known compiler pitfall, rounds 6-12)
    anc[0] = 0
    anc[1] = 0
    anc[2] = 0
    fm[0] = 0
    fm[1] = 0
    fm[2] = 0
    cur_dirty[0] = false
    cur_dirty[1] = false
    cur_dirty[2] = false
    nxt_dirty[0] = false
    nxt_dirty[1] = false
    nxt_dirty[2] = false
    f_nonempty[0] = false
    f_nonempty[1] = false
    f_nonempty[2] = false
    rclos[0] = false
    rclos[1] = false
    rclos[2] = false
    epm[0] = 0
    epm[1] = 0
    epm[2] = 0
    pm[0] = 0
    pm[1] = 0
    pm[2] = 0
    ep_id[0] = 0
    ep_id[1] = 0
    ep_id[2] = 0

    var n_fail: i64 = 0
@@CALLS@@
    if n_fail == 0 {
        println("ALL PASS")
        return 0
    }
    println("FAILURES:")
    println(n_fail)
    return 1
}
"""

# Per-target function body.  Tokens: @@NAME@@ (fn suffix), @@PATH@@,
# @@LABEL@@ (printed name), @@ABLATIONS@@ (ablation block or "").
TARGET_TEMPLATE = r"""
fn run_@@NAME@@() -> i64 with IO, Mut, Div, Panic {
    var n_fail = 0
    clear_all()

    // ── runtime data load (bypasses the ~24k-statement wall) ──────────
    let text = read_file("@@PATH@@")
    g_len = str_len(text)
    g_pos = 0
    let H = read_int(text)
    let NR = read_int(text)
    let NSUB = read_int(text)
    let NEX = read_int(text)
    let NDJ = read_int(text)
    let NRS = read_int(text)
    let NCH = read_int(text)
    let E_ATOM = read_int(text)
    let E_ROLE = read_int(text)
    let E_CONF = read_int(text)
    let E_ROUNDS = read_int(text)
    let E_NORC = read_int(text)
    let E_NORS = read_int(text)
    if H + 1 > HC || NR > NRC || NSUB > SUBC || NEX > EXC || NDJ > DJC || NRS > RSC || NCH > RCC {
        println("FAIL: @@LABEL@@ data exceeds compile-time capacities")
        n_fail = n_fail + 1
    }
    var k: i64 = 0
    var kind: i64 = 0
    var isub: i64 = 0
    var iex: i64 = 0
    var idj: i64 = 0
    var irs: i64 = 0
    var ich: i64 = 0
    while g_pos < g_len {
        kind = read_kind(text)
        if kind == 115 {
            sub_c[isub] = read_int(text)
            sub_p[isub] = read_int(text)
            isub = isub + 1
        }
        if kind == 120 {
            ex_c[iex] = read_int(text)
            ex_r[iex] = read_int(text)
            ex_f[iex] = read_int(text)
            iex = iex + 1
        }
        if kind == 100 {
            dj_a[idj] = read_int(text)
            dj_b[idj] = read_int(text)
            idj = idj + 1
        }
        if kind == 104 {
            rs_r[irs] = read_int(text)
            rs_s[irs] = read_int(text)
            irs = irs + 1
        }
        if kind == 107 {
            ch1[ich] = read_int(text)
            ch2[ich] = read_int(text)
            ch3[ich] = read_int(text)
            ich = ich + 1
        }
    }
    if isub != NSUB || iex != NEX || idj != NDJ || irs != NRS || ich != NCH {
        println("FAIL: @@LABEL@@ parsed axiom counts disagree with header")
        n_fail = n_fail + 1
    }

    // ── role-hierarchy closure (reflexive + transitive) ───────────────
    var r: i64 = 0
    while r < NR {
        rclos[r * NRC + r] = true
        r = r + 1
    }
    k = 0
    while k < NRS {
        rclos[rs_r[k] * NRC + rs_s[k]] = true
        k = k + 1
    }
    var changed2 = true
    var a: i64 = 0
    var b: i64 = 0
    var d: i64 = 0
    while changed2 {
        changed2 = false
        a = 0
        while a < NR {
            b = 0
            while b < NR {
                if rclos[a * NRC + b] {
                    d = 0
                    while d < NR {
                        if rclos[b * NRC + d] && !rclos[a * NRC + d] {
                            rclos[a * NRC + d] = true
                            changed2 = true
                        }
                        d = d + 1
                    }
                }
                b = b + 1
            }
            a = a + 1
        }
    }
    var nrp: i64 = 0
    a = 0
    while a < NR {
        b = 0
        while b < NR {
            if a != b && rclos[a * NRC + b] {
                rc_r[nrp] = a
                rc_s[nrp] = b
                nrp = nrp + 1
            }
            b = b + 1
        }
        a = a + 1
    }

    // ── topological order (Kahn; the is_a relation is a DAG) ──────────
    var c: i64 = 0
    while c < H {
        indeg[c] = 0
        coff[c] = 0
        poff[c] = 0
        c = c + 1
    }
    coff[H] = 0
    poff[H] = 0
    k = 0
    while k < NSUB {
        indeg[sub_c[k]] = indeg[sub_c[k]] + 1
        coff[sub_p[k] + 1] = coff[sub_p[k] + 1] + 1
        poff[sub_c[k] + 1] = poff[sub_c[k] + 1] + 1
        k = k + 1
    }
    c = 0
    while c < H {
        coff[c + 1] = coff[c + 1] + coff[c]
        poff[c + 1] = poff[c + 1] + poff[c]
        c = c + 1
    }
    c = 0
    while c < H {
        csr_cnt[c] = coff[c]
        c = c + 1
    }
    k = 0
    while k < NSUB {
        clist[csr_cnt[sub_p[k]]] = sub_c[k]
        csr_cnt[sub_p[k]] = csr_cnt[sub_p[k]] + 1
        k = k + 1
    }
    c = 0
    while c < H {
        csr_cnt[c] = poff[c]
        c = c + 1
    }
    k = 0
    while k < NSUB {
        plist[csr_cnt[sub_c[k]]] = sub_p[k]
        csr_cnt[sub_c[k]] = csr_cnt[sub_c[k]] + 1
        k = k + 1
    }
    var head: i64 = 0
    var tail: i64 = 0
    c = 0
    while c < H {
        if indeg[c] == 0 {
            queue[tail] = c
            tail = tail + 1
        }
        c = c + 1
    }
    var nord: i64 = 0
    var t: i64 = 0
    var ch: i64 = 0
    while head < tail {
        c = queue[head]
        head = head + 1
        order[nord] = c
        nord = nord + 1
        t = coff[c]
        while t < coff[c + 1] {
            ch = clist[t]
            indeg[ch] = indeg[ch] - 1
            if indeg[ch] == 0 {
                queue[tail] = ch
                tail = tail + 1
            }
            t = t + 1
        }
    }
    if nord != H {
        println("FAIL: @@LABEL@@ sub relation has a cycle (Kahn incomplete)")
        n_fail = n_fail + 1
    }

    // ── atomic closure: ancestor masks in topological order ───────────
    let one: i64 = 1
    var i: i64 = 0
    var w: i64 = 0
    var p: i64 = 0
    var dst: i64 = 0
    var src: i64 = 0
    while i < H {
        c = order[i]
        dst = c * WC
        let cw = c / 64
        anc[dst + cw] = anc[dst + cw] | (one << (c - cw * 64))
        let tw = H / 64
        anc[dst + tw] = anc[dst + tw] | (one << (H - tw * 64))
        t = poff[c]
        while t < poff[c + 1] {
            p = plist[t]
            src = p * WC
            w = 0
            while w < WC {
                let sv = anc[src + w]
                if sv != 0 {
                    anc[dst + w] = anc[dst + w] | sv
                }
                w = w + 1
            }
            t = t + 1
        }
        i = i + 1
    }
    var atomic_edges: i64 = 0
    c = 0
    while c < H {
        atomic_edges = atomic_edges + row_popcount(c * WC) - 1
        c = c + 1
    }
    if atomic_edges != E_ATOM {
        println("FAIL: @@LABEL@@ atomic closure edges disagree with mirror")
        n_fail = n_fail + 1
    }

    // ── role-aware closure: seed + expand + role fixpoint ─────────────
    seed_and_expand(NEX, H, NR)
    let rounds = role_fixpoint(1, 1, nrp, NCH, H, NR)
    let r_atom = count_role_edges(H, NR)
    if r_atom != E_ROLE {
        println("FAIL: @@LABEL@@ atom-source role edges disagree with mirror")
        n_fail = n_fail + 1
    }
    if rounds != E_ROUNDS {
        println("FAIL: @@LABEL@@ role fixpoint rounds disagree with mirror")
        n_fail = n_fail + 1
    }

    // ── conflicts over atomic ancestors (profile theorem) ─────────────
    // endpoint masks epm[c*NEPW + w] (which disjointness endpoints are
    // ancestors of c) and partner masks pm (union of partner bits);
    // word-generalised (UBERON has ~600 endpoints, GO cones < 64).
    var nep: i64 = 0
    k = 0
    while k < NDJ {
        a = dj_a[k]
        b = dj_b[k]
        if ep_id[a] == 0 {
            nep = nep + 1
            ep_id[a] = nep
        }
        if ep_id[b] == 0 {
            nep = nep + 1
            ep_id[b] = nep
        }
        k = k + 1
    }
    if nep > EPC {
        println("FAIL: @@LABEL@@ too many disjointness endpoints")
        n_fail = n_fail + 1
    }
    k = 0
    while k < NDJ {
        a = dj_a[k]
        b = dj_b[k]
        let aw = a / 64
        let abit = one << (a - aw * 64)
        let bw = b / 64
        let bbit = one << (b - bw * 64)
        let ka = ep_id[a] - 1
        let kb = ep_id[b] - 1
        let kaw = ka / 64
        let kab = ka - kaw * 64
        let kbw = kb / 64
        let kbb = kb - kbw * 64
        c = 0
        while c < H {
            let cw = c * WC
            if (anc[cw + aw] & abit) != 0 {
                epm[c * NEPW + kaw] = epm[c * NEPW + kaw] | (one << kab)
            }
            if (anc[cw + bw] & bbit) != 0 {
                epm[c * NEPW + kbw] = epm[c * NEPW + kbw] | (one << kbb)
            }
            c = c + 1
        }
        k = k + 1
    }
    // partner masks: a class with endpoint a gets the partner bit of b
    // and vice versa
    k = 0
    while k < NDJ {
        let ka = ep_id[dj_a[k]] - 1
        let kb = ep_id[dj_b[k]] - 1
        let kaw = ka / 64
        let kab = ka - kaw * 64
        let kbw = kb / 64
        let kbb = kb - kbw * 64
        c = 0
        while c < H {
            if (epm[c * NEPW + kaw] & (one << kab)) != 0 {
                pm[c * NEPW + kbw] = pm[c * NEPW + kbw] | (one << kbb)
            }
            if (epm[c * NEPW + kbw] & (one << kbb)) != 0 {
                pm[c * NEPW + kaw] = pm[c * NEPW + kaw] | (one << kab)
            }
            c = c + 1
        }
        k = k + 1
    }
    var nact: i64 = 0
    c = 0
    while c < H {
        var any_ep = false
        w = 0
        while w < NEPW {
            if epm[c * NEPW + w] != 0 || pm[c * NEPW + w] != 0 {
                any_ep = true
            }
            w = w + 1
        }
        if any_ep {
            actors[nact] = c
            nact = nact + 1
        }
        c = c + 1
    }
    var conf: i64 = 0
    var i1: i64 = 0
    var i2: i64 = 0
    var c1: i64 = 0
    var c2: i64 = 0
    while i1 < nact {
        c1 = actors[i1]
        i2 = 0
        while i2 < nact {
            c2 = actors[i2]
            if c1 != c2 {
                var hit = false
                w = 0
                while w < NEPW {
                    if (pm[c1 * NEPW + w] & epm[c2 * NEPW + w]) != 0 {
                        hit = true
                    }
                    w = w + 1
                }
                if hit {
                    conf = conf + 1
                }
            }
            i2 = i2 + 1
        }
        i1 = i1 + 1
    }
    if conf != E_CONF {
        println("FAIL: @@LABEL@@ conflicts disagree with mirror")
        n_fail = n_fail + 1
    }
@@ABLATIONS@@

    // ── summary ────────────────────────────────────────────────────────
    println("=== @@LABEL@@ : EL+ role-aware closure (round 13) ===")
    println("classes (H):")
    println(H)
    println("roles (NR):")
    println(NR)
    println("sub axioms:")
    println(NSUB)
    println("existential restrictions (exsub):")
    println(NEX)
    println("disjoint pairs:")
    println(NDJ)
    println("roleSub axioms:")
    println(NRS)
    println("roleComp chains:")
    println(NCH)
    println("atomic projection closure edges:")
    println(atomic_edges)
    println("atom-source role edges (= existential targets, stoR/RtoS):")
    println(r_atom)
    println("atomic conflicts (ordered pairs):")
    println(conf)
    println("role fixpoint rounds (full):")
    println(rounds)
    return n_fail
}
"""

ABLATION_TEMPLATE = r"""
    // ── ablations: attribute derived edges to the rule families ───────
    clear_f()
    seed_and_expand(NEX, H, NR)
    let rounds_norc = role_fixpoint(1, 0, nrp, NCH, H, NR)
    let re_norc = count_role_edges(H, NR)
    if re_norc != E_NORC {
        println("FAIL: @@LABEL@@ no-roleComp role edges disagree with mirror")
        n_fail = n_fail + 1
    }
    clear_f()
    seed_and_expand(NEX, H, NR)
    let rounds_nors = role_fixpoint(0, 1, nrp, NCH, H, NR)
    let re_nors = count_role_edges(H, NR)
    if re_nors != E_NORS {
        println("FAIL: @@LABEL@@ no-roleSub role edges disagree with mirror")
        n_fail = n_fail + 1
    }
    println("role edges without roleComp (ablation):")
    println(re_norc)
    println("role edges without roleSub (ablation):")
    println(re_nors)
    println("role fixpoint rounds (no-roleComp):")
    println(rounds_norc)
    println("role fixpoint rounds (no-roleSub):")
    println(rounds_nors)
"""


def emit_driver(path, title, targets, caps, with_ablations=True):
    """targets: list of (fn_suffix, packed_path, label)."""
    fns = []
    calls = []
    for suffix, packed, label in targets:
        abl = ABLATION_TEMPLATE.replace("@@LABEL@@", label) \
            if with_ablations else ""
        fns.append(TARGET_TEMPLATE
                   .replace("@@NAME@@", suffix)
                   .replace("@@PATH@@", packed)
                   .replace("@@LABEL@@", label)
                   .replace("@@ABLATIONS@@", abl))
        calls.append(f"    n_fail = n_fail + run_{suffix}()")
    src = DRIVER_TEMPLATE
    for tok, val in caps.items():
        src = src.replace("@@" + tok + "@@", str(val))
    src = src.replace("@@TITLE@@", title)
    src = src.replace("@@TARGETS@@", "".join(fns))
    src = src.replace("@@CALLS@@", "\n".join(calls))
    with open(path, "w") as f:
        f.write(src)
    print(f"  wrote {path}")


def caps_for(stats_list, hc_min=0):
    """Compile-time capacities from target stats (+ ~10% headroom)."""
    h = max([s["h"] for s in stats_list] + [hc_min]) + 1
    hc = ((h * 11 // 10) + 99) // 100 * 100
    wc = ((h + 63) // 64 + 8 + 7) // 8 * 8
    nr = max(s["nr"] for s in stats_list)
    nrc = ((nr * 11 // 10) + 7) // 8 * 8
    sub = max(s["nsub"] for s in stats_list)
    subc = ((sub * 11 // 10) + 99) // 100 * 100
    ex = max(s["nex"] for s in stats_list)
    exc = ((ex * 11 // 10) + 99) // 100 * 100
    dj = max(s["ndj"] for s in stats_list)
    djc = max(64, ((dj * 2) + 15) // 16 * 16)
    rs = max(s["nrs"] for s in stats_list)
    rsc = max(64, ((rs * 2) + 15) // 16 * 16)
    ch = max(s["nch"] for s in stats_list)
    rcc = max(64, ((ch * 2) + 15) // 16 * 16)
    nep = max(s["nep"] for s in stats_list)
    epc = max(64, ((nep * 2) + 63) // 64 * 64)
    nepw = (epc + 63) // 64
    return {"HC": hc, "HC1": hc + 1, "NRC": nrc, "WC": wc,
            "SUBC": subc, "EXC": exc, "DJC": djc, "RSC": rsc,
            "RCC": rcc, "EPC": epc, "NEPW": nepw, "EPMN": hc * nepw,
            "RCN": nrc * nrc, "FMN": nrc * hc * wc,
            "DCN": nrc * hc, "ANCN": hc * wc}


# ── main ─────────────────────────────────────────────────────────────────

P = "artifacts/ontology-frontiers/multi-ontology/"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--go-only", action="store_true",
                    help="GO cones only (no OWL parse)")
    ap.add_argument("--obo-only", action="store_true",
                    help="OBO ontologies only (no GO cone work)")
    args = ap.parse_args()
    if not args.obo_only:
        run_go_pipeline(args)
        if args.go_only:
            return
    run_obo_pipeline()


def run_go_pipeline(args):

    # ── GO cones ─────────────────────────────────────────────────────────
    H, NR, sub, exsub, disj, rsub, rcomp = load_go(
        "../real-data/go_full_elplus_tbox.txt")
    roots = find_go_root_ids()
    print(f"loaded full GO: H={H} NR={NR} sub={len(sub)} exsub={len(exsub)}"
          f" disj={len(disj)} roleSub={len(rsub)} roleComp={len(rcomp)}")

    cone_stats = {}
    cone_tboxes = {}
    cone_idmaps = {}
    descs = {}
    for name, goid, label in GO_ROOTS:
        suffix = goid.replace(":", "_")
        st, tbox = slice_cone(H, NR, sub, exsub, disj, rsub, rcomp,
                              roots[suffix])
        (h, nr, sub2, exsub2, disj2, rsub2, rcomp2, id_map, role_map) = tbox
        descs[name] = st["desc"]
        cone_idmaps[name] = id_map
        print(f"[{name}] cone of {goid} ({label}): H={h} NR={nr} "
              f"sub={len(sub2)} exsub={len(exsub2)} disj={len(disj2)} "
              f"roleSub={len(rsub2)} roleComp={len(rcomp2)} | orphan "
              f"parents={st['orphan_parents']} cross-cone exsub="
              f"{st['cross_exsub']} cross-cone disj={st['cross_disj']}")
        cone_stats[name] = st
        cone_tboxes[name] = (h, nr, sub2, exsub2, disj2, rsub2, rcomp2)

    # partition checks
    names = [n for n, _g, _l in GO_ROOTS]
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            ov = descs[names[i]] & descs[names[j]]
            if ov:
                sys.exit(f"GO CONES OVERLAP: {names[i]} / {names[j]} "
                         f"share {len(ov)} classes")
    union = set().union(*descs.values())
    leftover = set(range(H)) - union
    print(f"GO cones: partition OK, |union|={len(union)}, "
          f"leftover={len(leftover)} {sorted(leftover)[:10]}")
    if leftover:
        sys.exit("GO CONES DO NOT COVER the full class set")
    total_cross_ex = sum(cone_stats[n]["cross_exsub"] for n in names)
    total_orphan = sum(cone_stats[n]["orphan_parents"] for n in names)
    print(f"GO cones: total cross-cone exsub={total_cross_ex}, "
          f"orphan parents={total_orphan}")

    # mirror per cone
    mirror = {}
    for name, _g, _l in GO_ROOTS:
        mirror[name] = run_mirror(name, cone_tboxes[name])

    # decomposition identities vs the round-12 full-GO run
    sum_atom = sum(mirror[n]["atomic_edges"] for n in names)
    sum_role = sum(mirror[n]["role_edges_atom"] for n in names)
    sum_conf = sum(mirror[n]["conf"] for n in names)
    print(f"decomposition: sum(atomic)={sum_atom} vs full "
          f"{FULL_GO['atomic_edges']}; sum(role)={sum_role} vs full "
          f"{FULL_GO['role_edges_atom']}; sum(conf)={sum_conf}, "
          f"full conf {FULL_GO['conf']}")
    if sum_atom != FULL_GO["atomic_edges"]:
        sys.exit("DECOMPOSITION FAILED: atomic edges")
    if total_cross_ex == 0 and total_orphan == 0:
        if sum_role != FULL_GO["role_edges_atom"]:
            sys.exit("DECOMPOSITION FAILED: role edges")
    else:
        print("  (role-edge identity skipped: cross-cone axioms exist)")

    # full-GO conflicts recomputed from the cone masks (grouped counter,
    # independent of round 12): anc over full ids = per-cone anc lifted
    # back through the id maps; disj = all 55 full pairs.
    print("recomputing full-GO conflicts from cone masks (grouped) ...")
    anc_full = [0] * H
    for name in names:
        h, nr, sub2, exsub2, disj2, rsub2, rcomp2 = cone_tboxes[name]
        from gen_go_full_data import topo_order
        order, parents = topo_order(h, sub2)
        anc = [(1 << c) for c in range(h)]
        for c in order:
            m = anc[c]
            for p in parents[c]:
                m |= anc[p]
            anc[c] = m
        inv = {v: k for k, v in cone_idmaps[name].items()}
        for c in range(h):
            anc_full[inv[c]] = lift_mask(anc[c], inv, H)
    gconf_full, _ = grouped_conf(anc_full, disj, H)
    print(f"full-GO conflicts from cone masks: {gconf_full} vs round-12 "
          f"{FULL_GO['conf']}")
    if gconf_full != FULL_GO["conf"]:
        sys.exit("DECOMPOSITION FAILED: full-GO conflicts")
    cross_conf = gconf_full - sum_conf
    print(f"cross-cone conflicts: {cross_conf} "
          f"({100.0 * cross_conf / gconf_full:.2f}% of full)")

    for name, _g, _l in GO_ROOTS:
        st = dict(mirror[name])
        st.update({"sub2": cone_tboxes[name][2],
                   "exsub2": cone_tboxes[name][3],
                   "disj2": cone_tboxes[name][4],
                   "rsub2": cone_tboxes[name][5],
                   "rcomp2": cone_tboxes[name][6]})
        emit_packed(f"{name}_packed.txt", st)

    go_targets = [(n, P + n + "_packed.txt", n) for n, _g, _l in GO_ROOTS]
    emit_driver("go_roots_elplus_driver.sio",
                "Round 13: EL+ role-aware closure on the three GO root "
                "cones (GO:0008150 BP, GO:0005575 CC, GO:0003674 MF).",
                go_targets, caps_for([mirror[n] for n in names]))


def run_obo_pipeline():
    # ── OBO ontologies ───────────────────────────────────────────────────
    obo_stats = {}
    obo_targets = []
    for name, ns, owl in OBO_ONTS:
        st, tbox, labels_out, role_labels_out, iris_out, role_iris_out = \
            extract_obo(name, ns, owl, "downloads/ro.owl")
        m = run_mirror(name, tbox)
        obo_stats[name] = m
        st2 = dict(m)
        st2.update({"sub2": tbox[2], "exsub2": tbox[3], "disj2": tbox[4],
                    "rsub2": tbox[5], "rcomp2": tbox[6]})
        emit_packed(f"{name}_packed.txt", st2)
        emit_tsvs(name, tbox, labels_out, role_labels_out, iris_out,
                  role_iris_out)
        obo_targets.append((name, P + name + "_packed.txt", name))

    emit_driver("obo_elplus_driver.sio",
                "Round 13: EL+ role-aware closure on additional OBO "
                "ontologies (CL cell ontology, UBERON anatomy).",
                obo_targets, caps_for(list(obo_stats.values())))


def lift_mask(local_mask, inv, H):
    """Lift a cone-local ancestor bitmask back to full-GO bit positions."""
    out = 0
    m = local_mask
    while m:
        b = m & -m
        c = b.bit_length() - 1
        m ^= b
        out |= 1 << inv[c]
    return out


if __name__ == "__main__":
    main()
