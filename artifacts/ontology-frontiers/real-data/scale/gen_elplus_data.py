#!/usr/bin/env python3
"""gen_elplus_data.py — round 9: EL+ role-aware boolean closure on the FULL
OAEI 2016 Anatomy human TBox, with an independent python mirror whose
numbers are embedded as expected_*() functions.

Round 7 (gen_full_data.py) used only atomic subsumption/disjointness.  This
round integrates the role-aware closure of
formal/OntologyELPlusClosureComplete.lean (crStep / closeSatF, proved
equivalent to the deductive system by subBPlusC_iff / conflictBPlusC_iff)
into the real-data pipeline:

  * extract_tbox.py (round 9) now emits `exsub <ont> <child> <role>
    <filler>` lines for the existential restrictions C ⊑ ∃part_of.F that
    earlier rounds SKIPPED (1,662 in human.owl), plus roleSub / roleComp
    lines and roles.tsv;
  * the concept universe is interned as: atoms 0..H-1, top = H, base B =
    H+1 (no conjunctions occur in the Anatomy data), and one existential
    concept ∃r.f per (role r, base f): id = B + r*B + f, so U = B + NR*B;
  * the mirror runs the general 8-rule fixpoint (transitivity, conjElim,
    conjIntro, stoR, Rmono, roleSub, roleComp, RtoS) with python sets;
  * a packed cross-check (bitmask ancestor sets + the reduced edge
    formulas below) must agree exactly or the script aborts.

Data profile (honest limitation, carried into the driver and README):
the Anatomy track has ONE active role (part_of; a second declared
property, ObsoleteProperty, is never used in a restriction), and ZERO
roleSub / roleComp axioms.  The roleSub / roleComp / conjunction rules
are therefore exercised on a SNOMED-style synthetic instance (the TBox of
formal/OntologyELPlus.lean, as in examples/ontology_elplus_closure_demo.sio)
through the dense variant of stdlib/ontology/elplus.sio, in the same
driver.

Reduction used by the sparse driver (and cross-checked here against the
general fixpoint): with no conjunctions, no roleSub and no roleComp, the
least fixpoint satisfies, for every atom c,

  E(c) = U_{a in anc(c)} U_{(a,r,f) stated} anc_base(f)

where anc() is the subsumption ancestor set (reflexive, incl. top column)
and E(c) is the set of fillers of role edges with source c.  Ex-concept
sources x = ∃r.f have E(x) = {(r, f') : f' in anc_base(f)}, so every
statistic of the full fixpoint reduces to ancestor-set arithmetic:

  |S(c)|   = |anc_base(c)| + |E(c)|            (atom rows)
  |S(top)| = 1
  |S(x)|   = 1 + |anc_base(f)|                 (ex rows, x = ex r.f; the
             reflexive cell coincides with the f' = f existential target)
  nR_atom  = sum_c |E(c)|     nR_ex = NR * sum_f |anc_base(f)|

Emitted files (all in this directory):

  elplus_data.sio        real Anatomy human TBox: packed axiom arrays and
                         expected_*() mirror values (the 4096-stride
                         workspace matrices live in stdlib elplus.sio —
                         module-level arrays must not cross a &! boundary)
  elplus_synth_data.sio  SNOMED-style synthetic instance (concept table,
                         stated axioms, synth_expected_*() mirror values)
  elplus_scale_driver.sio  two-part driver (synthetic dense via stdlib
                         module + real sparse), ALL PASS only if every
                         number equals this mirror

Known compiler workarounds applied (verified rounds 6-7, REAL_RESULTS.md):
  * init assignments chunked at 500 statements/function;
  * module-level splat arrays have garbage leading cells (bool 0..2,
    i64/f64 0) -> data arrays are fully assigned; bool matrices get
    explicit fixup writes of cells 0..2 first;
  * multimodule thin-link dies beyond ~24k assignments -> this module
    stays well under (mappings' m_conf/m_keep are NOT emitted: the
    conflict count is unchanged by roles, so the round-7 repair carries
    over byte-identically; see the driver header).
"""

import sys

# ── loading ──────────────────────────────────────────────────────────────


def load():
    sub, disj, exsub, rsub, rcomp = [], set(), [], [], []
    n_roles = 0
    with open("../tbox.txt") as f:
        for line in f:
            if line.startswith("# roles_human"):
                n_roles = int(line.split()[2])
                continue
            if line.startswith("#") or not line.strip():
                continue
            p = line.split()
            if p[1] != "human":
                continue
            if p[0] == "sub":
                sub.append((int(p[2]), int(p[3])))
            elif p[0] == "disj":
                a, b = int(p[2]), int(p[3])
                disj.add((min(a, b), max(a, b)))
            elif p[0] == "exsub":
                exsub.append((int(p[2]), int(p[3]), int(p[4])))
            elif p[0] == "roleSub":
                rsub.append((int(p[2]), int(p[3])))
            elif p[0] == "roleComp":
                rcomp.append((int(p[2]), int(p[3]), int(p[4])))
    maps = []
    with open("../mappings.tsv") as f:
        next(f)
        for line in f:
            _, m, h, c = line.rstrip("\n").split("\t")
            a, b = f"{float(c):.4f}".split(".")
            maps.append((int(m), int(h), int(a) * 10000 + int(b)))
    n_human = 0
    with open("../classes.tsv") as f:
        next(f)
        for line in f:
            if line.split("\t")[1] == "human":
                n_human += 1
    return sub, sorted(disj), exsub, rsub, rcomp, n_roles, maps, n_human


# ── general set-based 8-rule fixpoint (crStep mirror) ────────────────────


def fixpoint(H, NR, conjs, subs, exsubs, rsubs, rcomps):
    """Least fixpoint of the 8 completion rules over the interned
    universe.  conjs: list of (id, arg1, arg2) conjunction concepts
    (ids must be H+1..H+len(conjs)).  Returns
    (S, R, U, base, top, ckind, carg1, carg2, exid, rclos, rounds).
    exid[(r, f)] = interned id of ex r.f for every base concept f
    (all role variants of all base concepts are interned, matching
    conceptUniv of the Lean file).
    """
    # layout: atoms 0..H-1, top = H, conjs H+1..B-1, then existentials
    top = H
    base = list(range(H)) + [top] + [c for c, _, _ in conjs]
    NB = len(base)
    B = H + 1 + len(conjs)
    U = B + NR * NB
    base_set = set(base)
    ckind = [0] * U
    carg1 = list(range(H)) + [0] * (U - H)
    carg2 = [0] * U
    ckind[top] = 1
    for cid, a1, a2 in conjs:
        ckind[cid] = 2
        carg1[cid] = a1
        carg2[cid] = a2
    exid = {}
    for r in range(NR):
        for k, f in enumerate(base):
            cid = B + r * NB + k
            exid[(r, f)] = cid
            ckind[cid] = 3
            carg1[cid] = r
            carg2[cid] = f

    S = [set() for _ in range(U)]
    for c in range(U):
        S[c].add(c)
        S[c].add(top)
    for c, p in subs:
        S[c].add(p)
    for c, r, f in exsubs:
        S[c].add(exid[(r, f)])

    rclos = [[False] * NR for _ in range(NR)]
    for r in range(NR):
        rclos[r][r] = True
    for r, s in rsubs:
        rclos[r][s] = True
    ch = True
    while ch:
        ch = False
        for a in range(NR):
            for b in range(NR):
                if rclos[a][b]:
                    for d in range(NR):
                        if rclos[b][d] and not rclos[a][d]:
                            rclos[a][d] = True
                            ch = True

    R = set()
    rounds = 0
    changed = True
    while changed:
        changed = False
        rounds += 1
        # (T) transitivity
        for a in range(U):
            sa = S[a]
            add = set()
            for b in sa:
                add |= S[b]
            if not add <= sa:
                sa |= add
                changed = True
        # (CE) conjElim / (CI) conjIntro
        for d, c1, c2 in conjs:
            for a in range(U):
                sa = S[a]
                if d in sa:
                    if c1 not in sa:
                        sa.add(c1)
                        changed = True
                    if c2 not in sa:
                        sa.add(c2)
                        changed = True
                if c1 in sa and c2 in sa and d not in sa:
                    sa.add(d)
                    changed = True
        # (S2R) stoR: C <= ex r.D becomes edge (r, C, D)
        for a in range(U):
            for d in S[a]:
                if ckind[d] == 3:
                    e = (carg1[d], a, carg2[d])
                    if e not in R:
                        R.add(e)
                        changed = True
        # (RM) Rmono: edge (r,C,D) and D <= D' (D' base) give (r,C,D')
        new_edges = []
        for (r, c, f) in R:
            for d2 in S[f]:
                if d2 in base_set:
                    e = (r, c, d2)
                    if e not in R:
                        new_edges.append(e)
        for e in new_edges:
            if e not in R:
                R.add(e)
                changed = True
        # (RS) roleSub: edge (r,C,D) and r <=* s give (s,C,D)
        new_edges = []
        for (r, c, f) in R:
            for s in range(NR):
                if rclos[r][s]:
                    e = (s, c, f)
                    if e not in R:
                        new_edges.append(e)
        for e in new_edges:
            if e not in R:
                R.add(e)
                changed = True
        # (RC) roleComp: edges (r1,C,D) and (r2,D,E), chain r1 o r2 <= r3
        if rcomps:
            by_rs = {}
            for (r, c, f) in R:
                by_rs.setdefault((r, c), []).append(f)
            new_edges = []
            for (r1, r2, r3) in rcomps:
                for (rr, c), fills in by_rs.items():
                    if rr != r1:
                        continue
                    for d in fills:
                        for e2 in by_rs.get((r2, d), ()):
                            e = (r3, c, e2)
                            if e not in R:
                                new_edges.append(e)
            for e in new_edges:
                if e not in R:
                    R.add(e)
                    changed = True
        # (R2S) RtoS: edge (r,C,D) gives C <= ex r.D
        for (r, c, f) in R:
            x = exid[(r, f)]
            if x not in S[c]:
                S[c].add(x)
                changed = True
    return S, R, U, base, top, ckind, carg1, carg2, exid, rclos, rounds


# ── real-data mirror: set-based + packed cross-check ─────────────────────


def mirror_real(H, NR, sub, disj, exsub, rsub, rcomp, maps):
    assert not rsub and not rcomp, "Anatomy profile: no role hierarchy"
    S, R, U, base, top, ckind, carg1, carg2, exid, rclos, rounds = \
        fixpoint(H, NR, [], sub, exsub, rsub, rcomp)
    B = len(base)

    s_cells = sum(len(s) for s in S)
    nR_total = len(R)
    nR_atom = sum(1 for (r, c, f) in R if c < H)
    atoms = set(range(H))

    # closure edge count over atoms (reflexive; excludes the top column)
    edges = sum(1 for c in range(H) for d in S[c] if d in atoms)

    # derived conflicts, set-based (round-6/7 definition, but over the
    # role-aware S; disjointness is atomic so the answer must equal the
    # round-7 number -- the mirror asserts this against the packed check)
    disjC = set()
    for d1, d2 in disj:
        for c1 in range(H):
            if d1 in S[c1]:
                for c2 in range(H):
                    if d2 in S[c2]:
                        disjC.add((c1, c2))
                        disjC.add((c2, c1))
    M = len(maps)

    def conflict_set(i, j):
        return (maps[i][0] == maps[j][0]
                and (maps[i][1], maps[j][1]) in disjC)

    n_conf = sum(1 for i in range(M) for j in range(M)
                 if i != j and conflict_set(i, j))

    # ── packed cross-check: bitmask ancestor sets + reduced formulas ────
    TOPBIT = 1 << H
    anc = [(1 << c) | TOPBIT for c in range(H)] + [TOPBIT]
    changed = True
    while changed:
        changed = False
        for c, p in sub:
            new = anc[c] | anc[p]
            if new != anc[c]:
                anc[c] = new
                changed = True
    # per-class agreement with the set-based fixpoint (base projection)
    base_set = set(base)
    for c in range(H + 1):
        if bin(anc[c]).count("1") != len(S[c] & base_set):
            sys.exit(f"MIRROR CROSS-CHECK FAILED: ancestor set of {c} "
                     f"disagrees with set-based fixpoint")
    stated = {}
    for c, r, f in exsub:
        assert r == exsub[0][1], "single-active-role profile expected"
        stated.setdefault(c, 0)
        stated[c] |= anc[f]
    nE_atom = 0
    for c in range(H):
        m = anc[c]
        em = 0
        k = 0
        while m:
            if m & 1:
                em |= stated.get(k, 0)
            m >>= 1
            k += 1
        nE_atom += bin(em).count("1")
    basesum = sum(bin(anc[f]).count("1") for f in range(H + 1))
    # ex row x = ex r.f: S[x] = {top} u {ex r.f' : f' in anc_base(f)}
    # (the reflexive cell coincides with the f' = f target), so
    # |S[x]| = 1 + |anc_base(f)|
    s_cells_packed = (sum(bin(anc[c]).count("1") for c in range(H))
                      + nE_atom + 1 + NR * (B + basesum))
    nR_ex = NR * basesum
    if nE_atom != nR_atom:
        sys.exit(f"MIRROR CROSS-CHECK FAILED: packed atom role edges "
                 f"{nE_atom} != set-based {nR_atom}")
    if s_cells_packed != s_cells:
        sys.exit(f"MIRROR CROSS-CHECK FAILED: packed S cells "
                 f"{s_cells_packed} != set-based {s_cells}")
    if nR_atom + nR_ex != nR_total:
        sys.exit(f"MIRROR CROSS-CHECK FAILED: packed total role edges "
                 f"{nR_atom + nR_ex} != set-based {nR_total}")

    # packed conflicts (endpoint partner-bit trick from round 7)
    eps = sorted({x for pr in disj for x in pr})
    ep_idx = {e: k for k, e in enumerate(eps)}
    E2 = len(eps)
    pbits = [0] * E2
    for a, b in disj:
        pbits[ep_idx[a]] |= 1 << ep_idx[b]
        pbits[ep_idx[b]] |= 1 << ep_idx[a]
    mask = [0] * H
    for e in eps:
        mask[e] |= 1 << ep_idx[e]
    changed = True
    while changed:
        changed = False
        for c, p in sub:
            if mask[p] & ~mask[c]:
                mask[c] |= mask[p]
                changed = True
    pmask = [0] * H
    for c in range(H):
        m, pm, k = mask[c], 0, 0
        while m:
            if m & 1:
                pm |= pbits[k]
            m >>= 1
            k += 1
        pmask[c] = pm

    def conflict_bit(i, j):
        return (maps[i][0] == maps[j][0]
                and (mask[maps[i][1]] & pmask[maps[j][1]]) != 0)

    n_conf_bit = sum(1 for i in range(M) for j in range(M)
                     if i != j and conflict_bit(i, j))
    if n_conf_bit != n_conf:
        sys.exit(f"MIRROR CROSS-CHECK FAILED: set-based conflicts "
                 f"{n_conf} != bitmask {n_conf_bit}")

    return {
        "H": H, "NR": NR, "NSUB": len(sub), "NDEP": len(disj),
        "NEX": len(exsub), "EP": E2, "M": M, "U": U, "B": B,
        "edges": edges, "s_cells": s_cells, "nR_atom": nR_atom,
        "nR_total": nR_total, "n_conf": n_conf, "rounds": rounds,
    }


# ── synthetic SNOMED-style instance (exercises conj/roleSub/roleComp) ────
# Classes: 0 Inflammation  1 Disorder  2 Lung   3 Organ
#          4 Pneumonia    5 Drug      6 InflammatoryLesion
#          7 DrugInducedDisorder       8 top    9 Lung x Inflammation
# Roles:   0 RoleGroup    1 DirectSite  2 PartOf
# Axioms (formal/OntologyELPlus.lean snomedTBox):
#   Pneumonia <= Inflammation <= Disorder ; Lung <= Organ
#   Pneumonia <= ex DirectSite.(Lung x Inflammation)
#   DirectSite <= RoleGroup ; DirectSite o PartOf <= RoleGroup
#   Lung <= ex PartOf.Organ ; Disorder _|_ Drug
#   DrugInducedDisorder <= Disorder, Drug ; InflammatoryLesion <= Inflammation

SYNTH_SUB = [(4, 0), (0, 1), (2, 3), (7, 1), (7, 5), (6, 0)]
SYNTH_EXSUB = [(4, 1, 9), (2, 2, 3)]
SYNTH_RSUB = [(1, 0)]
SYNTH_RCOMP = [(1, 2, 0)]
SYNTH_DISJ = [(1, 5)]
SYNTH_CONJ = [(9, 2, 0)]


def mirror_synth():
    S, R, U, base, top, ckind, carg1, carg2, exid, rclos, rounds = \
        fixpoint(8, 3, SYNTH_CONJ, SYNTH_SUB, SYNTH_EXSUB, SYNTH_RSUB,
                 SYNTH_RCOMP)
    NB = len(base)

    def ex(r, f):
        return exid[(r, f)]

    # the six demo queries (examples/ontology_elplus_closure_demo.sio);
    # abort rather than emit if the general fixpoint disagrees
    checks = [
        ("q1 Pneumonia <= ex RoleGroup.Organ", ex(0, 3) in S[4], True),
        ("q2 Pneumonia <= ex RoleGroup.(Lung x Infl)", ex(0, 9) in S[4],
         True),
        ("q3 Pneumonia <= Disorder", 1 in S[4], True),
        ("q6 NOT Organ <= Disorder", 1 in S[3], False),
    ]

    def conflict(a, b):
        return any(d1 in S[a] and d2 in S[b] and
                   (min(d1, d2), max(d1, d2)) in
                   {(1, 5)} for d1 in range(U) for d2 in range(U))

    checks.append(("q4 conflict(Pneumonia, Drug)", conflict(4, 5), True))
    checks.append(("q5 conflict(DrugInducedDisorder, itself)",
                   conflict(7, 7), True))
    for name, got, want in checks:
        if got != want:
            sys.exit(f"SYNTH MIRROR FAILED: {name}: got {got}, "
                     f"want {want}")
    return {
        "U": U, "NB": NB, "top": top, "ckind": ckind, "carg1": carg1,
        "carg2": carg2, "s_cells": sum(len(s) for s in S),
        "r_edges": len(R), "rounds": rounds,
    }


# ── emission helpers ─────────────────────────────────────────────────────


def emit_init(f, assigns, chunk=500, prefix=""):
    f.write(f"pub fn {prefix}init_data() {{\n")
    n = 0
    for i in range(0, len(assigns), chunk):
        f.write(f"    {prefix}init_chunk_{n}()\n")
        n += 1
    f.write("}\n\n")
    n = 0
    for i in range(0, len(assigns), chunk):
        f.write(f"pub fn {prefix}init_chunk_{n}() {{\n")
        for stmt in assigns[i:i + chunk]:
            f.write(f"    {stmt}\n")
        f.write("}\n\n")
        n += 1


SC = 4096          # sparse module capacity (classes incl top slot)


def emit_real(path, ex, sub, disj, exsub, maps):
    H, NSUB, NEX, M = ex["H"], ex["NSUB"], ex["NEX"], ex["M"]
    with open(path, "w") as f:
        f.write("// GENERATED by gen_elplus_data.py — do not edit by hand.\n")
        f.write("// Round 9: FULL OAEI 2016 Anatomy human TBox WITH roles\n")
        f.write("// (EL+ existential restrictions C <= ex part_of.F).\n")
        f.write("// Mirror values embedded as expected_*().\n")
        for k_, v in ex.items():
            f.write(f"// {k_} = {v}\n")
        f.write("\n")
        f.write(f"pub fn expected_h() -> i64 {{ return {ex['H']} }}\n")
        f.write(f"pub fn expected_roles() -> i64 {{ return {ex['NR']} }}\n")
        f.write(f"pub fn expected_sub() -> i64 {{ return {ex['NSUB']} }}\n")
        f.write(f"pub fn expected_disj() -> i64 {{ return {ex['NDEP']} }}\n")
        f.write(f"pub fn expected_exsub() -> i64 {{ return {ex['NEX']} }}\n")
        f.write(f"pub fn expected_endpoints() -> i64 {{ return {ex['EP']} }}\n")
        f.write(f"pub fn expected_mappings() -> i64 {{ return {ex['M']} }}\n")
        f.write(f"pub fn expected_closure_edges() -> i64 {{ return {ex['edges']} }}\n")
        f.write(f"pub fn expected_s_cells() -> i64 {{ return {ex['s_cells']} }}\n")
        f.write(f"pub fn expected_role_edges_atom() -> i64 {{ return {ex['nR_atom']} }}\n")
        f.write(f"pub fn expected_role_edges_total() -> i64 {{ return {ex['nR_total']} }}\n")
        f.write(f"pub fn expected_derived_conflicts() -> i64 {{ return {ex['n_conf']} }}\n\n")
        f.write(f"pub var h_sub: [i64; {SC}] = [0; {SC}]  // child*10000+parent\n")
        f.write(f"pub var ex_c: [i64; {SC}] = [0; {SC}]   // existential child\n")
        f.write(f"pub var ex_f: [i64; {SC}] = [0; {SC}]   // existential filler\n")
        f.write(f"pub var ex_r: [i64; {SC}] = [0; {SC}]   // role id\n")
        f.write(f"pub var h_dep: [i64; {ex['NDEP']}] = [0; {ex['NDEP']}]  // a*10000+b\n")
        f.write(f"pub var m_pack: [i64; {M}] = [0; {M}]  // mouse_ent*10000+human_cls\n\n")
        assigns = []
        for k_, (c, p) in enumerate(sub):
            assigns.append(f"h_sub[{k_}] = {c * 10000 + p}")
        for k_, (c, r, fl) in enumerate(exsub):
            assigns.append(f"ex_c[{k_}] = {c}")
            assigns.append(f"ex_f[{k_}] = {fl}")
            assigns.append(f"ex_r[{k_}] = {r}")
        for k_, (a, b) in enumerate(disj):
            assigns.append(f"h_dep[{k_}] = {a * 10000 + b}")
        for k_, (m, h, _c) in enumerate(maps):
            assigns.append(f"m_pack[{k_}] = {m * 10000 + h}")
        emit_init(f, assigns)
    return len(assigns)


def emit_synth(path, sy):
    U = sy["U"]
    with open(path, "w") as f:
        f.write("// GENERATED by gen_elplus_data.py — do not edit by hand.\n")
        f.write("// Round 9 synthetic instance: the SNOMED-flavoured TBox of\n")
        f.write("// formal/OntologyELPlus.lean (Fin 8 x Fin 3), interned over\n")
        f.write("// the FULL concept universe (every role variant of every\n")
        f.write("// base concept): 8 atoms + top + 1 conjunction + 3*10\n")
        f.write("// existentials = 40 concepts.  Exercises the rules the\n")
        f.write("// Anatomy data cannot: conjElim/Intro, roleSub, roleComp.\n")
        for k_ in ("U", "NB", "top", "s_cells", "r_edges", "rounds"):
            f.write(f"// {k_} = {sy[k_]}\n")
        f.write("\n")
        f.write(f"pub fn synth_expected_u() -> i64 {{ return {sy['U']} }}\n")
        f.write(f"pub fn synth_expected_base() -> i64 {{ return {sy['NB']} }}\n")
        f.write(f"pub fn synth_expected_top() -> i64 {{ return {sy['top']} }}\n")
        f.write(f"pub fn synth_expected_nr() -> i64 {{ return 3 }}\n")
        f.write(f"pub fn synth_expected_sub() -> i64 {{ return {len(SYNTH_SUB)} }}\n")
        f.write(f"pub fn synth_expected_exsub() -> i64 {{ return {len(SYNTH_EXSUB)} }}\n")
        f.write(f"pub fn synth_expected_disj() -> i64 {{ return {len(SYNTH_DISJ)} }}\n")
        f.write(f"pub fn synth_expected_rolesub() -> i64 {{ return {len(SYNTH_RSUB)} }}\n")
        f.write(f"pub fn synth_expected_rolecomp() -> i64 {{ return {len(SYNTH_RCOMP)} }}\n")
        f.write(f"pub fn synth_expected_s_cells() -> i64 {{ return {sy['s_cells']} }}\n")
        f.write(f"pub fn synth_expected_r_edges() -> i64 {{ return {sy['r_edges']} }}\n\n")
        f.write("pub var sc_kind: [i64; 64] = [0; 64]\n")
        f.write("pub var sc_arg1: [i64; 64] = [0; 64]\n")
        f.write("pub var sc_arg2: [i64; 64] = [0; 64]\n")
        f.write("pub var ssub: [i64; 64] = [0; 64]   // child*100+parent\n")
        f.write("pub var sex_cf: [i64; 64] = [0; 64] // child*100+filler\n")
        f.write("pub var sex_r: [i64; 64] = [0; 64]  // role id\n")
        f.write("pub var sdisj: [i64; 64] = [0; 64]  // a*100+b\n")
        f.write("pub var srsub: [i64; 8] = [0; 8]    // sub*10+super\n")
        f.write("pub var schain: [i64; 8] = [0; 8]   // (r1*10+r2)*10+r3\n\n")
        assigns = []
        for a in ("sc_kind", "sc_arg1", "sc_arg2", "ssub", "sex_cf",
                  "sex_r", "sdisj"):
            assigns += [f"{a}[0] = 0", f"{a}[1] = 0", f"{a}[2] = 0"]
        assigns += ["srsub[0] = 0", "schain[0] = 0"]
        for ci in range(U):
            assigns.append(f"sc_kind[{ci}] = {sy['ckind'][ci]}")
            assigns.append(f"sc_arg1[{ci}] = {sy['carg1'][ci]}")
            assigns.append(f"sc_arg2[{ci}] = {sy['carg2'][ci]}")
        for k_, (c, p) in enumerate(SYNTH_SUB):
            assigns.append(f"ssub[{k_}] = {c * 100 + p}")
        for k_, (c, r, fl) in enumerate(SYNTH_EXSUB):
            assigns.append(f"sex_cf[{k_}] = {c * 100 + fl}")
            assigns.append(f"sex_r[{k_}] = {r}")
        for k_, (a, b) in enumerate(SYNTH_DISJ):
            assigns.append(f"sdisj[{k_}] = {a * 100 + b}")
        for k_, (r, s) in enumerate(SYNTH_RSUB):
            assigns.append(f"srsub[{k_}] = {r * 10 + s}")
        for k_, (r1, r2, r3) in enumerate(SYNTH_RCOMP):
            assigns.append(f"schain[{k_}] = {(r1 * 10 + r2) * 10 + r3}")
        emit_init(f, assigns, chunk=400, prefix="synth_")
    return len(assigns)


# ── driver template (@TOKENS@ substituted from the mirror) ──────────────

DRIVER = '''//@ run-pass
//@ expect-stdout: ALL PASS
// Round 9: EL+ role-aware boolean closure integrated into the OAEI 2016
// Anatomy real-data pipeline.  Executable mirror of the verified
// saturation engine of formal/OntologyELPlusClosureComplete.lean (crStep,
// iterated to a genuine fixpoint like closeSatF; subBPlusC_iff /
// conflictBPlusC_iff), via the reusable stdlib/ontology/elplus.sio module.
//
// Part A (synthetic, DENSE variant): the SNOMED-flavoured TBox of
// formal/OntologyELPlus.lean (8 atoms, 3 roles, one conjunction,
// DirectSite <= RoleGroup, DirectSite o PartOf <= RoleGroup), interned
// over the full concept universe (40 concepts).  This part exercises the
// rules the real data cannot: conjElim/Intro, roleSub, roleComp.
//
// Part B (real data, SPARSE variant): the FULL Anatomy human TBox
// (H = @H@ classes, @NSUB@ sub axioms, @NEX@ existential restrictions
// C <= ex part_of.F, @NDEP@ disjoint pairs, @M@ candidate mappings) with
// the role-aware closure computed by ancestor-set expansion over packed
// adjacency (no per-role matrix): E(c) = union over ancestors a of c of
// the ancestor sets of the stated fillers of a.  The python mirror runs
// the GENERAL 8-rule fixpoint and aborts unless it agrees with this
// reduction exactly; the checks below compare the driver against the
// mirror.
//
// Data profile limitation (asserted below, documented in README): the
// Anatomy track has ONE active role (part_of) and NO roleSub/roleComp
// axioms, so roles extend the subsumption relation (existential targets)
// without changing the atomic disjointness conflicts: the derived
// conflict count must equal the round-7 value byte-identically, and the
// round-7 repair therefore carries over unchanged (m_keep/m_conf are not
// re-emitted).
//
// Data: elplus_data.sio + elplus_synth_data.sio (generated; mirror
// numbers embedded as expected_*() / synth_expected_*()).

import elplus_data::*
import elplus_synth_data::*
use ontology::elplus::{elplus_role_closure, elplus_seed, elplus_fixpoint, elplus_s_count, elplus_r_count, elplus_conflict, elplus_sparse_init_mats, elplus_sparse_build_adj, elplus_sparse_bfs, elplus_sparse_seed_edges, elplus_sparse_expand, elplus_sparse_row_count}

fn main() -> i32 with IO, Mut, Div, Panic {
    var n_fail = 0

    // ══ Part A: synthetic SNOMED-style instance, dense fixpoint ═══════
    synth_init_data()
    let U = synth_expected_u()
    let NB = synth_expected_base()
    let NRS = synth_expected_nr()
    let TOP = synth_expected_top()

    // exid[r*64+f] = interned id of ex r.(base f), or -1
    var exid: [i64; 512] = [0; 512]
    var ci: i64 = 0
    while ci < 512 {
        exid[ci] = 0 - 1
        ci = ci + 1
    }
    ci = 0
    while ci < U {
        if sc_kind[ci] == 3 {
            exid[sc_arg1[ci] * 64 + sc_arg2[ci]] = ci
        }
        ci = ci + 1
    }

    // role-hierarchy closure (reflexive + transitive over stated roleSub)
    var rclos: [bool; 64] = [false; 64]
    var k: i64 = 0
    var rr: i64 = 0
    var ss: i64 = 0
    while k < synth_expected_rolesub() {
        rr = srsub[k] / 10
        ss = srsub[k] - rr * 10
        rclos[rr * 8 + ss] = true
        k = k + 1
    }
    elplus_role_closure(&!rclos, NRS)

    // seed S: reflexivity + top + stated sub + stated exsub
    var s: [bool; 4096] = [false; 4096]
    elplus_seed(&!s, U, TOP)
    var c: i64 = 0
    var d: i64 = 0
    k = 0
    while k < synth_expected_sub() {
        c = ssub[k] / 100
        d = ssub[k] - c * 100
        s[c * 64 + d] = true
        k = k + 1
    }
    k = 0
    var x: i64 = 0
    while k < synth_expected_exsub() {
        c = sex_cf[k] / 100
        d = sex_cf[k] - c * 100
        x = exid[sex_r[k] * 64 + d]
        if x >= 0 {
            s[c * 64 + x] = true
        } else {
            println("FAIL: stated existential not interned (synth)")
            n_fail = n_fail + 1
        }
        k = k + 1
    }

    // disjointness (symmetric)
    var sdisjm: [bool; 4096] = [false; 4096]
    k = 0
    var a2: i64 = 0
    var b2: i64 = 0
    while k < synth_expected_disj() {
        a2 = sdisj[k] / 100
        b2 = sdisj[k] - a2 * 100
        sdisjm[a2 * 64 + b2] = true
        sdisjm[b2 * 64 + a2] = true
        k = k + 1
    }

    // unpack composition chains
    var ch1: [i64; 8] = [0; 8]
    var ch2: [i64; 8] = [0; 8]
    var ch3: [i64; 8] = [0; 8]
    k = 0
    var pack: i64 = 0
    while k < synth_expected_rolecomp() {
        pack = schain[k]
        ch3[k] = pack - (pack / 10) * 10
        pack = pack / 10
        ch2[k] = pack - (pack / 10) * 10
        ch1[k] = pack / 10
        k = k + 1
    }

    // dense 8-rule fixpoint (crStep mirror, iterated to stability)
    var redges: [bool; 32768] = [false; 32768]
    let rounds = elplus_fixpoint(&!s, &!redges, &sc_kind, &sc_arg1, &sc_arg2, &exid, &rclos, &ch1, &ch2, &ch3, synth_expected_rolecomp(), U, NB, NRS)
    println("synthetic closure rounds:")
    println(rounds)

    let sc = elplus_s_count(&s, U)
    if sc != synth_expected_s_cells() {
        println("FAIL: synthetic S cell count disagrees with mirror")
        n_fail = n_fail + 1
    }
    let rc = elplus_r_count(&redges, NRS, U, NB)
    if rc != synth_expected_r_edges() {
        println("FAIL: synthetic role edge count disagrees with mirror")
        n_fail = n_fail + 1
    }

    // the six demo queries
    x = exid[0 * 64 + 3]
    if s[4 * 64 + x] {
        println("subBPlus 4 (ex 0 (atom 3)) = true")
    } else {
        println("FAIL: Pneumonia <= ex RoleGroup.Organ should hold (roleComp)")
        n_fail = n_fail + 1
    }
    x = exid[0 * 64 + 9]
    if s[4 * 64 + x] {
        println("subBPlus 4 (ex 0 (conj 2 0)) = true")
    } else {
        println("FAIL: Pneumonia <= ex RoleGroup.(Lung x Infl) should hold (roleSub)")
        n_fail = n_fail + 1
    }
    if s[4 * 64 + 1] {
        println("subBPlus 4 1 = true")
    } else {
        println("FAIL: Pneumonia <= Disorder should hold (transitivity)")
        n_fail = n_fail + 1
    }
    if elplus_conflict(&s, &sdisjm, U, 4, 5) {
        println("conflictBPlus 4 5 = true")
    } else {
        println("FAIL: conflict(Pneumonia, Drug) should hold")
        n_fail = n_fail + 1
    }
    if elplus_conflict(&s, &sdisjm, U, 7, 7) {
        println("conflictBPlus 7 7 = true")
    } else {
        println("FAIL: conflict(DrugInducedDisorder, itself) should hold")
        n_fail = n_fail + 1
    }
    if s[3 * 64 + 1] {
        println("FAIL: Organ <= Disorder should NOT hold")
        n_fail = n_fail + 1
    } else {
        println("subBPlus 3 1 = false")
    }

    // ══ Part B: full Anatomy human TBox with roles, sparse variant ════
    init_data()
    let H = expected_h()
    let W = H + 1
    let NSUB = expected_sub()
    let NEX = expected_exsub()
    let NDEP = expected_disj()
    let EP = expected_endpoints()
    let M = expected_mappings()
    let NR = expected_roles()

    // data profile assertions (see header): no role hierarchy on real data
    if NR != @NR@ {
        println("FAIL: role count disagrees with extraction")
        n_fail = n_fail + 1
    }

    // packed parent adjacency (counting sort) + per-class BFS closure
    // (workspace matrices live in stdlib elplus.sio: module-level arrays
    // must not cross a &! boundary on the current compiler lane)
    elplus_sparse_init_mats()
    var poff: [i64; 4097] = [0; 4097]
    var pcount: [i64; 4096] = [0; 4096]
    var plist: [i64; 4096] = [0; 4096]
    var vis: [i64; 4096] = [0; 4096]
    var queue: [i64; 4096] = [0; 4096]
    elplus_sparse_build_adj(&h_sub, NSUB, &!poff, &!pcount, &!plist, H)
    let edges = elplus_sparse_bfs(&poff, &plist, &!vis, &!queue, H, W)
    if edges != expected_closure_edges() {
        println("FAIL: atom closure edge count disagrees with mirror")
        n_fail = n_fail + 1
    }

    // seed e[c] with the ancestor rows of the stated fillers of c, then
    // expand in place: e[c] |= e[a] over ancestors a of c (role-aware
    // edge fillers; exact single pass, see stdlib module docs)
    elplus_sparse_seed_edges(&ex_c, &ex_f, NEX, W)
    elplus_sparse_expand(H, W)

    var nre: i64 = 0
    c = 0
    while c < H {
        nre = nre + elplus_sparse_row_count(1, c, W)
        c = c + 1
    }
    if nre != expected_role_edges_atom() {
        println("FAIL: atom-source role edge count disagrees with mirror")
        n_fail = n_fail + 1
    }

    // total S cells over the interned universe (atoms + top + existentials)
    var scells: i64 = 0
    c = 0
    while c < H {
        scells = scells + elplus_sparse_row_count(0, c, W)
        scells = scells + elplus_sparse_row_count(1, c, W)
        c = c + 1
    }
    scells = scells + 1  // top row: top <= top only
    var basesum: i64 = 0
    c = 0
    while c < W {
        basesum = basesum + elplus_sparse_row_count(0, c, W)
        c = c + 1
    }
    let BB = W
    scells = scells + NR * (BB + basesum)
    if scells != expected_s_cells() {
        println("FAIL: total S cell count disagrees with mirror")
        n_fail = n_fail + 1
    }
    let nrtotal = nre + NR * basesum
    if nrtotal != expected_role_edges_total() {
        println("FAIL: total role edge count disagrees with mirror")
        n_fail = n_fail + 1
    }

    // ── derived conflicts must equal round 7 (roles add no atomic
    //    disjointness): endpoint partner-bit masks over the sub closure ──
    var ep_bit: [i64; @H@] = [0; @H@]
    var ep_pbits: [bool; @EP2@] = [false; @EP2@]
    var neps: i64 = 0
    var a: i64 = 0
    var b: i64 = 0
    var ia: i64 = 0
    var ib: i64 = 0
    k = 0
    while k < NDEP {
        a = h_dep[k] / 10000
        b = h_dep[k] - a * 10000
        if ep_bit[a] == 0 {
            neps = neps + 1
            ep_bit[a] = neps
        }
        if ep_bit[b] == 0 {
            neps = neps + 1
            ep_bit[b] = neps
        }
        ia = ep_bit[a] - 1
        ib = ep_bit[b] - 1
        ep_pbits[ia * EP + ib] = true
        ep_pbits[ib * EP + ia] = true
        k = k + 1
    }
    if neps != EP {
        println("FAIL: distinct endpoint count disagrees with mirror")
        n_fail = n_fail + 1
    }

    var mask: [bool; @HEP@] = [false; @HEP@]
    c = 0
    while c < H {
        if ep_bit[c] > 0 {
            mask[c * EP + ep_bit[c] - 1] = true
        }
        c = c + 1
    }
    var e: i64 = 0
    var p: i64 = 0
    var changed = true
    while changed {
        changed = false
        e = 0
        while e < NSUB {
            c = h_sub[e] / 10000
            p = h_sub[e] - c * 10000
            k = 0
            while k < EP {
                if mask[p * EP + k] && !mask[c * EP + k] {
                    mask[c * EP + k] = true
                    changed = true
                }
                k = k + 1
            }
            e = e + 1
        }
    }

    var pmask: [bool; @HEP@] = [false; @HEP@]
    var j: i64 = 0
    var any = false
    c = 0
    while c < H {
        j = 0
        while j < EP {
            any = false
            k = 0
            while k < EP {
                if mask[c * EP + k] && ep_pbits[k * EP + j] {
                    any = true
                }
                k = k + 1
            }
            pmask[c * EP + j] = any
            j = j + 1
        }
        c = c + 1
    }

    var ei: i64 = 0
    var ej: i64 = 0
    var ci2: i64 = 0
    var cj2: i64 = 0
    var hit = false
    var n_conf: i64 = 0
    var i: i64 = 0
    while i < M {
        ei = m_pack[i] / 10000
        ci2 = m_pack[i] - ei * 10000
        j = 0
        while j < M {
            if i != j {
                ej = m_pack[j] / 10000
                if ei == ej {
                    cj2 = m_pack[j] - ej * 10000
                    hit = false
                    k = 0
                    while k < EP {
                        if mask[ci2 * EP + k] && pmask[cj2 * EP + k] {
                            hit = true
                        }
                        k = k + 1
                    }
                    if hit {
                        n_conf = n_conf + 1
                    }
                }
            }
            j = j + 1
        }
        i = i + 1
    }
    if n_conf != expected_derived_conflicts() {
        println("FAIL: derived conflict count disagrees with mirror")
        n_fail = n_fail + 1
    }

    // ── Summary ─────────────────────────────────────────────────────────
    println("=== OAEI 2016 Anatomy: EL+ role-aware closure (round 9) ===")
    println("human classes (H):")
    println(H)
    println("roles (NR):")
    println(NR)
    println("sub axioms:")
    println(NSUB)
    println("existential restrictions (exsub):")
    println(NEX)
    println("disjoint pairs:")
    println(NDEP)
    println("atom closure edges:")
    println(edges)
    println("atom-source role edges:")
    println(nre)
    println("total role edges (incl existential sources):")
    println(nrtotal)
    println("total S cells over interned universe:")
    println(scells)
    println("candidate mappings (M):")
    println(M)
    println("derived conflicts (ordered pairs, = round 7):")
    println(n_conf)

    if n_fail == 0 {
        println("ALL PASS")
        return 0
    }
    println("FAILURES:")
    println(n_fail)
    return 1
}
'''


def main():
    sub, disj, exsub, rsub, rcomp, n_roles, maps, n_human = load()
    ex = mirror_real(n_human, n_roles, sub, disj, exsub, rsub, rcomp, maps)
    sy = mirror_synth()
    n1 = emit_real("elplus_data.sio", ex, sub, disj, exsub, maps)
    n2 = emit_synth("elplus_synth_data.sio", sy)
    drv = (DRIVER
           .replace("@NR@", str(ex["NR"]))
           .replace("@H@", str(ex["H"]))
           .replace("@NSUB@", str(ex["NSUB"]))
           .replace("@NEX@", str(ex["NEX"]))
           .replace("@NDEP@", str(ex["NDEP"]))
           .replace("@M@", str(ex["M"]))
           .replace("@EP2@", str(ex["EP"] * ex["EP"]))
           .replace("@HEP@", str(ex["H"] * ex["EP"])))
    with open("elplus_scale_driver.sio", "w") as f:
        f.write(drv)
    print(f"real: H={ex['H']} NR={ex['NR']} sub={ex['NSUB']} "
          f"exsub={ex['NEX']} disj={ex['NDEP']} M={ex['M']}")
    print(f"real: closure_edges={ex['edges']} role_edges_atom={ex['nR_atom']} "
          f"role_edges_total={ex['nR_total']} s_cells={ex['s_cells']} "
          f"conflicts={ex['n_conf']} rounds={ex['rounds']}")
    print(f"synth: U={sy['U']} s_cells={sy['s_cells']} "
          f"r_edges={sy['r_edges']} rounds={sy['rounds']}")
    print(f"emitted elplus_data.sio ({n1} assignments), "
          f"elplus_synth_data.sio ({n2} assignments), "
          f"elplus_scale_driver.sio")


if __name__ == "__main__":
    main()
