#!/usr/bin/env python3
"""gen_full_data.py — round-7 scale push: generate the FULL OAEI 2016
Anatomy human TBox data modules (NO ancestor cap; all 3,304 human classes)
plus two Sounio drivers, and run an independent python mirror whose numbers
are embedded as expected_*() functions.

Strategies (see SCALE_RESULTS.md):

  A. DENSE  — the round-6 algorithm unchanged: three H x H bool matrices
     (h_disj, clos, disj_c) and an O(H^2 + edges*H)-per-pass naive fixpoint.
     Emitted as dense_full_data.sio + dense_full_driver.sio.  At H = 3,304
     each matrix has 10,911,616 cells.

  B. SPARSE — no N^2 matrix anywhere.  Only 9 distinct classes appear in
     the 17 disjointWith axioms, so "disjoint-reachable" is computed as a
     bool[H*EP] ancestor-mask fixpoint over the 3,761 sub edges (EP = 9
     endpoint bits; an endpoint may be disjoint with several others, so
     each endpoint carries a partner-bit SET, ep_pbits).  The full closure
     edge count is obtained by per-class BFS over a packed parent-adjacency
     (counting sort).  Emitted as full_data.sio + full_scale_driver.sio.

Both drivers end with ALL PASS only if every internal sanity check holds
AND every number equals this mirror.  Both must agree with round 6 on the
referenced subgraph (the round-6 cap was ancestor-closed and therefore
lossless for this pipeline): 736 ordered conflicts, 6,392 kept, 246
dropped, top-5 dropped ids 45, 46, 52, 56, 77.

Mirror cross-check: conflicts are computed twice, set-based (N^2 disjC,
the round-6 definition) and bitmask-based (the sparse strategy); the
script aborts unless they agree exactly.

Known compiler workarounds applied (verified round 6, REAL_RESULTS.md):
  * init assignments chunked at 500 statements/function (>682 silently
    dropped);
  * module-level splat arrays have garbage leading cells (bool 0..2,
    i64/f64 0) -> data arrays are fully assigned; m_keep gets explicit
    fixup writes first;
  * f64 array stores outside main are no-ops -> confidences are exact i64
    per-10000;
  * multimodule thin-link dies beyond ~24k assignments -> pairs packed as
    a*10000+b (all ids < 10000).
"""

import sys

H_CAP_NONE = None


def load():
    sub, disj = [], set()
    with open("../tbox.txt") as f:
        for line in f:
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
    return sub, sorted(disj), maps, n_human


def mirror(H, sub, disj, maps):
    """Return dict of expected values; set-based + bitmask cross-check."""
    # full set-based closure (round-6 semantics: reflexive + sub edges)
    clos = [set([c]) for c in range(H)]
    for a, b in sub:
        clos[a].add(b)
    changed, passes = True, 0
    while changed:
        changed, passes = False, passes + 1
        for a in range(H):
            ca = clos[a]
            add = set()
            for b in ca:
                add |= clos[b]
            if not add <= ca:
                ca |= add
                changed = True
    edges = sum(len(s) for s in clos)

    disjC = set()
    for d1, d2 in disj:
        for c1 in range(H):
            if d1 in clos[c1]:
                for c2 in range(H):
                    if d2 in clos[c2]:
                        disjC.add((c1, c2))
                        disjC.add((c2, c1))

    M = len(maps)

    def conflict_set(i, j):
        return (maps[i][0] == maps[j][0]
                and (maps[i][1], maps[j][1]) in disjC)

    n_conf = sum(1 for i in range(M) for j in range(M)
                 if i != j and conflict_set(i, j))

    # bitmask cross-check (the sparse strategy, multi-partner endpoints)
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
    changed, mask_passes = True, 0
    while changed:
        changed, mask_passes = False, mask_passes + 1
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
        sys.exit(f"MIRROR CROSS-CHECK FAILED: set-based {n_conf} != "
                 f"bitmask {n_conf_bit}")

    # greedy repair (identical iteration order to the drivers)
    keep = [True] * M
    for i in range(M):
        for j in range(i + 1, M):
            if keep[i] and keep[j] and conflict_set(i, j):
                if maps[i][2] >= maps[j][2]:
                    keep[j] = False
                else:
                    keep[i] = False
    n_kept = sum(keep)
    dropped = sorted((maps[i][2], i) for i in range(M) if not keep[i])
    return {
        "H": H, "NSUB": len(sub), "NDEP": len(disj), "EP": E2,
        "M": M, "edges": edges, "passes": passes,
        "mask_passes": mask_passes, "n_conf": n_conf,
        "n_kept": n_kept, "n_dropped": M - n_kept,
        "drop5": [i for _, i in dropped[:5]],
    }


def emit_expected(f, ex):
    f.write(f"pub fn expected_h() -> i64 {{ return {ex['H']} }}\n")
    f.write(f"pub fn expected_sub() -> i64 {{ return {ex['NSUB']} }}\n")
    f.write(f"pub fn expected_disj() -> i64 {{ return {ex['NDEP']} }}\n")
    f.write(f"pub fn expected_endpoints() -> i64 {{ return {ex['EP']} }}\n")
    f.write(f"pub fn expected_mappings() -> i64 {{ return {ex['M']} }}\n")
    f.write(f"pub fn expected_closure_edges() -> i64 {{ return {ex['edges']} }}\n")
    f.write(f"pub fn expected_mask_passes() -> i64 {{ return {ex['mask_passes']} }}\n")
    f.write(f"pub fn expected_derived_conflicts() -> i64 {{ return {ex['n_conf']} }}\n")
    f.write(f"pub fn expected_kept() -> i64 {{ return {ex['n_kept']} }}\n")
    f.write(f"pub fn expected_dropped() -> i64 {{ return {ex['n_dropped']} }}\n")
    f.write("pub fn expected_drop5(k: i64) -> i64 {\n")
    for k, i in enumerate(ex["drop5"]):
        f.write(f"    if k == {k} {{ return {i} }}\n")
    f.write("    return 0 - 1\n}\n\n")


def emit_init(f, assigns, chunk=500):
    f.write("pub fn init_data() {\n")
    n = 0
    for i in range(0, len(assigns), chunk):
        f.write(f"    init_chunk_{n}()\n")
        n += 1
    f.write("}\n\n")
    n = 0
    for i in range(0, len(assigns), chunk):
        f.write(f"pub fn init_chunk_{n}() {{\n")
        for stmt in assigns[i:i + chunk]:
            f.write(f"    {stmt}\n")
        f.write("}\n\n")
        n += 1


def emit_sparse(path, ex, sub, disj, maps):
    H, NSUB, M = ex["H"], ex["NSUB"], ex["M"]
    with open(path, "w") as f:
        f.write("// GENERATED by gen_full_data.py — do not edit by hand.\n")
        f.write("// FULL OAEI 2016 Anatomy human TBox (no cap), SPARSE\n")
        f.write("// strategy: no H x H matrix; arrays are edge/endpoint/\n")
        f.write("// mapping lists only.  Mirror values embedded below.\n")
        for k_, v in (("H", ex["H"]), ("sub", NSUB), ("disj", ex["NDEP"]),
                      ("endpoints", ex["EP"]), ("M", M),
                      ("closure_edges", ex["edges"]),
                      ("conflicts", ex["n_conf"]),
                      ("kept", ex["n_kept"]), ("dropped", ex["n_dropped"])):
            f.write(f"// {k_} = {v}\n\n" if k_ == "H" else f"// {k_} = {v}\n")
        f.write("\n")
        emit_expected(f, ex)
        f.write(f"pub var h_sub: [i64; {NSUB}] = [0; {NSUB}]  // child*10000+parent\n")
        f.write(f"pub var h_dep: [i64; {ex['NDEP']}] = [0; {ex['NDEP']}]  // a*10000+b\n")
        f.write(f"pub var m_pack: [i64; {M}] = [0; {M}]  // mouse_ent*10000+human_cls\n")
        f.write(f"pub var m_conf: [i64; {M}] = [0; {M}]  // per-10000\n")
        f.write(f"pub var m_keep: [bool; {M}] = [true; {M}]\n\n")
        assigns = ["m_keep[0] = true", "m_keep[1] = true", "m_keep[2] = true"]
        for k_, (a, b) in enumerate(sub):
            assigns.append(f"h_sub[{k_}] = {a * 10000 + b}")
        for k_, (a, b) in enumerate(disj):
            assigns.append(f"h_dep[{k_}] = {a * 10000 + b}")
        for k_, (m, h, c) in enumerate(maps):
            assigns.append(f"m_pack[{k_}] = {m * 10000 + h}")
            assigns.append(f"m_conf[{k_}] = {c}")
        emit_init(f, assigns)
    return len(assigns)


def emit_dense(path, ex, sub, disj, maps):
    H, NSUB, M = ex["H"], ex["NSUB"], ex["M"]
    with open(path, "w") as f:
        f.write("// GENERATED by gen_full_data.py --dense — do not edit.\n")
        f.write("// FULL OAEI 2016 Anatomy human TBox (no cap), DENSE\n")
        f.write("// strategy: three H x H bool matrices (round-6 algorithm).\n")
        emit_expected(f, ex)
        f.write(f"pub var h_sub: [i64; {NSUB}] = [0; {NSUB}]  // child*10000+parent\n")
        f.write(f"pub var h_disj: [bool; {H*H}] = [false; {H*H}]\n")
        f.write(f"pub var clos: [bool; {H*H}] = [false; {H*H}]\n")
        f.write(f"pub var disj_c: [bool; {H*H}] = [false; {H*H}]\n")
        f.write(f"pub var m_pack: [i64; {M}] = [0; {M}]\n")
        f.write(f"pub var m_conf: [i64; {M}] = [0; {M}]\n")
        f.write(f"pub var m_keep: [bool; {M}] = [true; {M}]\n\n")
        assigns = [f"{a}[{i}] = false" for a in ("clos", "disj_c", "h_disj")
                   for i in range(3)]
        assigns += [f"m_keep[{i}] = true" for i in range(3)]
        for k_, (a, b) in enumerate(sub):
            assigns.append(f"h_sub[{k_}] = {a * 10000 + b}")
        for a, b in disj:
            assigns.append(f"h_disj[{a} * {H} + {b}] = true")
            assigns.append(f"h_disj[{b} * {H} + {a}] = true")
        for k_, (m, h, c) in enumerate(maps):
            assigns.append(f"m_pack[{k_}] = {m * 10000 + h}")
            assigns.append(f"m_conf[{k_}] = {c}")
        emit_init(f, assigns)
    return len(assigns)


# ── driver templates (@TOKENS@ substituted from the mirror) ──────────────

SPARSE_DRIVER = '''//@ run-pass
//@ expect-stdout: ALL PASS
// Round-7 scale push, SPARSE strategy: the verified pipeline (closure ->
// derived conflicts -> greedy epistemic repair) on the FULL OAEI 2016
// Anatomy human TBox (H = @H@ classes, NO cap) without any H x H matrix:
//
//   * only @EP@ distinct classes appear in the @NDEP@ disjointWith axioms,
//     so "disjoint-reachable" is a bool[H*EP] ancestor-mask fixpoint over
//     the @NSUB@ sub edges (an endpoint can be disjoint with several
//     others -> partner-bit SETS, ep_pbits);
//   * the full closure edge count comes from per-class BFS over a packed
//     parent adjacency (counting sort), NOT from an N^2 matrix;
//   * conflict(i,j) = same mouse entity AND
//     (mask[ci] & pmask[cj]) != 0  -- proved equivalent to the round-6
//     set-based definition by the mirror cross-check in gen_full_data.py.
//
// Data: full_data.sio (generated; mirror numbers embedded as expected_*).
// Same sanity checks as round 6: closure edge count + mask reflexivity,
// conflict symmetry + exact counts, conflict-free repair with maximality
// witnesses, top-5 dropped -- all equal to the independent python mirror.

import full_data::*

fn main() -> i32 with IO, Mut, Div, Panic {
    var n_fail = 0

    init_data()

    let H = expected_h()
    let NSUB = expected_sub()
    let NDEP = expected_disj()
    let EP = expected_endpoints()
    let M = expected_mappings()

    // ── Endpoint indexing + partner bit-sets ────────────────────────────
    var ep_bit: [i64; @H@] = [0; @H@]        // class -> endpoint bit+1 (0 = none)
    var ep_pbits: [bool; @EP2@] = [false; @EP2@]  // partner set per endpoint bit
    var neps: i64 = 0
    var a: i64 = 0
    var b: i64 = 0
    var ia: i64 = 0
    var ib: i64 = 0
    var k: i64 = 0
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

    // ── Ancestor-disjointness mask fixpoint (sparse closure) ────────────
    var mask: [bool; @HEP@] = [false; @HEP@]
    var c: i64 = 0
    while c < H {
        if ep_bit[c] > 0 {
            mask[c * EP + ep_bit[c] - 1] = true
        }
        c = c + 1
    }
    var e: i64 = 0
    var p: i64 = 0
    var mp: i64 = 0
    var changed = true
    while changed {
        changed = false
        mp = mp + 1
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
    if mp != expected_mask_passes() {
        println("FAIL: mask fixpoint passes disagree with mirror")
        n_fail = n_fail + 1
    }

    // ── Check (1b): mask reflexive on every endpoint ────────────────────
    c = 0
    while c < H {
        if ep_bit[c] > 0 {
            if !mask[c * EP + ep_bit[c] - 1] {
                println("FAIL: endpoint mask not reflexive")
                n_fail = n_fail + 1
            }
        }
        c = c + 1
    }

    // ── pmask[c][j] = OR_k ( mask[c][k] AND ep_pbits[k][j] ) ────────────
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

    // ── Full closure edge count: per-class BFS over parent adjacency ────
    var pcount: [i64; @H@] = [0; @H@]
    var poff: [i64; @H1@] = [0; @H1@]
    var plist: [i64; @NSUB@] = [0; @NSUB@]
    var vis: [i64; @H@] = [0; @H@]
    var queue: [i64; @H@] = [0; @H@]
    e = 0
    while e < NSUB {
        c = h_sub[e] / 10000
        pcount[c] = pcount[c] + 1
        e = e + 1
    }
    c = 0
    while c < H {
        poff[c + 1] = poff[c] + pcount[c]
        c = c + 1
    }
    c = 0
    while c < H {
        pcount[c] = poff[c]
        c = c + 1
    }
    e = 0
    while e < NSUB {
        c = h_sub[e] / 10000
        p = h_sub[e] - c * 10000
        plist[pcount[c]] = p
        pcount[c] = pcount[c] + 1
        e = e + 1
    }
    var total_edges: i64 = 0
    var head: i64 = 0
    var tail: i64 = 0
    var t: i64 = 0
    var s: i64 = 0
    while s < H {
        head = 0
        tail = 0
        queue[tail] = s
        tail = tail + 1
        vis[s] = s + 1
        while head < tail {
            b = queue[head]
            head = head + 1
            total_edges = total_edges + 1
            t = poff[b]
            while t < poff[b + 1] {
                p = plist[t]
                if vis[p] != s + 1 {
                    vis[p] = s + 1
                    queue[tail] = p
                    tail = tail + 1
                }
                t = t + 1
            }
        }
        s = s + 1
    }
    // ── Check (1a): closure edge count equals the mirror ────────────────
    if total_edges != expected_closure_edges() {
        println("FAIL: closure edge count disagrees with python mirror")
        n_fail = n_fail + 1
    }

    // ── Derived conflicts: same mouse entity, disjoint-reachable targets ─
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
        println("FAIL: derived conflict count disagrees with python mirror")
        n_fail = n_fail + 1
    }

    // ── Check (2): derived conflict relation is symmetric ───────────────
    var cij = false
    var cji = false
    i = 0
    while i < M {
        ei = m_pack[i] / 10000
        ci2 = m_pack[i] - ei * 10000
        j = i + 1
        while j < M {
            ej = m_pack[j] / 10000
            cj2 = m_pack[j] - ej * 10000
            if ei == ej {
                cij = false
                cji = false
                k = 0
                while k < EP {
                    if mask[ci2 * EP + k] && pmask[cj2 * EP + k] {
                        cij = true
                    }
                    if mask[cj2 * EP + k] && pmask[ci2 * EP + k] {
                        cji = true
                    }
                    k = k + 1
                }
                if (cij && !cji) || (!cij && cji) {
                    println("FAIL: derived conflict relation not symmetric")
                    n_fail = n_fail + 1
                }
            }
            j = j + 1
        }
        i = i + 1
    }

    // ── Greedy epistemic repair (drop weaker of each live conflict) ─────
    i = 0
    while i < M {
        ei = m_pack[i] / 10000
        ci2 = m_pack[i] - ei * 10000
        j = i + 1
        while j < M {
            if m_keep[i] && m_keep[j] {
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
                        if m_conf[i] >= m_conf[j] {
                            m_keep[j] = false
                        } else {
                            m_keep[i] = false
                        }
                    }
                }
            }
            j = j + 1
        }
        i = i + 1
    }

    var n_kept: i64 = 0
    i = 0
    while i < M {
        if m_keep[i] {
            n_kept = n_kept + 1
        }
        i = i + 1
    }
    let n_dropped = M - n_kept
    if n_kept != expected_kept() {
        println("FAIL: kept count disagrees with python mirror")
        n_fail = n_fail + 1
    }
    if n_dropped != expected_dropped() {
        println("FAIL: dropped count disagrees with python mirror")
        n_fail = n_fail + 1
    }

    // ── Check (3a): retained set is conflict-free ───────────────────────
    i = 0
    while i < M {
        ei = m_pack[i] / 10000
        ci2 = m_pack[i] - ei * 10000
        j = i + 1
        while j < M {
            if m_keep[i] && m_keep[j] {
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
                        println("FAIL: retained set still has a conflict")
                        n_fail = n_fail + 1
                    }
                }
            }
            j = j + 1
        }
        i = i + 1
    }

    // ── Check (3b): every dropped mapping has a maximality witness ──────
    var witnessed = false
    i = 0
    while i < M {
        if !m_keep[i] {
            ei = m_pack[i] / 10000
            ci2 = m_pack[i] - ei * 10000
            witnessed = false
            j = 0
            while j < M {
                if m_keep[j] {
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
                            if m_conf[j] >= m_conf[i] {
                                witnessed = true
                            }
                        }
                    }
                }
                j = j + 1
            }
            if !witnessed {
                println("FAIL: dropped mapping lacks witness")
                n_fail = n_fail + 1
            }
        }
        i = i + 1
    }

    // ── Check (4): top-5 dropped by confidence (lowest first) ───────────
    var chosen: [i64; 5] = [0 - 1; 5]
    var best_id: i64 = 0 - 1
    var best_conf: i64 = 20000
    var already = false
    k = 0
    while k < 5 {
        best_id = 0 - 1
        best_conf = 20000
        i = 0
        while i < M {
            if !m_keep[i] {
                already = false
                t = 0
                while t < k {
                    if chosen[t] == i {
                        already = true
                    }
                    t = t + 1
                }
                if !already {
                    if m_conf[i] < best_conf || (m_conf[i] == best_conf && (best_id < 0 || i < best_id)) {
                        best_conf = m_conf[i]
                        best_id = i
                    }
                }
            }
            i = i + 1
        }
        chosen[k] = best_id
        if best_id != expected_drop5(k) {
            println("FAIL: top-5 dropped disagrees with python mirror")
            n_fail = n_fail + 1
        }
        k = k + 1
    }

    // ── Summary ─────────────────────────────────────────────────────────
    println("=== OAEI 2016 Anatomy: FULL-TBox scale push (sparse) ===")
    println("human classes (H):")
    println(H)
    println("sub axioms:")
    println(NSUB)
    println("disjoint pairs:")
    println(NDEP)
    println("distinct disjoint endpoints:")
    println(EP)
    println("closure edges (full TBox, BFS count):")
    println(total_edges)
    println("mask fixpoint passes:")
    println(mp)
    println("candidate mappings (M):")
    println(M)
    println("derived conflicts (ordered pairs):")
    println(n_conf)
    println("kept:")
    println(n_kept)
    println("dropped:")
    println(n_dropped)
    println("top-5 dropped by confidence (lowest first; conf per-10000):")
    k = 0
    while k < 5 {
        println(chosen[k])
        println(m_conf[chosen[k]])
        k = k + 1
    }

    if n_fail == 0 {
        println("ALL PASS")
        return 0
    }
    println("FAILURES:")
    println(n_fail)
    return 1
}
'''

DENSE_DRIVER = '''//@ run-pass
//@ expect-stdout: ALL PASS
// Round-7 scale push, DENSE strategy: the round-6 algorithm UNCHANGED
// (three H x H bool matrices + naive fixpoint) on the FULL OAEI 2016
// Anatomy human TBox (H = @H@, no cap; H*H = @HH@ cells per matrix).
// This is the strategy-A ceiling probe: it documents whether the dense
// approach survives full scale.  Data: dense_full_data.sio (generated).

import dense_full_data::*

fn main() -> i32 with IO, Mut, Div, Panic {
    var n_fail = 0

    init_data()

    let H = expected_h()
    let N_SUB = expected_sub()
    let M = expected_mappings()

    // ── Subsumption closure fixpoint over the full human TBox ───────────
    var c: i64 = 0
    while c < H {
        clos[c * H + c] = true
        c = c + 1
    }
    var e: i64 = 0
    var pk: i64 = 0
    var lo: i64 = 0
    while e < N_SUB {
        pk = h_sub[e]
        lo = pk - (pk / 10000) * 10000
        clos[(pk / 10000) * H + lo] = true
        e = e + 1
    }
    var changed = true
    while changed {
        changed = false
        var a: i64 = 0
        while a < H {
            var b: i64 = 0
            while b < H {
                if clos[a * H + b] {
                    var d: i64 = 0
                    while d < H {
                        if clos[b * H + d] {
                            if !clos[a * H + d] {
                                clos[a * H + d] = true
                                changed = true
                            }
                        }
                        d = d + 1
                    }
                }
                b = b + 1
            }
            a = a + 1
        }
    }

    // ── Check (1a): closure edge count equals the python mirror ─────────
    var total_edges: i64 = 0
    var idx: i64 = 0
    while idx < H * H {
        if clos[idx] {
            total_edges = total_edges + 1
        }
        idx = idx + 1
    }
    if total_edges != expected_closure_edges() {
        println("FAIL: closure edge count disagrees with python mirror")
        n_fail = n_fail + 1
    }

    // ── Check (1b): closure reflexive on the diagonal ───────────────────
    c = 0
    while c < H {
        if !clos[c * H + c] {
            println("FAIL: closure not reflexive on diagonal")
            n_fail = n_fail + 1
        }
        c = c + 1
    }

    // ── Disjointness expanded through the closure ───────────────────────
    var d1: i64 = 0
    while d1 < H {
        var d2: i64 = 0
        while d2 < H {
            if h_disj[d1 * H + d2] {
                var c1: i64 = 0
                while c1 < H {
                    if clos[c1 * H + d1] {
                        var c2: i64 = 0
                        while c2 < H {
                            if clos[c2 * H + d2] {
                                disj_c[c1 * H + c2] = true
                                disj_c[c2 * H + c1] = true
                            }
                            c2 = c2 + 1
                        }
                    }
                    c1 = c1 + 1
                }
            }
            d2 = d2 + 1
        }
        d1 = d1 + 1
    }

    // ── Derived conflicts: same mouse entity, disjoint human targets ────
    var ei: i64 = 0
    var ci2: i64 = 0
    var ej: i64 = 0
    var cj2: i64 = 0
    var n_conf: i64 = 0
    var i: i64 = 0
    while i < M {
        ei = m_pack[i] / 10000
        ci2 = m_pack[i] - ei * 10000
        var j: i64 = 0
        while j < M {
            if i != j {
                ej = m_pack[j] / 10000
                if ei == ej {
                    cj2 = m_pack[j] - ej * 10000
                    if disj_c[ci2 * H + cj2] {
                        n_conf = n_conf + 1
                    }
                }
            }
            j = j + 1
        }
        i = i + 1
    }
    if n_conf != expected_derived_conflicts() {
        println("FAIL: derived conflict count disagrees with python mirror")
        n_fail = n_fail + 1
    }

    // ── Check (2): derived conflict relation is symmetric ───────────────
    var cij = false
    var cji = false
    i = 0
    while i < M {
        ei = m_pack[i] / 10000
        ci2 = m_pack[i] - ei * 10000
        var j: i64 = i + 1
        while j < M {
            ej = m_pack[j] / 10000
            cj2 = m_pack[j] - ej * 10000
            cij = ei == ej && disj_c[ci2 * H + cj2]
            cji = ei == ej && disj_c[cj2 * H + ci2]
            if (cij && !cji) || (!cij && cji) {
                println("FAIL: derived conflict relation not symmetric")
                n_fail = n_fail + 1
            }
            j = j + 1
        }
        i = i + 1
    }

    // ── Greedy epistemic repair (drop weaker of each live conflict) ─────
    i = 0
    while i < M {
        ei = m_pack[i] / 10000
        ci2 = m_pack[i] - ei * 10000
        var j: i64 = i + 1
        while j < M {
            if m_keep[i] && m_keep[j] {
                ej = m_pack[j] / 10000
                if ei == ej {
                    cj2 = m_pack[j] - ej * 10000
                    if disj_c[ci2 * H + cj2] {
                        if m_conf[i] >= m_conf[j] {
                            m_keep[j] = false
                        } else {
                            m_keep[i] = false
                        }
                    }
                }
            }
            j = j + 1
        }
        i = i + 1
    }

    var n_kept: i64 = 0
    i = 0
    while i < M {
        if m_keep[i] {
            n_kept = n_kept + 1
        }
        i = i + 1
    }
    let n_dropped = M - n_kept
    if n_kept != expected_kept() {
        println("FAIL: kept count disagrees with python mirror")
        n_fail = n_fail + 1
    }
    if n_dropped != expected_dropped() {
        println("FAIL: dropped count disagrees with python mirror")
        n_fail = n_fail + 1
    }

    // ── Check (3a): retained set is conflict-free ───────────────────────
    i = 0
    while i < M {
        ei = m_pack[i] / 10000
        ci2 = m_pack[i] - ei * 10000
        var j: i64 = i + 1
        while j < M {
            if m_keep[i] && m_keep[j] {
                ej = m_pack[j] / 10000
                if ei == ej {
                    cj2 = m_pack[j] - ej * 10000
                    if disj_c[ci2 * H + cj2] {
                        println("FAIL: retained set still has a conflict")
                        n_fail = n_fail + 1
                    }
                }
            }
            j = j + 1
        }
        i = i + 1
    }

    // ── Check (3b): every dropped mapping has a maximality witness ──────
    i = 0
    while i < M {
        if !m_keep[i] {
            ei = m_pack[i] / 10000
            ci2 = m_pack[i] - ei * 10000
            var witnessed = false
            var j: i64 = 0
            while j < M {
                if m_keep[j] {
                    ej = m_pack[j] / 10000
                    if ei == ej {
                        cj2 = m_pack[j] - ej * 10000
                        if disj_c[ci2 * H + cj2] {
                            if m_conf[j] >= m_conf[i] {
                                witnessed = true
                            }
                        }
                    }
                }
                j = j + 1
            }
            if !witnessed {
                println("FAIL: dropped mapping lacks witness")
                n_fail = n_fail + 1
            }
        }
        i = i + 1
    }

    // ── Check (4): top-5 dropped by confidence (lowest first) ───────────
    var chosen: [i64; 5] = [0 - 1; 5]
    var k: i64 = 0
    while k < 5 {
        var best_id: i64 = 0 - 1
        var best_conf: i64 = 20000
        i = 0
        while i < M {
            if !m_keep[i] {
                var already = false
                var t: i64 = 0
                while t < k {
                    if chosen[t] == i {
                        already = true
                    }
                    t = t + 1
                }
                if !already {
                    if m_conf[i] < best_conf || (m_conf[i] == best_conf && (best_id < 0 || i < best_id)) {
                        best_conf = m_conf[i]
                        best_id = i
                    }
                }
            }
            i = i + 1
        }
        chosen[k] = best_id
        if best_id != expected_drop5(k) {
            println("FAIL: top-5 dropped disagrees with python mirror")
            n_fail = n_fail + 1
        }
        k = k + 1
    }

    // ── Summary ─────────────────────────────────────────────────────────
    println("=== OAEI 2016 Anatomy: FULL-TBox scale push (dense) ===")
    println("human classes (H):")
    println(H)
    println("sub axioms:")
    println(N_SUB)
    println("disjoint pairs:")
    println(expected_disj())
    println("closure edges (full TBox):")
    println(total_edges)
    println("candidate mappings (M):")
    println(M)
    println("derived conflicts (ordered pairs):")
    println(n_conf)
    println("kept:")
    println(n_kept)
    println("dropped:")
    println(n_dropped)
    println("top-5 dropped by confidence (lowest first; conf per-10000):")
    k = 0
    while k < 5 {
        println(chosen[k])
        println(m_conf[chosen[k]])
        k = k + 1
    }

    if n_fail == 0 {
        println("ALL PASS")
        return 0
    }
    println("FAILURES:")
    println(n_fail)
    return 1
}
'''


def fill(template, ex):
    return (template
            .replace("@H1@", str(ex["H"] + 1))
            .replace("@HEP@", str(ex["H"] * ex["EP"]))
            .replace("@EP2@", str(ex["EP"] * ex["EP"]))
            .replace("@HH@", str(ex["H"] * ex["H"]))
            .replace("@H@", str(ex["H"]))
            .replace("@NSUB@", str(ex["NSUB"]))
            .replace("@NDEP@", str(ex["NDEP"]))
            .replace("@EP@", str(ex["EP"]))
            .replace("@M@", str(ex["M"])))


def main():
    sub, disj, maps, H = load()
    ex = mirror(H, sub, disj, maps)

    print(f"FULL TBox: H={ex['H']} sub={ex['NSUB']} disj={ex['NDEP']} "
          f"endpoints={ex['EP']} M={ex['M']}")
    print(f"mirror: closure edges={ex['edges']} passes={ex['passes']} "
          f"mask passes={ex['mask_passes']}")
    print(f"mirror: conflicts={ex['n_conf']} kept={ex['n_kept']} "
          f"dropped={ex['n_dropped']} drop5={ex['drop5']}")

    n = emit_sparse("full_data.sio", ex, sub, disj, maps)
    print(f"full_data.sio written ({n} init assignments)")
    with open("full_scale_driver.sio", "w") as f:
        f.write(fill(SPARSE_DRIVER, ex))
    print("full_scale_driver.sio written (sparse strategy)")

    n = emit_dense("dense_full_data.sio", ex, sub, disj, maps)
    print(f"dense_full_data.sio written ({n} init assignments, "
          f"H*H={ex['H']*ex['H']})")
    with open("dense_full_driver.sio", "w") as f:
        f.write(fill(DENSE_DRIVER, ex))
    print("dense_full_driver.sio written (dense strategy)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
