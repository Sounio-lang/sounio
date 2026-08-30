#!/usr/bin/env python3
"""gen_probes.py — round-7 ceiling probes (SYNTHETIC ontologies, not the
anatomy data).  Each probe is a single self-contained .sio file (no import,
so the multimodule thin-link statement limit is not a confounder) with a
trivially computable mirror, ending in ALL PASS.

Ontology shape: star hierarchy  sub c 0  for c in 1..N-1, one disjoint
pair (1, 2), M synthetic mappings (K entities x targets {1,2}).

Mirror:
  closure edges = 2N-1   (class 0: {0}; class c>0: {c, 0})
  disjC entries = 2      ((1,2) and (2,1); no subclasses under 1 or 2)
  derived ordered conflicts = 2K  (each entity maps to both 1 and 2)
  greedy repair: per entity, conf of target-1 mapping = 5000, target-2
  = 4000 -> drop the target-2 one: kept = K, dropped = K.

Dense probe: three N x N bool matrices, naive fixpoint (strategy A).
Sparse probe: endpoint-mask fixpoint + BFS edge count (strategy B).
"""

import sys


def chunks(assigns, chunk=500):
    out = []
    for i in range(0, len(assigns), chunk):
        out.append(assigns[i:i + chunk])
    return out


def emit_init(f, assigns):
    cs = chunks(assigns)
    f.write("fn init_data() {\n")
    for i in range(len(cs)):
        f.write(f"    init_chunk_{i}()\n")
    f.write("}\n\n")
    for i, c in enumerate(cs):
        f.write(f"fn init_chunk_{i}() {{\n")
        for stmt in c:
            f.write(f"    {stmt}\n")
        f.write("}\n\n")


def dense_probe(path, N, K):
    M = 2 * K
    with open(path, "w") as f:
        f.write(f"//@ run-pass\n//@ expect-stdout: ALL PASS\n")
        f.write(f"// SYNTHETIC dense ceiling probe: N={N} classes, star\n")
        f.write(f"// hierarchy, N*N={N*N} cells per matrix (x3).\n")
        f.write(f"var h_sub: [i64; {N-1}] = [0; {N-1}]  // child*1000000+parent\n")
        f.write(f"var h_disj: [bool; {N*N}] = [false; {N*N}]\n")
        f.write(f"var clos: [bool; {N*N}] = [false; {N*N}]\n")
        f.write(f"var disj_c: [bool; {N*N}] = [false; {N*N}]\n")
        f.write(f"var m_tgt: [i64; {M}] = [0; {M}]\n")
        f.write(f"var m_conf: [i64; {M}] = [0; {M}]\n")
        f.write(f"var m_keep: [bool; {M}] = [true; {M}]\n\n")
        assigns = [f"{a}[{i}] = false" for a in ("clos", "disj_c", "h_disj")
                   for i in range(3)]
        assigns += [f"m_keep[{i}] = true" for i in range(3)]
        for c in range(1, N):
            assigns.append(f"h_sub[{c-1}] = {c * 1000000}")
        assigns.append(f"h_disj[{N + 2}] = true")
        assigns.append(f"h_disj[{2 * N + 1}] = true")
        for i in range(M):
            assigns.append(f"m_tgt[{i}] = {1 + i % 2}")
            assigns.append(f"m_conf[{i}] = {5000 if i % 2 == 0 else 4000}")
        emit_init(f, assigns)
        f.write(DENSE_MAIN
                .replace("@N@", str(N)).replace("@M@", str(M))
                .replace("@K@", str(K)))


DENSE_MAIN = '''
fn main() -> i32 with IO, Mut, Div, Panic {
    var n_fail = 0
    init_data()
    let N: i64 = @N@
    let M: i64 = @M@

    var c: i64 = 0
    while c < N {
        clos[c * N + c] = true
        c = c + 1
    }
    var e: i64 = 0
    while e < N - 1 {
        c = h_sub[e] / 1000000
        clos[c * N + 0] = true
        e = e + 1
    }
    var changed = true
    while changed {
        changed = false
        var a: i64 = 0
        while a < N {
            var b: i64 = 0
            while b < N {
                if clos[a * N + b] {
                    var d: i64 = 0
                    while d < N {
                        if clos[b * N + d] {
                            if !clos[a * N + d] {
                                clos[a * N + d] = true
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
    var total: i64 = 0
    var idx: i64 = 0
    while idx < N * N {
        if clos[idx] {
            total = total + 1
        }
        idx = idx + 1
    }
    if total != 2 * N - 1 {
        println("FAIL: closure edges")
        n_fail = n_fail + 1
    }

    var d1: i64 = 0
    while d1 < N {
        var d2: i64 = 0
        while d2 < N {
            if h_disj[d1 * N + d2] {
                var c1: i64 = 0
                while c1 < N {
                    if clos[c1 * N + d1] {
                        var c2: i64 = 0
                        while c2 < N {
                            if clos[c2 * N + d2] {
                                disj_c[c1 * N + c2] = true
                                disj_c[c2 * N + c1] = true
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

    var n_conf: i64 = 0
    var i: i64 = 0
    while i < M {
        var j: i64 = 0
        while j < M {
            if i != j {
                if i / 2 == j / 2 {
                    if disj_c[m_tgt[i] * N + m_tgt[j]] {
                        n_conf = n_conf + 1
                    }
                }
            }
            j = j + 1
        }
        i = i + 1
    }
    if n_conf != 2 * @K@ {
        println("FAIL: conflicts")
        n_fail = n_fail + 1
    }

    i = 0
    while i < M {
        var j: i64 = i + 1
        while j < M {
            if m_keep[i] && m_keep[j] {
                if i / 2 == j / 2 {
                    if disj_c[m_tgt[i] * N + m_tgt[j]] {
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
    var kept: i64 = 0
    i = 0
    while i < M {
        if m_keep[i] {
            kept = kept + 1
        }
        i = i + 1
    }
    if kept != @K@ {
        println("FAIL: kept")
        n_fail = n_fail + 1
    }

    println("dense probe N:")
    println(N)
    println("closure edges:")
    println(total)
    println("conflicts:")
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


def sparse_probe(path, N, K):
    M = 2 * K
    with open(path, "w") as f:
        f.write(f"//@ run-pass\n//@ expect-stdout: ALL PASS\n")
        f.write(f"// SYNTHETIC sparse ceiling probe: N={N} classes, star\n")
        f.write(f"// hierarchy; no N*N matrix (endpoint masks + BFS).\n")
        f.write(f"var h_sub: [i64; {N-1}] = [0; {N-1}]  // child*1000000+parent\n")
        f.write(f"var m_tgt: [i64; {M}] = [0; {M}]\n")
        f.write(f"var m_conf: [i64; {M}] = [0; {M}]\n")
        f.write(f"var m_keep: [bool; {M}] = [true; {M}]\n\n")
        assigns = [f"m_keep[{i}] = true" for i in range(3)]
        for c in range(1, N):
            assigns.append(f"h_sub[{c-1}] = {c * 1000000}")
        for i in range(M):
            assigns.append(f"m_tgt[{i}] = {1 + i % 2}")
            assigns.append(f"m_conf[{i}] = {5000 if i % 2 == 0 else 4000}")
        emit_init(f, assigns)
        f.write(SPARSE_MAIN
                .replace("@NSUB@", str(N - 1))
                .replace("@N1@", str(N + 1))
                .replace("@N2@", str(N * 2))
                .replace("@N@", str(N)).replace("@M@", str(M))
                .replace("@K@", str(K)))


SPARSE_MAIN = '''
fn main() -> i32 with IO, Mut, Div, Panic {
    var n_fail = 0
    init_data()
    let N: i64 = @N@
    let M: i64 = @M@
    let EP: i64 = 2

    // endpoint masks: class 1 -> bit 0, class 2 -> bit 1; partners swapped
    var mask: [bool; @N2@] = [false; @N2@]
    mask[1 * EP + 0] = true
    mask[2 * EP + 1] = true
    var changed = true
    var e: i64 = 0
    var c: i64 = 0
    var p: i64 = 0
    var k: i64 = 0
    while changed {
        changed = false
        e = 0
        while e < N - 1 {
            c = h_sub[e] / 1000000
            p = h_sub[e] - c * 1000000
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
    // pmask: partner bits (bit 0 <-> bit 1)
    var pmask: [bool; @N2@] = [false; @N2@]
    c = 0
    while c < N {
        pmask[c * EP + 0] = mask[c * EP + 1]
        pmask[c * EP + 1] = mask[c * EP + 0]
        c = c + 1
    }

    // closure edge count via BFS over parent adjacency (star: cheap)
    var pcount: [i64; @N@] = [0; @N@]
    var poff: [i64; @N1@] = [0; @N1@]
    var plist: [i64; @NSUB@] = [0; @NSUB@]
    var vis: [i64; @N@] = [0; @N@]
    var queue: [i64; @N@] = [0; @N@]
    e = 0
    while e < N - 1 {
        c = h_sub[e] / 1000000
        pcount[c] = pcount[c] + 1
        e = e + 1
    }
    c = 0
    while c < N {
        poff[c + 1] = poff[c] + pcount[c]
        c = c + 1
    }
    c = 0
    while c < N {
        pcount[c] = poff[c]
        c = c + 1
    }
    e = 0
    while e < N - 1 {
        c = h_sub[e] / 1000000
        p = h_sub[e] - c * 1000000
        plist[pcount[c]] = p
        pcount[c] = pcount[c] + 1
        e = e + 1
    }
    var total: i64 = 0
    var head: i64 = 0
    var tail: i64 = 0
    var t: i64 = 0
    var s: i64 = 0
    var b: i64 = 0
    while s < N {
        head = 0
        tail = 0
        queue[tail] = s
        tail = tail + 1
        vis[s] = s + 1
        while head < tail {
            b = queue[head]
            head = head + 1
            total = total + 1
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
    if total != 2 * N - 1 {
        println("FAIL: closure edges")
        n_fail = n_fail + 1
    }

    // conflicts: same entity (i/2), mask & pmask
    var n_conf: i64 = 0
    var hit = false
    var i: i64 = 0
    while i < M {
        var j: i64 = 0
        while j < M {
            if i != j {
                if i / 2 == j / 2 {
                    hit = false
                    k = 0
                    while k < EP {
                        if mask[m_tgt[i] * EP + k] && pmask[m_tgt[j] * EP + k] {
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
    if n_conf != 2 * @K@ {
        println("FAIL: conflicts")
        n_fail = n_fail + 1
    }

    i = 0
    while i < M {
        var j: i64 = i + 1
        while j < M {
            if m_keep[i] && m_keep[j] {
                if i / 2 == j / 2 {
                    hit = false
                    k = 0
                    while k < EP {
                        if mask[m_tgt[i] * EP + k] && pmask[m_tgt[j] * EP + k] {
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
    var kept: i64 = 0
    i = 0
    while i < M {
        if m_keep[i] {
            kept = kept + 1
        }
        i = i + 1
    }
    if kept != @K@ {
        println("FAIL: kept")
        n_fail = n_fail + 1
    }

    println("sparse probe N:")
    println(N)
    println("closure edges:")
    println(total)
    println("conflicts:")
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
    for N in (4000, 6000, 8000, 10000, 15000, 20000):
        dense_probe(f"probe_dense_{N}.sio", N, 100)
        print(f"probe_dense_{N}.sio  (N*N={N*N})")
    for N in (10000, 30000, 100000, 300000):
        sparse_probe(f"probe_sparse_{N}.sio", N, 100)
        print(f"probe_sparse_{N}.sio")
    return 0


main()
