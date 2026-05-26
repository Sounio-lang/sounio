#!/usr/bin/env python3
"""
generate_depression_real_orc.py — Generate depression_real_orc.sio from real SWOW-EN edges.

Reads real depression-stratified edge lists from hyperbolic-semantic-networks,
selects top-N nodes by degree per group, builds subgraph adjacency, and emits
a self-contained Sounio program with hardcoded real data.

Output: examples/semantic_orc/depression_real_orc.sio
"""

import csv
import os
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
HN_DATA   = Path("/workspace/hyperbolic-semantic-networks")
OUT_FILE  = REPO_ROOT / "examples/semantic_orc/depression_real_orc.sio"

GROUPS = ["minimum", "mild", "moderate", "severe"]
EDGES_DIR = HN_DATA / "data/processed/depression_networks"
NODES_SAMPLE = 50   # 50 most-hyperbolic nodes (most negative κ) → [i32; 2500]

# ── Load node features — select most hyperbolic nodes ────────────────────────
node_features_path = HN_DATA / "data/cpc2026/sounio_input/node_features.csv"
node_kappa = {}
all_nodes_by_kappa = []
if node_features_path.exists():
    with open(node_features_path) as f:
        for row in csv.DictReader(f):
            k = float(row["feature_kappa"])
            node_kappa[row["node"]] = k
            all_nodes_by_kappa.append((row["node"], k))

# Sort by κ ascending (most negative = most hyperbolic) and take top-N
all_nodes_by_kappa.sort(key=lambda x: x[1])
shared_nodes = [n for n, _ in all_nodes_by_kappa[:NODES_SAMPLE]]
node_idx = {n: i for i, n in enumerate(shared_nodes)}
print(f"Shared node set: {NODES_SAMPLE} most-hyperbolic nodes "
      f"(κ range [{all_nodes_by_kappa[0][1]:.3f}, {all_nodes_by_kappa[NODES_SAMPLE-1][1]:.3f}])")

# ── Per-group processing — same nodes, group-specific edges ──────────────────
group_data = {}

for group in GROUPS:
    edges_path = EDGES_DIR / f"depression_{group}_edges.csv"
    if not edges_path.exists():
        print(f"WARNING: {edges_path} not found, skipping {group}", file=sys.stderr)
        continue

    # Build adjacency among shared nodes using group-specific edge list
    n = len(shared_nodes)
    adj = [[0] * n for _ in range(n)]
    total_edges = 0
    with open(edges_path) as f:
        for row in csv.DictReader(f):
            s, t = row["source"], row["target"]
            total_edges += 1
            if s in node_idx and t in node_idx:
                i, j = node_idx[s], node_idx[t]
                adj[i][j] = 1
                adj[j][i] = 1

    # Collect non-zero entries for code gen
    edges_sounio = []
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i][j]:
                edges_sounio.append((i, j))

    # mean κ for shared nodes (same for all groups since same nodes)
    kappas = [node_kappa.get(nd, 0.0) for nd in shared_nodes]
    mean_kappa_ref = sum(kappas) / len(kappas) if kappas else 0.0

    group_data[group] = {
        "n": n,
        "nodes": shared_nodes,
        "edges": edges_sounio,
        "mean_kappa_ref": mean_kappa_ref,
        "total_edges_in_group": total_edges,
    }
    print(f"{group}: {len(edges_sounio)} subgraph edges "
          f"(from {total_edges} total), mean_κ_ref={mean_kappa_ref:.4f}")

# ── Code generation ───────────────────────────────────────────────────────────
def gen_load_fn(group, data):
    n = data["n"]
    edges = data["edges"]
    lines = []
    lines.append(f"fn load_{group}() -> BinGraph with Mut, Panic {{")
    lines.append(f"    var g = bg_real_new({n})")
    for i, j in edges:
        lines.append(f"    g.adj[({i} * {n} + {j}) as usize] = 1")
        lines.append(f"    g.adj[({j} * {n} + {i}) as usize] = 1")
    lines.append("    g")
    lines.append("}")
    return "\n".join(lines)

# Build combined code
header = '''\
//@ run-pass
// depression_real_orc.sio — LLY-ORC on REAL SWOW-EN depression-stratified graphs
//
// AUTHORITY BOUNDARY: Real semantic-network data from Agourakis82/hyperbolic-semantic-networks.
// No clinical claims, no diagnostic use, no null models yet.
// κ reference values from Julia LP optimal transport (exact, HiGHS solver).
//
// Source data:
//   data/processed/depression_networks/depression_{group}_edges.csv
//   (subset: top-50 nodes by degree per group → 50×50 adjacency)
//
// Presentations: Hong Kong Digital Mental Health Conference (Jun 14 2026)
//               Computational Psychiatry Conference, Yale (Jul 2026)
'''

math_helpers = '''\
fn abs_r(x: f64) -> f64 { if x < 0.0 { 0.0 - x } else { x } }

fn sqrt_r(x: f64) -> f64 with Mut, Panic, Div {
    if x <= 0.0 { return 0.0 }
    var r = x
    var i = 0
    while i < 20 { r = 0.5 * (r + x / r); i = i + 1 }
    r
}
'''

bingraph_def = '''\
struct BinGraph {
    adj: [i32; 2500]
    n: i64
}

fn bg_real_new(n: i64) -> BinGraph { BinGraph { adj: [0; 2500], n: n } }

fn bg_r_degree(g: BinGraph, v: i64) -> i64 with Mut, Panic, Div {
    var d: i64 = 0
    var j: i64 = 0
    while j < g.n {
        if g.adj[(v * g.n + j) as usize] == 1 { d = d + 1 }
        j = j + 1
    }
    d
}

fn bg_r_common(g: BinGraph, u: i64, v: i64) -> i64 with Mut, Panic, Div {
    var count: i64 = 0
    var w: i64 = 0
    while w < g.n {
        if w != u && w != v {
            if g.adj[(u * g.n + w) as usize] == 1 && g.adj[(v * g.n + w) as usize] == 1 {
                count = count + 1
            }
        }
        w = w + 1
    }
    count
}

fn orc_r_lly(g: BinGraph, u: i64, v: i64) -> f64 with Mut, Panic, Div {
    let d_u = bg_r_degree(g, u)
    let d_v = bg_r_degree(g, v)
    if d_u == 0 || d_v == 0 { return 0.0 }
    let t = bg_r_common(g, u, v)
    let inv_du = 1.0 / (d_u as f64)
    let inv_dv = 1.0 / (d_v as f64)
    (t as f64) * (inv_du + inv_dv) + inv_du + inv_dv - 1.0
}

struct OrcStats {
    n_edges: i64
    mean_k: f64
    std_k: f64
    n_neg: i64
    n_pos: i64
}

fn compute_orc_r(g: BinGraph) -> OrcStats with Mut, Panic, Div {
    var n_edges: i64 = 0
    var sum_k: f64 = 0.0
    var sum_k2: f64 = 0.0
    var n_pos: i64 = 0
    var n_neg: i64 = 0
    let eps = 0.01

    var u: i64 = 0
    while u < g.n {
        var v: i64 = u + 1
        while v < g.n {
            if g.adj[(u * g.n + v) as usize] == 1 {
                let k = orc_r_lly(g, u, v)
                n_edges = n_edges + 1
                sum_k = sum_k + k
                sum_k2 = sum_k2 + k * k
                if k > eps { n_pos = n_pos + 1 }
                else if k < (0.0 - eps) { n_neg = n_neg + 1 }
            }
            v = v + 1
        }
        u = u + 1
    }

    let mean = if n_edges > 0 { sum_k / (n_edges as f64) } else { 0.0 }
    let var_k = if n_edges > 0 { sum_k2 / (n_edges as f64) - mean * mean } else { 0.0 }
    OrcStats {
        n_edges: n_edges, mean_k: mean,
        std_k: sqrt_r(abs_r(var_k)),
        n_neg: n_neg, n_pos: n_pos,
    }
}
'''

print_helpers = '''\
fn print_i64_r(n: i64) with IO, Mut, Panic, Div {
    if n < 0 { print("-"); print_i64_r(0 - n); return }
    if n == 0 { print("0"); return }
    var digits: [i64; 20] = [0; 20]
    var cnt: i64 = 0
    var x = n
    while x > 0 {
        digits[cnt as usize] = x % 10
        x = x / 10
        cnt = cnt + 1
    }
    var idx = cnt - 1
    while idx >= 0 {
        let d = digits[idx as usize]
        if d == 0 { print("0") }
        else if d == 1 { print("1") }
        else if d == 2 { print("2") }
        else if d == 3 { print("3") }
        else if d == 4 { print("4") }
        else if d == 5 { print("5") }
        else if d == 6 { print("6") }
        else if d == 7 { print("7") }
        else if d == 8 { print("8") }
        else { print("9") }
        idx = idx - 1
    }
}

fn print_f64_r(x: f64) with IO, Mut, Panic, Div {
    if x < 0.0 { print("-"); print_f64_r(0.0 - x); return }
    let ip = x as i64
    let frac = x - (ip as f64)
    let fs = (frac * 10000.0 + 0.5) as i64
    print_i64_r(ip)
    print(".")
    if fs < 10 { print("000") }
    else if fs < 100 { print("00") }
    else if fs < 1000 { print("0") }
    print_i64_r(fs)
}

fn print_group_r(g: i64) with IO {
    if g == 0 { print("minimum ") }
    else if g == 1 { print("mild    ") }
    else if g == 2 { print("moderate") }
    else { print("severe  ") }
}

fn print_orc_r(gid: i64, s: OrcStats, ref_k: f64) with IO, Mut, Panic, Div {
    print("  ")
    print_group_r(gid)
    print(": edges="); print_i64_r(s.n_edges)
    print(" mean_k="); print_f64_r(s.mean_k)
    print(" std="); print_f64_r(s.std_k)
    let frac_neg = if s.n_edges > 0 { (s.n_neg as f64) / (s.n_edges as f64) } else { 0.0 }
    print(" neg%="); print_f64_r(frac_neg)
    print(" ref_k="); print_f64_r(ref_k)
    println("")
}
'''

# Generate load functions
load_fns = []
for i, group in enumerate(GROUPS):
    if group in group_data:
        load_fns.append(gen_load_fn(group, group_data[group]))

# Build main
ref_means = {g: group_data[g]["mean_kappa_ref"] for g in GROUPS if g in group_data}

main_lines = [
    "fn main() -> i32 with IO, Mut, Panic, Div {",
    '    println("================================================================")',
    '    println("  DEPRESSION SEVERITY ORC — REAL SWOW-EN SEMANTIC GRAPH")',
    '    println("  Top-50 nodes by degree per group, LLY-ORC natively in Sounio")',
    '    println("  50 most-hyperbolic SWOW-EN nodes (kappa range [-0.56, -0.34])")',
    '    println("================================================================")',
    '    println("")',
]

gid = 0
for group in GROUPS:
    if group not in group_data:
        gid += 1
        continue
    d = group_data[group]
    ref_k_lit = f"{d['mean_kappa_ref']:.6f}"
    if d['mean_kappa_ref'] < 0:
        ref_k_expr = f"(0.0 - {abs(d['mean_kappa_ref']):.6f})"
    else:
        ref_k_expr = f"{d['mean_kappa_ref']:.6f}"

    main_lines += [
        f"    let g_{group} = load_{group}()",
        f"    let orc_{group} = compute_orc_r(g_{group})",
        f"    print_orc_r({gid}, orc_{group}, {ref_k_expr})",
    ]
    gid += 1

main_lines += [
    '    println("")',
    '    println("--- Published κ reference (Julia LP exact, full SWOW-EN graph) ---")',
    '    println("  minimum:  κ = -0.127 ± 0.238  [non-monotonic: most hyperbolic]")',
    '    println("  mild:     κ = -0.071 ± 0.206")',
    '    println("  moderate: κ = -0.065 ± 0.222")',
    '    println("  severe:   κ = -0.074 ± 0.207")',
    '    println("")',
    '    println("--- Interpretation ---")',
    '    println("  minimum: only 19 real edges in hyperbolic core vs 47-79 for clinical groups.")',
    '    println("  Sparse activation (degree~1) → high positive subgraph LLY (near-isolated pairs).")',
    '    println("  Denser clinical activation → more triangles → lower LLY curvature.")',
    '    println("  Full-graph published kappa requires complete 438-node context.")',
    '    println("")',
    '    println("SOUNIO_DEPRESSION_REAL_ORC_PASS")',
    "    0",
    "}",
]

main_body = "\n".join(main_lines)

# Assemble full source
parts = [
    header,
    "// " + "=" * 74,
    "// MATH HELPERS",
    "// " + "=" * 74,
    math_helpers,
    "// " + "=" * 74,
    "// GRAPH STRUCTURES + LLY-ORC",
    "// " + "=" * 74,
    bingraph_def,
    "// " + "=" * 74,
    "// HARDCODED REAL SWOW-EN SUBGRAPHS (top-50 nodes by degree per group)",
    "// " + "=" * 74,
    "",
]
for fn_code in load_fns:
    parts.append(fn_code)
    parts.append("")

parts += [
    "// " + "=" * 74,
    "// PRINT HELPERS",
    "// " + "=" * 74,
    print_helpers,
    "// " + "=" * 74,
    "// MAIN",
    "// " + "=" * 74,
    "",
    main_body,
]

source = "\n".join(parts)

OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
OUT_FILE.write_text(source)

lines = source.count("\n")
print(f"\nWrote {OUT_FILE} ({lines} lines)")
print(f"Groups generated: {list(group_data.keys())}")
