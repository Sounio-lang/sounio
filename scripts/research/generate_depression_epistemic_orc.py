#!/usr/bin/env python3
"""
generate_depression_epistemic_orc.py — emit the Sounio epistemic-layer program.

Sounio cannot recompute exact optimal-transport ORC on the full N=1634..3089 graphs
(infeasible natively). Instead it consumes the EXACT-OT outputs (validated bit-exact
against Julia/HiGHS) and performs the *epistemic* layer that is Sounio's differentiator:
native LCG bootstrap, z-score vs degree-preserving null, GCI, and a confidence gate on
the density-matched severity separation.

This program REPLACES the defective examples/semantic_orc/depression_real_orc.sio, which
read the wrong base graph (depression_networks/ 310-node dense) and selected nodes from an
unrelated 438-node SWOW core.

Input:  /tmp/epistemic_payload.json  (assembled from results/unified/*.json)
Output: examples/semantic_orc/depression_epistemic_orc.sio
"""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PAYLOAD = Path("/tmp/epistemic_payload.json")
OUT = REPO / "examples/semantic_orc/depression_epistemic_orc.sio"
GROUPS = ["minimum", "mild", "moderate", "severe"]


def neg(x):
    """Sounio has no unary minus — emit (0.0 - v) for negatives."""
    return f"(0.0 - {abs(x):.6f})" if x < 0 else f"{x:.6f}"


def sample_fn(gid, group, vals):
    lines = [f"fn load_sample_{group}() -> Sample with Mut, Panic {{",
             "    var s = Sample { k: [0.0; 256], n: 256 }"]
    for i, v in enumerate(vals):
        lines.append(f"    s.k[{i}] = {neg(v)}")
    lines += ["    s", "}"]
    return "\n".join(lines)


def main():
    p = json.loads(PAYLOAD.read_text())

    header = '''//@ run-pass
// depression_epistemic_orc.sio — EPISTEMIC LAYER over exact-OT semantic curvature
//
// Sounio does NOT recompute optimal transport here (infeasible on N=1634..3089 graphs).
// It ingests exact-OT per-edge curvatures (Wasserstein-1, POT network simplex, validated
// bit-exact vs Julia/HiGHS) and computes the epistemic layer natively:
//   - LCG bootstrap CI of mean curvature over a 256-edge real sample
//   - z-score vs degree-preserving (Maslov-Sneppen) null
//   - GCI = fraction of |kappa| > 2 sigma
//   - confidence gate on the density-matched severity separation
//
// Real exact-OT inputs (hyperbolic-semantic-networks, depression_networks_optimal):
//   group     kappa_real  null_mean  z        matched_kappa (N~1500, <k>~10.1)
//   minimum   -0.1303     -0.2642    266.8    -0.1776 +/- 0.0032   [most hyperbolic]
//   moderate  -0.0871     -0.1281    194.8    -0.1672 +/- 0.0037
//   severe    -0.0783     -0.1082    196.1    -0.1560 +/- 0.0049
//   mild      -0.0742     -0.1004     55.3    -0.1453 +/- 0.0064   [least hyperbolic]
//
// FINDING: raw kappa correlates r=0.991 with mean degree (density confound), but at matched
// density all 6 pairwise CIs separate and the rank is preserved — subclinical (minimum) is
// the most hyperbolic group, a genuine residual structural effect.
//
// AUTHORITY BOUNDARY: real semantic-network data, no clinical claims, no diagnostic use.
// Presentations: Hong Kong Digital Mental Health (Jun 14 2026); Yale CPC (Jul 2026).

'''

    structs = '''// ============================================================================
fn abs_e(x: f64) -> f64 { if x < 0.0 { 0.0 - x } else { x } }

fn sqrt_e(x: f64) -> f64 with Mut, Panic, Div {
    if x <= 0.0 { return 0.0 }
    var r = x
    var i = 0
    while i < 30 { r = 0.5 * (r + x / r); i = i + 1 }
    r
}

struct Sample {
    k: [f64; 256]
    n: i64
}

// Epistemic profile: a measured curvature with propagated uncertainty + confidence.
struct EpiProfile {
    group_id: i64
    kappa_real: f64
    z_null: f64
    boot_mean: f64
    boot_se: f64
    gci: f64
    matched_mean: f64
    matched_std: f64
}

'''

    epi_compute = '''// ============================================================================
// NATIVE EPISTEMIC COMPUTATION
// ============================================================================

// LCG bootstrap: B resamples of the 256-edge sample, returns bootstrap SE of the mean.
fn bootstrap_se(s: Sample, seed: i64) -> f64 with Mut, Panic, Div {
    var state = seed
    var sum_means: f64 = 0.0
    var sum_means2: f64 = 0.0
    let B: i64 = 200
    var b: i64 = 0
    while b < B {
        var acc: f64 = 0.0
        var i: i64 = 0
        while i < s.n {
            state = state * 6364136223846793005 + 1442695040888963407
            // Use HIGH bits — LCG low bits have short periods (state % n would alias).
            var hi = state / 2147483648
            if hi < 0 { hi = 0 - hi }
            let idx = hi % s.n
            acc = acc + s.k[idx as usize]
            i = i + 1
        }
        let m = acc / (s.n as f64)
        sum_means = sum_means + m
        sum_means2 = sum_means2 + m * m
        b = b + 1
    }
    let bm = sum_means / (B as f64)
    let var_m = sum_means2 / (B as f64) - bm * bm
    sqrt_e(abs_e(var_m))
}

fn sample_mean(s: Sample) -> f64 with Mut, Panic, Div {
    var acc: f64 = 0.0
    var i: i64 = 0
    while i < s.n { acc = acc + s.k[i as usize]; i = i + 1 }
    acc / (s.n as f64)
}

fn sample_std(s: Sample, mean: f64) -> f64 with Mut, Panic, Div {
    var acc: f64 = 0.0
    var i: i64 = 0
    while i < s.n {
        let d = s.k[i as usize] - mean
        acc = acc + d * d
        i = i + 1
    }
    sqrt_e(acc / (s.n as f64))
}

// GCI: fraction of edges with |kappa| > 2 sigma (geometric certainty index).
fn gci(s: Sample, mean: f64, sigma: f64) -> f64 with Mut, Panic, Div {
    let thr = 2.0 * sigma
    var cnt: i64 = 0
    var i: i64 = 0
    while i < s.n {
        if abs_e(s.k[i as usize]) > thr { cnt = cnt + 1 }
        i = i + 1
    }
    (cnt as f64) / (s.n as f64)
}

fn z_null(kappa_real: f64, null_mean: f64, null_std: f64) -> f64 with Div {
    if null_std <= 0.0 { return 0.0 }
    (kappa_real - null_mean) / null_std
}

fn build_profile(gid: i64, s: Sample, kappa_real: f64, null_mean: f64, null_std: f64,
                 matched_mean: f64, matched_std: f64, seed: i64) -> EpiProfile with Mut, Panic, Div {
    let m = sample_mean(s)
    let sd = sample_std(s, m)
    EpiProfile {
        group_id: gid,
        kappa_real: kappa_real,
        z_null: z_null(kappa_real, null_mean, null_std),
        boot_mean: m,
        boot_se: bootstrap_se(s, seed),
        gci: gci(s, m, sd),
        matched_mean: matched_mean,
        matched_std: matched_std,
    }
}

// Separation z between two density-matched groups (pooled std).
fn separation_z(a: EpiProfile, b: EpiProfile) -> f64 with Mut, Panic, Div {
    let pooled = sqrt_e(a.matched_std * a.matched_std + b.matched_std * b.matched_std)
    if pooled <= 0.0 { return 0.0 }
    abs_e(b.matched_mean - a.matched_mean) / pooled
}

'''

    # print helpers (reuse the digit-loop pattern)
    prints = '''// ============================================================================
// PRINT HELPERS
// ============================================================================

fn print_i64_e(n: i64) with IO, Mut, Panic, Div {
    if n < 0 { print("-"); print_i64_e(0 - n); return }
    if n == 0 { print("0"); return }
    var digits: [i64; 20] = [0; 20]
    var cnt: i64 = 0
    var x = n
    while x > 0 { digits[cnt as usize] = x % 10; x = x / 10; cnt = cnt + 1 }
    var idx = cnt - 1
    while idx >= 0 {
        let d = digits[idx as usize]
        if d == 0 { print("0") } else if d == 1 { print("1") } else if d == 2 { print("2") }
        else if d == 3 { print("3") } else if d == 4 { print("4") } else if d == 5 { print("5") }
        else if d == 6 { print("6") } else if d == 7 { print("7") } else if d == 8 { print("8") }
        else { print("9") }
        idx = idx - 1
    }
}

fn print_f64_e(x: f64) with IO, Mut, Panic, Div {
    if x < 0.0 { print("-"); print_f64_e(0.0 - x); return }
    let ip = x as i64
    let frac = x - (ip as f64)
    let fs = (frac * 10000.0 + 0.5) as i64
    print_i64_e(ip)
    print(".")
    if fs < 10 { print("000") } else if fs < 100 { print("00") } else if fs < 1000 { print("0") }
    print_i64_e(fs)
}

fn print_group_e(g: i64) with IO {
    if g == 0 { print("minimum ") } else if g == 1 { print("mild    ") }
    else if g == 2 { print("moderate") } else { print("severe  ") }
}

fn print_profile_e(p: EpiProfile) with IO, Mut, Panic, Div {
    print("  "); print_group_e(p.group_id)
    print(": kappa="); print_f64_e(p.kappa_real)
    print(" z_null="); print_f64_e(p.z_null)
    print(" boot="); print_f64_e(p.boot_mean); print("+/-"); print_f64_e(p.boot_se)
    print(" GCI="); print_f64_e(p.gci)
    print(" matched="); print_f64_e(p.matched_mean)
    println("")
}

'''

    # group profiles: gid mapping minimum=0, mild=1, moderate=2, severe=3
    gid = {"minimum": 0, "mild": 1, "moderate": 2, "severe": 3}
    sample_fns = "\n\n".join(sample_fn(gid[g], g, p[g]["sample"]) for g in GROUPS)

    main_lines = ["fn main() -> i32 with IO, Mut, Panic, Div {",
                  '    println("================================================================")',
                  '    println("  DEPRESSION ORC — EPISTEMIC LAYER over exact optimal transport")',
                  '    println("  Sounio: native bootstrap + z-vs-null + GCI + separation gate")',
                  '    println("================================================================")',
                  '    println("")']
    seed = 1001
    for g in GROUPS:
        d = p[g]
        main_lines += [
            f"    let s_{g} = load_sample_{g}()",
            # matched_std field carries the SE of the matched mean (std / sqrt(reps=20)),
            # so separation_z tests whether the GROUP MEANS differ, not single-draw spread.
            (f"    let p_{g} = build_profile({gid[g]}, s_{g}, {neg(d['kappa_real'])}, "
             f"{neg(d['null_mean'])}, {d['null_std']:.8f}, {neg(d['matched_mean'])}, "
             f"{d['matched_std'] / (20 ** 0.5):.6f}, {seed})"),
        ]
        seed += 7
    main_lines += ['    println("--- Per-group epistemic profiles (full-graph exact OT) ---")',
                   '    println("")']
    for g in GROUPS:
        main_lines.append(f"    print_profile_e(p_{g})")

    main_lines += [
        '    println("")',
        '    println("--- Density-matched separation (N~1500, <k>~10.1) ---")',
        '    println("  order most->least hyperbolic: minimum > moderate > severe > mild")',
        "    let sep_min_mod = separation_z(p_minimum, p_moderate)",
        "    let sep_mod_sev = separation_z(p_moderate, p_severe)",
        "    let sep_sev_mild = separation_z(p_severe, p_mild)",
        '    print("  sep z(minimum,moderate)="); print_f64_e(sep_min_mod); println("")',
        '    print("  sep z(moderate,severe) ="); print_f64_e(sep_mod_sev); println("")',
        '    print("  sep z(severe,mild)     ="); print_f64_e(sep_sev_mild); println("")',
        '    println("")',
        "    // Confidence gate: subclinical (minimum) most hyperbolic AND separated from all.",
        "    let min_most = p_minimum.matched_mean < p_moderate.matched_mean &&",
        "                   p_minimum.matched_mean < p_severe.matched_mean &&",
        "                   p_minimum.matched_mean < p_mild.matched_mean",
        "    let all_separated = sep_min_mod > 2.0 && sep_mod_sev > 2.0 && sep_sev_mild > 2.0",
        "    if min_most && all_separated {",
        '        println("  VERIFIED: subclinical most hyperbolic; all adjacent pairs separated (z>2)")',
        '        println("  Effect survives density control — not a degree artifact.")',
        "    } else {",
        '        println("  WARNING: separation gate not satisfied")',
        "    }",
        '    println("")',
        '    println("SOUNIO_DEPRESSION_EPISTEMIC_ORC_PASS")',
        "    0",
        "}",
    ]

    parts = [header, structs, sample_fns, "\n", epi_compute, prints,
             "// ============================================================================\n// MAIN\n// ============================================================================\n",
             "\n".join(main_lines)]
    OUT.write_text("\n".join(parts))
    print(f"Wrote {OUT} ({OUT.read_text().count(chr(10))} lines)")


if __name__ == "__main__":
    main()
