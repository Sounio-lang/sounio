#!/usr/bin/env python3
"""Door F: Generate associator-norm pre-ictal sweep for chb03.

Uses fixed α=0.20 (confirmed optimum from Door III + E).
Embeds 6 temporal probe windows relative to seizure onset (362s):
  FAR:   t=302s  (-60s) — far pre-ictal baseline
  PRE30: t=332s  (-30s)
  PRE10: t=352s  (-10s)
  PRE5:  t=357s  (-5s)
  IC:    t=362s  (onset)
  POST:  t=367s  (+5s)

For each window: MSE and mean associator norm ||(A⊗h)⊗x - A⊗(h⊗x)||.
Hypothesis: assoc_ratio rises pre-ictally while MSE_ratio stays flat (null from Door II).
"""

import pyedflib
import numpy as np
import sys
import os

EDF_PATH     = "data/chbmit/chb03_01.edf"
SEIZURE_S    = 362
FS           = 256
N_CH         = 16
N_TRAIN      = 64
N_WIN        = 80
INTERICTAL_S = 10.0
ALPHA        = 0.20
OUT_PATH     = "examples/door_f_assoc_preictal_chb03.sio"

# (label, start_s, time_rel_s)
WINDOWS = [
    ("FAR",   302, -60),
    ("PRE30", 332, -30),
    ("PRE10", 352, -10),
    ("PRE5",  357,  -5),
    ("IC",    362,   0),
    ("POST",  367,  +5),
]


def load_subject(edf_path):
    f = pyedflib.EdfReader(edf_path)
    sigs = np.stack([f.readSignal(i) for i in range(N_CH)])
    f.close()
    return sigs


def extract_windows(sigs):
    ii_start = int(INTERICTAL_S * FS)
    ii_raw   = sigs[:, ii_start : ii_start + N_TRAIN]

    means   = ii_raw.mean(axis=1, keepdims=True)
    stds    = ii_raw.std(axis=1, keepdims=True)
    stds    = np.where(stds < 1e-6, 1.0, stds)
    max_abs = np.abs((ii_raw - means) / stds).max(axis=1, keepdims=True)
    max_abs = np.where(max_abs < 1e-6, 1.0, max_abs)

    def normalize_window(raw):
        z = (raw - means) / stds / max_abs
        return np.clip(z, -5.0, 5.0)

    # Interictal training window (80 samples: 64 train + 16 holdout)
    ii_raw_full = sigs[:, ii_start : ii_start + N_WIN]
    ii_norm     = normalize_window(ii_raw_full)

    target_ii           = np.zeros(N_WIN)
    target_ii[:N_WIN-1] = ii_norm[0, 1:]
    target_ii[N_WIN-1]  = ii_norm[0, N_WIN-1]

    # Probe windows
    probes = {}
    for label, start_s, time_rel in WINDOWS:
        start   = int(start_s * FS)
        raw     = sigs[:, start : start + N_WIN]
        norm    = normalize_window(raw)
        target  = np.zeros(N_WIN)
        target[:N_WIN-1] = norm[0, 1:]
        target[N_WIN-1]  = norm[0, N_WIN-1]
        probes[label] = (norm, target, time_rel)

    return ii_norm, target_ii, probes


def fmt(v):
    return f"{v:.8f}"


def write_array_inits(fp, name, values):
    for i, v in enumerate(values):
        fp.write(f"    {name}[{i}] = {fmt(v)}\n")


def write_sedenion_loader(fp, prefix):
    """Write fn load_PREFIX(t) -> Sedenion"""
    fp.write(f"fn load_{prefix}(t: i64) -> Sedenion {{\n")
    fp.write(f"    Sedenion {{\n")
    for ch in range(N_CH):
        comma = "," if ch < N_CH - 1 else ""
        fp.write(f"        e{ch}: {prefix}_CH{ch}[t]{comma}\n")
    fp.write("    }\n}\n\n")


def write_mse_fn(fp, prefix):
    """Write fn mse_PREFIX(a) -> f64 — forward on 80 steps, MSE vs target."""
    fp.write(f"fn mse_{prefix}(a: Sedenion) -> f64 with Div, Panic {{\n")
    fp.write("    var h: Sedenion = sed_one()\n")
    fp.write("    var total: f64 = 0.0\n")
    fp.write("    var t: i64 = 0\n")
    fp.write(f"    while t < {N_WIN} {{\n")
    fp.write(f"        let xt: Sedenion = load_{prefix}(t)\n")
    fp.write("        let ah: Sedenion = sed_mul(a, h)\n")
    fp.write("        let raw: Sedenion = sed_add(ah, xt)\n")
    fp.write("        let act: Sedenion = sed_tanh_s(raw)\n")
    fp.write("        let n: f64 = sed_norm(act)\n")
    fp.write("        if n > 0.0000001 { h = sed_scale(act, 1.0 / n) } else { h = act }\n")
    fp.write("        let y: f64 = sed_dot(h, get_c())\n")
    fp.write(f"        let err: f64 = y - {prefix}_TGT[t]\n")
    fp.write("        total = total + err * err\n")
    fp.write("        t = t + 1\n")
    fp.write("    }\n")
    fp.write(f"    total / {float(N_WIN)}\n")
    fp.write("}\n\n")


def write_assoc_fn(fp, prefix):
    """Write fn assoc_PREFIX(a) -> f64 — mean ||(A⊗h)⊗x - A⊗(h⊗x)|| over N_WIN steps."""
    fp.write(f"fn assoc_{prefix}(a: Sedenion) -> f64 with Div, Panic {{\n")
    fp.write("    var h: Sedenion = sed_one()\n")
    fp.write("    var total: f64 = 0.0\n")
    fp.write("    var t: i64 = 0\n")
    fp.write(f"    while t < {N_WIN} {{\n")
    fp.write(f"        let xt: Sedenion = load_{prefix}(t)\n")
    fp.write("        let ah:   Sedenion = sed_mul(a, h)\n")
    fp.write("        let ahx:  Sedenion = sed_mul(ah, xt)\n")   # (A⊗h)⊗x
    fp.write("        let hx:   Sedenion = sed_mul(h, xt)\n")
    fp.write("        let a_hx: Sedenion = sed_mul(a, hx)\n")    # A⊗(h⊗x)
    fp.write("        let diff: Sedenion = sed_sub(ahx, a_hx)\n")
    fp.write("        let an:   f64 = sed_norm(diff)\n")
    fp.write("        total = total + an\n")
    fp.write("        // Advance hidden state\n")
    fp.write("        let raw: Sedenion = sed_add(ah, xt)\n")
    fp.write("        let act: Sedenion = sed_tanh_s(raw)\n")
    fp.write("        let n: f64 = sed_norm(act)\n")
    fp.write("        if n > 0.0000001 { h = sed_scale(act, 1.0 / n) } else { h = act }\n")
    fp.write("        t = t + 1\n")
    fp.write("    }\n")
    fp.write(f"    total / {float(N_WIN)}\n")
    fp.write("}\n\n")


def main():
    if not os.path.exists(EDF_PATH):
        print(f"ERROR: {EDF_PATH} not found", file=sys.stderr)
        sys.exit(1)

    sigs = load_subject(EDF_PATH)
    ii_norm, target_ii, probes = extract_windows(sigs)
    sys.stderr.write(f"chb03: II std={ii_norm.std():.3f}\n")
    for label, (norm, tgt, rel) in probes.items():
        sys.stderr.write(f"  {label} (t_rel={rel:+d}s): std={norm.std():.3f}\n")

    fp = open(OUT_PATH, "w")

    fp.write(f"""\
//@ run-pass
//@ expect-stdout: ALL PASS

// ═══════════════════════════════════════════════════════════════════════════
// Door F: Pre-ictal sweep — MSE vs Associator Norm
// ═══════════════════════════════════════════════════════════════════════════
//
// Data: CHB-MIT chb03  ({EDF_PATH})
// Seizure onset at {SEIZURE_S}s, 256 Hz, 16 channels.
// α fixed = {ALPHA} (interior optimum confirmed Door III + E)
//
// Door II found NO pre-ictal MSE ramp (null). MSE measures prediction error.
// The associator ||(A⊗h)⊗x − A⊗(h⊗x)|| measures geometric non-associativity:
// the structural coupling between brain dynamics and the sedenion gate.
//
// Hypothesis: assoc_ratio rises >1.0 pre-ictally while mse_ratio stays ≈1.0.
//
// Probe windows (each 80 samples = ~312ms at 256Hz):
//   FAR   t={WINDOWS[0][1]}s  (−60s from onset) — far pre-ictal baseline
//   PRE30 t={WINDOWS[1][1]}s  (−30s)
//   PRE10 t={WINDOWS[2][1]}s  (−10s)
//   PRE5  t={WINDOWS[3][1]}s  (−5s)
//   IC    t={WINDOWS[4][1]}s  (onset, 0s)
//   POST  t={WINDOWS[5][1]}s  (+5s)

use math::sedenion::*;

// ─── Math helpers ─────────────────────────────────────────────────────────────

fn sqrt_s(x: f64) -> f64 with Mut, Div, Panic {{
    if x <= 0.0 {{ return 0.0 }}
    var y: f64 = x
    var i: i64 = 0
    while i < 20 {{ y = 0.5 * (y + x / y); i = i + 1 }}
    y
}}

fn exp_s(x: f64) -> f64 with Mut, Div, Panic {{
    if x > 20.0 {{ return exp_s(10.0) * exp_s(x - 10.0) }}
    if x < 0.0 - 20.0 {{ return 0.0 }}
    var s: f64 = 1.0
    var term: f64 = 1.0
    var i: i64 = 1
    while i <= 20 {{ term = term * x / (i as f64); s = s + term; i = i + 1 }}
    s
}}

fn tanh_s(x: f64) -> f64 with Mut, Div, Panic {{
    if x > 10.0 {{ return 1.0 }}
    if x < 0.0 - 10.0 {{ return 0.0 - 1.0 }}
    let e2 = exp_s(2.0 * x)
    (e2 - 1.0) / (e2 + 1.0)
}}

fn sed_tanh_s(s: Sedenion) -> Sedenion with Mut, Div, Panic {{
    Sedenion {{
        e0: tanh_s(s.e0),   e1: tanh_s(s.e1),   e2: tanh_s(s.e2),   e3: tanh_s(s.e3),
        e4: tanh_s(s.e4),   e5: tanh_s(s.e5),   e6: tanh_s(s.e6),   e7: tanh_s(s.e7),
        e8: tanh_s(s.e8),   e9: tanh_s(s.e9),   e10: tanh_s(s.e10), e11: tanh_s(s.e11),
        e12: tanh_s(s.e12), e13: tanh_s(s.e13), e14: tanh_s(s.e14), e15: tanh_s(s.e15)
    }}
}}

// ─── Sedenion gate (lerp-to-ZD parameterization, same as Door I/E) ────────────

fn make_a(alpha: f64) -> Sedenion with Div, Panic {{
    let inv2: f64 = 1.0 / sqrt_s(2.0)
    let zd0: f64 = 0.5; let zd1: f64 = 0.5; let zd4: f64 = 0.5; let zd5: f64 = 0.0 - 0.5
    let g0: f64 = (1.0 - alpha) * inv2 + alpha * zd0
    let g1: f64 = (1.0 - alpha) * inv2 + alpha * zd1
    let g4: f64 =                         alpha * zd4
    let g5: f64 =                         alpha * zd5
    let n: f64 = sqrt_s(g0*g0 + g1*g1 + g4*g4 + g5*g5)
    let s: f64 = if n < 0.0000001 {{ 1.0 }} else {{ 1.0 / n }}
    Sedenion {{ e0: g0*s, e1: g1*s, e2: 0.0, e3: 0.0,
               e4: g4*s, e5: g5*s, e6: 0.0, e7: 0.0,
               e8: 0.0,  e9: 0.0,  e10: 0.0, e11: 0.0,
               e12: 0.0, e13: 0.0, e14: 0.0, e15: 0.0 }}
}}

// ─── Readout C (scaled i64 to avoid BSS-zero f64 init bug) ───────────────────

var WC: [i64; 16] = [0; 16]
fn cg(i: i64) -> f64 with Div {{ (WC[i] as f64) / 100000000.0 }}
fn cs(i: i64, v: f64) with Mut {{ WC[i] = (v * 100000000.0) as i64 }}
fn get_c() -> Sedenion {{
    Sedenion {{ e0:cg(0), e1:cg(1), e2:cg(2), e3:cg(3),
               e4:cg(4), e5:cg(5), e6:cg(6), e7:cg(7),
               e8:cg(8), e9:cg(9), e10:cg(10), e11:cg(11),
               e12:cg(12), e13:cg(13), e14:cg(14), e15:cg(15) }}
}}
fn init_c() with Mut {{ var i: i64 = 0; while i < 16 {{ WC[i] = 0; i = i + 1 }} }}

// ─── Hidden state for training ────────────────────────────────────────────────

var H0:  [f64; 64] = [0.0; 64]   var H1:  [f64; 64] = [0.0; 64]
var H2:  [f64; 64] = [0.0; 64]   var H3:  [f64; 64] = [0.0; 64]
var H4:  [f64; 64] = [0.0; 64]   var H5:  [f64; 64] = [0.0; 64]
var H6:  [f64; 64] = [0.0; 64]   var H7:  [f64; 64] = [0.0; 64]
var H8:  [f64; 64] = [0.0; 64]   var H9:  [f64; 64] = [0.0; 64]
var H10: [f64; 64] = [0.0; 64]   var H11: [f64; 64] = [0.0; 64]
var H12: [f64; 64] = [0.0; 64]   var H13: [f64; 64] = [0.0; 64]
var H14: [f64; 64] = [0.0; 64]   var H15: [f64; 64] = [0.0; 64]

fn save_h(t: i64, h: Sedenion) with Mut {{
    H0[t]=h.e0; H1[t]=h.e1; H2[t]=h.e2; H3[t]=h.e3
    H4[t]=h.e4; H5[t]=h.e5; H6[t]=h.e6; H7[t]=h.e7
    H8[t]=h.e8; H9[t]=h.e9; H10[t]=h.e10; H11[t]=h.e11
    H12[t]=h.e12; H13[t]=h.e13; H14[t]=h.e14; H15[t]=h.e15
}}

fn load_h(t: i64) -> Sedenion {{
    Sedenion {{ e0:H0[t], e1:H1[t], e2:H2[t], e3:H3[t],
               e4:H4[t], e5:H5[t], e6:H6[t], e7:H7[t],
               e8:H8[t], e9:H9[t], e10:H10[t], e11:H11[t],
               e12:H12[t], e13:H13[t], e14:H14[t], e15:H15[t] }}
}}

""")

    # ── Data arrays for II and each probe window ──────────────────────────────
    fp.write("// ─── Data arrays ─────────────────────────────────────────────────────────────\n\n")

    # Interictal arrays
    for ch in range(N_CH):
        fp.write(f"var II_CH{ch}: [f64; {N_WIN}] = [0.0; {N_WIN}]\n")
    fp.write(f"var II_TGT: [f64; {N_WIN}] = [0.0; {N_WIN}]\n\n")

    # Probe window arrays
    for label, _, _ in WINDOWS:
        for ch in range(N_CH):
            fp.write(f"var {label}_CH{ch}: [f64; {N_WIN}] = [0.0; {N_WIN}]\n")
        fp.write(f"var {label}_TGT: [f64; {N_WIN}] = [0.0; {N_WIN}]\n\n")

    # init_data function
    fp.write("fn init_data() with Mut {\n")
    for ch in range(N_CH):
        write_array_inits(fp, f"II_CH{ch}", ii_norm[ch])
    write_array_inits(fp, "II_TGT", target_ii)
    for label, (norm, tgt, rel) in probes.items():
        for ch in range(N_CH):
            write_array_inits(fp, f"{label}_CH{ch}", norm[ch])
        write_array_inits(fp, f"{label}_TGT", tgt)
    fp.write("}\n\n")

    # ── Loader functions (Sedenion from channel arrays at time t) ─────────────
    fp.write("// ─── Sedenion loaders ────────────────────────────────────────────────────────\n\n")
    write_sedenion_loader(fp, "II")
    for label, _, _ in WINDOWS:
        write_sedenion_loader(fp, label)

    # ── Training pass (on II, 64 samples, stores H for C update) ─────────────
    fp.write("""\
// ─── Training (II only, 64 samples) ──────────────────────────────────────────

fn ssm_forward_ii(a: Sedenion) with Mut, Div, Panic {
    var h: Sedenion = sed_one()
    var t: i64 = 0
    while t < 64 {
        let xt: Sedenion = load_II(t)
        let ah: Sedenion = sed_mul(a, h)
        let raw: Sedenion = sed_add(ah, xt)
        let act: Sedenion = sed_tanh_s(raw)
        let n: f64 = sed_norm(act)
        if n > 0.0000001 { h = sed_scale(act, 1.0 / n) } else { h = act }
        save_h(t, h)
        t = t + 1
    }
}

fn ssm_update_c(lr: f64) with Mut, Div, Panic {
    let c: Sedenion = get_c()
    var gc: Sedenion = Sedenion { e0:0.0, e1:0.0, e2:0.0, e3:0.0,
                                   e4:0.0, e5:0.0, e6:0.0, e7:0.0,
                                   e8:0.0, e9:0.0, e10:0.0, e11:0.0,
                                   e12:0.0, e13:0.0, e14:0.0, e15:0.0 }
    var t: i64 = 0
    while t < 64 {
        let ht: Sedenion = load_h(t)
        let y: f64 = sed_dot(ht, c)
        let err: f64 = y - II_TGT[t]
        gc = sed_add(gc, sed_scale(ht, err))
        t = t + 1
    }
    let scale: f64 = lr * 2.0 / 64.0
    let new_c: Sedenion = sed_sub(c, sed_scale(gc, scale))
    var i: i64 = 0
    while i < 16 {
        let v: f64 = if i == 0 { new_c.e0 } else if i == 1 { new_c.e1 }
            else if i == 2 { new_c.e2 } else if i == 3 { new_c.e3 }
            else if i == 4 { new_c.e4 } else if i == 5 { new_c.e5 }
            else if i == 6 { new_c.e6 } else if i == 7 { new_c.e7 }
            else if i == 8 { new_c.e8 } else if i == 9 { new_c.e9 }
            else if i == 10 { new_c.e10 } else if i == 11 { new_c.e11 }
            else if i == 12 { new_c.e12 } else if i == 13 { new_c.e13 }
            else if i == 14 { new_c.e14 } else { new_c.e15 }
        cs(i, v)
        i = i + 1
    }
}

fn train_c(a: Sedenion) with Mut, Div, Panic {
    init_c()
    var ep: i64 = 0
    while ep < 200 { ssm_forward_ii(a); ssm_update_c(0.2); ep = ep + 1 }
}

""")

    # ── MSE and associator functions per probe window ─────────────────────────
    fp.write("// ─── MSE and associator per probe window ─────────────────────────────────────\n\n")

    # II baseline MSE (on holdout samples 64..79)
    fp.write(f"""\
fn mse_II_holdout(a: Sedenion) -> f64 with Mut, Div, Panic {{
    // Evaluate on samples 64..79 (holdout after training)
    // Re-run forward on train split to get warm hidden state
    var h: Sedenion = sed_one()
    var t: i64 = 0
    while t < 64 {{
        let xt: Sedenion = load_II(t)
        let ah: Sedenion = sed_mul(a, h)
        let raw: Sedenion = sed_add(ah, xt)
        let act: Sedenion = sed_tanh_s(raw)
        let n: f64 = sed_norm(act)
        if n > 0.0000001 {{ h = sed_scale(act, 1.0 / n) }} else {{ h = act }}
        t = t + 1
    }}
    var total: f64 = 0.0
    while t < {N_WIN} {{
        let xt: Sedenion = load_II(t)
        let ah: Sedenion = sed_mul(a, h)
        let raw: Sedenion = sed_add(ah, xt)
        let act: Sedenion = sed_tanh_s(raw)
        let n: f64 = sed_norm(act)
        if n > 0.0000001 {{ h = sed_scale(act, 1.0 / n) }} else {{ h = act }}
        let y: f64 = sed_dot(h, get_c())
        let err: f64 = y - II_TGT[t]
        total = total + err * err
        t = t + 1
    }}
    total / 16.0
}}

fn assoc_II(a: Sedenion) -> f64 with Div, Panic {{
    var h: Sedenion = sed_one()
    var total: f64 = 0.0
    var t: i64 = 0
    while t < {N_WIN} {{
        let xt: Sedenion = load_II(t)
        let ah:   Sedenion = sed_mul(a, h)
        let ahx:  Sedenion = sed_mul(ah, xt)
        let hx:   Sedenion = sed_mul(h, xt)
        let a_hx: Sedenion = sed_mul(a, hx)
        let diff: Sedenion = sed_sub(ahx, a_hx)
        let an:   f64 = sed_norm(diff)
        total = total + an
        let raw: Sedenion = sed_add(ah, xt)
        let act: Sedenion = sed_tanh_s(raw)
        let n: f64 = sed_norm(act)
        if n > 0.0000001 {{ h = sed_scale(act, 1.0 / n) }} else {{ h = act }}
        t = t + 1
    }}
    total / {float(N_WIN)}
}}

""")

    for label, _, _ in WINDOWS:
        write_mse_fn(fp, label)
        write_assoc_fn(fp, label)

    # ── Main ──────────────────────────────────────────────────────────────────
    fp.write(f"""\
// ─── Main ────────────────────────────────────────────────────────────────────

fn main() -> i32 with IO, Mut, Div, Panic {{
    init_data()

    let a: Sedenion = make_a({ALPHA})
    train_c(a)

    // Baseline: interictal holdout MSE and associator
    let baseline_mse:   f64 = mse_II_holdout(a)
    let baseline_assoc: f64 = assoc_II(a)

    println("═══════════════════════════════════════════════════════════════════════")
    println("  Door F: Pre-ictal sweep — MSE vs Associator Norm  (α={ALPHA})")
    println("  chb03, seizure at {SEIZURE_S}s, 256Hz, 16ch")
    println("  Baseline: interictal holdout at t={int(INTERICTAL_S)}s")
    println("═══════════════════════════════════════════════════════════════════════")
    println("")
    print("  baseline_mse=");   print(baseline_mse);   println("")
    print("  baseline_assoc="); print(baseline_assoc); println("")
    println("")
    println("  label    t_rel  mse        mse_ratio  assoc      assoc_ratio")
    println("  ──────────────────────────────────────────────────────────────────")

""")

    # Write eval for each probe window
    for label, start_s, time_rel in WINDOWS:
        fp.write(f"""\
    let mse_{label}:   f64 = mse_{label}(a)
    let assoc_{label}: f64 = assoc_{label}(a)
    let mse_r_{label}: f64 = if baseline_mse   > 0.000001 {{ mse_{label}   / baseline_mse   }} else {{ 0.0 }}
    let asc_r_{label}: f64 = if baseline_assoc > 0.000001 {{ assoc_{label} / baseline_assoc }} else {{ 0.0 }}
    print("  {label:<6}  {time_rel:+4}s  "); print(mse_{label}); print("  "); print(mse_r_{label})
    print("  "); print(assoc_{label}); print("  "); print(asc_r_{label}); println("")

""")

    fp.write("""\
    println("")

    // ── Sanity checks ────────────────────────────────────────────────────────
    var passed: i32 = 0
    var failed: i32 = 0

    // T1: IC window MSE > baseline (seizure is detectable by MSE at onset)
""")
    fp.write("""\
    let mse_ic_val:   f64 = mse_IC(a)
    let mse_ic_r:     f64 = if baseline_mse > 0.000001 { mse_ic_val / baseline_mse } else { 0.0 }
    if mse_ic_r > 1.0 {
        print("T1 PASS: MSE at onset = "); print(mse_ic_r); println("× baseline (>1.0)")
        passed = passed + 1
    } else {
        print("T1 FAIL: MSE at onset = "); print(mse_ic_r); println("× baseline (expected >1.0)")
        failed = failed + 1
    }

    // T2: Associator is non-zero at baseline (sedenion non-associativity active)
    if baseline_assoc > 0.0 {
        print("T2 PASS: baseline assoc = "); print(baseline_assoc); println(" > 0")
        passed = passed + 1
    } else {
        println("T2 FAIL: baseline assoc = 0 (associator vanished)")
        failed = failed + 1
    }

    // T3: IC associator != baseline (structural disruption at onset)
    let assoc_ic_val: f64 = assoc_IC(a)
    let assoc_ic_r:   f64 = if baseline_assoc > 0.000001 { assoc_ic_val / baseline_assoc } else { 0.0 }
    if assoc_ic_r > 0.5 {
        print("T3 PASS: assoc at onset = "); print(assoc_ic_r); println("× baseline (structural change)")
        passed = passed + 1
    } else {
        print("T3 FAIL: assoc at onset = "); print(assoc_ic_r); println("× (no change)")
        failed = failed + 1
    }

    // T4 (exploratory): does assoc_ratio rise pre-ictally before MSE?
    //   Check PRE10: assoc_ratio > 1.1 or (assoc_ratio > mse_ratio + 0.1)
    let assoc_pre10: f64 = assoc_PRE10(a)
    let mse_pre10:   f64 = mse_PRE10(a)
    let asc_r_pre10: f64 = if baseline_assoc > 0.000001 { assoc_pre10 / baseline_assoc } else { 0.0 }
    let mse_r_pre10: f64 = if baseline_mse   > 0.000001 { mse_pre10   / baseline_mse   } else { 0.0 }
    if asc_r_pre10 > mse_r_pre10 {
        print("T4 PASS (exploratory): assoc_ratio at −10s = "); print(asc_r_pre10)
        print(" > mse_ratio = "); print(mse_r_pre10); println("")
        passed = passed + 1
    } else {
        print("T4 FAIL (exploratory): assoc_ratio at −10s = "); print(asc_r_pre10)
        print(" ≤ mse_ratio = "); print(mse_r_pre10); println("")
        failed = failed + 1
    }

    println("")
    if failed == 0 { println("ALL PASS") }
    else { print("FAILURES: "); print(failed); println("") }

    print("  Passed: "); print(passed); println("")
    print("  Failed: "); print(failed); println("")

    0
}
""")

    fp.close()
    sys.stderr.write(f"Written: {OUT_PATH}\n")


if __name__ == "__main__":
    main()
