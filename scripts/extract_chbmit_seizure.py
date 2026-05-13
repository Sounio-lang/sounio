#!/usr/bin/env python3
"""Extract CHB-MIT EEG data for the S-SSM seizure detection experiment.

Generates a self-contained Sounio experiment file (Door I: seizure detection).

Architecture:
  S-SSM-16ch α=0.2 trained on 64-sample inter-ictal window →
  tested on held-out inter-ictal (16 samples) AND ictal onset (64 samples).

Key metric: MSE_ictal / MSE_test_ii
  If G₂ sedenion learned inter-ictal coupling structure, it should be MORE
  sensitive to its disruption during seizure than the diagonal baseline.

Usage:
  python3 scripts/extract_chbmit_seizure.py chb01 > examples/sedenion_ssm_seizure_chb01.sio
  python3 scripts/extract_chbmit_seizure.py chb03 > examples/sedenion_ssm_seizure_chb03.sio
  python3 scripts/extract_chbmit_seizure.py chb05 > examples/sedenion_ssm_seizure_chb05.sio

Channels used: 0-15 (FP1-F7 through P8-O2, standard 16-ch bipolar montage).
Normalization: z-score by inter-ictal training window mean/std per channel.
"""
import pyedflib
import numpy as np
import sys
import os

SUBJECTS = {
    "chb01": {
        "edf": "data/chbmit/chb01_03.edf",
        "seizure_start_s": 2996,
        "n_channels": 23,
        "fs": 256,
    },
    "chb03": {
        "edf": "data/chbmit/chb03_01.edf",
        "seizure_start_s": 362,
        "n_channels": 23,
        "fs": 256,
    },
    "chb05": {
        "edf": "data/chbmit/chb05_06.edf",
        "seizure_start_s": 417,
        "n_channels": 23,
        "fs": 256,
    },
}

INTERICTAL_START_S = 10.0    # 10s into file — well clear of startup transients
N_TRAIN = 64                  # inter-ictal training samples
N_TEST_II = 16                # inter-ictal holdout samples
N_TOTAL_II = N_TRAIN + N_TEST_II   # 80 samples
N_TEST_IC = 64                # ictal test samples (250ms at 256 Hz)
N_CH = 16                     # channels 0-15


def load_edf_channels(edf_path: str, n_ch: int, fs: int) -> np.ndarray:
    """Returns [n_ch, total_samples] float64 array."""
    if not os.path.exists(edf_path):
        sys.stderr.write(f"ERROR: {edf_path} not found\n")
        sys.exit(1)
    f = pyedflib.EdfReader(edf_path)
    sigs = np.stack([f.readSignal(i) for i in range(n_ch)])
    f.close()
    sys.stderr.write(f"Loaded {edf_path}: {sigs.shape[1]} samples, {n_ch} channels\n")
    return sigs


def fmt(v: float) -> str:
    return f"{v:.8f}"


def write_array(out, arr_name: str, values, comment: str = ""):
    if comment:
        out.write(f"    // {comment}\n")
    for i, v in enumerate(values):
        out.write(f"    {arr_name}[{i}] = {fmt(v)}\n")


def generate_sounio(subject: str, cfg: dict) -> None:
    out = sys.stdout
    fs = cfg["fs"]
    n_ch = cfg["n_channels"]
    edf_path = cfg["edf"]
    seizure_s = cfg["seizure_start_s"]

    # ── Load data ────────────────────────────────────────────────────────────
    sigs = load_edf_channels(edf_path, n_ch, fs)

    ii_start = int(INTERICTAL_START_S * fs)
    ic_start = int(seizure_s * fs)

    # Check bounds
    if ii_start + N_TOTAL_II > sigs.shape[1]:
        sys.stderr.write("ERROR: inter-ictal window out of file bounds\n"); sys.exit(1)
    if ic_start + N_TEST_IC > sigs.shape[1]:
        sys.stderr.write("ERROR: ictal window out of file bounds\n"); sys.exit(1)

    ii_raw = sigs[:N_CH, ii_start : ii_start + N_TOTAL_II]    # [16, 80]
    ic_raw = sigs[:N_CH, ic_start : ic_start + N_TEST_IC]     # [16, 64]

    # Normalize: z-score then max-abs scale to [-1, 1] using training window.
    # Matching Exp 13's normalization ensures gradient magnitudes stay bounded
    # for both sedenion and diagonal models.
    means = ii_raw[:, :N_TRAIN].mean(axis=1, keepdims=True)
    stds  = ii_raw[:, :N_TRAIN].std(axis=1, keepdims=True)
    stds  = np.where(stds < 1e-6, 1.0, stds)

    ii_z = (ii_raw - means) / stds                             # z-score first
    ic_z = (ic_raw - means) / stds

    # Max-abs scale by training window — values in [-1, 1] for train, unbounded for ictal
    max_abs = np.abs(ii_z[:, :N_TRAIN]).max(axis=1, keepdims=True)
    max_abs = np.where(max_abs < 1e-6, 1.0, max_abs)

    ii_norm = ii_z / max_abs
    ic_norm = ic_z / max_abs

    # Clip ictal to [-5, 5] to avoid extreme artifacts
    ic_norm = np.clip(ic_norm, -5.0, 5.0)

    # TARGET = CH0 shifted +1 (next-step prediction)
    target_ii = np.zeros(N_TOTAL_II)
    target_ii[:N_TOTAL_II - 1] = ii_norm[0, 1:]
    target_ii[N_TOTAL_II - 1]  = ii_norm[0, N_TOTAL_II - 1]   # repeat last

    target_ic = np.zeros(N_TEST_IC)
    target_ic[:N_TEST_IC - 1] = ic_norm[0, 1:]
    target_ic[N_TEST_IC - 1]  = ic_norm[0, N_TEST_IC - 1]

    # Pad ic arrays to 80 (same as ii) to keep uniform declaration size
    ic_padded = np.zeros((N_CH, N_TOTAL_II))
    ic_padded[:, :N_TEST_IC] = ic_norm
    target_ic_padded = np.zeros(N_TOTAL_II)
    target_ic_padded[:N_TEST_IC] = target_ic

    # Stats for docstring
    corr_ii = np.corrcoef(ii_norm[0, :N_TRAIN], target_ii[:N_TRAIN])[0, 1]
    corr_ic = np.corrcoef(ic_norm[0, :N_TEST_IC], target_ic[:N_TEST_IC])[0, 1]
    ch_corr  = [np.corrcoef(ii_norm[0, :N_TRAIN], ii_norm[c, :N_TRAIN])[0, 1]
                for c in range(1, N_CH)]
    sys.stderr.write(f"Inter-ictal train/target corr: {corr_ii:.3f}\n")
    sys.stderr.write(f"Ictal test/target corr: {corr_ic:.3f}\n")
    sys.stderr.write(f"Mean cross-ch corr (train): {np.mean(np.abs(ch_corr)):.3f}\n")
    sys.stderr.write(f"Ictal std change vs II: "
                     f"{ic_norm[:, :N_TEST_IC].std():.3f} vs {ii_norm.std():.3f}\n")

    # ── Write Sounio file ─────────────────────────────────────────────────────
    out.write(f"""\
//@ run-pass
//@ expect-stdout: ALL PASS

// ═══════════════════════════════════════════════════════════════════════════
// Door I: S-SSM seizure detection — subject {subject}
// ═══════════════════════════════════════════════════════════════════════════
//
// CHB-MIT Scalp EEG 1.0.0  {subject}  ({edf_path})
// 16 channels (FP1-F7 through P8-O2), 256 Hz, seizure at {seizure_s}s.
//
// Design:
//   Train on 64-sample inter-ictal window (t={INTERICTAL_START_S}s, {N_TRAIN/fs*1000:.0f}ms at 256 Hz)
//   Test A: hold-out inter-ictal (samples 64-79, {N_TEST_II/fs*1000:.0f}ms)
//   Test B: ictal onset (t={seizure_s}s, {N_TEST_IC/fs*1000:.0f}ms, fresh hidden state)
//
// Key metric: ratio = MSE_ictal / MSE_test_ii
//   G₂ sedenion learns inter-ictal coupling structure.
//   Seizures produce hypersynchrony — massive coupling disruption.
//   Hypothesis: sedenion ratio > diagonal ratio
//     (S-SSM is MORE sensitive to coupling change → larger MSE jump on ictal).
//
// Normalization: z-score by inter-ictal train window (μ,σ per channel).
// Cross-channel corr (train): {np.mean(np.abs(ch_corr)):.3f} mean
// Ictal std ratio: {ic_norm[:, :N_TEST_IC].std():.3f} vs II {ii_norm.std():.3f}

use math::sedenion::*;

// ─── Math helpers ────────────────────────────────────────────────────────────

fn sqrt_s(x: f64) -> f64 with Mut, Div, Panic {{
    if x <= 0.0 {{ return 0.0 }}
    var y: f64 = x
    var i: i64 = 0
    while i < 20 {{ y = 0.5 * (y + x / y); i = i + 1 }}
    y
}}

fn abs_s(x: f64) -> f64 {{ if x < 0.0 {{ 0.0 - x }} else {{ x }} }}

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

// ─── EEG data arrays ─────────────────────────────────────────────────────────
// II = inter-ictal [80 samples: 64 train + 16 holdout]
// IC = ictal onset [64 samples padded to 80; only first 64 used]
// TARGET_* = CH0 shifted +1 (next-step prediction task)

""")

    # Declare arrays
    for prefix in ["II", "IC"]:
        for c in range(N_CH):
            out.write(f"var {prefix}_CH{c}: [f64; 80] = [0.0; 80]\n")
    out.write("var TARGET_II: [f64; 80] = [0.0; 80]\n")
    out.write("var TARGET_IC: [f64; 80] = [0.0; 80]\n")

    out.write(f"""
// ─── Data init ───────────────────────────────────────────────────────────────
// Generated by scripts/extract_chbmit_seizure.py from {edf_path}

fn init_data() with Mut {{
""")

    for c in range(N_CH):
        ch_label = f"ch{c}"
        write_array(out, f"II_CH{c}", ii_norm[c], f"II channel {c}")
        write_array(out, f"IC_CH{c}", ic_padded[c], f"IC channel {c}")

    write_array(out, "TARGET_II", target_ii, "next-step target from II CH0")
    write_array(out, "TARGET_IC", target_ic_padded, "next-step target from IC CH0")

    out.write("}\n\n")

    # Write the rest of the experiment (model code copied from Exp 13)
    out.write("""\
// ─── Sedenion state-space model ──────────────────────────────────────────────
// Architecture identical to Exp 13 (sedenion_ssm_16ch_eeg.sio).
// S-SSM-16ch: sedenion spatial input using all 16 EEG channels.
// S-SSM-scalar: sedenion state but only CH0 as input (no spatial context).
// Real-16ch: 16 parallel AR(1)s (diagonal baseline, no coupling).

fn make_a(alpha: f64) -> Sedenion with Div, Panic {
    // A(α) = normalize( lerp(a_generic, zd_norm, α) )
    // a_generic = normalize(e0+e1), zd_norm = normalized known ZD
    let inv2: f64 = 1.0 / sqrt_s(2.0)
    let zd0: f64 = 0.5; let zd1: f64 = 0.5; let zd4: f64 = 0.5; let zd5: f64 = 0.0 - 0.5
    let g0: f64 = (1.0 - alpha) * inv2 + alpha * zd0
    let g1: f64 = (1.0 - alpha) * inv2 + alpha * zd1
    let g4: f64 =                         alpha * zd4
    let g5: f64 =                         alpha * zd5
    let n: f64 = sqrt_s(g0*g0 + g1*g1 + g4*g4 + g5*g5)
    let s: f64 = if n < 0.0000001 { 1.0 } else { 1.0 / n }
    Sedenion { e0: g0*s, e1: g1*s, e2: 0.0, e3: 0.0,
               e4: g4*s, e5: g5*s, e6: 0.0, e7: 0.0,
               e8: 0.0,  e9: 0.0,  e10: 0.0, e11: 0.0,
               e12: 0.0, e13: 0.0, e14: 0.0, e15: 0.0 }
}

// ─── Hidden state storage (16 components × 64 time steps) ────────────────────

var H0:  [f64; 64] = [0.0; 64]
var H1:  [f64; 64] = [0.0; 64]
var H2:  [f64; 64] = [0.0; 64]
var H3:  [f64; 64] = [0.0; 64]
var H4:  [f64; 64] = [0.0; 64]
var H5:  [f64; 64] = [0.0; 64]
var H6:  [f64; 64] = [0.0; 64]
var H7:  [f64; 64] = [0.0; 64]
var H8:  [f64; 64] = [0.0; 64]
var H9:  [f64; 64] = [0.0; 64]
var H10: [f64; 64] = [0.0; 64]
var H11: [f64; 64] = [0.0; 64]
var H12: [f64; 64] = [0.0; 64]
var H13: [f64; 64] = [0.0; 64]
var H14: [f64; 64] = [0.0; 64]
var H15: [f64; 64] = [0.0; 64]

// Last hidden state (warm start for test_ii)
var LAST_H0:  f64 = 0.0
var LAST_H1:  f64 = 0.0
var LAST_H2:  f64 = 0.0
var LAST_H3:  f64 = 0.0
var LAST_H4:  f64 = 0.0
var LAST_H5:  f64 = 0.0
var LAST_H6:  f64 = 0.0
var LAST_H7:  f64 = 0.0
var LAST_H8:  f64 = 0.0
var LAST_H9:  f64 = 0.0
var LAST_H10: f64 = 0.0
var LAST_H11: f64 = 0.0
var LAST_H12: f64 = 0.0
var LAST_H13: f64 = 0.0
var LAST_H14: f64 = 0.0
var LAST_H15: f64 = 0.0

fn save_h(t: i64, h: Sedenion) with Mut {
    H0[t] = h.e0;   H1[t] = h.e1;   H2[t] = h.e2;   H3[t] = h.e3
    H4[t] = h.e4;   H5[t] = h.e5;   H6[t] = h.e6;   H7[t] = h.e7
    H8[t] = h.e8;   H9[t] = h.e9;   H10[t] = h.e10; H11[t] = h.e11
    H12[t] = h.e12; H13[t] = h.e13; H14[t] = h.e14; H15[t] = h.e15
}

fn save_last_h(h: Sedenion) with Mut {
    LAST_H0  = h.e0;  LAST_H1  = h.e1;  LAST_H2  = h.e2;  LAST_H3  = h.e3
    LAST_H4  = h.e4;  LAST_H5  = h.e5;  LAST_H6  = h.e6;  LAST_H7  = h.e7
    LAST_H8  = h.e8;  LAST_H9  = h.e9;  LAST_H10 = h.e10; LAST_H11 = h.e11
    LAST_H12 = h.e12; LAST_H13 = h.e13; LAST_H14 = h.e14; LAST_H15 = h.e15
}

fn load_last_h() -> Sedenion {
    Sedenion { e0: LAST_H0, e1: LAST_H1, e2: LAST_H2, e3: LAST_H3,
               e4: LAST_H4, e5: LAST_H5, e6: LAST_H6, e7: LAST_H7,
               e8: LAST_H8, e9: LAST_H9, e10: LAST_H10, e11: LAST_H11,
               e12: LAST_H12, e13: LAST_H13, e14: LAST_H14, e15: LAST_H15 }
}

fn load_h(t: i64) -> Sedenion {
    Sedenion { e0: H0[t], e1: H1[t], e2: H2[t], e3: H3[t],
               e4: H4[t], e5: H5[t], e6: H6[t], e7: H7[t],
               e8: H8[t], e9: H9[t], e10: H10[t], e11: H11[t],
               e12: H12[t], e13: H13[t], e14: H14[t], e15: H15[t] }
}

// ─── C (readout) — stored as scaled i64 to avoid BSS-zero f64 globals ────────

var WC: [i64; 16] = [0; 16]

fn cg(i: i64) -> f64 with Div { (WC[i] as f64) / 100000000.0 }
fn cs(i: i64, v: f64) with Mut { WC[i] = (v * 100000000.0) as i64 }
fn get_c() -> Sedenion {
    Sedenion { e0: cg(0), e1: cg(1), e2: cg(2), e3: cg(3),
               e4: cg(4), e5: cg(5), e6: cg(6), e7: cg(7),
               e8: cg(8), e9: cg(9), e10: cg(10), e11: cg(11),
               e12: cg(12), e13: cg(13), e14: cg(14), e15: cg(15) }
}
fn init_c() with Mut { var i: i64 = 0; while i < 16 { WC[i] = 0; i = i + 1 } }

// ─── S-SSM-16ch forward (trains on II arrays) ─────────────────────────────────

fn ssm16_forward_ii(a: Sedenion) with Mut, Div, Panic {
    var h: Sedenion = sed_one()
    var t: i64 = 0
    while t < 64 {
        let xt: Sedenion = Sedenion {
            e0:  II_CH0[t],  e1:  II_CH1[t],  e2:  II_CH2[t],  e3:  II_CH3[t],
            e4:  II_CH4[t],  e5:  II_CH5[t],  e6:  II_CH6[t],  e7:  II_CH7[t],
            e8:  II_CH8[t],  e9:  II_CH9[t],  e10: II_CH10[t], e11: II_CH11[t],
            e12: II_CH12[t], e13: II_CH13[t], e14: II_CH14[t], e15: II_CH15[t]
        }
        let ah: Sedenion = sed_mul(a, h)
        let raw: Sedenion = sed_add(ah, xt)
        let act: Sedenion = sed_tanh(raw)
        let n: f64 = sed_norm(act)
        var hn: Sedenion = act
        if n > 0.0000001 { hn = sed_scale(act, 1.0 / n) }
        h = hn
        save_h(t, h)
        t = t + 1
    }
    save_last_h(h)
}

fn ssm_update_c(lr: f64) with Mut, Div, Panic {
    let c: Sedenion = get_c()
    var gc: Sedenion = Sedenion { e0: 0.0, e1: 0.0, e2: 0.0, e3: 0.0,
                                   e4: 0.0, e5: 0.0, e6: 0.0, e7: 0.0,
                                   e8: 0.0, e9: 0.0, e10: 0.0, e11: 0.0,
                                   e12: 0.0, e13: 0.0, e14: 0.0, e15: 0.0 }
    var t: i64 = 0
    while t < 64 {
        let ht: Sedenion = load_h(t)
        let y: f64 = sed_dot(ht, c)
        let err: f64 = y - TARGET_II[t]
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

fn ssm_train_mse() -> f64 with Mut, Div, Panic {
    let c: Sedenion = get_c()
    var local_sum: f64 = 0.0
    var t: i64 = 0
    while t < 64 {
        let ht: Sedenion = load_h(t)
        let y: f64 = sed_dot(ht, c)
        let err: f64 = y - TARGET_II[t]
        local_sum = local_sum + err * err
        t = t + 1
    }
    local_sum / 64.0
}

// ─── Test on II holdout (warm start) ─────────────────────────────────────────

fn ssm16_test_ii(a: Sedenion) -> f64 with Mut, Div, Panic {
    let c: Sedenion = get_c()
    var h: Sedenion = load_last_h()
    var local_sum: f64 = 0.0
    var t: i64 = 64
    while t < 80 {
        let xt: Sedenion = Sedenion {
            e0:  II_CH0[t],  e1:  II_CH1[t],  e2:  II_CH2[t],  e3:  II_CH3[t],
            e4:  II_CH4[t],  e5:  II_CH5[t],  e6:  II_CH6[t],  e7:  II_CH7[t],
            e8:  II_CH8[t],  e9:  II_CH9[t],  e10: II_CH10[t], e11: II_CH11[t],
            e12: II_CH12[t], e13: II_CH13[t], e14: II_CH14[t], e15: II_CH15[t]
        }
        let ah: Sedenion = sed_mul(a, h)
        let raw: Sedenion = sed_add(ah, xt)
        let act: Sedenion = sed_tanh(raw)
        let n: f64 = sed_norm(act)
        var hn: Sedenion = act
        if n > 0.0000001 { hn = sed_scale(act, 1.0 / n) }
        h = hn
        let y: f64 = sed_dot(h, c)
        let err: f64 = y - TARGET_II[t]
        local_sum = local_sum + err * err
        t = t + 1
    }
    local_sum / 16.0
}

// ─── Test on IC (fresh hidden state — cross-condition transfer) ───────────────

fn ssm16_test_ic(a: Sedenion) -> f64 with Mut, Div, Panic {
    let c: Sedenion = get_c()
    var h: Sedenion = sed_one()      // fresh state — no prior knowledge of ictal
    var local_sum: f64 = 0.0
    var t: i64 = 0
    while t < 64 {
        let xt: Sedenion = Sedenion {
            e0:  IC_CH0[t],  e1:  IC_CH1[t],  e2:  IC_CH2[t],  e3:  IC_CH3[t],
            e4:  IC_CH4[t],  e5:  IC_CH5[t],  e6:  IC_CH6[t],  e7:  IC_CH7[t],
            e8:  IC_CH8[t],  e9:  IC_CH9[t],  e10: IC_CH10[t], e11: IC_CH11[t],
            e12: IC_CH12[t], e13: IC_CH13[t], e14: IC_CH14[t], e15: IC_CH15[t]
        }
        let ah: Sedenion = sed_mul(a, h)
        let raw: Sedenion = sed_add(ah, xt)
        let act: Sedenion = sed_tanh(raw)
        let n: f64 = sed_norm(act)
        var hn: Sedenion = act
        if n > 0.0000001 { hn = sed_scale(act, 1.0 / n) }
        h = hn
        let y: f64 = sed_dot(h, c)
        let err: f64 = y - TARGET_IC[t]
        local_sum = local_sum + err * err
        t = t + 1
    }
    local_sum / 64.0
}

// ─── S-SSM-scalar (CH0 only, no spatial context) ─────────────────────────────

fn ssm_scalar_forward_ii(a: Sedenion, b: Sedenion) with Mut, Div, Panic {
    var h: Sedenion = sed_one()
    var t: i64 = 0
    while t < 64 {
        let x: f64 = II_CH0[t]
        let ah: Sedenion = sed_mul(a, h)
        let bx: Sedenion = sed_scale(b, x)
        let raw: Sedenion = sed_add(ah, bx)
        let act: Sedenion = sed_tanh(raw)
        let n: f64 = sed_norm(act)
        var hn: Sedenion = act
        if n > 0.0000001 { hn = sed_scale(act, 1.0 / n) }
        h = hn
        save_h(t, h)
        t = t + 1
    }
    save_last_h(h)
}

fn ssm_scalar_test_ii(a: Sedenion, b: Sedenion) -> f64 with Mut, Div, Panic {
    let c: Sedenion = get_c()
    var h: Sedenion = load_last_h()
    var local_sum: f64 = 0.0
    var t: i64 = 64
    while t < 80 {
        let x: f64 = II_CH0[t]
        let ah: Sedenion = sed_mul(a, h)
        let bx: Sedenion = sed_scale(b, x)
        let raw: Sedenion = sed_add(ah, bx)
        let act: Sedenion = sed_tanh(raw)
        let n: f64 = sed_norm(act)
        var hn: Sedenion = act
        if n > 0.0000001 { hn = sed_scale(act, 1.0 / n) }
        h = hn
        let y: f64 = sed_dot(h, c)
        let err: f64 = y - TARGET_II[t]
        local_sum = local_sum + err * err
        t = t + 1
    }
    local_sum / 16.0
}

fn ssm_scalar_test_ic(a: Sedenion, b: Sedenion) -> f64 with Mut, Div, Panic {
    let c: Sedenion = get_c()
    var h: Sedenion = sed_one()
    var local_sum: f64 = 0.0
    var t: i64 = 0
    while t < 64 {
        let x: f64 = IC_CH0[t]
        let ah: Sedenion = sed_mul(a, h)
        let bx: Sedenion = sed_scale(b, x)
        let raw: Sedenion = sed_add(ah, bx)
        let act: Sedenion = sed_tanh(raw)
        let n: f64 = sed_norm(act)
        var hn: Sedenion = act
        if n > 0.0000001 { hn = sed_scale(act, 1.0 / n) }
        h = hn
        let y: f64 = sed_dot(h, c)
        let err: f64 = y - TARGET_IC[t]
        local_sum = local_sum + err * err
        t = t + 1
    }
    local_sum / 64.0
}

// ─── Real-16ch diagonal baseline ─────────────────────────────────────────────

var WA_R: [f64; 16] = [0.0; 16]
var WB_R: [f64; 16] = [0.0; 16]
var WC_R: [i64; 16] = [0; 16]
var H_REAL: [f64; 1024] = [0.0; 1024]
var LAST_HR: [f64; 16] = [0.0; 16]

fn cg_r(i: i64) -> f64 with Div { (WC_R[i] as f64) / 100000000.0 }
fn cs_r(i: i64, v: f64) with Mut { WC_R[i] = (v * 100000000.0) as i64 }

fn init_real() with Mut {
    var i: i64 = 0
    while i < 16 { WA_R[i] = 0.85; WB_R[i] = 0.1; WC_R[i] = 0; i = i + 1 }
}

fn real_forward_ii() with Mut, Div, Panic {
    var h: [f64; 16] = [0.0; 16]
    var t: i64 = 0
    while t < 64 {
        var k: i64 = 0
        while k < 16 {
            let ch_val: f64 = if k == 0 { II_CH0[t] }
                else if k == 1 { II_CH1[t] }  else if k == 2 { II_CH2[t] }
                else if k == 3 { II_CH3[t] }  else if k == 4 { II_CH4[t] }
                else if k == 5 { II_CH5[t] }  else if k == 6 { II_CH6[t] }
                else if k == 7 { II_CH7[t] }  else if k == 8 { II_CH8[t] }
                else if k == 9 { II_CH9[t] }  else if k == 10 { II_CH10[t] }
                else if k == 11 { II_CH11[t] } else if k == 12 { II_CH12[t] }
                else if k == 13 { II_CH13[t] } else if k == 14 { II_CH14[t] }
                else { II_CH15[t] }
            let hk: f64 = WA_R[k] * h[k] + WB_R[k] * ch_val
            h[k] = hk
            H_REAL[(t * 16 + k) as usize] = hk
            k = k + 1
        }
        t = t + 1
    }
    var k: i64 = 0
    while k < 16 { LAST_HR[k] = h[k]; k = k + 1 }
}

fn real_update_c(lr: f64) with Mut, Div, Panic {
    var gc: [f64; 16] = [0.0; 16]
    var t: i64 = 0
    while t < 64 {
        var local_y: f64 = 0.0
        var k: i64 = 0
        while k < 16 { local_y = local_y + cg_r(k) * H_REAL[(t * 16 + k) as usize]; k = k + 1 }
        let err: f64 = local_y - TARGET_II[t]
        k = 0
        while k < 16 { gc[k] = gc[k] + err * H_REAL[(t * 16 + k) as usize]; k = k + 1 }
        t = t + 1
    }
    let scale: f64 = lr * 2.0 / 64.0
    var k: i64 = 0
    while k < 16 { cs_r(k, cg_r(k) - scale * gc[k]); k = k + 1 }
}

fn real_train_mse() -> f64 with Div, Panic {
    var local_sum: f64 = 0.0
    var t: i64 = 0
    while t < 64 {
        var local_y: f64 = 0.0
        var k: i64 = 0
        while k < 16 { local_y = local_y + cg_r(k) * H_REAL[(t * 16 + k) as usize]; k = k + 1 }
        let err: f64 = local_y - TARGET_II[t]
        local_sum = local_sum + err * err
        t = t + 1
    }
    local_sum / 64.0
}

fn real_test_ii() -> f64 with Mut, Div, Panic {
    var h: [f64; 16] = [0.0; 16]
    var k: i64 = 0
    while k < 16 { h[k] = LAST_HR[k]; k = k + 1 }
    var local_sum: f64 = 0.0
    var t: i64 = 64
    while t < 80 {
        k = 0
        while k < 16 {
            let ch_val: f64 = if k == 0 { II_CH0[t] }
                else if k == 1 { II_CH1[t] }  else if k == 2 { II_CH2[t] }
                else if k == 3 { II_CH3[t] }  else if k == 4 { II_CH4[t] }
                else if k == 5 { II_CH5[t] }  else if k == 6 { II_CH6[t] }
                else if k == 7 { II_CH7[t] }  else if k == 8 { II_CH8[t] }
                else if k == 9 { II_CH9[t] }  else if k == 10 { II_CH10[t] }
                else if k == 11 { II_CH11[t] } else if k == 12 { II_CH12[t] }
                else if k == 13 { II_CH13[t] } else if k == 14 { II_CH14[t] }
                else { II_CH15[t] }
            h[k] = WA_R[k] * h[k] + WB_R[k] * ch_val
            k = k + 1
        }
        var local_y: f64 = 0.0
        k = 0
        while k < 16 { local_y = local_y + cg_r(k) * h[k]; k = k + 1 }
        let err: f64 = local_y - TARGET_II[t]
        local_sum = local_sum + err * err
        t = t + 1
    }
    local_sum / 16.0
}

fn real_test_ic() -> f64 with Mut, Div, Panic {
    var h: [f64; 16] = [0.0; 16]   // fresh state for ictal
    var local_sum: f64 = 0.0
    var t: i64 = 0
    while t < 64 {
        var k: i64 = 0
        while k < 16 {
            let ch_val: f64 = if k == 0 { IC_CH0[t] }
                else if k == 1 { IC_CH1[t] }  else if k == 2 { IC_CH2[t] }
                else if k == 3 { IC_CH3[t] }  else if k == 4 { IC_CH4[t] }
                else if k == 5 { IC_CH5[t] }  else if k == 6 { IC_CH6[t] }
                else if k == 7 { IC_CH7[t] }  else if k == 8 { IC_CH8[t] }
                else if k == 9 { IC_CH9[t] }  else if k == 10 { IC_CH10[t] }
                else if k == 11 { IC_CH11[t] } else if k == 12 { IC_CH12[t] }
                else if k == 13 { IC_CH13[t] } else if k == 14 { IC_CH14[t] }
                else { IC_CH15[t] }
            h[k] = WA_R[k] * h[k] + WB_R[k] * ch_val
            k = k + 1
        }
        var local_y: f64 = 0.0
        var k2: i64 = 0
        while k2 < 16 { local_y = local_y + cg_r(k2) * h[k2]; k2 = k2 + 1 }
        let err: f64 = local_y - TARGET_IC[t]
        local_sum = local_sum + err * err
        t = t + 1
    }
    local_sum / 64.0
}

// ─── Main ────────────────────────────────────────────────────────────────────
""")

    out.write(f"""\
fn main() -> i32 with IO, Mut, Div, Panic {{
    init_data()

    let a02: Sedenion = make_a(0.2)
    let b: Sedenion = sed_one()

    println("═══════════════════════════════════════════════════════════════")
    println("  Door I: S-SSM seizure detection  subject={subject}")
    println("  CHB-MIT {cfg['edf']}  seizure_start={seizure_s}s  256 Hz")
    println("  Train: 64 inter-ictal samples  Test: 16 II holdout + 64 ictal")
    println("═══════════════════════════════════════════════════════════════")
    println("")

    // ── S-SSM-16ch ───────────────────────────────────────────────────────────
    // lr=0.2, 200 epochs. Sedenion state normalization provides implicit
    // gradient stability; more epochs compensates for reduced lr.
    init_c()
    var ep: i64 = 0
    while ep < 200 {{ ssm16_forward_ii(a02); ssm_update_c(0.2); ep = ep + 1 }}
    ssm16_forward_ii(a02)
    let s16_train: f64 = ssm_train_mse()
    let s16_ii:    f64 = ssm16_test_ii(a02)
    let s16_ic:    f64 = ssm16_test_ic(a02)
    print("  S-SSM-16ch  α=0.2  train="); print(s16_train)
    print("  test_ii="); print(s16_ii)
    print("  test_ic="); print(s16_ic); println("")

    // ── S-SSM-scalar ─────────────────────────────────────────────────────────
    init_c()
    ep = 0
    while ep < 200 {{ ssm_scalar_forward_ii(a02, b); ssm_update_c(0.2); ep = ep + 1 }}
    ssm_scalar_forward_ii(a02, b)
    let sc_train: f64 = ssm_train_mse()
    let sc_ii:    f64 = ssm_scalar_test_ii(a02, b)
    let sc_ic:    f64 = ssm_scalar_test_ic(a02, b)
    print("  S-SSM-scalar α=0.2  train="); print(sc_train)
    print("  test_ii="); print(sc_ii)
    print("  test_ic="); print(sc_ic); println("")

    // ── Real-16ch ────────────────────────────────────────────────────────────
    // lr=0.01: diagonal AR has no state normalization; needs smaller lr to
    // prevent gradient explosion with 16-dim readout on real EEG.
    init_real()
    ep = 0
    while ep < 200 {{ real_forward_ii(); real_update_c(0.01); ep = ep + 1 }}
    real_forward_ii()
    let r16_train: f64 = real_train_mse()
    let r16_ii:    f64 = real_test_ii()
    let r16_ic:    f64 = real_test_ic()
    print("  Real-16ch (diag AR)  train="); print(r16_train)
    print("  test_ii="); print(r16_ii)
    print("  test_ic="); print(r16_ic); println("")
    println("")

    // ── Seizure sensitivity ratios ────────────────────────────────────────────
    let s16_ratio: f64 = if s16_ii > 0.000001 {{ s16_ic / s16_ii }} else {{ 0.0 }}
    let sc_ratio:  f64 = if sc_ii  > 0.000001 {{ sc_ic  / sc_ii  }} else {{ 0.0 }}
    let r16_ratio: f64 = if r16_ii > 0.000001 {{ r16_ic / r16_ii }} else {{ 0.0 }}

    print("  Seizure sensitivity ratio  (MSE_ic / MSE_ii):"); println("")
    print("    S-SSM-16ch  ratio="); print(s16_ratio); println("")
    print("    S-SSM-scalar ratio="); print(sc_ratio); println("")
    print("    Real-16ch   ratio="); print(r16_ratio); println("")
    println("")
    print("  G2 seizure amplification vs diagonal: ")
    let delta: f64 = s16_ratio - r16_ratio
    print(delta); println("")
    println("")

    var passed: i32 = 0
    var failed: i32 = 0

    // T1: all MSE values finite and non-negative
    if s16_train >= 0.0 && s16_ii >= 0.0 && s16_ic >= 0.0
        && r16_train >= 0.0 && r16_ii >= 0.0 && r16_ic >= 0.0 {{
        print("T1 PASS: all MSE values finite"); println("")
        passed = passed + 1
    }} else {{
        print("T1 FAIL: negative MSE"); println("")
        failed = failed + 1
    }}

    // T2: S-SSM-16ch trains to below unit variance (model does something)
    if s16_train < 1.5 {{
        print("T2 PASS: S-SSM-16ch trains (train MSE < 1.5)"); println("")
        passed = passed + 1
    }} else {{
        print("T2 FAIL: S-SSM-16ch did not converge"); println("")
        failed = failed + 1
    }}

    // T3: Real baseline trains
    if r16_train < 1.5 {{
        print("T3 PASS: Real-16ch trains"); println("")
        passed = passed + 1
    }} else {{
        print("T3 FAIL: Real-16ch did not converge"); println("")
        failed = failed + 1
    }}

    // T4: ratios are computed (not zero)
    if s16_ratio > 0.0 && r16_ratio > 0.0 {{
        print("T4 PASS: ratios computed"); println("")
        passed = passed + 1
    }} else {{
        print("T4 FAIL: ratio is zero"); println("")
        failed = failed + 1
    }}

    // T5 REPORT: seizure sensitivity amplification (main scientific result)
    print("T5 REPORT: G2 seizure amplification = "); print(delta)
    if delta > 0.0 {{
        println(" (S-SSM more seizure-sensitive than diagonal — G2 coupling signal)")
    }} else {{
        println(" (diagonal more seizure-sensitive — no G2 coupling signal)")
    }}
    passed = passed + 1

    println("")
    if failed == 0 {{ println("ALL PASS") }}
    else {{ print("FAILURES: "); print(failed); println("") }}

    println("")
    print("  Passed: "); print(passed); println("")
    print("  Failed: "); print(failed); println("")

    0
}}
""")


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in SUBJECTS:
        sys.stderr.write(f"Usage: {sys.argv[0]} <subject>\n")
        sys.stderr.write(f"  subjects: {', '.join(SUBJECTS.keys())}\n")
        sys.exit(1)
    subject = sys.argv[1]
    cfg = SUBJECTS[subject]
    generate_sounio(subject, cfg)


if __name__ == "__main__":
    main()
