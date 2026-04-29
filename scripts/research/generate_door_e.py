#!/usr/bin/env python3
"""Door E: Generate α-mechanism Sounio experiment for chb03.

Extracts chb03 data (same as Door I), then writes door_e_alpha_mechanism.sio:
  - Embeds II and IC data arrays
  - α sweep: {0.05, 0.10, 0.20, 0.30, 0.50, 0.80}
  - For each α: train C (200 epochs), then report:
      1. Seizure sensitivity ratio = MSE_ic / MSE_ii
      2. Hidden state separation  = sed_norm(h_ii_final - h_ic_final)
      3. Associator ratio         = mean_assoc_ic / mean_assoc_ii

Door H killed linear resonance. Door E exposes what α=0.20 actually optimizes:
the hidden state geometric separation between interictal and ictal regimes.

The associator ||(A⊗h)⊗x − A⊗(h⊗x)|| directly measures non-associativity
of the sedenion product at each step — a structural signal, not a prediction error.
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
N_TEST_II    = 16
N_TOTAL_II   = N_TRAIN + N_TEST_II
N_TEST_IC    = 64
INTERICTAL_S = 10.0
OUT_PATH     = "examples/door_e_alpha_mechanism.sio"

ALPHAS = [0.05, 0.10, 0.20, 0.30, 0.50, 0.80]


def load_subject(edf_path):
    f = pyedflib.EdfReader(edf_path)
    sigs = np.stack([f.readSignal(i) for i in range(N_CH)])
    f.close()
    return sigs


def normalize(sigs):
    ii_start = int(INTERICTAL_S * FS)
    ic_start = int(SEIZURE_S * FS)

    ii_raw = sigs[:, ii_start : ii_start + N_TOTAL_II]
    ic_raw = sigs[:, ic_start : ic_start + N_TEST_IC]

    means  = ii_raw[:, :N_TRAIN].mean(axis=1, keepdims=True)
    stds   = ii_raw[:, :N_TRAIN].std(axis=1, keepdims=True)
    stds   = np.where(stds < 1e-6, 1.0, stds)
    ii_z   = (ii_raw - means) / stds
    ic_z   = (ic_raw - means) / stds

    max_abs = np.abs(ii_z[:, :N_TRAIN]).max(axis=1, keepdims=True)
    max_abs = np.where(max_abs < 1e-6, 1.0, max_abs)

    ii_norm = ii_z / max_abs
    ic_norm = np.clip(ic_z / max_abs, -5.0, 5.0)

    target_ii = np.zeros(N_TOTAL_II)
    target_ii[:N_TOTAL_II - 1] = ii_norm[0, 1:]
    target_ii[N_TOTAL_II - 1]  = ii_norm[0, N_TOTAL_II - 1]

    target_ic = np.zeros(N_TOTAL_II)
    target_ic[:N_TEST_IC - 1] = ic_norm[0, 1:]
    target_ic[N_TEST_IC - 1]  = ic_norm[0, N_TEST_IC - 1]

    ic_padded = np.zeros((N_CH, N_TOTAL_II))
    ic_padded[:, :N_TEST_IC] = ic_norm

    return ii_norm, ic_padded, target_ii, target_ic


def fmt(v):
    return f"{v:.8f}"


def write_array_inits(f, name, values):
    for i, v in enumerate(values):
        f.write(f"    {name}[{i}] = {fmt(v)}\n")


def main():
    if not os.path.exists(EDF_PATH):
        print(f"ERROR: {EDF_PATH} not found", file=sys.stderr)
        sys.exit(1)

    sigs = load_subject(EDF_PATH)
    ii_norm, ic_padded, target_ii, target_ic = normalize(sigs)
    sys.stderr.write(f"chb03: II std={ii_norm.std():.3f}  IC std={ic_padded.std():.3f}\n")

    f = open(OUT_PATH, "w")

    alpha_list = ", ".join(str(a) for a in ALPHAS)
    f.write(f"""\
//@ run-pass
//@ expect-stdout: ALL PASS

// ═══════════════════════════════════════════════════════════════════════════
// Door E: α mechanism — hidden state separation + associator ratio
// ═══════════════════════════════════════════════════════════════════════════
//
// Data: CHB-MIT chb03  ({EDF_PATH})
// Seizure at {SEIZURE_S}s, 256 Hz, 16 channels.
//
// Door H proved α=0.20 is NOT a linear resonance phenomenon.
// Door III (Python) confirmed α=0.20 is the population interior optimum.
// This experiment probes WHY: for each α ∈ {{{alpha_list}}}, measure:
//   1. ratio      = MSE_ic / MSE_ii           (seizure sensitivity)
//   2. h_sep      = sed_norm(h_ii - h_ic)     (hidden state separation)
//   3. assoc_ii   = mean ||(A⊗h)⊗x - A⊗(h⊗x)|| over interictal steps
//   4. assoc_ic   = same quantity over ictal steps
//   5. assoc_ratio = assoc_ic / assoc_ii      (structural disruption)
//
// Hypothesis: h_sep and assoc_ratio both peak at α=0.20.

use math::sedenion::*;

// ─── Math helpers ────────────────────────────────────────────────────────────

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

// ─── Data arrays ─────────────────────────────────────────────────────────────

var II_CH0:  [f64; 80] = [0.0; 80]   var II_CH1:  [f64; 80] = [0.0; 80]
var II_CH2:  [f64; 80] = [0.0; 80]   var II_CH3:  [f64; 80] = [0.0; 80]
var II_CH4:  [f64; 80] = [0.0; 80]   var II_CH5:  [f64; 80] = [0.0; 80]
var II_CH6:  [f64; 80] = [0.0; 80]   var II_CH7:  [f64; 80] = [0.0; 80]
var II_CH8:  [f64; 80] = [0.0; 80]   var II_CH9:  [f64; 80] = [0.0; 80]
var II_CH10: [f64; 80] = [0.0; 80]   var II_CH11: [f64; 80] = [0.0; 80]
var II_CH12: [f64; 80] = [0.0; 80]   var II_CH13: [f64; 80] = [0.0; 80]
var II_CH14: [f64; 80] = [0.0; 80]   var II_CH15: [f64; 80] = [0.0; 80]

var IC_CH0:  [f64; 80] = [0.0; 80]   var IC_CH1:  [f64; 80] = [0.0; 80]
var IC_CH2:  [f64; 80] = [0.0; 80]   var IC_CH3:  [f64; 80] = [0.0; 80]
var IC_CH4:  [f64; 80] = [0.0; 80]   var IC_CH5:  [f64; 80] = [0.0; 80]
var IC_CH6:  [f64; 80] = [0.0; 80]   var IC_CH7:  [f64; 80] = [0.0; 80]
var IC_CH8:  [f64; 80] = [0.0; 80]   var IC_CH9:  [f64; 80] = [0.0; 80]
var IC_CH10: [f64; 80] = [0.0; 80]   var IC_CH11: [f64; 80] = [0.0; 80]
var IC_CH12: [f64; 80] = [0.0; 80]   var IC_CH13: [f64; 80] = [0.0; 80]
var IC_CH14: [f64; 80] = [0.0; 80]   var IC_CH15: [f64; 80] = [0.0; 80]

var TARGET_II: [f64; 80] = [0.0; 80]
var TARGET_IC: [f64; 80] = [0.0; 80]

fn init_data() with Mut {{
""")

    for ch in range(N_CH):
        write_array_inits(f, f"II_CH{ch}", ii_norm[ch])
    for ch in range(N_CH):
        write_array_inits(f, f"IC_CH{ch}", ic_padded[ch])
    write_array_inits(f, "TARGET_II", target_ii)
    write_array_inits(f, "TARGET_IC", target_ic)
    f.write("}\n\n")

    f.write("""\
// ─── Sedenion gate (same lerp-to-ZD parameterization as Door I) ──────────────

fn make_a(alpha: f64) -> Sedenion with Div, Panic {
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

// ─── Hidden state storage ─────────────────────────────────────────────────────

var H0:  [f64; 64] = [0.0; 64]   var H1:  [f64; 64] = [0.0; 64]
var H2:  [f64; 64] = [0.0; 64]   var H3:  [f64; 64] = [0.0; 64]
var H4:  [f64; 64] = [0.0; 64]   var H5:  [f64; 64] = [0.0; 64]
var H6:  [f64; 64] = [0.0; 64]   var H7:  [f64; 64] = [0.0; 64]
var H8:  [f64; 64] = [0.0; 64]   var H9:  [f64; 64] = [0.0; 64]
var H10: [f64; 64] = [0.0; 64]   var H11: [f64; 64] = [0.0; 64]
var H12: [f64; 64] = [0.0; 64]   var H13: [f64; 64] = [0.0; 64]
var H14: [f64; 64] = [0.0; 64]   var H15: [f64; 64] = [0.0; 64]

var LAST_H0: f64=0.0  var LAST_H1: f64=0.0  var LAST_H2:  f64=0.0  var LAST_H3:  f64=0.0
var LAST_H4: f64=0.0  var LAST_H5: f64=0.0  var LAST_H6:  f64=0.0  var LAST_H7:  f64=0.0
var LAST_H8: f64=0.0  var LAST_H9: f64=0.0  var LAST_H10: f64=0.0  var LAST_H11: f64=0.0
var LAST_H12: f64=0.0 var LAST_H13: f64=0.0 var LAST_H14: f64=0.0  var LAST_H15: f64=0.0

fn save_h(t: i64, h: Sedenion) with Mut {
    H0[t]=h.e0; H1[t]=h.e1; H2[t]=h.e2; H3[t]=h.e3
    H4[t]=h.e4; H5[t]=h.e5; H6[t]=h.e6; H7[t]=h.e7
    H8[t]=h.e8; H9[t]=h.e9; H10[t]=h.e10; H11[t]=h.e11
    H12[t]=h.e12; H13[t]=h.e13; H14[t]=h.e14; H15[t]=h.e15
}

fn save_last_h(h: Sedenion) with Mut {
    LAST_H0=h.e0; LAST_H1=h.e1; LAST_H2=h.e2; LAST_H3=h.e3
    LAST_H4=h.e4; LAST_H5=h.e5; LAST_H6=h.e6; LAST_H7=h.e7
    LAST_H8=h.e8; LAST_H9=h.e9; LAST_H10=h.e10; LAST_H11=h.e11
    LAST_H12=h.e12; LAST_H13=h.e13; LAST_H14=h.e14; LAST_H15=h.e15
}

fn load_last_h() -> Sedenion {
    Sedenion { e0:LAST_H0, e1:LAST_H1, e2:LAST_H2, e3:LAST_H3,
               e4:LAST_H4, e5:LAST_H5, e6:LAST_H6, e7:LAST_H7,
               e8:LAST_H8, e9:LAST_H9, e10:LAST_H10, e11:LAST_H11,
               e12:LAST_H12, e13:LAST_H13, e14:LAST_H14, e15:LAST_H15 }
}

fn load_h(t: i64) -> Sedenion {
    Sedenion { e0:H0[t], e1:H1[t], e2:H2[t], e3:H3[t],
               e4:H4[t], e5:H5[t], e6:H6[t], e7:H7[t],
               e8:H8[t], e9:H9[t], e10:H10[t], e11:H11[t],
               e12:H12[t], e13:H13[t], e14:H14[t], e15:H15[t] }
}

var WC: [i64; 16] = [0; 16]
fn cg(i: i64) -> f64 with Div { (WC[i] as f64) / 100000000.0 }
fn cs(i: i64, v: f64) with Mut { WC[i] = (v * 100000000.0) as i64 }
fn get_c() -> Sedenion {
    Sedenion { e0:cg(0), e1:cg(1), e2:cg(2), e3:cg(3),
               e4:cg(4), e5:cg(5), e6:cg(6), e7:cg(7),
               e8:cg(8), e9:cg(9), e10:cg(10), e11:cg(11),
               e12:cg(12), e13:cg(13), e14:cg(14), e15:cg(15) }
}
fn init_c() with Mut { var i: i64 = 0; while i < 16 { WC[i] = 0; i = i + 1 } }

// ─── S-SSM-16ch forward ───────────────────────────────────────────────────────

fn ssm16_forward_ii(a: Sedenion) with Mut, Div, Panic {
    var h: Sedenion = sed_one()
    var t: i64 = 0
    while t < 64 {
        let xt: Sedenion = Sedenion {
            e0: II_CH0[t],  e1: II_CH1[t],  e2: II_CH2[t],  e3: II_CH3[t],
            e4: II_CH4[t],  e5: II_CH5[t],  e6: II_CH6[t],  e7: II_CH7[t],
            e8: II_CH8[t],  e9: II_CH9[t],  e10: II_CH10[t], e11: II_CH11[t],
            e12: II_CH12[t], e13: II_CH13[t], e14: II_CH14[t], e15: II_CH15[t]
        }
        let ah: Sedenion = sed_mul(a, h)
        let raw: Sedenion = sed_add(ah, xt)
        let act: Sedenion = sed_tanh_s(raw)
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
    var gc: Sedenion = Sedenion { e0:0.0, e1:0.0, e2:0.0, e3:0.0,
                                   e4:0.0, e5:0.0, e6:0.0, e7:0.0,
                                   e8:0.0, e9:0.0, e10:0.0, e11:0.0,
                                   e12:0.0, e13:0.0, e14:0.0, e15:0.0 }
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

fn ssm16_mse_ii(a: Sedenion) -> f64 with Mut, Div, Panic {
    let c: Sedenion = get_c()
    var local_sum: f64 = 0.0
    var t: i64 = 64
    while t < 80 {
        let xt: Sedenion = Sedenion {
            e0: II_CH0[t],  e1: II_CH1[t],  e2: II_CH2[t],  e3: II_CH3[t],
            e4: II_CH4[t],  e5: II_CH5[t],  e6: II_CH6[t],  e7: II_CH7[t],
            e8: II_CH8[t],  e9: II_CH9[t],  e10: II_CH10[t], e11: II_CH11[t],
            e12: II_CH12[t], e13: II_CH13[t], e14: II_CH14[t], e15: II_CH15[t]
        }
        var h: Sedenion = load_last_h()
        let ah: Sedenion = sed_mul(a, h)
        let raw: Sedenion = sed_add(ah, xt)
        let act: Sedenion = sed_tanh_s(raw)
        let n: f64 = sed_norm(act)
        if n > 0.0000001 { h = sed_scale(act, 1.0 / n) } else { h = act }
        let y: f64 = sed_dot(h, c)
        let err: f64 = y - TARGET_II[t]
        local_sum = local_sum + err * err
        t = t + 1
    }
    local_sum / 16.0
}

fn ssm16_mse_ic(a: Sedenion) -> f64 with Mut, Div, Panic {
    let c: Sedenion = get_c()
    var h: Sedenion = sed_one()
    var local_sum: f64 = 0.0
    var t: i64 = 0
    while t < 64 {
        let xt: Sedenion = Sedenion {
            e0: IC_CH0[t],  e1: IC_CH1[t],  e2: IC_CH2[t],  e3: IC_CH3[t],
            e4: IC_CH4[t],  e5: IC_CH5[t],  e6: IC_CH6[t],  e7: IC_CH7[t],
            e8: IC_CH8[t],  e9: IC_CH9[t],  e10: IC_CH10[t], e11: IC_CH11[t],
            e12: IC_CH12[t], e13: IC_CH13[t], e14: IC_CH14[t], e15: IC_CH15[t]
        }
        let ah: Sedenion = sed_mul(a, h)
        let raw: Sedenion = sed_add(ah, xt)
        let act: Sedenion = sed_tanh_s(raw)
        let n: f64 = sed_norm(act)
        if n > 0.0000001 { h = sed_scale(act, 1.0 / n) } else { h = act }
        let y: f64 = sed_dot(h, c)
        let err: f64 = y - TARGET_IC[t]
        local_sum = local_sum + err * err
        t = t + 1
    }
    local_sum / 64.0
}

// ─── Final hidden state (for h_sep computation) ───────────────────────────────

fn hidden_final_ic(a: Sedenion) -> Sedenion with Div, Panic {
    // Run S-SSM fresh on ictal data, return final hidden state
    var h: Sedenion = sed_one()
    var t: i64 = 0
    while t < 64 {
        let xt: Sedenion = Sedenion {
            e0: IC_CH0[t],  e1: IC_CH1[t],  e2: IC_CH2[t],  e3: IC_CH3[t],
            e4: IC_CH4[t],  e5: IC_CH5[t],  e6: IC_CH6[t],  e7: IC_CH7[t],
            e8: IC_CH8[t],  e9: IC_CH9[t],  e10: IC_CH10[t], e11: IC_CH11[t],
            e12: IC_CH12[t], e13: IC_CH13[t], e14: IC_CH14[t], e15: IC_CH15[t]
        }
        let ah: Sedenion = sed_mul(a, h)
        let raw: Sedenion = sed_add(ah, xt)
        let act: Sedenion = sed_tanh_s(raw)
        let n: f64 = sed_norm(act)
        if n > 0.0000001 { h = sed_scale(act, 1.0 / n) } else { h = act }
        t = t + 1
    }
    h
}

fn h_sep(a: Sedenion) -> f64 with Mut, Div, Panic {
    // ||h_ii_final - h_ic_final||
    // h_ii is in LAST_H (from prior ssm16_forward_ii call)
    let h_ii: Sedenion = load_last_h()
    let h_ic: Sedenion = hidden_final_ic(a)
    sed_norm(sed_sub(h_ii, h_ic))
}

// ─── Associator norm ─────────────────────────────────────────────────────────
// For sedenion: assoc(A, h, x) = ||(A⊗h)⊗x − A⊗(h⊗x)||
// Measures geometric non-associativity at each step.
// Non-zero because sedenions are non-associative (unlike octonions: flexible).
// In interictal: brain has stable coupling → characteristic assoc signature.
// At ictal onset: hypersynchrony disrupts coupling → assoc changes.

fn assoc_ii(a: Sedenion) -> f64 with Div, Panic {
    var h: Sedenion = sed_one()
    var total: f64 = 0.0
    var t: i64 = 0
    while t < 64 {
        let xt: Sedenion = Sedenion {
            e0: II_CH0[t],  e1: II_CH1[t],  e2: II_CH2[t],  e3: II_CH3[t],
            e4: II_CH4[t],  e5: II_CH5[t],  e6: II_CH6[t],  e7: II_CH7[t],
            e8: II_CH8[t],  e9: II_CH9[t],  e10: II_CH10[t], e11: II_CH11[t],
            e12: II_CH12[t], e13: II_CH13[t], e14: II_CH14[t], e15: II_CH15[t]
        }
        let ah: Sedenion = sed_mul(a, h)
        let ahx: Sedenion = sed_mul(ah, xt)         // (A⊗h)⊗x
        let hx: Sedenion = sed_mul(h, xt)
        let a_hx: Sedenion = sed_mul(a, hx)         // A⊗(h⊗x)
        let diff: Sedenion = sed_sub(ahx, a_hx)
        let an: f64 = sed_norm(diff)
        total = total + an
        // Advance hidden state (same recurrence)
        let raw: Sedenion = sed_add(ah, xt)
        let act: Sedenion = sed_tanh_s(raw)
        let n: f64 = sed_norm(act)
        if n > 0.0000001 { h = sed_scale(act, 1.0 / n) } else { h = act }
        t = t + 1
    }
    total / 64.0
}

fn assoc_ic(a: Sedenion) -> f64 with Div, Panic {
    var h: Sedenion = sed_one()
    var total: f64 = 0.0
    var t: i64 = 0
    while t < 64 {
        let xt: Sedenion = Sedenion {
            e0: IC_CH0[t],  e1: IC_CH1[t],  e2: IC_CH2[t],  e3: IC_CH3[t],
            e4: IC_CH4[t],  e5: IC_CH5[t],  e6: IC_CH6[t],  e7: IC_CH7[t],
            e8: IC_CH8[t],  e9: IC_CH9[t],  e10: IC_CH10[t], e11: IC_CH11[t],
            e12: IC_CH12[t], e13: IC_CH13[t], e14: IC_CH14[t], e15: IC_CH15[t]
        }
        let ah: Sedenion = sed_mul(a, h)
        let ahx: Sedenion = sed_mul(ah, xt)
        let hx: Sedenion = sed_mul(h, xt)
        let a_hx: Sedenion = sed_mul(a, hx)
        let diff: Sedenion = sed_sub(ahx, a_hx)
        let an: f64 = sed_norm(diff)
        total = total + an
        let raw: Sedenion = sed_add(ah, xt)
        let act: Sedenion = sed_tanh_s(raw)
        let n: f64 = sed_norm(act)
        if n > 0.0000001 { h = sed_scale(act, 1.0 / n) } else { h = act }
        t = t + 1
    }
    total / 64.0
}

// ─── Run one α ───────────────────────────────────────────────────────────────

fn run_alpha(alpha: f64) with IO, Mut, Div, Panic {
    let a: Sedenion = make_a(alpha)

    // Train C on interictal (200 epochs)
    init_c()
    var ep: i64 = 0
    while ep < 200 { ssm16_forward_ii(a); ssm_update_c(0.2); ep = ep + 1 }
    ssm16_forward_ii(a)   // final pass to populate H and LAST_H

    // Metrics
    let mse_ii:   f64 = ssm16_mse_ii(a)
    let mse_ic:   f64 = ssm16_mse_ic(a)
    let ratio:    f64 = if mse_ii > 0.000001 { mse_ic / mse_ii } else { 0.0 }
    let sep:      f64 = h_sep(a)
    let a_ii:     f64 = assoc_ii(a)
    let a_ic:     f64 = assoc_ic(a)
    let a_ratio:  f64 = if a_ii > 0.000001 { a_ic / a_ii } else { 0.0 }

    print("  α="); print(alpha)
    print("  ratio="); print(ratio)
    print("  h_sep="); print(sep)
    print("  assoc_ii="); print(a_ii)
    print("  assoc_ic="); print(a_ic)
    print("  assoc_ratio="); print(a_ratio)
    println("")
}

// ─── Main ────────────────────────────────────────────────────────────────────

fn main() -> i32 with IO, Mut, Div, Panic {
    init_data()

    println("═══════════════════════════════════════════════════════════════════════")
    println("  Door E: α mechanism — chb03 seizure at 362s")
    println("  Metrics per α: sensitivity ratio | h_sep | assoc_ratio")
    println("  Null: flat landscape. Alt: interior peak at α=0.20 for all metrics.")
    println("═══════════════════════════════════════════════════════════════════════")
    println("")
    println("  α        ratio      h_sep      assoc_ii   assoc_ic   assoc_ratio")
    println("  ─────────────────────────────────────────────────────────────────")
""")

    # Write run_alpha calls for each α
    for alpha in ALPHAS:
        f.write(f"    run_alpha({alpha})\n")

    f.write("""
    println("")

    var passed: i32 = 0
    var failed: i32 = 0

    // T1: ratio at α=0.20 is positive (model works)
    let a02: Sedenion = make_a(0.2)
    init_c()
    var ep: i64 = 0
    while ep < 200 { ssm16_forward_ii(a02); ssm_update_c(0.2); ep = ep + 1 }
    ssm16_forward_ii(a02)
    let r02: f64 = ssm16_mse_ic(a02)
    let ii02: f64 = ssm16_mse_ii(a02)
    let ratio02: f64 = if ii02 > 0.000001 { r02 / ii02 } else { 0.0 }
    if ratio02 > 1.0 {
        print("T1 PASS: ratio at α=0.20 = "); print(ratio02); println(" > 1 (seizure detected)")
        passed = passed + 1
    } else {
        print("T1 FAIL: ratio at α=0.20 = "); print(ratio02); println(" ≤ 1")
        failed = failed + 1
    }

    // T2: associator is non-zero (sedenion non-associativity is real)
    let a_ii_02: f64 = assoc_ii(a02)
    if a_ii_02 > 0.0 {
        print("T2 PASS: assoc_ii at α=0.20 = "); print(a_ii_02); println(" > 0")
        passed = passed + 1
    } else {
        println("T2 FAIL: assoc_ii = 0 (unexpected)")
        failed = failed + 1
    }

    // T3: h_sep is non-zero (hidden states differ between II and IC)
    let sep02: f64 = h_sep(a02)
    if sep02 > 0.0 {
        print("T3 PASS: h_sep at α=0.20 = "); print(sep02); println(" > 0")
        passed = passed + 1
    } else {
        println("T3 FAIL: h_sep = 0 (II and IC hidden states identical)")
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

    f.close()
    sys.stderr.write(f"Written: {OUT_PATH}\n")


if __name__ == "__main__":
    main()
