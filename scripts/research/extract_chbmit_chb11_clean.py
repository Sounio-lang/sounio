#!/usr/bin/env python3
"""Door D: Extract chb11 data with cleaned channels → generate Sounio experiment.

chb11_82.edf has 28 channels. Positions 4,9,12,17,22 are blank '-' channels
(std=0). The Door C extractor naively used positions 0-15, which included blanks
at 4,9,12. This explains why S-SSM-scalar > S-SSM-16ch for chb11 in Door C.

Fix: select 16 real channels, skipping all blanks:
  ch_map = [0,1,2,3,5,6,7,8,10,11,13,14,15,16,18,19]

Generates: examples/sedenion_ssm_seizure_chb11.sio
Usage:
    python3 scripts/research/extract_chbmit_chb11_clean.py
"""
import pyedflib
import numpy as np
import sys
import os

EDF_PATH       = "data/chbmit/chb11_82.edf"
SEIZURE_S      = 298
N_CH_EDF       = 28
FS             = 256
INTERICTAL_S   = 10.0
N_TRAIN        = 64
N_TEST_II      = 16
N_TOTAL_II     = N_TRAIN + N_TEST_II
N_TEST_IC      = 64
N_CH           = 16
OUT_PATH       = "examples/sedenion_ssm_seizure_chb11.sio"

# Real channels in chb11_82.edf (skipping blanks at positions 4,9,12,17,22)
CH_MAP = [0, 1, 2, 3, 5, 6, 7, 8, 10, 11, 13, 14, 15, 16, 18, 19]
assert len(CH_MAP) == N_CH


def load_channels(edf_path):
    f = pyedflib.EdfReader(edf_path)
    sys.stderr.write(f"Loading {edf_path}: {f.signals_in_file} channels\n")
    sigs = np.stack([f.readSignal(i) for i in CH_MAP])  # (16, T)
    sr = int(f.getSampleFrequency(CH_MAP[0]))
    labels = [f.getSignalLabels()[i].strip() for i in CH_MAP]
    f.close()
    sys.stderr.write(f"  Selected channels: {labels}\n")
    sys.stderr.write(f"  Signal shape: {sigs.shape}  fs={sr}\n")
    return sigs, sr


def fmt(v):
    return f"{v:.8f}"


def write_array_inits(out, arr_name, values):
    for i, v in enumerate(values):
        out.write(f"    {arr_name}[{i}] = {fmt(v)}\n")


def generate_sio(sigs, sr):
    ii_start = int(INTERICTAL_S * sr)
    ic_start = int(SEIZURE_S * sr)

    ii_raw = sigs[:, ii_start : ii_start + N_TOTAL_II]   # (16, 80)
    ic_raw = sigs[:, ic_start : ic_start + N_TEST_IC]    # (16, 64)

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

    ch_corr = [np.corrcoef(ii_norm[0, :N_TRAIN], ii_norm[c, :N_TRAIN])[0, 1]
               for c in range(1, N_CH)]
    sys.stderr.write(f"  II train/test corr: {np.corrcoef(ii_norm[0,:N_TRAIN], target_ii[:N_TRAIN])[0,1]:.3f}\n")
    sys.stderr.write(f"  Mean cross-ch corr: {np.mean(np.abs(ch_corr)):.3f}\n")
    sys.stderr.write(f"  Ictal std: {ic_norm.std():.3f}  II std: {ii_norm.std():.3f}\n")

    out = open(OUT_PATH, "w")

    out.write(f"""\
//@ run-pass
//@ expect-stdout: ALL PASS

// ═══════════════════════════════════════════════════════════════════════════
// Door D: S-SSM seizure detection — subject chb11 (blank-channel-cleaned)
// ═══════════════════════════════════════════════════════════════════════════
//
// CHB-MIT Scalp EEG 1.0.0  chb11  ({EDF_PATH})
// 28 raw channels; 16 real channels selected (blanks at 4,9,12,17,22 removed).
// ch_map = [0,1,2,3,5,6,7,8,10,11,13,14,15,16,18,19]
// 256 Hz, seizure at {SEIZURE_S}s.
//
// Door C anomaly: scalar (16.5×) > S-SSM-16ch (8.2×) for chb11.
// Root cause: blank '-' channels at positions 4,9,12 within first 16
//   inject structured zeros into sedenion computation, degrading spatial coupling.
// Fix: map only real channels to positions 0-15.
// Hypothesis: after fix, S-SSM-16ch > scalar restores G₂ coupling signal.
//
// Cross-ch corr (train): {np.mean(np.abs(ch_corr)):.3f} mean
// Ictal std ratio: {ic_norm.std():.3f} vs II {ii_norm.std():.3f}

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

// ─── EEG data arrays ─────────────────────────────────────────────────────────
// II = inter-ictal [80 samples: 64 train + 16 holdout]
// IC = ictal onset [64 samples padded to 80]

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

    # Write channel data
    for ch in range(N_CH):
        write_array_inits(out, f"II_CH{ch}", ii_norm[ch])
    for ch in range(N_CH):
        write_array_inits(out, f"IC_CH{ch}", ic_padded[ch])
    write_array_inits(out, "TARGET_II", target_ii)
    write_array_inits(out, "TARGET_IC", target_ic)
    out.write("}\n\n")

    # Model code (identical to Door I sedenion_ssm_seizure_chb01.sio)
    out.write(SOUNIO_MODEL_CODE)
    out.close()
    sys.stderr.write(f"\nWritten: {OUT_PATH}\n")


# The Sounio model code is identical to Door I (chb01.sio) — same architecture,
# just different data. Embedded as a string constant.
SOUNIO_MODEL_CODE = """\
// ─── Sedenion gate ────────────────────────────────────────────────────────────

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

var LAST_H0: f64 = 0.0   var LAST_H1: f64 = 0.0   var LAST_H2:  f64 = 0.0   var LAST_H3:  f64 = 0.0
var LAST_H4: f64 = 0.0   var LAST_H5: f64 = 0.0   var LAST_H6:  f64 = 0.0   var LAST_H7:  f64 = 0.0
var LAST_H8: f64 = 0.0   var LAST_H9: f64 = 0.0   var LAST_H10: f64 = 0.0   var LAST_H11: f64 = 0.0
var LAST_H12: f64 = 0.0  var LAST_H13: f64 = 0.0  var LAST_H14: f64 = 0.0   var LAST_H15: f64 = 0.0

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

// ─── C readout (scaled i64 — avoids BSS f64 zero-init bug) ───────────────────

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

// ─── S-SSM-16ch ──────────────────────────────────────────────────────────────

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

fn ssm_train_mse() -> f64 with Div, Panic {
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

fn ssm16_test_ii(a: Sedenion) -> f64 with Mut, Div, Panic {
    let c: Sedenion = get_c()
    var h: Sedenion = load_last_h()
    var local_sum: f64 = 0.0
    var t: i64 = 64
    while t < 80 {
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
        let y: f64 = sed_dot(h, c)
        let err: f64 = y - TARGET_II[t]
        local_sum = local_sum + err * err
        t = t + 1
    }
    local_sum / 16.0
}

fn ssm16_test_ic(a: Sedenion) -> f64 with Mut, Div, Panic {
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

// ─── S-SSM-scalar (CH0 only) ─────────────────────────────────────────────────

fn ssm_scalar_forward_ii(a: Sedenion) with Mut, Div, Panic {
    var h: Sedenion = sed_one()
    var t: i64 = 0
    while t < 64 {
        let x: f64 = II_CH0[t]
        let ah: Sedenion = sed_mul(a, h)
        let bx: Sedenion = sed_scale(sed_one(), x)
        let raw: Sedenion = sed_add(ah, bx)
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

fn ssm_scalar_test_ii(a: Sedenion) -> f64 with Mut, Div, Panic {
    let c: Sedenion = get_c()
    var h: Sedenion = load_last_h()
    var local_sum: f64 = 0.0
    var t: i64 = 64
    while t < 80 {
        let x: f64 = II_CH0[t]
        let ah: Sedenion = sed_mul(a, h)
        let bx: Sedenion = sed_scale(sed_one(), x)
        let raw: Sedenion = sed_add(ah, bx)
        let act: Sedenion = sed_tanh_s(raw)
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

fn ssm_scalar_test_ic(a: Sedenion) -> f64 with Mut, Div, Panic {
    let c: Sedenion = get_c()
    var h: Sedenion = sed_one()
    var local_sum: f64 = 0.0
    var t: i64 = 0
    while t < 64 {
        let x: f64 = IC_CH0[t]
        let ah: Sedenion = sed_mul(a, h)
        let bx: Sedenion = sed_scale(sed_one(), x)
        let raw: Sedenion = sed_add(ah, bx)
        let act: Sedenion = sed_tanh_s(raw)
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

// ─── Real-16ch diagonal ──────────────────────────────────────────────────────

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
        while k < 16 { local_y = local_y + cg_r(k) * H_REAL[(t*16+k) as usize]; k = k + 1 }
        let err: f64 = local_y - TARGET_II[t]
        k = 0
        while k < 16 { gc[k] = gc[k] + err * H_REAL[(t*16+k) as usize]; k = k + 1 }
        t = t + 1
    }
    let scale: f64 = lr * 2.0 / 64.0
    var k: i64 = 0
    while k < 16 { cs_r(k, cg_r(k) - scale * gc[k]); k = k + 1 }
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
    var h: [f64; 16] = [0.0; 16]
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
fn main() -> i32 with IO, Mut, Div, Panic {
    init_data()

    let a02: Sedenion = make_a(0.2)

    println("═══════════════════════════════════════════════════════════════")
    println("  Door D: S-SSM seizure detection  subject=chb11 (channels cleaned)")
    println("  CHB-MIT chb11_82.edf  seizure=298s  16 real channels (blanks removed)")
    println("  Anomaly test: Door C showed scalar>16ch due to blank channel noise.")
    println("  After fix: expect S-SSM-16ch > S-SSM-scalar.")
    println("═══════════════════════════════════════════════════════════════")
    println("")

    // ── S-SSM-16ch ───────────────────────────────────────────────────────────
    init_c()
    var ep: i64 = 0
    while ep < 200 { ssm16_forward_ii(a02); ssm_update_c(0.2); ep = ep + 1 }
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
    while ep < 200 { ssm_scalar_forward_ii(a02); ssm_update_c(0.2); ep = ep + 1 }
    ssm_scalar_forward_ii(a02)
    let sc_ii: f64 = ssm_scalar_test_ii(a02)
    let sc_ic: f64 = ssm_scalar_test_ic(a02)
    print("  S-SSM-scalar α=0.2  test_ii="); print(sc_ii)
    print("  test_ic="); print(sc_ic); println("")

    // ── Real-16ch ────────────────────────────────────────────────────────────
    init_real()
    ep = 0
    while ep < 200 { real_forward_ii(); real_update_c(0.01); ep = ep + 1 }
    real_forward_ii()
    let r16_ii: f64 = real_test_ii()
    let r16_ic: f64 = real_test_ic()
    print("  Real-16ch (diag)  test_ii="); print(r16_ii)
    print("  test_ic="); print(r16_ic); println("")
    println("")

    // ── Ratios ───────────────────────────────────────────────────────────────
    let s16_ratio: f64 = if s16_ii > 0.000001 { s16_ic / s16_ii } else { 0.0 }
    let sc_ratio:  f64 = if sc_ii  > 0.000001 { sc_ic  / sc_ii  } else { 0.0 }
    let r16_ratio: f64 = if r16_ii > 0.000001 { r16_ic / r16_ii } else { 0.0 }

    print("  Seizure sensitivity (MSE_ic / MSE_ii):"); println("")
    print("    S-SSM-16ch  ratio="); print(s16_ratio); println("")
    print("    S-SSM-scalar ratio="); print(sc_ratio); println("")
    print("    Real-16ch   ratio="); print(r16_ratio); println("")
    println("")

    var passed: i32 = 0
    var failed: i32 = 0

    // T1: MSE values finite
    if s16_train >= 0.0 && s16_ii >= 0.0 && s16_ic >= 0.0 {
        println("T1 PASS: all MSE finite"); passed = passed + 1
    } else {
        println("T1 FAIL: negative MSE"); failed = failed + 1
    }

    // T2: S-SSM-16ch trains
    if s16_train < 1.5 {
        println("T2 PASS: S-SSM-16ch converged"); passed = passed + 1
    } else {
        println("T2 FAIL: S-SSM-16ch did not converge"); failed = failed + 1
    }

    // T3: S-SSM-16ch > S-SSM-scalar (Door C anomaly resolved)
    if s16_ratio > sc_ratio {
        print("T3 PASS: 16ch ("); print(s16_ratio)
        print(") > scalar ("); print(sc_ratio)
        println(") — blank-channel fix restores G2 coupling")
        passed = passed + 1
    } else {
        print("T3 FAIL: scalar ("); print(sc_ratio)
        print(") >= 16ch ("); print(s16_ratio)
        println(") — anomaly persists")
        failed = failed + 1
    }

    // T4: S-SSM-16ch > diagonal (G2 advantage over linear baseline)
    if s16_ratio > r16_ratio {
        print("T4 PASS: 16ch ("); print(s16_ratio)
        print(") > diagonal ("); print(r16_ratio)
        println(") — G2 coupling signal")
        passed = passed + 1
    } else {
        print("T4 FAIL: diagonal ("); print(r16_ratio)
        print(") >= 16ch ("); print(s16_ratio)
        println("")
        failed = failed + 1
    }

    println("")
    if failed == 0 { println("ALL PASS") }
    else { print("FAILURES: "); print(failed); println("") }

    print("  Passed: "); print(passed); println("")
    print("  Failed: "); print(failed); println("")

    0
}
"""


def main():
    if not os.path.exists(EDF_PATH):
        print(f"ERROR: {EDF_PATH} not found", file=sys.stderr)
        sys.exit(1)
    sigs, sr = load_channels(EDF_PATH)
    generate_sio(sigs, sr)


if __name__ == "__main__":
    main()
