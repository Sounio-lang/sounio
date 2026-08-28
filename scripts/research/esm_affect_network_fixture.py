#!/usr/bin/env python3
"""Per-window affect-item NETWORK fixture for temporal discrete-curvature EWS (I/O plumbing only).

Reads the real Kossakowski (2017) n=1 ESM data (1476 beeps, 12 mood items), slides a window over
the series, and within each window builds the between-item correlation network (nodes = mood items,
edges = |Pearson r| >= threshold, weight = |r|). Emits a self-contained Sounio fixture with per-window
edge lists, so Sounio can compute Forman-Ricci + Ollivier-Ricci curvature of the network over time and
test whether curvature drops at the depressive transition (geometric early-warning signal).

This is the "full network ORC over all mood items" named as future work in esm_real_data_orc.sio and
affect_network_orc.sio (which only did 4 independent per-node temporal self-transitions).

Classical CSD baselines (mean within-window variance, mean lag-1 autocorrelation) are emitted too, for
comparison — these are summary statistics, not the novel geometry.

Usage: python3 scripts/research/esm_affect_network_fixture.py [--window 50 --step 25 --thr 0.25]
"""
import argparse, csv, io, json, zipfile
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ESM_ZIP = ROOT / "data/external/ESMdata.zip"
OUT_SIO = ROOT / "examples/hyperbolic_semantic_networks"
MOOD = ["mood_relaxed", "mood_down", "mood_irritat", "mood_satisfi", "mood_lonely", "mood_anxious",
        "mood_enthus", "mood_suspic", "mood_cheerf", "mood_guilty", "mood_doubt", "mood_strong"]
MAXWIN = 256
MAXEDGE = 32768


def load_esm():
    z = zipfile.ZipFile(ESM_ZIP)
    name = [n for n in z.namelist() if n.lower().endswith("esmdata.csv") and "__MACOSX" not in n][0]
    rows = list(csv.DictReader(io.TextIOWrapper(z.open(name), "latin-1")))
    M = np.full((len(rows), len(MOOD)), np.nan)
    phase = np.zeros(len(rows), dtype=np.int64)
    for i, r in enumerate(rows):
        try: phase[i] = int(r["phase"])
        except Exception: phase[i] = 0
        for j, c in enumerate(MOOD):
            v = r[c].strip()
            if v not in ("", "NA"):
                try: M[i, j] = float(v)
                except Exception: pass
    return M, phase


def emit_temporal(M, phase, a):
    """Per-window per-node active-fraction marginals for the temporal-transition full-network ORC
    (direct heir of the 4-node affect_network_orc.sio: per-node distribution shift window t -> t+1)."""
    T, K = M.shape
    starts = list(range(0, T - a.window + 1, a.step))
    nW = len(starts)
    # binarization threshold per item: centered (min<0) -> >0, else 1-7 scale -> >4
    thr = []
    for j in range(K):
        col = M[:, j]
        col = col[np.isfinite(col)]
        thr.append(0.0 if col.min() < 0 else 4.0)
    pact, pvar, win_phase, var_mean, ac1_mean = [], [], [], [], []
    for st in starts:
        seg = M[st:st + a.window]
        ph = np.bincount(phase[st:st + a.window]).argmax(); win_phase.append(int(ph))
        col_mean = np.nanmean(seg, axis=0)
        seg2 = np.where(np.isnan(seg), col_mean, seg)
        var_mean.append(float(np.nanmean(np.nanvar(seg, axis=0))))
        ac = []
        for j in range(K):
            x = seg2[:, j]
            if x.std() > 1e-9: ac.append(float(np.corrcoef(x[:-1], x[1:])[0, 1]))
        ac1_mean.append(float(np.mean(ac)) if ac else 0.0)
        for j in range(K):
            x = seg[:, j]; x = x[np.isfinite(x)]
            n = max(len(x), 1)
            p = float((x > thr[j]).sum()) / n
            p = min(max(p, 0.001), 0.999)  # clamp off 0/1 for the log-domain Sinkhorn
            pact.append(p); pvar.append(p * (1.0 - p) / n)
    print(f"[temporal] {nW} windows x {K} nodes; thresholds {thr}")

    def fmt(v):
        if not np.isfinite(v): return "0.0"
        t = f"{v:.8g}"; return t if ("." in t or "e" in t) else t + ".0"
    def f64arr(name, vals, cap):
        lit = ", ".join(fmt(v) for v in vals) if vals else "0.0"
        s = f"fn {name}(out: &![f64; {cap}]) with Mut {{\n"
        if vals: s += f"    let src: [f64; {len(vals)}] = [{lit}]\n    var i: i64 = 0\n    while i < {len(vals)} {{ out[i]=src[i]; i=i+1 }}\n"
        return s + "}\n"
    def i64arr(name, vals, cap):
        lit = ", ".join(str(int(v)) for v in vals) if vals else "0"
        s = f"fn {name}(out: &![i64; {cap}]) with Mut {{\n"
        if vals: s += f"    let src: [i64; {len(vals)}] = [{lit}]\n    var i: i64 = 0\n    while i < {len(vals)} {{ out[i]=src[i]; i=i+1 }}\n"
        return s + "}\n"

    out = OUT_SIO / "affect_temporal_orc_data.sio"
    with open(out, "w") as f:
        f.write(f"//! GENERATED (temporal mode) — Kossakowski ESM. W={a.window} step={a.step} nW={nW} K={K}. DO NOT EDIT.\n\n")
        f.write(f"fn fx_n_nodes() -> i64 {{ {K} }}\nfn fx_n_windows() -> i64 {{ {nW} }}\n\n")
        f.write(f64arr("load_pact", pact, 4096))
        f.write(f64arr("load_pvar", pvar, 4096))
        f.write(i64arr("load_win_phase", win_phase, 256))
        f.write(f64arr("load_var_mean", var_mean, 256))
        f.write(f64arr("load_ac1_mean", ac1_mean, 256))
    (OUT_SIO / "affect_temporal_orc_manifest.json").write_text(json.dumps(
        {"mode": "temporal", "window": a.window, "step": a.step, "n_windows": nW, "n_nodes": K,
         "thresholds": thr, "mood_items": MOOD, "phase_per_window": win_phase,
         "binarize": "centered(min<0)->p(x>0); else 1-7 scale->p(x>4); clamped[0.001,0.999]"}, indent=2))
    print(f"[emit] {out.name} + manifest")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=int, default=50)
    ap.add_argument("--step", type=int, default=25)
    ap.add_argument("--thr", type=float, default=0.25)
    ap.add_argument("--mode", choices=["structural", "temporal"], default="structural")
    a = ap.parse_args()

    M, phase = load_esm()
    T, K = M.shape
    print(f"[esm] {T} beeps, {K} mood items")

    if a.mode == "temporal":
        return emit_temporal(M, phase, a)

    starts = list(range(0, T - a.window + 1, a.step))
    nW = len(starts)
    print(f"[windows] W={a.window} step={a.step} -> {nW} windows")

    # per-window network + CSD baselines
    win_phase, var_mean, ac1_mean, mean_absr = [], [], [], []
    all_ei, all_ej, all_ew, win_start, win_nedge = [], [], [], [], []
    for st in starts:
        seg = M[st:st + a.window]  # (W, K)
        # phase = mode of the window
        ph = np.bincount(phase[st:st + a.window]).argmax()
        win_phase.append(int(ph))
        # impute column means for correlation (low missingness)
        col_mean = np.nanmean(seg, axis=0)
        seg2 = np.where(np.isnan(seg), col_mean, seg)
        # classical CSD baselines
        var_mean.append(float(np.nanmean(np.nanvar(seg, axis=0))))
        ac1 = []
        for j in range(K):
            x = seg2[:, j]
            if x.std() > 1e-9 and len(x) > 2:
                ac1.append(float(np.corrcoef(x[:-1], x[1:])[0, 1]))
        ac1_mean.append(float(np.mean(ac1)) if ac1 else 0.0)
        # correlation network
        C = np.corrcoef(seg2.T)  # (K,K)
        # threshold-free density covariate: mean |r| over off-diagonal
        iu = np.triu_indices(K, 1)
        mean_absr.append(float(np.nanmean(np.abs(C[iu]))))
        win_start.append(len(all_ei))
        ne = 0
        for i in range(K):
            for j in range(i + 1, K):
                r = C[i, j]
                if np.isfinite(r) and abs(r) >= a.thr:
                    all_ei.append(i); all_ej.append(j); all_ew.append(abs(float(r))); ne += 1
        win_nedge.append(ne)
    total_edges = len(all_ei)
    print(f"[net] total edges across windows={total_edges} (avg {total_edges/nW:.1f}/window), thr={a.thr}")

    # ---- emit Sounio fixture ----
    def fmt(v):
        if not np.isfinite(v):
            return "0.0"
        t = f"{v:.8g}"
        return t if ("." in t or "e" in t) else t + ".0"
    def i64arr(name, vals, cap):
        lit = ", ".join(str(int(v)) for v in vals) if vals else "0"
        s = f"fn {name}(out: &![i64; {cap}]) with Mut {{\n"
        if vals:
            s += f"    let src: [i64; {len(vals)}] = [{lit}]\n    var i: i64 = 0\n    while i < {len(vals)} {{ out[i]=src[i]; i=i+1 }}\n"
        return s + "}\n"
    def f64arr(name, vals, cap):
        lit = ", ".join(fmt(v) for v in vals) if vals else "0.0"
        s = f"fn {name}(out: &![f64; {cap}]) with Mut {{\n"
        if vals:
            s += f"    let src: [f64; {len(vals)}] = [{lit}]\n    var i: i64 = 0\n    while i < {len(vals)} {{ out[i]=src[i]; i=i+1 }}\n"
        return s + "}\n"

    out = OUT_SIO / "affect_net_curv_data.sio"
    with open(out, "w") as f:
        f.write(f"//! GENERATED by scripts/research/esm_affect_network_fixture.py — Kossakowski ESM. DO NOT EDIT.\n")
        f.write(f"//! W={a.window} step={a.step} thr={a.thr} | n_windows={nW} n_nodes={K} total_edges={total_edges}\n\n")
        f.write(f"fn fx_n_nodes() -> i64 {{ {K} }}\n")
        f.write(f"fn fx_n_windows() -> i64 {{ {nW} }}\n\n")
        f.write(i64arr("load_win_start", win_start, MAXWIN))
        f.write(i64arr("load_win_nedge", win_nedge, MAXWIN))
        f.write(i64arr("load_win_phase", win_phase, MAXWIN))
        f.write(i64arr("load_ei", all_ei, MAXEDGE))
        f.write(i64arr("load_ej", all_ej, MAXEDGE))
        f.write(f64arr("load_ew", all_ew, MAXEDGE))
        f.write(f64arr("load_var_mean", var_mean, MAXWIN))
        f.write(f64arr("load_ac1_mean", ac1_mean, MAXWIN))
        f.write(f64arr("load_mean_absr", mean_absr, MAXWIN))

    man = {"window": a.window, "step": a.step, "thr": a.thr, "n_windows": nW, "n_nodes": K,
           "total_edges": total_edges, "mood_items": MOOD,
           "phase_per_window": win_phase,
           "network": "within-window Pearson |r|>=thr, weight=|r|, mean-imputed",
           "csd_baselines": "mean within-window variance, mean lag-1 autocorrelation"}
    (OUT_SIO / "affect_net_curv_manifest.json").write_text(json.dumps(man, indent=2))
    print(f"[emit] {out.name} ({out.stat().st_size//1024} KB) + manifest")
    print(f"[phase/window] {win_phase}")


if __name__ == "__main__":
    main()
