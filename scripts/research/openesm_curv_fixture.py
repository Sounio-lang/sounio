#!/usr/bin/env python3
"""Per-SUBJECT affect-network fixture for the preregistered curvature->depression test (I/O only).

For one openESM dataset: per subject, build the Pearson correlation network among the NETWORK items
(never the depression-outcome items), emit a Sounio fixture where each "unit" is a subject with an edge
list, a depression outcome (mean of outcome items over beeps), and density (mean|r|). The Sounio side
computes per-subject Forman + exact-OT Ollivier-Ricci curvature and the partial(curvature, depression |
density) across subjects.

Datasets/config frozen in PREREGISTRATION.md. Usage:
  python3 scripts/research/openesm_curv_fixture.py --dataset fisher|kuczynski|geschwind
"""
import argparse, csv, json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DDIR = ROOT / "data/external/openesm"
OUT = ROOT / "examples/hyperbolic_semantic_networks"
MIN_OBS = 30
THR = 0.25

CFG = {
    "fisher": {
        "file": "0033_fisher_ts.tsv",
        "outcome": ["down", "hopeless", "anhedonia"],
        "network": ["energetic", "enthusiastic", "content", "irritable", "restless", "worried",
                    "guilty", "afraid", "angry", "positive", "fatigue", "muscle_tension",
                    "difficulty_concentrating", "rumination", "avoid_activity", "avoid_people"],
    },
    "kuczynski": {
        "file": "0039_kuczynski_ts.tsv",
        "outcome": ["depressed", "anhedonia"],
        "network": ["loneliness", "left_out", "social_interaction", "vulnerability",
                    "perceived_responsiveness", "covid_anxiety"],
    },
    "geschwind": {
        "file": "0010_geschwind_ts.tsv",
        "outcome": ["sad"],
        "network": ["cheerful", "relaxed", "worried", "fearful"],
    },
}


def fnum(s):
    s = (s or "").strip()
    if s in ("", "NA", "NaN", "."): return np.nan
    try: return float(s)
    except Exception: return np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(CFG))
    a = ap.parse_args()
    cfg = CFG[a.dataset]
    rows = list(csv.DictReader(open(DDIR / cfg["file"]), delimiter="\t"))
    net, out_items = cfg["network"], cfg["outcome"]
    K = len(net)

    # group rows by subject id
    by = {}
    for r in rows:
        by.setdefault(r["id"], []).append(r)

    subj_edges_i, subj_edges_j, subj_edges_w = [], [], []
    win_start, win_nedge, outcome, density, nobs = [], [], [], [], []
    kept, excluded = 0, 0
    for sid, rs in by.items():
        net_mat = np.array([[fnum(r.get(c)) for c in net] for r in rs])  # (T, K)
        out_mat = np.array([[fnum(r.get(c)) for c in out_items] for r in rs])  # (T, n_out)
        valid = np.sum(np.all(np.isfinite(net_mat), axis=1))
        if valid < MIN_OBS:
            excluded += 1; continue
        # mean-impute columns
        cm = np.nanmean(net_mat, axis=0)
        net2 = np.where(np.isnan(net_mat), cm, net_mat)
        # drop zero-variance columns guard: corrcoef handles, but set those edges absent
        C = np.corrcoef(net2.T)
        iu = np.triu_indices(K, 1)
        dens = float(np.nanmean(np.abs(C[iu])))
        win_start.append(len(subj_edges_i)); ne = 0
        for i in range(K):
            for j in range(i + 1, K):
                r = C[i, j]
                if np.isfinite(r) and abs(r) >= THR:
                    subj_edges_i.append(i); subj_edges_j.append(j); subj_edges_w.append(abs(float(r))); ne += 1
        win_nedge.append(ne)
        outcome.append(float(np.nanmean(out_mat)))
        density.append(dens); nobs.append(int(valid)); kept += 1
    print(f"[{a.dataset}] kept {kept} subjects, excluded {excluded} (<{MIN_OBS} obs). K={K} network items.")
    print(f"  edges total={len(subj_edges_i)} | outcome range {min(outcome):.2f}-{max(outcome):.2f} | "
          f"density range {min(density):.3f}-{max(density):.3f}")

    def fmt(v):
        if not np.isfinite(v): return "0.0"
        t = f"{v:.8g}"; return t if ("." in t or "e" in t) else t + ".0"
    def i64a(name, vals, cap):
        lit = ", ".join(str(int(v)) for v in vals) if vals else "0"
        s = f"fn {name}(out: &![i64; {cap}]) with Mut {{\n"
        if vals: s += f"    let src: [i64; {len(vals)}] = [{lit}]\n    var i: i64 = 0\n    while i < {len(vals)} {{ out[i]=src[i]; i=i+1 }}\n"
        return s + "}\n"
    def f64a(name, vals, cap):
        lit = ", ".join(fmt(v) for v in vals) if vals else "0.0"
        s = f"fn {name}(out: &![f64; {cap}]) with Mut {{\n"
        if vals: s += f"    let src: [f64; {len(vals)}] = [{lit}]\n    var i: i64 = 0\n    while i < {len(vals)} {{ out[i]=src[i]; i=i+1 }}\n"
        return s + "}\n"

    MAXU, MAXE = 1024, 65536
    fpath = OUT / f"openesm_{a.dataset}_data.sio"
    with open(fpath, "w") as f:
        f.write(f"//! GENERATED openESM per-subject curvature fixture — {a.dataset}. DO NOT EDIT.\n")
        f.write(f"//! n_units(subjects)={kept} K(network nodes)={K} total_edges={len(subj_edges_i)}\n\n")
        f.write(f"fn fx_n_nodes() -> i64 {{ {K} }}\n")
        f.write(f"fn fx_n_units() -> i64 {{ {kept} }}\n\n")
        f.write(i64a("load_u_start", win_start, MAXU))
        f.write(i64a("load_u_nedge", win_nedge, MAXU))
        f.write(i64a("load_ei", subj_edges_i, MAXE))
        f.write(i64a("load_ej", subj_edges_j, MAXE))
        f.write(f64a("load_ew", subj_edges_w, MAXE))
        f.write(f64a("load_depression", outcome, MAXU))
        f.write(f64a("load_density", density, MAXU))
    (OUT / f"openesm_{a.dataset}_manifest.json").write_text(json.dumps(
        {"dataset": a.dataset, "n_units": kept, "excluded": excluded, "K": K, "min_obs": MIN_OBS,
         "thr": THR, "network_items": net, "outcome_items": out_items,
         "total_edges": len(subj_edges_i)}, indent=2))
    print(f"[emit] {fpath.name}")


if __name__ == "__main__":
    main()
