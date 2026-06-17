#!/usr/bin/env python3
"""Run acceptance gates for typological expansion (expansion/ only)."""
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx
from GraphRicciCurvature.OllivierRicci import OllivierRicci
from scipy.optimize import linprog

REPO = Path(__file__).resolve().parents[2]
EXP = REPO / "data/processed/expansion"
SOUC = REPO / "bin/souc"
SEED = 88172645463325252


def compile_run(sio_path: str, elf_path: Path) -> tuple[int, str]:
    p = subprocess.run(
        [str(SOUC), sio_path, str(elf_path)],
        cwd=REPO,
        capture_output=True,
        text=True,
        env={**dict(__import__("os").environ), "SOUNIO_STDLIB_PATH": str(REPO / "stdlib")},
    )
    if p.returncode != 0:
        return p.returncode, p.stdout + p.stderr
    elf_path.chmod(0o755)
    r = subprocess.run([str(elf_path)], cwd=REPO, capture_output=True, text=True)
    return r.returncode, r.stdout + r.stderr


def parse_native_orc(out: str) -> dict:
    m = re.search(
        r"LCC_N=(\d+) LCC_M=(\d+) E=(\d+) neg=(\d+).*?"
        r"kappa_mean_x1e9=(-?\d+).*?"
        r"ci95_lo_x1e9=(-?\d+) ci95_hi_x1e9=(-?\d+)",
        out,
        re.S,
    )
    if not m:
        raise ValueError(f"cannot parse native output:\n{out}")
    n, m_lcc, e, neg, km, lo, hi = m.groups()
    return {
        "LCC_N": int(n),
        "LCC_M": int(m_lcc),
        "E": int(e),
        "neg_edges": int(neg),
        "kappa_u": int(km) / 1e9,
        "ci95_lo": int(lo) / 1e9,
        "ci95_hi": int(hi) / 1e9,
    }


def weighted_graphricci(edge_csv: Path, alpha: float = 0.5) -> dict:
    df = pd.read_csv(edge_csv)
    g = nx.DiGraph()
    for _, row in df.iterrows():
        g.add_edge(row["source"], row["target"], weight=row["weight"])
    gu = g.to_undirected()
    if not nx.is_connected(gu):
        lcc = max(nx.connected_components(gu), key=len)
        gu = gu.subgraph(lcc).copy()
    orc = OllivierRicci(gu, alpha=alpha, verbose="ERROR")
    orc.compute_ricci_curvature()
    curv = [orc.G[u][v]["ricciCurvature"] for u, v in orc.G.edges()]
    return {"N": gu.number_of_nodes(), "E": gu.number_of_edges(), "kappa_w": float(np.mean(curv))}


def oracle_unweighted_kappa_mean(edge_csv: Path) -> float:
    df = pd.read_csv(edge_csv)
    nodes = sorted(set(df["source"]) | set(df["target"]))
    idx = {w: i for i, w in enumerate(nodes)}
    adj: dict[int, set[int]] = {i: set() for i in range(len(nodes))}
    for _, row in df.iterrows():
        u, v = idx[row["source"]], idx[row["target"]]
        adj[u].add(v)
        adj[v].add(u)
    g = nx.Graph()
    for u, nbrs in adj.items():
        for v in nbrs:
            if u < v:
                g.add_edge(u, v)
    lcc = max(nx.connected_components(g), key=len)
    lcc = set(lcc)
    lmap = {old: new for new, old in enumerate(sorted(lcc))}
    adj2: dict[int, set[int]] = {i: set() for i in range(len(lcc))}
    for u in lcc:
        for v in adj[u]:
            if v in lcc and lmap[u] < lmap[v]:
                a, b = lmap[u], lmap[v]
                adj2[a].add(b)
                adj2[b].add(a)

    def hop(a: int, b: int) -> int:
        if a == b:
            return 0
        if b in adj2[a]:
            return 1
        for x in adj2[a]:
            if b in adj2[x]:
                return 2
        return 3

    def w1_lp(u: int, v: int) -> float:
        du, dv = len(adj2[u]), len(adj2[v])
        ns, nd = du + 1, dv + 1
        su = [u] + sorted(adj2[u])
        sv = [v] + sorted(adj2[v])
        nvar = ns * nd
        c = np.array([hop(su[i], sv[j]) for i in range(ns) for j in range(nd)], dtype=float)
        aeq, beq = [], []
        for i in range(ns):
            row = np.zeros(nvar)
            row[i * nd : (i + 1) * nd] = 1
            aeq.append(row)
            beq.append(0.5 if i == 0 else 1 / (2 * du))
        for j in range(nd):
            row = np.zeros(nvar)
            row[j::nd] = 1
            aeq.append(row)
            beq.append(0.5 if j == 0 else 1 / (2 * dv))
        res = linprog(c, A_eq=np.array(aeq), b_eq=np.array(beq), bounds=[(0, 1)] * nvar, method="highs")
        if not res.success:
            raise RuntimeError(res.message)
        return res.fun

    kappas = []
    for u in adj2:
        for v in adj2[u]:
            if u < v:
                kappas.append(1 - w1_lp(u, v))
    return float(np.mean(kappas))


def gate_language(lang: str, edge_csv: Path, sio_orc: str, sio_smt: str) -> dict:
    row: dict = {"language": lang, "edge_file": str(edge_csv)}
    if not edge_csv.exists():
        return {**row, "verdict": "FAIL", "reason": "edge CSV missing"}

    elf_orc = Path(f"/tmp/orc_{lang}_expansion.elf")
    rc, out = compile_run(sio_orc, elf_orc)
    row["native_exit"] = rc
    if rc != 0:
        return {**row, "verdict": "FAIL", "reason": f"native ORC exit {rc}", "native_stdout": out}
    native = parse_native_orc(out)
    row.update({f"native_{k}": v for k, v in native.items()})

    oracle = oracle_unweighted_kappa_mean(edge_csv)
    row["oracle_kappa_u"] = oracle
    row["native_oracle_absdiff"] = abs(native["kappa_u"] - oracle)
    row["native_oracle_ok"] = row["native_oracle_absdiff"] <= 1e-9

    w = weighted_graphricci(edge_csv)
    row["weighted_kappa_w"] = w["kappa_w"]

    elf_smt = Path(f"/tmp/orc_{lang}_smt.elf")
    rc_s, out_s = compile_run(sio_smt, elf_smt)
    row["smt_exit"] = rc_s
    row["smt_stdout"] = out_s.strip()
    row["smt_ok"] = rc_s == 0 and "UNSAT_CERTIFIED" in out_s

    ci_ok = native["ci95_hi"] < 0
    passed = row["native_oracle_ok"] and native["kappa_u"] < 0 and w["kappa_w"] < 0 and ci_ok and row["smt_ok"]
    row["verdict"] = "PASS" if passed else "FAIL"
    if not passed:
        reasons = []
        if not row["native_oracle_ok"]:
            reasons.append(f"native!=oracle ({row['native_oracle_absdiff']:.2e})")
        if native["kappa_u"] >= 0:
            reasons.append(f"kappa_u={native['kappa_u']:+.6f}>=0")
        if w["kappa_w"] >= 0:
            reasons.append(f"kappa_w={w['kappa_w']:+.6f}>=0")
        if not ci_ok:
            reasons.append(f"CI95 hi={native['ci95_hi']:+.6f} not <0")
        if not row["smt_ok"]:
            reasons.append(f"SMT cert failed (exit={rc_s}; {out_s.strip()[:120]})")
        row["reason"] = "; ".join(reasons)
    return row


def main() -> int:
    results = {
        "producer": "typological-expansion-audit",
        "branch": "feat/typological-expansion",
        "r1_only": True,
        "bootstrap_B": 2000,
        "bootstrap_seed": SEED,
        "castle_untouched": True,
        "languages": {},
    }

    sl_csv = EXP / "slovenian_edges_FINAL.csv"
    if sl_csv.exists():
        results["languages"]["SL"] = gate_language(
            "SL",
            sl_csv,
            "repro/exact_orc/gen/orc_SL_expansion.sio",
            "repro/exact_orc/expansion/orc_SL_smt_max_edge.sio",
        )
        results["languages"]["SL"]["n_cues_source"] = 1000
        results["languages"]["SL"]["ci_width"] = (
            results["languages"]["SL"]["native_ci95_hi"] - results["languages"]["SL"]["native_ci95_lo"]
        )
    else:
        results["languages"]["SL"] = {"verdict": "FAIL", "reason": "not preprocessed"}

    results["languages"]["DE"] = {
        "verdict": "FAIL",
        "reason": "GATE A STOP: SWOW-DE 2025 R1 strength absent on disk; SWOW portal download requires manual registration (automated POST returned HTTP 405/500)",
    }
    results["languages"]["ZH_REFRESH"] = {
        "verdict": "FAIL",
        "reason": "GATE A STOP: SWOW-ZH23 post-preprocessing R1 file absent; SWOW portal download requires manual registration",
        "frozen_baseline_untouched": {
            "file": "data/processed/chinese_edges_FINAL.csv",
            "native_kappa_u": -0.143997243,
            "weighted_kappa_w": -0.189347,
            "delta_note": "cannot compute delta without ZH23 raw",
        },
    }

    results["typology"] = {
        "family_map": {
            "EN": "Germanic",
            "NL": "Germanic",
            "DE": "Germanic",
            "ES": "Romance",
            "ZH": "Sino-Tibetan",
            "ZH_REFRESH": "Sino-Tibetan",
            "SL": "Slavic",
        },
        "frozen_castle_hyperbolic": ["EN", "NL", "ES", "ZH"],
        "expansion_pass": [k for k, v in results["languages"].items() if v.get("verdict") == "PASS"],
        "verdict": (
            "3/4 frozen families hyperbolic (Germanic/Romance/Sino-Tibetan); "
            "DE/ZH_REFRESH blocked on missing raw; SL pending full SMT gate"
        ),
    }

    out_json = EXP / "AUDIT_TYPOLOGY_EXPANSION.json"
    out_json.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
