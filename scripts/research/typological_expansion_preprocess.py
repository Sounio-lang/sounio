#!/usr/bin/env python3
"""Typological expansion preprocessing — identical params to complete_all_4_languages_FINAL.py.

R1-only, R1.Strength >= 0.06, top-500 vocabulary, LCC stats reported.
Outputs ONLY under data/processed/expansion/ (never touches frozen castle files).
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import logging

import networkx as nx
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


# Copied verbatim from complete_all_4_languages_FINAL.py (param parity; no import
# to avoid module-level side effects that touch castle files).
def preprocess_strength(file_path, language, sep="\t"):
    """Process SWOW strength file."""
    logger.info(f"[{language}] Loading: {file_path}")

    try:
        df = pd.read_csv(file_path, sep=sep, quoting=1, on_bad_lines="skip")
        if "cue,response" in str(df.columns):
            df = pd.read_csv(file_path, sep=",", on_bad_lines="skip")
    except Exception:
        df = pd.read_csv(file_path, sep=",", on_bad_lines="skip")

    logger.info(f"[{language}] Loaded {len(df)} rows")

    if "R1.Strength" not in df.columns:
        logger.error(f"[{language}] Missing R1.Strength!")
        return pd.DataFrame()

    df = df[df["R1.Strength"] >= 0.06].copy()
    logger.info(f"[{language}] {len(df)} after R1.Strength >= 0.06")

    words = Counter()
    words.update(df["cue"].astype(str).str.lower())
    words.update(df["response"].astype(str).str.lower())

    top_500 = set([w for w, _ in words.most_common(500)])
    logger.info(f"[{language}] Top 500 from {len(words)} unique words")

    df["cue_clean"] = df["cue"].astype(str).str.lower()
    df["resp_clean"] = df["response"].astype(str).str.lower()

    df_filt = df[
        (df["cue_clean"].isin(top_500))
        & (df["resp_clean"].isin(top_500))
        & (df["cue_clean"] != df["resp_clean"])
    ].copy()

    edges = df_filt.groupby(["cue_clean", "resp_clean"])["R1.Strength"].max().reset_index()
    edges.columns = ["source", "target", "weight"]

    logger.info(f"[{language}] {len(edges)} unique edges")

    g = nx.DiGraph()
    for _, row in edges.iterrows():
        g.add_edge(row["source"], row["target"], weight=row["weight"])

    g_undir = g.to_undirected()
    logger.info(
        f"✅ [{language}] {g_undir.number_of_nodes()} nodes, "
        f"{g_undir.number_of_edges()} edges\n"
    )

    return edges

REPO = Path(__file__).resolve().parents[2]
EXP = REPO / "data/processed/expansion"
RAW = EXP / "raw"


def md5_file(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


DE_UNKNOWN_MARKERS = {
    "unbekanntes wort",
    "unbekannest wort",
    "unknown word",
}
DE_NO_FURTHER_MARKERS = {
    "keine eingaben mehr",
    "keine antowrt",
    "keine anwort",
    "?",
    "-",
}


def build_de_r1_strength(r55_path: Path, out_path: Path) -> dict:
    """R1-only strength from SWOW-DE 2025 R55 trial export (response_corrected_1 only).

    Mirrors SWOW strength-table semantics used by complete_all_4_languages_FINAL.py:
    R1.Strength(cue, response) = count(cue, R1=response) / sum_response count(cue, R1).
    Excludes unknown-word and no-further-response markers per Aeschbach et al. preprocessing.
    """
    df = pd.read_csv(r55_path, low_memory=False)
    r1_col = "response_corrected_1"
    if r1_col not in df.columns:
        raise ValueError(f"DE R55 missing {r1_col!r}; columns: {list(df.columns)[:20]}")
    if "cue" not in df.columns:
        raise ValueError("DE R55 missing cue column")

    work = df[["cue", r1_col]].copy()
    if "unknown_word" in df.columns:
        work = work[~df["unknown_word"].fillna(False).astype(bool)]
    if "no_further_response_1" in df.columns:
        work = work[~df["no_further_response_1"].fillna(False).astype(bool)]

    work["cue_l"] = work["cue"].astype(str).str.strip().str.lower()
    work["resp_l"] = work[r1_col].astype(str).str.strip().str.lower()
    work = work[work["resp_l"].notna() & (work["resp_l"] != "") & (work["resp_l"] != "nan")]
    work = work[~work["resp_l"].isin(DE_UNKNOWN_MARKERS | DE_NO_FURTHER_MARKERS)]
    work = work[work["cue_l"] != work["resp_l"]]

    counts = work.groupby(["cue_l", "resp_l"]).size().reset_index(name="n")
    cue_totals = counts.groupby("cue_l")["n"].sum()
    counts["R1.Strength"] = counts.apply(
        lambda r: r["n"] / cue_totals[r["cue_l"]], axis=1
    )
    out = counts[["cue_l", "resp_l", "R1.Strength"]].rename(
        columns={"cue_l": "cue", "resp_l": "response"}
    )
    out.to_csv(out_path, index=False)
    return {
        "rows_r1": int(len(out)),
        "unique_cues": int(out["cue"].nunique()),
        "trials_in": int(len(df)),
        "trials_valid_r1": int(len(work)),
        "md5": md5_file(out_path),
        "source_csv": str(r55_path),
        "source_csv_md5": md5_file(r55_path),
    }


def build_sl_r1_strength(stats_path: Path, out_path: Path) -> dict:
    """R1-only strength from SWOW-SL normalized statistics (F1 column only)."""
    df = pd.read_csv(stats_path, sep="\t")
    required = {"cue", "response", "F1"}
    if not required.issubset(df.columns):
        raise ValueError(f"SL stats missing columns: {required - set(df.columns)}")
    # R1-only: keep associations that appeared as FIRST response
    df = df[df["F1"] > 0].copy()
    # Exclude SWOW non-response markers
    bad = {"<unknownword>", "<nomorereplies>"}
    df["cue_l"] = df["cue"].astype(str).str.lower()
    df["resp_l"] = df["response"].astype(str).str.lower()
    df = df[~df["resp_l"].isin(bad)].copy()
    cue_totals = df.groupby("cue_l")["F1"].sum()
    df["R1.Strength"] = df.apply(
        lambda r: r["F1"] / cue_totals[r["cue_l"]], axis=1
    )
    out = df[["cue_l", "resp_l", "R1.Strength"]].rename(
        columns={"cue_l": "cue", "resp_l": "response"}
    )
    out.to_csv(out_path, index=False)
    return {
        "rows_r1": int(len(out)),
        "unique_cues": int(out["cue"].nunique()),
        "md5": md5_file(out_path),
    }


def graph_stats(edges: pd.DataFrame) -> dict:
    g = nx.DiGraph()
    for _, row in edges.iterrows():
        g.add_edge(row["source"], row["target"], weight=row["weight"])
    gu = g.to_undirected()
    if not nx.is_connected(gu):
        lcc = max(nx.connected_components(gu), key=len)
        gu = gu.subgraph(lcc).copy()
    n = gu.number_of_nodes()
    e = gu.number_of_edges()
    avg_k = (2 * e / n) if n else 0.0
    eta = nx.global_efficiency(gu) if n > 1 else 0.0
    return {"N": n, "E": e, "avg_k": float(avg_k), "eta": float(eta)}


def run_language(
    lang_code: str,
    strength_path: Path,
    provenance: dict,
    sep: str = ",",
) -> dict:
    """Run identical preprocess_strength call site; write expansion FINAL edges."""
    edges = preprocess_strength(str(strength_path), lang_code, sep=sep)
    if edges.empty:
        return {"status": "FAIL", "reason": "preprocess_strength returned empty"}
    out_csv = EXP / f"{lang_code.lower()}_edges_FINAL.csv"
    edges.to_csv(out_csv, index=False)
    stats = graph_stats(edges)
    prov_path = EXP / f"{lang_code.lower()}_provenance.json"
    provenance.update(
        {
            "artifact": out_csv.name,
            "preprocess_call": {
                "module": "complete_all_4_languages_FINAL.preprocess_strength",
                "params": {
                    "threshold": "R1.Strength >= 0.06",
                    "top_n": 500,
                    "sep": sep,
                    "r1_only": True,
                },
            },
            "graph_stats": stats,
            "output_md5": md5_file(out_csv),
            "generated_utc": datetime.now(timezone.utc).isoformat(),
        }
    )
    prov_path.write_text(json.dumps(provenance, indent=2) + "\n")
    return {"status": "PREPROCESSED", "out_csv": str(out_csv), **stats}


def main() -> int:
    EXP.mkdir(parents=True, exist_ok=True)
    results: dict = {}

    # --- SL (CLARIN on disk) ---
    sl_stats = RAW / "sl" / "SWOW-SL1.0_statistics_normalized.tsv"
    if sl_stats.exists():
        sl_r1 = EXP / "raw" / "sl" / "strength.SWOW-SL.R1.tsv"
        sl_r1.parent.mkdir(parents=True, exist_ok=True)
        sl_meta = build_sl_r1_strength(sl_stats, sl_r1)
        sl_prov = {
            "language": "SL",
            "family": "Slavic",
            "source": {
                "project": "SWOW-SL 1.0",
                "repository": "CLARIN.SI",
                "hdl": "http://hdl.handle.net/11356/1980",
                "url": "https://www.clarin.si/repository/xmlui/handle/11356/1980",
                "license": "CC BY-NC-ND 4.0",
                "citation": "Brglez, Mojca; Vintar, Špela; De Deyne, Simon (2024). Word association norms for Slovenian SWOW-SL 1.0.",
                "release_date": "2024-11-05",
                "n_cues": 1000,
            },
            "r1_derivation": {
                "column": "F1 only (first response frequency)",
                "normalized_responses": True,
                "formula": "R1.Strength = F1(cue,response) / sum_response F1(cue,response)",
                "excludes": ["<unknownWord>", "<noMoreReplies>"],
                "intermediate_strength_file": str(sl_r1),
                **sl_meta,
            },
        }
        results["SL"] = run_language("Slovenian", sl_r1, sl_prov, sep=",")
    else:
        results["SL"] = {"status": "FAIL", "reason": f"missing raw: {sl_stats}"}

    # --- DE (SWOW-DE 2025 R55 trial CSV or pre-built R1 strength table) ---
    de_strength = list(RAW.glob("de/strength*R1*")) + list(RAW.glob("de/*R55*R1*"))
    de_r55 = list(RAW.glob("de/SWOW_DE_2025_R55.csv")) + list(
        RAW.glob("de/**/SWOW_DE_2025_R55.csv")
    )
    if de_strength:
        de_path = de_strength[0]
        de_r1_meta: dict = {"intermediate_strength_file": str(de_path)}
        de_prov = {
            "language": "DE",
            "family": "Germanic",
            "source": {
                "project": "SWOW-DE 2025",
                "doi": "10.48550/arXiv.2604.19620",
                "code_repo": "https://github.com/samuelae/SWOW-DE-2025-Code",
                "url": "https://smallworldofwords.org/en/project/research",
                "license": "CC BY-NC-ND 3.0",
                "release_date": "2025-04-22",
                "response_set": "R55 first response only (NOT R123)",
            },
            "raw_md5": md5_file(de_path),
            "r1_derivation": {"prebuilt_strength_table": True, **de_r1_meta},
        }
        results["DE"] = run_language("German", de_path, de_prov, sep=",")
    elif de_r55:
        r55_path = de_r55[0]
        de_r1 = EXP / "raw" / "de" / "strength.SWOW-DE.R1.from_R55.csv"
        de_r1.parent.mkdir(parents=True, exist_ok=True)
        de_r1_meta = build_de_r1_strength(r55_path, de_r1)
        de_prov = {
            "language": "DE",
            "family": "Germanic",
            "source": {
                "project": "SWOW-DE 2025",
                "doi": "10.48550/arXiv.2604.19620",
                "code_repo": "https://github.com/samuelae/SWOW-DE-2025-Code",
                "url": "https://smallworldofwords.org/en/project/research",
                "license": "CC BY-NC-ND 3.0",
                "release_date": "2025-04-22",
                "response_set": "R55 first response only (NOT R123)",
                "trial_export": "SWOW_DE_2025_R55.csv (55 responses/cue, spell-corrected)",
            },
            "r1_derivation": {
                "column": "response_corrected_1 only",
                "formula": "R1.Strength = count(cue,R1) / sum_R1 count(cue,R1)",
                "excludes": sorted(DE_UNKNOWN_MARKERS | DE_NO_FURTHER_MARKERS),
                "filters": ["unknown_word", "no_further_response_1"],
                "note": (
                    "infer_network.R in samuelae/SWOW-DE-2025-Code pools R1+R2+R3; "
                    "this pipeline uses R1-only for castle parity"
                ),
                **de_r1_meta,
            },
        }
        results["DE"] = run_language("German", de_r1, de_prov, sep=",")
    else:
        results["DE"] = {
            "status": "FAIL",
            "reason": (
                "SWOW-DE 2025 data not on disk. Place SWOW_DE_2025_R55.csv under "
                "data/processed/expansion/raw/de/ (from SWOW portal; code-only repo "
                "https://github.com/samuelae/SWOW-DE-2025-Code has no CSV)"
            ),
        }

    # --- ZH refresh (SWOW-ZH23 post-preprocessing) ---
    zh_candidates = list(RAW.glob("zh/*R1*")) + list(RAW.glob("zh/strength*ZH*"))
    if zh_candidates:
        zh_path = zh_candidates[0]
        zh_prov = {
            "language": "ZH_REFRESH",
            "family": "Sino-Tibetan",
            "source": {
                "project": "SWOW-ZH23",
                "doi": "10.3758/s13428-024-02513-1",
                "url": "https://smallworldofwords.org/en/project/research",
                "license": "CC BY-NC-ND 3.0",
                "release_date": "2026-03-18",
                "response_set": "R1 only (post-preprocessing dataset)",
            },
            "raw_md5": md5_file(zh_path),
            "frozen_baseline": {
                "file": "data/processed/chinese_edges_FINAL.csv (UNTOUCHED)",
                "native_kappa_u": -0.143997243,
                "weighted_kappa": -0.189347,
            },
        }
        results["ZH_REFRESH"] = run_language("Chinese", zh_path, zh_prov, sep=",")
    else:
        results["ZH_REFRESH"] = {
            "status": "FAIL",
            "reason": "SWOW-ZH23 R1 strength file not on disk (place under data/processed/expansion/raw/zh/)",
        }

    print(json.dumps(results, indent=2))
    out = EXP / "preprocess_results.json"
    out.write_text(json.dumps(results, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
