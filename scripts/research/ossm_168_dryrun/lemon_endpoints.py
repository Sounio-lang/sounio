"""Build a per-subject endpoint manifest for the LEMON confirmatory run.

Merges the META file (Hamilton, BSL-23, SKID) with the emotion/personality
battery to produce primary and secondary endpoints aligned with the v3
preregistration hypotheses.

Endpoint choices (given LEMON instrument availability):
  H1 rumination  -> CERQ_Rumination (primary), NEOFFI_Neuroticism (secondary)
  H2 anhedonia   -> BAS_Drive reversed (primary), BAS_Reward reversed (secondary)
  H3 valence/dep -> Hamilton_Scale (primary), BSL23_sumscore (secondary)

BSL-23 coverage note:
  The only canonical source for BSL23_sumscore in LEMON is the META file
  (verified against the GWDG mirror: no BSL file exists in
  Behavioural_Data_MPILMBB_LEMON/Emotion_and_Personality_Test_Battery_LEMON
  or any sibling directory). LEMON administered BSL-23 selectively:
  169/227 META rows (74.4%), or 165/220 when intersected with
  subjects_all.txt. The H3s_F3_vs_BSL23 secondary is therefore
  preregistered as "planned missing" and runs on the available subset.

Run:
    python3 lemon_endpoints.py \
        --meta  /workspace/data/lemon/meta/META_File_IDs_Age_Gender_Education_Drug_Smoke_SKID_LEMON.csv \
        --phen  /workspace/data/lemon/phenotype \
        --out   /workspace/data/lemon/endpoints.csv
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict


def _num(s: str) -> float | None:
    s = (s or "").strip()
    if not s or s.lower() in {"na", "nan", "none"}:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _load_csv(path: Path, id_col: str = "ID") -> Dict[str, Dict[str, str]]:
    out: Dict[str, Dict[str, str]] = {}
    with path.open(encoding="utf-8-sig") as fh:
        rdr = csv.DictReader(fh)
        for row in rdr:
            sid = (row.get(id_col) or "").strip()
            if sid:
                out[sid] = row
    return out


def build_endpoints(meta_path: Path, phen_dir: Path) -> list[dict]:
    meta = _load_csv(meta_path, id_col="ID")
    cerq = _load_csv(phen_dir / "CERQ.csv")
    neo = _load_csv(phen_dir / "NEO_FFI.csv")
    bisbas = _load_csv(phen_dir / "BISBAS.csv")
    stai = _load_csv(phen_dir / "STAI_G_X2.csv")
    tas = _load_csv(phen_dir / "TAS.csv")

    subjects = sorted(meta.keys())
    rows: list[dict] = []
    for sid in subjects:
        m = meta[sid]
        rec = {
            "subject_id": sid,
            "age_bin": (m.get("Age") or "").strip(),
            "sex": (m.get("Gender_ 1=female_2=male") or "").strip(),
            "skid_raw": (m.get("SKID_Diagnoses") or "").strip(),
            "hamilton": _num(m.get("Hamilton_Scale", "")),
            "bsl23": _num(m.get("BSL23_sumscore", "")),
            "cerq_rumination": _num(
                cerq.get(sid, {}).get("CERQ_Rumination", "")),
            "cerq_catastrophizing": _num(
                cerq.get(sid, {}).get("CERQ_Catastrophizing", "")),
            "neo_neuroticism": _num(
                neo.get(sid, {}).get("NEOFFI_Neuroticism", "")),
            "bas_drive": _num(bisbas.get(sid, {}).get("BAS_Drive", "")),
            "bas_fun": _num(bisbas.get(sid, {}).get("BAS_Fun", "")),
            "bas_reward": _num(bisbas.get(sid, {}).get("BAS_Reward", "")),
            "bis": _num(bisbas.get(sid, {}).get("BIS", "")),
            "stai_trait": _num(
                stai.get(sid, {}).get("STAI_Trait_Anxiety", "")),
            "tas_total": _num(tas.get(sid, {}).get("TAS_OverallScore", "")),
        }
        rec["has_current_mdd"] = int(
            "current major depressive" in rec["skid_raw"].lower()
        )
        rec["has_past_mdd"] = int(
            "past major depressive" in rec["skid_raw"].lower()
        )
        rows.append(rec)
    return rows


def write_csv(rows: list[dict], out: Path) -> None:
    if not rows:
        raise SystemExit("no rows to write")
    fields = list(rows[0].keys())
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--meta", type=Path, required=True)
    ap.add_argument("--phen", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    rows = build_endpoints(args.meta, args.phen)
    write_csv(rows, args.out)
    n_ham = sum(1 for r in rows if r["hamilton"] is not None)
    n_cerq = sum(1 for r in rows if r["cerq_rumination"] is not None)
    n_bas = sum(1 for r in rows if r["bas_drive"] is not None)
    n_neo = sum(1 for r in rows if r["neo_neuroticism"] is not None)
    print(f"wrote {len(rows)} subjects -> {args.out}")
    print(f"  Hamilton      : {n_ham} / {len(rows)}")
    print(f"  CERQ_Rumination: {n_cerq} / {len(rows)}")
    print(f"  BAS_Drive     : {n_bas} / {len(rows)}")
    print(f"  NEO_Neuroticism: {n_neo} / {len(rows)}")


if __name__ == "__main__":
    main()
