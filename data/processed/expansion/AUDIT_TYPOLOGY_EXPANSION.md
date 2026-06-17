# AUDIT — Typological Expansion (SPECULATIVE, NO-COMMITMENT)

**Branch:** `feat/typological-expansion`  
**Date:** 2026-06-06  
**Isolation:** All new artefacts under `data/processed/expansion/` only. Frozen EN/ES/ZH/NL castle **not modified**.

## Pipeline spine (R1-only, identical to castle)

Preprocessing copies `complete_all_4_languages_FINAL.preprocess_strength` verbatim:

| Parameter | Value |
|-----------|-------|
| Response set | **R1 / first response only** (never R123) |
| Threshold | `R1.Strength >= 0.06` |
| Vocabulary | Top 500 by frequency |
| Edges | Dedup `groupby.max(R1.Strength)` |

## Per-language verdicts

```
DE:         FAIL — GATE A STOP: SWOW-DE 2025 R1 strength file not on disk (SWOW portal requires manual registration; automated fetch HTTP 405/500)
SL:         FAIL — SMT cert on max-|κ| edge: dual objective 23/30 ≠ W1 53/30 (native κ_u=-0.0943, κ_w=-0.1684, CI=[-0.1199,-0.0693]<0, oracle parity 2e-10; n=1000 cues, CI width=0.051)
ZH_REFRESH: FAIL — GATE A STOP: SWOW-ZH23 post-preprocessing R1 file not on disk; delta old→new not computable
```

### SL detail (only language with complete raw → gates B–E)

| Gate | Result |
|------|--------|
| A Provenance | PASS — CLARIN.SI hdl:11356/1980, normalized stats, F1-only R1 strength |
| B Edges | 432 LCC nodes / 564 edges, ⟨k⟩=2.61, η=0.154 |
| C Native exact ORC (α=½, unweighted hop) | κ_u = **−0.094272784** |
| C Weighted GraphRicci (α=0.5) | κ_w = **−0.168414310** |
| C Oracle parity | \|native − oracle\| = **2.0×10⁻¹⁰** ≤ 1e−9 |
| D SMT Farkas (max-\|κ\| edge) | **FAIL** — `modi_duals` dual obj 23/30 ≠ primal W1 53/30 |
| E Bootstrap B=2000, seed fixed | CI₉₅ = [−0.1199, −0.0693] **strictly < 0** |
| Slurm | Job **2335** (`cpu-ops`, `orc_SL_expansion`) |

### ZH_REFRESH frozen baseline (untouched)

| Metric | Frozen `chinese_edges_FINAL.csv` |
|--------|----------------------------------|
| κ_u (native exact) | −0.143997243 |
| κ_w (GraphRicci) | −0.189347 |

## Typology verdict

**Frozen castle (EN/NL/ES/ZH):** 3 families — Germanic, Romance, Sino-Tibetan — all hyperbolic (κ<0, frozen audit).  

**Expansion:** No candidate language **PASS** all gates. SL shows hyperbolic sign under both ORC definitions and strict CI, but **fails native SMT witness** on the max-|κ| edge (honest stop, no massaging). DE and ZH_REFRESH blocked before curvature gates.

> **Pattern intact among frozen languages; expansion inconclusive pending DE/ZH23 raw and SL SMT witness repair.**

## Castle integrity

```
md5 english_edges_FINAL.csv  369b5b6c1608aa0fce7b3de399eb907e  (unchanged)
md5 spanish_edges_FINAL.csv  84e03c8125a9b10151e53fafc044a62f  (unchanged)
md5 chinese_edges_FINAL.csv  702d9608c7a6bd790352792178d2f364  (unchanged)
```

No changes to deck, preprint, or `orc_definition_matrix.json`.

## Operator action to unblock

1. **DE:** Place SWOW-DE 2025 R55 **R1-only** strength file at `data/processed/expansion/raw/de/`
2. **ZH_REFRESH:** Place SWOW-ZH23 post-preprocessing R1 strength at `data/processed/expansion/raw/zh/`
3. Re-run: `python3 scripts/research/typological_expansion_preprocess.py` then `python3 scripts/research/typological_expansion_audit.py`
