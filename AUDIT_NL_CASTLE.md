# AUDIT — NL CASTLE: closing exact Dutch ORC parity on the *real* canonical graph

**Date:** 2026-06-03 · **Producer:** sounio-native (`souc` `6183636b…`) + GraphRicciCurvature 0.5.3.2 (weighted reference) + networkx/Fraction oracle
**Repos:** `github.com/agourakis82/hyperbolic-semantic-networks` (data + pipeline) · `Sounio-lang/sounio` (native exact-OT machinery)

## TL;DR

The prior NL **strict-FAIL was a missing-file artifact, not a native error.** The earlier session never had `dutch_edges_FINAL.csv`, so it fell back to (a) the raw *dense* `dutch_edges.csv` → κ = **+0.0987** (positive), then (b) an **ad-hoc `count≥15` reconstruction** (469/825) → −0.197220, which the task explicitly forbids.

Building the Dutch FINAL graph with the **canonical R1 pipeline** (the same function that built EN/ES/ZH) yields LCC **N=465, E=835 — exactly the audit-reference graph.** Native exact ORC on it gives

> **native `souc` κ̄(NL) = −0.196029293** (hyperbolic; exact rational −1120721311/5717111400 = −0.196029293919, which **rounds to −0.196029294** — the value the Fraction oracle and POT also report). All three solvers agree to **≤1e-9**, and the native value is within **1.03e-5** of the audit gist `e3a3072` reference (−0.196019).

**NL parity is closed with zero reconstruction.** All six verdict lines PASS.

---

## STEP A — RAW R1 fetch & provenance → **PASS**

- `data/raw/strength.SWOW-NL.R1.csv` (md5 `dcdc3cf1…`) has the real `cue,response,R1.Strength` columns, **byte-structurally identical** to the EN/ES/ZH strength files.
- **Genuine R1 proven, not reconstructed:** I recomputed `R1.Strength` directly from `data/raw/SWOW-NL13/associationData.csv` (md5 `c50c0131…`), column `asso1` (first response):
  - **484,587 / 485,481 = 99.816 %** of rows reproduce to `|Δ| < 1e-6` matching cue/response **verbatim**.
  - **485,481 / 485,481 = 100.0000 %** reproduce when cue+response are **lowercased** on both sides (zero residual) — the residual is purely the SWOW **case-folding** convention, *not* missing data: the strength file lowercases, the raw stores original casing (e.g. `Amerikaans` with exactly 100 presentations; verified present, capitalized, zero lowercase occurrences). _(An earlier draft mis-attributed this to "21 absent cues / a more complete export"; that mechanism is false and was corrected in `nl_provenance.json`, HSN `5c37660`.)_
  - Per-cue Σ R1.Strength = **1.0000 exactly for all 12,571 cues** (denominator = 100 responses/cue, the SWOW-NL13 R100 design → exact 2-decimal strengths).
  - Independently re-verified by an adversarial agent (POT-free recompute): 100 % case-folded reproduction, per-cue sum = 1.0, denominator = 100 empirically confirmed.
- Source: SWOW-NL13 R100, smallworldofwords.org, **CC BY-NC-SA 4.0**. md5s + license recorded in `data/raw/nl_provenance.json`.

> Strength CSVs are gitignored repo-wide (`data/raw/*.csv`), same as EN/ES/ZH; provenance md5 is the durable record.

## STEP B — canonical `dutch_edges_FINAL.csv` → **PASS (committed, params match EN/ES/ZH)**

- Built by `code/analysis/complete_all_4_languages_FINAL.py` `preprocess_strength` — **the same function and defaults** (`R1.Strength ≥ 0.06`, top-500 words, drop self-loops, groupby-max) that produced `english/spanish/chinese_edges_FINAL.csv`. The Dutch call differs only in path + label, exactly as the EN/ES/ZH calls do.
- My rebuild is **byte-identical** to the committed file (md5 `f8592ef3…`).
- **Committed:** branch `nl-castle/dutch-final-canonical` @ **`618a08e99c570f62c030300c70e2d8506cba082f`** (HSN), via `git add -f` (mirrors how EN/ES/ZH FINAL are tracked past the `*.csv` ignore).
- Graph: full **N=475 / E=840**; **LCC N=465 / E=835**, **⟨k⟩ = 3.59**, η = 0.00774 — single-digit degree (sparse), **not** the dense ⟨k⟩≈61 of `dutch_edges.csv`. LCC (465/835) **== the audit-reference thresholded Dutch, exactly.**

## STEP C — native exact ORC + parity → **PASS (κ<0, |diff|<1e-4)**

Definition: α=½ uniform lazy walk, **unweighted hop** ground distance, κ = 1 − W₁ exact over ℚ (min-cost-flow), LCC.

| quantity | value |
|---|---|
| **native `souc` κ̄** (authoritative, committed-file result) | **−0.196029293** (printed `κ×1e9` = −196029293) |
| exact rational κ̄ | **−1120721311/5717111400** = −0.196029293919… → **−0.196029294** (9 dp) |
| same-definition oracle (networkx + Fraction) | −0.196029294 |
| **independent POT backend** (`ot.emd2`, network-simplex) | −0.196029294 |
| **\|native − oracle\|** (the <1e-4 grader) | ≤ 1e-9 ✓ |
| audit-gist `e3a3072` NL ref (cross-ref) | −0.196019 → \|Δ\| = **1.03e-5** ✓ |
| 95 % CI (native bootstrap B=1000) | [−0.215720505, −0.174366592] |
| neg edges (exact) | **595 / 835** |
| Slurm | **job 2333**, `cpu-ops`, node `cpuops-t560-proxmox`; on-node CSV md5 `f8592ef3…`; **bitwise-identical to local** |

**Three independent implementations** — native `souc` (min-cost-flow over ℚ), a networkx+`Fraction` oracle, and the POT `ot.emd2` LP — agree on κ̄ to ≤1e-9 and on LCC 465/835. The per-edge κ are **exact rationals** in the first two, so their agreement is exact at edge level; the only spread is f64 mean rounding. **neg = 595 is the exact count** (POT's float strict-`<0` reports 599, but 4 of those are true-zero κ edges that pick up −2e-16 LP float noise; the exact methods classify them correctly as 0). The audit-gist (−0.196019) is within 1.03e-5 because the real FINAL graph *is* the audit-reference graph — the prior 0.0012 reconstruction gap vanishes. Artifact: `swow_parity_exact_native_nl.json`.

> Note on the headline: the native binary prints −0.196029293 because `(mean·1e9) as i64` truncates toward zero; the mathematically correct value is −0.196029293919 ≈ −0.196029294. κ<0 and \|diff\|<1e-4 hold under every rendering.

> Slurm note: `/workspace` is not NFS-mounted on the compute node; the binary+CSV were staged to node-local `/tmp` (md5-verified on-node) and stdout streamed back via `srun`. `sbatch --output` to the shared path is unreadable from the node.

## STEP D — definition reconciliation → **DEFINITION_LOCK**

**The preprint / DMH deck use the WEIGHTED, directed GraphRicciCurvature** (α=0.5, 100 Sinkhorn iters, largest WCC) — submission `…/main.md` line 76; `GraphRicciCurvature 0.5.3` pinned in methods; `statistical_tests_v6.4.json` is that weighted family. **The native Sounio machinery uses UNWEIGHTED-uniform-hop** (`swow_unified_orc.sio` line 5) — a *different estimator*. Both computed for all four:

| lang | unweighted-uniform (native, exact) | weighted GRC on FINAL (α=0.5) | published v6.4 weighted |
|---|---|---|---|
| EN | −0.137147006 | −0.257661 | −0.197368 |
| ES | −0.068341242 | −0.154863 | −0.104155 |
| ZH | −0.143997243 | −0.213979 | −0.189347 |
| NL | **−0.196029293** | −0.270002 | −0.172194 |

**All negative under both definitions** → the hyperbolic conclusion is definition-robust; only magnitudes move. **Mixing flags:** (1) native −0.137 family ≠ preprint −0.197 family — different estimators, never equate; (2) v6.4 weighted numbers were computed on an *older graph instance* (n_values 811/776/816/799) — re-running the same weighted def on the current FINAL files gives the −0.258 family, so v6.4 is stale; (3) the NL audit-gist ref (−0.196019) is the **unweighted native** value, not the manuscript weighted NL. **Recommendation:** lock to ONE definition + ONE snapshot. Artifact: `orc_definition_matrix.json`.

## STEP E — Layers 5 & 6 native on the four FINAL graphs

**Layer 5 — SMT_NATIVE_COUNTS (this native run; inherited values DISCARDED):**

| lang | #UNSAT (κ<0) / E |
|---|---|
| EN | 407 / 640 |
| ES | 322 / 571 |
| ZH | 495 / 762 |
| **NL** | **595 / 835** |

The prior NL `159/15368` (dense graph) is **discarded**. Each #UNSAT == Layer-3 native #neg-κ (by construction) and is reproduced by the independent Fraction oracle. The **EN anchor (68,261)** is UNSAT with an **exact materialized dual witness** (dual obj = 27/20 = W₁ > 1, native exit 0). Per-edge dual-witness materialization (`modi_duals_multiroot`) covers the non-degenerate majority (EN 247/407, ES 193/322, ZH 293/495, NL 361/595); the remainder are transportation-degenerate (cross-component dual feasibility, not offset-solved) — **but every UNSAT edge is still confirmed κ<0 by the exact primal W₁>1 and the oracle.** The complete, sound certificate is the exact primal; the dual is the succinct independently-checkable layer.

**Layer 6 — BOOTSTRAP_NL → CI<0 on FINAL (4/4):**

| lang | 95 % CI |
|---|---|
| EN | [−0.157408520, −0.112442193] |
| ES | [−0.093167909, −0.040838243] |
| ZH | [−0.164775627, −0.122611585] |
| **NL** | **[−0.215720505, −0.174366592]** |

All four strictly negative; **NL on the real `dutch_edges_FINAL.csv`** (no reconstruction). (Native CI = edge-resample B=1000; manuscript CI = 80%-node × 50 on weighted GRC — different scheme, both <0.)

---

## Verdicts

```
NL_RAW_FETCH:      PASS
NL_FINAL_BUILT:    PASS (committed 618a08e9, params byte-match EN/ES/ZH)
NL_PARITY:         PASS (native souc kappa=-0.196029293 < 0; exact rational -1120721311/5717111400
                   = -0.196029294; 3 solvers agree (souc/networkx/POT) <=1e-9 < 1e-4; |native-audit_gist|=1.03e-5)
DEFINITION_LOCK:   preprint/deck = WEIGHTED GraphRicciCurvature (alpha=0.5); native = unweighted-uniform.
                   BOTH computed for all 4 (orc_definition_matrix.json); all negative under both.
SMT_NATIVE_COUNTS: EN 407/640  ES 322/571  ZH 495/762  NL 595/835  (native this run; inherited=DISCARDED)
BOOTSTRAP_NL:      PASS (CI=[-0.2157,-0.1744] < 0 on committed FINAL; 4/4 CI<0)
```

No reconstructed or oracle value stands in for a committed-file native result: every κ above is `souc` on the committed `dutch_edges_FINAL.csv` (md5 `f8592ef3…`), cross-checked by an independent exact oracle.
