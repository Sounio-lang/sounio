# Stage G PBPK28 + TMDD + PD — Gap-Report Discovery Log

## Step 0 — Discovery

Audit run on worktree `/workspace/sounio-stage-g-gamma`, branch `dissertation/3d-frontend-stage-f`, HEAD `890971ea` (working tree clean, no `-dirty` suffix). The chapter draft files (§§4.1–4.6) referenced in the user's prompt are **not present** in the repo at any branch/commit; the audit therefore treats the 40 claims as a *specification of intent* and classifies each by implementation locus.

Codebase enumeration:

- `stdlib/darwin_pbpk/` contains 49 .sio sources organised under `core/`, `compartments/`, `drugs/`, `bbb/`, `pd/`, `ddi/`, `release/`, `scenarios/`, `validation/`, `fit/`, `io/`, `export/`, `schema/`, `population/`, plus a top-level `tsit5_pbpk14.sio`, `epistemic_pbpk14.sio`, `epistemic_pbpk14_hessian.sio`, `epistemic_sim.sio`, `simulation.sio`, `simulation_real.sio`, `aggregate_confidence.sio`, `absorption.sio`, `brain_plasma_tac.sio`, `constants.sio`, `lib.sio`. Everything is **PBPK14**. No `epistemic_pbpk28.sio`, no `tmdd/` subdirectory, no `pd/coronary_smc_prolif.sio` or `pd/bergman_glucose_insulin.sio`, no `core/tissue_composition.sio`, no `core/peptide_partitioning.sio`, no `release/higuchi.sio` (the only release file is `release/biomaterial_release.sio`).
- `website/src/lib/`: `pbpk14_core.mjs` (PBPK14 engine) and `pbpk28_core.mjs` (full PBPK28+TMDD+PD+multi-drug, 669 lines). Plus localisation/audience/proof TS modules.
- `website/src/components/dissertation/`: 16 .tsx components (DissertationViewer, Compartments, BloodFlowEdges, Stent, SCDepot, TmddPanel, PdReadoutPanel, ConfidenceGate, DrugSelector, GumBudgetBar, HessianHeatmap, OrganDetailModal, TimeScrubber, TourControls, CameraDirector, Silhouette, InfoPopover) + `compartments.ts` + `tours.ts`.
- `website/src/hooks/`: `usePBPK.ts` (PBPK28 driver), `usePBPK14.ts`, `useReducedMotion.ts`.
- `tests/run-pass/`: `dissertation_pbpk28_parity_ref_rapamycin.sio` (570 lines), `dissertation_pbpk28_parity_ref_semaglutide.sio` (475 lines), `dissertation_pbpk28_degenerate_parity_ref.sio`, `dissertation_pbpk_qss_analytical_ref.sio`, `dissertation_pbpk14_gum.sio`, `dissertation_pbpk14_hessian.sio`, `dissertation_frontend_parity_ref.sio`. These are **tests, not stdlib library modules**.
- `scripts/ci/dissertation_*`: `dissertation_pbpk28_parity_gate.sh` (the single mega-gate folding 9 cases — PBPK28+QSS+TMDD+PD+multi-drug+mass-balance into one script), `dissertation_dossier_gate.sh`, `dissertation_frontend_parity_gate.sh`, `dissertation_pbpk_hessian_gate.sh`, `dissertation_pbpk_suite_gate.sh`. **There is no separate `dissertation_tmdd_parity_gate.sh`, `dissertation_pkpd_parity_gate.sh`, or `dissertation_multi_drug_parity_gate.sh`** — those concerns are subsumed into `dissertation_pbpk28_parity_gate.sh`.
- `scripts/dissertation/`: `run_pbpk28_node.mjs`, `run_pbpk14_node.mjs`, `dossier_generator.sio`.
- `benchmarks/pbpk/`: `qss_residual.csv` (169 rows), `model_form_uc.csv` (169 rows), `hessian_budget.csv` (11 rows), `causal_intervention.sio`, `validate.sh`.
- `tests/golden/dissertation/`: `dossier_rapamycin_snapshot.md` (89 lines).

## Step 1 — Gap report

`gap_report.json` written next. The taxonomy assigns 40 ids across the 9 loci defined in the brief. Key locus-distribution observations recorded inline as items; numeric extraction performed against `website/src/lib/pbpk28_core.mjs` and `tests/run-pass/dissertation_pbpk28_parity_ref_{rapamycin,semaglutide}.sio` with both values reported when a claim spans loci.

## Step 2 — Sounio examples

`sounio_examples.json` extracts 8 constructs. Of those, only **5 are populated**: `ODEConfig` (from `stdlib/darwin_pbpk/tsit5_pbpk14.sio`), TMDD organ declaration (parity ref), confidence-gated control flow (BBB gate), parity-gate test invariant (parity ref), scenario file top-level (parity ref `main()`). `Knowledge<T>` declaration, `Validated<T>` introduction, and `pbpk_invariant_report` mass-balance audit invocation are recorded as **null** with notes — the codebase mentions `Knowledge<T>` only in a BBB-gate doc-comment as future work, has no `Validated<T>` usage, and `pbpk_invariant_report` checks non-negativity/finiteness but does not enforce a < 1% closure-error threshold.

## Step 3 — Test outputs

`test_outputs.json` enumerates the parity references, the three CSVs under `benchmarks/pbpk/`, the dossier snapshot, and the run-pass dissertation tests.

## Step 4 — Validation

All four JSON files validated with `python3 -c 'import json; json.load(open(...))'`. `gap_report.json` has exactly 40 items.

## Headlines

1. **The dissertation §4 cannot be honestly written from the current stdlib**. Of 40 claims, the Sounio stdlib (`stdlib/darwin_pbpk/`) implements **zero PBPK28-specific modules**. Everything PBPK28 / TMDD / PD lives either (a) in the JS engine `website/src/lib/pbpk28_core.mjs`, or (b) duplicated as the Sounio parity-ref tests in `tests/run-pass/`. The draft's claimed paths (`tmdd/fkbp12_mtorc1.sio`, `tmdd/glp1r.sio`, `pd/coronary_smc_prolif.sio`, `pd/bergman_glucose_insulin.sio`, `core/tissue_composition.sio`, `core/peptide_partitioning.sio`, `tmdd/qe_approximation.sio`, `epistemic_pbpk28.sio`, `epistemic_pbpk28_hessian.sio`) do not exist.

2. **Numeric inconsistencies between draft and code are severe — but JS and parity-ref are consistent with each other**. Rapamycin TMDD: draft says `K_d=0.2 nM, k_on=0.5, k_off=0.1, R_total={80,200,40} nM, ε={0.45,0.40,0.30}`; both JS and Sounio parity-ref have `K_d=0.10, k_on=0.10, k_off=0.010, R_total={50,25,30}` and **no ε field at all**. The TMDD parameters are uniform across organs in code (single scalar from `tmdd_kdeg_at(i)` etc.), not per-organ vectors as the draft claims. Rapamycin PD: draft `k_a=0.20, k_prolif=1.74e-4, k_apo=1.5e-4` vs code `k_a=1.0, k_prolif=5.0e-4, k_apo=3.0e-3`. Bergman: draft has 5 params (`S_G, S_I, p_2, γ, k_I`); code uses linearized 2-state model (`SG, SI, kI, Gb, Ib, alpha`) with no `p_2` and no `γ` — `k_sec(G)` glucose-dependent claim is false (code uses constant `kBase = kI · Ib`). HbA1c/eAG/Nathan-2008 conversion does **not exist anywhere** in code.

3. **ODE config field names differ**. Draft claims `rel_tol, abs_tol, h_init, h_min, h_max`; code uses `rtol, atol, dt_init, dt_min, dt_max` (struct `ODEConfig14`). Draft tolerances `rel_tol=1e-6 / abs_tol=1e-9` (default) and `1e-9 / 1e-12` (tight) vs code `0.01 / 1e-4` (default) and `1e-6 / 1e-10` (tight) — the "default" in the draft is the code's **tight** config, and the draft's "tight" is **two orders tighter** than anything in the code.

4. **Knowledge<T> + ε + prov pattern is dissertation-spec-only**. Searched all of `stdlib/darwin_pbpk/` and `tests/run-pass/dissertation_*`: zero callers, zero declarations. The only mention is `stdlib/darwin_pbpk/bbb/bbb_gate.sio:8` which explicitly notes `Knowledge<T> confidence propagation` is **future Sounio compiler work**, not in use. §4.1.2 cannot be sourced.

5. **CI-gate naming**. Draft claims four parity gates; code has one mega-gate (`dissertation_pbpk28_parity_gate.sh`) covering 9 cases. Honest framing: rename the dissertation §4.x descriptions to refer to the 9 cases of the single gate, or split the gate. Promotion cost is trivial (~30 LOC of bash shuffling) but the draft narrative needs revision regardless.

6. **The Higuchi singularity regulariser `ε_t = 1e-4 h`** claim conflicts with code (`Math.max(0.1, tHours)` — a 0.1 h clamp, 1000× looser). The K_H value (draft 4.18 µg·h⁻¹⸍², code 0.00417 mg·√h⁻¹) is consistent (same number, different units). Cypher `f_local = 0.3` to coronary_smc and late-lumen-loss `α = 1.00 mm/unit-N` are **not implemented anywhere** — the JS PD model produces a unitless neointimal `N(t)` only, with no spatial localisation step and no clinical-endpoint conversion.
