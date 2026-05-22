<!-- docs:meta
topic_id: repo.docs.dissertation.audit.discovery-log
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.dissertation.audit.discovery-log
-->

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

## Compiler items (added 2026-05-11, Stage G-ε-11)

7. **`Knowledge<T>` constructor expression rejected by `./bin/souc` at HEAD `ae2123ad`.** The AST plumbing is present (`self-hosted/test_knowledge.sio:T04` parses `Knowledge<f64>` shorthand into a `TypeKnowledge` node with `KnowledgeTypeInfo`); the *type annotation* `let x: Knowledge<f64>` compiles fine. But the canonical run-pass test `tests/run-pass/med/vancomycin_full_propagation.sio`, which uses the `Knowledge(value, ε=…, prov=…)` constructor expression, fails the typechecker:
   ```
   E200 `ε` at line 70
   E200 `prov` at line 70
   error: unknown identifier at line 70
   typecheck: failed
   ```
   Same error on a minimal repro (`let f = Foo { cl_hep: Knowledge(12.4, ε=0.85, prov="ferron_1997_cpt") }`). The Stage G stdlib promotion path therefore *cannot* introduce Knowledge<T>-wrapped parameters without a parallel compiler fix; the Stage G-ε-11+ ports use plain `f64` fields (matching the existing parity refs). Re-evaluate when the compiler self-hosting work lands the named-args parser for the Knowledge constructor.

## Reconciliation — 2026-05-21 (HEAD `8c6631a2a`, origin/main)

The Step 0–4 record above is an accurate point-in-time snapshot taken on 2026-05-11 at HEAD `890971ea` and is retained verbatim as history. This section reconciles it to the current tree; `gap_report.json` has been re-audited to match.

**Headline #1 is RETIRED.** The 2026-05-11 verdict — *"the dissertation §4 cannot be honestly written from the current stdlib; the stdlib implements zero PBPK28-specific modules"* — is no longer true. The 2026-05-17 commit chain (`652133d7d` "PBPK28 epistemic stack: first-order GUM, Hessian, Sobol/PCE" → `d1bd7bf30` → `88def17ae` → `8d020fe93` → `6b922d385` → `1105dcccf` → merges `55863178f`/`cb51778fa`) authored the modules whose absence the headline rested on:

- `stdlib/darwin_pbpk/epistemic_pbpk28.sio` (664 loc) and `epistemic_pbpk28_hessian.sio` (591 loc) — closes contributions (1) and (2) of `six_contributions_modules`.
- `stdlib/darwin_pbpk/tmdd/fkbp12_mtorc1.sio` and `tmdd/glp1r.sio` — closes the module-authoring sub-gap of `tmdd_rapamycin_path` / `tmdd_semaglutide_path`.
- `stdlib/darwin_pbpk/pd/coronary_smc_prolif.sio` (102 loc) and `pd/bergman_glucose_insulin.sio` (113 loc) — closes `pd_rapamycin_path` / `pd_semaglutide_path`.

Six items flip locus from `PARITY_REF_SOUNIO`/`NOT_AUTHORED` to `STDLIB_SOUNIO`. Summary distribution moves from NONE=5/MINOR=7/MAJOR=28 to NONE=9/MINOR=8/MAJOR=23. The newly-authored modules are mutually consistent with the JS engine and the parity refs on every numeric value re-checked; the remaining numeric divergences are draft-vs-implementation, not locus disagreements.

**Headline #4 (Knowledge<T>) STILL STANDS, re-verified.** `./bin/souc check` at this HEAD still rejects the named-arg `Knowledge(value, ε=…, prov=…)` constructor — confirmed on a minimal repro and on `tests/run-pass/med/vancomycin_full_propagation.sio` (both fail `E200 ε` / `E200 prov` / `E200 unknown identifier` / `typecheck: failed`). The PBPK28 modules above therefore use plain `f64`; every ε-bearing dissertation claim remains blocked on a compiler fix.

**Gaps re-verified as genuinely OPEN at this HEAD** (still absent / unimplemented; not promoted by the May-17 work):

- `stdlib/darwin_pbpk/core/tissue_composition.sio` — absent (values still inline in `rodgers_rowland.sio`).
- `stdlib/darwin_pbpk/core/peptide_partitioning.sio` — absent.
- `stdlib/darwin_pbpk/tmdd/qe_approximation.sio` — absent; the QE step is still not referenced anywhere.
- `rapamycin_pd_alpha` (N → mm LLL conversion), `cypher_f_local`, `coronary_smc` 5% carve-out, `eag_window`, `nathan_conversion` (HbA1c), G-dependent `k_sec`, the 1% closure-error auditor, and the operator-split parity case — all unimplemented. (`coronary_smc_prolif.sio` self-flags the α gap in its own header comment.)

**One-dissertation framing.** Per operator directive (2026-05-21), the PBPK28 biomaterials track and the broader Sounio epistemic track are a single dissertation. `RECONCILIATION_MEMO_2026-05-12.md`'s two-lane "binding authority" split is superseded (a banner has been added there; its body is retained for history). The audit and the claim truth table now present one unified evidence surface. The distinction between PBPK14 / PBPK28 / Hessian / K-AXI *gates* is retained as evidence-scoping discipline, not as a thesis-boundary.

**Gates re-run 2026-05-21 — all six PASS.** The verification sweep was executed in the worktree at `8c6631a2a` (`SKIP_BUILD=1`, `SOUNIO_STDLIB_PATH=$(pwd)/stdlib`):

- `dissertation_pbpk28_parity_gate.sh` — exit 0, 9/9 cases PASS (PBPK28 parity, mass conservation, TMDD {1,4,8}, PD heart, semaglutide PBPK28/GLP-1R-TMDD/glucose-insulin-PD), all within 1.0% RMSE.
- `dissertation_pbpk_suite_gate.sh` — exit 0, PASS (50/50 rapamycin PBPK tests + smoke demos), ~99 s.
- `dissertation_pbpk_hessian_gate.sh` — exit 0, PASS 5 / FAIL 0.
- `dissertation_frontend_parity_gate.sh` — exit 0, 14/14 compartments within 1.0% RMSE.
- `dissertation_dossier_gate.sh` — exit 0, PASS 5 / FAIL 0.
- `kretikos_kaxi_phase_y_gate.sh` — exit 0, 3/3 truth claims; ran on real CUDA hardware (`/usr/local/bin/ptxas`, `nvidia-smi`/A5000 present), GPU↔CPU f32 digests bit-exact at a 10M cohort (`c1=38a827a54cb778aa`, `v11=dbf63987c228e0e6`), ISO budget emitted.

The `repo-backed` rows in `pbpk_claim_truth_table.md` are confirmed green at this HEAD, not merely carried forward. This does not change the standing gaps above (the gates exercise what is implemented; they do not test the unimplemented tissue_composition / peptide_partitioning / qe_approximation modules or the blocked Knowledge<T> ε/prov constructor).
