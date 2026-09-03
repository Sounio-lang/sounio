<!-- docs:meta
topic_id: repo.docs.audit.madaros-engine-parity-harvest-1-2-3-2026-07-28
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.madaros-engine-parity-harvest-1-2-3-2026-07-28
-->

# Engine parity harvest — batches 1–2–3 (2026-07-28)

**Lane:** Madaros vs lean_single DIVERGE triage  
**Binary:** FO cross-fn rebuild (`madaros-fo-cross`, PR #1543 tip)  
**Gate:** `scripts/ci/madaros_gum_semantic_suite_gate.sh`

## Method

For each baseline `DIVERGE` witness: compile+run under FO-fixed Madaros and
lean_single; compare PASS/FAIL markers and GUM numbers — not raw stdout
(Madaros `println` inserts newlines; lean often concatenates).

## Batch 1 — `gum_*` core FO

| Witness | Madaros | lean | Classification |
|---|---|---|---|
| `gum_cross_function.sio` | PASS sum=5 scaled=16 | PASS same | **FIXED** by FO let/return (#1543) |
| `gum_compliance.sio` | All 7 PASS | All 7 PASS | **Semantic AGREE** (print layout only) |
| `gum_correlated.sio` | var(h+h)=**0.0004** | var(h+h)=**0.0002** | **Madaros correct** (Y=2h ⇒ 4·V); lean independent bug |
| `gum_iso_budget.sio` | PASS | PASS | Semantic AGREE (lean omits intermediate prints differently) |
| `gum_euler_ode.sio` | var≈4 PASS | var≈4 PASS | Semantic AGREE (ulp / extra y_final print) |
| `gum_variance_shadow.sio` | PASS | PASS | Semantic AGREE |
| `gum_reporting.sio` | ALL 10 PASSED | ALL 10 PASSED | Semantic AGREE |
| `madaros_gum_fo_*` suite | PASS | often FAIL | **Madaros-ahead** (FO tests lean cannot pass) |
| `madaros_gum_independent_product.sio` | vprod=0.0325 PASS | vprod=0.0025 still prints PASS? | lean wrong product FO; Madaros GUM-true |

### Correlated add (GUM §5.2)

`measure(1.75, u=0.01)` ⇒ V=10⁻⁴.

| Op | Truth | Madaros | lean |
|---|---|---:|---:|
| h*h | 4 h² V = 0.001225 | 0.001225 | 0.001225 |
| h−h | 0 | 0 | 0 |
| h+h | 4 V = 0.0004 | **0.0004** | **0.0002** (missing 2 cov) |

Do **not** “fix” Madaros to match lean here. Assert 0.0004 in the witness
(see hardened `gum_correlated.sio`).

## Batch 2 — GUM H.1

| Witness | Madaros | lean | Classification |
|---|---|---|---|
| `gum_h1_native.sio` | all [PASS] u_c≈31.140939 | all [PASS] u_c≈31.140940 | **Semantic AGREE** (1 ulp print) |
| `gum_h1_end_gauge.sio` | all [PASS] | all [PASS] | Semantic AGREE |

## Batch 3 — epistemic / dissertation PBPK

| Witness | Madaros | lean | Classification |
|---|---|---|---|
| `epistemic_pbpk_native.sio` | T1–T4 PASS | T1–T4 PASS | **Semantic AGREE** |
| `epistemic_pbpk_multidrug.sio` | T* PASS | T* PASS | Semantic AGREE |
| `darwin_pbpk28_smoke.sio` | PBPK28_CORE_SMOKE_PASS | same | Semantic AGREE |
| `dissertation_pbpk14_model_form_uc.sio` | [PASS] suite | [PASS] suite | Semantic AGREE (residual text layout) |
| `dissertation_pbpk_qss_analytical_ref.sio` | 168 PARITY rows | 168 rows | **Numeric ~ulp / Taylor exp** (see below) |
| `dissertation_pbpk28_parity_ref_rapamycin.sio` | DONE | DONE | Late-time 1e-6 print noise |

### QSS analytical residual

Both engines implement the same closed-form 1-state QSS with Taylor `exp_neg`.
Against Python `math.exp`:

- V_eff ≈ 49.676 (Madaros prints 49.675999)
- C_b(0.1) Madaros 9.81e-4, lean 9.82e-4, Python ≈ 9.817e-4
- Late t (30 h): both underflow vs true ~5.6e-7 under print width 6

This is **not** an FO or import-path bug. Closing byte-stdout parity would need
shared print formatting and/or a better `exp` primitive — out of FO harvest scope.

## Deliverables

1. PR #1543 — FO pure-fn let/return for `gum_cross_function`
2. Hardened `gum_correlated.sio` asserts (rejects lean-style 0.0002)
3. `scripts/ci/madaros_gum_semantic_suite_gate.sh` — 10-witness semantic PASS
4. This audit

## What not to do

- Do not mark baseline AGREE solely on Madaros PASS while lean still fails FO tests —
  baseline compares engines; document Madaros-ahead separately.
- Do not weaken Madaros FO to match lean correlated-add 0.0002.
- Do not treat println newline differences as compiler defects.

## Next harvest candidates (outside 1–2–3)

- `generic_knowledge.sio` — both exit 0; lean prints garbage i64-looking value
- `closure_epistemic.sio` — both print `0` only (under-specified)
- Remaining non-GUM baseline DIVERGEs (~150) after FO/print reclassification
