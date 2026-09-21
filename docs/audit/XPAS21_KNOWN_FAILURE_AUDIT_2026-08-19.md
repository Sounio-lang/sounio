<!-- docs:meta
topic_id: repo.docs.audit.xpas21-known-failure-audit-2026-08-19
authority: repo_only
audience: users
last_validated: 2026-08-19
validated_by: grok-cli3
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.xpas21-known-failure-audit-2026-08-19
-->

# 21 unexpected known-failure passes — which ones are real

Date: 2026-08-19
Lane: `xpas21-audit-20260819`
Dispatch: `/tmp/dispatch_xpas21_claude1.md`
Worktree: `/workspace/.wt/xpas21-audit` at `origin/main` `cdea9d7eef` (includes `#1939` / `7be969ed05`)
Companion: [`XPAS21_KNOWN_FAILURE_AUDIT_2026-08-19.tsv`](XPAS21_KNOWN_FAILURE_AUDIT_2026-08-19.tsv)

No `//@ known-failure` tag was edited. No sidecar row was removed. This is measurement.

---

## Semantic-Lane declaration

```text
Semantic-Lane-ID: xpas21-audit-20260819
Owner: grok-cli3
Concept-IDs: SOUNIO-EPISTEMIC-NUMERIC-VALUE, SOUNIO-ORDERED-PATH-PROVENANCE
Intent-Preserved: uncertainty crosses a function boundary without being erased; a CI green is not a live variance; compile success != runtime parity
Transformation: none — classification of 21 XPAS plus a re-run of the FO arity matrix
Types-Changed: none
Effects-Changed: none
IR-Changed: none
Claims-Introduced: CI Full Test Suite on PR #1960 ran lean_single stage2, not Madaros; on Madaros rebuilt from cdea9d7eef, ADD3=0.000000 and ADD4=0.000000; gum_fo_across_call prints CALL_var 0.000000
Claims-Forbidden: "FO is fixed"; "the arity hole closed"; "gum_fo_across_call XPASS means first-order variance survives a 3-arg call on Madaros"; "supports N-arg FO transfer"; "the 21 defects are gone"; "drop the gum_fo_across_call tag"
Assumptions: Madaros is the claim-oracle; lean_single is the bootstrap seed and has no semantic authority; #1939 claimed imported 1-2 arg only and did not lift the >2-param skip
Write-Set: docs/audit/XPAS21_KNOWN_FAILURE_AUDIT_2026-08-19.md; docs/audit/XPAS21_KNOWN_FAILURE_AUDIT_2026-08-19.tsv
Read-Set: tests/run-pass/gum_fo_across_call.sio; tests/run-pass/gum_fo_arity3_boundary.sio; tests/run-pass/gum_fo_import_boundary.sio; tests/run-pass/fo_call_boundary_arity3.sio; tests/run-pass/fo_call_boundary_neg.sio; docs/audit/repro/fo_var_{samefile,import,callee}.sio; self-hosted/ir/lower.sio fo_register_pure_fn_transfer; .github/workflows/ci.yml full-test-suite; tests/known_failures/hardened_diagnostics_full_suite.txt
Positive-Witness: none claimed for FO
Negative-Witness: ADD3=0 and ADD4=0 on rebuilt Madaros; gum_fo_across_call CALL_var=0; E001 still refuses
Acceptance-Gate: every one of the 21 is named CORRIGIDO / TESTEMUNHA-MORTA / INDETERMINADO; the FO matrix is a measured table against lean_single and rebuilt Madaros; a cell that must still fail did fail
Integration-Target: none
Authoritative-Only-If: a later Madaros rebuilt from source prints ADD3=14.0 and ADD4=14.25 on fo_var_samefile.sio
```

---

## What CI actually ran

PR [#1960](https://github.com/Sounio-lang/sounio/pull/1960) Full Test Suite
(`https://github.com/Sounio-lang/sounio/actions/runs/32246471872/job/96048482036`)
printed `Unexpected passes (stale known-failure): 21`.

That job is **not Madaros**. `.github/workflows/ci.yml` `full-test-suite` sets
`SOUNIO_TEST_SOUC_BIN: /tmp/souc-stage2`, the artifact of
`native-selfhost-linux-x86_64` / `scripts/ci/selfhost_host_gate.sh`. That is the
lean_single stage2 seed. Harness known-failure comes from two places:

1. in-file `//@ known-failure`
2. `tests/known_failures/hardened_diagnostics_full_suite.txt` (loaded only when
   `--format junit` and no filter — exactly the CI invocation)

17 of the 21 are sidecar rows with **no** in-file tag. Four have in-file tags.
`gum_fo_arity3_boundary.sio` still has an in-file tag, `requires: madaros`, and
was **not** among the 21 (skipped on lean).

The EffectKind enum on #1960 does not explain these 21. They XPASS on a
lean_single suite.

---

## Compilers used (mandatory)

| role | what | identity |
|---|---|---|
| Madaros (FO + 21) | **rebuilt from source** on Slurm | job **10338**, `gpuorangefs-r770-proxmox`, 32 CPUs, build rc=0 in 226s, ELF **100553240** bytes, source `cdea9d7eef` |
| lean_single (FO + 21) | committed seed via wrapper | `/workspace/.wt/xpas21-audit/bin/souc` + `SOUNIO_SOUC_ENGINE=lean_single`; seed `bin/souc-lean-single-x86_64` 2555805 B |
| shipped ELF | **not used** | `bin/madaros-linux-x86_64` is not today's tree |

`#1939` is an ancestor of `cdea9d7eef`. `self-hosted/ir/lower.sio:8698` still
reads `// >2 params: skip (unsupported transfer)` and returns without
registering a transfer.

---

## FO arity matrix (today's main)

Expected: ADD3 = 14.0, ADD4 = 14.25. Locals are variance of `measure` (u=2,1,3,0.5) → 4, 1, 9, 0.25.

| cell | expected | lean_single | Madaros rebuilt `cdea9d7eef` |
|---|---:|---:|---:|
| LOCAL_a | 4.0 | 4.000000 | 4.000000 |
| LOCAL_b | 1.0 | 1.000000 | 1.000000 |
| LOCAL_c | 9.0 | 9.000000 | 9.000000 |
| LOCAL_d | 0.25 | 0.250000 | 0.250000 |
| ID1 | 4.0 | 4.000000 | 4.000000 |
| RET_ONLY | 4.0 | 4.000000 | 4.000000 |
| ADD2 | 5.0 | 5.000000 | 5.000000 |
| **ADD3** | **14.0** | **14.000000** | **0.000000** |
| **ADD4** | **14.25** | **14.250000** | **0.000000** |
| IMP_ID1 | 4.0 | 4.000000 | 4.000000 |
| IMP_ADD2 | 5.0 | 5.000000 | 5.000000 |
| IMP_ADD3 | 14.0 | 14.000000 | **0.000000** |

Witness files:

| witness | lean | Madaros rebuilt |
|---|---|---|
| `gum_fo_across_call.sio` (3-arg `rhs`, same file) | CALL_var **0.000013** `GUM_FO_ACROSS_CALL_OK` rc=0 | CALL_var **0.000000** `GUM_FO_ACROSS_CALL_ZERO` rc=1 |
| `gum_fo_arity3_boundary.sio` | ADD3=14.000000 rc=0 | ADD3=**0.000000** rc=1 |
| `gum_fo_import_boundary.sio` (2-arg imported) | PEEL=4 IMP=5 rc=1 (peel window) | PEEL=5 IMP=5 `GUM_FO_IMPORT_BOUNDARY_OK` rc=0 |
| `fo_call_boundary_arity3.sio` | id3_var=1.000000 rc=0 | id3_var=**0.000000** rc=1 |
| `fo_call_boundary_neg.sio` | neg_var=1.000000 rc=0 | neg_var=**0.000000** rc=1 |

`#1939` did what it claimed: imported 1–2 arg helpers keep variance
(`IMP_ID1`, `IMP_ADD2`, `gum_fo_import_boundary`). It did **not** close the
arity hole. `gum_fo_across_call` is a 3-arg same-file helper plus `OpSub`.
On today's Madaros it is still exactly zero. The CI XPASS of that file is
lean_single doing what lean_single has always done.

The witness is not dead. Its assertion (`v > 1e-12`) still fires. It is
just not the instrument CI ran.

---

## Mandatory still-fail controls (instrument is not lying)

Written before the Madaros run:

1. **ADD3 must stay 0** if `fo_register_pure_fn_transfer` still skips `>2` params.
   Measured: ADD3=0.000000, ADD4=0.000000, IMP_ADD3=0.000000.
2. **E001 must still refuse** `let x: i64 = true`.
   Measured: Madaros compile rc=1 (incompatible types); lean compile rc=1
   (`Type mismatch — expected i64, got bool`).
3. **`gum_fo_arity3_boundary` must still fail on Madaros.**
   Measured: rc=1, `GUM_FO_ARITY3_BOUNDARY_ZERO`.

If ADD3 had printed 14.0, the hole would be closed and that would be said
loudly. It printed 0. The instrument is live.

A fourth control was *not* used as the honesty pin: Madaros **accepts**
`tests/compile-fail/refinement_f64_return_violation.sio` (compile rc=0,
run rc=0). Lean now rejects it. That is a real Madaros hole, recorded
below. It is not evidence the runner is greenwashing.

---

## The 21, one by one

Class is about **why the CI lean suite printed XPAS**. Madaros results sit
beside that class so nobody drops a tag that is still true on the oracle.

| # | file | tag | class | why |
|---|---|---|---|---|
| 1 | `parser_card_a_misc_patterns.sio` | sidecar | **CORRIGIDO** | Sidecar = native-artifact backlog 2026-06-12. Assertion is "it runs" (return 0). Runs on lean and on rebuilt Madaros. |
| 2 | `parser_card_a_refinement_predicates.sio` | sidecar | **CORRIGIDO** | Same backlog. Compound refinement predicates parse and evaluate on both engines. |
| 3 | `pbpk28_struct_return.sio` | sidecar | **INDETERMINADO** | Lean runs and prints `cl=12.400000` + `PASS`. Rebuilt Madaros compile fails (preflight). CI XPASS is lean-only; Madaros still cannot produce the ELF. |
| 4 | `rapamycin_iso_budget.sio` | sidecar | **TESTEMUNHA-MORTA** | `print("PASS\n")` is unconditional (line 227) after a Knowledge/Budget64 cross-check that **printed out of [0.9,1.1]** on lean. Harness only greps `PASS` and `inf`. Lean executes; Madaros thin-link fails (`ir_into_acc_failed`). The ISO-budget claim is not gated. |
| 5 | `rapamycin_rk4_budget.sio` | sidecar | **INDETERMINADO** | Stronger than #4: `FAMILY_A_VAR_LIVE` is gated on live `var_blood_k`. Lean prints it + `PASS`. Madaros thin-link fails. CI XPASS is lean execution, not a Madaros FO close. |
| 6 | `darwin_compartments_coronary_smc_smoke.sio` | sidecar | **CORRIGIDO** | Mass-balance smoke. `CORONARY_SMC_SMOKE_PASS` on lean and Madaros with the three `f_local` regimes. |
| 7 | `darwin_pd_coronary_smc_smoke.sio` | sidecar | **CORRIGIDO** | PD smoke. `PD_CORONARY_SMC_SMOKE_PASS` on both; N shrinks as `target_a` falls. |
| 8 | `dissertation_pbpk28_confidence_gate.sio` | sidecar | **CORRIGIDO** | `PBPK28_CONFIDENCE_GATE_PASS` on both. Method demonstration, not a dosing engine. |
| 9 | `refinement_f64_return_violation.sio` | sidecar + compile-fail | **CORRIGIDO** (lean only) | Lean now emits `refinement type violation` (the pattern the test names). Madaros **accepts** `return 1.7` and runs it. Dropping the sidecar from a lean XPASS would hide a live Madaros hole. |
| 10 | `turbofish_concrete_type_mismatch.sio` | in-file, lean-only soundness | **CORRIGIDO** | Tag said lean accepts `identity::<bool>(42)`. Today's lean rejects E001. Madaros still rejects. The lean gap the tag named is closed on this seed. |
| 11 | `test_pipeline_real_e2e.sio` | sidecar | **CORRIGIDO** | `SCIENCE_PBPK_OK` + live metrics on both engines. |
| 12 | `fo_call_boundary_arity3.sio` | in-file, Madaros FO >2 params | **INDETERMINADO** | CI engine is lean, which the tag already said keeps a live variance (`id3_var=1`). Madaros still prints 0. The defect the tag names is **not** fixed. |
| 13 | `fo_call_boundary_neg.sio` | in-file, Madaros no OpSub | **INDETERMINADO** | Same split: lean `neg_var=1`, Madaros 0. OpSub transfer is still absent. |
| 14 | `test_kaxi_fuse.sio` | sidecar | **INDETERMINADO** | Lean prints `kaxi_fuse: PASS` against the 11.6 / 0.8944 window. Rebuilt Madaros **SEGV 139** in `lower_array` during compile. |
| 15 | `test_core_e2e.sio` | sidecar | **INDETERMINADO** | Lean rc=0 (numeric `check_near` battery). Madaros compile preflight fails. |
| 16 | `test_hyper_math_e2e.sio` | sidecar | **CORRIGIDO** | `HYPER_MATH_OK` on both; oct_mul / sed flags printed. |
| 17 | `test_distributions_e2e.sio` | sidecar | **INDETERMINADO** | Lean `Passed: 7 Failed: 0`. Madaros compile preflight fails. |
| 18 | `test_log_path_cmp.sio` | sidecar | **CORRIGIDO** | Numeric log/path/cmp guards; rc=0 on both. |
| 19 | `gum_fo_across_call.sio` | in-file, Madaros Family A | **INDETERMINADO** | **This is the FO witness.** CI XPASS is lean (`CALL_var 0.000013`). Rebuilt Madaros: `CALL_var 0.000000`, `EPISTEMIC_FABRICATION`. Assertion is alive. Defect is alive. Do not drop the tag. |
| 20 | `associator_field_octonion.sio` | sidecar | **CORRIGIDO** | `ALL PASS` on both (Fano / non-Fano windows). |
| 21 | `knowledge_array.sio` | sidecar | **INDETERMINADO** | Lean prints `KNOWLEDGE_ARRAY_PASS` after the 800-slot stomp. Madaros compile ok, **run SEGV 139**. |

Counts on the CI question: **CORRIGIDO 10**, **TESTEMUNHA-MORTA 1**,
**INDETERMINADO 10**.

The three FO in-file tags in the 21 (rows 12, 13, 19) are INDETERMINADO
because the suite that printed XPAS was not the engine the tag accuses.
On the oracle they still fail.

---

## What `#1939` did and did not close

Title: `fix(madaros): imported 1-2 arg helpers keep first-order variance`.
Receipt on this rebuild:

- imported 1-arg and 2-arg: live (IMP_ID1=4, IMP_ADD2=5, import-boundary OK)
- same-file 1-arg and 2-arg: live (already were)
- same-file ≥3: **still zero**
- imported ≥3: **still zero**
- `0.0 - x` through a user fn: **still zero**
- `rhs(c, cl, fu)` 20-step loop: **still zero**

The 3-arg same-file witness did not start passing because `#1939` over-closed.
It "passes" in CI because CI is lean.

---

## What not to do next

- Do not drop `gum_fo_across_call`, `fo_call_boundary_arity3`,
  `fo_call_boundary_neg`, or `gum_fo_arity3_boundary` tags.
- Do not announce FO fixed. ADD3 is 0.000000 on rebuilt Madaros.
- Sidecar rows that are CORRIGIDO on **both** engines (1, 2, 6, 7, 8, 11, 16,
  18, 20) are the only ones a later lane may consider removing, and only after
  someone else decides. This lane does not touch them.
- `refinement_f64_return_violation` must not be removed from the sidecar on
  the strength of a lean XPASS.

---

## Semantic-Outcome

```text
Semantic-Outcome: measurement only; FO not declared fixed
Concept-Status-Before: #1939 claimed imported 1-2 arg FO; CI printed 21 XPAS including gum_fo_across_call
Concept-Status-After: imported 1-2 confirmed on rebuilt Madaros; arity ≥3 and OpSub still erase; the 21 XPAS are a lean_single Full Suite signal
Distinctions-Added: Full Test Suite stage2 != Madaros claim-oracle; CI XPASS != defect closed
Distinctions-Preserved: uncertainty != ignorance; compile success != runtime parity; lean_single is not the oracle
Distinctions-Erased: none
Evidence-Run: Slurm job 10338 rebuild 226s ELF 100553240; local lean_single matrix; CI job 96048482036
Fallback-Path: none
Legacy-Kept: every known-failure tag and every sidecar row
Conflicting-Lanes: grok-cli5 effect-enum-2a/2b (PR #1960 is the CI surface, not the FO surface)
Next-Semantic-Interface: a later lane may drop sidecar rows that passed on both engines; FO tags stay until ADD3=14.0 and ADD4=14.25 on a source-rebuilt Madaros
```
