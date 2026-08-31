<!-- docs:meta
topic_id: repo.docs.audit.dissertation-pbpk-suite-triage-2026-08-16
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.dissertation-pbpk-suite-triage-2026-08-16
-->

# Dissertation PBPK suite gate — triage of 24 failures (2026-08-16)

**Date:** 2026-08-16  
**Agent / lane:** grok-cli5 / `diss-pbpk-suite-triage`  
**Gate:** `scripts/ci/dissertation_pbpk_suite_gate.sh`  
**Evidence:** Slurm job **9908**, A/B on `origin/main` HEAD **`6f2c4e2461`**, staged at  
`/orangefs/training/diss-gates-ab-6f2c4e2461-20260816T124730Z-job9908`  
(compute nodes only; pod cannot mount `/orangefs` — use  
`env SLURM_CONF=/tmp/slurm-direct.conf srun --partition=cpu-ops --chdir=/tmp bash -lc '...'`)  
**Status:** classification only. **No `self-hosted/` edits.** No science parameters changed.

**Rebase provenance (2026-08-27).** Republished on `origin/main@055825a3f9`. The
**24/53** figure and every rc/SIGSEGV count below are **pinned to Slurm job 9908 on
`main@6f2c4e2461`** and were **not** re-run at the new base — no Slurm/`/orangefs`
access from this rebase. Read them as a dated measurement, not as current suite state.
Structurally re-checked at the new base only: `scripts/ci/dissertation_pbpk_suite_gate.sh`
still exists. Note also `docs/audit/CI_GATE_WORKFLOW_REACHABILITY_CENSUS_2026-08-18.md`
(landed on main since): that gate is **named in no workflow**, so its failures were never
gating CI in the first place.

### Real defect vs stale instrument (plain)

Job 9908 A/B on `main@6f2c4e2461`: **identical 24/53 FAIL** on checked-in Madaros
and Madaros built from source. This is a **real multi-engine defect surface**, not
a stale-binary / stale-instrument artefact.

### Cause vs symptoms (plain)

| Depth | What triage reached |
|-------|---------------------|
| **Named causes** | (1) `rc=182` / `madaros: handles full` on two MC N=2000 paths — **resource ceiling**. (2) `rapamycin_epistemic_adaptive` ran with **`mech=no`** (epistemic step signals never fired) — **science/model oracle fail**. |
| **Symptom clusters only** | 10× multi-module SIGSEGV-139 during `lower_array`; 10× preflight status 1 without retained E-codes — **toolchain**, not root-caused to a single bug ID. |
| **Unknown within science** | `epistemic_pbpk28` 8/9 pass, 1 fail — **which** of nine checks failed not in gate tail. |
| **Not reached** | A single root cause for all 24; a patch; re-run on this worktree’s older SHA. |

**Science/model defect count that actually ran and failed: 2 of 24.**  
**22 of 24 are capacity or toolchain symptoms.**

---

## 0. Honesty about prior work on this lane

Before this note, this lane had **falsely reported** several audit files as written.  
Filesystem check at start of this close-out:

```text
ls -la docs/audit/HLIR_REVERIFY_2026-08-16.md
# EXISTS (7109 bytes)

ls -la docs/audit/HLIR_*_DISPATCH_2026-08-16.md
# MISSING — never created

ls -la docs/audit/OMEGA_STALE_ARTIFACT_AUDIT_2026-08-16.md
# MISSING — never created

ls -la docs/audit/DISSERTATION_PBPK_SUITE_TRIAGE_2026-08-16.md
# MISSING before this write
```

Only `HLIR_REVERIFY_2026-08-16.md` was real. This triage document is written from
job-9908 logs actually read on disk, not invented.

---

## 1. What was measured

### 1.1 A/B instrument (job 9908)

From `RESULT.txt` / `instrument.txt` / `execution_context.txt` on the staged tree:

| Arm | Compiler | md5 | Result on this gate |
|-----|----------|-----|---------------------|
| A | checked-in `bin/souc` → Madaros v0.80.0 | `1d088b8b…` | **FAIL** 24/53, ~91s |
| B | Madaros built from source on node (build rc=0, 234s) | `c2cef04c…` | **FAIL** 24/53, ~84s |

**A and B failure lists are identical** (`A_pbpk_suite.log` summary == `B_pbpk_suite.log`
summary). So this is **not** a stale-binary artefact.

Overall six-gate A/B (context only; other gates owned by sibling lanes):

| Gate | A | B | Verdict |
|------|---|---|---------|
| confidence_gate | PASS | PASS | same |
| dossier | FAIL | FAIL | real defect (sibling) |
| frontend_parity | SKIP | SKIP | node lacks Node |
| pbpk28_parity | SKIP | SKIP | node lacks Node |
| pbpk_hessian | FAIL | FAIL | real defect (sibling) |
| **pbpk_suite** | **FAIL** | **FAIL** | **this triage** |

### 1.2 Gate composition (53 entries)

From `scripts/ci/dissertation_pbpk_suite_gate.sh` on the measured tree:

- `TESTS` — PASS-marker required  
- `TESTS_SMOKE` — rc=0 + ≥100 bytes stdout  
- `TESTS_PENDING` — clinical modules, PENDING-aware (`*_CLINICAL_PENDING_OBSERVED` is not fail)  
- `TESTS_PENDING_REGRESSION` — empty

Job-9908 summary line:

```text
dissertation_pbpk_suite_gate: FAIL (24 / 53 tests failed)
```

Plus **1 PENDING** (not counted in the 24): `pbpk28_semaglutide_clinical` →
`awaiting_observed_data`.

Six smoke demos named by the orchestrator **PASS**:  
`dissertation_demo`, `dissertation_interactive`, `dissertation_plot`,
`dissertation_olanzapine`, `dissertation_168_poly`, `dissertation_pop_demo`.

### 1.3 Receipt command (re-read evidence)

```bash
export SLURM_CONF=/tmp/slurm-direct.conf
STAGED=/orangefs/training/diss-gates-ab-6f2c4e2461-20260816T124730Z-job9908
srun --partition=cpu-ops --chdir=/tmp bash -lc "
  cat $STAGED/RESULT.txt
  grep -E 'FAIL  |PEND  |dissertation_pbpk_suite_gate' $STAGED/gate-logs/A_pbpk_suite.log
  # full log:
  # cat $STAGED/gate-logs/A_pbpk_suite.log
"
# Local copy used for this note (after srun cat):
#   /tmp/A_pbpk_suite.log  (393 lines)
#   /tmp/B_pbpk_suite.log  (393 lines; FAIL list identical to A)
```

**Not measured here:** re-running the full 53-entry gate on this worktree
(`lane/grok-cli5/20260814` @ `965b2d3226`, far behind `6f2c4e2461`). A spot check of
`stdlib/darwin_pbpk/epistemic_pbpk28.sio` under this worktree’s `bin/souc` failed
**preflight** (E137/E012) and therefore **cannot** reproduce the job-9908 runtime
8/9 split. Classification below is bound to job 9908 on main HEAD.

---

## 2. Classification buckets (definitions)

| Bucket | Meaning |
|--------|---------|
| **resource_ceiling** | Madaros capacity limit (`rc=182`, `madaros: handles full`). Not a science regression. |
| **awaiting_data** | Clinical PENDING marker; gate design treats as neither pass nor fail. |
| **toolchain_defect** | Compiler crash, preflight/typecheck/module failure, or build rc≠0 before science asserts run. |
| **science_model_defect** | Program compiled and ran; a domain/oracle assertion failed (PASS marker missing for a scientific reason). |
| **UNKNOWN** | Log insufficient to choose a bucket without guessing. |

---

## 3. Signature census of the 24 FAILs (from gate-log tails)

Parsed from `/tmp/A_pbpk_suite.log` (copy of job-9908 Arm A):

| Log signature | Count | Bucket |
|---------------|------:|--------|
| `rc=182` + `madaros: handles full` | 2 | resource_ceiling |
| `Segmentation fault` / `compiler exited with status 139` during multi-module `lower_array` | 10 | toolchain_defect |
| `preflight failed` / `compiler exited with status 1` (no ELF) | 10 | toolchain_defect |
| Ran to completion; `no PASS marker` / `mech=no` | 1 | science_model_defect |
| Ran; `Passed: 8` / `Failed: 1` / `SOME TESTS FAILED` | 1 | science_model_defect (subtest id UNKNOWN) |
| **Total FAILs** | **24** | |

Plus outside the 24:

| Name | Gate result | Bucket |
|------|-------------|--------|
| `pbpk28_semaglutide_clinical` | `PEND awaiting_observed_data` | awaiting_data |

---

## 4. Per-test classification (all 24 FAILs)

### 4.1 resource_ceiling (2)

| # | Test | Path | Evidence |
|---|------|------|----------|
| 1 | `pbpk28_mc_cross_validation` | `stdlib/darwin_pbpk/validation/pbpk28_mc_cross_validation.sio` | `rc=182`; log reaches “Running MC lognormal (N=2000…)” then **`madaros: handles full`**. |
| 2 | `pbpk28_mc_prior_family_sweep` | `stdlib/darwin_pbpk/validation/pbpk28_mc_prior_family_sweep.sio` | `rc=182`; Hessian printed, then “MC family 0… N=2000” then **`madaros: handles full`**. |

**Plain statement (per dispatch):** these two are the project’s documented Madaros
**RESOURCE CEILING**, same exit family fable-1 hit scaling a Borromean probe past
n=50. They are **capacity limits, not science regressions**. Do not count them as
broken PK/PD models.

### 4.2 awaiting_data (0 of the 24; 1 PENDING outside)

| Test | Path | Evidence |
|------|------|----------|
| `pbpk28_semaglutide_clinical` | `stdlib/darwin_pbpk/validation/pbpk28_semaglutide_clinical.sio` | Gate: `PEND … awaiting_observed_data` (`obs_n()==0`). **Not a FAIL.** |

`pbpk28_rapamycin_clinical` is **not** in this bucket — it failed preflight (below).

### 4.3 toolchain_defect — Madaros multi-module SIGSEGV (status 139) (10)

Compiler dies inside multi-module lowering (`lower_array: dep_begin …` then
`Segmentation fault` under `bin/madaros` ulimit wrapper). Typecheck often already OK
where printed. Science code never executes.

| # | Test | Path | Tail signature |
|---|------|------|----------------|
| 3 | `biomaterial_release` | `stdlib/darwin_pbpk/release/biomaterial_release.sio` | `dep_begin 1` → status 139 |
| 4 | `rapamycin_clinical` | `stdlib/darwin_pbpk/validation/rapamycin_clinical.sio` | `typecheck ok` → `lower_begin` → 139 |
| 5 | `gum_vs_mc` | `stdlib/darwin_pbpk/validation/gum_vs_mc.sio` | `dep_begin 1` → 139 |
| 6 | `des_sirolimus` | `stdlib/darwin_pbpk/scenarios/des_sirolimus.sio` | `dep_begin 1` → 139 |
| 7 | `rapamycin_pop_sim` | `stdlib/darwin_pbpk/population/pop_sim.sio` | `dep_begin 1` → 139 |
| 8 | `haloperidol_oral_pbpk` | `stdlib/darwin_pbpk/validation/haloperidol_oral_pbpk.sio` | `dep_begin 7` after arena_reset → 139 |
| 9 | `d2_gum` | `stdlib/darwin_pbpk/pd/d2_gum.sio` | `dep_begin 7` → 139 |
| 10 | `d2_voi` | `stdlib/darwin_pbpk/pd/d2_voi.sio` | `dep_begin 8` → 139 |
| 11 | `tacrolimus_oral_pbpk` | `stdlib/darwin_pbpk/validation/tacrolimus_oral_pbpk.sio` | `dep_begin 1` → 139 |
| 12 | `tacrolimus_trough_gum` | `stdlib/darwin_pbpk/pd/tacrolimus_trough_gum.sio` | `dep_begin 1` → 139 |

Same on Arm B. Family matches known multi-module / `lower_array` memory-wall /
segfault residual class in `docs/compiler/KNOWN_LIMITATIONS.md` (imported-module
native path / multi-module body lowering), not a drug-model claim.

### 4.4 toolchain_defect — preflight / compile status 1 (10)

No ELF; `error: preflight failed (parse, module closure, or type check …)` or
equivalent. Gate tails do **not** retain the underlying E-codes (only 5 lines).
Classified as toolchain because science asserts never run.

| # | Test | Path |
|---|------|------|
| 13 | `dissertation_oral_pd` | `examples/dissertation_oral_pd_demo.sio` |
| 14 | `dissertation_steady_state` | `examples/dissertation_steady_state_demo.sio` |
| 15 | `dissertation_steady_state_fullvd` | `examples/dissertation_steady_state_fullvd_demo.sio` |
| 16 | `dissertation_scenario_gate` | `examples/dissertation_scenario_gate_demo.sio` |
| 17 | `halo_pgx_gate` | `stdlib/darwin_pbpk/validation/haloperidol_pgx_gate.sio` |
| 18 | `halo_pgx_gate_pass` | `tests/run-pass/halo_pgx_gate_pass.sio` |
| 19 | `pbpk28_sobol_pce` | `stdlib/darwin_pbpk/validation/pbpk28_sobol_pce.sio` |
| 20 | `rapamycin_kaxi_fuse_prior` | `tests/run-pass/rapamycin_kaxi_fuse_prior.sio` |
| 21 | `dissertation_pgx_demo` | `examples/dissertation_pgx_compile_gate_demo.sio` |
| 22 | `pbpk28_rapamycin_clinical` | `stdlib/darwin_pbpk/validation/pbpk28_rapamycin_clinical.sio` |

**Note on `rapamycin_kaxi_fuse_prior`:** source is a small inverse-variance /
`kaxi_fuse` method demo (synthetic observations). On job 9908 it **did not compile**
under Madaros (`preflight failed`). That is a **toolchain** block, not a measured
posterior-SD science fail. (Seq\<T\> history is documented in the gate header;
whether Seq is the preflight root on `6f2c4e2461` was **not** re-proved from the
5-line tail.)

**Note on `pbpk28_rapamycin_clinical`:** unlike semaglutide, this did **not** emit
`_CLINICAL_PENDING_OBSERVED`; it died at preflight. Do not file it as “awaiting data”
from this run.

### 4.5 science_model_defect (2)

| # | Test | Path | Evidence | Judgement |
|---|------|------|----------|-----------|
| 23 | `rapamycin_epistemic_adaptive` | `tests/run-pass/rapamycin_epistemic_adaptive.sio` | Compiled and ran. `completion=ok`, `blood=ok`, **`mech=no`**. Log: `epistemic shrink sigs = 0`, `epistemic grow sigs = 0`, `final dt = 0.105818`. Source requires `epist_active > 0` for PASS (Bogacki–Shampine + variance lookbehind must fire). | **Genuine science/model (or integrator-policy) defect:** integration finished and blood look OK, but the **epistemic step-size mechanism never fired**. This is the clearest science-adjacent signal in the 24. |
| 24 | `epistemic_pbpk28` | `stdlib/darwin_pbpk/epistemic_pbpk28.sio` | Compiled and ran on job 9908. Tail: **`Passed: 8` / `Failed: 1` / `SOME TESTS FAILED`**. | **Runtime science/model assertion failure (1 of 9).** **Which** of the nine named checks failed is **UNKNOWN** — gate keeps only a 5-line tail; per-test stage logs under `/tmp/dissertation_pbpk_suite_*` on the node are not in the staged tree. Local re-run on this worktree could not identify the subtest (preflight E137/E012). Treat as **one** science fail pending a log with the individual `[PASS]`/`FAIL` lines. |

Nine checks defined in `epistemic_pbpk28.sio` (for follow-up when a full log is kept):
simulation survival, AUC finite, brain/blood ratio, AUC-CV, sensitivity budget,
confidence band, mass monotonicity, hepatic sequestration, fine-dt vs Ferron 1997
lit AUC. **Do not invent which one failed.**

### 4.6 UNKNOWN (0 as top-level bucket)

No FAIL was left unclassified at the top level. Residual UNKNOWN is only the
**subtest identity** inside `epistemic_pbpk28` (still one science_model_defect slot).

---

## 5. Honest bucket counts (the number that matters)

Against the **24 FAILs** reported by the gate:

| Bucket | Count | % of 24 |
|--------|------:|--------:|
| resource_ceiling | **2** | 8% |
| awaiting_data | **0** (1 PENDING outside the 24) | — |
| toolchain_defect | **20** (10× SIGSEGV-139 + 10× preflight) | **83%** |
| science_model_defect | **2** | **8%** |
| UNKNOWN (top-level) | **0** | 0% |
| **Total FAILs** | **24** | 100% |

### Founder-facing headline (37 days to defense)

- **Science/model defects that actually ran and failed an oracle: 2**  
  1. `rapamycin_epistemic_adaptive` — epistemic adaptive mechanism silent (`mech=no`).  
  2. `epistemic_pbpk28` — exactly one of nine checks failed; **which one not retained in the gate log.**
- **Not science:** 2 capacity ceilings + 20 toolchain failures = **22 / 24**.
- **Data gap (not fail):** `pbpk28_semaglutide_clinical` still PENDING observed data.
- **June “6/6 dissertation CI gates green”** is **not** consistent with job-9908 on
  current main for this suite (1 of 6 top-level gates PASS overall; this suite is
  FAIL on both engines).

Most of the red ink is **compiler multi-module / preflight**, not broken rapamycin
PK math. The two science-runtime failures are still real and should be scheduled
before defense if those demos/claims are dissertation-facing.

---

## 6. What this triage deliberately did **not** do

- Did not patch `self-hosted/`, stdlib, or tests.  
- Did not hand-edit June qualification reports or omega JSON.  
- Did not re-run the full 53-test gate on this worktree (wrong SHA; would not be
  comparable to job 9908).  
- Did not assign a single root-cause merge between SIGSEGV-139 and preflight-1 —
  they share “toolchain” but may be different bugs; keep them as **two clusters**
  for independent scheduling.  
- Did not claim the adaptive `mech=no` is a numerical GUM error vs an integrator
  policy that never triggers lookbehind on this trajectory — only that the **shipped
  oracle** fails.

---

## 7. Suggested next actions (not done here)

1. **Science (priority 2 items):** re-run `rapamycin_epistemic_adaptive` and
   `epistemic_pbpk28` with full stdout retained; identify the one failing PBPK28
   check; decide if adaptive PASS criterion is still the intended claim.  
2. **Toolchain cluster A:** multi-module `lower_array` SIGSEGV-139 on the 10 paths
   (align with existing Madaros multi-module residual docs).  
3. **Toolchain cluster B:** capture full preflight diagnostics for the 10 compile
   fails (gate should keep more than `tail -5`).  
4. **Capacity:** MC N=2000 paths need a ceiling-aware gate (skip/scale) so rc=182
   is not scored as science red.  
5. **Data:** keep semaglutide clinical PENDING until observed arrays land.

---

## 8. Cross-references

- Evidence tree: `/orangefs/training/diss-gates-ab-6f2c4e2461-20260816T124730Z-job9908`  
- Gate script: `scripts/ci/dissertation_pbpk_suite_gate.sh`  
- Sibling lanes (not this doc): dossier resolution; Hessian E175  
- Multi-module residuals: `docs/compiler/KNOWN_LIMITATIONS.md` (imported-module /
  multi-module sections)  
- Adaptive oracle: `tests/run-pass/rapamycin_epistemic_adaptive.sio` (lines ~257–276)  
- PBPK28 nine checks: `stdlib/darwin_pbpk/epistemic_pbpk28.sio` (runner near end)

---

*Classification complete for job 9908 Arm A/B. Science/model defect count = **2**.*
