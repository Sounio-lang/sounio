<!-- docs:meta
topic_id: repo.docs.audit.dissertation-pbpk-suite-remeasure-2026-08-17
authority: repo_only
audience: users
last_validated: 2026-08-17
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.dissertation-pbpk-suite-remeasure-2026-08-17
-->

# Dissertation PBPK suite re-measure — 2026-08-17

**Gate:** `scripts/ci/dissertation_pbpk_suite_gate.sh`  
**Source commit:** `d0c798e4ed` (`origin/main` tip including #1795 dual-path parity)  
**Compiler:** Madaros **built from source** on the node (`build_modular_madaros.sh` → md5 `fe91a596981d88fa…`, v0.80.0)  
**Prior table:** [`DISSERTATION_PBPK_SUITE_TRIAGE_2026-08-16.md`](DISSERTATION_PBPK_SUITE_TRIAGE_2026-08-16.md) — **24 FAIL / 28 PASS / 1 PEND** on job 9908 (`6f2c4e2461`)  
**This measure:** **19 FAIL / 33 PASS / 1 PEND / 0 SKIP / 0 UNKNOWN**  
**Author:** grok-cli2 / lane `pbpk-suite-remeasure-20260817`

---

## 0. The number this exercise exists to produce

### Whole suite (53 registered)

| Outcome | 2026-08-16 (job 9908) | **2026-08-17 (this run)** | Δ |
|---|---:|---:|---:|
| **PASS** | 28 | **33** | **+5** |
| **FAIL** | 24 | **19** | **−5** |
| **PEND** (awaiting data) | 1 | **1** | 0 |
| **SKIP** | 0 | **0** | 0 |
| **UNKNOWN** | 0 | **0** | 0 |

PEND is its own category. It is **not** a pass. SKIP did not occur. UNKNOWN did not occur.

### Among the **19 FAILs** only

| Bucket | N | Note |
|---|---:|---|
| resource ceiling (`rc=182` / handles full) | **7** | was 2; +5 from former lower_array SIGSEGV class |
| awaiting data | **0** | semaglutide is PEND, not FAIL |
| toolchain defect | **12** | preflight E175/E137/E170/E009/E011 + epistemic runtime corruption |
| **genuine science or model defect** | **0** | |
| UNKNOWN | **0** | |

**Founder-worry size (genuine science/model among fails): still `0`.**

---

## 1. Headline: the lower_array ten after #1799

August cluster L was **ten** tests dying in compile with `compiler exited with status 139` after `lower_array` progress. Today's landings that were expected to touch this surface: **#1799** (`println(bool)` / unclassified-scalar char\* SIGSEGV — “biomaterial_release + class”), plus earlier Madaros main motion.

### Measured fate of the ten

| # | Test | 2026-08-16 | **2026-08-17** | What changed |
|---:|---|---|---|---|
| 6 | `biomaterial_release` | FAIL SIGSEGV 139 | **PASS** | fully green |
| 7 | `rapamycin_clinical` | FAIL SIGSEGV 139 | **FAIL rc=182** | **compiles** (`Written to a.out`); dies mid GUM budget: `madaros: handles full` |
| 8 | `gum_vs_mc` | FAIL SIGSEGV 139 | **FAIL rc=182** | compiles; dies mid MC: `handles full` |
| 9 | `des_sirolimus` | FAIL SIGSEGV 139 | **PASS** | fully green |
| 10 | `rapamycin_pop_sim` | FAIL SIGSEGV 139 | **FAIL rc=182** | compiles; dies during 20-patient run: `handles full` |
| 12 | `haloperidol_oral_pbpk` | FAIL SIGSEGV 139 | **PASS** | fully green |
| 13 | `d2_gum` | FAIL SIGSEGV 139 | **FAIL rc=182** | compiles; dies at GUM budget start: `handles full` |
| 14 | `d2_voi` | FAIL SIGSEGV 139 | **FAIL rc=182** | compiles; dies at VoI start: `handles full` |
| 30 | `tacrolimus_oral_pbpk` | FAIL SIGSEGV 139 | **PASS** | fully green |
| 31 | `tacrolimus_trough_gum` | FAIL SIGSEGV 139 | **PASS** | fully green |

### Cluster L scorecard

| Metric | Value |
|---|---:|
| Fully green of the ten | **5** |
| Still FAIL of the ten | **5** |
| Still SIGSEGV 139 | **0** |
| Migrated to resource ceiling after successful native emit | **5** |

**What must be said in the number:** #1799 (and current main) did **not** turn the ten into ten passes. It turned **compile-time death into either pass or handle-ceiling**. That is a real, large step — half the class is dissertation-green — and also a refutation of “println was the sole remaining cause of all ten reds.” The five residual members now fail for a **different** reason (Madaros handle table / `rc=182`), after printing scientific headers and completing compile. That is more important than a false all-green claim would have been.

The five new PASSes are exactly the whole-suite Δ: **24 → 19 FAIL, 28 → 33 PASS**.

---

## 2. Receipt (how to re-read the evidence)

Evidence lives on the compute node’s OrangeFS mount (not visible from the workspace pod):

```text
STAGE=/orangefs/training/pbpk-suite-remeasure-d0c798e4edcd-20260817T221419Z
```

```bash
env SLURM_CONF=/tmp/slurm-direct.conf srun \
  --partition=cpu-ops --nodes=1 --ntasks=1 --time=00:05:00 --chdir=/tmp \
  bash -lc "
    cat $STAGE/RESULT.txt
    cat $STAGE/instrument.txt
    cat $STAGE/SOURCE_COMMIT.txt
    tail -80 $STAGE/logs/pbpk_suite.log
    # per-test logs:
    ls $STAGE/suite-stage/*.log | wc -l
  "
```

| Field | Value |
|---|---|
| Host | `cpuops-t560-proxmox` (via `srun --partition=cpu-ops`) |
| Source tree | `git archive` of `d0c798e4ed` streamed to `$STAGE/repo` |
| Madaros | `bash scripts/ci/build_modular_madaros.sh $STAGE/build/madaros` (not prebuilt `bin/souc`) |
| Madaros md5 | `fe91a596981d88fa8d38295a673e0253` |
| Suite log | `$STAGE/logs/pbpk_suite.log` |
| Per-test logs | `$STAGE/suite-stage/<name>.log` (53 files) |
| `SOUC_BIN` | source-built Madaros |
| `DPS_TIMEOUT_SECONDS` | 180 |
| Suite wall | ~87 s after build (~4 min build) |
| Finished UTC | 2026-08-17T22:21:35Z |
| Gate exit | `dissertation_pbpk_suite_gate: FAIL (19 / 53 tests failed)` |

**Instrument discipline:** prebuilt `bin/souc` was **not** trusted as the measurement engine. The seed ELF was only used to compile Madaros from `self-hosted/`.

---

## 3. Per-test table (all 53)

Order matches gate registration (`TESTS` → `TESTS_SMOKE` → `TESTS_PENDING`).  
**Aug** = 2026-08-16 job 9908. **Now** = this run.

### 3.1 Main tests (`TESTS`)

| # | Test | Aug | Now | Evidence (this run) | Bucket now |
|---:|---|---|---|---|---|
| 1 | `rapamycin_iso_budget` | PASS | PASS | — | — |
| 2 | `rapamycin_rk4_budget` | PASS | PASS | — | — |
| 3 | `rapamycin_epistemic_pbpk` | PASS | PASS | — | — |
| 4 | `rapamycin_epistemic_adaptive` | FAIL | FAIL | `var(blood/brain/periph)=0`; `epistemic * sigs = 0`; `FAIL: completion=ok blood=ok mech=no` | **toolchain** (GUM variance collapsed; sibling #3 still green) |
| 5 | `rapamycin_gum_vs_mc` | PASS | PASS | — | — |
| 6 | `biomaterial_release` | FAIL 139 | **PASS** | native emit + PASS | **recovered** (was L) |
| 7 | `rapamycin_clinical` | FAIL 139 | FAIL **182** | PART A clinical checks print PASS; PART B GUM → `handles full` | **resource ceiling** (was L) |
| 8 | `gum_vs_mc` | FAIL 139 | FAIL **182** | GUM SD printed; MC → `handles full` | **resource ceiling** (was L) |
| 9 | `des_sirolimus` | FAIL 139 | **PASS** | — | **recovered** (was L) |
| 10 | `rapamycin_pop_sim` | FAIL 139 | FAIL **182** | 20-patient header; `handles full` | **resource ceiling** (was L) |
| 11 | `haloperidol_d2_pet` | PASS | PASS | — | — |
| 12 | `haloperidol_oral_pbpk` | FAIL 139 | **PASS** | — | **recovered** (was L) |
| 13 | `d2_gum` | FAIL 139 | FAIL **182** | native emit; GUM header; `handles full` | **resource ceiling** (was L) |
| 14 | `d2_voi` | FAIL 139 | FAIL **182** | native emit; VoI header; `handles full` | **resource ceiling** (was L) |
| 15 | `dissertation_pbpk_rapamycin` | PASS | PASS | — | — |
| 16 | `dissertation_oral_pd` | FAIL | FAIL | **E175** private `drugs/rapamycin::rapamycin_mean_params` | **toolchain** (visibility) |
| 17 | `dissertation_steady_state` | FAIL | FAIL | **E175** private + **E008** return type + **E137** `print_i64` | **toolchain** |
| 18 | `dissertation_steady_state_fullvd` | FAIL | FAIL | same family as #17 | **toolchain** |
| 19 | `dissertation_scenario_gate` | FAIL | FAIL | **E175** `rapamycin_mean_params`; **E137** `print_i64` in `bbb_voi` | **toolchain** |
| 20 | `rodgers_rowland_kp` | PASS | PASS | — | — |
| 21 | `gnn_rapamycin_inference` | PASS | PASS | — | — |
| 22 | `hybrid_ode_rapamycin` | PASS | PASS | — | — |
| 23 | `dissertation_hybrid_demo` | PASS | PASS | — | — |
| 24 | `tirzepatide_sc_pbpk` | PASS | PASS | — | — |
| 25 | `glp1_gipr_gum` | PASS | PASS | — | — |
| 26 | `dissertation_tirzepatide_demo` | PASS | PASS | — | — |
| 27 | `vancomycin_icu_pbpk` | PASS | PASS | — | — |
| 28 | `vancomycin_auc_gum` | PASS | PASS | — | — |
| 29 | `dissertation_vancomycin_demo` | PASS | PASS | — | — |
| 30 | `tacrolimus_oral_pbpk` | FAIL 139 | **PASS** | — | **recovered** (was L) |
| 31 | `tacrolimus_trough_gum` | FAIL 139 | **PASS** | — | **recovered** (was L) |
| 32 | `tacrolimus_ddi_module` | PASS | PASS | — | — |
| 33 | `tacrolimus_ddi_clinical` | PASS | PASS | — | — |
| 34 | `cross_drug_iso_budget` | PASS | PASS | — | — |
| 35 | `halo_pgx_gate` | FAIL | FAIL | **E175** private `math/pure::sqrt` from `aggregate_confidence` | **toolchain** (E175 pub residual) |
| 36 | `halo_pgx_gate_pass` | FAIL | FAIL | **E170** `.value` needs `with Epistemic` / `acknowledge` | **toolchain** (effect surface) |
| 37 | `olanzapine_d2_mtor` | PASS | PASS | — | — |
| 38 | `pop_pbpk_pd` | PASS | PASS | — | — |
| 39 | `epistemic_pbpk28` | FAIL | FAIL | 8/9 internal PASS; TEST 6 `AUC confidence: 4604219396932172800` | **toolchain** (representation) |
| 40 | `epistemic_pbpk28_hessian` | PASS | PASS | — | — |
| 41 | `pbpk28_sobol_pce` | FAIL | FAIL | **E009** fn-type mismatch; **E035** missing `Epistemic` | **toolchain** |
| 42 | `pbpk28_mc_cross_validation` | FAIL 182 | FAIL **182** | Hessian `u_Hessian=0.295160`; MC N=2000 → `handles full` | **resource ceiling** |
| 43 | `pbpk28_mc_prior_family_sweep` | FAIL 182 | FAIL **182** | same class; family 0 N=2000 → `handles full` | **resource ceiling** |
| 44 | `rapamycin_kaxi_fuse_prior` | FAIL | FAIL | **E011**/**E013**/**E137** `acknowledge` on kaxi path | **toolchain** (Seq/kaxi surface) |

### 3.2 Smoke tests (`TESTS_SMOKE`)

| # | Test | Aug | Now | Evidence | Bucket now |
|---:|---|---|---|---|---|
| 45 | `dissertation_demo` | PASS | PASS | smoke rc=0 | — |
| 46 | `dissertation_interactive` | PASS | PASS | — | — |
| 47 | `dissertation_plot` | PASS | PASS | — | — |
| 48 | `dissertation_pgx_demo` | FAIL | FAIL | **E175** `math/pure::sqrt` (same as #35) | **toolchain** |
| 49 | `dissertation_olanzapine` | PASS | PASS | — | — |
| 50 | `dissertation_168_poly` | PASS | PASS | — | — |
| 51 | `dissertation_pop_demo` | PASS | PASS | — | — |

### 3.3 Clinical pending-aware (`TESTS_PENDING`)

| # | Test | Aug | Now | Evidence | Bucket now |
|---:|---|---|---|---|---|
| 52 | `pbpk28_rapamycin_clinical` | FAIL | FAIL | **E011** ontology/model methods — still never reaches data path | **toolchain** (ontology surface) |
| 53 | `pbpk28_semaglutide_clinical` | **PEND** | **PEND** | `awaiting_observed_data` / `*_CLINICAL_PENDING_OBSERVED` | **awaiting data** (not a pass) |

---

## 4. Family / cluster verdict (fell vs remain)

| Cluster | Aug FAIL | Now FAIL | Recovered (PASS) | Residual mode |
|---|---:|---:|---:|---|
| **L** lower_array SIGSEGV | **10** | **0 as SIGSEGV** | **5** green | **5** now **R** (handles full after native emit) |
| **V** multi-module visibility / E175 / print_i64 | **6** named in Aug table | **6** | **0** | oral_pd, steady_state×2, scenario_gate, halo_pgx, pgx_demo — still E175/E137/E008 |
| **E** epistemic / kaxi / effects | **3** | **3** | **0** | `halo_pgx_gate_pass` E170; `sobol_pce` E009/E035; `kaxi_fuse_prior` |
| **R** resource ceiling | **2** MC | **7** (= 2 MC + 5 ex-L) | **0** of historical MC | MC N=2000 still 182; former L now dies after native emit |
| **C** runtime epistemic corruption | **2** | **2** | **0** | adaptive zero variance; pbpk28 confidence ~4.6e18 |
| **O** ontology blocks clinical | **1** | **1** | **0** | `pbpk28_rapamycin_clinical` still E011 |

Check: 6V + 3E + 7R + 2C + 1O = **19**.  

E175 `pub` landings claimed today did **not** clear `rapamycin_mean_params` nor `math/pure::sqrt` for this suite surface.

**#1789 (dossier):** not a suite member; no row expected to flip from that PR alone.

---

## 5. What this means for defense readiness

1. **Suite is still FAIL** (19/53), but the August table is obsolete: **five fewer fails**, all from the former lower_array class.  
2. **The largest single movement in months on this gate** is: **5/10 of cluster L now PASS**, and **0/10 still die with SIGSEGV 139**. Say both numbers.  
3. **The incomplete half of L is not a silent non-event:** those five now **compile and start science**, then hit **handle ceiling**. Repair target shifted from “codegen println/class” to **runtime capacity (handle table / rc=182)** for clinical GUM, MC, pop sim, and D2 GUM/VoI.  
4. **E175 residuals remain the second wall** (demos + PGx + steady-state runner). minimax-cli2 pub work did not clear this suite’s callees.  
5. **Science-bucket size among fails remains 0.** No new genuine PK model fails appeared among previously green rows.  
6. **PEND stays PEND** — semaglutide is not a green tick.  
7. Combined with the six-gates story after #1795 (**3 PASS, 2 FAIL, 0 UNKNOWN**): the dissertation measurement surface is more honest and strictly greener on this suite, without inventing passes.

---

## 6. Suggested repair order (updated)

1. **R / handle ceiling** — unblocks 7 fails (5 former L + 2 MC), including dissertation-facing clinical GUM and D2 GUM/VoI.  
2. **V / E175 + print_i64** — unblocks 6–7 demos and PGx compile gates (`rapamycin_mean_params`, `math/pure::sqrt`, `print_i64`).  
3. **C** — adaptive zero GUM variance; pbpk28 confidence bit-pattern.  
4. **E** — kaxi/Seq surface; Epistemic effect on `halo_pgx_gate_pass`; sobol fn-type.  
5. **O** — ontology method surface for `pbpk28_rapamycin_clinical`.  
6. **Data** — semaglutide observed arrays (PEND → real clinical gate).

---

## 7. Method notes / non-claims

- Classifications use **only** this stage’s suite log + per-test logs on the source-built Madaros.  
- “Toolchain defect” still means: failure is not a scientific disagreement with clinical data.  
- No UNKNOWN rows.  
- Default gate `SOUC_BIN` (`souc-seq-leansingle.sh`) was **overridden** to source-built Madaros — matching the August A/B Madaros arm and the defense engine of record.  
- Pod did not run the suite; Slurm `cpu-ops` did (pod cannot see `/orangefs`).  
- This document does not claim #1799 alone caused the five greens without other main motion since `6f2c4e2461`; it claims the **measured** before/after on the registered suite at the two commits named.

---

## 8. Document control

| Date | Change |
|---|---|
| 2026-08-17 | Re-measure on `d0c798e4ed` with source-built Madaros; 19 FAIL / 33 PASS / 1 PEND; cluster L 5 green + 5→R. |
