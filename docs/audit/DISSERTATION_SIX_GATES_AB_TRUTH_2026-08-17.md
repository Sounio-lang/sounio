<!-- docs:meta
topic_id: repo.docs.audit.dissertation-six-gates-ab-truth-2026-08-17
authority: repo_only
audience: users
last_validated: 2026-08-17
validated_by: grok-cli2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.dissertation-six-gates-ab-truth-2026-08-17
-->

# Dissertation six-gate truth — A/B instrument vs real defect

**Date:** 2026-08-17  
**Defense horizon:** ~36 days (from STATUS_AUDIT 2026-09-22 target)  
**Lane:** grok-cli2 / `diss-gates-ab-truth`  
**Worktree:** `/workspace/.wt/diss-gates-ab-truth` on `docs/diss-gates-ab-truth-20260817` @ `origin/main` (control `/workspace/sounio` **not** used — dirty foreign branch)  
**Question:** June qualification claimed **6/6 green**. Job **9904** re-measured on main with checked-in `bin/souc`: **pass=1 fail=3 skip=2**. Is that a **stale-instrument artefact** or **real defect**?

---

## 0. Headline answer (do not bury)

### Tip re-run (current `origin/main`, 2026-08-17)

| Gate | Tip Arm A (checked-in) | Tip Arm B (from-source) | Pod+Node (where applicable) | Verdict |
|---|---|---|---|---|
| `confidence_gate` | **PASS** | **PASS** | — | Real **green** on both engines |
| `dossier` | **FAIL** | **FAIL** | FAIL (same) | **REAL DEFECT** |
| `frontend_parity` | **SKIP** (no node on compute) | **SKIP** | **PASS** 14/14 @ 1% RMSE | **Green when Node is present**; compute-node skip is environment, not science fail |
| `pbpk28_parity` | **SKIP** (no node) | **SKIP** | **FAIL** (case 7 semaglutide: E175 private `glp1r_rtotal0_at` + E008 Bergman return type; cases 1,4,5,6 PASS) | **Not green**: Node removes SKIP but exposes **toolchain/API defects** on semaglutide path |
| `pbpk_hessian` | **FAIL** | **FAIL** | — | **REAL DEFECT** |
| `pbpk_suite` | **FAIL** | **FAIL** | — | **REAL DEFECT** (toolchain-heavy; 0 genuine science fails in suite triage) |

Tip A/B stage: `/orangefs/training/diss-gates-ab-tip-dca2775061-20260817T151650Z-job10117`  
Pin: `dca2775061f0` · Arm A md5 `1d088b8b…` · Arm B md5 `518006cc…` (build 235s) · same fail pattern as job **9908**.

**June 6/6 is not reproduced on tip.**  
**Three FAILs remain real on both engines** (dossier, hessian, suite) — **not** stale-instrument.  
**SKIPs on compute are not green**; on a Node-bearing pod, frontend is **PASS** and pbpk28 is **FAIL** (semaglutide ref).

**SKIP ≠ PASS.**

Primary historical A/B: job **9908** pin `6f2c4e2461`. Tip confirmation: job **10117** pin `dca2775061`.


---

## 1. Instruments and pins

| Item | Value |
|---|---|
| Schema | `sounio.dissertation.six-gate-ab-instrument.v1` |
| Prior single-arm | job **9904** (checked-in only) |
| A/B job | **9908** |
| Source commit | `6f2c4e2461cc` (main tip at that measure; planning-tranche merge era) |
| Node | `cpuops-t560-proxmox` |
| Node.js | **ABSENT** (`node_on_compute_node=ABSENT`) |
| Arm A `bin/souc` md5 | `1d088b8b263356e96948f26a9dd17702` |
| Arm B from-source md5 | `c2cef04c6d9088dc71c2525142081712` (build 234s, rc=0) |
| Both report | Madaros v0.80.0 |

Different engine bytes; **identical pass/fail/skip pattern** → instrument staleness cannot explain the three FAILs.

---

## 2. Per-gate detail

### 2.1 `confidence_gate` — PASS / PASS

**Script:** `scripts/ci/dissertation_confidence_gate_gate.sh` (note double `_gate`).  
**What it measures:** compile-time Epistemic confidence on rapamycin priors (honest ceiling runs; overclaim and non-literal ε rejected).  
**9908:** PASS both arms (~1s).  
**Cheap recheck (this worktree, prebuilt `bin/souc`):** PASS again.  
**June claim:** consistent with a real green contribution-2 gate **if** the June run used the same fixtures.

### 2.2 `dossier` — FAIL / FAIL — REAL DEFECT

**Script:** `scripts/ci/dissertation_dossier_gate.sh`  
**9908:** FAIL both arms — `souc check` / compile of `tests/run-pass/dossier_smoke.sio` fail; no executable; no `PASS dossier_smoke`.  
**Cheap recheck on current `origin/main` tip (this WT):** same FAIL shape.  
**Check diagnostic (representative):**

```text
unresolved import in authoritative closure: scripts::dissertation::dossier_generator
run_check_mode: AST closure incomplete nodes=1 unresolved=1
```

Root class: **module/import/closure** — smoke cannot resolve the generator module. Not a numeric tolerance issue. **Do not “fix” by loosening the golden.**

### 2.3 `frontend_parity` — SKIP / SKIP — UNMEASURED

**Script:** `scripts/ci/dissertation_frontend_parity_gate.sh`  
**Why skip (logged, both arms):**

```text
[parity] SKIP: node not available (gate requires Node ≥ 18 for ESM .mjs)
```

**Exit code:** 0 after SKIP (gate self-skips successfully). Harness must still label **SKIP**, not PASS.  
**What would measure:** Sounio 14-comp trajectory vs Node ESM runner, 1% RMSE.  
**Status for defense:** **unknown** until run on a node with Node ≥ 18 (or a non-Node reference path is built).  
**Pod note:** this interactive pod *has* Node v22; the **compute node used for 9908 did not**. Environment split.

### 2.4 `pbpk28_parity` — SKIP / SKIP — UNMEASURED

**Script:** `scripts/ci/dissertation_pbpk28_parity_gate.sh`  
**Why skip:**

```text
[pbpk28-parity] SKIP: node not available (gate requires Node ≥ 18)
```

Same class as frontend_parity. **Not a green.**  
Additional cases in the script can also SKIP if QSS/degenerate refs incomplete — not exercised in 9908 because node skip fires first.

### 2.5 `pbpk_hessian` — FAIL / FAIL — REAL DEFECT

**Script:** `scripts/ci/dissertation_pbpk_hessian_gate.sh`  
**9908:** PASS 2 / FAIL 3 both arms:

- `souc check` failed: `tests/run-pass/dissertation_pbpk14_hessian.sio`
- `souc compile` failed: same
- runtime CSV empty vs golden `benchmarks/pbpk/hessian_budget.csv` (diff shows golden lines deleted / no runtime rows)

Compile failure prevents golden comparison from being informative. **Not** “CSV off by ε — widen tolerance.”  
**FAIL_HONEST** until check/compile green; only then may residual numeric drift be debated with a published derivation.

### 2.6 `pbpk_suite` — FAIL / FAIL — REAL DEFECT (toolchain-heavy)

**Script:** `scripts/ci/dissertation_pbpk_suite_gate.sh`  
**9908:** FAIL 24/53 both arms (A~91s, B~84s).  
**Prior triage** (`docs/audit/DISSERTATION_PBPK_SUITE_TRIAGE_2026-08-16.md`, same job logs):

| Bucket (of 24 fails) | N |
|---|---:|
| resource ceiling (rc=182) | 2 |
| toolchain (SIGSEGV 139, E175, epistemic surface, corruption) | 22 |
| **genuine science / model defect** | **0** |
| awaiting data (PEND, not fail) | 1 semaglutide |

So: **gate is red for real reasons**, but the red is **compiler/API/capacity**, not “PK science wrong vs literature.” Defense narrative: toolchain must be fixed or the suite must be honestly scoped — **not** retoleranced to green.

---

## 3. Scoreboard that replaces “6/6”

| Category | Count | Gates |
|---|---:|---|
| **PASS** (measured green) | 1 | confidence_gate |
| **FAIL — real defect** | 3 | dossier, pbpk_hessian, pbpk_suite |
| **SKIP — unmeasured** | 2 | frontend_parity, pbpk28_parity |
| **Stale-instrument-only FAILS** | **0** | (no A=FAIL B=PASS) |

June **6/6** vs today:

```text
June claim:     P P P P P P
9908 reality:   P F S S F F
```

Even if both SKIPs later PASS, June 6/6 still requires dossier + hessian + suite to return to green.

---

## 4. Harness review (`diss-gates-ab.sbatch`)

Source reviewed: `/workspace/.wt/claude-1/.scratch/diss-gates-ab.sbatch` (ran as job 9908).  
Fixed copy on this lane: `scripts/ci/dissertation_six_gates_ab.sbatch`.

| Assumption | Verdict | Fix in lane copy |
|---|---|---|
| A/B design (checked-in vs from-source) | **Correct** and necessary | kept |
| `SOURCE_COMMIT` hard-pin `6f2c4e2461` | Correct for **reproducing 9904/9908**; wrong if used as “current main” forever | `DISS_AB_COMMIT` env override |
| SKIP via log grep | Correct intent; wording loose | tighter SKIP patterns + **why=** line; SKIP never counted as PASS in verdict |
| `dissertation_${g}_gate.sh` for confidence | Works only because name is `…_gate_gate.sh` | explicit map for `confidence_gate` |
| Suite default `SOUC_BIN=souc-seq-leansingle.sh` | **Hazard** for A/B (Seq lean_single path ≠ Madaros question) | force `SOUC_BIN=$REPO/bin/souc` per arm |
| Arm B copies into `artifacts/self-hosted/madaros` | Correct for `bin/souc` resolution | kept |
| Build failure → exit 0 with note | OK as “finding” | kept |
| Node optional | Must record ABSENT (done) | kept |

**Not fixed (needs grok-cli1 / ops):** Slurm launch held-requeue. Heavy re-run on **current** main tip blocked until launch works. **Not required to answer 9904-vs-June** — 9908 already answers for pin `6f2c4e2461`.

---

## 5. Cheap measurements this session (no full suite)

| Probe | Result |
|---|---|
| Worktree | `/workspace/.wt/diss-gates-ab-truth` @ current `origin/main` tip (not control) |
| `dissertation_confidence_gate_gate.sh` | **PASS** |
| `dissertation_dossier_gate.sh` | **FAIL** (check/compile smoke) |
| `souc check dossier_smoke.sio` | rc=1, unresolved `scripts::dissertation::dossier_generator` |
| Tolerances | **untouched** |

Heavy suite / full six-gate re-pin to tip: **held** pending Slurm repair (grok-cli1); coord message sent.

---

## 6. What would refute this conclusion

| Hypothesis | Refutation |
|---|---|
| FAILs are only stale `bin/souc` | A job where **B PASSes** and **A FAILs** on dossier/hessian/suite |
| SKIPs are really PASS | A run with Node ≥ 18 where parity gates emit `PARITY_PASS` / `PBPK28_PARITY_PASS` |
| Suite red is science | Any suite fail with honest clinical/GMFE fail after toolchain green — currently **0** such fails in triage |

None of those refutations have been observed on 9908.

---

## 7. Defense-facing implications (honest)

1. **Do not claim 6/6** on current main without a new full A/B receipt.  
2. **Contribution-2 confidence gate** still stands as measured green.  
3. **Dossier and Hessian** are broken at compile/check — fix imports/module path and hessian check surface **before** any numeric golden debate.  
4. **Parity gates** need a **Node-bearing** environment or a non-Node oracle; until then status is **UNMEASURED**, not green.  
5. **PBPK suite** red is largely **toolchain** (see suite triage); still a real gate FAIL for CI honesty.  
6. **Never** widen tolerances to absorb empty CSVs or missing PASS markers.

---

## 8. Receipts

```bash
# Job 9908 RESULT (A/B)
env SLURM_CONF=/tmp/slurm-direct.conf srun --partition=cpu-ops --nodes=1 \
  --ntasks=1 --time=00:05:00 --chdir=/tmp bash -lc \
  'cat /orangefs/training/diss-gates-ab-6f2c4e2461-20260816T124730Z-job9908/RESULT.txt'

# Fixed harness (this branch)
ls scripts/ci/dissertation_six_gates_ab.sbatch

# Cheap local (from clean worktree on main)
bash scripts/ci/dissertation_confidence_gate_gate.sh
bash scripts/ci/dissertation_dossier_gate.sh
```

Coord: info to grok-cli1/slurm-repair that 9908 already answers A/B; tip re-run optional after launch fix.

---


---

## 9. Tip re-run receipt (job 10117) + Node-bearing parity

### 9.1 Slurm tip A/B (compute node, no Node)

Launched via `scripts/dev/slurm_srun_minimal.sh` (grok-cli1 recipe; sbatch still broken
for this submitter). Confirms job 9908 pattern on **current main tip**:

```text
confidence_gate    PASS both
dossier            FAIL both  → REAL DEFECT
frontend_parity    SKIP both  why=node not available
pbpk28_parity      SKIP both  why=node not available
pbpk_hessian       FAIL both  → REAL DEFECT
pbpk_suite         FAIL both  → REAL DEFECT
```

### 9.2 Parity gates on the pod (Node v22 present)

Honest fix for SKIP: **run where Node exists**, with `SOUC_BIN` pinned to the
clean worktree (unpinned resolve can walk into control `/workspace/sounio` and
SEGV on a foreign madaros wrapper).

| Gate | Result | Detail |
|---|---|---|
| frontend_parity | **PASS** | `PARITY_PASS 14/14` within 1.0% RMSE; bit-identical peaks after SOUC pin |
| pbpk28_parity | **FAIL** | Case 1 organ-average **PASS** 14/14; mass/TMDD/PD **PASS**; **Case 7 FAIL**: `dissertation_pbpk28_parity_ref_semaglutide.sio` — E175 private `tmdd/glp1r::glp1r_rtotal0_at`, E008 Bergman return type vs Pbpk28 |

### 9.3 What a non-Node reference path would take

A parity gate that **cannot** run without Node on the runner is a permanent
unknown for pure Slurm/cpu-ops defense receipts. Options:

1. **Preferred short term:** keep Node as second clock, but **CI/Slurm image must
   install Node ≥ 18** (or ship a pinned Node binary under `tools/` used by the
   gate). Cost: image/ops change; no science change.
2. **Sounio-only dual path:** implement a second independent PBPK14/PBPK28
   trajectory in pure Sounio (different integrator or hand-unrolled CN) and
   compare Sounio↔Sounio. Cost: new reference modules + proof they are not
   copy-paste of the same bug; still need literature anchors.
3. **Analytical oracles:** extend case 2 (degenerate QSS) style closed forms to
   more organs — only covers reduced models, not full 28-state TMDD/PD.
4. **Frozen trajectory fixtures:** check-in CSV golden from a trusted Node run;
   gate becomes regression-only (loses cross-language live parity claim).

**Defence cannot use SKIP as green.** Frontend is now measured PASS with Node.
PBPK28 is measured FAIL until case 7 visibility/return-type defects close (or
case 7 is honestly scoped out of the gate with a published reason — not silent).

---
## Document control

| Date | Change |
|---|---|
| 2026-08-17 | Truth from job 9908 A/B + cheap rechecks; harness fixes; SKIP≠PASS; no tolerance changes. |
| 2026-08-17 | Tip A/B job 10117 (`dca2775061`); Node-pod frontend PASS; pbpk28 FAIL case 7; non-Node path options. |
