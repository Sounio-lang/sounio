<!-- docs:meta
topic_id: repo.docs.audit.ci-absence-as-success-contract-2026-08-17
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.ci-absence-as-success-contract-2026-08-17
-->

# Absence-as-success — general form and Sounio instantiation

**Date:** 2026-08-17  
**Lane:** grok-cli5 / `ci-measurement-contract`  
**Related:** `docs/audit/VACUOUS_CI_GATE_CENSUS_2026-08-17.md` (N=6 empty-list greens),  
`scripts/ci/gate_vacuity_gate.sh` (static unguarded-extraction ratchet),  
`scripts/ci/gate_measurement_meta_gate.sh` (positive-measurement meta-gate).

---

## 0. Claim (plain)

**Absence of signal must never read as positive signal.**

A CI system that reports success for work it did not do is not “flaky green”; it is
a category error. Four faces confirmed live in this fleet in one day are **one
defect**:

| Face | What happened | How it reads |
|------|----------------|--------------|
| **Abort before measure** | Gate/stack aborts (e.g. exit 128 on missing `origin/main`) before evaluating product work | Ordinary **red** (looks like a real product fail) |
| **Empty input green** | Glob/list/dir empty → exit 0 (census **N=6**) | Ordinary **green** |
| **Suite absent, not red** | PR conflict / path missing → job skipped or matrix hole | **Not red** (looks like “nothing wrong”) |
| **Observer heuristic misfire** | A human or meta-check uses the wrong scope (check-count, job count, file count) and treats a correctly-scoped run as incomplete — or an incomplete run as complete | **False red or false green** about *whether measurement happened* |

Faces 1–3 are producer-side. Face 4 is consumer-side: the **observer’s criterion**
for “enough signal” is not the same as the gate’s work set \(W\), so absence or
presence of the *observer’s* preferred tokens is misread as (un)measured work.
All four share: **no distinguished unmeasured state tied to the actual work set.**

---

## 1. General form

### 1.1 Definitions

- **Work unit** — a finite set of checks the gate claims to perform (tests, greps,
  compiles, receipt fields, …).
- **Measurement** — an observation that at least one work unit was attempted and
  classified (pass/fail/skip-with-reason).
- **Outcome channel** — what CI surfaces: job exit code, check run conclusion,
  required-status aggregation.

### 1.2 Defect (success-without-work)

A reporting system exhibits **success-without-work (SWW)** when:

1. The **intended work set** \(W\) is empty, undefined, or never executed; and  
2. The **outcome channel** emits a value in the success class (exit 0, `success`,
   neutral skip that does not block merge); and  
3. There is **no orthogonal channel** that records “unmeasured” as distinct from
   “measured and passed.”

Formally: if \(M\) is the set of executed assertions and the UI maps
\(\mathrm{exit}=0 \Rightarrow \text{success}\), then SWW occurs whenever
\(\mathrm{exit}=0 \land |M|=0\).

### 1.3 Four faces (same equation)

| Face | \(W\) / \(M\) | Exit / UI | Why it fools |
|------|----------------|-----------|--------------|
| Die before measure | \(W\) never executed | ≠0 (red) | Looks like product failure; often **infra** |
| Empty input green | \(W=\emptyset\), loop runs zero times | 0 (green) | Looks like product pass |
| Absent suite | job not in the graph | missing check | Looks like “no news” |
| Observer heuristic misfire | True \(M\) fine or empty; observer counts the wrong thing | wrong colour | e.g. expects 16 checks, sees 4, calls run “incomplete” when scope was intentional — or sees many checks and misses empty \(W\) inside one green job |

The ambitious fix is not “more red.” It is **making unmeasured unrepresentable as
success**, binding success to a **receipt of \(|M|\ge 1\)** over the declared work
set (and, for face 4, binding observer expectations to that same receipt rather
than to ambient job counts).

### 1.4 What does *not* fix it

- More PASS banners without counts  
- Baselines that freeze “0 tests” as golden  
- Static linters alone (necessary, not sufficient — `gate_vacuity_gate` is the
  static half)  
- Hand-edited omega JSON that still says `pass` (fleet already burned)

---

## 2. Minimal contract that makes SWW impossible

**Measurement receipt contract (MRC-1):**

1. Every gate that may conclude **success** must emit exactly one machine line:

   ```text
   GATE_MEASURED schema=sounio.gate.measurement.v1 gate=<id> assertions=<N> status=pass|fail|skip [reason=<token>]
   ```

2. **`status=pass` ⇒ `assertions >= 1`.** Emitting pass with N=0 is a hard error
   inside the emitter (library refuses).

3. **`status=skip` with `assertions=0` ⇒ non-empty `reason=`.** Silent skip is
   forbidden (skip is not green-without-reason).

4. **A meta-gate** treats subject `exit=0` without a valid pass receipt as **fail**,
   regardless of banners like `ALL PASSED` or `0 fail`.

5. **Optional but strong:** merge required-check must include the meta-gate (or a
   workflow that parses receipts from all required jobs).

Under MRC-1, \(\mathrm{exit}=0 \land |M|=0\) cannot be produced by a conforming
gate: the emitter blocks it; the meta-gate blocks non-conforming greens.

### 2.1 Relation to the four faces

| Face | What forecloses it |
|------|---------------------|
| Empty input green | `require_min_count` / existence before success; **MRC-1** refuses `pass` with \(N=0\) |
| Abort before measure | Still red; receipt optional on fail. Wrappers should tag **infra vs product** (`reason=`) so red is not misread as measured product fail |
| Suite absent | Required-check set must include measurement meta-gate (or receipt aggregator); **missing required check ≠ merge success** |
| Observer heuristic misfire | Observers (humans, dashboards, meta-scripts) must key off **`GATE_MEASURED assertions=N`** (and declared `gate=`), not ambient “how many GitHub checks” or ad-hoc line counts. A correctly-scoped run with \(N\ge 1\) is complete; a green job with no receipt is not |

### 2.2 Why “assertions exercised” not “tests passed”

Pass count can be gamed (`0 fail`). **Assertions exercised** is a lower bound on
work attempted. A suite that proves it measured is categorically stronger than one
that only reports green. The author is not aware of a mainstream project that
**enforces** positive measurement receipts on success as a merge gate; this is the
novelty claim, scoped to the contract + meta-gate, not to “we fixed all 387 gates.”

---

## 3. Instantiation in this repository

| Piece | Path | Role |
|-------|------|------|
| Shared emitter/parser | `scripts/lib/gate_measurement_receipt.sh` | MRC-1 primitives |
| Shared emptiness helpers | `scripts/lib/gate_assert.sh` | `require_min_count`, etc. (prior art 2026-08-04) |
| Static ratchet | `scripts/ci/gate_vacuity_gate.sh` | Unguarded extraction debt; **now emits `GATE_MEASURED`** |
| **Meta-gate** | `scripts/ci/gate_measurement_meta_gate.sh` | Unit tests + pilot vacuous catch + enrolled live gate |
| Pilots | `scripts/ci/fixtures/gate_measurement_pilot_{ok,vacuous}.sh` | Specimen pass / specimen SWW |
| Empty-list census | `docs/audit/VACUOUS_CI_GATE_CENSUS_2026-08-17.md` | N=6 shapes + remedies |

### 3.1 Meta-gate behaviour (what it proves)

1. **A1–A5:** Library contract — pass-with-zero refused; valid receipt accepted;
   vacuous log rejected; silent skip refused.  
2. **B1–B2:** Pilot ok emits N≥1; pilot vacuous exit-0 is **caught**.  
3. **C1:** `gate_vacuity_gate` live run carries measurement (receipt or legacy
   `scanned N` during transition).  
4. **D:** Enrolled scripts reference the emit symbol.  
5. Emits its own `GATE_MEASURED assertions≥8 status=pass`.

### 3.2 Ratchet (ambition without false green)

- **Pilot enrollment** starts small (meta-gate + vacuity + fixtures).  
- Census N=6 gates are **debt**: they must not be “fixed” by adding a fake
  `assertions=1`; they need real `require_min_count` / existence checks, then
  enrollment.  
- Growing enrollment is the path to suite-wide “we measured.”

### 3.3 What this does not yet claim

- All of `scripts/ci/*_gate.sh` emit receipts (false).  
- Abort-before-measure is fully classified as infra (needs wrapper taxonomy).  
- GitHub required-check configuration is updated in this change (wire
  `gate_measurement_meta_gate.sh` next to `gate_vacuity_gate` in CI when ready).

---

## 4. Evidence commands

```bash
bash scripts/ci/gate_measurement_meta_gate.sh
bash scripts/ci/gate_vacuity_gate.sh | tee /tmp/vacuity.log
grep '^GATE_MEASURED ' /tmp/vacuity.log
# Specimen defect (must be rejected by the meta-gate's B2 logic):
bash scripts/ci/fixtures/gate_measurement_pilot_vacuous.sh; echo rc=$?
```

---

## 5. Bottom line

**General form:** SWW when exit∈success and executed assertions = 0 (or the job
never ran and absence is not blocking).

**Minimal contract:** MRC-1 — pass ⇒ positive `GATE_MEASURED` assertions; meta-gate
enforces it.

**Instantiation:** library + meta-gate + vacuity emit + census of the six empty-list
greens. A CI suite that can **prove it measured** is the SOTA++ step; green alone
is not.
