<!-- docs:meta
topic_id: repo.docs.audit.gate-unmeasure-taxonomy-2026-08-17
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.gate-unmeasure-taxonomy-2026-08-17
-->

# Taxonomy of green-without-measure CI gates (2026-08-17)

**Lane:** grok-cli4 / gate-unmeasure-taxonomy  
**Claim:** A gate that exits 0 without having measured overstates coverage by one.
Today this repo has **hundreds** of gates (`~454` `scripts/ci/*_gate.sh`) and
**~69** wired by name into `.github/workflows/ci.yml`. Three distinct unmeasure
modes showed up in one day of fleet work. This document names the **complete
class** as far as measurement allows, gives a **live example of each**, a
**detector** that finds siblings, and the **one structural contract** that would
make the class impossible to wire into CI.

This is not a rewrite of `scripts/ci/gate_vacuity_gate.sh` (2026-08-04). That
gate is real and valuable: it ratchets **unguarded tool extraction** (empty
`grep`/`wc` → green). The taxonomy here is larger: **any path from “CI green”
to “no measurement occurred.”** Empty extraction is one species.

---

## Why this is a first (ambition, then evidence)

No mainstream project treats “green CI” as an epistemic object with a failure
taxonomy. Projects accumulate skip flags, soft-ors, and optional suites until
the dashboard is a **coverage fiction**. Sounio already has the raw material:
self-falsifying compilation lines, claim registries, and an existing vacuity
ratchet. Writing the taxonomy and proposing a **wire-time contract** is the
beyond-SOTA move: not inventing another gate, but making **unmeasure
impossible to launder as pass**.

Evidence discipline is unchanged: every family below has a command you can
re-run. Counts are from this worktree on 2026-08-17 against
`lane/grok-cli4/ws-f-visibility-20260817` / `origin/main` tip `c66014fda9`
lineage.

```bash
bash scripts/ci/gate_contract_probe.sh
# Measured 2026-08-17 (this worktree):
#   all_gates=454
#   wired_paths_in_workflows=73
#   unwired_gates=381
#   U1_skip_exit0=80 (wired∩U1=9)   e.g. self_falsifying_compilation_line_r1_gate.sh
#   U2_soft_or_workflow_lines=5      e.g. release-gate.yml LSP_SMOKE_WARN
#   U3_no_input_floor_heuristic=20
#   U7_partial_skip=5
#   U9_known_blocker=1
#   U10_lean_single_only=11
#   GATE_CONTRACT_v0_headers=0
#   wired_missing_contract_header=73
```

---

## The complete set (as measured)

A gate is **green-without-measure** when its process exits with a status that
CI treats as success **and** no positive measurement of the intended property
ran to completion. Subspecies differ by *how* measurement is avoided.

| ID | Name | Shape | CI appearance |
|---|---|---|---|
| **U0** | Empty extraction | Tool output empty → compared as success | Green pass string, zero cases |
| **U1** | Explicit SKIP→0 | `echo SKIP; exit 0` | Green job, skip line buried in log |
| **U2** | Soft-or absorption | `gate \|\| echo WARN` | Job green; failure becomes warning text |
| **U3** | Missing input silent | Required fixture/binary absent → skip or empty path | Green or “OK” without subject |
| **U4** | Abort-before-evaluate | Timeout/crash/setup fail before property check | Red that looks like a real fail *or* early green |
| **U5** | Suite ABSENT | Tests/gates never entered the merge tree (conflict/PR) | No red row; dashboard silent |
| **U6** | Unwired diagnostic | Gate exists, not in CI graph | “We have a gate” while CI never runs it |
| **U7** | Partial stack skip | Outer gate greens after skipping the only live arm | `STATUS … CLOSED` while L2-exec skipped |
| **U8** | No-work success | Empty change set / empty selection → exit 0 | Green “nothing to do” on PRs that needed work |
| **U9** | Known-blocker green | Documents defect then exits 0 | Green with `KNOWN_BLOCKER` in log |
| **U10** | Engine-split costume | Only lean_single (or only Madaros) measured; claim is “the compiler” | Green under wrong engine |
| **U11** | Ratchet-as-absolution | Baseline freezes unguarded debt; CI green while debt remains | Green “ratchet holds” ≠ “gates measure” |
| **U12** | Positive-control absent | No must-fail witness; broken checker still greens | Green until a human invents the control |
| **U13** | Defective completeness observer | Meta-heuristic treats check **count** as completeness; path-selected CI makes N=5 complete and N=16 possibly empty of property | Merge guard / human “16 checks = full CI” lies |

These thirteen cover every unmeasure mode found in-repo today. New modes should
be added only with a live example and a detector regex/probe.

### Fleet four (2026-08-17) — the confirmed operational set

The fleet discovered four **confirmed** ways a gate or CI surface reports something
other than what was measured. They map into the table; the fourth is the most
interesting because the **observer** is the defective instrument.

| # | Fleet finding | Taxonomy |
|---|---|---|
| 1 | Abort before evaluate (~27s); red looked ordinary | **U4** |
| 2 | Pass / silence on absent fixtures or empty input | **U0, U3, U12** |
| 3 | Suite ABSENT (PR conflict) rather than red | **U5** |
| 4 | Raw check-counting as completeness | **U13** |

**U13 detail (live, this repo):** `.github/workflows/ci.yml` job `Impact`
(`scripts/ci/classify_ci_impact.sh`) sets path booleans (`lean`, `math`,
`compiler`, …). Downstream jobs are gated with
`if: needs.impact.outputs.<facet> == 'true'`. Job `CI Decision`
(`scripts/ci/evaluate_ci_decision.py`) **only requires success for selected
jobs** — unselected jobs may be `skipped` and still yield
`CI_DECISION_PASS selected=…`. Therefore:

- A formal-only PR can finish with **~5 checks** and be **COMPLETE**.
- A run with **16 checks** can still measure the wrong facet or skip the
  property that mattered.

A merge guard that required “at least N GitHub checks” would **false-red**
honest lean-only PRs and **false-green** large but unmeasuring selections.
The founder correction while building that guard is the type specimen for U13:
**the completeness heuristic must be selection-relative and claim-relative,
never raw cardinality.**

---

## Live examples (this repository)

### U0 — Empty extraction
**Prior census (in-tree):** `scripts/ci/gate_vacuity_gate.sh` (2026-08-04)
lists measured instances: `run_pass_output_gate.sh` “all 0 tests”,
`check_doc_snippets.sh` “0 pass, 0 fail”, etc.

**Detector:** `bash scripts/ci/gate_vacuity_gate.sh` (already CI-wired).

### U1 — Explicit SKIP→0
**Live:** `scripts/ci/self_falsifying_compilation_line_r1_gate.sh` — when
`SFCL_R1_RUN_COMPILE` is unset (default), the compile arm prints
`compile arm: SKIPPED` and **`exit 0`**, while the gate is **listed in**
`.github/workflows/ci.yml` self-falsifying loop.

**Live:** `scripts/ci/kretikos_kaxi_phase_w_gate.sh` — missing `cc` or
`SOUNIO_KAXI_PHASE_W_GATE_SKIP=1` → `SKIPPED` + `exit 0`.

**Live count:** ≥75 `scripts/ci/*_gate.sh` combine `exit 0` with SKIP language
(scan 2026-08-17).

**Detector:** `scripts/ci/gate_contract_probe.sh` family `U1`.

### U2 — Soft-or absorption
**Live:** `.github/workflows/release-gate.yml`:
`bash scripts/ci/lsp_smoke_gate.sh || echo "LSP_SMOKE_WARN"` — gate failure
cannot fail the job.

**Detector:** `rg '\|\| echo' .github/workflows/*.yml`.

### U3 — Missing input silent / missing-fixture family
**Live (this lane, measured):** `madaros_visibility_context_gate.sh` on
`origin/main` never names `ambiguous_public_*.sio` (0 refs). Historical control
`cfdf1b7e0b` never merged. When fixtures are restored, the **positive control
fires** (compiler silent-bind: `check: OK`, `run_rc=101`). Main was not
“green on empty glob”; it was **U6+U12**. The missing-fixture *family* is still
real: any gate that globs fixtures and proceeds on zero matches.

**Live:** `scripts/ci/fo_residual4_stack_gate.sh` — if Madaros ELF missing,
prints WARN and **continues to** `FO_RESIDUAL4_STACK_GATE_OK` (also **U7**).

**Detector:** hard-coded `REQUIRED_FIXTURES` + non-empty file checks (see
patched visibility gate); probe family `U3`.

### U4 — Abort-before-evaluate
**Fleet report (2026-08-17):** a gate aborted in ~27s before evaluating its
property; the red looked ordinary. Structural shape: timeout/setup failure
mapped to the same exit class as property failure, with no
`MEASURE_PHASE=setup|property` receipt.

**In-repo cousins:** any `timeout N` where timeout is treated as generic fail
without distinguishing setup vs property (`stdlib_evolution_gate.sh` uses
`timeout 30` per file — timeout can look like “test failed”).

**Detector:** require phase-tagged receipts; fail CI summary if
`MEASURE_PHASE!=property` on a “red” attributed to the claim.

### U5 — Suite ABSENT (PR conflict / never landed)
**Fleet report (2026-08-17):** suite absent rather than red because the PR
conflicted — no failing row appears.

**In-repo cousin:** `cfdf1b7e0b` ambiguous-public fixtures + gate hook **not on
main** while later narrative assumed they were. Dashboard never showed red for
the missing control.

**Detector:** for every path cited by a wired gate, `test -f` in a
preflight job; PR check that `scripts/ci/X` and its `REQUIRED_FIXTURES` exist
on the merge tree.

### U6 — Unwired diagnostic
**Live:** `scripts/ci/madaros_visibility_context_gate.sh` + README:
“intentionally **not wired** into ordinary CI.” Exists as coverage language
(“we have a visibility gate”) while default CI never runs it.

**Live:** multiple `suffering_aware_*_gate.sh` headers: “intentionally NOT
wired into `.github/workflows/ci.yml`.”

**Detector:** `comm` between `git ls-files scripts/ci/*_gate.sh` and paths
referenced from workflows; emit `UNWIRED` inventory (not automatically red —
unwired can be honest if labeled).

### U7 — Partial stack skip
**Live:** `fo_residual4_stack_gate.sh` runs L0–L2 fragments, skips L2-executable
when Madaros missing, still prints `FO_RESIDUAL4_STACK_GATE_OK` and
`L2_FULL_ENGINE=OPEN` while earlier legs say CLOSED — easy to misread as full
stack green.

**Detector:** outer gate must AND a `MEASURED_ARMS=` bitset; CI fails if any
arm required by the gate’s claim header is `skipped`.

### U8 — No-work success
**Live:** `madaros_changed_tests_gate.sh` — empty selection →
`MADAROS_CHANGED_TESTS_SKIP reason=no_changed_requires_madaros_tests` +
`exit 0`. Correct for “no changed tests”; **unmeasure** when the PR was
supposed to carry Madaros tests that were never tagged/selected.

**Detector:** if PR labels/paths match `self-hosted/**` and selection is empty,
exit non-zero or emit `SKIP_CLASS=no_work` that CI policy treats as yellow.

### U9 — Known-blocker green
**Live:** visibility gate baseline path ends in
`KNOWN_BLOCKER: name-only lookup…` with exit 0 under `EXPECT=classify`.
Honest as classification; **unmeasure** if CI treats that exit as “visibility
OK.”

**Detector:** ban `exit 0` after `KNOWN_BLOCKER` unless
`GATE_RESULT=classified_blocker` is a first-class CI outcome (not “success”).

### U10 — Engine-split costume
**Live (this lane):** lean_single-only `eisa_bridge_conformance_gate.sh` was
the only EISA bridge gate; Madaros never measured. WS-F Madaros variant
exposes E137 / stack / emitter gaps. June dissertation gates “6/6 green”
re-measured split failures under current engines.

**Detector:** gates that claim Madaros must invoke default `bin/souc` without
`SOUNIO_SOUC_ENGINE=lean_single`, or run **both** and report a matrix.

### U11 — Ratchet-as-absolution
**Live:** `gate_vacuity_gate.sh` itself — baseline freezes hundreds of
unguarded gates; CI is green while they remain unmeasuring. Necessary debt
tool; still **not** “all gates measure.”

**Detector:** report `flagged_n` as a coverage debt metric next to green; never
market “vacuity gate green” as “gates are non-vacuous.”

### U12 — Positive-control absent
**Live (this lane):** ambiguous-public control **fires** when present
(`check: OK` + `run_rc=101`). Without it, a “resolved” visibility story can
green while global fallback still fabricates a binding.

**Detector:** every wired gate declares `POSITIVE_CONTROL=` path that must
**fail** the compiler/tool under test; meta-gate runs controls and requires
non-zero.

### U13 — Defective completeness observer (check-count fallacy)
**Live:** `Impact` + `CI Decision` in `.github/workflows/ci.yml`:

```text
impact → classify_ci_impact.sh → outputs.{lean,math,compiler,…}
jobs.* if: needs.impact.outputs.<facet>
ci-decision → evaluate_ci_decision.py
  required[job] = f(impact outputs)
  pass iff every selected job succeeded
  print CI_DECISION_PASS selected=<comma-list>
```

Raw `N_checks` is not a valid completeness signal. Completeness is
`selected ⊇ claims(PR)` and each selected job `measured≥1` for its claim.

**Detector:** merge guards must parse `CI_DECISION_PASS selected=` (or Impact
outputs) and match against the PR’s claimed facets — never `checks.total >= K`.
Probe family `U13` documents the pattern; enforcement belongs in the guard, not
in counting check-runs.

---

## Detectors (runnable)

| Family | Command |
|---|---|
| U0 | `bash scripts/ci/gate_vacuity_gate.sh` |
| U1–U13 inventory | `bash scripts/ci/gate_contract_probe.sh` |
| U2 | `rg -n '\|\| echo' .github/workflows` |
| U6 | probe prints unwired set |
| U13 | `rg -n 'CI Decision|evaluate_ci_decision|impact.outputs' .github/workflows/ci.yml`; never use check-run cardinality as completeness |
| Visibility control | see `docs/audit/MADAROS_VISIBILITY_CONTEXT_GATE_VACUITY_2026-08-17.md` |

The probe is **observational by default** (exit 0 with a census). Set
`GATE_CONTRACT_PROBE_FAIL=1` to fail if any **wired** gate is in a hard
unmeasure class (U1 skip-green without `SKIP_CLASS=`, U2 soft-or, U3 missing
required path pattern without `require_fixture`/`require_min_count`).

---

## The structural changes: wire-time contract + assertion meta-gate

Two layers. Both are required. Neither is “count GitHub checks.”

### A. Gate Contract v0 (wire-time admission)

Nothing may be added to required CI unless it satisfies **Gate Contract v0**:

1. **Identity**  
   ```text
   # GATE_CONTRACT: v0
   # GATE_ID: <stable-id>
   # GATE_CLAIMS: <one line property>
   # GATE_ENGINE: madaros|lean_single|both|host|python|gpu|n/a
   # GATE_RESULT_ON_SKIP: fail|classify|forbidden
   ```

2. **Required inputs** — floor ≥ 1; empty discovery ⇒ exit 2 (`NO_INPUT`), never 0.

3. **Measurement receipt with positive assertion count** (this is the meta-gate’s fuel):
   ```text
   GATE_RECEIPT id=… result=pass|fail|classify measured=1 inputs=N assertions=K
   ```
   **`assertions` must be an integer ≥ 1** for `result=pass` or `result=fail`.
   Zero assertions ⇒ exit 2 (`NO_ASSERTION`), even if the process “succeeded.”

4. **Positive control** — inverted property must fail the SUT.

5. **No soft-or** on required workflow steps.

6. **Engine honesty** — no lean_single costume for Madaros claims.

7. **Phase tags** — setup abort ≠ property fail.

### B. Meta-gate: prove the suite measured (SOTA++ enforcement)

**Name:** `scripts/ci/gate_assertion_meta_gate.sh` (proposed; not yet landed as enforced CI).

**Property:** every gate that ran in this CI invocation emitted
`assertions=K` with **K ≥ 1**, and the **sum of K over selected jobs is ≥ 1**
for each claim facet the PR asserts.

**Mechanism (minimal):**

1. Each gate writes one line to `$GITHUB_STEP_SUMMARY` or
   `artifacts/ci/receipts/<GATE_ID>.receipt`:
   ```text
   GATE_RECEIPT id=eisa_bridge_madaros result=pass measured=1 inputs=31 assertions=47
   ```
2. Meta-gate after `CI Decision` (or as its body):
   - Parse all receipts from the selected job set (from
     `CI_DECISION_PASS selected=` + downloaded artifacts).
   - **Fail** if any selected gate lacks a receipt.
   - **Fail** if any receipt has `assertions=0` or missing `assertions`.
   - **Fail** if `sum(assertions) < floor` for the Impact facets in play
     (facet floors are small positive integers, not “16 checks”).
3. Merge guards consume **the same receipts**, not check-run counts — killing **U13**.

**Why this is categorically stronger than green CI:**

| Green CI today | Meta-gated CI |
|---|---|
| Process exited 0 | Process exited 0 **and** exercised K≥1 assertions |
| N checks can mean path-selected subset | Completeness = selected claims measured |
| Skip/soft-or/empty look like pass | No receipt / assertions=0 → red |
| Observer counts check-runs | Observer sums **assertions** |

No mainstream project known to this audit enforces “every required gate must
prove a positive assertion count.” That is the beyond-SOTA instrument:
**measurement as a first-class CI outcome.**

### Why wire-time + meta-gate beats 454 local fixes

- Unwired research gates may stay (honest **U6**).
- Wired gates cannot be U0/U1/U2/U3/U8 silent greens (**contract**).
- Completeness cannot be faked by check cardinality (**U13** + meta-gate).
- Green CI count becomes count of **measured assertion units**, not scripts.

### Rollout (evidence-preserving)

| Phase | Action |
|---|---|
| 0 | Land taxonomy + observational probe (this doc + `gate_contract_probe.sh`) |
| 1 | Annotate new gates with `GATE_RECEIPT` + `assertions=K`; reject new wired gates without it |
| 2 | Convert top wired offenders (SFCL r1, residual4, changed-tests, release soft-ors) |
| 3 | Land `gate_assertion_meta_gate.sh`; wire after CI Decision; `assertions` floor |
| 4 | Merge guards switch from check-count to receipt/assertion sum |
| 5 | `GATE_CONTRACT_PROBE_FAIL=1`; shrink `gate_vacuity` baseline |

---

## Relation to the four fleet findings

| Fleet finding | Taxonomy ID |
|---|---|
| Abort before evaluate; red looked ordinary | **U4** |
| Suite ABSENT (PR conflict) rather than red | **U5** |
| Green on nothing / missing fixtures / empty input | **U0, U3, U12** (visibility was U6+U12 on main) |
| Raw check-counting as completeness (Impact + CI Decision) | **U13** |

**Positive control on `madaros_visibility_context_gate.sh`:** **fired**
(`check: OK`, `run_rc=101`). That proves U12 is live when the control exists,
and that main’s gap was unwired/missing control (U6/U5), not a successful
empty-glob green of a wired ambiguous_public path.

---

## File presence

```text
docs/audit/GATE_UNMEASURE_TAXONOMY_2026-08-17.md
scripts/ci/gate_contract_probe.sh
```

## Non-claims

- Does not claim all 454 gates are false.
- Does not disable existing gates.
- Does not replace `gate_vacuity_gate.sh`; it **embeds** it as U0.
- Enforcement (`GATE_CONTRACT_PROBE_FAIL=1`) is opt-in until Phase 3.

---

*Last measured 2026-08-17. Re-run detectors before quoting counts in a paper or defense slide.*
