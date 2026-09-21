<!-- docs:meta
topic_id: repo.docs.audit.known-failure-xpas-gate-dispatch-2026-08-18
authority: repo_only
audience: users
last_validated: 2026-08-18
validated_by: grok-cli4
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.known-failure-xpas-gate-dispatch-2026-08-18
-->

# Dispatch: known-failure XPAS must not count as Pass

**Date:** 2026-08-18  
**Lane:** grok-cli4 / xpass-gate-dispatch-20260818  
**Coordinates with:** grok-cli3 (`lane/grok-cli3/known-failure-xpas-20260818`, audit
`docs/audit/KNOWN_FAILURE_XPAS_SIGNAL_2026-08-18.md` — **uncommitted** on that worktree)  
**Status:** confirmed on `origin/main`; implementation already drafted by grok-cli3; this dispatch is the shared contract + policy call for the founder

---

## 1. Confirmation (do not take the brief on faith)

### 1.1 `scripts/ci/compiler_stage_contract_gate.sh` — XPASS fails the gate

Measured on `origin/main`:

```113:131:scripts/ci/compiler_stage_contract_gate.sh
known_blocker_cmd() {
  ...
  if [[ $rc -eq 0 ]]; then
    xpass_case "$name" "xpass" "known_blocker_resolved_update_manifest:$reason" "$log_path" "$@"
  elif grep -Eq "$pattern" "$log_path"; then
    known_case "$name" "known_blocker" "$reason" "$log_path" "$@"
  ...
}
```

```246:248:scripts/ci/compiler_stage_contract_gate.sh
if [[ "$FAIL_COUNT" -ne 0 || "$XPASS_COUNT" -ne 0 ]]; then
  echo "COMPILER_STAGE_CONTRACT_GATE_FAIL pass=$PASS_COUNT known_blocker=$KNOWN_COUNT fail=$FAIL_COUNT xpass=$XPASS_COUNT"
  exit 1
fi
```

**Verdict:** the house already has a gate that treats “known blocker now passes” as
**hard fail** with reason prefix `known_blocker_resolved_update_manifest`. The
founder’s intuition is correct for this file.

### 1.2 `scripts/dev/run_sio_test_suite_v2.sh` — emits `status=xpas`, then swallows it

On `origin/main`, when `exit_code == 0` and `//@ known-failure` is set:

```517:519:scripts/dev/run_sio_test_suite_v2.sh
        if $is_known_failure; then
            status="xpas"
            category="known-failure"
```

Aggregation on `origin/main`:

```709:713:scripts/dev/run_sio_test_suite_v2.sh
        xpas)
            ((PASS++))
            if [[ "$VERBOSE" == "1" ]]; then
                echo "  XPAS  $name (known failure now passes)"
            fi
```

Exit policy on `origin/main`:

```863:871:scripts/dev/run_sio_test_suite_v2.sh
if [[ $FAIL -gt 0 ]]; then
    ...
    exit 1
fi

echo ""
echo "All tests passed!"
```

**Verdict:**

| claim | on `origin/main` |
|---|---|
| suite emits `status=xpas` in per-test JSON | **yes** |
| suite prints XPAS by default | **no** (verbose only) |
| suite counts XPAS separate from Pass | **no** (`((PASS++))`) |
| suite / Full Test Suite fails the job on XPAS | **no** (only `FAIL > 0`) |
| any CI job on main fails on suite XPAS | **no** (grep: only `compiler_stage_contract_gate` fails on XPASS; Full Test Suite uses `scripts/run_sio_test_suite.sh` → v2) |

So: **the signal exists in the data; nobody acts on it in the main suite path.**
The brief is correct. The stage-contract pattern is the template; the suite is the hole.

### 1.3 What is *not* in the suite

`tests/known_failures/` (directory of whole files) is **not** scanned by the suite
globs. Those cannot silently XPAS via this path; they also cannot be cleaned by
it. Out of scope for this gate (named residual).

Suite-visible `//@ known-failure` count on this tip: **47** files under the same
globs grok-cli3 used (rg over run-pass / compile-fail / ui / stdlib / gpu).

---

## 2. Cost to wire (measured, not invented)

### 2.1 grok-cli3 receipts (authoritative for wall clock)

From `KNOWN_FAILURE_XPAS_SIGNAL_2026-08-18.md` (grok-cli3 lane, N = **47** suite-visible):

| engine | wall | XPAS | harness rc today |
|---|--:|--:|---|
| lean_single (Full Test Suite path) | **~5 s** | 8 | 0 — `All tests passed!` |
| default Madaros | **~61 s** | **22** | 0 — `All tests passed!` |

The **22** is Madaros XPAS: tags the repo still declares as failures that pass
under the default engine. That is the number in the brief.

The earlier “239 in &lt;16 s” figure in the fleet conversation is the #1890
*untag census* cost class (already inside suite cost once tags are suite-visible),
not a separate heavy job. Gate-sized either way.

### 2.2 Independent structural cost (this lane)

| item | value |
|---|---|
| Suite-visible `//@ known-failure` files | **47** |
| Full suite already *runs* them (not skipped) | yes — so XPAS detection is free on every Full Test Suite job once the aggregator stops swallowing |
| Extra CI job that only rechecks `requires: madaros` + `known-failure` | grok-cli3 script; order **~1 min** under Madaros if not already paid by f64 job |
| New compile of the compiler | **not required** for the harness change |

**Conclusion:** linking the signal is **CI-cheap**. The expensive part was the
one-time census (#1890). The mechanism replaces the next census.

---

## 3. Designed exit shape (mirror stage-contract)

### 3.1 Harness (every suite run)

When `status=xpas`:

1. **Do not** increment Pass.
2. Increment `XPAS` and always print:
   ```
   XPAS  <file> (known failure now passes)
   ```
3. Summary section (always, not only verbose):
   ```
   === Known-failure tags that passed in THIS run ===
       <file>
     A known-failure that passes is a stale claim about this engine,
     not a green test. Drop the //@ known-failure tag, or add
     //@ requires: <engine> if the claim is about a different engine.
     Madaros decides a Madaros-named tag; lean_single decides whether
     the file needs requires: madaros.
   ```
4. JUnit: `<failure message="stale known-failure: test now passes on this engine"/>`
   (so CI UIs that only read failures see it).
5. Refuse `All tests passed!` when `XPAS > 0`:
   ```
   Suite finished with N stale known-failure tag(s) (not a silent pass).
   ```
6. Exit code: see §4 policy.

Message prefix aligned with stage-contract vocabulary:

```
known_blocker_resolved_update_manifest:<file>:<engine>
```

### 3.2 Madaros recheck gate (compiler-only PRs)

Full Test Suite is lean_single and **skips** `//@ requires: madaros` files.
`madaros_changed_tests_gate` only runs requires:madaros files **in the PR diff**.
A compiler-only PR that heals a tagged failure never re-runs that file → tag rots
(#1890’s 240).

**Designed gate:** `scripts/ci/known_failure_madaros_recheck.sh`

- Select every suite-visible file with both `//@ known-failure` and
  `//@ requires: madaros`.
- Run via the harness with the **current-source Madaros ELF** and
  `SOUNIO_XPAS_FATAL=1`.
- Wire from `madaros_changed_tests_gate.sh` (end of job) so every f64 /
  current-source Madaros job pays it.

grok-cli3 already has this script + one-line wire in their worktree
(**not on `origin/main`**).

### 3.3 What “update the manifest” means here

There is no separate TSV manifest for suite known-failures — the annotation
**is** the manifest. Actions on XPAS:

| situation | action |
|---|---|
| passes on **both** engines | drop `//@ known-failure` |
| passes only on lean_single; still fails Madaros | keep tag; add `//@ requires: madaros` if missing |
| passes only on Madaros; tag names lean_single gap (e.g. f128) | keep tag; do **not** drop because Madaros XPAS |
| reason text wrong | rewrite `//@ known-failure: …` |

---

## 4. Policy decision for the founder (named, not decided here)

**Question:** should suite XPAS **block merge** (exit 1) or only **warn** (exit 0 + loud print)?

| option | honesty | irritation | failure mode if chosen |
|---|---|---|---|
| **Block** (`SOUNIO_XPAS_FATAL=1` always) | high | high until residual XPASses classified | red Full Suite until ~8 lean + classification of cross-engine tags |
| **Warn only** | low | low | backlog of stale tags returns; #1890-class census becomes routine again |
| **Phased (recommended)** | high after phase-in | bounded | see below |

### Recommendation: **phased block** (honest end-state, not permanent warn)

1. **Land immediately (block):** Madaros known-failure recheck with
   `SOUNIO_XPAS_FATAL=1` on every current-source Madaros job. Scope is the
   `requires: madaros` subset; heals the #1890 class without painting Full
   Suite red for seed-only f128 tags.
2. **Land immediately (announce, not fatal on Full Suite):** harness always
   prints XPAS, separate count, JUnit failure element, no `All tests passed!`.
   Default `SOUNIO_XPAS_FATAL` **off** on Full Test Suite for one short window
   while residual seed XPASses owned by other lanes are classified
   (grok-cli3 named: `gum_fo_across_call.sio`, `turbofish_concrete_type_mismatch.sio`
   among others).
3. **Then flip Full Suite to block** (`SOUNIO_XPAS_FATAL=1` in
   `.github/workflows/ci.yml` Full Test Suite env) once the remaining seed
   XPASses are dropped or correctly `requires:`-scoped.

**Why not permanent warn:** warn is how 240 tags sat green. The stage-contract
gate already chose block for the same logical event. Permanent warn re-accumulates.

**Why not hard-block Full Suite on day one without classification:** 8 lean_single
XPASses + cross-engine f128 tags would red every PR for reasons that are
*tag hygiene*, not the PR’s diff. That trains people to ignore the suite.
Phased block keeps irritation proportional to ownership.

**Founder chooses** between: (A) phased as above, (B) hard-block everything
now, (C) warn-only. This dispatch recommends **A**.

---

## 5. Coordination — grok-cli3 already built it

| artifact | where | on `origin/main`? |
|---|---|---|
| Measurement + cross-tab (22 Madaros XPAS, 8 lean) | `docs/audit/KNOWN_FAILURE_XPAS_SIGNAL_2026-08-18.md` | **no** (untracked on grok-cli3 worktree) |
| Harness XPAS count + announce + `SOUNIO_XPAS_FATAL` | dirty `scripts/dev/run_sio_test_suite_v2.sh` | **no** |
| `known_failure_madaros_recheck.sh` | untracked on grok-cli3 | **no** |
| Wire from `madaros_changed_tests_gate.sh` | dirty on grok-cli3 | **no** |
| Partial tag classification (drop / requires: madaros) | dirty test files on grok-cli3 | **no** |

**Do not re-implement.** Next action is for grok-cli3 (or a landing lane with
their consent) to **commit + open PR** from
`lane/grok-cli3/known-failure-xpas-20260818` carrying:

1. harness change  
2. recheck script + wire  
3. their audit doc  
4. the tag fixes they already measured as safe  
5. this dispatch (or a pointer) for the founder policy line  

grok-cli4 will not open a competing implementation PR. Bus message sent to
`grok-cli3` / `imported-139-untag` and broadcast.

---

## 6. Acceptance gate (when the PR lands)

| check | expect |
|---|---|
| Known-failure file forced to pass under harness | status `xpas`, printed, not in Pass count |
| `SOUNIO_XPAS_FATAL=1` + XPAS &gt; 0 | exit 1, message lists files |
| `SOUNIO_XPAS_FATAL` unset + XPAS &gt; 0 | exit 0, **not** “All tests passed!” |
| Madaros recheck on a synthetic stale tag | job red with `known_blocker_resolved_update_manifest` / `XPAS_FATAL` |
| `compiler_stage_contract_gate` | unchanged behaviour (already blocks XPASS) |
| Full Test Suite wall for known-failure-only subset | stays gate-sized (≤ ~1–2 min lean; Madaros recheck ≤ ~2 min class) |

---

## 7. Out of scope

- Raising global timeouts / CAP / handle table  
- Scanning `tests/known_failures/` directory into the suite (separate decision)  
- Auto-deleting tags in CI (human or lane updates the annotation)  
- Claiming the 22 are all drop-safe without the two-engine cross-tab (they are not;
  grok-cli3’s table is required)

---

## 8. One-line summary for the founder

> Main already *emits* XPAS and already *blocks* XPASS in the compiler-stage
> contract gate; the suite still counts XPAS as Pass and prints “All tests
> passed!”. grok-cli3 measured 22 Madaros + 8 lean stale tags and drafted the
> fix (uncommitted). Cost to wire is minutes, not a census. Recommend **phased
> block**: fatal on Madaros known-failure recheck now; announce everywhere;
> fatal on Full Suite after residual seed XPASses are classified.

*End of dispatch. No harness change in this commit — ownership of the land is grok-cli3’s draft.*
