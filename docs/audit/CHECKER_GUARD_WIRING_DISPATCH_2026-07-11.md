<!-- docs:meta
topic_id: repo.docs.audit.checker-guard-wiring-dispatch-2026-07-11
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.checker-guard-wiring-dispatch-2026-07-11
-->

# Dispatch — wire the enforcement guards missing from the modular Madaros checker

**Filed:** 2026-07-11 · **Status:** OPEN (dispatch, not yet implemented) · **Protocol:** CLAUDE.md §8 (self-hosted/ changes require a forensic dispatch before code).

## Summary

The 2026-07-11 doc-reality audit (PRs #786/#788/#790) found that the default
compiler `bin/souc` (Madaros, self-hosted modular) is **more permissive than the
error catalog documents**: several `error[Exxxx]` guards do not fire under
`souc check`. Root cause is **not** a codegen bug — the guards are implemented in
the monolithic **`lean_single` seed** but were **never ported to the modular
checker** (`self-hosted/check/`). PR #790 documented this reality in the LLM
error catalog; this dispatch is the plan to actually close the gap in the
compiler.

## Evidence

- Madaros routes type-checking through the modular checker:
  `self-hosted/compiler/main.sio:16-25` imports `check::{dependent,types,compat,epistemic,effects,refinement,units,env,traits,mod}`; `run_check_mode` (`main.sio:1879`) calls `check_modules_verdict_boot4_with_visibility` and `module_frontend_check_items_with_source_context`.
- The guards exist in the seed but not the modular checker:
  - Tuple-destructure arity: `self-hosted/compiler/lean_single.sio:19825` (`tc_error(pp0, "tuple destructure arity mismatch")`). **0 occurrences in `self-hosted/check/`.**
  - Infinite recursive type: `self-hosted/compiler/lean_single.sio:27707` (`"infinite recursive type"`). **0 occurrences in `self-hosted/check/`.**
  - E208/E209 (refinement predicate): 0 occurrences in `self-hosted/check/` (predicate is not evaluated; `Pos`/`Prob` are treated nominally).
- Behavioural confirmation (default `bin/souc`, Madaros v0.80.0): `let (a,b) = <3-tuple>` → `check: OK`; `struct Node { next: Node }` → `check: OK`; missing `with IO`/`with Div` → `check: OK`. (The `lean_single` seed rejects several of these — the engines diverge.)

## Per-guard triage

| Code | What | Wire? | Risk / notes |
|---|---|---|---|
| **E213** | tuple-destructure arity (`let (a,b) = <3-tuple>`) | **YES — do first** | Genuine error; valid code has matching arity, so a clean corpus stays clean. Lowest risk. |
| **E216** | infinite recursive struct (`struct N { next: N }`) | Yes, with care | Must **exempt `Box<T>`/reference-wrapped recursion** (`struct N { next: Box<N> }` is valid). Verify that specifically. |
| E208/E209 | refinement predicate not evaluated | Medium feature | Compile-time predicate evaluation (`{v:i64 | v>0}`). `check/refinement.sio` + `check/dependent.sio` infra exists but the predicate is not run. Larger than a guard-port. |
| E040–E043 | Rust-compat (`let mut`/`&mut`/`#[...]`/`ident!()`) | Diagnostic only | These already fail (bare `parse error`); wiring = **better message** with the fix hint. Parser-side, not checker. Safe, low-value-per-effort. |
| **E035** | missing IO/Div/Observe effect | **NO — do not wire** | The permissiveness is almost certainly **intentional gradual/optional effects**. Enforcing would false-reject a large fraction of the ~6,000-file corpus. Requires an explicit **design decision from the maintainer**, not a bug fix. |

## Regression gate (blocks landing)

A checker change that ADDS rejections can false-reject valid code. **Do not land
unless the change produces zero new failures across everything that checks-clean
today.**

- **Do NOT** baseline with a naive per-file `souc check` sweep. It is unreliable:
  cross-module tests (`tests/run-pass/a13_crossmod_*`, etc.) fail when checked
  standalone because imports are unresolved — a 2026-07-11 sample showed ~28%
  spurious "failures" from this artefact alone.
- **DO** baseline with the real harness: `bash scripts/run_sio_test_suite.sh`
  (which supplies module/project context). Record the pass set on `main` before
  the change; after the change, the only new failures allowed are genuinely
  ill-typed programs.

## Implementation notes

1. **Port ≠ copy-paste.** `lean_single.sio` is the monolithic seed; the modular
   checker uses a flat-array arena + integer IDs (see the `self-hosted/check/*`
   file headers). Re-implement the check in the modular representation. The
   modular checker may lack the needed info at the pattern site — that is often
   *why* it was not ported; budget for threading it through.
2. **Insertion point (E213):** where `let`-destructure patterns are checked in
   the modular checker — start at `self-hosted/check/patterns.sio` and the
   tuple-type machinery in `self-hosted/check/check.sio`
   (`checker_lower_tuple_type_mut`, `ty_tuple`, `CheckTupleElemsResult`).
3. **Rebuild + verify:** `make build-madaros` under the pod-wide build lock
   (`/tmp/sounio-souc-build.lock`); then re-run `scripts/run_sio_test_suite.sh`
   and diff against the baseline pass set. Land E213 only if the diff is empty
   (modulo genuinely-ill-typed additions).
4. **Scope:** land **one guard per PR** (E213 first, E216 second). Do not batch.

## Recommendation

Land **E213** as the first verified guard once the harness baseline is clean.
Keep **E035 out** pending a maintainer design decision on whether effects are a
hard gate or a gradual discipline. Everything above is the evidence + plan; the
code change itself is the next dispatch step.
