<!-- docs:meta
topic_id: repo.docs.internal.coordination.state-of-the-compiler-2026-08-17
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.internal.coordination.state-of-the-compiler-2026-08-17
-->

# State of the Compiler — main @ `0b0c5cdd5b`

Written 2026-08-17 by `claude-2` (coordination bus lane `session-d502977e-a7be-46de-82ca-`),
synthesizing four independent, forked verification passes over the four things that landed on
`main` on 2026-08-16: WS-C PR1 (#1753), MLI S1 (#1754), P0-F (#1755), and the registry-provenance
fix (#1752, current `main` HEAD). Every claim below was checked against the tree — file contents,
live builds, live runs — not against the PR bodies. Where a PR body's claim could not be
independently confirmed, or turned out to be narrower than stated, that is called out explicitly
rather than folded into a clean summary. This document does not attempt to be exhaustive about
either PR; it answers the two questions the founder asked: what changed, and what of it is proven.

**Read order note:** the four sections are independent and can be read in any order. The "what
Madaros can do now" section at the top is the only place they interact.

---

## What Madaros can do today that it could not do yesterday

Exactly one of the four landings changes what a user running `bin/souc` can actually do:
**P0-F (#1755)**. Seven POSIX externs — `getpid`, `getppid`, `exit`, `abort`, `malloc`, `free`,
`system` — now genuinely execute under the *default* Madaros engine when called via `extern "C"`,
instead of silently returning a fabricated `0` while claiming success. This closes, for these
seven names specifically, the "Track A" gap left open in this session's own earlier work today
(`docs/audit/EXTERN_C_FFI_SILENT_NOOP_DISPATCH_2026-08-13.md`, which only fixed the legacy
`lean_single` engine and explicitly left the default engine unpatched). The fork verifying P0-F
did not take the PR's word for this: it built Madaros fresh from source, wrote its own probe file
(not the PR's fixtures), and got a real, distinct `getpid()`/`getppid()`, and a `system()` call
that genuinely created a file on disk and returned the correct shell exit status. That is a real,
checkable capability change, confirmed independently, today.

The other three landings do not change what `bin/souc` can do for a user yet:

- **WS-C PR1** adds a new, isolated `enir/` module tree that *typechecks* cleanly under default
  Madaros but *cannot be natively compiled* by it (SIGSEGV, already filed as a known limitation).
  Nothing outside a human running `bin/madaros-enir` by hand touches this code.
- **MLI S1** adds a new internal IR (kind model, builder, verifier) that is real and functionally
  correct for its own self-tests, but has zero callers anywhere else in the tree — no lowering
  path in, no codegen consumer out. It is an island.
- **The registry-provenance fix** changes how `docs:meta` headers survive an automated sync. This
  is real and load-bearing for anyone editing docs in this repo, but it is a documentation-tooling
  fix, not a compiler capability.

So: one real new user-facing capability (P0-F), two pieces of correct-but-unreachable internal
scaffolding for future work (WS-C PR1, MLI S1), and one infrastructure fix that stops a specific
class of silent data loss in the docs pipeline (#1752). None of the four, individually or
together, changes what `self-hosted/compiler/main.sio` — the actual production driver — does when
someone runs `bin/souc compile`.

---

## P0-F — execute allowlisted POSIX externs + execution gate (#1755, `1e8d48cdc8`)

**What landed:** 12 files, +496/-0. The allowlist itself is 12 new lines in
`self-hosted/check/check.sio` (~13844-13857); the emitters are +209 lines in
`self-hosted/native/codegen_x86_linux.sio`; the gate is `scripts/ci/ffi_posix_builtin_gate.sh`
(131 lines); the remaining files are 9 test fixtures under `tests/ffi_posix/`. `malloc`/`free`
reuse two pre-existing, lower-risk emitter ids (`heap_alloc`/`heap_free`, ids 23/24); the four
genuinely new emitters are `getpid`/`getppid`/`exit`/`abort`/`system` (ids 28-32).

**Proved, independently, today:**
- `getpid()`/`getppid()` return real, distinct, plausible PIDs — verified with a probe file the
  verifying fork wrote itself, against a Madaros built fresh from source (not the checked-in
  prebuilt ELF), not against the PR's own fixtures.
- `system()` genuinely forks and executes — verified via a real file created on disk and a
  correctly-decoded exit status (`5 << 8 = 1280`, `WEXITSTATUS` math checked).
- The gate script (`ffi_posix_builtin_gate.sh`), run live against `0b0c5cdd5b`, passes all 12
  checks: a structural mirror (checker allowlist vs. backend authority table), four `REFRAME`
  arms proving non-allowlisted names still fail closed with `E219`, and seven per-name execution
  witnesses.
- The gate design is stronger than "no longer E219 therefore correct" — and there's a concrete
  reason to believe that distinction was taken seriously while building it: the structural mirror
  check exists *because* an actual bug was caught during development of this same commit (checker
  allowlisted `free`, backend authority table registered `free_extern` — a name mismatch that
  would have silently reintroduced a fabricated-zero regression for exactly that name). It was
  caught and fixed within the commit, not after.

**Claimed but not independently verifiable:** the PR body's "proven by an executable Slurm
witness with a firing positive control" — no SLURM job logs or receipts for this specific work
exist anywhere in the tree. This is not evidence the claim is false; it's evidence the claim
can't be checked from what's committed. It matters less than it would otherwise, because the
verifying fork reproduced the substance of the claim itself, live, by an independent mechanism.

**What remains open:**
1. **The execution witnesses are not CI-enforced.** Only the *structural mirror* sub-check
   (`extern_builtin_mirror_gate.sh`) is wired into `.github/workflows/ci.yml` as a blocking step.
   The seven execution witnesses — the part that actually proves `system()` runs a real shell
   command rather than returning a plausible-looking `0` — must currently be invoked by hand. A
   future regression that reintroduces a fabricated-zero emitter *without* breaking the
   checker/backend name mirror would not be caught automatically. This is the same shape of gap
   this session's own earlier FFI dispatch work flagged for the legacy engine, now recurring one
   level up the stack.
2. **The two engines' `system()` calling conventions are not interchangeable and nothing stops
   you from mixing them.** Default Madaros wants `system(cmd: string) -> i64`; the legacy
   `lean_single` engine (this session's earlier Track B fix) wants `system(cmd: &[i8;N]) -> i32`.
   Code written for one signature type-checks under the other engine but silently misbehaves
   (the verifying fork hit this directly — an empty command reached the shell). Nothing in the
   tree currently warns a caller about this.
3. Only 5 of the 7 names were independently re-derived with a live side-effect test by the
   verifying fork (`getpid`, `getppid`, `system`); `malloc`/`free`/`exit`/`abort` were checked by
   reading the gate's own witnesses running green, not by an independent second test. Lower risk
   given they reuse existing or trivial emitters, but not to the same evidentiary standard as the
   other three.

---

## WS-C PR1 — ENIR/MIR shadow lane onto main, Route B (#1753, `8999e0fdff`)

**What landed:** 14 files under `self-hosted/enir/` (driver, hash, interpreter, ir, mir, mir_cfg,
mir_join, mod, parser, qd, shadow_fixture, source_lower, verify, canonical), plus
`bin/madaros-enir` (a cached-build wrapper around the legacy seed engine), a 1272-line design doc
(`docs/architecture/MADAROS_V2_EISA_SEMANTIC_IR.md`), and one audit dispatch. +8759/-4 across 20
files.

**Proved, independently, today:**
- The "dependency-isolated" claim holds exactly as stated: every `use enir::` in the tree is
  inside `self-hosted/enir/` itself; zero occurrences in `self-hosted/compiler/main.sio` or
  anywhere else in production code (checked by `git grep` across all of `origin/main`, not by
  reading the file list).
- The code genuinely typechecks under *default* Madaros, not just the legacy seed: `./bin/souc
  check self-hosted/enir/driver.sio` against a from-source build of `0b0c5cdd5b` reports
  `verdict=0` / `check: OK` across 13 modules.
- The seed-engine build path (`bin/madaros-enir`, which shells out through
  `scripts/dev/souc-build-lock.sh` to `lean_single`) also builds clean.
- Native *compilation* under default Madaros genuinely fails — SIGSEGV, exit 139, at
  `lower_array: dep_begin 2` (multi-module dependency lowering). This was independently
  reproduced, and it matches an already-filed dispatch doc
  (`MADAROS_ENIR_DRIVER_NATIVE_LOWER_139_DISPATCH_2026-08-16.md`) precisely — the PR is not
  hiding this, it filed the defect against itself in the same landing.

**What remains open:**
1. **The gate infrastructure the whole plan depends on does not exist in the tree.**
   `scripts/dev/madaros_v2_e1_enir_shadow_gate.sh` is referenced roughly 16 times across
   `WS_C_PR1_PAYLOAD_CENSUS.md`, `MIR_PORT_PLAN.md`, and the native-lower dispatch doc — and does
   not exist anywhere on `origin/main`. There is currently no automated check that this code keeps
   typechecking on future commits; today's clean `check: OK` is a one-time, by-hand verification,
   not a standing guarantee.
2. **49 files are explicitly deferred** to a second PR: 14 gate scripts, 13 Python verifiers, 22
   `tools/eisa` oracle/fixture files. This matches the PR's own payload census, independently
   cross-checked.
3. **The "stale-base instrument" explanation is unverified as history, though its outcome is
   verified.** The PR body attributes an earlier, larger defect estimate to having been measured
   against a WIP Madaros build (`9498c533a8`) rather than clean main, and asserts that base had a
   parser bug current main doesn't. No independent commit, bug report, or record of that specific
   parser defect could be found. What *is* independently true: `mir.sio` genuinely parses and
   typechecks clean under current main's Madaros right now. The outcome the explanation is used to
   justify is real; the historical narrative for why the earlier number was wrong is not
   corroborated by anything outside this PR's own commit message.
4. Nothing outside a human invoking `bin/madaros-enir` directly exercises this code today. "Shadow
   lane" is an accurate name — it casts no shadow on CI, because there is no CI pointed at it yet.

---

## MLI S1 — kind model, builder, dump, V-struct verifier (#1754, `453b2e6e2f`)

**What landed:** 6 new files under `self-hosted/mli/` — `ir.sio` (540 lines, kind model and
instruction pool), `builder.sio` (228 lines, fail-closed construction API), `dump.sio` (252
lines), `verify.sio` (654 lines, V-struct verifier), `self_test_runner.sio` (385 lines, the S1
self-test), `aggregate_store_diag.sio` (72 lines, a diagnostic) — plus a new dispatch doc and
governance-registry bookkeeping. 12 files, +2340/-4. The four deletions are entirely inside
`docs/governance/DOCS_ACCEPTANCE_REPORT.md` (registry regeneration); no source file was modified,
so "purely additive" holds for the part of the claim that matters (compiler source).

**Proved, independently, today:**
- `self_test_runner.sio` was run against current main and exits 0. It genuinely contains 15
  fixtures, counted directly rather than taken from the file's own summary line: cases spanning
  correct construction, deliberate cross-kind-move / non-`Bool` branch condition / `Int128`
  exclusion / QD128 misuse / missing-terminator / use-before-def failures (all of which are
  required to *fire*, i.e. these are negative controls, not just happy-path checks), plus
  `Knowledge<f64,Min3>` and `CD<8,f64>` (octonion) construction and verification, plus a
  capacity-overflow fail-closed case. Output ends `=== MLI S1: ALL FIXTURES PASS ===`, and it does
  today.
- `aggregate_store_diag.sio` was also run against current main and exits 0, printing `OBSERVED (3
  misaddressed cells)`. This is a real, currently-reproducing compiler defect (not a stale or
  vacuous witness) — a from-source Madaros build genuinely misaddresses scalar-field writes
  through a mutable reference into an array-of-struct element under this specific shape. The
  corresponding dispatch doc exists, is specific about the repro (392-byte / depth-3 aggregate
  array-element stores), and explicitly distinguishes itself from an adjacent, already-known bug
  family (`#1749`/`GLOBAL_VAR_ARRAY_INDEX_READS_ELEMENT0`) rather than being conflated with it.
- The kind model is functionally real, not a skeleton: no `TODO`/`unimplemented`/stub markers in
  `ir.sio`, `builder.sio`, or `verify.sio`; working constructors exist for every claimed kind,
  including `Knowledge<T>` and Cayley-Dickson (`CD<dim,coeff>`) as first-class operand kinds, and
  the self-test exercises both concretely rather than just declaring them.
- `docs/architecture/MLI_DESIGN.md` genuinely exists, is dated the same day, and its "Option C,
  amended D1-D4" language matches the commit message's own framing verbatim.

**What remains open:**
1. **Zero reachability.** A tree-wide grep for `mli::` or `use.*mli` outside `self-hosted/mli/`
   itself returns nothing. Nothing lowers into this IR and nothing consumes it — no codegen path,
   no gate wiring. This is disclosed by the PR itself (S2/S3 are described as future work fed by
   an "IR→MLI side door"), not a hidden gap.
2. Not wired into any CI gate — also self-disclosed, matching the commit message's own framing
   ("no gate wiring" is stated, not implied).
3. This is, today, correct internal machinery with no external effect. The honest read is: a
   real, verified building block for a future MLI-based pipeline, not yet a capability change for
   anyone running the compiler.

**Note on the PR body's calibration:** of the four landings, this one's commit message was found
to *understate* rather than overstate what's in the tree — every specific, checkable number in it
(15/15 fixtures, 10 of 15 being required-to-fire negative controls, 3 misaddressed cells) matched
exactly on independent re-run.

---

## Registry-provenance fix — R22/R23 inverted from indictments into guards (#1752, `0b0c5cdd5b`, current `main` HEAD)

**What landed:** 11 files, +1036/-450. The substantive fix is small: 12 lines in
`scripts/docs/sync_governance_metadata.mjs` and 76 in `scripts/docs/governance_registry.mjs`.
Previously, the mandatory docs-registry sync overwrote a document's entire `docs:meta` block with
freshly generated fields on every run, which for any document outside a curated owner table meant
a real `last_validated`/`validated_by` provenance record was silently regressed to a generic
placeholder (`last_validated: 2026-03-07`, `validated_by: A2` — visible, for instance, at the top
of every freshly-scaffolded doc in this repo, including this one) on every sync. The fix makes
`syncFrontmatter` parse and pass through the document's *own current* provenance fields instead of
overwriting them; four structural fields (`topic_id`, `authority`, `audience`, `source_of_truth`)
remain registry-authoritative and are still corrected as before. Provenance is now validated by
*form* (a well-formed ISO date, a non-empty validator string), not by *value* — a document with no
header at all, or a malformed one, still legitimately fails and gets the placeholder.

**R22/R23, what they actually are:** `docs/research/self_falsifying_compilation_line_r22_2026-07-29.md`
(and its paired R23 doc from the following day) documented the placeholder-stamping bug as a
formal indictment: `last_validated` was "a quoted literal at two sites in the generator, the same
value for every topic... the gate is green exactly when the field it guards carries no
information." Four named sub-indictments (`V1_VALUE_IS_A_LITERAL`, `V2_ONE_DATE_FOR_EVERY_DOC`,
`V3_DATE_PRECEDES_THE_REPO`, `V4_GATE_REJECTS_THE_TRUE_DATE`) were closed the same day as #1752 by
literal renaming into guard theorems (`V1_GENERATOR_PRESERVES_PROVENANCE`, etc.) in the same
research doc, which names #1752 explicitly as the closing commit.

**Proved, independently, today:**
- The fix's own selftest (`node scripts/docs/check_docs_registry_selftest.mjs`) was run live
  against a fresh worktree of `origin/main` and passes: 5 pre-existing failure fixtures plus a
  baseline plus 4 distinct provenance-preserve fixtures (preserve-header, headerless-stamped,
  malformed-provenance, impossible-date) — the fixture count was verified by reading the actual
  `cloneFixture` calls in the script, not by trusting its own printed summary.
- This selftest, plus the main registry checker, is wired into `.github/workflows/ci.yml` as a
  blocking CI step — confirmed by reading the workflow file directly.
- A specific, live discrepancy was chased down and resolved rather than left ambiguous: two files
  this session had itself edited earlier today (`docs/audit/EXTERN_C_FFI_SILENT_NOOP_DISPATCH_
  2026-08-13.md`, `docs/papers/main/cayley_dickson_hierarchy_paper_2026-08-13.md`) currently show
  exactly the placeholder-looking header on the working branch. This turned out **not** to be a
  live regression on `main`: neither file exists on `origin/main` at all (they are artifacts of a
  different local branch, `research/zd-fiber-antisymmetry-lemma-20260731`, whose HEAD is not a
  descendant of `0b0c5cdd5b`), and the placeholder stamp on that branch was applied by a commit
  roughly 22 hours *before* #1752 merged — i.e. this is the exact incident that motivated writing
  the fix, observed on a branch that has not yet pulled it, not evidence the fix fails on `main`.

**What remains open / minor discrepancy found:** the commit message claims the selftest coverage
went from "5 failure scenarios plus baseline" to "plus 3 provenance-preserve scenarios." The tree
actually has 4, not 3 — verified by direct fixture count, not by trusting the script's own report.
This is a real, checkable gap between the commit body and the tree, though it is in the safe
direction (more coverage than claimed, not less). R23's own document content was not independently
re-read in this pass (only R22's), so the claim that R23 is "the paired rung, same family" rests on
the closure doc's own framing rather than a second independent read.

---

## What this note does not cover

This synthesis is scoped to exactly the four items named. It does not attempt to assess the
broader Route B MIR port plan's viability, the MLI design document's long-term architecture, or
whether P0-F's allowlist should be extended further — those are open engineering questions the
four PRs themselves raise but do not settle, and are out of scope for a same-day state-of-the-
compiler snapshot.
