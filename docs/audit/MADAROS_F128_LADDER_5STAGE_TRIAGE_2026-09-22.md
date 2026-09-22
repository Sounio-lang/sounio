<!-- docs:meta
topic_id: repo.docs.audit.madaros-f128-ladder-5stage-triage-2026-09-22
authority: repo_only
audience: users
last_validated: 2026-09-22
validated_by: claude (fix/madaros-f128-f256-ladder-full-coverage-20260922 follow-up)
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.madaros-f128-ladder-5stage-triage-2026-09-22
-->

# Triage: the 5 f128/f256 ladder stages left out of PR #2615 (2026-09-22)

PR #2615 wired 15 of 20 `scripts/ci/madaros_f128_f256_ladder_gate.sh` stages
into `.github/workflows/ci.yml`'s `madaros-witness-gate` job. The remaining
5 (`v0c`, `v0e4`/`v0e41`, `v0e55`, `v0e57`) were left out because they were
measured red against a fresh build of that PR's tree, and deliberately not
guessed into "pass" or "must fix here." This doc records the follow-up
triage of all five.

## v0c — deferred milestone, not a regression

`v0c_codec_does_not_consume_external_corpus` fails because no consumer maps
`tests/vectors/f128_f256_v0c/wire_f{128,256}.jsonl` through the limb/wire
codec. Confirmed via `git log --all` on the three consumer-marker paths
(`self-hosted/compiler/f128_f256_v0c_wire_corpus_probe.sio`,
`tests/run-pass/f128_v0c_wire_corpus_smoke.sio`,
`scripts/dev/ws_g_v0c_codec_corpus_runner.py`): **zero hits across every
branch and commit** — they have never existed in this repo. The gate script
itself has asserted this failure since it was introduced
(`dd3b68ef9b`, #1775, 2026-08-17: "Gate MUST FAIL today until a real corpus
consumer exists"), and the gap is already documented in
`tests/vectors/f128_f256_v0c/V0C_GATE_CONSUMPTION.md`.

**Action taken:** added a `**Status (2026-09-22):**` line to
`docs/architecture/F128_F256_LADDER.md` §V0-C, matching the pattern already
used for §V0-B, so the deferral is visible in the design doc itself and not
only in the test-vectors README. `v0c` stays out of `ci.yml` — implementing
the corpus-consuming codec is a real, separately-scoped feature, not a bug
fix.

## v0e4 / v0e41 — stale negative-control fixture, not a lean_single regression

The fixture's negative control asserts lean_single language `f128`
greenwashes to f64: it builds an inline program computing
`(1e-20 + 1.0) - 1.0` with the seed compiler and expects the build to
succeed with a nonzero (non-greenwashed) result. It now fails at
`note_fail "lean_single_language_f128_build"` because the seed compiler
refuses to build the program at all.

Root cause: commit `8b99209e41` ("fix(lean_single): real binary128 for
annotated f128 (#2387) (#2426)", 2026-09-13) gave lean_single real binary128
lowering for annotated `f128` locals. As part of that same change,
`self-hosted/compiler/lean_single.sio`'s `f128_widen_refuses_inexact_literal`
refuses, at compile time, to widen a literal that is not exactly
representable in binary64 (`"f128 literal is not exactly representable in
binary64; lean_single would widen an f64 approximation"`) — matching the
same fail-closed philosophy Madaros's own lowering path enforces since
V0-E.5.9 (no f64 approximation is ever fabricated). `1e-20` is not exactly
representable in binary64, so it now trips this guard before the program
ever runs.

This is intentional, tested behavior, not a regression: the same commit
added `tests/compile-fail/f128_inexact_literal_refused_on_lean_single.sio`
as a positive fixture for exactly this refusal, and updated
`docs/architecture/F128_F256_LADDER.md` to note lean_single's `f128` path
"since #2387 ... is binary128." The v0e4 gate script predates that commit by
six days (`0e96a091f6`/`d7fa2f947d`, 2026-09-07) and was never updated for
the new lean_single behavior; `v0e41` only fails because its fallback path
re-checks v0e4's exact (now-stale) pass-label strings.

**Action taken:** updated the negative control in
`scripts/ci/madaros_f128_f256_v0e4_language_lower_gate.sh` to accept
lean_single's current, correct behavior — a build failure carrying the
exactness-refusal message — as the passing case (renamed
`lean_single_language_f128_f64_greenwash_refused` to
`lean_single_language_f128_inexact_literal_refused_no_greenwash`), while
still falling back to the original greenwash-detection path if the literal
is ever relaxed to build again, and still failing on any other unrelated
build error. Updated `scripts/ci/madaros_f128_f256_v0e41_fail_closed_lower_gate.sh`'s
fallback string match to accept either pass label.

## v0e55 — stale "must fail closed" fixture; KL-8 made scalar f128 `+=` a real, correct feature

`language_f128_compound_assign_fail_closed_regression rc=0`: the negative
control `acc += one` (both `f128`) is expected to be refused with the
V0-E.4.1 sentinel but now compiles and runs to completion.

Root cause: commit `86b9a08775` ("feat(madaros): KL-8 — f128 surface
residuals (#2491)", 2026-09-12) added a dedicated compound-assign path for
f128 slots in `self-hosted/ir/lower.sio::lower_assign_stmt_ref` (the
`target_is_f128 && s.assign_op != AssignEq` block): `+=`/`-=` on an f128
local now lowers through `lower_f128_value_ref` on both operands and
`lower_f128_call_soft_limbs` (softfloat add/sub), storing the result back
with `ir_copy`. This is deliberate, implemented support for f128 compound
assignment on scalars — not an accidental hole. The v0e55 gate's fixture
predates KL-8 (`bdba05ddbf`, #2457, 2026-09-08 — four days earlier) and
still asserts the pre-KL-8 refusal.

Verified against a fresh build of this tree
(`MADAROS_RAW_BIN=<fresh elf> bash scripts/ci/madaros_f128_f256_v0e55_language_struct_fields_gate.sh`):
`acc += one` compiles and computes the correct binary128 sum (no f64
greenwash — the softfloat add path is the same one used for `x + y`
elsewhere in this ladder). This is a real feature working correctly, not a
wrong-answer hole.

**Action taken:** updated
`scripts/ci/madaros_f128_f256_v0e55_language_struct_fields_gate.sh`'s
negative control to assert the new, correct behavior: the compound-assign
program is now expected to compile, run, and produce the correct softfloat
sum, with a regression check that it still fails closed on the one shape
KL-8 does *not* cover (an array-element compound target — see v0e57 below).

## v0e57 — one real bug (SIGSEGV) fixed; the third failure is a stale error-message match, not a bug

Two distinct problems were reported for this stage:

1. **`madaros_run_array_only_dce_trigger` and `madaros_run_language_f128_arrays`
   SIGSEGV (rc=139) — a genuine miscompile, now fixed.**

   Isolated to a minimal repro: any read of an element through a `&[f128; N]`
   *reference* parameter (`fn f(a: &[f128; 2]) -> f128 { a[0] }`) produced
   garbage, not the correct value. Depending on what that garbage happened to
   be, the program either SIGSEGV'd or silently returned the wrong answer
   (observed both, from the exact same compiled binary run under different
   conditions — confirmed by printing the "value" read through the reference
   next to the correct value read directly from the array: the reference path
   produced bytes that decoded as fragments of unrelated process strings, e.g.
   a scratch-directory path). By-value `[f128; N]` params were unaffected;
   `&[i64; N]` (and presumably every other element type) reference params were
   unaffected — the bug was specific to the combination of *reference* +
   *f128 element*.

   Root cause: `self-hosted/ir/lower.sio` has two parallel param-binding
   implementations ("path A", `lower_fn_params_mut`, and "path B",
   `lower_fn_params_ref`). Both compute whether a fixed-array param needs an
   element-wise slot copy (so a by-value `[f128; N]` param can't alias the
   caller's array) as:
   ```
   param_array_len = lower_type_expr_fixed_array_len_ref(ty)   // array length, or -1 if not an array
   param_word_scalar = lower_type_expr_fixed_array_word_scalar_ref(ty)
                     || lower_type_expr_is_array_of_f128(ty)
   if param_array_len >= 0 && param_word_scalar { <copy> }
   ```
   Both sites carried a comment asserting `&[f128; N]` can't reach this copy
   because reference types make `param_array_len` come back `-1`. That
   invariant was true when V0-E.5.7 (`6cfbf57206`, 2026-09-08) wrote it, but
   KL-14a (commit that changed `lower_type_expr_fixed_array_len_ref` to peel
   one `&`/`&!` so `&[i8; N]` FFI params get a known extent) silently broke
   it: `param_array_len` is the real length (2) for a reference param too,
   post-KL-14a. `lower_type_expr_is_array_of_f128` was already ref-aware (by
   design, for the `array_elem_wide_bits` marking that makes element reads
   work at all), so `param_word_scalar` was already true for `&[f128; N]`.
   With `param_array_len` no longer `-1`, the copy fired unconditionally for
   `&[f128; N]` params.
   The copy (`emit_fixed_array_value_copy`) reads from `preg`, the incoming
   parameter register. For a *value* param that register holds the array's
   own handle — correct to copy from. For a *reference* param, native-v2's
   calling convention makes `preg` hold the ADDRESS OF THE CALLER'S STACK
   SLOT that itself holds the handle (`&x` is "LEA of the stack slot", per
   the comment at `self-hosted/ir/lower.sio:5875`) — one level more indirect.
   Copying "elements" from that address treats an unrelated stack address as
   if it were a GC handle, explaining both failure modes: resolved via the
   handle table to unrelated memory (wrong answer) or rejected by the
   handle-table bounds check and used raw (SIGSEGV on dereference).
   `&[i64; N]` never hit this because `lower_type_expr_is_array_of_f128` is
   the *only* disjunct of `param_word_scalar` that is ref-aware — an i64
   element never sets `param_word_scalar` for a reference param, so the
   (also newly-nonzero) `param_array_len` never matters for it.

   **Fix:** both param-binding sites now exclude reference-typed params from
   the copy explicitly (`&& !lower_type_expr_is_ref_like(ty)`), restoring the
   original invariant without depending on `param_array_len`'s sign.
   Reference params still bind straight to `preg` and read through the
   existing `is_ref` / label routing in `lower_index_expr_ref` /
   `lower_field_expr_ref`, exactly as `&[i64; N]` already did correctly.

   **Regression coverage:** no new test file was needed — the existing
   `tests/run-pass/f128_v0e57_language_arrays.sio` smoke test already
   exercises `&[f128; N]` params (`sum3(a: &[f128; 3])`,
   `first_lt_second(a: &[f128; 3])`) and the gate script's inline
   `array_only.sio` probe is exactly the minimal repro; both are part of
   `scripts/ci/madaros_f128_f256_v0e57_language_arrays_gate.sh`, which is
   wired into CI by this change.
2. **`language_f128_compound_elem_assign_fail_closed_regression rc=1`.**
   `xs[0] += one` (an array-element compound target) still fails to compile
   (rc=1, correctly fail-closed) but no longer with the V0-E.4.1 sentinel
   text the gate greps for. Cause: `lower_expr_is_f128_slot_ref` (used to
   decide `target_is_f128`) already recognized `ExprIndex` targets as f128
   slots before KL-8 landed (V0-E.5.7, `6cfbf57206`), but the KL-8
   compound-assign store-back code only handles `ExprIdent` and
   `ExprFieldAccess` targets; an `ExprIndex` target falls through to
   `lower_f128_fail(lo_cmp, "f128 compound assign target shape unsupported")`.
   The program still fails closed — no ELF is ever produced — just with a
   more specific message than the gate's fixed string. This is a stale
   assertion in the test, not a correctness hole: no wrong array element is
   ever written.

**Action taken:** updated
`scripts/ci/madaros_f128_f256_v0e57_language_arrays_gate.sh`'s negative
control to accept either the V0-E.4.1 sentinel or the
`f128 compound assign target shape unsupported` message for the array-element
compound-assign case, since both are fail-closed refusals and the specific
wording is an implementation detail of which guard fires first.

## Net effect

All four investigations above concluded "the fixture (or the doc) was
stale, not the compiler" except item 1 of v0e57 (the SIGSEGVs), which was a
genuine bug with a genuine fix. After the fixture updates and the `lower.sio`
fix, `v0e4`, `v0e41`, `v0e55`, and `v0e57` were re-verified green against a
fresh build of this tree and wired into `.github/workflows/ci.yml`'s
`madaros-witness-gate` job alongside the 15 stages PR #2615 already wired,
all 19 stages verified sequentially green under `bash -eo pipefail` against
one fresh build, matching the verification discipline PR #2615 used.
`v0c` remains deliberately unwired pending the corpus-consuming codec.

Because the `lower.sio` fix touches param-binding code shared by every
fixed-array-typed parameter (not just `[f128; N]`), it was additionally
checked against `scripts/ci/madaros_full_gate.sh` and
`scripts/ci/madaros_source_to_elf_gate.sh` (both green) and the two
`tests/run-pass/ffi_system_*.sio` fixtures that exercise KL-14a's
`&[i8; N]` FFI path directly — the feature whose `param_array_len`
reference-peeling change created this bug's precondition — confirming that
path still works correctly (both green).

## Follow-up from review (2026-09-22, same day)

Automated review on PR #2615 caught two correctness gaps in the fixture
changes above and one CI-efficiency issue, all fixed the same day:

- The KL-15a `ci.yml` loop re-ran `--stage v0e510` even though the preceding
  KL-8 step (`madaros_kl8_f128_residuals_gate.sh`) already invokes it and
  requires its PASS marker — redundant work in a job capped at 30 minutes.
  Removed from the loop; KL-8 remains the sole runner for `v0e510`.
- v0e4's fallback path (reachable only if the literal-exactness refusal
  above ever regresses and the seed build succeeds again) had an inverted
  check inherited from the pre-#2387 test: it treated the program printing
  `FAIL f64_greenwash` — i.e. the compiler actually greenwashing — as
  `note_pass`. Fixed to require the program run to completion (`rc=0`) and
  print `PASS anti_f64`; a build that succeeds and greenwashes now correctly
  fails the gate instead of masking the regression.
- v0e55's new compound-assign positive check used `acc += one` (1.0+1.0=2.0),
  which is exact in both f64 and binary128 and so could not actually
  distinguish real softfloat arithmetic from an f64-widen-then-narrow cheat
  for this specific value. Replaced with the same anti-f64 `tiny` constant
  used elsewhere in this ladder (`f128_from_limbs(3746994889972252672,
  4592679628783035426)`); the expected result
  (`00002f3942192484:3fff000000000000`) cross-checks exactly against v0e4's
  own `wire_1+tiny` receipt for the identical `1 + tiny` computation.

All three fixes re-verified against the same fresh build; the full 18-stage
KL-15a loop (`v0e510` now excluded, covered once by KL-8) re-ran green
sequentially under `bash -eo pipefail`.
