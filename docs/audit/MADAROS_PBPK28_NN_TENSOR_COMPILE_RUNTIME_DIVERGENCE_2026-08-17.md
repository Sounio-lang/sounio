<!-- docs:meta
topic_id: repo.docs.audit.madaros-pbpk28-nn-tensor-compile-runtime-divergence-2026-08-17
authority: repo_only
audience: users
last_validated: 2026-08-17
validated_by: claude
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.madaros-pbpk28-nn-tensor-compile-runtime-divergence-2026-08-17
-->

# Three Madaros compiler bugs found auditing dissertation Madaros-vs-lean_single claims — dispatch

**Date:** 2026-08-17
**Toolchain:** `./bin/souc` → Madaros v0.80.0 (default engine); cross-checked against `SOUNIO_SOUC_ENGINE=lean_single`
**Owner:** claude (Defect A only, fixed 2026-08-17); Defects B and C unassigned
**Status:** Defect A **fixed and merged** (see its section for the commit and verification). Defects
B and C remain open, unpatched, dispatch only. Originally filed with no `self-hosted/` changes per
this repo's forensic dispatch protocol (CLAUDE.md §8); Defect A's fix followed as an explicit,
separately-authorized implementation pass on top of this same dispatch, not an ad hoc patch.

## Why this dispatch

PRs #1786 and #1809 audited Sounio's documentation for claims that state a fact about "the
compiler" when it's actually true of only one of Sounio's two engines (default Madaros vs the
`lean_single` bootstrap seed). While verifying dissertation `results/*.md` docs for that audit
(live-testing every cited harness under both engines, not trusting doc prose), three genuine
Madaros defects surfaced — not documentation gaps, compiler bugs. Each blocks part of the
dissertation's PBPK28/neural-network evidence surface from reproducing under the project's
default, user-facing engine. All three are independently reproduced below.

---

## Defect A — `read_f64`/`write_f64` are undeclared under Madaris; they are a lean_single-only special-cased builtin — **FIXED 2026-08-17**

### Repro

```
$ SOUNIO_STDLIB_PATH=stdlib ./bin/souc check tests/stdlib/nn/test_pinn_training_d6.sio
error[E137] in nn/optimizer::store_grad_read at 3872..3880: use of undeclared variable
  = name read_f64
error[E137] in nn/optimizer::store_grad_write at 4154..4163: use of undeclared variable
  = name write_f64
[... 40+ more E137 sites across stdlib/nn/optimizer.sio and stdlib/tensor/tape.sio ...]
error[E137] in tensor/tape::tape_grad_at at 40304..40312: use of undeclared variable
  = name read_f64
run_check_mode: verdict=1
run_check_mode: type checking failed across 8 module(s); the diagnostics above name the offending declarations
```

Every call site has the shape `read_f64(buffer_array, offset_i64) -> f64` /
`write_f64(buffer_array, offset_i64, value_f64)` — a raw flat-buffer memory accessor used
throughout the D.4 `ParameterStore`/Adam optimizer path (`stdlib/nn/optimizer.sio`) and the D.5
autograd tape (`stdlib/tensor/tape.sio`) to read/write parameter and gradient scalars by offset.
Neither file declares, imports, or `extern`s a function by this name — `grep -n "fn read_f64\|fn
write_f64" stdlib/nn/optimizer.sio stdlib/tensor/tape.sio` finds nothing.

Under `SOUNIO_SOUC_ENGINE=lean_single`, `tests/stdlib/nn/test_pinn_training_d6.sio` compiles
(tolerating several unrelated non-fatal warnings) and runs to completion:
`D6_PINN_TRAINING_LOOP_PASS`, `rc=0`.

### Root cause

`read_f64`/`write_f64` with this exact signature are a **lean_single-only special-cased
builtin**, not a real declared function anywhere in the source tree. `self-hosted/compiler/lean_single.sio`
recognizes the bare identifier textually and substitutes a hand-emitted memory access, independent
of normal name resolution:

```
self-hosted/compiler/lean_single.sio:13496:  if fn_find(ns, ne) < 0 && src_match(ns, ne - ns, "read_f64") {
self-hosted/compiler/lean_single.sio:13515:      emit_read_f64()
self-hosted/compiler/lean_single.sio:13546:  if fn_find(ns, ne) < 0 && src_match(ns, ne - ns, "write_f64") {
self-hosted/compiler/lean_single.sio:13561:      emit_write_f64()
```

(x86-64 emit sites at lines 9216/9231; a second, ARM64 pair `emit_read_f64_a64`/
`emit_write_f64_a64` exists at lines 31290/33683/33748 for the a64 codegen path.) This is the
same class of gap as `docs/audit/EXTERN_C_FFI_SILENT_NOOP_DISPATCH_2026-08-13.md`'s Track A/B
split (a lean_single-only stub-rewrite mechanism with no Madaros equivalent), and the same class
that P0-F (#1755) closed for `getpid`/`getppid`/`malloc`/`free`/`exit`/`abort`/`system` via
`name_is_native_backend_builtin` in `self-hosted/check/check.sio`. `read_f64`/`write_f64` are
**not** in that allowlist (confirmed: `grep -n '"read_f64"\|"write_f64"' self-hosted/check/check.sio
self-hosted/native/codegen_x86_linux.sio` finds nothing), and no equivalent name-match
special-case exists anywhere in Madaros's modular pipeline.

Two unrelated `fn read_f64(buf: [i8; 131072], pos: i64) -> (f64, i64)` / `fn write_f64(...)`
functions exist in `bootstrap/bootstrap_stage1.sio` and `self-hosted/ir/serialize.sio` — these are
a **different, incompatible signature** (byte-buffer serialization, not the offset-into-array
accessor `stdlib/nn/optimizer.sio` needs) and are not visible to `stdlib/nn/` regardless.

### Secondary, distinct defect found in the same module closure — borrow-checker strictness divergence

Checking the same file also surfaces 4 `error[E037]` sites in `stdlib/tensor/ops.sio`
(`sum_to_shape:160`, `tensor_sum:242,282`, `tensor_mean:299`), all the same shape:

```
tensor_set(&!out, dst, tensor_get(&out, dst) + tensor_get(x, i))
```

Madaros rejects this: `cannot borrow sharedly while exclusive borrow is active` — the `&!out`
exclusive borrow (first argument to `tensor_set`) and the `&out` shared borrow (nested inside the
second argument, computing the value to store) are ruled conflicting. lean_single accepts the
identical code. Not isolated further — flagged here because it was found in the same repro, not
because it's confirmed related to Defect A. Worth its own dispatch if pursued; the argument
evaluation-order/borrow-lifetime question (does the exclusive borrow's lifetime genuinely need to
span the second argument's evaluation, or is this an overly conservative Madaros borrow-checker
rule vs. lean_single's more permissive one?) is a real language-semantics question, not obviously
a "which engine is right" call.

### Fix (landed 2026-08-17)

Ported the full `read_i64`/`write_i64`/`read_f64`/`write_f64` family (builtin ids 33-36) into
Madaros: `name_is_read_i64`/`name_is_write_i64`/`name_is_read_f64`/`name_is_write_f64` plus two
new emitters (`emit_builtin_read_offset64_into`, `emit_builtin_write_offset64_into`) in
`self-hosted/native/codegen_x86_linux.sio`, wired at all 6 dispatch sites
(`native_v2_builtin_id_for_name_ref`, `native_v2_builtin_id_for_func_ref`, the third
id-lookup-by-name function, `native_v2_builtin_returns_float` (id 35 only), `native_v2_emit_builtin_by_id_into`,
and both name-based emit blocks), following the exact pattern P0-F (#1755) established. Added the
matching allowlist entries to `name_is_native_backend_builtin` in `self-hosted/check/check.sio`,
plus `write_i64`/`read_f64`/`write_f64` to `checker_collect_runtime_builtins_inplace` (`read_i64`
was already bound there — the sole entry present before this fix, which is why it alone never
E137'd).

Read and write share one emitter body each (`read_i64`/`read_f64` both compile to `mov
rax,[rdi+rsi*8]`; `write_i64`/`write_f64` both to `mov [rdi+rsi*8],rdx`) because Sounio's internal
call ABI passes every argument and return value as a raw 64-bit value through the general-purpose
registers regardless of Sounio-level type — confirmed against `emit_builtin_sqrt`, whose f64
argument arrives in `rdi` as bits and whose f64 result leaves in `rax` as bits, never touching
`xmm0` at the call boundary. They still need separate builtin ids: `read_f64`'s id had to be added
to `native_v2_builtin_returns_float` (drives `IR_FLOAT_REG_MARKER_FLAG` for correct downstream f64
arithmetic) while `read_i64`'s must not be, so a shared id across the two would have been wrong.

Also fixed, found while validating this fix against a real consumer test rather than only the
minimal repro: `heap_realloc` (used by `stdlib/data/bigframe.sio`, `stdlib/collections/heap_vec.sio`,
`stdlib/mem/box.sio`) was hitting the identical E137, but it is **not** part of this builtin
family — `stdlib/mem/box.sio` already has a real, portable implementation via `extern "C" realloc`.
It simply needed the same `checker_collect_runtime_builtins_inplace` registration `heap_alloc`/
`heap_free` already had. One-line fix, no codegen change, added to the same commit since it was
found validating this exact defect.

### Verification (built from source, off-pod on Slurm, `scripts/ci/build_modular_madaros.sh`)

- `scripts/ci/extern_builtin_mirror_gate.sh`: PASS, 38 builtin names, checker and backend agree.
- A minimal round-trip probe (`heap_alloc` an `f64`/`i64` buffer, `write_*`/`read_*` 10 elements
  each, compare): `DEFECT_A_ROUNDTRIP_PASS`, `rc=0`.
- `tests/stdlib/nn/test_pinn_training_d6.sio`: all 40+ `read_f64`/`write_f64` `error[E137]` sites
  gone. `check`/`run` still fail — but now *only* on the 4 `error[E037]` borrow-checker sites in
  `stdlib/tensor/ops.sio` documented above as a separate, unrelated defect. Not fixed by this
  change; that borrow-checker question is still open.
- `tests/stdlib/data/test_bigframe_ops_stdlib.sio`: `check` now passes cleanly (`verdict=0`,
  `check: OK`) — the `read_f64`/`write_f64`/`heap_realloc` E137s are gone. `run` still fails, on a
  **third, unrelated, pre-existing** limit: `error: function main needs 30937 IR instructions but
  IR_MAX_INSTRS is 16384` — a fixed per-function IR instruction cap this specific test's `main`
  exceeds, nothing to do with this fix. Not investigated further here.
- `self-hosted/compiler/main.sio` self-check: still passes (`self-check rc=0`) — the fix does not
  break Madaros's own self-hosting.

### Follow-up

`docs/dissertation/results/d6_pinn_training_v1.md`'s engine-dependency note (added in #1809)
should be revisited: the underlying `read_f64`/`write_f64` gap it described is closed, but the file
still does not check/run clean under Madaros because of the separate E037 borrow-checker defect —
the note needs re-wording, not removal.

---

## Defect B — `pbpk28_sobol_pce.sio` fails Madaros's multi-module check; the same imported function checks clean standalone

### Repro

```
$ SOUNIO_STDLIB_PATH=stdlib ./bin/souc check stdlib/darwin_pbpk/validation/pbpk28_sobol_pce.sio
run_check_mode: about to check 6 modules
error[E009] in validation/pbpk28_sobol_pce::sp28_selftest_main at 0..7271: argument type does not match parameter
  = expected fn#167
  = found fn#6
error[E009] in validation/pbpk28_sobol_pce::sp28_selftest_semaglutide_main at 0..17330: argument type does not match parameter
  = expected fn#167
  = found fn#11
error[E035] in darwin_pbpk/epistemic_pbpk28::main at 0..36190: effect not declared in function signature (missing: Epistemic) -- required by `ep28_selftest_main`
run_check_mode: verdict=1
```

The `E035` line is the striking part: it names `darwin_pbpk/epistemic_pbpk28::main`, a function in
a *different file* (`stdlib/darwin_pbpk/epistemic_pbpk28.sio`) than the one being checked. That
exact same file, checked **standalone**, passes cleanly:

```
$ SOUNIO_STDLIB_PATH=stdlib ./bin/souc check stdlib/darwin_pbpk/epistemic_pbpk28.sio
run_check_mode: about to check 3 modules
run_check_mode: verdict=0
check: OK
```

Under `SOUNIO_SOUC_ENGINE=lean_single`, `pbpk28_sobol_pce.sio` compiles (with two non-fatal
`tuple index out of bounds` warnings at `stdlib/epistemic/pce.sio:519-520`, tolerated per the
`lean_single` typecheck-tolerance behavior documented as #1494 in
`docs/compiler/KNOWN_LIMITATIONS.md`) and runs all 5 tests to `SOBOL_PCE_SEMAGLUTIDE_FULL_PASS`.

### Root cause — not isolated

Two independent-looking symptoms are bundled in one multi-module check run:

- The `E035` (missing `Epistemic` effect on `epistemic_pbpk28::main`) firing **only** when
  `epistemic_pbpk28.sio` is checked as an *imported dependency* of another module, never when
  checked standalone, points at a context-sensitivity bug in Madaros's multi-module effect
  inference — plausibly related to the "Imported-module native path" residuals already tracked in
  `docs/compiler/KNOWN_LIMITATIONS.md` §13 (D3 family: "multi-module memory-wall / exclusive-ref
  fragile chains"), though this dispatch did not confirm that specific connection.
- The two `E009` (`expected fn#167, found fn#6`/`fn#11`) errors are function-reference/pointer type
  mismatches at the call sites of `sp28_selftest_main`/`sp28_selftest_semaglutide_main` — consistent
  with a higher-order function argument (a callback passed by reference) being resolved to the
  wrong function-type slot when the module closure includes `epistemic_pbpk28.sio`'s functions.
  Not traced further; whether this is the *same* root cause as the E035 (one context-sensitivity
  bug producing two symptoms) or a second, independent bug was not determined.

### Proposed fix locus

`self-hosted/check/check.sio`'s multi-module effect-inference and function-reference-resolution
passes, specifically around how a module's own standalone-checked signature (effects, function
identity) is or isn't preserved when that module is re-checked as a dependency of a different
entry module. Needs a minimal repro isolated to 2 files (not `pbpk28_sobol_pce.sio`'s full 6-module
closure) before attempting a fix — this dispatch provides the symptom, not the minimal repro.

### Acceptance gate (proposed)

1. A 2-file minimal repro: module A declares a function requiring effect `X`; module B imports A
   and calls a *different* function in A that doesn't require `X`. Module A checks clean standalone
   and does not spuriously require `X` on unrelated functions when checked as B's dependency.
2. `pbpk28_sobol_pce.sio` checks and runs clean under default Madaros, matching lean_single
   (`SOBOL_PCE_SEMAGLUTIDE_FULL_PASS`).
3. `docs/dissertation/results/sobol_pce_semaglutide_v1.md`'s engine-dependency note (added in
   #1809) updated once this closes.

---

## Defect C — native-v2's fixed handle table is exhausted by large-N loops, aborting with `rc=182`

### Repro

```
$ SOUNIO_STDLIB_PATH=stdlib ./bin/souc run stdlib/darwin_pbpk/validation/pbpk28_mc_cross_validation.sio
Running MC lognormal (N=2000, seed=1729) ...
madaros: handles full
$ echo $?
182
```

Same symptom, same exit code, on the sibling harness:

```
$ SOUNIO_STDLIB_PATH=stdlib ./bin/souc run stdlib/darwin_pbpk/validation/pbpk28_mc_prior_family_sweep.sio
MC family 0: Gaussian(positive) N=2000 ...
madaros: handles full
$ echo $?
182
```

Both **compile clean** under Madaros (`check` passes, `verdict=0`) — the failure is purely at
runtime, partway through the N=2000 Monte Carlo loop. Both run to completion (`rc=0`, `PASS`,
matching gate markers) under `SOUNIO_SOUC_ENGINE=lean_single`.

### Root cause

`182` is not an arbitrary crash code — it's native-v2's own allocator-failure exit path,
`self-hosted/native/codegen_x86_linux.sio:7232`:

```
nc_core_emit_alloc_fail_into(nc, heap_slow_jnz, 181)     // heap exhausted
nc_core_emit_alloc_fail_into(nc, handle_slow_jnz, 182)   // handle table exhausted
```

Every native-v2 object allocation registers an entry in a fixed-capacity handle table
(`native_v2_handle_table_capacity_default()` in `self-hosted/native/gc.sio:64`, currently
`4194304` = 2²² entries, `native_v2_handle_entry_size()` = 48 bytes/entry). When a compiled
program allocates more live/tracked handles than that capacity over its run, the generated code
takes the `handle_slow_jnz` branch and the process exits 182. This is the same resource-ceiling
class already observed and named in `docs/internal/coordination/` context from the MLI S3 work
earlier the same day ("n=75 hit Madaros's resource ceiling, exit 182" — a different program,
same mechanism), but it is **not documented anywhere in `docs/compiler/KNOWN_LIMITATIONS.md`** as
a general constraint on loop/allocation scale.

Whether 2,000 iterations of this specific Monte Carlo loop *should* need anywhere near 4,194,304
handles is not established here — no per-iteration allocation count was measured. Two candidate
explanations, not distinguished: (a) each MC iteration genuinely allocates enough short-lived
tensor/struct objects that 2000 iterations plausibly exceeds 4M handles, in which case the fix is
either raising the default capacity or (better) making short-lived per-iteration allocations
handle-free/stack-allocated; or (b) handles from prior iterations are never released (a leak),
in which case even N=2000 shouldn't need anywhere close to 4M and the real bug is a missing
free/reuse path. This dispatch did not instrument the allocator to distinguish the two.

### Proposed fix locus

`self-hosted/native/gc.sio` (handle table capacity/lifecycle) and whatever governs
handle release for `stdlib/tensor/` / `stdlib/darwin_pbpk/` allocations in per-iteration MC/tensor
loops. Before any fix: instrument `runtime_context_field_handle_count()` (or add a debug print) to
measure actual handle-count growth per MC iteration on a small-N run, to distinguish leak from
genuine volume.

### Acceptance gate (proposed)

1. `pbpk28_mc_cross_validation.sio` and `pbpk28_mc_prior_family_sweep.sio` both run to completion
   (`rc=0`, matching gate markers) under default Madaros at N=2000, matching lean_single.
2. A regression test that runs a tight allocation loop past the current 4,194,304 handle
   capacity and asserts either a clean completion (if the real fix is handle reuse) or a
   documented, catchable resource-limit signal (if the real fix is a raised/configurable cap) —
   not a bare `rc=182` process exit with no diagnostic.
3. The six dissertation docs whose engine-dependency notes (added in #1809) cite this exact
   crash — `m6_prior_update_v1.md`, `mc_cross_validation_lognormal_v1/v2.md`,
   `mc_prior_family_sweep_v1/v2.md`, `prior_evolution_sprint_summary_v2.md`,
   `determinism_audit_summary_v1.md` — updated once this closes.

---

## Cross-reference

All three defects were found verifying dissertation `results/*.md` docs fixed in #1786/#1809; the
engine-dependency notes added there cite this dispatch doc's evidence but do not attempt to fix
the underlying compiler behavior — that's this document's job, left open per the forensic dispatch
protocol.
