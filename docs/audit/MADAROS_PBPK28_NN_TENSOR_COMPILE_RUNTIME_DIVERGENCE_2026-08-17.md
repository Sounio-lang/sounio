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
**Owner:** unassigned
**Status:** open, unpatched. Dispatch only — no `self-hosted/` files touched, per this repo's forensic
dispatch protocol (CLAUDE.md §8: "Do not patch `self-hosted/` ad hoc; record evidence and proposed
fix as a dispatch first.")

## Why this dispatch

PRs #1786 and #1809 audited Sounio's documentation for claims that state a fact about "the
compiler" when it's actually true of only one of Sounio's two engines (default Madaros vs the
`lean_single` bootstrap seed). While verifying dissertation `results/*.md` docs for that audit
(live-testing every cited harness under both engines, not trusting doc prose), three genuine
Madaros defects surfaced — not documentation gaps, compiler bugs. Each blocks part of the
dissertation's PBPK28/neural-network evidence surface from reproducing under the project's
default, user-facing engine. All three are independently reproduced below.

---

## Defect A — `read_f64`/`write_f64` are undeclared under Madaris; they are a lean_single-only special-cased builtin

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

### Proposed fix locus

Port `read_f64`/`write_f64` (the array-offset-accessor variant, matching lean_single's
`emit_read_f64`/`emit_write_f64`) into Madaros's builtin surface: add both names to
`name_is_native_backend_builtin` in `self-hosted/check/check.sio` and implement matching emitters
in `self-hosted/native/codegen_x86_linux.sio`, following the exact pattern P0-F (#1755) used for
the POSIX allowlist — each new name needs a working emitter or it silently returns a fabricated
value (E219 fail-closed only protects *undeclared* names; the E137 seen here is arguably the
*correct* current behavior given no builtin exists — the fix is adding the builtin, not loosening
the check).

### Acceptance gate (proposed)

1. `tests/stdlib/nn/test_pinn_training_d6.sio` and `tests/stdlib/nn/test_pinn_caputo_residual_d6.sio`
   both `check` and `run` clean under default Madaros, matching current lean_single behavior
   (`D6_PINN_TRAINING_LOOP_PASS`, `rc=0`).
2. A regression test alongside `tests/run-pass/ffi_integer_return.sio`'s pattern that exercises
   `read_f64`/`write_f64` directly (buffer array, in-bounds and out-of-bounds offset) under both
   engines.
3. `docs/dissertation/results/d6_pinn_training_v1.md`'s engine-dependency note (added in #1809)
   updated once this closes.

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
