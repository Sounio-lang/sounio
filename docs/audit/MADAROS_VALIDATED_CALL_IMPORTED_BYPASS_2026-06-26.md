<!-- docs:meta
topic_id: repo.docs.audit.madaros-validated-call-imported-bypass-2026-06-26
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.madaros-validated-call-imported-bypass-2026-06-26
-->

# Madaros dispatch — `ir_patch_validated_calls` imported-path bypass is load-bearing (2026-06-26)

*Status:* TRIAGED, FIX DEFERRED. Recorded as a forensic dispatch per CLAUDE.md §8
(do not patch `self-hosted/` ad hoc; record evidence + proposed fix first).

## Summary

The imported-module body-lowering entries in `self-hosted/ir/lower.sio`
deliberately **skip** the validated-call post-pass `ir_patch_validated_calls`:

- `lower_program_items_bodies_from_summary_with_epistemic_boxed_ref` (~L8884) —
  sets `patch_stats = ir_empty_validated_patch_stats()` and only emits a
  `body_stage skip_patch_for_imported_probe` trace under the `!had_error` branch.
- `lower_program_bodies_from_summary_flat_with_epistemic_ref` (~L8912) — sets
  `patch_stats = ir_empty_validated_patch_stats()` with no patch call.

These are the live multimodule emit paths (driven from
`self-hosted/compiler/module_frontend.sio`: seed via
`module_frontend_lower_program_items_box_traced_with_externs`, deps via
`module_frontend_lower_program_items_box_traced`, and the flat path via
`load_multimodule_lower_program_traced`).

Earlier solver-lane notes implied this skip is gratuitous debt removable by a
trivial "mirror the sibling `ir_patch_validated_calls` call." **That is wrong.
The skip is load-bearing.**

## Evidence (controlled rebuild, 2026-06-26)

Applied the sibling-mirror fix to both entries:

```sio
if !lo2.had_error {
    patch_stats = ir_patch_validated_calls(&! lo2.module)
}
```

Source `./bin/souc check self-hosted/ir/lower.sio` → `check: OK`. Rebuilt madaros
via `scripts/ci/build_modular_madaros.sh artifacts/self-hosted/madaros`.

| Source state | madaros sha256 (16) | `test_smt` | `solver_novelty_readiness` |
|---|---|---|---|
| baseline (skip present) | `40f9abc2c697…` | 6/6 PASS | 1/1 PASS |
| + sibling-mirror fix | `7aaa058c0568…` | 0/6 (run exited 139) | 0/1 (139) |
| reverted | `40f9abc2c697…` (bit-identical) | 6/6 PASS | 1/1 PASS |

The reverted source reproduced the **bit-identical** baseline artifact
(fixed-point determinism holds), so the sibling-mirror edit is the *sole* cause
of the SIGSEGV regression across the entire imported-SMT runtime.

## Why a logical no-op still crashes

`ir_call_needs_validated_patch` (lower.sio ~L8385) fires only when the callee's
`compile_strategy == IR_STRATEGY_INSTRUMENTED` (3) — i.e. a `Contest<T>`/
`Robust<T>` return type — **and** the call site is off-by-one on args.
`stdlib/theorem/smt.sio` has zero `Contest`/`Robust` functions, so the predicate
never fires and **no instruction is actually patched**. The damage is structural,
not logical: `ir_patch_validated_calls` declares two `[i64; 2048]` arrays (32 KB)
and `ir_patch_validated_calls_in_function` takes them **by value** — i.e. a 32 KB
copy per function. Invoked deep in the recursive multimodule lowering stack (the
seed/dep traversal in `module_frontend`), under the vmem ulimit the madaros
launcher sets, this is enough stack pressure to corrupt where the single-file
patch sites (`main.sio` L5311/L5347) have headroom. The prior `test31`
139-regression note in `docs/research/solver-novelty-readiness-2026-06-25.md`
(local restore change in `lower_block_ref`) is consistent with this fragile-stack
class.

## Reachability note

The imported INSTRUMENTED-callee case the patch exists for is currently
**unreachable** anyway: cross-module `Contest`/`Robust` fails type-checking
*before* IR lowering. `./bin/souc check tests/multimodule/thin_contest_main.sio`
→ `E137 use of undeclared variable` (the lib's `models`/`policy` declarations do
not resolve across the imported-IR merge). So the skip masks neither a solver bug
nor any observable bug today.

## Proposed safe fix (not yet applied)

Do **not** re-introduce the by-value `[i64; 2048]` pass on the imported path.
Candidate approaches, in order of preference:

1. Move `callee_strategies` / `callee_param_counts` to heap allocation (or pass
   by `&`), so `ir_patch_validated_calls{,_in_function}` no longer place 32 KB on
   the stack per call. This also benefits the single-file sites.
2. Only invoke the patch on the imported path when the module actually contains
   an `IR_STRATEGY_INSTRUMENTED` function (cheap pre-scan), so the common case
   (no Contest/Robust) never pays the cost.
3. Defer until cross-module `Contest`/`Robust` type-checking lands (E137 above);
   until then the patch is dead on this path and the skip is correct.

Any fix must be validated by the same controlled rebuild: rebuild madaros, then
`test_smt` 6/6 + `solver_novelty_readiness` 1/1 + multimodule witness 5/5, plus a
multimodule fixture that genuinely exercises an imported INSTRUMENTED callee
(blocked on E137 today).

## Until fixed

Leave the skip in place. It is correct-by-necessity on the current imported
lowering path.
