<!-- docs:meta
topic_id: repo.docs.audit.seed-fix-two-level-indexed-aggregate-store-2026-07-05
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.seed-fix-two-level-indexed-aggregate-store-2026-07-05
-->

# SEED_FIX: two-level indexed aggregate store materialization — 2026-07-05

Status: **root cause of the two-level indexed RMW miscompile class, fixed at the
source** (lean_single bootstrap compiler), seed re-converged to a byte-identical
fixed point, canonical gate green.

This closes the durable follow-up named by
[`MADAROS_SEED_BEGIN_DISPATCH_2026-07-05.md`](MADAROS_SEED_BEGIN_DISPATCH_2026-07-05.md)
("the lean_single two-level indexed load/store codegen repair") — the defect class
behind the 2026-06/07 cascade: seed_begin segfault (June 18/22 family), merged
IrCall zeroing (`0ba18481a`'s workaround target), and roughly ten shipped
one-level-idiom band-aids in `module_frontend.sio` / `lower.sio`.

## Defect

`compile_indexed_field_field_array_store_x86` (lean_single.sio, handler for
`outer.arr1[i].arr2[j] = v`) computed the target address correctly but emitted a
**scalar-only terminal store**: it wrote the pointer of the transient RHS local
directly into the leaf array slot. Array-of-struct slots hold pointers to
heap-materialized copies (see `materialize_aggregate_array_element_x86`); every
other store handler in the family materializes a fresh heap copy first — only
this handler (and its a64 twin `compile_indexed_field_field_array_store_a64`)
omitted it. Result: the slot aliases reused local storage; the next loop
iteration overwrites it and previously stored elements read back as zeros —
exactly the all-zero IrInstr records observed in the merged-module passes.

## Fix

Additive, in the style of the prior seed fixes (`e42857a78`, `d633b1cb4`): in
both handlers, before the terminal store,

```
emit_load_var(slot_val)
materialize_aggregate_array_element_x86(leaf_hash)   // no-op for scalar leaves
emit_store_var(slot_val)
```

placed before the index reload because the heap-alloc syscall clobbers `rcx`.
For scalar leaves `materialize_*` returns without emitting (compile-time no-op).

## Witnesses (tests/known_failures/lean_*)

| Witness | old seed | fixed stage1 |
|---|---|---|
| `lean_two_level_indexed_rmw_aggregate.sio` (W_A — the cascade shape) | FAIL, 39/48 mismatches (only last-written elements survive) | **PASS** |
| `lean_two_level_indexed_rmw_scalar_control.sio` (W_B) | PASS | PASS |
| `lean_two_level_indexed_store_single_norease.sio` (W_D — aliasing nature: single store, local still live) | PASS "by luck" | PASS |
| `lean_field_array_array_aggregate_store.sio` (W_C — `field[i][j]` aggregate) | SIGSEGV | **still SIGSEGV — separate defect** |

W_C is a distinct manifestation in `compile_field_array_array_store_x86`
(likely an init/layout mismatch for `[[Inner;N];M]` aggregate 2-D fields, not
just missing materialization) — left in `known_failures/` as its own dispatch.
Witness promotion to run-pass is deferred until their behavior under the
Madaros (default) engine is classified.

No regression on the neighboring existing witnesses
(`aggregate_array_field_assignment_witness`, `nested_struct_field_copy_assignment_witness`,
`nested_ref_field_array_store` PASS, `abi_ptr_scalar_local_rmw_42` exit 42).

## Reseed

Fixed-point bounce from the old committed seed:
`old_seed → s1 → s2 → s3`, `cmp s2 s3` **byte-identical**
(md5 `a0316bce554909138e183b7c7428e3b5`); `scripts/ci/canonical_compiler_gate.sh`
**PASS** with the new seed on the fixed source. `bin/souc-lean-single-x86_64`
resynced to s2 (paired resync commit). Note: `scripts/ci/lean_single_fixed_point_gate.sh`
currently fails on a pre-existing harness issue (its stage1 step targets the
`bin/souc` wrapper, now Madaros-routed) — identical failure with the old seed;
not a regression.

## Downstream proof — COMPLETE (commit `c30b3af0a`)

Two Madaros builds from the resynced seed (`06409ecb9`), in a clean worktree,
provenance-guarded:

- **Matrix A** (workarounds in place): 8/8 green — thin exit 7, smt witness
  end-to-end, 2× smt ALL PASS, 4/4 dd64 ALL PASS.
- **Matrix B** (reversion proof: `module_frontend.sio` restored to its
  pre-`0ba18481a` state — the original in-place two-level RMW in
  `ir_module_finalize_merged_calls` / `ir_module_compact_duplicate_fn_refs`
  that used to zero every touched IrCall): **8/8 green, identical verdicts.**

The natural two-level indexed RMW shape is proven safe under the fixed seed;
the reversion is committed (`c30b3af0a`) as the standing witness. Remaining
one-level-idiom band-aids are now incrementally revertible under the same
proof protocol (revert in a scratch worktree → rebuild → matrix → commit):
lower.sio `:9617` skip_patch (re-enable `ir_patch_validated_calls` on the
imported lane — note its `&!(*module).functions[i]` call sites are a
*different* miscompile class, `&!`-of-boxed-element, and need their own
witness before reverting), `:743/:787/:1364/:4088/:4475/:4576` single-level
idioms, and `restore_user_main_calls` + defensive `ir_function_deep_copy` in
module_frontend.sio.
