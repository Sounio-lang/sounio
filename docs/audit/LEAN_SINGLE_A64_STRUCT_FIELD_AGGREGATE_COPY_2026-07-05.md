<!-- docs:meta
topic_id: repo.docs.audit.lean-single-a64-struct-field-aggregate-copy-2026-07-05
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.lean-single-a64-struct-field-aggregate-copy-2026-07-05
-->

# lean_single forensic dispatch — aarch64 struct-literal aggregate field copy writes to the wrong slot range

Date: 2026-07-05
Branch: `main` (post-PR #635, issue #633)
Class: **aarch64-only codegen addressing bug** — the Release Gate's "Apple
Self-Host" native runtime acceptance job has failed on every run for 3+
weeks; this closes the specific failure current at dispatch time
(`abi_return_literal_array_field_42`) and, as a side effect, one more
(`bdf_stiff`)
Status: root-caused, fixed, verified — zero regressions on both the aarch64
native-runtime manifest (95/99 pass, up from 93/99, the 4 remaining failures
confirmed pre-existing and unrelated) and the full x86-64 suite (1314/0/124/689,
exact baseline match, since the fix is aarch64-only)

## Symptom

CI's Release Gate (`Apple Self-Host` job, `scripts/selfhost/
selfhost_native_acceptance_gate.sh` → `selfhost_native_runtime_proof.sh`)
segfaults compiling+running `tests/selfhost/native_runtime/
abi_return_literal_array_field_42.sio` on the source-bootstrapped aarch64
native compiler:

```
FAIL [abi_return_literal_array_field_42] unexpected exit (got=139 expected=42)
Segmentation fault: 11
```

The test constructs two structs, each containing an `[f64; 8]` array field
sandwiched between scalar fields, copying one struct's array field into
another via a struct literal (`Result { ..., vals: src.vals, ... }`), across
a function-return boundary.

## Reproduction (local, no macOS hardware needed)

The bug reproduces identically by cross-compiling to the `aarch64-linux`
target (a Linux ELF, not the CI's Darwin Mach-O, but the AArch64
instruction-level bug is OS-agnostic) and running under `qemu-aarch64-static`
(`apt-get install qemu-user-static`) — this cut the debug loop from
"push, wait for macOS CI" to seconds, locally:

```sio
struct Source { head: f64, vals: [f64; 8], tail: i64 }

fn main() -> i32 with IO, Mut, Panic, Div {
    var vals: [f64; 8] = [0.0; 8]
    vals[0] = 11.0
    vals[1] = 31.0
    let src = Source { head: 3.0, vals: vals, tail: 7 }
    println(src.vals[0])  // pre-fix: 0.000000 (should be 11.0)
    println(src.vals[1])  // pre-fix: 0.000000 (should be 31.0)
    0
}
```

No function-return boundary is even needed — a bare struct literal with an
array field copied from a local array variable already loses the data. The
CI test's extra severity (SIGSEGV rather than silently-wrong values) comes
from its larger, doubly-nested struct shape writing far enough outside the
intended stack range to hit an unmapped page.

## Root cause

`copy_agg_into_struct_slots_a64()` (`self-hosted/compiler/lean_single.sio`)
handles copying an aggregate value (array, `Option<T>` inline, or nested
struct) into a struct literal's field slots. For all three cases it computed
the destination's starting address as:

```sio
emit_lea_var_a64(dst_start - (nslots - 1))   // array / struct-like
emit_lea_var_a64(dst_start - 1)              // option_inline (nslots==2)
```

`dst_start` is the struct's slot number for this field, using the
compiler's "highest slot number = lowest address = element/byte 0"
convention (the same convention every other aggregate-slot computation in
this codebase uses). `emit_copy_words_x10_x11_a64()` then copies `nslots`
words starting from that address, **incrementing** both source and
destination pointers by 8 bytes each iteration — which walks from
`dst_start` down through `dst_start - (nslots - 1)` in slot-number terms
(since increasing address ⇔ decreasing slot number). Pre-subtracting
`(nslots - 1)` before the copy loop's own natural decrement therefore starts
the write `(nslots - 1)` slots too low, and the loop then walks **another**
`(nslots - 1)` slots past that — landing the entire copy in a range shifted
by `(nslots - 1)` slots from where every reader (struct field access,
`.field[idx]` indexing) expects the data, and, for a large enough field,
spilling into unrelated stack memory.

The x86-64 twin, `copy_agg_into_struct_slots_x86()`, does not have this bug:
its `nslots <= 32` path is a per-element loop that stores each source word
directly to `dst_start - i` for `i` in `0..nslots-1` — i.e. it starts at
`dst_start` itself (no pre-adjustment) and lets the loop's own index do the
walking. The a64 twin's single-LEA-plus-fixed-loop shape needed the
equivalent starting point (`dst_start`, unadjusted) but had an extra,
incorrect subtraction baked into the initial address instead.

Confirmed via runtime-injected instrumentation (temporarily emitting extra
machine code — `emit_print_int_a64()` calls wrapped in register-preserving
push/pop — into the generated binary itself, then running it under
`qemu-aarch64-static`) that the loaded source pointer and computed
destination pointer were both individually correct-looking addresses,
ruling out a "wrong pointer value" theory before finding the actual
off-by-`(nslots-1)`-slots addressing mistake by hand-tracing the copy loop's
iteration order against where the field-read side expects data to land.

## Fix

Removed the erroneous pre-subtraction from all three branches — each now
starts the destination LEA at `dst_start` directly, matching the x86-64
twin's effective starting point:

```sio
emit_lea_var_a64(dst_start)   // array / struct-like / option_inline, all three
```

## Verification

```bash
export SOUNIO_STDLIB_PATH="$(pwd)/stdlib"
scripts/dev/souc-build-lock.sh ./bin/souc-lean-single-x86_64 self-hosted/compiler/lean_single.sio /tmp/lean_fixed.elf

# Exact CI repro, cross-compiled + run under qemu-aarch64-static:
/tmp/lean_fixed.elf tests/selfhost/native_runtime/abi_return_literal_array_field_42.sio /tmp/t.elf --target aarch64-linux
qemu-aarch64-static /tmp/t.elf; echo $?   # 42 (was 139/SIGSEGV)

# Full aarch64 native_runtime manifest (99 cases) via aarch64-linux + qemu:
# 95 pass / 4 fail (was 93 pass / 6 fail) — abi_return_literal_array_field_42
# and bdf_stiff both newly fixed; zero new failures.

# Full x86-64 suite (fix is aarch64-only, unaffected by construction):
bash scripts/run_sio_test_suite.sh --format junit --jobs 8
# Pass: 1314  Fail: 0  Known failures: 124  Skip: 689  Total: 2127 — exact baseline
```

## Discovered but explicitly out of scope

Four failures remain in the aarch64 native_runtime manifest, confirmed
present identically **before** this fix (not introduced by it):

- `abi_nested_array_local_only_42`, `abi_return_nested_array_42` — both
  mutate an array field through a **nested struct field chain**
  (`s.inner.vals[0] = 10`), a different code path (nested-field-chain
  assignment) from the struct-literal-construction path fixed here. Traced
  `fill_repeat_struct_array_slots_a64()` (the array-*repeat*-literal
  initializer these tests also use) far enough to confirm its own
  `dst_start - i` addressing is already correct (no shared root cause with
  this dispatch) — the actual defect for these two is elsewhere,
  uninvestigated.
- `epistemic_ops_42`, `epistemic_propagation` — both segfault under
  `aarch64-linux`; not investigated, plausibly an unrelated aarch64
  epistemic-runtime gap.

None of these were in scope for the CI failure this dispatch was opened to
fix; each would need its own dispatch.

## Cross-references

- CI: `Release Gate` → `Apple Self-Host` job, `scripts/selfhost/
  selfhost_native_acceptance_gate.sh`. Confirmed via `gh run list --workflow
  "Release Gate"` that this job has failed on every run for 3+ weeks with a
  *different* specific failing test each time (test count also grew over
  that period) — consistent with ongoing aarch64 work outpacing new test
  coverage, not one static unfixed bug.
- `reference_a64_codegen_gotchas.md` (project memory) — aarch64 codegen
  gotchas reference; this dispatch's finding (an addressing off-by-`(nslots-1)`
  in a shared aggregate-copy helper) is a new entry for that reference.
