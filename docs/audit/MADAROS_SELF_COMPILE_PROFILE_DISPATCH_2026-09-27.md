# Madaros self-compile: where the 54 minutes go — dispatch

**Date:** 2026-09-27
**Status:** EVIDENCE + PROPOSED FIX (no `self-hosted/` change in this dispatch)
**Scope:** wall time of `madaros build self-hosted/compiler/main.sio` (the post-merge fixed-point rung, 56–62 min in CI)

## 1. Measurements

Both runs: DARWIN node `r770-proxmox`, 4 CPU / 48 GiB pod, `ulimit -s 524288`,
`SOUNIO_STDLIB_PATH` = checkout stdlib. No other heavy build on the pod.

| Run | Commit | Compiler | Wall | Artefacts (`sounio-ci-cache` PVC) |
|---|---|---|---:|---|
| Phase timing | `f141ad5d` | cached Madaros `3b1c11ec…` | 3242 s | `/cache/profile/20260927T004243/` |
| Function sampling | `897c3448` | Madaros rebuilt by the seed with `--debug-fn-map` (`sha256 f1f69e71…`) | 3149 s | `/cache/profile/fn2-20260927T023032/` |

Both runs exit rc=0 and write the output ELF.

### 1.1 Phases

From per-line timestamps on the compiler's own progress output:

| Phase | Span (s) | Duration | Share |
|---|---|---:|---:|
| `run_check_mode` (127 modules) | 5 → 139 | 134 s | 4% |
| `imported_compile` typecheck | 141 → 275 | 134 s | 4% |
| `lower_array` seed (main program) | 276 → 416 | 140 s | 4% |
| **`lower_array` deps, 126 modules, `into_acc`** | 416 → 2610 | **2194 s** | **68%** |
| **IR merge → `Merged IR: 13571 functions`** | 2610 → 3128 | **518 s** | **16%** |
| native emit + write | 3128 → 3241 | 112 s | 3% |

**Per-module time.**
- Per-module dep lowering sums to 2189 s over 126 modules: a mean of 17.4 s and a floor of about 10 s.
- The time does not scale with module size. Module 80 is the outlier at 128 s.
- It falls slowly from about 16 s (modules 1–40) to about 11 s (modules 85–126).
- That is the signature of a per-lookup cost that scales with a large, shared table, not with the module being lowered.

**Memory.**
- Peak RSS was 15.7 GB, growing monotonically from 5.3 GB.
- `arena_reset_skipped (call-arg scratch overflow)` fires for 126/126 modules, with `sites_reclaimed=0`.
- This is a separate memory defect, noted in §4, and does not explain the wall time.

### 1.2 Function-level profile

**Sampler.**
- Tool: `rip_sampler` (ptrace, 50 ms, RBP-chain unwind to depth 24).
- Symbolised against the seed's `--debug-fn-map`.
- Runtime address = `0x401000 + off`; `.text` loads at `0x400000` with a 4096-byte header.
- 62 799 samples; 11 720 functions mapped.

**Artefact.** `__native_parse_f64` and `main` show about 88% inclusive only because frames above `main` (process start-up stubs) resolve to the last fnmap entry. Ignore them.

**Top self time (whole run):**

| Self % | Samples | Function |
|---:|---:|---|
| **53.7** | 33 731 | `ir_fn_get` |
| 10.7 | 6 744 | `ir_function_name_at` |
| 6.7 | 4 228 | `lowerer_name_has_body_mut` |
| 5.8 | 3 662 | `fn_sig_table_find_prefer_module` |
| 4.0 | 2 492 | `ir_merge_find_function_name_index` |
| 3.3 | 2 069 | `lowerer_find_or_add_fn_id_mut` |
| 2.8 | 1 747 | `lowerer_lookup_fn_id_by_name_ref` |
| 1.9 | 1 162 | `ir_name_eq` |
| 1.7 | 1 060 | `ir_fn_table_chunk_at` |
| 1.7 / 1.2 | 1 070 / 778 | `ir_float_bit_mask` / `ir_float_bits_get` |

**Top inclusive:**

| Incl % | Function |
|---:|---|
| 62.8 | `lower_dep_program_items_into_acc_with_externs` |
| 50.2 | `lowerer_preseed_external_items_into_acc_mut` |
| 36.0 | `lowerer_name_has_body_mut` |
| 31.7 | `lowerer_dep_should_skip_fn_mut` |
| 17.5 | `lowerer_find_or_add_fn_id_mut` |
| 15.5 | `ir_module_finalize_merged_calls` |
| 15.3 | `ir_merge_find_function_name_index` |
| 14.9 | `lowerer_lookup_fn_id_by_name_ref` |

**Per-phase windows (top self):**

| Window | Dominant self |
|---|---|
| check + typecheck (0–275 s) | `fn_sig_table_find_prefer_module` **65–69%** |
| seed lowering (275–416 s) | `ir_fn_get` 74% (via `lowerer_lookup_fn_id_by_name_ref`) |
| dep lowering (416–2610 s) | `ir_fn_get` **71–73%**, `lowerer_name_has_body_mut` 9–10% |
| IR merge (2610–3128 s) | `ir_function_name_at` **57%**, `ir_merge_find_function_name_index` 21% (inside `ir_module_finalize_merged_calls` 82%) |
| native emit (3128 s–end) | `ir_float_bit_mask` / `ir_float_bits_get` 82% |

## 2. Diagnosis

About 90% of the self-compile is **linear name lookups over the function table, each iteration copying structs by value.**

1. **Lowerer lookups (dep lowering and seed lowering).**
   - Affected: `lowerer_lookup_fn_id_by_name_ref` (`ir/lower.sio:1730`), `lowerer_name_has_body_mut` (`:2066`) and `lowerer_find_or_add_fn_id_mut` (`:2634`).
   - Each scans `0..fn_count` and calls `ir_fn_get(m, i).name` per iteration.
   - `ir_fn_get` (`ir/ir.sio:5770`) returns an entire `IrFunction` by value.
   - An `IrFunction` is about 1.9 KB: two `Name` of 392 B each, plus `param_regs[64]` and `float_reg_bits[64]`.
   - `ir_name_eq(a: Name, b: Name)` then takes both `Name`s by value, another 784 B.
   - So each probe copies about 2.7 KB to compare what is usually a length mismatch.
2. **The call pattern is quadratic.**
   - `lowerer_preseed_external_items_into_acc_mut` runs `_dedup` → `lowerer_name_has_body_mut` for every external item of every dep module, against an accumulator that grows to 13 570 functions.
   - That is O(items × fn_count) per module.
   - This matches the flat 10–18 s per-module floor in §1.1.
3. **Merge.**
   - `ir_module_finalize_merged_calls` → `ir_merge_find_function_name_index` (`compiler/module_frontend.sio:1616`) does a full scan per call target.
   - Each probe goes through `ir_function_name_at`, which returns a 392 B `Name` by value.
   - That is O(call_sites × fn_count) ≈ 518 s.
4. **Checker.**
   - `fn_sig_table_find_prefer_module` (`check/defs.sio:1527`) runs up to three full passes over the sig table per call expression.
   - It accounts for 65–69% of the ~270 s of check + typecheck.

## 3. Proposed fix

Staged, so each stage is separately verifiable.

**Output invariant for every stage.**
- This is a pure performance change: emitted code must not move.
- Build `M_old` from the base commit and `M_new` from the patched commit (operating principle 15: no prebuilt binaries).
- Both compile the **same** input: `self-hosted/compiler/main.sio` at the base commit, plus the `run-pass` suite.
- Outputs must be **byte-identical** ELFs.
- Any diff is a stop.

**Stage A — stop copying (mechanical, no algorithmic change).**
- Add `ir_fn_name_eq_at(m: &IrModule, i: i64, name: &Name) -> bool`.
- It resolves the chunk once, compares `len` first, then bytes in place, with no `IrFunction` or `Name` copy.
- Use it in the four scans named in §2. Keep the scan order identical so first-match semantics are unchanged.
- Estimated effect: removes most of `ir_fn_get` + `ir_function_name_at` + `ir_name_eq` self time. That is ~66% of samples; the per-probe cost falls from ~2.7 KB copied to a length compare.
- This is an estimate, not a measurement; re-profile after landing.

**Stage B — index (algorithmic).**
- Add a name → fn-id hash index on `IrModule`, open-addressed on `ir_name_hash`.
- **A hash is not a key.** On a hash match, compare the full name and keep probing on mismatch, exactly as `ir_intern_name` does (`ir/ir.sio`: "Compare the NAME, not the hash"). Two colliding symbols must never alias one entry.
- **Each lookup keeps its own selection rule.** One entry per distinct name, holding:
  - `first_id`: the lowest id with that name. This is what the lowerer scans return: `lowerer_lookup_fn_id_by_name_ref`, `lowerer_find_or_add_fn_id_mut`, and `lowerer_name_has_body_mut`, which then reads `instr_count` of that first id live.
  - `last_body_id`: the **highest** id with `instr_count > 0`. `ir_merge_find_function_name_index` overwrites `found` on every matching body, so it returns the last body, not the first.
  - `first_stub_id`: the lowest id with `instr_count == 0`, the merge fallback.
- **Invalidate on body writes, not only name writes.** `ir_module_promote_canonical_into_stub_slots` (`compiler/module_frontend.sio`) turns a same-named stub into a body through `ir_fn_set`, which changes `last_body_id` and `first_stub_id` without touching the name. Route every write through `ir_fn_set` and have it refresh the entry whenever `name` or `instr_count > 0` changes. Otherwise keep the merge lookup linear, but Stage A-cheap.
- Guard with `SOUNIO_IR_FN_INDEX_VERIFY=1`: run indexed and linear lookup side by side and panic on any divergence. The CI rung runs once with it on.
- Stage A keeps the existing scans and their order, so it inherits all three rules unchanged.

**Stage C — checker sig table.** Same treatment for `FnSigTable`: a per-(name, module) index covering the three `prefer_module` passes.

**Minimal repro** (principle 12), to be added with Stage A:
- A generated `.sio` with *N* trivial `fn fK() -> i64 { K }` across *M* imported modules; time `madaros build` for N ∈ {1k, 2k, 4k, 8k}.
- Quadratic growth before, near-linear after.
- This isolates the lookup cost from everything else in `main.sio`.

## 4. Side findings (not part of this dispatch)

- **`arena_reset_skipped (call-arg scratch overflow)` on 126/126 dep modules, `sites_reclaimed=0`.** The per-module arena reset never runs, so RSS grows 5.3 → 15.7 GB. Separate dispatch.
- **Corrupt frame-size warning.** `warning: stack frame too large (4151395786548136 bytes)` on `compiler_mark_hlir_kernel_name` and `compiler_mark_hlir_kernels_from_items`. A 4.15e15-byte layout is a corrupt size computation, not a real frame.
- **Native emit hotspot.** `ir_float_bit_mask` / `ir_float_bits_get` dominate the last ~110 s; low priority.
- **`spec_dce_mm` refused.** `REFUSING to filter — mark set incomplete (marks 8192 of 8192)`: cross-module DCE is disabled on this input because the mark set saturates at 8192.

## 5. Reproduce

The sampler source and symboliser are in the ConfigMap `arc-runners/madaros-prof-tools`: `rip_sampler.cpp`, C++23, and `symbolize.pl`. The run is Job `madaros-fn-profile2`:

```bash
seed self-hosted/compiler/main.sio /tmp/m2 --debug-fn-map > seed.out   # seed = lean_single stage from the Madaros cache
chmod +x /tmp/m2                                                       # the seed does not set +x
perl -0777 -ne 'while(/fnmap fn=(\d+)\s+off=(-?\d+)\s+name=(\S+)/g){print "fnmap fn=$1 off=$2 name=$3\n"}' seed.out > fnmap.txt
rip_sampler --out samples.txt --interval-ms 50 --depth 24 -- /tmp/m2 build self-hosted/compiler/main.sio /tmp/gen2
perl symbolize.pl fnmap.txt samples.txt 401000
```
