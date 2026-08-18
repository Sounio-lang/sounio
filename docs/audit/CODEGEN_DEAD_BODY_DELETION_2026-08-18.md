<!-- docs:meta
topic_id: repo.docs.audit.codegen-dead-body-deletion-2026-08-18
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.codegen-dead-body-deletion-2026-08-18
-->

# Codegen dead-body deletion — option-3 result

**Date:** 2026-08-18
**Counsel:** minimax-cli2
**Working tree:** `codegen/v2-ref-deletion-20260818` at `/tmp/wt-codegen-deletion` (branched from `origin/main` at `2016efb8e4`).
**Companion priors:**
- `docs/audit/CODEGEN_DUPLICATION_REFUTED_HYPOTHESIS_2026-08-17.md` (homonym-cascade refutation)
- `docs/audit/CODEGEN_BODY_DIFF_GLOB_HOMONYMS_2026-08-17.md` (48-pair byte-diff classification)

## Scope (user's directive, restated)

Delete ONLY bodies the census proves unreachable, one body at a time with cheap `souc check` between each. Confirm first with a repo-wide `native::codegen::` importer search. Measure how many of the 25 residual E175-shaped homonyms drop. If any remain, name each: symbol, the two files, and whether the copies DIVERGE or are identical.

## Step 0 — Confirmation: importer search across the entire repo

`grep -rn 'native::codegen' --include='*.sio' .` (excluding `archive/` and `bootstrap/`) returned **24 importers** (17 glob `use native::codegen::*` and 7 qualified — see the doc that accompanies this one for the full list).

`grep -rn 'compile_ir_function_v2_ref' --include='*.sio' .` returned exactly **4 lines**:

| file | line | role |
|---|---|---|
| `self-hosted/native/codegen.sio` | 4187 | definition (`pub fn`) |
| `self-hosted/native/codegen.sio` | 4189 | internal call (from `compile_ir_function_v2_ref` body to `compile_ir_function_v2_into`) |
| `self-hosted/native/codegen_x86_linux.sio` | 8714 | definition (`pub fn`) |
| `self-hosted/native/codegen_x86_linux.sio` | 8716 | internal call (from `compile_ir_function_v2_ref` body to `compile_ir_function_v2_into`) |

**Zero external callers.** Confirmed dead in BOTH files before any edit.

## Step 0.5 — Premise verification on the user's `compile_ir_function_v2_into` claim

The dispatch stated: *"compile_ir_function_v2_into tem 7 chamadores no x86_linux e 0 directos no codegen.sio, onde só é alcançado por v2_ref, que também está morto."*

That is half right and half wrong:

- ✅ **True:** the INTERNAL codegen.sio reachability of `compile_ir_function_v2_into` is exactly one path — through `compile_ir_function_v2_ref`. Zero OTHER direct callers inside `codegen.sio`'s own body.
- ✅ **True:** `compile_ir_function_v2_ref` is dead (see Step 0).
- ❌ **Not 7, not reachable-from-outside-only via v2_ref:** `compile_ir_function_v2_into` is **externally LIVE** in BOTH files:

| external importer | imports from | call site |
|---|---|---|
| `self-hosted/compiler/module_loader.sio` (qualified, line 36) | `native::codegen` | 1845, 3742 |
| `self-hosted/native/wide_driver.sio` (glob, line 22) | `native::codegen::*` | 255 |
| `self-hosted/compiler/render_native_compile_driver_stable.sio` (glob, line 35) | `native::codegen::*` | 233 |
| `self-hosted/compiler/module_native_streaming.sio` (qualified, line 13) | `native::codegen_x86_linux` | 141 |

So in `codegen.sio`'s `compile_ir_function_v2_into` is reached from 4 distinct external call sites; in `codegen_x86_linux.sio`'s it is reached from 1.

**Operationally:** per the user's "APENAS os corpos que o censo prova inalcançáveis" rule, the only deletions authorised by the census are the two `compile_ir_function_v2_ref` bodies. **`compile_ir_function_v2_into` is NOT deletable** — it has live external callers in both modules.

## Step 1 — Delete `compile_ir_function_v2_ref` from `codegen.sio`

Body at lines 4187–4191 (5 lines). Replaced with a tombstone comment recording the deletion reason and the census evidence. Auto-compile (`souc check` on the file plus its 4 importers) showed E175 counts unchanged at 61 / 70 / 67 / 61 — confirming the prebuilt `bin/souc` couldn't see the source change (and so cannot refute the deletion), and that no importer was generating E175s about `v2_ref` even before the edit.

Commit: `dead-code(codegen): delete compile_ir_function_v2_ref from codegen.sio` (1 file, +8/-5).

## Step 2 — Delete `compile_ir_function_v2_ref` from `codegen_x86_linux.sio`

Body at lines 8714–8718. Mirror edit. Auto-compile on `codegen_x86_linux.sio` and `module_native_streaming.sio` (the only importers of `codegen_x86_linux::*` that could plausibly reach this symbol): E175 counts unchanged at 0 / 0.

Commit: `dead-code(codegen_x86_linux): delete compile_ir_function_v2_ref from codegen_x86_linux.sio` (1 file, +7/-5).

## Step 3 — Measure: how many of the 25 residual E175-shaped homonyms fell?

**Zero.** The two deleted bodies (`compile_ir_function_v2_ref` × 2) are NOT in the 25-residual set.

The 25-residual set is the 48-pair byte-diff census (§`CODEGEN_BODY_DIFF_GLOB_HOMONYMS_2026-08-17.md`) minus the 23 PUB-SWAP-ONLY pairs (whose bodies are byte-identical after stripping `pub`, so they're "no actual content difference" and don't behave differently as E175 generators). That leaves 24 IDENTICAL + 1 SUBSTANTIVE-DIVERGENT = **25 residual pairs whose content-shape could plausibly generate or fail to generate E175s**.

`v2_ref` is not one of those 25 names. So deleting it does not touch any residual.

**Re-census after deletion** (run on the same source files in `/tmp/wt-codegen-deletion`):

```
IDENTICAL (debt): 24
PUB_SWAP_ONLY (debt+vis): 23
SUBSTANTIVE_DIVERGENT (bug): 1

Sum of E175-residual (IDENTICAL + DIVERGENT): 25
```

**25 of 25 remain. 0 fell.**

## Step 4 — Name each of the 25 residuals (per user directive)

### 24 IDENTICAL pairs (debt — pure duplication, bodies byte-equal)

Both files literally carry the same line: e.g. `pub fn ARCH_X86_64() -> i64 { native_policy_arch_x86_64() }`. Deletion direction is blocked by glob-importers that depend on the name existing in the named module — see prior docs for the deletion-direction analysis.

| # | symbol | files | divergence |
|---:|---|---|---|
| 1 | `ARCH_RISCV64` | codegen.sio, codegen_x86_linux.sio | IDENTICAL |
| 2 | `ARCH_UNKNOWN` | codegen.sio, codegen_x86_linux.sio | IDENTICAL |
| 3 | `ARCH_X86_64` | codegen.sio, codegen_x86_linux.sio | IDENTICAL |
| 4 | `ERR_BACKEND_NOT_IMPLEMENTED` | codegen.sio, codegen_x86_linux.sio | IDENTICAL |
| 5 | `ERR_INVALID_MATRIX` | codegen.sio, codegen_x86_linux.sio | IDENTICAL |
| 6 | `ERR_INVALID_TARGET` | codegen.sio, codegen_x86_linux.sio | IDENTICAL |
| 7 | `ERR_OK` | codegen.sio, codegen_x86_linux.sio | IDENTICAL |
| 8 | `ERR_TARGET_FLAGS_REQUIRE_NATIVE` | codegen.sio, codegen_x86_linux.sio | IDENTICAL |
| 9 | `ERR_TRACE_WRITE_FAILED` | codegen.sio, codegen_x86_linux.sio | IDENTICAL |
| 10 | `FORMAT_ELF64` | codegen.sio, codegen_x86_linux.sio | IDENTICAL |
| 11 | `FORMAT_MACHO64` | codegen.sio, codegen_x86_linux.sio | IDENTICAL |
| 12 | `FORMAT_NONE` | codegen.sio, codegen_x86_linux.sio | IDENTICAL |
| 13 | `FORMAT_PE64` | codegen.sio, codegen_x86_linux.sio | IDENTICAL |
| 14 | `MACOS_SYS_OPEN` | codegen.sio, codegen_x86_linux.sio | IDENTICAL |
| 15 | `MATRIX_AME` | codegen.sio, codegen_x86_linux.sio | IDENTICAL |
| 16 | `MATRIX_APPLE_AMX` | codegen.sio, codegen_x86_linux.sio | IDENTICAL |
| 17 | `MATRIX_AUTO` | codegen.sio, codegen_x86_linux.sio | IDENTICAL |
| 18 | `MATRIX_IME` | codegen.sio, codegen_x86_linux.sio | IDENTICAL |
| 19 | `MATRIX_INTEL_AMX` | codegen.sio, codegen_x86_linux.sio | IDENTICAL |
| 20 | `MATRIX_OFF` | codegen.sio, codegen_x86_linux.sio | IDENTICAL |
| 21 | `MATRIX_VME` | codegen.sio, codegen_x86_linux.sio | IDENTICAL |
| 22 | `OS_LINUX` | codegen.sio, codegen_x86_linux.sio | IDENTICAL |
| 23 | `OS_UNKNOWN` | codegen.sio, codegen_x86_linux.sio | IDENTICAL |
| 24 | `OS_WINDOWS` | codegen.sio, codegen_x86_linux.sio | IDENTICAL |

### 1 SUBSTANTIVE-DIVERGENT pair (the bug — runtime divergence)

| # | symbol | files | divergence |
|---:|---|---|---|
| 25 | `name_is_get_arg_count` | codegen.sio (L948–L965, 18 lines), codegen_x86_linux.sio (L1199–L1229, 31 lines) | **SUBSTANTIVE-DIVERGENT** |

The 13-character `"get_arg_count"` byte-cascade is identical in both files. The x86_linux copy **also** matches a 9-character `"arg_count"` alias (with explicit comment: *"Source-level spelling is `arg_count`; retain the older `get_arg_count` backend alias for hand-built IR witnesses."*). The codegen copy **does not** include the 9-char alias. So `name_is_get_arg_count("arg_count")` returns `false` from `codegen.sio`'s version but `true` from `codegen_x86_linux.sio`'s version — runtime divergence depending on which module the call resolves through. This is the user-framed "bug à espera".

## Halt per FLEET_CONSTRAINTS

The user's directive also asked for a measurement of how many of the 25 residuals fall from this deletion. I have done that measurement in the form available on this pod:

- The prebuilt `./bin/souc` cannot see source edits (per FLEET_CONSTRAINTS: "**`./bin/souc` is PREBUILT.** Editing compiler source does not change it."). The cheap `souc check` baseline / post-edit E175 counts are identical — confirming the deletion didn't break anything in the prebuilt's view, but also confirming this measurement cannot see source-level deltas.
- The full self-compile gate (`scripts/ci/native_v2_driver_self_compile_gate.sh`) that would measure the residual E175 count change is forbidden on this pod per FLEET_CONSTRAINTS ("the k8s liveness probe recycles the pod under CPU saturation"). Slurm path is broken (`launch failed requeued held`).
- The static byte-diff census is the available measurement, and it says **0 of 25 fell** (because the deletions don't touch any of the 25).

This is the precisely-bounded refutation the protocol asks for: the deletions were safe (zero external callers), they did not touch the 25 residual homonym names, and the available measurement cannot detect a fall. To detect a fall, route through the self-compile gate (CI or Slurm) once those are operational.

## Files referenced (this worktree)

- `self-hosted/native/codegen.sio` — line 4187 replaced (8-line tombstone, 5-line body removed)
- `self-hosted/native/codegen_x86_linux.sio` — line 8714 replaced (7-line tombstone, 5-line body removed)
- `self-hosted/compiler/module_loader.sio` (importer, unchanged)
- `self-hosted/native/wide_driver.sio` (importer, unchanged)
- `self-hosted/compiler/render_native_compile_driver_stable.sio` (importer, unchanged)
- `self-hosted/compiler/module_native_streaming.sio` (importer of codegen_x86_linux, unchanged)
- `docs/audit/CODEGEN_DUPLICATION_REFUTED_HYPOTHESIS_2026-08-17.md`
- `docs/audit/CODEGEN_BODY_DIFF_GLOB_HOMONYMS_2026-08-17.md`
- `scripts/ci/native_v2_driver_self_compile_gate.sh` (NOT executed on this pod)

## Branch

`codegen/v2-ref-deletion-20260818` at `/tmp/wt-codegen-deletion`. Two commits:

```
838e250 dead-code(codegen_x86_linux): delete compile_ir_function_v2_ref from codegen_x86_linux.sio
3bbcb56 dead-code(codegen): delete compile_ir_function_v2_ref from codegen.sio
```

Not pushed. Awaiting user direction (push / iterate / close).
