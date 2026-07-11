<!-- docs:meta
topic_id: repo.docs.handoff.modular-madaros-build-greenup-scope
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.handoff.modular-madaros-build-greenup-scope
-->

# Follow-up scope — green the modular Madaros build (`build_modular_madaros.sh`)

**Tracking issue:** [#767](https://github.com/Sounio-lang/sounio/issues/767)
**Authored by:** Claude (EISA lane, `gpu/epistemic-tensor-core-next`), 2026-07-11
**Type:** compiler-internals (modular `main.sio` path). Serialized surfaces; fixed-point + output-verified gates mandatory.
**Priority:** non-blocking. The branch's *proven* compiler is `lean_single` (gen1→gen2→gen3 fixed point, md5 `6da577cadf74b50e8f3df7fa0813c62b`). This work only concerns the parallel modular `main.sio` compiler.

## Context — why this is a separate effort

The EISA↔main merge (PR #742, merge `3d560e8ff`, fix `c6321b788`) preserves the verified `lean_single` self-host. Investigating the `parser/items.sio:1086` typecheck failure revealed that the **modular** compiler (`self-hosted/compiler/main.sio` + its import graph) is **pre-existingly red** — it typechecks with ~140–160 errors from undefined symbols in **both** checkpoint-C's and `origin/main`'s file versions. It is unrelated to the merge; the modular `main.sio` path was never green on this branch.

Reproduce:
```
SOUC_BIN=/tmp/g3.elf bash scripts/ci/build_modular_madaros.sh /tmp/madaros_out
# (/tmp/g3.elf = a fresh gen3 lean_single seed from `make build` / the gen-sweep)
```
Full error log captured during the merge investigation: rebuild and grep `^error:`.

## Error inventory (162 errors, current tree)

| Bucket | Count | Files | Root cause |
|---|---:|---|---|
| **W1 wasm backend** | 136 | `wasm/lower.sio` (70), `wasm/encode.sio` (66) | Import-path mismatch, **not** missing code |
| **W2 codegen EISA symbols** | ✅ done | `native/codegen_x86_linux.sio` | Referenced-but-never-declared EISA symbols — fixed on `gpu/epistemic-tensor-core-next` (see W2 section) |
| **W3 ir type mismatches** | 3 | `ir/const_prop.sio:1676`, `ir/dce.sio:885`, `ir/opt_strategy.sio:229` | Field-init / call-arg type mismatch |
| **W4 gpu kernel_ir** | ✅ done | `gpu/kernel_ir.sio` | `missing field in struct literal` in `gpu_build_gemm_shared_ir` — fixed on `gpu/epistemic-tensor-core-next` (see W4 section) |
| **W5 kaxi buf API drift** | 2 | `compiler/main.sio:27616,27643` | `unknown field access` `kaxi.data` — `KaxiAsmBuf` refactored inline-array → heap-ptr; K-AXI write path not updated (see W5 section) |
| **W0 items.sio is_extern** | ✅ done | `parser/items.sio:1086` | Fixed in `c6321b788` |

Category totals (original): 93 `unknown identifier`, 33 `unknown field access`, 28 `value is not indexable`, 5 `private struct field access [struct=Lowerer]`, 2 `field initializer type mismatch`, 1 `E001`.

### Stacked verification (2026-07-11): 148 → 2

Applying **W1+W3 locally** (cherry-pick `67d05760c` + `8bf239a8d`, W3's `const_prop`/`dce` conflicts resolved to the loop-copy fix) on top of the committed **W2+W4** on `gpu/epistemic-tensor-core-next`, the modular typecheck drops from **148 → 2**. The only survivors are the 2 `<main>` residuals — which **persist with W1+W3 applied**, proving they are NOT W1/W3 fallout but a distinct bug (**W5**). The local stack was reverted after measurement (W1/W3 land via PRs #771/#773; re-apply = 2 cherry-picks). So once W1+W3 merge and W5 is fixed, the EISA modular typecheck is **clean (0)**.

## Work items

### W1 — wasm backend import wiring (136 errors, biggest, **shared with main**)
- **Root cause:** the 24 distinct undefined `wasm_*` helpers (`wasm_buf_byte_at`, `wasm_func_type_new`, `wasm_*_section_new/add`, `wasm_emit_full_module`, …) **are defined** — in `self-hosted/wasm/mod.sio`. But `wasm/encode.sio`, `wasm/lower.sio`, and `main.sio` import `use wasm::core::*` / `use wasm::encode::*`, which do not surface `mod.sio`'s definitions. The `value is not indexable` / `unknown field access` / `private struct field access [struct=Lowerer]` errors are downstream of the same missing types (wasm sections/buffers) not being in scope.
- **Fix direction:** align module boundaries — either move the `wasm_*` definitions from `wasm/mod.sio` into `wasm/core.sio` (where consumers import from), or re-export them from `core`, or repoint the `use` paths to the module that actually defines them. Then re-check the `Lowerer` field visibility (5 `private struct field access` → make the accessed fields `pub` or route through accessors).
- **main-parity:** `wasm/encode.sio` + `wasm/lower.sio` are **byte-identical to `origin/main`** → main has the identical errors. Fixing here does not diverge from main behaviour; it is a genuine shared bug. Consider landing the wasm wiring fix to main independently.
- **Effort:** medium. Mechanical once the intended module layout is decided, but touches the module system.

### W2 — codegen EISA symbols undeclared (✅ DONE, **EISA-specific**)
- **Root cause:** `native/codegen_x86_linux.sio` references EISA's own off-stack/flat-reloc + text-mirror machinery that was scaffolded but never completed. The true count was **23** (21 + 2 more found during the fix):
  - globals `NC_FLAT_RELOC_OFFSETS` / `_KIND_CODES` / `_IS_FUNCTIONS` / `_TARGET_INDICES` (`[i64; 65536]`, written at 1629, drained at 5181) + `NV2_RELOC_OVERFLOW` (bool overflow latch) — used but never declared.
  - `NATIVE_V2_TEXT_BUF` / `NATIVE_V2_TEXT_LEN` — the ELF-writer ".text mirror" (read at 10910/11141, preferred over `code.bytes` when `ti < LEN`) — used but never declared. **Populate path never built**: nothing assigns `NATIVE_V2_TEXT_LEN`, so it is `0` and the mirror is never read.
  - functions `native_v2_core_ir_trace_fail`, `native_v2_patch_label_forwards_text_mirror`, `native_v2_text_reset` — defined nowhere.
- **Resolution** (all provable from the code, no guessed codegen):
  - Declared the 7 globals module-scope next to `NC_BIG_ELF`/`NV2_SRET_*` (BSS, demand-paged).
  - `native_v2_core_ir_trace_fail(...) -> bool` → returns `false` (fail-path sentinel; the core-IR path carries no IO effect so it cannot print — the `reason` string documents each bail site).
  - `native_v2_patch_label_forwards_text_mirror(nc)` → **documented no-op**. `patch_label_forwards_mut(nc)` (called immediately before) patches the live buffer; the mirror is inert (`LEN ≡ 0`, never read), so patching it has no observable effect. Verified `NATIVE_V2_TEXT_LEN` is never assigned or address-taken anywhere in the tree.
  - `native_v2_text_reset()` → pins `NATIVE_V2_TEXT_LEN = 0` (self-enforces the inert-mirror invariant the no-op relies on) and clears `NV2_RELOC_OVERFLOW`.
- **Verified:** modular build → `codegen_x86_linux.sio` errors **23 → 0**, zero W2-family symbols remain. Edit is modular-native-v2-only (1 file, +53 lines); `lean_single.sio` contains 0 of these symbols and its seed still compiles → **gen2==gen3 unaffected by construction**.
- **Note:** the off-stack text-mirror remains an inert, unfinished feature (scaffold + read sites, no populate). Completing it (populate `NATIVE_V2_TEXT_BUF`, patch its label forwards) is a separate native-v2 backend task, NOT W2. If a future change sets `NATIVE_V2_TEXT_LEN > 0`, the no-op mirror-patch becomes a live miscompile — a guard-comment at the call site flags this.

### W3 — ir type mismatches (3 errors, small)
- `ir/const_prop.sio:1676` + `ir/dce.sio:885`: `field initializer type does not match struct field`. `ir/opt_strategy.sio:229`: `E001` call-argument type mismatch. These three files **differ from main** (EISA-modified), so the mismatches are EISA-carried.
- **Fix direction:** localized — inspect each struct-field/call type and correct the initializer/argument. Likely a widened/narrowed int type or a changed struct shape.
- **Effort:** small (hours).

### W4 — gpu kernel_ir missing struct fields (✅ DONE, **EISA-specific**)
- **Root cause:** `gpu/kernel_ir.sio` fn `gpu_build_gemm_shared_ir` has 5 `GpuOp { … }` literals that each omit one required field (Sounio struct literals must set all fields; `GpuOp` has 23). Not a shape mismatch — hand-written literals that dropped a field.
- **Resolution** (each value recovered from code comments + sibling literals, not defaulted — the missing fields are real operands, so `-1` would miscompile):
  - `GpuSetpLt` p0 (dst_reg:0) missing `rhs_reg` → **6** (`p0=(r1<m)`, m=param3→r6; siblings p1/p2 use r7/r8).
  - 3× `GpuAdd` (dst 9/10/11) missing `rhs_reg` → **2** (comment `rdN = rdK + r2`).
  - `GpuStoreSharedPred` (F32 branch, src_reg:1) missing `lhs_reg` → **0** (predicate p0; mirrors its `GpuLoadGlobalPred dst_reg:1 lhs_reg:0`; the F64-branch twin already had it).
- **Verified:** modular build → `kernel_ir.sio` errors 5→0, no new gpu errors. Edit is gpu-modular-only (1 file, 5 lines); `lean_single.sio` has 0 of these symbols → gen2==gen3 unaffected by construction.

### W5 — KaxiAsmBuf API drift in the K-AXI write path (2 errors, **EISA-specific, OPEN**)
- **Root cause:** `KaxiAsmBuf` (`gpu/kaxi_backend.sio:197`) was refactored from an inline `data: [i8; 2097152]` byte array to a **heap-ptr model** (`ptr: i64` into a 16 MiB `KAXI_ASM_CAP` block, `len`, `cap`; bytes accessed via `kaxi_asm_get` word-RMW because `*mut i8` isn't byte-indexable in the seed). But the K-AXI write path was not migrated:
  - `compiler/main.sio:27616` + `:27643` read `kaxi_all.data` / `kaxi.data` — **no such field** (`unknown field access`); `.len` also now private.
  - `io/file_write.sio` `io_write_kaxi_binary(path, bytes: [i8; 2097152], len)` still takes a 2 MB **inline array** (stale — even its doc-comment says "KaxiAsmBuf.data is [i8; 2097152]"), which is both the wrong representation and too small (buffer cap is 16 MiB). `write_file` is a builtin taking a fixed `[i8; N]` array (no ptr variant).
- **Fix direction:** add a heap-aware writer in `kaxi_backend.sio` (same module → can read the private `len`), e.g. `io_write_kaxi_buf(path, buf: KaxiAsmBuf) -> bool`, that copies `buf.len` bytes via `kaxi_asm_get` into a **module-scope BSS** `[i8; 16777216]` staging array (NOT a stack local — mirror `NC_BIG_ELF`), then `write_file(path, staging, buf.len)`. Repoint the 2 main.sio call sites to it and drop the stale `io_write_kaxi_binary`. Touches I/O correctness — verify the emitted `.kaxi` bytes round-trip.
- **Scope:** modular-main.sio-only (`lean_single.sio` has 0 of `io_write_kaxi_binary`/`kaxi_all.data`/`hlir_kernels_to_kaxi` → the proven compiler lacks this path; gen2==gen3 unaffected). This is the **last** blocker to a fully-clean EISA modular typecheck.

## Acceptance criteria

1. `SOUC_BIN=<fresh gen3 seed> bash scripts/ci/build_modular_madaros.sh <out>` → `typecheck: OK`, produces an executable `<out>`.
2. `<out>` compiles a hello-world and a struct+recursion program to runnable ELFs (Stage-0/0b equivalents).
3. **No regression to `lean_single`:** gen1→gen2→gen3 still md5 `6da577c…` (W1–W3 touch modular-only files not in the import-free `lean_single.sio` bundle, but re-run as the non-falsifiable gate).
4. W1 fix ideally landed to `main` too (shared wasm bug) rather than only on the EISA branch.

## Explicitly out of scope / already handled

- The merge itself (PR #742) is done, `MERGEABLE`, gen2==gen3-verified.
- `lean_single` decisions (a64 borrow engine subsumes #740; main's lean_single deletes EISA features — kept ours deliberately).
- `items.sio` `is_extern` (W0) — fixed in `c6321b788`.

See [[eisa-main-merge-landed]] (memory) and `docs/handoff/` for prior compiler-internals handoffs.
