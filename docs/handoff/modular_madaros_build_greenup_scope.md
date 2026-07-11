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
| **W2 codegen EISA symbols** | 21 | `native/codegen_x86_linux.sio` | Referenced-but-never-declared EISA symbols |
| **W3 ir type mismatches** | 3 | `ir/const_prop.sio:1676`, `ir/dce.sio:885`, `ir/opt_strategy.sio:229` | Field-init / call-arg type mismatch |
| **W0 items.sio is_extern** | ✅ done | `parser/items.sio:1086` | Fixed in `c6321b788` |
| `<main>` residual | 2 | bundled | Falls out once W1/W2 resolve |

Category totals: 93 `unknown identifier`, 33 `unknown field access`, 28 `value is not indexable`, 5 `private struct field access [struct=Lowerer]`, 2 `field initializer type mismatch`, 1 `E001`.

## Work items

### W1 — wasm backend import wiring (136 errors, biggest, **shared with main**)
- **Root cause:** the 24 distinct undefined `wasm_*` helpers (`wasm_buf_byte_at`, `wasm_func_type_new`, `wasm_*_section_new/add`, `wasm_emit_full_module`, …) **are defined** — in `self-hosted/wasm/mod.sio`. But `wasm/encode.sio`, `wasm/lower.sio`, and `main.sio` import `use wasm::core::*` / `use wasm::encode::*`, which do not surface `mod.sio`'s definitions. The `value is not indexable` / `unknown field access` / `private struct field access [struct=Lowerer]` errors are downstream of the same missing types (wasm sections/buffers) not being in scope.
- **Fix direction:** align module boundaries — either move the `wasm_*` definitions from `wasm/mod.sio` into `wasm/core.sio` (where consumers import from), or re-export them from `core`, or repoint the `use` paths to the module that actually defines them. Then re-check the `Lowerer` field visibility (5 `private struct field access` → make the accessed fields `pub` or route through accessors).
- **main-parity:** `wasm/encode.sio` + `wasm/lower.sio` are **byte-identical to `origin/main`** → main has the identical errors. Fixing here does not diverge from main behaviour; it is a genuine shared bug. Consider landing the wasm wiring fix to main independently.
- **Effort:** medium. Mechanical once the intended module layout is decided, but touches the module system.

### W2 — codegen EISA symbols undeclared (21 errors, **EISA-specific**)
- **Root cause:** `native/codegen_x86_linux.sio` references EISA's own off-stack/flat-reloc machinery that was never completed:
  - globals `NC_FLAT_RELOC_OFFSETS` / `_KIND_CODES` / `_IS_FUNCTIONS` / `_TARGET_INDICES` — **used** (e.g. lines 1629, 5181) but **never declared** (unlike the sibling `NC_BIG_CODE` global, which is declared).
  - functions `native_v2_core_ir_trace_fail`, `native_v2_patch_label_forwards_text_mirror`, `native_v2_text_reset` — **defined nowhere**.
- **Fix direction:** declare the four `NC_FLAT_RELOC_*` globals alongside `NC_BIG_CODE` (module-scope fixed arrays, e.g. `[i64; 65536]`), and implement or honestly stub the three `native_v2_*` helpers per their call sites. This is finishing EISA's own work, not adopting main (main lacks this machinery entirely).
- **Effort:** medium. Needs understanding the flat-reloc design intent; risk of masking a deeper incomplete feature.

### W3 — ir type mismatches (3 errors, small)
- `ir/const_prop.sio:1676` + `ir/dce.sio:885`: `field initializer type does not match struct field`. `ir/opt_strategy.sio:229`: `E001` call-argument type mismatch. These three files **differ from main** (EISA-modified), so the mismatches are EISA-carried.
- **Fix direction:** localized — inspect each struct-field/call type and correct the initializer/argument. Likely a widened/narrowed int type or a changed struct shape.
- **Effort:** small (hours).

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
