# LOCALISATION_REPORT — ModuleGraph silent-corruption defect

> **Date completed:** 2026-07-19.
> **Session worktree:** `/tmp/sounio-modgraph-diffhash-20260719` at `origin/main@32b6707cb`.
> **Defect record of record:** `module_graph_facade_vertical_20260719T132620Z.json` (canonical worktree `/tmp/sounio-modgraph-witness-20260719`).
> **Recert confirmation:** `module_graph_facade_vertical_20260719T134414Z.json` (recert worktree `/tmp/sounio-modgraph-recert-20260719`).
> **Status:** localised to a single dispatch site + a single emit function. Not yet patched. No merge.

---

## 1. Headline localisation

The silent corruption is **not in `module_frontend.sio` at all**. It is in **`self-hosted/compiler/module_native_driver.sio`**, at the dispatch site that chooses between two compile lanes:

```
self-hosted/compiler/module_native_driver.sio:1171-1193
```

```sio
if !optimize {
    print("module_native_driver: imported source uses compact modular IR table path\n")
    let imported_ir_status = load_multimodule_imported_simple_ir_global(main_path)
    if imported_ir_status.ok {
        let simple_rc = native_driver_write_imported_simple_ir_elf(output_path)
        if simple_rc == 0 { return 0 }   // ← returns without ever calling the full IR lane
    }
    ...
}
...
module_frontend_compile_imported_to_file(main_path, output_path, optimize)
```

When `optimize=false` (the default for `souc compile` and `souc run`), the dispatch takes a "compact IR table" fast path. **That fast path is a never-finished development stub** that emits a hardcoded `"42\n"` ELF whenever any function in the program matches a print pattern. The full IR lane — which contains the merge, finalize, promote, restore machinery audited in `IR_CAPTURE_POINTS_MAP.md` — is never entered.

When `optimize=true` (i.e. `souc compile -O` or raw Madaros with `-O`), the dispatch skips the stub and calls the full IR lane directly. **The full IR lane produces correct output for every mutation.**

## 2. The acceptance chain is GREEN via the full IR path

Same fixtures, same Madaros binary (`bin/madaros-linux-x86_64` SHA-256 `11e7730f01...`), only the lane differs:

| Case | Compact path stdout | Compact path ELF SHA-16 | Full IR path (-O) stdout | Full IR path (-O) ELF SHA-16 |
|---|---|---|---|---|
| CONTROL_A greet=42 | `42` | `553ea15fd8fb88da` | `42` ✓ | `36cb83ed2d495e52` |
| CONTROL_B greet=999 | `42` ❌ | `4aaa5bf9bc018f87` | `999` ✓ | `743988b8ab0ae5dc` |
| PROBE_A leaf=42 | `42` | `553ea15fd8fb88da` | `42` ✓ | `ad84f9dc6ef3feb2` |
| PROBE_B leaf=7 | `42` ❌ | `2514a484042546cd` | `7` ✓ | `f9d4d310a2552087` |

Compact path: every mutation that should change stdout produces a different ELF SHA (the bytes do vary because paths/timestamps are embedded) but stdout is invariant `42`. The acceptance chain fails at the last step.

Full IR path: every mutation produces a different ELF SHA **and** the correct stdout. The acceptance chain is fully green.

## 3. The stub mechanism

`self-hosted/compiler/module_native_driver.sio:1002-1006` inside `native_driver_write_imported_simple_ir_elf`:

```sio
if native_driver_imported_simple_has_print() {
    native_driver_put_u8(&! bytes, data_off, 52)     // ASCII '4'
    native_driver_put_u8(&! bytes, data_off + 1, 50) // ASCII '2'
    native_driver_put_u8(&! bytes, data_off + 2, 10) // ASCII '\n'
}
```

Whenever the program contains any function classified as `kind=3` (print-statement pattern), the data section of the emitted ELF is filled with the literal bytes `"42\n"`. The classification work (which correctly extracts `VALUES[fi]` for each function) is **discarded** for kind=3 — see `native_driver_emit_imported_simple_fn` at `module_native_driver.sio:517-534`:

```sio
if kind == 3 {
    native_driver_put_u8(bytes, off, 184)              // mov eax, 1
    native_driver_put_u8(bytes, off + 1, 1)
    native_driver_put_u8(bytes, off + 5, 191)          // mov edi, 1
    native_driver_put_u8(bytes, off + 6, 1)
    native_driver_put_u8(bytes, off + 10, 72)          // lea rsi, [rip+disp32]   ← always data_off
    native_driver_put_u8(bytes, off + 11, 141)
    native_driver_put_u8(bytes, off + 12, 53)
    native_driver_put_u32(bytes, off + 13, data_off - (off + 17))
    native_driver_put_u8(bytes, off + 17, 186)         // mov edx, 3              ← always length 3
    native_driver_put_u8(bytes, off + 18, 3)
    native_driver_put_u8(bytes, off + 22, 15)          // syscall (write)
    native_driver_put_u8(bytes, off + 23, 5)
    native_driver_put_u8(bytes, off + 24, 49)          // xor eax, eax
    native_driver_put_u8(bytes, off + 25, 192)
    native_driver_put_u8(bytes, off + 26, 195)         // ret
    return true
}
```

The emitted code writes 3 bytes from `data_off` to stdout, returns 0. It does not reference `VALUES[fi]` at all. For `kind=2` (return-constant functions), the emit at `:511-516` correctly uses `VALUES[fi]` (`mov eax, VALUES[fi]; ret`). The compact path is partially functional but the print lane is stubbed.

## 4. Why this masqueraded as success

The Sounio repo ships `examples/projects/hello_pkg/src/{main,greet}.sio` as the canonical multi-module example. `greet.sio` returns 42; `main.sio` prints `answer()` followed by newline. The expected output is exactly `"42\n"` — which matches the stub's hardcoded data section byte-for-byte. Every CI run, every smoke test, every tutorial invocation of `souc run examples/projects/hello_pkg/src/main.sio` saw the correct output and concluded the compiler worked.

The recert matrix in `tests/compiler/module_graph_facade_vertical_witness/` exposed the stub by mutating the imported function body. Because the stub ignores function bodies, the mutation did not propagate to stdout — the silent corruption became visible.

## 5. Prime suspect in `IR_CAPTURE_POINTS_MAP.md` retracted

The map identified `ir_module_promote_canonical_into_stub_slots` (`module_frontend.sio:1728`) as the prime suspect. That identification was wrong. Promotion, merge, finalize, deep_copy, restore, and `module_frontend_compile_imported_to_file` as a whole are **all proven correct** by the full IR path producing correct output for every mutation.

The bug is one architectural layer above the merge: in the dispatch that decides whether to enter the merge at all.

## 6. Fix options for the implementing agent (in increasing scope)

### Option A — Wrapper workaround (1-line behavior change)

Modify `bin/souc` to inject `-O` whenever it would invoke the multi-module lane. This forces every user-facing compile through the full IR path. Cost: the full IR path is slower (it actually lowers, merges, finalises) and runs `opt_cleanup_module_inplace`. Touches a shared control file (`bin/souc` is on the AGENTS.md high-risk list) — coordinate before editing.

### Option B — Disable the compact path (1-line source change)

In `module_native_driver.sio:1171`, change `if !optimize {` to `if false {`. The compact path becomes dead code; every multi-module compile goes through the full IR path. Smallest possible compiler-source patch that closes the defect. Loses the (theoretical) speedup the compact path was supposed to provide.

### Option C — Implement the kind=3 emit correctly (real fix)

Modify `native_driver_emit_imported_simple_fn` at `module_native_driver.sio:517-534` to honour `VALUES[fi]`:
- For kind=3 functions whose `VALUES[fi]` was set (e.g. from `module_frontend_try_eval_imported_function_i64`), emit a data section containing the ASCII representation of `VALUES[fi]` instead of the hardcoded `"42\n"`.
- Also fix `native_driver_write_imported_simple_ir_elf:1002-1006` to populate `data_off` from per-function data instead of the constant.

This is what "Program pinning" would do if it refers to pinning the per-function `VALUES[i]` table through to the emit stage.

### Option D — Delete the compact path

Remove `load_multimodule_imported_simple_ir_global`, `native_driver_write_imported_simple_ir_elf`, `module_frontend_lower_imported_simple_global_recursive`, and the globals `MODULE_FRONTEND_IMPORTED_SIMPLE_*`. This is the cleanup that follows once Option B has been in production long enough to confirm nothing depends on the compact path's existence. Not for this phase.

## 7. Recommended sequence

1. **Verify Option B in an isolated worktree** — change one line, rebuild Madaros, rerun the witness matrix. If green, the localisation is confirmed and the full IR path is exonerated.
2. **Do NOT merge Option B alone.** The compact path's existence suggests someone still wants its speedup. Bring Option C as the real fix.
3. **Run the canonical + recert matrix against Option C** to confirm both fixed cases and no regressions on the existing test suite.
4. **Then promote the witness to a permanent CI gate** per the canonical receipt's `promotion_path` field.

## 8. Reproducibility commands

```bash
# Compact path (BUGGY):
/workspace/.../bin/madaros-linux-x86_64 examples/projects/hello_pkg/src/main.sio -o /tmp/bad.elf
/tmp/bad.elf   # → "42"

/workspace/.../bin/madaros-linux-x86_64 examples/projects/hello_pkg/src/main.sio -o /tmp/bad.elf -O
/tmp/bad.elf   # → "42" or "999" depending on greet.sco content (CORRECT)
```

The presence of `-O` is the single switch that determines whether the bug manifests.

## 9. Boundary compliance

- [x] No compiler source modified during this localisation (the +60 line instrumentation patch was reverted after build issues; the localisation was achieved via existing `SOUNIO_DUMP_ALL_CALLS=1` env var and `-O` differential testing)
- [x] No binaries, resolvers, Makefile, CI, or governance files modified
- [x] No claim that any specific merge/finalize/promote function is faulty (those are exonerated)
- [x] Four canonical receipts untouched
- [x] Two recert receipts untouched
- [x] IR_CAPTURE_POINTS_MAP.md preserved (its prime-suspect section is retracted by this report but the capture-point coordinates remain valid for future instrumentation)
- [x] No merge to main
- [x] No prebuilt refresh
- [x] No claim of vertical milestone concluded

The implementing agent now has:
- A one-line repo change to verify the localisation (Option B)
- A surgical target for the real fix (`module_native_driver.sio:517-534` and `:1002-1006`)
- A clean acceptance criterion (the witness matrix going green via the compact path, not just the full IR path)

The "Program pinning" lane, if it refers to preserving per-function pinning through the compact emit stage, should target `native_driver_emit_imported_simple_fn` and `native_driver_write_imported_simple_ir_elf` directly — not `module_frontend.sio`.
