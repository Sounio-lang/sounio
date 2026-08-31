<!-- docs:meta
topic_id: repo.docs.audit.madaros-builtin-emission-2026-06-24
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.madaros-builtin-emission-2026-06-24
-->

# Madaros builtin emission — fix the by-value `NativeCompiler` corruption (2026-06-24)

*Branch off `main` (with string concat #422). The fallback builtin emitter corrupted
`fn_offsets`, so several builtins crashed when called; `str_slice` and `file_size` now work.*

## Root cause (same as str_concat #422)

`compile_ir_function_v2_from_ir_into` dispatches builtins: a few have dedicated `*mut`/persist
emitters (`str_len`, `str_char_at`, `str_eq`, and `str_concat` from #422), and the rest fall
through to `native_v2_emit_builtin_by_id_into`, which emitted each via
`(*nc) = emit_builtin_X((*nc))` — a **full by-value `NativeCompiler` replacement**. That large
aggregate is hit by the by-value-aggregate miscompile and **drops `fn_offsets`**, so the
function's recorded offset is wrong and every `call` to it jumps to a bad address (crash
*before* the builtin's first instruction).

`str_len` worked because it uses the `*mut` `emit_builtin_str_len_into`; `str_eq`/`str_concat`
work because they use `native_v2_persist_builtin_emit_into` (a *selective* code+relocs copy
onto the existing `nc`, preserving `fn_offsets`).

## Fix

Route **every** builtin in `native_v2_emit_builtin_by_id_into` through
`native_v2_persist_builtin_emit_into(nc, emit_builtin_X((*nc)))` instead of
`(*nc) = emit_builtin_X((*nc))`. Same proven workaround as #422, applied to the whole family.

## Verified (madaros from this source)
- **Now work (were crashing): `str_len(str_slice("hello",1,4))`, `file_size(path) -> 13`.**
  The `-> 4` recorded here for the `str_slice` call was the *defect*, not the
  expectation: `[1, 4)` of `"hello"` is `"ell"`, length **3**. This fix restored the
  call rather than crashing it, but the emitted stub still took only `(s, start)`
  and returned the whole suffix `"ello"`. Corrected while fixing #2244, which
  retags 3-argument call sites onto their own builtin; the value is now **3**.
  Nothing else on this page changes -- the emission-path fix it documents stands.
- Still correct: `str_char_at("hello",1) → 101`, `str_eq → 1`, string concat `"ab"+"cde" → 5`.
- No regression: 47/80 run-pass = prebuilt main +6, 0 regressed; madaros self-builds.

## Honest scope — what this does NOT fix
- **`read_file` still crashes (139)** — measured identical on the prebuilt main, so it is a
  **separate, pre-existing bug** in `read_file` itself (not the emission path). Out of scope.
- **`sqrt`, `starts_with`, `exp`, `log`, `sin`, `cos` fail with `E137` (parse error)** on both
  this build and the prebuilt — a **separate frontend/parser gap**, not the emission bug.
  Those builtins never reach codegen, so this fix cannot affect them.
- This fix removes the *emission-corruption* root for the fallback builtins; the remaining
  read_file / math-builtin failures are distinct roots to be triaged separately.

## AI disclosure
Fix by AI agent (Claude) under human direction; root localised by instrumenting the str_concat
builtin (the call crashed before the entry marker → corrupted `fn_offsets`), then generalised
to the whole fallback. Every claim backed by a re-runnable probe.
