<!-- docs:meta
topic_id: repo.docs.audit.madaros-string-concat-2026-06-24
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.madaros-string-concat-2026-06-24
-->

# Madaros string concatenation — working (2026-06-24)

*Branch off `main`. `"a" + "b"` and `s ++ t` on strings now produce the concatenated string;
previously string `+` lowered as an integer add of the two pointers (garbage), and the
`str_concat` builtin crashed when called.*

## Two fixes

### 1. Lowering: route string `+`/`++` to the `str_concat` builtin (`lower.sio`)
`OpAdd`/`OpConcat` fell through to a generic integer binop. Added string detection
(`expr_result_is_string_ref`: a string literal, a string-typed local tracked with the new
`scalar_kind = 3`, or a nested string concat) and, when either operand is a string,
`lower_binary_expr_ref` emits a call to the `str_concat` builtin instead of an integer add.
Numeric `+` is unaffected.

### 2. Codegen: the `str_concat` builtin was never reachable (`codegen_x86_linux.sio`)
Even a direct `str_concat("ab","cde")` crashed *before* the builtin's first instruction — the
`call` resolved to a bad address. `str_len` (also a builtin) worked because it is emitted via
the `*mut` `emit_builtin_str_len_into`; `str_concat` fell through to
`native_v2_emit_builtin_by_id_into` → `(*nc) = emit_builtin_str_concat((*nc))`, a **full
by-value `NativeCompiler` replacement** that the large-aggregate miscompile corrupts
(dropping `fn_offsets`), so the function's recorded offset was wrong and every call to it
jumped to garbage. Fixed by routing `str_concat` (builtin id 10) through
`native_v2_persist_builtin_emit_into` — the same selective code+relocs copy `str_eq` already
uses — preserving `fn_offsets`.

This is the **same by-value-aggregate miscompile** behind the struct/method/closure/method-
param roots; the established workaround (avoid the by-value copy of a large aggregate) applies
again.

## Verified (madaros from this source)
- `str_len(str_concat("ab","cde")) → 5` (builtin reachable + correct).
- `"ab"+"cde" → abcde`; `let a="foo"; a+"bar" → foobar`; string vars `s+t → xy`;
  nested `(a+b)+(c+d) → abcd`.
- No regression: numeric `40+2 → 42`, `str_len("abc") → 3`; 47/80 run-pass = prebuilt main
  +6, 0 regressed; madaros self-builds.

## Honest scope
- String detection covers string literals, string-typed locals (`scalar_kind 3`), and nested
  string concats. A string returned from a non-builtin function and then concatenated may not
  be detected (the local's kind would not be 3); such a value falls back to integer add.
- Array `++` (a true array concat, not via `str_concat`) is still unimplemented — `OpConcat`
  on non-string operands falls through to the (failing) integer binop. Separate gap.

## AI disclosure
Fix by AI agent (Claude) under human direction; the builtin root cause was localised by
instrumenting `emit_builtin_str_concat` with staged `exit()` markers (the call crashed before
the entry marker → bad call target → corrupted `fn_offsets`). Every claim backed by a
re-runnable probe.
