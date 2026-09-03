# Compiler bug discovered during LSP bringup (W1 day 1)

## Symptom

`/tmp/sounio-lsp` (compiled from `self-hosted/lsp/server.sio`) segfaults
(SIGSEGV, exit 139) **inside `read_line()`** the first time it is called,
*before* any stdin byte is consumed.

## Minimum repro

Identical Sounio source except for two string literals:

- `sb6.sio` — log messages are 23 chars (`"dbg: about to read_line"`). **SEGFAULT.**
- `sb7.sio` — log messages are 5 chars (`"about"`, `"after"`). **PASSES.**

```bash
./bin/souc compile artifacts/lsp_bringup/sb6.sio -o /tmp/sb6
printf 'hi\n' | /tmp/sb6   # exit 139
./bin/souc compile artifacts/lsp_bringup/sb7.sio -o /tmp/sb7
printf 'hi\n' | /tmp/sb7   # exit 0
```

Both files have:

- 2 × `var X: [i8; 262144] = [0; 262144]` BSS arrays
- 5 × `var X: i64 = 0` scalars
- `fn log_stderr_nl(msg: string)` issuing two `syscall6(1, 2, ...)` writes
- `fn read_headers()` which calls `log_stderr_nl` then `read_line()` inside
  a `while`

## Bisect findings

1. With **either** 0 small i64 globals **or** short log strings: PASSES.
2. With 5 small i64 globals **and** a single long string literal in the
   function that invokes `read_line()`: SEGFAULTS.
3. Removing the long string but leaving the layout: PASSES.
4. Removing the small globals but keeping the long string: PASSES.

So the bug requires the conjunction of:
- Mixed BSS layout (big i8 arrays interleaved/followed by scalar i64s), **and**
- A "long" (≥ ~20 char?) string literal referenced from inside the
  function that calls `read_line()`.

Smells like a constant-pool/string-rodata addressing bug where the
displacement clashes with the BSS layout the binary emits — perhaps the
inline cstr emission paths near `lean_single.sio:1349 emit_inline_cstr_*`
or the BSS layout in `lean_single.sio` around line ~22618.

## Impact on LSP

The W2 server (`self-hosted/lsp/server.sio`) needs many long string
literals (LSP method names like `"textDocument/semanticTokens/full"`,
JSON capability fragments). Cannot ship until this bug is fixed.

## Suggested investigation

- Run `souc compile sb6.sio -o /tmp/sb6 --show-types` and diff against
  `sb7.sio` to compare emitted layout.
- Disassemble `/tmp/sb6` near the call site of `read_line` and check the
  `lea rdi, [string_literal_addr]` / `mov` sequences right before the
  syscall — likely the literal address is being miscomputed when global
  scalars push the rodata anchor.
- Suspected file: `self-hosted/compiler/lean_single.sio`, the inline cstr
  emitters at lines 1349-1399 and the BSS / globals layout pass.
