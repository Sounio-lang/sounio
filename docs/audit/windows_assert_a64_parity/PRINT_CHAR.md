<!-- docs:meta
topic_id: repo.docs.audit.windows-assert-a64-parity.print-char
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.windows-assert-a64-parity.print-char
-->

# A64 PARITY — `print_char` builtin

**Opened / closed.** 2026-05-21.
**Status.** RESOLVED — CODE CHANGE LANDED.
**Class.** Codegen on `self-hosted/compiler/lean_single.sio` (new
`emit_print_char_a64` + dispatch in `compile_primary_a64`).
**Branch.** `feat/windows-assert-exit`.
**Found by.** The `_x86`-only helper scan (`print_char`/`str_eq`/`str_concat`/
`str_slice` were the four real language-feature gaps; this fixes the first).

---

## §1 — The gap

`print_char(n)` (write one byte by char code to stdout) existed on x86
(`emit_print_char_x86`) but was absent from `compile_primary_a64`, so on
`aarch64-*` it was an unknown identifier — compile error, no output. (a64 already
had `emit_print_char_lit_a64` for *literal* chars used internally, but no
user-facing `print_char` dispatch.)

## §2 — The fix

Added `emit_print_char_a64` — the dynamic counterpart of `emit_print_char_lit_a64`:
`mov x10, x0` (save the char code, since the write syscall clobbers x0),
`strb w10, [sp]`, then `write(1, sp, 1)` (Linux x8=64 / macOS x16=4). Added the
`print_char` dispatch to `compile_primary_a64` (integer-arg type check, EXPR_TY=5),
mirroring the existing `print_int` dispatch.

## §3 — Verification

- **Self-host fixed point.** PASS: stage1==stage2==stage3,
  `md5=ac5370b9b8a585594d68c74cad1a0f2a`; binary rebuilt.
- **Real Apple M3** (`aarch64-macos`): `print_char(72) print_char(105)
  print_char(c=90) print_char(33)` prints `HiZ!`, byte-matching x86 (literals and
  a variable argument).
- **x86 non-regression.** `epistemic_hessian_transcendentals`,
  `sensitivity_transcendental` exit 0.

## §4 — Remaining from the scan

`str_eq`, `str_concat`, `str_slice` are the other three confirmed a64 builtin
gaps (str_concat/str_slice need mmap+copy loops like read_line; str_eq is a
compare loop). First-class function calls/closures and nested field/index stores
are already handled on a64; the rest of the `_x86`-only helpers are arch
internals, the GPU subsystem (x86-host only), or Windows PE.
