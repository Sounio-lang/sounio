<!-- docs:meta
topic_id: repo.docs.audit.windows-assert-a64-parity.str-eq
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.windows-assert-a64-parity.str-eq
-->

# A64 PARITY — `str_eq` builtin

**Opened / closed.** 2026-05-21.
**Status.** RESOLVED — CODE CHANGE LANDED.
**Class.** Codegen on `self-hosted/compiler/lean_single.sio` (new
`emit_str_eq_regs_a64` + dispatch in `compile_primary_a64`).
**Branch.** `feat/windows-assert-exit`.
**From the scan.** 2nd of the 4 confirmed string/char builtin gaps (after
`print_char`).

---

## §1 — The gap

`str_eq(a, b)` (byte-compare two NUL-terminated strings → bool) existed on x86
(`emit_str_eq_regs_x86`) but was absent from `compile_primary_a64` — unknown
identifier on `aarch64-*`, silently false.

## §2 — The fix

Added `emit_str_eq_regs_a64` (x1 = ptr a, x0 = ptr b; result in x0): a byte loop —
`ldrb w2,[x1]` / `ldrb w3,[x0]` / `cmp w2,w3` → `b.ne` 0; `cmp w2,#0` (both bytes
equal and NUL ⇒ strings equal) → `b.eq` 1; else advance both cursors and loop.
Branches use `patch_branch_a64`; the backward loop branch uses an inline
`(loop_off-CL)/4` rel like `emit_str_len_a64`. Added the `str_eq` dispatch
mirroring x86: compile arg-a, push, compile arg-b, `pop_x1` (a), compare,
EXPR_TY=4 (bool).

## §3 — Verification

- **Self-host fixed point.** PASS: stage1==stage2==stage3,
  `md5=62228c90a78bec615b02438545613c66`; binary rebuilt.
- **Real Apple M3** (`aarch64-macos`): the 5 cases `("hello","hello")`,
  `("hello","world")`, `("ab","abc")`, `("","")`, `("abc","ab")` print `10010`,
  byte-matching x86 (equal / unequal / prefix-shorter / both-empty / prefix-longer).
- **x86 non-regression.** `text_interpolate`, `test_syscall_ffi` (both use
  `str_eq`) and `epistemic_hessian_transcendentals` exit 0.

## §4 — Remaining from the scan

`str_concat` and `str_slice` are the last two confirmed a64 builtin gaps; both
need `mmap` + a copy loop (like `read_line`/`read_file`).
