<!-- docs:meta
topic_id: repo.docs.audit.windows-assert-a64-parity.audit
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.windows-assert-a64-parity.audit
-->

# AUDIT — `emit_assert_fail_a64` parity with the Windows assert-exit fix

**Opened / closed.** 2026-05-21 (same session).
**Status.** RESOLVED — NO CODE CHANGE REQUIRED. The ARM64 abort path was
already correct; the audit produced empirical proof, not a fix.
**Predecessor.** `74543ca7c` / `8c6631a2a` (#178) — "windows:
target-dispatch `emit_assert_fail` exit (Tier-2; was hardcoded Linux
syscall 60)". That commit fixed the **x86** abort path.
**Class.** Audit only. No edit to
`self-hosted/compiler/lean_single.sio`.
**Branch.** `feat/windows-assert-exit`.

---

## §1 — Question

The Windows assert-exit work target-dispatched the **x86** `emit_assert_fail`.
Does the ARM64 counterpart `emit_assert_fail_a64` need the same treatment?

## §2 — The two gaps the x86 commit fixed, mapped onto ARM64

The x86 commit fixed two distinct problems. Both are **structurally absent**
on the ARM64 path:

| x86 gap fixed | ARM64 status |
|---|---|
| Hardcoded `exit(60)` for *every* OS (raw Linux syscall, wrong on a real PE host and wrong on macOS) | Already target-dispatched — see §3. |
| Hardcoded `jnz +12` skip-jump assumed a fixed 12-byte abort; broke once the abort became target-variable size, so it was changed to a backpatched `rel8`. | Every ARM64 caller already backpatches via `patch_branch_a64(site, CL)` — it never hardcoded an offset. See §4. |

## §3 — Reachable ARM64 targets (no `aarch64-windows`)

`--target` accepts exactly (CLI parse at `lean_single.sio:32291`):

| triple | `TARGET_OS` | `TARGET_ARCH` |
|---|---|---|
| `x86_64-windows` | 3 | 0 |
| `x86_64-macos`   | 2 | 0 |
| `aarch64-macos`  | 2 | 1 |
| `aarch64-linux`  | 0 | 1 |
| *(default)* `x86_64-linux` | 0 | 0 |

**There is no `aarch64-windows` target.** Windows (`TARGET_OS==3`) only ever
pairs with x86 (`TARGET_ARCH==0`), which uses the x86 `emit_assert_fail`.
So the ExitProcess / PE concern that drove the x86 fix is N/A for ARM by
construction; `emit_assert_fail_a64` only has to serve `aarch64-macos` and
`aarch64-linux`, and it already dispatches both:

```
fn emit_assert_fail_a64() with Mut {     // lean_single.sio:25062 (feat branch)
    em32(0xD2800020)                     // movz x0, #1   (exit code)
    if TARGET_OS == 2 {
        emit_imm64_reg_a64(16, 1)        // movz x16, #1  (macOS BSD exit)
    } else {
        em32(0xD2800000 | ((93 & 0xFFFF) << 5) | 8)  // movz x8, #93 (Linux exit)
    }
    emit_syscall_a64()                   // svc #0x80 (macOS) / svc #0 (Linux)
}
```

`emit_syscall_a64` dispatches the SVC immediate (`svc #0x80` macOS vs
`svc #0` Linux). `emit_imm64_reg_a64(16, 1)` emits a single `movz x16, #1`
(higher halfwords are zero), so both aborts are exactly 12 bytes — though
that symmetry is incidental: callers backpatch regardless of size.

## §4 — Callers already backpatch

All eight `emit_assert_fail_a64()` call sites (assert keyword, slice/bounds
checks, etc.) precede the abort with a conditional branch whose target is
fixed up *after* emission:

```
... cmp ...
let patch = CL
em32(0x54000001)        // b.<cond> placeholder
emit_assert_fail_a64()  // size doesn't matter — branch is backpatched below
patch_branch_a64(patch, CL)
```

This is the ARM64 equivalent of the x86 commit's "hardcoded `jnz +12` →
backpatched rel8" change, and it was already in place.

## §5 — Empirical verification

Built stage1 from `feat/windows-assert-exit` source
(`md5 479fad32fbb6521d9b438906f5a7c794`, **identical** to the shipped
`bin/souc-linux-x86_64`), compiled `assert(1 == 2)` to both ARM targets, and
disassembled the abort (llvm-objdump; cross-checked by raw little-endian
word decode):

**aarch64-linux** (`file`: ELF aarch64):
```
cmp  x0, #0x0
b.ne <skip>          ; falls through to abort iff assert fails
mov  x0, #0x1        ; d2800020
mov  x8, #0x5d  (93) ; d2800ba8
svc  #0              ; d4000001
```

**aarch64-macos** (`file`: Mach-O arm64):
```
cmp  x0, #0x0
b.ne <skip>
mov  x0,  #0x1       ; d2800020
mov  x16, #0x1       ; d2800030
svc  #0x80           ; d4001001
```

Both decode bit-for-bit to the expected `exit(1)` sequences. **PASS on both
reachable ARM targets.**

## §6 — Conclusion

`emit_assert_fail_a64` is at full parity with the post-fix x86
`emit_assert_fail` for every target the ARM64 backend can produce. No source
change is warranted. This document is the durable record of that finding.
