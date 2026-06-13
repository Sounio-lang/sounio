# macOS arm64 self-hosted compiler crash — root cause & fix (2026-06-13)

Branch: `fix/silent-typecheck-diag` · Fixes at `c95690b2f` and `e55d11729`.

## Symptom

The rebuilt `artifacts/self-hosted/souc-self-hosted-arm64-macos` crashed on macOS
arm64 (Apple Silicon) on every input — 0/4 compile proof, 0/95 runtime proof —
while the May-17 baseline (`a5ab12395`, 1,840,592 B) worked. Two distinct bugs
were stacked; fixing the first unmasked the second.

## Bug 1 — FN_RET_TY_TOK global init (fixed in `c95690b2f`)

`emit_global_inits_a64()` emitted `0xAA0003ED` (`MOV X13, X0`) instead of
`0xAA0203ED` (`MOV X13, X2`) as the splat fill-value register. Every non-zero
splat array (e.g. `[-1; 65536]`) was initialised to its own BSS address. Pass
1c.5 (now unconditional) then called `scan_type()` on those slots → out-of-bounds
`TK[0x117…]` → **SIGSEGV at 0x100033fd0**.

Fix: one byte of encoding. Verified: 285 init sites now `mov x13, x2`, 0 of the bug.

## Bug 2 — function epilogue sp restore (fixed in `e55d11729`)

Both arm64 epilogues (explicit `return` and implicit tail return) restored sp via
`add sp, sp, x9` (frame size), which is only correct if the expression stack is
perfectly balanced at the epilogue. Any control path with a transient imbalance
(a value pushed whose pop was skipped) left sp off, so `ldp x29, x30, [sp], #16`
read two **local** words as the saved `{x29,x30}` pair and `ret` jumped to garbage.

Signature on hardware: **SIGBUS, PC = LR = small constant (0x1, 0x51), FP = garbage,
x9 = frame_size identical across different inputs.** (`blr` cannot produce LR=PC=const;
`ret` with a corrupted x30 can — and x9 being a constant frame size explains the
identical `x9=0x20` across two unrelated programs.)

The working baseline `a5ab12395` anchored sp on the frame pointer (`mov sp, x29`,
`0x910003BF`) and carried a comment documenting exactly this hazard. The
`9b53bb8d4` merge replaced it with `add sp, sp, x9` (the same merge that dropped
the fn-ref dispatch later restored by `65dc60cd8`). The crash was masked by Bug 1
until `c95690b2f` let execution reach Pass 2.

Fix: restore `mov sp, x29` in both epilogues. Verified: new arm64 binary has 1698
epilogues using `mov sp, x29`, **zero** `add sp, sp, x9` (the only remaining
`add sp,sp,x9` is `emit_drop_call_args_a64`, a post-call arg pop, not an epilogue).

## Rebuild chain (deterministic)

```
fixed lean_single.sio
  → x86_64 seed                (souc-self-hosted-x86_64, gen2==gen3 fixed point ✓)
  → arm64 cross-compile        (souc-self-hosted-arm64-macos, 1,900,644 B)
```
arm64 sha256: `af598f49b99460063981ff499e937dbf7a3a756528b1a51a866780c7faf6a812`

## Mac verification (one round-trip)

```bash
git fetch origin fix/silent-typecheck-diag && git checkout fix/silent-typecheck-diag && git pull
codesign --force -s - artifacts/self-hosted/souc-self-hosted-arm64-macos

# Full acceptance gate
SOUC_NATIVE=artifacts/self-hosted/souc-self-hosted-arm64-macos \
  bash scripts/selfhost/selfhost_native_acceptance_gate.sh

# If anything still faults, name the faulting epilogue with the frame diag:
lldb --batch -s scripts/selfhost/macos_arm64_frame_diag.lldb \
  -- artifacts/self-hosted/souc-self-hosted-arm64-macos \
     tests/selfhost/native_runtime/expr_add_7_35.sio /tmp/out.macho --target aarch64-macos
```

A clean run prints `PROCESS COMPLETED WITHOUT FAULT`. A residual fault dumps
`sp/fp/lr/pc/x9` + the 16 bytes at `[sp]` + backtrace, which names the exact
function whose frame is mis-torn-down.
