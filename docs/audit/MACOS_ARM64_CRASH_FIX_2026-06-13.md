<!-- docs:meta
topic_id: repo.docs.audit.macos-arm64-crash-fix-2026-06-13
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.macos-arm64-crash-fix-2026-06-13
-->

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
`0x910003BF`) and carried a comment documenting exactly this hazard — the original
fix was commit `6d6231c02` ("compiler: restore arm64 epilogues from frame pointer").

Provenance (verified by opcode count + binary-blob OID across commits):
- The **source** lost `mov sp, x29` at `1b719e5f8` ("Merge origin/main into
  feat/i128-modular") — a merge-conflict resolution silently swapped both epilogues
  to `add sp, sp, x9`. (3×`mov sp,x29` → 0; 1×`add sp,sp,x9` → 3.)
- The committed **arm64 binary** stayed byte-identical (blob `dd655ca0`) from
  `a5ab12395` through `9b53bb8d4`, so the stale-but-correct binary kept working —
  the merge did NOT ship a crashing binary.
- The crash first surfaced when `65dc60cd8` **rebuilt** the binary from the
  regressed source (blob `988b8473`). Bug 1 then masked Bug 2 until `c95690b2f`
  let execution reach Pass 2.

Fix: restore `mov sp, x29` in both epilogues. Verified: new arm64 binary has 1698
epilogues using `mov sp, x29`, **zero** `add sp, sp, x9` (the only remaining
`add sp,sp,x9` is `emit_drop_call_args_a64`, a post-call arg pop, not an epilogue).

## Latent a64 merge-nop gap (hardening, fixed in a follow-up commit)

The x86 backend emits a merge-point `nop` (`em(0x90)`) after an if-without-else
(`8f0443a13`, line 20705) and after match arms (`998112bb8`, line 21689) so the
implicit-return guard (`last word == ret → skip epilogue`) cannot misfire. The a64
backend had **no** `0xD503201F` counterpart — and never did, including in the
working baseline (so it did not discriminate the crash; it was latent hardening).

Ported the a64 nop (`em32(0xD503201F)`) to the three a64 merge points identified
by workflow `wf_c68b673c-635` (Agent 3), each mirroring the x86 fix:

- if-without-else, after `patch_branch_a64(cbz_off, CL)` — ~line 31507
- if-let without else, after `if il_cbz_off > 0 { patch_branch_a64(il_cbz_off, CL) }` — ~line 31434
- match arms, after `while ai < arm_count { patch_branch_a64(match_ends[…], CL) }` — ~line 32050

Verified: a probe with `if x>0 { return 1 } x+10` and a `0 => { return 100 } _ => 200`
match emits the nop immediately before the `mov sp, x29` epilogue at each merge, and
runs to the correct result (311). Only a64 emission changed, so the x86-target output
is byte-identical and the bootstrap fixed point (gen2 == gen3) still holds.

**Known mirror gap (not fixed):** the x86 *if-let* no-else merge (line 20573) also
lacks its `em(0x90)` — the same latent defect on the x86 side. Left untouched to keep
the working x86 bootstrap seed's x86-target output unchanged; tracked as follow-up.

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
