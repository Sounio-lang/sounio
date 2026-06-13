# macOS arm64 self-hosted compiler crash — root cause & fix (2026-06-13)

Branch: `fix/silent-typecheck-diag` · Fixes at `c95690b2f`, `e55d11729`,
`f32f11faa`, `fe4827b72`.

## Symptom

The rebuilt `artifacts/self-hosted/souc-self-hosted-arm64-macos` crashed on macOS
arm64 (Apple Silicon) on every input — 0/4 compile proof, 0/95 runtime proof —
while the May-17 baseline (`a5ab12395`, 1,840,592 B) worked. Multiple distinct
bugs were stacked; each fix unmasked the next. All four share one provenance
pattern: a64-only codegen fixes silently dropped in cross-branch merge-conflict
resolutions, while the x86 sibling kept the fix.

| # | Bug | Fix | Surfaced by |
|---|-----|-----|-------------|
| 1 | FN_RET_TY_TOK splat init (`MOV X13,X0`→`X2`) | `c95690b2f` | first report (SIGSEGV) |
| 2 | epilogue sp restore (`add sp,sp,x9`→`mov sp,x29`) | `e55d11729` | Mac round 1 (SIGBUS in compiler) |
| 3 | a64 fall-through merge nops (latent hardening) | `f32f11faa` | workflow Agent 3 |
| 4 | fn-type param not marked closure → emitted-binary fn-ref call doesn't untag | `fe4827b72` | Mac round 2 (SIGBUS in emitted binary) |

Bugs 1–3 are in how the compiler itself runs; bug 4 is in the **code the compiler
emits** for user programs that pass functions as parameters.

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

## Bug 4 — fn-type parameter not marked closure (fixed in `fe4827b72`)

Surfaced by macOS arm64 hardware (round 2): the compiler now runs to completion
and arm64 compile-proof is 4/4, but the **emitted** arm64 Mach-O SIGBUSed at
runtime — `PC = 0x…255` (misaligned, bit 0 set), `x9 = tagged thunk pointer`.

A fn-typed value is materialized as a TAGGED thunk pointer (`adr x0,thunk` +
`orr x0,x0,#1`, bit 0 = fn-ref). The indirect-call dispatch picks one of two
paths by `call_is_closure`:
- Path A (`tbnz w0,#0; …; sub x9,x0,#1; blr x9`) — untags before branching;
- Path B (`mov x9,x0; blr x9`) — branches verbatim.

A tagged value reaching Path B jumps to an odd address → misaligned PC → SIGBUS.
The a64 Pass-2 **parameter** scan failed to mark fn-type params (`SCAN_TY == 9`)
as `VAR_IS_CLOSURE`, so a callee like `fn apply(f: fn(i64)->i64, v)` calling
`f(v)` took Path B. The x86 param scan already had the marking ("fn-type
parameters are always called with closure ABI"); the a64 line was dropped.

Fix: restore `if SCAN_TY == 9 { VAR_IS_CLOSURE[(VAR_COUNT-1)] = 1 }` in the a64
param scan. Verified: `closure_fn_ref` now emits 6 `tbnz` untag dispatches (was
0); `apply(inc,41)` → 42; only a64 emission changed (x86-target byte-identical).

## Known separate issue — x86_64-macos cross-compile from an arm64 host

The full acceptance gate also runs an **x86_64-macos** compile leg, which fails
on the Mac with a controlled error (exit 1, `refined: N in 2 passes` diagnostics)
and fail-fasts the gate before the arm64 runtime-proof phase. This is **not** one
of the four fixes above and not caused by them: the *x86 host* compiler emits
x86_64-macos Mach-O cleanly (verified on Linux), so it is an arm64-host-specific
gap in the x86_64 emission path — orthogonal to the arm64-target work. Tracked
separately; to verify the arm64 runtime in the meantime, bypass it (below).

## Rebuild chain (deterministic)

```
fixed lean_single.sio
  → x86_64 seed                (souc-self-hosted-x86_64, gen2==gen3 fixed point ✓)
  → arm64 cross-compile        (souc-self-hosted-arm64-macos, 1,917,028 B)
```
arm64 sha256: `80b46c56f827cc7e08b556b94d08c7c66e4f0c375eac2d3cf83a739506324e6d`

## Mac verification

```bash
git fetch origin fix/silent-typecheck-diag && git checkout fix/silent-typecheck-diag && git pull
shasum -a 256 artifacts/self-hosted/souc-self-hosted-arm64-macos
#   expected: 80b46c56f827cc7e08b556b94d08c7c66e4f0c375eac2d3cf83a739506324e6d
codesign --force -s - artifacts/self-hosted/souc-self-hosted-arm64-macos

# Decisive test for bug 4 — compiles AND RUNS emitted arm64 binaries (incl. closures):
SOUC_NATIVE=artifacts/self-hosted/souc-self-hosted-arm64-macos \
  bash scripts/selfhost/selfhost_native_runtime_proof.sh ; echo "runtime-exit=$?"

# arm64-only compile proof (skips the unrelated x86_64-macos leg):
TARGETS=aarch64-macos \
  SOUC_NATIVE=artifacts/self-hosted/souc-self-hosted-arm64-macos \
  bash scripts/selfhost/selfhost_macos_compile_proof.sh
```

Success = `SELFHOST_NATIVE_RUNTIME_PROOF_SUMMARY pass=95 … fail=0` (vs the prior
SIGBUS on every emitted binary). If anything still faults, the frame diag still
applies:

```bash
lldb --batch -s scripts/selfhost/macos_arm64_frame_diag.lldb \
  -- artifacts/self-hosted/souc-self-hosted-arm64-macos \
     tests/selfhost-driver-output/expr_add_7_35.sio /tmp/out.macho --target aarch64-macos
```
