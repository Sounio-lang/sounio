<!-- docs:meta
topic_id: repo.docs.handoff.apple-selfhost-release-gate-sigsegv-2026-07-10
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.handoff.apple-selfhost-release-gate-sigsegv-2026-07-10
-->

# Handoff — Apple Self-Host Release Gate SIGSEGV (arm64-macOS native backend)

**Filed:** 2026-07-10 · **Status:** OPEN · **Owner needed:** someone with Apple Silicon hardware
**Scope:** the **preview** arm64-macOS native backend only. Does **not** block main CI. Independent of any recent x86/Linux work.

---

## TL;DR

The `Release Gate` workflow's **Apple Self-Host** job has been RED on every daily run since
**2026-05-28**. The arm64-macOS *self-hosted compiler* (`souc-self-hosted-arm64-macos`)
**segfaults (SIGSEGV / "Segmentation fault: 11")** while typechecking a trivial fixture, instead
of emitting the expected diagnostic. This is a bug in the **preview Apple/arm64 native backend**
(`self-hosted/native/apple_arm64_preview.sio`), not in the shared frontend — the **Linux** native
self-host is green on the same commits with the same fixtures.

**It cannot be reproduced or fixed from an x86-linux environment** (the workspace + dev box are
both x86). It needs Apple Silicon to run the binary under `lldb`.

---

## Current failure (as of merge commit `d3f638535`, 2026-07-10)

Job step: **Release Gate → Apple Self-Host → "Native acceptance gate"** → `native typecheck proof`.

```
FAIL [undefined_var] missing expected diagnostic
--- expected pattern: E137
--- compiler output:
scripts/selfhost/selfhost_native_typecheck_proof.sh: line 28: 5885 Segmentation fault: 11  "$@"
SELFHOST_NATIVE_TYPECHECK_PROOF_SUMMARY pass=0 fail=1 skip=0
```

- Fixture: `tests/selfhost/native_typecheck/undefined_var.sio` = `fn main() -> i32 { x + 1 }`
  (`x` undefined → should emit **E137 undefined variable**).
- The arm64-macOS compiler **crashes during typecheck of an undefined-symbol lookup** rather than
  reporting E137. `fail_fast=1` → the run dies on case #1, so we don't know how many later cases
  also crash.
- Manifest: `tests/selfhost/native_typecheck/manifest.tsv` (18 diagnostic fixtures).

## History — this is long-standing, NOT a recent regression

- **Last green Release Gate:** 2026-05-28 (`b65cd4788`).
- **First red:** 2026-05-29 (`a32348c94`). Bisect the backend regression in
  **`b65cd4788..a32348c94`**.
- Last 200 Release Gate runs: **88 failure / 39 success**; continuous red since late May across
  dozens of unrelated commits.
- **The specific failing macOS sub-test has drifted** over the month (same root cause = macOS
  native compiler segfaults):
  - *2026-05-29 (earliest red):* `aarch64-macos` **and** `x86_64-macos` **compile proof** failed —
    `closure_lambda_lift` / `closure_fn_ref` compile failed (`exit=139`) via
    `scripts/selfhost/selfhost_aarch64_compile_proof.sh:20`.
  - *2026-07-10 (now):* `native typecheck proof` segfault on `undefined_var` (above).

## Why it's out of scope for x86-linux sessions

- **Linux native self-host is GREEN** on the same commits with the same E137 fixtures → the defect
  is in the **Apple/arm64 native codegen path**, not the shared frontend/IR.
- arm64-macOS is an explicit **preview** target: `self-hosted/native/apple_arm64_preview.sio`, with
  a trail of "apple preview" / "stabilize apple preview smoke path" commits.
- No Apple Silicon in the workspace or on the dev machine (both x86). A blind edit to arm64 codegen
  that can't be executed/verified would violate the project's byte-identity/verification discipline
  — do **not** attempt it sight-unseen.

## Repro (on Apple Silicon)

```bash
# after the source-bootstrap arm64-macOS artifact is present at
# artifacts/self-hosted/souc-self-hosted-arm64-macos
SOUC_NATIVE=artifacts/self-hosted/souc-self-hosted-arm64-macos \
SOUNIO_NATIVE_TARGET=aarch64-macos FAIL_FAST=1 \
bash scripts/selfhost/selfhost_native_typecheck_proof.sh

# minimal crashing input:
#   fn main() -> i32 { x + 1 }        (expects E137; currently SIGSEGVs)
# also historically crashing (compile proof, closures):
SOUNIO_NATIVE_TARGET=aarch64-macos \
bash scripts/selfhost/selfhost_aarch64_compile_proof.sh   # closure_lambda_lift / closure_fn_ref
```

Run the segfaulting compile under `lldb` to get the backtrace; the crash is in the arm64 native
codegen for undefined-symbol resolution during typecheck (and, historically, closure lambda-lift).

## Investigation 2026-07-10 (x86-linux) — localized to arm64 CODEGEN, with a local repro

Hard findings from actually attempting the fix (not just reading logs):

1. **The suspect commit is `9701710aa`** ("file:line source locator in tc_error"), which added
   `print_src_loc_for_tok(tok)` (lean_single.sio:3524), wired into `tc_error`/`tc_error_hard` and
   `tc_undefined_var` (:3919, the E137 path). **But the OOB hypothesis was FALSED by reading it:**
   every scan in `print_src_loc_for_tok` is bounds-guarded (`while scan >= 0`,
   `while marker_start >= 0 && …`, `pi < marker_end`, `nl_scan < pos`) and `pos` is range-checked
   at :3526. The shared logic is not a naive out-of-bounds.

2. **The crash is arm64-codegen-specific, PROVEN.** The arm64 binary is built by cross-compiling
   `lean_single.sio` (ci.yml:110: `souc-stage2 lean_single.sio --target aarch64-macos`). I built
   **origin/main's exact `lean_single.sio`** (37,671 lines) to a **Linux** native binary via the
   seed and ran the harness invocation (`--check undefined_var.sio`): it exits **rc=1 with no
   crash**. Same source → arm64 SIGSEGVs, x86-linux does not. ⇒ the fault is the **aarch64-macOS
   code generator mis-lowering some construct in the diagnostic/check path**, not shared logic.
   (Corollary: this is why it can't be fixed+verified from an x86-linux box — the codegen bug only
   *manifests* on arm64.)

3. **★ The arm64 binary reproduces LOCALLY via the seed — no Mac needed to BUILD it.** The seed
   cross-targets aarch64-macOS directly:
   ```
   ./bin/souc-lean-single-x86_64  <lean_single.sio>  ls_arm64_macos  --target aarch64-macos
   #  → Mach-O 64-bit arm64 executable  (macho_arm64 emit path)
   ```
   So the arm64 codegen path (`macho_arm64` / aarch64 emit functions **inside lean_single.sio
   itself**) can be disassembled on x86 with `llvm-objdump`/`objdump`. A candidate codegen fix can
   be **partially** verified statically (rebuild → disassemble → confirm correct instructions)
   before CI confirms the runtime fix. Only *running* the binary needs Apple Silicon.

4. **Second, separate observation:** lean_single's `--check` on Linux prints `error: no main` for
   this fixture and does **not** emit E137 at all — so the arm64 binary likely also fails the
   `grep E137` check independent of the crash. The green x86 typecheck-proof jobs use **`lean.sio`**
   (modular), not `lean_single.sio`, so they don't exercise this path. Worth confirming whether the
   Apple job's *real* fix needs both (a) the arm64 codegen crash fixed AND (b) lean_single `--check`
   actually emitting E137 — or whether the harness should run `lean.sio` on arm64 like x86 does.

## Update 2026-07-10 (later) — the arm64 target can't even COMPILE lean.sio (deeper than the crash)

Attempted a CI fix (PR #729): the native typecheck proof was feeding **lean_single** (no
`--check`, no E-codes) as the typecheck compiler where it should use **lean.sio** (like the x86
proof). The wiring fix is correct — but dispatching the Release Gate against the branch
(run 29103199769) **falsified its prerequisite**:

- `souc-stage2 self-hosted/compiler/lean.sio --target aarch64-macos` **fails to compile** —
  cascade of `unknown identifier` (VISIBILITY_PUB, VALIDITY_KIND_ALWAYS, FAIRNESS_CRITERION_*,
  DIST_*_EP, …) + `E001` + `no main`, exit 1. Reproduced locally (245+ errors, no artifact).
- The **x86** build of the *same* `lean.sio` (no `--target`) **succeeds** (37.6 MB, emits E137).

So the aarch64-macOS backend has **two** distinct maturity gaps, both blocking the Apple job:
1. **`lean_single` → arm64:** compiles, **SIGSEGVs at runtime** (the original crash).
2. **`lean.sio` → arm64:** **does not compile** — the target's frontend/symbol-resolution path
   chokes on the large multi-module program the x86 default target resolves fine.

Consequence: the Apple typecheck proof **cannot be greened by CI wiring alone**. It needs the arm64
backend to (a) compile `lean.sio` and (b) run it without SIGSEGV. PR #729 holds the correct wiring
as a **draft**, ready to un-draft once (a) is fixed. The `--check`/E137 diagnosis stands; it just
turned out to be gated on the same backend-maturity wall as the crash.

**Likely mechanism to investigate:** why does `--target aarch64-macos` change *symbol resolution*
(it shouldn't — resolution is pre-codegen)? The seed's aarch64 target path may route through a
leaner/divergent frontend or module-loader than the x86 default. Start at the seed's `--target`
dispatch and compare the module/pub-constant resolution path for aarch64-macos vs x86.

## Update 2026-07-10 (Apple Silicon repro session, macOS 27.0 / build 26A5368g)

Ran the CI arm64 artifact (`souc-self-hosted-arm64-macos`, run 29076751561) on a real M-series Mac
(macOS **27.0**, `ulimit -s` = 8176 KB, same ~8 MB as the CI runner). Findings:

1. **The crash does NOT reproduce on macOS 27.** `<bin> --check /tmp/uv.sio` (the exact CI
   invocation) exits cleanly rc=1 with `no main` — no SIGSEGV, no lldb stop. Also clean at
   `ulimit -s 4096`. So the gate-reddening crash is **specific to the CI `macos-15` runner**
   (loader/memory-layout/OS-version), not a deterministic fault — the CI runner is the only oracle
   for the crash itself.

2. **★ Precise miscompile signal — blank identifier name.** Positional real compile
   `<bin> /tmp/uv.sio /tmp/out` of `fn main()->i32{ x+1 }`:
   - **arm64:** `` E200 `` at line 1 `` — identifier name is **BLANK**, then `typecheck: failed`.
   - **x86 (control, same lean_single):** `` E200 `x` at line 1 `` — name **`x`** present.
   The name comes from `lean_single.sio` `compile_primary` (line ~15242):
   `var e200_i = ns; while e200_i < ne { print_char(SRC[e200_i]); e200_i += 1 }` where
   `ns/ne = TS[EP]/TE[EP]`. `ns/ne` are correct (the compiler *did* classify `x` as undefined, and
   the same `ns/ne` feed working `const_find_val`/`ne-ns==9` checks), and `print_char` works (all
   telemetry prints fine). So the arm64 codegen miscompiles **this specific loop** inside the giant
   `compile_primary` fn (~4640 lines) — almost certainly a **register-allocation/spill defect under
   high pressure in a huge function**. Benign (empty print) on macOS 27; plausibly the same bad
   read → unmapped access → SIGSEGV on macos-15.

3. **arm64-built compiler emits INVALID Mach-O.** `<bin> /tmp/ok.sio /tmp/okout` on a valid program
   (`fn main()->i32{7}`) compiles, but `/tmp/okout` → `zsh: exec format error` (ENOEXEC), even after
   `codesign -s -`. So when the `macho_arm64` emitter runs **as an arm64 binary** it produces a
   structurally-broken Mach-O — whereas the same emitter cross-built on x86 (souc-stage2) produces a
   valid one (the CI arm64 compiler itself runs). Another arm64-codegen self-inconsistency.

**Consequence for tooling:** binaries the arm64 compiler produces can't be run to verify fixes — but
binaries built by the **x86 seed** (souc-stage2 path) are valid arm64 Mach-O and DO run on-device.
And the blank-name bug is observable in the arm64 compiler's own stdout, so a fix can be verified by
re-running the compiler (no need to run its output). The crash-vs-green signal, however, only shows
on macos-15 (CI).

## Update 2026-07-10 (arm64 backend fixes — root causes found + fixed)

On-device debugging (Apple Silicon, macOS 27) with SHA-verified binary transfer isolated and fixed
**three** distinct aarch64-backend defects. The x86-64 walker (`compile_primary`) handles all of
these; the arm64 twin (`compile_primary_a64`, a ~4640-line parallel walker) did not.

1. **`print_char` builtin missing (PR #737).** The arm64 dispatcher had print/println/print_int/
   print_f64 but not print_char → blank output / SIGSEGV. Root cause of the "blank identifier name"
   (`E200 \`\``) and the native-typecheck-proof crash. Fixed: `emit_print_char_a64` + dispatcher
   handler. Confirmed on-device (P5 blank→`AAA`, P3 SIGSEGV→`xyz`).

2. **Imported const not resolved (branch `fix/arm64-multimodule-parity`).** `compile_primary_a64`'s
   bare-identifier fallthrough went straight to `tc_undefined_var` without calling `const_find_val`
   → every named const (incl. imported enum-style: VISIBILITY_PUB, FAIRNESS_CRITERION_*, DIST_*_EP)
   was a false "unknown identifier". Fixed by mirroring `compile_primary` ~15188 (int + f64 const).

3. **Effect check used unmasked FN_EFFECTS (same branch).** The a64 call handler's effect subset
   check used the callee's raw `FN_EFFECTS`; the x86 twin masks `& 1023` first. Imported/pub callees
   carry meta bits (pub=1024, import=2048) that looked like required effects → false "effect not
   declared". Fixed by masking.

**Verification:** a minimal 2-module reproducer (const + effectful imported fn) now compiles CLEAN on
aarch64 and produces a valid Mach-O; x86 program output byte-identical; 5-gen self-host fixed point
preserved. `lean_single.sio` itself (single-module) was never affected.

### Remaining: the `lean.sio` type-mismatch cascade (NOT yet fixed — dedicated effort)

`lean.sio` (multi-module, rich types) still fails to cross-compile to arm64 with ~1000+ residual
**E001 "Type mismatch in call argument"** errors. These come from the arm64 walker computing a
different `EXPR_TY` than x86 for specific expressions — the check itself (compile_primary_a64
~32342) is byte-identical to x86; the divergence is **upstream in expression typing**. The dominant
site is generic/enum/match/box typing, e.g. `lower.sio:947`:
```
match &head.fn_def {              // &Option<Box<FnDef>>
    Some(fd) => lowerer_preseed_fn_signature_mut(lo, m, &(*fd).params, ...)  // fd: &Box<FnDef>
}
```
i.e. the arm64 walker mis-types a **match-pattern binding over a generic enum** (`Some(fd)` from
`&Option<Box<FnDef>>`), so `&(*fd).params` gets a wrong ref-inner-hash → E001 at every such call.

This resists the minimal-repro methodology (simple structs/refs/fields already compile clean on
arm64; the failure needs lean.sio's internal box/generic AST). Closing it requires **direct tracing
inside the lean.sio compile** (instrument the arm64 match-binding/ref-hash path, ~2min build/iter)
and is a real multi-session parity effort — not a point fix. Branch `fix/arm64-multimodule-parity`
is the base to continue from; each fix follows the pattern: find the x86↔arm64 type-computation
divergence, mirror x86, verify byte-neutrality + fixed point + on-device.

## Suggested first steps for the Mac owner

1. Reproduce the `undefined_var` SIGSEGV under `lldb`; capture the backtrace + the faulting
   instruction in `apple_arm64_preview.sio` (or wherever the typecheck error path lands for arm64).
2. Bisect the backend break in `b65cd4788..a32348c94` (2026-05-28→29). The window mixes Erdős/SAT,
   modular-cap, and diagnostics-locator commits — the diagnostics/error-emission changes
   (`9701710aa` file:line source locator in `tc_error`, `a32348c94` drop SITE_A debug tag) are the
   most suspicious for an *error-path* crash, but confirm by bisect rather than by guess.
3. Check whether the closure compile-proof cases (`closure_lambda_lift`, `closure_fn_ref`) still
   crash too — if so, it's a broader arm64 native-codegen instability, not just the error path.

## Relationship to PR #724 (EISA native backend)

None. #724 (merged `d3f638535`) touches the **x86-linux** native backend (`codegen_x86_linux.sio`,
`encode.sio`, `frame.sio`), IR caps (`ir.sio`), the merge dedup (`module_frontend.sio`), and the
lean_single spill threshold. The Apple job was already red on the pre-merge commit `497604168` with
the identical signature. The main CI (the gate that blocks merges) is green.
