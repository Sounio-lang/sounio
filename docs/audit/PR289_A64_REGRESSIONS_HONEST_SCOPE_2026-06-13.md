# PR #289 — arm64 self-host codegen regressions: honest scope

**Date:** 2026-06-13
**Branch:** `fix/silent-typecheck-diag` → `main`
**PR:** https://github.com/Sounio-lang/sounio/pull/289 (supersedes already-merged #280; new PR because #280's branch slice is merged)

This document declares — without masking — exactly what PR #289 contains, what it
deliberately leaves out, and the future work that remains PENDING. It exists so the
scope is recorded in-repo, not only in the PR description.

## What landed (in scope)

28 commits, **a64-emit + self-host codegen correctness only**. These restore real
compiler regressions affecting `main` and the modular **Madaros** compiler in
production — they are not tese-only.

Core a64 codegen fixes:

- nested-aggregate field copy anchored at `dst_start` (re-land of 47c1a4246, silently
  reverted by a cross-merge) — `b52fd2233`
- closure/fn-ref calling convention for >7 args — `71a55a936`
- calling convention for calls with >8 args — `47fe1588e`
- f64 comparison compared RHS to itself (`fcmp d1,d1`) — `184ecd6d1`
- variance-shadow branch used x86-style placeholder dword — `69e0b85a4`
- full closure/lambda codegen restored (was a `mov x0,#0` stub) — `fcbf1ac98`
- local array subscript clobbered index reg via `emit_load_var` — `81484b1be`
- fn-type parameters marked as closures so calls untag the fn-ref — `fe4827b72`
- fall-through merge nop ported to a64 if/if-let/match merges — `f32f11faa`
- frame-pointer `sp` restore in function epilogues — `e55d11729`
- fill-value register in global array init — `c95690b2f`
- tagged arm64 fn-ref dispatch restored (lost via a main merge) — `65dc60cd8`

Plus: macOS runtime proof (codesign emitted binaries, `--debug-fn-map` on the arm64
emit path, `arm64_host_symbols.sh` recovery), rebuilt self-host artifacts reflecting
the calling-convention fixes, the Stage1-compiler "Madaros" branding, and the
docs/audit root-cause writeups.

## Out of scope (deliberately excluded, not masked)

- The large **uncommitted `fix/silent-typecheck-diag` working tree** (~834 insertions
  across `check.sio`, `items.sio`, `parser/ast.sio`, plus a rebuilt `bin/souc`) is real
  in-progress checker/parser diagnostics work. It is intentionally left uncommitted and
  is **not** part of this PR. Pushing the committed history did not touch it.

## Future work — PENDING

- Lean proof offload (descarga das provas Lean)
- GPU PBPK14 Tsit5
- Clinical cohort (coorte clínica)
- Identifiability (identificabilidade)
- Model-form / RK4 (G-α-δ, stiff integrator)
- `Seq<T>` / kaxi

## Independent verification (2026-06-14, 12-agent workflow)

A Sonnet fan-out (git archaeology / triage / authorship) + Opus adversarial
verification was run over the branch. Headline: **the commit diffs are sound for what
they claim; the pass found one newly-observable real bug and several disclosure gaps —
recorded here unmasked.**

Opus adversarial verdicts on the 5 most delicate codegen commits:

| commit | claim | verdict |
| --- | --- | --- |
| `b52fd2233` nested-agg copy anchor | dst_start anchor correct | **SOUND** — proven against the x86 oracle slot-for-slot |
| `e55d11729` epilogue `mov sp,x29` | sp restored without corruption | **SOUND** — x29-reserved-as-frame-pointer invariant verified |
| `71a55a936` >7-arg closure convention | reg/stack placement correct | **SOUND** — no refutation found, composes with 47fe1588e |
| `184ecd6d1` f64 `fcmp d0,d1` fix | f64 comparisons correct | **UNSOUND (unmasks real bug)** — see below |
| `fcbf1ac98` closure-stub restore | matches x86 semantics | **UNSOUND (pre-existing edge case)** — see below |

**`184ecd6d1` — newly-observable real bug (recommended follow-up, NOT a blocker on the
fix itself):** the fcmp fix is correct, but it unmasks a pre-existing latent defect: the
`f64_gt` **method builtin** (`lean_single.sio:29423`) uses cset `le` (`0x9A9FC7E0`,
computes `a<=b`) where it should use `hi` (`0x9A9F97E0`, `a>b`). Under the old buggy
`fcmp d1,d1` the flags were always EQUAL so `f64_gt` always returned `1`; with real
flags it now returns `a<=b` — systematically wrong for `.gt()`/`f64_gt` method-form
calls on a64. The operator-form `>` is correct (uses `hi`); only the method builtin is
affected. Requires a SLURM rebuild to verify, so deferred to follow-up, not fixed here.

**`fcbf1ac98` — pre-existing edge case, re-landed verbatim (not introduced by this PR):**
annotated-braceless closures (e.g. `|x| -> i64 x + 1`) emit a spurious `mov x0,#0` in
the a64 epilogue when `LAST_STMT_HAS_VALUE` is stale, clobbering the return value. The
x86 path has no such zeroing. Faithfully copied from baseline `a5ab12395`; braced and
unannotated closures are safe (which is why the green tests don't expose it).

All other commit groups (float-and-array codegen, closure/fn-ref dispatch, epilogue,
macOS runtime proof, artifacts+branding) reviewed **genuine-fix** or **docs-or-tooling**
with instruction encodings mechanically decoded and verified.

### Disclosure gaps the verification flagged (unmasked)

- **Commit count is 30, not 28** — `a1f767dfb` and `fc30bc56c` post-date this doc's
  original count. (PR title still says 28.)
- **Authorship trailers:** 28/30 commits carry no `Co-Authored-By` trailer though the
  work was AI-produced; repo convention on `main` uses Anthropic trailers. `fe4827b72`
  carries a malformed `Co-authored-by: macOS arm64 hardware verification (Apple Silicon,
  macOS 26.6)` — not a valid identity, and it implies hardware verification that did not
  occur (the Mac target was ssh_unreachable; verification was via qemu-aarch64-static).
- **Misleading provenance:** `b52fd2233`'s commit message + `A64_NESTED_AGG_COPY_REGRESSION`
  doc blame merge `9b53bb8d4` for the silent revert; static archaeology shows the real
  culprit is merge `5f1e397a2` (kept the broken side of the conflict).
- **Stale `.syms`:** the shipped `souc-self-hosted-arm64-macos.syms` was regenerated
  before two later artifact rebuilds — all its lldb breakpoint addresses are now wrong.
- **Scope creep:** `f8b7dc215` (Madaros branding, 19 files) and the `--debug-fn-map` /
  `arm64_host_symbols.sh` debug tooling arguably belong in separate PRs. The Madaros
  `g6` identity gate can trigger a heavy local `make build-madaros` on a fresh checkout
  (SLURM-directive risk).
- **Binary artifacts** are committed and cannot be statically verified; Mac arm64
  runtime correctness remains author-reported (qemu), not hardware-proven.
- A documented-but-unfixed regression (`67107e250`, x86_64-from-arm64-host E200 flood)
  rides in this PR, out of scope.

## Methodology note

The prior session that produced these fixes used a workflow + subagent structure with
model routing: ~7 Sonnet subagents for the lower-stakes fan-out (git archaeology, patch
application, authorship/co-author attribution, paper-reference checks, triage) and Opus
reserved for the delicate compiler surgery and verification. This split saved tokens
significantly. Recorded here as the method of record; this doc itself was written in the
follow-up push+PR session and does not re-run that fan-out.
