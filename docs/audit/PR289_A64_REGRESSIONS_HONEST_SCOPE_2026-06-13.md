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

## Methodology note

The prior session that produced these fixes used a workflow + subagent structure with
model routing: ~7 Sonnet subagents for the lower-stakes fan-out (git archaeology, patch
application, authorship/co-author attribution, paper-reference checks, triage) and Opus
reserved for the delicate compiler surgery and verification. This split saved tokens
significantly. Recorded here as the method of record; this doc itself was written in the
follow-up push+PR session and does not re-run that fan-out.
