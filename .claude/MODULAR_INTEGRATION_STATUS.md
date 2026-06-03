# Modular-compiler integration — STATUS & PROTOCOL (live)

**Last updated: 2026-06-03 ~15:10 UTC.** This file coordinates the multiple
agents working the modular self-hosted compiler. Read it before pushing modular
fixes. It exists because parallel lanes were re-solving the same problems and
the trunk was a moving target.

## THE integration lane (single point of convergence)

**`codegen/nested-mut-write-fix`** is the trunk / integration lane. ALL
modular-compiler fixes land here. Do **not** fork new long-lived `modular/*`,
`g1/*`, `codegen/*`, `work/*` branches for compiler fixes — commit small,
validated changes to the lane (or a short-lived branch fast-forwarded into it
same-day). `bin/souc` on this lane is the canonical lean_single fixed point.

## Push discipline (prevents the moving-target + CPU-eviction problems)

1. `git pull --ff-only` immediately before any push (the trunk moves every few
   minutes).
2. Validate before pushing: `bash scripts/ci/canonical_compiler_gate.sh` (PASS)
   **and** an A/B `--check` over `tests/run-pass/*.sio` showing **0 regressions**
   vs the pre-change trunk. Build mc only via
   `scripts/dev/souc-build-lock.sh` (un-locked concurrent `souc main.sio` builds
   saturate CPU and trigger the k8s liveness eviction — CLAUDE.md §4; the pod has
   fallen twice this way).
3. If you re-bootstrap `bin/souc`, re-pin the fixed point
   (`bin/souc lean_single.sio /tmp/s1 && /tmp/s1 … /tmp/s2 && /tmp/s2 … /tmp/s3`,
   require `s2==s3`) and confirm the canonical gate before pushing.

## What is DONE — do NOT re-solve (verified 2026-06-03)

- **E008 over-strict return narrowing: CLEARED, 0/504** (int+float narrowing,
  sci-notation lexer, algebra/study keyword). `--check` corpus ~191+ OK.
- **Typed-closure / crasher class: closed by the trunk's `_inplace` work.** The
  `*mut` dispatch + boundary fixes (PRs #230/#228, `g1/qualify-bare-patterns`)
  are **SUPERSEDED** — they produce 0 net census change on the trunk. Close them,
  don't re-merge. (Evidence: A/B `--check` identical.)
- **native-v2 back-half (IR→ELF): WORKS.** `emit13` + 9 witnesses pass
  (`native_v2_e2e_*_gate.sh`), semantic exit-code-correct.

## The ONE remaining blocker — focus here

**SRET large-struct-return, in a CHAIN.** A function returning a large aggregate
by value triggers an epilogue `rep movsq` overrun / "caller sret pointer dropped"
(gdb-pinned), frame-sensitive. It blocks: the 72 `--check` corpus crashers AND
the source→native-v2 bridge (`--native-v2-compile`). 4 prior branches attempted
it and could not close it.

Two fix families (pick ONE, coordinate before starting — this is where effort
should go, not new parallel lanes):
- **(a) `*mut`-route the return chain** (no re-bootstrap, no brick risk; all in
  `main.sio`/`check.sio`). Top level done on `feat/native-v2-bridge-sret`
  (`compiler_preflight_ir_load_into` + BSS global); next offender pinned:
  `let ir_result = load_multimodule_ir(path)` (also returns a big struct by
  value). Continue down the chain one level at a time, rebuilding mc each step.
- **(b) lean_single codegen fix** (`COMPILE_SRET_FORWARD`/`emit_reload_sret_ptr`,
  17 partial hits already). Real root fix but re-bootstrap + multi-day +
  brick-risk. Do not open casually.

## Housekeeping

- Kill orphaned corpus-sweep shells in `/workspace/sounio-move-codegen` (worktree
  removed; ~1-day-old `souc` loops still burning CPU).
- Worktree count was pruned 36→23; keep it down (one worktree per agent).
