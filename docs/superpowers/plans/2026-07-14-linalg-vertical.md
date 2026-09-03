<!-- docs:meta
topic_id: repo.docs.superpowers.plans.2026-07-14-linalg-vertical
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.superpowers.plans.2026-07-14-linalg-vertical
-->

# Linalg (matnm) Hardening — Implementation Plan

> Execute task-by-task; compile-and-run is the gate (`check:OK` ≠ runs).

**Goal:** Make `stdlib/linalg/matnm.sio` importable + run-proven with a matrix printer, so a program can do matrix algebra and print matrices/solutions. No compiler changes.

**Compiler workarounds:** wildcard `use linalg::matnm::*`; print floats with `print`/`println` not `print_f64`; inline logic into `main` in importing programs.

**Ground rules:** never touch `self-hosted/`/`bootstrap/`; no change to existing `pub` signatures; additive (leave `matrix.sio`/`vector.sio`/broken files); EN-UK; atomic commits; no AI attribution.

Spec: `docs/superpowers/specs/2026-07-14-linalg-vertical-design.md`.

## Task 1 — Header note + verify import/run
- [ ] Add a usage note to the top comment of `stdlib/linalg/matnm.sio` (wildcard import; `print`/`println` not `print_f64`; inline in importing mains; ref multimodule audit).
- [ ] Prove import+run: scratch `use linalg::matnm::*` main building A=[[2,1],[1,3]] and `println(matnm_det(a))` → 5, exit 0.
- [ ] `check` stays green. Commit.

## Task 2 — `matnm_show`
- [ ] Append to `matnm.sio` (before its `main`): `matnm_show(label: string, m: MatNM) with IO, Mut, Div, Panic` — print `label (rows×cols):\n`, then for each i in 0..rows print each j in 0..cols `print(matnm_get(m,i,j)); print(" ")`, `print("\n")`. Uses private `m.rows`/`m.cols`.
- [ ] `check` green; smoke via importing driver showing a 2×2 → compile+run, exit 0. Commit.

## Task 3 — Run-proof driver
- [ ] `tests/stdlib/linalg/test_matnm_stdlib.sio` (inline in `main`, wildcard import). Assert on `matnm_get` (tol 1e-6; 1e-5 for solve/inverse):
  - identity(3): [0,0]=1, [0,1]=0.
  - A=[[2,1],[1,3]]: det=5, trace=5, norm_fro=√15≈3.872983.
  - inverse: A·A⁻¹ ≈ I (four entries).
  - solve Ax=[4,5]ᵀ: x0=1.4, x1=1.2.
  - mul [[1,2],[3,4]]·[[5,6],[7,8]] = [[19,22],[43,50]] (check all four).
  - transpose: (Aᵀ)[0,1]==A[1,0].
  - scale 2·[[1,2]] row: [0,0]=2, [0,1]=4.
  - then `matnm_show` a matrix, `print("MATNM_STDLIB_OK\n")`, return 0.
- [ ] Compile+run → `MATNM_STDLIB_OK`, exit 0. No tolerance-retrofit. Commit.

## Task 4 — Consumer example
- [ ] `examples/linalg/solve_report.sio` — build a 3×3 system, `matnm_solve`, `matnm_show` A, b, x. Compile+run, exit 0. Commit.

## Task 5 — Gate
- [ ] `scripts/linalg_gate.sh` — check `matnm.sio`; compile+run driver (grep `MATNM_STDLIB_OK`); compile+run example; end `LINALG_GATE_OK`. (`matnm.sio` is a pure library with no `main`, so it is `check`ed, not run.) Run it. Commit.

## Task 6 — Math-review + PR
- [ ] `bin/llm-offload -t math-review -p xai` on the textbook LA identities; append to `.claude/llm_offload_log.md`.
- [ ] `node scripts/docs/sync_governance_metadata.mjs`; commit governance + docs.
- [ ] Push; PR to `main`; ensure `Contracts`/`CI Decision` green; merge.
