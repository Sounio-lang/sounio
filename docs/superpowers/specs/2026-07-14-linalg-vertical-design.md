<!-- docs:meta
topic_id: repo.docs.superpowers.specs.2026-07-14-linalg-vertical-design
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.superpowers.specs.2026-07-14-linalg-vertical-design
-->

# Design — Harden the linalg (matnm) vertical

**Status:** approved design, pre-implementation
**Date:** 2026-07-14
**Constraint:** No compiler changes (Madaros owned by CODEX-2). Work in `stdlib/`, `examples/`, `tests/`, `scripts/`.
**Orthography:** EN-UK.

## 1. Why
Third application of the playbook that landed the GUM (#860) and units (#873) verticals: take a module
inside the current compiler's working surface (stdout + pure compute, fixed slabs, no file/heap/argv), make
it **importable → run-proven against textbook values → gated → PR**. `linalg` is the healthiest module and
foundational for scientific computing.

## 2. Verified starting state
- `stdlib/linalg/matnm.sio` (554 lines) is the importable, green, general-purpose core: `MatNM` (dense up
  to 64×64, flat `[f64; 4096]`, stride 64), with `matnm_new/zeros/identity/get/set`, `add/sub/scale/mul/
  transpose/trace/norm_fro`, and `lu/solve/det/inv/qr`.
- It **imports + runs** as native ELF (verified: A=[[2,1],[1,3]] → det=5, inv[0,0]=0.6, solve Ax=[4,5]ᵀ →
  x=[1.4,1.2] — all textbook-correct).
- `matrix.sio`/`vector.sio` are `//@ run-pass` self-test files (ops private, only `main` is `pub`) — not
  importable; left untouched. `epistemic_matrix.sio`/`shaped.sio` fail `check` — left untouched.
- **Gap:** there is **no way to display a matrix** — no `matnm_show`/print. You can solve a system but
  cannot print the matrix or solution. This is the real-world-usability hole this work closes.
- `matnm.sio` is a pure library (no `main`); its arithmetic is **correct** (verified: `matnm_inv([[2,1],
  [1,3]]) = [[0.6,-0.2],[-0.2,0.4]]` because `A·A⁻¹ = I` exactly).
- **Compiler bug found (display-only):** `print(f64)` renders **negative** floats as `-0.000000`
  (magnitude dropped); positives are fine. The values are intact (comparisons/arithmetic correct). Filed
  as `docs/audit/MADAROS_PRINT_NEGATIVE_F64_2026-07-14.md`; `matnm_show` works around it (print sign +
  positive magnitude, inlined). This also latently affects the merged `gum_report`/`quantity_show`.

## 3. Goal
A program can `use linalg::matnm::*`, do matrix algebra (mul, transpose, det, inverse, solve), and **print
matrices/solutions**, proven by compile-and-run against textbook values, gated.

## 4. Scope
### In
1. **Verify + document** the import idiom in the module header (no signature change).
2. **`matnm_show`** — a print-based labelled matrix printer (rows × cols grid of `print(f64)`), inside
   `matnm.sio` (needs the private `rows`/`cols` fields).
3. **Run-proof driver** — textbook assertions: identity, add, scale, mul, transpose, trace, norm_fro, det,
   inverse (A·A⁻¹ = I), solve (Ax=b).
4. **Runnable gate**.
### Out
- No fix to `matrix.sio`/`vector.sio` (self-test files) or the broken `epistemic_matrix.sio`/`shaped.sio`.
- No new algorithms; expose/harden what exists. QR/LU beyond a smoke check are future work.
- No compiler edits; output stdout only.
- Math-review is **not strictly mandatory** here (plain LA is not in CLAUDE.md §10's list) but a quality
  math-review of the textbook identities is run and logged for consistency with the playbook.

## 5. Design
### 5.1 Import idiom (item 1)
Add a header note (wildcard `use linalg::matnm::*`; `print`/`println` not `print_f64`; inline logic into
`main` in importing programs — the known Madaros multi-module quirks).

### 5.2 `matnm_show` (item 2)
`matnm_show(label: string, m: MatNM) with IO, Mut, Div, Panic` — prints `label (rows×cols):` then each row
as space-separated values, newline per row. Uses the private `m.rows`/`m.cols` and `matnm_get`.
**Negative-safe**: prints `-` + positive magnitude for negatives (the `print(f64)` negative bug),
inlined (a private helper called cross-module segfaults). Print-based; no string assembly.

### 5.3 Run-proof (items 3, 4)
Driver `tests/stdlib/linalg/test_matnm_stdlib.sio`, all inline in `main`, asserts on `matnm_get` (tol 1e-6
unless noted; 1e-5 for solve/inverse round-off):
- identity(3): diag 1, off-diag 0.
- A=[[2,1],[1,3]]: det=5; trace=5; norm_fro=√(4+1+1+9)=√15≈3.872983.
- inverse: A·A⁻¹ = I (check the four entries ≈ identity, tol 1e-5).
- solve Ax=[4,5]ᵀ: x=[1.4,1.2] (tol 1e-5).
- mul: [[1,2],[3,4]]·[[5,6],[7,8]] = [[19,22],[43,50]].
- transpose: (Aᵀ)[0,1] == A[1,0].
- add/scale: ([[1,2]]+[[3,4]])·… and 2·[[1,2]] = [[2,4]].
Prints a `matnm_show` line, then `MATNM_STDLIB_OK`.

## 6. Module layout
```
stdlib/linalg/matnm.sio                       (modify: header note + matnm_show)
tests/stdlib/linalg/test_matnm_stdlib.sio     (new: run-proof driver)
examples/linalg/solve_report.sio              (new: consumer example — solve + show)
scripts/linalg_gate.sh                        (new: compile+run gate)
```

## 7. Verification
- `souc check stdlib/linalg/matnm.sio` green (unchanged public signatures).
- `souc compile … && ./elf` for driver + example (never `check` alone).
- `scripts/linalg_gate.sh` → `LINALG_GATE_OK` (checks matnm.sio + runs driver + example; does NOT run the
  pre-existing-crashing embedded self-test).
- Quality math-review logged.

## 8. Success criteria
1. A standalone program `use`s `linalg::matnm`, runs as ELF, does matrix algebra, and prints matrices.
2. Run-proof asserts textbook values (det/inverse/solve/mul/…) and passes.
3. Gate green.
4. No compiler files touched; self-test/broken files untouched.

## 9. Risks
| Risk | Mitigation |
|---|---|
| Inverse/solve round-off fails a tight tolerance | Use 1e-5 for solve/inverse (documented), tighter for exact ops. |
| `matnm_show` trips a multi-module quirk | Print-based, lives in the module, `print`/`println` only. |
| 64×64 cap surprises a user | Documented in the module header (flat [f64;4096]); out-of-range is the module's existing behaviour. |
