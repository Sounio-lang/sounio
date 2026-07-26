<!-- docs:meta
topic_id: repo.docs.superpowers.specs.2026-07-19-linalg-parity-design
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.superpowers.specs.2026-07-19-linalg-parity-design
-->

# linalg↔mpmath parity vertical — `stdlib/linalg` (matnm + eigen)

**Status:** design (approved 2026-07-19)
**Owner:** Data & Science verticals
**Reuses & extends:** the special/stats parity harness (#1210/#1218) — same
bit-exact `f64_to_bits` bridge, `stdlib/parity/emit.sio`, lean_single-locked
gate/comparator pattern — **extended to matrix I/O + reconstruction invariants**.

## 1. Goal

An honest, reproducible accuracy map of Sounio's general dense linear-algebra
operations (`stdlib/linalg/matnm.sio` — the `MatNM` N×M matrix type) plus
symmetric eigendecomposition (`stdlib/linalg/eigen.sio`), vs an arbitrary-precision
**mpmath (dps=30)** reference (numpy as an optional cross-check). Same philosophy:
measure genuine achieved error per operation, gate on it, fix root-caused defects
the map surfaces. Matrix ops with non-unique factorizations are checked by a
**reconstruction invariant**, not by comparing ambiguous factors directly.

## 2. Scope — three tiers of operations

| Tier | Operations | Comparison |
|---|---|---|
| **Scalar-output** | `matnm_det(A)`, `matnm_trace(A)`, `matnm_norm_fro(A)` | single value vs mpmath (like special/stats) |
| **Matrix-output, well-defined** | `matnm_mul(A,B)`, `matnm_transpose(A)`, `matnm_inv(A)`, `matnm_solve(A,b)` | **element-wise** vs mpmath (no ambiguity) |
| **Decompositions (non-unique)** | `matnm_lu(A)` → `LUResult{l,u,piv,sign}`; `matnm_qr(A)` → `QRResult_NM{q,r}`; `eigen_symmetric(A)` → eigenvalues+eigenvectors | **reconstruction invariant** (below) |

**Decomposition invariants** (robust to sign/permutation/order ambiguity):
- LU: reconstruct `L·U` and compare to `P·A` (P from `piv`); also check `L` lower-unit-triangular, `U` upper-triangular structurally (zeros where required).
- QR: reconstruct `Q·R` and compare to `A`; also check `Qᵀ·Q = I` (orthogonality).
- eig: eigenvalues compared as a **sorted set** to mpmath's; each eigenvector via the residual `‖A·vᵢ − λᵢ·vᵢ‖` (sign/scale-invariant if normalized).

Reconstruction (`L·U`, `Q·R`, `A·v`) is computed **in Python from the emitted
factor elements**, not by Sounio's matmul — so a decomposition bug can't be
masked by (or blamed on) `matnm_mul`. `matnm_mul` is separately tested in tier 2.

**Out of scope (this vertical):** the specialized types (`mat16`, `ematrix`/
`etensor` epistemic, `tensor16`, `sparse`), BLAS FFI paths, non-symmetric eigen,
SVD (no `matnm_svd`; `mat16_singular_values` is a different type — a follow-up).

## 3. Interfaces (read during design — the harness depends on these)

- **MatNM** (struct): `matnm_new/zeros(rows,cols)`, `matnm_identity(n)`,
  `matnm_set(m,i,j,val)->MatNM` (functional — returns a new matrix),
  `matnm_get(m,i,j)->f64`. Ops return `MatNM` or `f64`. `matnm_lu(m)->LUResult{l:MatNM,
  u:MatNM, piv:[i64;64], n, sign}`; `matnm_qr(a)->QRResult_NM{q:MatNM, r:MatNM,
  rows, cols}`. Confirm row/col-major and the `piv` permutation encoding in Phase 0.
- **eigen_symmetric** (flat-array + out-params): `eigen_symmetric(mat: &![f64;65536],
  n: i32, eigenvalues: &![f64;256], eigenvectors: &![f64;65536]) -> bool`. Build the
  flat `mat[i*n+j]`, pass mutable out-param arrays, read `eigenvalues[i]` /
  `eigenvectors[...]` after. Confirm eigenvector storage layout (row vs column) in
  Phase 0 — this is the #1 convention risk.

## 4. Architecture — extend the parity harness to matrices

### 4.1 Wire format (extended, self-describing)

Per emitted matrix element, one line:
```
<op> <case> <role> <i> <j> <val_bits>
```
- `op` — operation name (`det`, `mul`, `lu`, `qr`, `eig`, …).
- `case` — integer test-case index (several test matrices per op).
- `role` — `A`/`B` (inputs), `R` (result), `L`/`U`/`P` (LU factors; `P` emits the
  pivot as an integer, `j=0`), `Q`/`RR` (QR factors), `EVAL` (eigenvalue vector,
  `j=0`), `EVEC` (eigenvector matrix). Scalars use role `R`, `i=j=0`.
- `i`,`j` — element indices. `val_bits` — `f64_to_bits(value)` (signed i64);
  for `P` it's the integer pivot value directly (still via `print_int`).

Self-describing: the emitter emits the inputs (`A`,`B`) too, so Python needs no
shared test-matrix table — it reconstructs every matrix from the stream and
computes references itself. Reuses `f64_to_bits`/`print_int` (builtins) and the
`stdlib/parity/emit.sio` scalar helpers; adds a small `emit_mat(op,case,role,m)`
Sounio helper that loops `matnm_get` and prints one line per element.

### 4.2 Sounio emitters

- `tests/parity/linalg_parity_matnm.sio` — constructs each test `MatNM` via
  `matnm_set`, runs the tier-1/2/3 matnm ops, emits inputs + results/factors.
- `tests/parity/linalg_parity_eigen.sio` — the flat-array eigen path (separate
  because the interface differs). Emits `A`, `EVAL`, `EVEC`.
Both compile+run under `SOUNIO_SOUC_ENGINE=lean_single`. Bit f64 on LOCALS.

### 4.3 Python comparator `scripts/parity/linalg_parity_ref.py`

Whole-stream tokenizer (5 tokens/line). Groups by `(op, case)`, assembles the
role matrices (dict of {(i,j): value}) into mpmath matrices. Per op:
- scalar/element-wise: compute the mpmath reference from `A` (and `B`), compare
  the emitted `R` element-wise; report `max_rel_err` per op.
- LU: build `L`,`U`,`P` from roles; compare `L·U` vs `P·A` (mpmath matmul);
  report the reconstruction `max_rel_err`.
- QR: compare `Q·R` vs `A` and `Qᵀ·Q` vs `I`.
- eig: sort emitted `EVAL` vs mpmath eigenvalues (matched by sorted order),
  report eigenvalue `max_rel_err`; report the worst eigenvector residual
  `‖A·v−λv‖/‖v‖`.
`--require-all` coverage assertion + `--selftest`. mpmath only (numpy optional).

### 4.4 Gate `scripts/linalg_parity_gate.sh`

Mirrors the special/stats gates: `SOUNIO_SOUC_ENGINE=lean_single`, compile+run
the emitters, pipe to the comparator with `--require-all`, emit
`LINALG_PARITY_GATE_OK`. Deterministic; dev-tier (needs mpmath).

### 4.5 Report `docs/research/2026-07-19-linalg-parity.md`

Per-operation accuracy map, the decomposition invariants used, convention
findings (piv encoding, eigenvector layout, row/col-major), defects found+fixed.

## 5. Test matrices

Small (2×2, 3×3, 4×4) with known-good conditioning: an SPD matrix (for eigen,
cholesky-like), a general invertible matrix, a near-identity, a moderately
ill-conditioned one (documented). For solve, an (A, b) pair. Avoid singular/
ill-posed inputs unless deliberately testing the error path. ~4–8 cases per op.

## 6. Tolerance philosophy (same as special/stats)

Measure genuine `max_rel_err`; fail loudly only on gross error (> 1e-2, likely a
bug); calibrate a function's threshold to its real accuracy only after confirming
it's an approximation/conditioning limit, never to hide a bug. Ill-conditioned
cases get a documented, looser threshold tied to the condition number. Exact
anchors (trace, transpose, identity·A) held tight. **Fix-as-found:** root-caused
≈1-line defects fixed in `stdlib/linalg/*.sio` (separate `fix(linalg):` commits,
re-verified, no regression in existing `tests/stdlib/linalg/**`); non-trivial
ones flagged.

## 7. Phase-0 confirmations (before the emitters)

- The matrix bit-exact bridge works: a 1-element `emit_mat` round-trips under
  lean_single.
- `MatNM` indexing is row-major; `matnm_get(set(m,i,j,v),i,j)==v`.
- `LUResult.piv` permutation encoding (partial-pivot row swaps vs a permutation
  vector) — decode it correctly for `P·A`.
- `eigen_symmetric` eigenvector layout (row i = i-th eigenvector, or column) —
  verify with a known 2×2 symmetric matrix whose eigenpairs are exact.
- Whether these compile/run under lean_single at all (matnm/eigen are larger
  modules — if a specific op segfaults, record and isolate it).

## 8. Non-goals

- Not the specialized/epistemic/sparse/BLAS-FFI surfaces, non-symmetric eigen, SVD.
- Not installing scipy (mpmath is the reference; numpy optional cross-check).
- Nothing ships into the prebuilt; stdlib fixes reach users on the normal build path.
