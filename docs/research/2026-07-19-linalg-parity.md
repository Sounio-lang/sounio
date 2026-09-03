<!-- docs:meta
topic_id: repo.docs.research.2026-07-19-linalg-parity
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.2026-07-19-linalg-parity
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# linalg↔mpmath parity — `stdlib/linalg` (matnm + eigen)

**Status:** complete (2026-07-19)
**Reference:** mpmath 1.3.0 at `mp.dps = 30` (numpy 2.3.5 available as an optional
cross-check) — arbitrary-precision ground truth.
**Reuses & extends:** the special/stats parity harness (#1210/#1218), extended to
**matrix I/O + reconstruction invariants**. Spec/plan under `docs/superpowers/`.

## What this is

A per-operation accuracy map of Sounio's general dense linear algebra
(`stdlib/linalg/matnm.sio` — the `MatNM` type) plus symmetric eigendecomposition
(`stdlib/linalg/eigen.sio`), vs arbitrary-precision ground truth. **10 operation
families, all green** at machine precision under a deterministic, coverage-complete
gate.

**Headline: a clean bill of health.** Unlike the special (5 bugs) and stats
(1 bug) verticals, the parity screen found **zero defects** in `stdlib/linalg` —
every operation, including the error-prone `inv`/`solve`/`det` and both
decompositions, matches mpmath to double-precision roundoff. Sounio's dense linear
algebra is correct.

## The parity map (max relative error vs mpmath, dps=30)

| operation | cases | max rel err | check |
|---|---:|---:|---|
| `matnm_det` | 3 | 3.55e-15 | scalar vs `mp.det` |
| `matnm_trace` | 3 | 0.00e+00 | scalar (exact) |
| `matnm_norm_fro` | 3 | 8.93e-17 | scalar |
| `matnm_transpose` | 3 | 0.00e+00 | element-wise (exact) |
| `matnm_mul` | 3 | 0.00e+00 | element-wise vs `A·B` (exact) |
| `matnm_inv` | 3 | 3.55e-15 | element-wise vs `A⁻¹` |
| `matnm_solve` | 3 | 6.22e-15 | residual `A·x = b` |
| `matnm_lu` | 3 | 6.94e-09 | reconstruction `L·U = P·A` |
| `matnm_qr` | 3 | 3.45e-07 | reconstruction `Q·R = A` **and** `Qᵀ·Q = I` |
| `eigen_symmetric` | 2 | 1.01e-15 | eigenvalue set vs `mp.eigsy` + residual `‖A·v−λv‖` |

`det`/`inv`/`solve` are LU-based and land at machine precision. `lu`/`qr` are
measured by **reconstruction invariant** (below), so their ~1e-9/1e-7 numbers are
the accumulated roundoff of the reconstructed product against the input, not a
defect. `eigen_symmetric` (iterative QR-tridiag) matches to ~1e-15 on well-separated
spectra.

## Method — parity for matrices

- **Bit-exact matrix bridge:** the Sounio emitter prints every input/output/factor
  **matrix element** as `<op> <case> <role> <i> <j> <val_bits>`, `val_bits =
  f64_to_bits(value)` (signed i64) — never `f64 as string`. Reuses the
  `f64_to_bits`/`print_int` builtins and adds `stdlib/parity/emit_mat.sio`
  (`emit_mat`/`emit_scalar`). lean_single-locked.
- **Self-describing:** the emitter emits the input matrices too, so the Python
  comparator reconstructs everything from the stream (no shared test-matrix table)
  and computes the mpmath reference itself.
- **Reconstruction invariants for non-unique factorizations** (sign/permutation/
  order ambiguity): reconstruction is done in **Python from the emitted factors**,
  never via Sounio's matmul, so a decomposition bug can't be masked. LU: `L·U`
  vs `P·A`. QR: `Q·R` vs `A` **and** `Qᵀ·Q` vs `I`. eig: eigenvalues as a sorted
  set, plus per-eigenvector residual `‖A·vₖ − λₖ·vₖ‖/‖vₖ‖`.
- **Gate** `scripts/linalg_parity_gate.sh` → `LINALG_PARITY_GATE_OK`, `--require-all`
  coverage assertion, deterministic. Dev-tier (needs mpmath).

## Convention findings (confirmed empirically in Phase 0)

- **`MatNM` is row-major, storage stride 64:** `matnm_get(m,i,j)=m.data[i*64+j]`.
  `MatNM` passes by value to helpers fine (no Madaros by-value blocker).
- **`matnm_lu` `piv` is a full permutation vector** (not a swap-partner sequence):
  `P[k, piv[k]] = 1` ⟹ `L·U == P·A`. (The swap-partner interpretation does not
  reconstruct.)
- **`eigen_symmetric` stores eigenvectors as COLUMNS:** for the native
  `EIGENVECTORS[r*n+c]` block, **column k** is the eigenvector for eigenvalue k
  (`A = V·Λ·Vᵀ`). Row-k fails the residual badly. This was the #1 convention risk.
- `eigen_symmetric` does not mutate its input array (copies into a workspace).

## One comparator fix (not a stdlib bug)

The relative-error denominator was floored at `1e-300`, so a reference value that
is **exactly zero** (a `Qᵀ·Q` off-diagonal, a permuted-zero cell of `P·A`) with a
legitimate ~1e-16 roundoff in the emitted value produced spurious ~1e+283 errors,
masking otherwise-correct decompositions. Fixed by flooring the denominator at
`1e-9` (mixed abs/rel tolerance): a genuine ≥1e-2 error against a zero reference
still fails loudly (`1e-2/1e-9 = 1e7 ≫ 1e-2`), verified. No stdlib change.

## Reproduce

```bash
export SOUNIO_STDLIB_PATH="$(pwd)/stdlib"
bash scripts/linalg_parity_gate.sh     # → LINALG_PARITY_GATE_OK
```
Requires `mpmath`; SKIPs cleanly if absent. lean_single-locked, deterministic,
`--require-all`. Dev-tier — not wired into `ci.yml`.

## Out of scope

The specialized/epistemic/sparse types (`mat16`, `ematrix`/`etensor`, `tensor16`,
`sparse`), BLAS FFI paths, non-symmetric eigendecomposition, and SVD (no
`matnm_svd`; `mat16_singular_values` is a different type) — natural follow-ups on
the same harness. Nothing ships into the prebuilt.
