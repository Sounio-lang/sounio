<!-- docs:meta
topic_id: repo.docs.research.sedenion-dynamics
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.sedenion-dynamics
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# The substrate dynamics: spanning-tree complexity of the ZD-geometry graphs (executed)

**One line.** Frente B, vector 4 (the pre-geometric jump), second step: the Laplacian `L = D − A` is the
generator of the substrate **dynamics** (heat flow `e^{−tL}`, random walk). Two of its exact,
integer-certified invariants — the **spanning-tree complexity** `τ` (Matrix-Tree theorem) and the
**random-walk return counts** `(A^k)_{ii}` — are computed over ℤ, fraction-free, and cross-verified on
three independent legs. Headline: `τ(fiber) = 393216`, `τ(2·K₇) = 1075648`.

## From the spectrum to the dynamics

Vector 4/1 (`sedenion_spectra.md`) proved the ZD-geometry graphs have **integral** Laplacian spectra:
the fiber `K_{6,6}−3·K_{2,2}` (12 vertices, degree 4) has `L`-spectrum `{0, 2², 4⁶, 6², 8}`; the `2·K₇`
fiber-incidence graph (7 vertices, off-diagonal 2, degree 12) has `L`-spectrum `{0, 14⁶}`. The spectrum
*is* the dynamics: `e^{−tL}` is heat flow, the walk operator governs diffusion, and the number of
**spanning trees** is the graph's global connectivity/complexity — the normalization constant of the
uniform-spanning-tree measure and (via Matrix-Tree/Kirchhoff) a determinant of the Laplacian.

### 1. Spanning-tree complexity `τ` (Matrix-Tree, exact over ℤ)

Kirchhoff's Matrix-Tree theorem: `τ` = the determinant of **any** principal `(n−1)×(n−1)` cofactor of
`L` (delete one row and its column). We compute that determinant by **fraction-free (Bareiss) integer
Gaussian elimination** — no rationals, no float; every intermediate is an exact `i64` (the Bareiss
identity guarantees each division is exact, and for these connected graphs the reduced Laplacian is
positive-definite, so no pivot is ever zero).

| Graph | `L`-spectrum | `τ = (1/n)·∏(nonzero eigenvalues)` | Bareiss `det` |
|---|---|---|---|
| fiber `K_{6,6}−3K_{2,2}` (12 v) | `{0, 2², 4⁶, 6², 8}` | `(2²·4⁶·6²·8)/12` | **393216** |
| `2·K₇` incidence (7 v) | `{0, 14⁶}` | `14⁶/7` | **1075648** |

The right two columns are computed **independently** — the middle from the integral spectrum of
vector 4/1, the right by exact integer elimination on the Laplacian cofactor — and they agree. This ties
the dynamics directly back to the metric: `τ = ∏(nonzero Laplacian eigenvalues)/n`.

### 2. Random-walk return counts `(A^k)_{ii}` (secondary)

`(A^k)_{ii}` is the number of closed walks of length `k` from vertex `i`. Both graphs are
vertex-transitive, so these return counts are the same at every vertex:

- fiber: `(A²)_{00} = 4 = degree`, `(A⁴)_{00} = 48`.
- `2·K₇`: `(A²)_{00} = 24 = 2²·6`, `(A⁴)_{00} = 2976`.

**A subtlety worth flagging.** For a simple `0/1` graph, `(A²)_{ii}` equals the degree — this is why the
fiber gives `4 = degree`. For `2·K₇` the adjacency entries are `2`, so `(A²)_{00} = Σ_j A_{0j}² = 6·2² =
24`, which is **not** the degree (`12 = Σ_j A_{0j}`). The closed-2-walk count and the degree coincide only
in the `0/1` case; we certify the true return count `24`.

## Certification (exact, over ℤ — three independent legs)

- **souc**: `tests/run-pass/sedenion_dynamics.sio` → `DYNAMICS OK`. Self-contained (the Cayley–Dickson
  sign `cd_sigma`, `prim_prod`, `is_zero_pair` copied verbatim from `sedenion_zd_fibers.sio`, avoiding
  the stdlib-engine import defect #637). Runs correctly under **both** `bin/souc` and the fresh stage2.
- **Python oracle**: `scripts/research/sedenion_dynamics_oracle.py` (same Bareiss + matrix-power values);
  CI gate `scripts/ci/sedenion_dynamics_gate.sh` diffs the souc value lines against the oracle.
- **Lean `native_decide`**: `formal/lean4/SounioSedenionDynamics.lean` → `fiber_spantree`,
  `k7_spantree`, `fiber_walk2/walk4`, `k7_walk2`, `fiber_verts` (Mathlib-free, no `sorry`; Bareiss with
  structural fuel recursion). `lake build SounioSedenionDynamics` in <4 s.

### A compiler note (recorded for lineage)

A single monolithic `main()` computing **both** graphs SIGSEGVs under `bin/souc` (a codegen defect: the
values are all computed correctly, the crash is during output; the fiber and `2·K₇` halves each run clean
in isolation, and it is not a stack-size issue). Splitting the two graphs into separate functions (so
`main` stays tiny) sidesteps it entirely, and the brick then runs correctly under **both** `bin/souc` and
stage2 — so no stage2-only carve-out is needed here (unlike `sedenion_automorphism_168`, where `bin/souc`
genuinely miscompiles the value). This is a concrete instance of the standing rule: never trust a bare
souc pass; every number here is cross-checked against the Python oracle and the Lean kernel.

## Toward the jump (vectors 4/3)

Spanning-tree complexity and the walk operator are the *dynamical* data of the substrate; the integral
spectrum (`sedenion_spectra.md`) is its *metric* data; the Fano/`PSL(2,7)` symmetry
(`sedenion_fano_fibers.md`) is its *algebraic* data. A spinor/spacetime construction (vector 4/3,
Dixon/Furey-style) would build on all three. This note delivers the exact dynamical datum.

## Reproduce

```bash
SOUNIO_STDLIB_PATH=$PWD/stdlib ./bin/souc run tests/run-pass/sedenion_dynamics.sio
python3 scripts/research/sedenion_dynamics_oracle.py
bash scripts/ci/sedenion_dynamics_gate.sh
(cd formal/lean4 && lake build SounioSedenionDynamics)
```
