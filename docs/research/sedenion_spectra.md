<!-- docs:meta
topic_id: repo.docs.research.sedenion-spectra
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.sedenion-spectra
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# The emergent metric: the sedenion ZD-geometry graphs have integral spectra (executed)

**One line.** Frente B, vector 4 (the pre-geometric jump), first step: the zero-divisor geometry is not
only combinatorial — its graphs carry an exact **spectral (metric) structure**, and it is **integral**.
A fiber (`K_{6,6} − 3·K_{2,2}`) has adjacency spectrum `{4, 2², 0⁶, −2², −4}` (Laplacian `{0, 2², 4⁶, 6², 8}`,
algebraic connectivity **2**); the `2·K₇` fiber-incidence graph has adjacency `{12, −2⁶}` (Laplacian
`{0, 14⁶}`, algebraic connectivity **14**). Certified exactly over ℤ by spectral moments.

## From combinatorics to a metric

The Laplacian `L = D − A` of a graph is the discrete metric/diffusion operator: its spectrum is the
graph's "shape" — the algebraic connectivity (smallest nonzero eigenvalue) is a curvature/expansion
scale, and `e^{−tL}` is heat flow. Both graphs of the ZD geometry have **integral** Laplacian spectra —
the hallmark of a highly symmetric ("crystallographic") structure, consistent with the Fano
collineation symmetry (`sedenion_fano_fibers.md`).

| Graph | adjacency spectrum | Laplacian spectrum | algebraic connectivity |
|---|---|---|---|
| fiber `K_{6,6}−3K_{2,2}` (12 v, deg 4) | `{4, 2², 0⁶, −2², −4}` | `{0, 2², 4⁶, 6², 8}` | **2** |
| `2·K₇` incidence (7 v, deg 12) | `{12, −2⁶}` | `{0, 14⁶}` | **14** |

(A curiosity: the `2·K₇` second moment `trace(A²) = 168`, the census number itself.)

## Certification (exact, over ℤ — no eigen-solver)

We certify the integral spectra by their **spectral moments** `m_k = trace(A^k) = Σ λ^k`, computed by
exact integer matrix powers. Matching `m_k` for `k` beyond the number of distinct eigenvalues pins the
spectrum. Verified `m₂, m₄, m₆ = 48, 576, 8448` (fiber) and `m₂, m₃, m₄ = 168, 1680, 20832` (`2·K₇`) —
exactly the moments of the proposed integral spectra.

- **souc**: `tests/run-pass/sedenion_spectra.sio` → `SPECTRA OK`. Runs under `bin/souc` and stage2.
- **Python oracle**: `scripts/research/sedenion_spectra_oracle.py`; CI gate `scripts/ci/sedenion_spectra_gate.sh` (souc == oracle).
- **Lean `native_decide`**: `formal/lean4/SounioSedenionSpectra.lean` → `fiber_m2/m4/m6`, `k7_m2/m3`, `fiber_verts`.

## Toward the jump (vectors 4/2, 4/3)

The Laplacian is the generator of the substrate **dynamics** (heat flow `e^{−tL}`, random walk) — the
next step (vector 4/2). The integral spectrum + the Fano/`PSL(2,7)` symmetry are the algebraic data a
**spinor/spacetime** construction (vector 4/3, Dixon/Furey-style) would build on. Those are the open
frontier; this note delivers the exact metric datum they start from.

## Reproduce

```bash
SOUNIO_STDLIB_PATH=$PWD/stdlib ./bin/souc run tests/run-pass/sedenion_spectra.sio
python3 scripts/research/sedenion_spectra_oracle.py
bash scripts/ci/sedenion_spectra_gate.sh
(cd formal/lean4 && lake build SounioSedenionSpectra)
```
