<!-- docs:meta
topic_id: repo.docs.research.furey-octonion-generation
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.furey-octonion-generation
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Sounio reproduces Furey's octonion → one Standard-Model generation (executed, exact)

**One line.** Frente B, vector 4/3 Part A: the complex octonions ℂ⊗𝕆 carry a fermionic ladder algebra of
three modes whose number operator gives the electric charges of exactly **one generation** of quarks and
leptons, with the ×3 multiplicities of **SU(3) colour**. Sounio **executes** this established result
**exactly over the Gaussian integers ℤ[i]** — no float — and it is cross-verified on three independent
legs (souc, a Python oracle, and a Lean `native_decide` kernel proof).

## Established result, not a new claim

This is the *textbook / "trivial" end* of the octonion–Standard-Model program (C. Furey, *"Standard model
physics from an algebra?"*, and the broader Dixon/Furey ℂ⊗𝕆 line of work). We claim **no new physics**.
The point is that Sounio's exact-algebra substrate can carry a real particle-physics construction end to
end and certify every integer, reproducing the known one-generation charge spectrum on the nose.

## The construction (over Sounio's `cd_sigma` convention)

Work with left-multiplication operators. For the octonion unit `e_a` (`e_0 = 1`, `e_1..e_7`), the
**left-multiplication matrix** `L_a` is the 8×8 integer matrix with `L_a[a^b][b] = cd_sigma(a,b,3)` in
column `b`, because `e_a · e_b = cd_sigma(a,b,3) · e_{a^b}` (`cd_sigma` = the standard Cayley–Dickson sign,
here at `bits = 3`, the *same* recursion used at `bits = 4` for the sedenion bricks).

Furey's ladder operators over ℂ⊗𝕆 are

```
alpha_i = 1/2 ( -L_{a_i} + i · L_{b_i} ),   (a_1,b_1),(a_2,b_2),(a_3,b_3) = (1,2),(3,4),(5,6)
```

**relative to Sounio's `cd_sigma` sign convention** (the specific index pairs are convention-dependent;
these three pairs are the ones that close the algebra here). To stay in exact integers we scale by 2 and
work with `A_i = 2·alpha_i`, the complex 8×8 matrix with **real part `−L_{a_i}`** and **imaginary part
`L_{b_i}`** — Gaussian-integer entries. A complex matrix is represented as **two** integer arrays
`(Re, Im)`; the adjoint is the conjugate-transpose `A† = (Reᵀ, −Imᵀ)`, and the complex product is
`(Xr+iXi)(Yr+iYi) = (XrYr − XiYi) + i(XrYi + XiYr)`.

### Claim 1 — the fermionic ladder algebra closes, exactly over ℤ[i]

For all `i, j ∈ {1,2,3}` (18 anticommutator relations):

| relation | value |
|---|---|
| `{A_i, A_j} = A_iA_j + A_jA_i` | `0` (both Re and Im all-zero) |
| `{A_i, A_j†} = A_iA_j† + A_j†A_i` | `4·δ_ij·I` (Re = `4·I` iff `i=j`, else 0; Im all-zero) |

The factor **4** appears because `A = 2·alpha`; the physical `alpha_i` satisfy `{alpha_i, alpha_j†} =
δ_ij` — the canonical anticommutation relations of three fermionic creation/annihilation modes. Every
entry is computed by exact integer complex matrix multiplication; there is no rounding anywhere.

### Claim 2 — the charge spectrum of one generation

Given Claim 1, the rest is combinatorics. The fermionic Fock space of the three modes has occupation
number `n ∈ {0,1,2,3}` with multiplicities `C(3,n) = {0:1, 1:3, 2:3, 3:1}`. The **electric charge** is
`Q = N/3`, so it takes the values

```
Q ∈ { 0,  1/3,  2/3,  1 }   with multiplicities { 1, 3, 3, 1 }.
```

The `×3` multiplicities on `Q = 1/3` and `Q = 2/3` are exactly the **SU(3) colour triplets** — the three
colours of the down-type and up-type quarks. `Q = 0` is the neutrino, `Q = 1` the positron (or, in the
conjugate ideal, `Q = −1` the electron). The conjugate minimal ideal supplies
`{0, −1/3, −2/3, −1}` with the same `{1,3,3,1}`, completing the **full 16-state generation** (8 particles +
8 anti-particles). The brick emits `Q×3 → multiplicity`, i.e. `CHARGE3_0 1`, `CHARGE3_1 3`, `CHARGE3_2 3`,
`CHARGE3_3 1`.

## Certification (exact, over ℤ[i] — three independent legs)

- **souc**: `tests/run-pass/furey_octonion_generation.sio` → `FUREY OK`. Self-contained (the
  Cayley–Dickson sign `cd_sigma` copied verbatim; no stdlib import). `main` stays tiny (the 8×8 complex
  matmul lives in helper functions), and the brick runs with **identical values** under **both**
  `bin/souc` and the fresh stage2 compiler — no bin/souc miscompile here.
- **Python oracle**: `scripts/research/furey_octonion_oracle.py` (independent implementation of the same
  `L_a`, complex matmul, adjoint, and Fock counting). CI gate
  `scripts/ci/furey_octonion_gate.sh` diffs the souc value lines against the oracle.
- **Lean `native_decide`**: `formal/lean4/SounioFureyOctonion.lean` → `ladder_closes` (Claim 1, all 18
  relations over ℤ[i]) and `charge_multiplicities` (Claim 2, `C(3,n) = [1,3,3,1]`). Mathlib-free, no
  `sorry`; `lake build SounioFureyOctonion` in <4 s.

## Reproduce

```bash
SOUNIO_STDLIB_PATH=$PWD/stdlib ./bin/souc run tests/run-pass/furey_octonion_generation.sio
python3 scripts/research/furey_octonion_oracle.py
bash scripts/ci/furey_octonion_gate.sh
(cd formal/lean4 && lake build SounioFureyOctonion)
```
