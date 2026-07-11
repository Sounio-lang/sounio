<!-- docs:meta
topic_id: repo.docs.research.gresnigt-g2s3
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.gresnigt-g2s3
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Aut(𝕊) = G₂ × S₃, executed — the direct-product structure Erratum E1 rests on

**One line.** Reconstructed both generators of Gresnigt's family `S₃ = ⟨ψ, ϵ⟩` (arXiv:2306.13098 §4.3,
eq 47–55) and certified the **commuting-factors (direct-product) structure** `Aut(𝕊) = G₂ × S₃`: the
color-triplet Weyl element `φ ∈ G₂` commutes with **both** family generators, `[φ,ψ] = [φ,ϵ] = 0`, while
`⟨ψ,ϵ⟩` satisfies the `S₃` braid relation. This turns Brown's theorem — cited as the foundation of
Erratum E1 — into a certified computation, and pins **why** the zero-divisor monomial-168 carries no
generation structure: color and family live in independent factors.

## The two family generators (eq 47–55)

`Aut(𝕊) = Aut(𝕆) × S₃ = G₂ × S₃`, with `S₃ = ⟨ψ, ϵ⟩`:

- **`ψ`** (order 3, eq 51–54): a 120° rotation in each plane `{e_i, e_{i+8}}` (`i=1..7`), fixing `e₀, e₈` —
  `ψ(e_i) = −½ e_i − (√3/2) e_{i+8}`. **Non-monomial** (uses `√3`); the generation-cycling generator.
- **`ϵ`** (order 2, eq 55/49): `ϵ(e_i)=e_i` for `i≤7`, `ϵ(e_i)=−e_i` for `i≥8` — a **monomial**
  (diagonal-sign) involution; does not permute generations.

Certified over `ℤ[√3]` (souc/oracle/Lean):

| Check | Meaning |
|---|---|
| `EPS_AUTO` | `ϵ` is a genuine sedenion automorphism |
| `S3_REL` | `ϵ∘ψ = ψ²∘ϵ` — the `S₃` braid relation (with `ϵ²=ψ³=1`, cf. #708) |
| `COMM_PHI_PSI`, `COMM_PHI_EPS` | `[φ,ψ]=0` and `[φ,ϵ]=0` — the color-Weyl `φ∈G₂` commutes with the whole family `S₃` |
| `PSI_NONMONO`, `EPS_MONO` | `ψ` non-monomial (√3), `ϵ` monomial |
| `ONLY_PSI_MIXES` | only `ψ` mixes octonion units (`≤7`) with the new sedenion units (`≥8`); `ϵ, φ` do not |

## What this establishes

`[φ,ψ]=[φ,ϵ]=0` executes the **direct-product** independence: the color element (in `G₂ ⊃ SU(3)_C`) and
the family `S₃` commute, so color and family are separate factors — exactly the `G₂ × S₃` structure
Erratum E1 invokes (from Brown 1967) but did not compute. Combined with `ψ` being non-monomial (#708),
the picture is complete and executed: **the generation-cycling symmetry is a rotation in an independent
family factor, disjoint from the monomial group the zero-divisor geometry inhabits.** No bridge — now
grounded, not cited.

## Honest boundary

- This does **not** reprove all of Brown's theorem (that `Aut(𝕊)` is *exactly* `G₂ × S₃`, with nothing
  more). It certifies that `ψ, ϵ` are the stated automorphisms, generate an `S₃`, and commute with the
  color-Weyl `φ ∈ G₂` — the commuting-factor structure the "no bridge" conclusion uses.
- **`ϵ` is monomial**, so the family `S₃` is not *entirely* non-monomial. What matters is that its
  **generation-cycling** element `ψ` is non-monomial; `ϵ` does not permute generations (it supplies an
  extra degree of freedom — chirality/handedness, per the paper). So Erratum E1 and #708 stand: the
  *generation* symmetry is rotational, hence outside the ZD monomial-168.
- The physical status of the three-generation model (electroweak sector, etc.) remains the authors' open
  problem, unchanged here.

## Certification (3 legs)
- **souc** (`ℤ[√3]`): `tests/run-pass/gresnigt_g2s3.sio` → `G2S3 OK` (bin/souc AND stage2 agree).
- **Python oracle** (`ℚ(√3)`): `scripts/research/gresnigt_g2s3_oracle.py`; gate
  `scripts/ci/gresnigt_g2s3_gate.sh`.
- **Lean `native_decide`** (`ℚ(√3)`): `formal/lean4/SounioGresnigtG2S3.lean` — `eps_automorphism`,
  `s3_braid`, `phi_commutes_psi`, `phi_commutes_eps`, `psi_nonmonomial_eps_monomial`.

## Reproduce
```bash
SOUNIO_STDLIB_PATH=$PWD/stdlib ./bin/souc run tests/run-pass/gresnigt_g2s3.sio
python3 scripts/research/gresnigt_g2s3_oracle.py
bash scripts/ci/gresnigt_g2s3_gate.sh
(cd formal/lean4 && lake build SounioGresnigtG2S3)
```
Source: Gresnigt NG, arXiv:2306.13098 (EPJC 83:747, 2023) §4.3, eq (47)–(55); Brown RB, Pac. J. Math.
20 (1967) 415.
