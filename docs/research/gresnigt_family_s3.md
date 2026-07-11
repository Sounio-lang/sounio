<!-- docs:meta
topic_id: repo.docs.research.gresnigt-family-s3
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.gresnigt-family-s3
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Gresnigt's family S₃ generator is non-monomial: the fermion family symmetry is not in the ZD-168

**One line.** Reconstructed, exactly and against the primary source (Gresnigt arXiv:2306.13098 §5,
eq 70–76), the order-3 automorphism `ψ` that generates the three fermion generations, and proved it is
**non-monomial** (it uses `√3`). Since a signed permutation cannot produce `√3` coefficients, the fermion
**family symmetry lies outside the zero-divisor monomial-168** — the decisive answer to the whole arc:
**there is no bridge from the ZD-168 census to fermion generations.** Erratum E1 is vindicated, now with
the family generator explicit rather than inferred.

## The family generator ψ (frame-independent core)

Gresnigt builds one generation from three sedenion octonion subalgebras `𝕆₁,𝕆₂,𝕆₃`
(`sedenion_gresnigt_octonions.md`) via ladder operators `A†_i = ½√2(e_i + i e_{i+4} + e_{i+8} + i e_{i+12})`,
and generates the other two generations with an **order-3 automorphism `ψ`** (eq 70–76). Reading off
the paper, `ψ` is a **120° rotation in each plane `{e_i, e_{i+8}}`** (`i=1..7`), fixing `e₀, e₈`:

```
ψ(e_i) = −½ e_i − (√3/2) e_{i+8},   ψ(e_{i+8}) = (√3/2) e_i − ½ e_{i+8}.
```

Working over `ℤ[√3]` (scaling `2ψ` to integer coefficients in `{1, √3}`), we certify:

1. **`ψ` is a genuine sedenion automorphism** — `ψ(e_j e_k) = ψ(e_j) ψ(e_k)` for all 256 basis pairs;
2. **`ψ` has order 3** — `ψ³ = id`;
3. **`ψ` is NON-MONOMIAL** — some `ψ(e_j)` has a nonzero `√3` component, so it is not a signed
   permutation of basis units;
4. **`ψ` maps the gen-1 ladder to the gen-2 ladder** — `ψ(A†_1) = B†_1` with
   `a=(√3−1)/2, b=(−√3−1)/2` exactly (eq 70–76), confirming this is Gresnigt's family generator.

Fact (3) is frame-independent and is the load-bearing claim: **the family symmetry is genuinely
rotational, not a signed permutation, hence not an element of the monomial automorphism group (the 168 =
|PSL(2,7)| that the zero-divisor geometry inhabits).** There is no bridge.

## The mechanism (frame-relative)

With Gresnigt's 16-dim number operator `N = Σ_i A†_i A_i` (`Q = N/3` the electric charge), two
complementary facts hold and are cross-verified (Lean + oracle):

- the **monomial** `φ = (e₁e₂e₃)(e₅e₆e₇)(e₉e₁₀e₁₁)(e₁₃e₁₄e₁₅)` (from `sedenion_gresnigt_octonions.md`)
  **commutes with `N`** — it permutes the three ladder indices `A₁→A₂→A₃`, i.e. it acts as a
  **color-triplet Weyl-`S₃`** element (in the normalizer of the color Cartan);
- the family generator `ψ` does **not** commute with `N` — it **carries `N` to the gen-2 operator
  `Σ B†_i B_i`** (equal spectrum, *different* operator), as a family symmetry relating generations must.

These commutator statements are **frame-relative** (they depend on which number operator); the
frame-independent fact is that `ψ` is non-monomial and `φ` is monomial. Together they exhibit the clean
separation: color permutation (monomial, in `G₂ ⊃ SU(3)_C`) vs family permutation (rotational, the Brown
`S₃` factor of `Aut(𝕊)=G₂×S₃`).

## Relation to #707 (cross-reference, not a correction)

`furey_charge_g2.md` (#707) computed, in the **8-dim base-octonion Furey frame** (ladder pairs
`(1,2),(3,4),(5,6)`), that `φ` does not commute with *that* charge, and explicitly left the family
question **open** pending "Brown's rotational `S₃`". This brick **delivers** that `S₃` and closes the
question. The two are consistent: they concern *different operators in different dimensions*. #707's
literal claims hold; the physically relevant charge for the three-generation construction is the 16-dim
Gresnigt `N` used here, under which `φ` is the color-Weyl element.

## Honest boundary
The family symmetry being non-monomial settles that the ZD-168 does not carry it. A *positive* bridge —
were one to exist — would still require identifying some structure of the ZD geometry with the
three-generation ideals themselves; the census and the family symmetry are group-theoretically disjoint.
The physical status of the three-generation model (electroweak sector, etc.) is the authors' open
problem, unchanged here.

## Certification
- **souc** (frame-independent core, `ℤ[√3]`): `tests/run-pass/gresnigt_family_s3.sio` → `FAMILYS3 OK`
  (bin/souc AND stage2 agree): `PSI_AUTO/PSI_ORD3/PSI_NONMONO/PSI_MAPS_AB`.
- **Python oracle** (core + commutators, exact `ℚ(√3)` / `ℚ(√3,i)`):
  `scripts/research/gresnigt_family_s3_oracle.py`; gate `scripts/ci/gresnigt_family_s3_gate.sh`.
- **Lean `native_decide`** (`ℚ(√3)`): `formal/lean4/SounioGresnigtFamilyS3.lean` — `psi_automorphism`,
  `psi_order_3`, `psi_non_monomial`, `psi_maps_A_to_B`, `phi_commutes_charge`, `psi_not_commute_charge`.

## Reproduce
```bash
SOUNIO_STDLIB_PATH=$PWD/stdlib ./bin/souc run tests/run-pass/gresnigt_family_s3.sio
python3 scripts/research/gresnigt_family_s3_oracle.py
bash scripts/ci/gresnigt_family_s3_gate.sh
(cd formal/lean4 && lake build SounioGresnigtFamilyS3)
```
Source: Gresnigt NG, "Three generations of colored fermions with S₃ family symmetry", EPJC 83:747
(2023), arXiv:2306.13098 — §5, eq (56)–(76).
