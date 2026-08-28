<!-- docs:meta
topic_id: repo.docs.research.sedenion-seam-bridge
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.sedenion-seam-bridge
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# The e₈-seam bridge — zero-division and operator non-alternativity are one locus in 𝕊

**One line.** Two *a priori* independent structures of the sedenions — the state-level zero-divisor
census (the e₈-boundary geometry of Paper 1/2) and the operator-level failure of the left-multiplications
to anticommute (the Cℓ(6)→Cℓ(8) fingerprint) — are shown to be **the same off-seam locus**, and the
anticommutator is the **exact obstruction** to zero-division. This closes the previously disconnected
state-level and operator-level strata of Vector 4/3.

## The theorem (seam trichotomy)

Let `𝕊 = 𝕆 ⊕ 𝕆·e₈`, lower indices `L={1..7}`, upper `U={8..15}`, `L_i` = left multiplication by `e_i`
(`L_i²=−I`). The **e₈ seam** is `{(l,u) : u=8 or l⊕u=8}` (14 pairs); its complement in the 56 lower×upper
pairs is the **off-seam** set (42 pairs). For every lower×upper pair `(l,u)` the following are equivalent:

1. `{L_l, L_u} ≠ 0`  (the left-multiplications fail to anticommute);
2. `(L_l L_u)² ≠ −I`;
3. `+1 ∈ spec(L_l L_u)`;
4. `L_l + L_u` is singular;
5. `e_l + e_u` is a (left) zero divisor of 𝕊;
6. `(l,u)` is off the e₈ seam.

Consequently the three independently defined 42-element sets — **non-anticommuting operator-pairs**,
**zero-divisor-participating directions**, and **off-seam index-pairs** — coincide.

### Mechanism (forward implication, proved)

`e_l + e_u` is a left zero divisor iff `L_l + L_u` is singular, i.e. iff some `w ≠ 0` has `L_l w = −L_u w`;
applying `L_l` and using `L_l²=−I` gives `L_l L_u w = w`, so `w` is a `+1`-eigenvector of `L_l L_u`. If
`{L_l,L_u}=0`, then

```
(L_l L_u)² = L_l (L_u L_l) L_u = −L_l² L_u² = −(−I)(−I) = −I,
```

so `spec(L_l L_u) ⊆ {±i}`, `+1` is excluded, `L_l+L_u` is nonsingular, and `e_l+e_u` is **not** a zero
divisor. **Anticommutation obstructs zero-division.** The converse holds for all 42 off-seam pairs by
direct computation.

## Quartet incidence (resolution of the "42 = 42")

The 84 participating primitives (42 off-seam directions × 2 signs) carry 168 unordered zero-divisor edges,
grouping by support-union into 42 quartets `{l₁,l₂,u₁,u₂}` of four edges each (`168 = 42×4`, the Paper 1
census). Certified: **every quartet contains exactly four off-seam cross sub-pairs, and every off-seam
pair lies in exactly four quartets.** The coincidence `|operator-pairs| = |quartets| = 42` is therefore
not a bijection but a **4-regular, self-paired incidence** with 168 incidences — both sides being the
off-seam locus, joined by the 168 zero-divisor edges.

## Significance

The zero-divisor geometry (a property of the multiplication on *states*) and the non-alternativity of the
left-adjoint action (a property of *operators*) are two faces of the same off-seam locus, with the
anticommutator as the precise obstruction. The 42 non-anticommuting pairs, the 42 ZD-quartets, and the
84↔84 dagger structure are one object viewed along three axes. Intrinsic to 𝕊; independent of any
particle-physics interpretation.

## Certification (3 legs)
- **souc** (sparse core, ℤ): `tests/run-pass/sedenion_seam_bridge.sio` → `BRIDGE OK` (bin/souc AND stage2):
  `{L_l,L_u}=0 ⟺ (L_lL_u)²=−I ⟺ not-ZD ⟺ on-seam`, all 56 pairs; the three 42-sets.
- **Python oracle** (full six-way incl. the det/spec formulations via exact integer Bareiss determinants,
  + the 4-regular quartet incidence): `scripts/research/sedenion_seam_bridge_oracle.py`; gate
  `scripts/ci/sedenion_seam_bridge_gate.sh` (asserts `SIXWAY_OK` and `INCIDENCE_OK`).
- **Lean `native_decide`**: `formal/lean4/SounioSeamBridge.lean` — `seam_equivalence`, `three_42_sets`.

The sparse identities used (no matrix products): `{L_l,L_u}=0 ⟺ σ(l,u⊕c)σ(u,c)+σ(u,l⊕c)σ(l,c)=0 ∀c`;
`(L_lL_u)²=−I ⟺ σ(l,l⊕c)σ(u,l⊕u⊕c)σ(l,u⊕c)σ(u,c)=−1 ∀c`.

## Reproduce
```bash
SOUNIO_STDLIB_PATH=$PWD/stdlib ./bin/souc run tests/run-pass/sedenion_seam_bridge.sio
python3 scripts/research/sedenion_seam_bridge_oracle.py
bash scripts/ci/sedenion_seam_bridge_gate.sh
(cd formal/lean4 && lake build SounioSeamBridge)
```

## References
- Paper 1 (`docs/papers/exact-168-executable.md`), Paper 2 (`docs/papers/sedenion-fano-geometry.md`).
- Moreno G, Bol Soc Mat Mexicana 4 (1998) 13; Cawagas RE, Discuss. Math. 24 (2004) 251; Biss/Dugger/
  Isaksen, Commun. Algebra 36 (2008) 632.
