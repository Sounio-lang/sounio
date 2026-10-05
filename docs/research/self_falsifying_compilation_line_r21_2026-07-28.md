<!-- docs:meta
topic_id: repo.docs.research.self-falsifying-compilation-line-r21-2026-07-28
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.self-falsifying-compilation-line-r21-2026-07-28
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Self-falsifying compilation R21 — the lemma, proved: R16's inference is a theorem

**Date:** 2026-07-28
**Orthography:** EN-UK
**Status:** `EXECUTABLE` — `EQUIVARIANCE_PROVED__R16_INFERENCE_IS_A_THEOREM`
**Parents:** `self_falsifying_compilation_line_r19_2026-07-28.md` (the lemma this closes), `self_falsifying_compilation_line_r16_2026-07-28.md` (the inference this upgrades), `self_falsifying_compilation_line_r20_2026-07-28.md` (which restored the objects the proof needs)
**Harness:** `scripts/research/self_falsifying_compilation_line_r21_contract.py`
**Gate:** `scripts/ci/self_falsifying_compilation_line_r21_gate.sh`

---

## 1. Result

R16 measured that the count-preserving flip σ(H/2, H+H/2) preserves the whole
partition of fibers into spectrum-classes, and said plainly that the step from
"uniform local change" to "classification preserved" was **inferred, not
established**. R19 derived the locality half and reduced what was left to one
statement. This closes it.

> **Both relations that generate the blocks are F₂-linear and fix h. Each
> therefore carries the added edge to the added edge of the image fiber, so an
> isomorphism of the unperturbed graphs is an isomorphism of the perturbed ones.
> The partition is preserved. R16's inference is a theorem.**

Verdict: `SELF_FALSIFYING_R21_VERDICT EQUIVARIANCE_PROVED__R16_INFERENCE_IS_A_THEOREM`.

## 2. The argument

**(1) What the flip does — R19, derived.** In the Z₂ signed double-cover
presentation (each annihilation graph is the double cover of a *lo-graph*
(R, ε) — `cd_tower_collapse_isomorphism.py`), the flip does exactly one thing:
it adds the edge `{h, h ^ Llo}` to the lo-graph with ε = −1. And
`h XOR (h ^ Llo) = Llo`: the added edge is the unique edge anchored at `h` whose
endpoint-XOR is the fiber's own label.

**(2) What generates the blocks.** Two relations, both prior in-repo results and
re-verified here against the measured spectra (§3): the frozen
PSL(2,7) ≅ GL(3,2) orbit action, and the parity-collapse map Φ for even-weight
seams.

**(3) Both are F₂-linear and fix `h`.**

- **Orbit action.** `A = g ⊕ Id`: GL(3,2) on the octonion bits {0,1,2},
  **identity on every seam bit {3…n−2}**. And `h = 2^(n−2)` *is* a seam bit.
- **Collapse.** `τ = swap(bit 0, bit lsb(Y))`, a coordinate transposition, hence
  linear. And **`lsb(Y) ≠ n−2`**: `Y` has *even* weight with bits confined to
  {3…n−2}, so `lsb(Y) = n−2` would force `Y = 2^(n−2)`, of weight 1 — odd.
  Therefore `τ(h) = h`.

**(4) Conclusion.** For any `g` relating same-block fibers: `g` linear,
`g(h) = h`, `g(Llo) = Llo′`, hence

```
g({h, h ^ Llo}) = {g(h), g(h) ^ g(Llo)} = {h, h ^ Llo′}
```

— the added edge of the image fiber. So `g`, already an isomorphism of the
unperturbed graphs, is an isomorphism of the perturbed ones. ∎

The parity side-condition in (3) is the crux, and it is why the *even*-weight
restriction in the parity-collapse law — which looked like a fact about seams —
is exactly what protects `h`.

## 3. Verified, and how

| clause | |
|---|---|
| `V1_H_IS_A_SEAM_BIT` | arithmetic, n = 5…12 |
| `V2_TAU_FIXES_H` | 11 collapse pairs at n = 6, 7, 8 against the restored Φ: `lsb(Y)` is never `n−2`, `τ(h) = h`, added edge carried |
| `V3_BLOCKS_ARE_ORBITS_PLUS_COLLAPSE` | measured spectrum-blocks at n = 5, 6, 7 are exactly Fano orbits, seam fixed points, and orbits with **even**-weight seams collapsed in |

`V3` is computed **independently of the orbit contract**, which cannot run in
this tree: R20 found that its oracle was never committed to any branch. Relying
on a script that cannot execute would have been the failure this line studies.

## 4. What this is NOT

- **Not a proof from first principles.** It rests on two prior in-repo results
  taken as given: the orbit theorem (proven ∀n) and the parity-collapse law
  (verified n ≤ 8). What is proved here is the *equivariance of the
  perturbation* with respect to them.
- **Not ∀n.** The orbit half is ∀n; the collapse half is n ≤ 8, because that is
  where the parity-collapse law is established. So the theorem holds exactly on
  the range where the block structure itself does.
- **Not a statement about the real Cayley–Dickson tower.** As throughout R15–R19,
  a perturbed sign table is not a CD algebra. This is about the reach of a check.
- **Not a compiler change.**

## 5. Reproduce

```bash
python3 scripts/research/self_falsifying_compilation_line_r21_contract.py
bash scripts/ci/self_falsifying_compilation_line_r21_gate.sh
```

Needs `cd_tower_collapse_isomorphism.py`, restored in R20 — the proof was not
available while that artifact was missing, which is the concrete cost of the
provenance defect R20 found.

## 6. AI disclosure

Argument, contract, gate and spec drafted under human direction (2026-07-28).
The parity side-condition and the seam-bit observation were derived by hand and
then checked mechanically; V3 is machine-measured. The dependence on two prior
results is stated in §4 rather than absorbed. No clinical content.
GAIDeT-ICMJE 2025.
