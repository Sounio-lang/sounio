<!-- docs:meta
topic_id: repo.docs.research.cd-tower-seam
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.cd-tower-seam
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# The e_top-seam coincidence across the Cayley–Dickson tower

**One line.** The sedenion e₈-seam bridge (`sedenion_seam_bridge.md`) is not special to 𝕊: the same
**coincidence of two independently-defined sets** — the operator-level *non-anticommuting* pairs and the
state-level *zero-divisor* pairs — holds, and equals the *off-seam* set, at dim 16, 32 and 64 of the
Cayley–Dickson tower. Executably certified; not claimed novel (this is Moreno/de-Marrais/Cawagas
zero-divisor territory).

## Setup
For the 2ⁿ-dim CD algebra `A_n` with top imaginary unit `e_top` (`top = 2ⁿ⁻¹`) and lower×upper index
pairs `(l ∈ 1..top-1, u ∈ top..2ⁿ-1)`, define the *e_top seam* `{(l,u) : u=top or l⊕u=top}`.

## What is proved vs. computed vs. open

**Proved — dimension-independent linear algebra (no per-level computation), *given* `L_i²=−I`:**
- `{L_l,L_u}=0 ⟹ (L_lL_u)² = −L_l²L_u² = −I ⟹ +1 ∉ spec(L_lL_u) ⟹ L_l+L_u nonsingular ⟹ e_l+e_u is
  not a left zero divisor.` The anticommutator is the exact obstruction to zero-division — in **every**
  CD algebra.
- `L_l+L_u singular ⟺ +1 ∈ spec(L_lL_u) ⟺ e_l+e_u is a ZD` is also pure linear algebra. So the
  spectral and singular members of the equivalence carry no computational content; only the two members
  below are computed.

**Computed — the actual content (two independently-defined sets coincide):** the *non-anticommuting*
set `{(l,u) : {L_l,L_u}≠0}`, the *zero-divisor* set `{(l,u) : e_l+e_u is a ZD}`, and the *off-seam* set
are **equal** — verified at dim 16, 32, 64. (Their common size is `(top-1)(top-2)` = 42, 210, 930 — but
that is just the number of off-seam pairs *by definition*; a footnote, not the result. The content is
that the two *a-priori-unrelated* sets land on it.)

**The base fact `L_i²=−I`** is the CD cocycle identity `σ(i,j)·σ(i,i⊕j) = −1` (all `i≥1`, all `j`),
certified here at dim 16/32/64. A Mathlib-free general induction on `bits` would upgrade the forward
obstruction to a theorem for *all* n. **First step done** (`formal/lean4/SounioCDCocycle.lean`): the CD
sign is reformulated on explicit bit-lists (structural XOR; verified to agree with the Nat `cdSigma` at
dim 16/32), and **`e_i²=−1` (diag) is proved for ALL n** by structural induction. The full `L_i²=−I`
and basis-unit anticommutation close on paper (a simultaneous induction over the sign's four branches)
but their Mathlib-free formalization is **still open** — they remain certified at n=4,5,6 above.

**Open (all n): the converse.** `off-seam ⟹ e_l+e_u is a ZD` is *verified* at n=4,5,6, not proved — a
tower-wide conjecture (the sedenion "natural next lemma" of `sedenion_seam_bridge.md`, now at tower
scale). The tower brick establishes breadth and the forward half; it does not close the converse.

## Certification (3 legs)
- **souc** (bin/souc AND stage2): `tests/run-pass/cd_tower_seam.sio` → `TOWER OK`. Full four-member
  coincidence + counts at dim 16 & 32; at dim 64 the cocycle lemma + operator/seam coincidence + counts
  (the O(n³) ZD scan at dim 64 is left to the oracle to keep the test fast).
- **Python oracle**: `scripts/research/cd_tower_seam_oracle.py` — same, INCLUDING the full dim-64 ZD scan
  (`ZDEQ64`); gate `scripts/ci/cd_tower_seam_gate.sh`.
- **Lean `native_decide`**: `formal/lean4/SounioCDTowerSeam.lean` — `lsq_16/32/64` (cocycle lemma),
  `coincidence_16` (four members), `coincidence_32` (operator/seam).

## Reproduce
```bash
SOUNIO_STDLIB_PATH=$PWD/stdlib ./bin/souc run tests/run-pass/cd_tower_seam.sio
python3 scripts/research/cd_tower_seam_oracle.py
bash scripts/ci/cd_tower_seam_gate.sh
(cd formal/lean4 && lake build SounioCDTowerSeam)
```

## References
- Sedenion case: `docs/research/sedenion_seam_bridge.md`.
- Moreno G, Bol Soc Mat Mexicana 4 (1998) 13; de Marrais R, "box-kites"; Cawagas RE, Discuss. Math. 24
  (2004) 251.
