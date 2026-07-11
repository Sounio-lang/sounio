<!-- docs:meta
topic_id: repo.docs.research.sedenion-clifford8
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.sedenion-clifford8
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# The sedenion left-multiplication algebra is Cℓ(8) — executed and cross-verified

**One line.** The left-multiplication algebra of the complexified sedenions ℂ⊗𝕊 is the Clifford
algebra **`Cℓ(8) ≅ ℂ(16)`** (256 = 2⁸ maps) — the exact rank-4 step up from the octonionic **`Cℓ(6)`**
(64 = 2⁶) that underlies Furey's one-generation Standard-Model construction. Certified exactly over ℤ
(and ℤ/p for the full dimension), it is the **algebraic basis of the peer-reviewed sedenion
three-generation program** (Gresnigt et al.). The *physical* three-generation interpretation is a
modelling assumption and is flagged **open**, following the authors.

## What is established (peer-reviewed) and what this brick executes

The octonion→Standard-Model construction (Furey arXiv:1405.4601, 1611.09182; Dixon; Stoica
arXiv:1702.04336): the left-action algebra of ℂ⊗𝕆 is `Cℓ(6)`, and one SM generation
(SU(3)_C × U(1)_em) is two minimal left ideals built on 3 fermionic ladder pairs — the rank-3 picture
Sounio reproduces exactly (`furey_octonion_generation.sio`). Going to the sedenions
(Gresnigt EPJC 2019 `s10052-019-6967-1`, 2024 `s10052-024-13476-0`; arXiv:2306.13098): the
left-multiplication algebra of ℂ⊗𝕊 is **`Cℓ(8)`**, and the three-generation extension is driven by the
**S3 automorphism** (`Aut(𝕊) = G₂ × S3`, the S3 novel at the 𝕆→𝕊 doubling) / equivalently the
**triality** that `Cℓ(8)` admits but `Cℓ(6)` does not — **not** by the zero-divisor census, and **not**
by naive spinor doubling.

This brick executes and certifies the **algebraic core**:

| Result | Value |
|---|---|
| `Cℓ(8)` presentation: `L_1..L_8` each square to `−I`, all 28 pairs anticommute | ✓ (8 anticommuting √(−I) generators) |
| the 8th generator is `L_8 = e_8` | the octonion→sedenion **doubling unit** — the extra Clifford generator |
| full algebra `dim⟨L_1..L_15⟩` | **256** = `M_16(ℂ)` = `Cℓ(8,ℂ)` (rank mod p) |
| maximal mutually-anticommuting set / **ladder rank** | 8 generators ⟹ **rank 4** (octonion: 6 ⟹ rank 3) |
| non-anticommuting pairs `{L_i,L_j} ≠ 0` | **42 of 105**, and **all 42 are lower-upper** (touch the doubling half {8..15}) |
| Gresnigt S3-invariant charge `Q_1 = (N_1+N_2+N_3 − 3N_4)/3` on the 2⁴ Fock space | multiset **{0, ±1/3, ±2/3, ±1}** = SM electric charges |

The user's own independent computation (this repo) that the maximal single-generator anticommuting
ladder rank is exactly 4 for sedenions vs 3 for octonions **is precisely the maximal-isotropic
dimension of `Cℓ(8)` vs `Cℓ(6)`**, independently corroborating the peer-reviewed identification. The
`e_8`-as-8th-generator fact is the same e₈ throughline as the zero-divisor geometry: e₈ bounds the ZD
set, carries the extra 168 on the associator side, is the unique fixed point of the Fano symmetry
(`sedenion_fano_fibers.md`), and here is the extra Clifford generator of `Cℓ(6)→Cℓ(8)`.

## Honest boundary (firewall)

**Certifiable (algebraic, exact):** the `Cℓ(6)`/`Cℓ(8)` identifications; the generator presentation;
the ladder ranks; the anticommutation fingerprint; the `Q_1` electric-charge multiset under the
**unbroken** SU(3)_C × U(1)_em.

**Flagged OPEN (physical modelling assumptions):** that the three algebraic copies **are** three
physical fermion generations; the electroweak SU(2)_L × U(1)_Y sector and symmetry breaking are **not**
derived (Furey flags SU(2)/U(1) "to be approached with caution"); the naive electric-charge number
operator **fails** at three generations without a generalized action; the 2019 sedenion construction
had a documented defect repaired only by later S3 work (2023/2024). **No peer-reviewed source links the
zero-divisor "168-census" or the associator "1848 = 11×168" to fermion generations or gauge groups** —
those are this program's own geometric invariants (algebraically real, physically uninterpreted).

## Certification

- **souc**: `tests/run-pass/sedenion_clifford8.sio` → `CL8 OK` (bin/souc AND stage2 agree).
- **Python oracle**: `scripts/research/sedenion_clifford8_oracle.py` (exact port of the operator's numpy
  reference; includes the `dim = 256` rank-mod-p check). CI gate `scripts/ci/sedenion_clifford8_gate.sh`.
- **Lean `native_decide`**: `formal/lean4/SounioSedenionClifford8.lean` → `gens_square_negI`,
  `gens_anticommute`, `nonanti_42`, `nonanti_all_lohi`.

## Reproduce

```bash
SOUNIO_STDLIB_PATH=$PWD/stdlib ./bin/souc run tests/run-pass/sedenion_clifford8.sio
python3 scripts/research/sedenion_clifford8_oracle.py
bash scripts/ci/sedenion_clifford8_gate.sh
(cd formal/lean4 && lake build SounioSedenionClifford8)
```

## Next defensible target
Certify the 2024 EPJC (Gourlay–Gresnigt) construction over ℚ: the S3-split of ℂ⊗𝕊 into three ℂ⊗𝕆
(shared quaternionic subalgebra), the ideal decomposition `T₁ = (Cℓ(6) ⊕ Cℓ(6)·a₄†)·v₁`, and the
`Q₁` charge assignment — turning a peer-reviewed three-generation construction into an executable,
certified artifact, with the physical interpretation flagged as the authors do.
