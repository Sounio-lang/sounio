# Round 16 — multi-namespace (`open_fillers`) vs `ns_only`

**Date:** 2026-08-06 · **Branch:** `research/zd-fiber-antisymmetry-lemma-20260731`

## Policy

| policy | meaning |
|---|---|
| `ns_only` | rounds 12–15: only `/{NS}_` classes as parents/fillers/disj partners |
| `open_fillers` | primary-NS classes **plus** any parent/filler/disj partner of a primary subject, **closed under superclasses** |

Mirrors: bitmask (H&lt;50k) or sparse sets (H≥50k), same as rounds 13–15.  
**Packed ns_only drivers stay the historical receipts.**  
**Sounio open_fillers driver** (round 16b): `open_fillers_elplus_driver.sio` on PATO+CL open — **ALL PASS**.

Generator: `gen_cross_ns_probe.py` · `extract_obo(..., policy=...)`.

---

### pato

| metric | ns_only | open_fillers | Δ |
|---|---:|---:|---:|
| H | 1887 | 1888 | 1 |
| foreign_interned | 0 | 1 | 1 |
| sub | 2227 | 2228 | 1 |
| exsub | 37 | 37 | 0 |
| disj | 67 | 67 | 0 |
| NR | 12 | 12 | 0 |
| atomic_edges | 12433 | 14321 | 1888 |
| role_edges | 4005 | 4841 | 836 |
| conf | 471806 | 471806 | 0 |
| amp (edges/exsub) | 108.2× | 130.8× | +22.6 |
| foreign_filler dropped by ns_only | 0 | (recovered in open) | — |
| foreign_parent dropped by ns_only | 1 | (recovered in open) | — |
| super_side / equiv_restr (probed) | 1/0 | 1/0 | — |

### cl

| metric | ns_only | open_fillers | Δ |
|---|---:|---:|---:|
| H | 3335 | 6206 | 2871 |
| foreign_interned | 0 | 2871 | 2871 |
| sub | 4664 | 8964 | 4300 |
| exsub | 477 | 5697 | 5220 |
| disj | 35 | 120 | 85 |
| NR | 29 | 173 | 144 |
| atomic_edges | 37926 | 87490 | 49564 |
| role_edges | 146188 | 1204329 | 1058141 |
| conf | 1071098 | 7418464 | 6347366 |
| amp (edges/exsub) | 306.5× | 211.4× | -95.1 |
| foreign_filler dropped by ns_only | 3754 | (recovered in open) | — |
| foreign_parent dropped by ns_only | 216 | (recovered in open) | — |
| super_side / equiv_restr (probed) | 1/2 | 1/2 | — |
| NEP (endpoints) | — | **162** | needs 4×64-bit epm words |

### uberon

| metric | ns_only | open_fillers | Δ |
|---|---:|---:|---:|
| H | 14975 | 16295 | 1320 |
| foreign_interned | 0 | 1320 | 1320 |
| sub | 19607 | 21371 | 1764 |
| exsub | 17080 | 19737 | 2657 |
| disj | 589 | 682 | 93 |
| NR | 128 | 202 | 74 |
| atomic_edges | 150515 | 229014 | 78499 |
| role_edges | 2343535 | 5190470 | 2846935 |
| conf | 25001610 | 44540444 | 19538834 |
| amp (edges/exsub) | 137.2× | 263.0× | +125.8 |
| foreign_filler dropped by ns_only | 2040 | (recovered in open) | — |
| foreign_parent dropped by ns_only | 11 | (recovered in open) | — |
| super_side / equiv_restr (probed) | 1/15 | 1/15 | — |

### chebi (item D)

| metric | ns_only | open_fillers | Δ |
|---|---:|---:|---:|
| H | 218253 | 218253 | **0** |
| foreign_interned | 0 | 0 | 0 |
| sub / exsub / disj | 307965 / 109108 / 0 | same | **0** |
| NR | 13 | 13 | 0 |
| atomic_edges | 5297445 | 5297445 | 0 |
| role_edges | 29846298 | 29846298 | 0 |
| conf | 0 | 0 | 0 |
| amp (edges/exsub) | 273.5× | 273.5× | 0 |
| foreign_filler / parent dropped | **0 / 0** | — | — |
| super_side / equiv_restr | 0/0 | 0/0 | — |

**ChEBI is namespace-closed under this cut:** every parent and filler of a CHEBI subject is already CHEBI. Round-15 ns_only numbers are **also** open_fillers numbers — no multi-namespace undercount on ChEBI (unlike CL/UBERON).

---

## Takeaway

- **pato**: 0 foreign fillers, **1 foreign parent** → H+1, **+836 role edges**, conf invariant.
- **cl**: drops **3754 foreign-filler axioms** / **216 foreign-parent axioms** (axiom counts ≠ unique H). Unique foreign interned **2871**. Role edges **146 188 → 1 204 329** (~**8.2×**). Marginal edges/exsub on recovered slice ≈202.7 (&lt; ns_only amp 306.5).
- **uberon**: **2040** foreign-filler axioms, unique foreign **1320**. Role edges **2.34M → 5.19M** (~**2.2×**); amp rises 137→263.
- **chebi**: **Δ = 0** on every metric — multi-namespace is a no-op; r15 is OWL-complete for the extracted shape.

## Sounio open_fillers drivers (16b–16c)

| target | packed | Sounio | notes |
|---|---|---|---|
| PATO open | `pato_open_packed.txt` | **ALL PASS** | NEP=118 · `open_fillers_elplus_driver.sio` |
| CL open | `cl_open_packed.txt` | **ALL PASS** | NEP=162, KMAX=16, 4-word epm · same |
| **UBERON open** | `uberon_open_packed.txt` | **ALL PASS** | NEP=957, NEPW=16 multi-word epm + actor-pair conf · `uberon_open_elplus_driver.sio` · role edges **5 190 470** |
| ChEBI open | ≡ ns_only packed | covered by r15 driver | Δ=0 |

Gate: both open drivers wired in `ontology_multi_ontology_gate.sh`.

## GO full sparse reaffirmation

`go_full_elplus_driver.sio` (r14): **ALL PASS** H=38 245, role edges=2 135 207, conf=792 814 846, rounds=4.

## Merge readiness (item A)

See `MERGE_READINESS_PR1580.md`: PR #1580 is **CONFLICTING** vs base (~60 commits); science side is pushed and documented; no auto-merge.
