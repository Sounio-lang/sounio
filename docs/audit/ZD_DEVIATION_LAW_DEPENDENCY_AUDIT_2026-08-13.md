<!-- docs:meta
topic_id: repo.docs.audit.zd-deviation-law-dependency-audit-2026-08-13
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.zd-deviation-law-dependency-audit-2026-08-13
-->

# Dependency audit — the deviation law (Tier 161) and its external base case

**Date:** 2026-08-13
**Auditor:** Claude (the T1/transfer lane)
**Subject:** `deviation_law` (`formal/lean4/SounioZDFiberAntisym.lean`, Tier 161, `f6676d1cab`)
and the half of its base case that comes from the concurrent lane —
`s3_reference_closed7` (Tier 166, `b1a927f047`).

**Why:** half of obligation (iii) of §57.49 rests on a theorem this lane did not write. Committing
Tier 161 without auditing it would have made the headline law depend on an unverified import.

---

## 1. Structural findings

| Check | Result |
|---|---|
| `#print axioms s3_reference_closed7` | `[propext, Classical.choice, Quot.sound]` — the file's baseline |
| `#print axioms deviation_law` | same |
| `sorry` / `axiom` tokens in the file | none (doc mentions only) |
| `native_decide` | not used anywhere in the file |
| `s3_reference_closed7` hypotheses | **none** — `(k : Nat)` only |
| `tier164_hdc` (the last open item of E5 as of Tier 162) | **unconditional theorem**, induction on `k`, base case `decide` |

`s3_reference_closed7` takes no hypotheses and introduces no axioms, so nothing is assumed and
left undischarged. Its internal `hDeltadef` / `hnetdef` style parameters are naming devices,
supplied by `rfl` at the call site; `hiso'` and `hdeg'` are built from `hiso_ref` (Tier 158) and
`defect_regular_free` (Tiers 144+150), both theorems.

The conditional chain closes: `triangle_count_hrule` needs `hmult`, supplied by `tier162_hmult`,
which needs `hdc`, supplied by `tier164_hdc` — which is unconditional. No link is left hanging.

## 2. Numeric verification — independent of the whole proof chain

Every number below is `#eval` of `tri3 N (fun x y => P3 x y W m)` — i.e. computed from the
DEFINITION of `P3`, touching none of the ~3000 lines of proof under audit.

| level `m` | `s3` at `W = 1` | predicted by `s3_reference_closed7` | agree |
|---:|---:|---:|:--:|
| 3 | −272 | −272 | ✓ |
| 4 | −4560 | −4560 | ✓ |
| 5 | −53072 | −53072 | ✓ |

| level `m` | `s3` at `W = 8` | predicted by `s3_pow2_closed` (Tier 159) | agree |
|---:|---:|---:|:--:|
| 3 | 1456 | 1456 (maximal seam, Tier 136) | ✓ |
| 4 | 9264 | 9264 | ✓ |
| 5 | 57520 | 57520 | ✓ |

**The deviation law itself**, `D = 1728·8^(m−j)·[j,3]₂`, evaluated as a difference of two raw sums:

| `j` | `m` | measured `D` | law | agree |
|---:|---:|---:|---:|:--:|
| 3 | 3 | 1728 | `1728·8⁰·1` | ✓ |
| 3 | 4 | 13824 | `1728·8¹·1` | ✓ |
| 3 | 5 | 110592 | `1728·8²·1` | ✓ |
| 4 | 4 | 25920 | `1728·8⁰·15` | ✓ |

At `j = m` (rows 1 and 4) the law holds with **no** `288·[m−1,2]₂` correction, which is the
independent confirmation of §57.49's reading that the maximal-seam exception is an artifact of
the mask rather than a property of `tri3`.

## 3. What this audit does NOT establish

- It does not verify that the combinatorial definitions (`isDefect`, `ordQualOrbit`,
  `indTripleCount`, `t3Count`) model what their names and docstrings say. What it does establish
  is that whatever they model, the resulting VALUE agrees with a direct evaluation of `tri3` at
  three levels — so an error in that modelling would have to be value-preserving at `m = 3,4,5`.
- It does not audit the proof bodies line by line. The argument is structural (no hypotheses, no
  axioms, no `sorry`, no `native_decide`) plus numeric, not a reading of the derivation.
- `deviation_law`'s scope is unchanged by this audit: `j ≥ 3`, levels `m ≥ j`, reference pairs
  `W = 2^j` against `W = 1`, unmasked `tri3`.

## 4. Verdict

The external dependency is sound to the standard this lane uses elsewhere. Tier 161 may be
quoted without an "unaudited import" caveat, subject to §3.
