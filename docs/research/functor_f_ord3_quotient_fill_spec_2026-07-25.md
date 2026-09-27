<!-- docs:meta
topic_id: repo.docs.research.functor-f-ord3-quotient-fill-spec-2026-07-25
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.functor-f-ord3-quotient-fill-spec-2026-07-25
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Functor F — ord-3 follow-up: the 2-dim quotient is reachable but not canonically fillable

**Date:** 2026-07-25
**Orthography:** EN-UK
**Status:** `EXECUTABLE` — `NO_CANONICAL_FILL` (sharpens, does not overturn, `SECONDARY_TERNARY_LOCATED`)
**Parent:** `functor_f_ord3_secondary_ternary_spec_2026-07-25.md` (`SECONDARY_TERNARY_LOCATED`)
**Harness:** `scripts/research/functor_f_ord3_quotient_fill_contract.py`

---

## 0. The question and the test

The ord-3 rung located a nontrivial 2-dim quotient `Q = 𝕊/(a·𝕊 + 𝕊·c)` on the sedenion
ZD fibre where a secondary ternary invariant could live. Does the sedenion algebra fill
`Q` with a **canonical** nonzero value, or does any fill require a **chosen** datum?

> **Falsifiability rule (advisor).** A value is canonical iff it does not move as its
> auxiliary datum ranges. Three outcomes were named **before computing**:
> constant-nonzero → a canonical invariant (genuine positive); constant-zero → the
> algebra lands at the origin; datum-varying → no canonical fill. The verdict follows the
> sweep, not the reverse.

---

## 1. Results

| Clause | Result | Reading |
|---|---|---|
| `U1_QUOTIENT_2DIM` | `dim Q = 2` | the room for a secondary value (re-confirmed). |
| `U2_CONSECUTIVE_VANISH` | `proj_Q(a·c) = proj_Q(c·a) = 0` | the naive fill (the surviving non-consecutive product) falls into the indeterminacy. |
| `U3_INTRINSIC_REACHES_Q` | the composites `(a·c)·b, b·(a·c), (c·a)·b, b·(c·a)` have nonzero Q-image (`min ‖·‖ = 2.5`) | `Q` **is** reachable by intrinsic ternary products — this is **not** "the algebra lands at the origin". |
| `U4_BRACKETING_IS_DATUM` | those four composites point in **different** Q-directions and span **rank 2** (the full quotient), **robustly across all 42 ZD** | the auxiliary datum is the **bracketing/ordering** of the ternary composite — forced to matter by non-associativity — and it **sweeps all of Q**. The fill is *selected*, not *forced*. |
| `U5_NO_DIFFERENTIAL` | `a·b`, `b·c` are **strict** zeros, not `d(u)=a·b` | 𝕊 has no differential, so the classical Massey construction — which picks a distinguished representative mod indeterminacy — cannot run. |

Verdict: `FUNCTOR_F_ORD3FILL_VERDICT NO_CANONICAL_FILL`.

---

## 2. The honest near-miss (recorded deliberately)

A first look found that `(a·c)·b` is a choice-free, nonzero element of `Q` orthogonal to
the trivial `[b]`-echo — which *looked* like a canonical secondary invariant. The
bracketing test refuted it: `(a·c)·b` and `b·(a·c)` and `(c·a)·b` land in **different**
Q-directions, and together the orderings sweep the whole quotient. So there is no
distinguished composite; "the secondary value" is not well-defined without choosing a
bracketing, and 𝕊 offers nothing to make that choice. Reporting the near-miss is the
point: a positive here would have been a bracketing chosen to look canonical.

---

## 3. What this means

`Q` is genuinely 2-dimensional and genuinely reachable — but the sedenion algebra, being
non-associative **and** differential-free, has no canonical way to place a value in it:
every intrinsic ternary product is bracketing-dependent, and the dependence covers all
of `Q`. So the ord-3 secondary structure **remains located but unfilled**: filling it
requires *imposed* A∞ structure (a differential / a distinguished `m₃`), which is exactly
the honest positive follow-up left open by the parent rung. `SECONDARY_TERNARY_LOCATED`
stands, now sharpened: the 2-dim room is real, and empty of a canonical occupant.

---

## 4. What this is NOT

- **Not** a canonical secondary invariant (the `(a·c)·b` candidate is bracketing-selected).
- **Not** "the algebra lands at the origin" — `Q` is reachable (`U3`).
- **Not** a Massey product — 𝕊 has no differential (`U5`); only the definedness pattern
  transferred, and here it stops at non-canonicity.
- **Not** a claim that no `A∞`/differential structure could *ever* fill `Q` canonically —
  imposing such structure and testing whether it lands nonzero is the open positive
  follow-up; this rung shows only that the **bare** sedenion algebra does not.
- **Not** D3, not clinical, not an identity.

---

## 5. Reproduce

```bash
python3 scripts/research/functor_f_ord3_quotient_fill_contract.py
# expect: U0..U5 PASS, FUNCTOR_F_ORD3FILL_VERDICT NO_CANONICAL_FILL
```

Pure Python (numpy); Cayley-Dickson `bits=4`; embeds the `U0` core axiom-audit.

---

## 6. AI disclosure

Probe, contract, and note produced under human direction (2026-07-25) across three
advisor rounds (annihilator dim-count; generic baseline for the indeterminacy;
canonical-vs-imposed by datum sweep). A first computation suggested a canonical secondary
invariant; a bracketing-consistency test refuted it, and that near-miss is recorded in §2
rather than suppressed. Claims bounded by the six named clauses. Commit gated on the §10
math-review offload. No clinical content. GAIDeT-ICMJE 2025.
