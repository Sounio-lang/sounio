<!-- docs:meta
topic_id: repo.docs.research.functor-f-ord3-secondary-ternary-spec-2026-07-25
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.functor-f-ord3-secondary-ternary-spec-2026-07-25
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Functor F — ord-3 cross-column: the secondary ternary operation lives on the ord-2 ZD fibre

**Date:** 2026-07-25
**Orthography:** EN-UK
**Status:** `EXECUTABLE` — `SECONDARY_TERNARY_LOCATED` (a located bridge with a stated definability bound; not an identity)
**Parents:** `functor_f_g2_tower_closure_spec_2026-07-25.md` (`Q_GREEN`), `cd-tower` seam-coincidence / `lo⊕hi` ZD-fibre work (merged, PR #720 line)
**Empirical sibling:** `docs/gpu/BORROMEAN_AINFINITY.md` (a *different*, ML/path-signature argument reaching a compatible conclusion — see §6)
**Harness:** `scripts/research/functor_f_ord3_secondary_ternary_contract.py`

---

## 0. The question, and the object named honestly

The programme's ord-3 row is *Massey / Borromean* — higher-homotopy ternary linking,
where pairwise products vanish and only a secondary operation carries the signal. Does
the algebraic column reach it?

> **What this computes (named for what it is, per advisor).** Not "the Massey product":
> a Massey product requires a differential, and a plain (non-DG) algebra like 𝕊 has
> none. The object here is a **secondary ternary operation on the two-sided annihilator
> of a zero-divisor**, defined on the slice where the *primary* associator vanishes
> identically. What transfers from Massey theory is only the **definedness pattern**
> (secondary lives where primary vanishes), not the construction. Verdict type fixed in
> advance: operational, never an identity (D3 respected).

---

## 1. Results (all measured exactly; indeterminacy carries a generic baseline)

| Clause | Result | Reading |
|---|---|---|
| `T1_OCTONION_DOMAIN_EMPTY` | 𝕆 has no zero-divisors (`max dim ker L_b = 0`) | the slice `a·b=0` is **empty** in 𝕆 — the secondary-ternary domain does not exist. This is the **algebraic** reason the octonion associator (a *total, primary* operation) cannot be a secondary/Massey object. |
| `T2_SEDENION_DOMAIN_ON_ZD_FIBRE` | 𝕊: `42/105` of `e_i+e_j` are ZD; for each, `ker L_b = ker R_b = ` a **4-dim fibre**, uniform | the slice is nonempty and it **is the ord-2 ZD fibre** — the same `lo⊕hi` fibre of the merged `seam_coincidence` work. ord-3 attaches to ord-2. |
| `T3_PRIMARY_BLIND` | on the slice `[a,b,c] = (a·b)·c − a·(b·c) = 0` | both products vanish, so the **primary associator is blind** — any ord-3 signal there is genuinely secondary. |
| `T4_SECONDARY_DISTINGUISHABLE` | `dim(a·𝕊 + 𝕊·c) = 14` on the ZD slice vs **`16` for generic pairs** | the indeterminacy is *smaller* than generic, so the quotient where a secondary invariant would live is **2-dim (ZD) vs 0-dim (generic)** — distinguishable structure, **not** indeterminacy-swamped. |
| `T5_NO_BORROMEAN` | in-fibre annihilator-of-`v₀` has dim `0`; the triple is consecutive-only (`a·b=0, b·c=0, a·c≠0`) | **no all-pairwise-annihilating (Borromean) triple** exists within a fibre; and 𝕊 has no differential to fix a canonical value in the 2-dim quotient. |

Verdict: `FUNCTOR_F_ORD3_VERDICT SECONDARY_TERNARY_LOCATED`.

---

## 2. What this is — the first non-obstruction cross-column

Unlike the two prior cross-column edges (`B_OBSTRUCTED` algebra→Petitot,
`M_CHARACTERISED` algebra→ORC), this one is **not a pure obstruction**. It **locates**
the ord-3 secondary-ternary structure precisely:

- **Empty in 𝕆** (division algebra) — so the octonion associator is the *wrong* object
  (it is primary and total).
- **Nonempty in 𝕊, and equal to the ord-2 ZD fibre** — the secondary operation's domain
  is exactly where the algebra first develops zero-divisors, tying ord-3 to ord-2.
- **Carrying a nontrivial 2-dim quotient** (vs 0-dim generic) — there is genuine room
  for a secondary invariant.

The bridge is real up to a **definability bound**: without a differential 𝕊 does not
canonically fill that 2-dim quotient, and no Borromean (all-pairwise) triple exists — so
this is a *located domain with distinguishable structure*, not a canonical ord-3
invariant. Honest, bounded, and cross-order (ord-3 ↔ ord-2), which the other two edges
were not.

---

## 3. The structural spine (independently strong)

`ker L_b = ker R_b`, dim `4`, **uniform across all 42 zero-divisors** is the cleanest
fact here and it is not new-in-isolation: it is the two-sided annihilator = the `lo⊕hi`
ZD fibre of the merged `seam_coincidence` result (`SounioCDConverse.lean`, PR #720
line). This rung shows that fibre is *also* the natural home of the ord-3 secondary
operation — a second role for an already-certified object.

---

## 4. Contract clauses

(as in §1; `T0_CORE_AUDIT` additionally verifies both the octonion `bits=3` and
sedenion `bits=4` cores — identity, `e_i²=−1`, anticommutativity — before use.)

---

## 5. What this is NOT

- **Not the Massey product** — 𝕊 has no differential; only the definedness pattern
  transfers (§0).
- **Not a canonical ord-3 invariant** — the 2-dim quotient is *available*, not *filled*.
- **Not Borromean** — no all-pairwise-annihilating triple in a fibre (`T5`).
- **Not** `∀n` — this is the octonion→sedenion step (`bits=3→4`).
- **Not** an identity (`ord-3 ≡ ZD` is not claimed) — D3 respected.

---

## 6. Relation to the empirical `BORROMEAN_AINFINITY` result

`docs/gpu/BORROMEAN_AINFINITY.md` reached a compatible conclusion by a **different**
route: an ML path-signature task on which the octonion associator scored at chance
(48.9%) while the level-3 iterated-integral (Massey) feature defined the label. That is
an *empirical* statement about a *path* Massey product. This rung is the **algebraic**
complement: the octonion associator cannot be that object because its secondary-ternary
domain is *empty* (division algebra), independent of any dataset. The two arguments
agree; the algebraic one is stated here on its own terms and does **not** lean on the
48.9% number.

---

## 7. Place in the ladder

```
G_GREEN..Q_GREEN            algebraic column, closed (F equivariant functor; tower terminates ord-2)
B_OBSTRUCTED               algebra->Petitot (ord-P): cusp canonical, butterfly obstructed
M_CHARACTERISED            algebra->Ollivier-Ricci (ord-M): symmetry-coincidence obstruction
SECONDARY_TERNARY_LOCATED  algebra->Massey/Borromean (ord-3): located on the ord-2 ZD fibre
```

Three cross-column probes: two clean obstructions (ord-P, ord-M) and one **located,
bounded bridge** (ord-3 ↔ ord-2). The frontier is no longer uniformly negative.

---

## 8. Suggested next edges

1. **Fill the 2-dim quotient** — introduce the minimal structure (a differential / an
   A∞ `m₃` on the ZD fibre) that would define a canonical secondary value there, and
   test whether it is nonzero. This is the honest positive follow-up.
2. **`∀n` behaviour** — the ZD fibre and its dim-4 annihilator recur up the tower
   (`cd-tower` orbit theorem); does the 2-dim quotient persist at 𝕋 (bits=5)?
3. **External write-up** — fold this into the paper skeleton as the third cross-column
   result (the non-obstruction one).

---

## 9. Reproduce

```bash
python3 scripts/research/functor_f_ord3_secondary_ternary_contract.py
# expect: T0..T5 PASS, FUNCTOR_F_ORD3_VERDICT SECONDARY_TERNARY_LOCATED
```

Pure Python (numpy); Cayley-Dickson to `bits=4`; embeds the `T0` core axiom-audit.

---

## 10. AI disclosure

Probe, contract, and note produced under human direction (2026-07-25). Two advisor
rounds fixed this rung: the first mandated the annihilator dimension count before any
Borromean search; the second corrected the framing (secondary ternary operation, not
Massey — 𝕊 has no differential) and required the indeterminacy number to carry a generic
baseline (14 vs 16) before it could be called distinguishable. The pre-existing
`BORROMEAN_AINFINITY` negative is empirical and is cited as a *different* argument, not
as evidence for the algebraic claim. Claims bounded by the six named clauses. Commit
gated on the §10 math-review offload. No clinical content. GAIDeT-ICMJE 2025.
