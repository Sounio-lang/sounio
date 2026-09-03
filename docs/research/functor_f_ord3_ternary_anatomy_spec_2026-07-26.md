<!-- docs:meta
topic_id: repo.docs.research.functor-f-ord3-ternary-anatomy-spec-2026-07-26
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.functor-f-ord3-ternary-anatomy-spec-2026-07-26
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Functor F — anatomy of the ord-3 secondary operation: it is fixed by known octonion/sedenion structure

**Date:** 2026-07-26
**Orthography:** EN-UK
**Status:** `EXECUTABLE` — `ORD3_TERNARY_ANATOMY`
**Parent:** `functor_f_ord3_module_decomp_spec_2026-07-26.md` (the module deflation → attack the operation itself)
**Harness:** `scripts/research/functor_f_ord3_ternary_anatomy_contract.py`

---

## 0. The result — an honest negative

The push "find something genuinely uncomputed" was aimed, finally, at the ord-3 operation itself
`(x,y) ↦ (x·y)·b` on the sedenion zero-divisor fibres. Dissected fully and pruned by advisor + §10
review + a symbolic proof, the honest verdict is a **clean negative**:

> **The ord-3 secondary operation harbours no genuinely-new deep invariant — it is fixed by
> standard octonion/sedenion structure.** Its every structural feature reduces to two known facts:
> the **two-sided annihilation** `F(b)=ker L_b=ker R_b` (Moreno 1998, Kivunge 2004) and the
> **flexibility** of every Cayley-Dickson algebra (classical).

Concretely: the annihilation kills 4 of the 8 triple bracketings; the surviving **reversal law**
`b·(x·y) = (y·x)·b` (equivalently `{b,x·y} = −2⟨x,y⟩b`), which a first §10 pass had tentatively
called "a new computed relation", is proved here to be a **3-line corollary of flexibility** — the
flexibility law's shadow on the annihilator fibres. The operation splits as symmetric + commutator
`(x·y)·b = S + ½[x,y]·b`, an honest sym/antisym decomposition. Nothing here is new.

This is worth recording precisely *because* it is negative: it says where the genuinely-new
invariant is **not** — the ord-3 vein is structurally determined, so the arc should look elsewhere.

---

## 1. Results

| Clause | Result | Status |
|---|---|---|
| `T1_TWO_SIDED_ANNIHILATION` | all 42 ZD: `F(b)=ker L_b=ker R_b` (dim 4), `max‖x·b‖,‖b·x‖=1.1e-14` | **KNOWN** (Moreno 1998, Kivunge 2004). |
| `T2_BRACKETING_COLLAPSE` | the 4 "b-first" bracketings `x(yb),(xb)y,x(by),(bx)y` vanish | consequence of `T1`. |
| `T3_REVERSAL_LAW` | all 42 ZD: `b(xy)=(yx)b` (dev `5.6e-16`); anticommutator `{b,xy}=−2⟨x,y⟩b` | true, **but** proved below to be derivable. |
| `T3b_PROOF_VIA_FLEXIBILITY` | sedenions flexible; `b(xy)=−[b,x,y]`, `(yx)b=[y,x,b]`, `[b,x,y]=−[y,x,b]` | **3-line proof** ⟹ `T3` is a corollary, **not new**. |
| `T4_OPERATION_SPLIT` | `(xy)b = S + C`, `‖S‖=18.3`, `‖C‖=41.0` (`=½[x,y]b`); both nonzero | honest sym/antisym split (not an irreducibility claim). |

Verdict: `FUNCTOR_F_ORD3TERN_VERDICT ORD3_TERNARY_ANATOMY`.

---

## 2. The 3-line proof (reversal law = flexibility's shadow)

Every Cayley-Dickson algebra is **flexible**: `[a,b,c] = −[c,b,a]` (classical; verified for the
sedenions). With the two-sided annihilation `bx = xb = 0` for `x ∈ F(b)`:

```
(bx)y = 0   ⟹   b(xy) = −[b,x,y]          [since [b,x,y] = (bx)y − b(xy)]
y(xb) = 0   ⟹   (yx)b =  [y,x,b]          [since [y,x,b] = (yx)b − y(xb)]
flexibility ⟹   [b,x,y] = −[y,x,b]
∴  b(xy) − (yx)b = −[b,x,y] − [y,x,b] = 0.                              ∎
```

So the reversal law is a corollary of two documented facts. §10 (Grok) first flagged it as new,
reasoning from conjugation/alternativity and missing flexibility; the proof settles it. (Literature:
flexibility of CD algebras is classical — Schafer, nLab; sedenion ZD annihilators — Moreno 1998,
Kivunge 2004, arXiv:2411.18881 (2024), arXiv:math/0702075.)

---

## 3. The higher-temperature path, and what it burned

Running hot produced genuine reach and two honest retractions — both recorded:

- **Retracted (this session):** `dim Hom_G(D,M) = 6` ("a 6-dim space of `G`-equivariant ternary
  operations") — the domain `D` tied the operation to `b`-**vectors** (orbit 24), not the clean
  6-**fibre** orbit, so `D` was not a `G`-module; count ill-posed (Grok `[WRONG]` + orbit check).
- **Downgraded (this rung):** the reversal law from "new" to "3-line flexibility corollary."

Combined with the earlier `2·V₃` module-fingerprint overclaim (deflated to CD-doubling of the
coordinate space), the whole ord-3 "genuinely-new" chase closed negative — honestly, with proofs.

---

## 4. What this is / is NOT

- **Is:** a complete, honest anatomy showing the ord-3 operation is determined by known structure
  (two-sided annihilation + flexibility), with the reversal law proved as a corollary.
- **Not** a new identity or invariant; **not** the retracted `dim Hom` count; **not** an
  irreducibility claim for `S,C`; **not** symbolic beyond the elementwise numerical check of a
  human-readable proof; **not** the Petitot conjecture (`D3`); **not** clinical.

---

## 5. Reproduce

```bash
python3 scripts/research/functor_f_ord3_ternary_anatomy_contract.py
# expect: T0..T4 (incl. T3b) PASS, FUNCTOR_F_ORD3TERN_VERDICT ORD3_TERNARY_ANATOMY
```

---

## 6. AI disclosure

Probe, contract, note, and proof produced under human direction (2026-07-26), closing the "find
something genuinely uncomputed" push at high risk. **Honest outcome: negative.** The ord-3 operation
is fixed by known octonion/sedenion structure — two-sided annihilation (Moreno/Kivunge) + flexibility
(classical) — and the reversal law that looked new is a 3-line corollary (proof `T3b`). One hot-path
sub-claim (`dim Hom_G(D,M)=6`) was retracted as ill-posed. §10 Grok reviewed the findings (and its own
"new relation" read is corrected here by the flexibility proof). Numerical certificate of a
human-readable proof; no new invariant, no semantic claim, no clinical content. GAIDeT-ICMJE 2025.
