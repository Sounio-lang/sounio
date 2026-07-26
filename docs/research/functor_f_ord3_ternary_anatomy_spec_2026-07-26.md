# Functor F — anatomy of the ord-3 secondary operation: two-sided annihilation, a new reversal law, and the S+C split

**Date:** 2026-07-26
**Orthography:** EN-UK
**Status:** `EXECUTABLE` — `ORD3_TERNARY_ANATOMY`
**Parent:** `functor_f_ord3_module_decomp_spec_2026-07-26.md` (the module deflation → attack the operation itself)
**Harness:** `scripts/research/functor_f_ord3_ternary_anatomy_contract.py`

---

## 0. The result

After the module `M` deflated to the class coordinate space, the genuinely operation-specific
object is not the *image* but the **operation as an algebraic map** `(x,y) ↦ (x·y)·b` on the
sedenion zero-divisor fibres. Dissecting it — at higher risk, then pruned by advisor + §10 review —
yields one **new** identity, cleanly separated from what is **known** and what had to be **retracted**:

> **New (reversal law).** For every one of the 42 zero-divisors `b=e_i+e_j` and all `x,y ∈ F(b)`:
> `b·(x·y) = (y·x)·b`. Equivalently — since `y·x = −x·y − 2⟨x,y⟩e₀` for imaginary `x,y` — the clean
> anticommutator identity `{b, x·y} = −2⟨x,y⟩·b`: **`b`'s anticommutator with its own fibre's
> internal products stays along `b`.** §10 (Grok) judged this "a new computed relation", not a known
> consequence of conjugation/alternativity.

The operation then splits as `(x·y)·b = S + C` — a symmetric part `S` and a commutator part
`C = ½[x,y]·b` — both nonzero, reconnecting the ord-3 operation to the **associator/commutator**
theme the whole functor-F arc began with.

---

## 1. Results

| Clause | Result | Status |
|---|---|---|
| `T1_TWO_SIDED_ANNIHILATION` | all 42 ZD: `F(b)=ker L_b=ker R_b` (dim 4), `max‖x·b‖,‖b·x‖=1.1e-14` | **KNOWN** (Moreno 1998, Kivunge 2004) — cited, not claimed. |
| `T2_BRACKETING_COLLAPSE` | the 4 "b-first" bracketings `x(yb),(xb)y,x(by),(bx)y` vanish identically | consequence of `T1`; only "multiply the pair `x·y` first" survives. |
| `T3_REVERSAL_LAW` | all 42 ZD: `b(xy)=(yx)b` (dev `5.6e-16`); anticommutator form `{b,xy}=−2⟨x,y⟩b` (dev `6.3e-16`) | **NEW** (Grok) — a genuinely new computed identity (see §3 caveat). |
| `T4_OPERATION_SPLIT` | `(xy)b = S + C`; `‖S‖=18.3` (sym), `‖C‖=41.0` (commutator `½[x,y]b`); both nonzero | honest sym/antisym split (**not** a claim `S,C` are irreducible submodules). |

Verdict: `FUNCTOR_F_ORD3TERN_VERDICT ORD3_TERNARY_ANATOMY`.

---

## 2. The higher-temperature path — and the retraction it produced

The push "find something genuinely uncomputed" ran hot, and it is honest to record what it burned:

- **Retracted:** a probe claimed `dim Hom_G(D,M) = 6` — "a 6-dimensional space of `G`-equivariant
  ternary operations." The domain `D = ⊕_b F(b)⊗F(b)` tied the operation to specific `b`-**vectors**,
  whose `G`-orbit is **24** (four sign-variants `±e_i±e_j` per fibre) — while the **fibre** orbit is a
  clean **6**. So `D` as built was **not** a clean `G`-module and the count was ill-posed. §10 (Grok)
  flagged it `[WRONG]`; a direct orbit computation confirmed. Claim **withdrawn** — the second
  retraction this session (after the `2·V₃` module-fingerprint overclaim), and the discipline working.

What survived that pruning is `T1`–`T4`: modest, but real and honestly bounded.

---

## 3. What this is / is NOT

- **Is:** an honest anatomy — known two-sided annihilation ⟹ bracketing collapse ⟹ a **new**
  reversal/anticommutator identity ⟹ the operation `= S + C` (symmetric + commutator).
- **Caveat on the reversal law's novelty:** it holds for **all** 42 ZD (not only the `G`-stabilised
  fibres Grok assumed), so its generality means it may ultimately be a **derivable/known** ZD identity;
  it is flagged for a literature/citation check rather than claimed as deep.
- **Not** the retracted `dim Hom` count; **not** a claim `S,C` are irreducible `G`-submodules
  (would need a separate check); **not** symbolic (numerical certificate, machine precision);
  **not** the Petitot conjecture (`D3`-quarantined); **not** clinical.

---

## 4. Reproduce

```bash
python3 scripts/research/functor_f_ord3_ternary_anatomy_contract.py
# expect: T0..T4 PASS, FUNCTOR_F_ORD3TERN_VERDICT ORD3_TERNARY_ANATOMY
```

Pure Python (numpy); embeds a core axiom-audit and verifies `T1`–`T4` on all 42 sedenion ZD.

---

## 5. AI disclosure

Probe, contract, and note produced under human direction (2026-07-26), continuing the "find
something genuinely uncomputed" push at deliberately higher risk. Outcome, honestly bounded: the
**two-sided annihilation** (`T1`) is a known sedenion ZD fact (Moreno/Kivunge), cited not claimed;
the **bracketing collapse** (`T2`) is its consequence; the **reversal law / anticommutator identity**
(`T3`, `b(xy)=(yx)b ⟺ {b,xy}=−2⟨x,y⟩b`, all 42 ZD) is §10-reviewed as a new computed relation, with
a self-imposed caveat that its generality may make it a derivable/known identity (flagged for
citation); the **`S+C` split** (`T4`) is an honest sym/antisym decomposition, not an irreducibility
claim. One sub-claim from the hot path (`dim Hom_G(D,M)=6`) was **retracted** as ill-posed (domain not
a clean `G`-module — Grok `[WRONG]` + orbit check). §10 Grok reviewed all findings. Numerical
certificate; no semantic claim; no clinical content. GAIDeT-ICMJE 2025.
