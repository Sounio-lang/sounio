# Functor F — the exact representation of the ord-3 secondary operation: `M = 2·V₃`

**Date:** 2026-07-26
**Orthography:** EN-UK
**Status:** `EXECUTABLE` — `ORD3_MODULE_IS_2xV3`
**Parent:** `functor_f_ord3_symmetry_fill_spec_2026-07-25.md` (`NO_INVARIANT_FILL` — the ord-3 secondary op is an invariant-free module)
**Harness:** `scripts/research/functor_f_ord3_module_decomp_contract.py`

---

## 0. The result

The previous rung proved the ord-3 secondary operation is *invariant-free* (no canonical
scalar fill exists, at any level). This rung asks the natural next question — *then what,
exactly, is that module?* — and answers it in closed representation-theoretic form:

> The ord-3 secondary ternary operation on the sedenion zero-divisor fibres,
> `(x, y) ↦ (x·y)·b`, restricted to one Fano-line support-class, spans a **6-dimensional real
> module `M`** for its symmetry group `G = 2³:S₄` (order 192, inside the full signed-octonion-
> automorphism group `1344 = (ℤ2)³:PSL(2,7)`). That module is **exactly**
> `M ≅ 2·V₃` — **two copies of a single 3-dimensional absolutely-irreducible real
> representation `V₃`, with no trivial constituent.**

This is a **new fact about a bespoke object.** The group and its irreps are entirely standard;
what had not been computed is the decomposition of *this* module — the ord-3 secondary
operation is this programme's own construction, so its representation-theoretic fingerprint is
ours to compute. The certificate is **numerical** (double precision, machine-epsilon
tolerances), not a symbolic proof.

`V₃` being **non-trivial** is *why* `NO_INVARIANT_FILL` holds: a module with no trivial summand
has no invariant line, so no canonical secondary value can exist. `2·V₃` is the precise
representation-theoretic **form** of that no-go.

---

## 1. Results

| Clause | Result | Reading |
|---|---|---|
| `M1_MODULE` | `|G| = 192 = (ℤ2)³⋊S₄`; the secondary module `M` has `dim = 6`; `G`-stable (dev `1.8e-15`) | the object is a genuine 6-dim `G`-module. |
| `M2_CHARACTER` | `⟨χ_M, χ_M⟩ = 4` (Σ of squared multiplicities); `⟨χ_M, 1⟩ = 0` (no invariant vector) | `M` is a sum of irreps with `Σ mᵢ² = 4` and **no trivial part**. |
| `M3_COMMUTANT` | `End_G(M)` **non-abelian**, `dim = 4` (`‖[T₁,T₂]‖ = 0.58`); a generic self-adjoint commutant element splits `{3, 3}` | `End_G(M) ≅ M₂(ℝ)` ⟺ **one** absolutely-irreducible constituent at **multiplicity 2**; the `{3,3}` split rules out the quaternionic (`ℍ`) alternative (which would give `{6}`). |
| `M4_V3_NONTRIVIAL` | `M = 2·V₃`, `V₃` a non-trivial 3-dim irrep ⟹ no trivial summand ⟹ invariant-free | the exact representation-theoretic **form** of `NO_INVARIANT_FILL`. |

Verdict: `FUNCTOR_F_ORD3MOD_VERDICT ORD3_MODULE_IS_2xV3`.

---

## 2. Why the inference is tight (not merely suggestive)

For a real orthogonal module, `dim_ℝ End_G(M)` and the eigenstructure of its self-adjoint part
pin the decomposition:

- `End_G(M)` non-abelian of `dim 4` ⟹ it is `M₂(ℝ)` **or** `ℍ` (the two non-abelian real
  algebras of dimension 4 that arise as a commutant). `M₂(ℝ)` ⟺ a single **absolutely-
  irreducible** constituent at **multiplicity 2**; `ℍ` ⟺ a single **quaternionic-type**
  6-dim irreducible.
- **The `{3,3}` split decides it.** In the `ℍ` case the only *symmetric* commutant elements are
  `ℝ·id` (the quaternion units `i,j,k` act skew), so a generic self-adjoint commutant element
  would be **scalar → one eigenspace of dim 6**. We observe **two eigenspaces of dim 3**, so the
  commutant is `M₂(ℝ)` and `V₃` is **absolutely irreducible** (real/orthogonal type).
- **Consistency.** `⟨χ_M,χ_M⟩ = 4 = 2²·1` (`⟨χ_{V₃},χ_{V₃}⟩ = 1` since `V₃` absolutely
  irreducible) and `⟨χ_M,1⟩ = 2·0 = 0` (`V₃` non-trivial) — both exactly the multiplicity-2
  absolutely-irreducible signature.

§10 math-review (Grok `[OK]`) confirmed all three links: the decomposition inference, the
`M₂(ℝ)`-vs-`ℍ` distinction, and the character arithmetic.

---

## 3. What this is NOT

- **Not** a claim that the *group* `2³:S₄` / `1344 = PSL(2,7)` or its irreps are new — they are
  entirely standard. The novelty is the decomposition of **this bespoke module**.
- **Not** a symbolic proof — a **numerical** certificate (double precision, `1e-9`..`1e-15`
  tolerances; the group elements are exact `0,±1` matrices, the SVD/eigenvalue steps are
  floating-point).
- **Not** the Petitot semantic conjecture (`petitot §4`, quarantined `D3`-class); **not** an
  identity; **not** clinical.

---

## 4. Reproduce

```bash
python3 scripts/research/functor_f_ord3_module_decomp_contract.py
# expect: M1..M4 PASS, FUNCTOR_F_ORD3MOD_VERDICT ORD3_MODULE_IS_2xV3
```

Pure Python (numpy). Builds the order-192 group from the signed-octonion collineation
transversal, forms the ord-3 secondary module `M`, and reads off `⟨χ,χ⟩`, `⟨χ,1⟩`, the
commutant dimension/commutativity, and the generic-eigenvalue split.

---

## 5. AI disclosure

Probe, contract, and note produced under human direction (2026-07-26), continuing the ord-3
thread ("find something genuinely uncomputed"). The result is a **new fact about a bespoke
object**: `M = 2·V₃` is the exact decomposition of *this programme's* ord-3 secondary module —
the group and its irreps are standard, the module is ours. The certificate is **numerical**
(machine precision), not symbolic. §10 math-review (Grok `[OK]` on the decomposition logic, the
`M₂(ℝ)`-vs-`ℍ` distinction, and the character arithmetic; framing tightened from "uncomputed"
to "this module realises `2·V₃`"; numerical-certificate caveat recorded). No new group, no
semantic claim, no clinical content. GAIDeT-ICMJE 2025.
