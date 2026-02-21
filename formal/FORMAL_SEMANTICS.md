# Sounio Formal Semantics — Phase 8 Reference

Lean 4 formalization of the Sounio type system.
Source: `formal/lean4/` — no Mathlib, no `sorry`, `lake build` < 1 min.

---

## §1 Linear Type System — Modality Lattice

### Modalities

| Modality       | Use count | Weakening | Contraction | Must use |
|----------------|-----------|-----------|-------------|----------|
| `Linear`       | exactly 1 | ✗         | ✗           | ✓        |
| `Affine`       | ≤ 1       | ✓         | ✗           | ✗        |
| `Relevant`     | ≥ 1       | ✗         | ✓           | ✓        |
| `Unrestricted` | any       | ✓         | ✓           | ✗        |

### Lattice diagram (ordered by resource permissiveness)

```
       Unrestricted       ← top (join identity = Linear)
          /      \
      Affine    Relevant   ← incomparable (Affine ⊓ Relevant = Linear)
          \      /
          Linear           ← bottom (meet identity = Unrestricted)
```

`meet` = greatest lower bound (most restrictive).
`join` = least upper bound (most permissive).

Mirrors `Modality::combine` / `Modality::join` in `crates/souc/src/linear/modality.rs`.

---

## §2 Linear Typing Rules (natural-deduction style)

### Var rule

```
  x : τ [m] ∈ Γ      wellUsed m One
  ─────────────────────────────────── (Var)
            Γ ⊢ x : τ
```

### Weakening (Affine / Unrestricted only)

```
  Γ ⊢ e : τ      m.allowsWeakening     usage(x) = Zero
  ────────────────────────────────────────────────────── (Weak)
              x : σ [m], Γ ⊢ e : τ
```

### Contraction (Relevant / Unrestricted only)

```
  x : τ [m], x : τ [m], Γ ⊢ e : σ      m.allowsContraction
  ─────────────────────────────────────────────────────────── (Contr)
                   x : τ [m], Γ ⊢ e : σ
```

### Function introduction / elimination

```
  x : τ [m], Γ ⊢ e : σ
  ──────────────────────── (Fun-I)
  Γ ⊢ λx.e : τ →[m] σ

  Γ₁ ⊢ f : τ →[m] σ      Γ₂ ⊢ a : τ      Γ₁ ⊗ Γ₂ well-formed
  ──────────────────────────────────────────────────────────────── (Fun-E)
                      Γ₁ ∪ Γ₂ ⊢ f a : σ
```

### Tensor product

```
  Γ₁ ⊢ e₁ : τ₁      Γ₂ ⊢ e₂ : τ₂
  ─────────────────────────────────── (⊗-I)
       Γ₁ ∪ Γ₂ ⊢ (e₁, e₂) : τ₁ ⊗ τ₂

  Γ₁ ⊢ e : τ₁ ⊗ τ₂      x : τ₁ [m], y : τ₂ [m], Γ₂ ⊢ body : σ
  ──────────────────────────────────────────────────────────────── (⊗-E)
              Γ₁ ∪ Γ₂ ⊢ let (x, y) = e in body : σ
```

### Bang promotion / dereliction (Seely isomorphism)

```
  !Γ ⊢ e : τ      Γ.allUnrestricted
  ──────────────────────────────────── (Bang-I)
          !Γ ⊢ box e : !τ

  Γ ⊢ e : !τ      x : τ [Unrestricted], Γ' ⊢ body : σ
  ───────────────────────────────────────────────────── (Bang-E / Dereliction)
                 Γ ∪ Γ' ⊢ let !x = e in body : σ
```

Proved in Lean: `promotion_exhausted_any` — if every variable in Γ has modality
`Unrestricted`, then `Ctx.exhausted Γ ρ` holds for any usage environment `ρ`.

---

## §3 Algebraic Effect System

### Effect rows as characteristic functions

Effect rows are modelled as `EffectRow := Effect → Bool` — a decidable predicate
over the finite type `Effect`. This is mathematically equivalent to `Finset Effect`
via function extensionality, but requires no Mathlib.

### Named effects

| Effect      | Description                             | Rust constant               |
|-------------|-----------------------------------------|-----------------------------|
| `IO`        | Console, file, network I/O              | `types/core.rs:IO`          |
| `Mut`       | Mutable state                           | `types/core.rs:Mut`         |
| `Alloc`     | Memory allocation                       | `types/core.rs:Alloc`       |
| `Prob`      | Probabilistic sampling                  | `types/core.rs:Prob`        |
| `GPU`       | GPU kernel launch                       | `types/core.rs:GPU`         |
| `Epistemic` | Confidence / provenance operations      | `types/core.rs:Epistemic`   |
| `Div`       | Division (may be by zero)               | `types/core.rs:Div`         |
| `Exn`       | Exceptions                              | `types/core.rs:Exn`         |
| `Async`     | Asynchronous computation                | `types/core.rs:Async`       |
| `FFI`       | Foreign function interface              | `types/core.rs:FFI`         |

### Effect typing rules

```
  ─────────────────────────────────────── (Pure)
        Γ ⊢ e : τ  with  pureRow

  Γ ⊢ e : τ  with  ρ      e ∈ᵣ ρ'      ρ ⊆ ρ'
  ──────────────────────────────────────────────── (Subsumption)
               Γ ⊢ e : τ  with  ρ'

  Γ ⊢ e : τ  with  ρ ∪ {e}      handler h handles e
  ─────────────────────────────────────────────────── (Handle)
           Γ ⊢ handle h e : τ  with  ρ
```

The Handle rule is formalised by `handler_reduces_effects`:
`effectSubrow (mask r e) r` — the masked row is a subrow of the original.

### Row polymorphism

```
  Γ ⊢ f : (τ → σ  with  ρ)      ρ ⊆ ρ'
  ────────────────────────────────────── (Row-Poly)
       Γ ⊢ f : (τ → σ  with  ρ')
```

---

## §4 Proven Lean 4 Theorems

### SounioLinear — modality lattice

| Theorem | Informal statement |
|---------|-------------------|
| `modality_meet_comm` | `m ⊓ n = n ⊓ m` |
| `modality_meet_assoc` | `(m ⊓ n) ⊓ k = m ⊓ (n ⊓ k)` |
| `modality_join_comm` | `m ⊔ n = n ⊔ m` |
| `modality_join_assoc` | `(m ⊔ n) ⊔ k = m ⊔ (n ⊔ k)` |
| `modality_meet_idempotent` | `m ⊓ m = m` |
| `modality_join_idempotent` | `m ⊔ m = m` |
| `modality_absorption_meet` | `m ⊓ (m ⊔ n) = m` |
| `modality_absorption_join` | `m ⊔ (m ⊓ n) = m` |
| `modality_linear_is_bottom` | `Linear ⊓ m = Linear` |
| `modality_unrestricted_is_top` | `Unrestricted ⊔ m = Unrestricted` |
| `modality_unrestricted_meet_id` | `Unrestricted ⊓ m = m` |
| `modality_linear_join_id` | `Linear ⊔ m = m` |
| `modality_affine_relevant_incomparable` | `Affine ⊓ Relevant = Linear` (strictly incomparable) |
| `weakening_monotone` | If `m₁ ≤ m₂` and `m₁` allows weakening, so does `m₂` |

### SounioLinear — structural rules and usage

| Theorem | Informal statement |
|---------|-------------------|
| `linear_no_weakening` | Linear forbids weakening |
| `linear_no_contraction` | Linear forbids contraction |
| `affine_allows_weakening` | Affine permits weakening |
| `affine_no_contraction` | Affine forbids contraction |
| `relevant_no_weakening` | Relevant forbids weakening |
| `relevant_allows_contraction` | Relevant permits contraction |
| `unrestricted_both_rules` | Unrestricted allows both |
| `affine_zero_well_used` | `wellUsed Affine Zero` |
| `linear_zero_not_well_used` | `¬wellUsed Linear Zero` |
| `linear_many_not_well_used` | `¬wellUsed Linear Many` |
| `relevant_zero_not_well_used` | `¬wellUsed Relevant Zero` |
| `unrestricted_always_well_used` | `wellUsed Unrestricted u` for all `u` |
| `weakening_lemma` | Affine/Unrestricted entries can be dropped |
| `no_weakening_lemma` | Linear/Relevant entries force non-zero usage |
| `promotion_exhausted_any` | All-Unrestricted context is always exhausted (Bang-I) |

### SounioLinear — distributive lattice and structural monotonicity

| Theorem | Informal statement |
|---------|-------------------|
| `modality_meet_distrib` | `m ⊓ (n ⊔ k) = (m ⊓ n) ⊔ (m ⊓ k)` |
| `modality_join_distrib` | `m ⊔ (n ⊓ k) = (m ⊔ n) ⊓ (m ⊔ k)` |
| `contraction_monotone` | If `m₁ ≤ m₂` and `m₁` allows contraction, so does `m₂` |
| `allowsWeakening_join` | Weakening is preserved under join |
| `allowsContraction_join` | Contraction is preserved under join |
| `mustUse_meet` | MustUse is preserved under meet |
| `wellUsed_one_always` | Every modality accepts exactly-one usage |
| `wellUsed_join_left` | If `m` accepts `u`, so does `m ⊔ n` |
| `not_wellUsed_meet` | If `m` rejects `u`, so does `m ⊓ n` |
| `usage_many_absorb_left` | `Many + u = Many` |
| `usage_many_absorb_right` | `u + Many = Many` |
| `usage_one_plus_one` | `One + One = Many` |
| `usage_add_zero_iff` | `u + v = Zero ↔ u = Zero ∧ v = Zero` |
| `usage_add_many_iff` | `u + v = Many ↔ u = Many ∨ v = Many ∨ (u = One ∧ v = One)` |

### SounioEffects — effect row algebra (core)

| Theorem | Informal statement |
|---------|-------------------|
| `mask_idempotent` | `mask (mask r e) e = mask r e` |
| `mask_comm` | `mask (mask r e₁) e₂ = mask (mask r e₂) e₁` |
| `mask_absent_noop` | `e ∉ r → mask r e = r` |
| `mask_removes` | `e ∉ mask r e` |
| `mask_preserves_other` | `f ≠ e → (f ∈ mask r e ↔ f ∈ r)` |
| `rowUnion_comm` | `r₁ ∪ r₂ = r₂ ∪ r₁` |
| `rowUnion_assoc` | `(r₁ ∪ r₂) ∪ r₃ = r₁ ∪ (r₂ ∪ r₃)` |
| `rowUnion_idempotent` | `r ∪ r = r` |
| `rowUnion_pure_left/right` | `∅ ∪ r = r`, `r ∪ ∅ = r` |
| `effectSubrow_refl` | `r ⊆ r` |
| `effectSubrow_trans` | `r₁ ⊆ r₂ → r₂ ⊆ r₃ → r₁ ⊆ r₃` |
| `effectSubrow_antisymm` | `r₁ ⊆ r₂ → r₂ ⊆ r₁ → r₁ = r₂` |
| `handler_reduces_effects` | `mask r e ⊆ r` |
| `handler_idempotent` | `mask (mask r e) e = mask r e` |
| `handler_order_independence` | handlers for distinct effects commute |
| `single_mask_pure` | `mask (singleRow e) e = pureRow` |
| `effectSubrow_union_left/right` | `r₁ ⊆ r₁ ∪ r₂`, `r₂ ⊆ r₁ ∪ r₂` |
| `effectSubrow_union_lub` | union is the least upper bound |
| `mask_union_right` | `r₁ e = false → mask (r₁ ∪ r₂) e = r₁ ∪ mask r₂ e` |

### SounioEffects — singleton membership and intersection

| Theorem | Informal statement |
|---------|-------------------|
| `singleRow_member` | `e ∈ singleRow e` |
| `singleRow_not_member` | `f ≠ e → f ∉ singleRow e` |
| `memberOf_single_iff` | `(e ∈ singleRow f) ↔ e = f` |
| `mask_single_other` | masking a different effect from a singleton is a no-op |
| `memberOf_pure_false` | no effect belongs to the pure row |
| `memberOf_union_iff` | `(e ∈ r₁ ∪ r₂) ↔ e ∈ r₁ ∨ e ∈ r₂` |
| `memberOf_inter_iff` | `(e ∈ r₁ ∩ r₂) ↔ e ∈ r₁ ∧ e ∈ r₂` |
| `pureRow_unique` | `(∀ e, e ∉ r) → r = ∅` |
| `rowInter_comm` | `r₁ ∩ r₂ = r₂ ∩ r₁` |
| `rowInter_assoc` | `(r₁ ∩ r₂) ∩ r₃ = r₁ ∩ (r₂ ∩ r₃)` |
| `rowInter_idempotent` | `r ∩ r = r` |
| `rowInter_pure_left/right` | `∅ ∩ r = ∅`, `r ∩ ∅ = ∅` |
| `effectSubrow_inter_left/right` | `r₁ ∩ r₂ ⊆ r₁`, `r₁ ∩ r₂ ⊆ r₂` |
| `effectSubrow_inter_glb` | intersection is the greatest lower bound |
| `rowInter_union_distrib` | `r₁ ∩ (r₂ ∪ r₃) = (r₁ ∩ r₂) ∪ (r₁ ∩ r₃)` |
| `rowUnion_inter_distrib` | `r₁ ∪ (r₂ ∩ r₃) = (r₁ ∪ r₂) ∩ (r₁ ∪ r₃)` |
| `rowInter_union_absorb` | `r₁ ∩ (r₁ ∪ r₂) = r₁` |
| `rowUnion_inter_absorb` | `r₁ ∪ (r₁ ∩ r₂) = r₁` |

### SounioEffects — monotonicity, complement, all-effects, disjointness

| Theorem | Informal statement |
|---------|-------------------|
| `effectSubrow_union_mono_left/right` | union is monotone in each argument |
| `effectSubrow_mask_mono` | masking is monotone in the subrow order |
| `effectSubrow_inter_mono_left` | intersection is monotone on the left |
| `rowComplement_involution` | `¬¬r = r` (double complement) |
| `rowComplement_union` | `¬(r₁ ∪ r₂) = ¬r₁ ∩ ¬r₂` (De Morgan) |
| `rowComplement_inter` | `¬(r₁ ∩ r₂) = ¬r₁ ∪ ¬r₂` (De Morgan) |
| `rowUnion_complement_full` | `r ∪ ¬r = ⊤` |
| `rowInter_complement_pure` | `r ∩ ¬r = ∅` |
| `effectSubrow_complement_antimono` | complement reverses the subrow order |
| `effectSubrow_allEffects` | every row is a subrow of `⊤` |
| `rowUnion_allEffects_left/right` | `⊤ ∪ r = ⊤`, `r ∪ ⊤ = ⊤` |
| `rowInter_allEffects_left/right` | `⊤ ∩ r = r`, `r ∩ ⊤ = r` |
| `rowComplement_pure_is_all` | `¬∅ = ⊤` |
| `rowComplement_all_is_pure` | `¬⊤ = ∅` |
| `rowDisjoint_comm` | disjointness is symmetric |
| `rowDisjoint_pure_left/right` | `∅ ⊥ r`, `r ⊥ ∅` |
| `rowDisjoint_single_absent` | `e ∉ r → {e} ⊥ r` |
| `rowDisjoint_single_mask` | `{e} ⊥ mask r e` |
| `rowDisjoint_union_inter` | `(r₁ ⊥ r₃) ∧ (r₂ ⊥ r₃) → (r₁ ∪ r₂) ⊥ r₃` |
| `maskAll_pure` | applying handlers for all 10 effects yields the pure row |
| `mask_union_left` | `r₁ e = false → mask (r₁ ∪ r₂) e = r₁ ∪ mask r₂ e` |
| `mask_inter` | `mask (r₁ ∩ r₂) e = (mask r₁ e) ∩ (mask r₂ e)` |
| `handler_chain_reduces` | `mask (mask r e₁) e₂ ⊆ r` |
| `mask_union_single_superset` | `r ⊆ (mask r e) ∪ {e}` |

---

## §5 References

- Girard, J.-Y. (1987). "Linear Logic." *Theoretical Computer Science* 50(1):1–102.
- Wadler, P. (1990). "Linear Types Can Change the World!" *IFIP TC 2 Working Conference.*
- Bierman, G. (1994). "On Intuitionistic Linear Logic." PhD thesis, University of Cambridge.
- Plotkin, G. and Pretnar, M. (2009). "Handlers of Algebraic Effects." *ESOP.*
- Leijen, D. (2014). "Koka: Programming with Row Polymorphic Effect Types." *HOPE.*
- Lindley, S. and Cheney, J. (2012). "Row-based Effect Types for Database Integration." *TLDI.*
