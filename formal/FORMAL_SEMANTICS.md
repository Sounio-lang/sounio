# Sounio Formal Semantics — Phases 8–12 Reference

Lean 4 formalization of the Sounio type system.
Source: `formal/lean4/` — no Mathlib, no `sorry`, `lake build` < 1 min (25 jobs).

**12 libraries, ~5539 lines, ~601 theorems.**

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

---

## §3 Algebraic Effect System

### Effect rows as characteristic functions

Effect rows are modelled as `EffectRow := Effect → Bool` — a decidable predicate
over the finite type `Effect`. This is mathematically equivalent to `Finset Effect`
via function extensionality, but requires no Mathlib.

### Named effects

| Effect      | Description                             |
|-------------|-----------------------------------------|
| `IO`        | Console, file, network I/O              |
| `Mut`       | Mutable state                           |
| `Alloc`     | Memory allocation                       |
| `Prob`      | Probabilistic sampling                  |
| `GPU`       | GPU kernel launch                       |
| `Epistemic` | Confidence / provenance operations      |
| `Div`       | Division (may be by zero)               |
| `Exn`       | Exceptions                              |
| `Async`     | Asynchronous computation                |
| `FFI`       | Foreign function interface              |

---

## §4 CBV Operational Semantics (Phase 11)

### Small-step reduction (`Step e e'`)

11 rules: β-reduction, letP-β, letB-β, + 8 congruences (app-left, app-right,
pair-left, pair-right, box, letP-scrutinee, letB-scrutinee).

### Values (`IsValue`)

Three forms: `lam`, `pair` (of values), `box` (of value).

### Key theorems

| Theorem | Statement |
|---------|-----------|
| `step_deterministic` | CBV has at most one reduct |
| `value_irreducible` | Values do not step |
| `multistep_one` | Single step lifts to multi-step |
| `multistep_trans` | Multi-step is transitive |

---

## §5 Progress Theorem (Phase 12)

### Canonical forms

| Theorem | Statement |
|---------|-----------|
| `canonical_fun_form` | Value of function type → lambda |
| `canonical_prod_form` | Value of product type → pair |
| `canonical_bang_form` | Value of bang type → box |

### Progress

| Theorem | Statement |
|---------|-----------|
| `progress` | `Typing [] e τ ρ → IsValue e ∨ ∃ e', Step e e'` |
| `progress_xor` | The dichotomy is exclusive |
| `progress_not_stuck` | Closed well-typed terms are never stuck |
| `closed_normal_is_value` | Normal forms of closed well-typed terms are values |
| `app_always_steps` | Applications always step |
| `letP_always_steps` | Let-pairs always step |
| `letB_always_steps` | Let-bangs always step |

---

## §6 Substitution Theory (Phase 12)

### Free variables

| Definition/Theorem | Statement |
|---------|-----------|
| `freeVars` | Free variable computation |
| `subst_not_free` | Substituting non-free variable is identity |
| `freeVars_lam_shadow` | Bound variable not in free vars |

### Context operations

| Theorem | Statement |
|---------|-----------|
| `typing_append_right` | Context extension on the right |
| `typing_closed_weakening` | Closed terms typable in any context |
| `substitution_closed_value` | Substitution lemma for closed values |

---

## §7 Type Safety Framework (Phase 12)

### Preservation (parametric)

Progress and preservation are axiomatized as `Prop` hypotheses following
Wright-Felleisen (1994). The safety theorems hold for ANY calculus satisfying both.

| Theorem | Statement |
|---------|-----------|
| `type_safety` | `Pres → Prog → Typing [] e τ ρ → MultiStep e e' → ¬Stuck e'` |
| `typed_normal_is_value` | Multi-step normal forms are values |
| `evaluatesTo_unique` | Evaluation is deterministic |
| `multistep_preservation` | Single-step preservation lifts to multi-step |
| `Preserves` combinator | Generic property preservation under reduction |

### Multi-step counted reduction

| Theorem | Statement |
|---------|-----------|
| `MultiStepN` | Counted reduction (step count tracked) |
| `multiStepN_deterministic` | Same count + same start = same result |
| `multiStepN_trans` | Composition with count addition |

---

## §8 Epistemic Type Theory (Phase 12, NOVEL)

Formalizes `Knowledge[T, ε]` as a bounded confidence lattice with provenance tracking.
No prior Lean 4 project covers epistemic types.

### Confidence lattice

- `confMeet` / `confJoin`: bounded lattice (comm, assoc, idempotent, absorption)
- `confBot`: minimum confidence (0)
- Full lattice laws proved (8 theorems)

### Knowledge operations

| Operation | Description |
|-----------|-------------|
| `measure c` | Direct measurement: confidence `c`, provenance 0 |
| `derive k d` | Derived knowledge: confidence degrades by `d` |
| `combine k₁ k₂` | Conservative: min confidence, max provenance |
| `strengthen k₁ k₂` | Optimistic: max confidence, min provenance |
| `bayesianUpdate prior evidence bound` | Posterior update |

### Knowledge ordering

`knowledgeLeq k₁ k₂` = higher confidence AND shorter provenance chain.
Proved: partial order (refl, trans, antisymm).

### Uncertainty propagation

`uncertainty k bound = bound - k.confidence`

| Theorem | Statement |
|---------|-----------|
| `uncertainty_zero_is_certain` | Maximum confidence → zero uncertainty |
| `combine_uncertainty_ge` | Combining increases uncertainty |
| `strengthen_uncertainty_le` | Strengthening decreases uncertainty |

### Effect system connection

- `epistemicRow`: epistemic effect singleton row
- `epistemic_handled`: masking epistemic yields pure
- `epistemic_disjoint_gpu`: epistemic and GPU are independent

---

## §9 Causal Type Theory (Phase 12, NOVEL)

Formalizes Pearl's structural causal models (SCMs) as a type-level discipline.
No prior Lean 4 project covers causal inference as types.

### Structural causal models

- `SCM n`: graph with `n` variables + topological ordering (acyclicity)
- `isTopological`: edges go from lower to higher index
- `topo_no_self_loop`: no self-loops in DAGs

### Reachability (ancestral relation)

| Theorem | Statement |
|---------|-----------|
| `reachable_trans` | Reachability is transitive |
| `reachable_lt` | Reachable implies strict index ordering |
| `reachable_irrefl` | No variable reaches itself (acyclicity) |
| `reachable_antisymm` | Antisymmetric in DAGs |

### Intervention (do-operator)

`intervene edges t` removes all incoming edges to `t`.

| Theorem | Statement |
|---------|-----------|
| `intervene_removes_parents` | All parents severed |
| `intervene_preserves_other` | Non-target edges unchanged |
| `intervene_idempotent` | Double intervention is identity |
| `intervene_comm` | Interventions commute |
| `intervene_preserves_topo` | Topological ordering preserved |
| `intervene_unreachable_target` | Target unreachable after intervention |

### d-Separation

| Theorem | Statement |
|---------|-----------|
| `dsep_empty_iff_no_path` | d-separated by ∅ iff no path |
| `dsep_target_in_z` | Target in Z → d-separated |
| `dsep_monotone` | Enlarging Z preserves d-separation |

### Pearl's do-calculus (simplified)

| Rule | Condition | Statement |
|------|-----------|-----------|
| Rule 1 | `rule1Holds` | Z irrelevant for Y given do(X) |
| Rule 2 | `rule2Holds` | do(Z) = observe(Z) under conditions |
| Rule 3 | `rule3Holds` | do(Z) can be removed |

### Identification criteria

- `backDoorCriterion`: blocks confounding paths
- `frontDoorCriterion`: mediator identification
- `instrumentalVariable`: IV identification

### Effect system connection

- `causalEffectRow = singleRow .Mut`: interventions require Mut
- `observationalRow = pureRow`: observations are pure
- `handle_mut_reduces_to_obs`: handling Mut yields observational

---

## §10 References

- Girard, J.-Y. (1987). "Linear Logic." *Theoretical Computer Science* 50(1):1–102.
- Wadler, P. (1990). "Linear Types Can Change the World!" *IFIP TC 2.*
- Bierman, G. (1994). "On Intuitionistic Linear Logic." PhD, Cambridge.
- Plotkin, G. and Pretnar, M. (2009). "Handlers of Algebraic Effects." *ESOP.*
- Wright, A. and Felleisen, M. (1994). "A Syntactic Approach to Type Soundness." *I&C* 115(1):38–94.
- Pierce, B. (2002). "Types and Programming Languages." MIT Press.
- Pearl, J. (2009). "Causality: Models, Reasoning, and Inference." 2nd ed. Cambridge.
- van Ditmarsch, H. et al. (2007). "Dynamic Epistemic Logic." Springer.
- Halpern, J. (2003). "Reasoning About Uncertainty." MIT Press.
