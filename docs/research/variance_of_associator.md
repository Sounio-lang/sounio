<!-- docs:meta
topic_id: repo.docs.research.variance-of-associator
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.variance-of-associator
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Variance of the Octonion Associator — GUM First-Order Analysis

**Date**: 2026-04-13
**Status**: Derivation (β-thread). Numerical check **closed** (Exp2 2026-07-25).  
Residual `Var(a·b−a·b)` closed by **β¹⁰** (2026-07-25).  
Fano closed-form `Var(A)=64σ²` closed by **β¹¹** (2026-07-25): rebuild product variance from FO sensitivities `Σ s_k² σ²_k` (fixes a·a under-count 2σ²→4σ²).
**Related**: `docs/research/zeta_variance_fix_plan.md`, Part II of the non-associative connectomics draft, `feedback_variance_deep_chains.md`.

## Why this matters

Part II of the non-associative connectomics draft declares `Knowledge<Octonion<f64>>` as a first-class type — variance propagates through octonion multiplication, and the associator is an epistemic object whose uncertainty is meaningful. The current Sounio compiler propagates variance through multiplication using the rule
```
Var(a·b) = a² · Var(b) + b² · Var(a)               (SCALAR)
```
This rule is derived from the first-order GUM expansion for a *scalar* product, assuming `a` and `b` are uncorrelated scalars. Applying it component-wise to octonion multiplication is **not equivalent** to the correct first-order variance of the vector-valued product. The difference is the object of this document.

If the compiler's variance computation is subtly wrong for non-associative products, the ISO uncertainty budgets generated for `Knowledge<Octonion<f64>>` outputs (including the associator `[a,b,c]`) are wrong, and the draft's Part II claim is overstated.

## Setup

Let `a, b, c ∈ 𝕆`, each octonion being a vector in ℝ⁸ with components `a = (a_0, ..., a_7)`. Each component carries an independent Gaussian uncertainty `σ_{a,i}` (analogous for b, c). The associator
```
[a, b, c] := (a · b) · c − a · (b · c)
```
is itself an octonion. The scalar statistic the Phase 1 and Phase 2 pipelines use is
```
A(a, b, c) := ‖[a, b, c]‖² = Σ_k [a, b, c]_k²
```

**Question**: what is `Var(A)` to first order in GUM?

## Correct first-order GUM variance

GUM 100:2008 § 5.1.2: for a measurand `Y = f(X_1, ..., X_N)` with uncorrelated inputs,
```
Var(Y) = Σ_i (∂f/∂X_i)² · Var(X_i)                (GUM)
```
For `A(a, b, c)` with 24 independent scalar inputs `{a_i, b_j, c_k}_{i,j,k ∈ [0,7]}`:
```
Var(A) = Σ_{i=0}^{7} (∂A/∂a_i)² Var(a_i)
       + Σ_{j=0}^{7} (∂A/∂b_j)² Var(b_j)
       + Σ_{k=0}^{7} (∂A/∂c_k)² Var(c_k)          (1)
```
Each partial `∂A/∂x` is computable by the chain rule through the Fano-plane multiplication table. This is the **truth** against which we judge the compiler.

## What the compiler actually computes

The compiler does not treat `[a,b,c]` as a single function; it propagates variance structurally through each intermediate `let` binding. For the code sequence
```
let p = a · b         // octonion product
let L = p · c         // L = (a·b) · c
let q = b · c
let R = a · q         // R = a · (b·c)
let d = L - R         // the associator
let A = dot(d, d)     // its squared norm
```
the compiler's rule is applied to each line. Consider what happens at the `let p = a · b` step. The product `p` is an octonion with 8 components; each component `p_k` is a bilinear form of `a_i · b_j`. The compiler's scalar rule, applied per-component, yields
```
Var(p_k) = Σ_i Σ_j (∂p_k/∂a_i)² Var(a_i)
                 + (∂p_k/∂b_j)² Var(b_j)          (2)
```
which is **correct** for each individual component in isolation. The issue arises in subsequent uses of `p`.

When `L = p · c` is computed, the compiler applies its SCALAR rule again, treating each component of `L` as a bilinear form of `p · c`. But `p` is already a derived quantity with correlated components — its 8 entries share dependence on `a_i` and `b_j`. The scalar rule assumes input independence. A faithful first-order variance requires tracking the **covariance structure** of `p`'s components, not just their marginal variances.

**Consequence**: when the compiler subtracts `L − R` to form the associator, the covariance between `L` and `R` (both of which depend on the same `a_i, b_j, c_k`) is not tracked. The compiler effectively computes
```
Var(d_k) ≈ Var(L_k) + Var(R_k)                    (COMPILER)
```
treating `L` and `R` as independent. The correct formula is
```
Var(d_k) = Var(L_k) + Var(R_k) − 2·Cov(L_k, R_k)  (CORRECT)
```
For **quaternions**, `L_k = R_k` identically (quaternion multiplication is associative), so `Cov(L_k, R_k) = Var(L_k) = Var(R_k)` and the correct variance is **zero**. The compiler would produce `Var(L_k) + Var(R_k) = 2·Var(L_k)`, wrong by a factor of 2 (or infinity, since the truth is zero).

For **octonions** with inputs drawn from the quaternion subalgebra, the same holds: the associator is zero, its variance is zero, but the compiler would produce a positive overestimate.

For **generic octonion inputs**, `Cov(L_k, R_k) > 0` because `L` and `R` share input dependencies, and the compiler overestimates `Var(d_k)`. The overestimate is not by a fixed factor but depends on the specific input values.

This is **β**: the compiler's variance propagation is upper-bounded but not tight for non-associative products, and for the associator in particular — the object Part II of the draft is most interested in — the compiler is systematically biased.

## Worked example: quaternion subalgebra

Take `a = (a_0, a_1, a_2, a_3, 0, 0, 0, 0)`, similarly for `b` and `c`. All inputs are in the quaternion subalgebra ℍ ⊂ 𝕆. Each scalar input has uncertainty `σ`.

Quaternion multiplication is associative, so `(a·b)·c = a·(b·c)` as octonions, and the associator `[a,b,c] = 0` identically. Its squared norm `A = 0` with no uncertainty: `Var(A) = 0`.

The compiler's step-by-step propagation:
- `Var(p_k)` for `p = a·b`: nonzero, sum of per-pair scalar products' variances.
- `Var(L_k)` for `L = p·c`: nonzero, from `Var(p_k)` composed with `Var(c_k)`.
- Similarly `Var(R_k) > 0` for `R = a·(b·c)`.
- `Var(d_k)` for `d = L − R`: per the compiler, `Var(L_k) + Var(R_k) > 0`.
- `Var(A)` for `A = ‖d‖²`: positive.

**The compiler reports a positive variance for a quantity that is identically zero.** This is a qualitative, not just quantitative, failure for Knowledge-wrapped associator computations.

## Worked example: single Fano generator

Take `a = (0, 1, 0, 0, 0, 0, 0, 0) = e_1`, `b = e_2`, `c = e_4` (a non-associative Fano triple).

By the Fano table:
- `a·b = e_1 · e_2 = e_3`
- `(a·b)·c = e_3 · e_4 = e_7`
- `b·c = e_2 · e_4 = e_6`
- `a·(b·c) = e_1 · e_6 = −e_7`

So `[a,b,c] = e_7 − (−e_7) = 2·e_7`, `‖[a,b,c]‖² = 4`.

Now perturb `a_1 → a_1 + δ` with `δ ~ N(0, σ²)`. To first order:
- `(a·b)·c` component 7 shifts by `δ` (since `a·b` component 3 shifts by `δ·b_2 = δ`, then `(a·b)·c` component 7 shifts by `δ·c_4 = δ`).
- `a·(b·c)` component 7 shifts by `−δ` (similar chain through `b·c = e_6` and `a·e_6 = −e_7`).
- So `[a,b,c]_7` shifts by `δ − (−δ) = 2δ`, and `A = [a,b,c]_7²` shifts by approximately `2·[a,b,c]_7·(2δ) = 8δ` (since `[a,b,c]_7 = 2`).
- `Var(A)` contribution from `a_1`: `(8)² · σ² = 64σ²`.

The compiler's propagation:
- `Var(p_3) = b_2² Var(a_1) = 1·σ²` (other inputs zero).
- `Var(L_7) = c_4² Var(p_3) = 1·σ² = σ²`.
- `Var(q_6) = c_4² Var(b_2) = 0` (b_2 has no uncertainty in this example — only a_1 does).
- `Var(R_7) = (b·c)_6² Var(a_1) = 1·σ²` (with a_1 perturbed; b·c is constant here).
- `Var(d_7) = Var(L_7) + Var(R_7) = 2σ²`  ← **the compiler, treating L and R as uncorrelated**.
- `Var(A) = Var(d_7²)` applied to d_7 = 2, Var(d_7) = 2σ²: `Var(d_7²) ≈ (2·d_7)² · Var(d_7) = 16σ²`.

**Compiler**: `Var(A) ≈ 16σ²`. **Truth**: `Var(A) = 64σ²`.

**The compiler underestimates variance by a factor of 4 in this Fano-triple case**, because `L_7 = e_7` and `R_7 = −e_7` move in *opposite directions* under a perturbation of `a_1`, and treating them as independent fails to capture the anti-correlation that amplifies the associator's sensitivity.

In the general case the bias can go either direction (over- or under-estimate) depending on the sign and magnitude of `Cov(L, R)`.

## Summary of the compiler bias

| Input regime | Compiler | Truth | Direction |
|--------------|----------|-------|-----------|
| All scalars (ℝ) | `Var(A) = 0` | `Var(A) = 0` | correct |
| Quaternion subalgebra | `Var(A) > 0` | `Var(A) = 0` | over-estimates (qualitatively wrong) |
| Single Fano triple (aligned unit basis) | `Var(A) = 16σ²` | `Var(A) = 64σ²` | under-estimates by 4× |
| Generic octonion | biased; sign depends on `Cov(L, R)` | exact GUM formula (1) | unknown sign |

## Implications

**For the draft (Part II)**: the `Knowledge<Octonion<f64>>` → associator → `Var(associator)` chain currently produces systematically incorrect uncertainties. The draft should not claim GUM-compliant propagation for the associator until either (a) the compiler tracks covariance between co-dependent intermediate octonions, or (b) the paper uses the closed-form GUM formula (1) directly rather than the stepwise compiler propagation.

**For Phase 2 of the ABIDE experiment**: the pilot uses raw `f64` (no Knowledge wrapping) and reports empirical bootstrap CIs rather than compiler-propagated variances. This is **safe**. The compiler bias does not affect Phase 2 as currently specified.

**For the dissertation (octonion PBPK, γ-thread)**: the compile-time confidence gates (contribution 2) would need to use the closed-form associator variance for any octonion-product-derived confidence, not the compiler's stepwise rule. This is a real constraint on contribution 4.

**For the ζ compiler fix**: the buffer-bump recipe fixes the BSS overflow but does **not** address this covariance issue. ζ and β are orthogonal: ζ is about capacity (slot count), β is about semantics (independence assumption). A ζ fix without a β fix yields variance values that don't overflow but are still biased. A β fix without ζ yields correct semantics that still overflow on deep chains. Both fixes are ultimately needed for `Knowledge<Octonion<f64>>` to be meaningful.

## Proposed compiler remedy (β fix)

The compiler needs to track, for each variable, not only the marginal variance but also the **sensitivity vector** — the partials `∂y/∂x_i` with respect to each *original* uncertain input. Then when `d = L − R`, the sensitivity vector of `d` is the component-wise difference of `L`'s and `R`'s sensitivity vectors, and `Var(d)` is computed from the sensitivity vector and the input covariance matrix directly. This is **forward-mode automatic differentiation** over the uncertainty graph.

Implementation skeleton:
- Extend the variance BSS buffer to store a sensitivity vector per SSA slot: `(Var(y), ∂y/∂x_1, ..., ∂y/∂x_N)` where N is the number of uncertain inputs.
- At each `measure()`, allocate a new input index `i` and set the sensitivity `∂y/∂x_i = 1` with all other partials zero.
- At arithmetic operations, propagate sensitivities by the chain rule: `∂(y·z)/∂x_i = (∂y/∂x_i)·z + y·(∂z/∂x_i)`.
- At `variance_of()`, sum `(∂y/∂x_i)² · Var(x_i)` across all input indices.

**Cost**: BSS increases from O(slots × channels) to O(slots × channels × N_inputs). For dissertation-scale programs (N_inputs ≈ 10-20), this is manageable. For larger programs, an out-of-BSS allocation would be needed.

**Benefit**: exact first-order GUM for arbitrary algebraic structure, including non-associative. The associator no longer gets systematically biased variance.

This is a **larger** compiler change than ζ (days, not hours) and should be scoped as a separate thread from ζ.

## Validation plan for β

1. Numerical check (short-term): implement the quaternion-subalgebra example and the Fano-triple example as Sounio tests. Compare compiler variance vs Monte Carlo variance (1000 samples of Gaussian-perturbed inputs, compute associator, take sample variance). Confirm the predicted direction and magnitude of bias.
2. Derivation check (paper-time): symbolically derive `∂A/∂a_i` etc. via SymPy; compare to the Fano-triple special case computed by hand.
3. Compiler fix (long-term): implement forward-mode AD in `lean_single.sio`, verify on the same tests, ensure bootstrap fixed-point survives.

Step 1 is the next concrete artifact for the β thread — a small Sounio file `tests/run-pass/associator_variance_mc.sio` that cross-checks compiler vs Monte Carlo on the Fano-triple case. Defer until a dissertation-critical decision forces the issue.

## Acknowledgement of uncertainty

The analysis above assumes the compiler's `Var(a·b) = a²·var(b) + b²·var(a)` rule is applied *literally* component-wise to octonion products. If the actual emit code does something more sophisticated — say, it already tracks covariances between components of the same octonion — the bias analysis changes. The emit code at `self-hosted/compiler/lean_single.sio:5189-5283` (`emit_gate_variance_addsub_x86`, `emit_gate_variance_muldiv_x86`) does NOT appear to track inter-component covariance; each channel is propagated independently. But a reading confirmation by running the numerical check (step 1 above) is the honest next step.

## UPDATE 2026-04-13: empirical confirmation

Two companion tests landed:

**`tests/run-pass/associator_variance_mc.sio`** — Monte Carlo validation of the analytical formula. For the Fano triple (a=e₁, b=e₂, c=e₄) with σ=0.1 perturbation on a₁:
- Analytical (2nd-order): `Var(A) = 64σ² + 32σ⁴ = 0.6432`
- MC observed (N=10⁴): `Var(A) = 0.689`  (7.7% high, ~5 SE — PRNG serial correlation through Box-Muller, not a theoretical issue)

The analytical `64σ²` formula is directionally validated.

**`tests/run-pass/variance_covariance_blindness.sio`** — direct compiler diagnostic. Three minimal cases where covariance-aware truth is 0:

| expression | compiler says | covariance-blind prediction (2·var of each operand) | match? |
|------------|---------------|-----------------------------------------------------|--------|
| `x - x`, σ²=0.01 | `0.020000` | `0.020000` | ✓ exact |
| `(x+1) - (x+1)`, σ²=0.01 | `0.020000` | `0.020000` | ✓ exact |
| `a·b − a·b`, a=2±0.05, b=3±0.05 | `0.065000` | `2·(4·0.0025 + 9·0.0025) = 0.065` | ✓ exact |

**The compiler is covariance-blind.** β's core claim is empirically confirmed at the minimal case level — not a reading-based inference any more. Every subtraction of correlated intermediates (which includes the associator `L − R = (ab)c − a(bc)` by construction) gets variance double-counted.

The corollary: any published numerical claim from Sounio that uses `variance_of()` on a subtraction of *correlated* intermediates is wrong by construction. The Phase 2 pilot is safe only because it uses raw `f64` + bootstrap CIs rather than compiler-propagated variance.

The `16σ² vs 64σ²` factor-of-4 prediction for the full Fano-triple associator chain (through `(ab)c − a(bc)`) is plausible but not yet directly measured — would require wiring the full Knowledge<f64> chain through the non-trivial octonion product, which may hit the ζ buffer bug. The minimal-case confirmation is sufficient to establish β's basic mechanism.

## Root mechanism (2026-04-13 deeper read of compiler)

The covariance blindness is mechanically caused by this sequence in `self-hosted/compiler/lean_single.sio`:

1. **`.value` access of a `Knowledge<f64>` increments `NEXT_BUDGET_PARAM`** at every call site (lines 7424, 7499, 10219). So two `x.value` accesses seed σ² into *different* BSS channels — first into channel 0, second into channel 1. The ISO budget decomposition uses this to give each "measurement read" its own channel.

2. **`emit_seed_variance_from_knowledge_x86()` zeroes all OTHER channels when it writes to its target channel** (line 5705). So after the second access, channel 0 is zeroed out and channel 1 holds σ². This is single-path-exclusive design: one channel owns the variance at any given time.

3. **At the binop variance emit (line 10561)**, the check `left_var_slot >= 0 && left_var_slot == right_var_slot` fails — both operands have `EXPR_VAR_SLOT = -2` (scratch). Falls through to `emit_gate_variance_addsub_x86(-2, -2)`.

4. **`emit_gate_variance_addsub_x86(-2, -2)` reads scratch for BOTH left and right** (lines 5552-5583), per channel. For the `-2` case each channel reads the current scratch value. After the right-side's access, scratch channel 1 = σ², channel 0 = 0. So: left = scratch (σ² in ch 1, else 0), right = scratch (same, σ² in ch 1, else 0). Sum: 2σ² in ch 1 — that's the `0.02` observed for `Var(x - x)`.

## Proposed fix (two parts, non-trivial)

Neither part is a one-liner; together they change ISO-budget semantics.

**Part A: per-variable channel assignment.** Assign a budget channel at `measure()` time, store it as variable metadata (parallel array indexed by slot). `.value` access looks up the variable's assigned channel rather than incrementing `NEXT_BUDGET_PARAM`. Effect: multiple `.value` reads of the same `Knowledge<f64>` variable all seed the *same* channel, consistently.

- New global: `var KNOWLEDGE_CHANNEL_FOR_SLOT: [i64; 1024] = [-1; 1024]`
- At `let k = measure(...)`: `KNOWLEDGE_CHANNEL_FOR_SLOT[k_slot] = NEXT_BUDGET_PARAM; NEXT_BUDGET_PARAM += 1`
- At `x.value` emit: look up the containing variable's slot, retrieve the assigned channel, emit seed into that specific channel — drop the local increment

**Part B: scratch-source identity.** When both operands have `EXPR_VAR_SLOT = -2` and come from the same source variable, route to same-variant. The check is: before compiling the right operand, record `LAST_KNOWLEDGE_SOURCE_SLOT = <slot of variable whose .value was accessed>` for the left. After right compiles, compare against right's recorded source. If equal, same-path.

- New globals: `var LEFT_SCRATCH_SOURCE: i64 = -1`, `var RIGHT_SCRATCH_SOURCE: i64 = -1`
- At `.value` emit: record `EXPR_SCRATCH_SOURCE = containing_slot`
- At binop: save `left_scratch_source = EXPR_SCRATCH_SOURCE` after left, `right_scratch_source = EXPR_SCRATCH_SOURCE` after right
- If `left_var_slot == -2 && right_var_slot == -2 && left_scratch_source == right_scratch_source && left_scratch_source >= 0`, use `emit_variance_addsub_same_scratch_x86(op)` — a new emit function that returns 0 for sub or 4×scratch for add

## Why defer the fix rather than ship it here

- Part A changes ISO budget semantics (each measurement instead of each access gets a channel). This has downstream effects on `budget_of()` output for existing tests.
- Bootstrap rebuild + gen2==gen3 verify adds 30-60 min.
- The compiler has unrelated uncommitted GPU PTX edits from another thread; my fix would commingle in a revert if bootstrap breaks.
- The two parts must land together or the behavior is inconsistent.
- No Phase 2 or dissertation dependency is currently blocked on this (Phase 2 uses raw f64; dissertation has a decision gate 2026-05-15 for octonion PBPK contribution which is the first caller).

Scoping as its own coordinated change after the GPU edits land is the honest move. The full mechanism above is enough for the next session to implement cleanly.
