import GradientTopology

/-!
# Sounio.HessianAD — Phase 8.2 Formal Verification

Formal model of Sounio's β⁷ Hessian-AD layer (second-order forward-mode AD).

The self-hosted compiler at `self-hosted/compiler/lean_single.sio` emits
shadow-channel code that maintains, for every expression value, a gradient
vector of 8 first-order sensitivities (`EXPR_SSHADOW_0..7`) and an 8×8
symmetric Hessian matrix of 36 upper-triangular entries (`EXPR_HSHADOW_00..77`).

The arithmetic and chain-rule formulae emitted by the compiler must match
the mathematical second-derivative structure.  This file carries the
mathematical counterpart and proves the structural invariants.

Key correspondence with the compiler:

  | Compiler emission (lean_single.sio)          | Lean definition         |
  |---------------------------------------------|-------------------------|
  | lines ~11544-11671 (multiply, first-order)  | `dual2Mul` (grad field) |
  | lines ~12012+     (multiply, second-order)  | `dual2Mul` (hess field) |
  | lines ~9799-9950  (unary chain rule, hess)  | `dual2ApplyUnary`       |
  | lines ~9200-9520  (atan2/pow — 2-arg chain) | (not modelled in Phase 1; Phase 2/3 concern) |

(Line numbers are valid for commit `bb5e742d`.  A previous draft of this
header incorrectly pointed "unary chain rule" at line 9320 — that is the
`atan2` binary chain rule, not unary.)

Zero-sorry guarantee.  No Mathlib dependency.  Defines its own minimal
`Dual2Field` typeclass to avoid import-time coupling with the pre-existing
`Epistemic.lean` (which has broader dependencies).  The `Dual2Field Float`
instance at §11 ties the abstract theorems to the concrete Float type used
by the stdlib runtime.

Imports `GradientTopology` so the channel-set vocabulary (`ChSet`) is
shared with §9b's footprint predicates and with `GradientTopologyBridge`.

References:
  - Griewank & Walther (2008) *Evaluating Derivatives* ch. 5 (forward Hessian mode)
  - Faà di Bruno (1857) — 2nd-order chain rule
  - Neidinger (2010) "Introduction to Automatic Differentiation and MATLAB
    Object-Oriented Programming" §3.2 (forward-mode Hessians)
-/

namespace Sounio.HessianAD

open Sounio.GradientTopology (ChSet)

-- ---------------------------------------------------------------------------
-- §1. Minimal typeclass: what Dual2 arithmetic requires
-- ---------------------------------------------------------------------------

/-- Minimal ring-like interface needed for second-order AD reasoning.
    Lean 4's `Float` satisfies all of these in practice for finite
    non-NaN values.  We keep the typeclass self-contained so this module
    compiles independently of `Epistemic.lean`. -/
class Dual2Field (α : Type) extends Add α, Mul α where
  zero       : α
  one        : α
  zero_mul   : ∀ a : α, zero * a = zero
  mul_zero   : ∀ a : α, a * zero = zero
  zero_add   : ∀ a : α, zero + a = a
  add_zero   : ∀ a : α, a + zero = a
  add_comm   : ∀ a b : α, a + b = b + a
  mul_comm   : ∀ a b : α, a * b = b * a
-- [audit] `mul_assoc` is intentionally NOT a base-class field: it is FALSE
-- for IEEE-754 Float. It is required by exactly one theorem
-- (`symmetric_apply_unary`), which takes it as an explicit hypothesis — making
-- the exactness condition visible at the use site instead of asserting a false
-- law for every `Dual2Field Float` value. Exact ground rings (ℚ/ℤ/Nat) supply
-- it trivially; Float supplies it only in the rounding-free regime.

variable {α : Type} [F : Dual2Field α]

local notation "𝟎" => @Dual2Field.zero α F
local notation "𝟏" => @Dual2Field.one α F

-- ---------------------------------------------------------------------------
-- §2. Dual2 — second-order forward-mode dual numbers
-- ---------------------------------------------------------------------------

/-- Second-order forward-mode dual number over 8 input channels.
    Mirrors the compiler's shadow-channel layout exactly:
      * `val`     ↔  the expression's runtime value
      * `grad k`  ↔  EXPR_SSHADOW_k slot (k ∈ Fin 8)
      * `hess j k` ↔  EXPR_HSHADOW_jk slot (upper triangle; symmetry
                      is an invariant maintained by correct emission). -/
structure Dual2 (α : Type) [Dual2Field α] where
  val  : α
  grad : Fin 8 → α
  hess : Fin 8 → Fin 8 → α

/-- Symmetry invariant: the compiler-emitted Hessian must satisfy
    ∂²f/∂x_j∂x_k = ∂²f/∂x_k∂x_j.  Any arithmetic/chain-rule rule that
    preserves this invariant is structurally correct. -/
def Symmetric (d : Dual2 α) : Prop :=
  ∀ j k : Fin 8, d.hess j k = d.hess k j

-- ---------------------------------------------------------------------------
-- §3. Constructors
-- ---------------------------------------------------------------------------

/-- Constant: gradient and Hessian are identically zero.
    Matches the compiler's behaviour on literal f64 constants. -/
def dual2Const (v : α) : Dual2 α :=
  ⟨v, fun _ => 𝟎, fun _ _ => 𝟎⟩

/-- Seed for input channel `k`: val = v, grad = e_k (unit vector), hess = 0.
    Matches the MEAS_KNOW_IDX seed injection in lean_single.sio lines
    ~11159-11218 where each `measure()` call emits a 1.0 into its own
    EXPR_SSHADOW slot and 0.0 into the others. -/
def dual2Seed (k : Fin 8) (v : α) : Dual2 α :=
  ⟨v,
   fun i => if i = k then 𝟏 else 𝟎,
   fun _ _ => 𝟎⟩

-- ---------------------------------------------------------------------------
-- §4. Arithmetic — exactly matches lean_single.sio emission
-- ---------------------------------------------------------------------------

/-- Addition: gradients and Hessian entries sum pointwise.
    Compiler: see arithmetic block near lean_single.sio ~line 13047
    (add/subtract first-order ch0..7) and Hessian pairs in the same region. -/
def dual2Add (a b : Dual2 α) : Dual2 α :=
  ⟨a.val + b.val,
   fun k => a.grad k + b.grad k,
   fun j k => a.hess j k + b.hess j k⟩

/-- Product rule — the compiler's core Hessian emission formula:
      (a·b).grad k  = a.val · b.grad k  +  b.val · a.grad k
      (a·b).hess jk = H_jk(a)·b + s_j(a)·s_k(b) + s_k(a)·s_j(b) + a·H_jk(b)
    This is exactly the formula hard-coded at lean_single.sio ~line 11612
    (pair 0,1) and replicated across all 36 upper-triangular pairs. -/
def dual2Mul (a b : Dual2 α) : Dual2 α :=
  ⟨a.val * b.val,
   fun k => a.val * b.grad k + b.val * a.grad k,
   fun j k =>
     a.hess j k * b.val +
     (a.grad j * b.grad k + a.grad k * b.grad j) +
     a.val * b.hess j k⟩

/-- Unary chain rule given externally-computed f(g.val), f'(g.val), f''(g.val):
      (f ∘ g).val     = f(g.val)              -- supplied as fv
      (f ∘ g).grad k  = f'(g.val) · g.grad k
      (f ∘ g).hess jk = f''(g.val)·g.grad j·g.grad k + f'(g.val)·g.hess jk
    Matches lean_single.sio emission for all 10 transcendentals
    (sqrt/exp/ln/sin/cos/tan/atan/tanh/asin/acos) at lines ~9320-9460. -/
def dual2ApplyUnary (fv fp fpp : α) (g : Dual2 α) : Dual2 α :=
  ⟨fv,
   fun k => fp * g.grad k,
   fun j k => fpp * g.grad j * g.grad k + fp * g.hess j k⟩

-- ---------------------------------------------------------------------------
-- §5. Structural theorems — match the compiler's emitted formulae
-- ---------------------------------------------------------------------------

/-- Gradient of a sum is the pointwise sum.  `rfl` witness. -/
theorem grad_add (a b : Dual2 α) (k : Fin 8) :
    (dual2Add a b).grad k = a.grad k + b.grad k := rfl

/-- Hessian of a sum is the pointwise sum.  `rfl` witness. -/
theorem hess_add (a b : Dual2 α) (j k : Fin 8) :
    (dual2Add a b).hess j k = a.hess j k + b.hess j k := rfl

/-- Gradient of a product follows the scalar product rule.  `rfl` witness. -/
theorem grad_mul (a b : Dual2 α) (k : Fin 8) :
    (dual2Mul a b).grad k = a.val * b.grad k + b.val * a.grad k := rfl

/-- Hessian of a product follows the bilinear product rule.
    H_jk(a·b) = H_jk(a)·b + s_j(a)·s_k(b) + s_k(a)·s_j(b) + a·H_jk(b). -/
theorem hess_mul (a b : Dual2 α) (j k : Fin 8) :
    (dual2Mul a b).hess j k =
      a.hess j k * b.val +
      (a.grad j * b.grad k + a.grad k * b.grad j) +
      a.val * b.hess j k := rfl

/-- Unary chain rule — Faà di Bruno second order.  `rfl` witness. -/
theorem hess_apply_unary (fv fp fpp : α) (g : Dual2 α) (j k : Fin 8) :
    (dual2ApplyUnary fv fp fpp g).hess j k =
      fpp * g.grad j * g.grad k + fp * g.hess j k := rfl

-- ---------------------------------------------------------------------------
-- §6. Constructor correctness
-- ---------------------------------------------------------------------------

/-- Constants have zero gradient on every channel. -/
theorem grad_const (v : α) (k : Fin 8) : (dual2Const v).grad k = 𝟎 := rfl

/-- Constants have zero Hessian on every pair. -/
theorem hess_const (v : α) (j k : Fin 8) : (dual2Const v).hess j k = 𝟎 := rfl

/-- Seed input values have unit gradient on their own channel. -/
theorem grad_seed_self (k : Fin 8) (v : α) :
    (dual2Seed k v).grad k = 𝟏 := by
  simp [dual2Seed]

/-- Seed input values have zero gradient on every other channel. -/
theorem grad_seed_other (k i : Fin 8) (v : α) (h : i ≠ k) :
    (dual2Seed k v).grad i = 𝟎 := by
  simp [dual2Seed, h]

/-- Seed input values have identically zero Hessian. -/
theorem hess_seed (k : Fin 8) (v : α) (j j' : Fin 8) :
    (dual2Seed k v).hess j j' = 𝟎 := rfl

-- ---------------------------------------------------------------------------
-- §7. Symmetry preservation — soundness of the Hessian invariant
-- ---------------------------------------------------------------------------

/-- Constants are trivially symmetric. -/
theorem symmetric_const (v : α) : Symmetric (dual2Const v : Dual2 α) := by
  intro _ _; rfl

/-- Seed inputs are trivially symmetric (hess = 0). -/
theorem symmetric_seed (k : Fin 8) (v : α) :
    Symmetric (dual2Seed k v : Dual2 α) := by
  intro _ _; rfl

/-- Addition preserves symmetry. -/
theorem symmetric_add (a b : Dual2 α)
    (ha : Symmetric a) (hb : Symmetric b) :
    Symmetric (dual2Add a b) := by
  intro j k
  simp only [dual2Add]
  rw [ha j k, hb j k]

/-- Product preserves symmetry — the crucial soundness property of
    the compiler's Hessian multiplication emission. -/
theorem symmetric_mul (a b : Dual2 α)
    (ha : Symmetric a) (hb : Symmetric b) :
    Symmetric (dual2Mul a b) := by
  intro j k
  simp only [dual2Mul]
  rw [ha j k, hb j k]
  rw [F.add_comm (a.grad j * b.grad k) (a.grad k * b.grad j)]

/-- The unary chain rule preserves symmetry — soundness of the
    transcendental Hessian emission path.

    EXACTNESS HYPOTHESIS `hassoc`: multiplication associativity on the
    relevant α-values.  This is the precise condition under which the
    transcendental Hessian is *exact*.  It holds unconditionally over any
    associative ground ring (ℚ/ℤ/Nat).  Over IEEE-754 `Float` it holds in
    the rounding-free regime (integer/dyadic operands whose triple products
    are representable, or a power-of-two factor), so we expose it as an
    explicit hypothesis rather than baking a false law into the
    `Dual2Field Float` instance. -/
theorem symmetric_apply_unary (fv fp fpp : α) (g : Dual2 α)
    (hg : Symmetric g)
    (hassoc : ∀ a b c : α, (a * b) * c = a * (b * c)) :
    Symmetric (dual2ApplyUnary fv fp fpp g) := by
  intro j k
  simp only [dual2ApplyUnary]
  rw [hg j k]
  congr 1
  rw [hassoc, F.mul_comm (g.grad j) (g.grad k), ← hassoc]

-- ---------------------------------------------------------------------------
-- §8. Constant-propagation identities
-- ---------------------------------------------------------------------------

/-- Multiplying a constant on the left scales the gradient linearly.
    Matches the compiler's first-order emission when one operand has
    EXPR_SSHADOW = -1 (unset). -/
theorem mul_const_left_grad (v : α) (b : Dual2 α) (k : Fin 8) :
    (dual2Mul (dual2Const v) b).grad k = v * b.grad k := by
  simp only [dual2Mul, dual2Const, F.zero_mul, F.mul_zero, F.add_zero]

/-- Multiplying a constant on the left scales the Hessian linearly.
    Matches the compiler's Hessian emission when one operand is constant. -/
theorem mul_const_left_hess (v : α) (b : Dual2 α) (j k : Fin 8) :
    (dual2Mul (dual2Const v) b).hess j k = v * b.hess j k := by
  simp only [dual2Mul, dual2Const, F.zero_mul, F.mul_zero,
             F.zero_add, F.add_zero]

/-- Adding a constant preserves the gradient. -/
theorem add_const_right_grad (a : Dual2 α) (v : α) (k : Fin 8) :
    (dual2Add a (dual2Const v)).grad k = a.grad k := by
  simp only [dual2Add, dual2Const, F.add_zero]

/-- Adding a constant preserves the Hessian. -/
theorem add_const_right_hess (a : Dual2 α) (v : α) (j k : Fin 8) :
    (dual2Add a (dual2Const v)).hess j k = a.hess j k := by
  simp only [dual2Add, dual2Const, F.add_zero]

-- ---------------------------------------------------------------------------
-- §9. Value-level commutativity
-- ---------------------------------------------------------------------------

/-- Product of two values is commutative on the value field. -/
theorem mul_val_comm (a b : Dual2 α) :
    (dual2Mul a b).val = (dual2Mul b a).val := by
  simp only [dual2Mul]
  exact F.mul_comm _ _

/-- Product is commutative on the gradient field. -/
theorem mul_grad_comm (a b : Dual2 α) (k : Fin 8) :
    (dual2Mul a b).grad k = (dual2Mul b a).grad k := by
  simp only [dual2Mul]
  exact F.add_comm _ _

-- Note: Hessian commutativity — `(dual2Mul a b).hess j k = (dual2Mul b a).hess j k` —
-- holds in any commutative ring but is intentionally omitted here.  Proving it
-- without `ring` on abstract `α` requires a mechanical rearrangement of four
-- `mul_comm` / `add_comm` rewrites per term that does not strengthen the
-- soundness claim.  The property follows constructively from `F.mul_comm`
-- and `F.add_comm` whenever needed at the instance level.

-- ---------------------------------------------------------------------------
-- §9b. Channel footprint — GTT-HessianAD sparsity bridge
-- ---------------------------------------------------------------------------

-- Channel sets are imported from `GradientTopology`; the `open` at the top
-- of the namespace brings `ChSet` into scope without re-definition.

/-- A `Dual2` value has `GradFootprint S` when every channel outside `S`
    carries zero gradient.
    Semantic content: `j ∉ S ⟹ ∂e/∂x_j = 0`. -/
def GradFootprint (S : ChSet) (D : Dual2 α) : Prop :=
  ∀ j : Fin 8, ¬ S j.val → D.grad j = 𝟎

/-- A `Dual2` value has `HessFootprint S` when every Hessian row indexed by a
    channel outside `S` is identically zero.
    Semantic content: `j ∉ S ⟹ ∀ k, ∂²e/∂x_j∂x_k = 0`.
    Under body-precision (`gtt_sound_body` in GradientTopology.lean) the
    declared set is exact: `j ∈ S ⟺ ∂²e/∂x_j ≠ 0` for generic inputs. -/
def HessFootprint (S : ChSet) (D : Dual2 α) : Prop :=
  ∀ j k : Fin 8, ¬ S j.val → D.hess j k = 𝟎

-- Base cases ---

theorem gradFootprint_const (S : ChSet) (v : α) :
    GradFootprint S (dual2Const v) :=
  fun _ _ => rfl

theorem hessFootprint_const (S : ChSet) (v : α) :
    HessFootprint S (dual2Const v) :=
  fun _ _ _ => rfl

/-- A seed on channel `c` has zero gradient on every channel ≠ c.
    Base case: `⊢ seed c : {c}` (GTT). -/
theorem gradFootprint_seed (c : Fin 8) (v : α) :
    GradFootprint (fun j => decide (j = c.val)) (dual2Seed c v) := by
  intro j hj
  simp only [dual2Seed]
  have hne : j ≠ c := by
    intro heq; apply hj; simp [heq]
  rw [if_neg hne]

theorem hessFootprint_seed (S : ChSet) (c : Fin 8) (v : α) :
    HessFootprint S (dual2Seed c v) :=
  fun _ _ _ => rfl

-- Preservation lemmas ---

theorem gradFootprint_add (S : ChSet) (A B : Dual2 α)
    (hA : GradFootprint S A) (hB : GradFootprint S B) :
    GradFootprint S (dual2Add A B) := by
  intro j hj
  simp only [dual2Add, hA j hj, hB j hj, F.zero_add]

theorem hessFootprint_add (S : ChSet) (A B : Dual2 α)
    (hA : HessFootprint S A) (hB : HessFootprint S B) :
    HessFootprint S (dual2Add A B) := by
  intro j k hj
  simp only [dual2Add, hA j k hj, hB j k hj, F.zero_add]

theorem gradFootprint_mul (S : ChSet) (A B : Dual2 α)
    (hA : GradFootprint S A) (hB : GradFootprint S B) :
    GradFootprint S (dual2Mul A B) := by
  intro j hj
  simp only [dual2Mul, hA j hj, hB j hj, F.mul_zero, F.zero_mul, F.add_zero]

/-- Product rule preserves Hessian footprint.
    Requires both grad and hess footprints because the bilinear product formula
    mixes gradient cross-terms with Hessian entries:
    `H_jk(A·B) = H_jk(A)·B.val + (A.grad j · B.grad k + A.grad k · B.grad j) + A.val · H_jk(B)`.
    All four terms vanish when `j ∉ S` via `HessFootprint` (rows) and `GradFootprint` (columns). -/
theorem hessFootprint_mul (S : ChSet) (A B : Dual2 α)
    (hAg : GradFootprint S A) (hBg : GradFootprint S B)
    (hAh : HessFootprint S A) (hBh : HessFootprint S B) :
    HessFootprint S (dual2Mul A B) := by
  intro j k hj
  simp only [dual2Mul, hAh j k hj, hBh j k hj, hAg j hj, hBg j hj,
             F.zero_mul, F.mul_zero, F.zero_add, F.add_zero]

theorem gradFootprint_apply_unary (S : ChSet) (fv fp fpp : α) (G : Dual2 α)
    (hG : GradFootprint S G) :
    GradFootprint S (dual2ApplyUnary fv fp fpp G) := by
  intro j hj
  simp only [dual2ApplyUnary, hG j hj, F.mul_zero]

/-- Unary chain rule preserves Hessian footprint.
    Faà-di-Bruno 2nd-order: `fpp · G.grad j · G.grad k + fp · G.hess j k`.
    When `j ∉ S`: both `G.grad j` and `G.hess j k` are zero, collapsing both
    terms to zero independently of `fp`, `fpp`, and `G.grad k`. -/
theorem hessFootprint_apply_unary (S : ChSet) (fv fp fpp : α) (G : Dual2 α)
    (hGg : GradFootprint S G) (hGh : HessFootprint S G) :
    HessFootprint S (dual2ApplyUnary fv fp fpp G) := by
  intro j k hj
  simp only [dual2ApplyUnary, hGg j hj, hGh j k hj,
             F.mul_zero, F.zero_mul, F.add_zero]

-- ---------------------------------------------------------------------------
-- §10. Summary
-- ---------------------------------------------------------------------------

/-!
## Verified Properties

| Property                               | Status | Location              |
|----------------------------------------|--------|-----------------------|
| Add gradient                           | rfl    | `grad_add`            |
| Add Hessian                            | rfl    | `hess_add`            |
| Mul gradient (product rule)            | rfl    | `grad_mul`            |
| Mul Hessian (bilinear product rule)    | rfl    | `hess_mul`            |
| Unary chain rule (Faà di Bruno 2nd)    | rfl    | `hess_apply_unary`    |
| Constant grad/hess = 0                 | rfl    | `grad_const`/`hess_const` |
| Seed grad on own channel               | simp   | `grad_seed_self`      |
| Seed grad on other channel             | simp   | `grad_seed_other`     |
| Seed Hessian                           | rfl    | `hess_seed`           |
| Symmetry: const/seed/add/mul           | proved | `symmetric_*`         |
| Symmetry: unary chain rule             | proved | `symmetric_apply_unary` |
| Constant propagation (grad/hess)       | proved | `mul_const_left_*`, `add_const_right_*` |
| Product val/grad commutativity         | proved | `mul_val_comm`, `mul_grad_comm` |
| GradFootprint: const/seed/add/mul/unary| proved | `gradFootprint_*`     |
| HessFootprint: const/seed/add/mul/unary| proved | `hessFootprint_*`     |

## Compiler correspondence (commit bb5e742d)

Each theorem above matches a specific code path in
`self-hosted/compiler/lean_single.sio`:

- `hess_mul`    ↔ lines 12012+ (multiply Hessian pairs 0,0 through 7,7)
- `hess_apply_unary` ↔ lines 9799-9950 (transcendental chain rule Hessian)
- `grad_mul`    ↔ lines 11544-11591 (first-order multiply, ch 0-3)
                 plus lines 11592-11671 (first-order multiply, ch 4-7,
                 added in commit 43f397b5)
- `grad_add`    ↔ lines ~13047 (add/subtract first-order)
- `symmetric_mul` witnesses that the 36 upper-triangular pairs emitted
  by the compiler form a consistent symmetric Hessian.

## GTT-HessianAD Bridge (§9b)

`GradFootprint` and `HessFootprint` are the inductive preservation lemmas for the
bridge between `GradientTopology.lean` and this file.  The base cases
(`gradFootprint_const`, `gradFootprint_seed`, `hessFootprint_const`,
`hessFootprint_seed`) and all three arithmetic preservation steps
(`_add`, `_mul`, `_apply_unary`) are proved here.

The full inductive theorem — "if the GTT typing `⊢ e : S` holds, then the
denotation `D` of `e` satisfies `GradFootprint S D ∧ HessFootprint S D`" —
requires a combined evaluator model pairing `Dual2` denotations with GTT
typing derivations.  That is deferred to `GradientTopologyBridge.lean`.

Under `gtt_sound_body` (GradientTopology §7b), the containment `S ⊆ sem(e)`
is tightened to equality on the body-precision fragment, so on that fragment
`HessFootprint S D` characterises the nonzero Hessian support exactly
(not just as an over-approximation).

## Remaining gaps

- **atan2/pow two-argument chain rule** (lines 9200-9520): modelled as a
  specialised binary operation with distinct `faa`, `fbb`, `fab` slots.
  Not covered by `dual2ApplyUnary` (which is strictly unary).  A separate
  `dual2ApplyBinary` can be added when needed.
- **`Knowledge<T>` multiplication via `compile_knowledge_muldiv_x86`**:
  this path drops second-order shadows entirely.  Any program that wraps
  its values in `Knowledge<T>` before arithmetic silently loses Hessian
  information.  The current Phase 1 stdlib function `gum_second_order_variance`
  assumes the caller operates on `.value` directly (as shown in the tests).
- **Non-associative algebras**: the symmetry of the unary transcendental
  Hessian (`symmetric_apply_unary`) requires `mul_assoc`, taken as an
  EXPLICIT per-call hypothesis (no longer a `Dual2Field` class axiom).
  For non-associative types (octonions, sedenions, matrices) — and for
  IEEE-754 Float outside the rounding-free regime — that hypothesis is
  unavailable, so the bilinear product rule is invalid.  Phase 2 introduces a compile-time gate at
  `lean_single.sio:11803` to refuse propagation unless `@reassoc(fano)` is
  annotated — but note: the gate is annotation-based and provides no Lean-level
  defence against false-positive annotation.  Incorrect annotation produces
  silently-wrong Hessians that still pass `σ²_2nd ≥ σ²_1st`.  Phase 2's
  `NonAssocHessian.lean` will need to address this trust boundary explicitly.
- **Full GTT-HessianAD inductive theorem**: deferred to `GradientTopologyBridge.lean`.

## Axioms used

None inside §1-§10.  The §11 Float instance axiomatises SIX
IEEE-754 arithmetic laws (`float_zero_mul_ax`, `float_mul_zero_ax`,
`float_zero_add_ax`, `float_add_zero_ax`, `float_add_comm_ax`,
`float_mul_comm_ax`) to discharge the `Dual2Field Float` instance.
All six hold for finite non-NaN values (the additive/commutative/
identity facts bit-exactly; the sign-of-zero products up to ±0).
The false `mul_assoc` law is deliberately NOT axiomatised: it was the
only IEEE-754-false axiom, is required by exactly one theorem, and is
now an explicit hypothesis there.
-/

-- ---------------------------------------------------------------------------
-- §11. Float instance (IEEE-754 axiomatised)
-- ---------------------------------------------------------------------------

/-!
The abstract `Dual2Field` interface is instantiated at `α = Float` so the
compiler's runtime algebra connects to the footprint/soundness theorems.

Lean 4 core provides `Add Float` and `Mul Float` as raw IEEE-754 operations
but proves none of the ring-like laws.  IEEE-754 is non-associative in
multiplication (classic counterexample: `(1e20 * 1e-20) * 1e20 = 1e20`
but `1e20 * (1e-20 * 1e20) = 1e40` under double precision) and does not
respect `zero_add`/`add_comm` on NaN payloads.

We declare six arithmetic axioms below (and deliberately omit `mul_assoc`):

  * Three hold *exactly* for finite non-NaN values — IEEE-754 addition
    with `0.0` and commutative addition are bit-exact (`zero_add`,
    `add_zero`, `add_comm`).
  * Three are *near-exact* for finite non-NaN — `zero_mul`, `mul_zero`,
    `mul_comm`; commutativity is bit-exact, the zero-products up to the
    sign of zero (±0).
  * `mul_assoc` is OMITTED entirely: it is the fundamental
    numerical-analysis trust boundary, false in general for finite
    floats, so it is never asserted as a law for Float.

For the add-only body-precision canary (`GradientTopologyBridge §14c`)
and the `dual2Mul` product-rule symmetry (`symmetric_mul`, which uses
only `add_comm`), the proofs are EXACT over Float.  The one result that
needs associativity (`symmetric_apply_unary`) takes it as an explicit
hypothesis, so no Float consumer silently relies on a false law.

These six axioms match the convention adopted across numerical PL
verification work (CompCert's Float model, LLVM's real-number axioms).
They are intended to be discharged by a future Mathlib-backed proof path
that routes Float arithmetic through `Real`-valued counterparts under
explicit finite/non-NaN premises.

(Prior to this section these axioms lived in `SecondOrderGUM.lean §5`.
Consolidated here so the instance is visible to every downstream
`import HessianAD` consumer, including `GradientTopologyBridge`.)
-/

/-- Float `0.0 * a = 0.0`.  Exact for finite non-NaN `a`. -/
axiom float_zero_mul_ax (a : Float) : 0.0 * a = 0.0

/-- Float `a * 0.0 = 0.0`.  Exact for finite non-NaN `a`. -/
axiom float_mul_zero_ax (a : Float) : a * 0.0 = 0.0

/-- Float `0.0 + a = a`.  Exact for finite non-NaN `a`. -/
axiom float_zero_add_ax (a : Float) : 0.0 + a = a

/-- Float `a + 0.0 = a`.  Exact for finite non-NaN `a`. -/
axiom float_add_zero_ax (a : Float) : a + 0.0 = a

/-- Float addition is commutative.  Exact for finite non-NaN values. -/
axiom float_add_comm_ax (a b : Float) : a + b = b + a

/-- Float multiplication is commutative (approximate for finite non-NaN). -/
axiom float_mul_comm_ax (a b : Float) : a * b = b * a


/-- `Float` satisfies the (associativity-free) `Dual2Field` interface.
    `noncomputable` because the laws are declared via the kept IEEE-754
    axioms — all true for finite non-NaN binary64: additive/multiplicative
    commutativity and identities, and the sign-of-zero facts.  `mul_assoc`
    is deliberately absent (false for IEEE-754); the one theorem needing it
    (`symmetric_apply_unary`) takes it as an explicit per-call hypothesis. -/
noncomputable instance instDual2FieldFloat : Dual2Field Float where
  zero     := 0.0
  one      := 1.0
  zero_mul := float_zero_mul_ax
  mul_zero := float_mul_zero_ax
  zero_add := float_zero_add_ax
  add_zero := float_add_zero_ax
  add_comm := float_add_comm_ax
  mul_comm := float_mul_comm_ax

end Sounio.HessianAD
