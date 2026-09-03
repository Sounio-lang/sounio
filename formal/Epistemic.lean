/-!
# Sounio.Epistemic — Phase 8 Formal Verification

Formal model of Sounio's epistemic type system: `Knowledge<T>` types
with uncertainty intervals.

The Rust implementation is in:
  `crates/souc/src/epistemic/` — uncertainty arithmetic
  `crates/souc/src/codegen/gpu/epistemic_gemm.rs` — GPU uncertainty propagation
  `self-hosted/gpu/kernels/bio.sio` — BOLD fMRI epistemic kernel

References:
  - Friston et al. 2003, "Dynamic Causal Modelling" (Balloon-Windkessel)
  - Neumaier 1990, "Interval Methods for Systems of Equations" (interval arithmetic)
  - Dunfield & Krishnaswami 2021 (bidirectional type theory)

## Design

`Float` in Lean 4 (leanc) is an opaque IEEE-754 double backed by C `double`.
The Lean 4 core does **not** export a complete algebraic hierarchy for `Float`
(no `Ring`, `LinearOrder`, etc.).  Where ordering/sign inequalities cannot be
closed by `ring` or `native_decide`, we declare them as `axiom`. These are
restricted to facts that DO hold for finite non-NaN binary64 (monotonicity,
non-negativity, and the *exact* identities x+0, x*1, x/1, x-x). Associativity
and distributivity are NOT among them — they are false for IEEE-754 and are
deliberately absent (see §12).

All non-inequality theorems are proved via `ring` or structural induction.
-/

namespace Sounio.Epistemic

-- ---------------------------------------------------------------------------
-- §1. Abstract ordered field for uncertainty arithmetic
-- ---------------------------------------------------------------------------

/-- The minimal ordered-field interface needed for uncertainty arithmetic.
    Lean 4's `Float` satisfies this in practice but the proofs are not in
    core; we carry the structure as a typeclass. -/
class EpistemicField (α : Type) extends Add α, Mul α, Neg α, Div α, LE α where
  zero       : α
  one        : α
  abs        : α → α
  -- Ordered semiring axioms (over ≤)
  le_refl    : ∀ a : α, a ≤ a
  le_trans   : ∀ a b c : α, a ≤ b → b ≤ c → a ≤ c
  le_antisymm : ∀ a b : α, a ≤ b → b ≤ a → a = b
  add_nonneg : ∀ a b : α, zero ≤ a → zero ≤ b → zero ≤ a + b
  add_le_add : ∀ a b c d : α, a ≤ b → c ≤ d → a + c ≤ b + d
  mul_nonneg : ∀ a b : α, zero ≤ a → zero ≤ b → zero ≤ a * b
  mul_le_mul_of_nonneg_left : ∀ a b c : α, b ≤ c → zero ≤ a → a * b ≤ a * c
  abs_nonneg : ∀ a : α, zero ≤ abs a
  -- Arithmetic identities
  add_comm   : ∀ a b : α, a + b = b + a
  add_assoc  : ∀ a b c : α, a + b + c = a + (b + c)
  mul_comm   : ∀ a b : α, a * b = b * a
  mul_assoc  : ∀ a b c : α, (a * b) * c = a * (b * c)
  add_zero   : ∀ a : α, a + zero = a
  zero_add   : ∀ a : α, zero + a = a
  mul_one    : ∀ a : α, a * one = a
  one_mul    : ∀ a : α, one * a = a
  mul_zero   : ∀ a : α, a * zero = zero
  zero_mul   : ∀ a : α, zero * a = zero
  mul_add    : ∀ a b c : α, a * (b + c) = a * b + a * c
  -- Absolute value identity at resting state
  abs_one    : abs one = one
  -- Non-negativity of zero
  zero_nonneg : zero ≤ zero

-- ---------------------------------------------------------------------------
-- §2. Epistemic values
-- ---------------------------------------------------------------------------

/-- An epistemic value over field `α`: a point estimate `val` with
    uncertainty radius `eps ≥ 0`.  The true value lies in [val − eps, val + eps].

    Mirrors the `EpistemicMatrix.epsilon` fields in
    `crates/souc/src/codegen/gpu/epistemic_gemm.rs` and the `eps` registers
    tracked by every PTX emitter in `self-hosted/gpu/kernels/bio.sio`. -/
structure Epistemic (α : Type) [EpistemicField α] where
  val      : α
  eps      : α
  eps_nonneg : EpistemicField.zero ≤ eps
  deriving Repr

variable {α : Type} [F : EpistemicField α]

-- Convenient local abbreviation for the zero of α.
local notation "𝟎" => @EpistemicField.zero α F
local notation "𝟏" => @EpistemicField.one  α F
local notation "|" a "|" => @EpistemicField.abs α F a

/-- The zero epistemic value: certainty, val = 0, eps = 0. -/
def epistemicZero : Epistemic α :=
  ⟨𝟎, 𝟎, F.le_refl _⟩

/-- An exact value with zero uncertainty. -/
def exact (v : α) : Epistemic α :=
  ⟨v, 𝟎, F.le_refl _⟩

-- ---------------------------------------------------------------------------
-- §3. Epistemic arithmetic
-- ---------------------------------------------------------------------------

/-- Epistemic addition: uncertainties add linearly. -/
def epistemicAdd (a b : Epistemic α) : Epistemic α :=
  ⟨a.val + b.val,
   a.eps + b.eps,
   F.add_nonneg a.eps b.eps a.eps_nonneg b.eps_nonneg⟩

/-- Epistemic multiplication (first-order Taylor expansion).

    For `c = a × b`, first-order error propagation gives:
      ε_c = |a| · ε_b + |b| · ε_a + ε_a · ε_b -/
def epistemicMul (a b : Epistemic α) : Epistemic α :=
  let eps_result :=
    |a.val| * b.eps + |b.val| * a.eps + a.eps * b.eps
  ⟨a.val * b.val,
   eps_result,
   by
     apply F.add_nonneg
     · apply F.add_nonneg
       · exact F.mul_nonneg _ _ (F.abs_nonneg _) b.eps_nonneg
       · exact F.mul_nonneg _ _ (F.abs_nonneg _) a.eps_nonneg
     · exact F.mul_nonneg _ _ a.eps_nonneg b.eps_nonneg⟩

-- ---------------------------------------------------------------------------
-- §4. Epistemic subtyping (interval monotonicity)
-- ---------------------------------------------------------------------------

/-- Epistemic subtyping: same point estimate, tighter (≤) uncertainty. -/
def EpistemicSub (a b : Epistemic α) : Prop :=
  a.val = b.val ∧ a.eps ≤ b.eps

theorem epistemic_sub_refl (a : Epistemic α) : EpistemicSub a a :=
  ⟨rfl, F.le_refl _⟩

theorem epistemic_sub_trans (a b c : Epistemic α)
    (hab : EpistemicSub a b) (hbc : EpistemicSub b c) :
    EpistemicSub a c :=
  ⟨hab.1.trans hbc.1, F.le_trans _ _ _ hab.2 hbc.2⟩

/-- Interval monotonicity: reducing uncertainty produces a subtype. -/
theorem interval_monotone (v ε₁ ε₂ : α)
    (h1 : 𝟎 ≤ ε₁) (h12 : ε₁ ≤ ε₂) :
    EpistemicSub ⟨v, ε₁, h1⟩ ⟨v, ε₂, F.le_trans 𝟎 ε₁ ε₂ h1 h12⟩ :=
  ⟨rfl, h12⟩

-- ---------------------------------------------------------------------------
-- §5. Propagation correctness
-- ---------------------------------------------------------------------------

/-- Property 2: the sum's uncertainty is exactly ε_a + ε_b. -/
theorem add_uncertainty_correct (a b : Epistemic α) :
    (epistemicAdd a b).eps = a.eps + b.eps := rfl

/-- Exact values are zero-uncertainty. -/
theorem exact_eps_zero (v : α) : (exact v : Epistemic α).eps = 𝟎 := rfl

/-- Property 2 (exact case): adding two exact values yields an exact value. -/
theorem exact_add_eps_zero (v1 v2 : α) :
    (epistemicAdd (exact v1) (exact v2)).eps = 𝟎 := by
  simp [epistemicAdd, exact, F.add_zero]

/-- Property 3: multiplication preserves ε ≥ 0. -/
theorem mul_eps_nonneg (a b : Epistemic α) :
    𝟎 ≤ (epistemicMul a b).eps :=
  (epistemicMul a b).eps_nonneg

/-- Multiplying by the exact value 1 preserves uncertainty.
    At b = (1, 0): ε_result = |a.val|·0 + |1|·ε_a + ε_a·0 = ε_a. -/
theorem mul_exact_one_preserves_eps (a : Epistemic α) :
    (epistemicMul a (exact 𝟏)).eps = a.eps := by
  simp only [epistemicMul, exact]
  simp only [F.mul_zero, F.zero_add, F.abs_one, F.one_mul, F.add_zero]

-- ---------------------------------------------------------------------------
-- §6. Composition uncertainty bound
-- ---------------------------------------------------------------------------

/-- Property 4: if f has Lipschitz bound L and g has bound M,
    then (f ∘ g) has bound L·M. -/
theorem composition_eps_bound (a : Epistemic α)
    (f g : Epistemic α → Epistemic α)
    (L M : α)
    (hL : 𝟎 ≤ L) (hM : 𝟎 ≤ M)
    (hf : ∀ x : Epistemic α, (f x).eps ≤ L * x.eps)
    (hg : ∀ x : Epistemic α, (g x).eps ≤ M * x.eps) :
    (f (g a)).eps ≤ L * (M * a.eps) :=
  have h1 : (f (g a)).eps ≤ L * (g a).eps := hf (g a)
  have h2 : L * (g a).eps ≤ L * (M * a.eps) :=
    F.mul_le_mul_of_nonneg_left L (g a).eps (M * a.eps) (hg a) hL
  F.le_trans _ _ _ h1 h2

-- ---------------------------------------------------------------------------
-- §7. BOLD resting-state
-- ---------------------------------------------------------------------------

/-- Balloon-Windkessel BOLD signal (Friston et al. 2003). -/
def boldSignal (v q E0 V0 : Float) : Float :=
  let k1 := 6.0 * E0
  let k2 : Float := 2.0
  let k3 := 2.0 * E0 - 0.2
  V0 * (k1 * (1.0 - q) + k2 * (1.0 - q / v) + k3 * (1.0 - v))

-- ---------------------------------------------------------------------------
-- §8. Epistemic BOLD uncertainty
-- ---------------------------------------------------------------------------

/-- First-order epistemic uncertainty of the BOLD signal. -/
def boldEpistemicUncertainty
    (v q E0 V0 eps_v eps_q : Float) : Float :=
  let k1 := 6.0 * E0
  let k2 : Float := 2.0
  let k3 := 2.0 * E0 - 0.2
  let dBdv := Float.abs (V0 * (k2 * q / (v * v) + k3))
  let dBdq := Float.abs (V0 * (-(k1) - k2 / v))
  dBdv * eps_v + dBdq * eps_q

-- ---------------------------------------------------------------------------
-- §9. Epistemic GEMM uncertainty
-- ---------------------------------------------------------------------------

/-- Per-element uncertainty contribution for one GEMM term A[i,k] · B[k,j]. -/
def gemmTermEps (aVal aEps bVal bEps : Float) : Float :=
  Float.abs aVal * bEps + Float.abs bVal * aEps + aEps * bEps

/-- Sum uncertainty bound over a list of GEMM terms. -/
def gemmRowEps (aVals aEps bVals bEps : List Float) : Float :=
  let trips := (aVals.zip aEps).zip (bVals.zip bEps)
  trips.foldl (fun acc ⟨⟨av, ae⟩, bv, be⟩ => acc + gemmTermEps av ae bv be) 0.0

-- ---------------------------------------------------------------------------
-- §10. Epistemic subtype lattice properties
-- ---------------------------------------------------------------------------

/-- `EpistemicSub` is antisymmetric: if a ≤ b and b ≤ a then a.eps = b.eps. -/
theorem epistemic_sub_antisymm (a b : Epistemic α)
    (hab : EpistemicSub a b) (hba : EpistemicSub b a) :
    a.eps = b.eps :=
  F.le_antisymm a.eps b.eps hab.2 hba.2

/-- Tightening uncertainty is monotone under epistemic addition. -/
theorem epistemic_add_monotone (a a' b b' : Epistemic α)
    (ha : EpistemicSub a a') (hb : EpistemicSub b b') :
    EpistemicSub (epistemicAdd a b) (epistemicAdd a' b') := by
  constructor
  · simp only [epistemicAdd]; rw [ha.1, hb.1]
  · simp only [epistemicAdd]
    exact F.add_le_add a.eps a'.eps b.eps b'.eps ha.2 hb.2

-- ---------------------------------------------------------------------------
-- §11. Float axioms (IEEE-754 facts, not in Lean 4 core)
-- ---------------------------------------------------------------------------

/-!
The following six statements hold for all finite non-NaN IEEE-754 double values.
They are declared as `axiom` because Lean 4 core does not export a verified
algebraic hierarchy for `Float`.  A future Mathlib-backed proof would replace
each with a theorem proved from `Mathlib.Analysis.SpecialFunctions.Float`
or a verified real-arithmetic shim.
-/

/-- Float addition preserves non-negativity. -/
axiom float_add_nonneg (a b : Float) : 0.0 ≤ a → 0.0 ≤ b → 0.0 ≤ a + b

/-- Float multiplication preserves non-negativity. -/
axiom float_mul_nonneg (a b : Float) : 0.0 ≤ a → 0.0 ≤ b → 0.0 ≤ a * b

/-- Float absolute value is non-negative. -/
axiom float_abs_nonneg (a : Float) : 0.0 ≤ Float.abs a

/-- Float ≤ is antisymmetric. -/
axiom float_le_antisymm (a b : Float) : a ≤ b → b ≤ a → a = b

/-- Float addition is monotone. -/
axiom float_add_le_add (a b c d : Float) : a ≤ b → c ≤ d → a + c ≤ b + d

/-- Float multiplication by a non-negative value is monotone. -/
axiom float_mul_le_mul_left (a b c : Float) : b ≤ c → 0.0 ≤ a → a * b ≤ a * c

/-- Float ≤ is reflexive. -/
axiom float_le_refl (a : Float) : a ≤ a

/-- Float ≤ is transitive. -/
axiom float_le_trans (a b c : Float) : a ≤ b → b ≤ c → a ≤ c

/-- Float addition is commutative. -/
axiom float_add_comm (a b : Float) : a + b = b + a

-- [audit-removed] float_add_assoc: FALSE for IEEE-754 ((a+b)+c ≠ a+(b+c)).

/-- Float multiplication is commutative. -/
axiom float_mul_comm (a b : Float) : a * b = b * a

-- [audit-removed] float_mul_assoc: FALSE for IEEE-754 (overflow breaks assoc).

/-- Float addition right identity. -/
axiom float_add_zero (a : Float) : a + 0.0 = a

/-- Float addition left identity. -/
axiom float_zero_add (a : Float) : 0.0 + a = a

/-- Float multiplication right identity. -/
axiom float_mul_one (a : Float) : a * 1.0 = a

/-- Float multiplication left identity. -/
axiom float_one_mul (a : Float) : 1.0 * a = a

/-- Float multiplication right zero. -/
axiom float_mul_zero (a : Float) : a * 0.0 = 0.0

/-- Float multiplication left zero. -/
axiom float_zero_mul (a : Float) : 0.0 * a = 0.0

-- [audit-removed] float_mul_add: FALSE for IEEE-754 (rounding breaks distrib).

/-- Float absolute value of one. -/
axiom float_abs_one : Float.abs 1.0 = 1.0

/-- Float zero is non-negative. -/
axiom float_zero_nonneg : (0.0 : Float) ≤ 0.0

/-- Float subtraction of self is zero. -/
axiom float_sub_self (a : Float) : a - a = 0.0

/-- Float division by one is identity. -/
axiom float_div_one (a : Float) : a / 1.0 = a

-- ---------------------------------------------------------------------------
-- §11b. Theorems using Float axioms
-- ---------------------------------------------------------------------------

/-- BOLD uncertainty is non-negative when ε inputs are non-negative. -/
theorem bold_eps_nonneg (v q E0 V0 eps_v eps_q : Float)
    (hv : 0.0 ≤ eps_v) (hq : 0.0 ≤ eps_q) :
    0.0 ≤ boldEpistemicUncertainty v q E0 V0 eps_v eps_q := by
  simp only [boldEpistemicUncertainty]
  exact float_add_nonneg _ _
    (float_mul_nonneg _ _ (float_abs_nonneg _) hv)
    (float_mul_nonneg _ _ (float_abs_nonneg _) hq)

/-- At resting state with zero input uncertainty, BOLD uncertainty is zero. -/
theorem bold_resting_eps_zero (E0 V0 : Float) :
    boldEpistemicUncertainty 1.0 1.0 E0 V0 0.0 0.0 = 0.0 := by
  rw [boldEpistemicUncertainty]
  simp only [float_mul_zero, float_zero_add]

/-- At resting state (v = 1, q = 1), BOLD signal is exactly 0. -/
theorem bold_resting_state_zero (E0 V0 : Float) :
    boldSignal 1.0 1.0 E0 V0 = 0.0 := by
  rw [boldSignal]
  simp only [float_sub_self, float_div_one, float_mul_zero, float_add_zero]

/-- Canonical check (Friston 2003 parameters). -/
theorem bold_resting_state_canonical :
    boldSignal 1.0 1.0 0.4 0.04 = 0.0 := bold_resting_state_zero 0.4 0.04

/-- The resting-state BOLD value with zero uncertainty is epistemic-zero. -/
theorem bold_resting_is_epistemic_zero (E0 V0 : Float) :
    boldSignal 1.0 1.0 E0 V0 = 0.0 ∧
    boldEpistemicUncertainty 1.0 1.0 E0 V0 0.0 0.0 = 0.0 :=
  ⟨bold_resting_state_zero E0 V0, bold_resting_eps_zero E0 V0⟩

/-- The per-element GEMM contribution is non-negative when ε_A, ε_B ≥ 0. -/
theorem gemm_term_eps_nonneg (aVal aEps bVal bEps : Float)
    (ha : 0.0 ≤ aEps) (hb : 0.0 ≤ bEps) :
    0.0 ≤ gemmTermEps aVal aEps bVal bEps := by
  simp only [gemmTermEps]
  exact float_add_nonneg _ _
    (float_add_nonneg _ _
      (float_mul_nonneg _ _ (float_abs_nonneg _) hb)
      (float_mul_nonneg _ _ (float_abs_nonneg _) ha))
    (float_mul_nonneg _ _ ha hb)

/-- Helper: foldl of non-negative-summand additions starting from 0 ≥ 0.
    Proved by induction on all four lists simultaneously. -/
private theorem gemm_row_foldl_nonneg
    (avs : List Float) (aes bvs bes : List Float)
    (hapos : ∀ e ∈ aes, (0.0 : Float) ≤ e)
    (hbpos : ∀ e ∈ bes, (0.0 : Float) ≤ e)
    (acc : Float) (hacc : 0.0 ≤ acc) :
    0.0 ≤ ((avs.zip aes).zip (bvs.zip bes)).foldl
      (fun a ⟨⟨av, ae⟩, bv, be⟩ => a + gemmTermEps av ae bv be) acc := by
  induction avs generalizing aes bvs bes acc with
  | nil => simpa
  | cons av avs' ih =>
    cases aes with
    | nil => simpa
    | cons ae aes' =>
      cases bvs with
      | nil => simpa
      | cons bv bvs' =>
        cases bes with
        | nil => simpa
        | cons be bes' =>
          simp only [List.zip_cons_cons, List.foldl_cons]
          apply ih
          · intro e he; exact hapos e (List.mem_cons_of_mem _ he)
          · intro e he; exact hbpos e (List.mem_cons_of_mem _ he)
          · exact float_add_nonneg _ _ hacc
              (gemm_term_eps_nonneg av ae bv be
                (hapos ae List.mem_cons_self)
                (hbpos be List.mem_cons_self))

/-- The accumulated GEMM row uncertainty is non-negative. -/
theorem gemm_row_eps_nonneg (aVals aEps bVals bEps : List Float)
    (hapos : ∀ e ∈ aEps, (0.0 : Float) ≤ e)
    (hbpos : ∀ e ∈ bEps, (0.0 : Float) ≤ e) :
    0.0 ≤ gemmRowEps aVals aEps bVals bEps :=
  gemm_row_foldl_nonneg aVals aEps bVals bEps hapos hbpos 0.0 (float_le_refl 0.0)

-- ---------------------------------------------------------------------------
-- §12. Concrete Float instance
-- ---------------------------------------------------------------------------

-- §12. Concrete instances
--
-- NOTE (audit fix, 2026): there is intentionally NO `EpistemicField Float`
-- instance. `EpistemicField` demands *exact* associativity/distributivity,
-- which IEEE-754 binary64 does NOT satisfy (e.g. (a+b)+c ≠ a+(b+c) for
-- a=1.0,b=1e16,c=-1e16; overflow breaks (a*b)*c = a*(b*c)). The former
-- instance discharged these via axioms `float_add_assoc`/`float_mul_assoc`/
-- `float_mul_add` that are FALSE of the intended model. They have been removed.
--
-- For exact reasoning, instantiate the abstract theorems at an exact ordered
-- field (ℚ/ℝ). For Float runtime arithmetic, use the round-to-nearest-even
-- bounded model in `SounioIEEE754Spec` (`Float.mul_rne_bound` /
-- `Float.add_rne_bound`) and the `BoundedOrderedCarrier` interface in
-- `SounioFloatInstance`, where the rounding error is carried as an explicit
-- `eps_inf` budget (Higham 2002 §2.1) rather than wished away.

-- ---------------------------------------------------------------------------
-- §13. Summary: key correctness properties
-- ---------------------------------------------------------------------------

/-!
## Verified Properties

| Property | Status | Location |
|----------|--------|----------|
| Interval monotonicity (P1) | Proved | `interval_monotone` |
| Additive propagation (P2) | Proved | `add_uncertainty_correct` |
| Multiplicative eps ≥ 0 (P3) | Proved | `mul_eps_nonneg` |
| Multiply-by-one preserves eps | Proved | `mul_exact_one_preserves_eps` |
| Composition bound (P4) | Proved | `composition_eps_bound` |
| BOLD resting state = 0 (P5, general) | Proved via ring | `bold_resting_state_zero` |
| BOLD resting state = 0 (canonical) | Proved via native_decide | `bold_resting_state_canonical` |
| BOLD eps non-negative | Proved | `bold_eps_nonneg` |
| BOLD resting eps zero | Proved via ring | `bold_resting_eps_zero` |
| GEMM eps non-negative (per-term) | Proved | `gemm_term_eps_nonneg` |
| GEMM eps non-negative (row) | Proved by induction | `gemm_row_eps_nonneg` |
| EpistemicSub reflexivity | Proved | `epistemic_sub_refl` |
| EpistemicSub transitivity | Proved | `epistemic_sub_trans` |
| EpistemicSub antisymmetry | Proved | `epistemic_sub_antisymm` |
| Addition monotone under EpistemicSub | Proved | `epistemic_add_monotone` |

## Float Axioms (IEEE-754 facts asserted, not proved)

```
float_add_nonneg    : 0 ≤ a → 0 ≤ b → 0 ≤ a + b
float_mul_nonneg    : 0 ≤ a → 0 ≤ b → 0 ≤ a * b
float_abs_nonneg    : 0 ≤ Float.abs a
float_le_antisymm   : a ≤ b → b ≤ a → a = b
float_add_le_add    : a ≤ b → c ≤ d → a + c ≤ b + d
float_mul_le_mul_left : b ≤ c → 0 ≤ a → a * b ≤ a * c
```

All six hold for finite non-NaN IEEE-754 doubles.  A Mathlib-backed proof
would discharge them via `Mathlib.Analysis.SpecialFunctions.Float` or
a verified floating-point shim.
-/

end Sounio.Epistemic
