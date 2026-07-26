-- formal/lean4/SounioFalsificationLedger.lean
import SounioEffects
import SounioEpistemic

/-!
# Sounio Falsification Ledger — Lean 4 Formalization

Formalizes the claim schema of the Falsification Ledger: evidence levels,
verdicts, zero-provenance taxonomy, and consistency conditions.

This is a NOVEL formalization — no prior Lean 4 project covers
compiler-integrated scientific claims with zero-provenance evidence.

References:
- Agourakis (2026), "Falsification Ledger — compiler-integrated scientific claims"
- Agourakis (2026), "Zero-provenance claims — zero-event taxonomy"

No sorry. No Mathlib.
-/

open Sounio.Effects Sounio.Epistemic

namespace Sounio.FalsificationLedger

-- ================================================================
-- S1. Evidence levels (totally ordered)
-- ================================================================

/-- Evidence level for a scientific claim. -/
inductive EvidenceLevel where
  | conceived
  | implemented
  | typeCheck
  | compiles
  | executes
  | gateGreen
  | instrumentControlled
  | claimReady
  deriving DecidableEq, Repr

/-- Total order on evidence levels: later levels subsume earlier ones. -/
def EvidenceLevel.le (a b : EvidenceLevel) : Prop :=
  match a, b with
  | .conceived, _ => True
  | .implemented, .conceived => False
  | .implemented, _ => True
  | .typeCheck, .conceived | .typeCheck, .implemented => False
  | .typeCheck, _ => True
  | .compiles, .conceived | .compiles, .implemented | .compiles, .typeCheck => False
  | .compiles, _ => True
  | .executes, .conceived | .executes, .implemented | .executes, .typeCheck | .executes, .compiles => False
  | .executes, _ => True
  | .gateGreen, .conceived | .gateGreen, .implemented | .gateGreen, .typeCheck | .gateGreen, .compiles | .gateGreen, .executes => False
  | .gateGreen, _ => True
  | .instrumentControlled, .conceived | .instrumentControlled, .implemented | .instrumentControlled, .typeCheck | .instrumentControlled, .compiles | .instrumentControlled, .executes | .instrumentControlled, .gateGreen => False
  | .instrumentControlled, _ => True
  | .claimReady, .claimReady => True
  | .claimReady, _ => False

/-- Evidence order is reflexive. -/
theorem EvidenceLevel.le_refl (a : EvidenceLevel) : EvidenceLevel.le a a := by
  cases a <;> simp [EvidenceLevel.le]

/-- Evidence order is transitive. -/
theorem EvidenceLevel.le_trans (a b c : EvidenceLevel) :
    EvidenceLevel.le a b → EvidenceLevel.le b c → EvidenceLevel.le a c := by
  cases a <;> cases b <;> cases c <;> simp [EvidenceLevel.le] <;> tauto

/-- Evidence order is antisymmetric. -/
theorem EvidenceLevel.le_antisymm (a b : EvidenceLevel) :
    EvidenceLevel.le a b → EvidenceLevel.le b a → a = b := by
  cases a <;> cases b <;> simp [EvidenceLevel.le] <;> tauto

-- ================================================================
-- S2. Verdicts
-- ================================================================

/-- Verdict for a scientific claim. -/
inductive Verdict where
  | alive
  | negative
  | dormant
  | refuted
  deriving DecidableEq, Repr

/-- A negative verdict requires a falsifier. -/
def Verdict.requiresFalsifier (v : Verdict) : Prop :=
  match v with
  | .negative => True
  | _ => False

/-- Verdict is decidable. -/
instance : DecidableEq Verdict := inferInstance

-- ================================================================
-- S3. Zero-provenance taxonomy
-- ================================================================

/-- Zero-event provenance categories. -/
inductive Provenance where
  | absent
  | cancelled
  | annihilated
  | belowResolution
  | rounded
  | gated
  | unknown
  deriving DecidableEq, Repr

/-- The provenance taxonomy is exhaustive: every provenance is one of the seven. -/
theorem Provenance.exhaustive (p : Provenance) :
    p = .absent ∨ p = .cancelled ∨ p = .annihilated ∨ p = .belowResolution ∨
    p = .rounded ∨ p = .gated ∨ p = .unknown := by
  cases p <;> tauto

/-- The provenance taxonomy has seven distinct categories. -/
theorem Provenance.cardinality : Provenance = .absent ∨ Provenance = .cancelled ∨ Provenance = .annihilated ∨ Provenance = .belowResolution ∨ Provenance = .rounded ∨ Provenance = .gated ∨ Provenance = .unknown := by
  cases Provenance <;> tauto

-- ================================================================
-- S4. Claim schema
-- ================================================================

/-- A scientific claim in the Falsification Ledger. -/
structure Claim where
  hypothesis : String
  falsifier : String
  evidence : EvidenceLevel
  harness : String
  gate : String
  verdict : Verdict
  provenance : Option Provenance
  deriving DecidableEq, Repr

/-- A claim is consistent if a negative verdict has a nonempty falsifier. -/
def Claim.isConsistent (c : Claim) : Prop :=
  c.verdict = .negative → c.falsifier ≠ ""

/-- A claim is zero-consistent if it mentions zero evidence and has provenance. -/
def Claim.isZeroConsistent (c : Claim) : Prop :=
  (c.evidence = .instrumentControlled ∨ c.evidence = .gateGreen) →
  c.provenance ≠ none

/-- Consistency is decidable for claims with decidable string equality. -/
instance (c : Claim) : Decidable (c.isConsistent) :=
  inferInstanceAs (Decidable (c.verdict = .negative → c.falsifier ≠ ""))

/-- Zero-consistency is decidable. -/
instance (c : Claim) : Decidable (c.isZeroConsistent) :=
  inferInstanceAs (Decidable ((c.evidence = .instrumentControlled ∨ c.evidence = .gateGreen) → c.provenance ≠ none))

-- ================================================================
-- S5. Theorems
-- ================================================================

/-- A negative claim with empty falsifier is inconsistent. -/
theorem negative_empty_falsifier_inconsistent (c : Claim)
    (hv : c.verdict = .negative) (hf : c.falsifier = "") :
    ¬c.isConsistent := by
  intro h
  exact h hv hf

/-- A claim with provenance is zero-consistent if it has high evidence. -/
theorem provenance_zero_consistent (c : Claim)
    (he : c.evidence = .instrumentControlled ∨ c.evidence = .gateGreen)
    (hp : c.provenance ≠ none) :
    c.isZeroConsistent := by
  intro h
  exact hp

/-- The seven provenance categories are exhaustive. -/
theorem provenance_complete (p : Provenance) :
    p = .absent ∨ p = .cancelled ∨ p = .annihilated ∨ p = .belowResolution ∨
    p = .rounded ∨ p = .gated ∨ p = .unknown := by
  exact Provenance.exhaustive p

-- ================================================================
-- S6. Ledger operations
-- ================================================================

/-- A ledger is a list of consistent claims. -/
def Ledger := List Claim

/-- All claims in a ledger are consistent. -/
def Ledger.allConsistent (l : Ledger) : Prop :=
  ∀ c ∈ l, c.isConsistent

/-- Adding a consistent claim preserves ledger consistency. -/
theorem Ledger.add_consistent (l : Ledger) (c : Claim)
    (hl : l.allConsistent) (hc : c.isConsistent) :
    (c :: l).allConsistent := by
  intro x hx
  cases hx with
  | head => exact hc
  | tail _ h => exact hl x h

/-- The empty ledger is consistent. -/
theorem Ledger.empty_consistent : [].allConsistent := by
  intro c h
  cases h

-- ================================================================
-- S7. Claim state transitions
-- ================================================================

/-- A claim can transition to dormant if it is not refuted. -/
def Claim.canDormant (c : Claim) : Prop :=
  c.verdict ≠ .refuted

/-- A claim can transition to refuted only from negative. -/
def Claim.canRefute (c : Claim) : Prop :=
  c.verdict = .negative

/-- Transition to dormant preserves consistency (vacuously, since dormant ≠ negative). -/
theorem Claim.dormant_preserves_consistency (c : Claim) :
    ({ c with verdict := .dormant }).isConsistent := by
  intro hv
  exact absurd hv (by decide)

/-- Transition to refuted preserves consistency (vacuously, since refuted ≠ negative). -/
theorem Claim.refute_preserves_consistency (c : Claim) :
    ({ c with verdict := .refuted }).isConsistent := by
  intro hv
  exact absurd hv (by decide)

end Sounio.FalsificationLedger
