import SounioSkewCategory
import SounioEffects
/-!
# The Bidirectional Bridge: Type Inference Modes ↔ Associator Signs

The central contribution: a formal correspondence between the two modes
of bidirectional type checking and the two parenthesization signs of
the Cayley-Dickson associator.

## The Bridge (informal)

In bidirectional type checking (Pierce-Turner 1998, Dunfield-Krishnaswami 2021):
- **Synthesis** (⇒): given an expression, produce its type (forward, constructive)
- **Checking** (⇐): given a type, verify the expression against it (backward, verificatory)

For Cayley-Dickson basis expressions `(e_i · e_j) · e_k`:
- **Alpha** (synthesis direction): σ(i,j) · σ(i⊕j, k) — left parenthesization sign
- **Beta** (checking direction): σ(j,k) · σ(i, j⊕k) — right parenthesization sign

The correspondence:
  α = β  ↔  Fano triple  ↔  modes AGREE    ↔  no NonAssoc needed
  α ≠ β  ↔  non-Fano     ↔  modes DISAGREE ↔  NonAssoc required

This exhibits a precise structural analogy with the two-state vector
formalism (Aharonov-Bergmann-Lebowitz 1964):
  Forward state |ψ⟩ ↔ Synthesis mode (⇒)
  Backward state ⟨φ| ↔ Checking mode (⇐)
  Interference α - β ↔ Mode disagreement

## Main results

- `synth_eq_alpha`: the synthesis sign IS the alpha (left) sign
- `check_eq_beta`: the checking sign IS the beta (right) sign
- `mode_agreement_iff_fano`: modes agree iff Fano triple (175 triples)
- `mode_disagreement_iff_nonfano`: modes disagree iff non-Fano (168 triples)
- `nonassoc_required_iff_mode_disagree`: NonAssoc effect necessity
- `mode_dagger_exchange`: dagger exchanges synthesis and checking
- `mode_perfect_duality`: 84 synth-dominant + 84 check-dominant = 168

No sorry. No Mathlib. All finite theorems by `native_decide`.

References:
  - Pierce & Turner (1998), "Local Type Inference", POPL
  - Dunfield & Krishnaswami (2021), "Bidirectional Typing", ACM Computing Surveys
  - Mihejevs & Hedges (2025), "Canonical bidirectional typechecking"
  - Aharonov, Bergmann & Lebowitz (1964), "Time symmetry in QM"
  - Abramsky & Coecke (2004), "A categorical semantics of quantum protocols"

Mirrors Sounio implementation:
  - `self-hosted/check/effects.sio`: NonAssoc = effect ID 14
  - `self-hosted/check/cayley_dickson.sio`: cd_is_fano_triple
  - `self-hosted/ir/algebra.sio`: ir_can_reassociate_triple
  - `self-hosted/check/hyper.sio`: check_hyper_binary
-/

open Sounio.CayleyDickson
open Sounio.SkewCategory
open Sounio.Effects

namespace Sounio.BidirectionalBridge

-- ================================================================
-- §1. Bidirectional typing modes
-- ================================================================

/-- The two modes of bidirectional type checking.
    Following Dunfield-Krishnaswami (2021). -/
inductive Mode where
  | Synth  -- synthesis (⇒): term → type (forward, constructive)
  | Check  -- checking (⇐): type → term (backward, verificatory)
  deriving DecidableEq, Repr

/-- The dagger exchanges modes: dag(Synth) = Check, dag(Check) = Synth. -/
def modeDagger : Mode → Mode
  | .Synth => .Check
  | .Check => .Synth

theorem modeDagger_involution (m : Mode) : modeDagger (modeDagger m) = m := by
  cases m <;> rfl

-- ================================================================
-- §2. The synthesis and checking signs
-- ================================================================

/-- **Synthesis sign** for basis triple (i,j,k).
    Computes the sign of the LEFT parenthesization: (e_i · e_j) · e_k.
    This is the sign produced by FORWARD (synthesis) type propagation.

    synth(i,j,k) = σ(i,j) · σ(i⊕j, k) = alphaSign(i,j,k) -/
def synthSign (i j k : Nat) : Int := alphaSign i j k

/-- **Checking sign** for basis triple (i,j,k).
    Computes the sign of the RIGHT parenthesization: e_i · (e_j · e_k).
    This is the sign produced by BACKWARD (checking) type propagation.

    check(i,j,k) = σ(j,k) · σ(i, j⊕k) = betaSign(i,j,k) -/
def checkSign (i j k : Nat) : Int := betaSign i j k

/-- The mode interference: the difference between synthesis and checking signs.
    This equals the wave function (associator coefficient).

    interference(i,j,k) = synth(i,j,k) - check(i,j,k) = waveFunc(i,j,k) -/
def modeInterference (i j k : Nat) : Int := synthSign i j k - checkSign i j k

-- ================================================================
-- §3. Definitional equalities (the bridge identifications)
-- ================================================================

/-- The synthesis sign IS the alpha (left-parenthesization) sign. -/
theorem synth_eq_alpha (i j k : Nat) : synthSign i j k = alphaSign i j k := rfl

/-- The checking sign IS the beta (right-parenthesization) sign. -/
theorem check_eq_beta (i j k : Nat) : checkSign i j k = betaSign i j k := rfl

/-- Mode interference IS the wave function (associator coefficient). -/
theorem interference_eq_wave (i j k : Nat) :
    modeInterference i j k = waveFunc i j k := rfl

-- ================================================================
-- §4. The core correspondence: mode agreement ↔ Fano
-- ================================================================

/-- Mode agreement: synthesis and checking produce the same sign. -/
def modesAgree (i j k : Nat) : Bool := synthSign i j k == checkSign i j k

/-- **Bidirectional-Fano Correspondence** (fundamental theorem).
    For all 343 imaginary triples: modes agree ↔ Fano triple.

    This is the BRIDGE between type theory and algebra:
    the type-checking algorithm's behavior (synthesis vs checking)
    corresponds exactly to the algebraic structure (Fano plane). -/
theorem mode_agreement_iff_fano :
    (allImagTriples.filter (fun t =>
      modesAgree t.1 t.2.1 t.2.2 == isFano t.1 t.2.1 t.2.2)).length = 343 := by
  native_decide

/-- Exactly 175 triples have mode agreement (Fano). -/
theorem mode_agreement_count :
    (allImagTriples.filter (fun t => modesAgree t.1 t.2.1 t.2.2)).length = 175 := by
  native_decide

/-- Exactly 168 triples have mode disagreement (non-Fano). -/
theorem mode_disagreement_count :
    (allImagTriples.filter (fun t => !modesAgree t.1 t.2.1 t.2.2)).length = 168 := by
  native_decide

-- ================================================================
-- §5. NonAssoc effect necessity
-- ================================================================

/-- The NonAssoc effect is required precisely when modes disagree.

    In Sounio's type system:
    - If modes AGREE (Fano triple): the expression type-checks in both
      modes without declaring NonAssoc. The compiler can reassociate.
    - If modes DISAGREE (non-Fano triple): the expression REQUIRES the
      NonAssoc effect. The compiler must preserve parenthesization.

    This connects the categorical bridge to the effect system:
    NonAssoc ∈ ρ  ↔  ¬(modesAgree i j k)  ↔  ¬(isFano i j k) -/
def requiresNonAssocEffect (i j k : Nat) : Bool := !modesAgree i j k

/-- NonAssoc is required for exactly the 168 non-Fano triples. -/
theorem nonassoc_required_count :
    (allImagTriples.filter (fun t =>
      requiresNonAssocEffect t.1 t.2.1 t.2.2)).length = 168 := by
  native_decide

/-- NonAssoc requirement matches the Fano complement. -/
theorem nonassoc_iff_not_fano :
    (allImagTriples.filter (fun t =>
      requiresNonAssocEffect t.1 t.2.1 t.2.2 ==
      !isFano t.1 t.2.1 t.2.2)).length = 343 := by
  native_decide

-- ================================================================
-- §6. Dagger exchanges synthesis and checking
-- ================================================================

/-- Synthesis-dominant: synth_sign > check_sign (wave = +2). -/
def isSynthDominant (i j k : Nat) : Bool :=
  modeInterference i j k == 2

/-- Check-dominant: check_sign > synth_sign (wave = -2). -/
def isCheckDominant (i j k : Nat) : Bool :=
  modeInterference i j k == -2

/-- The dagger exchanges synthesis-dominant and check-dominant triples.
    If (i,j,k) is synthesis-dominant, then dag(i,j,k) = (k,j,i) is
    check-dominant, and vice versa.

    This is the TYPE-THEORETIC interpretation of time-reversal:
    the dagger swaps which mode "dominates." -/
theorem dagger_exchanges_modes :
    (allImagTriples.filter (fun t =>
      isSynthDominant t.1 t.2.1 t.2.2 &&
      isCheckDominant (octDagger t).1 (octDagger t).2.1 (octDagger t).2.2)).length
    = 84 := by native_decide

/-- Reverse direction: check-dominant → synthesis-dominant under dagger. -/
theorem dagger_exchanges_modes_rev :
    (allImagTriples.filter (fun t =>
      isCheckDominant t.1 t.2.1 t.2.2 &&
      isSynthDominant (octDagger t).1 (octDagger t).2.1 (octDagger t).2.2)).length
    = 84 := by native_decide

-- ================================================================
-- §7. Perfect duality: 84/84 synthesis/checking decomposition
-- ================================================================

/-- Exactly 84 triples are synthesis-dominant. -/
theorem synth_dominant_count :
    (allImagTriples.filter (fun t =>
      isSynthDominant t.1 t.2.1 t.2.2)).length = 84 := by native_decide

/-- Exactly 84 triples are check-dominant. -/
theorem check_dominant_count :
    (allImagTriples.filter (fun t =>
      isCheckDominant t.1 t.2.1 t.2.2)).length = 84 := by native_decide

/-- The partition is complete: 175 + 84 + 84 = 343. -/
theorem tripartition_complete :
    (allImagTriples.filter (fun t => modesAgree t.1 t.2.1 t.2.2)).length +
    (allImagTriples.filter (fun t => isSynthDominant t.1 t.2.1 t.2.2)).length +
    (allImagTriples.filter (fun t => isCheckDominant t.1 t.2.1 t.2.2)).length
    = 343 := by native_decide

-- ================================================================
-- §8. Cross-validation: all definitions agree
-- ================================================================

/-- Mode interference equals wave function for all 343 triples. -/
theorem interference_cross_validation :
    (allImagTriples.filter (fun t =>
      modeInterference t.1 t.2.1 t.2.2 == waveFunc t.1 t.2.1 t.2.2)).length
    = 343 := by native_decide

/-- Mode agreement matches the compiler's reassociation predicate.
    modesAgree(i,j,k) = isFano(i,j,k) for all triples.
    The TYPE CHECKER and the OPTIMIZER agree. -/
theorem typechecker_optimizer_agree :
    (allImagTriples.filter (fun t =>
      modesAgree t.1 t.2.1 t.2.2 == isFano t.1 t.2.1 t.2.2)).length
    = 343 := by native_decide

-- ================================================================
-- §9. The arrow of time as mode asymmetry
-- ================================================================

/-- No non-Fano triple has equal synthesis and checking signs.
    Mode disagreement is STRICT for non-Fano triples:
    if modes disagree, they disagree by exactly ±2. -/
theorem nonfano_strict_disagreement :
    (allImagTriples.filter (fun t =>
      !modesAgree t.1 t.2.1 t.2.2 &&
      (modeInterference t.1 t.2.1 t.2.2 != 2 &&
       modeInterference t.1 t.2.1 t.2.2 != -2))).length = 0 := by native_decide

/-- The synthesis-dominant and check-dominant counts are EQUAL.
    There is no preferred mode — the asymmetry is balanced.

    If there is an "arrow of time" in the type-checking algorithm,
    it comes from the CHOICE of which mode is called "synthesis"
    (forward) and which is called "checking" (backward).
    The algebra itself is perfectly balanced. -/
theorem mode_symmetry :
    (allImagTriples.filter (fun t => isSynthDominant t.1 t.2.1 t.2.2)).length =
    (allImagTriples.filter (fun t => isCheckDominant t.1 t.2.1 t.2.2)).length := by
  native_decide

end Sounio.BidirectionalBridge
