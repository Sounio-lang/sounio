/-
  FORMAL_PARITY for the missing basis-fixed rebase in Pireus V13.

  Pulling an 11-bit basis-fixed potential through an arbitrary XOR-linear lane
  equivalence preserves its coboundary but generally makes the four standard
  basis values nonzero. The constructed linear character with those four
  basis values is subtracted (XORed) from the pullback. Its coboundary is zero, so
  the rebased potential defines the same gauge transformation and is again
  representable by the frozen 11-bit GaugeWord.

  This closes the abstract linear-rebase obligation. The 65536 matrix-code
  predicate still has to be connected to concrete XorLaneEquiv witnesses, and
  the 20160-code enumeration and outer 40320-view minimum remain open.
-/
import SounioPireusLinearSwapGaugeDescent

namespace SounioPireusBasisFixedGaugeRebase

open SounioPireusGaugeCoboundaryAction
open SounioPireusLinearSwapGaugeDescent

def lane0 : Lane := ⟨0, by decide⟩
def lane1 : Lane := ⟨1, by decide⟩
def lane2 : Lane := ⟨2, by decide⟩
def lane3 : Lane := ⟨3, by decide⟩
def lane4 : Lane := ⟨4, by decide⟩
def lane5 : Lane := ⟨5, by decide⟩
def lane6 : Lane := ⟨6, by decide⟩
def lane7 : Lane := ⟨7, by decide⟩
def lane8 : Lane := ⟨8, by decide⟩
def lane9 : Lane := ⟨9, by decide⟩
def lane10 : Lane := ⟨10, by decide⟩
def lane11 : Lane := ⟨11, by decide⟩
def lane12 : Lane := ⟨12, by decide⟩
def lane13 : Lane := ⟨13, by decide⟩
def lane14 : Lane := ⟨14, by decide⟩
def lane15 : Lane := ⟨15, by decide⟩

private theorem gauge_rank_0 : gaugeRank? 0 = none := by decide
private theorem gauge_rank_1 : gaugeRank? 1 = none := by decide
private theorem gauge_rank_2 : gaugeRank? 2 = none := by decide
private theorem gauge_rank_3 : gaugeRank? 3 = some 0 := by decide
private theorem gauge_rank_4 : gaugeRank? 4 = none := by decide
private theorem gauge_rank_5 : gaugeRank? 5 = some 1 := by decide
private theorem gauge_rank_6 : gaugeRank? 6 = some 2 := by decide
private theorem gauge_rank_7 : gaugeRank? 7 = some 3 := by decide
private theorem gauge_rank_8 : gaugeRank? 8 = none := by decide
private theorem gauge_rank_9 : gaugeRank? 9 = some 4 := by decide
private theorem gauge_rank_10 : gaugeRank? 10 = some 5 := by decide
private theorem gauge_rank_11 : gaugeRank? 11 = some 6 := by decide
private theorem gauge_rank_12 : gaugeRank? 12 = some 7 := by decide
private theorem gauge_rank_13 : gaugeRank? 13 = some 8 := by decide
private theorem gauge_rank_14 : gaugeRank? 14 = some 9 := by decide
private theorem gauge_rank_15 : gaugeRank? 15 = some 10 := by decide

def selectBit (coefficient value : Bool) : Bool := coefficient && value

@[simp] theorem select_bit_xor (left right value : Bool) :
    selectBit (left ^^ right) value =
      (selectBit left value ^^ selectBit right value) := by
  cases left <;> cases right <;> cases value <;> decide

private theorem test_bits_0 :
    Nat.testBit 0 0 = false ∧ Nat.testBit 0 1 = false ∧
      Nat.testBit 0 2 = false ∧ Nat.testBit 0 3 = false := by decide

private theorem test_bits_1 :
    Nat.testBit 1 0 = true ∧ Nat.testBit 1 1 = false ∧
      Nat.testBit 1 2 = false ∧ Nat.testBit 1 3 = false := by decide

private theorem test_bits_2 :
    Nat.testBit 2 0 = false ∧ Nat.testBit 2 1 = true ∧
      Nat.testBit 2 2 = false ∧ Nat.testBit 2 3 = false := by decide

private theorem test_bits_4 :
    Nat.testBit 4 0 = false ∧ Nat.testBit 4 1 = false ∧
      Nat.testBit 4 2 = true ∧ Nat.testBit 4 3 = false := by decide

private theorem test_bits_8 :
    Nat.testBit 8 0 = false ∧ Nat.testBit 8 1 = false ∧
      Nat.testBit 8 2 = false ∧ Nat.testBit 8 3 = true := by decide

def basisCharacter (potential : Potential) : Potential :=
  fun lane =>
    selectBit (lane.val.testBit 0) (potential lane1) ^^
      selectBit (lane.val.testBit 1) (potential lane2) ^^
      selectBit (lane.val.testBit 2) (potential lane4) ^^
      selectBit (lane.val.testBit 3) (potential lane8)

theorem basis_character_zero (potential : Potential) :
    basisCharacter potential lane0 = false := by
  rcases test_bits_0 with ⟨h0, h1, h2, h3⟩
  simp [basisCharacter, selectBit, lane0, h0, h1, h2, h3]

theorem basis_character_on_basis (potential : Potential) :
    basisCharacter potential lane1 = potential lane1 ∧
      basisCharacter potential lane2 = potential lane2 ∧
      basisCharacter potential lane4 = potential lane4 ∧
      basisCharacter potential lane8 = potential lane8 := by
  rcases test_bits_1 with ⟨h10, h11, h12, h13⟩
  rcases test_bits_2 with ⟨h20, h21, h22, h23⟩
  rcases test_bits_4 with ⟨h40, h41, h42, h43⟩
  rcases test_bits_8 with ⟨h80, h81, h82, h83⟩
  refine ⟨?_, ?_, ?_, ?_⟩
  · change (selectBit (Nat.testBit 1 0) (potential lane1) ^^
        selectBit (Nat.testBit 1 1) (potential lane2) ^^
        selectBit (Nat.testBit 1 2) (potential lane4) ^^
        selectBit (Nat.testBit 1 3) (potential lane8)) = potential lane1
    simp [selectBit, h10, h11, h12, h13]
  · change (selectBit (Nat.testBit 2 0) (potential lane1) ^^
        selectBit (Nat.testBit 2 1) (potential lane2) ^^
        selectBit (Nat.testBit 2 2) (potential lane4) ^^
        selectBit (Nat.testBit 2 3) (potential lane8)) = potential lane2
    simp [selectBit, h20, h21, h22, h23]
  · change (selectBit (Nat.testBit 4 0) (potential lane1) ^^
        selectBit (Nat.testBit 4 1) (potential lane2) ^^
        selectBit (Nat.testBit 4 2) (potential lane4) ^^
        selectBit (Nat.testBit 4 3) (potential lane8)) = potential lane4
    simp [selectBit, h40, h41, h42, h43]
  · change (selectBit (Nat.testBit 8 0) (potential lane1) ^^
        selectBit (Nat.testBit 8 1) (potential lane2) ^^
        selectBit (Nat.testBit 8 2) (potential lane4) ^^
        selectBit (Nat.testBit 8 3) (potential lane8)) = potential lane8
    simp [selectBit, h80, h81, h82, h83]

theorem basis_character_xor
    (potential : Potential) (left right : Lane) :
    basisCharacter potential (left ^^^ right) =
      (basisCharacter potential left ^^ basisCharacter potential right) := by
  unfold basisCharacter
  rw [lane_xor_val]
  simp only [Nat.testBit_xor, select_bit_xor]
  simp only [Bool.xor_assoc, Bool.xor_left_comm]

theorem basis_character_coboundary_zero
    (potential : Potential) (cell : Cell) :
    unrestrictedCoboundary (basisCharacter potential) cell = false := by
  simp [unrestrictedCoboundary, basis_character_xor]

def rebasedPotential (potential : Potential) : Potential :=
  fun lane => potential lane ^^ basisCharacter potential lane

def IsBasisFixedPotential (potential : Potential) : Prop :=
  potential lane0 = false ∧
    potential lane1 = false ∧
    potential lane2 = false ∧
    potential lane4 = false ∧
    potential lane8 = false

theorem rebased_potential_is_basis_fixed
    (potential : Potential) (originZero : potential lane0 = false) :
    IsBasisFixedPotential (rebasedPotential potential) := by
  rcases basis_character_on_basis potential with ⟨h1, h2, h4, h8⟩
  simp [IsBasisFixedPotential, rebasedPotential, basis_character_zero,
    originZero, h1, h2, h4, h8]

theorem rebased_potential_preserves_coboundary
    (potential : Potential) :
    unrestrictedCoboundary (rebasedPotential potential) =
      unrestrictedCoboundary potential := by
  funext cell
  simp [unrestrictedCoboundary, rebasedPotential, basis_character_xor,
    Bool.xor_left_comm, Bool.xor_comm]

def gaugeLane (rank : Fin 11) : Lane :=
  match rank.val with
  | 0 => lane3
  | 1 => lane5
  | 2 => lane6
  | 3 => lane7
  | 4 => lane9
  | 5 => lane10
  | 6 => lane11
  | 7 => lane12
  | 8 => lane13
  | 9 => lane14
  | _ => lane15

def basisFixedBits (potential : Potential) : BitVec 11 :=
  BitVec.cons (potential lane15)
    (BitVec.cons (potential lane14)
      (BitVec.cons (potential lane13)
        (BitVec.cons (potential lane12)
          (BitVec.cons (potential lane11)
            (BitVec.cons (potential lane10)
              (BitVec.cons (potential lane9)
                (BitVec.cons (potential lane7)
                  (BitVec.cons (potential lane6)
                    (BitVec.cons (potential lane5)
                      (BitVec.cons (potential lane3) BitVec.nil))))))))))

def basisFixedWord (potential : Potential) : GaugeWord :=
  (basisFixedBits potential).toFin

private theorem basis_fixed_bits_0 (potential : Potential) :
    BitVec.getLsbD (basisFixedBits potential) 0 = potential lane3 := by
  simp [basisFixedBits, BitVec.getElem_cons]

private theorem basis_fixed_bits_1 (potential : Potential) :
    BitVec.getLsbD (basisFixedBits potential) 1 = potential lane5 := by
  simp [basisFixedBits, BitVec.getElem_cons]

private theorem basis_fixed_bits_2 (potential : Potential) :
    BitVec.getLsbD (basisFixedBits potential) 2 = potential lane6 := by
  simp [basisFixedBits, BitVec.getElem_cons]

private theorem basis_fixed_bits_3 (potential : Potential) :
    BitVec.getLsbD (basisFixedBits potential) 3 = potential lane7 := by
  simp [basisFixedBits, BitVec.getElem_cons]

private theorem basis_fixed_bits_4 (potential : Potential) :
    BitVec.getLsbD (basisFixedBits potential) 4 = potential lane9 := by
  simp [basisFixedBits, BitVec.getElem_cons]

private theorem basis_fixed_bits_5 (potential : Potential) :
    BitVec.getLsbD (basisFixedBits potential) 5 = potential lane10 := by
  simp [basisFixedBits, BitVec.getElem_cons]

private theorem basis_fixed_bits_6 (potential : Potential) :
    BitVec.getLsbD (basisFixedBits potential) 6 = potential lane11 := by
  simp [basisFixedBits, BitVec.getElem_cons]

private theorem basis_fixed_bits_7 (potential : Potential) :
    BitVec.getLsbD (basisFixedBits potential) 7 = potential lane12 := by
  simp [basisFixedBits, BitVec.getElem_cons]

private theorem basis_fixed_bits_8 (potential : Potential) :
    BitVec.getLsbD (basisFixedBits potential) 8 = potential lane13 := by
  simp [basisFixedBits, BitVec.getElem_cons]

private theorem basis_fixed_bits_9 (potential : Potential) :
    BitVec.getLsbD (basisFixedBits potential) 9 = potential lane14 := by
  simp [basisFixedBits, BitVec.getElem_cons]

private theorem basis_fixed_bits_10 (potential : Potential) :
    BitVec.getLsbD (basisFixedBits potential) 10 = potential lane15 := by
  simp [basisFixedBits, BitVec.getElem_cons]

theorem basis_fixed_word_testBit (potential : Potential) (rank : Fin 11) :
    (basisFixedWord potential).val.testBit rank.val =
      potential (gaugeLane rank) := by
  change BitVec.getLsbD (basisFixedBits potential) rank.val =
    potential (gaugeLane rank)
  have allRanks : ∀ index : Fin 11,
      BitVec.getLsbD (basisFixedBits potential) index.val =
        potential (gaugeLane index) := by
    simp only [Fin.forall_fin_succ, Fin.forall_fin_zero, and_true]
    exact ⟨basis_fixed_bits_0 potential,
      basis_fixed_bits_1 potential,
      basis_fixed_bits_2 potential,
      basis_fixed_bits_3 potential,
      basis_fixed_bits_4 potential,
      basis_fixed_bits_5 potential,
      basis_fixed_bits_6 potential,
      basis_fixed_bits_7 potential,
      basis_fixed_bits_8 potential,
      basis_fixed_bits_9 potential,
      basis_fixed_bits_10 potential⟩
  exact allRanks rank

theorem basis_fixed_word_roundtrip
    (potential : Potential) (basisFixed : IsBasisFixedPotential potential) :
    gaugeWordPotential (basisFixedWord potential) = potential := by
  rcases basisFixed with ⟨h0, h1, h2, h4, h8⟩
  funext lane
  rcases lane with ⟨value, valueLt⟩
  have cases16 : value = 0 ∨ value = 1 ∨ value = 2 ∨ value = 3 ∨
      value = 4 ∨ value = 5 ∨ value = 6 ∨ value = 7 ∨
      value = 8 ∨ value = 9 ∨ value = 10 ∨ value = 11 ∨
      value = 12 ∨ value = 13 ∨ value = 14 ∨ value = 15 := by
    omega
  rcases cases16 with h | h | h | h | h | h | h | h |
      h | h | h | h | h | h | h | h <;> subst value
  · simpa only [gaugeWordPotential, gaugeValue, gauge_rank_0, lane0]
      using h0.symm
  · simpa only [gaugeWordPotential, gaugeValue, gauge_rank_1, lane1]
      using h1.symm
  · simpa only [gaugeWordPotential, gaugeValue, gauge_rank_2, lane2]
      using h2.symm
  · simpa only [gaugeWordPotential, gaugeValue, gauge_rank_3, gaugeLane, lane3]
      using basis_fixed_word_testBit potential (⟨0, by decide⟩ : Fin 11)
  · simpa only [gaugeWordPotential, gaugeValue, gauge_rank_4, lane4]
      using h4.symm
  · simpa only [gaugeWordPotential, gaugeValue, gauge_rank_5, gaugeLane, lane5]
      using basis_fixed_word_testBit potential (⟨1, by decide⟩ : Fin 11)
  · simpa only [gaugeWordPotential, gaugeValue, gauge_rank_6, gaugeLane, lane6]
      using basis_fixed_word_testBit potential (⟨2, by decide⟩ : Fin 11)
  · simpa only [gaugeWordPotential, gaugeValue, gauge_rank_7, gaugeLane, lane7]
      using basis_fixed_word_testBit potential (⟨3, by decide⟩ : Fin 11)
  · simpa only [gaugeWordPotential, gaugeValue, gauge_rank_8, lane8]
      using h8.symm
  · simpa only [gaugeWordPotential, gaugeValue, gauge_rank_9, gaugeLane, lane9]
      using basis_fixed_word_testBit potential (⟨4, by decide⟩ : Fin 11)
  · simpa only [gaugeWordPotential, gaugeValue, gauge_rank_10, gaugeLane, lane10]
      using basis_fixed_word_testBit potential (⟨5, by decide⟩ : Fin 11)
  · simpa only [gaugeWordPotential, gaugeValue, gauge_rank_11, gaugeLane, lane11]
      using basis_fixed_word_testBit potential (⟨6, by decide⟩ : Fin 11)
  · simpa only [gaugeWordPotential, gaugeValue, gauge_rank_12, gaugeLane, lane12]
      using basis_fixed_word_testBit potential (⟨7, by decide⟩ : Fin 11)
  · simpa only [gaugeWordPotential, gaugeValue, gauge_rank_13, gaugeLane, lane13]
      using basis_fixed_word_testBit potential (⟨8, by decide⟩ : Fin 11)
  · simpa only [gaugeWordPotential, gaugeValue, gauge_rank_14, gaugeLane, lane14]
      using basis_fixed_word_testBit potential (⟨9, by decide⟩ : Fin 11)
  · simpa only [gaugeWordPotential, gaugeValue, gauge_rank_15, gaugeLane, lane15]
      using basis_fixed_word_testBit potential (⟨10, by decide⟩ : Fin 11)

theorem pulled_gauge_potential_origin_zero
    (linear : XorLaneEquiv) (word : GaugeWord) :
    pullPotential linear (gaugeWordPotential word) lane0 = false := by
  have mappedZero : linear.toFun lane0 = lane0 := by
    have lane0_eq_zero : lane0 = 0 := by
      apply Fin.ext
      rfl
    rw [lane0_eq_zero]
    exact linear.mapZero
  rw [pullPotential, mappedZero]
  simp only [gaugeWordPotential, lane0, gaugeValue, gauge_rank_0]

def rebasedGaugeWord (linear : XorLaneEquiv) (word : GaugeWord) : GaugeWord :=
  basisFixedWord
    (rebasedPotential (pullPotential linear (gaugeWordPotential word)))

theorem rebased_gauge_word_potential
    (linear : XorLaneEquiv) (word : GaugeWord) :
    gaugeWordPotential (rebasedGaugeWord linear word) =
      rebasedPotential (pullPotential linear (gaugeWordPotential word)) := by
  apply basis_fixed_word_roundtrip
  exact rebased_potential_is_basis_fixed _
    (pulled_gauge_potential_origin_zero linear word)

theorem pulled_gauge_coboundary_reencoded
    (linear : XorLaneEquiv) (word : GaugeWord) :
    unrestrictedCoboundary
        (pullPotential linear (gaugeWordPotential word)) =
      gaugeCoboundary (rebasedGaugeWord linear word) := by
  rw [← rebased_potential_preserves_coboundary,
    ← rebased_gauge_word_potential,
    unrestricted_coboundary_of_gauge_word]

theorem raw_action_transports_basis_fixed_gauge
    (action : LinearSwapAction) (word : GaugeWord) (table : SignTable) :
    rawAct action (gaugeAct word table) =
      gaugeAct (rebasedGaugeWord action.linear word) (rawAct action table) := by
  rw [raw_action_transports_basis_fixed_gauge_to_unrestricted_potential]
  funext cell
  simp [potentialGaugeAct, gaugeAct,
    pulled_gauge_coboundary_reencoded]

structure BasisFixedRebaseBoundary where
  parentLinearSwapDescentProved : Bool
  basisCharacterLinearityProved : Bool
  basisCharacterCoboundaryZeroProved : Bool
  rebasedPotentialBasisFixedProved : Bool
  rebasedPotentialPreservesCoboundaryProved : Bool
  basisFixedWordEncoderRoundtripProved : Bool
  basisFixedGaugeRebaseAfterLinearMapProved : Bool
  concreteMatrixCodeToXorEquivBridgeProved : Bool
  concreteGL4ActionInstantiated : Bool
  outer40320ViewMinimumProved : Bool
  concreteCanonicalEqualityIffFullDeclaredOrbitProved : Bool
  formalTarget03Closed : Bool
  formalParityClosed : Bool
  claimReady : Bool
deriving Repr, BEq, DecidableEq

def basisFixedRebaseBoundary : BasisFixedRebaseBoundary :=
  { parentLinearSwapDescentProved := true
  , basisCharacterLinearityProved := true
  , basisCharacterCoboundaryZeroProved := true
  , rebasedPotentialBasisFixedProved := true
  , rebasedPotentialPreservesCoboundaryProved := true
  , basisFixedWordEncoderRoundtripProved := true
  , basisFixedGaugeRebaseAfterLinearMapProved := true
  , concreteMatrixCodeToXorEquivBridgeProved := false
  , concreteGL4ActionInstantiated := false
  , outer40320ViewMinimumProved := false
  , concreteCanonicalEqualityIffFullDeclaredOrbitProved := false
  , formalTarget03Closed := false
  , formalParityClosed := false
  , claimReady := false }

theorem basis_fixed_rebase_progress_does_not_close_v13_target03 :
    basisFixedRebaseBoundary.parentLinearSwapDescentProved &&
      basisFixedRebaseBoundary.basisCharacterLinearityProved &&
      basisFixedRebaseBoundary.basisCharacterCoboundaryZeroProved &&
      basisFixedRebaseBoundary.rebasedPotentialBasisFixedProved &&
      basisFixedRebaseBoundary.rebasedPotentialPreservesCoboundaryProved &&
      basisFixedRebaseBoundary.basisFixedWordEncoderRoundtripProved &&
      basisFixedRebaseBoundary.basisFixedGaugeRebaseAfterLinearMapProved &&
      !basisFixedRebaseBoundary.concreteMatrixCodeToXorEquivBridgeProved &&
      !basisFixedRebaseBoundary.concreteGL4ActionInstantiated &&
      !basisFixedRebaseBoundary.outer40320ViewMinimumProved &&
      !basisFixedRebaseBoundary.concreteCanonicalEqualityIffFullDeclaredOrbitProved &&
      !basisFixedRebaseBoundary.formalTarget03Closed &&
      !basisFixedRebaseBoundary.formalParityClosed &&
      !basisFixedRebaseBoundary.claimReady := by
  decide

end SounioPireusBasisFixedGaugeRebase
