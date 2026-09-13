/-
  FORMAL_PARITY for the direct gauge section used by the frozen Sounio Pireus
  V13 normalizer.

  Sounio does not enumerate 2048 gauge words for every GL4/input-swap view.
  It decodes one 11-bit word along the ascending XOR tree, applies that word,
  and compares only the resulting 40320 normalized GL4 x swap views. This
  file proves that direct gauge normalization is constant exactly on the
  basis-fixed gauge orbits and returns the unique zero-section-bit
  representative. No lexicographic minimality claim is implied by that name.

  This does not yet instantiate input swap or GL(4,F2), prove the outer
  40320-view fold realizes a lawful minimum, or close Target-03.
-/
import SounioPireusGaugeCoboundaryFaithfulness

namespace SounioPireusGaugeSectionCanonicalization

open SounioPireusFiniteActionCanonicalization
open SounioPireusGaugeCoboundaryAction
open SounioPireusGaugeCoboundaryFaithfulness

def sectionBit (table : SignTable) (rank : Fin 11) : Bool :=
  treeSectionBit (tableAt table) rank

def xorTable (left right : SignTable) : SignTable :=
  fun cell => left cell ^^ right cell

private theorem section_bit_xor_0 (left right : SignTable) :
    sectionBit (xorTable left right) (⟨0, by decide⟩ : Fin 11) =
      (sectionBit left ⟨0, by decide⟩ ^^ sectionBit right ⟨0, by decide⟩) := by
  simp [sectionBit, xorTable, treeSectionBit, tableAt]

private theorem section_bit_xor_1 (left right : SignTable) :
    sectionBit (xorTable left right) (⟨1, by decide⟩ : Fin 11) =
      (sectionBit left ⟨1, by decide⟩ ^^ sectionBit right ⟨1, by decide⟩) := by
  simp [sectionBit, xorTable, treeSectionBit, tableAt]

private theorem section_bit_xor_2 (left right : SignTable) :
    sectionBit (xorTable left right) (⟨2, by decide⟩ : Fin 11) =
      (sectionBit left ⟨2, by decide⟩ ^^ sectionBit right ⟨2, by decide⟩) := by
  simp [sectionBit, xorTable, treeSectionBit, tableAt]

private theorem section_bit_xor_3 (left right : SignTable) :
    sectionBit (xorTable left right) (⟨3, by decide⟩ : Fin 11) =
      (sectionBit left ⟨3, by decide⟩ ^^ sectionBit right ⟨3, by decide⟩) := by
  simp [sectionBit, xorTable, treeSectionBit, tableAt,
    Bool.xor_left_comm, Bool.xor_comm]

private theorem section_bit_xor_4 (left right : SignTable) :
    sectionBit (xorTable left right) (⟨4, by decide⟩ : Fin 11) =
      (sectionBit left ⟨4, by decide⟩ ^^ sectionBit right ⟨4, by decide⟩) := by
  simp [sectionBit, xorTable, treeSectionBit, tableAt]

private theorem section_bit_xor_5 (left right : SignTable) :
    sectionBit (xorTable left right) (⟨5, by decide⟩ : Fin 11) =
      (sectionBit left ⟨5, by decide⟩ ^^ sectionBit right ⟨5, by decide⟩) := by
  simp [sectionBit, xorTable, treeSectionBit, tableAt]

private theorem section_bit_xor_6 (left right : SignTable) :
    sectionBit (xorTable left right) (⟨6, by decide⟩ : Fin 11) =
      (sectionBit left ⟨6, by decide⟩ ^^ sectionBit right ⟨6, by decide⟩) := by
  simp [sectionBit, xorTable, treeSectionBit, tableAt,
    Bool.xor_left_comm, Bool.xor_comm]

private theorem section_bit_xor_7 (left right : SignTable) :
    sectionBit (xorTable left right) (⟨7, by decide⟩ : Fin 11) =
      (sectionBit left ⟨7, by decide⟩ ^^ sectionBit right ⟨7, by decide⟩) := by
  simp [sectionBit, xorTable, treeSectionBit, tableAt]

private theorem section_bit_xor_8 (left right : SignTable) :
    sectionBit (xorTable left right) (⟨8, by decide⟩ : Fin 11) =
      (sectionBit left ⟨8, by decide⟩ ^^ sectionBit right ⟨8, by decide⟩) := by
  simp [sectionBit, xorTable, treeSectionBit, tableAt,
    Bool.xor_left_comm, Bool.xor_comm]

private theorem section_bit_xor_9 (left right : SignTable) :
    sectionBit (xorTable left right) (⟨9, by decide⟩ : Fin 11) =
      (sectionBit left ⟨9, by decide⟩ ^^ sectionBit right ⟨9, by decide⟩) := by
  simp [sectionBit, xorTable, treeSectionBit, tableAt,
    Bool.xor_left_comm, Bool.xor_comm]

private theorem section_bit_xor_10 (left right : SignTable) :
    sectionBit (xorTable left right) (⟨10, by decide⟩ : Fin 11) =
      (sectionBit left ⟨10, by decide⟩ ^^ sectionBit right ⟨10, by decide⟩) := by
  simp [sectionBit, xorTable, treeSectionBit, tableAt,
    Bool.xor_left_comm, Bool.xor_comm]

theorem section_bit_xor (left right : SignTable) (rank : Fin 11) :
    sectionBit (xorTable left right) rank =
      (sectionBit left rank ^^ sectionBit right rank) := by
  have allRanks : ∀ index : Fin 11,
      sectionBit (xorTable left right) index =
        (sectionBit left index ^^ sectionBit right index) := by
    simp only [Fin.forall_fin_succ, Fin.forall_fin_zero, and_true]
    exact ⟨section_bit_xor_0 left right,
      section_bit_xor_1 left right,
      section_bit_xor_2 left right,
      section_bit_xor_3 left right,
      section_bit_xor_4 left right,
      section_bit_xor_5 left right,
      section_bit_xor_6 left right,
      section_bit_xor_7 left right,
      section_bit_xor_8 left right,
      section_bit_xor_9 left right,
      section_bit_xor_10 left right⟩
  exact allRanks rank

theorem section_bit_gauge_action
    (word : GaugeWord) (table : SignTable) (rank : Fin 11) :
    sectionBit (gaugeAct word table) rank =
      (sectionBit table rank ^^ word.val.testBit rank.val) := by
  have actionAsXor : gaugeAct word table = xorTable table (gaugeCoboundary word) := rfl
  rw [actionAsXor, section_bit_xor]
  exact congrArg (fun bit => sectionBit table rank ^^ bit)
    (tree_section_bit_roundtrip word rank)

def directSectionBits (table : SignTable) : BitVec 11 :=
  BitVec.cons (sectionBit table ⟨10, by decide⟩)
    (BitVec.cons (sectionBit table ⟨9, by decide⟩)
      (BitVec.cons (sectionBit table ⟨8, by decide⟩)
        (BitVec.cons (sectionBit table ⟨7, by decide⟩)
          (BitVec.cons (sectionBit table ⟨6, by decide⟩)
            (BitVec.cons (sectionBit table ⟨5, by decide⟩)
              (BitVec.cons (sectionBit table ⟨4, by decide⟩)
                (BitVec.cons (sectionBit table ⟨3, by decide⟩)
                  (BitVec.cons (sectionBit table ⟨2, by decide⟩)
                    (BitVec.cons (sectionBit table ⟨1, by decide⟩)
                      (BitVec.cons (sectionBit table ⟨0, by decide⟩)
                        BitVec.nil))))))))))

def directSectionWord (table : SignTable) : GaugeWord :=
  (directSectionBits table).toFin

private theorem direct_section_bits_0 (table : SignTable) :
    BitVec.getLsbD (directSectionBits table) 0 = sectionBit table ⟨0, by decide⟩ := by
  simp [directSectionBits, BitVec.getElem_cons]

private theorem direct_section_bits_1 (table : SignTable) :
    BitVec.getLsbD (directSectionBits table) 1 = sectionBit table ⟨1, by decide⟩ := by
  simp [directSectionBits, BitVec.getElem_cons]

private theorem direct_section_bits_2 (table : SignTable) :
    BitVec.getLsbD (directSectionBits table) 2 = sectionBit table ⟨2, by decide⟩ := by
  simp [directSectionBits, BitVec.getElem_cons]

private theorem direct_section_bits_3 (table : SignTable) :
    BitVec.getLsbD (directSectionBits table) 3 = sectionBit table ⟨3, by decide⟩ := by
  simp [directSectionBits, BitVec.getElem_cons]

private theorem direct_section_bits_4 (table : SignTable) :
    BitVec.getLsbD (directSectionBits table) 4 = sectionBit table ⟨4, by decide⟩ := by
  simp [directSectionBits, BitVec.getElem_cons]

private theorem direct_section_bits_5 (table : SignTable) :
    BitVec.getLsbD (directSectionBits table) 5 = sectionBit table ⟨5, by decide⟩ := by
  simp [directSectionBits, BitVec.getElem_cons]

private theorem direct_section_bits_6 (table : SignTable) :
    BitVec.getLsbD (directSectionBits table) 6 = sectionBit table ⟨6, by decide⟩ := by
  simp [directSectionBits, BitVec.getElem_cons]

private theorem direct_section_bits_7 (table : SignTable) :
    BitVec.getLsbD (directSectionBits table) 7 = sectionBit table ⟨7, by decide⟩ := by
  simp [directSectionBits, BitVec.getElem_cons]

private theorem direct_section_bits_8 (table : SignTable) :
    BitVec.getLsbD (directSectionBits table) 8 = sectionBit table ⟨8, by decide⟩ := by
  simp [directSectionBits, BitVec.getElem_cons]

private theorem direct_section_bits_9 (table : SignTable) :
    BitVec.getLsbD (directSectionBits table) 9 = sectionBit table ⟨9, by decide⟩ := by
  simp [directSectionBits, BitVec.getElem_cons]

private theorem direct_section_bits_10 (table : SignTable) :
    BitVec.getLsbD (directSectionBits table) 10 = sectionBit table ⟨10, by decide⟩ := by
  simp [directSectionBits, BitVec.getElem_cons]

theorem direct_section_word_testBit (table : SignTable) (rank : Fin 11) :
    (directSectionWord table).val.testBit rank.val = sectionBit table rank := by
  change BitVec.getLsbD (directSectionBits table) rank.val = sectionBit table rank
  have allRanks : ∀ index : Fin 11,
      BitVec.getLsbD (directSectionBits table) index.val = sectionBit table index := by
    simp only [Fin.forall_fin_succ, Fin.forall_fin_zero, and_true]
    exact ⟨direct_section_bits_0 table,
      direct_section_bits_1 table,
      direct_section_bits_2 table,
      direct_section_bits_3 table,
      direct_section_bits_4 table,
      direct_section_bits_5 table,
      direct_section_bits_6 table,
      direct_section_bits_7 table,
      direct_section_bits_8 table,
      direct_section_bits_9 table,
      direct_section_bits_10 table⟩
  exact allRanks rank

theorem direct_section_word_gauge_action
    (word : GaugeWord) (table : SignTable) :
    directSectionWord (gaugeAct word table) =
      composeGauge (directSectionWord table) word := by
  apply gauge_word_eq_of_section_bits
  intro rank
  rw [direct_section_word_testBit, section_bit_gauge_action]
  simp [composeGauge, Fin.xor_val_of_two_pow, Nat.testBit_xor,
    direct_section_word_testBit]

def normalizeGauge (table : SignTable) : SignTable :=
  gaugeAct (directSectionWord table) table

def IsDirectGaugeSection (table : SignTable) : Prop :=
  ∀ rank : Fin 11, sectionBit table rank = false

theorem normalize_gauge_has_zero_section_bits (table : SignTable) :
    IsDirectGaugeSection (normalizeGauge table) := by
  intro rank
  simp [normalizeGauge, section_bit_gauge_action,
    direct_section_word_testBit]

theorem direct_section_word_zero_of_section
    {table : SignTable} (inSection : IsDirectGaugeSection table) :
    directSectionWord table = zeroGauge := by
  apply gauge_word_eq_of_section_bits
  intro rank
  rw [direct_section_word_testBit, inSection rank]
  simp [zeroGauge]

theorem normalize_gauge_fixed_of_section
    {table : SignTable} (inSection : IsDirectGaugeSection table) :
    normalizeGauge table = table := by
  rw [normalizeGauge, direct_section_word_zero_of_section inSection]
  exact gauge_action_identity table

theorem normalize_gauge_idempotent (table : SignTable) :
    normalizeGauge (normalizeGauge table) = normalizeGauge table := by
  exact normalize_gauge_fixed_of_section (normalize_gauge_has_zero_section_bits table)

theorem normalize_gauge_invariant
    (word : GaugeWord) (table : SignTable) :
    normalizeGauge (gaugeAct word table) = normalizeGauge table := by
  rw [normalizeGauge, direct_section_word_gauge_action]
  rw [gauge_action_compose]
  have cancelled : gaugeAct word (gaugeAct word table) = table := by
    simpa [inverseGauge] using gauge_action_inverse word table
  rw [cancelled]
  rfl

def SameGaugeOrbit (left right : SignTable) : Prop :=
  ∃ word : GaugeWord, gaugeAct word left = right

theorem same_gauge_orbit_refl (table : SignTable) :
    SameGaugeOrbit table table := by
  exact ⟨zeroGauge, gauge_action_identity table⟩

theorem same_gauge_orbit_symm
    {left right : SignTable} (sameOrbit : SameGaugeOrbit left right) :
    SameGaugeOrbit right left := by
  rcases sameOrbit with ⟨word, acts⟩
  refine ⟨inverseGauge word, ?_⟩
  rw [← acts]
  exact gauge_action_inverse word left

theorem same_gauge_orbit_trans
    {left middle right : SignTable}
    (leftMiddle : SameGaugeOrbit left middle)
    (middleRight : SameGaugeOrbit middle right) :
    SameGaugeOrbit left right := by
  rcases leftMiddle with ⟨leftWord, leftActs⟩
  rcases middleRight with ⟨rightWord, rightActs⟩
  refine ⟨composeGauge rightWord leftWord, ?_⟩
  calc
    gaugeAct (composeGauge rightWord leftWord) left =
        gaugeAct rightWord (gaugeAct leftWord left) :=
      gauge_action_compose rightWord leftWord left
    _ = gaugeAct rightWord middle := congrArg (gaugeAct rightWord) leftActs
    _ = right := rightActs

theorem normalize_gauge_equal_of_same_orbit
    {left right : SignTable} (sameOrbit : SameGaugeOrbit left right) :
    normalizeGauge left = normalizeGauge right := by
  rcases sameOrbit with ⟨word, acts⟩
  calc
    normalizeGauge left = normalizeGauge (gaugeAct word left) :=
      (normalize_gauge_invariant word left).symm
    _ = normalizeGauge right := congrArg normalizeGauge acts

theorem same_orbit_of_normalize_gauge_equal
    {left right : SignTable}
    (normalEqual : normalizeGauge left = normalizeGauge right) :
    SameGaugeOrbit left right := by
  let leftWord := directSectionWord left
  let rightWord := directSectionWord right
  refine ⟨composeGauge rightWord leftWord, ?_⟩
  change gaugeAct (composeGauge rightWord leftWord) left = right
  calc
    gaugeAct (composeGauge rightWord leftWord) left =
        gaugeAct rightWord (gaugeAct leftWord left) :=
      gauge_action_compose rightWord leftWord left
    _ = gaugeAct rightWord (gaugeAct rightWord right) := by
      exact congrArg (gaugeAct rightWord) normalEqual
    _ = right := by
      simpa [inverseGauge] using gauge_action_inverse rightWord right

theorem normalize_gauge_equal_iff_same_orbit (left right : SignTable) :
    normalizeGauge left = normalizeGauge right ↔ SameGaugeOrbit left right := by
  exact ⟨same_orbit_of_normalize_gauge_equal, normalize_gauge_equal_of_same_orbit⟩

theorem direct_section_is_unique_orbit_representative
    {left right : SignTable}
    (leftSection : IsDirectGaugeSection left)
    (rightSection : IsDirectGaugeSection right)
    (sameOrbit : SameGaugeOrbit left right) :
    left = right := by
  calc
    left = normalizeGauge left := (normalize_gauge_fixed_of_section leftSection).symm
    _ = normalizeGauge right := normalize_gauge_equal_of_same_orbit sameOrbit
    _ = right := normalize_gauge_fixed_of_section rightSection

structure GaugeSectionBoundary where
  parentGaugeFaithfulnessProved : Bool
  directSectionWordConstructed : Bool
  directSectionEquivarianceProved : Bool
  normalizedSectionBitsZeroProved : Bool
  normalizationIdempotentProved : Bool
  normalizationInvariantOnGaugeOrbitProved : Bool
  sameGaugeOrbitEquivalenceProved : Bool
  normalizationEqualityIffGaugeOrbitProved : Bool
  uniqueDirectSectionRepresentativeProved : Bool
  treeSectionLexicographicMinimumProved : Bool
  concreteInputSwapActionInstantiated : Bool
  concreteGL4ActionInstantiated : Bool
  outer40320ViewMinimumProved : Bool
  concreteCanonicalEqualityIffDeclaredOrbitProved : Bool
  formalTarget03Closed : Bool
  formalParityClosed : Bool
  claimReady : Bool
deriving Repr, BEq, DecidableEq

def gaugeSectionBoundary : GaugeSectionBoundary :=
  { parentGaugeFaithfulnessProved := true
  , directSectionWordConstructed := true
  , directSectionEquivarianceProved := true
  , normalizedSectionBitsZeroProved := true
  , normalizationIdempotentProved := true
  , normalizationInvariantOnGaugeOrbitProved := true
  , sameGaugeOrbitEquivalenceProved := true
  , normalizationEqualityIffGaugeOrbitProved := true
  , uniqueDirectSectionRepresentativeProved := true
  , treeSectionLexicographicMinimumProved := false
  , concreteInputSwapActionInstantiated := false
  , concreteGL4ActionInstantiated := false
  , outer40320ViewMinimumProved := false
  , concreteCanonicalEqualityIffDeclaredOrbitProved := false
  , formalTarget03Closed := false
  , formalParityClosed := false
  , claimReady := false }

theorem gauge_section_progress_does_not_close_v13_target03 :
    gaugeSectionBoundary.parentGaugeFaithfulnessProved &&
      gaugeSectionBoundary.directSectionWordConstructed &&
      gaugeSectionBoundary.directSectionEquivarianceProved &&
      gaugeSectionBoundary.normalizedSectionBitsZeroProved &&
      gaugeSectionBoundary.normalizationIdempotentProved &&
      gaugeSectionBoundary.normalizationInvariantOnGaugeOrbitProved &&
      gaugeSectionBoundary.sameGaugeOrbitEquivalenceProved &&
      gaugeSectionBoundary.normalizationEqualityIffGaugeOrbitProved &&
      gaugeSectionBoundary.uniqueDirectSectionRepresentativeProved &&
      !gaugeSectionBoundary.treeSectionLexicographicMinimumProved &&
      !gaugeSectionBoundary.concreteInputSwapActionInstantiated &&
      !gaugeSectionBoundary.concreteGL4ActionInstantiated &&
      !gaugeSectionBoundary.outer40320ViewMinimumProved &&
      !gaugeSectionBoundary.concreteCanonicalEqualityIffDeclaredOrbitProved &&
      !gaugeSectionBoundary.formalTarget03Closed &&
      !gaugeSectionBoundary.formalParityClosed &&
      !gaugeSectionBoundary.claimReady := by
  decide

end SounioPireusGaugeSectionCanonicalization
