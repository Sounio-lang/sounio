/-
  FORMAL_PARITY for the basis-fixed gauge decoder already executed by the
  frozen Sounio Pireus V13 semantic artifact.

  The eleven section bits are recovered along the ascending XOR tree. For a
  coboundary table, each pivot cancels the already recovered parent value and
  returns the corresponding bit of the normalized gauge word. This gives a
  symbolic left inverse, so distinct gauge words induce distinct coboundaries
  and distinct translations on every Boolean sign table.

  This does not prove tree-section lexicographic minimality, instantiate the
  input-swap or GL(4,F2) actions, or close Target-03.
-/
import SounioPireusGaugeCoboundaryAction

namespace SounioPireusGaugeCoboundaryFaithfulness

open SounioPireusGaugeCoboundaryAction

def tableAt (table : SignTable) (left right : Nat) : Bool :=
  if leftBound : left < 16 then
    if rightBound : right < 16 then
      table (⟨left, leftBound⟩, ⟨right, rightBound⟩)
    else false
  else false

def treeSectionBit (table : Nat -> Nat -> Bool) (rank : Fin 11) : Bool :=
  match rank.val with
  | 0 => table 1 2
  | 1 => table 1 4
  | 2 => table 2 4
  | 3 => table 3 4 ^^ table 1 2
  | 4 => table 1 8
  | 5 => table 2 8
  | 6 => table 3 8 ^^ table 1 2
  | 7 => table 4 8
  | 8 => table 5 8 ^^ table 1 4
  | 9 => table 6 8 ^^ table 2 4
  | 10 => table 7 8 ^^ table 3 4 ^^ table 1 2
  | _ => false

@[simp] private theorem gauge_value_1 (word : GaugeWord) : gaugeValue word 1 = false := rfl
@[simp] private theorem gauge_value_2 (word : GaugeWord) : gaugeValue word 2 = false := rfl
@[simp] private theorem gauge_value_3 (word : GaugeWord) :
    gaugeValue word 3 = word.val.testBit 0 := rfl
@[simp] private theorem gauge_value_4 (word : GaugeWord) : gaugeValue word 4 = false := rfl
@[simp] private theorem gauge_value_5 (word : GaugeWord) :
    gaugeValue word 5 = word.val.testBit 1 := rfl
@[simp] private theorem gauge_value_6 (word : GaugeWord) :
    gaugeValue word 6 = word.val.testBit 2 := rfl
@[simp] private theorem gauge_value_7 (word : GaugeWord) :
    gaugeValue word 7 = word.val.testBit 3 := rfl
@[simp] private theorem gauge_value_8 (word : GaugeWord) : gaugeValue word 8 = false := rfl
@[simp] private theorem gauge_value_9 (word : GaugeWord) :
    gaugeValue word 9 = word.val.testBit 4 := rfl
@[simp] private theorem gauge_value_10 (word : GaugeWord) :
    gaugeValue word 10 = word.val.testBit 5 := rfl
@[simp] private theorem gauge_value_11 (word : GaugeWord) :
    gaugeValue word 11 = word.val.testBit 6 := rfl
@[simp] private theorem gauge_value_12 (word : GaugeWord) :
    gaugeValue word 12 = word.val.testBit 7 := rfl
@[simp] private theorem gauge_value_13 (word : GaugeWord) :
    gaugeValue word 13 = word.val.testBit 8 := rfl
@[simp] private theorem gauge_value_14 (word : GaugeWord) :
    gaugeValue word 14 = word.val.testBit 9 := rfl
@[simp] private theorem gauge_value_15 (word : GaugeWord) :
    gaugeValue word 15 = word.val.testBit 10 := rfl

private theorem tree_section_bit_roundtrip_0 (word : GaugeWord) :
    treeSectionBit (tableAt (gaugeCoboundary word)) (⟨0, by decide⟩ : Fin 11) =
      word.val.testBit 0 := by
  simp [treeSectionBit, tableAt, gaugeCoboundary]

private theorem tree_section_bit_roundtrip_1 (word : GaugeWord) :
    treeSectionBit (tableAt (gaugeCoboundary word)) (⟨1, by decide⟩ : Fin 11) =
      word.val.testBit 1 := by
  simp [treeSectionBit, tableAt, gaugeCoboundary]

private theorem tree_section_bit_roundtrip_2 (word : GaugeWord) :
    treeSectionBit (tableAt (gaugeCoboundary word)) (⟨2, by decide⟩ : Fin 11) =
      word.val.testBit 2 := by
  simp [treeSectionBit, tableAt, gaugeCoboundary]

private theorem tree_section_bit_roundtrip_3 (word : GaugeWord) :
    treeSectionBit (tableAt (gaugeCoboundary word)) (⟨3, by decide⟩ : Fin 11) =
      word.val.testBit 3 := by
  simp [treeSectionBit, tableAt, gaugeCoboundary,
    Bool.xor_comm]

private theorem tree_section_bit_roundtrip_4 (word : GaugeWord) :
    treeSectionBit (tableAt (gaugeCoboundary word)) (⟨4, by decide⟩ : Fin 11) =
      word.val.testBit 4 := by
  simp [treeSectionBit, tableAt, gaugeCoboundary]

private theorem tree_section_bit_roundtrip_5 (word : GaugeWord) :
    treeSectionBit (tableAt (gaugeCoboundary word)) (⟨5, by decide⟩ : Fin 11) =
      word.val.testBit 5 := by
  simp [treeSectionBit, tableAt, gaugeCoboundary]

private theorem tree_section_bit_roundtrip_6 (word : GaugeWord) :
    treeSectionBit (tableAt (gaugeCoboundary word)) (⟨6, by decide⟩ : Fin 11) =
      word.val.testBit 6 := by
  simp [treeSectionBit, tableAt, gaugeCoboundary,
    Bool.xor_comm]

private theorem tree_section_bit_roundtrip_7 (word : GaugeWord) :
    treeSectionBit (tableAt (gaugeCoboundary word)) (⟨7, by decide⟩ : Fin 11) =
      word.val.testBit 7 := by
  simp [treeSectionBit, tableAt, gaugeCoboundary]

private theorem tree_section_bit_roundtrip_8 (word : GaugeWord) :
    treeSectionBit (tableAt (gaugeCoboundary word)) (⟨8, by decide⟩ : Fin 11) =
      word.val.testBit 8 := by
  simp [treeSectionBit, tableAt, gaugeCoboundary,
    Bool.xor_comm]

private theorem tree_section_bit_roundtrip_9 (word : GaugeWord) :
    treeSectionBit (tableAt (gaugeCoboundary word)) (⟨9, by decide⟩ : Fin 11) =
      word.val.testBit 9 := by
  simp [treeSectionBit, tableAt, gaugeCoboundary,
    Bool.xor_comm]

private theorem tree_section_bit_roundtrip_10 (word : GaugeWord) :
    treeSectionBit (tableAt (gaugeCoboundary word)) (⟨10, by decide⟩ : Fin 11) =
      word.val.testBit 10 := by
  simp [treeSectionBit, tableAt, gaugeCoboundary,
    Bool.xor_comm]

theorem tree_section_bit_roundtrip (word : GaugeWord) :
    ∀ rank : Fin 11,
      treeSectionBit (tableAt (gaugeCoboundary word)) rank =
        word.val.testBit rank.val := by
  simp only [Fin.forall_fin_succ, Fin.forall_fin_zero, and_true]
  exact ⟨tree_section_bit_roundtrip_0 word,
    tree_section_bit_roundtrip_1 word,
    tree_section_bit_roundtrip_2 word,
    tree_section_bit_roundtrip_3 word,
    tree_section_bit_roundtrip_4 word,
    tree_section_bit_roundtrip_5 word,
    tree_section_bit_roundtrip_6 word,
    tree_section_bit_roundtrip_7 word,
    tree_section_bit_roundtrip_8 word,
    tree_section_bit_roundtrip_9 word,
    tree_section_bit_roundtrip_10 word⟩

theorem gauge_word_eq_of_section_bits
    (left right : GaugeWord)
    (bitsEqual : ∀ rank : Fin 11,
      left.val.testBit rank.val = right.val.testBit rank.val) :
    left = right := by
  apply Fin.ext
  apply Nat.eq_of_testBit_eq
  intro index
  by_cases bounded : index < 11
  · exact bitsEqual ⟨index, bounded⟩
  · have exponentLe : 11 ≤ index := by omega
    have powersLe : 2 ^ 11 ≤ 2 ^ index :=
      Nat.pow_le_pow_right Nat.zero_lt_two exponentLe
    have leftCarrierBound : left.val < 2 ^ 11 := by simpa [gaugeBits] using left.isLt
    have rightCarrierBound : right.val < 2 ^ 11 := by simpa [gaugeBits] using right.isLt
    have leftBound : left.val < 2 ^ index := Nat.lt_of_lt_of_le leftCarrierBound powersLe
    have rightBound : right.val < 2 ^ index := Nat.lt_of_lt_of_le rightCarrierBound powersLe
    rw [Nat.testBit_lt_two_pow leftBound, Nat.testBit_lt_two_pow rightBound]

theorem gaugeCoboundary_injective : Function.Injective gaugeCoboundary := by
  intro left right tablesEqual
  apply gauge_word_eq_of_section_bits left right
  intro rank
  calc
    left.val.testBit rank.val =
        treeSectionBit (tableAt (gaugeCoboundary left)) rank :=
      (tree_section_bit_roundtrip left rank).symm
    _ = treeSectionBit (tableAt (gaugeCoboundary right)) rank := by rw [tablesEqual]
    _ = right.val.testBit rank.val := tree_section_bit_roundtrip right rank

theorem distinct_gauge_words_induce_distinct_coboundaries
    (left right : GaugeWord) (different : left ≠ right) :
    gaugeCoboundary left ≠ gaugeCoboundary right := by
  exact fun equal => different (gaugeCoboundary_injective equal)

theorem gaugeAct_injective_in_word (table : SignTable) :
    Function.Injective fun word => gaugeAct word table := by
  intro left right actionsEqual
  apply gaugeCoboundary_injective
  funext cell
  have cellEqual := congrFun actionsEqual cell
  cases table cell <;> simpa [gaugeAct] using cellEqual

theorem distinct_gauge_words_act_distinctly_on_every_table
    (left right : GaugeWord) (different : left ≠ right) (table : SignTable) :
    gaugeAct left table ≠ gaugeAct right table := by
  exact fun equal => different (gaugeAct_injective_in_word table equal)

theorem gauge_action_free_on_every_sign_table
    (word : GaugeWord) (table : SignTable)
    (fixed : gaugeAct word table = table) :
    word = zeroGauge := by
  apply gaugeAct_injective_in_word table
  calc
    gaugeAct word table = table := fixed
    _ = gaugeAct zeroGauge table := (gauge_action_identity table).symm

structure GaugeFaithfulnessBoundary where
  parentGaugeActionInstantiated : Bool
  triangularSectionRoundtripProved : Bool
  coboundaryMapInjectiveProved : Bool
  gaugeActionFaithfulProved : Bool
  gaugeActionFreeProved : Bool
  treeSectionEqualsGaugeOrbitMinimumProved : Bool
  concreteInputSwapActionInstantiated : Bool
  concreteGL4ActionInstantiated : Bool
  concreteExecutedNormalizerEqualsAbstractMinimumProved : Bool
  concreteCanonicalEqualityIffDeclaredOrbitProved : Bool
  formalTarget03Closed : Bool
  formalParityClosed : Bool
  claimReady : Bool
deriving Repr, BEq, DecidableEq

def gaugeFaithfulnessBoundary : GaugeFaithfulnessBoundary :=
  { parentGaugeActionInstantiated := true
  , triangularSectionRoundtripProved := true
  , coboundaryMapInjectiveProved := true
  , gaugeActionFaithfulProved := true
  , gaugeActionFreeProved := true
  , treeSectionEqualsGaugeOrbitMinimumProved := false
  , concreteInputSwapActionInstantiated := false
  , concreteGL4ActionInstantiated := false
  , concreteExecutedNormalizerEqualsAbstractMinimumProved := false
  , concreteCanonicalEqualityIffDeclaredOrbitProved := false
  , formalTarget03Closed := false
  , formalParityClosed := false
  , claimReady := false }

theorem gauge_faithfulness_does_not_close_v13_target03 :
    gaugeFaithfulnessBoundary.parentGaugeActionInstantiated &&
      gaugeFaithfulnessBoundary.triangularSectionRoundtripProved &&
      gaugeFaithfulnessBoundary.coboundaryMapInjectiveProved &&
      gaugeFaithfulnessBoundary.gaugeActionFaithfulProved &&
      gaugeFaithfulnessBoundary.gaugeActionFreeProved &&
      !gaugeFaithfulnessBoundary.treeSectionEqualsGaugeOrbitMinimumProved &&
      !gaugeFaithfulnessBoundary.concreteInputSwapActionInstantiated &&
      !gaugeFaithfulnessBoundary.concreteGL4ActionInstantiated &&
      !gaugeFaithfulnessBoundary.concreteExecutedNormalizerEqualsAbstractMinimumProved &&
      !gaugeFaithfulnessBoundary.concreteCanonicalEqualityIffDeclaredOrbitProved &&
      !gaugeFaithfulnessBoundary.formalTarget03Closed &&
      !gaugeFaithfulnessBoundary.formalParityClosed &&
      !gaugeFaithfulnessBoundary.claimReady := by
  decide

end SounioPireusGaugeCoboundaryFaithfulness
