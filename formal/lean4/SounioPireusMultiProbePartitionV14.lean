/-
  FORMAL_PARITY for Pireus V14 multi-probe block certification schema.

  Formalizes the structural partitioning and domain certification schema:
  - 64 blocks of width 630 partitioning [0, 40320)
  - Exact cover theorem: start(0) = 0, stop(63) = 40320, contiguous adjacency
  - Positive witness theorem for block 63: [39690, 40320)
  - Injective domain serialization and disjoint block domains
  - Extension to all 32 admitted V13 classes (AllProbesCertified)
-/

namespace SounioPireusMultiProbePartitionV14

def totalActions : Nat := 40320
def blockCount : Nat := 64
def blockWidth : Nat := 630
def probeCount : Nat := 32

def blockStart (k : Nat) : Nat := k * blockWidth
def blockEnd (k : Nat) : Nat := (k + 1) * blockWidth

theorem block_width_pos : blockWidth > 0 := by decide
theorem block_count_pos : blockCount > 0 := by decide

theorem partition_dimension_exact : blockCount * blockWidth = totalActions := by
  rfl

theorem block_start_zero : blockStart 0 = 0 := by
  rfl

theorem block_end_last : blockEnd (blockCount - 1) = totalActions := by
  rfl

theorem block_contiguous_adjacency (k : Nat) : blockEnd k = blockStart (k + 1) := by
  dsimp [blockStart, blockEnd]

theorem block_bounds_step (k : Nat) : blockEnd k = blockStart k + blockWidth := by
  dsimp [blockStart, blockEnd]
  rw [Nat.succ_mul]

theorem block_start_strict_mono {k₁ k₂ : Nat} (h : k₁ < k₂) :
    blockStart k₁ < blockStart k₂ := by
  dsimp [blockStart]
  exact Nat.mul_lt_mul_of_pos_right h (by decide)

theorem block_63_start_exact : blockStart 63 = 39690 := by
  rfl

theorem block_63_end_exact : blockEnd 63 = 40320 := by
  rfl

theorem positive_witness_block_63 :
    blockStart 63 = 39690 ∧ blockEnd 63 = 40320 ∧ blockEnd 63 - blockStart 63 = 630 := by
  decide

theorem negative_witness_block_63_mismatch :
    ¬ (blockStart 63 = 39689 ∧ blockEnd 63 = 40319) := by
  decide

structure BlockDomain where
  probeId : Fin probeCount
  blockIndex : Fin blockCount
deriving DecidableEq, Repr

def serializeDomain (d : BlockDomain) : Nat :=
  d.probeId.val * blockCount + d.blockIndex.val

theorem serialize_domain_injective :
    Function.Injective serializeDomain := by
  intro ⟨p1, b1⟩ ⟨p2, b2⟩ h
  dsimp [serializeDomain, blockCount] at h
  have hb1 : b1.val < 64 := b1.isLt
  have hb2 : b2.val < 64 := b2.isLt
  have hp_val : p1.val = p2.val := by
    have h1 : (p1.val * 64 + b1.val) / 64 = p1.val := by
      rw [Nat.add_comm, Nat.add_mul_div_right b1.val p1.val (by decide), Nat.div_eq_of_lt hb1, Nat.zero_add]
    have h2 : (p2.val * 64 + b2.val) / 64 = p2.val := by
      rw [Nat.add_comm, Nat.add_mul_div_right b2.val p2.val (by decide), Nat.div_eq_of_lt hb2, Nat.zero_add]
    rw [← h1, ← h2, h]
  have hp : p1 = p2 := Fin.ext hp_val
  have hb_val : b1.val = b2.val := by
    have h1 : (p1.val * 64 + b1.val) % 64 = b1.val := by
      rw [Nat.add_comm, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt hb1]
    have h2 : (p2.val * 64 + b2.val) % 64 = b2.val := by
      rw [Nat.add_comm, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt hb2]
    rw [← h1, ← h2, h]
  have hb : b1 = b2 := Fin.ext hb_val
  subst hp
  subst hb
  rfl

def InBlockInterval (k : Nat) (view : Nat) : Prop :=
  blockStart k ≤ view ∧ view < blockEnd k

instance (k view : Nat) : Decidable (InBlockInterval k view) :=
  inferInstanceAs (Decidable (blockStart k ≤ view ∧ view < blockEnd k))

theorem blocks_disjoint {k₁ k₂ : Nat} (h : k₁ < k₂) (view : Nat) :
    InBlockInterval k₁ view → ¬ InBlockInterval k₂ view := by
  intro ⟨hstart1, hend1⟩ ⟨hstart2, hend2⟩
  dsimp [InBlockInterval, blockStart, blockEnd] at *
  have h_bound : (k₁ + 1) * blockWidth ≤ k₂ * blockWidth :=
    Nat.mul_le_mul_right blockWidth h
  omega

theorem block_cover_exists (view : Nat) (h : view < totalActions) :
    ∃ k : Fin blockCount, InBlockInterval k.val view := by
  have hwidth : blockWidth = 630 := rfl
  have htotal : totalActions = 40320 := rfl
  let k_val := view / 630
  have hk_lt : k_val < 64 := by
    dsimp [k_val]
    omega
  refine ⟨⟨k_val, hk_lt⟩, ?_⟩
  dsimp [InBlockInterval, blockStart, blockEnd, k_val, blockWidth]
  constructor
  · rw [Nat.mul_comm]
    exact Nat.mul_div_le view 630
  · have := Nat.div_add_mod view 630
    have := Nat.mod_lt view (by decide : 630 > 0)
    omega

def ExistsUnique (p : Fin blockCount → Prop) : Prop :=
  ∃ x, p x ∧ ∀ y, p y → y = x

theorem exact_cover_partition (view : Nat) (h : view < totalActions) :
    ExistsUnique (fun k => InBlockInterval k.val view) := by
  obtain ⟨k, hk⟩ := block_cover_exists view h
  refine ⟨k, hk, ?_⟩
  intro k' hk'
  rcases Nat.lt_trichotomy k'.val k.val with hlt | heq | hgt
  · exfalso
    exact blocks_disjoint hlt view hk' hk
  · exact Fin.ext heq
  · exfalso
    exact blocks_disjoint hgt view hk hk'

structure BlockCertification (d : BlockDomain) where
  blockCoverValid : blockEnd d.blockIndex.val - blockStart d.blockIndex.val = blockWidth
  blockContiguous : d.blockIndex.val + 1 < blockCount →
                    blockEnd d.blockIndex.val = blockStart (d.blockIndex.val + 1)
  injectiveDomain : serializeDomain d < probeCount * blockCount

theorem certifyBlock (d : BlockDomain) : BlockCertification d where
  blockCoverValid := by
    dsimp [blockStart, blockEnd, blockWidth]
    omega
  blockContiguous := by
    intro _
    exact block_contiguous_adjacency d.blockIndex.val
  injectiveDomain := by
    dsimp [serializeDomain, probeCount, blockCount]
    have hp : d.probeId.val < 32 := d.probeId.isLt
    have hb : d.blockIndex.val < 64 := d.blockIndex.isLt
    omega

def AllProbesCertified : Prop :=
  ∀ d : BlockDomain, BlockCertification d

theorem all_32_classes_v14_certified : AllProbesCertified := by
  intro d
  exact certifyBlock d

structure MultiProbeBlockCertificationBoundary where
  exactCover64x630Proved : Bool
  positiveWitnessBlock63Proved : Bool
  negativeWitnessRejected : Bool
  injectiveDomainSerializationProved : Bool
  disjointBlockDomainsProved : Bool
  all32ClassesCertified : Bool
  formalParityClosed : Bool
  claimReady : Bool
deriving Repr, BEq, DecidableEq

def multiProbeBlockCertificationBoundary : MultiProbeBlockCertificationBoundary :=
  { exactCover64x630Proved := true
  , positiveWitnessBlock63Proved := true
  , negativeWitnessRejected := true
  , injectiveDomainSerializationProved := true
  , disjointBlockDomainsProved := true
  , all32ClassesCertified := true
  , formalParityClosed := true
  , claimReady := false }

theorem multiprobe_v14_closed_without_claim_promotion :
    (multiProbeBlockCertificationBoundary.exactCover64x630Proved &&
     multiProbeBlockCertificationBoundary.positiveWitnessBlock63Proved &&
     multiProbeBlockCertificationBoundary.negativeWitnessRejected &&
     multiProbeBlockCertificationBoundary.injectiveDomainSerializationProved &&
     multiProbeBlockCertificationBoundary.disjointBlockDomainsProved &&
     multiProbeBlockCertificationBoundary.all32ClassesCertified &&
     multiProbeBlockCertificationBoundary.formalParityClosed &&
     !multiProbeBlockCertificationBoundary.claimReady) = true := by
  decide

end SounioPireusMultiProbePartitionV14
