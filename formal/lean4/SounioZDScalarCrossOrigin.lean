import SounioZDScalarBit

/-! Necessary arithmetic conditions for an unresolved cross-origin scalar collision.
No bit-free classification is asserted. -/
namespace Sounio.ZDScalarCrossOrigin
open Sounio.ZDSignedState Sounio.ZDScalarInvariant Sounio.ZDScalarCore
set_option maxRecDepth 8192
set_option maxHeartbeats 16000000

def gap {d : Nat} (x : Code d) : Nat := capacity d - edges x

def headT : {d : Nat} → Code d → Nat
  | _, .t x => headT x + 1
  | _, _ => 0

theorem capacity_pos (d : Nat) : 0 < capacity d := by
  have h := (capacity_bounds d).1
  have hp : 0 < 4^d := Nat.pow_pos (by decide)
  omega

theorem capacity_odd (d : Nat) : capacity d % 2 = 1 := by
  have h := capacity_scale d
  omega

theorem gap_add_edges {d : Nat} (x : Code d) :
    gap x + edges x = capacity d := by
  have h := edges_le_capacity x
  simp only [gap]
  omega

theorem gap_t {d : Nat} (x : Code d) : gap (.t x) = 4 * gap x := by
  have h := gap_add_edges x
  have h' := gap_add_edges (.t x)
  simp only [capacity, edges] at h'
  omega

theorem gap_c {d : Nat} (x : Code d) :
    gap (.c x) = 3 * order d + 4 * gap x := by
  have h := gap_add_edges x
  have h' := gap_add_edges (.c x)
  simp only [capacity, edges] at h'
  omega

theorem gap_reset (d : Nat) : gap (Code.reset (d:=d)) = 0 := by
  simp only [gap, capacity_reset, Nat.sub_self]

theorem gap_shape {d : Nat} (x : Code d) :
    gap x = 0 ∨ ∃ u, gap x = 2^(2*headT x) * (2*u+1) := by
  induction x with
  | k => left; rfl
  | z => right; exact ⟨1, by decide⟩
  | reset => left; exact gap_reset _
  | t x ih =>
    rw [gap_t]
    rcases ih with hz | ⟨u,hu⟩
    · left; omega
    · right
      refine ⟨u, ?_⟩
      simp only [headT, Nat.mul_add, Nat.mul_one, Nat.pow_add]
      rw [hu]
      grind
  | @c d x ih =>
    right
    have hq := order_odd d
    rw [gap_c]
    simp only [headT, Nat.mul_zero, Nat.pow_zero, Nat.one_mul]
    exact ⟨(3*order d+4*gap x)/2, by omega⟩

theorem pow_two_odd_unique (k l a b : Nat)
    (ha : a%2=1) (hb : b%2=1)
    (he : 2^k*a=2^l*b) : k=l := by
  induction k generalizing l with
  | zero =>
    cases l with
    | zero => rfl
    | succ l =>
      simp only [Nat.pow_zero, Nat.one_mul, Nat.pow_succ] at he
      have hh : a=2*(2^l*b) := by grind
      omega
  | succ k ih =>
    cases l with
    | zero =>
      simp only [Nat.pow_zero, Nat.one_mul, Nat.pow_succ] at he
      have hh : 2*(2^k*a)=b := by grind
      omega
    | succ l =>
      have hh : 2^k*a=2^l*b := by
        simp only [Nat.pow_succ] at he
        have h2 : 2*(2^k*a)=2*(2^l*b) := by grind
        omega
      exact congrArg Nat.succ (ih l hh)

theorem reset_eq_gap {d : Nat} (x : Code d) (a j : Nat)
    (he : value a (4*2^j) (Code.reset (d:=d)) =
      value a (4*2^j) (.t x)) :
    a * gap x = 2^(j+1) * (3*edges x+4*positives x) := by
  have h := gap_add_edges x
  simp only [value, positives, Nat.mul_zero, Nat.add_zero] at he
  rw [capacity_reset] at he
  simp only [Nat.pow_succ]
  simp only [capacity, edges] at he
  grind

/-- Any reset/T equality forces an exact match between the coefficient exponent
and the number of exterior T constructors. No slope or origin hypothesis. -/
theorem reset_collision_exponent {d : Nat} (x : Code d) (a j : Nat)
    (ha : a%2=1)
    (he : value a (4*2^j) (Code.reset (d:=d)) =
      value a (4*2^j) (.t x)) :
    j+1 = 2*headT x := by
  have hg := reset_eq_gap x a j he
  have hs := gap_add_edges x
  have hc := capacity_odd d
  have hp := capacity_pos d
  have hj := Nat.two_pow_pos (j+1)
  have hn : (3*edges x+4*positives x)%2=1 := by
    have heven : (a*gap x)%2=0 := by
      rw [hg]
      simp [Nat.pow_succ,Nat.mul_mod]
    have hgap : gap x%2=0 := by
      simpa [Nat.mul_mod,ha] using heven
    omega
  rcases gap_shape x with hz | ⟨u,hu⟩
  · rw [hz] at hg
    simp only [Nat.mul_zero] at hg
    have hnpos : 0<3*edges x+4*positives x := by omega
    have hprod := Nat.mul_pos hj hnpos
    omega
  · have hodd : (a*(2*u+1))%2=1 := by simp [Nat.mul_mod,Nat.add_mod,ha]
    have heq : 2^(2*headT x)*(a*(2*u+1)) =
        2^(j+1)*(3*edges x+4*positives x) := by
      rw [hu] at hg
      grind
    exact (pow_two_odd_unique _ _ _ _ hodd hn heq).symm

theorem reset_ne_even_exponent {d : Nat} (x : Code d) (a j : Nat)
    (ha : a%2=1) (hj : j%2=0) :
    value a (4*2^j) (Code.reset (d:=d)) ≠
      value a (4*2^j) (.t x) := by
  intro he
  have h := reset_collision_exponent x a j ha he
  omega

theorem order_mod_three (d : Nat) : order d%3=0 ∨ order d%3=1 := by
  induction d with
  | zero => decide
  | succ d ih =>
    rcases ih with h | h <;> simp [order,Nat.add_mod,Nat.mul_mod,h]

theorem edges_mod_three {d : Nat} (x : Code d) : edges x%3=0 := by
  induction x with
  | k => decide
  | z => decide
  | @reset d =>
    rcases order_mod_three d with h | h <;> simp [edges,Nat.mul_mod,Nat.add_mod,h]
  | t x ih => simp [edges,Nat.add_mod,Nat.mul_mod,ih]
  | c x ih => simp [edges,Nat.mul_mod,ih]

theorem positives_mod_eighteen {d : Nat} (x : Code d) : positives x%18=0 := by
  induction x with
  | k => decide
  | z => decide
  | reset => rfl
  | t x ih =>
    have hm := edges_mod_three x
    have he : 6*edges x%18=0 := by omega
    simp [positives,Nat.add_mod,Nat.mul_mod,ih,he]
  | c x ih => simp [positives,Nat.mul_mod,ih]

theorem weighted_numerator_mod_nine {d : Nat} (x : Code d) :
    (3*edges x+4*positives x)%9=0 := by
  have hm := edges_mod_three x
  have hp := positives_mod_eighteen x
  omega

theorem reset_collision_gap_mod_nine {d : Nat} (x : Code d) (a j : Nat)
    (ha : a%3≠0)
    (he : value a (4*2^j) (Code.reset (d:=d)) =
      value a (4*2^j) (.t x)) : gap x%9=0 := by
  have h := reset_eq_gap x a j he
  have hn := weighted_numerator_mod_nine x
  have hprod : (a*gap x)%9=0 := by
    rw [h]
    simp [Nat.mul_mod,hn]
  have ha9 : a%9=1 ∨ a%9=2 ∨ a%9=4 ∨ a%9=5 ∨ a%9=7 ∨ a%9=8 := by omega
  rcases ha9 with ha9 | ha9 | ha9 | ha9 | ha9 | ha9 <;>
    rw [Nat.mul_mod,ha9] at hprod <;> omega



def normalizedGap {d : Nat} (x : Code d) : Nat := gap x / 3

/-- Count C constructors situated at even absolute depth, measured from depth zero. -/
def selectedCs : {d : Nat} → Code d → Nat
  | _, .t x => selectedCs x
  | d+1, .c x => selectedCs x + if d%2=1 then 1 else 0
  | _, _ => 0

theorem capacity_mod_three (d : Nat) : capacity d%3=0 := by
  induction d with
  | zero => decide
  | succ d ih => simp [capacity,Nat.add_mod,Nat.mul_mod,ih]

theorem gap_mod_three {d : Nat} (x : Code d) : gap x%3=0 := by
  have h := gap_add_edges x
  have hc := capacity_mod_three d
  have hm := edges_mod_three x
  omega

theorem normalized_gap_t {d : Nat} (x : Code d) :
    normalizedGap (.t x)=4*normalizedGap x := by
  have hg := gap_t x
  have hm := gap_mod_three x
  simp only [normalizedGap]
  omega

theorem normalized_gap_c {d : Nat} (x : Code d) :
    normalizedGap (.c x)=order d+4*normalizedGap x := by
  have hg := gap_c x
  have hm := gap_mod_three x
  simp only [normalizedGap]
  omega

theorem order_mod_three_parity (d : Nat) :
    order d%3 = if d%2=0 then 0 else 1 := by
  induction d with
  | zero => decide
  | succ d ih =>
    simp only [order]
    by_cases hd : d%2=0
    · have hs : (d+1)%2≠0 := by omega
      rw [if_pos hd] at ih
      rw [if_neg hs]
      omega
    · have hs : (d+1)%2=0 := by omega
      rw [if_neg hd] at ih
      rw [if_pos hs]
      omega

theorem zero_gap_selected_cs {d : Nat} (x : Code d)
    (hx : coreOrigin x=false) :
    normalizedGap x%3=(1+selectedCs x)%3 := by
  induction x with
  | k => simp [coreOrigin] at hx
  | z => decide
  | reset => simp [coreOrigin] at hx
  | t x ih =>
    have h := ih hx
    rw [normalized_gap_t]
    simp only [selectedCs]
    omega
  | @c d x ih =>
    have h := ih hx
    have hq := order_mod_three_parity d
    rw [normalized_gap_c]
    simp only [selectedCs]
    by_cases hd : d%2=0
    · have hs : d%2≠1 := by omega
      rw [if_pos hd] at hq
      rw [if_neg hs]
      omega
    · have hs : d%2=1 := by omega
      rw [if_neg hd] at hq
      rw [if_pos hs]
      omega

theorem reset_collision_selected_cs {d : Nat} (x : Code d) (a j : Nat)
    (hx : coreOrigin x=false) (ha : a%3≠0)
    (he : value a (4*2^j) (Code.reset (d:=d)) =
      value a (4*2^j) (.t x)) :
    selectedCs x%3=2 := by
  have hg := reset_collision_gap_mod_nine x a j ha he
  have h := zero_gap_selected_cs x hx
  simp only [normalizedGap] at h
  omega

/-- The remaining obligation, restricted by proved necessary conditions.
It remains a hypothesis, not a theorem. -/
def ResidualSeparation : Prop :=
  ∀ (d a j : Nat) (x : Code d), a%2=1 → a%3≠0 → a<6*2^j →
    coreOrigin x=false → j+1=2*headT x → selectedCs x%3=2 →
    value a (4*2^j) (Code.reset (d:=d)) ≠ value a (4*2^j) (.t x)

theorem reset_separation_iff_residual :
    ResetSeparation ↔ ResidualSeparation := by
  constructor
  · intro h d a j x ha h3 hlt _ _ _
    exact h d a (4*2^j) x ⟨j,rfl,ha,h3,hlt⟩
  · intro h d a b x hc
    obtain ⟨j,rfl,ha,h3,hlt⟩ := hc
    intro he
    cases hx : coreOrigin x with
    | true =>
      have hg : CoreGood a (4*2^j) := by
        refine ⟨j+2,?_,ha,by omega⟩
        simp only [Nat.pow_succ]
        omega
      exact reset_ne_t_core x hx hg he
    | false =>
      exact h d a j x ha h3 hlt hx
        (reset_collision_exponent x a j ha he)
        (reset_collision_selected_cs x a j hx h3 he) he

theorem native_iso_iff_sum_of_residual (hres : ResidualSeparation)
    (d W V : Nat) (hW : W<2^(d+3)) (hW0 : W≠0)
    (hV : V<2^(d+3)) (hV0 : V≠0) :
    Nonempty (Sounio.ZDTwinCover.GraphIso
      (Sounio.ZDTwinCover.NativeVertex (d+3) W)
      (Sounio.ZDTwinCover.NativeVertex (d+3) V)
      (Sounio.ZDTwinCover.NativeAdj (d+3) W)
      (Sounio.ZDTwinCover.NativeAdj (d+3) V)) ↔
    nativeEdges d W hW hW0+nativeTriangles d W hW hW0 =
      nativeEdges d V hV hV0+nativeTriangles d V hV hV0 := by
  exact native_iso_iff_sum_of_reset_separation
    (reset_separation_iff_residual.mpr hres) d W V hW hW0 hV hV0

/-- A zero-origin low-ratio word defeats a naive unconditional 3-adic shortcut.
Its odd denominator factor 197 still obstructs a genuine dyadic collision. -/
theorem residual_scope_control :
    let x : Code 6 := .t (.t (.c (.t (.c (.c .z)))))
    coreOrigin x=false ∧ headT x=2 ∧ selectedCs x=2 ∧ gap x=28368 ∧
    3*edges x+4*positives x=68427 ∧
    (gap x/3)%9=6 ∧ ((3*edges x+4*positives x)/3)%9=3 ∧
    (22809 : Nat)%591≠0 := by decide

#print axioms capacity_pos
#print axioms capacity_odd
#print axioms gap_add_edges
#print axioms gap_t
#print axioms gap_c
#print axioms gap_reset
#print axioms gap_shape
#print axioms pow_two_odd_unique
#print axioms reset_eq_gap
#print axioms reset_collision_exponent
#print axioms reset_ne_even_exponent
#print axioms order_mod_three
#print axioms edges_mod_three
#print axioms positives_mod_eighteen
#print axioms weighted_numerator_mod_nine
#print axioms reset_collision_gap_mod_nine
#print axioms reset_separation_iff_residual
#print axioms native_iso_iff_sum_of_residual
#print axioms residual_scope_control
#print axioms capacity_mod_three
#print axioms gap_mod_three
#print axioms normalized_gap_t
#print axioms normalized_gap_c
#print axioms order_mod_three_parity
#print axioms zero_gap_selected_cs
#print axioms reset_collision_selected_cs
end Sounio.ZDScalarCrossOrigin
