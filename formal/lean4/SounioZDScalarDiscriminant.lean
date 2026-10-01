import SounioZDScalarInnerBound

/-! Integer-root rigidity and exact dyadic-depth constraint.
No uniform low-cone exclusion or ResidualSeparation theorem is asserted. -/
namespace Sounio.ZDScalarDiscriminant
open Sounio.ZDScalarInnerBound
set_option maxRecDepth 8192
set_option maxHeartbeats 16000000

def slopeCore (v s t : Int) : Int :=
  2*s*s*t*t*v*v-6*s*s*t*t*v+4*s*s*t*t+6*s*t*t*v-6*s*t*t
  -3*s*t*v+3*s*t+2*t*t-3*t+1

def oddFactor (u v s t a : Int) : Int :=
  21*(a*(s*v-s+1)+t*slopeCore v s t)-14*a*s*v*u

theorem slope_factor (v s t : Int) :
    numSlope v s t = 21*s*t*v*slopeCore v s t := by
  unfold numSlope slopeCore
  grind

theorem polynomial_factor (u v s t a : Int) :
    collisionPoly u v s t a =
      coeffC v s t a-s*v*u*oddFactor u v s t a := by
  rw [polynomial_quadratic]
  unfold coeffA coeffB oddFactor
  rw [slope_factor]
  grind

theorem odd_factor (u v s t a : Int)
    (ha : a%2=1) (hs : s%2=0) (ht : t%2=0) :
    oddFactor u v s t a % 2 = 1 := by
  simp [oddFactor, Int.add_emod, Int.sub_emod, Int.mul_emod, ha, hs, ht]

theorem collision_constant_factor (u v s t a : Int)
    (hf : collisionPoly u v s t a=0) :
    coeffC v s t a=s*v*u*oddFactor u v s t a := by
  rw [polynomial_factor] at hf
  omega

theorem integer_root_unique (u₁ u₂ v s t a : Int)
    (ha : a%2=1) (hs : s%2=0) (ht : t%2=0)
    (hsv : s*v≠0)
    (h₁ : collisionPoly u₁ v s t a=0)
    (h₂ : collisionPoly u₂ v s t a=0) : u₁=u₂ := by
  by_cases hne : u₁=u₂
  · exact hne
  exfalso
  rw [polynomial_quadratic] at h₁ h₂
  have hz : (u₁-u₂)*(coeffA v s a*(u₁+u₂)+coeffB v s t a)=0 := by
    grind
  have hsum : coeffA v s a*(u₁+u₂)+coeffB v s t a=0 := by
    rcases Int.mul_eq_zero.mp hz with h | h
    · omega
    · exact h
  have hfactor : ∀ u v s t a : Int,
      coeffA v s a*u+coeffB v s t a =
      s*v*(14*a*s*v*u-21*(a*(s*v-s+1)+t*slopeCore v s t)) := by
    clear u₁ u₂ v s t a ha hs ht hsv h₁ h₂ hne hz hsum
    intro u v s t a
    unfold coeffA coeffB
    rw [slope_factor]
    grind
  have hfact : s*v*(14*a*s*v*(u₁+u₂)-21*(a*(s*v-s+1)+t*slopeCore v s t))=0 := by
    rw [← hfactor]
    exact hsum
  have ho : oddFactor (u₁+u₂) v s t a=0 := by
    have h := (Int.mul_eq_zero.mp hfact).resolve_left hsv
    unfold oddFactor
    omega
  have hp := odd_factor (u₁+u₂) v s t a ha hs ht
  rw [ho] at hp
  contradiction

theorem zero_constant_no_nonzero_root (u v s t a : Int)
    (ha : a%2=1) (hs : s%2=0) (ht : t%2=0)
    (hs0 : s≠0) (hv0 : v≠0) (hu0 : u≠0)
    (hc : coeffC v s t a=0) : collisionPoly u v s t a≠0 := by
  intro hf
  have h := collision_constant_factor u v s t a hf
  have ho := odd_factor u v s t a ha hs ht
  rw [hc] at h
  have hn : oddFactor u v s t a≠0 := by intro hz; rw [hz] at ho; contradiction
  have hp := Int.mul_ne_zero (Int.mul_ne_zero (Int.mul_ne_zero hs0 hv0) hu0) hn
  exact hp h.symm

def ExactTwoVal (n : Int) (e : Nat) : Prop :=
  ∃ z : Int, z%2=1 ∧ n=(2:Int)^e*z

theorem exact_two_val_unique (n : Int) (e f : Nat)
    (he : ExactTwoVal n e) (hf : ExactTwoVal n f) : e=f := by
  obtain ⟨a,ha,he⟩ := he
  obtain ⟨b,hb,hf⟩ := hf
  have h : (2:Int)^e*a=(2:Int)^f*b := he.symm.trans hf
  clear he hf n
  induction e generalizing f a b with
  | zero =>
    cases f with
    | zero => rfl
    | succ f =>
      simp only [Int.pow_zero, Int.one_mul, Int.pow_succ] at h
      have heven : ((2:Int)^f*2*b)%2=0 := by simp [Int.mul_emod]
      rw [← h] at heven
      omega
  | succ e ih =>
    cases f with
    | zero =>
      simp only [Int.pow_zero, Int.one_mul, Int.pow_succ] at h
      have heven : ((2:Int)^e*2*a)%2=0 := by simp [Int.mul_emod]
      rw [h] at heven
      omega
    | succ f =>
      simp only [Int.pow_succ] at h
      have hh : (2:Int)^e*a=(2:Int)^f*b := by grind
      exact congrArg Nat.succ (ih f a ha b hb hh)

theorem pow_two_even (c : Nat) (hc : 1≤c) : (2:Int)^c%2=0 := by
  obtain ⟨n,rfl⟩ := Nat.exists_eq_add_of_le hc
  rw [Int.pow_add]
  simp

theorem dyadic_constant_valuation (r h c k a : Nat)
    (hc : 1≤c) (hk : 1≤k) (ha : a%2=1)
    (hf : collisionPoly ((2:Int)^(r+1)) ((2:Int)^h)
      ((2:Int)^c) ((2:Int)^k) a=0) :
    ExactTwoVal (coeffC ((2:Int)^h) ((2:Int)^c) ((2:Int)^k) a)
      (r+c+h+1) := by
  have hs := pow_two_even c hc
  have ht := pow_two_even k hk
  have hai : (a:Int)%2=1 := by omega
  refine ⟨oddFactor ((2:Int)^(r+1)) ((2:Int)^h)
    ((2:Int)^c) ((2:Int)^k) a, odd_factor _ _ _ _ _ hai hs ht, ?_⟩
  have he := collision_constant_factor _ _ _ _ _ hf
  have hp : (2:Int)^c*(2:Int)^h*(2:Int)^(r+1)=(2:Int)^(r+c+h+1) := by
    rw [← Int.pow_add, ← Int.pow_add]
    congr 1
    omega
  rw [hp] at he
  exact he

theorem depth_from_constant (r h c k a e : Nat)
    (hc : 1≤c) (hk : 1≤k) (ha : a%2=1)
    (he : ExactTwoVal (coeffC ((2:Int)^h) ((2:Int)^c) ((2:Int)^k) a) e)
    (hf : collisionPoly ((2:Int)^(r+1)) ((2:Int)^h)
      ((2:Int)^c) ((2:Int)^k) a=0) : e=r+c+h+1 := by
  exact exact_two_val_unique _ _ _ he (dyadic_constant_valuation r h c k a hc hk ha hf)



def slopeOdd (v s t a : Int) : Int :=
  a*(s*v-s+1)+t*slopeCore v s t

def reducedDiscriminant (v s t a : Int) : Int :=
  441*slopeOdd v s t a*slopeOdd v s t a-56*a*coeffC v s t a

theorem discriminant_square_factor (v s t a : Int) :
    coeffB v s t a*coeffB v s t a-4*coeffA v s a*coeffC v s t a =
      (s*v)*(s*v)*reducedDiscriminant v s t a := by
  unfold coeffA coeffB reducedDiscriminant slopeOdd
  rw [slope_factor]
  grind

theorem slope_odd (v s t a : Int)
    (ha : a%2=1) (hs : s%2=0) (ht : t%2=0) :
    slopeOdd v s t a%2=1 := by
  simp [slopeOdd, Int.add_emod, Int.sub_emod, Int.mul_emod, ha, hs, ht]

theorem reduced_discriminant_mod_eight (v s t a : Int)
    (ha : a%2=1) (hs : s%2=0) (ht : t%2=0) :
    reducedDiscriminant v s t a%8=1 := by
  have ho := slope_odd v s t a ha hs ht
  have he : slopeOdd v s t a%8=1 ∨ slopeOdd v s t a%8=3 ∨
      slopeOdd v s t a%8=5 ∨ slopeOdd v s t a%8=7 := by omega
  rcases he with he | he | he | he <;>
    simp [reducedDiscriminant, Int.sub_emod, Int.mul_emod, he]

theorem reduced_discriminant_mod_three (v s t a : Int)
    (ha : a%3≠0) (hs : s%3≠0) (hv : v%3≠0) :
    reducedDiscriminant v s t a%3=1 := by
  have ha3 : a%3=1 ∨ a%3=2 := by omega
  have hs3 : s%3=1 ∨ s%3=2 := by omega
  have hv3 : v%3=1 ∨ v%3=2 := by omega
  rcases ha3 with ha3 | ha3 <;>
    rcases hs3 with hs3 | hs3 <;>
      rcases hv3 with hv3 | hv3 <;>
        simp [reducedDiscriminant, coeffC, numConstant,
          Int.add_emod, Int.sub_emod, Int.mul_emod, ha3, hs3, hv3]

theorem reduced_discriminant_mod_twenty_four (v s t a : Int)
    (ha : a%2=1) (hs : s%2=0) (ht : t%2=0)
    (ha3 : a%3≠0) (hs3 : s%3≠0) (hv3 : v%3≠0) :
    reducedDiscriminant v s t a%24=1 := by
  have h8 := reduced_discriminant_mod_eight v s t a ha hs ht
  have h3 := reduced_discriminant_mod_three v s t a ha3 hs3 hv3
  omega

theorem pow_two_mod_three (k : Nat) : (2:Int)^k%3≠0 := by
  have aux : ∀ k : Nat, (2:Int)^k%3=1 ∨ (2:Int)^k%3=2 := by
    intro k
    induction k with
    | zero => decide
    | succ k ih =>
      rw [Int.pow_succ]
      rcases ih with ih | ih <;> simp [Int.mul_emod, ih]
  rcases aux k with h | h <;> omega

theorem two_block_local_square_residues (h c k a : Nat)
    (hc : 1≤c) (hk : 1≤k) (ha : a%2=1) (ha3 : a%3≠0) :
    reducedDiscriminant ((2:Int)^h) ((2:Int)^c) ((2:Int)^k) a%24=1 := by
  exact reduced_discriminant_mod_twenty_four _ _ _ _
    (by omega) (pow_two_even c hc) (pow_two_even k hk)
    (by omega) (pow_two_mod_three c) (pow_two_mod_three h)



theorem balanced_half_root (t a : Int) (he : 7*a=21*t*t-9*t-3) :
    coeffA 2 t a+2*coeffB 2 t t a+4*coeffC 2 t t a=0 := by
  unfold coeffA coeffB coeffC numSlope numConstant
  grind

theorem balanced_constant (t a : Int) (he : 7*a=21*t*t-9*t-3) :
    coeffC 2 t t a=9*t*(14*t*t*t*t-t*t-t-1) := by
  unfold coeffC numConstant
  grind

private theorem half_discriminant_identity (A B C : Int) (h : A+2*B+4*C=0) :
    B*B-4*A*C=(A+B)*(A+B) := by grind

theorem balanced_discriminant_square (t a : Int) (he : 7*a=21*t*t-9*t-3) :
    coeffB 2 t t a*coeffB 2 t t a-4*coeffA 2 t a*coeffC 2 t t a =
      (coeffA 2 t a+coeffB 2 t t a)*(coeffA 2 t a+coeffB 2 t t a) := by
  exact half_discriminant_identity _ _ _ (balanced_half_root t a he)

private theorem cancel_nonzero_factor (t x y : Int) (ht : t≠0)
    (he : t*x=t*y) : x=y := by
  have hz : t*(x-y)=0 := by grind
  have h := (Int.mul_eq_zero.mp hz).resolve_left ht
  omega

theorem balanced_no_integer_root (u t a : Int)
    (ht : t%2=0) (ht0 : t≠0) (he : 7*a=21*t*t-9*t-3) :
    collisionPoly u 2 t t a≠0 := by
  intro hf
  have hc := balanced_constant t a he
  have hroot := collision_constant_factor u 2 t t a hf
  have heq : t*(9*(14*t*t*t*t-t*t-t-1)) =
      t*(2*u*oddFactor u 2 t t a) := by
    rw [hc] at hroot
    clear he hf hc
    grind
  have h := cancel_nonzero_factor _ _ _ ht0 heq
  have ho : (9*(14*t*t*t*t-t*t-t-1))%2=1 := by
    simp [Int.sub_emod, Int.mul_emod, ht]
  have hzero : (2*u*oddFactor u 2 t t a)%2=0 := by simp [Int.mul_emod]
  rw [h] at ho
  omega

def balancedT (n : Nat) : Int := (2:Int)^(3*n+1)
def balancedA (n : Nat) : Int :=
  3*(balancedT n*balancedT n-(3*balancedT n+1)/7)

theorem balanced_t_mod_seven (n : Nat) : balancedT n%7=2 := by
  induction n with
  | zero => decide
  | succ n ih =>
    have he : 3*(n+1)+1=(3*n+1)+3 := by omega
    unfold balancedT at *
    rw [he, Int.pow_add]
    simp [Int.mul_emod, ih]

private theorem balanced_coefficient_bounds (t a : Int) (ht : 2≤t)
    (he : 7*a=21*t*t-9*t-3) : 0<a ∧ a<3*t*t := by
  have hs : 2*t≤t*t := Int.mul_le_mul_of_nonneg_right ht (by omega)
  have he' : 7*a=21*(t*t)-9*t-3 := by simpa [Int.mul_assoc] using he
  constructor
  · omega
  · rw [Int.mul_assoc]
    omega

theorem balanced_parameters (n : Nat) :
    balancedT n%2=0 ∧ 2≤balancedT n ∧
    balancedA n%2=1 ∧ balancedA n%3=0 ∧
    0<balancedA n ∧ balancedA n<3*balancedT n*balancedT n ∧
    7*balancedA n=21*balancedT n*balancedT n-9*balancedT n-3 := by
  have ht := pow_two_even (3*n+1) (by omega)
  change balancedT n%2=0 at ht
  have ht2 : 2≤balancedT n := by
    unfold balancedT
    rw [Int.pow_succ]
    have hp : 0<(2:Int)^(3*n) := Int.pow_pos (by decide)
    omega
  have ht7 := balanced_t_mod_seven n
  have hd : (3*balancedT n+1)%7=0 := by
    simp [Int.add_emod, Int.mul_emod, ht7]
  have hq : 7*((3*balancedT n+1)/7)=3*balancedT n+1 := by omega
  have hqodd : ((3*balancedT n+1)/7)%2=1 := by omega
  have haodd : balancedA n%2=1 := by
    simp [balancedA, Int.sub_emod, Int.mul_emod, ht, hqodd]
  have ha3 : balancedA n%3=0 := by simp [balancedA]
  have hae : 7*balancedA n=21*balancedT n*balancedT n-9*balancedT n-3 := by
    unfold balancedA
    grind
  obtain ⟨hap,halt⟩ := balanced_coefficient_bounds _ _ ht2 hae
  exact ⟨ht,ht2,haodd,ha3,hap,halt,hae⟩

theorem balanced_infinite_controls (n : Nat) :
    (∃ d : Int, coeffB 2 (balancedT n) (balancedT n) (balancedA n)*
      coeffB 2 (balancedT n) (balancedT n) (balancedA n)-
      4*coeffA 2 (balancedT n) (balancedA n)*
      coeffC 2 (balancedT n) (balancedT n) (balancedA n)=d*d) ∧
    (∀ u : Int, collisionPoly u 2 (balancedT n) (balancedT n) (balancedA n)≠0) := by
  obtain ⟨ht,ht2,ha,h3,hp,hlt,he⟩ := balanced_parameters n
  constructor
  · exact ⟨_, balanced_discriminant_square _ _ he⟩
  · intro u
    exact balanced_no_integer_root _ _ _ ht (by omega) he



theorem balanced_checksum_factors (t a : Int) (he : 7*a=21*t*t-9*t-3) :
    7*a+3*(t-1)*(2*t-1)*(3*t+1)=9*t*(2*t*t-1) ∧
    innerChecksum t t a=3*t*(8*t*t*t*t*t-14*t*t*t*t+9*t*t+t-3) := by
  constructor
  · grind
  · unfold innerChecksum
    grind

theorem balanced_passes_checksums (n : Nat) (hn : 1≤n) :
    balancedT n ∣ 7*balancedA n+
      3*(balancedT n-1)*(2*balancedT n-1)*(3*balancedT n+1) ∧
    (4:Int) ∣ innerChecksum (balancedT n) (balancedT n) (balancedA n) := by
  have hp := balanced_parameters n
  have he := hp.2.2.2.2.2.2
  have hfs := balanced_checksum_factors _ _ he
  constructor
  · rw [hfs.1]
    exact ⟨9*(2*balancedT n*balancedT n-1), by grind⟩
  · have ht : (4:Int) ∣ balancedT n := by
      obtain ⟨j,hj⟩ := Nat.exists_eq_add_of_le (show 2≤3*n+1 by omega)
      have heq : 3*n+1=2+j := by omega
      refine ⟨(2:Int)^j, ?_⟩
      simp [balancedT, heq, Int.pow_add]
    obtain ⟨j,hj⟩ := ht
    rw [hfs.2]
    refine ⟨3*j*(8*balancedT n*balancedT n*balancedT n*balancedT n*balancedT n-
      14*balancedT n*balancedT n*balancedT n*balancedT n+
      9*balancedT n*balancedT n+balancedT n-3), ?_⟩
    clear hp he hfs hn
    grind

#print axioms balanced_checksum_factors
#print axioms balanced_passes_checksums

#print axioms balanced_half_root
#print axioms balanced_constant
#print axioms half_discriminant_identity
#print axioms balanced_discriminant_square
#print axioms cancel_nonzero_factor
#print axioms balanced_no_integer_root
#print axioms balanced_t_mod_seven
#print axioms balanced_coefficient_bounds
#print axioms balanced_parameters
#print axioms balanced_infinite_controls

#print axioms discriminant_square_factor
#print axioms slope_odd
#print axioms reduced_discriminant_mod_eight
#print axioms reduced_discriminant_mod_three
#print axioms reduced_discriminant_mod_twenty_four
#print axioms pow_two_mod_three
#print axioms two_block_local_square_residues

#print axioms slope_factor
#print axioms polynomial_factor
#print axioms odd_factor
#print axioms collision_constant_factor
#print axioms integer_root_unique
#print axioms zero_constant_no_nonzero_root
#print axioms exact_two_val_unique
#print axioms pow_two_even
#print axioms dyadic_constant_valuation
#print axioms depth_from_constant
end Sounio.ZDScalarDiscriminant
