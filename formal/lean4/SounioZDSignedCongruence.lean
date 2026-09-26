import SounioZDTwinCover

/-!
Explicit signed bijections for the two Cayley-Dickson matrix constructors.
Congruence means B(f r,f s) = epsilon(r) epsilon(s) A(r,s).
No finiteness, symmetry, or zero-diagonal assumption is needed for the generic
transport lemmas; the native specialization uses the certified actual matrices.
The full edge/triangle classification and bibliographic priority remain separate.
-/
namespace Sounio.ZDSignedCongruence
open SounioCDCocycle Sounio.ZDAlgebraBridge Sounio.ZDRecursion
open Sounio.ZDMatrixAssembly Sounio.ZDTwinCover
set_option maxRecDepth 4096
set_option maxHeartbeats 8000000

structure IndexIso (R S : Type) where
  toFun : R → S
  invFun : S → R
  left_inv : ∀ r, invFun (toFun r) = r
  right_inv : ∀ s, toFun (invFun s) = s

def IndexIso.refl (R : Type) : IndexIso R R :=
  ⟨id,id,fun _ => rfl,fun _ => rfl⟩

theorem IndexIso.injective {R S : Type} (f : IndexIso R S) {r s : R}
    (h : f.toFun r = f.toFun s) : r = s := by
  have e := congrArg f.invFun h
  simpa [f.left_inv] using e

theorem IndexIso.eq_iff {R S : Type} (f : IndexIso R S) (r s : R) :
    f.toFun r = f.toFun s ↔ r = s :=
  ⟨f.injective,fun h => congrArg f.toFun h⟩

structure SignedCongruence {R S : Type} (A : R → R → Int) (B : S → S → Int) where
  index : IndexIso R S
  weight : R → Int
  weight_sign : ∀ r, Sign (weight r)
  entry : ∀ r s, B (index.toFun r) (index.toFun s) = weight r * weight s * A r s

def SignedCongruence.refl {R : Type} (A : R → R → Int) : SignedCongruence A A where
  index := IndexIso.refl R
  weight := fun _ => 1
  weight_sign := fun _ => Or.inl rfl
  entry := by intros; simp [IndexIso.refl]

theorem sign_square (s : Int) (hs : Sign s) : s*s = 1 := by
  rcases hs with h | h <;> simp [h]

def signBit (s : Int) : Bool := decide (s = -1)

theorem hub_switch (e t u : Int) (he : Sign e) (ht : Sign t) (hu : Sign u)
    (α : Bool) :
    (if xor α (signBit (e*t*u)) then -u else u) = e * (if α then -t else t) := by
  rcases he with h | h <;> rcases ht with j | j <;> rcases hu with k | k <;>
    cases α <;> simp [h,j,k,signBit]

/-- Duplicate a bijection, optionally interchanging the two copies at each source index. -/
def liftIndex {R S : Type} (f : IndexIso R S) (δ : R → Bool) :
    IndexIso (Option (Bool × R)) (Option (Bool × S)) where
  toFun
    | none => none
    | some (α,r) => some (xor α (δ r),f.toFun r)
  invFun
    | none => none
    | some (β,s) => some (xor β (δ (f.invFun s)),f.invFun s)
  left_inv := by
    intro x
    cases x with
    | none => rfl
    | some x =>
      rcases x with ⟨α,r⟩
      simp [f.left_inv]
  right_inv := by
    intro x
    cases x with
    | none => rfl
    | some x =>
      rcases x with ⟨α,r⟩
      simp [f.right_inv]

def liftWeight {R : Type} (e : R → Int) : Option (Bool × R) → Int
  | none => 1
  | some (_,r) => e r

theorem liftWeight_sign {R : Type} (e : R → Int) (he : ∀ r, Sign (e r))
    (x : Option (Bool × R)) : Sign (liftWeight e x) := by
  cases x with
  | none => exact Or.inl rfl
  | some x => exact he x.2

/-- All signed hub vectors are allowed. Local pair swaps absorb their mismatch. -/
def tee_congr {R S : Type} [DecidableEq R] [DecidableEq S]
    {A : R → R → Int} {B : S → S → Int} (c : SignedCongruence A B)
    (t : R → Int) (u : S → Int) (ht : ∀ r, Sign (t r)) (hu : ∀ s, Sign (u s)) :
    SignedCongruence (tee A t) (tee B u) where
  index := liftIndex c.index (fun r => signBit (c.weight r * t r * u (c.index.toFun r)))
  weight := liftWeight c.weight
  weight_sign := liftWeight_sign c.weight c.weight_sign
  entry := by
    intro x y
    cases x with
    | none =>
      cases y with
      | none => simp [liftIndex,liftWeight,tee]
      | some y =>
        rcases y with ⟨β,s⟩
        simpa [liftIndex,liftWeight,tee] using
          hub_switch (c.weight s) (t s) (u (c.index.toFun s))
            (c.weight_sign s) (ht s) (hu _) β
    | some x =>
      rcases x with ⟨α,r⟩
      cases y with
      | none =>
        simpa [liftIndex,liftWeight,tee] using
          hub_switch (c.weight r) (t r) (u (c.index.toFun r))
            (c.weight_sign r) (ht r) (hu _) α
      | some y =>
        rcases y with ⟨β,s⟩
        by_cases h : r = s
        · subst s
          have hw := sign_square (c.weight r) (c.weight_sign r)
          simp [liftIndex,liftWeight,tee,hw]
        · have hf : c.index.toFun r ≠ c.index.toFun s := fun e => h (c.index.injective e)
          simpa [liftIndex,liftWeight,tee,h,hf] using c.entry r s

/-- Hub normalization is a pure permutation: every switching weight is one. -/
def normalize_tee {R : Type} [DecidableEq R] (A : R → R → Int)
    (t : R → Int) (ht : ∀ r, Sign (t r)) :
    SignedCongruence (tee A t) (tee A (fun _ => 1)) :=
  tee_congr (SignedCongruence.refl A) t (fun _ => 1) ht (fun _ => Or.inl rfl)

theorem normalize_tee_weight {R : Type} [DecidableEq R] (A : R → R → Int)
    (t : R → Int) (ht : ∀ r, Sign (t r)) (x : Option (Bool × R)) :
    (normalize_tee A t ht).weight x = 1 := by
  cases x <;> rfl

def cee_congr {R S : Type} {A : R → R → Int} {B : S → S → Int}
    (c : SignedCongruence A B) : SignedCongruence (cee A) (cee B) where
  index := liftIndex c.index (fun _ => false)
  weight := liftWeight c.weight
  weight_sign := liftWeight_sign c.weight c.weight_sign
  entry := by
    intro x y
    cases x with
    | none => cases y <;> simp [liftIndex,liftWeight,cee]
    | some x =>
      cases y with
      | none => simp [liftIndex,liftWeight,cee]
      | some y => simpa [liftIndex,liftWeight,cee] using c.entry x.2 y.2

def channelWeight {R : Type} : Option (Bool × R) → Int
  | none => 1
  | some (α,_) => if α then -1 else 1

/-- For Y=-N the high branch is minus switchedCee, not switchedCee itself. -/
def high_channel_congr {R : Type} (A : R → R → Int) :
    SignedCongruence (fun x y => -switchedCee A x y) (cee A) where
  index := IndexIso.refl _
  weight := channelWeight
  weight_sign := by
    intro x
    cases x with
    | none => exact Or.inl rfl
    | some x => cases x.1 <;> simp [channelWeight,Sign]
  entry := by
    intro x y
    cases x with
    | none => cases y <;> simp [IndexIso.refl,channelWeight,switchedCee,cee]
    | some x =>
      rcases x with ⟨α,r⟩
      cases y with
      | none => simp [IndexIso.refl,channelWeight,switchedCee,cee]
      | some y =>
        rcases y with ⟨β,s⟩
        cases α <;> cases β <;> simp [IndexIso.refl,channelWeight,switchedCee,cee]

#print axioms tee_congr
#print axioms normalize_tee
#print axioms cee_congr
#print axioms high_channel_congr

/-- Composition keeps the exact signed-permutation witness. -/
def SignedCongruence.trans {R S U : Type} {A : R → R → Int}
    {B : S → S → Int} {C : U → U → Int}
    (c : SignedCongruence A B) (d : SignedCongruence B C) : SignedCongruence A C where
  index := {
    toFun := fun r => d.index.toFun (c.index.toFun r)
    invFun := fun u => c.index.invFun (d.index.invFun u)
    left_inv := by intro r; simp [d.index.left_inv,c.index.left_inv]
    right_inv := by intro u; simp [c.index.right_inv,d.index.right_inv] }
  weight := fun r => d.weight (c.index.toFun r) * c.weight r
  weight_sign := fun r => mul_sign (d.weight_sign _) (c.weight_sign r)
  entry := by
    intro r s
    rw [d.entry,c.entry]
    simp [Int.mul_assoc,Int.mul_comm,Int.mul_left_comm]

theorem weighted_product_iff (e f x y z : Int) (he : Sign e) (hf : Sign f) :
    (e*x)*(f*y) = e*f*z ↔ x*y = z := by
  rcases he with h | h <;> rcases hf with j | j <;>
    simp [h,j,Int.neg_mul,Int.mul_neg]

structure CoverVertex (R K : Type) where
  index : R
  spin : Spin
  copy : K

def CoverAdj {R K : Type} (A : R → R → Int) (x y : CoverVertex R K) : Prop :=
  x.spin.val * y.spin.val = A x.index y.index

def coverMap {R S K : Type} {A : R → R → Int} {B : S → S → Int}
    (c : SignedCongruence A B) (x : CoverVertex R K) : CoverVertex S K :=
  ⟨c.index.toFun x.index,
   ⟨c.weight x.index * x.spin.val,mul_sign (c.weight_sign _) x.spin.property⟩,x.copy⟩

def coverInv {R S K : Type} {A : R → R → Int} {B : S → S → Int}
    (c : SignedCongruence A B) (y : CoverVertex S K) : CoverVertex R K :=
  ⟨c.index.invFun y.index,
   ⟨c.weight (c.index.invFun y.index) * y.spin.val,
    mul_sign (c.weight_sign _) y.spin.property⟩,y.copy⟩

theorem cover_ext {R K : Type} (x y : CoverVertex R K) (hi : x.index = y.index)
    (hs : x.spin.val = y.spin.val) (hc : x.copy = y.copy) : x = y := by
  cases x; cases y
  simp_all
  exact Subtype.ext hs

/-- A signed congruence preserves the cover graph, with any unchanged copy set K. -/
def cover_iso {R S K : Type} {A : R → R → Int} {B : S → S → Int}
    (c : SignedCongruence A B) :
    GraphIso (CoverVertex R K) (CoverVertex S K) (CoverAdj A) (CoverAdj B) where
  toFun := coverMap c
  invFun := coverInv c
  left_inv := by
    intro x
    apply cover_ext
    · exact c.index.left_inv _
    · simp [coverInv,coverMap,c.index.left_inv,← Int.mul_assoc,sign_square _ (c.weight_sign _)]
    · rfl
  right_inv := by
    intro y
    apply cover_ext
    · exact c.index.right_inv _
    · simp [coverInv,coverMap,← Int.mul_assoc,sign_square _ (c.weight_sign _)]
    · rfl
  map_adj_iff := by
    intro x y
    change x.spin.val*y.spin.val = A x.index y.index ↔
      (c.weight x.index*x.spin.val)*(c.weight y.index*y.spin.val) =
        B (c.index.toFun x.index) (c.index.toFun y.index)
    rw [c.entry]
    exact (weighted_product_iff _ _ _ _ _ (c.weight_sign _) (c.weight_sign _)).symm

def GraphIso.trans {R S U : Type} {A : R → R → Prop} {B : S → S → Prop}
    {C : U → U → Prop} (f : GraphIso R S A B) (g : GraphIso S U B C) :
    GraphIso R U A C where
  toFun := fun x => g.toFun (f.toFun x)
  invFun := fun z => f.invFun (g.invFun z)
  left_inv := by intro x; rw [g.left_inv,f.left_inv]
  right_inv := by intro z; rw [f.right_inv,g.right_inv]
  map_adj_iff := fun x y => (f.map_adj_iff x y).trans (g.map_adj_iff _ _)

def GraphIso.symm {R S : Type} {A : R → R → Prop} {B : S → S → Prop}
    (f : GraphIso R S A B) : GraphIso S R B A where
  toFun := f.invFun
  invFun := f.toFun
  left_inv := f.right_inv
  right_inv := f.left_inv
  map_adj_iff := by
    intro x y
    have h := f.map_adj_iff (f.invFun x) (f.invFun y)
    simpa [f.right_inv] using h.symm

def repMatrix (C : Int → Int → Int) (n W : Nat) : RepIndex n W → RepIndex n W → Int :=
  fun r s => matrix C n W r.val s.val

def twin_cover_iso (n W : Nat) :
    GraphIso (TwinVertex n W) (CoverVertex (RepIndex n W) Bool)
      (TwinAdj n W) (CoverAdj (repMatrix M n W)) where
  toFun := fun x => ⟨x.rep,x.spin,x.copy⟩
  invFun := fun x => ⟨x.index,x.spin,x.copy⟩
  left_inv := fun _ => rfl
  right_inv := fun _ => rfl
  map_adj_iff := fun _ _ => Iff.rfl

/-- Congruence of the actual M matrices gives an isomorphism of literal native graphs. -/
def native_congruence_iso (n W m V : Nat) (hn : 1 ≤ n) (hm : 1 ≤ m)
    (hW : W < 2^n) (hW0 : W ≠ 0) (hV : V < 2^m) (hV0 : V ≠ 0)
    (c : SignedCongruence (repMatrix M n W) (repMatrix M m V)) :
    GraphIso (NativeVertex n W) (NativeVertex m V) (NativeAdj n W) (NativeAdj m V) :=
  GraphIso.trans
    (GraphIso.trans (native_twin_iso n W hn hW hW0) (twin_cover_iso n W))
    (GraphIso.trans (cover_iso c)
      (GraphIso.symm
        (GraphIso.trans (native_twin_iso m V hm hV hV0) (twin_cover_iso m V))))

def lowX (n v : Nat) : Coord n v → Coord n v → Int :=
  fun x y => matrix M (n+1) v (lowValue n v x) (lowValue n v y)
def lowY (n v : Nat) : Coord n v → Coord n v → Int :=
  fun x y => -matrix N (n+1) v (lowValue n v x) (lowValue n v y)
def highX (n v : Nat) : Coord n v → Coord n v → Int :=
  fun x y => matrix M (n+1) (2^n+v) (highValue n v x) (highValue n v y)
def highY (n v : Nat) : Coord n v → Coord n v → Int :=
  fun x y => -matrix N (n+1) (2^n+v) (highValue n v x) (highValue n v y)

def lowX_normalized (n v : Nat) (hv : v < 2^n) (hv0 : v ≠ 0) :
    SignedCongruence (lowX n v) (tee (repMatrix M n v) (fun _ => 1)) := by
  unfold lowX
  rw [(low_block_identity n v hv hv0).1]
  exact normalize_tee _ _ (fun r => sgn_sign _ _ (by simp [bitsOf_length]))

theorem cee_neg {R : Type} (A : R → R → Int) :
    (fun x y => -cee A x y) = cee (fun r s => -A r s) := by
  funext x y
  cases x <;> cases y <;> simp [cee]

def lowY_normalized (n v : Nat) (hv : v < 2^n) (hv0 : v ≠ 0) :
    SignedCongruence (lowY n v) (cee (fun r s => -repMatrix N n v r s)) := by
  have h := congrArg (fun Z : Coord n v → Coord n v → Int => fun x y => -Z x y)
    (low_block_identity n v hv hv0).2
  change lowY n v = (fun x y => -cee (repMatrix N n v) x y) at h
  rw [h,cee_neg]
  exact SignedCongruence.refl _

def highX_normalized (n v : Nat) (hv : v < 2^n) (hv0 : v ≠ 0) :
    SignedCongruence (highX n v) (tee (fun r s => -repMatrix N n v r s) (fun _ => 1)) := by
  unfold highX
  rw [(high_block_identity n v hv hv0).1]
  exact normalize_tee _ _ (fun r => sgn_sign _ _ (by simp [bitsOf_length]))

def highY_normalized (n v : Nat) (hv : v < 2^n) (hv0 : v ≠ 0) :
    SignedCongruence (highY n v) (cee (repMatrix M n v)) := by
  have h := congrArg (fun Z : Coord n v → Coord n v → Int => fun x y => -Z x y)
    (high_block_identity n v hv hv0).2
  change highY n v = (fun x y => -switchedCee (repMatrix M n v) x y) at h
  rw [h]
  exact high_channel_congr _

#print axioms SignedCongruence.trans
#print axioms cover_iso
#print axioms native_congruence_iso
#print axioms lowX_normalized
#print axioms lowY_normalized
#print axioms highX_normalized
#print axioms highY_normalized

def SignedCongruence.symm {R S : Type} {A : R → R → Int} {B : S → S → Int}
    (c : SignedCongruence A B) : SignedCongruence B A where
  index := ⟨c.index.invFun,c.index.toFun,c.index.right_inv,c.index.left_inv⟩
  weight := fun s => c.weight (c.index.invFun s)
  weight_sign := fun s => c.weight_sign _
  entry := by
    intro r s
    have h := c.entry (c.index.invFun r) (c.index.invFun s)
    simp only [c.index.right_inv] at h
    rw [h]
    have hr := c.weight_sign (c.index.invFun r)
    have hs := c.weight_sign (c.index.invFun s)
    rcases hr with hr | hr <;> rcases hs with hs | hs <;> simp [hr,hs]

/-- Total executable inverse candidates; only bounded valid representatives are certified. -/
def lowInverse (n v a : Nat) : Coord n v :=
  if a = 2^n then none
  else if h : Rep n v a then some (false,⟨a,h⟩)
  else if h : Rep n v (a-2^n) then some (true,⟨a-2^n,h⟩)
  else none

def highInverse (n v a : Nat) : Coord n v :=
  if a = v then none
  else if h : Rep n v a then some (false,⟨a,h⟩)
  else if h : Rep n v (a ^^^ v) then some (true,⟨a ^^^ v,h⟩)
  else none

theorem lowValue_inverse (n v a : Nat) (hv : v < 2^n) (hv0 : v ≠ 0)
    (ha : Rep (n+1) v a) : lowValue n v (lowInverse n v a) = a := by
  rcases (low_partition n v a hv hv0).mp ha with h | h | ⟨r,hr,h⟩
  · subst a; simp [lowInverse,lowValue]
  · have hne : a ≠ 2^n := by have := h.1; omega
    simp [lowInverse,hne,h,lowValue,lift]
  · subst a
    have hnot : ¬Rep n v (2^n+r) := by intro h; have := h.1; omega
    simp [lowInverse,hnot,hr,hr.2.1,lowValue,lift]

theorem highValue_inverse (n v a : Nat) (hv : v < 2^n) (hv0 : v ≠ 0)
    (ha : Rep (n+1) (2^n+v) a) : highValue n v (highInverse n v a) = a := by
  rcases (high_partition n v a hv hv0).mp ha with h | h | ⟨r,hr,h⟩
  · subst a; simp [highInverse,highValue]
  · simp [highInverse,rep_ne_label h,h,highValue]
  · subst a
    have hne : r ^^^ v ≠ v := by
      intro e
      have z := congrArg (fun x : Nat => x ^^^ v) e
      exact hr.2.1 (by simpa [Nat.xor_assoc] using z)
    have hnot : ¬Rep n v (r ^^^ v) := by
      intro h
      exact rep_cross hr h (by simp [Nat.xor_assoc])
    simp [highInverse,hne,hnot,hr,highValue,Nat.xor_assoc]

def lowIndexIso (n v : Nat) (hv : v < 2^n) (hv0 : v ≠ 0) :
    IndexIso (Coord n v) (RepIndex (n+1) v) where
  toFun := fun x => ⟨lowValue n v x,lowValue_rep n v hv hv0 x⟩
  invFun := fun a => lowInverse n v a.val
  left_inv := by
    intro x
    apply lowValue_injective n v
    exact lowValue_inverse n v _ hv hv0 (lowValue_rep n v hv hv0 x)
  right_inv := fun a => Subtype.ext (lowValue_inverse n v a.val hv hv0 a.property)

def highIndexIso (n v : Nat) (hv : v < 2^n) (hv0 : v ≠ 0) :
    IndexIso (Coord n v) (RepIndex (n+1) (2^n+v)) where
  toFun := fun x => ⟨highValue n v x,highValue_rep n v hv hv0 x⟩
  invFun := fun a => highInverse n v a.val
  left_inv := by
    intro x
    apply highValue_injective n v
    exact highValue_inverse n v _ hv hv0 (highValue_rep n v hv hv0 x)
  right_inv := fun a => Subtype.ext (highValue_inverse n v a.val hv hv0 a.property)

def reindex_low (n v : Nat) (hv : v < 2^n) (hv0 : v ≠ 0) :
    SignedCongruence (lowX n v) (repMatrix M (n+1) v) where
  index := lowIndexIso n v hv hv0
  weight := fun _ => 1
  weight_sign := fun _ => Or.inl rfl
  entry := by intros; simp [lowIndexIso,lowX,repMatrix]

def reindex_high (n v : Nat) (hv : v < 2^n) (hv0 : v ≠ 0) :
    SignedCongruence (highX n v) (repMatrix M (n+1) (2^n+v)) where
  index := highIndexIso n v hv hv0
  weight := fun _ => 1
  weight_sign := fun _ => Or.inl rfl
  entry := by intros; simp [highIndexIso,highX,repMatrix]

def low_native_matrix_normalized (n v : Nat) (hv : v < 2^n) (hv0 : v ≠ 0) :
    SignedCongruence (repMatrix M (n+1) v) (tee (repMatrix M n v) (fun _ => 1)) :=
  (reindex_low n v hv hv0).symm.trans (lowX_normalized n v hv hv0)

def high_native_matrix_normalized (n v : Nat) (hv : v < 2^n) (hv0 : v ≠ 0) :
    SignedCongruence (repMatrix M (n+1) (2^n+v))
      (tee (fun r s => -repMatrix N n v r s) (fun _ => 1)) :=
  (reindex_high n v hv hv0).symm.trans (highX_normalized n v hv hv0)

/-- Complete explicit low-branch native graph recursion, including coordinate inverse. -/
def low_native_normalized_iso (n v : Nat) (hv : v < 2^n) (hv0 : v ≠ 0) :
    GraphIso (NativeVertex (n+1) v) (CoverVertex (Coord n v) Bool)
      (NativeAdj (n+1) v) (CoverAdj (tee (repMatrix M n v) (fun _ => 1))) := by
  have hv' : v < 2^(n+1) := by rw [Nat.pow_succ]; omega
  exact GraphIso.trans
    (GraphIso.trans (native_twin_iso (n+1) v (by omega) hv' hv0) (twin_cover_iso (n+1) v))
    (cover_iso (low_native_matrix_normalized n v hv hv0))

/-- Complete explicit high-branch native graph recursion; the parent channel is -N. -/
def high_native_normalized_iso (n v : Nat) (hv : v < 2^n) (hv0 : v ≠ 0) :
    GraphIso (NativeVertex (n+1) (2^n+v)) (CoverVertex (Coord n v) Bool)
      (NativeAdj (n+1) (2^n+v))
      (CoverAdj (tee (fun r s => -repMatrix N n v r s) (fun _ => 1))) := by
  have hv' : 2^n+v < 2^(n+1) := by rw [Nat.pow_succ]; omega
  have hv0' : 2^n+v ≠ 0 := by have := Nat.two_pow_pos n; omega
  exact GraphIso.trans
    (GraphIso.trans (native_twin_iso (n+1) (2^n+v) (by omega) hv' hv0')
      (twin_cover_iso (n+1) (2^n+v)))
    (cover_iso (high_native_matrix_normalized n v hv hv0))

#print axioms SignedCongruence.symm
#print axioms lowIndexIso
#print axioms highIndexIso
#print axioms low_native_matrix_normalized
#print axioms high_native_matrix_normalized
#print axioms low_native_normalized_iso
#print axioms high_native_normalized_iso

def reindex_low_Y (n v : Nat) (hv : v < 2^n) (hv0 : v ≠ 0) :
    SignedCongruence (lowY n v) (fun r s => -repMatrix N (n+1) v r s) where
  index := lowIndexIso n v hv hv0
  weight := fun _ => 1
  weight_sign := fun _ => Or.inl rfl
  entry := by intros; simp [lowIndexIso,lowY,repMatrix]

def reindex_high_Y (n v : Nat) (hv : v < 2^n) (hv0 : v ≠ 0) :
    SignedCongruence (highY n v) (fun r s => -repMatrix N (n+1) (2^n+v) r s) where
  index := highIndexIso n v hv hv0
  weight := fun _ => 1
  weight_sign := fun _ => Or.inl rfl
  entry := by intros; simp [highIndexIso,highY,repMatrix]

def low_native_Y_normalized (n v : Nat) (hv : v < 2^n) (hv0 : v ≠ 0) :
    SignedCongruence (fun r s => -repMatrix N (n+1) v r s)
      (cee (fun r s => -repMatrix N n v r s)) :=
  (reindex_low_Y n v hv hv0).symm.trans (lowY_normalized n v hv hv0)

def high_native_Y_normalized (n v : Nat) (hv : v < 2^n) (hv0 : v ≠ 0) :
    SignedCongruence (fun r s => -repMatrix N (n+1) (2^n+v) r s) (cee (repMatrix M n v)) :=
  (reindex_high_Y n v hv hv0).symm.trans (highY_normalized n v hv hv0)

#print axioms low_native_Y_normalized
#print axioms high_native_Y_normalized

end Sounio.ZDSignedCongruence
