import SounioZDRecursion
namespace Sounio.ZDMatrixAssembly
open SounioCDCocycle Sounio.ZDAlgebraBridge Sounio.ZDRecursion
set_option maxRecDepth 4096
set_option maxHeartbeats 8000000

def lift (n : Nat) (α : Bool) (a : Nat) := if α then 2^n+a else a

theorem lift_lt (n a : Nat) (ha : a < 2^n) (α : Bool) :
    lift n α a < 2^(n+1) := by
  cases α <;> simp only [lift, Bool.false_eq_true, ↓reduceIte, Nat.pow_succ] <;> omega

theorem bits_lift (n a : Nat) (ha : a < 2^n) (α : Bool) :
    bitsOf (n+1) (lift n α a) = α :: bitsOf n a := by
  cases α
  · exact bitsOf_lower n a ha
  · exact bitsOf_upper n a ha

theorem xor_lift (n a b : Nat) (ha : a < 2^n) (hb : b < 2^n)
    (α β : Bool) :
    lift n α a ^^^ lift n β b = lift n (xor α β) (a ^^^ b) := by
  apply bitsOf_injective (n+1) _ _
    (Nat.xor_lt_two_pow (lift_lt n a ha α) (lift_lt n b hb β))
    (lift_lt n (a ^^^ b) (Nat.xor_lt_two_pow ha hb) (xor α β))
  rw [← xorL_bitsOf (n+1) _ _ (lift_lt n a ha α) (lift_lt n b hb β),
    bits_lift n a ha α, bits_lift n b hb β,
    bits_lift n (a ^^^ b) (Nat.xor_lt_two_pow ha hb) (xor α β)]
  simp [xorL, xorL_bitsOf n a b ha hb]

def Rep (n W r : Nat) : Prop := r < 2^n ∧ r ≠ 0 ∧ r < r ^^^ W
instance (n W r : Nat) : Decidable (Rep n W r) := inferInstanceAs (Decidable (_ ∧ _ ∧ _))

theorem rep_ne_label {n W r : Nat} (h : Rep n W r) : r ≠ W := by
  intro e
  subst r
  simp [Rep] at h

theorem rep_partner {n W r : Nat} (hW : W < 2^n) (h : Rep n W r) :
    r ^^^ W < 2^n ∧ r ^^^ W ≠ 0 ∧ ¬Rep n W (r ^^^ W) := by
  refine ⟨Nat.xor_lt_two_pow h.1 hW, ?_, ?_⟩
  · intro e
    have := h.2.2
    omega
  · intro h'
    have e : (r ^^^ W) ^^^ W = r := by simp [Nat.xor_assoc]
    have := h'.2.2
    rw [e] at this
    have := h.2.2
    omega

theorem rep_cross {n W r s : Nat} (hr : Rep n W r) (hs : Rep n W s) :
    r ≠ s ^^^ W := by
  intro e
  have es : r ^^^ W = s := by rw [e]; simp [Nat.xor_assoc]
  have := hr.2.2
  have := hs.2.2
  omega

theorem rep_orientation {n W r : Nat} (hW : W < 2^n) (hW0 : W ≠ 0)
    (hr : r < 2^n) (hr0 : r ≠ 0) (hrW : r ≠ W) :
    Rep n W r ∨ Rep n W (r ^^^ W) := by
  have hp := Nat.xor_lt_two_pow hr hW
  have hp0 : r ^^^ W ≠ 0 := fun e => hrW ((xor_zero_iff r W).mp e)
  have hne : r ≠ r ^^^ W := by
    intro e
    have e' := congrArg (fun x : Nat => r ^^^ x) e
    have : W = 0 := by simpa [← Nat.xor_assoc] using e'.symm
    exact hW0 this
  by_cases hlt : r < r ^^^ W
  · exact Or.inl ⟨hr, hr0, hlt⟩
  · exact Or.inr ⟨hp, hp0, by simpa [Nat.xor_assoc] using (show r ^^^ W < r by omega)⟩

#print axioms xor_lift
#print axioms rep_orientation

theorem low_rep_lift (n v r : Nat) (hv : v < 2^n) (hr : r < 2^n) (α : Bool) :
    Rep (n+1) v (lift n α r) ↔ r < r ^^^ v ∧ (r ≠ 0 ∨ α = true) := by
  have hx := xor_lift n r v hr hv α false
  have hp := Nat.xor_lt_two_pow hr hv
  have hg := Nat.two_pow_pos n
  have hbound := lift_lt n r hr α
  cases α <;> simp [lift] at hx hbound ⊢
  · simp [Rep, hbound, and_comm]
  · simp only [Rep, hbound, true_and, hx]
    omega

theorem low_partition (n v x : Nat) (hv : v < 2^n) (hv0 : v ≠ 0) :
    Rep (n+1) v x ↔
      x = 2^n ∨ Rep n v x ∨ ∃ r, Rep n v r ∧ x = 2^n+r := by
  constructor
  · intro h
    by_cases hx : x < 2^n
    · have e := (low_rep_lift n v x hv hx false).mp h
      exact Or.inr (Or.inl ⟨hx, by simpa using e.2, e.1⟩)
    · have hr : x-2^n < 2^n := by have := h.1; rw [Nat.pow_succ] at this; omega
      have he : lift n true (x-2^n) = x := by simp only [lift,↓reduceIte]; omega
      have e := (low_rep_lift n v (x-2^n) hv hr true).mp (by rw [he]; exact h)
      by_cases hz : x-2^n = 0
      · exact Or.inl (by omega)
      · exact Or.inr (Or.inr ⟨x-2^n,⟨hr,hz,e.1⟩,by omega⟩)
  · intro h
    rcases h with h | h | ⟨r,hr,rfl⟩
    · subst x
      have h0 := Nat.two_pow_pos n
      have e := (low_rep_lift n v 0 hv h0 true).mpr
        ⟨by simpa using (show 0 < v by omega), Or.inr rfl⟩
      simpa [lift] using e
    · exact (low_rep_lift n v x hv h.1 false).mpr ⟨h.2.2,Or.inl h.2.1⟩
    · exact (low_rep_lift n v r hv hr.1 true).mpr ⟨hr.2.2,Or.inl hr.2.1⟩

theorem high_reps (n v x : Nat) (hv : v < 2^n) :
    Rep (n+1) (2^n+v) x ↔ x < 2^n ∧ x ≠ 0 := by
  constructor
  · intro h
    refine ⟨?_,h.2.1⟩
    by_cases hx : x < 2^n
    · exact hx
    have hr : x-2^n < 2^n := by have := h.1; rw [Nat.pow_succ] at this; omega
    have he : 2^n+(x-2^n) = x := by omega
    have e := xor_lift n (x-2^n) v hr hv true true
    simp only [lift,Bool.xor_self,↓reduceIte,Bool.false_eq_true] at e
    rw [he] at e
    have hp := Nat.xor_lt_two_pow hr hv
    have := h.2.2
    omega
  · rintro ⟨hx,hx0⟩
    have e := xor_lift n x v hx hv false true
    simp only [lift,Bool.false_xor,↓reduceIte,Bool.false_eq_true] at e
    refine ⟨lift_lt n x hx false,hx0,?_⟩
    rw [e]
    omega

theorem high_partition (n v x : Nat) (hv : v < 2^n) (hv0 : v ≠ 0) :
    Rep (n+1) (2^n+v) x ↔
      x = v ∨ Rep n v x ∨ ∃ r, Rep n v r ∧ x = r ^^^ v := by
  rw [high_reps n v x hv]
  constructor
  · rintro ⟨hx,hx0⟩
    by_cases hxv : x = v
    · exact Or.inl hxv
    · rcases rep_orientation hv hv0 hx hx0 hxv with h | h
      · exact Or.inr (Or.inl h)
      · exact Or.inr (Or.inr ⟨x ^^^ v,h,by simp [Nat.xor_assoc]⟩)
  · intro h
    rcases h with rfl | h | ⟨r,hr,rfl⟩
    · exact ⟨hv,hv0⟩
    · exact ⟨h.1,h.2.1⟩
    · exact ⟨(rep_partner hv hr).1,(rep_partner hv hr).2.1⟩

#print axioms low_partition
#print axioms high_partition

def lowList (n : Nat) (rs : List Nat) := 2^n :: (rs ++ rs.map (fun r => 2^n+r))
def highList (v : Nat) (rs : List Nat) := v :: (rs ++ rs.map (fun r => r ^^^ v))

theorem mem_lowList (n v x : Nat) (hv : v < 2^n) (hv0 : v ≠ 0)
    (rs : List Nat) (hmem : ∀ r, r ∈ rs ↔ Rep n v r) :
    x ∈ lowList n rs ↔ Rep (n+1) v x := by
  rw [low_partition n v x hv hv0]
  simp [lowList, hmem, eq_comm]

theorem mem_highList (n v x : Nat) (hv : v < 2^n) (hv0 : v ≠ 0)
    (rs : List Nat) (hmem : ∀ r, r ∈ rs ↔ Rep n v r) :
    x ∈ highList v rs ↔ Rep (n+1) (2^n+v) x := by
  rw [high_partition n v x hv hv0]
  simp [highList, hmem, eq_comm]

theorem nodup_double (h : Nat) (f : Nat → Nat) (rs : List Nat)
    (hn : rs.Nodup) (hf : ∀ a b, f a = f b → a = b)
    (hh : ∀ a, a ∈ rs → h ≠ a ∧ h ≠ f a)
    (hc : ∀ a, a ∈ rs → ∀ b, b ∈ rs → a ≠ f b) :
    (h :: (rs ++ rs.map f)).Nodup := by
  rw [List.nodup_cons, List.nodup_append]
  refine ⟨?_,hn,?_,?_⟩
  · simp only [List.mem_append, List.mem_map]
    rintro (hm | ⟨a,ha,ea⟩)
    · exact (hh h hm).1 rfl
    · exact (hh a ha).2 ea.symm
  · apply List.pairwise_map.mpr
    exact hn.imp (fun hab e => hab (hf _ _ e))
  · intro a ha fb hfb
    obtain ⟨b,hb,rfl⟩ := List.mem_map.mp hfb
    exact hc a ha b hb

theorem nodup_lowList (n v : Nat) (rs : List Nat)
    (hn : rs.Nodup) (hm : ∀ r, r ∈ rs → Rep n v r) :
    (lowList n rs).Nodup := by
  apply nodup_double _ _ rs hn
  · intro a b h; omega
  · intro a ha
    have h := hm a ha
    exact ⟨by have := h.1; omega, by have := h.2.1; omega⟩
  · intro a ha b hb
    have := (hm a ha).1
    omega

theorem nodup_highList (n v : Nat) (hv : v < 2^n) (rs : List Nat)
    (hn : rs.Nodup) (hm : ∀ r, r ∈ rs → Rep n v r) :
    (highList v rs).Nodup := by
  apply nodup_double _ _ rs hn
  · intro a b h
    have := congrArg (fun x : Nat => x ^^^ v) h
    simpa [Nat.xor_assoc] using this
  · intro a ha
    have h := hm a ha
    refine ⟨Ne.symm (rep_ne_label h),?_⟩
    intro e
    have := congrArg (fun x : Nat => x ^^^ v) e
    have : a = 0 := by simpa [Nat.xor_assoc] using this.symm
    exact h.2.1 this
  · intro a ha b hb
    exact rep_cross (hm a ha) (hm b hb)

def representatives : Nat → Nat → List Nat
  | 0, _ => []
  | n+1, W =>
    if W < 2^n then
      lowList n (representatives n W)
    else if W = 2^n then
      List.range' 1 (2^n-1)
    else
      highList (W-2^n) (representatives n (W-2^n))


/-- A complete, duplicate-free enumeration with its cardinality, at every width. -/
theorem representatives_certificate : ∀ n W, W ≠ 0 → W < 2^n →
    (representatives n W).Nodup ∧
    (∀ r, r ∈ representatives n W ↔ Rep n W r) ∧
    2*(representatives n W).length+2 = 2^n := by
  intro n
  induction n with
  | zero =>
    intro W h0 hW
    simp only [Nat.pow_zero] at hW
    omega
  | succ n ih =>
    intro W h0 hW
    have hg := Nat.two_pow_pos n
    by_cases hlo : W < 2^n
    · obtain ⟨hnd,hm,hlen⟩ := ih W h0 hlo
      simp only [representatives,hlo,↓reduceIte]
      refine ⟨nodup_lowList n W _ hnd (fun r h => (hm r).mp h),
        fun r => mem_lowList n W r hlo h0 _ hm,?_⟩
      simp only [lowList,List.length_cons,List.length_append,List.length_map,Nat.pow_succ]
      omega
    · by_cases heq : W = 2^n
      · subst W
        simp only [representatives,Nat.lt_irrefl,↓reduceIte]
        refine ⟨List.nodup_range',?_,?_⟩
        · intro r
          have hh := high_reps n 0 r hg
          simp only [Nat.add_zero] at hh
          rw [hh,List.mem_range']
          simp only [Nat.one_mul]
          constructor
          · rintro ⟨i,hi,rfl⟩; omega
          · rintro ⟨hr,hr0⟩
            exact ⟨r-1,by omega,by omega⟩
        · simp only [List.length_range',Nat.pow_succ]
          omega
      · have hv : W-2^n < 2^n := by rw [Nat.pow_succ] at hW; omega
        have hv0 : W-2^n ≠ 0 := by omega
        have he : 2^n+(W-2^n) = W := by omega
        obtain ⟨hnd,hm,hlen⟩ := ih (W-2^n) hv0 hv
        simp only [representatives,hlo,heq,↓reduceIte]
        refine ⟨nodup_highList n (W-2^n) hv _ hnd (fun r h => (hm r).mp h),?_,?_⟩
        · intro r
          rw [mem_highList n (W-2^n) r hv hv0 _ hm,he]
        · simp only [highList,List.length_cons,List.length_append,List.length_map,Nat.pow_succ]
          omega

theorem representatives_length (n W : Nat) (hW0 : W ≠ 0) (hW : W < 2^(n+1)) :
    (representatives (n+1) W).length = 2^n-1 := by
  have h := (representatives_certificate (n+1) W hW0 hW).2.2
  rw [Nat.pow_succ] at h
  omega

abbrev RepIndex (n W : Nat) := {r : Nat // Rep n W r}
abbrev Coord (n W : Nat) := Option (Bool × RepIndex n W)

def lowValue (n v : Nat) : Coord n v → Nat
  | none => 2^n
  | some (α,r) => lift n α r.val

def highValue (n v : Nat) : Coord n v → Nat
  | none => v
  | some (α,r) => if α then r.val ^^^ v else r.val

theorem lowValue_rep (n v : Nat) (hv : v < 2^n) (hv0 : v ≠ 0) (x : Coord n v) :
    Rep (n+1) v (lowValue n v x) := by
  cases x with
  | none => exact (low_partition n v _ hv hv0).mpr (Or.inl rfl)
  | some x =>
    exact (low_rep_lift n v x.2.val hv x.2.property.1 x.1).mpr
      ⟨x.2.property.2.2,Or.inl x.2.property.2.1⟩

theorem highValue_rep (n v : Nat) (hv : v < 2^n) (hv0 : v ≠ 0) (x : Coord n v) :
    Rep (n+1) (2^n+v) (highValue n v x) := by
  apply (high_partition n v _ hv hv0).mpr
  cases x with
  | none => exact Or.inl rfl
  | some x =>
    rcases x with ⟨α,r⟩
    cases α
    · exact Or.inr (Or.inl r.property)
    · exact Or.inr (Or.inr ⟨r.val,r.property,rfl⟩)

theorem lowValue_surjective (n v a : Nat) (hv : v < 2^n) (hv0 : v ≠ 0)
    (ha : Rep (n+1) v a) : ∃ x : Coord n v, lowValue n v x = a := by
  rcases (low_partition n v a hv hv0).mp ha with h | h | ⟨r,hr,h⟩
  · exact ⟨none,h.symm⟩
  · exact ⟨some (false,⟨a,h⟩),rfl⟩
  · exact ⟨some (true,⟨r,hr⟩),h.symm⟩

theorem highValue_surjective (n v a : Nat) (hv : v < 2^n) (hv0 : v ≠ 0)
    (ha : Rep (n+1) (2^n+v) a) : ∃ x : Coord n v, highValue n v x = a := by
  rcases (high_partition n v a hv hv0).mp ha with h | h | ⟨r,hr,h⟩
  · exact ⟨none,h.symm⟩
  · exact ⟨some (false,⟨a,h⟩),rfl⟩
  · exact ⟨some (true,⟨r,hr⟩),h.symm⟩

theorem lowValue_injective (n v : Nat) (x y : Coord n v)
    (h : lowValue n v x = lowValue n v y) : x = y := by
  cases x with
  | none =>
    cases y with
    | none => rfl
    | some y =>
      rcases y with ⟨β,s⟩
      have hs := s.property
      cases β <;> simp only [lowValue,lift,↓reduceIte,Bool.false_eq_true] at h
      all_goals have := hs.1; have := hs.2.1; omega
  | some x =>
    rcases x with ⟨α,r⟩
    cases y with
    | none =>
      have hr := r.property
      cases α <;> simp only [lowValue,lift,↓reduceIte,Bool.false_eq_true] at h
      all_goals have := hr.1; have := hr.2.1; omega
    | some y =>
      rcases y with ⟨β,s⟩
      have hr := r.property.1
      have hs := s.property.1
      cases α <;> cases β <;> simp only [lowValue,lift,↓reduceIte,Bool.false_eq_true] at h
      all_goals first
        | have e : r = s := Subtype.ext (by omega)
          subst s; rfl
        | omega

theorem highValue_injective (n v : Nat) (x y : Coord n v)
    (h : highValue n v x = highValue n v y) : x = y := by
  have hn (r : RepIndex n v) : v ≠ r.val ^^^ v := by
    intro e
    have := congrArg (fun x : Nat => x ^^^ v) e
    exact r.property.2.1 (by simpa [Nat.xor_assoc] using this.symm)
  cases x with
  | none =>
    cases y with
    | none => rfl
    | some y =>
      rcases y with ⟨β,s⟩
      cases β <;> simp only [highValue,↓reduceIte,Bool.false_eq_true] at h
      · exact False.elim ((rep_ne_label s.property) h.symm)
      · exact False.elim (hn s h)
  | some x =>
    rcases x with ⟨α,r⟩
    cases y with
    | none =>
      cases α <;> simp only [highValue,↓reduceIte,Bool.false_eq_true] at h
      · exact False.elim ((rep_ne_label r.property) h)
      · exact False.elim (hn r h.symm)
    | some y =>
      rcases y with ⟨β,s⟩
      cases α <;> cases β <;> simp only [highValue,↓reduceIte,Bool.false_eq_true] at h
      · have e : r = s := Subtype.ext h
        subst s; rfl
      · exact False.elim (rep_cross r.property s.property h)
      · exact False.elim (rep_cross s.property r.property h.symm)
      · have e' := congrArg (fun x : Nat => x ^^^ v) h
        have e : r = s := Subtype.ext (by simpa [Nat.xor_assoc] using e')
        subst s; rfl

#print axioms representatives_certificate
#print axioms representatives_length
#print axioms lowValue_surjective
#print axioms lowValue_injective
#print axioms highValue_surjective
#print axioms highValue_injective

/-- The actual bounded basis pair; neither signs nor entries are recurrence oracles. -/
def basisPair (n W a : Nat) := (bitsOf n a, bitsOf n (a ^^^ W))
def matrix (channel : Int → Int → Int) (n W a b : Nat) : Int :=
  if a = b then 0 else
    channel (pairT (basisPair n W a).1 (basisPair n W a).2
      (basisPair n W b).1 (basisPair n W b).2)
      (pairQ (basisPair n W a).1 (basisPair n W a).2
        (basisPair n W b).1 (basisPair n W b).2)

theorem pair_symmetry (d : Generic) :
    pairT d.b d.bp d.a d.ap = oldT d ∧
    pairQ d.b d.bp d.a d.ap = oldQ d := by
  have hBA := antisym d.b d.a d.lb d.zb d.za (Ne.symm d.ab)
  have hBB := antisym d.bp d.ap (d.lbp.trans d.lap.symm) d.zbp d.zap (Ne.symm d.apbp)
  have hBC := antisym d.bp d.a d.lbp d.zbp d.za (Ne.symm d.abp)
  have hBD := antisym d.b d.ap (d.lb.trans d.lap.symm) d.zb d.zap (Ne.symm d.apb)
  simp [pairT,pairQ,oldT,oldQ,hBA,hBB,hBC,hBD,
    Int.neg_mul,Int.mul_neg,Int.mul_comm,Int.mul_left_comm,Int.mul_assoc]

def repGeneric (n v : Nat) (hv : v < 2^n) (r s : RepIndex n v)
    (hne : r ≠ s) : Generic :=
  genericOfNat n v r.val s.val hv r.property.1 s.property.1
    r.property.2.1 s.property.2.1 (rep_ne_label r.property) (rep_ne_label s.property)
    (fun h => hne (Subtype.ext h)) (rep_cross r.property s.property)

theorem matrix_symmetry (channel : Int → Int → Int) (n v : Nat) (hv : v < 2^n)
    (r s : RepIndex n v) :
    matrix channel n v r.val s.val = matrix channel n v s.val r.val := by
  by_cases h : r = s
  · subst s; rfl
  · have hval : r.val ≠ s.val := fun e => h (Subtype.ext e)
    have hs := pair_symmetry (repGeneric n v hv r s h)
    dsimp [repGeneric,genericOfNat,oldT,oldQ] at hs
    simp only [matrix,hval,Ne.symm hval,↓reduceIte,basisPair]
    rw [hs.1,hs.2]

theorem basisPair_low_some (n v : Nat) (hv : v < 2^n) (α : Bool) (r : RepIndex n v) :
    basisPair (n+1) v (lowValue n v (some (α,r))) =
      (α :: bitsOf n r.val, α :: bitsOf n (r.val ^^^ v)) := by
  have hx := xor_lift n r.val v r.property.1 hv α false
  simp only [Bool.xor_false,lift,Bool.false_eq_true,↓reduceIte] at hx
  unfold basisPair lowValue
  rw [bits_lift n r.val r.property.1 α]
  change (α :: bitsOf n r.val, bitsOf (n+1) ((if α then 2^n+r.val else r.val) ^^^ v)) = _
  rw [hx]
  exact congrArg (fun z => (α :: bitsOf n r.val,z))
    (bits_lift n (r.val ^^^ v) (Nat.xor_lt_two_pow r.property.1 hv) α)

theorem basisPair_high_some (n v : Nat) (hv : v < 2^n) (α : Bool) (r : RepIndex n v) :
    basisPair (n+1) (2^n+v) (highValue n v (some (α,r))) =
      (false :: pick α (bitsOf n r.val) (bitsOf n (r.val ^^^ v)),
       true :: pick (!α) (bitsOf n r.val) (bitsOf n (r.val ^^^ v))) := by
  have hp := Nat.xor_lt_two_pow r.property.1 hv
  have hx := xor_lift n r.val v r.property.1 hv false true
  have hy := xor_lift n (r.val ^^^ v) v hp hv false true
  simp only [lift,Bool.false_xor,↓reduceIte,Bool.false_eq_true] at hx hy
  have hcancel : (r.val ^^^ v) ^^^ v = r.val := by simp [Nat.xor_assoc]
  rw [hcancel] at hy
  cases α <;> simp [basisPair,highValue,pick,hx,hy,
    bitsOf_lower n r.val r.property.1,bitsOf_lower n (r.val ^^^ v) hp,
    bitsOf_upper n r.val r.property.1,bitsOf_upper n (r.val ^^^ v) hp]

theorem basisPair_low_hub (n v : Nat) (hv : v < 2^n) :
    basisPair (n+1) v (lowValue n v none) =
      (true :: bitsOf n 0, true :: bitsOf n v) := by
  have hg := Nat.two_pow_pos n
  have hx := xor_lift n 0 v hg hv true false
  simp [lift] at hx
  simp [basisPair,lowValue,hx,bitsOf_upper n v hv,
    show bitsOf (n+1) (2^n) = true :: bitsOf n 0 by simpa using bitsOf_upper n 0 hg]

theorem basisPair_high_hub (n v : Nat) (hv : v < 2^n) :
    basisPair (n+1) (2^n+v) (highValue n v none) =
      (false :: bitsOf n v, true :: bitsOf n 0) := by
  have hg := Nat.two_pow_pos n
  have hx := xor_lift n v v hv hv false true
  simp [lift] at hx
  simp [basisPair,highValue,hx,bitsOf_lower n v hv,
    show bitsOf (n+1) (2^n) = true :: bitsOf n 0 by simpa using bitsOf_upper n 0 hg]

def tee {R : Type} [DecidableEq R] (Z : R → R → Int) (t : R → Int) :
    Option (Bool × R) → Option (Bool × R) → Int
  | none,none => 0
  | none,some (α,r) | some (α,r),none => if α then -t r else t r
  | some (α,r),some (β,s) => if r = s then (if α = β then 0 else 1) else Z r s

def cee {R : Type} (Z : R → R → Int) :
    Option (Bool × R) → Option (Bool × R) → Int
  | some (_,r),some (_,s) => Z r s
  | _,_ => 0

def switchedCee {R : Type} (Z : R → R → Int) :
    Option (Bool × R) → Option (Bool × R) → Int
  | some (α,r),some (β,s) => if xor α β then Z r s else -Z r s
  | _,_ => 0

theorem lowValue_eq_iff (n v : Nat) (x y : Coord n v) :
    lowValue n v x = lowValue n v y ↔ x = y :=
  ⟨lowValue_injective n v x y,fun h => congrArg (lowValue n v) h⟩

theorem highValue_eq_iff (n v : Nat) (x y : Coord n v) :
    highValue n v x = highValue n v y ↔ x = y :=
  ⟨highValue_injective n v x y,fun h => congrArg (highValue n v) h⟩

#print axioms matrix_symmetry
#print axioms basisPair_low_some
#print axioms basisPair_high_some

theorem low_matrix_sym (C : Int → Int → Int) (n v : Nat)
    (hv : v < 2^n) (hv0 : v ≠ 0) (x y : Coord n v) :
    matrix C (n+1) v (lowValue n v x) (lowValue n v y) =
    matrix C (n+1) v (lowValue n v y) (lowValue n v x) :=
  matrix_symmetry C (n+1) v (lift_lt n v hv false)
    ⟨_,lowValue_rep n v hv hv0 x⟩ ⟨_,lowValue_rep n v hv hv0 y⟩

theorem high_matrix_sym (C : Int → Int → Int) (n v : Nat)
    (hv : v < 2^n) (hv0 : v ≠ 0) (x y : Coord n v) :
    matrix C (n+1) (2^n+v) (highValue n v x) (highValue n v y) =
    matrix C (n+1) (2^n+v) (highValue n v y) (highValue n v x) :=
  matrix_symmetry C (n+1) (2^n+v) (lift_lt n v hv true)
    ⟨_,highValue_rep n v hv hv0 x⟩ ⟨_,highValue_rep n v hv hv0 y⟩

theorem low_matrix_generic (n v : Nat) (hv : v < 2^n)
    (r s : RepIndex n v) (hne : r ≠ s) (α β : Bool) :
    matrix M (n+1) v (lowValue n v (some (α,r))) (lowValue n v (some (β,s))) =
      matrix M n v r.val s.val ∧
    matrix N (n+1) v (lowValue n v (some (α,r))) (lowValue n v (some (β,s))) =
      matrix N n v r.val s.val := by
  have he : (some (α,r) : Coord n v) ≠ some (β,s) := by
    intro e; exact hne (Option.some.inj e |> congrArg Prod.snd)
  have hvv : r.val ≠ s.val := fun e => hne (Subtype.ext e)
  simp only [matrix,lowValue_eq_iff,he,hvv,↓reduceIte,
    basisPair_low_some n v hv]
  exact low_channels (repGeneric n v hv r s hne) α β

theorem high_matrix_generic (n v : Nat) (hv : v < 2^n)
    (r s : RepIndex n v) (hne : r ≠ s) (α β : Bool) :
    matrix M (n+1) (2^n+v) (highValue n v (some (α,r))) (highValue n v (some (β,s))) =
      -matrix N n v r.val s.val ∧
    matrix N (n+1) (2^n+v) (highValue n v (some (α,r))) (highValue n v (some (β,s))) =
      (if xor α β then matrix M n v r.val s.val else -matrix M n v r.val s.val) := by
  have he : (some (α,r) : Coord n v) ≠ some (β,s) := by
    intro e; exact hne (Option.some.inj e |> congrArg Prod.snd)
  have hvv : r.val ≠ s.val := fun e => hne (Subtype.ext e)
  simp only [matrix,highValue_eq_iff,he,hvv,↓reduceIte,
    basisPair_high_some n v hv]
  exact high_channels (repGeneric n v hv r s hne) α β

theorem low_matrix_pair (n v : Nat) (hv : v < 2^n) (hv0 : v ≠ 0) (r : RepIndex n v) :
    matrix M (n+1) v (lowValue n v (some (false,r))) (lowValue n v (some (true,r))) = 1 ∧
    matrix N (n+1) v (lowValue n v (some (false,r))) (lowValue n v (some (true,r))) = 0 := by
  have hne : bitsOf n r.val ≠ bitsOf n (r.val ^^^ v) := by
    intro e
    have h := bitsOf_injective n _ _ r.property.1 (Nat.xor_lt_two_pow r.property.1 hv) e
    have := r.property.2.2
    omega
  have h := low_pair (bitsOf n r.val) (bitsOf n (r.val ^^^ v))
    (by simp [bitsOf_length]) (isZ_bitsOf_false n _ r.property.1 r.property.2.1)
    (isZ_bitsOf_false n _ (Nat.xor_lt_two_pow r.property.1 hv) (rep_partner hv r.property).2.1) hne
  simp only [matrix,lowValue_eq_iff,Option.some.injEq,Prod.mk.injEq,Bool.false_eq_true,
    false_and,↓reduceIte,basisPair_low_some n v hv]
  rw [h.1,h.2]
  decide

theorem high_matrix_pair (n v : Nat) (hv : v < 2^n) (r : RepIndex n v) :
    matrix M (n+1) (2^n+v) (highValue n v (some (false,r))) (highValue n v (some (true,r))) = 1 ∧
    matrix N (n+1) (2^n+v) (highValue n v (some (false,r))) (highValue n v (some (true,r))) = 0 := by
  have h := high_pair (bitsOf n r.val) (bitsOf n (r.val ^^^ v))
    (by simp [bitsOf_length]) (isZ_bitsOf_false n _ r.property.1 r.property.2.1)
    (isZ_bitsOf_false n _ (Nat.xor_lt_two_pow r.property.1 hv) (rep_partner hv r.property).2.1)
  simp only [matrix,highValue_eq_iff,Option.some.injEq,Prod.mk.injEq,Bool.false_eq_true,
    false_and,↓reduceIte,basisPair_high_some n v hv,pick,Bool.not_false,Bool.not_true]
  rw [h.1,h.2]
  decide

theorem low_matrix_hub (n v : Nat) (hv : v < 2^n) (hv0 : v ≠ 0)
    (r : RepIndex n v) (α : Bool) :
    matrix M (n+1) v (lowValue n v none) (lowValue n v (some (α,r))) =
      (if α then -sgn (bitsOf n v) (bitsOf n (r.val ^^^ v))
       else sgn (bitsOf n v) (bitsOf n (r.val ^^^ v))) ∧
    matrix N (n+1) v (lowValue n v none) (lowValue n v (some (α,r))) = 0 := by
  let d := hubOfNat n v r.val hv r.property.1 hv0 r.property.2.1 (Ne.symm (rep_ne_label r.property))
  have h := low_hub d α
  dsimp [d,hubOfNat] at h
  rw [Nat.xor_comm v r.val] at h
  simp only [matrix,lowValue_eq_iff,Option.noConfusion,↓reduceIte,
    basisPair_low_hub n v hv,basisPair_low_some n v hv]
  rw [h.1,h.2]
  simp [M,N]

theorem high_matrix_hub (n v : Nat) (hv : v < 2^n) (hv0 : v ≠ 0)
    (r : RepIndex n v) (α : Bool) :
    matrix M (n+1) (2^n+v) (highValue n v none) (highValue n v (some (α,r))) =
      (if α then -sgn (bitsOf n v) (bitsOf n r.val)
       else sgn (bitsOf n v) (bitsOf n r.val)) ∧
    matrix N (n+1) (2^n+v) (highValue n v none) (highValue n v (some (α,r))) = 0 := by
  let d := hubOfNat n v r.val hv r.property.1 hv0 r.property.2.1 (Ne.symm (rep_ne_label r.property))
  have h := high_hub d α
  dsimp [d,hubOfNat] at h
  rw [Nat.xor_comm v r.val] at h
  simp only [matrix,highValue_eq_iff,Option.noConfusion,↓reduceIte,
    basisPair_high_hub n v hv,basisPair_high_some n v hv]
  rw [h.1,h.2]
  simp [M,N]

#print axioms low_matrix_generic
#print axioms high_matrix_generic
#print axioms low_matrix_pair
#print axioms high_matrix_pair
#print axioms low_matrix_hub
#print axioms high_matrix_hub

@[simp] theorem matrix_diagonal (C : Int → Int → Int) (n W a : Nat) :
    matrix C n W a a = 0 := by simp [matrix]

/-- Full low-label block identity, including the zero diagonal and both orientations. -/
theorem low_matrix (n v : Nat) (hv : v < 2^n) (hv0 : v ≠ 0) (x y : Coord n v) :
    matrix M (n+1) v (lowValue n v x) (lowValue n v y) =
      tee (fun r s : RepIndex n v => matrix M n v r.val s.val)
        (fun r => sgn (bitsOf n v) (bitsOf n (r.val ^^^ v))) x y ∧
    matrix N (n+1) v (lowValue n v x) (lowValue n v y) =
      cee (fun r s : RepIndex n v => matrix N n v r.val s.val) x y := by
  cases x with
  | none =>
    cases y with
    | none => simp [tee,cee]
    | some y =>
      exact low_matrix_hub n v hv hv0 y.2 y.1
  | some x =>
    rcases x with ⟨α,r⟩
    cases y with
    | none =>
      rw [low_matrix_sym M n v hv hv0,low_matrix_sym N n v hv hv0]
      exact low_matrix_hub n v hv hv0 r α
    | some y =>
      rcases y with ⟨β,s⟩
      by_cases he : r = s
      · subst s
        cases α <;> cases β
        · simp [tee,cee]
        · simpa [tee,cee] using low_matrix_pair n v hv hv0 r
        · rw [low_matrix_sym M n v hv hv0,low_matrix_sym N n v hv hv0]
          simpa [tee,cee] using low_matrix_pair n v hv hv0 r
        · simp [tee,cee]
      · simpa [tee,cee,he] using low_matrix_generic n v hv r s he α β

/-- Full high-label block identity. The second channel retains its copy switch. -/
theorem high_matrix (n v : Nat) (hv : v < 2^n) (hv0 : v ≠ 0) (x y : Coord n v) :
    matrix M (n+1) (2^n+v) (highValue n v x) (highValue n v y) =
      tee (fun r s : RepIndex n v => -matrix N n v r.val s.val)
        (fun r => sgn (bitsOf n v) (bitsOf n r.val)) x y ∧
    matrix N (n+1) (2^n+v) (highValue n v x) (highValue n v y) =
      switchedCee (fun r s : RepIndex n v => matrix M n v r.val s.val) x y := by
  cases x with
  | none =>
    cases y with
    | none => simp [tee,switchedCee]
    | some y =>
      exact high_matrix_hub n v hv hv0 y.2 y.1
  | some x =>
    rcases x with ⟨α,r⟩
    cases y with
    | none =>
      rw [high_matrix_sym M n v hv hv0,high_matrix_sym N n v hv hv0]
      exact high_matrix_hub n v hv hv0 r α
    | some y =>
      rcases y with ⟨β,s⟩
      by_cases he : r = s
      · subst s
        cases α <;> cases β
        · simp [tee,switchedCee]
        · simpa [tee,switchedCee] using high_matrix_pair n v hv r
        · rw [high_matrix_sym M n v hv hv0,high_matrix_sym N n v hv hv0]
          simpa [tee,switchedCee] using high_matrix_pair n v hv r
        · simp [tee,switchedCee]
      · simpa [tee,switchedCee,he] using high_matrix_generic n v hv r s he α β

/-- At a reset label, the first channel is the negative complete graph and the second is zero. -/
theorem reset_matrix (n : Nat) (r s : RepIndex (n+1) (2^n)) :
    matrix M (n+1) (2^n) r.val s.val = (if r = s then 0 else -1) ∧
    matrix N (n+1) (2^n) r.val s.val = 0 := by
  have hg := Nat.two_pow_pos n
  have hr := (high_reps n 0 r.val hg).mp (by simpa using r.property)
  have hs := (high_reps n 0 s.val hg).mp (by simpa using s.property)
  by_cases he : r = s
  · subst s; simp
  · have hev : r.val ≠ s.val := fun e => he (Subtype.ext e)
    have hx (a : Nat) (ha : a < 2^n) :
        basisPair (n+1) (2^n) a = (false::bitsOf n a,true::bitsOf n a) := by
      have e := xor_lift n a 0 ha hg false true
      simp [lift] at e
      simp [basisPair,e,bitsOf_lower n a ha,bitsOf_upper n a ha]
    have h := reset_entry (bitsOf n r.val) (bitsOf n s.val)
      (by simp [bitsOf_length]) (isZ_bitsOf_false n _ hr.1 hr.2)
      (isZ_bitsOf_false n _ hs.1 hs.2)
      (fun e => hev (bitsOf_injective n _ _ hr.1 hs.1 e))
    simp only [matrix,hev,he,↓reduceIte,hx r.val hr.1,hx s.val hs.1]
    rw [h.1,h.2]
    decide

#print axioms low_matrix
#print axioms high_matrix
#print axioms reset_matrix

/-- Function equality on the entire explicit coordinate type. -/
theorem low_block_identity (n v : Nat) (hv : v < 2^n) (hv0 : v ≠ 0) :
    (fun x y : Coord n v => matrix M (n+1) v (lowValue n v x) (lowValue n v y)) =
      tee (fun r s : RepIndex n v => matrix M n v r.val s.val)
        (fun r => sgn (bitsOf n v) (bitsOf n (r.val ^^^ v))) ∧
    (fun x y : Coord n v => matrix N (n+1) v (lowValue n v x) (lowValue n v y)) =
      cee (fun r s : RepIndex n v => matrix N n v r.val s.val) :=
  ⟨funext (fun x => funext (fun y => (low_matrix n v hv hv0 x y).1)),
   funext (fun x => funext (fun y => (low_matrix n v hv hv0 x y).2))⟩

theorem high_block_identity (n v : Nat) (hv : v < 2^n) (hv0 : v ≠ 0) :
    (fun x y : Coord n v =>
      matrix M (n+1) (2^n+v) (highValue n v x) (highValue n v y)) =
      tee (fun r s : RepIndex n v => -matrix N n v r.val s.val)
        (fun r => sgn (bitsOf n v) (bitsOf n r.val)) ∧
    (fun x y : Coord n v =>
      matrix N (n+1) (2^n+v) (highValue n v x) (highValue n v y)) =
      switchedCee (fun r s : RepIndex n v => matrix M n v r.val s.val) :=
  ⟨funext (fun x => funext (fun y => (high_matrix n v hv hv0 x y).1)),
   funext (fun x => funext (fun y => (high_matrix n v hv hv0 x y).2))⟩

theorem sign_eq_M (t q u : Int) (hu : Sign u) :
    u = M t q ↔ q = -1 ∧ u = t := by
  by_cases hq : q = -1
  · simp [M,hq]
  · rcases hu with rfl | rfl <;> simp [M,hq]

/-- The assembled M matrix decides both ordered native products on representatives.
This is the signed-cover adjacency criterion; the full native twin blow-up is separate. -/
theorem native_matrix_adjacency (n W : Nat) (hn : 1 ≤ n) (hW : W < 2^n)
    (r u : RepIndex n W) (s t : Int) (hs : Sign s) (ht : Sign t) :
    (nativeProduct n W r.val u.val s t = (zeroCoeff,zeroCoeff) ∧
     nativeProduct n W u.val r.val t s = (zeroCoeff,zeroCoeff)) ↔
      s*t = matrix M n W r.val u.val := by
  rw [native_adjacency_bit_iff n W r.val u.val s t hn hW
    r.property.1 u.property.1 r.property.2.1 u.property.2.1
    (rep_ne_label r.property) (rep_ne_label u.property) hs ht]
  have hst := mul_sign hs ht
  have hcross := rep_cross r.property u.property
  by_cases he : r.val = u.val
  · rcases hst with h | h <;> simp [he,matrix,h]
  · simp only [matrix,he,↓reduceIte,basisPair]
    have hc : (r.val ≠ u.val ∧ r.val ≠ u.val ^^^ W) := ⟨he,hcross⟩
    constructor
    · intro h
      exact (sign_eq_M _ _ (s*t) hst).mpr h.2.2
    · intro h
      exact ⟨he,hcross,(sign_eq_M _ _ (s*t) hst).mp h⟩

#print axioms low_block_identity
#print axioms high_block_identity
#print axioms native_matrix_adjacency
end Sounio.ZDMatrixAssembly
