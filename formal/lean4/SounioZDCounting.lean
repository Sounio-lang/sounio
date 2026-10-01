import SounioZDSignedCongruence
import SounioZDUnsignedCode

/-! Finite counting identities for the certified CD matrix constructors.
Integer ordered sums are used before any division by graph automorphism factors.
No classification or priority claim is implied by the generic counting lemmas. -/
namespace Sounio.ZDCounting
open SounioCDCocycle Sounio.ZDAlgebraBridge Sounio.ZDRecursion
open Sounio.ZDMatrixAssembly Sounio.ZDTwinCover Sounio.ZDSignedCongruence
set_option maxRecDepth 4096
set_option maxHeartbeats 16000000

def total {R : Type} : List R → (R → Int) → Int
  | [], _ => 0
  | r::rs, f => f r + total rs f

@[simp] theorem total_nil {R : Type} (f : R → Int) : total [] f = 0 := rfl
@[simp] theorem total_cons {R : Type} (r : R) (rs : List R) (f : R → Int) :
    total (r::rs) f = f r + total rs f := rfl
theorem total_ext {R : Type} (rs : List R) (f g : R → Int)
    (h : ∀ r, r ∈ rs → f r = g r) : total rs f = total rs g := by
  induction rs with
  | nil => rfl
  | cons r rs ih =>
    rw [total_cons,total_cons,h r (by simp),ih (fun s hs => h s (by simp [hs]))]

@[simp] theorem total_zero {R : Type} (rs : List R) : total rs (fun _ => 0) = 0 := by
  induction rs <;> simp_all [total]
theorem total_add {R : Type} (rs : List R) (f g : R → Int) :
    total rs (fun r => f r + g r) = total rs f + total rs g := by
  induction rs <;> simp_all [total] <;> omega
theorem total_mul {R : Type} (rs : List R) (a : Int) (f : R → Int) :
    total rs (fun r => a*f r) = a*total rs f := by
  induction rs <;> simp_all [total,Int.mul_add]
theorem total_mul_right {R : Type} (rs : List R) (f : R → Int) (a : Int) :
    total rs (fun r => f r*a) = total rs f*a := by
  simpa [Int.mul_comm] using total_mul rs a f
@[simp] theorem total_const {R : Type} (rs : List R) (a : Int) :
    total rs (fun _ => a) = (rs.length : Int)*a := by
  induction rs <;> simp_all [total,Int.add_mul,Int.add_comm]
theorem total_map {R S : Type} (rs : List R) (f : R → S) (g : S → Int) :
    total (rs.map f) g = total rs (fun r => g (f r)) := by
  induction rs <;> simp_all [total]
theorem total_append {R : Type} (rs ss : List R) (f : R → Int) :
    total (rs++ss) f = total rs f + total ss f := by
  induction rs <;> simp_all [total,Int.add_assoc]
theorem total_swap {R S : Type} (rs : List R) (ss : List S) (f : R → S → Int) :
    total rs (fun r => total ss (f r)) = total ss (fun s => total rs (fun r => f r s)) := by
  induction rs <;> simp_all [total,total_add]

theorem total_delta {R : Type} [DecidableEq R] (rs : List R) (hn : rs.Nodup)
    (r : R) (hr : r ∈ rs) (f : R → Int) :
    total rs (fun s => if r=s then f s else 0) = f r := by
  induction rs with
  | nil => simp at hr
  | cons a rs ih =>
    have ha := (List.nodup_cons.mp hn).1
    have hs := (List.nodup_cons.mp hn).2
    rcases List.mem_cons.mp hr with h | h
    · subst r
      have hz : total rs (fun s => if a=s then f s else 0) = 0 := by
        apply (total_ext rs _ (fun _ => 0) ?_).trans (total_zero rs)
        intro s hmem
        have hne : a ≠ s := by intro e; subst s; exact ha hmem
        simp [hne]
      simp [total,hz]
    · have hne : r ≠ a := by intro e; subst r; exact ha h
      simp [total,hne,ih hs h]

def lifted {R : Type} (rs : List R) : List (Option (Bool × R)) :=
  none :: (rs.map (fun r => some (false,r)) ++ rs.map (fun r => some (true,r)))

theorem total_lifted {R : Type} (rs : List R) (f : Option (Bool × R) → Int) :
    total (lifted rs) f =
      f none + total rs (fun r => f (some (false,r)) + f (some (true,r))) := by
  simp [lifted,total,total_append,total_map,total_add]

def moment2 {R : Type} (rs : List R) (A : R → R → Int) : Int :=
  total rs (fun r => total rs (fun s => A r s * A s r))
def moment3 {R : Type} (rs : List R) (A : R → R → Int) : Int :=
  total rs (fun r => total rs (fun s => total rs (fun t => A r s * A s t * A t r)))

theorem cee_moment2 {R : Type} (rs : List R) (A : R → R → Int) :
    moment2 (lifted rs) (cee A) = 4*moment2 rs A := by
  simp [moment2,total_lifted,cee,total_add,total_mul,total_mul_right]
  grind

theorem cee_moment3 {R : Type} (rs : List R) (A : R → R → Int) :
    moment3 (lifted rs) (cee A) = 8*moment3 rs A := by
  simp [moment3,total_lifted,cee,total_add,total_mul,total_mul_right]
  grind

def tblock {R : Type} [DecidableEq R] (A : R → R → Int)
    (a b : Bool) (r s : R) : Int := tee A (fun _ => 1) (some (a,r)) (some (b,s))
def pairSum (f : Bool → Bool → Int) : Int :=
  f false false + f false true + f true false + f true true
def tripleSum (f : Bool → Bool → Bool → Int) : Int :=
  pairSum (fun a b => f a b false + f a b true)

theorem tblock_pair {R : Type} [DecidableEq R] (A : R → R → Int)
    (hd : ∀ r, A r r = 0) (r s : R) :
    pairSum (fun a b => tblock A a b r s * tblock A b a s r) =
      4*(A r s*A s r) + if r=s then 2 else 0 := by
  by_cases h : r=s
  · subst s; simp [pairSum,tblock,tee,hd]
  · have h' := Ne.symm h
    simp [pairSum,tblock,tee,h,h']
    grind

theorem tblock_hub {R : Type} [DecidableEq R] (A : R → R → Int) (r s : R) :
    pairSum (fun a b => (if a then -1 else 1) * tblock A a b r s *
      (if b then -1 else 1)) = if r=s then -2 else 0 := by
  by_cases h : r=s <;> simp [pairSum,tblock,tee,h] <;> omega

theorem tblock_triple {R : Type} [DecidableEq R] (A : R → R → Int)
    (hd : ∀ r, A r r = 0) (r s t : R) :
    tripleSum (fun a b c =>
      tblock A a b r s * tblock A b c s t * tblock A c a t r) =
      8*(A r s*A s t*A t r) +
      (if r=s then 4*(A r t*A t r) else 0) +
      (if s=t then 4*(A r s*A s r) else 0) +
      (if t=r then 4*(A r s*A s r) else 0) := by
  by_cases hrs : r=s
  · subst s
    by_cases hrt : r=t
    · subst t; simp [tripleSum,pairSum,tblock,tee,hd]
    · simp [tripleSum,pairSum,tblock,tee,hd,hrt,Ne.symm hrt]
      grind
  · by_cases hst : s=t
    · subst t
      simp [tripleSum,pairSum,tblock,tee,hd,hrs,Ne.symm hrs]
      grind
    · by_cases htr : t=r
      · subst t
        simp [tripleSum,pairSum,tblock,tee,hd,hrs,Ne.symm hrs]
        grind
      · simp [tripleSum,pairSum,tblock,tee,hd,hrs,hst,htr,Ne.symm hrs,Ne.symm hst,Ne.symm htr]
        grind

#print axioms cee_moment2
#print axioms cee_moment3
#print axioms tblock_triple
theorem total_diagonal {R : Type} [DecidableEq R] (rs : List R) (hn : rs.Nodup)
    (f : R → Int) :
    total rs (fun r => total rs (fun s => if r=s then f r else 0)) = total rs f := by
  apply total_ext
  intro r hr
  exact total_delta rs hn r hr (fun _ => f r)

theorem triple_diag12 {R : Type} [DecidableEq R] (rs : List R) (hn : rs.Nodup)
    (A : R → R → Int) :
    total rs (fun r => total rs (fun s => total rs (fun t => if r=s then A r t else 0))) =
      total rs (fun r => total rs (A r)) := by
  apply total_ext
  intro r hr
  have h : ∀ s, total rs (fun t => if r=s then A r t else 0) =
      if r=s then total rs (A r) else 0 := by
    intro s; by_cases e : r=s <;> simp [e]
  simp only [h]
  exact total_delta rs hn r hr (fun _ => total rs (A r))

theorem triple_diag23 {R : Type} [DecidableEq R] (rs : List R) (hn : rs.Nodup)
    (A : R → R → Int) :
    total rs (fun r => total rs (fun s => total rs (fun t => if s=t then A r s else 0))) =
      total rs (fun r => total rs (A r)) := by
  apply total_ext
  intro r _
  apply total_ext
  intro s hs
  exact total_delta rs hn s hs (fun _ => A r s)

theorem triple_diag31 {R : Type} [DecidableEq R] (rs : List R) (hn : rs.Nodup)
    (A : R → R → Int) :
    total rs (fun r => total rs (fun s => total rs (fun t => if t=r then A r s else 0))) =
      total rs (fun r => total rs (A r)) := by
  apply total_ext
  intro r hr
  apply total_ext
  intro s _
  simpa only [eq_comm] using total_delta rs hn r hr (fun _ => A r s)

theorem moment2_lifted {R : Type} (rs : List R)
    (Z : Option (Bool × R) → Option (Bool × R) → Int) (h0 : Z none none = 0) :
    moment2 (lifted rs) Z =
      total rs (fun r =>
        Z none (some (false,r))*Z (some (false,r)) none +
        Z none (some (true,r))*Z (some (true,r)) none +
        Z (some (false,r)) none*Z none (some (false,r)) +
        Z (some (true,r)) none*Z none (some (true,r))) +
      total rs (fun r => total rs (fun s =>
        pairSum (fun a b => Z (some (a,r)) (some (b,s))*Z (some (b,s)) (some (a,r))))) := by
  simp [moment2,total_lifted,h0,pairSum,total_add]
  grind

theorem moment3_lifted {R : Type} (rs : List R)
    (Z : Option (Bool × R) → Option (Bool × R) → Int) (h0 : Z none none = 0) :
    moment3 (lifted rs) Z =
      total rs (fun r => total rs (fun s => pairSum (fun a b =>
        Z none (some (a,r))*Z (some (a,r)) (some (b,s))*Z (some (b,s)) none +
        Z (some (a,r)) none*Z none (some (b,s))*Z (some (b,s)) (some (a,r)) +
        Z (some (a,r)) (some (b,s))*Z (some (b,s)) none*Z none (some (a,r))))) +
      total rs (fun r => total rs (fun s => total rs (fun t => tripleSum (fun a b c =>
        Z (some (a,r)) (some (b,s))*Z (some (b,s)) (some (c,t))*Z (some (c,t)) (some (a,r)))))) := by
  simp [moment3,total_lifted,h0,pairSum,tripleSum,total_add]
  grind

theorem tblock_hubs {R : Type} [DecidableEq R] (A : R → R → Int) (r s : R) :
    pairSum (fun a b =>
      (if a then -1 else 1)*tblock A a b r s*(if b then -1 else 1) +
      (if a then -1 else 1)*(if b then -1 else 1)*tblock A b a s r +
      tblock A a b r s*(if b then -1 else 1)*(if a then -1 else 1)) =
      if r=s then -6 else 0 := by
  by_cases h : r=s
  · subst s; simp [pairSum,tblock,tee]
  · simp [pairSum,tblock,tee,h,Ne.symm h]
    omega

theorem tee_moment2 {R : Type} [DecidableEq R] (rs : List R) (hn : rs.Nodup)
    (A : R → R → Int) (hd : ∀ r, A r r = 0) :
    moment2 (lifted rs) (tee A (fun _ => 1)) =
      6*(rs.length : Int) + 4*moment2 rs A := by
  rw [moment2_lifted rs _ rfl]
  have hp : total rs (fun r => total rs (fun s =>
      pairSum (fun a b =>
        tee A (fun _ => 1) (some (a,r)) (some (b,s)) *
        tee A (fun _ => 1) (some (b,s)) (some (a,r))))) =
      4*moment2 rs A + 2*(rs.length : Int) := by
    have h : ∀ r s, pairSum (fun a b =>
        tee A (fun _ => 1) (some (a,r)) (some (b,s)) *
        tee A (fun _ => 1) (some (b,s)) (some (a,r))) =
        4*(A r s*A s r) + if r=s then 2 else 0 := tblock_pair A hd
    simp only [h,total_add,total_mul]
    rw [total_diagonal rs hn]
    simp [moment2,Int.mul_comm]
  rw [hp]
  simp [tee]
  omega

theorem tee_moment3 {R : Type} [DecidableEq R] (rs : List R) (hn : rs.Nodup)
    (A : R → R → Int) (hd : ∀ r, A r r = 0) :
    moment3 (lifted rs) (tee A (fun _ => 1)) =
      -6*(rs.length : Int) + 12*moment2 rs A + 8*moment3 rs A := by
  rw [moment3_lifted rs _ rfl]
  have hh : total rs (fun r => total rs (fun s => pairSum (fun a b =>
      tee A (fun _ => 1) none (some (a,r))*tee A (fun _ => 1) (some (a,r)) (some (b,s))*tee A (fun _ => 1) (some (b,s)) none +
      tee A (fun _ => 1) (some (a,r)) none*tee A (fun _ => 1) none (some (b,s))*tee A (fun _ => 1) (some (b,s)) (some (a,r)) +
      tee A (fun _ => 1) (some (a,r)) (some (b,s))*tee A (fun _ => 1) (some (b,s)) none*tee A (fun _ => 1) none (some (a,r))))) =
      -6*(rs.length : Int) := by
    have h : ∀ r s, pairSum (fun a b =>
        tee A (fun _ => 1) none (some (a,r))*tee A (fun _ => 1) (some (a,r)) (some (b,s))*tee A (fun _ => 1) (some (b,s)) none +
        tee A (fun _ => 1) (some (a,r)) none*tee A (fun _ => 1) none (some (b,s))*tee A (fun _ => 1) (some (b,s)) (some (a,r)) +
        tee A (fun _ => 1) (some (a,r)) (some (b,s))*tee A (fun _ => 1) (some (b,s)) none*tee A (fun _ => 1) none (some (a,r))) =
        if r=s then -6 else 0 := by
      intro r s
      exact tblock_hubs A r s
    simp only [h]
    rw [total_diagonal rs hn]
    simp [Int.mul_comm]
  rw [hh]
  have ht : total rs (fun r => total rs (fun s => total rs (fun t =>
      tripleSum (fun a b c =>
        tee A (fun _ => 1) (some (a,r)) (some (b,s)) *
        tee A (fun _ => 1) (some (b,s)) (some (c,t)) *
        tee A (fun _ => 1) (some (c,t)) (some (a,r)))))) =
      8*moment3 rs A + 12*moment2 rs A := by
    have h : ∀ r s t, tripleSum (fun a b c =>
        tee A (fun _ => 1) (some (a,r)) (some (b,s)) *
        tee A (fun _ => 1) (some (b,s)) (some (c,t)) *
        tee A (fun _ => 1) (some (c,t)) (some (a,r))) =
        8*(A r s*A s t*A t r) +
        (if r=s then 4*(A r t*A t r) else 0) +
        (if s=t then 4*(A r s*A s r) else 0) +
        (if t=r then 4*(A r s*A s r) else 0) := tblock_triple A hd
    simp only [h,total_add]
    rw [triple_diag12 rs hn, triple_diag23 rs hn, triple_diag31 rs hn]
    simp [total_mul,moment3,moment2]
    omega
  rw [ht]
  omega

#print axioms tee_moment2
#print axioms tee_moment3

theorem total_neg {R : Type} (rs : List R) (f : R → Int) :
    total rs (fun r => -f r) = -total rs f := by
  simpa using total_mul rs (-1) f
theorem total_sub {R : Type} (rs : List R) (f g : R → Int) :
    total rs (fun r => f r-g r) = total rs f-total rs g := by
  simp [Int.sub_eq_add_neg,total_add,total_neg]

theorem congr_moment2 {R S : Type} (rs : List R) {A : R → R → Int} {B : S → S → Int}
    (c : SignedCongruence A B) :
    moment2 (rs.map c.index.toFun) B = moment2 rs A := by
  simp only [moment2,total_map]
  apply total_ext
  intro r _
  apply total_ext
  intro s _
  rw [c.entry,c.entry]
  have hr := c.weight_sign r
  have hs := c.weight_sign s
  rcases hr with h | h <;> rcases hs with j | j <;> simp [h,j,Int.neg_mul,Int.mul_neg]

theorem congr_moment3 {R S : Type} (rs : List R) {A : R → R → Int} {B : S → S → Int}
    (c : SignedCongruence A B) :
    moment3 (rs.map c.index.toFun) B = moment3 rs A := by
  simp only [moment3,total_map]
  apply total_ext
  intro r _
  apply total_ext
  intro s _
  apply total_ext
  intro t _
  rw [c.entry,c.entry,c.entry]
  have hr := c.weight_sign r
  have hs := c.weight_sign s
  have ht := c.weight_sign t
  rcases hr with h | h <;> rcases hs with j | j <;> rcases ht with k | k <;>
    simp [h,j,k,Int.neg_mul,Int.mul_neg]

def Binary (a : Int) : Prop := a=0 ∨ a=1
def Ternary (a : Int) : Prop := a=0 ∨ Sign a
def support {R : Type} (A : R → R → Int) (r s : R) : Int := if A r s=0 then 0 else 1
def complement {R : Type} [DecidableEq R] (A : R → R → Int) (r s : R) : Int :=
  if r=s then 0 else 1-A r s

theorem binary_ternary {a : Int} (h : Binary a) : Ternary a := by
  rcases h with h | h
  · exact Or.inl h
  · exact Or.inr (Or.inl h)
theorem support_binary {R : Type} (A : R → R → Int) (r s : R) :
    Binary (support A r s) := by
  unfold support
  split <;> simp [Binary]
theorem support_zero {R : Type} (A : R → R → Int) (hd : ∀ r, A r r=0) (r : R) :
    support A r r=0 := by simp [support,hd]
theorem support_sym {R : Type} (A : R → R → Int) (hs : ∀ r s, A r s=A s r) (r s : R) :
    support A r s = support A s r := by simp [support,hs r s]

theorem support_moment2 {R : Type} (rs : List R) (A : R → R → Int)
    (ha : ∀ r s, Ternary (A r s)) (hs : ∀ r s, A r s=A s r) :
    moment2 rs (support A) = moment2 rs A := by
  unfold moment2
  apply total_ext
  intro r _
  apply total_ext
  intro s _
  have h := ha r s
  rcases h with h | h | h <;> simp [support,← hs r s,h]

theorem complement_binary {R : Type} [DecidableEq R] (A : R → R → Int)
    (hb : ∀ r s, Binary (A r s)) (r s : R) : Binary (complement A r s) := by
  by_cases h : r=s
  · simp [complement,h,Binary]
  · rcases hb r s with ha | ha <;> simp [complement,h,ha,Binary]
theorem complement_zero {R : Type} [DecidableEq R] (A : R → R → Int) (r : R) :
    complement A r r=0 := by simp [complement]
theorem complement_sym {R : Type} [DecidableEq R] (A : R → R → Int)
    (hs : ∀ r s, A r s=A s r) (r s : R) : complement A r s=complement A s r := by
  simp [complement,hs r s,eq_comm]

theorem complement_moment2 {R : Type} [DecidableEq R] (rs : List R) (hn : rs.Nodup)
    (A : R → R → Int) (hb : ∀ r s, Binary (A r s))
    (hd : ∀ r, A r r=0) (hs : ∀ r s, A r s=A s r) :
    moment2 rs (complement A) =
      (rs.length : Int)*((rs.length : Int)-1)-moment2 rs A := by
  have hp : ∀ r s, complement A r s * complement A s r =
      1-(if r=s then 1 else 0)-A r s*A s r := by
    intro r s
    by_cases h : r=s
    · subst s; simp [complement,hd]
    · rcases hb r s with ha | ha <;> simp [complement,h,Ne.symm h,← hs r s,ha]
  simp only [moment2,hp,total_sub]
  rw [total_diagonal rs hn]
  simp
  grind

theorem tee_zero {R : Type} [DecidableEq R] (A : R → R → Int)
    (x : Option (Bool × R)) : tee A (fun _ => 1) x x=0 := by
  cases x <;> simp [tee]
theorem tee_sym {R : Type} [DecidableEq R] (A : R → R → Int)
    (hs : ∀ r s, A r s=A s r) (x y : Option (Bool × R)) :
    tee A (fun _ => 1) x y=tee A (fun _ => 1) y x := by
  cases x with
  | none => cases y <;> simp [tee]
  | some x =>
    rcases x with ⟨a,r⟩
    cases y with
    | none => simp [tee]
    | some y =>
      rcases y with ⟨b,s⟩
      simp [tee,hs r s,eq_comm]
theorem tee_ternary {R : Type} [DecidableEq R] (A : R → R → Int)
    (ha : ∀ r s, Ternary (A r s)) (x y : Option (Bool × R)) :
    Ternary (tee A (fun _ => 1) x y) := by
  cases x with
  | none =>
    cases y with
    | none => exact Or.inl rfl
    | some y => cases y.1 <;> simp [tee,Ternary,Sign]
  | some x =>
    rcases x with ⟨a,r⟩
    cases y with
    | none => cases a <;> simp [tee,Ternary,Sign]
    | some y =>
      rcases y with ⟨b,s⟩
      by_cases h : r=s
      · by_cases h' : a=b <;> simp [tee,h,h',Ternary,Sign]
      · simpa [tee,h] using ha r s

/-- The ordinary unsigned cone of paired cliques is the support of tee. -/
def conePairs {R : Type} [DecidableEq R] (A : R → R → Int) :
    Option (Bool × R) → Option (Bool × R) → Int := support (tee A (fun _ => 1))

theorem conePairs_moment2 {R : Type} [DecidableEq R] (rs : List R) (hn : rs.Nodup)
    (A : R → R → Int) (hb : ∀ r s, Binary (A r s))
    (hd : ∀ r, A r r=0) (hs : ∀ r s, A r s=A s r) :
    moment2 (lifted rs) (conePairs A) = 6*(rs.length : Int)+4*moment2 rs A := by
  rw [conePairs,support_moment2 _ _ (tee_ternary A (fun r s => binary_ternary (hb r s))) (tee_sym A hs)]
  exact tee_moment2 rs hn A hd

@[reducible] def WordIndex : Nat → Type
  | 0 => Fin 3
  | d+1 => Option (Bool × WordIndex d)
instance wordIndexDecidableEq (d : Nat) : DecidableEq (WordIndex d) := by
  induction d with
  | zero => exact inferInstanceAs (DecidableEq (Fin 3))
  | succ d ih => exact inferInstanceAs (DecidableEq (Option (Bool × WordIndex d)))

def wordRoster : (d : Nat) → List (WordIndex d)
  | 0 => ([0,1,2] : List (Fin 3))
  | d+1 => lifted (wordRoster d)

theorem lifted_nodup {R : Type} (rs : List R) (hn : rs.Nodup) : (lifted rs).Nodup := by
  have hm (a : Bool) : (rs.map (fun r => some (a,r))).Nodup :=
    List.Pairwise.map _ (fun _ _ h e => h (Prod.mk.inj (Option.some.inj e)).2) hn
  rw [lifted,List.nodup_cons,List.nodup_append]
  refine ⟨by simp,hm false,hm true,?_⟩
  intro x hx y hy e
  obtain ⟨r,_,rfl⟩ := List.mem_map.mp hx
  obtain ⟨s,_,rfl⟩ := List.mem_map.mp hy
  cases e

theorem wordRoster_nodup (d : Nat) : (wordRoster d).Nodup := by
  induction d with
  | zero => decide
  | succ d ih => exact lifted_nodup _ ih

theorem wordRoster_complete (d : Nat) (r : WordIndex d) : r ∈ wordRoster d := by
  induction d with
  | zero =>
    change r ∈ ([0,1,2] : List (Fin 3))
    have h := r.isLt
    have hv : r.val=0 ∨ r.val=1 ∨ r.val=2 := by omega
    rcases hv with h | h | h
    all_goals simp [List.mem_cons,Fin.ext_iff,h]
  | succ d ih =>
    change r ∈ lifted (wordRoster d)
    cases r with
    | none => exact List.mem_cons_self
    | some r =>
      rcases r with ⟨a,r⟩
      apply List.mem_cons_of_mem
      cases a
      · exact List.mem_append_left _ (List.mem_map.mpr ⟨r,ih r,rfl⟩)
      · exact List.mem_append_right _ (List.mem_map.mpr ⟨r,ih r,rfl⟩)

theorem wordRoster_size (d : Nat) :
    ((wordRoster d).length : Int) = Sounio.ZDUnsignedCode.parentOrder d := by
  induction d with
  | zero => decide
  | succ d ih =>
    change ((lifted (wordRoster d)).length : Int) = Sounio.ZDUnsignedCode.parentOrder (d+1)
    simp only [lifted,List.length_cons,List.length_append,List.length_map,
      Int.natCast_add,Int.natCast_one,ih]
    unfold Sounio.ZDUnsignedCode.parentOrder
    rw [show d+1+2=(d+2)+1 by omega,Int.pow_succ]
    grind

def wordMatrix : (bs : List Bool) → WordIndex bs.length → WordIndex bs.length → Int
  | [] => fun r s => if r=s then 0 else 1
  | b::bs => conePairs (if b then complement (wordMatrix bs) else wordMatrix bs)

theorem wordMatrix_properties (bs : List Bool) :
    (∀ r s, Binary (wordMatrix bs r s)) ∧
    (∀ r, wordMatrix bs r r=0) ∧
    (∀ r s, wordMatrix bs r s=wordMatrix bs s r) := by
  induction bs with
  | nil =>
    refine ⟨?_,?_,?_⟩
    · intro r s; by_cases h : r=s <;> simp [wordMatrix,h,Binary]
    · intro r; simp [wordMatrix]
    · intro r s; simp [wordMatrix,eq_comm]
  | cons b bs ih =>
    have hb : ∀ r s, Binary ((if b then complement (wordMatrix bs) else wordMatrix bs) r s) := by
      cases b
      · exact ih.1
      · exact complement_binary _ ih.1
    have hs : ∀ r s,
        (if b then complement (wordMatrix bs) else wordMatrix bs) r s =
        (if b then complement (wordMatrix bs) else wordMatrix bs) s r := by
      cases b
      · exact ih.2.2
      · exact complement_sym _ ih.2.2
    refine ⟨?_,?_,?_⟩
    · exact support_binary _
    · exact support_zero _ (tee_zero _)
    · exact support_sym _ (tee_sym _ hs)

/-- Actual finite support matrices now realize the existing arithmetic word code. -/
theorem wordMatrix_code (bs : List Bool) :
    moment2 (wordRoster bs.length) (wordMatrix bs) = 2*Sounio.ZDUnsignedCode.edgeCode bs := by
  induction bs with
  | nil => decide
  | cons b bs ih =>
    have hp := wordMatrix_properties bs
    have hn := wordRoster_nodup bs.length
    have hq := wordRoster_size bs.length
    have hc := Sounio.ZDUnsignedCode.parentCapacity_identity bs.length
    cases b
    · simp only [wordMatrix,wordRoster,List.length_cons,Bool.false_eq_true,↓reduceIte]
      rw [conePairs_moment2 _ hn _ hp.1 hp.2.1 hp.2.2,ih,hq]
      simp [Sounio.ZDUnsignedCode.edgeCode,Sounio.ZDUnsignedCode.step]
      grind
    · simp only [wordMatrix,wordRoster,List.length_cons,↓reduceIte]
      rw [conePairs_moment2 _ hn _ (complement_binary _ hp.1) (complement_zero _) (complement_sym _ hp.2.2)]
      rw [complement_moment2 _ hn _ hp.1 hp.2.1 hp.2.2,ih,hq]
      simp [Sounio.ZDUnsignedCode.edgeCode,Sounio.ZDUnsignedCode.step]
      grind

def wordEdges (bs : List Bool) : Int := moment2 (wordRoster bs.length) (wordMatrix bs)/2
theorem wordEdges_code (bs : List Bool) : wordEdges bs = Sounio.ZDUnsignedCode.edgeCode bs := by
  unfold wordEdges
  rw [wordMatrix_code]
  omega

theorem decode_wordEdges (bs : List Bool) :
    Sounio.ZDUnsignedCode.decode bs.length (wordEdges bs) = bs := by
  rw [wordEdges_code]
  exact Sounio.ZDUnsignedCode.decode_edgeCode bs

theorem wordEdges_injective (xs ys : List Bool) (hl : xs.length=ys.length)
    (he : wordEdges xs=wordEdges ys) : xs=ys := by
  rw [wordEdges_code,wordEdges_code] at he
  exact Sounio.ZDUnsignedCode.edgeCode_injective xs ys hl he

#print axioms congr_moment3
#print axioms complement_moment2
#print axioms wordRoster_complete
#print axioms wordMatrix_code
#print axioms decode_wordEdges

def spinVal (a : Bool) : Int := if a then -1 else 1
def spinEdge (z : Int) (a b : Bool) : Int := if spinVal a*spinVal b=z then 1 else 0

theorem spin_edge_sum (z : Int) (hz : Ternary z) :
    pairSum (spinEdge z) = 2*(z*z) := by
  rcases hz with h | h | h <;> simp [pairSum,spinEdge,spinVal,h]

theorem spin_triangle_sum (a b c : Int) (ha : Ternary a) (hb : Ternary b) (hc : Ternary c) :
    tripleSum (fun x y z => spinEdge a x y * spinEdge b y z * spinEdge c z x) =
      2*(if a*b*c=1 then 1 else 0) := by
  rcases ha with h | h | h <;> rcases hb with j | j | j <;> rcases hc with k | k | k <;>
    simp [tripleSum,pairSum,spinEdge,spinVal,h,j,k]

def coverVertex {R : Type} (r : R) (a k : Bool) : CoverVertex R Bool :=
  ⟨r,⟨spinVal a,by cases a <;> simp [spinVal,Sign]⟩,k⟩
def coverRoster {R : Type} (rs : List R) : List (CoverVertex R Bool) :=
  rs.map (fun r => coverVertex r false false) ++ rs.map (fun r => coverVertex r false true) ++
  rs.map (fun r => coverVertex r true false) ++ rs.map (fun r => coverVertex r true true)
def coverIndicator {R : Type} (A : R → R → Int) (x y : CoverVertex R Bool) : Int :=
  if x.spin.val*y.spin.val=A x.index y.index then 1 else 0

theorem coverIndicator_iff {R : Type} (A : R → R → Int) (x y : CoverVertex R Bool) :
    coverIndicator A x y=1 ↔ CoverAdj A x y := by
  simp [coverIndicator,CoverAdj]

def edgeMass {R : Type} (rs : List R) (A : R → R → Int) : Int :=
  total rs (fun r => total rs (fun s => A r s))
def triangleMass {R : Type} (rs : List R) (A : R → R → Int) : Int := moment3 rs A
def positiveMass {R : Type} (rs : List R) (A : R → R → Int) : Int :=
  total rs (fun r => total rs (fun s => total rs (fun t => if A r s*A s t*A t r=1 then 1 else 0)))

theorem cover_edge_expansion {R : Type} (rs : List R) (A : R → R → Int) :
    edgeMass (coverRoster rs) (coverIndicator A) =
      4*total rs (fun r => total rs (fun s => pairSum (spinEdge (A r s)))) := by
  simp [edgeMass,coverRoster,total_append,total_map,coverIndicator,coverVertex,spinEdge,
    pairSum,spinVal,total_add,total_mul]
  grind

theorem cover_triangle_expansion {R : Type} (rs : List R) (A : R → R → Int) :
    triangleMass (coverRoster rs) (coverIndicator A) =
      8*total rs (fun r => total rs (fun s => total rs (fun t => tripleSum (fun a b c =>
        spinEdge (A r s) a b * spinEdge (A s t) b c * spinEdge (A t r) c a)))) := by
  simp [triangleMass,moment3,coverRoster,total_append,total_map,coverIndicator,coverVertex,
    spinEdge,tripleSum,pairSum,spinVal,total_add,total_mul]
  grind

/-- Ordered edge incidence count, including both independent native-twin copies. -/
theorem cover_edge_mass {R : Type} (rs : List R) (A : R → R → Int)
    (ha : ∀ r s, Ternary (A r s)) (hs : ∀ r s, A r s=A s r) :
    edgeMass (coverRoster rs) (coverIndicator A) = 8*moment2 rs A := by
  rw [cover_edge_expansion]
  simp only [spin_edge_sum _ (ha _ _),total_mul]
  have h : total rs (fun r => total rs (fun s => A r s*A r s)) = moment2 rs A := by
    unfold moment2
    apply total_ext
    intro r _
    apply total_ext
    intro s _
    rw [hs s r]
  rw [h]
  omega

/-- Ordered triangle count: only positive signed triangles lift. -/
theorem cover_triangle_mass {R : Type} (rs : List R) (A : R → R → Int)
    (ha : ∀ r s, Ternary (A r s)) :
    triangleMass (coverRoster rs) (coverIndicator A) = 16*positiveMass rs A := by
  rw [cover_triangle_expansion]
  simp only [spin_triangle_sum _ _ _ (ha _ _) (ha _ _) (ha _ _),total_mul]
  unfold positiveMass
  omega

theorem signed_triangle_identity {R : Type} (rs : List R) (A : R → R → Int)
    (ha : ∀ r s, Ternary (A r s)) :
    moment3 rs A = 2*positiveMass rs A - moment3 rs (support A) := by
  have h : ∀ r s t, A r s*A s t*A t r =
      2*(if A r s*A s t*A t r=1 then 1 else 0) -
      support A r s*support A s t*support A t r := by
    intro r s t
    rcases ha r s with h | h | h <;> rcases ha s t with j | j | j <;>
      rcases ha t r with k | k | k <;> simp [support,h,j,k]
  have he : moment3 rs A = total rs (fun r => total rs (fun s => total rs (fun t =>
      2*(if A r s*A s t*A t r=1 then 1 else 0) -
      support A r s*support A s t*support A t r))) := by
    unfold moment3
    apply total_ext
    intro r _
    apply total_ext
    intro s _
    apply total_ext
    intro t _
    exact h r s t
  rw [he]
  simp [total_sub,total_mul,moment3,positiveMass]

theorem matrix_M_ternary (n W : Nat) (r s : RepIndex n W) :
    Ternary (repMatrix M n W r s) := by
  unfold repMatrix matrix
  split
  · exact Or.inl rfl
  · unfold M
    split
    · apply Or.inr
      apply mul_sign
      · exact sgn_sign _ _ (by simp [basisPair,bitsOf_length])
      · exact sgn_sign _ _ (by simp [basisPair,bitsOf_length])
    · exact Or.inl rfl

def nativeCoverIso (n W : Nat) (hn : 1 ≤ n) (hW : W < 2^n) (hW0 : W ≠ 0) :
    GraphIso (NativeVertex n W) (CoverVertex (RepIndex n W) Bool)
      (NativeAdj n W) (CoverAdj (repMatrix M n W)) :=
  GraphIso.trans (native_twin_iso n W hn hW hW0) (twin_cover_iso n W)

def nativeRoster (n W : Nat) (hn : 1 ≤ n) (hW : W < 2^n) (hW0 : W ≠ 0)
    (rs : List (RepIndex n W)) : List (NativeVertex n W) :=
  (coverRoster rs).map (nativeCoverIso n W hn hW hW0).invFun

/-- Computable indicator; the following iff ties it to BOTH literal products. -/
def nativeIndicator (n W : Nat) (hn : 1 ≤ n) (hW : W < 2^n) (hW0 : W ≠ 0)
    (x y : NativeVertex n W) : Int :=
  coverIndicator (repMatrix M n W)
    ((nativeCoverIso n W hn hW hW0).toFun x) ((nativeCoverIso n W hn hW hW0).toFun y)

theorem nativeIndicator_iff (n W : Nat) (hn : 1 ≤ n) (hW : W < 2^n) (hW0 : W ≠ 0)
    (x y : NativeVertex n W) : nativeIndicator n W hn hW hW0 x y=1 ↔ NativeAdj n W x y := by
  unfold nativeIndicator
  rw [coverIndicator_iff]
  exact ((nativeCoverIso n W hn hW hW0).map_adj_iff x y).symm

theorem native_edge_mass (n W : Nat) (hn : 1 ≤ n) (hW : W < 2^n) (hW0 : W ≠ 0)
    (rs : List (RepIndex n W)) :
    edgeMass (nativeRoster n W hn hW hW0 rs) (nativeIndicator n W hn hW hW0) =
      8*moment2 rs (repMatrix M n W) := by
  have h : edgeMass (nativeRoster n W hn hW hW0 rs) (nativeIndicator n W hn hW hW0) =
      edgeMass (coverRoster rs) (coverIndicator (repMatrix M n W)) := by
    simp [edgeMass,nativeRoster,total_map,nativeIndicator,(nativeCoverIso n W hn hW hW0).right_inv]
  rw [h]
  exact cover_edge_mass _ _ (matrix_M_ternary n W) (fun r s => matrix_symmetry M n W hW r s)

theorem native_triangle_mass (n W : Nat) (hn : 1 ≤ n) (hW : W < 2^n) (hW0 : W ≠ 0)
    (rs : List (RepIndex n W)) :
    triangleMass (nativeRoster n W hn hW hW0 rs) (nativeIndicator n W hn hW hW0) =
      16*positiveMass rs (repMatrix M n W) := by
  have h : triangleMass (nativeRoster n W hn hW hW0 rs) (nativeIndicator n W hn hW hW0) =
      triangleMass (coverRoster rs) (coverIndicator (repMatrix M n W)) := by
    simp [triangleMass,moment3,nativeRoster,total_map,nativeIndicator,
      (nativeCoverIso n W hn hW hW0).right_inv]
  rw [h]
  exact cover_triangle_mass _ _ (matrix_M_ternary n W)

#print axioms cover_triangle_mass
#print axioms signed_triangle_identity
#print axioms nativeIndicator_iff
#print axioms native_edge_mass
#print axioms native_triangle_mass

theorem coverVertex_inj {R : Type} (r s : R) (a b k l : Bool) :
    coverVertex r a k = coverVertex s b l ↔ r=s ∧ a=b ∧ k=l := by
  constructor
  · intro h
    have hi := congrArg CoverVertex.index h
    have hs := congrArg (fun x => x.spin.val) h
    have hk := congrArg CoverVertex.copy h
    refine ⟨hi,?_,hk⟩
    cases a <;> cases b <;> simp_all [coverVertex,spinVal]
  · rintro ⟨rfl,rfl,rfl⟩; rfl

theorem coverRoster_nodup {R : Type} (rs : List R) (hn : rs.Nodup) :
    (coverRoster rs).Nodup := by
  have hm (a k : Bool) : (rs.map (fun r => coverVertex r a k)).Nodup :=
    List.Pairwise.map _ (fun _ _ h e => h ((coverVertex_inj _ _ _ _ _ _).mp e).1) hn
  simp [coverRoster,List.nodup_append,hm,List.mem_append,List.mem_map,coverVertex_inj]
  grind only [coverVertex_inj]

theorem coverRoster_complete {R : Type} (rs : List R) (h : ∀ r, r ∈ rs)
    (x : CoverVertex R Bool) : x ∈ coverRoster rs := by
  rcases x with ⟨r,⟨s,hs⟩,k⟩
  cases k <;> rcases hs with hs | hs
  all_goals
    subst s
    simp [coverRoster,List.mem_append,List.mem_map,coverVertex,spinVal,h r]

theorem nativeRoster_nodup (n W : Nat) (hn : 1 ≤ n) (hW : W < 2^n) (hW0 : W ≠ 0)
    (rs : List (RepIndex n W)) (hr : rs.Nodup) :
    (nativeRoster n W hn hW hW0 rs).Nodup := by
  apply List.Pairwise.map _ _ (coverRoster_nodup rs hr)
  intro x y h e
  apply h
  have he := congrArg (nativeCoverIso n W hn hW hW0).toFun e
  simpa only [(nativeCoverIso n W hn hW hW0).right_inv] using he

theorem nativeRoster_complete (n W : Nat) (hn : 1 ≤ n) (hW : W < 2^n) (hW0 : W ≠ 0)
    (rs : List (RepIndex n W)) (hr : ∀ r, r ∈ rs) (x : NativeVertex n W) :
    x ∈ nativeRoster n W hn hW hW0 rs := by
  exact List.mem_map.mpr ⟨(nativeCoverIso n W hn hW hW0).toFun x,
    coverRoster_complete rs hr _,(nativeCoverIso n W hn hW hW0).left_inv x⟩

#print axioms nativeRoster_nodup
#print axioms nativeRoster_complete

theorem edgeMass_even {R : Type} (rs : List R) (A : R → R → Int)
    (hs : ∀ r s, A r s=A s r) (hd : ∀ r, A r r=0) :
    edgeMass rs A % 2=0 := by
  induction rs with
  | nil => rfl
  | cons r rs ih =>
    have hc : edgeMass (r::rs) A = 2*total rs (A r)+edgeMass rs A := by
      have he : total rs (fun s => A s r)=total rs (A r) := by
        apply total_ext; intro s _; exact hs s r
      simp only [edgeMass,total_cons,total_add,hd,Int.zero_add]
      rw [he]
      grind
    rw [hc]
    omega

def tripleTotal {R : Type} (rs : List R) (F : R → R → R → Int) : Int :=
  total rs (fun r => total rs (fun s => total rs (F r s)))

theorem tripleTotal_six {R : Type} (rs : List R) (F : R → R → R → Int)
    (hc : ∀ r s t, F r s t=F s t r) (hs : ∀ r s t, F r s t=F s r t)
    (hd : ∀ r s, F r r s=0) : tripleTotal rs F % 6=0 := by
  induction rs with
  | nil => rfl
  | cons r rs ih =>
    have h1 : ∀ s, F r s r=0 := by intro s; rw [hc, hc, hd]
    have h2 : ∀ s, F s r r=0 := by intro s; rw [hc, hd]
    have h3 : ∀ s t, F s r t=F r s t := fun s t => hs s r t
    have h4 : ∀ s t, F s t r=F r s t := by intro s t; rw [hc, hc]
    have hrr : F r r = fun _ => 0 := funext (hd r)
    have h3' : ∀ s, F s r = F r s := fun s => funext (h3 s)
    have hcons : tripleTotal (r::rs) F =
        3*edgeMass rs (fun s t => F r s t)+tripleTotal rs F := by
      simp only [tripleTotal,total_cons,total_add,hd,h1,h2,h3,h4,hrr,h3',total_zero,
        Int.zero_add,Int.add_zero]
      unfold edgeMass
      grind
    have he : edgeMass rs (fun s t => F r s t) % 2=0 := by
      apply edgeMass_even
      · intro s t; rw [hs r s t,hc s r t]
      · intro s; rw [hc,hd]
    rw [hcons]
    omega

theorem moment2_even {R : Type} (rs : List R) (A : R → R → Int)
    (hd : ∀ r, A r r=0) : moment2 rs A % 2=0 := by
  apply edgeMass_even
  · intro r s; exact Int.mul_comm _ _
  · intro r; simp [hd]

theorem moment3_six {R : Type} (rs : List R) (A : R → R → Int)
    (hs : ∀ r s, A r s=A s r) (hd : ∀ r, A r r=0) :
    moment3 rs A % 6=0 := by
  apply tripleTotal_six
  · intro r s t; simp [Int.mul_assoc,Int.mul_comm,Int.mul_left_comm]
  · intro r s t; simp [hs,Int.mul_assoc,Int.mul_comm,Int.mul_left_comm]
  · intro r s; simp [hd]

theorem positiveMass_six {R : Type} (rs : List R) (A : R → R → Int)
    (hs : ∀ r s, A r s=A s r) (hd : ∀ r, A r r=0) :
    positiveMass rs A % 6=0 := by
  apply tripleTotal_six
  · intro r s t; simp [Int.mul_assoc,Int.mul_comm,Int.mul_left_comm]
  · intro r s t; simp [hs,Int.mul_assoc,Int.mul_comm,Int.mul_left_comm]
  · intro r s; simp [hd]

/-- Weighted second moment divided by two; a support-edge count for symmetric ternary matrices. -/
def matrixEdges {R : Type} (rs : List R) (A : R → R → Int) : Int := moment2 rs A/2
def signedTriangles {R : Type} (rs : List R) (A : R → R → Int) : Int := moment3 rs A/6
def positiveTriangles {R : Type} (rs : List R) (A : R → R → Int) : Int := positiveMass rs A/6
def graphEdges {R : Type} (rs : List R) (A : R → R → Int) : Int := edgeMass rs A/2
def graphTriangles {R : Type} (rs : List R) (A : R → R → Int) : Int := triangleMass rs A/6

theorem tee_counts {R : Type} [DecidableEq R] (rs : List R) (hn : rs.Nodup)
    (A : R → R → Int) (hs : ∀ r s, A r s=A s r) (hd : ∀ r, A r r=0) :
    matrixEdges (lifted rs) (tee A (fun _ => 1)) = 3*(rs.length : Int)+4*matrixEdges rs A ∧
    signedTriangles (lifted rs) (tee A (fun _ => 1)) =
      -(rs.length : Int)+4*matrixEdges rs A+8*signedTriangles rs A := by
  have h2 := moment2_even rs A hd
  have h3 := moment3_six rs A hs hd
  unfold matrixEdges signedTriangles
  rw [tee_moment2 rs hn A hd,tee_moment3 rs hn A hd]
  omega

theorem cee_counts {R : Type} (rs : List R) (A : R → R → Int)
    (hs : ∀ r s, A r s=A s r) (hd : ∀ r, A r r=0) :
    matrixEdges (lifted rs) (cee A) = 4*matrixEdges rs A ∧
    signedTriangles (lifted rs) (cee A) = 8*signedTriangles rs A := by
  have h2 := moment2_even rs A hd
  have h3 := moment3_six rs A hs hd
  unfold matrixEdges signedTriangles
  rw [cee_moment2,cee_moment3]
  omega

theorem native_edge_count (n W : Nat) (hn : 1 ≤ n) (hW : W < 2^n) (hW0 : W ≠ 0)
    (rs : List (RepIndex n W)) :
    graphEdges (nativeRoster n W hn hW hW0 rs) (nativeIndicator n W hn hW hW0) =
      8*matrixEdges rs (repMatrix M n W) := by
  have h := moment2_even rs (repMatrix M n W) (fun r => matrix_diagonal M n W r.val)
  unfold graphEdges matrixEdges
  rw [native_edge_mass]
  omega

theorem native_triangle_count (n W : Nat) (hn : 1 ≤ n) (hW : W < 2^n) (hW0 : W ≠ 0)
    (rs : List (RepIndex n W)) :
    graphTriangles (nativeRoster n W hn hW hW0 rs) (nativeIndicator n W hn hW hW0) =
      16*positiveTriangles rs (repMatrix M n W) := by
  have h := positiveMass_six rs (repMatrix M n W)
    (fun r s => matrix_symmetry M n W hW r s) (fun r => matrix_diagonal M n W r.val)
  unfold graphTriangles triangleMass positiveTriangles
  rw [show moment3 (nativeRoster n W hn hW hW0 rs) (nativeIndicator n W hn hW hW0) =
    16*positiveMass rs (repMatrix M n W) from native_triangle_mass n W hn hW hW0 rs]
  omega

theorem signed_triangle_count_identity {R : Type} (rs : List R) (A : R → R → Int)
    (ha : ∀ r s, Ternary (A r s)) (hs : ∀ r s, A r s=A s r) (hd : ∀ r, A r r=0) :
    signedTriangles rs A = 2*positiveTriangles rs A-signedTriangles rs (support A) := by
  have hp := positiveMass_six rs A hs hd
  have hu := moment3_six rs (support A) (support_sym A hs) (support_zero A hd)
  unfold signedTriangles positiveTriangles
  rw [signed_triangle_identity rs A ha]
  omega

#print axioms tripleTotal_six
#print axioms tee_counts
#print axioms cee_counts
#print axioms native_edge_count
#print axioms native_triangle_count
#print axioms signed_triangle_count_identity

theorem binary_matrixEdges {R : Type} (rs : List R) (A : R → R → Int)
    (hb : ∀ r s, Binary (A r s)) (hs : ∀ r s, A r s=A s r) :
    matrixEdges rs A=graphEdges rs A := by
  have h : moment2 rs A=edgeMass rs A := by
    unfold moment2 edgeMass
    apply total_ext
    intro r _
    apply total_ext
    intro s _
    rcases hb r s with h | h <;> simp [← hs r s,h]
  exact congrArg (fun z : Int => z/2) h

theorem wordEdges_actual (bs : List Bool) :
    wordEdges bs=graphEdges (wordRoster bs.length) (wordMatrix bs) := by
  exact binary_matrixEdges _ _ (wordMatrix_properties bs).1 (wordMatrix_properties bs).2.2

theorem decode_support_graph_edges (bs : List Bool) :
    Sounio.ZDUnsignedCode.decode bs.length
      (graphEdges (wordRoster bs.length) (wordMatrix bs)) = bs := by
  rw [← wordEdges_actual]
  exact decode_wordEdges bs

#print axioms wordEdges_actual
#print axioms decode_support_graph_edges

/-- Canonical complete representative roster, from the certified actual enumeration. -/
def representativeRoster (n W : Nat) (hW : W < 2^n) (hW0 : W ≠ 0) :
    List (RepIndex n W) :=
  (representatives n W).attach.map (fun r =>
    ⟨r.val,((representatives_certificate n W hW0 hW).2.1 r.val).mp r.property⟩)

theorem representativeRoster_values (n W : Nat) (hW : W < 2^n) (hW0 : W ≠ 0) :
    (representativeRoster n W hW hW0).map Subtype.val=representatives n W := by
  simp only [representativeRoster,List.map_map]
  exact List.attach_map_subtype_val _

theorem representativeRoster_nodup (n W : Nat) (hW : W < 2^n) (hW0 : W ≠ 0) :
    (representativeRoster n W hW hW0).Nodup := by
  have hm : ((representativeRoster n W hW hW0).map Subtype.val).Nodup := by
    rw [representativeRoster_values]
    exact (representatives_certificate n W hW0 hW).1
  have hp := List.pairwise_map.mp hm
  exact hp.imp (fun h e => h (congrArg Subtype.val e))

theorem representativeRoster_complete (n W : Nat) (hW : W < 2^n) (hW0 : W ≠ 0)
    (r : RepIndex n W) : r ∈ representativeRoster n W hW hW0 := by
  have hr : r.val ∈ representatives n W :=
    ((representatives_certificate n W hW0 hW).2.1 r.val).mpr r.property
  exact List.mem_map.mpr ⟨⟨r.val,hr⟩,by simp,Subtype.ext rfl⟩

def principalVertices (n W : Nat) (hn : 1 ≤ n) (hW : W < 2^n) (hW0 : W ≠ 0) :
    List (NativeVertex n W) :=
  nativeRoster n W hn hW hW0 (representativeRoster n W hW hW0)

theorem principalVertices_nodup (n W : Nat) (hn : 1 ≤ n) (hW : W < 2^n) (hW0 : W ≠ 0) :
    (principalVertices n W hn hW hW0).Nodup :=
  nativeRoster_nodup n W hn hW hW0 _ (representativeRoster_nodup n W hW hW0)

theorem principalVertices_complete (n W : Nat) (hn : 1 ≤ n) (hW : W < 2^n) (hW0 : W ≠ 0)
    (x : NativeVertex n W) : x ∈ principalVertices n W hn hW hW0 :=
  nativeRoster_complete n W hn hW hW0 _ (representativeRoster_complete n W hW hW0) x

theorem nativeIndicator_binary (n W : Nat) (hn : 1 ≤ n) (hW : W < 2^n) (hW0 : W ≠ 0)
    (x y : NativeVertex n W) : Binary (nativeIndicator n W hn hW hW0 x y) := by
  unfold nativeIndicator coverIndicator
  split <;> simp [Binary]

theorem nativeIndicator_sym (n W : Nat) (hn : 1 ≤ n) (hW : W < 2^n) (hW0 : W ≠ 0)
    (x y : NativeVertex n W) :
    nativeIndicator n W hn hW hW0 x y=nativeIndicator n W hn hW hW0 y x := by
  unfold nativeIndicator coverIndicator repMatrix
  rw [matrix_symmetry M n W hW]
  simp [Int.mul_comm]

theorem nativeIndicator_zero (n W : Nat) (hn : 1 ≤ n) (hW : W < 2^n) (hW0 : W ≠ 0)
    (x : NativeVertex n W) : nativeIndicator n W hn hW hW0 x x=0 := by
  unfold nativeIndicator coverIndicator
  rw [sign_square _ ((nativeCoverIso n W hn hW hW0).toFun x).spin.property]
  simp [repMatrix,matrix_diagonal]

/-- Whole principal graph counts; no caller-supplied enumeration obligation remains. -/
theorem principal_graph_counts (n W : Nat) (hn : 1 ≤ n) (hW : W < 2^n) (hW0 : W ≠ 0) :
    graphEdges (principalVertices n W hn hW hW0) (nativeIndicator n W hn hW hW0) =
      8*matrixEdges (representativeRoster n W hW hW0) (repMatrix M n W) ∧
    graphTriangles (principalVertices n W hn hW hW0) (nativeIndicator n W hn hW hW0) =
      16*positiveTriangles (representativeRoster n W hW hW0) (repMatrix M n W) :=
  ⟨native_edge_count n W hn hW hW0 _,native_triangle_count n W hn hW hW0 _⟩

#print axioms representativeRoster_complete
#print axioms principalVertices_nodup
#print axioms principalVertices_complete
#print axioms nativeIndicator_sym
#print axioms nativeIndicator_zero
#print axioms principal_graph_counts

end Sounio.ZDCounting
