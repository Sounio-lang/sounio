import SounioZDSupportWord

/-!
Canonical two-channel states for the principal main-sequence family.
The catalogue keeps signed information that unsigned support words erase.
-/
namespace Sounio.ZDSignedState
open SounioCDCocycle Sounio.ZDAlgebraBridge Sounio.ZDRecursion
open Sounio.ZDMatrixAssembly Sounio.ZDTwinCover Sounio.ZDSignedCongruence
open Sounio.ZDCounting Sounio.ZDSupportWord
set_option maxRecDepth 8192
set_option maxHeartbeats 16000000

inductive Code : Nat → Type
  | k : Code 0
  | z : Code 0
  | reset {d : Nat} : Code (d+1)
  | t {d : Nat} : Code d → Code (d+1)
  | c {d : Nat} : Code d → Code (d+1)
deriving DecidableEq, Repr

def order : Nat → Nat
  | 0 => 3
  | d+1 => 2*order d+1

theorem order_ge (d : Nat) : 3 ≤ order d := by
  induction d with
  | zero => decide
  | succ d ih => simp only [order]; omega

theorem order_odd (d : Nat) : order d % 2=1 := by
  cases d <;> simp [order,Nat.add_mod]

theorem order_pow (d : Nat) : order d+1=2^(d+2) := by
  induction d with
  | zero => decide
  | succ d ih =>
    rw [show d+1+2=(d+2)+1 by omega,Nat.pow_succ]
    simp only [order]
    omega

def edges : {d : Nat} → Code d → Nat
  | _, .k => 3
  | _, .z => 0
  | d+1, .reset => order d*(2*order d+1)
  | d+1, .t a => 3*order d+4*edges a
  | _, .c a => 4*edges a

def positives : {d : Nat} → Code d → Nat
  | _, .k => 0
  | _, .z => 0
  | _, .reset => 0
  | _, .t a => 6*edges a+8*positives a
  | _, .c a => 8*positives a

def isX : {d : Nat} → Code d → Bool
  | _, .k => true
  | _, .z => false
  | _, .reset => true
  | _, .t _ => true
  | _, .c _ => false

theorem reset_gt (d : Nat) : 3*order d < order d*(2*order d+1) := by
  have h := order_ge d
  have e := Nat.mul_lt_mul_of_pos_left (show 3 < 2*order d+1 by omega)
    (show 0 < order d by omega)
  simpa [Nat.mul_comm] using e

theorem edges_parity {d : Nat} (a : Code d) :
    edges a % 2 = if isX a then 1 else 0 := by
  cases a <;> simp [edges,isX,Nat.add_mod,Nat.mul_mod,order_odd]

theorem invariant_injective {d : Nat} (a b : Code d)
    (he : edges a=edges b) (hp : positives a=positives b) : a=b := by
  induction d with
  | zero =>
    cases a <;> cases b <;> simp_all [edges]
  | succ d ih =>
    have ha := edges_parity a
    have hb := edges_parity b
    have hg := reset_gt d
    cases a with
    | reset =>
      cases b with
      | reset => rfl
      | t b => simp only [edges,positives] at he hp; omega
      | c b => simp only [isX,Bool.false_eq_true,ite_true,ite_false] at ha hb; omega
    | t a =>
      cases b with
      | reset => simp only [edges,positives] at he hp; omega
      | t b =>
        simp only [edges,positives] at he hp
        have e : edges a=edges b := by omega
        have p : positives a=positives b := by omega
        exact congrArg Code.t (ih a b e p)
      | c b => simp only [isX,Bool.false_eq_true,ite_true,ite_false] at ha hb; omega
    | c a =>
      cases b with
      | reset => simp only [isX,Bool.false_eq_true,ite_true,ite_false] at ha hb; omega
      | t b => simp only [isX,Bool.false_eq_true,ite_true,ite_false] at ha hb; omega
      | c b =>
        simp only [edges,positives] at he hp
        exact congrArg Code.c (ih a b (by omega) (by omega))

def zeroCode : (d : Nat) → Code d
  | 0 => .z
  | d+1 => .c (zeroCode d)

def codes : (d : Nat) → List (Code d)
  | 0 => [.k,.z]
  | d+1 => [.reset] ++ (codes d).map Code.t ++ (codes d).map Code.c

theorem codes_complete {d : Nat} (a : Code d) : a ∈ codes d := by
  induction a with
  | k => simp [codes]
  | z => simp [codes]
  | reset => simp [codes]
  | t a ih => simp [codes,ih]
  | c a ih => simp [codes,ih]

theorem codes_nodup (d : Nat) : (codes d).Nodup := by
  induction d with
  | zero => decide
  | succ d ih =>
    change (.reset :: ((codes d).map Code.t ++ (codes d).map Code.c)).Nodup
    rw [List.nodup_cons]
    constructor
    · simp only [List.mem_append,List.mem_map]
      rintro (⟨x,_,h⟩ | ⟨x,_,h⟩) <;> cases h
    · rw [List.nodup_append]
      refine ⟨List.Pairwise.map _ (fun _ _ h e => h (Code.t.inj e)) ih,List.Pairwise.map _ (fun _ _ h e => h (Code.c.inj e)) ih,?_⟩
      intro a ha b hb e
      obtain ⟨x,_,rfl⟩ := List.mem_map.mp ha
      obtain ⟨y,_,rfl⟩ := List.mem_map.mp hb
      cases e

theorem codes_size (d : Nat) : (codes d).length+1=3*2^d := by
  induction d with
  | zero => decide
  | succ d ih =>
    simp only [codes,List.length_append,List.length_cons,List.length_nil,List.length_map,Nat.pow_succ]
    omega

def xCodes (d : Nat) : List (Code d) := (codes d).filter isX

theorem xCodes_succ (d : Nat) :
    xCodes (d+1) = .reset :: (codes d).map Code.t := by
  change List.filter isX (.reset :: ((codes d).map Code.t ++ (codes d).map Code.c)) = _
  rw [List.filter_cons]
  simp only [isX,ite_true,List.filter_append,List.filter_map]
  simp only [Function.comp_def,isX]
  have ht (l : List (Code d)) : l.filter (fun _ => true)=l := by
    induction l <;> simp_all [List.filter]
  have hf (l : List (Code d)) : l.filter (fun _ => false)=[] := by
    induction l <;> simp_all [List.filter]
  rw [ht,hf]
  simp

theorem xCodes_size (d : Nat) : (xCodes (d+1)).length=3*2^d := by
  rw [xCodes_succ,List.length_cons,List.length_map]
  exact codes_size d

def labelCodes : (d : Nat) → Nat → Code d × Code d
  | 0, _ => (.k,.z)
  | d+1, W =>
    if W < 2^(d+3) then
      let p := labelCodes d W
      (.t p.1,.c p.2)
    else if W=2^(d+3) then (.reset,zeroCode (d+1))
    else
      let p := labelCodes d (W-2^(d+3))
      (.t p.2,.c p.1)

theorem zeroCode_mode (d : Nat) : isX (zeroCode d)=false := by
  cases d <;> rfl

theorem labelCodes_mode (d W : Nat) :
    isX (labelCodes d W).1=true ∧ isX (labelCodes d W).2=false := by
  cases d with
  | zero => exact ⟨rfl,rfl⟩
  | succ d =>
    unfold labelCodes
    split
    · exact ⟨rfl,rfl⟩
    · split <;> exact ⟨rfl,rfl⟩

#print axioms invariant_injective
#print axioms codes_nodup
#print axioms codes_size
#print axioms xCodes_size

def negComplete (R : Type) [DecidableEq R] (r s : R) : Int := if r=s then 0 else -1

def mat : {d : Nat} → Code d → WordIndex d → WordIndex d → Int
  | _, .k => negComplete _
  | _, .z => fun _ _ => 0
  | _, .reset => negComplete _
  | _, .t a => tee (mat a) (fun _ => 1)
  | _, .c a => cee (mat a)

theorem mat_zeroCode (d : Nat) : mat (zeroCode d)=fun _ _ => 0 := by
  induction d with
  | zero => rfl
  | succ d ih =>
    simp only [zeroCode,mat,ih]
    funext r s
    cases r <;> cases s <;> rfl

def negCompleteIso {R S : Type} [DecidableEq R] [DecidableEq S] (f : IndexIso R S) :
    SignedCongruence (negComplete R) (negComplete S) where
  index := f
  weight := fun _ => 1
  weight_sign := fun _ => Or.inl rfl
  entry := by intro r s; simp [negComplete,f.eq_iff]

def zeroIso {R S : Type} (f : IndexIso R S) :
    SignedCongruence (fun (_ _ : R) => (0:Int)) (fun (_ _ : S) => (0:Int)) where
  index := f
  weight := fun _ => 1
  weight_sign := fun _ => Or.inl rfl
  entry := by intros; simp

def actualIndexIso (d W : Nat) (hW : W<2^(d+3)) (hW0 : W≠0) :
    IndexIso (RepIndex (d+3) W) (WordIndex d) := by
  simpa only [List.length_replicate] using
    rosterWordIso d W hW hW0 (List.replicate d false) (by simp)

def baseWeight (W r : Nat) : Int :=
  let anchor := if W=1 then 2 else 1
  if r=anchor then 1 else -matrix M 3 W anchor r

theorem base_signed_finite : ∀ W r s : Fin 8,
    W.val≠0 → Rep 3 W.val r.val → Rep 3 W.val s.val →
    Sign (baseWeight W.val r.val) ∧
    (if r.val=s.val then (0:Int) else -1) =
      baseWeight W.val r.val * baseWeight W.val s.val * matrix M 3 W.val r.val s.val ∧
    matrix N 3 W.val r.val s.val=0 := by
  unfold Sign
  decide +kernel

def baseSigned (W : Nat) (hW : W<2^3) (hW0 : W≠0) :
    SignedCongruence (repMatrix M 3 W) (negComplete (RepIndex 3 W)) where
  index := IndexIso.refl _
  weight := fun r => baseWeight W r.val
  weight_sign := by
    intro r
    exact (base_signed_finite ⟨W,hW⟩ ⟨r.val,r.property.1⟩ ⟨r.val,r.property.1⟩
      hW0 r.property r.property).1
  entry := by
    intro r s
    have h := (base_signed_finite ⟨W,hW⟩ ⟨r.val,r.property.1⟩ ⟨s.val,s.property.1⟩
      hW0 r.property s.property).2.1
    change (if r=s then (0:Int) else -1) = _
    by_cases he : r=s
    · subst s
      simpa only [if_pos rfl,repMatrix] using h
    · have hv : r.val≠s.val := fun e => he (Subtype.ext e)
      simpa only [if_neg he,if_neg hv,repMatrix] using h

theorem baseY (W : Nat) (hW : W<2^3) (hW0 : W≠0) :
    (fun r s => -repMatrix N 3 W r s)=fun _ _ => 0 := by
  funext r s
  have h := (base_signed_finite ⟨W,hW⟩ ⟨r.val,r.property.1⟩ ⟨s.val,s.property.1⟩
    hW0 r.property s.property).2.2
  change -matrix N 3 W r.val s.val=0
  rw [h]; rfl

theorem resetX (d : Nat) :
    repMatrix M (d+4) (2^(d+3))=negComplete (RepIndex (d+4) (2^(d+3))) := by
  funext r s
  exact (reset_matrix (d+3) r s).1

theorem resetY (d : Nat) :
    (fun r s => -repMatrix N (d+4) (2^(d+3)) r s)=fun _ _ => 0 := by
  funext r s
  have h := (reset_matrix (d+3) r s).2
  change -matrix N (d+4) (2^(d+3)) r.val s.val=0
  rw [h]; rfl

/-- Each channel has its own signed permutation; no common switching is asserted. -/
def actualNormalizations : (d W : Nat) → (hW : W<2^(d+3)) → (hW0 : W≠0) →
    SignedCongruence (repMatrix M (d+3) W) (mat (labelCodes d W).1) ×
    SignedCongruence (fun r s => -repMatrix N (d+3) W r s) (mat (labelCodes d W).2)
  | 0,W,hW,hW0 => by
    constructor
    · exact (baseSigned W hW hW0).trans (negCompleteIso (actualIndexIso 0 W hW hW0))
    · change SignedCongruence (fun r s => -repMatrix N 3 W r s) (fun _ _ => 0)
      rw [baseY W hW hW0]
      exact zeroIso (actualIndexIso 0 W hW hW0)
  | d+1,W,hW,hW0 => by
    by_cases hlo : W<2^(d+3)
    · have hc : labelCodes (d+1) W=(.t (labelCodes d W).1,.c (labelCodes d W).2) := by
        rw [labelCodes,if_pos hlo]
      rw [hc]
      have p := actualNormalizations d W hlo hW0
      exact ⟨(low_native_matrix_normalized (d+3) W hlo hW0).trans
          (tee_congr p.1 (fun _ => 1) (fun _ => 1) (fun _ => Or.inl rfl) (fun _ => Or.inl rfl)),
        (low_native_Y_normalized (d+3) W hlo hW0).trans (cee_congr p.2)⟩
    · by_cases heq : W=2^(d+3)
      · subst W
        have hc : labelCodes (d+1) (2^(d+3))=(.reset,zeroCode (d+1)) := by
          rw [labelCodes,if_neg (by omega),if_pos rfl]
        rw [hc]
        constructor
        · change SignedCongruence (repMatrix M (d+4) (2^(d+3))) (negComplete _)
          rw [resetX]
          exact negCompleteIso (actualIndexIso (d+1) _ hW hW0)
        · change SignedCongruence (fun r s => -repMatrix N (d+4) (2^(d+3)) r s) (mat (zeroCode (d+1)))
          rw [resetY,mat_zeroCode]
          exact zeroIso (actualIndexIso (d+1) _ hW hW0)
      · let v := W-2^(d+3)
        have hpow : 2^(d+1+3)=2*2^(d+3) := by rw [show d+1+3=(d+3)+1 by omega,Nat.pow_succ]; omega
        have hv : v<2^(d+3) := by omega
        have hv0 : v≠0 := by omega
        have hev : W=2^(d+3)+v := by omega
        have hc : labelCodes (d+1) W=(.t (labelCodes d v).2,.c (labelCodes d v).1) := by
          rw [labelCodes,if_neg hlo,if_neg heq]
        rw [hc]
        have p := actualNormalizations d v hv hv0
        have fx := (high_native_matrix_normalized (d+3) v hv hv0).trans
          (tee_congr p.2 (fun _ => 1) (fun _ => 1) (fun _ => Or.inl rfl) (fun _ => Or.inl rfl))
        have fy := (high_native_Y_normalized (d+3) v hv hv0).trans (cee_congr p.1)
        rw [← hev] at fx fy
        exact ⟨fx,fy⟩

def actualNativeIso (d W : Nat) (hW : W<2^(d+3)) (hW0 : W≠0) :
    GraphIso (NativeVertex (d+3) W) (CoverVertex (WordIndex d) Bool)
      (NativeAdj (d+3) W) (CoverAdj (mat (labelCodes d W).1)) :=
  GraphIso.trans (nativeCoverIso (d+3) W (by omega) hW hW0)
    (cover_iso (actualNormalizations d W hW hW0).1)

#print axioms base_signed_finite
#print axioms actualNormalizations
#print axioms actualNativeIso

theorem mat_properties {d : Nat} (a : Code d) :
    (∀ r s, Ternary (mat a r s)) ∧ (∀ r, mat a r r=0) ∧
      (∀ r s, mat a r s=mat a s r) := by
  induction a with
  | k =>
    refine ⟨?_,?_,?_⟩
    · intro r s; by_cases h : r=s <;> simp [mat,negComplete,h,Ternary,Sign]
    · intro r; simp [mat,negComplete]
    · intro r s; simp [mat,negComplete,eq_comm]
  | z => exact ⟨fun _ _ => Or.inl rfl,fun _ => rfl,fun _ _ => rfl⟩
  | reset =>
    refine ⟨?_,?_,?_⟩
    · intro r s; by_cases h : r=s <;> simp [mat,negComplete,h,Ternary,Sign]
    · intro r; simp [mat,negComplete]
    · intro r s; simp [mat,negComplete,eq_comm]
  | t a ih =>
    exact ⟨tee_ternary _ ih.1,tee_zero _,tee_sym _ ih.2.2⟩
  | c a ih =>
    refine ⟨?_,?_,?_⟩
    · intro r s; cases r <;> cases s <;> simp only [mat,cee]
      all_goals first | exact Or.inl rfl | exact ih.1 _ _
    · intro r; cases r <;> simp [mat,cee,ih.2.1]
    · intro r s; cases r <;> cases s <;> simp [mat,cee,ih.2.2]

def pos3 {R : Type} (A : R → R → Int) (r s t : R) : Int :=
  if A r s*A s t*A t r=1 then 1 else 0

theorem positive_lifted {R : Type} (rs : List R)
    (Z : Option (Bool × R) → Option (Bool × R) → Int) (h0 : Z none none=0) :
    positiveMass (lifted rs) Z =
      total rs (fun r => total rs (fun s => pairSum (fun a b =>
        pos3 Z none (some (a,r)) (some (b,s)) +
        pos3 Z (some (a,r)) none (some (b,s)) +
        pos3 Z (some (a,r)) (some (b,s)) none))) +
      total rs (fun r => total rs (fun s => total rs (fun t => tripleSum (fun a b c =>
        pos3 Z (some (a,r)) (some (b,s)) (some (c,t)))))) := by
  simp [positiveMass,pos3,total_lifted,h0,pairSum,tripleSum,total_add]
  grind

theorem tee_positive_hubs {R : Type} [DecidableEq R] (A : R → R → Int)
    (ha : ∀ r s, Ternary (A r s)) (hd : ∀ r, A r r=0)
    (hs : ∀ r s, A r s=A s r) (r s : R) :
    pairSum (fun a b =>
      pos3 (tee A (fun _ => 1)) none (some (a,r)) (some (b,s)) +
      pos3 (tee A (fun _ => 1)) (some (a,r)) none (some (b,s)) +
      pos3 (tee A (fun _ => 1)) (some (a,r)) (some (b,s)) none) =
      6*(A r s*A s r) := by
  by_cases h : r=s
  · subst s; simp [pairSum,pos3,tee,hd]
  · rcases ha r s with e | e | e <;>
      simp [pairSum,pos3,tee,h,Ne.symm h,← hs r s,e]

theorem tee_positive_block {R : Type} [DecidableEq R] (A : R → R → Int)
    (ha : ∀ r s, Ternary (A r s)) (hd : ∀ r, A r r=0)
    (hs : ∀ r s, A r s=A s r) (r s t : R) :
    tripleSum (fun a b c =>
      pos3 (tee A (fun _ => 1)) (some (a,r)) (some (b,s)) (some (c,t))) =
      8*pos3 A r s t +
      (if r=s then 4*(A r t*A t r) else 0) +
      (if s=t then 4*(A r s*A s r) else 0) +
      (if t=r then 4*(A r s*A s r) else 0) := by
  by_cases hrs : r=s
  · subst s
    by_cases hrt : r=t
    · subst t; simp [tripleSum,pairSum,pos3,tee,hd]
    · rcases ha r t with e | e | e <;>
        simp [tripleSum,pairSum,pos3,tee,hd,hrt,Ne.symm hrt,← hs r t,e]
  · by_cases hst : s=t
    · subst t
      rcases ha r s with e | e | e <;>
        simp [tripleSum,pairSum,pos3,tee,hd,hrs,Ne.symm hrs,← hs r s,e]
    · by_cases htr : t=r
      · subst t
        rcases ha r s with e | e | e <;>
          simp [tripleSum,pairSum,pos3,tee,hd,hrs,Ne.symm hrs,← hs r s,e]
      · simp [tripleSum,pairSum,pos3,tee,hrs,hst,htr]
        omega

theorem tee_positive_mass {R : Type} [DecidableEq R] (rs : List R) (hn : rs.Nodup)
    (A : R → R → Int) (ha : ∀ r s, Ternary (A r s))
    (hd : ∀ r, A r r=0) (hs : ∀ r s, A r s=A s r) :
    positiveMass (lifted rs) (tee A (fun _ => 1)) =
      18*moment2 rs A+8*positiveMass rs A := by
  rw [positive_lifted rs _ rfl]
  simp only [tee_positive_hubs A ha hd hs,tee_positive_block A ha hd hs,total_add]
  rw [triple_diag12 rs hn,triple_diag23 rs hn,triple_diag31 rs hn]
  simp [total_mul,moment2,positiveMass,pos3]
  omega

theorem cee_positive_mass {R : Type} (rs : List R) (A : R → R → Int) :
    positiveMass (lifted rs) (cee A)=8*positiveMass rs A := by
  rw [positive_lifted rs _ rfl]
  have hh : ∀ r s, pairSum (fun a b =>
      pos3 (cee A) none (some (a,r)) (some (b,s)) +
      pos3 (cee A) (some (a,r)) none (some (b,s)) +
      pos3 (cee A) (some (a,r)) (some (b,s)) none)=0 := by
    intros; simp [pos3,cee,pairSum]
  have ht : ∀ r s t, tripleSum (fun a b c =>
      pos3 (cee A) (some (a,r)) (some (b,s)) (some (c,t)))=8*pos3 A r s t := by
    intro r s t
    have he : ∀ a b c, pos3 (cee A) (some (a,r)) (some (b,s)) (some (c,t))=pos3 A r s t := by
      intros; rfl
    simp only [he,tripleSum,pairSum]
    omega
  simp only [hh,ht,total_mul]
  simp
  rfl

theorem negComplete_moment2 {R : Type} [DecidableEq R] (rs : List R) (hn : rs.Nodup) :
    moment2 rs (negComplete R)=(rs.length : Int)*((rs.length : Int)-1) := by
  have h := complement_moment2 rs hn (fun _ _ => 0) (fun _ _ => Or.inl rfl)
    (fun _ => rfl) (fun _ _ => rfl)
  have he : ∀ r s, negComplete R r s*negComplete R s r =
      complement (fun _ _ => 0) r s*complement (fun _ _ => 0) s r := by
    intro r s
    by_cases h : r=s
    · subst s; simp [negComplete,complement]
    · simp [negComplete,complement,h,Ne.symm h]
  simpa [moment2,he] using h

theorem negComplete_positive {R : Type} [DecidableEq R] (rs : List R) :
    positiveMass rs (negComplete R)=0 := by
  have h : ∀ r s t, (if negComplete R r s*negComplete R s t*negComplete R t r=1 then (1:Int) else 0)=0 := by
    intro r s t
    simp only [negComplete]
    split <;> split <;> split <;> decide
  simp [positiveMass,h]

theorem roster_order (d : Nat) : (wordRoster d).length=order d := by
  have h := wordRoster_size d
  have hp := order_pow d
  unfold Sounio.ZDUnsignedCode.parentOrder at h
  have hp' : (order d : Int)+1=(2:Int)^(d+2) := by exact_mod_cast hp
  omega

theorem mat_counts {d : Nat} (a : Code d) :
    moment2 (wordRoster d) (mat a)=2*(edges a : Int) ∧
    positiveMass (wordRoster d) (mat a)=6*(positives a : Int) := by
  induction a with
  | k => decide +kernel
  | z => decide +kernel
  | @reset d =>
    constructor
    · change moment2 (wordRoster (d+1)) (negComplete _) = 2*((order d*(2*order d+1) : Nat) : Int)
      rw [negComplete_moment2 _ (wordRoster_nodup _),roster_order (d+1)]
      simp only [order,Int.natCast_add,Int.natCast_mul,Int.natCast_one]
      grind
    · exact negComplete_positive _
  | @t d a ih =>
    have h := mat_properties a
    simp only [mat,wordRoster,tee_moment2 _ (wordRoster_nodup _) _ h.2.1,
      tee_positive_mass _ (wordRoster_nodup _) _ h.1 h.2.1 h.2.2,ih.1,ih.2,
      edges,positives,roster_order,Int.natCast_add,Int.natCast_mul]
    constructor <;> omega
  | @c d a ih =>
    simp only [mat,wordRoster,cee_moment2,cee_positive_mass,ih.1,ih.2,
      edges,positives,Int.natCast_mul]
    constructor <;> omega

#print axioms tee_positive_mass
#print axioms mat_counts

theorem moment2_perm {R : Type} {rs ss : List R} (h : rs.Perm ss) (A : R → R → Int) :
    moment2 rs A=moment2 ss A := by
  unfold moment2
  rw [total_perm h]
  apply total_ext
  intro r _
  exact total_perm h _

theorem moment3_perm {R : Type} {rs ss : List R} (h : rs.Perm ss) (A : R → R → Int) :
    moment3 rs A=moment3 ss A := by
  unfold moment3
  rw [total_perm h]
  apply total_ext
  intro r _
  rw [total_perm h]
  apply total_ext
  intro s _
  exact total_perm h _

theorem positive_perm {R : Type} {rs ss : List R} (h : rs.Perm ss) (A : R → R → Int) :
    positiveMass rs A=positiveMass ss A := by
  unfold positiveMass
  rw [total_perm h]
  apply total_ext
  intro r _
  rw [total_perm h]
  apply total_ext
  intro s _
  exact total_perm h _

theorem congr_positive {R S : Type} (rs : List R) {A : R → R → Int} {B : S → S → Int}
    (c : SignedCongruence A B) :
    positiveMass (rs.map c.index.toFun) B=positiveMass rs A := by
  simp only [positiveMass,total_map]
  apply total_ext
  intro r _
  apply total_ext
  intro s _
  apply total_ext
  intro t _
  have h : B (c.index.toFun r) (c.index.toFun s) *
      B (c.index.toFun s) (c.index.toFun t) * B (c.index.toFun t) (c.index.toFun r) =
      A r s*A s t*A t r := by
    rw [c.entry,c.entry,c.entry]
    rcases c.weight_sign r with h | h <;> rcases c.weight_sign s with j | j <;>
      rcases c.weight_sign t with k | k <;> simp [h,j,k,Int.neg_mul,Int.mul_neg]
  rw [h]

theorem congr_counts {R S : Type} {A : R → R → Int} {B : S → S → Int}
    (f : SignedCongruence A B) (rs : List R) (ss : List S)
    (hr : rs.Nodup) (hs : ss.Nodup) (cr : ∀ r, r∈rs) (cs : ∀ s, s∈ss) :
    moment2 rs A=moment2 ss B ∧ positiveMass rs A=positiveMass ss B := by
  have hp := roster_map_perm f.index rs ss hr hs cr cs
  exact ⟨(congr_moment2 rs f).symm.trans (moment2_perm hp B),
    (congr_positive rs f).symm.trans (positive_perm hp B)⟩

theorem actual_matrix_counts (d W : Nat) (hW : W<2^(d+3)) (hW0 : W≠0) :
    matrixEdges (representativeRoster (d+3) W hW hW0) (repMatrix M (d+3) W) =
      (edges (labelCodes d W).1 : Int) ∧
    positiveTriangles (representativeRoster (d+3) W hW hW0) (repMatrix M (d+3) W) =
      (positives (labelCodes d W).1 : Int) := by
  have h := congr_counts (actualNormalizations d W hW hW0).1
    (representativeRoster (d+3) W hW hW0) (wordRoster d)
    (representativeRoster_nodup _ _ hW hW0) (wordRoster_nodup _)
    (representativeRoster_complete _ _ hW hW0) (wordRoster_complete _)
  have hc := mat_counts (labelCodes d W).1
  unfold matrixEdges positiveTriangles
  rw [h.1,h.2,hc.1,hc.2]
  omega

def nativeEdges (d W : Nat) (hW : W<2^(d+3)) (hW0 : W≠0) : Int :=
  graphEdges (principalVertices (d+3) W (by omega) hW hW0)
    (nativeIndicator (d+3) W (by omega) hW hW0)

def nativeTriangles (d W : Nat) (hW : W<2^(d+3)) (hW0 : W≠0) : Int :=
  graphTriangles (principalVertices (d+3) W (by omega) hW hW0)
    (nativeIndicator (d+3) W (by omega) hW hW0)

theorem actual_native_counts (d W : Nat) (hW : W<2^(d+3)) (hW0 : W≠0) :
    nativeEdges d W hW hW0=8*(edges (labelCodes d W).1 : Int) ∧
    nativeTriangles d W hW hW0=16*(positives (labelCodes d W).1 : Int) := by
  have h := principal_graph_counts (d+3) W (by omega) hW hW0
  have hc := actual_matrix_counts d W hW hW0
  exact ⟨h.1.trans (congrArg (fun x => 8*x) hc.1),
    h.2.trans (congrArg (fun x => 16*x) hc.2)⟩

theorem entryIso_triangles {R S : Type} {A : R → R → Int} {B : S → S → Int}
    (f : EntryIso A B) (rs : List R) (ss : List S)
    (hr : rs.Nodup) (hs : ss.Nodup) (cr : ∀ r, r∈rs) (cs : ∀ s, s∈ss) :
    graphTriangles rs A=graphTriangles ss B := by
  have hp := roster_map_perm f.index rs ss hr hs cr cs
  have h' : moment3 (rs.map f.index.toFun) B=moment3 rs A := by
    simp [moment3,total_map,f.entry]
  unfold graphTriangles triangleMass
  rw [← h',moment3_perm hp]

theorem native_iso_iff_counts (d W V : Nat)
    (hW : W<2^(d+3)) (hW0 : W≠0) (hV : V<2^(d+3)) (hV0 : V≠0) :
    Nonempty (GraphIso (NativeVertex (d+3) W) (NativeVertex (d+3) V)
      (NativeAdj (d+3) W) (NativeAdj (d+3) V)) ↔
    nativeEdges d W hW hW0=nativeEdges d V hV hV0 ∧
      nativeTriangles d W hW hW0=nativeTriangles d V hV hV0 := by
  constructor
  · rintro ⟨f⟩
    let A := nativeIndicator (d+3) W (by omega) hW hW0
    let B := nativeIndicator (d+3) V (by omega) hV hV0
    let g : GraphIso _ _ (fun r s => A r s=1) (fun r s => B r s=1) := {
      toFun := f.toFun
      invFun := f.invFun
      left_inv := f.left_inv
      right_inv := f.right_inv
      map_adj_iff := by
        intro r s
        exact (nativeIndicator_iff (d+3) W (by omega) hW hW0 r s).trans
          ((f.map_adj_iff r s).trans
            (nativeIndicator_iff (d+3) V (by omega) hV hV0 _ _).symm) }
    let e := ofGraph (nativeIndicator_binary (d+3) W (by omega) hW hW0)
      (nativeIndicator_binary (d+3) V (by omega) hV hV0) g
    exact ⟨e.edges _ _ (principalVertices_nodup _ _ _ hW hW0)
        (principalVertices_nodup _ _ _ hV hV0)
        (principalVertices_complete _ _ _ hW hW0) (principalVertices_complete _ _ _ hV hV0),
      entryIso_triangles e _ _ (principalVertices_nodup _ _ _ hW hW0)
        (principalVertices_nodup _ _ _ hV hV0)
        (principalVertices_complete _ _ _ hW hW0) (principalVertices_complete _ _ _ hV hV0)⟩
  · rintro ⟨he,hp⟩
    have hw := actual_native_counts d W hW hW0
    have hv := actual_native_counts d V hV hV0
    have ec : edges (labelCodes d W).1=edges (labelCodes d V).1 := by omega
    have pc : positives (labelCodes d W).1=positives (labelCodes d V).1 := by omega
    have hc := invariant_injective _ _ ec pc
    have g := actualNativeIso d V hV hV0
    rw [← hc] at g
    exact ⟨GraphIso.trans (actualNativeIso d W hW hW0) (GraphIso.symm g)⟩

#print axioms actual_matrix_counts
#print axioms actual_native_counts
#print axioms native_iso_iff_counts

def labelFor : {d : Nat} → Code d → Nat
  | _, .k => 1
  | _, .z => 1
  | d+1, .reset => 2^(d+3)
  | d+1, .t a => if isX a then labelFor a else 2^(d+3)+labelFor a
  | d+1, .c a => if isX a then 2^(d+3)+labelFor a else labelFor a

theorem labelFor_valid {d : Nat} (a : Code d) :
    labelFor a<2^(d+3) ∧ labelFor a≠0 := by
  induction a with
  | k => decide
  | z => decide
  | @reset d =>
    have hp := Nat.two_pow_pos (d+3)
    have hpow : 2^(d+1+3)=2*2^(d+3) := by rw [show d+1+3=(d+3)+1 by omega,Nat.pow_succ]; omega
    change 2^(d+3)<2^(d+1+3) ∧ 2^(d+3)≠0
    rw [hpow]
    omega
  | @t d a ih =>
    have hp := Nat.two_pow_pos (d+3)
    have hpow : 2^(d+1+3)=2*2^(d+3) := by rw [show d+1+3=(d+3)+1 by omega,Nat.pow_succ]; omega
    simp only [labelFor,hpow]
    split <;> omega
  | @c d a ih =>
    have hp := Nat.two_pow_pos (d+3)
    have hpow : 2^(d+1+3)=2*2^(d+3) := by rw [show d+1+3=(d+3)+1 by omega,Nat.pow_succ]; omega
    simp only [labelFor,hpow]
    split <;> omega

theorem labelFor_realizes {d : Nat} (a : Code d) :
    (if isX a then (labelCodes d (labelFor a)).1 else (labelCodes d (labelFor a)).2)=a := by
  induction a with
  | k => rfl
  | z => rfl
  | @reset d =>
    change (labelCodes (d+1) (2^(d+3))).1=.reset
    rw [labelCodes,if_neg (by omega),if_pos rfl]
  | @t d a ih =>
    have hv := labelFor_valid a
    have hp := Nat.two_pow_pos (d+3)
    change (labelCodes (d+1) (labelFor (.t a))).1=.t a
    by_cases h : isX a=true
    · simp only [h,ite_true] at ih
      rw [labelFor,if_pos h,labelCodes,if_pos hv.1]
      exact congrArg Code.t ih
    · simp only [h,Bool.false_eq_true,ite_false] at ih
      rw [labelFor,if_neg h,labelCodes,if_neg (by omega),if_neg (by omega)]
      simpa only [Nat.add_sub_cancel_left] using congrArg Code.t ih
  | @c d a ih =>
    have hv := labelFor_valid a
    have hp := Nat.two_pow_pos (d+3)
    change (labelCodes (d+1) (labelFor (.c a))).2=.c a
    by_cases h : isX a=true
    · simp only [h,ite_true] at ih
      rw [labelFor,if_pos h,labelCodes,if_neg (by omega),if_neg (by omega)]
      simpa only [Nat.add_sub_cancel_left] using congrArg Code.c ih
    · simp only [h,Bool.false_eq_true,ite_false] at ih
      rw [labelFor,if_neg h,labelCodes,if_pos hv.1]
      exact congrArg Code.c ih

def nativePair {d : Nat} (a : Code d) : Int × Int :=
  (8*(edges a : Int),16*(positives a : Int))

theorem nativePair_injective {d : Nat} (a b : Code d) (h : nativePair a=nativePair b) : a=b := by
  have he := congrArg Prod.fst h
  have hp := congrArg Prod.snd h
  simp only [nativePair] at he hp
  apply invariant_injective a b <;> omega

def nativeCatalog (d : Nat) : List (Int × Int) := (xCodes d).map nativePair

theorem nativeCatalog_nodup (d : Nat) : (nativeCatalog d).Nodup := by
  have hn : (xCodes d).Nodup := (codes_nodup d).filter _
  exact List.Pairwise.map _ (fun a b h e => h (nativePair_injective a b e)) hn

theorem nativeCatalog_size (d : Nat) : (nativeCatalog (d+1)).length=3*2^d := by
  simp only [nativeCatalog,List.length_map,xCodes_size]

theorem nativeCatalog_base : (nativeCatalog 0).length=1 := by decide +kernel

theorem nativeCatalog_exact (d : Nat) (e p : Int) :
    (e,p)∈nativeCatalog d ↔ ∃ W, ∃ hW : W<2^(d+3), ∃ hW0 : W≠0,
      nativeEdges d W hW hW0=e ∧ nativeTriangles d W hW hW0=p := by
  constructor
  · intro h
    obtain ⟨a,ha,he⟩ := List.mem_map.mp h
    have hx : isX a=true := (List.mem_filter.mp ha).2
    have hv := labelFor_valid a
    have hc := labelFor_realizes a
    simp only [hx,ite_true] at hc
    refine ⟨labelFor a,hv.1,hv.2,?_⟩
    have hn := actual_native_counts d (labelFor a) hv.1 hv.2
    rw [hc] at hn
    have he1 := congrArg Prod.fst he
    have he2 := congrArg Prod.snd he
    exact ⟨hn.1.trans he1,hn.2.trans he2⟩
  · rintro ⟨W,hW,hW0,he,hp⟩
    apply List.mem_map.mpr
    refine ⟨(labelCodes d W).1,?_,?_⟩
    · exact List.mem_filter.mpr ⟨codes_complete _,(labelCodes_mode d W).1⟩
    · have hn := actual_native_counts d W hW hW0
      apply Prod.ext <;> simp only [nativePair]
      · omega
      · omega

/-- Exact finite catalogue and complete native graph invariant, at fixed depth. -/
theorem native_classification (d : Nat) :
    (nativeCatalog d).Nodup ∧
    (nativeCatalog d).length=(if d=0 then 1 else 3*2^(d-1)) ∧
    (∀ e p, (e,p)∈nativeCatalog d ↔ ∃ W, ∃ hW : W<2^(d+3), ∃ hW0 : W≠0,
      nativeEdges d W hW hW0=e ∧ nativeTriangles d W hW hW0=p) := by
  refine ⟨nativeCatalog_nodup d,?_,nativeCatalog_exact d⟩
  cases d with
  | zero => exact nativeCatalog_base
  | succ d => simpa using nativeCatalog_size d

#print axioms labelFor_valid
#print axioms labelFor_realizes
#print axioms nativeCatalog_exact
#print axioms native_classification

end Sounio.ZDSignedState
