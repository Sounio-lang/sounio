import SounioZDCounting

/-! Actual Cayley-Dickson supports represented by explicit operation words.
The classification of signed native graphs and bibliographic priority are separate.
The word family starts at representative width 3 (native algebra dimension 16).
-/
namespace Sounio.ZDSupportWord
open SounioCDCocycle Sounio.ZDAlgebraBridge Sounio.ZDRecursion
open Sounio.ZDMatrixAssembly Sounio.ZDTwinCover Sounio.ZDSignedCongruence Sounio.ZDCounting
set_option maxRecDepth 8192
set_option maxHeartbeats 16000000

/-- An exact entry-preserving bijection, carrying both computational directions. -/
structure EntryIso {R S : Type} (A : R → R → Int) (B : S → S → Int) where
  index : IndexIso R S
  entry : ∀ r s, B (index.toFun r) (index.toFun s) = A r s

def EntryIso.refl {R : Type} (A : R → R → Int) : EntryIso A A :=
  ⟨IndexIso.refl R,fun _ _ => rfl⟩

def EntryIso.trans {R S U : Type} {A : R → R → Int} {B : S → S → Int}
    {C : U → U → Int} (f : EntryIso A B) (g : EntryIso B C) : EntryIso A C where
  index := {
    toFun := fun r => g.index.toFun (f.index.toFun r)
    invFun := fun u => f.index.invFun (g.index.invFun u)
    left_inv := by intro r; simp [g.index.left_inv,f.index.left_inv]
    right_inv := by intro u; simp [f.index.right_inv,g.index.right_inv] }
  entry := by intro r s; rw [g.entry,f.entry]

def EntryIso.symm {R S : Type} {A : R → R → Int} {B : S → S → Int}
    (f : EntryIso A B) : EntryIso B A where
  index := ⟨f.index.invFun,f.index.toFun,f.index.right_inv,f.index.left_inv⟩
  entry := by intro r s; simpa [f.index.right_inv] using (f.entry (f.index.invFun r) (f.index.invFun s)).symm

def EntryIso.toSigned {R S : Type} {A : R → R → Int} {B : S → S → Int}
    (f : EntryIso A B) : SignedCongruence A B :=
  ⟨f.index,fun _ => 1,fun _ => Or.inl rfl,by intro r s; simpa using f.entry r s⟩

def ofSigned {R S : Type} {A : R → R → Int} {B : S → S → Int}
    (f : SignedCongruence A B) : EntryIso (support A) (support B) where
  index := f.index
  entry := by
    intro r s
    have hr := f.weight_sign r
    have hs := f.weight_sign s
    unfold support
    rw [f.entry]
    rcases hr with hr | hr <;> rcases hs with hs | hs <;> simp [hr,hs]

def EntryIso.compl {R S : Type} [DecidableEq R] [DecidableEq S]
    {A : R → R → Int} {B : S → S → Int} (f : EntryIso A B) :
    EntryIso (complement A) (complement B) where
  index := f.index
  entry := by intro r s; simp [complement,f.index.eq_iff,f.entry]

def EntryIso.cone {R S : Type} [DecidableEq R] [DecidableEq S]
    {A : R → R → Int} {B : S → S → Int} (f : EntryIso A B) :
    EntryIso (conePairs A) (conePairs B) :=
  ofSigned (tee_congr f.toSigned (fun _ => 1) (fun _ => 1)
    (fun _ => Or.inl rfl) (fun _ => Or.inl rfl))

def EntryIso.graph {R S : Type} {A : R → R → Int} {B : S → S → Int}
    (f : EntryIso A B) : GraphIso R S (fun r s => A r s=1) (fun r s => B r s=1) :=
  ⟨f.index.toFun,f.index.invFun,f.index.left_inv,f.index.right_inv,
   by intro r s; rw [f.entry]⟩

theorem support_tee {R : Type} [DecidableEq R] (A : R → R → Int) :
    support (tee A (fun _ => 1)) = conePairs (support A) := by
  funext x y
  cases x with
  | none => cases y with
    | none => rfl
    | some y => cases y.1 <;> simp [conePairs,support,tee]
  | some x =>
    rcases x with ⟨a,r⟩
    cases y with
    | none => cases a <;> simp [conePairs,support,tee]
    | some y =>
      rcases y with ⟨b,s⟩
      by_cases h : r=s
      · subst s; cases a <;> cases b <;> simp [conePairs,support,tee]
      · by_cases hz : A r s=0 <;> simp [conePairs,support,tee,h,hz]

theorem support_neg {R : Type} (A : R → R → Int) :
    support (fun r s => -A r s) = support A := by funext r s; simp [support]

/-- The two actual channels partition all off-diagonal representative pairs. -/
theorem channels_complement (n W : Nat) (hW : W < 2^n) :
    support (fun r s => -repMatrix N n W r s) = complement (support (repMatrix M n W)) := by
  funext r s
  by_cases he : r=s
  · subst s; simp [support,repMatrix,complement]
  · have hev : r.val ≠ s.val := fun h => he (Subtype.ext h)
    let a := basisPair n W r.val
    let b := basisPair n W s.val
    have ht : Sign (pairT a.1 a.2 b.1 b.2) :=
      mul_sign (sgn_sign _ _ (by simp [a,b,basisPair,bitsOf_length]))
        (sgn_sign _ _ (by simp [a,b,basisPair,bitsOf_length]))
    have hq : Sign (pairQ a.1 a.2 b.1 b.2) :=
      mul_sign (mul_sign ht (sgn_sign _ _ (by simp [a,b,basisPair,bitsOf_length])))
        (sgn_sign _ _ (by simp [a,b,basisPair,bitsOf_length]))
    change (if -(if r.val=s.val then 0 else N (pairT a.1 a.2 b.1 b.2)
      (pairQ a.1 a.2 b.1 b.2))=0 then 0 else 1) =
      if r=s then 0 else 1-(if (if r.val=s.val then 0 else M (pairT a.1 a.2 b.1 b.2)
        (pairQ a.1 a.2 b.1 b.2))=0 then 0 else 1)
    rcases ht with ht | ht <;> rcases hq with hq | hq <;> simp [he,hev,M,N,ht,hq]

def lowSupport (n v : Nat) (hv : v < 2^n) (hv0 : v ≠ 0) :
    EntryIso (support (repMatrix M (n+1) v)) (conePairs (support (repMatrix M n v))) := by
  rw [← support_tee]
  exact ofSigned (low_native_matrix_normalized n v hv hv0)

def highSupport (n v : Nat) (hv : v < 2^n) (hv0 : v ≠ 0) :
    EntryIso (support (repMatrix M (n+1) (2^n+v)))
      (conePairs (complement (support (repMatrix M n v)))) := by
  rw [← channels_complement n v hv,← support_tee]
  exact ofSigned (high_native_matrix_normalized n v hv hv0)

/-- Number the supplied complete duplicate-free list; no choice-based inverse. -/
def enumerationIso {R : Type} [DecidableEq R] (rs : List R)
    (hn : rs.Nodup) (hc : ∀ r, r ∈ rs) : IndexIso R (Fin rs.length) where
  toFun := fun r => ⟨rs.idxOf r,List.idxOf_lt_length_of_mem (hc r)⟩
  invFun := fun i => rs[i.val]
  left_inv := by intro r; exact List.getElem_idxOf (List.idxOf_lt_length_of_mem (hc r))
  right_inv := by intro i; apply Fin.ext; exact hn.idxOf_getElem i.val i.isLt

def sameSizeIso {R S : Type} [DecidableEq R] [DecidableEq S]
    (rs : List R) (ss : List S) (hr : rs.Nodup) (hs : ss.Nodup)
    (cr : ∀ r, r∈rs) (cs : ∀ s, s∈ss) (hlen : rs.length=ss.length) : IndexIso R S where
  toFun := fun r => ss[(enumerationIso rs hr cr).toFun r |>.val]'(by
    rw [← hlen]; exact ((enumerationIso rs hr cr).toFun r).isLt)
  invFun := fun s => rs[(enumerationIso ss hs cs).toFun s |>.val]'(by
    rw [hlen]; exact ((enumerationIso ss hs cs).toFun s).isLt)
  left_inv := by
    intro r
    simp only [enumerationIso]
    simp [hs.idxOf_getElem,List.getElem_idxOf,cr]
  right_inv := by
    intro s
    simp only [enumerationIso]
    simp [hr.idxOf_getElem,List.getElem_idxOf,cs]

def completeMatrix (R : Type) [DecidableEq R] (r s : R) : Int := if r=s then 0 else 1

def completeIso {R S : Type} [DecidableEq R] [DecidableEq S] (f : IndexIso R S) :
    EntryIso (completeMatrix R) (completeMatrix S) :=
  ⟨f,by intro r s; simp [completeMatrix,f.eq_iff]⟩

theorem allLow_complete (d : Nat) :
    wordMatrix (List.replicate d false) = completeMatrix (WordIndex (List.replicate d false).length) := by
  induction d with
  | zero => rfl
  | succ d ih =>
    funext x y
    simp only [List.replicate_succ,wordMatrix,Bool.false_eq_true,↓reduceIte]
    rw [ih]
    cases x with
    | none => cases y with
      | none => rfl
      | some y =>
        rcases y with ⟨b,s⟩
        cases b <;> simp [conePairs,support,tee,completeMatrix]
    | some x =>
      rcases x with ⟨a,r⟩
      cases y with
      | none => cases a <;> simp [conePairs,support,tee,completeMatrix]
      | some y =>
        rcases y with ⟨b,s⟩
        by_cases h : r=s
        · subst s; cases a <;> cases b <;> simp [conePairs,support,tee,completeMatrix]
        · simp [conePairs,support,tee,completeMatrix,h]

theorem representative_word_length (d W : Nat) (hW : W < 2^(d+3)) (hW0 : W ≠ 0)
    (bs : List Bool) (hb : bs.length=d) :
    (representativeRoster (d+3) W hW hW0).length=(wordRoster bs.length).length := by
  have hr := (representatives_certificate (d+3) W hW0 hW).2.2
  have hv := congrArg List.length (representativeRoster_values (d+3) W hW hW0)
  simp only [List.length_map] at hv
  have hw := wordRoster_size bs.length
  rw [hb] at hw
  unfold Sounio.ZDUnsignedCode.parentOrder at hw
  have hp : (2 : Int)^(d+3)=2*(2 : Int)^(d+2) := by
    rw [show d+3=(d+2)+1 by omega,Int.pow_succ]; omega
  have hi : (2 : Int)*((representativeRoster (d+3) W hW hW0).length:Int)+2=(2:Int)^(d+3) := by
    exact_mod_cast (hv.symm ▸ hr)
  rw [hb]
  omega

def rosterWordIso (d W : Nat) (hW : W < 2^(d+3)) (hW0 : W ≠ 0)
    (bs : List Bool) (hb : bs.length=d) : IndexIso (RepIndex (d+3) W) (WordIndex bs.length) :=
  sameSizeIso (representativeRoster (d+3) W hW hW0) (wordRoster bs.length)
    (representativeRoster_nodup _ _ hW hW0) (wordRoster_nodup _)
    (representativeRoster_complete _ _ hW hW0) (wordRoster_complete _)
    (representative_word_length d W hW hW0 bs hb)

theorem base_support_finite : ∀ W r s : Fin 8,
    W.val ≠ 0 → Rep 3 W.val r.val → Rep 3 W.val s.val →
    (if matrix M 3 W.val r.val s.val=0 then (0:Int) else 1) =
      (if r.val=s.val then 0 else 1) := by decide +kernel

theorem base_support (W : Nat) (hW : W < 2^3) (hW0 : W ≠ 0) :
    support (repMatrix M 3 W) = completeMatrix (RepIndex 3 W) := by
  funext r s
  have h := base_support_finite ⟨W,hW⟩ ⟨r.val,r.property.1⟩ ⟨s.val,s.property.1⟩ hW0 r.property s.property
  change (if matrix M 3 W r.val s.val=0 then (0:Int) else 1) =
    (if r.val=s.val then 0 else 1) at h
  change (if matrix M 3 W r.val s.val=0 then (0:Int) else 1) =
    (if r=s then 0 else 1)
  by_cases he : r=s
  · subst s
    simpa only [if_pos rfl] using h
  · have hv : r.val ≠ s.val := fun e => he (Subtype.ext e)
    simp only [if_neg he]
    simpa only [if_neg hv] using h

theorem reset_support (d : Nat) :
    support (repMatrix M (d+4) (2^(d+3))) = completeMatrix (RepIndex (d+4) (2^(d+3))) := by
  funext r s
  have h := (reset_matrix (d+3) r s).1
  change (if matrix M (d+4) (2^(d+3)) r.val s.val=0 then 0 else 1) = _
  rw [h]
  by_cases he : r=s <;> simp [he,completeMatrix]

#print axioms channels_complement
#print axioms lowSupport
#print axioms highSupport
#print axioms enumerationIso
#print axioms allLow_complete
#print axioms representative_word_length
#print axioms base_support
#print axioms reset_support

/-- Operation word for an actual label. Invalid labels are merely totalized data. -/
def labelWord : Nat → Nat → List Bool
  | 0, _ => []
  | d+1, W =>
    if W < 2^(d+3) then false :: labelWord d W
    else if W = 2^(d+3) then List.replicate (d+1) false
    else true :: labelWord d (W-2^(d+3))

theorem labelWord_length (d W : Nat) : (labelWord d W).length=d := by
  induction d generalizing W with
  | zero => rfl
  | succ d ih =>
    simp only [labelWord]
    split
    · simp [ih]
    · split <;> simp [ih]

/-- Every actual support at and above the sedenions has a computational word isomorphism. -/
def supportWordIso : (d W : Nat) → (hW : W < 2^(d+3)) → (hW0 : W ≠ 0) →
    EntryIso (support (repMatrix M (d+3) W)) (wordMatrix (labelWord d W))
  | 0, W, hW, hW0 => by
    rw [base_support W hW hW0]
    exact completeIso (rosterWordIso 0 W hW hW0 [] rfl)
  | d+1, W, hW, hW0 => by
    by_cases hlo : W < 2^(d+3)
    · have hlw : labelWord (d+1) W = false :: labelWord d W := by
        rw [labelWord,if_pos hlo]
      rw [hlw]
      exact (lowSupport (d+3) W hlo hW0).trans
        (supportWordIso d W hlo hW0).cone
    · by_cases he : W=2^(d+3)
      · subst W
        rw [labelWord,if_neg hlo,if_pos rfl,reset_support,allLow_complete]
        exact completeIso (rosterWordIso (d+1) (2^(d+3)) hW hW0
          (List.replicate (d+1) false) (by simp))
      · let v := W-2^(d+3)
        have hg := Nat.two_pow_pos (d+3)
        have hw : W < 2^(d+3)*2 := by simpa [Nat.pow_succ,Nat.add_assoc] using hW
        have hv : v < 2^(d+3) := by dsimp [v]; omega
        have hv0 : v ≠ 0 := by dsimp [v]; omega
        have hev : 2^(d+3)+v=W := by dsimp [v]; omega
        have f := (highSupport (d+3) v hv hv0).trans
          (supportWordIso d v hv hv0).compl.cone
        rw [hev] at f
        have hlw : labelWord (d+1) W = true :: labelWord d v := by
          rw [labelWord,if_neg hlo,if_neg he]
        rw [hlw]
        exact f

def actualSupportGraphIso (d W : Nat) (hW : W < 2^(d+3)) (hW0 : W ≠ 0) :
    GraphIso (RepIndex (d+3) W) (WordIndex (labelWord d W).length)
      (fun r s => support (repMatrix M (d+3) W) r s=1)
      (fun r s => wordMatrix (labelWord d W) r s=1) :=
  (supportWordIso d W hW hW0).graph

#print axioms labelWord_length
#print axioms supportWordIso
#print axioms actualSupportGraphIso

theorem total_perm {R : Type} {rs ss : List R} (h : rs.Perm ss) (f : R → Int) :
    total rs f=total ss f := by
  induction h with
  | nil => rfl
  | cons a h ih => simp [total,ih]
  | swap a b rs => simp [total]; omega
  | trans h k ih ik => exact ih.trans ik

theorem roster_map_perm {R S : Type} (f : IndexIso R S)
    (rs : List R) (ss : List S) (hr : rs.Nodup) (hs : ss.Nodup)
    (cr : ∀ r, r∈rs) (cs : ∀ s, s∈ss) : (rs.map f.toFun).Perm ss := by
  have hm : (rs.map f.toFun).Nodup :=
    List.Pairwise.map _ (fun _ _ h e => h (f.injective e)) hr
  apply (List.perm_ext_iff_of_nodup hm hs).mpr
  intro s
  exact ⟨fun _ => cs s,fun _ => List.mem_map.mpr ⟨f.invFun s,cr _,f.right_inv s⟩⟩

theorem EntryIso.edges {R S : Type} {A : R → R → Int} {B : S → S → Int}
    (f : EntryIso A B) (rs : List R) (ss : List S)
    (hr : rs.Nodup) (hs : ss.Nodup) (cr : ∀ r, r∈rs) (cs : ∀ s, s∈ss) :
    graphEdges rs A=graphEdges ss B := by
  have hp := roster_map_perm f.index rs ss hr hs cr cs
  have h : edgeMass (rs.map f.index.toFun) B = edgeMass ss B := by
    unfold edgeMass
    rw [total_perm hp]
    apply total_ext
    intro s _
    exact total_perm hp (B s)
  have h' : edgeMass (rs.map f.index.toFun) B = edgeMass rs A := by
    simp [edgeMass,total_map,f.entry]
  unfold graphEdges
  rw [← h',h]

def actualEdges (d W : Nat) (hW : W < 2^(d+3)) (hW0 : W ≠ 0) : Int :=
  graphEdges (representativeRoster (d+3) W hW hW0) (support (repMatrix M (d+3) W))

theorem actualEdges_code (d W : Nat) (hW : W < 2^(d+3)) (hW0 : W ≠ 0) :
    actualEdges d W hW hW0 = Sounio.ZDUnsignedCode.edgeCode (labelWord d W) := by
  unfold actualEdges
  rw [(supportWordIso d W hW hW0).edges
    (representativeRoster (d+3) W hW hW0) (wordRoster (labelWord d W).length)
    (representativeRoster_nodup _ _ hW hW0) (wordRoster_nodup _)
    (representativeRoster_complete _ _ hW hW0) (wordRoster_complete _)]
  rw [← wordEdges_actual,wordEdges_code]

theorem decode_actualEdges (d W : Nat) (hW : W < 2^(d+3)) (hW0 : W ≠ 0) :
    Sounio.ZDUnsignedCode.decode d (actualEdges d W hW hW0) = labelWord d W := by
  rw [actualEdges_code]
  simpa only [labelWord_length] using Sounio.ZDUnsignedCode.decode_edgeCode (labelWord d W)

theorem native_edges_code (d W : Nat) (hW : W < 2^(d+3)) (hW0 : W ≠ 0) :
    graphEdges (principalVertices (d+3) W (by omega) hW hW0)
      (nativeIndicator (d+3) W (by omega) hW hW0) =
      8*Sounio.ZDUnsignedCode.edgeCode (labelWord d W) := by
  rw [(principal_graph_counts (d+3) W (by omega) hW hW0).1]
  have hm : matrixEdges (representativeRoster (d+3) W hW hW0) (repMatrix M (d+3) W) =
      actualEdges d W hW hW0 := by
    let A := repMatrix M (d+3) W
    have hs : ∀ r s, A r s=A s r := matrix_symmetry M (d+3) W hW
    have ht : ∀ r s, Ternary (A r s) := matrix_M_ternary (d+3) W
    calc
      matrixEdges (representativeRoster (d+3) W hW hW0) A =
          matrixEdges (representativeRoster (d+3) W hW hW0) (support A) := by
        unfold matrixEdges
        rw [support_moment2 _ A ht hs]
      _ = actualEdges d W hW hW0 :=
        binary_matrixEdges _ (support A) (support_binary A) (support_sym A hs)
  rw [hm,actualEdges_code]

theorem decode_native_edges (d W : Nat) (hW : W < 2^(d+3)) (hW0 : W ≠ 0) :
    Sounio.ZDUnsignedCode.decode d
      (graphEdges (principalVertices (d+3) W (by omega) hW hW0)
        (nativeIndicator (d+3) W (by omega) hW hW0) / 8) = labelWord d W := by
  rw [native_edges_code]
  have h : (8*Sounio.ZDUnsignedCode.edgeCode (labelWord d W))/8 =
      Sounio.ZDUnsignedCode.edgeCode (labelWord d W) := by omega
  rw [h]
  simpa only [labelWord_length] using Sounio.ZDUnsignedCode.decode_edgeCode (labelWord d W)

/-- Every word is realized by a valid actual label; reset cases are not needed for surjectivity. -/
def realizeLabel : List Bool → Nat
  | [] => 1
  | b::bs => if b then 2^(bs.length+3)+realizeLabel bs else realizeLabel bs

theorem realizeLabel_valid (bs : List Bool) :
    realizeLabel bs ≠ 0 ∧ realizeLabel bs < 2^(bs.length+3) := by
  induction bs with
  | nil => decide
  | cons b bs ih =>
    have hp : 2^((b::bs).length+3)=2^(bs.length+3)*2 := by
      simp only [List.length_cons]
      rw [show bs.length+1+3=(bs.length+3)+1 by omega,Nat.pow_succ]
    rw [hp]
    cases b <;> simp only [realizeLabel,Bool.false_eq_true,↓reduceIte]
    all_goals have := Nat.two_pow_pos (bs.length+3); omega

theorem labelWord_realizeLabel (bs : List Bool) :
    labelWord bs.length (realizeLabel bs)=bs := by
  induction bs with
  | nil => rfl
  | cons b bs ih =>
    have hv := realizeLabel_valid bs
    cases b
    · simp only [realizeLabel,Bool.false_eq_true,↓reduceIte,List.length_cons,labelWord,if_pos hv.2,ih]
    · have hlo : ¬2^(bs.length+3)+realizeLabel bs < 2^(bs.length+3) := by omega
      have he : 2^(bs.length+3)+realizeLabel bs ≠ 2^(bs.length+3) := by omega
      simp only [realizeLabel,Bool.false_eq_true,↓reduceIte,List.length_cons,labelWord,if_neg hlo,if_neg he,Nat.add_sub_cancel_left,ih]

theorem actual_support_realizes_every_word (bs : List Bool) :
    Nonempty (EntryIso
      (support (repMatrix M (bs.length+3) (realizeLabel bs))) (wordMatrix bs)) := by
  have f := supportWordIso bs.length (realizeLabel bs) (realizeLabel_valid bs).2 (realizeLabel_valid bs).1
  rw [labelWord_realizeLabel] at f
  exact ⟨f⟩

/-- For binary matrices, an ordinary graph isomorphism is exact entry preservation. -/
def ofGraph {R S : Type} {A : R → R → Int} {B : S → S → Int}
    (ha : ∀ r s, Binary (A r s)) (hb : ∀ r s, Binary (B r s))
    (f : GraphIso R S (fun r s => A r s=1) (fun r s => B r s=1)) : EntryIso A B where
  index := ⟨f.toFun,f.invFun,f.left_inv,f.right_inv⟩
  entry := by
    intro r s
    have h := f.map_adj_iff r s
    rcases ha r s with ha | ha <;> rcases hb (f.toFun r) (f.toFun s) with hb | hb <;>
      simp_all

theorem actual_support_iso_iff_edges (d W V : Nat)
    (hW : W < 2^(d+3)) (hW0 : W ≠ 0) (hV : V < 2^(d+3)) (hV0 : V ≠ 0) :
    Nonempty (GraphIso (RepIndex (d+3) W) (RepIndex (d+3) V)
      (fun r s => support (repMatrix M (d+3) W) r s=1)
      (fun r s => support (repMatrix M (d+3) V) r s=1)) ↔
      actualEdges d W hW hW0=actualEdges d V hV hV0 := by
  constructor
  · rintro ⟨f⟩
    exact (ofGraph (support_binary _) (support_binary _) f).edges
      (representativeRoster (d+3) W hW hW0) (representativeRoster (d+3) V hV hV0)
      (representativeRoster_nodup _ _ hW hW0) (representativeRoster_nodup _ _ hV hV0)
      (representativeRoster_complete _ _ hW hW0) (representativeRoster_complete _ _ hV hV0)
  · intro h
    have hw : labelWord d W=labelWord d V := by
      rw [← decode_actualEdges d W hW hW0,← decode_actualEdges d V hV hV0,h]
    have f := supportWordIso d W hW hW0
    rw [hw] at f
    exact ⟨(f.trans (supportWordIso d V hV hV0).symm).graph⟩

#print axioms actualEdges_code
#print axioms decode_actualEdges
#print axioms native_edges_code
#print axioms decode_native_edges
#print axioms labelWord_realizeLabel
#print axioms actual_support_realizes_every_word
#print axioms actual_support_iso_iff_edges

theorem actual_edges_range (d : Nat) (e : Int) :
    e ∈ Sounio.ZDUnsignedCode.codes d ↔
      ∃ W, ∃ (hW : W < 2^(d+3)), ∃ (hW0 : W ≠ 0), actualEdges d W hW hW0=e := by
  constructor
  · intro h
    obtain ⟨bs,hb,he⟩ := (Sounio.ZDUnsignedCode.mem_codes_iff d e).mp h
    have hv := realizeLabel_valid bs
    have hv' : realizeLabel bs < 2^(d+3) := by simpa only [hb] using hv.2
    refine ⟨realizeLabel bs,hv',hv.1,?_⟩
    rw [actualEdges_code,← hb,labelWord_realizeLabel,he]
  · rintro ⟨W,hW,hW0,rfl⟩
    exact (Sounio.ZDUnsignedCode.mem_codes_iff d _).mpr
      ⟨labelWord d W,labelWord_length d W,(actualEdges_code d W hW hW0).symm⟩

/-- Exact distinct edge-value catalog. Combined with the isomorphism iff, it counts support classes. -/
theorem support_class_catalog (d : Nat) :
    (Sounio.ZDUnsignedCode.codes d).Nodup ∧
    (Sounio.ZDUnsignedCode.codes d).length=2^d ∧
    (∀ e, e∈Sounio.ZDUnsignedCode.codes d ↔
      ∃ W, ∃ (hW : W < 2^(d+3)), ∃ (hW0 : W ≠ 0), actualEdges d W hW hW0=e) :=
  ⟨Sounio.ZDUnsignedCode.codes_nodup d,Sounio.ZDUnsignedCode.codes_length d,actual_edges_range d⟩

#print axioms actual_edges_range
#print axioms support_class_catalog

end Sounio.ZDSupportWord
