import SounioZDMatrixAssembly

/-!
Explicit native Cayley-Dickson graph isomorphism to the signed cover of M,
with two independent twins per cover vertex. Vertices are normalized
two-term native elements; adjacency is literal vanishing of both products.
The full two-count classification and maximal-twin uniqueness are separate.
-/
namespace Sounio.ZDTwinCover
open SounioCDCocycle Sounio.ZDAlgebraBridge Sounio.ZDRecursion Sounio.ZDMatrixAssembly
set_option maxRecDepth 4096
set_option maxHeartbeats 8000000

def flip (α : Bool) (s : Int) : Int := if α then -s else s
def pickIndex (W : Nat) (α : Bool) (a : Nat) : Nat := if α then a ^^^ W else a

theorem flip_sign (α : Bool) (s : Int) (hs : Sign s) : Sign (flip α s) := by
  cases α
  · exact hs
  · exact neg_sign hs

@[simp] theorem flip_twice (α : Bool) (s : Int) : flip α (flip α s) = s := by
  cases α <;> simp [flip]

@[simp] theorem pickIndex_xor (W a : Nat) (α : Bool) :
    pickIndex W α a ^^^ W = pickIndex W (!α) a := by
  cases α <;> simp [pickIndex,Nat.xor_assoc]

theorem pickIndex_bits (n W a : Nat) (α : Bool) :
    bitsOf n (pickIndex W α a) = pick α (bitsOf n a) (bitsOf n (a ^^^ W)) := by
  cases α <;> rfl

/-- Swapping one XOR partner preserves Q and multiplies T by Q. -/
theorem partner_swap (d : Generic) (α β : Bool) :
    pairT (pick α d.a d.ap) (pick (!α) d.a d.ap)
      (pick β d.b d.bp) (pick (!β) d.b d.bp) =
        oldT d * (if xor α β then oldQ d else 1) ∧
    pairQ (pick α d.a d.ap) (pick (!α) d.a d.ap)
      (pick β d.b d.bp) (pick (!β) d.b d.bp) = oldQ d := by
  have hA := sgn_sign d.a d.b d.lb.symm
  have hB := sgn_sign d.ap d.bp (d.lap.trans d.lbp.symm)
  have hC := sgn_sign d.a d.bp d.lbp.symm
  have hD := sgn_sign d.ap d.b (d.lap.trans d.lb.symm)
  cases α <;> cases β <;>
    simp only [pick,Bool.not_true,Bool.not_false,Bool.false_eq_true,↓reduceIte,
      Bool.xor_self,Bool.false_xor,Bool.true_xor,pairQ,pairT,oldQ,oldT]
  all_goals
    rcases hA with hA | hA <;> rcases hB with hB | hB <;>
    rcases hC with hC | hC <;> rcases hD with hD | hD <;>
      simp [pairT,pairQ,hA,hB,hC,hD,oldT,oldQ]

theorem partner_criterion (t q s u : Int) (α β : Bool) :
    (q = -1 ∧ flip α s * flip β u = t * (if xor α β then q else 1)) ↔
      q = -1 ∧ s*u = t := by
  by_cases hq : q = -1
  · cases α <;> cases β <;> simp [flip,hq,Int.neg_mul,Int.mul_neg]
  · simp [hq]

abbrev Spin := {s : Int // Sign s}
abbrev NativeIndex (n W : Nat) := {a : Nat // a < 2^n ∧ a ≠ 0 ∧ a ≠ W}

structure NativeVertex (n W : Nat) where
  index : NativeIndex n W
  spin : Spin
deriving DecidableEq

structure TwinVertex (n W : Nat) where
  rep : RepIndex n W
  spin : Spin
  copy : Bool
deriving DecidableEq

def NativeAdj (n W : Nat) (x y : NativeVertex n W) : Prop :=
  nativeProduct n W x.index.val y.index.val x.spin.val y.spin.val = (zeroCoeff,zeroCoeff) ∧
  nativeProduct n W y.index.val x.index.val y.spin.val x.spin.val = (zeroCoeff,zeroCoeff)

/-- The adjacency of B(M)[two independent vertices], independent of copy bits. -/
def TwinAdj (n W : Nat) (x y : TwinVertex n W) : Prop :=
  x.spin.val * y.spin.val = matrix M n W x.rep.val y.rep.val

theorem picked_admissible (n W : Nat) (hW : W < 2^n)
    (r : RepIndex n W) (α : Bool) :
    pickIndex W α r.val < 2^n ∧ pickIndex W α r.val ≠ 0 ∧ pickIndex W α r.val ≠ W := by
  cases α
  · exact ⟨r.property.1,r.property.2.1,rep_ne_label r.property⟩
  · refine ⟨(rep_partner hW r.property).1,(rep_partner hW r.property).2.1,?_⟩
    intro e
    have h := congrArg (fun a : Nat => a ^^^ W) e
    exact r.property.2.1 (by simpa [pickIndex,Nat.xor_assoc] using h)

def encode (n W : Nat) (hW : W < 2^n) (x : TwinVertex n W) : NativeVertex n W :=
  ⟨⟨pickIndex W x.copy x.rep.val,picked_admissible n W hW x.rep x.copy⟩,
   ⟨flip x.copy x.spin.val,flip_sign x.copy x.spin.val x.spin.property⟩⟩

theorem reverse_rep (n W : Nat) (hW : W < 2^n) (hW0 : W ≠ 0)
    (a : NativeIndex n W) (ha : ¬a.val < a.val ^^^ W) :
    Rep n W (a.val ^^^ W) := by
  rcases rep_orientation hW hW0 a.property.1 a.property.2.1 a.property.2.2 with h | h
  · exact False.elim (ha h.2.2)
  · exact h

/-- Computable inverse: compare the actual index with its XOR partner. -/
def decode (n W : Nat) (hW : W < 2^n) (hW0 : W ≠ 0)
    (x : NativeVertex n W) : TwinVertex n W :=
  if h : x.index.val < x.index.val ^^^ W then
    ⟨⟨x.index.val,⟨x.index.property.1,x.index.property.2.1,h⟩⟩,x.spin,false⟩
  else
    ⟨⟨x.index.val ^^^ W,reverse_rep n W hW hW0 x.index h⟩,
     ⟨-x.spin.val,neg_sign x.spin.property⟩,true⟩

#print axioms partner_swap
#print axioms partner_criterion
#print axioms encode
#print axioms decode

theorem native_ext (x y : NativeVertex n W)
    (hi : x.index.val = y.index.val) (hs : x.spin.val = y.spin.val) : x = y := by
  cases x with
  | mk xi xs =>
    cases y with
    | mk yi ys =>
      have hi' : xi = yi := Subtype.ext hi
      have hs' : xs = ys := Subtype.ext hs
      subst yi; subst ys; rfl

theorem twin_ext (x y : TwinVertex n W)
    (hr : x.rep.val = y.rep.val) (hs : x.spin.val = y.spin.val)
    (hc : x.copy = y.copy) : x = y := by
  cases x with
  | mk xr xs xc =>
    cases y with
    | mk yr ys yc =>
      have hr' : xr = yr := Subtype.ext hr
      have hs' : xs = ys := Subtype.ext hs
      subst yr; subst ys
      cases hc
      rfl

theorem encode_decode (n W : Nat) (hW : W < 2^n) (hW0 : W ≠ 0)
    (x : NativeVertex n W) :
    encode n W hW (decode n W hW hW0 x) = x := by
  by_cases h : x.index.val < x.index.val ^^^ W
  · apply native_ext <;> simp [decode,encode,h,pickIndex,flip]
  · apply native_ext <;> simp [decode,encode,h,pickIndex,flip,Nat.xor_assoc]

theorem decode_encode (n W : Nat) (hW : W < 2^n) (hW0 : W ≠ 0)
    (x : TwinVertex n W) :
    decode n W hW hW0 (encode n W hW x) = x := by
  cases x with
  | mk r s α =>
    have hr := r.property.2.2
    cases α
    · apply twin_ext <;> simp [decode,encode,pickIndex,flip,hr]
    · have hn : ¬(r.val ^^^ W) < r.val := by omega
      apply twin_ext <;> simp [decode,encode,pickIndex,flip,Nat.xor_assoc,hn]

theorem encode_injective (n W : Nat) (hW : W < 2^n) (hW0 : W ≠ 0)
    (x y : TwinVertex n W) (h : encode n W hW x = encode n W hW y) : x = y := by
  have e := congrArg (decode n W hW hW0) h
  simpa only [decode_encode] using e

theorem encode_surjective (n W : Nat) (hW : W < 2^n) (hW0 : W ≠ 0)
    (x : NativeVertex n W) : ∃ y, encode n W hW y = x :=
  ⟨decode n W hW hW0 x,encode_decode n W hW hW0 x⟩

theorem picked_cross (n W : Nat) (r u : RepIndex n W) (hne : r ≠ u) (α β : Bool) :
    pickIndex W α r.val ≠ pickIndex W β u.val ∧
    pickIndex W α r.val ≠ pickIndex W β u.val ^^^ W := by
  have hru : r.val ≠ u.val := fun e => hne (Subtype.ext e)
  have hcross := rep_cross r.property u.property
  have hcross' := Ne.symm (rep_cross u.property r.property)
  have hp : r.val ^^^ W ≠ u.val ^^^ W := by
    intro e
    have h := congrArg (fun a : Nat => a ^^^ W) e
    exact hru (by simpa [Nat.xor_assoc] using h)
  cases α <;> cases β <;> simp [pickIndex,Nat.xor_assoc,hru,hcross,hcross',hp]

theorem picked_same_coset (W a : Nat) (α β : Bool) :
    pickIndex W α a = pickIndex W β a ∨
      pickIndex W α a = pickIndex W β a ^^^ W := by
  cases α <;> cases β <;> simp [pickIndex,Nat.xor_assoc]

/-- The full encoded adjacency law, with same-coset cases explicitly excluded by products. -/
theorem encode_adjacency (n W : Nat) (hn : 1 ≤ n) (hW : W < 2^n)
    (x y : TwinVertex n W) :
    NativeAdj n W (encode n W hW x) (encode n W hW y) ↔ TwinAdj n W x y := by
  rw [NativeAdj,native_adjacency_bit_iff n W _ _ _ _ hn hW
    (encode n W hW x).index.property.1 (encode n W hW y).index.property.1
    (encode n W hW x).index.property.2.1 (encode n W hW y).index.property.2.1
    (encode n W hW x).index.property.2.2 (encode n W hW y).index.property.2.2
    (encode n W hW x).spin.property (encode n W hW y).spin.property]
  by_cases he : x.rep = y.rep
  · have same := picked_same_coset W x.rep.val x.copy y.copy
    have hprod := mul_sign x.spin.property y.spin.property
    have hv : x.rep.val = y.rep.val := congrArg Subtype.val he
    have hright : ¬TwinAdj n W x y := by
      unfold TwinAdj
      rw [hv,matrix_diagonal]
      rcases hprod with hp | hp <;> simp [hp]
    constructor
    · intro h
      apply False.elim
      rcases same with hsame | hsame
      · exact h.1 (by simpa only [encode,hv] using hsame)
      · exact h.2.1 (by simpa only [encode,hv] using hsame)
    · intro h
      exact False.elim (hright h)
  · have hc := picked_cross n W x.rep y.rep he x.copy y.copy
    have hp := partner_swap (repGeneric n W hW x.rep y.rep he) x.copy y.copy
    dsimp [repGeneric,genericOfNat] at hp
    dsimp [encode]
    rw [pickIndex_xor,pickIndex_xor,pickIndex_bits,pickIndex_bits,pickIndex_bits,pickIndex_bits]
    rw [hp.1,hp.2]
    have heval : x.rep.val ≠ y.rep.val := fun e => he (Subtype.ext e)
    change (_ ∧ _ ∧ oldQ (repGeneric n W hW x.rep y.rep he) = -1 ∧
      flip x.copy x.spin.val * flip y.copy y.spin.val =
        oldT (repGeneric n W hW x.rep y.rep he) *
          (if xor x.copy y.copy then oldQ (repGeneric n W hW x.rep y.rep he) else 1)) ↔ _
    constructor
    · rintro ⟨_,_,hadj⟩
      have hbase := (partner_criterion _ _ _ _ x.copy y.copy).mp hadj
      change x.spin.val*y.spin.val = matrix M n W x.rep.val y.rep.val
      simp only [matrix,heval,↓reduceIte,basisPair]
      exact (sign_eq_M _ _ _ (mul_sign x.spin.property y.spin.property)).mpr hbase
    · intro h
      have hbase : oldQ (repGeneric n W hW x.rep y.rep he) = -1 ∧
          x.spin.val*y.spin.val = oldT (repGeneric n W hW x.rep y.rep he) := by
        change x.spin.val*y.spin.val = matrix M n W x.rep.val y.rep.val at h
        simp only [matrix,heval,↓reduceIte,basisPair] at h
        exact (sign_eq_M _ _ _ (mul_sign x.spin.property y.spin.property)).mp h
      exact ⟨hc.1,by simpa only [pickIndex_xor] using hc.2,(partner_criterion _ _ _ _ x.copy y.copy).mpr hbase⟩

#print axioms encode_decode
#print axioms decode_encode
#print axioms encode_adjacency

/-- An explicit graph isomorphism with a computable forward map and inverse. -/
structure GraphIso (V U : Type) (A : V → V → Prop) (B : U → U → Prop) where
  toFun : V → U
  invFun : U → V
  left_inv : ∀ x, invFun (toFun x) = x
  right_inv : ∀ y, toFun (invFun y) = y
  map_adj_iff : ∀ x y, A x y ↔ B (toFun x) (toFun y)

/-- Universal isomorphism G = B(M)[two independent twins]. -/
def native_twin_iso (n W : Nat) (hn : 1 ≤ n) (hW : W < 2^n) (hW0 : W ≠ 0) :
    GraphIso (NativeVertex n W) (TwinVertex n W) (NativeAdj n W) (TwinAdj n W) where
  toFun := decode n W hW hW0
  invFun := encode n W hW
  left_inv := encode_decode n W hW hW0
  right_inv := decode_encode n W hW hW0
  map_adj_iff := by
    intro x y
    have h := encode_adjacency n W hn hW (decode n W hW hW0 x) (decode n W hW hW0 y)
    simpa only [encode_decode] using h

def mate (x : TwinVertex n W) : TwinVertex n W := ⟨x.rep,x.spin,!x.copy⟩

theorem mate_ne (x : TwinVertex n W) : mate x ≠ x := by
  intro h
  have e := congrArg TwinVertex.copy h
  cases hc : x.copy <;> simp [mate,hc] at e

@[simp] theorem mate_mate (x : TwinVertex n W) : mate (mate x) = x := by
  apply twin_ext <;> simp [mate]

theorem mate_independent (n W : Nat) (x : TwinVertex n W) :
    ¬TwinAdj n W x (mate x) := by
  have hprod := mul_sign x.spin.property x.spin.property
  simp only [TwinAdj,mate,matrix_diagonal]
  rcases hprod with h | h <;> simp [h]

theorem mate_same_neighbors (n W : Nat) (x y : TwinVertex n W) :
    TwinAdj n W (mate x) y ↔ TwinAdj n W x y := Iff.rfl

/-- The corresponding explicit nontrivial involution on actual native vertices. -/
def nativeMate (n W : Nat) (hW : W < 2^n) (hW0 : W ≠ 0)
    (x : NativeVertex n W) : NativeVertex n W :=
  encode n W hW (mate (decode n W hW hW0 x))

theorem nativeMate_twice (n W : Nat) (hW : W < 2^n) (hW0 : W ≠ 0)
    (x : NativeVertex n W) :
    nativeMate n W hW hW0 (nativeMate n W hW hW0 x) = x := by
  simp only [nativeMate,decode_encode,mate_mate,encode_decode]

theorem nativeMate_ne (n W : Nat) (hW : W < 2^n) (hW0 : W ≠ 0)
    (x : NativeVertex n W) : nativeMate n W hW hW0 x ≠ x := by
  intro h
  have e := congrArg (decode n W hW hW0) h
  simp only [nativeMate,decode_encode] at e
  exact mate_ne (decode n W hW hW0 x) e

theorem nativeMate_independent (n W : Nat) (hn : 1 ≤ n)
    (hW : W < 2^n) (hW0 : W ≠ 0) (x : NativeVertex n W) :
    ¬NativeAdj n W x (nativeMate n W hW hW0 x) := by
  rw [(native_twin_iso n W hn hW hW0).map_adj_iff]
  simp only [native_twin_iso,nativeMate,decode_encode]
  exact mate_independent n W (decode n W hW hW0 x)

theorem nativeMate_same_neighbors (n W : Nat) (hn : 1 ≤ n)
    (hW : W < 2^n) (hW0 : W ≠ 0) (x y : NativeVertex n W) :
    NativeAdj n W (nativeMate n W hW hW0 x) y ↔ NativeAdj n W x y := by
  rw [(native_twin_iso n W hn hW hW0).map_adj_iff,
    (native_twin_iso n W hn hW hW0).map_adj_iff]
  simp only [native_twin_iso,nativeMate,decode_encode]
  exact mate_same_neighbors n W _ _

#print axioms native_twin_iso
#print axioms nativeMate_twice
#print axioms nativeMate_ne
#print axioms nativeMate_independent
#print axioms nativeMate_same_neighbors

/-- In native coordinates the mate is exactly (a XOR W, -s). -/
theorem nativeMate_formula (n W : Nat) (hW : W < 2^n) (hW0 : W ≠ 0)
    (x : NativeVertex n W) :
    (nativeMate n W hW hW0 x).index.val = x.index.val ^^^ W ∧
    (nativeMate n W hW hW0 x).spin.val = -x.spin.val := by
  by_cases h : x.index.val < x.index.val ^^^ W
  · simp [nativeMate,mate,decode,encode,h,pickIndex,flip]
  · simp [nativeMate,mate,decode,encode,h,pickIndex,flip,Nat.xor_assoc]

/-- The actual coefficient functions of the normalized two-term element. -/
def nativeValue (W : Nat) (x : NativeVertex n W) : Coeff × Coeff :=
  (termValue (native W x.index.val x.spin.val).lower,
   termValue (native W x.index.val x.spin.val).upper)

theorem nativeValue_injective (W : Nat) (x y : NativeVertex n W)
    (h : nativeValue W x = nativeValue W y) : x = y := by
  have hi : x.index.val = y.index.val := by
    have e := congrArg (fun p : Coeff × Coeff => p.1 x.index.val) h
    by_cases he : x.index.val = y.index.val
    · exact he
    · simp [nativeValue,native,termValue,mono,he] at e
  have hs : x.spin.val = y.spin.val := by
    have e := congrArg (fun p : Coeff × Coeff => p.2 (x.index.val ^^^ W)) h
    simpa [nativeValue,native,termValue,mono,hi] using e
  exact native_ext x y hi hs

theorem nativeValue_nonzero (W : Nat) (x : NativeVertex n W) :
    nativeValue W x ≠ (zeroCoeff,zeroCoeff) := by
  intro h
  have e := congrArg (fun p : Coeff × Coeff => p.1 x.index.val) h
  simp [nativeValue,native,termValue,mono,zeroCoeff] at e

theorem nativeAdj_symmetric (n W : Nat) (x y : NativeVertex n W) :
    NativeAdj n W x y ↔ NativeAdj n W y x := And.comm

theorem nativeAdj_irreflexive (n W : Nat) (hn : 1 ≤ n) (hW : W < 2^n)
    (x : NativeVertex n W) : ¬NativeAdj n W x x := by
  intro h
  exact native_self_nonzero n W x.index.val x.spin.val hn hW
    x.index.property.1 x.index.property.2.1 x.index.property.2.2 x.spin.property h.1

#print axioms nativeMate_formula
#print axioms nativeValue_injective
#print axioms nativeValue_nonzero
#print axioms nativeAdj_symmetric
#print axioms nativeAdj_irreflexive
end Sounio.ZDTwinCover
