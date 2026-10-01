import SounioZDAlgebraBridge
/-!
Universal entrywise Cayley-Dickson doubling identities.
Bit lists are the existing all-level basis representation, not an abstract
recurrence oracle. This file does not yet assemble graph isomorphisms.
-/
namespace Sounio.ZDRecursion
open SounioCDCocycle Sounio.ZDAlgebraBridge

def kap (a : List Bool) : Int := if isZ a then 1 else -1

/-- The full doubling table, including zero and repeated parent indices. -/
theorem sgn_double (a b : List Bool) (hl : a.length = b.length) (α β : Bool) :
    sgn (α :: a) (β :: b) =
      match α,β with
      | false,false => sgn a b
      | false,true => sgn b a
      | true,false => kap b * sgn a b
      | true,true => -kap b * sgn b a := by
  have hf (x y : List Bool) (hxy : x.length = y.length)
      (hz : isZ x = true ∨ isZ y = true) : sgn x y = 1 :=
    sgn_isZ x y hxy hz
  cases α <;> cases β
  · by_cases ha : isZ a = true
    · rw [hf (false::a) (false::b) (by simpa using hl)
        (Or.inl (by simp [isZ,ha])), hf a b hl (Or.inl ha)]
    · have ha0 : isZ a = false := by simpa using ha
      by_cases hb : isZ b = true
      · rw [hf (false::a) (false::b) (by simpa using hl)
          (Or.inr (by simp [isZ,hb])), hf a b hl (Or.inr hb)]
      · exact sgn_ff a b ha0 (by simpa using hb)
  · by_cases ha : isZ a = true
    · rw [hf (false::a) (true::b) (by simpa using hl)
        (Or.inl (by simp [isZ,ha])), hf b a hl.symm (Or.inr ha)]
    · exact sgn_ft a b (by simpa using ha)
  · by_cases hb : isZ b = true
    · rw [hf (true::a) (false::b) (by simpa using hl)
        (Or.inr (by simp [isZ,hb])), hf a b hl (Or.inr hb)]
      simp [kap,hb]
    · rw [sgn_tf a b (by simpa using hb)]
      simp [kap,hb]
  · rw [sgn_tt a b hl]
    by_cases hb : isZ b = true
    · rw [hf b a hl.symm (Or.inl hb)]
      simp [kap,hb]
    · simp [kap,hb]

theorem sgn_sign_width : ∀ n a b, a.length = n → b.length = n →
    Sign (sgn a b) := by
  intro n
  induction n with
  | zero =>
    intro a b ha hb
    have ea : a = [] := by cases a with
      | nil => rfl
      | cons x xs => simp at ha
    have eb : b = [] := by cases b with
      | nil => rfl
      | cons x xs => simp at hb
    subst a; subst b
    exact Or.inl (by simp [sgn])
  | succ n ih =>
    intro a b ha hb
    cases a with
    | nil => simp at ha
    | cons α a =>
      cases b with
      | nil => simp at hb
      | cons β b =>
        have ha' : a.length = n := by simpa using ha
        have hb' : b.length = n := by simpa using hb
        rw [sgn_double a b (ha'.trans hb'.symm)]
        have hab := ih a b ha' hb'
        have hba := ih b a hb' ha'
        cases α <;> cases β
        · exact hab
        · exact hba
        · by_cases hz : isZ b = true
          · simpa [kap,hz] using hab
          · simpa [kap,hz] using neg_sign hab
        · by_cases hz : isZ b = true
          · simpa [kap,hz] using neg_sign hba
          · simpa [kap,hz] using hba

theorem sgn_sign (a b : List Bool) (hl : a.length = b.length) :
    Sign (sgn a b) := sgn_sign_width a.length a b rfl hl.symm

theorem sgn_sq (a b : List Bool) (hl : a.length = b.length) :
    sgn a b * sgn a b = 1 := by
  rcases sgn_sign a b hl with h | h <;> rw [h] <;> decide

def pairT (a ap b bp : List Bool) : Int := sgn a b * sgn ap bp
def pairQ (a ap b bp : List Bool) : Int :=
  pairT a ap b bp * sgn a bp * sgn ap b

/-- Concrete relation to the canonical Nat-indexed quantities of the previous bridge. -/
theorem pairT_bitsOf (n W a b : Nat) (hn : 1 ≤ n)
    (hW : W < 2^n) (ha : a < 2^n) (hb : b < 2^n) :
    pairT (bitsOf n a) (bitsOf n (a ^^^ W))
      (bitsOf n b) (bitsOf n (b ^^^ W)) = T n W a b := by
  unfold pairT T
  rw [sgn_eq_cdSigma n a b hn ha hb,
    sgn_eq_cdSigma n (a ^^^ W) (b ^^^ W) hn
      (Nat.xor_lt_two_pow ha hW) (Nat.xor_lt_two_pow hb hW)]

theorem pairQ_bitsOf (n W a b : Nat) (hn : 1 ≤ n)
    (hW : W < 2^n) (ha : a < 2^n) (hb : b < 2^n) :
    pairQ (bitsOf n a) (bitsOf n (a ^^^ W))
      (bitsOf n b) (bitsOf n (b ^^^ W)) = Q n W a b := by
  unfold pairQ Q
  rw [pairT_bitsOf n W a b hn hW ha hb,
    sgn_eq_cdSigma n a (b ^^^ W) hn ha (Nat.xor_lt_two_pow hb hW),
    sgn_eq_cdSigma n (a ^^^ W) b hn (Nat.xor_lt_two_pow ha hW) hb]

#print axioms sgn_double
#print axioms sgn_sign
#print axioms pairT_bitsOf
#print axioms pairQ_bitsOf
theorem sgn_generic (a b : List Bool) (hl : a.length = b.length)
    (ha : isZ a = false) (hb : isZ b = false) (hne : a ≠ b) (α β : Bool) :
    sgn (α :: a) (β :: b) = if α || β then -sgn a b else sgn a b := by
  have hab := antisym b a hl.symm hb ha (Ne.symm hne)
  rw [sgn_double a b hl]
  cases α <;> cases β <;> simp [kap, ha, hb, hab]

structure Generic where
  a : List Bool
  ap : List Bool
  b : List Bool
  bp : List Bool
  lap : ap.length = a.length
  lb : b.length = a.length
  lbp : bp.length = a.length
  za : isZ a = false
  zap : isZ ap = false
  zb : isZ b = false
  zbp : isZ bp = false
  ab : a ≠ b
  abp : a ≠ bp
  apb : ap ≠ b
  apbp : ap ≠ bp

def oldT (d : Generic) := pairT d.a d.ap d.b d.bp
def oldQ (d : Generic) := pairQ d.a d.ap d.b d.bp
def pick (α : Bool) (a ap : List Bool) := if α then ap else a

/-- Generic low-label entries: both channels preserve the parent T,Q. -/
theorem low_generic (d : Generic) (α β : Bool) :
    pairT (α::d.a) (α::d.ap) (β::d.b) (β::d.bp) = oldT d ∧
    pairQ (α::d.a) (α::d.ap) (β::d.b) (β::d.bp) = oldQ d := by
  simp only [pairQ,pairT,oldQ,oldT,
    sgn_generic d.a d.b d.lb.symm d.za d.zb d.ab,
    sgn_generic d.ap d.bp (d.lap.trans d.lbp.symm) d.zap d.zbp d.apbp,
    sgn_generic d.a d.bp d.lbp.symm d.za d.zbp d.abp,
    sgn_generic d.ap d.b (d.lap.trans d.lb.symm) d.zap d.zb d.apb]
  cases α <;> cases β <;> simp [Int.neg_mul,Int.mul_neg]

/-- Generic high-label entries. Partner indices acquire the new top bit. -/
theorem high_generic (d : Generic) (α β : Bool) :
    pairT (false::pick α d.a d.ap) (true::pick (!α) d.a d.ap)
      (false::pick β d.b d.bp) (true::pick (!β) d.b d.bp) =
        -oldT d * (if xor α β then oldQ d else 1) ∧
    pairQ (false::pick α d.a d.ap) (true::pick (!α) d.a d.ap)
      (false::pick β d.b d.bp) (true::pick (!β) d.b d.bp) = -oldQ d := by
  have hBA := antisym d.b d.a d.lb d.zb d.za (Ne.symm d.ab)
  have hBB := antisym d.bp d.ap (d.lbp.trans d.lap.symm) d.zbp d.zap (Ne.symm d.apbp)
  have hBC := antisym d.bp d.a d.lbp d.zbp d.za (Ne.symm d.abp)
  have hBD := antisym d.b d.ap (d.lb.trans d.lap.symm) d.zb d.zap (Ne.symm d.apb)
  have hA := sgn_sign d.a d.b d.lb.symm
  have hB := sgn_sign d.ap d.bp (d.lap.trans d.lbp.symm)
  have hC := sgn_sign d.a d.bp d.lbp.symm
  have hD := sgn_sign d.ap d.b (d.lap.trans d.lb.symm)
  cases α <;> cases β <;>
    simp only [pick,Bool.not_true,Bool.not_false,Bool.false_eq_true,↓reduceIte,Bool.xor_self,
      Bool.false_xor,Bool.true_xor,pairQ,pairT,oldQ,oldT]
  all_goals
    simp only [sgn_double,d.lap,d.lb,d.lbp,
      kap,d.za,d.zap,d.zb,d.zbp,↓reduceIte,hBA,hBB,hBC,hBD]
    rcases hA with hA | hA <;> rcases hB with hB | hB <;>
    rcases hC with hC | hC <;> rcases hD with hD | hD <;>
      simp [pairT,pairQ,hA,hB,hC,hD,oldT,oldQ]

/-- The off-diagonal connection between the two copies of one representative. -/
theorem low_pair (a ap : List Bool) (hl : a.length = ap.length)
    (ha : isZ a = false) (hap : isZ ap = false) (hne : a ≠ ap) :
    pairT (false::a) (false::ap) (true::a) (true::ap) = 1 ∧
    pairQ (false::a) (false::ap) (true::a) (true::ap) = -1 := by
  have hd := diag a ha
  have hdp := diag ap hap
  have hr := antisym ap a hl.symm hap ha (Ne.symm hne)
  have hsign := sgn_sign a ap hl
  simp only [pairT,pairQ,sgn_double a a rfl,sgn_double a ap hl,
    sgn_double ap a hl.symm,sgn_double ap ap rfl,kap,ha,hap,↓reduceIte,hd,hdp,hr]
  rcases hsign with h | h <;> simp [h]

theorem high_pair (a ap : List Bool) (hl : a.length = ap.length)
    (ha : isZ a = false) (hap : isZ ap = false) :
    pairT (false::a) (true::ap) (false::ap) (true::a) = 1 ∧
    pairQ (false::a) (true::ap) (false::ap) (true::a) = -1 := by
  have hd := diag a ha
  have hdp := diag ap hap
  have hsign := sgn_sign a ap hl
  simp only [pairT,pairQ,sgn_double a a rfl,sgn_double a ap hl,
    sgn_double ap a hl.symm,sgn_double ap ap rfl,kap,ha,hap,↓reduceIte,hd,hdp]
  rcases hsign with h | h <;> simp [h]

theorem reset_entry (a b : List Bool) (hl : a.length = b.length)
    (ha : isZ a = false) (hb : isZ b = false) (hne : a ≠ b) :
    pairT (false::a) (true::a) (false::b) (true::b) = -1 ∧
    pairQ (false::a) (true::a) (false::b) (true::b) = -1 := by
  have hr := antisym b a hl.symm hb ha (Ne.symm hne)
  have hsign := sgn_sign a b hl
  simp only [pairT,pairQ,sgn_double a b hl,kap,ha,hb,↓reduceIte,hr]
  rcases hsign with h | h <;> simp [h]

#print axioms low_generic
#print axioms high_generic
#print axioms low_pair
#print axioms high_pair
#print axioms reset_entry

structure Hub where
  z : List Bool
  v : List Bool
  r : List Bool
  u : List Bool
  lz : z.length = v.length
  lr : r.length = v.length
  lu : u.length = v.length
  zz : isZ z = true
  zv : isZ v = false
  zr : isZ r = false
  zu : isZ u = false
  vr : v ≠ r
  vu : v ≠ u
  xu : u = xorL v r

theorem hub_relation (d : Hub) :
    sgn d.v d.r * sgn d.v d.u = -1 := by
  rw [d.xu]
  exact Lsq d.v d.r d.lr.symm d.zv

theorem hub_flip (d : Hub) : sgn d.v d.u = -sgn d.v d.r := by
  have h := hub_relation d
  rcases sgn_sign d.v d.r d.lr.symm with hs | hs
  · rw [hs] at h ⊢
    simp at h
    exact h
  · rw [hs] at h ⊢
    simp at h ⊢
    omega

/-- Hub entries in the low branch, with both copy signs and Q=-1. -/
theorem low_hub (d : Hub) (α : Bool) :
    pairT (true::d.z) (true::d.v) (α::d.r) (α::d.u) =
      (if α then -sgn d.v d.u else sgn d.v d.u) ∧
    pairQ (true::d.z) (true::d.v) (α::d.r) (α::d.u) = -1 := by
  have hzr := sgn_isZ d.z d.r (d.lz.trans d.lr.symm) (Or.inl d.zz)
  have hrz := sgn_isZ d.r d.z (d.lr.trans d.lz.symm) (Or.inr d.zz)
  have hzu := sgn_isZ d.z d.u (d.lz.trans d.lu.symm) (Or.inl d.zz)
  have huz := sgn_isZ d.u d.z (d.lu.trans d.lz.symm) (Or.inr d.zz)
  have hrv := antisym d.r d.v d.lr d.zr d.zv (Ne.symm d.vr)
  have huv := antisym d.u d.v d.lu d.zu d.zv (Ne.symm d.vu)
  have hrel := hub_relation d
  cases α <;>
    simp only [pairT,pairQ,sgn_double d.z d.r (d.lz.trans d.lr.symm),
      sgn_double d.v d.u d.lu.symm,sgn_double d.z d.u (d.lz.trans d.lu.symm),
      sgn_double d.v d.r d.lr.symm,kap,d.zr,d.zu,↓reduceIte,
      hzr,hrz,hzu,huz,hrv,huv]
  all_goals
    rcases sgn_sign d.v d.r d.lr.symm with hA | hA <;>
    rcases sgn_sign d.v d.u d.lu.symm with hB | hB <;>
      simp [hA,hB] at hrel ⊢

/-- Hub entries in the high branch; the second copy has the opposite sign. -/
theorem high_hub (d : Hub) (α : Bool) :
    pairT (false::d.v) (true::d.z)
      (false::pick α d.r d.u) (true::pick (!α) d.r d.u) =
      (if α then -sgn d.v d.r else sgn d.v d.r) ∧
    pairQ (false::d.v) (true::d.z)
      (false::pick α d.r d.u) (true::pick (!α) d.r d.u) = -1 := by
  have hzr := sgn_isZ d.z d.r (d.lz.trans d.lr.symm) (Or.inl d.zz)
  have hrz := sgn_isZ d.r d.z (d.lr.trans d.lz.symm) (Or.inr d.zz)
  have hzu := sgn_isZ d.z d.u (d.lz.trans d.lu.symm) (Or.inl d.zz)
  have huz := sgn_isZ d.u d.z (d.lu.trans d.lz.symm) (Or.inr d.zz)
  have hrv := antisym d.r d.v d.lr d.zr d.zv (Ne.symm d.vr)
  have huv := antisym d.u d.v d.lu d.zu d.zv (Ne.symm d.vu)
  have hrel := hub_relation d
  cases α <;>
    simp only [pick,Bool.not_true,Bool.not_false,Bool.false_eq_true,↓reduceIte,pairT,pairQ,
      sgn_double,d.lz,d.lr,d.lu,kap,d.zr,d.zu,↓reduceIte,
      hzr,hrz,hzu,huz,hrv,huv]
  all_goals
    rcases sgn_sign d.v d.r d.lr.symm with hA | hA <;>
    rcases sgn_sign d.v d.u d.lu.symm with hB | hB <;>
      simp [hA,hB] at hrel ⊢

theorem bitsOf_lower (n a : Nat) (ha : a < 2^n) :
    bitsOf (n+1) a = false :: bitsOf n a := by
  simp [bitsOf, Nat.not_le_of_gt ha, Nat.mod_eq_of_lt ha]

theorem bitsOf_upper (n a : Nat) (ha : a < 2^n) :
    bitsOf (n+1) (2^n+a) = true :: bitsOf n a := by
  have hge : 2^n+a ≥ 2^n := by omega
  simp [bitsOf,hge,Nat.add_mod,Nat.mod_eq_of_lt ha]

#print axioms low_hub
#print axioms high_hub
#print axioms bitsOf_lower
#print axioms bitsOf_upper

def M (t q : Int) : Int := if q = -1 then t else 0
def N (t q : Int) : Int := if q = 1 then t else 0

theorem mul_sign {a b : Int} (ha : Sign a) (hb : Sign b) : Sign (a*b) := by
  rcases ha with rfl | rfl <;> rcases hb with rfl | rfl <;> simp [Sign]

theorem oldQ_sign (d : Generic) : Sign (oldQ d) :=
  mul_sign (mul_sign
    (mul_sign (sgn_sign d.a d.b d.lb.symm)
      (sgn_sign d.ap d.bp (d.lap.trans d.lbp.symm)))
    (sgn_sign d.a d.bp d.lbp.symm))
    (sgn_sign d.ap d.b (d.lap.trans d.lb.symm))

/-- Q reversal interchanges the two support channels, with the required signs. -/
theorem high_channel_law (t q : Int) (hq : Sign q) (α β : Bool) :
    M (-t * (if xor α β then q else 1)) (-q) = -N t q ∧
    N (-t * (if xor α β then q else 1)) (-q) =
      (if xor α β then M t q else -M t q) := by
  rcases hq with rfl | rfl <;> cases α <;> cases β <;> simp [M,N]

theorem high_channels (d : Generic) (α β : Bool) :
    let t := pairT (false::pick α d.a d.ap) (true::pick (!α) d.a d.ap)
      (false::pick β d.b d.bp) (true::pick (!β) d.b d.bp)
    let q := pairQ (false::pick α d.a d.ap) (true::pick (!α) d.a d.ap)
      (false::pick β d.b d.bp) (true::pick (!β) d.b d.bp)
    M t q = -N (oldT d) (oldQ d) ∧
      N t q = (if xor α β then M (oldT d) (oldQ d) else -M (oldT d) (oldQ d)) := by
  dsimp only
  rw [(high_generic d α β).1,(high_generic d α β).2]
  exact high_channel_law (oldT d) (oldQ d) (oldQ_sign d) α β

theorem low_channels (d : Generic) (α β : Bool) :
    let t := pairT (α::d.a) (α::d.ap) (β::d.b) (β::d.bp)
    let q := pairQ (α::d.a) (α::d.ap) (β::d.b) (β::d.bp)
    M t q = M (oldT d) (oldQ d) ∧ N t q = N (oldT d) (oldQ d) := by
  dsimp only
  rw [(low_generic d α β).1,(low_generic d α β).2]
  exact ⟨rfl,rfl⟩

#print axioms high_channels
#print axioms low_channels

/-- The bit-list quantities are the same ones deciding actual native products. -/
theorem native_adjacency_bit_iff (n W a b : Nat) (s t : Int)
    (hn : 1 ≤ n) (hW : W < 2^n)
    (ha : a < 2^n) (hb : b < 2^n)
    (ha0 : a ≠ 0) (hb0 : b ≠ 0)
    (haW : a ≠ W) (hbW : b ≠ W)
    (hs : Sign s) (ht : Sign t) :
    (nativeProduct n W a b s t = (zeroCoeff,zeroCoeff) ∧
     nativeProduct n W b a t s = (zeroCoeff,zeroCoeff)) ↔
      a ≠ b ∧ a ≠ b ^^^ W ∧
        pairQ (bitsOf n a) (bitsOf n (a ^^^ W))
          (bitsOf n b) (bitsOf n (b ^^^ W)) = -1 ∧
        s*t = pairT (bitsOf n a) (bitsOf n (a ^^^ W))
          (bitsOf n b) (bitsOf n (b ^^^ W)) := by
  rw [pairQ_bitsOf n W a b hn hW ha hb,pairT_bitsOf n W a b hn hW ha hb]
  exact native_adjacency_iff n W a b s t hn hW ha hb ha0 hb0 haW hbW hs ht

#print axioms native_adjacency_bit_iff

/-- Equal bounded basis bit lists denote the same index. -/
theorem bitsOf_injective (n a b : Nat) (ha : a < 2^n) (hb : b < 2^n)
    (h : bitsOf n a = bitsOf n b) : a = b := by
  have hz := (xorL_isZ_iff (bitsOf n a) (bitsOf n b) (by simp [bitsOf_length])).mpr h
  rw [xorL_bitsOf n a b ha hb] at hz
  have hx := (isZ_bitsOf n (a ^^^ b) (Nat.xor_lt_two_pow ha hb)).mp hz
  exact (xor_zero_iff a b).mp hx

/-- Construct the generic table's hypotheses from actual admissible XOR partners. -/
def genericOfNat (n W a b : Nat)
    (hW : W < 2^n) (ha : a < 2^n) (hb : b < 2^n)
    (ha0 : a ≠ 0) (hb0 : b ≠ 0) (haW : a ≠ W) (hbW : b ≠ W)
    (hab : a ≠ b) (habW : a ≠ b ^^^ W) : Generic where
  a := bitsOf n a
  ap := bitsOf n (a ^^^ W)
  b := bitsOf n b
  bp := bitsOf n (b ^^^ W)
  lap := by simp [bitsOf_length]
  lb := by simp [bitsOf_length]
  lbp := by simp [bitsOf_length]
  za := isZ_bitsOf_false n a ha ha0
  zap := isZ_bitsOf_false n (a ^^^ W) (Nat.xor_lt_two_pow ha hW)
    (fun h => haW ((xor_zero_iff a W).mp h))
  zb := isZ_bitsOf_false n b hb hb0
  zbp := isZ_bitsOf_false n (b ^^^ W) (Nat.xor_lt_two_pow hb hW)
    (fun h => hbW ((xor_zero_iff b W).mp h))
  ab := fun h => hab (bitsOf_injective n a b ha hb h)
  abp := fun h => habW (bitsOf_injective n a (b ^^^ W) ha (Nat.xor_lt_two_pow hb hW) h)
  apb := fun h => (xor_ne_reverse a b W habW)
    (bitsOf_injective n (a ^^^ W) b (Nat.xor_lt_two_pow ha hW) hb h).symm
  apbp := by
    intro h
    have e := bitsOf_injective n (a ^^^ W) (b ^^^ W)
      (Nat.xor_lt_two_pow ha hW) (Nat.xor_lt_two_pow hb hW) h
    have e' := congrArg (fun x : Nat => x ^^^ W) e
    exact hab (by simpa [Nat.xor_assoc] using e')

/-- Construct hub data from actual parent indices, without assumed sign relations. -/
def hubOfNat (n v r : Nat) (hv : v < 2^n) (hr : r < 2^n)
    (hv0 : v ≠ 0) (hr0 : r ≠ 0) (hne : v ≠ r) : Hub where
  z := bitsOf n 0
  v := bitsOf n v
  r := bitsOf n r
  u := bitsOf n (v ^^^ r)
  lz := by simp [bitsOf_length]
  lr := by simp [bitsOf_length]
  lu := by simp [bitsOf_length]
  zz := (isZ_bitsOf n 0 (Nat.two_pow_pos n)).mpr rfl
  zv := isZ_bitsOf_false n v hv hv0
  zr := isZ_bitsOf_false n r hr hr0
  zu := isZ_bitsOf_false n (v ^^^ r) (Nat.xor_lt_two_pow hv hr)
    (fun h => hne ((xor_zero_iff v r).mp h))
  vr := fun h => hne (bitsOf_injective n v r hv hr h)
  vu := by
    intro h
    have e := bitsOf_injective n v (v ^^^ r) hv (Nat.xor_lt_two_pow hv hr) h
    have e' := congrArg (fun x : Nat => v ^^^ x) e
    exact hr0 (by simpa [← Nat.xor_assoc] using e'.symm)
  xu := (xorL_bitsOf n v r hv hr).symm

#print axioms genericOfNat
#print axioms hubOfNat

end Sounio.ZDRecursion
