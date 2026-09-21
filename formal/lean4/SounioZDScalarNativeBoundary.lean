import SounioZDScalarThirdBoundary

/-! Explicit native fixed-XOR graph form of the third-boundary obstruction.
The algebra/count bridge is inherited from ZDSignedState, not newly assumed.
These are normalized basis-pair graphs in the principal Cayley-Dickson
convention, at fixed parent level. No global scalar completeness is asserted. -/
namespace Sounio.ZDScalarNativeBoundary
open Sounio.ZDSignedState Sounio.ZDScalarInvariant Sounio.ZDScalarCore
open Sounio.ZDScalarBit Sounio.ZDScalarCrossOrigin Sounio.ZDScalarDivisibility
open Sounio.ZDScalarLowCone Sounio.ZDScalarTwoBlock Sounio.ZDScalarThirdBoundary
open Sounio.ZDAlgebraBridge Sounio.ZDTwinCover
set_option maxRecDepth 16384
set_option maxHeartbeats 16000000

def depth (r c k : Nat) : Nat := (((r+1)+c)+k)+1
def resetCode (r c k : Nat) : Code (depth r c k) := .reset
def branchCode (r c k : Nat) : Code (depth r c k) := .t (twoBlock r 1 c k)

def codeEdges {d : Nat} (x : Code d) : Int :=
  nativeEdges d (labelFor x) (labelFor_valid x).1 (labelFor_valid x).2
def codeTriangles {d : Nat} (x : Code d) : Int :=
  nativeTriangles d (labelFor x) (labelFor_valid x).1 (labelFor_valid x).2

/-- Actual native graph counts; the X-channel condition is essential. -/
theorem code_counts {d : Nat} (x : Code d) (hx : isX x=true) :
    codeEdges x=8*(edges x : Int) ∧ codeTriangles x=16*(positives x : Int) := by
  have hc := labelFor_realizes x
  simp only [hx,ite_true] at hc
  have hn := actual_native_counts d (labelFor x) (labelFor_valid x).1 (labelFor_valid x).2
  rw [hc] at hn
  exact hn

def weightedCount {d : Nat} (x : Code d) (a b : Nat) : Int :=
  (a:Int)*codeEdges x+(b:Int)*codeTriangles x

theorem weighted_count_model {d : Nat} (x : Code d) (hx : isX x=true) (a b : Nat) :
    weightedCount x a b=8*(value a (2*b) x : Int) := by
  have h := code_counts x hx
  simp only [weightedCount,h.1,h.2,value,Int.natCast_add,Int.natCast_mul]
  grind only

theorem weighted_count_collision_iff {d : Nat} (x y : Code d)
    (hx : isX x=true) (hy : isX y=true) (a b : Nat) :
    weightedCount x a b=weightedCount y a b ↔ value a (2*b) x=value a (2*b) y := by
  rw [weighted_count_model x hx,weighted_count_model y hy]
  omega

theorem zeroAt_label (r : Nat) : labelFor (zeroAt r)=1 ∧ isX (zeroAt r)=false := by
  induction r with
  | zero => exact ⟨rfl,rfl⟩
  | succ r ih =>
    change (if isX (zeroAt r) then 2^(r+3)+labelFor (zeroAt r) else labelFor (zeroAt r))=1 ∧ false=false
    simp [ih.1,ih.2]

theorem towerC_label {d : Nat} (x : Code d) (c : Nat) (hc : 1≤c) :
    labelFor (towerC c x)=(if isX x then 2^(d+3)+labelFor x else labelFor x) ∧
    isX (towerC c x)=false := by
  cases c with
  | zero => omega
  | succ c =>
    induction c with
    | zero => exact ⟨rfl,rfl⟩
    | succ c ih =>
      have H := ih (by omega)
      constructor
      · change (if isX (towerC (c+1) x) then 2^((d+(c+1))+3)+labelFor (towerC (c+1) x) else labelFor (towerC (c+1) x))=_
        simpa only [H.2,Bool.false_eq_true,ite_false] using H.1
      · rfl

theorem towerT_label {d : Nat} (x : Code d) (k : Nat) (hk : 1≤k) :
    labelFor (towerT k x)=(if isX x then labelFor x else 2^(d+3)+labelFor x) ∧
    isX (towerT k x)=true := by
  cases k with
  | zero => omega
  | succ k =>
    induction k with
    | zero => exact ⟨rfl,rfl⟩
    | succ k ih =>
      have H := ih (by omega)
      constructor
      · change (if isX (towerT (k+1) x) then labelFor (towerT (k+1) x) else 2^((d+(k+1))+3)+labelFor (towerT (k+1) x))=_
        simpa only [H.2,ite_true] using H.1
      · rfl

theorem reset_label (r c k : Nat) :
    labelFor (resetCode r c k)=2^(r+c+k+4) := by
  change 2^((((r+1)+c)+k)+3)=2^(r+c+k+4)
  congr 1 <;> omega

theorem branch_label (r c k : Nat) (hc : 1≤c) (hk : 1≤k) :
    labelFor (branchCode r c k)=1+3*2^(r+3)+2^(r+c+4) := by
  have hz := zeroAt_label r
  have ht := towerT_label (zeroAt r) 1 (by decide)
  simp only [hz.2,Bool.false_eq_true,ite_false,hz.1] at ht
  have hct := towerC_label (towerT 1 (zeroAt r)) c hc
  simp only [ht.1,ht.2,ite_true] at hct
  have hout := towerT_label (towerC c (towerT 1 (zeroAt r))) k hk
  simp only [hct.1,hct.2,Bool.false_eq_true,ite_false] at hout
  change labelFor (.t (twoBlock r 1 c k))=_
  simp only [labelFor,show isX (twoBlock r 1 c k)=true from hout.2,ite_true]
  change labelFor (towerT k (towerC c (towerT 1 (zeroAt r))))=_
  rw [hout.1]
  have ht2 : (2:Nat)^(r+1+3)=2*2^(r+3) := by
    rw [show r+1+3=(r+3)+1 by omega,Nat.pow_succ]; omega
  rw [ht2,show r+1+c+3=r+c+4 by omega]
  omega

theorem explicit_labels_valid (r c k : Nat) (hc : 1≤c) (hk : 1≤k) :
    0<(2:Nat)^(r+c+k+4) ∧ (2:Nat)^(r+c+k+4)<2^(r+c+k+5) ∧
    0<1+3*(2:Nat)^(r+3)+2^(r+c+4) ∧
    1+3*(2:Nat)^(r+3)+2^(r+c+4)<2^(r+c+k+5) := by
  have hr := labelFor_valid (resetCode r c k)
  have hb := labelFor_valid (branchCode r c k)
  rw [reset_label] at hr
  rw [branch_label r c k hc hk] at hb
  have hd : depth r c k+3=r+c+k+5 := by simp only [depth]; omega
  rw [hd] at hr hb
  exact ⟨Nat.pos_of_ne_zero hr.2,hr.1,Nat.pos_of_ne_zero hb.2,hb.1⟩

theorem native_boundary_collision_region (r c k a j : Nat)
    (hc : 1≤c) (hk : 1≤k) (ha : a%2=1) (hab : a<6*2^j)
    (he : weightedCount (resetCode r c k) a (2^(j+1)) =
      weightedCount (branchCode r c k) a (2^(j+1))) :
    j+1=2*k ∧ c<k ∧ 2*c<k ∧ 3*c≤k+2 ∧
      k≤r ∧ r+2*c≤3*k+4 ∧ r≤3*k+1 := by
  have h := (weighted_count_collision_iff (resetCode r c k) (branchCode r c k)
    rfl rfl a (2^(j+1))).mp he
  have hb : 2*2^(j+1)=4*2^j := by rw [Nat.pow_succ]; omega
  rw [hb] at h
  have hj := reset_collision_exponent (twoBlock r 1 c k) a j ha h
  have hhead : headT (twoBlock r 1 c k)=k := by
    simp only [twoBlock,headT_tower,towerC_headT _ c hc,Nat.zero_add]
  rw [hhead] at hj
  exact ⟨hj,one_inner_third_boundary_value_collision_region r c k a j hc hk ha hab h⟩


theorem native_reset_t_counts {d : Nat} (x : Code d) :
    codeEdges (Code.reset (d:=d))-codeEdges (.t x)=32*(gap x : Int) ∧
    codeTriangles (Code.reset (d:=d))=0 ∧
    codeTriangles (.t x)=32*((3*edges x+4*positives x : Nat) : Int) := by
  have hr := code_counts (Code.reset (d:=d)) rfl
  have ht := code_counts (.t x) rfl
  have hg := gap_add_edges x
  have hgi : (gap x : Int)+(edges x : Int)=(capacity d : Int) := by exact_mod_cast hg
  rw [capacity_reset] at hr
  simp only [capacity,edges,positives,Int.natCast_add,Int.natCast_mul] at hr ht
  refine ⟨by omega,by omega,?_⟩
  simp only [Int.natCast_add,Int.natCast_mul]
  omega


private theorem weighted_gap_algebra (ER EB TR TB a b G N : Int)
    (hE : ER-EB=32*G) (hR : TR=0) (hT : TB=32*N) :
    a*ER+b*TR=a*EB+b*TB ↔ a*G=b*N := by
  constructor <;> intro h <;> grind only

/-- No parity or cone hypothesis is needed for this exact native graph equation. -/
theorem weighted_reset_collision_iff {d : Nat} (x : Code d) (a b : Nat) :
    weightedCount (Code.reset (d:=d)) a b=weightedCount (.t x) a b ↔
      a*gap x=b*(3*edges x+4*positives x) := by
  have h := native_reset_t_counts x
  have hi := weighted_gap_algebra _ _ _ _ (a:Int) (b:Int) _ _ h.1 h.2.1 h.2.2
  change weightedCount (Code.reset (d:=d)) a b=weightedCount (.t x) a b ↔
    (a:Int)*(gap x:Int)=(b:Int)*((3*edges x+4*positives x:Nat):Int) at hi
  simpa only [←Int.natCast_mul,Int.ofNat_inj] using hi

theorem native_fixed_weight_reduction (r c k a : Nat)
    (he : weightedCount (resetCode r c k) a (4^k) =
      weightedCount (branchCode r c k) a (4^k)) :
    3*edges (twoBlock r 1 c k)+4*positives (twoBlock r 1 c k)=
      a*gap (towerC c (towerT 1 (zeroAt r))) := by
  have h := (weighted_reset_collision_iff (twoBlock r 1 c k) a (4^k)).mp he
  have hg := tower_gap (towerC c (towerT 1 (zeroAt r))) k
  change gap (twoBlock r 1 c k)=4^k*gap (towerC c (towerT 1 (zeroAt r))) at hg
  rw [hg] at h
  have hid : (4^k)*(a*gap (towerC c (towerT 1 (zeroAt r))))=
      (4^k)*(3*edges (twoBlock r 1 c k)+4*positives (twoBlock r 1 c k)) := by
    simpa only [Nat.mul_left_comm] using h
  exact (Nat.eq_of_mul_eq_mul_left (Nat.pow_pos (by decide)) hid).symm

theorem native_fixed_weight_collision_odd (r c k a : Nat)
    (hk : 1≤k)
    (he : weightedCount (resetCode r c k) a (4^k) =
      weightedCount (branchCode r c k) a (4^k)) : a%2=1 := by
  have h := native_fixed_weight_reduction r c k a he
  have hout := towerT_label (towerC c (towerT 1 (zeroAt r))) k hk
  have hpar := edges_parity (twoBlock r 1 c k)
  have hx : isX (twoBlock r 1 c k)=true := hout.2
  simp only [hx,ite_true] at hpar
  have hm := congrArg (fun n:Nat => n%2) h
  by_cases ha : a%2=0
  · simp [Nat.add_mod,Nat.mul_mod,hpar,ha] at hm
  · omega

/-- The weights a and4^k are expressed directly in native edge/triangle counts. -/
theorem native_fixed_weight_collision_region (r c k a : Nat)
    (hc : 1≤c) (hk : 1≤k) (ha : 0<a) (hab : a<3*4^k)
    (he : weightedCount (resetCode r c k) a (4^k) =
      weightedCount (branchCode r c k) a (4^k)) :
    c<k ∧ 2*c<k ∧ 3*c≤k+2 ∧ k≤r ∧ r+2*c≤3*k+4 ∧ r≤3*k+1 := by
  have h := native_fixed_weight_reduction r c k a he
  have hp : (4:Nat)^k=(2^k)^2 := by
    rw [show (4:Nat)=2*2 by rfl,Nat.mul_pow]
    simp only [Nat.pow_succ,Nat.pow_zero,Nat.one_mul]
  rw [hp] at hab
  exact one_inner_third_boundary_collision_region r c k a hc hk ha hab h

theorem native_fixed_weight_no_collision (r c k a : Nat)
    (hc : 1≤c) (hk : 1≤k) (ha : 0<a) (hab : a<3*4^k)
    (hboundary : k+3=3*c) :
    weightedCount (resetCode r c k) a (4^k) ≠
      weightedCount (branchCode r c k) a (4^k) := by
  intro he
  have h := native_fixed_weight_collision_region r c k a hc hk ha hab he
  omega

theorem native_collision_dimension_bound (r c k a : Nat)
    (hc : 1≤c) (hk : 1≤k) (ha : 0<a) (hab : a<3*4^k)
    (he : weightedCount (resetCode r c k) a (4^k) =
      weightedCount (branchCode r c k) a (4^k)) :
    13≤r+c+k+6 ∧ 7*c+2≤r+c+k+6 := by
  have h := native_fixed_weight_collision_region r c k a hc hk ha hab he
  omega

/-- Both codes name actual X-channel native graphs; they are never isomorphic. -/
theorem native_graphs_nonisomorphic (r c k : Nat) :
    ¬Nonempty (GraphIso
      (NativeVertex (depth r c k+3) (labelFor (resetCode r c k)))
      (NativeVertex (depth r c k+3) (labelFor (branchCode r c k)))
      (NativeAdj (depth r c k+3) (labelFor (resetCode r c k)))
      (NativeAdj (depth r c k+3) (labelFor (branchCode r c k)))) := by
  intro hiso
  have he := (native_iso_iff_counts (depth r c k) _ _
    (labelFor_valid (resetCode r c k)).1 (labelFor_valid (resetCode r c k)).2
    (labelFor_valid (branchCode r c k)).1 (labelFor_valid (branchCode r c k)).2).mp hiso
  have hr := code_counts (resetCode r c k) rfl
  have hb := code_counts (branchCode r c k) rfl
  have he1 := he.1
  have he2 := he.2
  change codeEdges (resetCode r c k)=codeEdges (branchCode r c k) at he1
  change codeTriangles (resetCode r c k)=codeTriangles (branchCode r c k) at he2
  have em : edges (resetCode r c k)=edges (branchCode r c k) := by omega
  have ep : positives (resetCode r c k)=positives (branchCode r c k) := by omega
  have H := invariant_injective _ _ em ep
  cases H

#print axioms code_counts
#print axioms weighted_count_model
#print axioms weighted_count_collision_iff
#print axioms zeroAt_label
#print axioms towerC_label
#print axioms towerT_label
#print axioms reset_label
#print axioms branch_label
#print axioms explicit_labels_valid
#print axioms native_boundary_collision_region
#print axioms native_reset_t_counts
#print axioms weighted_gap_algebra
#print axioms weighted_reset_collision_iff
#print axioms native_fixed_weight_reduction
#print axioms native_fixed_weight_collision_odd
#print axioms native_fixed_weight_collision_region
#print axioms native_fixed_weight_no_collision
#print axioms native_collision_dimension_bound
#print axioms native_graphs_nonisomorphic

end Sounio.ZDScalarNativeBoundary
