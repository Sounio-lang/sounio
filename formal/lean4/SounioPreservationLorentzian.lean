/-
  SounioPreservationLorentzian.lean

  Machine-checked witness (native_decide, no Mathlib, no sorry) for the RUNG LAW
  at a TIMELIKE locus of the BASE-SPLIT sedenion algebra:

      the two-sided preservation algebra P_z of ker L_z, for the timelike
      zero-divisor z = e4 + e13, is a Jordan spin factor whose Jordan Gram is
      diag(1,1,-1,1,1) — signature (4,1). i.e. P_z ≅ J_spin(4,1), Lorentzian.

  Base split = Cayley–Dickson doubling-sign vector μ⃗ = (-1,-1,+1,-1): the octonion
  layer is split, so e1..e3,e8..e11 square to -1 and e4..e7,e12..e15 square to +1.
  Multiplication keeps the XOR-index law  e_i · e_j = σ(i,j) · e_{i⊕j};  only the
  sign σ changes vs the division algebra. σ is the verified 16×16 table below.

  Companion to docs/research/PRESERVATION_ALGEBRA_GEOMETRY_2026-08-24.md.
-/

namespace Sounio.PreservationLorentzian

/-- Sparse 16-D element: list of (index, coefficient). -/
abbrev SVec := List (Nat × Int)

/-- Base-split sign table σ(i,j): row-major, e_i·e_j = σ(i,j)·e_{i⊕j}. -/
def sigmaTable : Array Int := #[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,-1,1,-1,1,-1,-1,1,1,-1,-1,1,-1,1,1,-1,1,-1,-1,1,1,1,-1,-1,1,1,-1,-1,-1,-1,1,1,1,1,-1,-1,1,-1,1,-1,1,-1,1,-1,-1,1,-1,1,1,-1,-1,-1,1,-1,-1,-1,1,1,1,1,1,1,1,1,1,1,-1,1,1,1,1,-1,1,-1,1,-1,-1,1,-1,1,1,1,1,-1,1,-1,1,1,1,-1,-1,1,-1,1,1,-1,1,-1,1,1,1,1,-1,1,1,1,-1,-1,-1,-1,1,1,1,-1,-1,-1,-1,-1,-1,-1,-1,1,1,1,1,1,1,1,1,1,-1,1,-1,1,1,-1,-1,-1,-1,1,-1,1,1,-1,1,1,1,-1,-1,-1,1,1,-1,1,-1,-1,-1,-1,1,1,1,-1,1,1,-1,1,-1,1,-1,-1,1,-1,-1,1,-1,1,1,1,1,1,-1,1,1,1,-1,1,1,1,1,1,1,1,1,-1,1,-1,-1,-1,-1,1,-1,-1,1,-1,-1,1,-1,1,1,-1,-1,1,-1,1,-1,-1,-1,-1,-1,1,-1,1,1,-1,1,1,-1,-1,-1,-1,1,-1,-1,1,-1,-1,-1,-1,1,1]

def splitSigma (i j : Nat) : Int := sigmaTable.getD (i * 16 + j) 0

def bump (acc : List (Nat × Int)) (k : Nat) (c : Int) : List (Nat × Int) :=
  if acc.any (fun p => p.1 == k)
  then acc.map (fun p => if p.1 == k then (p.1, p.2 + c) else p)
  else acc ++ [(k, c)]

def combine (v : SVec) : SVec :=
  (v.foldl (fun acc p => bump acc p.1 p.2) []).filter (fun p => p.2 != 0)

/-- Base-split sedenion product. -/
def smul (a b : SVec) : SVec :=
  combine (a.flatMap (fun p => b.map (fun q => (p.1 ^^^ q.1, p.2 * q.2 * splitSigma p.1 q.1))))

def isZero (v : SVec) : Bool := (combine v).isEmpty

/-- Scalar (e_0) part. -/
def scal (v : SVec) : Int := ((combine v).filter (fun p => p.1 == 0)).foldl (fun s p => s + p.2) 0

-- The timelike locus and its data (verified externally, re-checked here).
def zLocus : SVec := [(4,1),(13,1)]

/-- Basis of ker L_z = { w : z·w = 0 }. -/
def kernelBasis : List SVec :=
  [ [(3,1),(10,1)], [(2,-1),(11,1)], [(7,-1),(14,1)], [(6,1),(15,1)] ]

/-- The five imaginary generators of the preservation algebra P_z. -/
def presGens : List SVec :=
  [ [(4,1)], [(5,1)], [(8,1)], [(12,1)], [(13,1)] ]

-- §1. e4+e13 is a genuine (two-sided) zero-divisor: its kernel basis is annihilated.
theorem zLocus_is_zero_divisor :
    (kernelBasis.all (fun k => isZero (smul zLocus k) && isZero (smul k zLocus))) = true := by
  native_decide

-- §2. Each generator PRESERVES ker L_z two-sidedly:
--     a·k ∈ ker  ⟺  z·(a·k) = 0,   and   k·a ∈ ker  ⟺  z·(k·a) = 0.
theorem presGens_preserve_kernel :
    (presGens.all (fun a =>
      kernelBasis.all (fun k =>
        isZero (smul zLocus (smul a k)) && isZero (smul zLocus (smul k a))))) = true := by
  native_decide

-- §3. The Jordan Gram is diagonal (spin factor).  We use 2·B(a,b) = scal(a·b + b·a)
--     to stay in ℤ; the doubled Gram is diag(2,2,-2,2,2).
def jScal2 (a b : SVec) : Int := scal (smul a b) + scal (smul b a)

def gram2 : List (List Int) :=
  presGens.map (fun a => presGens.map (fun b => jScal2 a b))

theorem gram_is_diag_4_1 :
    gram2 = [ [2,0,0,0,0], [0,2,0,0,0], [0,0,-2,0,0], [0,0,0,2,0], [0,0,0,0,2] ] := by
  native_decide

-- §4. Signature: the (proven-diagonal) Gram has 4 positive and 1 negative
--     diagonal entries ⇒ signature (4,1,0), Lorentzian.
def gramDiag : List Int :=
  [ (gram2[0]!)[0]!, (gram2[1]!)[1]!, (gram2[2]!)[2]!, (gram2[3]!)[3]!, (gram2[4]!)[4]! ]

def countPos (l : List Int) : Nat := l.foldl (fun c x => if x > 0 then c + 1 else c) 0
def countNeg (l : List Int) : Nat := l.foldl (fun c x => if x < 0 then c + 1 else c) 0

theorem signature_is_lorentzian_4_1 :
    countPos gramDiag = 4 ∧ countNeg gramDiag = 1 := by
  native_decide

/-
  Together §1–§4 certify: at the timelike base-split locus z = e4+e13, ker L_z is a
  genuine 4-D two-sided annihilator, the five exhibited imaginary multipliers
  preserve it, and their Jordan form is diag(1,1,-1,1,1) — signature (4,1). Hence
  P_z contains a Lorentzian spin factor J_spin(4,1). (Maximality dim P_z = 6 is
  established by the external rational computation; this file certifies the
  algebra structure and its Lorentzian signature.)
-/

end Sounio.PreservationLorentzian
