/-
  SounioSedenionLadderExtension — the SEDENION EXTENSION of the Furey ladder (Frente B, vector 4/3 B),
  by Lean native_decide. Mathlib-free, no sorry.

  Over the sedenion LEFT-multiplication matrices L_a (16x16, L_a[a^b][b] = cd_sigma(a,b,4)), a single-pair
  ladder op is the complex 16x16 matrix A = alpha(a,b) = (-L_a, L_b) (Re, Im) — Furey's A = 2*alpha, hence
  the factor 4. adjoint(Re,Im) = (Re^T, -Im^T); anticommutator {X,Y} = XY + YX.

    B1 (octonion generation persists): the three octonion ladder ops (1,2),(3,4),(5,6), now 16x16, still
       satisfy {A_i,A_j}=0 and {A_i,A_j†}=4*delta_ij*I_16.                             (b1_persists)
    B2 (maximal fermionic rank 3 -> 4): greedy max mutually-fermionic single-pair ladder ops is 3 for the
       octonion (indices 1..7) and 4 for the sedenion (indices 1..15).       (oct_rank_3 / sed_rank_4)

  The doubling adds EXACTLY ONE fermionic mode — NOT a clean second generation (which needs 6).

  All three native_decide sweeps (the 16x16 complex greedy over indices 1..15) build in ~10s, so this is
  a @[default_target]. Companion to tests/run-pass/sedenion_ladder_extension.sio.
-/
namespace SounioSedenionLadderExtension

def cdSigma (a b : Nat) : Nat → Int
  | 0 => -1
  | 1 => if a == 0 || b == 0 then 1 else -1
  | (n+2) =>
      if a == 0 || b == 0 then 1
      else
        let half := 2 ^ (n+1)
        let aHi := a ≥ half; let bHi := b ≥ half; let aLo := a % half; let bLo := b % half
        if !aHi && !bHi then cdSigma aLo bLo (n+1)
        else if !aHi && bHi then cdSigma bLo aLo (n+1)
        else if aHi && !bHi then (if bLo == 0 then cdSigma aLo 0 (n+1) else - cdSigma aLo bLo (n+1))
        else (if bLo == 0 then - cdSigma 0 aLo (n+1) else cdSigma bLo aLo (n+1))
def sedSigma (a b : Nat) : Int := cdSigma a b 4

-- a 16x16 integer matrix is a flat List Int of length 256, row-major.
abbrev Mat := List Int

-- sign * L_a : L_a[a^j][j] = cd_sigma(a,j,4).
def signedL (a : Nat) (sign : Int) : Mat :=
  (List.range 256).map (fun idx =>
    let i := idx / 16; let j := idx % 16
    if i == (a ^^^ j) then sign * sedSigma a j else 0)

-- complex ladder op alpha(a,b) = (-L_a, L_b) as a pair (Re, Im).
def alphaRe (a : Nat) : Mat := signedL a (-1)
def alphaIm (b : Nat) : Mat := signedL b 1

def transp (A : Mat) : Mat :=
  (List.range 256).map (fun idx => let i := idx / 16; let j := idx % 16; A.getD (j * 16 + i) 0)
def negTransp (A : Mat) : Mat :=
  (List.range 256).map (fun idx => let i := idx / 16; let j := idx % 16; - A.getD (j * 16 + i) 0)

-- 16x16 integer matmul (flat).
def mm (A B : Mat) : Mat :=
  (List.range 256).map (fun idx =>
    let i := idx / 16; let j := idx % 16
    (List.range 16).foldl (fun s k => s + A.getD (i * 16 + k) 0 * B.getD (k * 16 + j) 0) 0)
def madd (A B : Mat) : Mat := (List.range 256).map (fun idx => A.getD idx 0 + B.getD idx 0)
def msub (A B : Mat) : Mat := (List.range 256).map (fun idx => A.getD idx 0 - B.getD idx 0)

-- {X,Y} = XY + YX for complex X=(xr,xi), Y=(yr,yi); returns (Re, Im).
--   re = xr yr - xi yi + yr xr - yi xi ;  im = xr yi + xi yr + yr xi + yi xr
def acRe (xr xi yr yi : Mat) : Mat :=
  msub (madd (mm xr yr) (mm yr xr)) (madd (mm xi yi) (mm yi xi))
def acIm (xr xi yr yi : Mat) : Mat :=
  madd (madd (mm xr yi) (mm xi yr)) (madd (mm yr xi) (mm yi xr))

def isZero (M : Mat) : Bool := M.all (· == 0)
def isScalar (M : Mat) (v : Int) : Bool :=
  (List.range 256).all (fun idx => let i := idx / 16; let j := idx % 16;
    M.getD idx 0 == (if i == j then v else 0))

-- {alpha(a,b), alpha(c,d)} == 0 ?
def pairAnticommZero (a b c d : Nat) : Bool :=
  let xr := alphaRe a; let xi := alphaIm b; let yr := alphaRe c; let yi := alphaIm d
  isZero (acRe xr xi yr yi) && isZero (acIm xr xi yr yi)

-- {alpha(a,b), alpha(c,d)†} : adj(c,d) = (Re^T, -Im^T) with Re=-L_c, Im=L_d.
-- returns 1 if 4*I, 0 if zero, 2 otherwise.
def adjAnticommKind (a b c d : Nat) : Nat :=
  let xr := alphaRe a; let xi := alphaIm b
  let yr := transp (alphaRe c); let yi := negTransp (alphaIm d)
  let re := acRe xr xi yr yi; let im := acIm xr xi yr yi
  if !isZero im then 2
  else if isScalar re 4 then 1
  else if isZero re then 0
  else 2

def selfFermionic (a b : Nat) : Bool :=
  adjAnticommKind a b a b == 1 && pairAnticommZero a b a b
def crossFermionic (a b c d : Nat) : Bool :=
  pairAnticommZero a b c d && adjAnticommKind a b c d == 0

-- greedy over pairs (a<b) in 1..hi; store chosen pairs; count.
def greedyRank (hi : Nat) : Nat := Id.run do
  let mut chosen : List (Nat × Nat) := []
  for a in List.range (hi + 1) do
    if a ≥ 1 then
      for b in List.range (hi + 1) do
        if b > a then
          if selfFermionic a b && chosen.all (fun p => crossFermionic a b p.1 p.2) then
            chosen := chosen ++ [(a, b)]
  return chosen.length

-- B1: the three octonion ladder ops (1,2),(3,4),(5,6) still fermionic as 16x16 matrices.
def b1Persists : Bool := Id.run do
  let pa := [1, 3, 5]; let pb := [2, 4, 6]
  for i in List.range 3 do
    for j in List.range 3 do
      let ai := pa.getD i 0; let bi := pb.getD i 0
      let aj := pa.getD j 0; let bj := pb.getD j 0
      if !pairAnticommZero ai bi aj bj then return false
      let want := if i == j then 1 else 0
      if adjAnticommKind ai bi aj bj != want then return false
  return true

theorem b1_persists : b1Persists = true := by native_decide
theorem oct_rank_3 : greedyRank 7 = 3 := by native_decide
theorem sed_rank_4 : greedyRank 15 = 4 := by native_decide

end SounioSedenionLadderExtension
