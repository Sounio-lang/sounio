# reuse the exact parser, then emit Lean (Int-scaled coords + declared edges)
import importlib.util, math
spec=importlib.util.spec_from_file_location("c","certify510.py")
# can't import (it runs). re-parse inline:
from fractions import Fraction as Fr
exec(open('certify510.py').read().split('verts=[]')[0])  # load F, parse, sqrt_rat, etc.
verts=[]
for line in open('cnp510.vtx'):
    line=line.strip()
    if not line: continue
    inner=line[1:-1]; depth=0; pos=None
    for i,ch in enumerate(inner):
        if ch in '([': depth+=1
        elif ch in ')]': depth-=1
        elif ch==',' and depth==0: pos=i; break
    verts.append((parse(inner[:pos]),parse(inner[pos+1:])))
# python idx (bits 3,5,11) -> lean idx (bits 3,5,7,11 ; 7 unused)
def pyl(m): return (m&1)*1 + ((m>>1)&1)*2 + ((m>>2)&1)*8
# common denominator D
from math import gcd
def lcm(a,b): return a*b//gcd(a,b)
D=1
for vx,vy in verts:
    for comp in (vx,vy):
        for v in comp.c.values(): D=lcm(D, v.denominator)
print("scale D =",D,"  unitSq = D^2 =",D*D)
def emit(comp):
    terms=[]
    for idx,v in sorted(comp.c.items()):
        sv=v*D; assert sv.denominator==1, (v,D)
        terms.append(f"({pyl(idx)},{int(sv)})")
    return "["+", ".join(terms)+"]"
rows=[]
for vx,vy in verts:
    rows.append(f"  (({emit(vx)} : KV), ({emit(vy)} : KV))")
coords_arr="#[\n"+",\n".join(rows)+"\n]"
edges=[]
for line in open('cnp510.edge'):
    if line.startswith('e '):
        _,a,b=line.split(); a=int(a)-1; b=int(b)-1
        edges.append((min(a,b),max(a,b)))
edge_list="[" + ", ".join(f"({a},{b})" for a,b in edges) + "]"
open('/workspace/.cnp-phaseB/formal/lean4/SounioHeule510.lean','w').write(f'''import SounioUnitDistanceField
/-!
# HeuleGraph510 — machine-checked exact unit-distance certification (Phase B)

Auto-generated from Heule's CNP-SAT `vtx/510.vtx` + `edge/510.edge`
(https://github.com/marijnheule/CNP-SAT). The 510 vertex coordinates are parsed
exactly into the number field K = ℚ(√3,√5,√11) (the data's realised field; √7 is
absent — dropped in the Heule trim of de Grey's original). Coordinates are scaled by
the global denominator D = {D} so all field coefficients are integers; a unit edge
then has exact squared length D² = {D*D}.

Reuses the exact XOR-graded field arithmetic of `SounioUnitDistanceField` (degree-16
basis; the degree-8 data populates only the √7-free indices).

* `heule510_soundness` — every one of the 2504 declared edges is at exact unit
  distance ‖·‖² = D² in K.
* `heule510_converse` — exactly 2504 of all C(510,2) vertex pairs are at exact unit
  distance: the declared edge set IS the complete exact unit-distance graph (no
  undeclared unit pair, none missing).

χ ≥ 5 is NOT addressed here (that is the Phase-C UNSAT certificate); this file
certifies the *geometry* — that HeuleGraph510 is a genuine unit-distance graph in
ℚ(√3,√5,√11), the property floating-point computation cannot guarantee.
-/
namespace Sounio.Heule510
open Sounio.UnitDistanceField

def unitSq : Int := {D*D}

/-- 510 vertex coordinates, scaled ×{D}, as exact field (x,y) pairs. -/
def C : Array (KV × KV) := {coords_arr}

/-- The 2504 declared edges (0-indexed) of HeuleGraph510. -/
def E : List (Nat × Nat) := {edge_list}

def cx (i : Nat) : KV × KV := C.getD i ([], [])
def isUnitV (i j : Nat) : Bool := isScalar (d2 (cx i) (cx j)) unitSq

/-- **Soundness.** Every declared edge is at exact unit distance in ℚ(√3,√5,√11). -/
theorem heule510_soundness : E.all (fun e => isUnitV e.1 e.2) := by native_decide

def allPairs : List (Nat × Nat) :=
  (List.range 510).flatMap (fun i => (List.range 510).filterMap
    (fun j => if i < j then some (i,j) else none))

/-- **Converse audit.** Exactly 2504 vertex pairs are at exact unit distance — the
    declared edge set is precisely the exact unit-distance graph. -/
theorem heule510_converse :
    (allPairs.filter (fun p => isUnitV p.1 p.2)).length = 2504 := by native_decide

/-- Vertex/edge census. -/
theorem heule510_census : C.size = 510 ∧ E.length = 2504 := by native_decide

end Sounio.Heule510
''')
print("emitted SounioHeule510.lean  (coords:",len(rows),"edges:",len(edges),")")
