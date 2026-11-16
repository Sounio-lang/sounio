/-
The DAG semantics of Sporos (decision 21): the same machine as Sporos.Step, with
every provenance a node of an append-only arena instead of a list. A primitive
computes its words exactly as in the list semantics and describes each
provenance it produces by a plan (`PE`) over existing nodes; an allocator
(`Alloc`) then builds the plan into the arena. The machine is parametric in the
allocator: Sporos.DagRun proves the two semantics agree for every allocator
that meets `Alloc.spec`.
-/
import Sporos.Step
import Sporos.Dag

namespace Sporos

/-- A provenance to build: an existing node, one origin, or a union. -/
inductive PE where
  | nil
  | node (n : Nat)
  | leaf (o : Origin)
  | app (a b : PE)
deriving Repr

/-- Allocate a plan. Node 0 of every arena is `empty`. -/
def build : PE → Arena → Nat × Arena
  | .nil, ar => (0, ar)
  | .node n, ar => (n, ar)
  | .leaf o, ar => (ar.length, ar ++ [.leaf o])
  | .app a b, ar =>
    let (x, ar1) := build a ar
    let (y, ar2) := build b ar1
    (ar2.length, ar2 ++ [.join x y])

/-- What a plan stands for, given what each node stands for. -/
def PE.den (σ : Nat → List Origin) : PE → List Origin
  | .nil => []
  | .node n => σ n
  | .leaf o => [o]
  | .app a b => a.den σ ++ b.den σ

/-- `ar'` extends `ar`. -/
def Ext (ar ar' : Arena) : Prop := ∃ x, ar' = ar ++ x

theorem Ext.refl (ar : Arena) : Ext ar ar := ⟨[], by simp⟩

theorem Ext.trans {a b c : Arena} (h1 : Ext a b) (h2 : Ext b c) : Ext a c := by
  obtain ⟨x, rfl⟩ := h1; obtain ⟨y, rfl⟩ := h2; exact ⟨x ++ y, by simp⟩

theorem Ext.len {a b : Arena} (h : Ext a b) : a.length ≤ b.length := by
  obtain ⟨x, rfl⟩ := h; simp

/-- A good arena: joins point backwards, node 0 is `empty`. -/
def Good (ar : Arena) : Prop := AWF ar ∧ ar[0]? = some .empty

theorem Good.leaves_ext {a b : Arena} (ha : Good a) (h : Ext a b) {i : Nat} (hi : i < a.length) :
    leaves b i = leaves a i := by
  obtain ⟨x, rfl⟩ := h; exact leaves_append ha.1 x hi

/-- Every node a plan names exists. -/
def PE.ok (n : Nat) : PE → Prop
  | .nil => True
  | .node i => i < n
  | .leaf _ => True
  | .app a b => a.ok n ∧ b.ok n

theorem PE.ok_mono {e : PE} {n m : Nat} (h : e.ok n) (hm : n ≤ m) : e.ok m := by
  induction e with
  | nil => trivial
  | node i => exact Nat.lt_of_lt_of_le h hm
  | leaf _ => trivial
  | app a b iha ihb => exact ⟨iha h.1, ihb h.2⟩

theorem PE.den_ext {e : PE} {a b : Arena} (ha : Good a) (h : Ext a b) (he : e.ok a.length) :
    e.den (leaves b) = e.den (leaves a) := by
  induction e with
  | nil => rfl
  | node i => exact ha.leaves_ext h he
  | leaf _ => rfl
  | app x y ihx ihy => simp only [PE.den]; rw [ihx he.1, ihy he.2]


/-- An allocator: builds a plan into an arena. Any allocator with this
specification will do; `Alloc.plain` appends one node per union, the hash-consing
one in `Sporos.DagHash` reuses existing nodes. -/
structure Alloc where
  build : PE → Arena → Nat × Arena
  spec : ∀ (e : PE) (ar : Arena), Good ar → e.ok ar.length →
    Good (build e ar).2 ∧ Ext ar (build e ar).2 ∧ (build e ar).1 < (build e ar).2.length ∧
      leaves (build e ar).2 (build e ar).1 = e.den (leaves ar)

structure DVal where
  w : Nat
  n : Nat
deriving DecidableEq, Repr

structure DFrame where
  word : Nat
  pc : Nat
  locs : List DVal
  base : Nat
  abase : Nat

structure DSt where
  ds    : List DVal
  aux   : List DVal
  rs    : List DFrame
  mem   : Nat → DVal
  msize : Nat
  hp    : Nat
  tabs  : List (List (Nat × DVal))
  inp   : List Nat
  ipos  : Nat
  out   : List (Nat × Nat)
  pc    : Nat
  step  : Nat
  ar    : Arena

/-- The provenance of the operands, the context label and the instruction. -/
def qPlan (args : List DVal) (pc : Nat) (o : Origin) : PE :=
  .app (.app (args.foldr (fun v acc => .app (.node v.n) acc) .nil) (.node pc)) (.leaf o)

def tagPlan (pc : Nat) (o : Origin) (v : DVal) : PE := .app (.app (.node v.n) (.node pc)) (.leaf o)

/-- Read `n` bytes: the word, and the plan of the bytes' provenances. -/
def DSt.ld (s : DSt) (a n : Nat) : Option (Nat × PE) :=
  if a + n ≤ s.msize then
    some ((List.range n).foldr (fun i acc => acc * 256 + (s.mem (a + i)).w % 256) 0,
          (List.range n).foldr (fun i acc => .app (.node (s.mem (a + i)).n) acc) .nil)
  else none

def DSt.st (s : DSt) (a n v : Nat) (q : Nat) : Option DSt :=
  if a + n ≤ s.msize then
    some { s with mem := (fun x => if a ≤ x ∧ x < a + n then ⟨v / 256 ^ (x - a) % 256, q⟩ else s.mem x) }
  else none

/-- A primitive's result before allocation: each pushed value as a word and a plan. -/
inductive DPR where
  | ok (res : List (Nat × PE)) (s : DSt)
  | halt (code : Nat)
  | fail (e : Err)

section
variable (A : Alloc)

/-- Push the plans of `res` in order, allocating as needed. -/
def buildAll : List (Nat × PE) → Arena → List DVal × Arena
  | [], ar => ([], ar)
  | (w, e) :: r, ar =>
    let (n, ar1) := A.build e ar
    let (vs, ar2) := buildAll r ar1
    (⟨w, n⟩ :: vs, ar2)

def tfind' : List (Nat × DVal) → Nat → Option DVal
  | [], _ => none
  | (k, v) :: r, x => if k = x then some v else tfind' r x

/-- The primitives. The node `qn` is the built `qPlan`, which stores and table
writes record. -/
def execPrimD (ph : Phys) (p : Prim) (args : List DVal) (s : DSt) (o : Origin) (qn : Nat) : DPR :=
  let q := qPlan args s.pc o
  let mk (w : Nat) : Nat × PE := (wrap w, q)
  let t (v : DVal) : Nat × PE := (v.w, tagPlan s.pc o v)
  match p, args with
  | .dup, [a] => .ok [t a, t a] s
  | .drop, [_] => .ok [] s
  | .swap, [b, a] => .ok [t a, t b] s
  | .over, [b, a] => .ok [t a, t b, t a] s
  | .rot, [c, b, a] => .ok [t a, t c, t b] s
  | .nip, [b, _] => .ok [t b] s
  | .tuck, [b, a] => .ok [t b, t a, t b] s
  | .add, [b, a] => .ok [mk (a.w + b.w)] s
  | .sub, [b, a] => .ok [mk (a.w + M - b.w % M)] s
  | .mul, [b, a] => .ok [mk (a.w * b.w)] s
  | .divu, [b, a] => if b.w % M = 0 then .fail .divZero else .ok [mk (a.w / b.w)] s
  | .modu, [b, a] => if b.w % M = 0 then .fail .divZero else .ok [mk (a.w % b.w)] s
  | .divs, [b, a] =>
    if b.w % M = 0 then .fail .divZero else .ok [mk (ofInt (Int.tdiv (toSigned a.w) (toSigned b.w)))] s
  | .mods, [b, a] =>
    if b.w % M = 0 then .fail .divZero else .ok [mk (ofInt (Int.tmod (toSigned a.w) (toSigned b.w)))] s
  | .neg, [a] => .ok [mk (M - a.w % M)] s
  | .band, [b, a] => .ok [mk (Nat.land a.w b.w)] s
  | .bor, [b, a] => .ok [mk (Nat.lor a.w b.w)] s
  | .bxor, [b, a] => .ok [mk (Nat.xor a.w b.w)] s
  | .bnot, [a] => .ok [mk (M - 1 - a.w % M)] s
  | .shl, [b, a] => .ok [mk (a.w * 2 ^ (b.w % 64))] s
  | .shr, [b, a] => .ok [mk (a.w / 2 ^ (b.w % 64))] s
  | .sar, [b, a] => .ok [mk (ofInt (toSigned a.w / (2 ^ (b.w % 64) : Nat)))] s
  | .eq, [b, a] => .ok [mk (bool (a.w = b.w))] s
  | .ne, [b, a] => .ok [mk (bool (a.w ≠ b.w))] s
  | .ltu, [b, a] => .ok [mk (bool (a.w < b.w))] s
  | .lts, [b, a] => .ok [mk (bool (toSigned a.w < toSigned b.w))] s
  | .leu, [b, a] => .ok [mk (bool (a.w ≤ b.w))] s
  | .les, [b, a] => .ok [mk (bool (toSigned a.w ≤ toSigned b.w))] s
  | .ld8, [a] => match s.ld a.w 1 with | some (v, pm) => .ok [(v, .app q pm)] s | none => .fail .bounds
  | .ld16, [a] => match s.ld a.w 2 with | some (v, pm) => .ok [(v, .app q pm)] s | none => .fail .bounds
  | .ld32, [a] => match s.ld a.w 4 with | some (v, pm) => .ok [(v, .app q pm)] s | none => .fail .bounds
  | .ld64, [a] => match s.ld a.w 8 with | some (v, pm) => .ok [(v, .app q pm)] s | none => .fail .bounds
  | .st8, [a, v] => match s.st a.w 1 v.w qn with | some s' => .ok [] s' | none => .fail .bounds
  | .st16, [a, v] => match s.st a.w 2 v.w qn with | some s' => .ok [] s' | none => .fail .bounds
  | .st32, [a, v] => match s.st a.w 4 v.w qn with | some s' => .ok [] s' | none => .fail .bounds
  | .st64, [a, v] => match s.st a.w 8 v.w qn with | some s' => .ok [] s' | none => .fail .bounds
  | .copy, [n, dst, src] =>
    -- each copied byte keeps its own provenance joined with q: one node per byte
    if src.w + n.w ≤ s.msize ∧ dst.w + n.w ≤ s.msize then
      let base := s.ar.length
      let ar' := s.ar ++ (List.range n.w).map fun i => DNode.join (s.mem (src.w + i)).n qn
      .ok [] { s with ar := ar', mem := (fun x =>
        if dst.w ≤ x ∧ x < dst.w + n.w then ⟨(s.mem (src.w + (x - dst.w))).w, base + (x - dst.w)⟩
        else s.mem x) }
    else .fail .bounds
  | .fill, [b, n, dst] =>
    if dst.w + n.w ≤ s.msize then
      .ok [] { s with mem := (fun x => if dst.w ≤ x ∧ x < dst.w + n.w then ⟨b.w % 256, qn⟩ else s.mem x) }
    else .fail .bounds
  | .mcmp, [n, y, x] =>
    if x.w + n.w ≤ s.msize ∧ y.w + n.w ≤ s.msize then
      let eqb := (List.range n.w).all fun i => (s.mem (x.w + i)).w % 256 == (s.mem (y.w + i)).w % 256
      let pm := (List.range n.w).foldr (fun i acc => .app (.app (.node (s.mem (x.w + i)).n) (.node (s.mem (y.w + i)).n)) acc) .nil
      .ok [(bool eqb, .app q pm)] s
    else .fail .bounds
  | .alloc, [n] =>
    if s.hp + n.w ≤ s.msize then .ok [mk s.hp] { s with hp := s.hp + n.w } else .fail .bounds
  | .mark, [] => .ok [mk s.hp] s
  | .release, [m] => if m.w ≤ s.hp then .ok [] { s with hp := m.w } else .fail .bounds
  | .streq, [l2, a2, l1, a1] =>
    if a1.w + l1.w ≤ s.msize ∧ a2.w + l2.w ≤ s.msize then
      let eqb := l1.w == l2.w && (List.range l1.w).all fun i =>
        (s.mem (a1.w + i)).w % 256 == (s.mem (a2.w + i)).w % 256
      let pm : PE := if l1.w = l2.w then
        (List.range l1.w).foldr (fun i acc => .app (.app (.node (s.mem (a1.w + i)).n) (.node (s.mem (a2.w + i)).n)) acc) .nil
        else .nil
      .ok [(bool eqb, .app q pm)] s
    else .fail .bounds
  | .strhash, [l, a] =>
    if a.w + l.w ≤ s.msize then
      let pm : PE := (List.range l.w).foldr (fun i acc => .app (.node (s.mem (a.w + i)).n) acc) .nil
      .ok [(wrap (fnv ((List.range l.w).map fun i => (s.mem (a.w + i)).w % 256)), .app q pm)] s
    else .fail .bounds
  | .tnew, [] => .ok [mk s.tabs.length] { s with tabs := s.tabs ++ [[]] }
  | .tput, [v, k, tb] =>
    if h : tb.w < s.tabs.length then
      let nd := s.ar.length
      .ok [] { s with ar := s.ar ++ [.join v.n qn],
                      tabs := s.tabs.set tb.w ((k.w, ⟨v.w, nd⟩) :: s.tabs[tb.w]) }
    else .fail .badTable
  | .tget, [k, tb] =>
    if h : tb.w < s.tabs.length then
      match tfind' s.tabs[tb.w] k.w with
      | some v => .ok [mk 1, (v.w, .app (.node v.n) q)] s
      | none => .ok [mk 0, mk 0] s
    else .fail .badTable
  | .getb, [] =>
    match s.inp with
    | [] => .ok [mk (M - 1)] s
    | c :: rest =>
      .ok [(c % 256, .app (.node s.pc) (.leaf { o with inp := some s.ipos }))]
        { s with inp := rest, ipos := s.ipos + 1 }
  | .putb, [b] => .ok [] { s with out := (b.w % 256, qn) :: s.out }
  | .putn, [n] => .ok [] { s with out := ((List.range 8).map fun i => (n.w / 256 ^ i % 256, qn)).reverse ++ s.out }
  | .exit, [c] => .halt (c.w % 256)
  | .clock, [] => .ok [mk (ph.clock s.step)] s
  | .core, [] => .ok [mk (ph.core s.step)] s
  | .sensor, [i] => .ok [mk (ph.sensor i.w s.step)] s
  | .provget, [a] => .ok [mk (leaves s.ar a.n).length, t a] s
  | .confget, [a] => .ok [mk (ph.conf (leaves s.ar a.n)), t a] s
  | .demand, [th, a] => if ph.conf (leaves s.ar a.n) < th.w then .fail .demand else .ok [t a] s
  | _, _ => .fail .underflow

def DSt.setTop (s : DSt) (f : DFrame) : DSt :=
  match s.rs with
  | [] => s
  | _ :: r => { s with rs := f :: r }

inductive DRes where
  | next (s : DSt)
  | halt (code : Nat) (out : List (Nat × Nat)) (ar : Arena)
  | fail (e : Err)

/-- Push one value whose provenance is plan `e`. -/
def DSt.push (s : DSt) (w : Nat) (e : PE) : DSt :=
  let (n, ar) := A.build e s.ar
  { s with ar := ar, ds := ⟨w, n⟩ :: s.ds }

def dstep (pr : Prog) (ph : Phys) (s : DSt) : DRes :=
  match s.rs with
  | [] => .fail .noFrame
  | f :: rest =>
    match fetch pr f.word f.pc with
    | none => .fail .badJump
    | some ins =>
      let o : Origin := ⟨none, f.word, f.pc, ph.core s.step⟩
      let s := { s with step := s.step + 1 }
      let adv (s : DSt) : DSt := s.setTop { f with pc := f.pc + 1 }
      match ins with
      | .prim p =>
        let k := p.eff.1
        if s.ds.length < k then .fail .underflow else
        let args := s.ds.take k
        let (qn, ar1) := A.build (qPlan args s.pc o) s.ar
        match execPrimD ph p args { s with ar := ar1 } o qn with
        | .ok res s' =>
          let (vs, ar2) := buildAll A res s'.ar
          .next (adv { s' with ar := ar2, ds := vs ++ s.ds.drop k })
        | .halt c => .halt c s.out s.ar
        | .fail e => .fail e
      | .lit n => .next (adv (s.push A (wrap n) (.app (.node s.pc) (.leaf o))))
      | .str i =>
        match pr.data[i]? with
        | none => .fail .bounds
        | some d =>
          let addr := ((pr.data.take i).map List.length).foldl (· + ·) 0
          .next (adv ((s.push A addr (.app (.node s.pc) (.leaf o))).push A d.length (.app (.node s.pc) (.leaf o))))
      | .pick n =>
        match s.ds[n]? with
        | none => .fail .underflow
        | some v => .next (adv (s.push A v.w (tagPlan s.pc o v)))
      | .tor =>
        match s.ds with
        | [] => .fail .underflow
        | v :: r =>
          let (n, ar) := A.build (tagPlan s.pc o v) s.ar
          .next (adv { s with ds := r, aux := ⟨v.w, n⟩ :: s.aux, ar := ar })
      | .fromr =>
        match s.aux with
        | [] => .fail .auxUnderflow
        | v :: r => .next (adv ({ s with aux := r }.push A v.w (tagPlan s.pc o v)))
      | .rfetch =>
        match s.aux with
        | [] => .fail .auxUnderflow
        | v :: _ => .next (adv (s.push A v.w (tagPlan s.pc o v)))
      | .locget i =>
        match f.locs[i]? with
        | none => .fail .badLocal
        | some v => .next (adv (s.push A v.w (tagPlan s.pc o v)))
      | .locset i =>
        match s.ds with
        | [] => .fail .underflow
        | v :: r =>
          if i < f.locs.length then
            let (n, ar) := A.build (tagPlan s.pc o v) s.ar
            .next ({ s with ds := r, ar := ar }.setTop { f with pc := f.pc + 1, locs := f.locs.set i ⟨v.w, n⟩ })
          else .fail .badLocal
      | .jmp t => .next (s.setTop { f with pc := t })
      | .jz t =>
        match s.ds with
        | [] => .fail .underflow
        | c :: r =>
          let (n, ar) := A.build (.app (.app (.node s.pc) (.node c.n)) (.leaf o)) s.ar
          let s := { s with ds := r, pc := n, ar := ar }
          .next (s.setTop { f with pc := if c.w % M = 0 then t else f.pc + 1 })
      | .jnz t =>
        match s.ds with
        | [] => .fail .underflow
        | c :: r =>
          let (n, ar) := A.build (.app (.app (.node s.pc) (.node c.n)) (.leaf o)) s.ar
          let s := { s with ds := r, pc := n, ar := ar }
          .next (s.setTop { f with pc := if c.w % M = 0 then f.pc + 1 else t })
      | .call w =>
        match pr.words[w]? with
        | none => .fail .badWord
        | some wd =>
          if s.ds.length < wd.ins then .fail .underflow else
          let (n, ar) := A.build (.app (.node s.pc) (.leaf o)) s.ar
          let callee : DFrame := ⟨w, 0, List.replicate wd.locals ⟨0, n⟩, s.ds.length - wd.ins, s.aux.length⟩
          .next { s with ar := ar, rs := callee :: { f with pc := f.pc + 1 } :: rest }
      | .ret =>
        match rest with
        | [] => .halt 0 s.out s.ar
        | _ => .next { s with rs := rest }

end

end Sporos
