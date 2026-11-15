/-
One step of Sporos. Every value an instruction produces carries the origins of
its operands, the context label `pc`, and the origin of the instruction itself
(input position if it read one, word, index, core). Shuffles are instructions
too: a value moved by `swap` gains the origin of that `swap`, since the silicon
that executed it could have corrupted it.
-/
import Sporos.Core

namespace Sporos

/-- Result of a primitive on its operands (top first). -/
inductive PR where
  | ok (res : List Val) (s : St)
  | halt (code : Nat)
  | fail (e : Err)

/-- Tag a value moved by an instruction. -/
def tag (pc : Prov) (o : Origin) (v : Val) : Val := ⟨v.w, v.p ++ pc ++ [o]⟩

def bool (b : Bool) : Nat := if b then 1 else 0

/-- Find a key in a table. -/
def tfind : List (Nat × Val) → Nat → Option Val
  | [], _ => none
  | (k, v) :: r, x => if k = x then some v else tfind r x

def execPrim (ph : Phys) (p : Prim) (args : List Val) (s : St) (o : Origin) : PR :=
  let q : Prov := (args.foldr (fun v acc => v.p ++ acc) []) ++ s.pc ++ [o]
  let mk (w : Nat) : Val := ⟨wrap w, q⟩
  let t := tag s.pc o
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
  | .ld8, [a] => match s.ld a.w 1 with | some (v, pm) => .ok [⟨v, q ++ pm⟩] s | none => .fail .bounds
  | .ld16, [a] => match s.ld a.w 2 with | some (v, pm) => .ok [⟨v, q ++ pm⟩] s | none => .fail .bounds
  | .ld32, [a] => match s.ld a.w 4 with | some (v, pm) => .ok [⟨v, q ++ pm⟩] s | none => .fail .bounds
  | .ld64, [a] => match s.ld a.w 8 with | some (v, pm) => .ok [⟨v, q ++ pm⟩] s | none => .fail .bounds
  | .st8, [a, v] => match s.st a.w 1 v.w q with | some s' => .ok [] s' | none => .fail .bounds
  | .st16, [a, v] => match s.st a.w 2 v.w q with | some s' => .ok [] s' | none => .fail .bounds
  | .st32, [a, v] => match s.st a.w 4 v.w q with | some s' => .ok [] s' | none => .fail .bounds
  | .st64, [a, v] => match s.st a.w 8 v.w q with | some s' => .ok [] s' | none => .fail .bounds
  | .copy, [n, dst, src] =>
    -- ( src dst n -- ): every destination byte is read from the memory before the copy,
    -- so overlapping blocks behave as memmove (`copy_overlap`)
    if src.w + n.w ≤ s.msize ∧ dst.w + n.w ≤ s.msize then
      .ok [] { s with mem := (fun x =>
        if dst.w ≤ x ∧ x < dst.w + n.w then ⟨(s.mem (src.w + (x - dst.w))).w, (s.mem (src.w + (x - dst.w))).p ++ q⟩
        else s.mem x) }
    else .fail .bounds
  | .fill, [b, n, dst] =>
    -- ( dst n b -- )
    if dst.w + n.w ≤ s.msize then
      .ok [] { s with mem := (fun x => if dst.w ≤ x ∧ x < dst.w + n.w then ⟨b.w % 256, q⟩ else s.mem x) }
    else .fail .bounds
  | .mcmp, [n, y, x] =>
    -- ( a b n -- f ): 1 when the two blocks are equal
    if x.w + n.w ≤ s.msize ∧ y.w + n.w ≤ s.msize then
      let eqb := (List.range n.w).all fun i => (s.mem (x.w + i)).w % 256 == (s.mem (y.w + i)).w % 256
      let pm := (List.range n.w).foldr (fun i acc => (s.mem (x.w + i)).p ++ (s.mem (y.w + i)).p ++ acc) []
      .ok [⟨bool eqb, q ++ pm⟩] s
    else .fail .bounds
  | .alloc, [n] =>
    if s.hp + n.w ≤ s.msize then .ok [mk s.hp] { s with hp := s.hp + n.w } else .fail .bounds
  | .mark, [] => .ok [mk s.hp] s
  | .release, [m] => if m.w ≤ s.hp then .ok [] { s with hp := m.w } else .fail .bounds
  | .streq, [l2, a2, l1, a1] =>
    if a1.w + l1.w ≤ s.msize ∧ a2.w + l2.w ≤ s.msize then
      let eqb := l1.w == l2.w && (List.range l1.w).all fun i =>
        (s.mem (a1.w + i)).w % 256 == (s.mem (a2.w + i)).w % 256
      -- the bytes compared determine the result: when the lengths differ none is read
      let pm := if l1.w = l2.w then
        (List.range l1.w).foldr (fun i acc => (s.mem (a1.w + i)).p ++ (s.mem (a2.w + i)).p ++ acc) [] else []
      .ok [⟨bool eqb, q ++ pm⟩] s
    else .fail .bounds
  | .strhash, [l, a] =>
    if a.w + l.w ≤ s.msize then
      let pm := (List.range l.w).foldr (fun i acc => (s.mem (a.w + i)).p ++ acc) []
      .ok [⟨wrap (fnv ((List.range l.w).map fun i => (s.mem (a.w + i)).w % 256)), q ++ pm⟩] s
    else .fail .bounds
  | .tnew, [] => .ok [mk s.tabs.length] { s with tabs := s.tabs ++ [[]] }
  | .tput, [v, k, tb] =>
    -- ( t k v -- )
    if h : tb.w < s.tabs.length then
      .ok [] { s with tabs := s.tabs.set tb.w ((k.w, ⟨v.w, v.p ++ q⟩) :: s.tabs[tb.w]) }
    else .fail .badTable
  | .tget, [k, tb] =>
    -- ( t k -- v found )
    if h : tb.w < s.tabs.length then
      match tfind s.tabs[tb.w] k.w with
      | some v => .ok [mk 1, ⟨v.w, v.p ++ q⟩] s
      | none => .ok [mk 0, mk 0] s
    else .fail .badTable
  | .getb, [] =>
    match s.inp with
    | [] => .ok [mk (M - 1)] s
    | c :: rest =>
      let oi : Origin := { o with inp := some s.ipos }
      .ok [⟨c % 256, s.pc ++ [oi]⟩] { s with inp := rest, ipos := s.ipos + 1 }
  | .putb, [b] => .ok [] { s with out := (b.w % 256, q) :: s.out }
  | .putn, [n] =>
    .ok [] { s with out := ((List.range 8).map fun i => (n.w / 256 ^ i % 256, q)).reverse ++ s.out }
  | .exit, [c] => .halt (c.w % 256)
  | .clock, [] => .ok [mk (ph.clock s.step)] s
  | .core, [] => .ok [mk (ph.core s.step)] s
  | .sensor, [i] => .ok [mk (ph.sensor i.w s.step)] s
  | .provget, [a] => .ok [mk a.p.length, t a] s
  | .confget, [a] => .ok [mk (ph.conf a.p), t a] s
  | .demand, [th, a] => if ph.conf a.p < th.w then .fail .demand else .ok [t a] s
  | _, _ => .fail .underflow

/-- The instruction at `pc` of word `w`. -/
def fetch (pr : Prog) (w pc : Nat) : Option Instr := do
  let wd ← pr.words[w]?
  wd.code[pc]?

def St.setTop (s : St) (f : Frame) : St :=
  match s.rs with
  | [] => s
  | _ :: r => { s with rs := f :: r }

def step (pr : Prog) (ph : Phys) (s : St) : Res :=
  match s.rs with
  | [] => .fail .noFrame
  | f :: rest =>
    match fetch pr f.word f.pc with
    | none => .fail .badJump
    | some ins =>
      let o : Origin := ⟨none, f.word, f.pc, ph.core s.step⟩
      let s := { s with step := s.step + 1 }
      let adv (s : St) : St := s.setTop { f with pc := f.pc + 1 }
      match ins with
      | .prim p =>
        let k := p.eff.1
        if s.ds.length < k then .fail .underflow else
        match execPrim ph p (s.ds.take k) s o with
        | .ok res s' => .next (adv { s' with ds := res ++ s.ds.drop k })
        | .halt c => .halt c s.out
        | .fail e => .fail e
      | .lit n => .next (adv { s with ds := ⟨wrap n, s.pc ++ [o]⟩ :: s.ds })
      | .str i =>
        match pr.data[i]? with
        | none => .fail .bounds
        | some d =>
          let addr := ((pr.data.take i).map List.length).foldl (· + ·) 0
          .next (adv { s with ds := ⟨d.length, s.pc ++ [o]⟩ :: ⟨addr, s.pc ++ [o]⟩ :: s.ds })
      | .pick n =>
        match s.ds[n]? with
        | none => .fail .underflow
        | some v => .next (adv { s with ds := tag s.pc o v :: s.ds })
      | .tor =>
        match s.ds with
        | [] => .fail .underflow
        | v :: r => .next (adv { s with ds := r, aux := tag s.pc o v :: s.aux })
      | .fromr =>
        match s.aux with
        | [] => .fail .auxUnderflow
        | v :: r => .next (adv { s with aux := r, ds := tag s.pc o v :: s.ds })
      | .rfetch =>
        match s.aux with
        | [] => .fail .auxUnderflow
        | v :: _ => .next (adv { s with ds := tag s.pc o v :: s.ds })
      | .locget i =>
        match f.locs[i]? with
        | none => .fail .badLocal
        | some v => .next (adv { s with ds := tag s.pc o v :: s.ds })
      | .locset i =>
        match s.ds with
        | [] => .fail .underflow
        | v :: r =>
          if i < f.locs.length then
            .next ({ s with ds := r }.setTop { f with pc := f.pc + 1, locs := f.locs.set i (tag s.pc o v) })
          else .fail .badLocal
      | .jmp t => .next (s.setTop { f with pc := t })
      | .jz t =>
        match s.ds with
        | [] => .fail .underflow
        | c :: r =>
          -- the condition's origins enter the context label, for good
          let s := { s with ds := r, pc := s.pc ++ c.p ++ [o] }
          .next (s.setTop { f with pc := if c.w % M = 0 then t else f.pc + 1 })
      | .jnz t =>
        match s.ds with
        | [] => .fail .underflow
        | c :: r =>
          let s := { s with ds := r, pc := s.pc ++ c.p ++ [o] }
          .next (s.setTop { f with pc := if c.w % M = 0 then f.pc + 1 else t })
      | .call w =>
        match pr.words[w]? with
        | none => .fail .badWord
        | some wd =>
          if s.ds.length < wd.ins then .fail .underflow else
          let callee : Frame := ⟨w, 0, List.replicate wd.locals ⟨0, s.pc ++ [o]⟩, s.ds.length - wd.ins, s.aux.length⟩
          .next { s with rs := callee :: { f with pc := f.pc + 1 } :: rest }
      | .ret =>
        match rest with
        | [] => .halt 0 s.out
        | _ => .next { s with rs := rest }

end Sporos
