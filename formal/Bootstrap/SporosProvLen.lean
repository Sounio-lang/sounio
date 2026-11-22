/-
SporosProvLen: exact observational bisimulation between `Sporos.run pr Sporos.noPhys`
and provenance tracked solely by 64-bit list length modulo `2^64` (`LVal.plen < 2^64`),
closing the formal conjecture documented in `bootstrap/model/sporos_fast.cpp:10-17`.

Under `Sporos.noPhys`:
- `clock`, `core`, and `sensor` return `0`
- `conf` is constantly `0` (`confget` pushes `0`, `demand` fails iff `0 < th.w`)
- `provget` observes only `a.p.length % 2^64` (via `mk a.p.length`)
- every provenance concatenation `++` reduces modulo `2^64` to addition of lengths

Main results:
1. `linit_rel`: `Rel (Sporos.init pr msize inp) (linit pr msize inp)`.
2. `lstep_sim`: every `Sporos.step pr Sporos.noPhys s` matches `lstep pr ls`
   under `Rel` (`.next` preserves `Rel`, `.halt c out` matches `c` and `out.map Prod.fst`,
   `.fail e` matches `e`).
3. `lrun_sim`, `lrun_sim_init`, `lrun_sim_halt`, `lrun_sim_fail`, `lrun_sim_next`:
   universal observational equivalence for all `pr`, `msize`, `inp`, `n`.
4. `lrun_safe`: composition with `Sporos.run_safe` (`checkProg pr = true` rules out
   all static errors in `lrun`).
5. `linit_wordsLt`, `lstep_wordsLt`, `lrun_wordsLt`: every `LVal` word stays `< 2^64`
   whenever `pr.data.flatten.length < 2^64` (and `plen < 2^64`, `pc < 2^64` unconditionally).
6. Boundary lemmas for `sporos_fast.cpp` (`divs`/`mods` including `INT64_MIN / -1`,
   `streq` unequal-length provenance non-read, `copy` memmove snapshot, `strhash` FNV-1a,
   `tnew`/`tput`/`tget` LIFO shadowing, `clock`/`core`/`sensor`/`provget`/`confget`/`demand`)
   and kernel-checked (`decide +kernel`) witnesses on `echo`, `hello`, `upper`, `fib`,
   and a non-trivial `provget` + `confget` + `demand` program.
-/
import Sporos.Run
import Sporos.Examples

namespace SporosProvLen

open Sporos

/-! ### Provenance-length values, frames, memory, and machine state -/

/-- A machine value carrying its word `w` and 64-bit provenance list length `plen`. -/
structure LVal where
  w    : Nat
  plen : Nat
deriving DecidableEq, Repr

/-- Call-stack frame for the provenance-length semantics. -/
structure LFrame where
  word  : Nat
  pc    : Nat
  locs  : List LVal
  base  : Nat
  abase : Nat
deriving DecidableEq, Repr

/-- Byte-addressed memory map carrying word bytes and 64-bit provenance lengths. -/
abbrev LMem := Nat → LVal

/-- Machine state tracking provenance solely by 64-bit list length modulo `2^64`. -/
structure LSt where
  ds    : List LVal
  aux   : List LVal
  rs    : List LFrame
  mem   : LMem
  msize : Nat
  hp    : Nat
  tabs  : List (List (Nat × LVal))
  inp   : List Nat
  ipos  : Nat
  out   : List Nat
  pc    : Nat

/-- Result of executing a single primitive on `LSt`. -/
inductive LPR where
  | ok (res : List LVal) (s : LSt)
  | halt (code : Nat)
  | fail (e : Err)

/-- Outcome of one step or run of `lstep`/`lrun`. -/
inductive LRes where
  | next (s : LSt)
  | halt (code : Nat) (out : List Nat)
  | fail (e : Err)

/-- Tag a value moved by an instruction with context length `lpc` and instruction origin (+1). -/
def ltag (lpc : Nat) (v : LVal) : LVal :=
  ⟨v.w, (v.plen + lpc + 1) % M⟩

/-- Combined provenance length `(sum args.plen + lpc + 1) mod 2^64`. -/
def lq (args : List LVal) (lpc : Nat) : Nat :=
  (args.foldr (fun v acc => v.plen + acc) 0 + lpc + 1) % M

/-- Lookup the newest matching key in a table (LIFO shadowing). -/
def ltfind : List (Nat × LVal) → Nat → Option LVal
  | [], _ => none
  | (k, v) :: r, x => if k = x then some v else ltfind r x

/-- Read `n` little-endian bytes from memory, summing byte provenance lengths. -/
def LSt.ld (s : LSt) (a n : Nat) : Option (Nat × Nat) :=
  if a + n ≤ s.msize then
    some ((List.range n).foldr (fun i acc => acc * 256 + (s.mem (a + i)).w % 256) 0,
          (List.range n).foldr (fun i acc => (s.mem (a + i)).plen + acc) 0)
  else none

/-- Store `n` little-endian bytes of `v` with provenance length `q`. -/
def LSt.st (s : LSt) (a n v q : Nat) : Option LSt :=
  if a + n ≤ s.msize then
    some { s with mem := (fun x => if a ≤ x ∧ x < a + n then ⟨v / 256 ^ (x - a) % 256, q⟩ else s.mem x) }
  else none

/-- Replace the top call frame if non-empty. -/
def LSt.setTop (s : LSt) (f : LFrame) : LSt :=
  match s.rs with
  | [] => s
  | _ :: r => { s with rs := f :: r }

/-- Read-only data laid out from address 0 with provenance length 0. -/
def lloadData (data : List (List Nat)) : LMem := fun a =>
  match (data.flatten)[a]? with
  | some b => ⟨b % 256, 0⟩
  | none => ⟨0, 0⟩

/-- Initial provenance-length state corresponding to `Sporos.init`. -/
def linit (pr : Prog) (msize : Nat) (inp : List Nat) : LSt :=
  let locals := match pr.words[pr.main]? with | some m => m.locals | none => 0
  { ds := [], aux := [], rs := [⟨pr.main, 0, List.replicate locals ⟨0, 0⟩, 0, 0⟩],
    mem := lloadData pr.data, msize := msize, hp := pr.data.flatten.length, tabs := [],
    inp := inp, ipos := 0, out := [], pc := 0 }

/-- Execute one primitive under `noPhys` using 64-bit provenance lengths. -/
def lexecPrim (p : Prim) (args : List LVal) (s : LSt) : LPR :=
  let q : Nat := lq args s.pc
  let mk (w : Nat) : LVal := ⟨wrap w, q⟩
  let t := ltag s.pc
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
  | .ld8, [a] => match s.ld a.w 1 with | some (v, pm) => .ok [⟨v, (q + pm) % M⟩] s | none => .fail .bounds
  | .ld16, [a] => match s.ld a.w 2 with | some (v, pm) => .ok [⟨v, (q + pm) % M⟩] s | none => .fail .bounds
  | .ld32, [a] => match s.ld a.w 4 with | some (v, pm) => .ok [⟨v, (q + pm) % M⟩] s | none => .fail .bounds
  | .ld64, [a] => match s.ld a.w 8 with | some (v, pm) => .ok [⟨v, (q + pm) % M⟩] s | none => .fail .bounds
  | .st8, [a, v] => match s.st a.w 1 v.w q with | some s' => .ok [] s' | none => .fail .bounds
  | .st16, [a, v] => match s.st a.w 2 v.w q with | some s' => .ok [] s' | none => .fail .bounds
  | .st32, [a, v] => match s.st a.w 4 v.w q with | some s' => .ok [] s' | none => .fail .bounds
  | .st64, [a, v] => match s.st a.w 8 v.w q with | some s' => .ok [] s' | none => .fail .bounds
  | .copy, [n, dst, src] =>
    if src.w + n.w ≤ s.msize ∧ dst.w + n.w ≤ s.msize then
      .ok [] { s with mem := (fun x =>
        if dst.w ≤ x ∧ x < dst.w + n.w then
          ⟨(s.mem (src.w + (x - dst.w))).w, ((s.mem (src.w + (x - dst.w))).plen + q) % M⟩
        else s.mem x) }
    else .fail .bounds
  | .fill, [b, n, dst] =>
    if dst.w + n.w ≤ s.msize then
      .ok [] { s with mem := (fun x => if dst.w ≤ x ∧ x < dst.w + n.w then ⟨b.w % 256, q⟩ else s.mem x) }
    else .fail .bounds
  | .mcmp, [n, y, x] =>
    if x.w + n.w ≤ s.msize ∧ y.w + n.w ≤ s.msize then
      let eqb := (List.range n.w).all fun i => (s.mem (x.w + i)).w % 256 == (s.mem (y.w + i)).w % 256
      let pm := (List.range n.w).foldr (fun i acc => (s.mem (x.w + i)).plen + (s.mem (y.w + i)).plen + acc) 0
      .ok [⟨bool eqb, (q + pm) % M⟩] s
    else .fail .bounds
  | .alloc, [n] =>
    if s.hp + n.w ≤ s.msize then .ok [mk s.hp] { s with hp := s.hp + n.w } else .fail .bounds
  | .mark, [] => .ok [mk s.hp] s
  | .release, [m] => if m.w ≤ s.hp then .ok [] { s with hp := m.w } else .fail .bounds
  | .streq, [l2, a2, l1, a1] =>
    if a1.w + l1.w ≤ s.msize ∧ a2.w + l2.w ≤ s.msize then
      let eqb := l1.w == l2.w && (List.range l1.w).all fun i =>
        (s.mem (a1.w + i)).w % 256 == (s.mem (a2.w + i)).w % 256
      let pm := if l1.w = l2.w then
        (List.range l1.w).foldr (fun i acc => (s.mem (a1.w + i)).plen + (s.mem (a2.w + i)).plen + acc) 0 else 0
      .ok [⟨bool eqb, (q + pm) % M⟩] s
    else .fail .bounds
  | .strhash, [l, a] =>
    if a.w + l.w ≤ s.msize then
      let pm := (List.range l.w).foldr (fun i acc => (s.mem (a.w + i)).plen + acc) 0
      .ok [⟨wrap (fnv ((List.range l.w).map fun i => (s.mem (a.w + i)).w % 256)), (q + pm) % M⟩] s
    else .fail .bounds
  | .tnew, [] => .ok [mk s.tabs.length] { s with tabs := s.tabs ++ [[]] }
  | .tput, [v, k, tb] =>
    if h : tb.w < s.tabs.length then
      .ok [] { s with tabs := s.tabs.set tb.w ((k.w, ⟨v.w, (v.plen + q) % M⟩) :: s.tabs[tb.w]) }
    else .fail .badTable
  | .tget, [k, tb] =>
    if h : tb.w < s.tabs.length then
      match ltfind s.tabs[tb.w] k.w with
      | some v => .ok [mk 1, ⟨v.w, (v.plen + q) % M⟩] s
      | none => .ok [mk 0, mk 0] s
    else .fail .badTable
  | .getb, [] =>
    match s.inp with
    | [] => .ok [mk (M - 1)] s
    | c :: rest =>
      .ok [⟨c % 256, (s.pc + 1) % M⟩] { s with inp := rest, ipos := s.ipos + 1 }
  | .putb, [b] => .ok [] { s with out := (b.w % 256) :: s.out }
  | .putn, [n] =>
    .ok [] { s with out := ((List.range 8).map fun i => n.w / 256 ^ i % 256).reverse ++ s.out }
  | .exit, [c] => .halt (c.w % 256)
  | .clock, [] => .ok [mk 0] s
  | .core, [] => .ok [mk 0] s
  | .sensor, [_] => .ok [mk 0] s
  | .provget, [a] => .ok [mk a.plen, t a] s
  | .confget, [a] => .ok [mk 0, t a] s
  | .demand, [th, a] => if 0 < th.w then .fail .demand else .ok [t a] s
  | _, _ => .fail .underflow

/-- One step of the provenance-length semantics covering all `Instr` constructors. -/
def lstep (pr : Prog) (s : LSt) : LRes :=
  match s.rs with
  | [] => .fail .noFrame
  | f :: rest =>
    match fetch pr f.word f.pc with
    | none => .fail .badJump
    | some ins =>
      let adv (s : LSt) : LSt := s.setTop { f with pc := f.pc + 1 }
      match ins with
      | .prim p =>
        let k := p.eff.1
        if s.ds.length < k then .fail .underflow else
        match lexecPrim p (s.ds.take k) s with
        | .ok res s' => .next (adv { s' with ds := res ++ s.ds.drop k })
        | .halt c => .halt c s.out
        | .fail e => .fail e
      | .lit n => .next (adv { s with ds := ⟨wrap n, (s.pc + 1) % M⟩ :: s.ds })
      | .str i =>
        match pr.data[i]? with
        | none => .fail .bounds
        | some d =>
          let addr := ((pr.data.take i).map List.length).foldl (· + ·) 0
          .next (adv { s with ds := ⟨d.length, (s.pc + 1) % M⟩ :: ⟨addr, (s.pc + 1) % M⟩ :: s.ds })
      | .pick n =>
        match s.ds[n]? with
        | none => .fail .underflow
        | some v => .next (adv { s with ds := ltag s.pc v :: s.ds })
      | .tor =>
        match s.ds with
        | [] => .fail .underflow
        | v :: r => .next (adv { s with ds := r, aux := ltag s.pc v :: s.aux })
      | .fromr =>
        match s.aux with
        | [] => .fail .auxUnderflow
        | v :: r => .next (adv { s with aux := r, ds := ltag s.pc v :: s.ds })
      | .rfetch =>
        match s.aux with
        | [] => .fail .auxUnderflow
        | v :: _ => .next (adv { s with ds := ltag s.pc v :: s.ds })
      | .locget i =>
        match f.locs[i]? with
        | none => .fail .badLocal
        | some v => .next (adv { s with ds := ltag s.pc v :: s.ds })
      | .locset i =>
        match s.ds with
        | [] => .fail .underflow
        | v :: r =>
          if i < f.locs.length then
            .next ({ s with ds := r }.setTop { f with pc := f.pc + 1, locs := f.locs.set i (ltag s.pc v) })
          else .fail .badLocal
      | .jmp t => .next (s.setTop { f with pc := t })
      | .jz t =>
        match s.ds with
        | [] => .fail .underflow
        | c :: r =>
          let s := { s with ds := r, pc := (s.pc + c.plen + 1) % M }
          .next (s.setTop { f with pc := if c.w % M = 0 then t else f.pc + 1 })
      | .jnz t =>
        match s.ds with
        | [] => .fail .underflow
        | c :: r =>
          let s := { s with ds := r, pc := (s.pc + c.plen + 1) % M }
          .next (s.setTop { f with pc := if c.w % M = 0 then f.pc + 1 else t })
      | .call w =>
        match pr.words[w]? with
        | none => .fail .badWord
        | some wd =>
          if s.ds.length < wd.ins then .fail .underflow else
          let callee : LFrame := ⟨w, 0, List.replicate wd.locals ⟨0, (s.pc + 1) % M⟩, s.ds.length - wd.ins, s.aux.length⟩
          .next { s with rs := callee :: { f with pc := f.pc + 1 } :: rest }
      | .ret =>
        match rest with
        | [] => .halt 0 s.out
        | _ => .next { s with rs := rest }

/-- Run `lstep` for up to `n` steps. -/
def lrun (pr : Prog) : Nat → LSt → LRes
  | 0, s => .next s
  | n + 1, s =>
    match lstep pr s with
    | .next t => lrun pr n t
    | r => r

/-! ### Abstraction maps and state bisimulation relation -/

/-- Project a reference `Val` to its `LVal` (word, provenance length mod `2^64`). -/
def absVal (v : Val) : LVal := ⟨v.w, v.p.length % M⟩

/-- Project a reference `Frame` to `LFrame`. -/
def absFrame (f : Frame) : LFrame := ⟨f.word, f.pc, f.locs.map absVal, f.base, f.abase⟩

/-- Project a reference `St` to `LSt`. -/
def absSt (s : St) : LSt :=
  { ds := s.ds.map absVal
    aux := s.aux.map absVal
    rs := s.rs.map absFrame
    mem := fun a => absVal (s.mem a)
    msize := s.msize
    hp := s.hp
    tabs := s.tabs.map (List.map fun kv => (kv.1, absVal kv.2))
    inp := s.inp
    ipos := s.ipos
    out := s.out.map Prod.fst
    pc := s.pc.length % M }

/-- Project a reference `PR` to `LPR`. -/
def absPR : PR → LPR
  | .ok res s => .ok (res.map absVal) (absSt s)
  | .halt c => .halt c
  | .fail e => .fail e

/-- Project a reference `Res` to `LRes`. -/
def absRes : Res → LRes
  | .next s => .next (absSt s)
  | .halt c out => .halt c (out.map Prod.fst)
  | .fail e => .fail e

/-- Value-level bisimulation relation: identical word, provenance length mod `2^64`. -/
def ValRel (v : Val) (lv : LVal) : Prop :=
  v.w = lv.w ∧ v.p.length % M = lv.plen

/-- Frame-level bisimulation relation. -/
def FrameRel (f : Frame) (lf : LFrame) : Prop :=
  f.word = lf.word ∧
  f.pc = lf.pc ∧
  f.locs.map absVal = lf.locs ∧
  f.base = lf.base ∧
  f.abase = lf.abase

/-- State bisimulation relation between `Sporos.St` and `LSt`. -/
structure Rel (s : St) (ls : LSt) : Prop where
  ds    : s.ds.map absVal = ls.ds
  aux   : s.aux.map absVal = ls.aux
  rs    : s.rs.map absFrame = ls.rs
  mem   : ∀ a, ValRel (s.mem a) (ls.mem a)
  msize : s.msize = ls.msize
  hp    : s.hp = ls.hp
  tabs  : s.tabs.map (List.map fun kv => (kv.1, absVal kv.2)) = ls.tabs
  inp   : s.inp = ls.inp
  ipos  : s.ipos = ls.ipos
  out   : s.out.map Prod.fst = ls.out
  pc    : s.pc.length % M = ls.pc

/-- Outcome bisimulation relation between `Sporos.Res` and `LRes`. -/
inductive ResRel : Res → LRes → Prop
  | next {s : St} {ls : LSt} : Rel s ls → ResRel (.next s) (.next ls)
  | halt {c : Nat} {out : List (Nat × Prov)} {lout : List Nat} :
      out.map Prod.fst = lout → ResRel (.halt c out) (.halt c lout)
  | fail {e : Err} : ResRel (.fail e) (.fail e)

theorem M_pos : 0 < M := by decide

theorem absVal_plen_lt (v : Val) : (absVal v).plen < M :=
  Nat.mod_lt _ M_pos

theorem Rel.pc_lt {s : St} {ls : LSt} (h : Rel s ls) : ls.pc < M := by
  rw [← h.pc]; exact Nat.mod_lt _ M_pos

theorem Rel.mem_plen_lt {s : St} {ls : LSt} (h : Rel s ls) (a : Nat) : (ls.mem a).plen < M := by
  rw [← (h.mem a).2]; exact Nat.mod_lt _ M_pos

theorem absSt_rel (s : St) : Rel s (absSt s) where
  ds := rfl
  aux := rfl
  rs := rfl
  mem _ := ⟨rfl, rfl⟩
  msize := rfl
  hp := rfl
  tabs := rfl
  inp := rfl
  ipos := rfl
  out := rfl
  pc := rfl

theorem Rel.eq_abs {s : St} {ls : LSt} (h : Rel s ls) : ls = absSt s := by
  rcases ls with ⟨lds, laux, lrs, lmem, lms, lhp, ltabs, linp, lipos, lout, lpc⟩
  obtain ⟨hds, haux, hrs, hmem, hms, hhp, htabs, hinp, hipos, hout, hpc⟩ := h
  dsimp only at hds haux hrs hms hhp htabs hinp hipos hout hpc
  have hm : lmem = (fun a => absVal (s.mem a)) := by
    funext a
    obtain ⟨hw, hp⟩ := hmem a
    dsimp only at hw hp
    cases hma : lmem a
    rw [hma] at hw hp
    simp only [absVal, hw, hp]
  subst hds; subst haux; subst hrs; subst hm; subst hms; subst hhp
  subst htabs; subst hinp; subst hipos; subst hout; subst hpc
  rfl

theorem absRes_rel (r : Res) : ResRel r (absRes r) := by
  cases r with
  | next s => exact ResRel.next (absSt_rel s)
  | halt c out => exact ResRel.halt rfl
  | fail e => exact ResRel.fail

/-! ### Modular provenance-length lemmas -/

private theorem mod_add3 (a b c : Nat) : (a + b + c) % M = (a % M + b % M + c) % M := by
  rw [Nat.add_assoc, Nat.add_mod, Nat.add_mod b c M, Nat.add_mod_mod, ← Nat.add_assoc, Nat.add_mod_mod]

theorem foldr_prov_length (f : Nat → Prov) (l : List Nat) :
    (l.foldr (fun i acc => f i ++ acc) []).length % M =
      (l.foldr (fun i acc => (f i).length % M + acc) 0) % M := by
  induction l with
  | nil => rfl
  | cons i r ih =>
    simp only [List.foldr, List.length_append]
    rw [Nat.add_mod, ih, Nat.add_mod_mod]

theorem foldr_prov2_length (f g : Nat → Prov) (l : List Nat) :
    (l.foldr (fun i acc => f i ++ g i ++ acc) []).length % M =
      (l.foldr (fun i acc => (f i).length % M + (g i).length % M + acc) 0) % M := by
  induction l with
  | nil => rfl
  | cons i r ih =>
    simp only [List.foldr, List.length_append]
    rw [Nat.add_assoc (f i).length, Nat.add_mod (f i).length,
        Nat.add_mod (g i).length, ih, Nat.add_mod_mod ((g i).length % M),
        Nat.add_mod_mod, ← Nat.add_assoc ((f i).length % M)]

theorem args_foldr_length (args : List Val) :
    (args.foldr (fun (v : Val) acc => v.p ++ acc) []).length % M =
      ((args.map absVal).foldr (fun v acc => v.plen + acc) 0) % M := by
  induction args with
  | nil => rfl
  | cons v r ih =>
    simp only [List.map_cons, List.foldr, List.length_append, absVal]
    rw [Nat.add_mod, ih, Nat.add_mod_mod]

theorem q_length_mod (args : List Val) (pc : Prov) (o : Origin) :
    ((args.foldr (fun (v : Val) acc => v.p ++ acc) []) ++ pc ++ [o]).length % M =
      lq (args.map absVal) (pc.length % M) := by
  simp only [lq, List.length_append, List.length_singleton]
  rw [mod_add3, args_foldr_length]
  rw [Nat.add_assoc, Nat.mod_add_mod, ← Nat.add_assoc]

theorem absVal_tag (pc : Prov) (o : Origin) (v : Val) :
    absVal (tag pc o v) = ltag (pc.length % M) (absVal v) := by
  simp only [absVal, tag, ltag, List.length_append, List.length_singleton]
  congr 1
  exact mod_add3 v.p.length pc.length 1

theorem ltfind_map (l : List (Nat × Val)) (k : Nat) :
    ltfind (l.map fun kv => (kv.1, absVal kv.2)) k = (tfind l k).map absVal := by
  induction l with
  | nil => rfl
  | cons kv r ih =>
    obtain ⟨k', v⟩ := kv
    simp only [List.map, ltfind, tfind]
    split <;> simp_all

theorem absSt_st (s : St) (a n v : Nat) (q : Prov) :
    (absSt s).st a n v (q.length % M) = (s.st a n v q).map absSt := by
  simp only [St.st, LSt.st]
  have hms : (absSt s).msize = s.msize := rfl
  rw [hms]
  by_cases hb : a + n ≤ s.msize
  · rw [if_pos hb, if_pos hb]
    simp only [Option.map_some, Option.some.injEq, absSt]
    congr 1
    funext x
    by_cases hx : a ≤ x ∧ x < a + n
    · rw [if_pos hx, if_pos hx]; rfl
    · rw [if_neg hx, if_neg hx]
  · rw [if_neg hb, if_neg hb]; rfl

theorem absSt_ld_some {s : St} {a n v : Nat} {pm : Prov} (h : s.ld a n = some (v, pm)) :
    ∃ lpm, (absSt s).ld a n = some (v, lpm) ∧ pm.length % M = lpm % M := by
  simp only [St.ld, LSt.ld] at h ⊢
  have hms : (absSt s).msize = s.msize := rfl
  rw [hms]
  by_cases hb : a + n ≤ s.msize
  · rw [if_pos hb] at h
    rw [if_pos hb]
    simp only [Option.some.injEq, Prod.mk.injEq] at h
    obtain ⟨rfl, rfl⟩ := h
    exact ⟨_, rfl, foldr_prov_length (fun i => (s.mem (a + i)).p) (List.range n)⟩
  · rw [if_neg hb] at h; cases h

theorem absSt_ld_none {s : St} {a n : Nat} (h : s.ld a n = none) :
    (absSt s).ld a n = none := by
  simp only [St.ld, LSt.ld] at h ⊢
  have hms : (absSt s).msize = s.msize := rfl
  rw [hms]
  by_cases hb : a + n ≤ s.msize
  · rw [if_pos hb] at h; cases h
  · rw [if_neg hb]

theorem absSt_setTop (s : St) (f : Frame) :
    absSt (s.setTop f) = (absSt s).setTop (absFrame f) := by
  unfold St.setTop LSt.setTop
  cases h : s.rs with
  | nil => simp [absSt, h]
  | cons _ _ => simp [absSt, h]

/-! ### Simulation of `linit`, `lexecPrim`, `lstep`, and `lrun` -/

theorem linit_abs (pr : Prog) (msize : Nat) (inp : List Nat) :
    linit pr msize inp = absSt (Sporos.init pr msize inp) := by
  unfold linit Sporos.init absSt
  cases hm : pr.words[pr.main]? <;>
    simp only [absFrame, absVal, List.map, List.map_replicate, List.length_nil]
  all_goals
    congr 1
    funext a
    dsimp only [lloadData, loadData, absVal]
    cases pr.data.flatten[a]? <;> rfl

theorem linit_rel (pr : Prog) (msize : Nat) (inp : List Nat) :
    Rel (Sporos.init pr msize inp) (linit pr msize inp) := by
  rw [linit_abs]; exact absSt_rel _

private theorem lexecPrim_ld_abs (s : St) (a : Val) (k : Nat) (q : Prov)
    (hq : q.length % M = lq [absVal a] (absSt s).pc) :
    (match (absSt s).ld a.w k with
      | some (v, pm) => LPR.ok [⟨v, (lq [absVal a] (absSt s).pc + pm) % M⟩] (absSt s)
      | none => LPR.fail .bounds) =
    absPR (match s.ld a.w k with
      | some (v, pm) => PR.ok [⟨v, q ++ pm⟩] s
      | none => PR.fail .bounds) := by
  cases hld : s.ld a.w k with
  | none =>
    rw [absSt_ld_none hld]; rfl
  | some vp =>
    obtain ⟨v, pm⟩ := vp
    obtain ⟨lpm, hlpm, hmod⟩ := absSt_ld_some hld
    rw [hlpm]
    simp only [absPR, List.map]
    congr 3
    rw [Nat.add_mod, ← hq, ← hmod, Nat.mod_mod, ← Nat.add_mod]
    simp only [List.length_append]

private theorem absSt_tabs_len (s : St) : (absSt s).tabs.length = s.tabs.length :=
  List.length_map ..

private theorem absSt_tabs_get (s : St) (i : Nat) (hi : i < s.tabs.length)
    (hi' : i < (absSt s).tabs.length) :
    (absSt s).tabs[i]'hi' = s.tabs[i].map fun kv => (kv.1, absVal kv.2) := by
  simp [absSt]

theorem lexecPrim_abs (p : Prim) (args : List Val) (s : St) (o : Origin) :
    lexecPrim p (args.map absVal) (absSt s) = absPR (execPrim noPhys p args s o) := by
  have hq0 : lq [] (absSt s).pc = ([] ++ s.pc ++ [o]).length % M :=
    (q_length_mod [] s.pc o).symm
  have hq1 : ∀ a : Val,
      lq [absVal a] (absSt s).pc =
        (([a].foldr (fun (v : Val) acc => v.p ++ acc) []) ++ s.pc ++ [o]).length % M := by
    intro a; change lq ([a].map absVal) (s.pc.length % M) = _; exact (q_length_mod [a] s.pc o).symm
  have hq2 : ∀ a b : Val,
      lq [absVal a, absVal b] (absSt s).pc =
        (([a, b].foldr (fun (v : Val) acc => v.p ++ acc) []) ++ s.pc ++ [o]).length % M := by
    intro a b; change lq ([a, b].map absVal) (s.pc.length % M) = _; exact (q_length_mod [a, b] s.pc o).symm
  have hq3 : ∀ a b c : Val,
      lq [absVal a, absVal b, absVal c] (absSt s).pc =
        (([a, b, c].foldr (fun (v : Val) acc => v.p ++ acc) []) ++ s.pc ++ [o]).length % M := by
    intro a b c; change lq ([a, b, c].map absVal) (s.pc.length % M) = _; exact (q_length_mod [a, b, c] s.pc o).symm
  have hq4 : ∀ a b c d : Val,
      lq [absVal a, absVal b, absVal c, absVal d] (absSt s).pc =
        (([a, b, c, d].foldr (fun (v : Val) acc => v.p ++ acc) []) ++ s.pc ++ [o]).length % M := by
    intro a b c d; change lq ([a, b, c, d].map absVal) (s.pc.length % M) = _; exact (q_length_mod [a, b, c, d] s.pc o).symm
  cases p <;>
    rcases args with _ | ⟨a, _ | ⟨b, _ | ⟨c, _ | ⟨d, _ | ⟨e, _⟩⟩⟩⟩⟩ <;>
    try rfl
  · -- dup [a]
    simp only [lexecPrim, execPrim, List.map_cons, List.map_nil, absPR, absVal_tag]; rfl
  · -- swap [a, b]
    simp only [lexecPrim, execPrim, List.map_cons, List.map_nil, absPR, absVal_tag]; rfl
  · -- over [a, b]
    simp only [lexecPrim, execPrim, List.map_cons, List.map_nil, absPR, absVal_tag]; rfl
  · -- rot [a, b, c]
    simp only [lexecPrim, execPrim, List.map_cons, List.map_nil, absPR, absVal_tag]; rfl
  · -- nip [a, b]
    simp only [lexecPrim, execPrim, List.map_cons, List.map_nil, absPR, absVal_tag]; rfl
  · -- tuck [a, b]
    simp only [lexecPrim, execPrim, List.map_cons, List.map_nil, absPR, absVal_tag]; rfl
  · -- add
    simp only [lexecPrim, execPrim, List.map_cons, List.map_nil]
    rw [hq2]; simp only [absPR, List.map_cons, List.map_nil, absVal]
  · -- sub
    simp only [lexecPrim, execPrim, List.map_cons, List.map_nil]
    rw [hq2]; simp only [absPR, List.map_cons, List.map_nil, absVal]
  · -- mul
    simp only [lexecPrim, execPrim, List.map_cons, List.map_nil]
    rw [hq2]; simp only [absPR, List.map_cons, List.map_nil, absVal]
  · -- divu
    simp only [lexecPrim, execPrim, List.map_cons, List.map_nil]
    rw [hq2]; simp only [absVal]
    by_cases hz : a.w % M = 0 <;> simp [hz, absPR, absVal]
  · -- modu
    simp only [lexecPrim, execPrim, List.map_cons, List.map_nil]
    rw [hq2]; simp only [absVal]
    by_cases hz : a.w % M = 0 <;> simp [hz, absPR, absVal]
  · -- divs
    simp only [lexecPrim, execPrim, List.map_cons, List.map_nil]
    rw [hq2]; simp only [absVal]
    by_cases hz : a.w % M = 0 <;> simp [hz, absPR, absVal]
  · -- mods
    simp only [lexecPrim, execPrim, List.map_cons, List.map_nil]
    rw [hq2]; simp only [absVal]
    by_cases hz : a.w % M = 0 <;> simp [hz, absPR, absVal]
  · -- neg
    simp only [lexecPrim, execPrim, List.map_cons, List.map_nil]
    rw [hq1]; simp only [absPR, List.map_cons, List.map_nil, absVal]
  · -- band
    simp only [lexecPrim, execPrim, List.map_cons, List.map_nil]
    rw [hq2]; simp only [absPR, List.map_cons, List.map_nil, absVal]
  · -- bor
    simp only [lexecPrim, execPrim, List.map_cons, List.map_nil]
    rw [hq2]; simp only [absPR, List.map_cons, List.map_nil, absVal]
  · -- bxor
    simp only [lexecPrim, execPrim, List.map_cons, List.map_nil]
    rw [hq2]; simp only [absPR, List.map_cons, List.map_nil, absVal]
  · -- bnot
    simp only [lexecPrim, execPrim, List.map_cons, List.map_nil]
    rw [hq1]; simp only [absPR, List.map_cons, List.map_nil, absVal]
  · -- shl
    simp only [lexecPrim, execPrim, List.map_cons, List.map_nil]
    rw [hq2]; simp only [absPR, List.map_cons, List.map_nil, absVal]
  · -- shr
    simp only [lexecPrim, execPrim, List.map_cons, List.map_nil]
    rw [hq2]; simp only [absPR, List.map_cons, List.map_nil, absVal]
  · -- sar
    simp only [lexecPrim, execPrim, List.map_cons, List.map_nil]
    rw [hq2]; simp only [absPR, List.map_cons, List.map_nil, absVal]
  · -- eq
    simp only [lexecPrim, execPrim, List.map_cons, List.map_nil]
    rw [hq2]; simp only [absPR, List.map_cons, List.map_nil, absVal]; rfl
  · -- ne
    simp only [lexecPrim, execPrim, List.map_cons, List.map_nil]
    rw [hq2]; simp only [absPR, List.map_cons, List.map_nil, absVal]; rfl
  · -- ltu
    simp only [lexecPrim, execPrim, List.map_cons, List.map_nil]
    rw [hq2]; simp only [absPR, List.map_cons, List.map_nil, absVal]; rfl
  · -- lts
    simp only [lexecPrim, execPrim, List.map_cons, List.map_nil]
    rw [hq2]; simp only [absPR, List.map_cons, List.map_nil, absVal]; rfl
  · -- leu
    simp only [lexecPrim, execPrim, List.map_cons, List.map_nil]
    rw [hq2]; simp only [absPR, List.map_cons, List.map_nil, absVal]; rfl
  · -- les
    simp only [lexecPrim, execPrim, List.map_cons, List.map_nil]
    rw [hq2]; simp only [absPR, List.map_cons, List.map_nil, absVal]; rfl
  · -- ld8
    simp only [lexecPrim, execPrim, List.map_cons, List.map_nil]
    exact lexecPrim_ld_abs s a 1 _ (hq1 a).symm
  · -- st8
    simp only [lexecPrim, execPrim, List.map_cons, List.map_nil]
    rw [hq2]; simp only [absVal]; rw [absSt_st]
    cases s.st a.w 1 b.w _ <;> rfl
  · -- ld16
    simp only [lexecPrim, execPrim, List.map_cons, List.map_nil]
    exact lexecPrim_ld_abs s a 2 _ (hq1 a).symm
  · -- st16
    simp only [lexecPrim, execPrim, List.map_cons, List.map_nil]
    rw [hq2]; simp only [absVal]; rw [absSt_st]
    cases s.st a.w 2 b.w _ <;> rfl
  · -- ld32
    simp only [lexecPrim, execPrim, List.map_cons, List.map_nil]
    exact lexecPrim_ld_abs s a 4 _ (hq1 a).symm
  · -- st32
    simp only [lexecPrim, execPrim, List.map_cons, List.map_nil]
    rw [hq2]; simp only [absVal]; rw [absSt_st]
    cases s.st a.w 4 b.w _ <;> rfl
  · -- ld64
    simp only [lexecPrim, execPrim, List.map_cons, List.map_nil]
    exact lexecPrim_ld_abs s a 8 _ (hq1 a).symm
  · -- st64
    simp only [lexecPrim, execPrim, List.map_cons, List.map_nil]
    rw [hq2]; simp only [absVal]; rw [absSt_st]
    cases s.st a.w 8 b.w _ <;> rfl
  · -- copy [a, b, c] = [n, dst, src]
    simp only [lexecPrim, execPrim, List.map_cons, List.map_nil]
    rw [hq3]
    have hms : (absSt s).msize = s.msize := rfl
    have haw : (absVal a).w = a.w := rfl
    have hbw : (absVal b).w = b.w := rfl
    have hcw : (absVal c).w = c.w := rfl
    rw [hms, haw, hbw, hcw]
    by_cases hb : c.w + a.w ≤ s.msize ∧ b.w + a.w ≤ s.msize
    · rw [if_pos hb, if_pos hb]
      simp only [absPR, List.map_nil]
      congr 1
      simp only [absSt]
      congr 1
      funext x
      split
      · simp only [absVal, List.length_append]
        exact congrArg (LVal.mk _) (Nat.add_mod _ _ M).symm
      · rfl
    · rw [if_neg hb, if_neg hb]; rfl
  · -- fill [a, b, c] = [b, n, dst]
    simp only [lexecPrim, execPrim, List.map_cons, List.map_nil]
    rw [hq3]
    have hms : (absSt s).msize = s.msize := rfl
    have hbw : (absVal b).w = b.w := rfl
    have hcw : (absVal c).w = c.w := rfl
    rw [hms, hbw, hcw]
    by_cases hb : c.w + b.w ≤ s.msize
    · rw [if_pos hb, if_pos hb]
      simp only [absPR, List.map_nil]
      congr 1
      simp only [absSt]
      congr 1
      funext x
      split
      · simp only [absVal]
      · rfl
    · rw [if_neg hb, if_neg hb]; rfl
  · -- mcmp [a, b, c] = [n, y, x]
    simp only [lexecPrim, execPrim, List.map_cons, List.map_nil]
    rw [hq3]
    have hms : (absSt s).msize = s.msize := rfl
    have haw : (absVal a).w = a.w := rfl
    have hbw : (absVal b).w = b.w := rfl
    have hcw : (absVal c).w = c.w := rfl
    rw [hms, haw, hbw, hcw]
    by_cases hb : c.w + a.w ≤ s.msize ∧ b.w + a.w ≤ s.msize
    · rw [if_pos hb, if_pos hb]
      simp only [absPR, List.map_cons, List.map_nil, absVal, absSt, List.length_append]
      congr 3
      rw [Nat.add_mod, Nat.mod_mod, ← foldr_prov2_length, ← Nat.add_mod]
    · rw [if_neg hb, if_neg hb]; rfl
  · -- alloc [a]
    simp only [lexecPrim, execPrim, List.map_cons, List.map_nil]
    rw [hq1]
    have hms : (absSt s).msize = s.msize := rfl
    have hhp : (absSt s).hp = s.hp := rfl
    have haw : (absVal a).w = a.w := rfl
    rw [hms, hhp, haw]
    by_cases hb : s.hp + a.w ≤ s.msize
    · rw [if_pos hb, if_pos hb]
      simp only [absPR, List.map_cons, List.map_nil, absVal, absSt]
    · rw [if_neg hb, if_neg hb]; rfl
  · -- mark []
    simp only [lexecPrim, execPrim, List.map_nil]
    rw [hq0]; simp only [absPR, List.map_cons, List.map_nil, absVal, absSt]; rfl
  · -- release [a]
    simp only [lexecPrim, execPrim, List.map_cons, List.map_nil]
    have hhp : (absSt s).hp = s.hp := rfl
    have haw : (absVal a).w = a.w := rfl
    rw [hhp, haw]
    by_cases hb : a.w ≤ s.hp
    · rw [if_pos hb, if_pos hb]; rfl
    · rw [if_neg hb, if_neg hb]; rfl
  · -- streq [a, b, c, d] = [l2, a2, l1, a1]
    simp only [lexecPrim, execPrim, List.map_cons, List.map_nil]
    rw [hq4]
    have hms : (absSt s).msize = s.msize := rfl
    have haw : (absVal a).w = a.w := rfl
    have hbw : (absVal b).w = b.w := rfl
    have hcw : (absVal c).w = c.w := rfl
    have hdw : (absVal d).w = d.w := rfl
    rw [hms, haw, hbw, hcw, hdw]
    by_cases hb : d.w + c.w ≤ s.msize ∧ b.w + a.w ≤ s.msize
    · rw [if_pos hb, if_pos hb]
      simp only [absPR, List.map_cons, List.map_nil, absVal, absSt, List.length_append]
      congr 3
      by_cases hl : c.w = a.w
      · rw [if_pos hl, if_pos hl]
        rw [Nat.add_mod, Nat.mod_mod, ← foldr_prov2_length, ← Nat.add_mod]
      · rw [if_neg hl, if_neg hl]
        simp only [List.length_nil, Nat.add_zero, Nat.mod_mod]
    · rw [if_neg hb, if_neg hb]; rfl
  · -- strhash [a, b] = [l, a']
    simp only [lexecPrim, execPrim, List.map_cons, List.map_nil]
    rw [hq2]
    have hms : (absSt s).msize = s.msize := rfl
    have haw : (absVal a).w = a.w := rfl
    have hbw : (absVal b).w = b.w := rfl
    rw [hms, haw, hbw]
    by_cases hb : b.w + a.w ≤ s.msize
    · rw [if_pos hb, if_pos hb]
      simp only [absPR, List.map_cons, List.map_nil, absVal, absSt, List.length_append]
      congr 3
      rw [Nat.add_mod, Nat.mod_mod, ← foldr_prov_length, ← Nat.add_mod]
    · rw [if_neg hb, if_neg hb]; rfl
  · -- tnew []
    simp only [lexecPrim, execPrim, List.map_nil]
    rw [hq0]
    simp only [absPR, List.map_cons, List.map_nil, absVal, absSt, List.length_map, List.map_append]; rfl
  · -- tput [a, b, c] = [v, k, tb]
    simp only [lexecPrim, execPrim, List.map_cons, List.map_nil]
    rw [hq3]
    have htl : (absSt s).tabs.length = s.tabs.length := absSt_tabs_len s
    have hcw : (absVal c).w = c.w := rfl
    rw [hcw]
    by_cases hb : c.w < s.tabs.length
    · have hb' : c.w < (absSt s).tabs.length := by rw [htl]; exact hb
      rw [dif_pos hb', dif_pos hb]
      simp only [absPR, List.map_nil, absSt]
      congr 2
      rw [List.map_set]
      congr 1
      · simp only [List.getElem_map, List.map_cons, absVal, List.length_append]
        congr 3
        exact (Nat.add_mod _ _ M).symm
    · have hb' : ¬ c.w < (absSt s).tabs.length := by rw [htl]; exact hb
      rw [dif_neg hb', dif_neg hb]; rfl
  · -- tget [a, b] = [k, tb]
    simp only [lexecPrim, execPrim, List.map_cons, List.map_nil]
    rw [hq2]
    have htl : (absSt s).tabs.length = s.tabs.length := absSt_tabs_len s
    have hbw : (absVal b).w = b.w := rfl
    have haw : (absVal a).w = a.w := rfl
    rw [hbw, haw]
    by_cases hb : b.w < s.tabs.length
    · have hb' : b.w < (absSt s).tabs.length := by rw [htl]; exact hb
      rw [dif_pos hb', dif_pos hb]
      rw [absSt_tabs_get s b.w hb hb', ltfind_map]
      cases htf : tfind s.tabs[b.w] a.w with
      | none =>
        simp only [Option.map_none, absPR, List.map_cons, List.map_nil, absVal]
      | some v =>
        simp only [Option.map_some, absPR, List.map_cons, List.map_nil, absVal, List.length_append]
        congr 4
        exact (Nat.add_mod _ _ M).symm
    · have hb' : ¬ b.w < (absSt s).tabs.length := by rw [htl]; exact hb
      rw [dif_neg hb', dif_neg hb]; rfl
  · -- getb []
    simp only [lexecPrim, execPrim, List.map_nil]
    rw [hq0]
    have hinp : (absSt s).inp = s.inp := rfl
    cases hc : s.inp with
    | nil =>
      have hinp' : (absSt s).inp = [] := by rw [hinp, hc]
      rw [hinp']
      simp only [absPR, List.map_cons, List.map_nil, absVal, absSt, hc]; rfl
    | cons x rest =>
      have hinp' : (absSt s).inp = x :: rest := by rw [hinp, hc]
      rw [hinp']
      simp only [absPR, List.map_cons, List.map_nil, absVal, absSt, List.length_append, List.length_singleton]
      congr 3
      exact Nat.mod_add_mod _ _ _
  · -- clock []
    simp only [lexecPrim, execPrim, List.map_nil, noPhys]
    rw [hq0]; simp only [absPR, List.map_cons, List.map_nil, absVal, absSt]; rfl
  · -- core []
    simp only [lexecPrim, execPrim, List.map_nil, noPhys]
    rw [hq0]; simp only [absPR, List.map_cons, List.map_nil, absVal, absSt]; rfl
  · -- sensor [a]
    simp only [lexecPrim, execPrim, List.map_cons, List.map_nil, noPhys]
    rw [hq1]; simp only [absPR, List.map_cons, List.map_nil, absVal, absSt]
  · -- provget [a]
    simp only [lexecPrim, execPrim, List.map_cons, List.map_nil]
    rw [hq1]; simp only [absPR, List.map_cons, List.map_nil, absVal_tag]
    simp only [absVal, wrap, Nat.mod_mod]; rfl
  · -- confget [a]
    simp only [lexecPrim, execPrim, List.map_cons, List.map_nil, noPhys]
    rw [hq1]; simp only [absPR, List.map_cons, List.map_nil, absVal_tag]
    simp only [absVal]; rfl
  · -- demand [a, b] = [th, a']
    simp only [lexecPrim, execPrim, List.map_cons, List.map_nil]
    have haw : (absVal a).w = a.w := rfl
    have hconf : noPhys.conf b.p = 0 := rfl
    rw [haw, hconf]
    by_cases hd : 0 < a.w
    · rw [if_pos hd, if_pos hd]; rfl
    · rw [if_neg hd, if_neg hd]
      simp only [absPR, List.map_cons, List.map_nil, absVal_tag]; rfl

theorem lstep_abs (pr : Prog) (s : St) :
    lstep pr (absSt s) = absRes (step pr noPhys s) := by
  rcases s with ⟨ds, aux, rs, mem, msize, hp, tabs, inp, ipos, out, pc, stp⟩
  unfold lstep step
  cases hrs : rs with
  | nil => simp [absSt, absRes]
  | cons f rest =>
    subst hrs
    simp only [absSt, List.map_cons]
    have hfword : (absFrame f).word = f.word := rfl
    have hfpc : (absFrame f).pc = f.pc := rfl
    rw [hfword, hfpc]
    cases hf : fetch pr f.word f.pc with
    | none => simp [absRes]
    | some ins =>
      dsimp only
      cases ins with
      | prim p =>
        dsimp only
        have hlen : (ds.map absVal).length = ds.length := List.length_map ..
        rw [hlen]
        by_cases hk : ds.length < p.eff.1
        · rw [if_pos hk, if_pos hk]; rfl
        · rw [if_neg hk, if_neg hk]
          have htake : (ds.map absVal).take p.eff.1 = (ds.take p.eff.1).map absVal := by
            rw [List.map_take]
          rw [htake]
          have hprim := lexecPrim_abs p (ds.take p.eff.1)
            ⟨ds, aux, f :: rest, mem, msize, hp, tabs, inp, ipos, out, pc, stp + 1⟩
            ⟨none, f.word, f.pc, noPhys.core stp⟩
          have hst : absSt ⟨ds, aux, f :: rest, mem, msize, hp, tabs, inp, ipos, out, pc, stp + 1⟩ =
              { ds := ds.map absVal, aux := aux.map absVal, rs := absFrame f :: rest.map absFrame,
                mem := fun a => absVal (mem a), msize := msize, hp := hp,
                tabs := tabs.map (List.map fun kv => (kv.1, absVal kv.2)),
                inp := inp, ipos := ipos, out := out.map Prod.fst, pc := pc.length % M } := by
            simp [absSt]
          rw [hst] at hprim
          rw [hprim]
          cases execPrim noPhys p (ds.take p.eff.1)
              ⟨ds, aux, f :: rest, mem, msize, hp, tabs, inp, ipos, out, pc, stp + 1⟩
              ⟨none, f.word, f.pc, noPhys.core stp⟩ with
          | ok res s' =>
            simp only [absPR, absRes]
            have hds : absSt { s' with ds := res ++ ds.drop p.eff.1 } =
                { absSt s' with ds := res.map absVal ++ (ds.map absVal).drop p.eff.1 } := by
              simp [absSt, List.map_append, List.map_drop]
            rw [absSt_setTop, hds]
            rfl
          | halt c => simp [absPR, absRes]
          | fail e => simp [absPR, absRes]
      | lit n =>
        dsimp only
        simp only [LSt.setTop, St.setTop, absRes, absSt, absFrame, absVal,
          List.map_cons, List.length_append, List.length_singleton]
        congr 4
        rw [Nat.mod_add_mod]
      | str i =>
        dsimp only
        cases hdi : pr.data[i]? with
        | none => rfl
        | some d =>
          dsimp only
          simp only [LSt.setTop, St.setTop, absRes, absSt, absFrame, absVal,
            List.map_cons, List.length_append, List.length_singleton]
          congr 4 <;> rw [Nat.mod_add_mod]
      | pick n =>
        dsimp only
        rw [List.getElem?_map]
        cases hdn : ds[n]? with
        | none => rfl
        | some v =>
          dsimp only
          simp only [Option.map_some, LSt.setTop, St.setTop, absRes, absSt, absFrame,
            List.map_cons, absVal_tag]
      | tor =>
        dsimp only
        cases hds : ds with
        | nil => simp [absRes]
        | cons v r =>
          simp only [LSt.setTop, St.setTop, absRes, absSt, absFrame, List.map_cons, absVal_tag]
      | fromr =>
        dsimp only
        cases hax : aux with
        | nil => simp [absRes]
        | cons v r =>
          simp only [LSt.setTop, St.setTop, absRes, absSt, absFrame, List.map_cons, absVal_tag]
      | rfetch =>
        dsimp only
        cases hax : aux with
        | nil => simp [absRes]
        | cons v r =>
          simp only [LSt.setTop, St.setTop, absRes, absSt, absFrame, List.map_cons, absVal_tag]
      | locget i =>
        dsimp only
        have hloc : (absFrame f).locs[i]? = f.locs[i]?.map absVal := by simp [absFrame, List.getElem?_map]
        rw [hloc]
        cases hli : f.locs[i]? with
        | none => rfl
        | some v =>
          dsimp only
          simp only [Option.map_some, LSt.setTop, St.setTop, absRes, absSt, absFrame,
            List.map_cons, absVal_tag]
      | locset i =>
        dsimp only
        cases hds : ds with
        | nil => simp [absRes]
        | cons v r =>
          have hllen : (absFrame f).locs.length = f.locs.length := List.length_map ..
          simp only [List.map_cons, hllen]
          by_cases hi : i < f.locs.length
          · rw [if_pos hi, if_pos hi]
            simp only [LSt.setTop, St.setTop, absRes, absSt, absFrame, List.map_cons,
              List.map_set, absVal_tag]
          · rw [if_neg hi, if_neg hi]; rfl
      | jmp t =>
        dsimp only
        simp only [LSt.setTop, St.setTop, absRes, absSt, absFrame, List.map_cons]
      | jz t =>
        dsimp only
        cases hds : ds with
        | nil => simp [absRes]
        | cons c r =>
          simp only [LSt.setTop, St.setTop, absRes, absSt, absFrame, absVal,
            List.map_cons, List.length_append, List.length_singleton]
          congr 2
          exact (mod_add3 pc.length c.p.length 1).symm
      | jnz t =>
        dsimp only
        cases hds : ds with
        | nil => simp [absRes]
        | cons c r =>
          simp only [LSt.setTop, St.setTop, absRes, absSt, absFrame, absVal,
            List.map_cons, List.length_append, List.length_singleton]
          congr 2
          exact (mod_add3 pc.length c.p.length 1).symm
      | call w =>
        dsimp only
        cases hw : pr.words[w]? with
        | none => rfl
        | some wd =>
          dsimp only
          have hlen : (ds.map absVal).length = ds.length := List.length_map ..
          rw [hlen]
          by_cases hl : ds.length < wd.ins
          · rw [if_pos hl, if_pos hl]; rfl
          · rw [if_neg hl, if_neg hl]
            simp only [absRes, absSt, List.map_cons, absFrame, List.map_replicate, absVal,
              List.length_append, List.length_singleton, List.length_map, Nat.mod_add_mod]
      | ret =>
        dsimp only
        cases rest with
        | nil => simp [absRes]
        | cons g r' => simp [absRes, absSt]

/-- Universal single-step bisimulation: every `Sporos.step pr Sporos.noPhys s`
matches `lstep pr ls` under `Rel`. -/
theorem lstep_sim (pr : Prog) {s : St} {ls : LSt} (h : Rel s ls) :
    ResRel (step pr noPhys s) (lstep pr ls) := by
  rw [h.eq_abs, lstep_abs]
  exact absRes_rel _

theorem lrun_abs (pr : Prog) : ∀ (n : Nat) (s : St),
    lrun pr n (absSt s) = absRes (run pr noPhys n s)
  | 0, _ => rfl
  | n + 1, s => by
    simp only [lrun, run, lstep_abs]
    cases hs : step pr noPhys s with
    | next t => exact lrun_abs pr n t
    | halt c out => rfl
    | fail e => rfl

/-- Universal multi-step bisimulation from any related pair of states. -/
theorem lrun_sim (pr : Prog) (n : Nat) {s : St} {ls : LSt} (h : Rel s ls) :
    ResRel (run pr noPhys n s) (lrun pr n ls) := by
  rw [h.eq_abs, lrun_abs]
  exact absRes_rel _

/-- Universal whole-run bisimulation from initial states. -/
theorem lrun_sim_init (pr : Prog) (msize : Nat) (inp : List Nat) (n : Nat) :
    ResRel (run pr noPhys n (Sporos.init pr msize inp)) (lrun pr n (linit pr msize inp)) :=
  lrun_sim pr n (linit_rel pr msize inp)

theorem lrun_sim_halt (pr : Prog) (msize : Nat) (inp : List Nat) (n code : Nat) (bytes : List Nat) :
    lrun pr n (linit pr msize inp) = .halt code bytes ↔
      ∃ out, run pr noPhys n (Sporos.init pr msize inp) = .halt code out ∧ out.map Prod.fst = bytes := by
  rw [linit_abs, lrun_abs]
  cases run pr noPhys n (Sporos.init pr msize inp) with
  | next _ => simp [absRes]
  | halt c out =>
    simp only [absRes, LRes.halt.injEq]
    constructor
    · intro ⟨hc, ho⟩; subst hc; exact ⟨out, rfl, ho⟩
    · intro ⟨_, h1, ho⟩; cases h1; exact ⟨rfl, ho⟩
  | fail _ => simp [absRes]

theorem lrun_sim_fail (pr : Prog) (msize : Nat) (inp : List Nat) (n : Nat) (err : Err) :
    lrun pr n (linit pr msize inp) = .fail err ↔
      run pr noPhys n (Sporos.init pr msize inp) = .fail err := by
  rw [linit_abs, lrun_abs]
  cases run pr noPhys n (Sporos.init pr msize inp) <;> simp [absRes]

theorem lrun_sim_next (pr : Prog) (msize : Nat) (inp : List Nat) (n : Nat) :
    (∃ ls', lrun pr n (linit pr msize inp) = .next ls') ↔
      (∃ s', run pr noPhys n (Sporos.init pr msize inp) = .next s') := by
  rw [linit_abs, lrun_abs]
  cases hr : run pr noPhys n (Sporos.init pr msize inp) with
  | next s' => exact ⟨fun _ => ⟨s', rfl⟩, fun _ => ⟨absSt s', rfl⟩⟩
  | halt _ _ => simp [absRes]
  | fail _ => simp [absRes]

/-- Composition with `Sporos.run_safe`: any checked program never hits a static
checker error (`underflow`, `auxUnderflow`, `badLocal`, `badJump`, `badWord`, `noFrame`)
in `lrun`, on any input and memory size. -/
theorem lrun_safe (pr : Prog) (hp : checkProg pr = true) (msize : Nat) (inp : List Nat)
    (n : Nat) (e : Err) (h : lrun pr n (linit pr msize inp) = .fail e) : e.static = false :=
  Sporos.run_safe pr noPhys hp msize inp n e ((lrun_sim_fail pr msize inp n e).mp h)

/-! ### 64-bit word and provenance-length invariants -/

/-- Every `LVal` in `s` has `w < 2^64` and `plen < 2^64`, and `s.pc < 2^64`. -/
def LSt.WordsLt (s : LSt) : Prop :=
  (∀ v ∈ s.ds, v.w < M ∧ v.plen < M) ∧
  (∀ v ∈ s.aux, v.w < M ∧ v.plen < M) ∧
  (∀ f ∈ s.rs, ∀ v ∈ f.locs, v.w < M ∧ v.plen < M) ∧
  (∀ a, (s.mem a).w < M ∧ (s.mem a).plen < M) ∧
  (∀ tb ∈ s.tabs, ∀ kv ∈ tb, kv.2.w < M ∧ kv.2.plen < M) ∧
  s.pc < M

theorem linit_wordsLt (pr : Prog) (msize : Nat) (inp : List Nat) :
    (linit pr msize inp).WordsLt := by
  refine ⟨by simp [linit], by simp [linit], ?_, ?_, by simp [linit], by simp [linit, M_pos]⟩
  · intro f hf v hv
    simp only [linit] at hf
    cases hm : pr.words[pr.main]? <;> rw [hm] at hf <;>
      simp only [List.mem_cons, List.not_mem_nil, or_false] at hf <;> subst hf <;>
      simp only [List.mem_replicate] at hv <;> rw [hv.2] <;> exact ⟨M_pos, M_pos⟩
  · intro a
    simp only [linit, lloadData]
    split
    · exact ⟨Nat.lt_trans (Nat.mod_lt _ (by decide)) (by decide), M_pos⟩
    · exact ⟨M_pos, M_pos⟩

private theorem foldr_bytes_lt_aux (f : Nat → Nat) (l : List Nat) :
    l.foldr (fun i acc => acc * 256 + f i % 256) 0 < 256 ^ l.length := by
  induction l with
  | nil => simp
  | cons i r ih =>
    simp only [List.foldr_cons, List.length_cons, Nat.pow_succ]
    have hm : f i % 256 < 256 := Nat.mod_lt _ (by decide)
    omega

theorem LSt.ld_lt {s : LSt} {a n v pm : Nat} (hn : n ≤ 8) (h : s.ld a n = some (v, pm)) :
    v < M := by
  unfold LSt.ld at h
  split at h
  · simp only [Option.some.injEq, Prod.mk.injEq] at h
    obtain ⟨rfl, -⟩ := h
    have hlt := foldr_bytes_lt_aux (fun i => (s.mem (a + i)).w) (List.range n)
    rw [List.length_range] at hlt
    have hp : 256 ^ n ≤ M := by
      have : 256 ^ n ≤ 256 ^ 8 := Nat.pow_le_pow_right (by decide) hn
      exact Nat.le_trans this (by decide)
    exact Nat.lt_of_lt_of_le hlt hp
  · cases h

theorem ltfind_mem : ∀ (l : List (Nat × LVal)) (k : Nat) {v : LVal},
    ltfind l k = some v → ∃ kv ∈ l, kv.2 = v
  | [], _, _, h => by simp [ltfind] at h
  | (k', v') :: r, k, v, h => by
    simp only [ltfind] at h
    split at h
    · exact ⟨(k', v'), by simp, Option.some.inj h⟩
    · obtain ⟨kv, hkv, he⟩ := ltfind_mem r k h
      exact ⟨kv, by simp [hkv], he⟩

private theorem str_addr_le (data : List (List Nat)) (i : Nat) :
    ((data.take i).map List.length).foldl (· + ·) 0 ≤ data.flatten.length := by
  have hfold : ∀ (l : List (List Nat)) (acc : Nat),
      (l.map List.length).foldl (· + ·) acc = acc + l.flatten.length := by
    intro l
    induction l with
    | nil => intro acc; simp
    | cons x xs ih =>
      intro acc
      simp only [List.map_cons, List.foldl_cons, List.flatten_cons, List.length_append]
      rw [ih]
      omega
  rw [hfold, Nat.zero_add]
  induction data generalizing i with
  | nil => simp
  | cons x xs ih =>
    cases i with
    | zero => simp
    | succ i =>
      simp only [List.take_succ_cons, List.flatten_cons, List.length_append]
      exact Nat.add_le_add_left (ih i) x.length

private theorem str_len_le {data : List (List Nat)} {i : Nat} {d : List Nat}
    (h : data[i]? = some d) : d.length ≤ data.flatten.length := by
  induction data generalizing i with
  | nil => simp at h
  | cons x xs ih =>
    cases i with
    | zero => simp only [List.getElem?_cons_zero, Option.some.injEq] at h; subst h; simp
    | succ i =>
      simp only [List.getElem?_cons_succ] at h
      simp only [List.flatten_cons, List.length_append]
      exact Nat.le_trans (ih h) (Nat.le_add_left _ _)

private theorem bool_lt_M (b : Bool) : Sporos.bool b < M := by
  cases b <;> decide

private theorem forall_mem_nil {α : Type} {P : α → Prop} : ∀ v ∈ ([] : List α), P v := by
  intro _ h; cases h

private theorem forall_mem_one {α : Type} {P : α → Prop} {a : α} (ha : P a) :
    ∀ v ∈ [a], P v := by
  intro v h; simp at h; subst h; exact ha

private theorem forall_mem_two {α : Type} {P : α → Prop} {a b : α} (ha : P a) (hb : P b) :
    ∀ v ∈ [a, b], P v := by
  intro v h; simp at h; rcases h with rfl | rfl <;> assumption

private theorem forall_mem_three {α : Type} {P : α → Prop} {a b c : α}
    (ha : P a) (hb : P b) (hc : P c) : ∀ v ∈ [a, b, c], P v := by
  intro v h; simp at h; rcases h with rfl | rfl | rfl <;> assumption

private theorem ld_lt_aux {s : LSt} {a n v pm : Nat}
    (h : s.ld a n = some (v, pm)) (hn : n ≤ 8) : v < M :=
  LSt.ld_lt hn h

private theorem ltfind_lt {s : LSt} {i k : Nat} {v : LVal}
    (htab : ∀ tb ∈ s.tabs, ∀ kv ∈ tb, kv.2.w < M ∧ kv.2.plen < M)
    (hi : i < s.tabs.length) (h : ltfind s.tabs[i] k = some v) : v.w < M := by
  obtain ⟨kv, hkv, rfl⟩ := ltfind_mem _ _ h
  exact (htab _ (List.getElem_mem hi) kv hkv).1

private theorem lexecPrim_wordsLt {p : Prim} {args : List LVal} {s s' : LSt} {res : List LVal}
    (hs : s.WordsLt) (ha : ∀ v ∈ args, v.w < M ∧ v.plen < M)
    (h : lexecPrim p args s = .ok res s') :
    (∀ v ∈ res, v.w < M ∧ v.plen < M) ∧ s'.WordsLt := by
  have htag : ∀ v, v.w < M → (ltag s.pc v).w < M ∧ (ltag s.pc v).plen < M :=
    fun v hv => ⟨hv, Nat.mod_lt _ M_pos⟩
  obtain ⟨hds, haux, hrs, hmem, htab, hpc⟩ := hs
  cases p <;>
    rcases args with _ | ⟨a, _ | ⟨b, _ | ⟨c, _ | ⟨d, _ | ⟨e, _⟩⟩⟩⟩⟩ <;>
    simp only [lexecPrim] at h <;> (repeat' split at h) <;> (try cases h) <;>
    (try (rename_i hst; unfold LSt.st at hst; split at hst <;> cases hst))
  all_goals
    (refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
     · first
         | exact forall_mem_nil
         | ((first | apply forall_mem_one | apply forall_mem_two | apply forall_mem_three) <;>
            first
              | exact ⟨Nat.mod_lt _ M_pos, Nat.mod_lt _ M_pos⟩
              | (apply htag; apply (ha _ _).1; simp)
              | exact ⟨bool_lt_M _, Nat.mod_lt _ M_pos⟩
              | (refine ⟨Nat.lt_trans (Nat.mod_lt _ ?_) ?_, Nat.mod_lt _ M_pos⟩ <;> decide)
              | (refine ⟨?_, Nat.mod_lt _ M_pos⟩; rename_i heq; apply ld_lt_aux heq; decide)
              | (refine ⟨?_, Nat.mod_lt _ M_pos⟩; rename_i hb _ _ heq; have hlt := ltfind_lt htab hb heq; exact hlt))
     · exact hds
     · exact haux
     · exact hrs
     · first
         | exact hmem
         | (intro x; dsimp only; split
            · first
                | (refine ⟨Nat.lt_trans (Nat.mod_lt _ ?_) ?_, Nat.mod_lt _ M_pos⟩ <;> decide)
                | exact ⟨(hmem _).1, Nat.mod_lt _ M_pos⟩
            · exact hmem x)
     · first
         | exact htab
         | (intro tb htb kv hkv
            simp only [List.mem_append, List.mem_cons, List.not_mem_nil, or_false] at htb
            rcases htb with htb | rfl
            · exact htab tb htb kv hkv
            · simp at hkv)
         | (intro tb htb kv hkv
            rcases List.mem_or_eq_of_mem_set htb with htb | rfl
            · exact htab tb htb kv hkv
            · simp only [List.mem_cons] at hkv
              rcases hkv with rfl | hkv
              · exact ⟨(ha a (by simp)).1, Nat.mod_lt _ M_pos⟩
              · exact htab _ (List.getElem_mem _) kv hkv)
     · exact hpc)

private theorem wordsLt_setTop {s : LSt} {f : LFrame} (hs : s.WordsLt)
    (hf : ∀ v ∈ f.locs, v.w < M ∧ v.plen < M) : (s.setTop f).WordsLt := by
  obtain ⟨hds, haux, hrs, hmem, htab, hpc⟩ := hs
  unfold LSt.setTop
  cases hr : s.rs with
  | nil => exact ⟨hds, haux, hrs, hmem, htab, hpc⟩
  | cons _ r =>
    refine ⟨hds, haux, ?_, hmem, htab, hpc⟩
    intro g hg v hv
    simp only [List.mem_cons] at hg
    rcases hg with rfl | hg
    · exact hf v hv
    · exact hrs g (by rw [hr]; simp [hg]) v hv

private theorem lexecPrim_rs {p : Prim} {args : List LVal} {s s' : LSt} {res : List LVal}
    (h : lexecPrim p args s = .ok res s') : s'.rs = s.rs := by
  cases p <;>
    rcases args with _ | ⟨a, _ | ⟨b, _ | ⟨c, _ | ⟨d, _ | ⟨e, _⟩⟩⟩⟩⟩ <;>
    simp only [lexecPrim, LSt.st] at h <;> (repeat' split at h) <;>
    (try (cases h <;> rfl)) <;>
    (rename_i heq; split at heq <;> cases heq <;> cases h <;> rfl)

theorem lstep_wordsLt {pr : Prog} (hdata : pr.data.flatten.length < M) {s s' : LSt}
    (hs : s.WordsLt) (h : lstep pr s = .next s') : s'.WordsLt := by
  unfold lstep at h
  cases hrs0 : s.rs with
  | nil => rw [hrs0] at h; cases h
  | cons f rest =>
    rw [hrs0] at h
    dsimp only at h
    cases hf0 : fetch pr f.word f.pc with
    | none => rw [hf0] at h; cases h
    | some ins =>
      rw [hf0] at h
      dsimp only at h
      have hf : ∀ v ∈ f.locs, v.w < M ∧ v.plen < M := hs.2.2.1 f (by rw [hrs0]; simp)
      cases ins with
      | prim p =>
        dsimp only at h
        split at h
        · cases h
        · cases hxp : lexecPrim p (s.ds.take p.eff.1) s with
          | halt _ => rw [hxp] at h; cases h
          | fail _ => rw [hxp] at h; cases h
          | ok res s1 =>
            rw [hxp] at h
            simp only [LRes.next.injEq] at h; subst h
            obtain ⟨hres, hs1⟩ := lexecPrim_wordsLt hs
              (fun v hv => hs.1 v (List.mem_of_mem_take hv)) hxp
            have hrs1 : s1.rs = f :: rest := (lexecPrim_rs hxp).trans hrs0
            have hf1 : ∀ v ∈ f.locs, v.w < M ∧ v.plen < M := hs1.2.2.1 f (by rw [hrs1]; simp)
            refine wordsLt_setTop ?_ (by simpa using hf1)
            refine ⟨?_, hs1.2.1, hs1.2.2.1, hs1.2.2.2.1, hs1.2.2.2.2.1, hs1.2.2.2.2.2⟩
            intro v hv
            rcases List.mem_append.1 hv with hv | hv
            · exact hres v hv
            · exact hs.1 v (List.mem_of_mem_drop hv)
      | lit n =>
        simp only [LRes.next.injEq] at h; subst h
        refine wordsLt_setTop ?_ (by simpa using hf)
        refine ⟨?_, hs.2.1, fun g hg => hs.2.2.1 g (by rw [hrs0]; exact hg), hs.2.2.2.1, hs.2.2.2.2.1, hs.2.2.2.2.2⟩
        intro v hv
        simp only [List.mem_cons] at hv
        rcases hv with rfl | hv
        · exact ⟨Nat.mod_lt _ M_pos, Nat.mod_lt _ M_pos⟩
        · exact hs.1 v hv
      | str i =>
        dsimp only at h
        cases hdi : pr.data[i]? with
        | none => rw [hdi] at h; cases h
        | some d =>
          rw [hdi] at h
          simp only [LRes.next.injEq] at h; subst h
          refine wordsLt_setTop ?_ (by simpa using hf)
          refine ⟨?_, hs.2.1, fun g hg => hs.2.2.1 g (by rw [hrs0]; exact hg), hs.2.2.2.1, hs.2.2.2.2.1, hs.2.2.2.2.2⟩
          intro v hv
          simp only [List.mem_cons] at hv
          rcases hv with rfl | rfl | hv
          · exact ⟨Nat.lt_of_le_of_lt (str_len_le hdi) hdata, Nat.mod_lt _ M_pos⟩
          · exact ⟨Nat.lt_of_le_of_lt (str_addr_le pr.data i) hdata, Nat.mod_lt _ M_pos⟩
          · exact hs.1 v hv
      | pick n =>
        dsimp only at h
        cases hdn : s.ds[n]? with
        | none => rw [hdn] at h; cases h
        | some u =>
          rw [hdn] at h
          simp only [LRes.next.injEq] at h; subst h
          refine wordsLt_setTop ?_ (by simpa using hf)
          refine ⟨?_, hs.2.1, fun g hg => hs.2.2.1 g (by rw [hrs0]; exact hg), hs.2.2.2.1, hs.2.2.2.2.1, hs.2.2.2.2.2⟩
          intro v hv
          simp only [List.mem_cons] at hv
          rcases hv with rfl | hv
          · exact ⟨(hs.1 u (List.mem_of_getElem? hdn)).1, Nat.mod_lt _ M_pos⟩
          · exact hs.1 v hv
      | tor =>
        dsimp only at h
        cases hds : s.ds with
        | nil => rw [hds] at h; cases h
        | cons u r =>
          rw [hds] at h
          simp only [LRes.next.injEq] at h; subst h
          refine wordsLt_setTop ?_ (by simpa using hf)
          refine ⟨fun v hv => hs.1 v (by rw [hds]; simp [hv]), ?_, fun g hg => hs.2.2.1 g (by rw [hrs0]; exact hg), hs.2.2.2.1, hs.2.2.2.2.1, hs.2.2.2.2.2⟩
          intro v hv
          simp only [List.mem_cons] at hv
          rcases hv with rfl | hv
          · exact ⟨(hs.1 u (by rw [hds]; simp)).1, Nat.mod_lt _ M_pos⟩
          · exact hs.2.1 v hv
      | fromr =>
        dsimp only at h
        cases hax : s.aux with
        | nil => rw [hax] at h; cases h
        | cons u r =>
          rw [hax] at h
          simp only [LRes.next.injEq] at h; subst h
          refine wordsLt_setTop ?_ (by simpa using hf)
          refine ⟨?_, fun v hv => hs.2.1 v (by rw [hax]; simp [hv]), fun g hg => hs.2.2.1 g (by rw [hrs0]; exact hg), hs.2.2.2.1, hs.2.2.2.2.1, hs.2.2.2.2.2⟩
          intro v hv
          simp only [List.mem_cons] at hv
          rcases hv with rfl | hv
          · exact ⟨(hs.2.1 u (by rw [hax]; simp)).1, Nat.mod_lt _ M_pos⟩
          · exact hs.1 v hv
      | rfetch =>
        dsimp only at h
        cases hax : s.aux with
        | nil => rw [hax] at h; cases h
        | cons u r =>
          rw [hax] at h
          simp only [LRes.next.injEq] at h; subst h
          refine wordsLt_setTop ?_ (by simpa using hf)
          refine ⟨?_, fun v hv => hs.2.1 v (by rw [hax]; exact hv), fun g hg => hs.2.2.1 g (by rw [hrs0]; exact hg), hs.2.2.2.1, hs.2.2.2.2.1, hs.2.2.2.2.2⟩
          intro v hv
          simp only [List.mem_cons] at hv
          rcases hv with rfl | hv
          · exact ⟨(hs.2.1 u (by rw [hax]; simp)).1, Nat.mod_lt _ M_pos⟩
          · exact hs.1 v hv
      | locget i =>
        dsimp only at h
        cases hli : f.locs[i]? with
        | none => rw [hli] at h; cases h
        | some u =>
          rw [hli] at h
          simp only [LRes.next.injEq] at h; subst h
          refine wordsLt_setTop ?_ (by simpa using hf)
          refine ⟨?_, hs.2.1, fun g hg => hs.2.2.1 g (by rw [hrs0]; exact hg), hs.2.2.2.1, hs.2.2.2.2.1, hs.2.2.2.2.2⟩
          intro v hv
          simp only [List.mem_cons] at hv
          rcases hv with rfl | hv
          · exact ⟨(hf u (List.mem_of_getElem? hli)).1, Nat.mod_lt _ M_pos⟩
          · exact hs.1 v hv
      | locset i =>
        dsimp only at h
        cases hds : s.ds with
        | nil => rw [hds] at h; cases h
        | cons u r =>
          rw [hds] at h
          dsimp only at h
          split at h
          · simp only [LRes.next.injEq] at h; subst h
            refine wordsLt_setTop
              ⟨fun v hv => hs.1 v (by rw [hds]; exact List.mem_cons_of_mem _ hv),
               hs.2.1, fun g hg => hs.2.2.1 g (by rw [hrs0]; exact hg), hs.2.2.2.1, hs.2.2.2.2.1, hs.2.2.2.2.2⟩ ?_
            intro v hv
            rcases List.mem_or_eq_of_mem_set hv with hv | rfl
            · exact hf v hv
            · exact ⟨(hs.1 u (by rw [hds]; simp)).1, Nat.mod_lt _ M_pos⟩
          · cases h
      | jmp t =>
        simp only [LRes.next.injEq] at h; subst h
        exact wordsLt_setTop hs hf
      | jz t =>
        dsimp only at h
        cases hds : s.ds with
        | nil => rw [hds] at h; cases h
        | cons c r =>
          rw [hds] at h
          simp only [LRes.next.injEq] at h; subst h
          exact wordsLt_setTop ⟨fun v hv => hs.1 v (by rw [hds]; exact List.mem_cons_of_mem _ hv),
            hs.2.1, fun g hg => hs.2.2.1 g (by rw [hrs0]; exact hg), hs.2.2.2.1, hs.2.2.2.2.1, Nat.mod_lt _ M_pos⟩ hf
      | jnz t =>
        dsimp only at h
        cases hds : s.ds with
        | nil => rw [hds] at h; cases h
        | cons c r =>
          rw [hds] at h
          simp only [LRes.next.injEq] at h; subst h
          exact wordsLt_setTop ⟨fun v hv => hs.1 v (by rw [hds]; exact List.mem_cons_of_mem _ hv),
            hs.2.1, fun g hg => hs.2.2.1 g (by rw [hrs0]; exact hg), hs.2.2.2.1, hs.2.2.2.2.1, Nat.mod_lt _ M_pos⟩ hf
      | call w =>
        dsimp only at h
        cases hw : pr.words[w]? with
        | none => rw [hw] at h; cases h
        | some wd =>
          rw [hw] at h
          dsimp only at h
          split at h
          · cases h
          · simp only [LRes.next.injEq] at h; subst h
            refine ⟨hs.1, hs.2.1, ?_, hs.2.2.2.1, hs.2.2.2.2.1, hs.2.2.2.2.2⟩
            intro g hg v hv
            simp only [List.mem_cons] at hg
            rcases hg with rfl | rfl | hg
            · simp only [List.mem_replicate] at hv
              rw [hv.2]; exact ⟨M_pos, Nat.mod_lt _ M_pos⟩
            · exact hf v hv
            · exact hs.2.2.1 g (by rw [hrs0]; simp [hg]) v hv
      | ret =>
        dsimp only at h
        cases rest with
        | nil => cases h
        | cons g r' =>
          simp only [LRes.next.injEq] at h; subst h
          exact ⟨hs.1, hs.2.1, fun f' hf' => hs.2.2.1 f' (by rw [hrs0]; simp [hf']),
            hs.2.2.2.1, hs.2.2.2.2.1, hs.2.2.2.2.2⟩

theorem lrun_wordsLt_from {pr : Prog} (hdata : pr.data.flatten.length < M) :
    ∀ (n : Nat) {s s' : LSt}, s.WordsLt → lrun pr n s = .next s' → s'.WordsLt
  | 0, _, _, hs, h => by simp [lrun] at h; subst h; exact hs
  | n + 1, s, s', hs, h => by
    simp only [lrun] at h
    cases hst : lstep pr s with
    | next t => rw [hst] at h; exact lrun_wordsLt_from hdata n (lstep_wordsLt hdata hs hst) h
    | halt _ _ => rw [hst] at h; cases h
    | fail _ => rw [hst] at h; cases h

theorem lrun_wordsLt {pr : Prog} (hdata : pr.data.flatten.length < M) (msize : Nat)
    (inp : List Nat) (n : Nat) {s' : LSt} (h : lrun pr n (linit pr msize inp) = .next s') :
    s'.WordsLt :=
  lrun_wordsLt_from hdata n (linit_wordsLt pr msize inp) h

/-! ### `sporos_fast.cpp` semantic boundary theorems -/

/-- Explicit C++ `sporos_fast.cpp` signed division formula handling `INT64_MIN / -1` wrapping. -/
def cppDivs (a b : Nat) : Nat :=
  if a = 2 ^ 63 ∧ b = M - 1 then a
  else ofInt (Int.tdiv (toSigned a) (toSigned b))

/-- Explicit C++ `sporos_fast.cpp` signed remainder formula handling `% -1 = 0`. -/
def cppMods (a b : Nat) : Nat :=
  if b = M - 1 then 0
  else ofInt (Int.tmod (toSigned a) (toSigned b))

theorem cppDivs_eq (a b : Nat) :
    wrap (cppDivs a b) = wrap (ofInt (Int.tdiv (toSigned a) (toSigned b))) := by
  unfold cppDivs
  split
  · rename_i h
    rw [h.1, h.2]
    decide
  · rfl

private theorem int_tmod_one (x : Int) : Int.tmod x 1 = 0 := by
  cases x <;> simp [Int.tmod, Nat.mod_one]

theorem cppMods_eq (a b : Nat) :
    wrap (cppMods a b) = wrap (ofInt (Int.tmod (toSigned a) (toSigned b))) := by
  unfold cppMods
  split
  · rename_i h
    rw [h]
    have hts : toSigned (M - 1) = (-1 : Int) := by decide
    rw [hts, Int.tmod_neg, int_tmod_one]
    rfl
  · rfl

theorem lexecPrim_divs_int64_min_neg_one (pa pb : Nat) (s : LSt) :
    lexecPrim .divs [⟨M - 1, pb⟩, ⟨2 ^ 63, pa⟩] s =
      .ok [⟨2 ^ 63, (pb + pa + s.pc + 1) % M⟩] s := by
  simp only [lexecPrim, lq, List.foldr, Nat.add_zero]
  have hne : (M - 1) % M ≠ 0 := by decide
  rw [if_neg hne]
  have hw : wrap (ofInt (Int.tdiv (toSigned (2 ^ 63)) (toSigned (M - 1)))) = 2 ^ 63 := by decide
  rw [hw]

theorem lexecPrim_mods_neg_one (a : LVal) (pb : Nat) (s : LSt) :
    lexecPrim .mods [⟨M - 1, pb⟩, a] s =
      .ok [⟨0, (pb + a.plen + s.pc + 1) % M⟩] s := by
  simp only [lexecPrim, lq, List.foldr, Nat.add_zero]
  have hne : (M - 1) % M ≠ 0 := by decide
  rw [if_neg hne]
  have hts : toSigned (M - 1) = (-1 : Int) := by decide
  have hw : wrap (ofInt (Int.tmod (toSigned a.w) (toSigned (M - 1)))) = 0 := by
    rw [hts, Int.tmod_neg, int_tmod_one]; rfl
  rw [hw]

theorem lexecPrim_streq_diff_len (l2 a2 l1 a1 : LVal) (s : LSt)
    (hb : a1.w + l1.w ≤ s.msize ∧ a2.w + l2.w ≤ s.msize) (hne : l1.w ≠ l2.w) :
    lexecPrim .streq [l2, a2, l1, a1] s =
      .ok [⟨0, lq [l2, a2, l1, a1] s.pc⟩] s := by
  have hbeq : (l1.w == l2.w) = false := by
    cases h : l1.w == l2.w
    · rfl
    · exact absurd (beq_iff_eq.mp h) hne
  simp only [lexecPrim, if_pos hb, if_neg hne, hbeq, Bool.false_and, Nat.add_zero, lq, Nat.mod_mod]
  rfl

theorem lexecPrim_copy_snapshot (n dst src : LVal) (s : LSt)
    (hb : src.w + n.w ≤ s.msize ∧ dst.w + n.w ≤ s.msize) :
    ∃ s', lexecPrim .copy [n, dst, src] s = .ok [] s' ∧
      ∀ x, s'.mem x =
        if dst.w ≤ x ∧ x < dst.w + n.w then
          ⟨(s.mem (src.w + (x - dst.w))).w, ((s.mem (src.w + (x - dst.w))).plen + lq [n, dst, src] s.pc) % M⟩
        else s.mem x := by
  simp only [lexecPrim, if_pos hb]
  exact ⟨_, rfl, fun _ => rfl⟩

theorem lexecPrim_strhash_fnv (l a : LVal) (s : LSt) (hb : a.w + l.w ≤ s.msize) :
    lexecPrim .strhash [l, a] s =
      .ok [⟨wrap (fnv ((List.range l.w).map fun i => (s.mem (a.w + i)).w % 256)),
            (lq [l, a] s.pc + (List.range l.w).foldr (fun i acc => (s.mem (a.w + i)).plen + acc) 0) % M⟩] s := by
  simp only [lexecPrim, if_pos hb]

theorem lexecPrim_noPhys_observables (a th i : LVal) (s : LSt) :
    lexecPrim .clock [] s = .ok [⟨0, (s.pc + 1) % M⟩] s ∧
    lexecPrim .core [] s = .ok [⟨0, (s.pc + 1) % M⟩] s ∧
    lexecPrim .sensor [i] s = .ok [⟨0, (i.plen + s.pc + 1) % M⟩] s ∧
    lexecPrim .provget [a] s =
      .ok [⟨wrap a.plen, (a.plen + s.pc + 1) % M⟩, ⟨a.w, (a.plen + s.pc + 1) % M⟩] s ∧
    lexecPrim .confget [a] s =
      .ok [⟨0, (a.plen + s.pc + 1) % M⟩, ⟨a.w, (a.plen + s.pc + 1) % M⟩] s ∧
    (0 < th.w → lexecPrim .demand [th, a] s = .fail .demand) ∧
    (th.w = 0 → lexecPrim .demand [th, a] s = .ok [⟨a.w, (a.plen + s.pc + 1) % M⟩] s) := by
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_⟩ <;>
    first
      | (simp [lexecPrim, lq, ltag, wrap]; done)
      | (intro h; simp [lexecPrim, h])

theorem lexecPrim_tput_tget_shadow (tb k v1 v2 : LVal) (s : LSt) (htb : tb.w < s.tabs.length) :
    match lexecPrim .tput [v1, k, tb] s with
    | .ok _ s1 =>
      match lexecPrim .tput [v2, k, tb] s1 with
      | .ok _ s2 =>
        lexecPrim .tget [k, tb] s2 =
          .ok [⟨1, lq [k, tb] s2.pc⟩,
               ⟨v2.w, ((v2.plen + lq [v2, k, tb] s1.pc) % M + lq [k, tb] s2.pc) % M⟩] s2
      | _ => False
    | _ => False := by
  have htb1 : tb.w < (s.tabs.set tb.w ((k.w, ⟨v1.w, (v1.plen + lq [v1, k, tb] s.pc) % M⟩) :: s.tabs[tb.w])).length := by
    simp [htb]
  simp only [lexecPrim, dif_pos htb, dif_pos htb1, List.getElem_set_self, ltfind, ↓reduceIte]
  rw [dif_pos (by simp [htb])]
  rfl

/-! ### Ground `decide +kernel` witnesses (`echo`, `hello`, `upper`, `fib`, `provDemo`) -/

/-- Non-trivial program exercising `getb`, `dup`, `jz` (context `pc` growth), `confget`,
`provget`, and `demand` (both passing at `th = 0` and failing `.demand` at `th = 1`). -/
def provDemo : Prog where
  words := [{
    ins := 0, outs := 0, locals := 0,
    code := [.prim .getb, .prim .dup, .jz 5, .lit 1, .prim .add,
             .prim .confget, .prim .putb, .prim .provget, .prim .putb,
             .prim .getb, .prim .demand, .prim .putb, .lit 0, .prim .exit],
    ann := [(5, ⟨1, 0⟩)] }]
  data := []
  main := 0

theorem provDemo_checks : checkProg provDemo = true := by decide +kernel

theorem provDemo_roundtrip : (decode provDemo.enc).map Prog.enc = some provDemo.enc := by
  decide +kernel

private def haltEq (r : LRes) (c : Nat) (stdout : List Nat) : Bool :=
  match r with
  | .halt rc out => rc == c && out.reverse == stdout
  | _ => false

private def refHaltEq (r : Res) (c : Nat) (stdout : List Nat) : Bool :=
  match r with
  | .halt rc out => rc == c && out.reverse.map Prod.fst == stdout
  | _ => false

private def failEq (r : LRes) (e : Err) : Bool :=
  match r with
  | .fail err => err == e
  | _ => false

private def refFailEq (r : Res) (e : Err) : Bool :=
  match r with
  | .fail err => err == e
  | _ => false

theorem witness_echo_lrun :
    haltEq (lrun echo 40 (linit echo 0 [104, 105])) 0 [104, 105] = true := by
  decide +kernel

theorem witness_echo_agree :
    refHaltEq (run echo noPhys 40 (Sporos.init echo 0 [104, 105])) 0 [104, 105] = true := by
  decide +kernel

theorem witness_hello_lrun :
    haltEq (lrun Examples.hello 300 (linit Examples.hello 64 [])) 0
      ("Hello from Sporos!\n".toList.map Char.toNat) = true := by
  decide +kernel

theorem witness_hello_agree :
    refHaltEq (run Examples.hello noPhys 300 (Sporos.init Examples.hello 64 [])) 0
      ("Hello from Sporos!\n".toList.map Char.toNat) = true := by
  decide +kernel

theorem witness_upper_lrun :
    haltEq (lrun Examples.upper 100 (linit Examples.upper 0 [97, 66, 49, 122])) 0 [65, 66, 49, 90] = true := by
  decide +kernel

theorem witness_upper_agree :
    refHaltEq (run Examples.upper noPhys 100 (Sporos.init Examples.upper 0 [97, 66, 49, 122])) 0 [65, 66, 49, 90] = true := by
  decide +kernel

private def fibExpected : List Nat :=
  ("0\n1\n1\n2\n3\n5\n8\n13\n21\n34\n55\n89\n144\n233\n377\n610\n987\n1597\n2584\n4181\n".toList.map Char.toNat)

theorem witness_fib_lrun :
    haltEq (lrun Examples.fib 2000 (linit Examples.fib 0 [])) 0 fibExpected = true := by
  decide +kernel

theorem witness_fib_ref :
    ∃ out, run Examples.fib noPhys 2000 (Sporos.init Examples.fib 0 []) = .halt 0 out ∧
      out.reverse.map Prod.fst = fibExpected := by
  have h : lrun Examples.fib 2000 (linit Examples.fib 0 []) = .halt 0 fibExpected.reverse := by
    have h0 := witness_fib_lrun
    simp only [haltEq] at h0
    rcases hr : lrun Examples.fib 2000 (linit Examples.fib 0 []) with _ | ⟨c, o⟩ | _ <;>
      rw [hr] at h0 <;> simp at h0
    obtain ⟨rfl, ho⟩ := h0
    rw [← ho, List.reverse_reverse]
  obtain ⟨out, hout, hmap⟩ := (lrun_sim_halt Examples.fib 0 [] 2000 0 fibExpected.reverse).mp h
  refine ⟨out, hout, ?_⟩
  rw [List.map_reverse, hmap, List.reverse_reverse]

theorem witness_provDemo_nonzero_ok :
    haltEq (lrun provDemo 30 (linit provDemo 0 [65, 0])) 0 [0, 14, 66] = true ∧
    refHaltEq (run provDemo noPhys 30 (Sporos.init provDemo 0 [65, 0])) 0 [0, 14, 66] = true := by
  decide +kernel

theorem witness_provDemo_zero_ok :
    haltEq (lrun provDemo 30 (linit provDemo 0 [0, 0])) 0 [0, 6, 0] = true ∧
    refHaltEq (run provDemo noPhys 30 (Sporos.init provDemo 0 [0, 0])) 0 [0, 6, 0] = true := by
  decide +kernel

theorem witness_provDemo_demand_fail :
    failEq (lrun provDemo 30 (linit provDemo 0 [65, 1])) .demand = true ∧
    refFailEq (run provDemo noPhys 30 (Sporos.init provDemo 0 [65, 1])) .demand = true := by
  decide +kernel

theorem witness_copy_overlap :
    (match lexecPrim .copy [⟨3, 0⟩, ⟨1, 0⟩, ⟨0, 0⟩]
        { ds := [], aux := [], rs := [],
          mem := fun a => ⟨([1, 2, 3, 4][a]?).getD 0, 0⟩,
          msize := 16, hp := 8, tabs := [], inp := [], ipos := 0, out := [], pc := 0 } with
      | .ok _ s' => ((List.range 4).map fun i => (s'.mem i).w) == [1, 1, 2, 3]
      | _ => false) = true := by
  decide +kernel

end SporosProvLen
