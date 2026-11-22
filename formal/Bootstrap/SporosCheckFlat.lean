/-
SporosCheckFlat: flat byte-slice specification of the `SPR1` (`Sporos.decode` +
`Sporos.checkProg`) verifier, universal soundness & completeness for all
`bs : List Nat`, and 3-pass + `TMRVoter` + dual-`ACCEPT`-guard single-fault
false-accept immunity over `X86Int.pstep` / `X86Int.prun`.
-/
import Sporos.Bytecode
import Sporos.Check
import Sporos.SafeStep
import Sporos.Run
import Sporos.Examples
import Bootstrap.TMRVoter


namespace SporosCheckFlat

section FlatSpec

open Sporos

def W : Nat := 2 ^ 64
def HMAX : Nat := 2 ^ 48
def ACCEPT : Nat := 0x5acce970a55a3c96
def REJECT : Nat := 0xdeadbeef11110001
def MAGIC : Nat := 0x31525053

/-! ### Little-endian readers and opcode lengths -/

def rdU32 (bs : List Nat) (pos : Nat) : Option Nat := do
  let b0 ← bs[pos]?
  let b1 ← bs[pos + 1]?
  let b2 ← bs[pos + 2]?
  let b3 ← bs[pos + 3]?
  pure (b0 + 256 * b1 + 65536 * b2 + 16777216 * b3)

def rdU64 (bs : List Nat) (pos : Nat) : Option Nat := do
  let lo ← rdU32 bs pos
  let hi ← rdU32 bs (pos + 4)
  pure (lo + 4294967296 * hi)

def instrLen (op : Nat) : Option Nat :=
  if op < 57 then some 1
  else if op == 0x80 then some 9
  else if op == 0x81 || op == 0x82 || op == 0x86 || op == 0x87 ||
          op == 0x88 || op == 0x89 || op == 0x8a || op == 0x8b then some 5
  else if op == 0x83 || op == 0x84 || op == 0x85 || op == 0x8c then some 1
  else none

def skipInstr (bs : List Nat) (pos : Nat) : Option Nat := do
  let op ← bs[pos]?
  let len ← instrLen op
  if pos + len ≤ bs.length then some (pos + len) else none

def skipCode (bs : List Nat) : Nat → Nat → Option Nat
  | 0, pos => some pos
  | n + 1, pos => do
    let pos' ← skipInstr bs pos
    skipCode bs n pos'

def skipData (bs : List Nat) : Nat → Nat → Option Nat
  | 0, pos => some pos
  | n + 1, pos => do
    let len ← rdU32 bs pos
    let pos1 := pos + 4
    if pos1 + len ≤ bs.length then skipData bs n (pos1 + len) else none

def skipWords (bs : List Nat) : Nat → Nat → Option Nat
  | 0, pos => some pos
  | n + 1, pos =>
    if pos + 20 ≤ bs.length then do
      let nc ← rdU32 bs (pos + 12)
      let na ← rdU32 bs (pos + 16)
      let annOff ← skipCode bs nc (pos + 20)
      let annBytes := na * 12
      if annOff + annBytes ≤ bs.length then
        skipWords bs n (annOff + annBytes)
      else none
    else none

/-- Parse the `SPR1` header, `nd` data items, and `nw` words, returning
`(main, nw, wordsStart)`. -/
def parse (bs : List Nat) : Option (Nat × Nat × Nat) := do
  if bs.length < 16 then none else
  let magic ← rdU32 bs 0
  if magic != MAGIC then none else
  let main ← rdU32 bs 4
  let nw ← rdU32 bs 8
  let nd ← rdU32 bs 12
  let ws ← skipData bs nd 16
  let endPos ← skipWords bs nw ws
  if endPos == bs.length then some (main, nw, ws) else none

/-! ### Streaming word index, annotation lookup, and stack-effect checker -/

structure WordSlice where
  ins : Nat
  outs : Nat
  locals : Nat
  nc : Nat
  na : Nat
  codeOff : Nat
  annOff : Nat
  nextOff : Nat
deriving DecidableEq, Repr

def readWordAt (bs : List Nat) (pos : Nat) : Option WordSlice :=
  if pos + 20 ≤ bs.length then do
    let ins ← rdU32 bs pos
    let outs ← rdU32 bs (pos + 4)
    let locals ← rdU32 bs (pos + 8)
    let nc ← rdU32 bs (pos + 12)
    let na ← rdU32 bs (pos + 16)
    let codeOff := pos + 20
    let annOff ← skipCode bs nc codeOff
    let annBytes := na * 12
    if annOff + annBytes ≤ bs.length then
      some ⟨ins, outs, locals, nc, na, codeOff, annOff, annOff + annBytes⟩
    else none
  else none

def wordAt (bs : List Nat) (ws : Nat) : Nat → Option WordSlice
  | 0 => readWordAt bs ws
  | w + 1 => do
    let wm ← readWordAt bs ws
    wordAt bs wm.nextOff w

def lookupLoop (bs : List Nat) (t : Nat) : Nat → Nat → Option (Nat × Nat)
  | 0, _ => none
  | k + 1, off => do
    let idx ← rdU32 bs off
    if idx == t then
      let ad ← rdU32 bs (off + 4)
      let aa ← rdU32 bs (off + 8)
      pure (ad, aa)
    else
      lookupLoop bs t k (off + 12)

def lookup (bs : List Nat) (annOff na t : Nat) : Option (Nat × Nat) :=
  lookupLoop bs t na annOff

/-- Packed `(in << 4) | out` stack effects for the 57 primitives in `Prim.all`. -/
def primEffBytes : List Nat :=
  [ 0x12, 0x10, 0x22, 0x23, 0x33, 0x21, 0x23
  , 0x21, 0x21, 0x21, 0x21, 0x21, 0x21, 0x21, 0x11
  , 0x21, 0x21, 0x21, 0x11, 0x21, 0x21, 0x21
  , 0x21, 0x21, 0x21, 0x21, 0x21, 0x21
  , 0x11, 0x20, 0x11, 0x20, 0x11, 0x20, 0x11, 0x20
  , 0x30, 0x30, 0x31
  , 0x11, 0x01, 0x10
  , 0x41, 0x21
  , 0x01, 0x30, 0x22
  , 0x01, 0x10, 0x10, 0x10
  , 0x01, 0x01, 0x11
  , 0x12, 0x12, 0x21 ]

theorem primEffBytes_len : primEffBytes.length = 57 := by decide +kernel
theorem primAll_len : Prim.all.length = 57 := by decide +kernel

theorem primEffBytes_spec :
    (primEffBytes.map fun b => (b / 16, b % 16)) = Prim.all.map Prim.eff := by
  decide +kernel

def primEff (op : Nat) : Option (Nat × Nat) :=
  primEffBytes[op]?.map fun b => (b / 16, b % 16)

inductive StepRes where
  | overflow
  | reject
  | noFall
  | next (d a : Nat)
deriving DecidableEq, Repr

inductive CheckRes where
  | ok
  | reject
  | overflow
deriving DecidableEq, Repr

def hstepF (bs : List Nat) (nw ws outs locals annOff na hd ha op imm : Nat) : StepRes :=
  match primEff op with
  | some (eIn, eOut) =>
    if hd < eIn then .reject
    else if op == 50 then .noFall
    else
      let d' := (hd - eIn) + eOut
      if d' ≥ HMAX then .overflow else .next d' ha
  | none =>
    if op == 0x80 then
      let d' := hd + 1
      if d' ≥ HMAX then .overflow else .next d' ha
    else if op == 0x81 then
      let d' := hd + 2
      if d' ≥ HMAX then .overflow else .next d' ha
    else if op == 0x82 then
      if imm ≥ hd then .reject
      else
        let d' := hd + 1
        if d' ≥ HMAX then .overflow else .next d' ha
    else if op == 0x83 then
      if hd < 1 then .reject
      else
        let a' := ha + 1
        if a' ≥ HMAX then .overflow else .next (hd - 1) a'
    else if op == 0x84 then
      if ha < 1 then .reject
      else
        let d' := hd + 1
        if d' ≥ HMAX then .overflow else .next d' (ha - 1)
    else if op == 0x85 then
      if ha < 1 then .reject
      else
        let d' := hd + 1
        if d' ≥ HMAX then .overflow else .next d' ha
    else if op == 0x86 then
      if imm ≥ locals then .reject
      else
        let d' := hd + 1
        if d' ≥ HMAX then .overflow else .next d' ha
    else if op == 0x87 then
      if hd < 1 || imm ≥ locals then .reject
      else .next (hd - 1) ha
    else if op == 0x88 then
      match lookup bs annOff na imm with
      | some (ad, aa) => if ad == hd && aa == ha then .noFall else .reject
      | none => .reject
    else if op == 0x89 || op == 0x8a then
      if hd < 1 then .reject
      else
        let d1 := hd - 1
        match lookup bs annOff na imm with
        | some (ad, aa) => if ad == d1 && aa == ha then .next d1 ha else .reject
        | none => .reject
    else if op == 0x8b then
      if imm ≥ nw then .reject
      else match wordAt bs ws imm with
        | none => .reject
        | some c =>
          if hd < c.ins then .reject
          else
            let d' := (hd - c.ins) + c.outs
            if d' ≥ HMAX then .overflow else .next d' ha
    else if op == 0x8c then
      if hd == outs && ha == 0 then .noFall else .reject
    else .reject

def walk (bs : List Nat) (nw ws outs locals nc annOff na : Nat) :
    Nat → Nat → Nat → Nat → Nat → CheckRes
  | 0, _, _, _, _ => .ok
  | rem + 1, i, p, curD, curA =>
    match bs[p]? with
    | none => .reject
    | some op =>
      match instrLen op with
      | none => .reject
      | some ilen =>
        if !(p + ilen ≤ bs.length) then .reject
        else
          match (if ilen == 5 then rdU32 bs (p + 1) else some 0) with
          | none => .reject
          | some imm =>
            let (hd, ha) := (lookup bs annOff na i).getD (curD, curA)
            match hstepF bs nw ws outs locals annOff na hd ha op imm with
            | .overflow => .overflow
            | .reject => .reject
            | .noFall => walk bs nw ws outs locals nc annOff na rem (i + 1) (p + ilen) 0 0
            | .next nd na' =>
              if i + 1 ≥ nc then .reject
              else
                let okNext := match lookup bs annOff na (i + 1) with
                  | none => true
                  | some (ad, aa) => ad == nd && aa == na'
                if okNext then walk bs nw ws outs locals nc annOff na rem (i + 1) (p + ilen) nd na'
                else .reject

def annsOK (bs : List Nat) (nc annOff na : Nat) : Nat → Nat → Bool
  | 0, _ => true
  | k + 1, off =>
    match rdU32 bs off, rdU32 bs (off + 4), rdU32 bs (off + 8) with
    | some t, some ad, some aa =>
      t < nc && lookup bs annOff na t == some (ad, aa) &&
        annsOK bs nc annOff na k (off + 12)
    | _, _, _ => false

def checkWords (bs : List Nat) (nw ws : Nat) : Nat → Nat → CheckRes
  | 0, _ => .ok
  | rem + 1, off =>
    match readWordAt bs off with
    | none => .reject
    | some wm =>
      if wm.nc == 0 then .reject
      else
        let ok0 := match lookup bs wm.annOff wm.na 0 with
          | none => true
          | some (d0, a0) => d0 == wm.ins && a0 == 0
        if !ok0 then .reject
        else match walk bs nw ws wm.outs wm.locals wm.nc wm.annOff wm.na wm.nc 0 wm.codeOff wm.ins 0 with
          | .overflow => .overflow
          | .reject => .reject
          | .ok =>
            if annsOK bs wm.nc wm.annOff wm.na wm.na wm.annOff then
              checkWords bs nw ws rem wm.nextOff
            else .reject

def checkFlatRes (bs : List Nat) : Option CheckRes :=
  if !bs.all (· < 256) || bs.length ≥ W then none
  else some <|
    match parse bs with
    | none => .reject
    | some (main, nw, ws) =>
      if main ≥ nw then .reject
      else match wordAt bs ws main with
        | none => .reject
        | some mainWm =>
          if mainWm.ins != 0 then .reject
          else checkWords bs nw ws nw ws

def checkFlat (bs : List Nat) : Option Bool := do
  match ← checkFlatRes bs with
  | .ok => some true
  | .reject => some false
  | .overflow => none

/-! ### Block A: Little-endian byte slice lemmas (`rdU32`, `rdU64` vs `Sporos.rdLE`) -/

theorem rdU32_bounds {bs : List Nat} {pos v : Nat} (h : rdU32 bs pos = some v) :
    pos + 4 ≤ bs.length := by
  simp only [rdU32, bind, Option.bind, pure] at h
  cases h0 : bs[pos]? <;> rw [h0] at h
  · cases h
  cases h1 : bs[pos + 1]? <;> rw [h1] at h
  · cases h
  cases h2 : bs[pos + 2]? <;> rw [h2] at h
  · cases h
  cases h3 : bs[pos + 3]? <;> rw [h3] at h
  · cases h
  have := (List.getElem?_eq_some_iff.1 h3).1
  omega

theorem rdU32_lt {bs : List Nat} (hb : ∀ b ∈ bs, b < 256) {pos v : Nat} (h : rdU32 bs pos = some v) :
    v < 2 ^ 32 := by
  simp only [rdU32, bind, Option.bind, pure] at h
  cases h0 : bs[pos]? with
  | none => rw [h0] at h; cases h
  | some b0 =>
    rw [h0] at h
    cases h1 : bs[pos + 1]? with
    | none => rw [h1] at h; cases h
    | some b1 =>
      rw [h1] at h
      cases h2 : bs[pos + 2]? with
      | none => rw [h2] at h; cases h
      | some b2 =>
        rw [h2] at h
        cases h3 : bs[pos + 3]? with
        | none => rw [h3] at h; cases h
        | some b3 =>
          rw [h3] at h; cases h
          have hb0 := hb b0 (List.mem_of_getElem? h0)
          have hb1 := hb b1 (List.mem_of_getElem? h1)
          have hb2 := hb b2 (List.mem_of_getElem? h2)
          have hb3 := hb b3 (List.mem_of_getElem? h3)
          omega

theorem rdU32_magic_iff {bs : List Nat} (hb : ∀ b ∈ bs, b < 256) :
    rdU32 bs 0 = some MAGIC ↔
      bs[0]? = some 0x53 ∧ bs[1]? = some 0x50 ∧ bs[2]? = some 0x52 ∧ bs[3]? = some 0x31 := by
  constructor
  · intro h
    simp only [rdU32, bind, Option.bind, pure] at h
    cases h0 : bs[0]? with
    | none => rw [h0] at h; cases h
    | some b0 =>
      rw [h0] at h
      cases h1 : bs[1]? with
      | none => rw [h1] at h; cases h
      | some b1 =>
        rw [h1] at h
        cases h2 : bs[2]? with
        | none => rw [h2] at h; cases h
        | some b2 =>
          rw [h2] at h
          cases h3 : bs[3]? with
          | none => rw [h3] at h; cases h
          | some b3 =>
            rw [h3] at h
            simp only [Option.some.injEq] at h
            have hb0 := hb b0 (List.mem_of_getElem? h0)
            have hb1 := hb b1 (List.mem_of_getElem? h1)
            have hb2 := hb b2 (List.mem_of_getElem? h2)
            have hb3 := hb b3 (List.mem_of_getElem? h3)
            have : b0 = 0x53 ∧ b1 = 0x50 ∧ b2 = 0x52 ∧ b3 = 0x31 := by
              unfold MAGIC at *
              omega
            simp [this]
  · rintro ⟨h0, h1, h2, h3⟩
    simp [rdU32, h0, h1, h2, h3, MAGIC]

private theorem drop_take_four (bs : List Nat) (pos : Nat) (hl : pos + 4 ≤ bs.length) :
    (bs.drop pos).take 4 = [bs[pos], bs[pos + 1], bs[pos + 2], bs[pos + 3]] := by
  have h0 : pos < bs.length := by omega
  have h1 : pos + 1 < bs.length := by omega
  have h2 : pos + 2 < bs.length := by omega
  have h3 : pos + 3 < bs.length := by omega
  rw [List.drop_eq_getElem_cons h0, List.drop_eq_getElem_cons h1,
      List.drop_eq_getElem_cons h2, List.drop_eq_getElem_cons h3]
  rfl

private theorem drop_take_eight (bs : List Nat) (pos : Nat) (hl : pos + 8 ≤ bs.length) :
    (bs.drop pos).take 8 =
      [bs[pos], bs[pos + 1], bs[pos + 2], bs[pos + 3],
       bs[pos + 4], bs[pos + 5], bs[pos + 6], bs[pos + 7]] := by
  have h0 : pos < bs.length := by omega
  have h1 : pos + 1 < bs.length := by omega
  have h2 : pos + 2 < bs.length := by omega
  have h3 : pos + 3 < bs.length := by omega
  have h4 : pos + 4 < bs.length := by omega
  have h5 : pos + 5 < bs.length := by omega
  have h6 : pos + 6 < bs.length := by omega
  have h7 : pos + 7 < bs.length := by omega
  rw [List.drop_eq_getElem_cons h0, List.drop_eq_getElem_cons h1,
      List.drop_eq_getElem_cons h2, List.drop_eq_getElem_cons h3,
      List.drop_eq_getElem_cons h4, List.drop_eq_getElem_cons h5,
      List.drop_eq_getElem_cons h6, List.drop_eq_getElem_cons h7]
  rfl

private theorem rdU32_eq_some {bs : List Nat} {pos : Nat} (hl : pos + 4 ≤ bs.length) :
    rdU32 bs pos =
      some (bs[pos] + 256 * bs[pos + 1] + 65536 * bs[pos + 2] + 16777216 * bs[pos + 3]) := by
  have h0 : bs[pos]? = some bs[pos] := List.getElem?_eq_getElem (by omega)
  have h1 : bs[pos + 1]? = some bs[pos + 1] := List.getElem?_eq_getElem (by omega)
  have h2 : bs[pos + 2]? = some bs[pos + 2] := List.getElem?_eq_getElem (by omega)
  have h3 : bs[pos + 3]? = some bs[pos + 3] := List.getElem?_eq_getElem (by omega)
  simp only [rdU32, bind, Option.bind, pure, h0, h1, h2, h3]

private theorem foldl_zip4 (a0 a1 a2 a3 : Nat) (f : Nat → Nat × Nat → Nat) (init : Nat) :
    (([a0, a1, a2, a3].zip (List.range 4)).foldl f init) =
      f (f (f (f init (a0, 0)) (a1, 1)) (a2, 2)) (a3, 3) := by
  have hr : List.range 4 = [0, 1, 2, 3] := rfl
  rw [hr]
  simp only [List.zip_cons_cons, List.zip_nil_right, List.foldl_cons, List.foldl_nil]

private theorem rdLE4_val (a0 a1 a2 a3 : Nat) :
    (([a0, a1, a2, a3].zip (List.range 4)).foldl (fun acc (b, i) => acc + b * 256 ^ i) 0) =
      a0 + 256 * a1 + 65536 * a2 + 16777216 * a3 := by
  rw [foldl_zip4]
  dsimp only
  simp only [Nat.zero_add, Nat.pow_zero, Nat.mul_one,
    show (256 : Nat) ^ 1 = 256 from rfl,
    show (256 : Nat) ^ 2 = 65536 from rfl,
    show (256 : Nat) ^ 3 = 16777216 from rfl]
  rw [Nat.mul_comm a1 256, Nat.mul_comm a2 65536, Nat.mul_comm a3 16777216]

theorem rdLE4_drop_iff (bs : List Nat) (pos v : Nat) (rest : List Nat) :
    Sporos.rdLE (bs.drop pos) 4 = some (v, rest) ↔
      rdU32 bs pos = some v ∧ rest = bs.drop (pos + 4) := by
  unfold Sporos.rdLE
  simp only [List.length_drop, List.drop_drop]
  by_cases hl : pos + 4 ≤ bs.length
  · have hnl : ¬ (bs.length - pos < 4) := by omega
    rw [if_neg hnl, drop_take_four bs pos hl, rdLE4_val, rdU32_eq_some hl]
    simp only [Option.some.injEq, Prod.mk.injEq]
    constructor
    · rintro ⟨h1, h2⟩
      exact ⟨h1, by rw [← h2, Nat.add_comm]⟩
    · rintro ⟨h1, h2⟩
      exact ⟨h1, by rw [h2, Nat.add_comm]⟩
  · have hlt : bs.length - pos < 4 := by omega
    rw [if_pos hlt]
    constructor
    · intro h; cases h
    · rintro ⟨h, _⟩
      exfalso; exact hl (rdU32_bounds h)

theorem rdU64_bounds {bs : List Nat} {pos v : Nat} (h : rdU64 bs pos = some v) :
    pos + 8 ≤ bs.length := by
  simp only [rdU64, bind, Option.bind, pure] at h
  cases hlo : rdU32 bs pos <;> rw [hlo] at h
  · cases h
  cases hhi : rdU32 bs (pos + 4) <;> rw [hhi] at h
  · cases h
  have := rdU32_bounds hhi
  omega

private theorem rdU64_eq_some {bs : List Nat} {pos : Nat} (hl : pos + 8 ≤ bs.length) :
    rdU64 bs pos =
      some ((bs[pos] + 256 * bs[pos + 1] + 65536 * bs[pos + 2] + 16777216 * bs[pos + 3]) +
        4294967296 * (bs[pos + 4] + 256 * bs[pos + 5] + 65536 * bs[pos + 6] + 16777216 * bs[pos + 7])) := by
  have h0 : bs[pos]? = some bs[pos] := List.getElem?_eq_getElem (by omega)
  have h1 : bs[pos + 1]? = some bs[pos + 1] := List.getElem?_eq_getElem (by omega)
  have h2 : bs[pos + 2]? = some bs[pos + 2] := List.getElem?_eq_getElem (by omega)
  have h3 : bs[pos + 3]? = some bs[pos + 3] := List.getElem?_eq_getElem (by omega)
  have h4 : bs[pos + 4]? = some bs[pos + 4] := List.getElem?_eq_getElem (by omega)
  have h5 : bs[pos + 4 + 1]? = some bs[pos + 5] := List.getElem?_eq_getElem (by omega)
  have h6 : bs[pos + 4 + 2]? = some bs[pos + 6] := List.getElem?_eq_getElem (by omega)
  have h7 : bs[pos + 4 + 3]? = some bs[pos + 7] := List.getElem?_eq_getElem (by omega)
  simp only [rdU64, rdU32, bind, Option.bind, pure, h0, h1, h2, h3, h4, h5, h6, h7]

private theorem rdLE8_poly (a0 a1 a2 a3 a4 a5 a6 a7 : Nat) :
    ((((((((0 + a0 * 256 ^ 0) + a1 * 256 ^ 1) + a2 * 256 ^ 2) + a3 * 256 ^ 3) +
      a4 * 256 ^ 4) + a5 * 256 ^ 5) + a6 * 256 ^ 6) + a7 * 256 ^ 7) =
    a0 + 256 * a1 + 65536 * a2 + 16777216 * a3 +
      4294967296 * (a4 + 256 * a5 + 65536 * a6 + 16777216 * a7) := by
  simp only [Nat.zero_add, Nat.pow_zero, Nat.mul_one, Nat.mul_add, ← Nat.mul_assoc,
    show (256 : Nat) ^ 1 = 256 from rfl,
    show (256 : Nat) ^ 2 = 65536 from rfl,
    show (256 : Nat) ^ 3 = 16777216 from rfl,
    show (256 : Nat) ^ 4 = 4294967296 from rfl,
    show (256 : Nat) ^ 5 = 1099511627776 from rfl,
    show (256 : Nat) ^ 6 = 281474976710656 from rfl,
    show (256 : Nat) ^ 7 = 72057594037927936 from rfl,
    show (4294967296 : Nat) * 256 = 1099511627776 from rfl,
    show (4294967296 : Nat) * 65536 = 281474976710656 from rfl,
    show (4294967296 : Nat) * 16777216 = 72057594037927936 from rfl]
  rw [Nat.mul_comm a1 256, Nat.mul_comm a2 65536, Nat.mul_comm a3 16777216,
      Nat.mul_comm a4 4294967296, Nat.mul_comm a5 1099511627776,
      Nat.mul_comm a6 281474976710656, Nat.mul_comm a7 72057594037927936]
  simp only [Nat.add_assoc]

private theorem foldl_zip8 (a0 a1 a2 a3 a4 a5 a6 a7 : Nat)
    (f : Nat → Nat × Nat → Nat) (init : Nat) :
    (([a0, a1, a2, a3, a4, a5, a6, a7].zip (List.range 8)).foldl f init) =
      f (f (f (f (f (f (f (f init (a0, 0)) (a1, 1)) (a2, 2)) (a3, 3))
        (a4, 4)) (a5, 5)) (a6, 6)) (a7, 7) := by
  have hr : List.range 8 = [0, 1, 2, 3, 4, 5, 6, 7] := rfl
  rw [hr]
  simp only [List.zip_cons_cons, List.zip_nil_right, List.foldl_cons, List.foldl_nil]

private theorem rdLE8_val (a0 a1 a2 a3 a4 a5 a6 a7 : Nat) :
    (([a0, a1, a2, a3, a4, a5, a6, a7].zip (List.range 8)).foldl (fun acc (b, i) => acc + b * 256 ^ i) 0) =
      a0 + 256 * a1 + 65536 * a2 + 16777216 * a3 +
        4294967296 * (a4 + 256 * a5 + 65536 * a6 + 16777216 * a7) := by
  rw [foldl_zip8]
  dsimp only
  exact rdLE8_poly a0 a1 a2 a3 a4 a5 a6 a7

theorem rdLE8_drop_iff (bs : List Nat) (pos v : Nat) (rest : List Nat) :
    Sporos.rdLE (bs.drop pos) 8 = some (v, rest) ↔
      rdU64 bs pos = some v ∧ rest = bs.drop (pos + 8) := by
  unfold Sporos.rdLE
  simp only [List.length_drop, List.drop_drop]
  by_cases hl : pos + 8 ≤ bs.length
  · have hnl : ¬ (bs.length - pos < 8) := by omega
    rw [if_neg hnl, drop_take_eight bs pos hl, rdLE8_val, rdU64_eq_some hl]
    simp only [Option.some.injEq, Prod.mk.injEq]
    constructor
    · rintro ⟨h1, h2⟩
      exact ⟨h1, by rw [← h2, Nat.add_comm]⟩
    · rintro ⟨h1, h2⟩
      exact ⟨h1, by rw [h2, Nat.add_comm]⟩
  · have hlt : bs.length - pos < 8 := by omega
    rw [if_pos hlt]
    constructor
    · intro h; cases h
    · rintro ⟨h, _⟩
      exfalso; exact hl (rdU64_bounds h)

/-! ### Block B: Structural decoding equivalence (`skip*`, `parse` vs `Sporos.decode`) -/

def instrOfOpImm (op imm : Nat) : Option Instr :=
  if h : op < Prim.all.length then some (.prim (Prim.all[op]'h))
  else match op with
    | 0x80 => some (.lit imm)
    | 0x81 => some (.str imm)
    | 0x82 => some (.pick imm)
    | 0x83 => some .tor
    | 0x84 => some .fromr
    | 0x85 => some .rfetch
    | 0x86 => some (.locget imm)
    | 0x87 => some (.locset imm)
    | 0x88 => some (.jmp imm)
    | 0x89 => some (.jz imm)
    | 0x8a => some (.jnz imm)
    | 0x8b => some (.call imm)
    | 0x8c => some .ret
    | _ => none

private theorem instrLen_ok {op len : Nat} (h : instrLen op = some len) :
    (len = 1 ∨ len = 5 ∨ len = 9) ∧ ∀ imm, ∃ ins, instrOfOpImm op imm = some ins := by
  by_cases h57 : op < 57
  · simp [instrLen, h57] at h; cases h
    have h' : op < Prim.all.length := by rwa [primAll_len]
    exact ⟨Or.inl rfl, fun _ => ⟨.prim (Prim.all[op]'h'), by simp [instrOfOpImm, h']⟩⟩
  by_cases h80 : op = 0x80; · subst h80; simp [instrLen] at h; cases h; exact ⟨Or.inr (Or.inr rfl), fun _ => ⟨_, rfl⟩⟩
  by_cases h81 : op = 0x81; · subst h81; simp [instrLen] at h; cases h; exact ⟨Or.inr (Or.inl rfl), fun _ => ⟨_, rfl⟩⟩
  by_cases h82 : op = 0x82; · subst h82; simp [instrLen] at h; cases h; exact ⟨Or.inr (Or.inl rfl), fun _ => ⟨_, rfl⟩⟩
  by_cases h83 : op = 0x83; · subst h83; simp [instrLen] at h; cases h; exact ⟨Or.inl rfl, fun _ => ⟨_, rfl⟩⟩
  by_cases h84 : op = 0x84; · subst h84; simp [instrLen] at h; cases h; exact ⟨Or.inl rfl, fun _ => ⟨_, rfl⟩⟩
  by_cases h85 : op = 0x85; · subst h85; simp [instrLen] at h; cases h; exact ⟨Or.inl rfl, fun _ => ⟨_, rfl⟩⟩
  by_cases h86 : op = 0x86; · subst h86; simp [instrLen] at h; cases h; exact ⟨Or.inr (Or.inl rfl), fun _ => ⟨_, rfl⟩⟩
  by_cases h87 : op = 0x87; · subst h87; simp [instrLen] at h; cases h; exact ⟨Or.inr (Or.inl rfl), fun _ => ⟨_, rfl⟩⟩
  by_cases h88 : op = 0x88; · subst h88; simp [instrLen] at h; cases h; exact ⟨Or.inr (Or.inl rfl), fun _ => ⟨_, rfl⟩⟩
  by_cases h89 : op = 0x89; · subst h89; simp [instrLen] at h; cases h; exact ⟨Or.inr (Or.inl rfl), fun _ => ⟨_, rfl⟩⟩
  by_cases h8a : op = 0x8a; · subst h8a; simp [instrLen] at h; cases h; exact ⟨Or.inr (Or.inl rfl), fun _ => ⟨_, rfl⟩⟩
  by_cases h8b : op = 0x8b; · subst h8b; simp [instrLen] at h; cases h; exact ⟨Or.inr (Or.inl rfl), fun _ => ⟨_, rfl⟩⟩
  by_cases h8c : op = 0x8c; · subst h8c; simp [instrLen] at h; cases h; exact ⟨Or.inl rfl, fun _ => ⟨_, rfl⟩⟩
  unfold instrLen at h
  simp only [h57, Bool.or_eq_true, beq_iff_eq, h80, h81, h82, h83, h84, h85, h86, h87, h88, h89,
    h8a, h8b, h8c, or_self, ↓reduceIte] at h
  cases h

theorem skipInstr_bounds {bs : List Nat} {pos pos' : Nat} (h : skipInstr bs pos = some pos') :
    pos < pos' ∧ pos' ≤ bs.length := by
  simp only [skipInstr, bind, Option.bind] at h
  cases hop : bs[pos]? with
  | none => rw [hop] at h; cases h
  | some op =>
    rw [hop] at h; dsimp only at h
    cases hlen : instrLen op with
    | none => rw [hlen] at h; cases h
    | some len =>
      rw [hlen] at h; dsimp only at h
      have hpos : 0 < len := by
        obtain ⟨hlens, _⟩ := instrLen_ok hlen
        rcases hlens with rfl | rfl | rfl <;> omega
      split at h
      · cases h; omega
      · cases h

private theorem decInstr_match_eq (op : Nat) (rest : List Nat) (h57 : ¬ op < Prim.all.length) :
    Sporos.decInstr (op :: rest) =
      let imm (k : Nat) (f : Nat → Instr) : Option (Instr × List Nat) :=
        (Sporos.rdLE rest k).map fun x => (f x.1, x.2)
      if op = 0x80 then imm 8 .lit
      else if op = 0x81 then imm 4 .str
      else if op = 0x82 then imm 4 .pick
      else if op = 0x83 then some (.tor, rest)
      else if op = 0x84 then some (.fromr, rest)
      else if op = 0x85 then some (.rfetch, rest)
      else if op = 0x86 then imm 4 .locget
      else if op = 0x87 then imm 4 .locset
      else if op = 0x88 then imm 4 .jmp
      else if op = 0x89 then imm 4 .jz
      else if op = 0x8a then imm 4 .jnz
      else if op = 0x8b then imm 4 .call
      else if op = 0x8c then some (.ret, rest)
      else none := by
  simp only [Sporos.decInstr, dif_neg h57]
  unfold Sporos.decInstr.match_1
  by_cases h80 : op = 0x80; · subst h80; rfl
  rw [dif_neg h80, if_neg h80]
  by_cases h81 : op = 0x81; · subst h81; rfl
  rw [dif_neg h81, if_neg h81]
  by_cases h82 : op = 0x82; · subst h82; rfl
  rw [dif_neg h82, if_neg h82]
  by_cases h83 : op = 0x83; · subst h83; rfl
  rw [dif_neg h83, if_neg h83]
  by_cases h84 : op = 0x84; · subst h84; rfl
  rw [dif_neg h84, if_neg h84]
  by_cases h85 : op = 0x85; · subst h85; rfl
  rw [dif_neg h85, if_neg h85]
  by_cases h86 : op = 0x86; · subst h86; rfl
  rw [dif_neg h86, if_neg h86]
  by_cases h87 : op = 0x87; · subst h87; rfl
  rw [dif_neg h87, if_neg h87]
  by_cases h88 : op = 0x88; · subst h88; rfl
  rw [dif_neg h88, if_neg h88]
  by_cases h89 : op = 0x89; · subst h89; rfl
  rw [dif_neg h89, if_neg h89]
  by_cases h8a : op = 0x8a; · subst h8a; rfl
  rw [dif_neg h8a, if_neg h8a]
  by_cases h8b : op = 0x8b; · subst h8b; rfl
  rw [dif_neg h8b, if_neg h8b]
  by_cases h8c : op = 0x8c; · subst h8c; rfl
  rw [dif_neg h8c, if_neg h8c]

private theorem decInstr_imm4_aux {bs : List Nat} {pos : Nat} {op : Nat}
    {f : Nat → Instr} {ins : Instr} {rest : List Nat}
    (hpos : bs[pos]? = some op) (hlen : instrLen op = some 5)
    (hf : instrOfOpImm op = fun imm => some (f imm))
    (h : (Sporos.rdLE (bs.drop (pos + 1)) 4).map (fun x => (f x.1, x.2)) = some (ins, rest)) :
    ∃ len imm,
      instrLen op = some len ∧
      pos + len ≤ bs.length ∧
      skipInstr bs pos = some (pos + len) ∧
      (if len == 5 then rdU32 bs (pos + 1) else if len == 9 then rdU64 bs (pos + 1) else some 0) = some imm ∧
      instrOfOpImm op imm = some ins ∧
      rest = bs.drop (pos + len) := by
  simp only [Option.map_eq_some_iff] at h
  obtain ⟨⟨v, r⟩, hrd, hv⟩ := h
  rw [rdLE4_drop_iff] at hrd
  obtain ⟨hrd4, rfl⟩ := hrd
  cases hv
  have hb := rdU32_bounds hrd4
  refine ⟨5, v, hlen, by omega, ?_, by simp [hrd4], ?_, ?_⟩
  · simp [skipInstr, hpos, hlen]; omega
  · rw [hf]
  · simp [Nat.add_assoc]

def decFlatAt (bs : List Nat) (pos : Nat) : Option (Instr × Nat × Nat × Nat) := do
  let op ← bs[pos]?
  let len ← instrLen op
  if pos + len ≤ bs.length then do
    let imm ← if len == 5 then rdU32 bs (pos + 1)
              else if len == 9 then rdU64 bs (pos + 1)
              else some 0
    let ins ← instrOfOpImm op imm
    pure (ins, pos + len, op, imm)
  else none

private theorem rdU32_of_lt {bs : List Nat} {pos : Nat} (h : pos + 4 ≤ bs.length) :
    ∃ v, rdU32 bs pos = some v :=
  ⟨_, rdU32_eq_some h⟩

private theorem rdU64_of_lt {bs : List Nat} {pos : Nat} (h : pos + 8 ≤ bs.length) :
    ∃ v, rdU64 bs pos = some v :=
  ⟨_, rdU64_eq_some h⟩

private theorem decFlatAt_step {bs : List Nat} {pos op len : Nat}
    (hpos : bs[pos]? = some op) (hlen : instrLen op = some len) :
    decFlatAt bs pos =
      if pos + len ≤ bs.length then do
        let imm ← if len == 5 then rdU32 bs (pos + 1)
                  else if len == 9 then rdU64 bs (pos + 1)
                  else some 0
        let ins ← instrOfOpImm op imm
        pure (ins, pos + len, op, imm)
      else none := by
  simp only [decFlatAt, bind, Option.bind, pure, hpos, hlen]

private theorem skipInstr_eq_if {bs : List Nat} {pos op len : Nat}
    (hpos : bs[pos]? = some op) (hlen : instrLen op = some len) :
    skipInstr bs pos = if pos + len ≤ bs.length then some (pos + len) else none := by
  simp only [skipInstr, bind, Option.bind, hpos, hlen]

private theorem rdLE4_eq_rdU32_bind {bs : List Nat} {pos : Nat} {f : Nat → Instr}
    (hlen : pos + 5 ≤ bs.length) :
    (Sporos.rdLE (bs.drop (pos + 1)) 4).map (fun x => (f x.1, x.2)) =
      (rdU32 bs (pos + 1)).map (fun imm => (f imm, bs.drop (pos + 5))) := by
  obtain ⟨v, hv⟩ := rdU32_of_lt (bs := bs) (pos := pos + 1) (by omega)
  have hrdle : Sporos.rdLE (bs.drop (pos + 1)) 4 = some (v, bs.drop (pos + 5)) := by
    rw [rdLE4_drop_iff]; exact ⟨hv, by simp [Nat.add_assoc]⟩
  rw [hv, hrdle]; rfl

private theorem rdLE8_eq_rdU64_bind {bs : List Nat} {pos : Nat} {f : Nat → Instr}
    (hlen : pos + 9 ≤ bs.length) :
    (Sporos.rdLE (bs.drop (pos + 1)) 8).map (fun x => (f x.1, x.2)) =
      (rdU64 bs (pos + 1)).map (fun imm => (f imm, bs.drop (pos + 9))) := by
  obtain ⟨v, hv⟩ := rdU64_of_lt (bs := bs) (pos := pos + 1) (by omega)
  have hrdle : Sporos.rdLE (bs.drop (pos + 1)) 8 = some (v, bs.drop (pos + 9)) := by
    rw [rdLE8_drop_iff]; exact ⟨hv, by simp [Nat.add_assoc]⟩
  rw [hv, hrdle]; rfl

private theorem decInstr_imm4_eq {bs : List Nat} {pos op : Nat} {f : Nat → Instr}
    (hpos : bs[pos]? = some op) (hlen : instrLen op = some 5)
    (hf : ∀ imm, instrOfOpImm op imm = some (f imm)) :
    (Sporos.rdLE (bs.drop (pos + 1)) 4).map (fun x => (f x.1, x.2)) =
      (decFlatAt bs pos).map (fun (ins, pos', _, _) => (ins, bs.drop pos')) := by
  rw [decFlatAt_step hpos hlen]
  by_cases hle : pos + 5 ≤ bs.length
  · rw [if_pos hle, rdLE4_eq_rdU32_bind hle]
    simp only [show (5 == 5) = true from rfl, ↓reduceIte, bind, Option.bind, pure]
    cases hrd : rdU32 bs (pos + 1) with
    | none => rfl
    | some imm => simp [hf imm]
  · rw [if_neg hle]
    have hnone : Sporos.rdLE (bs.drop (pos + 1)) 4 = none := by
      unfold Sporos.rdLE; rw [if_pos (by simp; omega)]
    rw [hnone]; rfl

private theorem decInstr_imm8_eq {bs : List Nat} {pos op : Nat} {f : Nat → Instr}
    (hpos : bs[pos]? = some op) (hlen : instrLen op = some 9)
    (hf : ∀ imm, instrOfOpImm op imm = some (f imm)) :
    (Sporos.rdLE (bs.drop (pos + 1)) 8).map (fun x => (f x.1, x.2)) =
      (decFlatAt bs pos).map (fun (ins, pos', _, _) => (ins, bs.drop pos')) := by
  rw [decFlatAt_step hpos hlen]
  by_cases hle : pos + 9 ≤ bs.length
  · rw [if_pos hle, rdLE8_eq_rdU64_bind hle]
    simp only [show (9 == 5) = false from rfl, show (9 == 9) = true from rfl,
      Bool.false_eq_true, ↓reduceIte, bind, Option.bind, pure]
    cases hrd : rdU64 bs (pos + 1) with
    | none => rfl
    | some imm => simp [hf imm]
  · rw [if_neg hle]
    have hnone : Sporos.rdLE (bs.drop (pos + 1)) 8 = none := by
      unfold Sporos.rdLE; rw [if_pos (by simp; omega)]
    rw [hnone]; rfl

private theorem decInstr_imm0_eq {bs : List Nat} {pos op : Nat} {ins : Instr}
    (hpos : bs[pos]? = some op) (hlen : instrLen op = some 1)
    (hlt : pos < bs.length) (hf : instrOfOpImm op 0 = some ins) :
    some (ins, bs.drop (pos + 1)) =
      (decFlatAt bs pos).map (fun (ins, pos', _, _) => (ins, bs.drop pos')) := by
  rw [decFlatAt_step hpos hlen, if_pos (by omega)]
  simp only [show (1 == 5) = false from rfl, show (1 == 9) = false from rfl,
    Bool.false_eq_true, ↓reduceIte, bind, Option.bind, pure, hf, Option.map_some]

theorem decInstr_drop_eq_decFlatAt (bs : List Nat) (pos : Nat) :
    Sporos.decInstr (bs.drop pos) =
      (decFlatAt bs pos).map fun (ins, pos', _, _) => (ins, bs.drop pos') := by
  cases hpos : bs[pos]? with
  | none =>
    have hge : bs.length ≤ pos := List.getElem?_eq_none_iff.1 hpos
    rw [List.drop_of_length_le hge]
    simp [Sporos.decInstr, decFlatAt, hpos]
  | some op =>
    have hlt : pos < bs.length := (List.getElem?_eq_some_iff.1 hpos).1
    have hdrop : bs.drop pos = op :: bs.drop (pos + 1) := by
      have : op = bs[pos] := (List.getElem?_eq_some_iff.1 hpos).2.symm
      rw [this]; exact List.drop_eq_getElem_cons hlt
    rw [hdrop]
    by_cases h57 : op < Prim.all.length
    · have h57' : op < 57 := by rwa [← primAll_len]
      have hlen : instrLen op = some 1 := by simp [instrLen, h57']
      have hf : instrOfOpImm op 0 = some (.prim (Prim.all[op]'h57)) := by
        simp [instrOfOpImm, h57]
      rw [Sporos.decInstr, dif_pos h57]
      exact decInstr_imm0_eq hpos hlen hlt hf
    · rw [decInstr_match_eq op (bs.drop (pos + 1)) h57]
      have hn57 : ¬ op < 57 := by rwa [← primAll_len]
      dsimp only
      by_cases h80 : op = 0x80; · subst h80; rw [if_pos rfl]; exact decInstr_imm8_eq hpos rfl (fun _ => rfl)
      rw [if_neg h80]
      by_cases h81 : op = 0x81; · subst h81; rw [if_pos rfl]; exact decInstr_imm4_eq hpos rfl (fun _ => rfl)
      rw [if_neg h81]
      by_cases h82 : op = 0x82; · subst h82; rw [if_pos rfl]; exact decInstr_imm4_eq hpos rfl (fun _ => rfl)
      rw [if_neg h82]
      by_cases h83 : op = 0x83; · subst h83; rw [if_pos rfl]; exact decInstr_imm0_eq hpos rfl hlt rfl
      rw [if_neg h83]
      by_cases h84 : op = 0x84; · subst h84; rw [if_pos rfl]; exact decInstr_imm0_eq hpos rfl hlt rfl
      rw [if_neg h84]
      by_cases h85 : op = 0x85; · subst h85; rw [if_pos rfl]; exact decInstr_imm0_eq hpos rfl hlt rfl
      rw [if_neg h85]
      by_cases h86 : op = 0x86; · subst h86; rw [if_pos rfl]; exact decInstr_imm4_eq hpos rfl (fun _ => rfl)
      rw [if_neg h86]
      by_cases h87 : op = 0x87; · subst h87; rw [if_pos rfl]; exact decInstr_imm4_eq hpos rfl (fun _ => rfl)
      rw [if_neg h87]
      by_cases h88 : op = 0x88; · subst h88; rw [if_pos rfl]; exact decInstr_imm4_eq hpos rfl (fun _ => rfl)
      rw [if_neg h88]
      by_cases h89 : op = 0x89; · subst h89; rw [if_pos rfl]; exact decInstr_imm4_eq hpos rfl (fun _ => rfl)
      rw [if_neg h89]
      by_cases h8a : op = 0x8a; · subst h8a; rw [if_pos rfl]; exact decInstr_imm4_eq hpos rfl (fun _ => rfl)
      rw [if_neg h8a]
      by_cases h8b : op = 0x8b; · subst h8b; rw [if_pos rfl]; exact decInstr_imm4_eq hpos rfl (fun _ => rfl)
      rw [if_neg h8b]
      by_cases h8c : op = 0x8c; · subst h8c; rw [if_pos rfl]; exact decInstr_imm0_eq hpos rfl hlt rfl
      rw [if_neg h8c]
      have hnone : instrLen op = none := by
        unfold instrLen
        simp [hn57, h80, h81, h82, h83, h84, h85, h86, h87, h88, h89, h8a, h8b, h8c]
      simp [decFlatAt, hpos, hnone]

theorem decInstr_drop_some {bs : List Nat} {pos : Nat} {ins : Instr} {rest : List Nat}
    (h : Sporos.decInstr (bs.drop pos) = some (ins, rest)) :
    ∃ op len imm,
      bs[pos]? = some op ∧
      instrLen op = some len ∧
      pos + len ≤ bs.length ∧
      skipInstr bs pos = some (pos + len) ∧
      (if len == 5 then rdU32 bs (pos + 1) else if len == 9 then rdU64 bs (pos + 1) else some 0) = some imm ∧
      instrOfOpImm op imm = some ins ∧
      rest = bs.drop (pos + len) := by
  rw [decInstr_drop_eq_decFlatAt] at h
  simp only [Option.map_eq_some_iff] at h
  obtain ⟨⟨ins', pos', op, imm⟩, hf, hEq⟩ := h
  cases hEq
  cases hpos : bs[pos]? with
  | none => simp [decFlatAt, hpos] at hf
  | some op' =>
    cases hlen : instrLen op' with
    | none => simp [decFlatAt, hpos, hlen] at hf
    | some len =>
      rw [decFlatAt_step hpos hlen] at hf
      split at hf
      · rename_i hle
        obtain ⟨hlens, _⟩ := instrLen_ok hlen
        rcases hlens with rfl | rfl | rfl
        · simp only [show (1 == 5) = false from rfl, show (1 == 9) = false from rfl,
            Bool.false_eq_true, ↓reduceIte, bind, Option.bind, pure] at hf
          cases hins : instrOfOpImm op' 0 with
          | none => rw [hins] at hf; cases hf
          | some ins'' =>
            rw [hins] at hf; cases hf
            exact ⟨op, 1, 0, rfl, hlen, hle, by rw [skipInstr_eq_if hpos hlen, if_pos hle],
              rfl, hins, rfl⟩
        · simp only [show (5 == 5) = true from rfl, ↓reduceIte, bind, Option.bind, pure] at hf
          cases hv : rdU32 bs (pos + 1) with
          | none => rw [hv] at hf; cases hf
          | some v =>
            rw [hv] at hf; dsimp only at hf
            cases hins : instrOfOpImm op' v with
            | none => rw [hins] at hf; cases hf
            | some ins'' =>
              rw [hins] at hf; cases hf
              exact ⟨op, 5, imm, rfl, hlen, hle, by rw [skipInstr_eq_if hpos hlen, if_pos hle],
                by simp, hins, rfl⟩
        · simp only [show (9 == 5) = false from rfl, show (9 == 9) = true from rfl,
            Bool.false_eq_true, ↓reduceIte, bind, Option.bind, pure] at hf
          cases hv : rdU64 bs (pos + 1) with
          | none => rw [hv] at hf; cases hf
          | some v =>
            rw [hv] at hf; dsimp only at hf
            cases hins : instrOfOpImm op' v with
            | none => rw [hins] at hf; cases hf
            | some ins'' =>
              rw [hins] at hf; cases hf
              exact ⟨op, 9, imm, rfl, hlen, hle, by rw [skipInstr_eq_if hpos hlen, if_pos hle],
                by simp, hins, rfl⟩
      · cases hf

theorem skipInstr_to_decInstr {bs : List Nat} {pos pos' : Nat} (h : skipInstr bs pos = some pos') :
    ∃ op len imm8 imm4 ins,
      bs[pos]? = some op ∧
      instrLen op = some len ∧
      pos' = pos + len ∧
      pos + len ≤ bs.length ∧
      (if len == 5 then rdU32 bs (pos + 1) else if len == 9 then rdU64 bs (pos + 1) else some 0) = some imm8 ∧
      (if len == 5 then rdU32 bs (pos + 1) else some 0) = some imm4 ∧
      instrOfOpImm op imm8 = some ins ∧
      Sporos.decInstr (bs.drop pos) = some (ins, bs.drop pos') := by
  simp only [skipInstr, bind, Option.bind] at h
  rcases hpos : bs[pos]? with _ | op
  · rw [hpos] at h; cases h
  · rw [hpos] at h; dsimp only at h
    rcases hlen : instrLen op with _ | len
    · rw [hlen] at h; cases h
    · rw [hlen] at h; dsimp only at h
      split at h
      · rename_i hle
        cases h
        obtain ⟨hlens, hins⟩ := instrLen_ok hlen
        have hdrop : Sporos.decInstr (bs.drop pos) =
            (decFlatAt bs pos).map (fun (ins, pos', _, _) => (ins, bs.drop pos')) :=
          decInstr_drop_eq_decFlatAt bs pos
        rcases hlens with rfl | rfl | rfl
        · obtain ⟨ins, hf⟩ := hins 0
          refine ⟨op, 1, 0, 0, ins, rfl, hlen, rfl, hle, rfl, rfl, hf, ?_⟩
          rw [hdrop, decFlatAt_step hpos hlen, if_pos hle]
          simp only [show (1 == 5) = false from rfl, show (1 == 9) = false from rfl,
            Bool.false_eq_true, ↓reduceIte, bind, Option.bind, pure, hf, Option.map_some]
        · obtain ⟨v, hv⟩ := rdU32_of_lt (by omega : (pos + 1) + 4 ≤ bs.length)
          obtain ⟨ins, hf⟩ := hins v
          refine ⟨op, 5, v, v, ins, rfl, hlen, rfl, hle, by simp [hv], by simp [hv], hf, ?_⟩
          rw [hdrop, decFlatAt_step hpos hlen, if_pos hle]
          simp only [show (5 == 5) = true from rfl, ↓reduceIte, bind, Option.bind, pure, hv, hf, Option.map_some]
        · obtain ⟨v, hv⟩ := rdU64_of_lt (by omega : (pos + 1) + 8 ≤ bs.length)
          obtain ⟨ins, hf⟩ := hins v
          refine ⟨op, 9, v, 0, ins, rfl, hlen, rfl, hle, by simp [hv], rfl, hf, ?_⟩
          rw [hdrop, decFlatAt_step hpos hlen, if_pos hle]
          simp only [show (9 == 5) = false from rfl, show (9 == 9) = true from rfl,
            Bool.false_eq_true, ↓reduceIte, bind, Option.bind, pure, hv, hf, Option.map_some]
      · cases h

theorem skipCode_mono {bs : List Nat} :
    ∀ {n pos pos' : Nat}, skipCode bs n pos = some pos' → pos ≤ pos'
  | 0, _, _, h => by cases h; exact Nat.le_refl _
  | n + 1, pos, pos', h => by
    simp only [skipCode, bind, Option.bind] at h
    cases hs : skipInstr bs pos with
    | none => rw [hs] at h; cases h
    | some pos1 =>
      rw [hs] at h; dsimp only at h
      have h1 := (skipInstr_bounds hs).1
      have h2 := skipCode_mono h
      omega

theorem skipCode_le {bs : List Nat} :
    ∀ {n pos pos' : Nat}, pos ≤ bs.length → skipCode bs n pos = some pos' → pos' ≤ bs.length
  | 0, _, _, hle, h => by cases h; exact hle
  | n + 1, pos, pos', _, h => by
    simp only [skipCode, bind, Option.bind] at h
    cases hs : skipInstr bs pos with
    | none => rw [hs] at h; cases h
    | some pos1 =>
      rw [hs] at h; dsimp only at h
      exact skipCode_le (skipInstr_bounds hs).2 h

theorem skipCode_to_decMany {bs : List Nat} :
    ∀ {n pos pos' : Nat}, pos ≤ bs.length → skipCode bs n pos = some pos' →
      ∃ code, Sporos.decMany Sporos.decInstr n (bs.drop pos) = some (code, bs.drop pos') ∧
        code.length = n ∧ pos ≤ pos' ∧ pos' ≤ bs.length
  | 0, pos, pos', hle, h => by
    cases h
    exact ⟨[], rfl, rfl, Nat.le_refl _, hle⟩
  | n + 1, pos, pos', _, h => by
    simp only [skipCode, bind, Option.bind] at h
    cases hs : skipInstr bs pos with
    | none => rw [hs] at h; cases h
    | some pos1 =>
      rw [hs] at h; dsimp only at h
      obtain ⟨_, _, _, _, ins, _, _, rfl, hlen1, _, _, _, hdec⟩ := skipInstr_to_decInstr hs
      obtain ⟨h1, hle1⟩ := skipInstr_bounds hs
      obtain ⟨code, hmany, hclen, hmono, hle'⟩ := skipCode_to_decMany hle1 h
      refine ⟨ins :: code, ?_, by simp [hclen], by omega, hle'⟩
      simp only [Sporos.decMany, bind, Option.bind, pure, hdec, hmany]

theorem decMany_decInstr_to_skipCode {bs : List Nat} :
    ∀ {n pos : Nat} {code : List Instr} {rest : List Nat},
      pos ≤ bs.length →
      Sporos.decMany Sporos.decInstr n (bs.drop pos) = some (code, rest) →
      ∃ pos', skipCode bs n pos = some pos' ∧ rest = bs.drop pos' ∧
        code.length = n ∧ pos ≤ pos' ∧ pos' ≤ bs.length
  | 0, pos, code, rest, hle, h => by
    cases h
    exact ⟨pos, rfl, rfl, rfl, Nat.le_refl _, hle⟩
  | n + 1, pos, code, rest, _, h => by
    simp only [Sporos.decMany, bind, Option.bind, pure] at h
    cases hd : Sporos.decInstr (bs.drop pos) with
    | none => rw [hd] at h; cases h
    | some p =>
      rw [hd] at h; dsimp only at h
      rcases p with ⟨ins, r1⟩
      obtain ⟨op, len, imm, _, _, hle1, hskip, _, _, rfl⟩ := decInstr_drop_some hd
      cases hm : Sporos.decMany Sporos.decInstr n (bs.drop (pos + len)) with
      | none => rw [hm] at h; cases h
      | some p2 =>
        rw [hm] at h; dsimp only at h
        cases h
        rcases p2 with ⟨code', r2⟩
        obtain ⟨pos', hsc, rfl, hclen, hmono, hle'⟩ := decMany_decInstr_to_skipCode hle1 hm
        refine ⟨pos', ?_, rfl, by simp [hclen], by omega, hle'⟩
        simp only [skipCode, bind, Option.bind, hskip, hsc]

theorem skipData_mono {bs : List Nat} :
    ∀ {n pos pos' : Nat}, skipData bs n pos = some pos' → pos ≤ pos'
  | 0, _, _, h => by cases h; exact Nat.le_refl _
  | n + 1, pos, pos', h => by
    simp only [skipData, bind, Option.bind] at h
    cases hrd : rdU32 bs pos with
    | none => rw [hrd] at h; cases h
    | some len =>
      rw [hrd] at h; dsimp only at h
      split at h
      · have := skipData_mono h; omega
      · cases h

theorem skipData_le {bs : List Nat} :
    ∀ {n pos pos' : Nat}, pos ≤ bs.length → skipData bs n pos = some pos' → pos' ≤ bs.length
  | 0, _, _, hle, h => by cases h; exact hle
  | n + 1, pos, pos', _, h => by
    simp only [skipData, bind, Option.bind] at h
    cases hrd : rdU32 bs pos with
    | none => rw [hrd] at h; cases h
    | some len =>
      rw [hrd] at h; dsimp only at h
      split at h
      · rename_i hle'
        exact skipData_le hle' h
      · cases h

theorem decData_drop_iff (bs : List Nat) (pos : Nat) (d rest : List Nat) :
    Sporos.decData (bs.drop pos) = some (d, rest) ↔
      ∃ len, rdU32 bs pos = some len ∧ pos + 4 + len ≤ bs.length ∧
        d = (bs.drop (pos + 4)).take len ∧ rest = bs.drop (pos + 4 + len) := by
  unfold Sporos.decData
  simp only [bind, Option.bind, pure]
  cases hrd : Sporos.rdLE (bs.drop pos) 4 with
  | none =>
    constructor
    · intro h; cases h
    · rintro ⟨len, hu, _⟩
      have h' : Sporos.rdLE (bs.drop pos) 4 = some (len, bs.drop (pos + 4)) :=
        (rdLE4_drop_iff bs pos len (bs.drop (pos + 4))).2 ⟨hu, rfl⟩
      rw [hrd] at h'; cases h'
  | some p =>
    rcases p with ⟨len, r1⟩
    have hpair := (rdLE4_drop_iff bs pos len r1).1 hrd
    rcases hpair with ⟨hu, rfl⟩
    have hb4 := rdU32_bounds hu
    simp only [List.length_drop]
    by_cases hlt : pos + 4 + len ≤ bs.length
    · have hnl : ¬ (bs.length - (pos + 4) < len) := by omega
      rw [if_neg hnl]
      simp only [Option.some.injEq, Prod.mk.injEq, List.drop_drop]
      constructor
      · rintro ⟨rfl, rfl⟩
        exact ⟨len, hu, hlt, rfl, rfl⟩
      · rintro ⟨len', hu', _, hd, hr⟩
        rw [hu] at hu'; cases hu'
        exact ⟨hd.symm, by rw [hr]⟩
    · have hlt' : bs.length - (pos + 4) < len := by omega
      rw [if_pos hlt']
      constructor
      · intro h; cases h
      · rintro ⟨len', hu', hle', _⟩
        rw [hu] at hu'; cases hu'
        exact False.elim (hlt hle')

theorem skipData_to_decMany {bs : List Nat} :
    ∀ {n pos pos' : Nat}, pos ≤ bs.length → skipData bs n pos = some pos' →
      ∃ data, Sporos.decMany Sporos.decData n (bs.drop pos) = some (data, bs.drop pos') ∧
        data.length = n ∧ pos ≤ pos' ∧ pos' ≤ bs.length
  | 0, pos, pos', hle, h => by
    cases h
    exact ⟨[], rfl, rfl, Nat.le_refl _, hle⟩
  | n + 1, pos, pos', _, h => by
    simp only [skipData, bind, Option.bind] at h
    cases hrd : rdU32 bs pos with
    | none => rw [hrd] at h; cases h
    | some len =>
      rw [hrd] at h; dsimp only at h
      split at h
      · rename_i hle1
        have hdd : Sporos.decData (bs.drop pos) =
            some ((bs.drop (pos + 4)).take len, bs.drop (pos + 4 + len)) :=
          (decData_drop_iff bs pos _ _).2 ⟨len, hrd, hle1, rfl, rfl⟩
        obtain ⟨data, hmany, hdlen, hm, hle'⟩ := skipData_to_decMany hle1 h
        refine ⟨(bs.drop (pos + 4)).take len :: data, ?_, by simp [hdlen], by omega, hle'⟩
        simp only [Sporos.decMany, bind, Option.bind, pure, hdd, hmany]
      · cases h

theorem decMany_decData_to_skipData {bs : List Nat} :
    ∀ {n pos : Nat} {data : List (List Nat)} {rest : List Nat},
      pos ≤ bs.length →
      Sporos.decMany Sporos.decData n (bs.drop pos) = some (data, rest) →
      ∃ pos', skipData bs n pos = some pos' ∧ rest = bs.drop pos' ∧
        data.length = n ∧ pos ≤ pos' ∧ pos' ≤ bs.length
  | 0, pos, data, rest, hle, h => by
    cases h
    exact ⟨pos, rfl, rfl, rfl, Nat.le_refl _, hle⟩
  | n + 1, pos, data, rest, _, h => by
    simp only [Sporos.decMany, bind, Option.bind, pure] at h
    cases hd : Sporos.decData (bs.drop pos) with
    | none => rw [hd] at h; cases h
    | some p =>
      rw [hd] at h; dsimp only at h
      rcases p with ⟨d, r1⟩
      obtain ⟨len, hu, hle1, _, rfl⟩ := (decData_drop_iff bs pos d r1).1 hd
      cases hm : Sporos.decMany Sporos.decData n (bs.drop (pos + 4 + len)) with
      | none => rw [hm] at h; cases h
      | some p2 =>
        rw [hm] at h; dsimp only at h
        cases h
        rcases p2 with ⟨data', r2⟩
        obtain ⟨pos', hsd, rfl, hdlen, hmono, hle'⟩ := decMany_decData_to_skipData hle1 hm
        refine ⟨pos', ?_, rfl, by simp [hdlen], by omega, hle'⟩
        simp only [skipData, bind, Option.bind, hu, if_pos hle1, hsd]

theorem decAnn_drop_eq (bs : List Nat) (pos : Nat) (ann : Nat × Height) (rest : List Nat) :
    Sporos.decAnn (bs.drop pos) = some (ann, rest) ↔
      rdU32 bs pos = some ann.1 ∧ rdU32 bs (pos + 4) = some ann.2.d ∧
        rdU32 bs (pos + 8) = some ann.2.a ∧ rest = bs.drop (pos + 12) := by
  unfold Sporos.decAnn
  simp only [bind, Option.bind, pure]
  constructor
  · intro h
    cases hr1 : Sporos.rdLE (bs.drop pos) 4 with
    | none => rw [hr1] at h; cases h
    | some p1 =>
      rw [hr1] at h; dsimp only at h
      rcases p1 with ⟨t, r1⟩
      obtain ⟨hu1, rfl⟩ := (rdLE4_drop_iff bs pos t r1).1 hr1
      cases hr2 : Sporos.rdLE (bs.drop (pos + 4)) 4 with
      | none => rw [hr2] at h; cases h
      | some p2 =>
        rw [hr2] at h; dsimp only at h
        rcases p2 with ⟨d, r2⟩
        obtain ⟨hu2, rfl⟩ := (rdLE4_drop_iff bs (pos + 4) d r2).1 hr2
        cases hr3 : Sporos.rdLE (bs.drop (pos + 4 + 4)) 4 with
        | none => rw [hr3] at h; cases h
        | some p3 =>
          rw [hr3] at h; dsimp only at h
          rcases p3 with ⟨a, r3⟩
          obtain ⟨hu3, rfl⟩ := (rdLE4_drop_iff bs (pos + 4 + 4) a r3).1 hr3
          cases h
          exact ⟨hu1, hu2, hu3, rfl⟩
  · rintro ⟨hu1, hu2, hu3, hr⟩
    have hr1 : Sporos.rdLE (bs.drop pos) 4 = some (ann.1, bs.drop (pos + 4)) :=
      (rdLE4_drop_iff bs pos ann.1 _).2 ⟨hu1, rfl⟩
    have hr2 : Sporos.rdLE (bs.drop (pos + 4)) 4 = some (ann.2.d, bs.drop (pos + 8)) :=
      (rdLE4_drop_iff bs (pos + 4) ann.2.d _).2 ⟨hu2, rfl⟩
    have hr3 : Sporos.rdLE (bs.drop (pos + 8)) 4 = some (ann.2.a, bs.drop (pos + 12)) :=
      (rdLE4_drop_iff bs (pos + 8) ann.2.a _).2 ⟨hu3, rfl⟩
    rw [hr1]; dsimp only
    rw [hr2]; dsimp only
    rw [hr3]; dsimp only
    rw [hr]

theorem decMany_decAnn_of_le (bs : List Nat) :
    ∀ (k pos : Nat), pos + k * 12 ≤ bs.length →
      ∃ anns, Sporos.decMany Sporos.decAnn k (bs.drop pos) = some (anns, bs.drop (pos + k * 12)) ∧
        anns.length = k
  | 0, pos, _ => ⟨[], rfl, rfl⟩
  | k + 1, pos, hle => by
    obtain ⟨t, ht⟩ := rdU32_of_lt (bs := bs) (pos := pos) (by omega)
    obtain ⟨d, hd⟩ := rdU32_of_lt (bs := bs) (pos := pos + 4) (by omega)
    obtain ⟨a, ha⟩ := rdU32_of_lt (bs := bs) (pos := pos + 8) (by omega)
    have hann : Sporos.decAnn (bs.drop pos) = some ((t, ⟨d, a⟩), bs.drop (pos + 12)) :=
      (decAnn_drop_eq bs pos (t, ⟨d, a⟩) _).2 ⟨ht, hd, ha, rfl⟩
    obtain ⟨anns, hmany, hlen⟩ := decMany_decAnn_of_le bs k (pos + 12) (by omega)
    refine ⟨(t, ⟨d, a⟩) :: anns, ?_, by simp [hlen]⟩
    simp only [Sporos.decMany, bind, Option.bind, pure, hann, hmany]
    rw [show pos + 12 + k * 12 = pos + (k + 1) * 12 by omega]

theorem decMany_decAnn_some {bs : List Nat} :
    ∀ {k pos : Nat} {anns : List (Nat × Height)} {rest : List Nat},
      pos ≤ bs.length →
      Sporos.decMany Sporos.decAnn k (bs.drop pos) = some (anns, rest) →
      pos + k * 12 ≤ bs.length ∧ rest = bs.drop (pos + k * 12) ∧ anns.length = k
  | 0, pos, anns, rest, hle, h => by
    cases h
    exact ⟨hle, rfl, rfl⟩
  | k + 1, pos, anns, rest, _, h => by
    simp only [Sporos.decMany, bind, Option.bind, pure] at h
    cases ha : Sporos.decAnn (bs.drop pos) with
    | none => rw [ha] at h; cases h
    | some p1 =>
      rw [ha] at h; dsimp only at h
      rcases p1 with ⟨ann, r1⟩
      obtain ⟨hu1, hu2, hu3, rfl⟩ := (decAnn_drop_eq bs pos ann r1).1 ha
      have hb12 := rdU32_bounds hu3
      cases hm : Sporos.decMany Sporos.decAnn k (bs.drop (pos + 12)) with
      | none => rw [hm] at h; cases h
      | some p2 =>
        rw [hm] at h; dsimp only at h
        cases h
        rcases p2 with ⟨anns', r2⟩
        obtain ⟨hle', rfl, hlen⟩ := decMany_decAnn_some (by omega) hm
        refine ⟨by omega, ?_, by simp [hlen]⟩
        rw [show pos + (k + 1) * 12 = pos + 12 + k * 12 by omega]

def WordSliceMatches (bs : List Nat) (wm : WordSlice) (wd : Word) : Prop :=
  wd.ins = wm.ins ∧
  wd.outs = wm.outs ∧
  wd.locals = wm.locals ∧
  wd.code.length = wm.nc ∧
  wd.ann.length = wm.na ∧
  wm.codeOff ≤ wm.annOff ∧
  wm.nextOff = wm.annOff + wm.na * 12 ∧
  wm.nextOff ≤ bs.length ∧
  Sporos.decMany Sporos.decInstr wm.nc (bs.drop wm.codeOff) = some (wd.code, bs.drop wm.annOff) ∧
  Sporos.decMany Sporos.decAnn wm.na (bs.drop wm.annOff) = some (wd.ann, bs.drop wm.nextOff)

theorem readWordAt_bounds {bs : List Nat} {pos : Nat} {wm : WordSlice}
    (h : readWordAt bs pos = some wm) :
    pos + 20 ≤ bs.length ∧ wm.codeOff = pos + 20 ∧ wm.codeOff ≤ wm.annOff ∧
      wm.nextOff = wm.annOff + wm.na * 12 ∧ wm.nextOff ≤ bs.length ∧ pos < wm.nextOff := by
  unfold readWordAt at h
  split at h
  · rename_i hle20
    simp only [bind, Option.bind] at h
    cases hi : rdU32 bs pos <;> rw [hi] at h; · cases h
    cases ho : rdU32 bs (pos + 4) <;> rw [ho] at h; · cases h
    cases hl : rdU32 bs (pos + 8) <;> rw [hl] at h; · cases h
    cases hc : rdU32 bs (pos + 12) <;> rw [hc] at h; · cases h
    cases ha : rdU32 bs (pos + 16) <;> rw [ha] at h; · cases h
    dsimp only at h
    cases hsc : skipCode bs _ (pos + 20) <;> rw [hsc] at h; · cases h
    dsimp only at h
    split at h
    · rename_i hna
      cases h
      have hmono := skipCode_mono hsc
      exact ⟨hle20, rfl, hmono, rfl, hna, by dsimp only; omega⟩
    · cases h
  · cases h

theorem readWordAt_to_decWord {bs : List Nat} {pos : Nat} {wm : WordSlice}
    (h : readWordAt bs pos = some wm) :
    ∃ wd, Sporos.decWord (bs.drop pos) = some (wd, bs.drop wm.nextOff) ∧
      WordSliceMatches bs wm wd := by
  unfold readWordAt at h
  split at h
  · rename_i hle20
    simp only [bind, Option.bind] at h
    cases hi : rdU32 bs pos <;> rw [hi] at h; · cases h
    cases ho : rdU32 bs (pos + 4) <;> rw [ho] at h; · cases h
    cases hl : rdU32 bs (pos + 8) <;> rw [hl] at h; · cases h
    cases hc : rdU32 bs (pos + 12) <;> rw [hc] at h; · cases h
    cases ha : rdU32 bs (pos + 16) <;> rw [ha] at h; · cases h
    dsimp only at h
    cases hsc : skipCode bs _ (pos + 20) <;> rw [hsc] at h; · cases h
    dsimp only at h
    split at h
    · rename_i hna
      cases h
      rename_i ins outs locals nc na annOff
      obtain ⟨code, hcode, hclen, hmono, _⟩ := skipCode_to_decMany (by omega) hsc
      obtain ⟨ann, hann, halen⟩ := decMany_decAnn_of_le bs na annOff hna
      have hr0 : Sporos.rdLE (bs.drop pos) 4 = some (ins, bs.drop (pos + 4)) :=
        (rdLE4_drop_iff bs pos ins _).2 ⟨hi, rfl⟩
      have hr1 : Sporos.rdLE (bs.drop (pos + 4)) 4 = some (outs, bs.drop (pos + 8)) :=
        (rdLE4_drop_iff bs (pos + 4) outs _).2 ⟨ho, rfl⟩
      have hr2 : Sporos.rdLE (bs.drop (pos + 8)) 4 = some (locals, bs.drop (pos + 12)) :=
        (rdLE4_drop_iff bs (pos + 8) locals _).2 ⟨hl, rfl⟩
      have hr3 : Sporos.rdLE (bs.drop (pos + 12)) 4 = some (nc, bs.drop (pos + 16)) :=
        (rdLE4_drop_iff bs (pos + 12) nc _).2 ⟨hc, rfl⟩
      have hr4 : Sporos.rdLE (bs.drop (pos + 16)) 4 = some (na, bs.drop (pos + 20)) :=
        (rdLE4_drop_iff bs (pos + 16) na _).2 ⟨ha, rfl⟩
      let wd : Word := ⟨ins, outs, locals, code, ann⟩
      refine ⟨wd, ?_, rfl, rfl, rfl, hclen, halen, hmono, rfl, hna, hcode, hann⟩
      simp only [Sporos.decWord, bind, Option.bind, pure, hr0, hr1, hr2, hr3, hr4, hcode, hann]
      rfl
    · cases h
  · cases h

theorem decWord_to_readWordAt {bs : List Nat} {pos : Nat} {wd : Word} {rest : List Nat}
    (_ : pos ≤ bs.length) (h : Sporos.decWord (bs.drop pos) = some (wd, rest)) :
    ∃ wm, readWordAt bs pos = some wm ∧ rest = bs.drop wm.nextOff ∧ WordSliceMatches bs wm wd := by
  unfold Sporos.decWord at h
  simp only [bind, Option.bind, pure] at h
  cases hr0 : Sporos.rdLE (bs.drop pos) 4 with
  | none => rw [hr0] at h; cases h
  | some p0 =>
    rw [hr0] at h; dsimp only at h
    rcases p0 with ⟨ins, r0⟩
    obtain ⟨hu0, rfl⟩ := (rdLE4_drop_iff bs pos ins r0).1 hr0
    cases hr1 : Sporos.rdLE (bs.drop (pos + 4)) 4 with
    | none => rw [hr1] at h; cases h
    | some p1 =>
      rw [hr1] at h; dsimp only at h
      rcases p1 with ⟨outs, r1⟩
      obtain ⟨hu1, rfl⟩ := (rdLE4_drop_iff bs (pos + 4) outs r1).1 hr1
      cases hr2 : Sporos.rdLE (bs.drop (pos + 8)) 4 with
      | none => rw [hr2] at h; cases h
      | some p2 =>
        rw [hr2] at h; dsimp only at h
        rcases p2 with ⟨locals, r2⟩
        obtain ⟨hu2, rfl⟩ := (rdLE4_drop_iff bs (pos + 8) locals r2).1 hr2
        cases hr3 : Sporos.rdLE (bs.drop (pos + 12)) 4 with
        | none => rw [hr3] at h; cases h
        | some p3 =>
          rw [hr3] at h; dsimp only at h
          rcases p3 with ⟨nc, r3⟩
          obtain ⟨hu3, rfl⟩ := (rdLE4_drop_iff bs (pos + 12) nc r3).1 hr3
          cases hr4 : Sporos.rdLE (bs.drop (pos + 16)) 4 with
          | none => rw [hr4] at h; cases h
          | some p4 =>
            rw [hr4] at h; dsimp only at h
            rcases p4 with ⟨na, r4⟩
            obtain ⟨hu4, rfl⟩ := (rdLE4_drop_iff bs (pos + 16) na r4).1 hr4
            have hle20 : pos + 20 ≤ bs.length := by
              have := rdU32_bounds hu4; omega
            cases hc : Sporos.decMany Sporos.decInstr nc (bs.drop (pos + 20)) with
            | none => rw [hc] at h; cases h
            | some pc =>
              rw [hc] at h; dsimp only at h
              rcases pc with ⟨code, rc⟩
              obtain ⟨annOff, hsc, rfl, hclen, hmono, hle_ann⟩ :=
                decMany_decInstr_to_skipCode hle20 hc
              cases ha : Sporos.decMany Sporos.decAnn na (bs.drop annOff) with
              | none => rw [ha] at h; cases h
              | some pa =>
                rw [ha] at h; dsimp only at h
                cases h
                rcases pa with ⟨ann, ra⟩
                obtain ⟨hna, rfl, halen⟩ := decMany_decAnn_some (by omega) ha
                let wm : WordSlice := ⟨ins, outs, locals, nc, na, pos + 20, annOff, annOff + na * 12⟩
                refine ⟨wm, ?_, rfl, rfl, rfl, rfl, hclen, halen, hmono, rfl, hna, hc, ha⟩
                simp only [readWordAt, if_pos hle20, bind, Option.bind,
                  hu0, hu1, hu2, hu3, hu4, hsc, if_pos hna]
                rfl

theorem skipWords_step (bs : List Nat) (n pos : Nat) :
    skipWords bs (n + 1) pos =
      match readWordAt bs pos with
      | none => none
      | some wm => skipWords bs n wm.nextOff := by
  by_cases hle20 : pos + 20 ≤ bs.length
  · obtain ⟨v0, hv0⟩ := rdU32_of_lt (bs := bs) (pos := pos) (by omega)
    obtain ⟨v1, hv1⟩ := rdU32_of_lt (bs := bs) (pos := pos + 4) (by omega)
    obtain ⟨v2, hv2⟩ := rdU32_of_lt (bs := bs) (pos := pos + 8) (by omega)
    obtain ⟨v3, hv3⟩ := rdU32_of_lt (bs := bs) (pos := pos + 12) (by omega)
    obtain ⟨v4, hv4⟩ := rdU32_of_lt (bs := bs) (pos := pos + 16) (by omega)
    simp only [skipWords, readWordAt, if_pos hle20, bind, Option.bind, hv0, hv1, hv2, hv3, hv4]
    cases hsc : skipCode bs v3 (pos + 20) with
    | none => rfl
    | some annOff =>
      dsimp only
      by_cases hna : annOff + v4 * 12 ≤ bs.length
      · rw [if_pos hna, if_pos hna]
      · rw [if_neg hna, if_neg hna]
  · simp only [skipWords, readWordAt, if_neg hle20]

theorem skipWords_mono {bs : List Nat} :
    ∀ {n pos pos' : Nat}, skipWords bs n pos = some pos' → pos ≤ pos'
  | 0, _, _, h => by cases h; exact Nat.le_refl _
  | n + 1, pos, pos', h => by
    rw [skipWords_step] at h
    cases hr : readWordAt bs pos with
    | none => rw [hr] at h; cases h
    | some wm =>
      rw [hr] at h
      rcases readWordAt_bounds hr with ⟨_, _, _, _, _, h1⟩
      have h2 := skipWords_mono h
      omega

theorem skipWords_le {bs : List Nat} :
    ∀ {n pos pos' : Nat}, pos ≤ bs.length → skipWords bs n pos = some pos' → pos' ≤ bs.length
  | 0, _, _, hle, h => by cases h; exact hle
  | n + 1, pos, pos', _, h => by
    rw [skipWords_step] at h
    cases hr : readWordAt bs pos with
    | none => rw [hr] at h; cases h
    | some wm =>
      rw [hr] at h
      rcases readWordAt_bounds hr with ⟨_, _, _, _, hne, _⟩
      exact skipWords_le hne h

theorem skipWords_to_decMany {bs : List Nat} :
    ∀ {n pos pos' : Nat}, pos ≤ bs.length → skipWords bs n pos = some pos' →
      ∃ words, Sporos.decMany Sporos.decWord n (bs.drop pos) = some (words, bs.drop pos') ∧
        words.length = n ∧ pos ≤ pos' ∧ pos' ≤ bs.length
  | 0, pos, pos', hle, h => by
    cases h
    exact ⟨[], rfl, rfl, Nat.le_refl _, hle⟩
  | n + 1, pos, pos', _, h => by
    rw [skipWords_step] at h
    cases hr : readWordAt bs pos with
    | none => rw [hr] at h; cases h
    | some wm =>
      rw [hr] at h
      obtain ⟨wd, hdw, hm⟩ := readWordAt_to_decWord hr
      rcases readWordAt_bounds hr with ⟨_, _, _, _, hne, hlt⟩
      obtain ⟨words, hmany, hwlen, hmono, hle'⟩ := skipWords_to_decMany hne h
      refine ⟨wd :: words, ?_, by simp [hwlen], by omega, hle'⟩
      simp only [Sporos.decMany, bind, Option.bind, pure, hdw, hmany]

theorem decMany_decWord_to_skipWords {bs : List Nat} :
    ∀ {n pos : Nat} {words : List Word} {rest : List Nat},
      pos ≤ bs.length →
      Sporos.decMany Sporos.decWord n (bs.drop pos) = some (words, rest) →
      ∃ pos', skipWords bs n pos = some pos' ∧ rest = bs.drop pos' ∧
        words.length = n ∧ pos ≤ pos' ∧ pos' ≤ bs.length
  | 0, pos, words, rest, hle, h => by
    cases h
    exact ⟨pos, rfl, rfl, rfl, Nat.le_refl _, hle⟩
  | n + 1, pos, words, rest, hle, h => by
    simp only [Sporos.decMany, bind, Option.bind, pure] at h
    cases hd : Sporos.decWord (bs.drop pos) with
    | none => rw [hd] at h; cases h
    | some p =>
      rw [hd] at h; dsimp only at h
      rcases p with ⟨wd, r1⟩
      obtain ⟨wm, hr, rfl, hm⟩ := decWord_to_readWordAt hle hd
      rcases readWordAt_bounds hr with ⟨_, _, _, _, hne, hlt⟩
      cases hm2 : Sporos.decMany Sporos.decWord n (bs.drop wm.nextOff) with
      | none => rw [hm2] at h; cases h
      | some p2 =>
        rw [hm2] at h; dsimp only at h
        cases h
        rcases p2 with ⟨words', r2⟩
        obtain ⟨pos', hsw, rfl, hwlen, hmono, hle'⟩ := decMany_decWord_to_skipWords hne hm2
        refine ⟨pos', ?_, rfl, by simp [hwlen], by omega, hle'⟩
        rw [skipWords_step, hr]; exact hsw

private theorem bs_magic_cons_iff {bs : List Nat} :
    (bs.length ≥ 4 ∧ bs[0]? = some 0x53 ∧ bs[1]? = some 0x50 ∧ bs[2]? = some 0x52 ∧ bs[3]? = some 0x31) ↔
      ∃ tail, bs = 83 :: 80 :: 82 :: 49 :: tail ∧ tail = bs.drop 4 := by
  constructor
  · rintro ⟨hlen, h0, h1, h2, h3⟩
    rcases bs with _ | ⟨b0, _ | ⟨b1, _ | ⟨b2, _ | ⟨b3, tail⟩⟩⟩⟩ <;> simp at hlen
    simp at h0 h1 h2 h3
    subst h0 h1 h2 h3
    exact ⟨tail, rfl, rfl⟩
  · rintro ⟨tail, rfl, _⟩
    refine ⟨by simp, rfl, rfl, rfl, rfl⟩

theorem parse_to_decode {bs : List Nat} (hb : ∀ b ∈ bs, b < 256)
    {main nw ws : Nat} (hp : parse bs = some (main, nw, ws)) :
    ∃ words data,
      Sporos.decode bs = some ⟨words, data, main⟩ ∧
      words.length = nw ∧
      ws ≤ bs.length ∧
      skipWords bs nw ws = some bs.length ∧
      Sporos.decMany Sporos.decWord nw (bs.drop ws) = some (words, []) := by
  unfold parse at hp
  split at hp
  · cases hp
  rename_i hn16
  have h16 : 16 ≤ bs.length := by omega
  simp only [bind, Option.bind] at hp
  cases hm : rdU32 bs 0 <;> rw [hm] at hp; · cases hp
  dsimp only at hp
  split at hp; · cases hp
  rename_i magic hneq
  have hmag : magic = MAGIC := by
    simp [bne_iff_ne] at hneq
    exact hneq
  subst hmag
  obtain ⟨h0, h1, h2, h3⟩ := (rdU32_magic_iff hb).1 hm
  obtain ⟨tail, rfl, htail⟩ := bs_magic_cons_iff.1 ⟨by omega, h0, h1, h2, h3⟩
  cases hmain : rdU32 (83 :: 80 :: 82 :: 49 :: tail) 4 with
  | none => rw [hmain] at hp; cases hp
  | some main' =>
    rw [hmain] at hp; dsimp only at hp
    cases hnw : rdU32 (83 :: 80 :: 82 :: 49 :: tail) 8 with
    | none => rw [hnw] at hp; cases hp
    | some nw' =>
      rw [hnw] at hp; dsimp only at hp
      cases hnd : rdU32 (83 :: 80 :: 82 :: 49 :: tail) 12 with
      | none => rw [hnd] at hp; cases hp
      | some nd =>
        rw [hnd] at hp; dsimp only at hp
        cases hsd : skipData (83 :: 80 :: 82 :: 49 :: tail) nd 16 with
        | none => rw [hsd] at hp; cases hp
        | some ws' =>
          rw [hsd] at hp; dsimp only at hp
          cases hsw : skipWords (83 :: 80 :: 82 :: 49 :: tail) nw' ws' with
          | none => rw [hsw] at hp; cases hp
          | some endPos =>
            rw [hsw] at hp; dsimp only at hp
            split at hp
            · rename_i hend
              cases hp
              have hend' : endPos = (83 :: 80 :: 82 :: 49 :: tail).length := beq_iff_eq.1 hend
              subst hend'
              have hws_le := skipData_le h16 hsd
              obtain ⟨data, hdata, _⟩ := skipData_to_decMany h16 hsd
              obtain ⟨words, hwords, hwlen, _⟩ := skipWords_to_decMany hws_le hsw
              have hr1 : Sporos.rdLE tail 4 =
                  some (main, (83 :: 80 :: 82 :: 49 :: tail).drop 8) := by
                rw [htail]
                exact (rdLE4_drop_iff (83 :: 80 :: 82 :: 49 :: tail) 4 main _).2 ⟨hmain, rfl⟩
              have hr2 : Sporos.rdLE ((83 :: 80 :: 82 :: 49 :: tail).drop 8) 4 =
                  some (nw, (83 :: 80 :: 82 :: 49 :: tail).drop 12) :=
                (rdLE4_drop_iff (83 :: 80 :: 82 :: 49 :: tail) 8 nw _).2 ⟨hnw, rfl⟩
              have hr3 : Sporos.rdLE ((83 :: 80 :: 82 :: 49 :: tail).drop 12) 4 =
                  some (nd, (83 :: 80 :: 82 :: 49 :: tail).drop 16) :=
                (rdLE4_drop_iff (83 :: 80 :: 82 :: 49 :: tail) 12 nd _).2 ⟨hnd, rfl⟩
              have hdrop_len :
                  (83 :: 80 :: 82 :: 49 :: tail).drop (83 :: 80 :: 82 :: 49 :: tail).length = [] :=
                List.drop_length
              rw [hdrop_len] at hwords
              refine ⟨words, data, ?_, hwlen, hws_le, hsw, hwords⟩
              unfold Sporos.decode
              simp only [bind, Option.bind]
              rw [hr1]; dsimp only
              rw [hr2]; dsimp only
              rw [hr3]; dsimp only
              rw [hdata]; dsimp only
              rw [hwords]; dsimp only
              rfl
            · cases hp

theorem decode_to_parse {bs : List Nat} {pr : Prog} (hd : Sporos.decode bs = some pr) :
    ∃ ws,
      parse bs = some (pr.main, pr.words.length, ws) ∧
      ws ≤ bs.length ∧
      skipWords bs pr.words.length ws = some bs.length ∧
      Sporos.decMany Sporos.decWord pr.words.length (bs.drop ws) = some (pr.words, []) := by
  unfold Sporos.decode at hd
  split at hd
  · rename_i tail
    simp only [bind, Option.bind, pure] at hd
    have htail : tail = (83 :: 80 :: 82 :: 49 :: tail).drop 4 := rfl
    cases hr1 : Sporos.rdLE tail 4 with
    | none => rw [hr1] at hd; cases hd
    | some p1 =>
      rw [hr1] at hd; dsimp only at hd
      rcases p1 with ⟨main, r1⟩
      have hr1' : Sporos.rdLE ((83 :: 80 :: 82 :: 49 :: tail).drop 4) 4 = some (main, r1) := hr1
      obtain ⟨hu1, rfl⟩ := (rdLE4_drop_iff (83 :: 80 :: 82 :: 49 :: tail) 4 main r1).1 hr1'
      cases hr2 : Sporos.rdLE ((83 :: 80 :: 82 :: 49 :: tail).drop 8) 4 with
      | none => rw [hr2] at hd; cases hd
      | some p2 =>
        rw [hr2] at hd; dsimp only at hd
        rcases p2 with ⟨nw, r2⟩
        obtain ⟨hu2, rfl⟩ := (rdLE4_drop_iff (83 :: 80 :: 82 :: 49 :: tail) 8 nw r2).1 hr2
        cases hr3 : Sporos.rdLE ((83 :: 80 :: 82 :: 49 :: tail).drop 12) 4 with
        | none => rw [hr3] at hd; cases hd
        | some p3 =>
          rw [hr3] at hd; dsimp only at hd
          rcases p3 with ⟨nd, r3⟩
          obtain ⟨hu3, rfl⟩ := (rdLE4_drop_iff (83 :: 80 :: 82 :: 49 :: tail) 12 nd r3).1 hr3
          have h16 : 16 ≤ (83 :: 80 :: 82 :: 49 :: tail).length := rdU32_bounds hu3
          cases hdata : Sporos.decMany Sporos.decData nd ((83 :: 80 :: 82 :: 49 :: tail).drop 16) with
          | none => rw [hdata] at hd; cases hd
          | some pd =>
            rw [hdata] at hd; dsimp only at hd
            rcases pd with ⟨data, r4⟩
            obtain ⟨ws, hsd, rfl, _, _, hws_le⟩ := decMany_decData_to_skipData h16 hdata
            cases hwords : Sporos.decMany Sporos.decWord nw ((83 :: 80 :: 82 :: 49 :: tail).drop ws) with
            | none => rw [hwords] at hd; cases hd
            | some pw =>
              rw [hwords] at hd; dsimp only at hd
              rcases pw with ⟨words, r5⟩
              obtain ⟨endPos, hsw, rfl, hwlen, _, hend_le⟩ := decMany_decWord_to_skipWords hws_le hwords
              split at hd
              · rename_i heq
                cases hd
                have hemp : ((83 :: 80 :: 82 :: 49 :: tail).drop endPos).length = 0 := by
                  dsimp only at heq
                  rw [List.isEmpty_iff.1 heq]; rfl
                have hend_eq : endPos = (83 :: 80 :: 82 :: 49 :: tail).length := by
                  simp only [List.length_drop] at hemp; omega
                subst hend_eq
                have hdrop_len :
                    (83 :: 80 :: 82 :: 49 :: tail).drop (83 :: 80 :: 82 :: 49 :: tail).length = [] :=
                  List.drop_length
                rw [hdrop_len] at hwords
                have hm0 : rdU32 (83 :: 80 :: 82 :: 49 :: tail) 0 = some MAGIC := by
                  simp [rdU32, MAGIC]
                refine ⟨ws, ?_, hws_le, ?_, ?_⟩
                · simp only [parse, if_neg (by omega : ¬ (83 :: 80 :: 82 :: 49 :: tail).length < 16),
                    bind, Option.bind, hm0, bne_self_eq_false, Bool.false_eq_true, ↓reduceIte,
                    hu1, hu2, hu3, hsd, hsw, hwlen, beq_self_eq_true]
                · simpa [hwlen] using hsw
                · simpa [hwlen] using hwords
              · cases hd
  · cases hd

theorem decode_none_of_parse_none {bs : List Nat} (hp : parse bs = none) :
    Sporos.decode bs = none := by
  cases hd : Sporos.decode bs with
  | none => rfl
  | some pr =>
    obtain ⟨_, hp', _⟩ := decode_to_parse hd
    rw [hp] at hp'; cases hp'

/-! ### Block C: Semantic equivalence (`wordAt`, `lookup`, `hstepF`, `walk`, `checkWords`) -/

theorem wordAt_of_decMany {bs : List Nat} :
    ∀ {nw ws : Nat} {words : List Word} {rest : List Nat},
      ws ≤ bs.length →
      Sporos.decMany Sporos.decWord nw (bs.drop ws) = some (words, rest) →
      ∀ {k : Nat}, k < nw →
        ∃ wm wd,
          wordAt bs ws k = some wm ∧
          words[k]? = some wd ∧
          WordSliceMatches bs wm wd
  | 0, _, _, _, _, _, k, hk => by omega
  | nw + 1, ws, words, rest, hle, h, k, hk => by
    simp only [Sporos.decMany, bind, Option.bind, pure] at h
    cases hd : Sporos.decWord (bs.drop ws) with
    | none => rw [hd] at h; cases h
    | some p =>
      rw [hd] at h; dsimp only at h
      rcases p with ⟨wd0, r1⟩
      obtain ⟨wm0, hr0, rfl, hm0⟩ := decWord_to_readWordAt hle hd
      have hle_next := (readWordAt_bounds hr0).2.2.2.2.1
      cases hm : Sporos.decMany Sporos.decWord nw (bs.drop wm0.nextOff) with
      | none => rw [hm] at h; cases h
      | some p2 =>
        rw [hm] at h; dsimp only at h
        cases h
        rcases p2 with ⟨words', r2⟩
        cases k with
        | zero =>
          exact ⟨wm0, wd0, by simp [wordAt, hr0], rfl, hm0⟩
        | succ k' =>
          obtain ⟨wm, wd, hwa, hget, hm'⟩ := wordAt_of_decMany hle_next hm (by omega : k' < nw)
          refine ⟨wm, wd, ?_, hget, hm'⟩
          simp [wordAt, hr0, hwa]

theorem wordAt_spec {bs : List Nat} {nw ws k : Nat} {words : List Word} {rest : List Nat} {wm : WordSlice}
    (hle : ws ≤ bs.length)
    (hwords : Sporos.decMany Sporos.decWord nw (bs.drop ws) = some (words, rest))
    (hk : k < nw) (hwa : wordAt bs ws k = some wm) :
    ∃ wd, words[k]? = some wd ∧ WordSliceMatches bs wm wd := by
  obtain ⟨wm', wd, hwa', hget, hm⟩ := wordAt_of_decMany hle hwords hk
  rw [hwa] at hwa'; cases hwa'
  exact ⟨wd, hget, hm⟩

theorem lookupLoop_of_decMany {bs : List Nat} (t : Nat) :
    ∀ {k off : Nat} {anns : List (Nat × Height)} {rest : List Nat},
      off ≤ bs.length →
      Sporos.decMany Sporos.decAnn k (bs.drop off) = some (anns, rest) →
      ((lookupLoop bs t k off).map (fun p => Height.mk p.1 p.2)) = anns.lookup t
  | 0, _, anns, _, _, h => by cases h; rfl
  | k + 1, off, anns, rest, _, h => by
    simp only [Sporos.decMany, bind, Option.bind, pure] at h
    cases ha : Sporos.decAnn (bs.drop off) with
    | none => rw [ha] at h; cases h
    | some p1 =>
      rw [ha] at h; dsimp only at h
      rcases p1 with ⟨ann, r1⟩
      obtain ⟨hu1, hu2, hu3, rfl⟩ := (decAnn_drop_eq bs off ann r1).1 ha
      have hle12 := rdU32_bounds hu3
      cases hm : Sporos.decMany Sporos.decAnn k (bs.drop (off + 12)) with
      | none => rw [hm] at h; cases h
      | some p2 =>
        rw [hm] at h; dsimp only at h
        cases h
        rcases p2 with ⟨anns', r2⟩
        simp only [lookupLoop, bind, Option.bind, hu1]
        by_cases heq : ann.1 = t
        · have hb1 : (ann.1 == t) = true := beq_iff_eq.2 heq
          have hb2 : (t == ann.1) = true := beq_iff_eq.2 heq.symm
          simp [hb1, hb2, hu2, hu3, List.lookup]
        · have hb1 : (ann.1 == t) = false := beq_false_of_ne heq
          have hb2 : (t == ann.1) = false := beq_false_of_ne (Ne.symm heq)
          simp only [hb1, hb2, Bool.false_eq_true, ↓reduceIte, List.lookup]
          exact lookupLoop_of_decMany t hle12 hm

theorem lookup_of_matches {bs : List Nat} {wm : WordSlice} {wd : Word}
    (hm : WordSliceMatches bs wm wd) (t : Nat) :
    ((lookup bs wm.annOff wm.na t).map (fun p => Height.mk p.1 p.2)) = wd.ann.lookup t := by
  have hle : wm.annOff ≤ bs.length := by rcases hm with ⟨_, _, _, _, _, _, hne, hleb, _, _⟩; omega
  exact lookupLoop_of_decMany t hle hm.2.2.2.2.2.2.2.2.2

theorem lookup_eq_some_iff {bs : List Nat} {wm : WordSlice} {wd : Word}
    (hm : WordSliceMatches bs wm wd) (t ad aa : Nat) :
    lookup bs wm.annOff wm.na t = some (ad, aa) ↔ wd.ann.lookup t = some ⟨ad, aa⟩ := by
  have h := lookup_of_matches hm t
  cases hl : lookup bs wm.annOff wm.na t with
  | none =>
    rw [hl] at h
    simp only [Option.map_none] at h
    rw [← h]
    constructor <;> (intro hc; cases hc)
  | some p =>
    rw [hl] at h
    simp only [Option.map_some] at h
    rw [← h]
    constructor
    · intro hs; cases hs; rfl
    · intro hs; cases hs; rfl

theorem lookup_eq_none_iff {bs : List Nat} {wm : WordSlice} {wd : Word}
    (hm : WordSliceMatches bs wm wd) (t : Nat) :
    lookup bs wm.annOff wm.na t = none ↔ wd.ann.lookup t = none := by
  have h := lookup_of_matches hm t
  cases hl : lookup bs wm.annOff wm.na t <;> rw [hl] at h <;> simp at h <;> simp [← h]

theorem annsOK_aux {bs : List Nat} {wm : WordSlice} {wd : Word} (hm : WordSliceMatches bs wm wd) :
    ∀ {k off : Nat} {anns : List (Nat × Height)} {rest : List Nat},
      off ≤ bs.length →
      Sporos.decMany Sporos.decAnn k (bs.drop off) = some (anns, rest) →
      (annsOK bs wm.nc wm.annOff wm.na k off = true ↔
        ∀ p ∈ anns, p.1 < wd.code.length ∧ wd.ann.lookup p.1 = some p.2)
  | 0, _, anns, _, _, h => by cases h; simp [annsOK]
  | k + 1, off, anns, rest, _, h => by
    simp only [Sporos.decMany, bind, Option.bind, pure] at h
    cases ha : Sporos.decAnn (bs.drop off) with
    | none => rw [ha] at h; cases h
    | some p1 =>
      rw [ha] at h; dsimp only at h
      rcases p1 with ⟨ann, r1⟩
      obtain ⟨hu1, hu2, hu3, rfl⟩ := (decAnn_drop_eq bs off ann r1).1 ha
      have hle12 := rdU32_bounds hu3
      cases hrem : Sporos.decMany Sporos.decAnn k (bs.drop (off + 12)) with
      | none => rw [hrem] at h; cases h
      | some p2 =>
        rw [hrem] at h; dsimp only at h
        cases h
        rcases p2 with ⟨anns', r2⟩
        have ih := annsOK_aux hm hle12 hrem
        simp only [annsOK, hu1, hu2, hu3, Bool.and_eq_true, decide_eq_true_eq, beq_iff_eq]
        rw [ih, hm.2.2.2.1, lookup_eq_some_iff hm ann.1 ann.2.d ann.2.a]
        constructor
        · rintro ⟨⟨ht, hl⟩, hall⟩ p hp
          rcases List.mem_cons.1 hp with rfl | hp'
          · exact ⟨ht, by rwa [show (⟨p.2.d, p.2.a⟩ : Height) = p.2 from rfl] at hl⟩
          · exact hall p hp'
        · intro hall
          have h0 := hall ann (by simp)
          exact ⟨⟨h0.1, h0.2⟩, fun p hp => hall p (by simp [hp])⟩

theorem annsOK_spec {bs : List Nat} {wm : WordSlice} {wd : Word} (hm : WordSliceMatches bs wm wd) :
    annsOK bs wm.nc wm.annOff wm.na wm.na wm.annOff = true ↔
      ∀ p ∈ wd.ann, p.1 < wd.code.length ∧ wd.ann.lookup p.1 = some p.2 := by
  have hle : wm.annOff ≤ bs.length := by rcases hm with ⟨_, _, _, _, _, _, hne, hleb, _, _⟩; omega
  exact annsOK_aux hm hle hm.2.2.2.2.2.2.2.2.2

theorem infer_go_cons (pr : Prog) (wd : Word) (ins : Instr) (rest : List Instr) (i : Nat) (cur : Height) :
    Sporos.infer.go pr wd (ins :: rest) i cur =
      let h := (wd.ann.lookup i).getD cur
      let next := match Sporos.hstep pr wd h ins with
        | some (some h') => h'
        | _ => (wd.ann.lookup (i + 1)).getD ⟨0, 0⟩
      h :: Sporos.infer.go pr wd rest (i + 1) next := by
  simp only [Sporos.infer.go]
  rfl

theorem infer_go_getD_idemp (pr : Prog) (wd : Word) :
    ∀ (rest : List Instr) (i : Nat) (cur : Height),
      Sporos.infer.go pr wd rest i ((wd.ann.lookup i).getD cur) =
        Sporos.infer.go pr wd rest i cur
  | [], _, _ => rfl
  | ins :: rest, i, cur => by
    rw [infer_go_cons, infer_go_cons]
    cases wd.ann.lookup i <;> rfl

theorem infer_go_cons_zero (pr : Prog) (wd : Word) (ins : Instr) (rest : List Instr) (i : Nat) (cur : Height) :
    Sporos.infer.go pr wd (ins :: rest) i cur =
      let h := (wd.ann.lookup i).getD cur
      let next := match Sporos.hstep pr wd h ins with
        | some (some h') => h'
        | _ => ⟨0, 0⟩
      h :: Sporos.infer.go pr wd rest (i + 1) next := by
  rw [infer_go_cons]
  dsimp only
  cases Sporos.hstep pr wd ((wd.ann.lookup i).getD cur) ins with
  | none => rw [infer_go_getD_idemp]
  | some m =>
    cases m with
    | none => rw [infer_go_getD_idemp]
    | some h' => rfl

theorem infer_go_length (pr : Prog) (wd : Word) :
    ∀ (code : List Instr) (i : Nat) (cur : Height),
      (Sporos.infer.go pr wd code i cur).length = code.length
  | [], _, _ => rfl
  | ins :: rest, i, cur => by
    rw [infer_go_cons]; simp [infer_go_length pr wd rest]

theorem infer_go_at (pr : Prog) (wd : Word) :
    ∀ (code : List Instr) (i j : Nat) (cur a : Height),
      j < code.length →
      wd.ann.lookup (i + j) = some a →
      (Sporos.infer.go pr wd code i cur)[j]? = some a
  | [], _, _, _, _, hj, _ => by simp at hj
  | ins :: rest, i, 0, cur, a, _, hlk => by
    rw [infer_go_cons]
    simp only [Nat.add_zero] at hlk
    simp [hlk]
  | ins :: rest, i, j + 1, cur, a, hj, hlk => by
    rw [infer_go_cons]
    have hj' : j < rest.length := by simp at hj; omega
    have hlk' : wd.ann.lookup ((i + 1) + j) = some a := by
      rwa [show (i + 1) + j = i + (j + 1) by omega]
    exact infer_go_at pr wd rest (i + 1) j _ a hj' hlk'

private theorem lookup_isSome_of_mem {α β : Type} [BEq α] [LawfulBEq α] :
    ∀ (l : List (α × β)) (k : α) (v : β), (k, v) ∈ l → ∃ v', l.lookup k = some v'
  | [], _, _, h => by cases h
  | (k', v') :: t, k, v, h => by
    by_cases heq : k = k'
    · subst heq; exact ⟨v', by simp [List.lookup]⟩
    · rcases List.mem_cons.1 h with h1 | h2
      · cases h1; exact False.elim (heq rfl)
      · obtain ⟨w, hw⟩ := lookup_isSome_of_mem t k v h2
        exact ⟨w, by simp [List.lookup, beq_false_of_ne heq, hw]⟩

theorem ann_all_iff_lookup (pr : Prog) (wd : Word) :
    (wd.ann.all fun (t, a) => decide (t < wd.code.length) && (Sporos.infer pr wd)[t]? == some a) = true ↔
      ∀ p ∈ wd.ann, p.1 < wd.code.length ∧ wd.ann.lookup p.1 = some p.2 := by
  simp only [List.all_eq_true, Bool.and_eq_true, decide_eq_true_eq, beq_iff_eq]
  constructor
  · intro h p hp
    rcases p with ⟨t, a⟩
    obtain ⟨ht, hat⟩ := h (t, a) hp
    obtain ⟨a', ha'⟩ := lookup_isSome_of_mem wd.ann t a hp
    have hinfer := infer_go_at pr wd wd.code 0 t ⟨wd.ins, 0⟩ a' ht (by simpa using ha')
    change (Sporos.infer pr wd)[t]? = some a' at hinfer
    rw [hat] at hinfer; cases hinfer
    exact ⟨ht, ha'⟩
  · intro h p hp
    rcases p with ⟨t, a⟩
    obtain ⟨ht, hl⟩ := h (t, a) hp
    have hinfer := infer_go_at pr wd wd.code 0 t ⟨wd.ins, 0⟩ a ht (by simpa using hl)
    exact ⟨ht, hinfer⟩

theorem primEff_of_lt {op : Nat} (h : op < 57) :
    primEff op = some (Prim.all[op]'(by rw [primAll_len]; exact h)).eff := by
  have hpe : op < primEffBytes.length := by rw [primEffBytes_len]; exact h
  have hall : ∀ i : Fin 57,
      (primEffBytes[i.1]'(by rw [primEffBytes_len]; exact i.2) / 16,
       primEffBytes[i.1]'(by rw [primEffBytes_len]; exact i.2) % 16) =
      (Prim.all[i.1]'(by rw [primAll_len]; exact i.2)).eff := by
    decide +kernel
  simp only [primEff, List.getElem?_eq_getElem hpe, Option.map_some]
  exact congrArg some (hall ⟨op, h⟩)

theorem primEff_none_of_ge {op : Nat} (h : ¬ op < 57) :
    primEff op = none := by
  have hnone : primEffBytes[op]? = none := List.getElem?_eq_none (by rw [primEffBytes_len]; omega)
  simp only [primEff, hnone, Option.map_none]

theorem prim_exit_iff {op : Nat} (h : op < 57) :
    Prim.all[op]'(by rw [primAll_len]; exact h) = Prim.exit ↔ op = 50 := by
  have hall : ∀ i : Fin 57,
      (Prim.all[i.1]'(by rw [primAll_len]; exact i.2) == Prim.exit) = (i.1 == 50) := by
    decide +kernel
  have hi := congrArg (· = true) (hall ⟨op, h⟩)
  simp only [beq_iff_eq] at hi
  exact iff_of_eq hi

theorem prim_eff_le {p : Prim} : p.eff.1 ≤ 4 ∧ p.eff.2 ≤ 3 := by
  have hall : ∀ i : Fin 57,
      (Prim.all[i.1]'(by rw [primAll_len]; exact i.2)).eff.1 ≤ 4 ∧
      (Prim.all[i.1]'(by rw [primAll_len]; exact i.2)).eff.2 ≤ 3 := by
    decide +kernel
  have hm := Sporos.Prim.mem_all p
  obtain ⟨k, hk, heq⟩ := List.getElem_of_mem hm
  have hk57 : k < 57 := by rw [← primAll_len]; exact hk
  have h := hall ⟨k, hk57⟩
  rwa [heq] at h

theorem readWordAt_u32_lt {bs : List Nat} (hb : ∀ b ∈ bs, b < 256)
    {pos : Nat} {wm : WordSlice} (h : readWordAt bs pos = some wm) :
    wm.ins < 2 ^ 32 ∧ wm.outs < 2 ^ 32 ∧ wm.locals < 2 ^ 32 ∧ wm.nc < 2 ^ 32 ∧ wm.na < 2 ^ 32 := by
  unfold readWordAt at h
  split at h
  · simp only [bind, Option.bind] at h
    cases hi : rdU32 bs pos <;> rw [hi] at h; · cases h
    cases ho : rdU32 bs (pos + 4) <;> rw [ho] at h; · cases h
    cases hl : rdU32 bs (pos + 8) <;> rw [hl] at h; · cases h
    cases hc : rdU32 bs (pos + 12) <;> rw [hc] at h; · cases h
    cases ha : rdU32 bs (pos + 16) <;> rw [ha] at h; · cases h
    dsimp only at h
    cases hsc : skipCode bs _ (pos + 20) <;> rw [hsc] at h; · cases h
    dsimp only at h
    split at h
    · cases h
      exact ⟨rdU32_lt hb hi, rdU32_lt hb ho, rdU32_lt hb hl, rdU32_lt hb hc, rdU32_lt hb ha⟩
    · cases h
  · cases h

theorem wordAt_u32_lt {bs : List Nat} (hb : ∀ b ∈ bs, b < 256) :
    ∀ {k ws : Nat} {wm : WordSlice}, wordAt bs ws k = some wm →
      wm.ins < 2 ^ 32 ∧ wm.outs < 2 ^ 32
  | 0, _, _, h => by
    have hu := readWordAt_u32_lt hb h
    exact ⟨hu.1, hu.2.1⟩
  | k + 1, ws, wm, h => by
    simp only [wordAt, bind, Option.bind] at h
    cases hr : readWordAt bs ws with
    | none => rw [hr] at h; cases h
    | some wm0 =>
      rw [hr] at h; dsimp only at h
      exact wordAt_u32_lt hb h

theorem lookupLoop_u32_lt {bs : List Nat} (hb : ∀ b ∈ bs, b < 256) :
    ∀ {t k off ad aa : Nat}, lookupLoop bs t k off = some (ad, aa) →
      ad < 2 ^ 32 ∧ aa < 2 ^ 32
  | _, 0, _, _, _, h => by cases h
  | t, k + 1, off, ad, aa, h => by
    simp only [lookupLoop, bind, Option.bind, pure] at h
    cases hi : rdU32 bs off with
    | none => rw [hi] at h; cases h
    | some idx =>
      rw [hi] at h; dsimp only at h
      split at h
      · cases hd : rdU32 bs (off + 4) <;> rw [hd] at h; · cases h
        cases ha : rdU32 bs (off + 8) <;> rw [ha] at h; · cases h
        cases h
        exact ⟨rdU32_lt hb hd, rdU32_lt hb ha⟩
      · exact lookupLoop_u32_lt hb h

theorem skipCode_add_le {bs : List Nat} :
    ∀ {n pos pos' : Nat}, skipCode bs n pos = some pos' → pos + n ≤ pos'
  | 0, _, _, h => by cases h; omega
  | n + 1, pos, pos', h => by
    simp only [skipCode, bind, Option.bind] at h
    cases hs : skipInstr bs pos with
    | none => rw [hs] at h; cases h
    | some pos1 =>
      rw [hs] at h; dsimp only at h
      have h1 := (skipInstr_bounds hs).1
      have h2 := skipCode_add_le h
      omega

theorem readWordAt_nc_le {bs : List Nat} {pos : Nat} {wm : WordSlice}
    (h : readWordAt bs pos = some wm) : wm.nc ≤ bs.length := by
  unfold readWordAt at h
  split at h
  · simp only [bind, Option.bind] at h
    cases hi : rdU32 bs pos <;> rw [hi] at h; · cases h
    cases ho : rdU32 bs (pos + 4) <;> rw [ho] at h; · cases h
    cases hl : rdU32 bs (pos + 8) <;> rw [hl] at h; · cases h
    cases hc : rdU32 bs (pos + 12) <;> rw [hc] at h; · cases h
    cases ha : rdU32 bs (pos + 16) <;> rw [ha] at h; · cases h
    dsimp only at h
    cases hsc : skipCode bs _ (pos + 20) <;> rw [hsc] at h; · cases h
    dsimp only at h
    split at h
    · rename_i hna
      cases h
      have hadd := skipCode_add_le hsc
      dsimp only
      omega
    · cases h
  · cases h

def walkSpec (pr : Prog) (wd : Word) : List Instr → Nat → Height → Bool
  | [], _, _ => true
  | ins :: rest, i, cur =>
    let h := (wd.ann.lookup i).getD cur
    match Sporos.hstep pr wd h ins with
    | none => false
    | some none => walkSpec pr wd rest (i + 1) ⟨0, 0⟩
    | some (some h') =>
      decide (i + 1 < wd.code.length) &&
      (match wd.ann.lookup (i + 1) with
       | none => true
       | some a => a == h') &&
      walkSpec pr wd rest (i + 1) h'

theorem walkSpec_iff_stepAll (pr : Prog) (wd : Word) :
    ∀ (code : List Instr) (i : Nat) (cur : Height),
      code = wd.code.drop i →
      (walkSpec pr wd code i cur = true ↔
        ∀ off, off < code.length →
          (match code[off]?, (Sporos.infer.go pr wd code i cur)[off]? with
           | some ins, some h =>
             match Sporos.hstep pr wd h ins with
             | none => false
             | some none => true
             | some (some h') =>
               decide (i + off + 1 < wd.code.length) &&
                 (Sporos.infer.go pr wd code i cur)[off + 1]? == some h'
           | _, _ => false) = true)
  | [], _, _, _ => by simp [walkSpec]
  | ins :: rest, i, cur, hdrop => by
    have hrest : rest = wd.code.drop (i + 1) := by
      have := congrArg List.tail hdrop
      simpa [List.tail_drop, Nat.add_comm 1 i] using this
    have hlen : rest.length + i + 1 = wd.code.length := by
      have hl := congrArg List.length hdrop
      simp only [List.length_cons, List.length_drop] at hl
      have hi : i < wd.code.length := by
        by_cases hge : wd.code.length ≤ i
        · have he : wd.code.drop i = [] := List.drop_of_length_le hge
          rw [← hdrop] at he; cases he
        · omega
      omega
    have ih := walkSpec_iff_stepAll pr wd rest (i + 1)
    rw [walkSpec, infer_go_cons_zero]
    cases hs : Sporos.hstep pr wd ((wd.ann.lookup i).getD cur) ins with
    | none =>
      simp only [Bool.false_eq_true, false_iff]
      intro hall
      have h0 := hall 0 (by simp)
      simp [hs] at h0
    | some m =>
      cases m with
      | none =>
        have ih0 := ih ⟨0, 0⟩ hrest
        dsimp only
        rw [ih0]
        constructor
        · intro h1 off hoff
          cases off with
          | zero => simp [hs]
          | succ off' =>
            have h1' := h1 off' (by simp at hoff; omega)
            simpa [hs, Nat.add_assoc, Nat.add_left_comm] using h1'
        · intro h1 off' hoff'
          have h1' := h1 (off' + 1) (by simp; omega)
          simpa [hs, Nat.add_assoc, Nat.add_left_comm] using h1'
      | some h' =>
        have ih' := ih h' hrest
        dsimp only
        rw [Bool.and_eq_true, Bool.and_eq_true, decide_eq_true_eq, ih']
        constructor
        · rintro ⟨⟨hi1, hnext⟩, hall⟩ off hoff
          cases off with
          | zero =>
            simp only [List.getElem?_cons_zero, hs, Nat.add_zero, Bool.and_eq_true, decide_eq_true_eq]
            cases rest with
            | nil => simp at hlen; omega
            | cons ins1 rest1 =>
              rw [infer_go_cons_zero]
              simp only [List.getElem?_cons_succ, List.getElem?_cons_zero]
              refine ⟨hi1, ?_⟩
              cases hlk : wd.ann.lookup (i + 1) with
              | none => simp
              | some a =>
                rw [hlk] at hnext
                simp only [Option.getD_some]
                simpa [beq_iff_eq] using hnext
          | succ off' =>
            have hall' := hall off' (by simp at hoff; omega)
            simpa [hs, Nat.add_assoc, Nat.add_left_comm] using hall'
        · intro hall
          have h0 := hall 0 (by simp)
          simp only [List.getElem?_cons_zero, hs, Nat.add_zero, Bool.and_eq_true,
            decide_eq_true_eq, beq_iff_eq] at h0
          rcases h0 with ⟨hi1, hsuc⟩
          refine ⟨⟨hi1, ?_⟩, fun off' hoff' => ?_⟩
          · cases hrest_eq : rest with
            | nil => rw [hrest_eq] at hlen; simp at hlen; omega
            | cons ins1 rest1 =>
              rw [hrest_eq, infer_go_cons_zero] at hsuc
              simp only [Nat.zero_add, List.getElem?_cons_succ, List.getElem?_cons_zero, Option.some.injEq] at hsuc
              cases hlk : wd.ann.lookup (i + 1) with
              | none => rfl
              | some a =>
                rw [hlk] at hsuc
                exact beq_iff_eq.2 hsuc
          · have h1' := hall (off' + 1) (by simp; omega)
            simpa [hs, Nat.add_assoc, Nat.add_left_comm] using h1'

private theorem bool_eq_of_iff {a b : Bool} (h : a = true ↔ b = true) : a = b := by
  cases a <;> cases b <;> simp at h ⊢

private theorem height_beq_eq (d1 a1 d2 a2 : Nat) :
    (({ d := d1, a := a1 } : Height) == { d := d2, a := a2 }) = (d1 == d2 && a1 == a2) := by
  apply bool_eq_of_iff
  simp [beq_iff_eq]

theorem walkSpec_iff_checkWord (pr : Prog) (wd : Word) (hne : ¬ wd.code.isEmpty) :
    ((match wd.ann.lookup 0 with | none => true | some a => a == ⟨wd.ins, 0⟩) &&
      walkSpec pr wd wd.code 0 ⟨wd.ins, 0⟩ &&
      (wd.ann.all fun ⟨t, a⟩ => decide (t < wd.code.length) && (Sporos.infer pr wd)[t]? == some a)) = true ↔
      Sporos.checkWord pr wd (Sporos.infer pr wd) = true := by
  have hlen : ((Sporos.infer pr wd).length == wd.code.length) = true :=
    beq_iff_eq.2 (infer_go_length pr wd wd.code 0 ⟨wd.ins, 0⟩)
  have hcode : ∃ ins0 rest, wd.code = ins0 :: rest := by
    cases h : wd.code with
    | nil => simp [h] at hne
    | cons x xs => exact ⟨x, xs, rfl⟩
  rcases hcode with ⟨ins0, rest, hceq⟩
  have h0_eq : ((Sporos.infer pr wd)[0]? == some ⟨wd.ins, 0⟩) =
      match wd.ann.lookup 0 with | none => true | some a => a == ⟨wd.ins, 0⟩ := by
    rw [Sporos.infer, hceq, infer_go_cons_zero]
    cases hlk : wd.ann.lookup 0 with
    | none => simp
    | some a =>
      simp only [Option.getD_some, List.getElem?_cons_zero]
      apply bool_eq_of_iff
      simp [beq_iff_eq]
  have hstep_iff := walkSpec_iff_stepAll pr wd wd.code 0 ⟨wd.ins, 0⟩ rfl
  have hrange :
      ((List.range wd.code.length).all fun idx =>
        match wd.code[idx]?, (Sporos.infer pr wd)[idx]? with
        | some ins, some h =>
          match Sporos.hstep pr wd h ins with
          | none => false
          | some none => true
          | some (some h') => idx + 1 < wd.code.length && (Sporos.infer pr wd)[idx + 1]? == some h'
        | _, _ => false) = true ↔
      walkSpec pr wd wd.code 0 ⟨wd.ins, 0⟩ = true := by
    rw [hstep_iff]
    simp only [Sporos.infer, List.all_eq_true, List.mem_range, Nat.zero_add]
  have hneb : (!wd.code.isEmpty) = true := by simp [hne]
  simp only [Sporos.checkWord, hlen, hneb, h0_eq, Bool.true_and, Bool.and_eq_true]
  constructor
  · rintro ⟨⟨h1, h2⟩, h3⟩
    exact ⟨⟨h1, hrange.2 h2⟩, h3⟩
  · rintro ⟨⟨h1, h2⟩, h3⟩
    exact ⟨⟨h1, hrange.1 h2⟩, h3⟩

def hstepSound (res : StepRes) (sp : Option (Option Height)) : Prop :=
  match res with
  | .overflow => True
  | .reject => sp = none
  | .noFall => sp = some none
  | .next nd na => sp = some (some ⟨nd, na⟩) ∧ nd < HMAX ∧ na < HMAX

private theorem hstepF_80 (bs : List Nat) (nw ws outs locals annOff na hd ha imm : Nat) :
    hstepF bs nw ws outs locals annOff na hd ha 0x80 imm =
      if hd + 1 ≥ HMAX then .overflow else .next (hd + 1) ha := by
  simp only [hstepF, primEff_none_of_ge (by decide : ¬ 0x80 < 57)]; rfl

private theorem hstepF_81 (bs : List Nat) (nw ws outs locals annOff na hd ha imm : Nat) :
    hstepF bs nw ws outs locals annOff na hd ha 0x81 imm =
      if hd + 2 ≥ HMAX then .overflow else .next (hd + 2) ha := by
  simp only [hstepF, primEff_none_of_ge (by decide : ¬ 0x81 < 57)]; rfl

private theorem hstepF_82 (bs : List Nat) (nw ws outs locals annOff na hd ha imm : Nat) :
    hstepF bs nw ws outs locals annOff na hd ha 0x82 imm =
      if imm ≥ hd then .reject else if hd + 1 ≥ HMAX then .overflow else .next (hd + 1) ha := by
  simp only [hstepF, primEff_none_of_ge (by decide : ¬ 0x82 < 57)]; rfl

private theorem hstepF_83 (bs : List Nat) (nw ws outs locals annOff na hd ha imm : Nat) :
    hstepF bs nw ws outs locals annOff na hd ha 0x83 imm =
      if hd < 1 then .reject else if ha + 1 ≥ HMAX then .overflow else .next (hd - 1) (ha + 1) := by
  simp only [hstepF, primEff_none_of_ge (by decide : ¬ 0x83 < 57)]; rfl

private theorem hstepF_84 (bs : List Nat) (nw ws outs locals annOff na hd ha imm : Nat) :
    hstepF bs nw ws outs locals annOff na hd ha 0x84 imm =
      if ha < 1 then .reject else if hd + 1 ≥ HMAX then .overflow else .next (hd + 1) (ha - 1) := by
  simp only [hstepF, primEff_none_of_ge (by decide : ¬ 0x84 < 57)]; rfl

private theorem hstepF_85 (bs : List Nat) (nw ws outs locals annOff na hd ha imm : Nat) :
    hstepF bs nw ws outs locals annOff na hd ha 0x85 imm =
      if ha < 1 then .reject else if hd + 1 ≥ HMAX then .overflow else .next (hd + 1) ha := by
  simp only [hstepF, primEff_none_of_ge (by decide : ¬ 0x85 < 57)]; rfl

private theorem hstepF_86 (bs : List Nat) (nw ws outs locals annOff na hd ha imm : Nat) :
    hstepF bs nw ws outs locals annOff na hd ha 0x86 imm =
      if imm ≥ locals then .reject else if hd + 1 ≥ HMAX then .overflow else .next (hd + 1) ha := by
  simp only [hstepF, primEff_none_of_ge (by decide : ¬ 0x86 < 57)]; rfl

private theorem hstepF_87 (bs : List Nat) (nw ws outs locals annOff na hd ha imm : Nat) :
    hstepF bs nw ws outs locals annOff na hd ha 0x87 imm =
      if hd < 1 || imm ≥ locals then .reject else .next (hd - 1) ha := by
  simp only [hstepF, primEff_none_of_ge (by decide : ¬ 0x87 < 57)]; rfl

private theorem hstepF_88 (bs : List Nat) (nw ws outs locals annOff na hd ha imm : Nat) :
    hstepF bs nw ws outs locals annOff na hd ha 0x88 imm =
      match lookup bs annOff na imm with
      | some (ad, aa) => if ad == hd && aa == ha then .noFall else .reject
      | none => .reject := by
  simp only [hstepF, primEff_none_of_ge (by decide : ¬ 0x88 < 57)]; rfl

private theorem hstepF_89 (bs : List Nat) (nw ws outs locals annOff na hd ha imm : Nat) :
    hstepF bs nw ws outs locals annOff na hd ha 0x89 imm =
      if hd < 1 then .reject
      else match lookup bs annOff na imm with
        | some (ad, aa) => if ad == hd - 1 && aa == ha then .next (hd - 1) ha else .reject
        | none => .reject := by
  simp only [hstepF, primEff_none_of_ge (by decide : ¬ 0x89 < 57)]; rfl

private theorem hstepF_8a (bs : List Nat) (nw ws outs locals annOff na hd ha imm : Nat) :
    hstepF bs nw ws outs locals annOff na hd ha 0x8a imm =
      if hd < 1 then .reject
      else match lookup bs annOff na imm with
        | some (ad, aa) => if ad == hd - 1 && aa == ha then .next (hd - 1) ha else .reject
        | none => .reject := by
  simp only [hstepF, primEff_none_of_ge (by decide : ¬ 0x8a < 57)]; rfl

private theorem hstepF_8b (bs : List Nat) (nw ws outs locals annOff na hd ha imm : Nat) :
    hstepF bs nw ws outs locals annOff na hd ha 0x8b imm =
      if imm ≥ nw then .reject
      else match wordAt bs ws imm with
        | none => .reject
        | some c =>
          if hd < c.ins then .reject
          else if (hd - c.ins) + c.outs ≥ HMAX then .overflow else .next ((hd - c.ins) + c.outs) ha := by
  simp only [hstepF, primEff_none_of_ge (by decide : ¬ 0x8b < 57)]; rfl

private theorem hstepF_8c (bs : List Nat) (nw ws outs locals annOff na hd ha imm : Nat) :
    hstepF bs nw ws outs locals annOff na hd ha 0x8c imm =
      if hd == outs && ha == 0 then .noFall else .reject := by
  simp only [hstepF, primEff_none_of_ge (by decide : ¬ 0x8c < 57)]; rfl

theorem hstepSound_prim {bs : List Nat} {nw ws outs locals annOff na hd ha op imm : Nat}
    {pr : Prog} {wd : Word} (h57 : op < 57) (_hdlt : hd < HMAX) (halt : ha < HMAX) :
    hstepSound (hstepF bs nw ws outs locals annOff na hd ha op imm)
      (Sporos.hstep pr wd ⟨hd, ha⟩ (.prim (Prim.all[op]'(by rw [primAll_len]; exact h57)))) := by
  have hpe := primEff_of_lt h57
  unfold hstepF Sporos.hstep
  rw [hpe]; dsimp only
  split
  · rename_i hlt
    rw [if_neg (by omega)]; simp [hstepSound]
  · rename_i hnlt
    have hle : (Prim.all[op]'(by rw [primAll_len]; exact h57)).eff.1 ≤ hd := by omega
    rw [if_pos hle]
    by_cases h50 : op = 50
    · have hex : Prim.all[op]'(by rw [primAll_len]; exact h57) = Prim.exit := (prim_exit_iff h57).2 h50
      have h50b : (op == 50) = true := beq_iff_eq.2 h50
      rw [if_pos hex, if_pos h50b]; rfl
    · have hex : Prim.all[op]'(by rw [primAll_len]; exact h57) ≠ Prim.exit :=
        fun h => h50 ((prim_exit_iff h57).1 h)
      have h50b : (op == 50) = false := beq_false_of_ne h50
      rw [if_neg hex]
      simp only [h50b, Bool.false_eq_true, ↓reduceIte]
      split
      · simp [hstepSound]
      · rename_i hge
        exact ⟨rfl, by omega, halt⟩

theorem hstepSound_spec {bs : List Nat} {nw ws : Nat} {pr : Prog} {wm : WordSlice} {wd : Word}
    (hm : WordSliceMatches bs wm wd)
    (hwords : Sporos.decMany Sporos.decWord nw (bs.drop ws) = some (pr.words, []))
    (hws : ws ≤ bs.length) (hwlen : pr.words.length = nw)
    {p hd ha op ilen imm8 imm4 : Nat} {ins : Instr}
    (hdlt : hd < HMAX) (halt : ha < HMAX)
    (hlen : instrLen op = some ilen)
    (hi8 : (if ilen == 5 then rdU32 bs (p + 1) else if ilen == 9 then rdU64 bs (p + 1) else some 0) = some imm8)
    (hi4 : (if ilen == 5 then rdU32 bs (p + 1) else some 0) = some imm4)
    (hins : instrOfOpImm op imm8 = some ins) :
    hstepSound (hstepF bs nw ws wm.outs wm.locals wm.annOff wm.na hd ha op imm4)
      (Sporos.hstep pr wd ⟨hd, ha⟩ ins) := by
  by_cases h57 : op < 57
  · have h' : op < Prim.all.length := by rw [primAll_len]; exact h57
    simp only [instrOfOpImm, dif_pos h', Option.some.injEq] at hins
    subst hins
    exact hstepSound_prim h57 hdlt halt
  have him48 : ∀ (_ : ilen = 5), imm4 = imm8 := by
    intro h5; subst h5; simp only [show (5 == 5) = true from rfl, ↓reduceIte] at hi4 hi8
    rw [hi4] at hi8; cases hi8; rfl
  rcases hm with ⟨hins_w, houts_w, hloc_w, hclen, halen, hca, hnex, hleb, hdc, hda⟩
  have hm' : WordSliceMatches bs wm wd :=
    ⟨hins_w, houts_w, hloc_w, hclen, halen, hca, hnex, hleb, hdc, hda⟩
  by_cases h80 : op = 0x80
  · subst h80; simp only [instrOfOpImm, show ¬ (0x80 < Prim.all.length) by rw [primAll_len]; omega,
      ↓reduceDIte, Option.some.injEq] at hins; subst hins
    rw [hstepF_80]; unfold Sporos.hstep; split <;> simp [hstepSound] <;> exact ⟨by omega, halt⟩
  by_cases h81 : op = 0x81
  · subst h81; simp only [instrOfOpImm, show ¬ (0x81 < Prim.all.length) by rw [primAll_len]; omega,
      ↓reduceDIte, Option.some.injEq] at hins; subst hins
    rw [hstepF_81]; unfold Sporos.hstep; split <;> simp [hstepSound] <;> exact ⟨by omega, halt⟩
  by_cases h82 : op = 0x82
  · subst h82; simp only [instrLen] at hlen; cases hlen
    have heq := him48 rfl; subst heq
    simp only [instrOfOpImm, show ¬ (0x82 < Prim.all.length) by rw [primAll_len]; omega,
      ↓reduceDIte, Option.some.injEq] at hins; subst hins
    rw [hstepF_82]; unfold Sporos.hstep
    by_cases hge : imm4 ≥ hd
    · simp [hge, show ¬ (imm4 < hd) by omega, hstepSound]
    · simp only [hge, ↓reduceIte, show imm4 < hd by omega]
      split <;> simp [hstepSound] <;> exact ⟨by omega, halt⟩
  by_cases h83 : op = 0x83
  · subst h83; simp only [instrOfOpImm, show ¬ (0x83 < Prim.all.length) by rw [primAll_len]; omega,
      ↓reduceDIte, Option.some.injEq] at hins; subst hins
    rw [hstepF_83]; unfold Sporos.hstep
    by_cases h1 : hd < 1
    · simp [h1, show ¬ (1 ≤ hd) by omega, hstepSound]
    · simp only [h1, ↓reduceIte, show 1 ≤ hd by omega]
      split <;> simp [hstepSound] <;> exact ⟨by omega, by omega⟩
  by_cases h84 : op = 0x84
  · subst h84; simp only [instrOfOpImm, show ¬ (0x84 < Prim.all.length) by rw [primAll_len]; omega,
      ↓reduceDIte, Option.some.injEq] at hins; subst hins
    rw [hstepF_84]; unfold Sporos.hstep
    by_cases h1 : ha < 1
    · simp [h1, show ¬ (1 ≤ ha) by omega, hstepSound]
    · simp only [h1, ↓reduceIte, show 1 ≤ ha by omega]
      split <;> simp [hstepSound] <;> exact ⟨by omega, by omega⟩
  by_cases h85 : op = 0x85
  · subst h85; simp only [instrOfOpImm, show ¬ (0x85 < Prim.all.length) by rw [primAll_len]; omega,
      ↓reduceDIte, Option.some.injEq] at hins; subst hins
    rw [hstepF_85]; unfold Sporos.hstep
    by_cases h1 : ha < 1
    · simp [h1, show ¬ (1 ≤ ha) by omega, hstepSound]
    · simp only [h1, ↓reduceIte, show 1 ≤ ha by omega]
      split <;> simp [hstepSound] <;> exact ⟨by omega, halt⟩
  by_cases h86 : op = 0x86
  · subst h86; simp only [instrLen] at hlen; cases hlen
    have heq := him48 rfl; subst heq
    simp only [instrOfOpImm, show ¬ (0x86 < Prim.all.length) by rw [primAll_len]; omega,
      ↓reduceDIte, Option.some.injEq] at hins; subst hins
    rw [hstepF_86]; unfold Sporos.hstep
    by_cases hge : imm4 ≥ wm.locals
    · simp [hge, hloc_w, show ¬ (imm4 < wm.locals) by omega, hstepSound]
    · simp only [hge, ↓reduceIte, hloc_w, show imm4 < wm.locals by omega]
      split <;> simp [hstepSound] <;> exact ⟨by omega, halt⟩
  by_cases h87 : op = 0x87
  · subst h87; simp only [instrLen] at hlen; cases hlen
    have heq := him48 rfl; subst heq
    simp only [instrOfOpImm, show ¬ (0x87 < Prim.all.length) by rw [primAll_len]; omega,
      ↓reduceDIte, Option.some.injEq] at hins; subst hins
    rw [hstepF_87]
    by_cases hbad : hd < 1 ∨ imm4 ≥ wm.locals
    · have hb : (hd < 1 || imm4 ≥ wm.locals) = true := by
        simp only [Bool.or_eq_true, decide_eq_true_eq]; exact hbad
      rw [if_pos hb]
      simp [Sporos.hstep, hstepSound, show ¬ (1 ≤ hd ∧ imm4 < wd.locals) by omega]
    · have hb : (hd < 1 || imm4 ≥ wm.locals) = false := by
        simp only [Bool.or_eq_false_iff, decide_eq_false_iff_not]; omega
      rw [if_neg (by rw [hb]; decide)]
      simp [Sporos.hstep, hstepSound, show 1 ≤ hd ∧ imm4 < wd.locals by omega]
      exact ⟨by omega, halt⟩
  by_cases h88 : op = 0x88
  · subst h88; simp only [instrLen] at hlen; cases hlen
    have heq := him48 rfl; subst heq
    simp only [instrOfOpImm, show ¬ (0x88 < Prim.all.length) by rw [primAll_len]; omega,
      ↓reduceDIte, Option.some.injEq] at hins; subst hins
    rw [hstepF_88]; unfold Sporos.hstep
    cases hl : lookup bs wm.annOff wm.na imm4 with
    | none => simp [(lookup_eq_none_iff hm' imm4).1 hl, hstepSound]
    | some q =>
      rcases q with ⟨ad, aa⟩
      have hl' := (lookup_eq_some_iff hm' imm4 ad aa).1 hl
      by_cases he : ad = hd ∧ aa = ha
      · rcases he with ⟨rfl, rfl⟩; simp [hl', hstepSound]
      · have hb : (ad == hd && aa == ha) = false := by simp [beq_iff_eq]; omega
        have hne : (⟨ad, aa⟩ : Height) ≠ ⟨hd, ha⟩ := by intro h; cases h; exact he ⟨rfl, rfl⟩
        simp [hl', hb, hne, hstepSound]
  by_cases h89 : op = 0x89
  · subst h89; simp only [instrLen] at hlen; cases hlen
    have heq := him48 rfl; subst heq
    simp only [instrOfOpImm, show ¬ (0x89 < Prim.all.length) by rw [primAll_len]; omega,
      ↓reduceDIte, Option.some.injEq] at hins; subst hins
    rw [hstepF_89]; unfold Sporos.hstep
    by_cases h1 : hd < 1
    · simp [h1, show ¬ (1 ≤ hd) by omega, hstepSound]
    · simp only [h1, ↓reduceIte, show 1 ≤ hd by omega]
      cases hl : lookup bs wm.annOff wm.na imm4 with
      | none => simp [(lookup_eq_none_iff hm' imm4).1 hl, hstepSound]
      | some q =>
        rcases q with ⟨ad, aa⟩
        have hl' := (lookup_eq_some_iff hm' imm4 ad aa).1 hl
        by_cases he : ad = hd - 1 ∧ aa = ha
        · rcases he with ⟨rfl, rfl⟩; simp [hl', hstepSound]; exact ⟨by omega, halt⟩
        · have hb : (ad == hd - 1 && aa == ha) = false := by simp [beq_iff_eq]; omega
          have hne : (⟨ad, aa⟩ : Height) ≠ ⟨hd - 1, ha⟩ := by intro h; cases h; exact he ⟨rfl, rfl⟩
          simp [hl', hb, hne, hstepSound]
  by_cases h8a : op = 0x8a
  · subst h8a; simp only [instrLen] at hlen; cases hlen
    have heq := him48 rfl; subst heq
    simp only [instrOfOpImm, show ¬ (0x8a < Prim.all.length) by rw [primAll_len]; omega,
      ↓reduceDIte, Option.some.injEq] at hins; subst hins
    rw [hstepF_8a]; unfold Sporos.hstep
    by_cases h1 : hd < 1
    · simp [h1, show ¬ (1 ≤ hd) by omega, hstepSound]
    · simp only [h1, ↓reduceIte, show 1 ≤ hd by omega]
      cases hl : lookup bs wm.annOff wm.na imm4 with
      | none => simp [(lookup_eq_none_iff hm' imm4).1 hl, hstepSound]
      | some q =>
        rcases q with ⟨ad, aa⟩
        have hl' := (lookup_eq_some_iff hm' imm4 ad aa).1 hl
        by_cases he : ad = hd - 1 ∧ aa = ha
        · rcases he with ⟨rfl, rfl⟩; simp [hl', hstepSound]; exact ⟨by omega, halt⟩
        · have hb : (ad == hd - 1 && aa == ha) = false := by simp [beq_iff_eq]; omega
          have hne : (⟨ad, aa⟩ : Height) ≠ ⟨hd - 1, ha⟩ := by intro h; cases h; exact he ⟨rfl, rfl⟩
          simp [hl', hb, hne, hstepSound]
  by_cases h8b : op = 0x8b
  · subst h8b; simp only [instrLen] at hlen; cases hlen
    have heq := him48 rfl; subst heq
    simp only [instrOfOpImm, show ¬ (0x8b < Prim.all.length) by rw [primAll_len]; omega,
      ↓reduceDIte, Option.some.injEq] at hins; subst hins
    rw [hstepF_8b]; unfold Sporos.hstep
    by_cases hge : imm4 ≥ nw
    · have hnone : pr.words[imm4]? = none := List.getElem?_eq_none (by omega)
      simp [hge, hnone, hstepSound]
    · have hlt : imm4 < nw := by omega
      rw [if_neg hge]
      obtain ⟨cm, cd, hwa, hget, hcm⟩ := wordAt_of_decMany hws hwords hlt
      rcases hcm with ⟨hc_ins, hc_outs, _⟩
      rw [hwa]; dsimp only
      cases hpw : pr.words[imm4]? with
      | none => rw [hpw] at hget; cases hget
      | some cd' =>
        rw [hpw] at hget; cases hget
        dsimp only
        by_cases hbad : hd < cm.ins
        · simp [hbad, hc_ins, hstepSound]
        · simp only [hbad, ↓reduceIte, hc_ins, hc_outs]
          rw [if_pos (show cm.ins ≤ hd by omega)]
          split
          · simp [hstepSound]
          · simp [hstepSound]; exact ⟨by omega, halt⟩
  by_cases h8c : op = 0x8c
  · subst h8c; simp only [instrOfOpImm, show ¬ (0x8c < Prim.all.length) by rw [primAll_len]; omega,
      ↓reduceDIte, Option.some.injEq] at hins; subst hins
    rw [hstepF_8c]; unfold Sporos.hstep; rw [houts_w]
    by_cases he : hd = wm.outs ∧ ha = 0
    · rcases he with ⟨rfl, rfl⟩; simp [hstepSound]
    · have hb : (hd == wm.outs && ha == 0) = false := by simp [beq_iff_eq]; omega
      have hne : (⟨hd, ha⟩ : Height) ≠ ⟨wm.outs, 0⟩ := by intro h; cases h; exact he ⟨rfl, rfl⟩
      simp [hb, hne, hstepSound]
  exfalso
  unfold instrOfOpImm at hins
  have hn57 : ¬ op < Prim.all.length := by rw [primAll_len]; exact h57
  rw [dif_neg hn57] at hins
  split at hins <;> first | contradiction | cases hins | omega | simp_all

theorem hstepF_ne_overflow_of_lt {bs : List Nat} {nw ws : Nat} {pr : Prog} {wm : WordSlice} {wd : Word}
    (hm : WordSliceMatches bs wm wd)
    (hwords : Sporos.decMany Sporos.decWord nw (bs.drop ws) = some (pr.words, []))
    (hws : ws ≤ bs.length) (_hwlen : pr.words.length = nw)
    {p hd ha op ilen imm8 imm4 : Nat} {ins : Instr}
    (hlen : instrLen op = some ilen)
    (hi8 : (if ilen == 5 then rdU32 bs (p + 1) else if ilen == 9 then rdU64 bs (p + 1) else some 0) = some imm8)
    (hi4 : (if ilen == 5 then rdU32 bs (p + 1) else some 0) = some imm4)
    (hins : instrOfOpImm op imm8 = some ins)
    (hnext : ∀ h', Sporos.hstep pr wd ⟨hd, ha⟩ ins = some (some h') → h'.d < HMAX ∧ h'.a < HMAX) :
    hstepF bs nw ws wm.outs wm.locals wm.annOff wm.na hd ha op imm4 ≠ .overflow := by
  by_cases h57 : op < 57
  · have h' : op < Prim.all.length := by rw [primAll_len]; exact h57
    simp only [instrOfOpImm, dif_pos h', Option.some.injEq] at hins
    subst hins
    have hpe := primEff_of_lt h57
    unfold hstepF; rw [hpe]; dsimp only
    split; · simp
    by_cases h50 : op = 50
    · simp [h50]
    · have hex : Prim.all[op]'(by rw [primAll_len]; exact h57) ≠ Prim.exit :=
        fun h => h50 ((prim_exit_iff h57).1 h)
      rename_i hge
      have hle : (Prim.all[op]'(by rw [primAll_len]; exact h57)).eff.1 ≤ hd := by omega
      have hsp := hnext ⟨hd - (Prim.all[op]'(by rw [primAll_len]; exact h57)).eff.1 +
          (Prim.all[op]'(by rw [primAll_len]; exact h57)).eff.2, ha⟩
        (by simp [Sporos.hstep, hle, hex])
      have h50b : (op == 50) = false := beq_false_of_ne h50
      rw [if_neg (by simp [h50b]), if_neg (by dsimp only at hsp; omega)]; simp
  have him48 : ∀ (_ : ilen = 5), imm4 = imm8 := by
    intro h5; subst h5; simp only [show (5 == 5) = true from rfl, ↓reduceIte] at hi4 hi8
    rw [hi4] at hi8; cases hi8; rfl
  rcases hm with ⟨_, _, hloc_w, _, _, _, _, _, _, _⟩
  by_cases h80 : op = 0x80
  · subst h80; simp only [instrOfOpImm, show ¬ (0x80 < Prim.all.length) by rw [primAll_len]; omega,
      ↓reduceDIte, Option.some.injEq] at hins; subst hins
    have hsp := hnext ⟨hd + 1, ha⟩ rfl
    rw [hstepF_80, if_neg (by dsimp at hsp; omega)]; simp
  by_cases h81 : op = 0x81
  · subst h81; simp only [instrOfOpImm, show ¬ (0x81 < Prim.all.length) by rw [primAll_len]; omega,
      ↓reduceDIte, Option.some.injEq] at hins; subst hins
    have hsp := hnext ⟨hd + 2, ha⟩ rfl
    rw [hstepF_81, if_neg (by dsimp at hsp; omega)]; simp
  by_cases h82 : op = 0x82
  · subst h82; simp only [instrLen] at hlen; cases hlen
    have heq := him48 rfl; subst heq
    simp only [instrOfOpImm, show ¬ (0x82 < Prim.all.length) by rw [primAll_len]; omega,
      ↓reduceDIte, Option.some.injEq] at hins; subst hins
    rw [hstepF_82]
    by_cases hge : imm4 ≥ hd
    · simp [hge]
    · have hsp := hnext ⟨hd + 1, ha⟩ (by simp [Sporos.hstep, show imm4 < hd by omega])
      rw [if_neg hge, if_neg (by dsimp at hsp; omega)]; simp
  by_cases h83 : op = 0x83
  · subst h83; simp only [instrOfOpImm, show ¬ (0x83 < Prim.all.length) by rw [primAll_len]; omega,
      ↓reduceDIte, Option.some.injEq] at hins; subst hins
    rw [hstepF_83]
    by_cases h1 : hd < 1
    · simp [h1]
    · have hsp := hnext ⟨hd - 1, ha + 1⟩ (by simp [Sporos.hstep, show 1 ≤ hd by omega])
      rw [if_neg h1, if_neg (by dsimp at hsp; omega)]; simp
  by_cases h84 : op = 0x84
  · subst h84; simp only [instrOfOpImm, show ¬ (0x84 < Prim.all.length) by rw [primAll_len]; omega,
      ↓reduceDIte, Option.some.injEq] at hins; subst hins
    rw [hstepF_84]
    by_cases h1 : ha < 1
    · simp [h1]
    · have hsp := hnext ⟨hd + 1, ha - 1⟩ (by simp [Sporos.hstep, show 1 ≤ ha by omega])
      rw [if_neg h1, if_neg (by dsimp at hsp; omega)]; simp
  by_cases h85 : op = 0x85
  · subst h85; simp only [instrOfOpImm, show ¬ (0x85 < Prim.all.length) by rw [primAll_len]; omega,
      ↓reduceDIte, Option.some.injEq] at hins; subst hins
    rw [hstepF_85]
    by_cases h1 : ha < 1
    · simp [h1]
    · have hsp := hnext ⟨hd + 1, ha⟩ (by simp [Sporos.hstep, show 1 ≤ ha by omega])
      rw [if_neg h1, if_neg (by dsimp at hsp; omega)]; simp
  by_cases h86 : op = 0x86
  · subst h86; simp only [instrLen] at hlen; cases hlen
    have heq := him48 rfl; subst heq
    simp only [instrOfOpImm, show ¬ (0x86 < Prim.all.length) by rw [primAll_len]; omega,
      ↓reduceDIte, Option.some.injEq] at hins; subst hins
    rw [hstepF_86]
    by_cases hge : imm4 ≥ wm.locals
    · simp [hge]
    · have hsp := hnext ⟨hd + 1, ha⟩ (by simp [Sporos.hstep, hloc_w, show imm4 < wm.locals by omega])
      rw [if_neg hge, if_neg (by dsimp at hsp; omega)]; simp
  by_cases h87 : op = 0x87
  · subst h87; rw [hstepF_87]; split <;> simp
  by_cases h88 : op = 0x88
  · subst h88; rw [hstepF_88]; cases lookup bs wm.annOff wm.na imm4 <;> simp; split <;> simp
  by_cases h89 : op = 0x89
  · subst h89; rw [hstepF_89]; split <;> simp; cases lookup bs wm.annOff wm.na imm4 <;> simp; split <;> simp
  by_cases h8a : op = 0x8a
  · subst h8a; rw [hstepF_8a]; split <;> simp; cases lookup bs wm.annOff wm.na imm4 <;> simp; split <;> simp
  by_cases h8b : op = 0x8b
  · subst h8b; simp only [instrLen] at hlen; cases hlen
    have heq := him48 rfl; subst heq
    simp only [instrOfOpImm, show ¬ (0x8b < Prim.all.length) by rw [primAll_len]; omega,
      ↓reduceDIte, Option.some.injEq] at hins; subst hins
    rw [hstepF_8b]
    by_cases hge : imm4 ≥ nw
    · simp [hge]
    · have hlt : imm4 < nw := by omega
      obtain ⟨cm, cd, hwa, hget, hc⟩ := wordAt_of_decMany hws hwords hlt
      rcases hc with ⟨hc_ins, hc_outs, _⟩
      rw [if_neg hge, hwa]; dsimp only
      by_cases hbad : hd < cm.ins
      · simp [hbad]
      · have hsp := hnext ⟨hd - cm.ins + cm.outs, ha⟩ (by
          simp [Sporos.hstep, hget, hc_ins, hc_outs, show cm.ins ≤ hd by omega])
        rw [if_neg hbad, if_neg (by dsimp at hsp; omega)]; simp
  by_cases h8c : op = 0x8c
  · subst h8c; rw [hstepF_8c]; split <;> simp
  exfalso
  unfold instrOfOpImm at hins
  have hn57 : ¬ op < Prim.all.length := by rw [primAll_len]; exact h57
  rw [dif_neg hn57] at hins
  split at hins <;> first | contradiction | cases hins | omega | simp_all

theorem hstepF_no_overflow_of_small {bs : List Nat} (hb : ∀ b ∈ bs, b < 256)
    {nw ws outs locals annOff na hd ha op imm : Nat}
    (hdlt : hd + 2 ^ 32 ≤ HMAX) (halt : ha + 1 < HMAX) :
    hstepF bs nw ws outs locals annOff na hd ha op imm ≠ .overflow ∧
      ∀ nd na', hstepF bs nw ws outs locals annOff na hd ha op imm = .next nd na' →
        nd < hd + 2 ^ 32 ∧ na' ≤ ha + 1 := by
  by_cases h57 : op < 57
  · have hpe := primEff_of_lt h57
    unfold hstepF; rw [hpe]; dsimp only
    by_cases hlt : hd < (Prim.all[op]'(by rw [primAll_len]; exact h57)).eff.1
    · simp [hlt]
    · by_cases h50 : op = 50
      · rw [if_neg hlt]; simp [h50]
      · have h50b : (op == 50) = false := beq_false_of_ne h50
        have hp3 : (Prim.all[op]'(by rw [primAll_len]; exact h57)).eff.2 ≤ 3 := prim_eff_le.2
        have hd'_lt : (hd - (Prim.all[op]'(by rw [primAll_len]; exact h57)).eff.1) +
            (Prim.all[op]'(by rw [primAll_len]; exact h57)).eff.2 < HMAX := by
          unfold HMAX at *; omega
        simp [hlt, h50b, show ¬ ((hd - (Prim.all[op]'(by rw [primAll_len]; exact h57)).eff.1) +
            (Prim.all[op]'(by rw [primAll_len]; exact h57)).eff.2 ≥ HMAX) by omega]
        omega
  by_cases h80 : op = 0x80
  · subst h80; have : hd + 1 < HMAX := by unfold HMAX at *; omega
    rw [hstepF_80]; simp [show ¬ (hd + 1 ≥ HMAX) by omega]
  by_cases h81 : op = 0x81
  · subst h81; have : hd + 2 < HMAX := by unfold HMAX at *; omega
    rw [hstepF_81]; simp [show ¬ (hd + 2 ≥ HMAX) by omega]
  by_cases h82 : op = 0x82
  · subst h82; have : hd + 1 < HMAX := by unfold HMAX at *; omega
    rw [hstepF_82]; by_cases hge : imm ≥ hd <;> simp [hge, show ¬ (hd + 1 ≥ HMAX) by omega]
  by_cases h83 : op = 0x83
  · subst h83; rw [hstepF_83]
    by_cases h1 : hd < 1 <;> simp [h1, show ¬ (ha + 1 ≥ HMAX) by omega]; omega
  by_cases h84 : op = 0x84
  · subst h84; have : hd + 1 < HMAX := by unfold HMAX at *; omega
    rw [hstepF_84]; by_cases h1 : ha < 1 <;> simp [h1, show ¬ (hd + 1 ≥ HMAX) by omega]; omega
  by_cases h85 : op = 0x85
  · subst h85; have : hd + 1 < HMAX := by unfold HMAX at *; omega
    rw [hstepF_85]; by_cases h1 : ha < 1 <;> simp [h1, show ¬ (hd + 1 ≥ HMAX) by omega]
  by_cases h86 : op = 0x86
  · subst h86; have : hd + 1 < HMAX := by unfold HMAX at *; omega
    rw [hstepF_86]; by_cases hge : imm ≥ locals <;> simp [hge, show ¬ (hd + 1 ≥ HMAX) by omega]
  by_cases h87 : op = 0x87
  · subst h87; rw [hstepF_87]; split <;> simp; omega
  by_cases h88 : op = 0x88
  · subst h88; rw [hstepF_88]; cases lookup bs annOff na imm <;> simp; split <;> simp
  by_cases h89 : op = 0x89
  · subst h89; rw [hstepF_89]; split <;> simp; cases lookup bs annOff na imm <;> simp; split <;> simp; omega
  by_cases h8a : op = 0x8a
  · subst h8a; rw [hstepF_8a]; split <;> simp; cases lookup bs annOff na imm <;> simp; split <;> simp; omega
  by_cases h8b : op = 0x8b
  · subst h8b; rw [hstepF_8b]
    by_cases hge : imm ≥ nw
    · simp [hge]
    · simp only [hge, ↓reduceIte]
      cases hwa : wordAt bs ws imm with
      | none => simp
      | some c =>
        have hcu := (wordAt_u32_lt hb hwa).2
        by_cases hbad : hd < c.ins
        · simp [hbad]
        · have hd'_lt : (hd - c.ins) + c.outs < HMAX := by unfold HMAX at *; omega
          simp [hbad, show ¬ ((hd - c.ins) + c.outs ≥ HMAX) by omega]; omega
  by_cases h8c : op = 0x8c
  · subst h8c; rw [hstepF_8c]; split <;> simp
  have hpe := primEff_none_of_ge h57
  unfold hstepF
  simp [hpe, h80, h81, h82, h83, h84, h85, h86, h87, h88, h89, h8a, h8b, h8c]

private theorem decMany_length {α : Type} {d : List Nat → Option (α × List Nat)} :
    ∀ {n : Nat} {bs : List Nat} {xs : List α} {rest : List Nat},
      Sporos.decMany d n bs = some (xs, rest) → xs.length = n
  | 0, _, _, _, h => by cases h; rfl
  | n + 1, bs, _, _, h => by
    simp only [Sporos.decMany, bind, Option.bind, pure] at h
    cases hd : d bs <;> rw [hd] at h; · cases h
    dsimp only at h
    cases hm : Sporos.decMany d n _ <;> rw [hm] at h; · cases h
    dsimp only at h
    cases h
    simp [decMany_length hm]

private theorem walk_succ (bs : List Nat) (nw ws outs locals nc annOff na rem i p curD curA : Nat) :
    walk bs nw ws outs locals nc annOff na (rem + 1) i p curD curA =
      match bs[p]? with
      | none => .reject
      | some op =>
        match instrLen op with
        | none => .reject
        | some ilen =>
          if !(p + ilen ≤ bs.length) then .reject
          else match (if ilen == 5 then rdU32 bs (p + 1) else some 0) with
            | none => .reject
            | some imm =>
              let (hd, ha) := (lookup bs annOff na i).getD (curD, curA)
              match hstepF bs nw ws outs locals annOff na hd ha op imm with
              | .overflow => .overflow
              | .reject => .reject
              | .noFall => walk bs nw ws outs locals nc annOff na rem (i + 1) (p + ilen) 0 0
              | .next nd na' =>
                if i + 1 ≥ nc then .reject
                else
                  let okNext := match lookup bs annOff na (i + 1) with
                    | none => true
                    | some (ad, aa) => ad == nd && aa == na'
                  if okNext then walk bs nw ws outs locals nc annOff na rem (i + 1) (p + ilen) nd na'
                  else .reject := rfl

theorem walk_sound_refuse {bs : List Nat} (hb : ∀ b ∈ bs, b < 256)
    {nw ws : Nat} {pr : Prog} {wm : WordSlice} {wd : Word}
    (hm : WordSliceMatches bs wm wd)
    (hwords : Sporos.decMany Sporos.decWord nw (bs.drop ws) = some (pr.words, []))
    (hws : ws ≤ bs.length) (hwlen : pr.words.length = nw) :
    ∀ (rem i p curD curA : Nat) (code : List Instr),
      curD < HMAX → curA < HMAX →
      i + rem = wm.nc →
      p ≤ bs.length →
      Sporos.decMany Sporos.decInstr rem (bs.drop p) = some (code, bs.drop wm.annOff) →
      (walk bs nw ws wm.outs wm.locals wm.nc wm.annOff wm.na rem i p curD curA = .ok →
        walkSpec pr wd code i ⟨curD, curA⟩ = true) ∧
      (walk bs nw ws wm.outs wm.locals wm.nc wm.annOff wm.na rem i p curD curA = .reject →
        walkSpec pr wd code i ⟨curD, curA⟩ = false)
  | 0, i, p, curD, curA, code, _, _, _, _, hcode => by
    rcases Prod.mk.inj (Option.some.inj hcode) with ⟨rfl, _⟩
    simp [walk, walkSpec]
  | rem + 1, i, p, curD, curA, code, hcd, hca, hsum, hle, hcode => by
    simp only [Sporos.decMany, bind, Option.bind, pure] at hcode
    cases hdi : Sporos.decInstr (bs.drop p) with
    | none => rw [hdi] at hcode; cases hcode
    | some p1 =>
      rw [hdi] at hcode; dsimp only at hcode
      rcases p1 with ⟨ins, r1⟩
      obtain ⟨op, ilen, imm8, hpos, hlen, hle_p, _, hi8, hins, rfl⟩ := decInstr_drop_some hdi
      cases hm2 : Sporos.decMany Sporos.decInstr rem (bs.drop (p + ilen)) with
      | none => rw [hm2] at hcode; cases hcode
      | some p2 =>
        rw [hm2] at hcode; dsimp only at hcode
        rcases p2 with ⟨rest, r2⟩
        cases hcode
        obtain ⟨hlens, _⟩ := instrLen_ok hlen
        have hi4 : ∃ imm4, (if ilen == 5 then rdU32 bs (p + 1) else some 0) = some imm4 := by
          rcases hlens with rfl | rfl | rfl
          · exact ⟨0, rfl⟩
          · obtain ⟨v, hv⟩ := rdU32_of_lt (by omega : (p + 1) + 4 ≤ bs.length)
            exact ⟨v, by simp [hv]⟩
          · exact ⟨0, rfl⟩
        rcases hi4 with ⟨imm4, hi4⟩
        have hwa_eq :
            walk bs nw ws wm.outs wm.locals wm.nc wm.annOff wm.na (rem + 1) i p curD curA =
              let (hd, ha) := (lookup bs wm.annOff wm.na i).getD (curD, curA)
              match hstepF bs nw ws wm.outs wm.locals wm.annOff wm.na hd ha op imm4 with
              | .overflow => .overflow
              | .reject => .reject
              | .noFall => walk bs nw ws wm.outs wm.locals wm.nc wm.annOff wm.na rem (i + 1) (p + ilen) 0 0
              | .next nd na' =>
                if i + 1 ≥ wm.nc then .reject
                else
                  let okNext := match lookup bs wm.annOff wm.na (i + 1) with
                    | none => true
                    | some (ad, aa) => ad == nd && aa == na'
                  if okNext then walk bs nw ws wm.outs wm.locals wm.nc wm.annOff wm.na rem (i + 1) (p + ilen) nd na'
                  else .reject := by
          rw [walk_succ, hpos]; dsimp only; rw [hlen]; dsimp only
          rw [if_neg (by simp [hle_p]), hi4]
        rw [hwa_eq]
        have hgetD :
            let p_lk := (lookup bs wm.annOff wm.na i).getD (curD, curA)
            (⟨p_lk.1, p_lk.2⟩ : Height) = (wd.ann.lookup i).getD ⟨curD, curA⟩ ∧
              p_lk.1 < HMAX ∧ p_lk.2 < HMAX := by
          cases hlk : lookup bs wm.annOff wm.na i with
          | none => dsimp only; simp [(lookup_eq_none_iff hm i).1 hlk, hcd, hca]
          | some q =>
            rcases q with ⟨ad, aa⟩
            have hu := lookupLoop_u32_lt hb hlk
            have h32 : 2 ^ 32 ≤ HMAX := by decide
            dsimp only; simp only [Option.getD_some]
            exact ⟨by simp [(lookup_eq_some_iff hm i ad aa).1 hlk], by omega, by omega⟩
        rcases hgetD with ⟨hh_eq, hdlt, halt⟩
        let pair := (lookup bs wm.annOff wm.na i).getD (curD, curA)
        let hd := pair.1
        let ha := pair.2
        have hsound := hstepSound_spec hm hwords hws hwlen hdlt halt hlen hi8 hi4 hins
        rw [hh_eq] at hsound
        rw [walkSpec]
        dsimp only
        have hlen_wd : wd.code.length = wm.nc := hm.2.2.2.1
        cases hsf : hstepF bs nw ws wm.outs wm.locals wm.annOff wm.na hd ha op imm4 with
        | overflow => simp
        | reject =>
          rw [hsf] at hsound
          dsimp only [hstepSound] at hsound
          simp [hsound]
        | noFall =>
          rw [hsf] at hsound
          dsimp only [hstepSound] at hsound
          rw [hsound]; dsimp only
          exact walk_sound_refuse hb hm hwords hws hwlen rem (i + 1) (p + ilen) 0 0 rest
            (by decide) (by decide) (by omega) hle_p hm2
        | next nd na' =>
          rw [hsf] at hsound
          dsimp only [hstepSound] at hsound
          rcases hsound with ⟨hsp, hndl, hnal⟩
          rw [hsp]; dsimp only
          by_cases hrem0 : rem = 0
          · have hge : i + 1 ≥ wm.nc := by omega
            have hnlt : ¬ (i + 1 < wd.code.length) := by omega
            simp [hge, hnlt]
          · have hnge : ¬ (i + 1 ≥ wm.nc) := by omega
            have hlt : i + 1 < wd.code.length := by omega
            rw [if_neg hnge]
            have hok_eq :
                (match lookup bs wm.annOff wm.na (i + 1) with
                 | none => true
                 | some (ad, aa) => ad == nd && aa == na') =
                (match wd.ann.lookup (i + 1) with
                 | none => true
                 | some a => a == ({ d := nd, a := na' } : Height)) := by
              cases hl : lookup bs wm.annOff wm.na (i + 1) with
              | none => simp [(lookup_eq_none_iff hm (i + 1)).1 hl]
              | some q =>
                rcases q with ⟨ad, aa⟩
                simp [(lookup_eq_some_iff hm (i + 1) ad aa).1 hl, height_beq_eq]
            rw [hok_eq]
            cases hok : (match wd.ann.lookup (i + 1) with | none => true | some a => a == ({ d := nd, a := na' } : Height)) with
            | false => simp
            | true =>
              simp only [if_true, hlt, decide_true, Bool.true_and]
              exact walk_sound_refuse hb hm hwords hws hwlen rem (i + 1) (p + ilen) nd na' rest
                hndl hnal (by omega) hle_p hm2

theorem walk_complete {bs : List Nat} (hb : ∀ b ∈ bs, b < 256)
    {nw ws : Nat} {pr : Prog} {wm : WordSlice} {wd : Word}
    (hm : WordSliceMatches bs wm wd)
    (hwords : Sporos.decMany Sporos.decWord nw (bs.drop ws) = some (pr.words, []))
    (hws : ws ≤ bs.length) (hwlen : pr.words.length = nw) :
    ∀ (rem i p curD curA : Nat) (code : List Instr),
      curD < HMAX → curA < HMAX →
      i + rem = wm.nc →
      p ≤ bs.length →
      Sporos.decMany Sporos.decInstr rem (bs.drop p) = some (code, bs.drop wm.annOff) →
      walkSpec pr wd code i ⟨curD, curA⟩ = true →
      (∀ h ∈ Sporos.infer.go pr wd code i ⟨curD, curA⟩, h.d < HMAX ∧ h.a < HMAX) →
      walk bs nw ws wm.outs wm.locals wm.nc wm.annOff wm.na rem i p curD curA = .ok
  | 0, i, p, curD, curA, code, _, _, _, _, hcode, _, _ => by
    rcases Prod.mk.inj (Option.some.inj hcode) with ⟨rfl, _⟩
    rfl
  | rem + 1, i, p, curD, curA, code, hcd, hca, hsum, hle, hcode, hwspec, hinf => by
    simp only [Sporos.decMany, bind, Option.bind, pure] at hcode
    cases hdi : Sporos.decInstr (bs.drop p) with
    | none => rw [hdi] at hcode; cases hcode
    | some p1 =>
      rw [hdi] at hcode; dsimp only at hcode
      rcases p1 with ⟨ins, r1⟩
      obtain ⟨op, ilen, imm8, hpos, hlen, hle_p, _, hi8, hins, rfl⟩ := decInstr_drop_some hdi
      cases hm2 : Sporos.decMany Sporos.decInstr rem (bs.drop (p + ilen)) with
      | none => rw [hm2] at hcode; cases hcode
      | some p2 =>
        rw [hm2] at hcode; dsimp only at hcode
        rcases p2 with ⟨rest, r2⟩
        cases hcode
        obtain ⟨hlens, _⟩ := instrLen_ok hlen
        have hi4 : ∃ imm4, (if ilen == 5 then rdU32 bs (p + 1) else some 0) = some imm4 := by
          rcases hlens with rfl | rfl | rfl
          · exact ⟨0, rfl⟩
          · obtain ⟨v, hv⟩ := rdU32_of_lt (by omega : (p + 1) + 4 ≤ bs.length)
            exact ⟨v, by simp [hv]⟩
          · exact ⟨0, rfl⟩
        rcases hi4 with ⟨imm4, hi4⟩
        have hgetD :
            let p_lk := (lookup bs wm.annOff wm.na i).getD (curD, curA)
            (⟨p_lk.1, p_lk.2⟩ : Height) = (wd.ann.lookup i).getD ⟨curD, curA⟩ := by
          cases hlk : lookup bs wm.annOff wm.na i with
          | none => simp [(lookup_eq_none_iff hm i).1 hlk]
          | some q => rcases q with ⟨ad, aa⟩; simp [(lookup_eq_some_iff hm i ad aa).1 hlk]
        let pair := (lookup bs wm.annOff wm.na i).getD (curD, curA)
        let hd := pair.1
        let ha := pair.2
        have hinf_hd : hd < HMAX ∧ ha < HMAX := by
          have hmem : (wd.ann.lookup i).getD ⟨curD, curA⟩ ∈
              Sporos.infer.go pr wd (ins :: rest) i ⟨curD, curA⟩ := by
            rw [infer_go_cons]; simp
          have h0 := hinf _ hmem
          rwa [← hgetD] at h0
        have hnext_lt : ∀ h', Sporos.hstep pr wd ⟨hd, ha⟩ ins = some (some h') →
            h'.d < HMAX ∧ h'.a < HMAX := by
          intro h' hsp
          rw [walkSpec, ← hgetD] at hwspec
          dsimp only at hwspec
          rw [hsp] at hwspec
          simp only [Bool.and_eq_true, decide_eq_true_eq] at hwspec
          rcases hwspec with ⟨⟨hi1, hokn⟩, _⟩
          have hrlen : rest.length = rem := decMany_length hm2
          have hwdlen : wd.code.length = wm.nc := hm.2.2.2.1
          cases hrest_eq : rest with
          | nil => rw [hrest_eq] at hrlen; simp at hrlen; omega
          | cons ins1 rest1 =>
            have hhead : (wd.ann.lookup (i + 1)).getD h' = h' := by
              cases hlk : wd.ann.lookup (i + 1) with
              | none => rfl
              | some a =>
                rw [hlk] at hokn
                simp only [beq_iff_eq] at hokn
                simp [hokn]
            have hmem : h' ∈ Sporos.infer.go pr wd (ins :: rest) i ⟨curD, curA⟩ := by
              rw [infer_go_cons, ← hgetD]
              dsimp only
              rw [hsp, hrest_eq, infer_go_cons, hhead]
              simp
            exact hinf h' hmem
        have hne_ov := hstepF_ne_overflow_of_lt hm hwords hws hwlen hlen hi8 hi4 hins hnext_lt
        have hwa_eq :
            walk bs nw ws wm.outs wm.locals wm.nc wm.annOff wm.na (rem + 1) i p curD curA =
              match hstepF bs nw ws wm.outs wm.locals wm.annOff wm.na hd ha op imm4 with
              | .overflow => .overflow
              | .reject => .reject
              | .noFall => walk bs nw ws wm.outs wm.locals wm.nc wm.annOff wm.na rem (i + 1) (p + ilen) 0 0
              | .next nd na' =>
                if i + 1 ≥ wm.nc then .reject
                else
                  let okNext := match lookup bs wm.annOff wm.na (i + 1) with
                    | none => true
                    | some (ad, aa) => ad == nd && aa == na'
                  if okNext then walk bs nw ws wm.outs wm.locals wm.nc wm.annOff wm.na rem (i + 1) (p + ilen) nd na'
                  else .reject := by
          rw [walk_succ, hpos]; dsimp only; rw [hlen]; dsimp only
          rw [if_neg (by simp [hle_p]), hi4]
        rw [hwa_eq]
        have hsound := hstepSound_spec hm hwords hws hwlen hinf_hd.1 hinf_hd.2 hlen hi8 hi4 hins
        rw [walkSpec, ← hgetD] at hwspec
        dsimp only at hwspec
        cases hsf : hstepF bs nw ws wm.outs wm.locals wm.annOff wm.na hd ha op imm4 with
        | overflow => exact absurd hsf hne_ov
        | reject =>
          rw [hsf] at hsound
          dsimp only [hstepSound] at hsound
          rw [hsound] at hwspec; simp at hwspec
        | noFall =>
          rw [hsf] at hsound
          dsimp only [hstepSound] at hsound
          rw [hsound] at hwspec; dsimp only at hwspec ⊢
          have hinf' : ∀ h ∈ Sporos.infer.go pr wd rest (i + 1) ⟨0, 0⟩, h.d < HMAX ∧ h.a < HMAX := by
            intro h' hmem'
            apply hinf
            rw [infer_go_cons_zero, ← hgetD]
            dsimp only
            rw [hsound]
            simp [hmem']
          exact walk_complete hb hm hwords hws hwlen rem (i + 1) (p + ilen) 0 0 rest
            (by decide) (by decide) (by omega) hle_p hm2 hwspec hinf'
        | next nd na' =>
          rw [hsf] at hsound
          dsimp only [hstepSound] at hsound
          rcases hsound with ⟨hsp, hndl, hnal⟩
          rw [hsp] at hwspec; dsimp only at hwspec ⊢
          simp only [Bool.and_eq_true, decide_eq_true_eq] at hwspec
          rcases hwspec with ⟨⟨hi1, hokn⟩, hwrest⟩
          have hwdlen : wd.code.length = wm.nc := hm.2.2.2.1
          have hnge : ¬ (i + 1 ≥ wm.nc) := by omega
          have hok_eq :
              (match lookup bs wm.annOff wm.na (i + 1) with
               | none => true
               | some (ad, aa) => ad == nd && aa == na') = true := by
            cases hl : lookup bs wm.annOff wm.na (i + 1) with
            | none => rfl
            | some q =>
              rcases q with ⟨ad, aa⟩
              rw [(lookup_eq_some_iff hm (i + 1) ad aa).1 hl] at hokn
              simpa [height_beq_eq, beq_iff_eq] using hokn
          rw [if_neg hnge, if_pos hok_eq]
          have hinf' : ∀ h ∈ Sporos.infer.go pr wd rest (i + 1) ⟨nd, na'⟩, h.d < HMAX ∧ h.a < HMAX := by
            intro h' hmem'
            apply hinf
            rw [infer_go_cons_zero, ← hgetD]
            dsimp only
            rw [hsp]
            simp [hmem']
          exact walk_complete hb hm hwords hws hwlen rem (i + 1) (p + ilen) nd na' rest
            hndl hnal (by omega) hle_p hm2 hwrest hinf'

theorem walk_no_overflow_of_small {bs : List Nat} (hb : ∀ b ∈ bs, b < 256)
    (nw ws outs locals nc annOff na : Nat) (hnc : nc ≤ 65535) :
    ∀ (rem i p curD curA : Nat),
      rem ≤ nc →
      curD < (65536 - rem) * 2 ^ 32 →
      curA < (65536 - rem) * 2 ^ 32 →
      walk bs nw ws outs locals nc annOff na rem i p curD curA ≠ .overflow
  | 0, _, _, _, _, _, _, _ => by simp [walk]
  | rem + 1, i, p, curD, curA, hrem, hd_lt, ha_lt => by
    unfold walk
    cases hpos : bs[p]? with
    | none => simp
    | some op =>
      dsimp only
      cases hlen : instrLen op with
      | none => simp
      | some ilen =>
        dsimp only
        by_cases hle : p + ilen ≤ bs.length
        · rw [if_neg (by simp [hle])]
          cases himm : (if ilen == 5 then rdU32 bs (p + 1) else some 0) with
          | none => simp
          | some imm =>
            dsimp only
            let pair := (lookup bs annOff na i).getD (curD, curA)
            have hpair_lt : pair.1 < (65536 - (rem + 1)) * 2 ^ 32 ∧
                pair.2 < (65536 - (rem + 1)) * 2 ^ 32 := by
              cases hlk : lookup bs annOff na i with
              | none => dsimp only [pair]; rw [hlk, Option.getD_none]; exact ⟨hd_lt, ha_lt⟩
              | some q =>
                have hu := lookupLoop_u32_lt hb hlk
                have h65 : 1 ≤ 65536 - (rem + 1) := by omega
                have hmul := Nat.mul_le_mul_right (2 ^ 32) h65
                dsimp only [pair]; rw [hlk, Option.getD_some]
                omega
            have hhd_le : pair.1 + 2 ^ 32 ≤ HMAX := by
              have h65 : 65536 - (rem + 1) + 1 ≤ 65536 := by omega
              have h1 := Nat.mul_le_mul_right (2 ^ 32) h65
              have h2 : (65536 - (rem + 1) + 1) * 2 ^ 32 =
                  (65536 - (rem + 1)) * 2 ^ 32 + 2 ^ 32 := Nat.add_mul _ 1 _
              have h3 : 65536 * 2 ^ 32 = HMAX := by decide
              omega
            have hha_lt : pair.2 + 1 < HMAX := by
              have h1 := Nat.mul_le_mul_right (2 ^ 32) (show 65536 - (rem + 1) ≤ 65536 by omega)
              have h3 : 65536 * 2 ^ 32 = HMAX := by decide
              omega
            obtain ⟨hne, hnext⟩ := hstepF_no_overflow_of_small hb hhd_le hha_lt
            cases hsf : hstepF bs nw ws outs locals annOff na
                ((lookup bs annOff na i).getD (curD, curA)).1
                ((lookup bs annOff na i).getD (curD, curA)).2 op imm with
            | overflow => exact False.elim (hne hsf)
            | reject => dsimp only; simp
            | noFall =>
              dsimp only
              have hpos0 : 0 < (65536 - rem) * 2 ^ 32 :=
                Nat.mul_pos (by omega) (by decide)
              exact walk_no_overflow_of_small hb nw ws outs locals nc annOff na hnc
                rem (i + 1) (p + ilen) 0 0 (by omega) hpos0 hpos0
            | next nd na' =>
              obtain ⟨hnd, hna⟩ := hnext nd na' hsf
              dsimp only
              by_cases hge : i + 1 ≥ nc
              · simp [hge]
              · rw [if_neg hge]
                by_cases hok : (match lookup bs annOff na (i + 1) with
                    | none => true | some (ad, aa) => ad == nd && aa == na') = true
                · rw [if_pos hok]
                  have hexp : (65536 - rem) * 2 ^ 32 =
                      (65536 - (rem + 1)) * 2 ^ 32 + 2 ^ 32 := by
                    rw [show 65536 - rem = (65536 - (rem + 1)) + 1 by omega, Nat.add_mul, Nat.one_mul]
                  exact walk_no_overflow_of_small hb nw ws outs locals nc annOff na hnc
                    rem (i + 1) (p + ilen) nd na' (by omega) (by omega) (by omega)
                · rw [if_neg hok]; simp
        · rw [if_pos (by simp [hle])]
          simp

theorem checkWords_sound_refuse {bs : List Nat} (hb : ∀ b ∈ bs, b < 256)
    {nw ws : Nat} {pr : Prog}
    (hwords : Sporos.decMany Sporos.decWord nw (bs.drop ws) = some (pr.words, []))
    (hws : ws ≤ bs.length) (hwlen : pr.words.length = nw) :
    ∀ (rem off : Nat) (tailWords : List Word) (rest : List Nat),
      off ≤ bs.length →
      Sporos.decMany Sporos.decWord rem (bs.drop off) = some (tailWords, rest) →
      (checkWords bs nw ws rem off = .ok →
        (tailWords.all fun wd => Sporos.checkWord pr wd (Sporos.infer pr wd)) = true) ∧
      (checkWords bs nw ws rem off = .reject →
        (tailWords.all fun wd => Sporos.checkWord pr wd (Sporos.infer pr wd)) = false)
  | 0, _, tailWords, _, _, hrem => by
    cases hrem; simp [checkWords]
  | rem + 1, off, tailWords, rest, hle, hrem => by
    simp only [Sporos.decMany, bind, Option.bind, pure] at hrem
    cases hd : Sporos.decWord (bs.drop off) with
    | none => rw [hd] at hrem; cases hrem
    | some p1 =>
      rw [hd] at hrem; dsimp only at hrem
      rcases p1 with ⟨wd, r1⟩
      obtain ⟨wm, hr, rfl, hm⟩ := decWord_to_readWordAt hle hd
      rcases readWordAt_bounds hr with ⟨_, _, _, _, hne, _⟩
      cases hm2 : Sporos.decMany Sporos.decWord rem (bs.drop wm.nextOff) with
      | none => rw [hm2] at hrem; cases hrem
      | some p2 =>
        rw [hm2] at hrem; dsimp only at hrem
        cases hrem
        rcases p2 with ⟨restWords, r2⟩
        have ih := checkWords_sound_refuse hb hwords hws hwlen rem wm.nextOff restWords r2 hne hm2
        have hcode_len : wd.code.length = wm.nc := hm.2.2.2.1
        have hwm_u := readWordAt_u32_lt hb hr
        have hc_eq :
            checkWords bs nw ws (rem + 1) off =
              if wm.nc == 0 then .reject
              else
                let ok0 := match lookup bs wm.annOff wm.na 0 with
                  | none => true
                  | some (d0, a0) => d0 == wm.ins && a0 == 0
                if !ok0 then .reject
                else match walk bs nw ws wm.outs wm.locals wm.nc wm.annOff wm.na wm.nc 0 wm.codeOff wm.ins 0 with
                  | .overflow => .overflow
                  | .reject => .reject
                  | .ok =>
                    if annsOK bs wm.nc wm.annOff wm.na wm.na wm.annOff then
                      checkWords bs nw ws rem wm.nextOff
                    else .reject := by
          simp [checkWords, hr]
        rw [hc_eq]
        by_cases hnc0 : wm.nc = 0
        · have he : wd.code.isEmpty = true := List.isEmpty_iff.2 (List.eq_nil_of_length_eq_zero (by omega))
          simp [hnc0, Sporos.checkWord, he]
        · have hnc0b : (wm.nc == 0) = false := beq_false_of_ne hnc0
          have hne_code : ¬ wd.code.isEmpty := fun h => hnc0 (by rw [← hcode_len, List.isEmpty_iff.1 h]; rfl)
          rw [if_neg (by simp [hnc0b])]
          have hok0_eq :
              (match lookup bs wm.annOff wm.na 0 with
               | none => true
               | some (d0, a0) => d0 == wm.ins && a0 == 0) =
              (match wd.ann.lookup 0 with
               | none => true
               | some a => a == ({ d := wd.ins, a := 0 } : Height)) := by
            cases hl : lookup bs wm.annOff wm.na 0 with
            | none => simp [(lookup_eq_none_iff hm 0).1 hl]
            | some q =>
              rcases q with ⟨d0, a0⟩
              simp [(lookup_eq_some_iff hm 0 d0 a0).1 hl, hm.1, height_beq_eq]
          have hc_iff := walkSpec_iff_checkWord pr wd hne_code
          have h32 : 2 ^ 32 ≤ HMAX := by decide
          have hwsr := walk_sound_refuse hb hm hwords hws hwlen wm.nc 0 wm.codeOff wm.ins 0 wd.code
            (by omega) (by decide) (Nat.zero_add _)
            (by rcases readWordAt_bounds hr with ⟨_, hc, hca, _, hna, _⟩; omega)
            hm.2.2.2.2.2.2.2.2.1
          have hanns := annsOK_spec hm
          have hanns_eq : annsOK bs wm.nc wm.annOff wm.na wm.na wm.annOff =
              (wd.ann.all fun ⟨t, a⟩ => decide (t < wd.code.length) && (Sporos.infer pr wd)[t]? == some a) := by
            cases h1 : annsOK bs wm.nc wm.annOff wm.na wm.na wm.annOff with
            | true =>
              rw [hanns, ← ann_all_iff_lookup] at h1
              exact h1.symm
            | false =>
              cases h2 : wd.ann.all fun ⟨t, a⟩ => decide (t < wd.code.length) && (Sporos.infer pr wd)[t]? == some a with
              | false => rfl
              | true => rw [ann_all_iff_lookup, ← hanns] at h2; rw [h2] at h1; cases h1
          rw [hok0_eq]
          cases hok0 : (match wd.ann.lookup 0 with | none => true | some a => a == ({ d := wd.ins, a := 0 } : Height)) with
          | false =>
            have hcw : Sporos.checkWord pr wd (Sporos.infer pr wd) = false := by
              cases hc : Sporos.checkWord pr wd (Sporos.infer pr wd)
              · rfl
              · rw [← hc_iff] at hc; simp [hok0] at hc
            simp [hcw]
          | true =>
            dsimp only
            rw [if_neg (by simp)]
            cases hwk : walk bs nw ws wm.outs wm.locals wm.nc wm.annOff wm.na wm.nc 0 wm.codeOff wm.ins 0 with
            | overflow => simp
            | reject =>
              have hwsf := hwsr.2 hwk
              rw [← hm.1] at hwsf
              have hcw : Sporos.checkWord pr wd (Sporos.infer pr wd) = false := by
                cases hc : Sporos.checkWord pr wd (Sporos.infer pr wd)
                · rfl
                · rw [← hc_iff] at hc; simp [hwsf] at hc
              simp [hcw]
            | ok =>
              have hwst := hwsr.1 hwk
              rw [← hm.1] at hwst
              rw [hanns_eq]
              cases hannb : wd.ann.all fun ⟨t, a⟩ => decide (t < wd.code.length) && (Sporos.infer pr wd)[t]? == some a with
              | false =>
                have hcw : Sporos.checkWord pr wd (Sporos.infer pr wd) = false := by
                  cases hc : Sporos.checkWord pr wd (Sporos.infer pr wd)
                  · rfl
                  · rw [← hc_iff] at hc; simp [hannb] at hc
                simp [hcw]
              | true =>
                have hcw : Sporos.checkWord pr wd (Sporos.infer pr wd) = true := by
                  rw [← hc_iff]; simp [hok0, hwst, hannb]
                simp only [List.all_cons, hcw, Bool.true_and]
                exact ih

/-- Explicit 48-bit height capacity and 64-bit byte-slice capacity predicate. -/
def Fits48 (bs : List Nat) (pr : Prog) : Prop :=
  (∀ b ∈ bs, b < 256) ∧
  bs.length < W ∧
  ∀ wd ∈ pr.words,
    wd.ins < HMAX ∧
    wd.outs < HMAX ∧
    (∀ p ∈ wd.ann, p.2.d < HMAX ∧ p.2.a < HMAX) ∧
    (∀ h ∈ Sporos.infer pr wd, h.d < HMAX ∧ h.a < HMAX)

theorem checkWords_complete {bs : List Nat} (hb : ∀ b ∈ bs, b < 256)
    {nw ws : Nat} {pr : Prog}
    (hwords : Sporos.decMany Sporos.decWord nw (bs.drop ws) = some (pr.words, []))
    (hws : ws ≤ bs.length) (hwlen : pr.words.length = nw)
    (hfit : ∀ wd ∈ pr.words, ∀ h ∈ Sporos.infer pr wd, h.d < HMAX ∧ h.a < HMAX) :
    ∀ (rem off : Nat) (tailWords : List Word) (rest : List Nat),
      off ≤ bs.length →
      Sporos.decMany Sporos.decWord rem (bs.drop off) = some (tailWords, rest) →
      (∀ wd ∈ tailWords, wd ∈ pr.words) →
      (tailWords.all fun wd => Sporos.checkWord pr wd (Sporos.infer pr wd)) = true →
      checkWords bs nw ws rem off = .ok
  | 0, _, tailWords, _, _, hrem, _, _ => by cases hrem; rfl
  | rem + 1, off, tailWords, rest, hle, hrem, hsub, hall => by
    simp only [Sporos.decMany, bind, Option.bind, pure] at hrem
    cases hd : Sporos.decWord (bs.drop off) with
    | none => rw [hd] at hrem; cases hrem
    | some p1 =>
      rw [hd] at hrem; dsimp only at hrem
      rcases p1 with ⟨wd, r1⟩
      obtain ⟨wm, hr, rfl, hm⟩ := decWord_to_readWordAt hle hd
      rcases readWordAt_bounds hr with ⟨_, _, _, _, hne, _⟩
      cases hm2 : Sporos.decMany Sporos.decWord rem (bs.drop wm.nextOff) with
      | none => rw [hm2] at hrem; cases hrem
      | some p2 =>
        rw [hm2] at hrem; dsimp only at hrem
        cases hrem
        rcases p2 with ⟨restWords, r2⟩
        simp only [List.all_cons, Bool.and_eq_true] at hall
        have hwd_mem : wd ∈ pr.words := hsub wd (by simp)
        have hwd_cw : Sporos.checkWord pr wd (Sporos.infer pr wd) = true := hall.1
        have hne_code : ¬ wd.code.isEmpty := by
          unfold Sporos.checkWord at hwd_cw
          simp only [Bool.and_eq_true] at hwd_cw
          rcases hwd_cw with ⟨⟨⟨⟨_, hne⟩, _⟩, _⟩, _⟩
          simpa using hne
        have hc_iff := (walkSpec_iff_checkWord pr wd hne_code).2 hwd_cw
        simp only [Bool.and_eq_true] at hc_iff
        rcases hc_iff with ⟨⟨hok0, hwspec⟩, hann_all⟩
        have hcode_len : wd.code.length = wm.nc := hm.2.2.2.1
        have hnc0 : (wm.nc == 0) = false := beq_false_of_ne fun h => hne_code (by
          have : wd.code = [] := List.eq_nil_of_length_eq_zero (by omega)
          simp [this])
        have hok0_flat :
            (match lookup bs wm.annOff wm.na 0 with
             | none => true
             | some (d0, a0) => d0 == wm.ins && a0 == 0) = true := by
          cases hl : lookup bs wm.annOff wm.na 0 with
          | none => rfl
          | some q =>
            rcases q with ⟨d0, a0⟩
            rw [(lookup_eq_some_iff hm 0 d0 a0).1 hl] at hok0
            simpa [hm.1, height_beq_eq] using hok0
        have hwm_u := readWordAt_u32_lt hb hr
        have hwk_ok :
            walk bs nw ws wm.outs wm.locals wm.nc wm.annOff wm.na wm.nc 0 wm.codeOff wm.ins 0 = .ok := by
          rw [← hm.1]
          exact walk_complete hb hm hwords hws hwlen wm.nc 0 wm.codeOff wd.ins 0 wd.code
            (by rw [hm.1]; unfold HMAX; omega) (by unfold HMAX; omega) (Nat.zero_add _)
            (by rcases readWordAt_bounds hr with ⟨_, _, _, _, _, _⟩; omega)
            hm.2.2.2.2.2.2.2.2.1 hwspec (hfit wd hwd_mem)
        have hanns_ok : annsOK bs wm.nc wm.annOff wm.na wm.na wm.annOff = true :=
          (annsOK_spec hm).2 ((ann_all_iff_lookup pr wd).1 hann_all)
        have hih := checkWords_complete hb hwords hws hwlen hfit rem wm.nextOff restWords r2 hne hm2
          (fun w hw => hsub w (by simp [hw])) hall.2
        simp [checkWords, hr, hnc0, hok0_flat, hwk_ok, hanns_ok, hih]

theorem checkWords_no_overflow_of_small {bs : List Nat} (hb : ∀ b ∈ bs, b < 256)
    (hlen : bs.length < 65536) (nw ws : Nat) :
    ∀ (rem off : Nat), checkWords bs nw ws rem off ≠ .overflow
  | 0, _ => by simp [checkWords]
  | rem + 1, off => by
    unfold checkWords
    cases hr : readWordAt bs off with
    | none => simp
    | some wm =>
      dsimp only
      by_cases hnc0 : (wm.nc == 0) = true
      · simp [hnc0]
      · rw [if_neg hnc0]
        by_cases hok0 : (match lookup bs wm.annOff wm.na 0 with
            | none => true | some (d0, a0) => d0 == wm.ins && a0 == 0) = true
        · rw [if_neg (by simp [hok0])]
          have hnc : wm.nc ≤ 65535 := by have := readWordAt_nc_le hr; omega
          have hwm_u := readWordAt_u32_lt hb hr
          have hd_lt : wm.ins < (65536 - wm.nc) * 2 ^ 32 := by
            have h1 : 1 ≤ 65536 - wm.nc := by omega
            have h2 := Nat.mul_le_mul_right (2 ^ 32) h1
            omega
          have ha_lt : 0 < (65536 - wm.nc) * 2 ^ 32 := Nat.mul_pos (by omega) (by decide)
          have hne := walk_no_overflow_of_small hb nw ws wm.outs wm.locals wm.nc wm.annOff wm.na hnc
            wm.nc 0 wm.codeOff wm.ins 0 (Nat.le_refl _) hd_lt ha_lt
          cases hwk : walk bs nw ws wm.outs wm.locals wm.nc wm.annOff wm.na wm.nc 0 wm.codeOff wm.ins 0 with
          | overflow => exact absurd hwk hne
          | reject => simp
          | ok =>
            by_cases hann : annsOK bs wm.nc wm.annOff wm.na wm.na wm.annOff = true
            · rw [if_pos hann]; exact checkWords_no_overflow_of_small hb hlen nw ws rem wm.nextOff
            · rw [if_neg hann]; simp
        · rw [if_pos (by simp [hok0])]; simp

/-! ### Universal soundness, safety, refusal soundness, and completeness theorems -/

theorem checkFlat_sound {bs : List Nat} (h : checkFlat bs = some true) :
    ∃ pr, Sporos.decode bs = some pr ∧ Sporos.checkProg pr = true := by
  simp only [checkFlat, bind, Option.bind] at h
  cases hres : checkFlatRes bs with
  | none => rw [hres] at h; cases h
  | some res =>
    rw [hres] at h; dsimp only at h
    cases res <;> simp at h
    unfold checkFlatRes at hres
    split at hres; · cases hres
    rename_i hpre
    simp only [Bool.or_eq_true, Bool.not_eq_true', not_or,
      decide_eq_true_eq, Nat.not_le] at hpre
    have hba : bs.all (· < 256) = true := by
      cases h2 : bs.all (· < 256) <;> simp_all
    have hb : ∀ b ∈ bs, b < 256 := fun b hmem =>
      of_decide_eq_true (List.all_eq_true.1 hba b hmem)
    cases hp : parse bs with
    | none => rw [hp] at hres; cases hres
    | some parsed =>
      rw [hp] at hres; dsimp only at hres
      rcases parsed with ⟨main, nw, ws⟩
      dsimp only at hres
      split at hres; · cases hres
      rename_i hn_main
      have hlt_main : main < nw := by omega
      cases hwa : wordAt bs ws main with
      | none => rw [hwa] at hres; cases hres
      | some mainWm =>
        rw [hwa] at hres; dsimp only at hres
        split at hres; · cases hres
        rename_i hmain_ins
        have hmain0 : mainWm.ins = 0 := by
          simp only [bne_iff_ne] at hmain_ins
          exact Classical.byContradiction hmain_ins
        have hcw_ok : checkWords bs nw ws nw ws = .ok := Option.some.inj hres
        obtain ⟨words, data, hdec, hwlen, hws_le, _, hwords⟩ := parse_to_decode hb hp
        let pr : Prog := ⟨words, data, main⟩
        obtain ⟨mwd, hmget, hm_match⟩ := wordAt_spec hws_le hwords (by omega) hwa
        have hcw := (checkWords_sound_refuse (pr := pr) hb hwords hws_le hwlen nw ws words [] hws_le hwords).1 hcw_ok
        refine ⟨pr, hdec, ?_⟩
        unfold Sporos.checkProg
        rw [Bool.and_eq_true]
        refine ⟨by simpa [pr, List.all_eq_true] using hcw, ?_⟩
        simp only [pr, hmget, hm_match.1, hmain0, beq_self_eq_true]

theorem checkFlat_safe {bs : List Nat} (h : checkFlat bs = some true) (ph : Phys) :
    ∃ pr, Sporos.decode bs = some pr ∧
      ∀ (s : St), Sporos.Inv pr s → Sporos.Safe pr (Sporos.step pr ph s) := by
  obtain ⟨pr, hdec, hchk⟩ := checkFlat_sound h
  exact ⟨pr, hdec, fun s hi => Sporos.step_safe pr ph hchk s hi⟩

theorem checkFlat_run_safe {bs : List Nat} (h : checkFlat bs = some true)
    (ph : Phys) (msize : Nat) (inp : List Nat) (n : Nat) :
    ∃ pr, Sporos.decode bs = some pr ∧
      ∀ e, Sporos.run pr ph n (Sporos.init pr msize inp) = .fail e → e.static = false := by
  obtain ⟨pr, hdec, hchk⟩ := checkFlat_sound h
  exact ⟨pr, hdec, fun e hr => Sporos.run_safe pr ph hchk msize inp n e hr⟩

theorem checkFlat_refuse_sound {bs : List Nat} (h : checkFlat bs = some false) :
    ∀ pr, Sporos.decode bs = some pr → Sporos.checkProg pr = false := by
  intro pr hdec
  simp only [checkFlat, bind, Option.bind] at h
  cases hres : checkFlatRes bs with
  | none => rw [hres] at h; cases h
  | some res =>
    rw [hres] at h; dsimp only at h
    cases res <;> simp at h
    unfold checkFlatRes at hres
    split at hres; · cases hres
    rename_i hpre
    simp only [Bool.or_eq_true, Bool.not_eq_true', not_or,
      decide_eq_true_eq, Nat.not_le] at hpre
    have hba : bs.all (· < 256) = true := by
      cases h2 : bs.all (· < 256) <;> simp_all
    have hb : ∀ b ∈ bs, b < 256 := fun b hmem =>
      of_decide_eq_true (List.all_eq_true.1 hba b hmem)
    obtain ⟨ws, hp, hws_le, _, hwords⟩ := decode_to_parse hdec
    rw [hp] at hres; dsimp only at hres
    by_cases hge : pr.main ≥ pr.words.length
    · have hnone : pr.words[pr.main]? = none := List.getElem?_eq_none (by omega)
      simp [Sporos.checkProg, hnone]
    · rw [if_neg hge] at hres
      obtain ⟨mwm, mwd, hwa, hget, hm⟩ := wordAt_of_decMany hws_le hwords (by omega : pr.main < pr.words.length)
      rw [hwa] at hres; dsimp only at hres
      by_cases hins0 : mwm.ins = 0
      · have hnot : ¬ ((mwm.ins != 0) = true) := by simp [hins0]
        rw [if_neg hnot] at hres
        have hcw_rej : checkWords bs pr.words.length ws pr.words.length ws = .reject :=
          Option.some.inj hres
        have hcw := (checkWords_sound_refuse hb hwords hws_le rfl pr.words.length ws pr.words [] hws_le hwords).2 hcw_rej
        unfold Sporos.checkProg
        rw [Bool.and_eq_false_iff]
        exact Or.inl (by simpa [List.all_eq_true] using hcw)
      · have hne0 : (mwd.ins == 0) = false := beq_false_of_ne (by rw [hm.1]; exact hins0)
        simp [Sporos.checkProg, hget, hne0]

theorem checkFlat_complete {bs : List Nat} {pr : Prog}
    (hdec : Sporos.decode bs = some pr)
    (hchk : Sporos.checkProg pr = true)
    (hfit : Fits48 bs pr) :
    checkFlat bs = some true := by
  rcases hfit with ⟨hb, hlen, hwords_fit⟩
  have hba : bs.all (· < 256) = true := List.all_eq_true.2 fun b hm => decide_eq_true (hb b hm)
  obtain ⟨ws, hp, hws_le, _, hwords⟩ := decode_to_parse hdec
  unfold Sporos.checkProg at hchk
  simp only [Bool.and_eq_true, List.all_eq_true] at hchk
  rcases hchk with ⟨hall, hmain_chk⟩
  cases hget : pr.words[pr.main]? with
  | none => rw [hget] at hmain_chk; cases hmain_chk
  | some mwd =>
    rw [hget] at hmain_chk
    have hm_lt : pr.main < pr.words.length := (List.getElem?_eq_some_iff.1 hget).1
    obtain ⟨mwm, mwd', hwa, hget', hm_m⟩ := wordAt_of_decMany hws_le hwords hm_lt
    rw [hget] at hget'; cases hget'
    have hins0 : mwm.ins = 0 := by
      rw [← hm_m.1]
      exact beq_iff_eq.1 hmain_chk
    have hcw_ok : checkWords bs pr.words.length ws pr.words.length ws = .ok :=
      checkWords_complete hb hwords hws_le rfl (fun wd hmem => (hwords_fit wd hmem).2.2.2)
        pr.words.length ws pr.words [] hws_le hwords (fun _ h => h) (by simpa [List.all_eq_true] using hall)
    have hres_ok : checkFlatRes bs = some .ok := by
      unfold checkFlatRes
      rw [if_neg (by simp [hba]; omega), hp]; dsimp only
      rw [if_neg (by omega), hwa]; dsimp only
      rw [if_neg (by simp [hins0])]
      exact congrArg some hcw_ok
    simp [checkFlat, hres_ok]

theorem checkFlat_small_iff {bs : List Nat} (hb : ∀ b ∈ bs, b < 256) (hlen : bs.length < 65536) :
    checkFlat bs = some (match Sporos.decode bs with | some pr => Sporos.checkProg pr | none => false) := by
  have hba : bs.all (· < 256) = true := List.all_eq_true.2 fun b hm => decide_eq_true (hb b hm)
  have hltW : bs.length < W := by unfold W; omega
  have hno_ov : checkFlatRes bs ≠ some .overflow := by
    unfold checkFlatRes
    rw [if_neg (by simp [hba]; omega)]
    cases hp : parse bs with
    | none => simp
    | some parsed =>
      rcases parsed with ⟨main, nw, ws⟩
      dsimp only
      by_cases hge : main ≥ nw
      · simp [hge]
      · rw [if_neg hge]
        cases hwa : wordAt bs ws main with
        | none => simp
        | some mainWm =>
          dsimp only
          by_cases hins : (mainWm.ins != 0) = true
          · simp [hins]
          · rw [if_neg hins]
            intro h
            exact checkWords_no_overflow_of_small hb hlen nw ws nw ws (Option.some.inj h)
  cases hres : checkFlatRes bs with
  | none =>
    unfold checkFlatRes at hres
    rw [if_neg (by simp [hba]; omega)] at hres
    cases hres
  | some res =>
    cases res with
    | overflow => exact False.elim (hno_ov hres)
    | ok =>
      have hc : checkFlat bs = some true := by simp [checkFlat, hres]
      obtain ⟨pr, hdec, hchk⟩ := checkFlat_sound hc
      simp [hc, hdec, hchk]
    | reject =>
      have hc : checkFlat bs = some false := by simp [checkFlat, hres]
      have href := checkFlat_refuse_sound hc
      cases hdec : Sporos.decode bs with
      | none => simp [hc]
      | some pr => simp [hc, href pr hdec]

/-! ### Tail voter + dual-`ACCEPT`-guard (`tailCode`, 143 bytes) and single-fault immunity -/

def OVERFLOW : Nat := 0x0bad0f1e33c3c3a5

/-- Canonical 143-byte x86-64 tail voter + dual-`ACCEPT` guard + exit stubs:
bytes `0x00..0x25` equal `TMRVoter.voterCode` with byte `0x19` patched `0xfe → 0x0c`
to jump to `0x26`; bytes `0x26..0x8e` verify both `rax` and `rbx` equal `ACCEPT`
before `lea rdi, [rbx - 0x96]; syscall`, else dispatch `OVERFLOW` to `exit(6)` and
all other non-`ACCEPT` values to `exit(5)`. -/
def tailCode : List Nat :=
  [ 0x48, 0x39, 0xd8              -- 0x00: cmp rax, rbx
  , 0x74, 0x0d                    -- 0x03: je  +0x0d -> 0x12
  , 0x48, 0x39, 0xc8              -- 0x05: cmp rax, rcx
  , 0x74, 0x08                    -- 0x08: je  +0x08 -> 0x12
  , 0x48, 0x39, 0xcb              -- 0x0a: cmp rbx, rcx
  , 0x75, 0x0b                    -- 0x0d: jne +0x0b -> 0x1a
  , 0x48, 0x89, 0xd8              -- 0x0f: mov rax, rbx
  , 0x48, 0x89, 0xc3              -- 0x12: mov rbx, rax
  , 0x48, 0x89, 0xc1              -- 0x15: mov rcx, rax
  , 0xeb, 0x0c                    -- 0x18: jmp +0x0c -> 0x26
  , 0xbf, 0x03, 0x00, 0x00, 0x00  -- 0x1a: mov edi, 3
  , 0xb8, 0x3c, 0x00, 0x00, 0x00  -- 0x1f: mov eax, 60
  , 0x0f, 0x05                    -- 0x24: syscall
  , 0xbf, 0x05, 0x00, 0x00, 0x00  -- 0x26: mov edi, 5
  , 0x48, 0xba, 0x6a, 0xc3, 0xa5, 0x5a, 0x8f, 0x16, 0x33, 0xa5 -- 0x2b: mov rdx, -ACCEPT
  , 0x48, 0x01, 0xc2              -- 0x35: add rdx, rax
  , 0x75, 0x1f                    -- 0x38: jne +0x1f -> 0x59
  , 0x48, 0xbe, 0x6a, 0xc3, 0xa5, 0x5a, 0x8f, 0x16, 0x33, 0xa5 -- 0x3a: mov rsi, -ACCEPT
  , 0x48, 0x01, 0xde              -- 0x44: add rsi, rbx
  , 0x75, 0x10                    -- 0x47: jne +0x10 -> 0x59
  , 0xb8, 0x3c, 0x00, 0x00, 0x00  -- 0x49: mov eax, 60
  , 0x48, 0x8d, 0xbb, 0x6a, 0xff, 0xff, 0xff -- 0x4e: lea rdi, [rbx - 0x96]
  , 0x0f, 0x05                    -- 0x55: syscall
  , 0xeb, 0x25                    -- 0x57: jmp +0x25 -> 0x7e
  , 0xbf, 0x05, 0x00, 0x00, 0x00  -- 0x59: mov edi, 5
  , 0x48, 0xba, 0x5b, 0x3c, 0x3c, 0xcc, 0xe1, 0xf0, 0x52, 0xf4 -- 0x5e: mov rdx, -OVERFLOW
  , 0x48, 0x01, 0xc2              -- 0x68: add rdx, rax
  , 0x75, 0x11                    -- 0x6b: jne +0x11 -> 0x7e
  , 0xbf, 0x06, 0x00, 0x00, 0x00  -- 0x6d: mov edi, 6
  , 0xb8, 0x3c, 0x00, 0x00, 0x00  -- 0x72: mov eax, 60
  , 0xbf, 0x06, 0x00, 0x00, 0x00  -- 0x77: mov edi, 6
  , 0x0f, 0x05                    -- 0x7c: syscall
  , 0xbf, 0x05, 0x00, 0x00, 0x00  -- 0x7e: mov edi, 5
  , 0xb8, 0x3c, 0x00, 0x00, 0x00  -- 0x83: mov eax, 60
  , 0xbf, 0x05, 0x00, 0x00, 0x00  -- 0x88: mov edi, 5
  , 0x0f, 0x05                    -- 0x8d: syscall
  ]

theorem tailCode_length : tailCode.length = 143 := by decide +kernel

theorem tailCode_voter_prefix :
    tailCode.take 38 = TMRVoter.voterCode.set 0x19 0x0c := by
  decide +kernel

end FlatSpec

section TailProofs

open X86Env (Mem Kern)
open X86Int (Ins St Flags Out Alts Rates Fault lift)

def tailImAt (base : Nat) : Nat → Option (Ins × Nat) :=
  X86Dec.imageOf tailCode base

def tailAltsAt (base : Nat) : Alts := fun a =>
  if a < base then []
  else match tailImAt base a with
    | some (_, len) => X86Dec.altsAt (fun i => tailCode[a - base + i]?) a len
    | none => []

def shiftIns (base : Nat) : Ins → Ins
  | .jmp t => .jmp ((base + t) % X86Env.W)
  | .jcc cc t => .jcc cc ((base + t) % X86Env.W)
  | .call t => .call ((base + t) % X86Env.W)
  | ins => ins

private theorem tgt_shift (base a len rel : Nat) :
    ((base + a) + len + rel) % X86Env.W = (base + (a + len + rel) % X86Env.W) % X86Env.W := by
  rw [Nat.add_assoc base a, Nat.add_assoc base (a + len), Nat.add_mod_mod]

private theorem optMap_bind {α β γ : Type} (x : Option α) (k : α → Option β) (g : β → γ) :
    Option.map g (Option.bind x k) = Option.bind x (fun a => Option.map g (k a)) := by
  cases x <;> rfl

private theorem optMap_ite {α β : Type} (c : Prop) [Decidable c] (a b : Option α) (g : α → β) :
    (if c then a else b).map g = if c then a.map g else b.map g := by
  by_cases h : c <;> simp [h]

theorem decode_shift (b : Nat → Option Nat) (base a : Nat) :
    X86Dec.decode b (base + a) =
      (X86Dec.decode b a).map (fun (ins, len) => (shiftIns base ins, len)) := by
  unfold X86Dec.decode
  simp only [bind, pure, optMap_bind, optMap_ite, Option.map_some, Option.map_none,
    shiftIns, tgt_shift]
  congr 1; funext b0
  congr 1; funext o
  by_cases hff : o = 255
  · subst hff
    cases hrex : (decide (64 ≤ b0) && decide (b0 ≤ 79))
    · simp only [Bool.false_eq_true, ↓reduceIte, Nat.reduceLeDiff, Nat.reduceBeqDiff,
        Bool.or_self, decide_false, Bool.and_false, beq_self_eq_true,
        Nat.zero_div, Nat.zero_mod, Nat.zero_mul, Nat.zero_add]
      split; · rfl
      cases b (0 + 1) with
      | none => rfl
      | some m =>
        dsimp only [Option.bind]
        split; · rfl
        cases b (0 + 2) with
        | none => rfl
        | some s =>
          dsimp only [Option.bind]
          split; · rfl
          split; · rfl
          split; · rfl
          cases m / 64 with
          | zero => rfl
          | succ k =>
            cases k
            · cases b 3 <;> rfl
            · cases X86Dec.leB b 3 4 <;> rfl
    · simp only [Bool.false_eq_true, ↓reduceIte, Nat.reduceLeDiff, Nat.reduceBeqDiff,
        Bool.or_self, decide_false, Bool.and_false, beq_self_eq_true]
      split; · rfl
      cases b (1 + 1) with
      | none => rfl
      | some m =>
        dsimp only [Option.bind]
        split; · rfl
        cases b (1 + 2) with
        | none => rfl
        | some s =>
          dsimp only [Option.bind]
          cases hx : ((b0 - 64) / 2 % 2 == 1) <;>
            simp only [Bool.false_eq_true, ↓reduceIte]
          all_goals
            split; · rfl
            split; · rfl
            split; · rfl
            cases m / 64 with
            | zero => rfl
            | succ k =>
              cases k
              · cases b (1 + 3) <;> rfl
              · cases X86Dec.leB b (1 + 3) 4 <;> rfl
  · simp [beq_false_of_ne hff]

theorem tailImAt_eq (base off : Nat) :
    tailImAt base (base + off) =
      (tailImAt 0 off).map (fun (ins, len) => (shiftIns base ins, len)) := by
  simp only [tailImAt, X86Dec.imageOf, Nat.not_lt.2 (Nat.le_add_right base off), ↓reduceIte,
    Nat.add_sub_cancel_left, Nat.sub_zero]
  exact decode_shift _ base off

theorem tailImAt_lt {base a : Nat} (h : a < base) : tailImAt base a = none := by
  simp [tailImAt, X86Dec.imageOf, h]

theorem tailImAt_ge {base a : Nat} (h : base + 143 ≤ a) : tailImAt base a = none := by
  have hlt : ¬ a < base := by omega
  have hget : tailCode[a - base]? = none := List.getElem?_eq_none (by simp [tailCode]; omega)
  simp only [tailImAt, X86Dec.imageOf, if_neg hlt, X86Dec.decode, bind, Option.bind, Nat.add_zero, hget]

theorem tailAltsAt_eq (base off : Nat) :
    tailAltsAt base (base + off) =
      (tailAltsAt 0 off).map (Option.map fun (ins, len) => (shiftIns base ins, len)) := by
  simp only [tailAltsAt, Nat.not_lt.2 (Nat.le_add_right base off), ↓reduceIte,
    Nat.add_sub_cancel_left, Nat.sub_zero, tailImAt_eq]
  cases tailImAt 0 off with
  | none => rfl
  | some p =>
    simp only [Option.map_some, X86Dec.altsAt, Nat.not_lt_zero, ↓reduceIte, List.map_map]
    exact List.map_congr_left fun k _ => decode_shift _ base off

def preFaultOffs : List Nat :=
  [0x00, 0x03, 0x12, 0x15, 0x18, 0x26, 0x2b, 0x35, 0x38,
   0x59, 0x5e, 0x68, 0x6b, 0x6d, 0x72, 0x77, 0x7c, 0x7e, 0x83, 0x88, 0x8d]

def altJumpRangeOk : Ins → Bool
  | .jmp t => t < 2 ^ 31 || (X86Env.W - 256 ≤ t && t < X86Env.W)
  | .jcc _ t => t < 2 ^ 31 || (X86Env.W - 256 ≤ t && t < X86Env.W)
  | .call t => t < 2 ^ 31 || (X86Env.W - 256 ≤ t && t < X86Env.W)
  | _ => true

def nonTrivialTailOffs : List Nat :=
  [0x00, 0x03, 0x05, 0x08, 0x0a, 0x0d, 0x0f, 0x12, 0x14, 0x15, 0x16, 0x18,
   0x1a, 0x1f, 0x24, 0x26, 0x2b, 0x2e, 0x35, 0x38, 0x3a, 0x3d, 0x44, 0x47,
   0x49, 0x4e, 0x50, 0x55, 0x7c, 0x8d]

def jumpTargetAllowedFromPre (off t : Nat) : Bool :=
  if 143 ≤ t then true
  else
    (t ∉ [0x50, 0x55] || 0x2b ≤ off) &&
    (t ∈ [0x0a, 0x12, 0x14, 0x26, 0x49, 0x51, 0x55, 0x57, 0x59, 0x66, 0x7e] ||
      (tailImAt 0 t).isNone)

def nxSafeGeneric (nx : Nat) : Bool :=
  nx ≥ 143 ||
  ((tailImAt 0 nx).isNone && nx ∉ [0x14, 0x16, 0x2e, 0x3d, 0x50])

def altSafeFromPre (off : Nat) (alt : Option (Ins × Nat)) : Bool :=
  match alt with
  | none => true
  | some (ins, len) =>
    if !altJumpRangeOk ins then false
    else
      let nx := off + len
      match ins with
      | .jmp t => jumpTargetAllowedFromPre off t
      | .jcc _ t =>
        jumpTargetAllowedFromPre off t &&
          (nx ≥ 143 || ((tailImAt 0 nx).isNone && (nx != 0x50 || 0x2b ≤ off)) || nx ∈ [0x05, 0x3a, 0x6d])
      | .call t =>
        jumpTargetAllowedFromPre off t && (143 ≤ t || t ∉ [0x14, 0x2e, 0x3d])
      | .ret | .jmpR _ | .jtab _ _ _ | .syscall => false
      | .push _ | .pushI _ => nxSafeGeneric nx
      | .pop d =>
        d == 0 && (nxSafeGeneric nx || nx ∈ [0x13, 0x16, 0x2c, 0x5f])
      | .st _ _ _ => nxSafeGeneric nx || nx == 0x16
      | .ld _ _ _ | .ld32 _ _ _ | .ld8 _ _ _ | .st8 _ _ _ => false
      | .cmp _ _ | .cmpI _ _ | .test _ _ =>
        nx ≥ 143 || ((tailImAt 0 nx).isNone && (nx != 0x50 || 0x2b ≤ off)) || nx ∈ [0x03, 0x38, 0x6b]
      | .movI d imm =>
        if nx ≥ 143 || ((tailImAt 0 nx).isNone && (nx != 0x50 || 0x2b ≤ off) && (nx ∉ [0x14, 0x16, 0x2e, 0x3d, 0x50] || (d != 3 && d != 4 && d != 7))) then true
        else if off == 0x00 then nx == 0x0a && d != 0 && d != 3
        else if off == 0x26 then nx == 0x2b
        else if off == 0x2b then nx == 0x35 && d ∈ [0, 2, 3, 6, 10] && (d != 3 || imm % X86Env.W != ACCEPT)
        else if off == 0x77 then nx == 0x7c && (d != 7 || imm % 256 != 0)
        else if off == 0x88 then nx == 0x8d && (d != 7 || imm % 256 != 0)
        else off ∈ [0x59, 0x5e, 0x6d, 0x72, 0x7e, 0x83] && (d != 7 || (off ∈ [0x59, 0x6d, 0x7e] && nx != 0x7c && nx != 0x8d)) && nx ∉ [0x00, 0x03, 0x05, 0x08, 0x0a, 0x0d, 0x0f, 0x12, 0x14, 0x15, 0x16, 0x18, 0x1a, 0x1f, 0x24, 0x26, 0x2b, 0x2e, 0x35, 0x38, 0x3a, 0x3d, 0x44, 0x47, 0x49, 0x4e, 0x50, 0x55]
      | .mov d s =>
        if nxSafeGeneric nx then true
        else if off == 0x12 then nx == 0x15 && (d != 0 || s == 3)
        else if off == 0x15 then nx == 0x18 && (d != 0 && d != 3 || s ∈ [0, 1, 3])
        else false
      | .mov32 d s =>
        if nxSafeGeneric nx then true
        else if off == 0x12 then nx == 0x15 && (d != 0 || s == 3)
        else if off == 0x15 then nx == 0x18 && d != 0 && d != 3
        else false
      | .alu op d s =>
        if nxSafeGeneric nx then true
        else if off == 0x00 then nx == 0x03 && d == 0 && s == 3 && (op == .xor || op == .sub)
        else if off == 0x12 then nx == 0x15 && d != 0
        else if off == 0x15 then nx == 0x18 && d != 0 && d != 3
        else if off == 0x35 then nx == 0x38 && ((d != 3 && d != 0) || (d == 3 && op == .add && s == 0) || (d == 0 && op == .add && s == 0))
        else if off == 0x68 then nx == 0x6b
        else false
      | .aluI _ _ _ | .shlI _ _ | .shrI _ _ | .lea _ _ _ =>
        nxSafeGeneric nx || nx ∉ nonTrivialTailOffs

theorem tailAlts_preFault_safe :
    (preFaultOffs.all fun off =>
      (tailAltsAt 0 off).all (altSafeFromPre off)) = true := by
  decide +kernel

/-! ### Part 1: Architectural tail execution theorems (arbitrary `base`) -/

def base_ok (base : Nat) : Prop := 256 ≤ base ∧ base + 2 ^ 32 ≤ X86Env.W

def tailInit (base : Nat) (s0 : St) (a b c : Nat) : St :=
  { TMRVoter.voterInit s0 a b c with rip := base }

def tailSpec : List (Nat × Ins × Nat) :=
  [ (0x00, .cmp 0 3, 3)
  , (0x03, .jcc 4 0x12, 2)
  , (0x05, .cmp 0 1, 3)
  , (0x08, .jcc 4 0x12, 2)
  , (0x0a, .cmp 3 1, 3)
  , (0x0d, .jcc 5 0x1a, 2)
  , (0x0f, .mov 0 3, 3)
  , (0x12, .mov 3 0, 3)
  , (0x15, .mov 1 0, 3)
  , (0x18, .jmp 0x26, 2)
  , (0x1a, .movI 7 3, 5)
  , (0x1f, .movI 0 60, 5)
  , (0x24, .syscall, 2)
  , (0x26, .movI 7 5, 5)
  , (0x2b, .movI 2 0xa533168f5aa5c36a, 10)
  , (0x35, .alu .add 2 0, 3)
  , (0x38, .jcc 5 0x59, 2)
  , (0x3a, .movI 6 0xa533168f5aa5c36a, 10)
  , (0x44, .alu .add 6 3, 3)
  , (0x47, .jcc 5 0x59, 2)
  , (0x49, .movI 0 60, 5)
  , (0x4e, .lea 7 3 (X86Env.W - 0x96), 7)
  , (0x55, .syscall, 2)
  , (0x57, .jmp 0x7e, 2)
  , (0x59, .movI 7 5, 5)
  , (0x5e, .movI 2 0xf452f0e1cc3c3c5b, 10)
  , (0x68, .alu .add 2 0, 3)
  , (0x6b, .jcc 5 0x7e, 2)
  , (0x6d, .movI 7 6, 5)
  , (0x72, .movI 0 60, 5)
  , (0x77, .movI 7 6, 5)
  , (0x7c, .syscall, 2)
  , (0x7e, .movI 7 5, 5)
  , (0x83, .movI 0 60, 5)
  , (0x88, .movI 7 5, 5)
  , (0x8d, .syscall, 2)
  ]

theorem tailSpec_ok :
    tailSpec.all (fun e => tailImAt 0 e.1 == some (e.2.1, e.2.2)) = true := by
  decide +kernel

theorem tailImAt_spec {off : Nat} {ins : Ins} {len : Nat}
    (h : (off, ins, len) ∈ tailSpec) : tailImAt 0 off = some (ins, len) := by
  have := List.all_eq_true.1 tailSpec_ok (off, ins, len) h
  simpa [beq_iff_eq] using this

theorem tailImAt_base_spec {base off : Nat} {ins : Ins} {len : Nat}
    (h : (off, ins, len) ∈ tailSpec) :
    tailImAt base (base + off) = some (shiftIns base ins, len) := by
  rw [tailImAt_eq, tailImAt_spec h]; rfl

private theorem lift_some (s : St) : X86Int.lift (some s) = .ok s := rfl

private theorem run_step_at {im : Nat → Option (Ins × Nat)} {n : Nat} {s : St} {a : Nat}
    {ins : Ins} {len : Nat} (hrip : s.rip = a) (hi : im a = some (ins, len)) :
    X86Int.run im (n + 1) (.ok s) = X86Int.run im n (X86Int.exec ins len s) := by
  subst hrip; simp [X86Int.run, X86Int.step, hi]

private theorem mod_W_3 : 3 % X86Env.W = 3 := by decide
private theorem mod_W_5 : 5 % X86Env.W = 5 := by decide
private theorem mod_W_6 : 6 % X86Env.W = 6 := by decide
private theorem mod_W_60 : 60 % X86Env.W = 60 := by decide

private theorem run_add (im : Nat → Option (Ins × Nat)) :
    ∀ m n o, X86Int.run im (m + n) o = X86Int.run im n (X86Int.run im m o)
  | 0, n, o => by rw [Nat.zero_add]; rfl
  | m + 1, n, .ok s => by
    rw [Nat.add_right_comm]
    simp only [X86Int.run, run_add im m n (X86Int.step im s)]
  | _ + 1, n, .exit c => by cases n <;> simp [X86Int.run]
  | _ + 1, n, .stuck => by cases n <;> simp [X86Int.run]

private theorem run_exit (im : Nat → Option (Ins × Nat)) (c : Nat) :
    ∀ n, X86Int.run im n (.exit c) = .exit c
  | 0 => rfl
  | _ + 1 => rfl

private theorem run_stuck (im : Nat → Option (Ins × Nat)) :
    ∀ n, X86Int.run im n .stuck = .stuck
  | 0 => rfl
  | _ + 1 => rfl

private theorem run_ge_exit {im : Nat → Option (Ins × Nat)} {k c n : Nat} {o : Out}
    (h : X86Int.run im k o = .exit c) (hk : k ≤ n) :
    X86Int.run im n o = .exit c := by
  obtain ⟨d, rfl⟩ := Nat.exists_eq_add_of_le hk
  rw [run_add, h, run_exit]

private theorem aluF_sub_zf_eq (x : Nat) :
    (X86Int.aluF .sub x x).2.zf = some true := by
  simp [X86Int.aluF, Nat.add_sub_cancel_left]

private theorem aluF_sub_zf_ne {x y : Nat} (hx : x < X86Env.W) (hy : y < X86Env.W) (hne : x ≠ y) :
    (X86Int.aluF .sub x y).2.zf = some false := by
  simp only [X86Int.aluF]
  have hpos : 0 < x + X86Env.W - y := by omega
  have hlt : x + X86Env.W - y < 2 * X86Env.W := by omega
  have hneW : x + X86Env.W - y ≠ X86Env.W := by omega
  have hnz : (x + X86Env.W - y) % X86Env.W ≠ 0 := by
    intro hmod
    obtain ⟨k, hk⟩ := Nat.dvd_of_mod_eq_zero hmod
    rcases k with _ | _ | k
    · simp at hk; omega
    · simp at hk; exact hneW hk
    · have hk2 : 2 ≤ k + 2 := by omega
      have hge : 2 * X86Env.W ≤ X86Env.W * (k + 2) := by
        simpa [Nat.mul_comm] using Nat.mul_le_mul_left X86Env.W hk2
      have hkeq : x + X86Env.W - y = X86Env.W * (k + 2) := hk
      omega
  simp [hnz]

private theorem aluF_add_compl_zf {tgt v : Nat} (htgt0 : 0 < tgt) (htgtW : tgt < X86Env.W) (hv : v < X86Env.W) :
    ((X86Int.aluF .add ((X86Env.W - tgt) % X86Env.W) v).2.zf = some true ↔ v = tgt) ∧
    ((X86Int.aluF .add ((X86Env.W - tgt) % X86Env.W) v).2.zf = some false ↔ v ≠ tgt) := by
  have hlt : X86Env.W - tgt < X86Env.W := by omega
  rw [Nat.mod_eq_of_lt hlt]
  simp only [X86Int.aluF]
  have hpos : 0 < (X86Env.W - tgt) + v := by omega
  have h2W : (X86Env.W - tgt) + v < 2 * X86Env.W := by omega
  have hmod : ((X86Env.W - tgt) + v) % X86Env.W = 0 ↔ v = tgt := by
    constructor
    · intro h0
      obtain ⟨k, hk⟩ := Nat.dvd_of_mod_eq_zero h0
      rcases k with _ | _ | k
      · simp at hk; omega
      · simp at hk; omega
      · have hk2 : 2 ≤ k + 2 := by omega
        have hge : 2 * X86Env.W ≤ X86Env.W * (k + 2) := by
          simpa [Nat.mul_comm] using Nat.mul_le_mul_left X86Env.W hk2
        have hkeq : (X86Env.W - tgt) + v = X86Env.W * (k + 2) := hk
        omega
    · intro rfl
      have hsum : (X86Env.W - v) + v = X86Env.W := by omega
      rw [hsum, Nat.mod_self]
  constructor
  · simp [hmod]
  · simp [hmod]

private theorem shift_tgt_lt {base t : Nat} (hb : base_ok base) (ht : t < 143) :
    (base + t) % X86Env.W = base + t :=
  Nat.mod_eq_of_lt (by have := hb.2; omega)

private theorem arch_vote_stage {base : Nat} (hb : base_ok base)
    (s0 : St) {a b c v : Nat} (ha : a < X86Env.W) (hbv : b < X86Env.W) (hc : c < X86Env.W)
    (hv : TMR.vote a b c = some v) :
    ∃ (k : Nat) (fl' : Flags), k ≤ 10 ∧
      X86Int.run (tailImAt base) k (.ok (tailInit base s0 a b c)) =
        .ok { s0 with regs := TMRVoter.voterRegs s0 v v v, rip := base + 0x26, fl := fl' } ∧
      v < X86Env.W := by
  have h12 : ∀ fl1 v1 b1 c1,
      X86Int.run (tailImAt base) 3
        (.ok { s0 with regs := TMRVoter.voterRegs s0 v1 b1 c1, rip := base + 0x12, fl := fl1 }) =
      .ok { s0 with regs := TMRVoter.voterRegs s0 v1 v1 v1, rip := base + 0x26, fl := fl1 } := by
    intro fl1 v1 b1 c1
    have i12 := tailImAt_base_spec (base := base) (by decide : (0x12, .mov 3 0, 3) ∈ tailSpec)
    have i15 := tailImAt_base_spec (base := base) (by decide : (0x15, .mov 1 0, 3) ∈ tailSpec)
    have i18 := tailImAt_base_spec (base := base) (by decide : (0x18, .jmp 0x26, 2) ∈ tailSpec)
    simp only [shiftIns, shift_tgt_lt hb (by decide : 0x26 < 143)] at i12 i15 i18
    rw [run_step_at rfl i12]
    simp only [X86Int.exec, lift_some, bind, Option.bind, pure, St.reg, St.set, TMRVoter.voterRegs,
      List.getElem?_cons_zero, Option.getD_some, List.set]
    rw [run_step_at (by dsimp) i15]
    simp only [X86Int.exec, lift_some, bind, Option.bind, pure, St.reg, St.set,
      List.getElem?_cons_zero, Option.getD_some, List.set]
    rw [run_step_at (by dsimp) i18]
    simp only [X86Int.exec, X86Int.run]
  by_cases hab : a = b
  · subst hab
    have hva : v = a := by unfold TMR.vote at hv; simp at hv; exact hv.symm
    subst hva
    refine ⟨5, (X86Int.aluF .sub v v).2, by omega, ?_, ha⟩
    have i00 := tailImAt_base_spec (base := base) (by decide : (0x00, .cmp 0 3, 3) ∈ tailSpec)
    have i03 := tailImAt_base_spec (base := base) (by decide : (0x03, .jcc 4 0x12, 2) ∈ tailSpec)
    simp only [shiftIns, shift_tgt_lt hb (by decide : 0x12 < 143)] at i00 i03
    rw [show (5 : Nat) = 2 + 3 from rfl, run_add]
    let st12 : St := ⟨TMRVoter.voterRegs s0 v v c, base + 0x12, (X86Int.aluF .sub v v).2, s0.mem, s0.k⟩
    have hpre : X86Int.run (tailImAt base) 2 (.ok (tailInit base s0 v v c)) = .ok st12 := by
      rw [run_step_at (by simp [tailInit, TMRVoter.voterInit]) i00]
      simp only [X86Int.exec, lift_some, bind, Option.bind, pure, St.reg, tailInit, TMRVoter.voterInit,
        TMRVoter.voterRegs, List.getElem?_cons_zero, List.getElem?_cons_succ, Option.getD_some]
      rw [run_step_at (by dsimp) i03]
      simp [X86Int.exec, X86Int.cond, aluF_sub_zf_eq, lift_some, St.reg, TMRVoter.voterRegs, X86Int.run, st12]
    rw [hpre]
    exact h12 _ v v c
  · have hzf00 : (X86Int.aluF .sub a b).2.zf = some false := aluF_sub_zf_ne ha hbv hab
    have i00 := tailImAt_base_spec (base := base) (by decide : (0x00, .cmp 0 3, 3) ∈ tailSpec)
    have i03 := tailImAt_base_spec (base := base) (by decide : (0x03, .jcc 4 0x12, 2) ∈ tailSpec)
    simp only [shiftIns, shift_tgt_lt hb (by decide : 0x12 < 143)] at i00 i03
    have hstep0 : X86Int.run (tailImAt base) 2 (.ok (tailInit base s0 a b c)) =
        .ok { s0 with regs := TMRVoter.voterRegs s0 a b c, rip := base + 0x05, fl := (X86Int.aluF .sub a b).2 } := by
      rw [run_step_at (by simp [tailInit, TMRVoter.voterInit]) i00]
      simp only [X86Int.exec, lift_some, bind, Option.bind, pure, St.reg, tailInit, TMRVoter.voterInit,
        TMRVoter.voterRegs, List.getElem?_cons_zero, List.getElem?_cons_succ, Option.getD_some]
      rw [run_step_at (by dsimp) i03]
      simp [X86Int.exec, X86Int.cond, hzf00, lift_some, X86Int.run]
    by_cases hac : a = c
    · subst hac
      have hva : v = a := by unfold TMR.vote at hv; simp [hab] at hv; exact hv.symm
      subst hva
      refine ⟨7, (X86Int.aluF .sub v v).2, by omega, ?_, ha⟩
      have i05 := tailImAt_base_spec (base := base) (by decide : (0x05, .cmp 0 1, 3) ∈ tailSpec)
      have i08 := tailImAt_base_spec (base := base) (by decide : (0x08, .jcc 4 0x12, 2) ∈ tailSpec)
      simp only [shiftIns, shift_tgt_lt hb (by decide : 0x12 < 143)] at i05 i08
      rw [show (7 : Nat) = 2 + (2 + 3) from rfl, run_add, hstep0, run_add]
      have hstep1 : X86Int.run (tailImAt base) 2
          (.ok { s0 with regs := TMRVoter.voterRegs s0 v b v, rip := base + 0x05, fl := (X86Int.aluF .sub v b).2 }) =
          .ok { s0 with regs := TMRVoter.voterRegs s0 v b v, rip := base + 0x12, fl := (X86Int.aluF .sub v v).2 } := by
        rw [run_step_at rfl i05]
        simp only [X86Int.exec, lift_some, bind, Option.bind, pure, St.reg, TMRVoter.voterRegs,
          List.getElem?_cons_zero, List.getElem?_cons_succ, Option.getD_some]
        rw [run_step_at (by dsimp) i08]
        simp [X86Int.exec, X86Int.cond, aluF_sub_zf_eq, lift_some, X86Int.run]
      rw [hstep1]
      exact h12 _ v b v
    · have hzf05 : (X86Int.aluF .sub a c).2.zf = some false := aluF_sub_zf_ne ha hc hac
      have hbc : b = c := by
        by_cases hbc : b = c
        · exact hbc
        · exfalso; unfold TMR.vote at hv; simp [hab, hac, hbc] at hv
      subst hbc
      have hvb : v = b := by unfold TMR.vote at hv; simp [hab] at hv; exact hv.symm
      subst hvb
      refine ⟨10, (X86Int.aluF .sub v v).2, by omega, ?_, hbv⟩
      have i05 := tailImAt_base_spec (base := base) (by decide : (0x05, .cmp 0 1, 3) ∈ tailSpec)
      have i08 := tailImAt_base_spec (base := base) (by decide : (0x08, .jcc 4 0x12, 2) ∈ tailSpec)
      have i0a := tailImAt_base_spec (base := base) (by decide : (0x0a, .cmp 3 1, 3) ∈ tailSpec)
      have i0d := tailImAt_base_spec (base := base) (by decide : (0x0d, .jcc 5 0x1a, 2) ∈ tailSpec)
      have i0f := tailImAt_base_spec (base := base) (by decide : (0x0f, .mov 0 3, 3) ∈ tailSpec)
      simp only [shiftIns, shift_tgt_lt hb (by decide : 0x12 < 143),
        shift_tgt_lt hb (by decide : 0x1a < 143)] at i05 i08 i0a i0d i0f
      rw [show (10 : Nat) = 2 + (5 + 3) from rfl, run_add, hstep0, run_add]
      have hmid : X86Int.run (tailImAt base) 5
          (.ok { s0 with regs := TMRVoter.voterRegs s0 a v v, rip := base + 0x05, fl := (X86Int.aluF .sub a v).2 }) =
          .ok { s0 with regs := TMRVoter.voterRegs s0 v v v, rip := base + 0x12, fl := (X86Int.aluF .sub v v).2 } := by
        rw [run_step_at rfl i05]
        simp only [X86Int.exec, lift_some, bind, Option.bind, pure, St.reg, TMRVoter.voterRegs,
          List.getElem?_cons_zero, List.getElem?_cons_succ, Option.getD_some]
        rw [run_step_at (by dsimp) i08]
        simp [X86Int.exec, X86Int.cond, hzf05, lift_some]
        rw [run_step_at (by dsimp) i0a]
        simp only [X86Int.exec, lift_some, bind, Option.bind, pure, St.reg,
          List.getElem?_cons_zero, List.getElem?_cons_succ, Option.getD_some]
        rw [run_step_at (by dsimp) i0d]
        simp [X86Int.exec, X86Int.cond, aluF_sub_zf_eq, lift_some]
        rw [run_step_at (by dsimp) i0f]
        simp [X86Int.exec, lift_some, bind, Option.bind, pure, St.reg, St.set,
          List.set, X86Int.run]
      rw [hmid]
      exact h12 _ v v v

theorem tail_arch_split {base : Nat} (hb : base_ok base)
    (s0 : St) {a b c : Nat} (ha : a < X86Env.W) (hbv : b < X86Env.W) (hc : c < X86Env.W)
    (hv : TMR.vote a b c = none) :
    ∀ n, 9 ≤ n → X86Int.run (tailImAt base) n (.ok (tailInit base s0 a b c)) = .exit 3 := by
  have hab : a ≠ b := by intro h; subst h; unfold TMR.vote at hv; simp at hv
  have hac : a ≠ c := by intro h; subst h; unfold TMR.vote at hv; simp at hv
  have hbc : b ≠ c := by intro h; subst h; unfold TMR.vote at hv; simp [hab] at hv
  have i00 := tailImAt_base_spec (base := base) (by decide : (0x00, .cmp 0 3, 3) ∈ tailSpec)
  have i03 := tailImAt_base_spec (base := base) (by decide : (0x03, .jcc 4 0x12, 2) ∈ tailSpec)
  have i05 := tailImAt_base_spec (base := base) (by decide : (0x05, .cmp 0 1, 3) ∈ tailSpec)
  have i08 := tailImAt_base_spec (base := base) (by decide : (0x08, .jcc 4 0x12, 2) ∈ tailSpec)
  have i0a := tailImAt_base_spec (base := base) (by decide : (0x0a, .cmp 3 1, 3) ∈ tailSpec)
  have i0d := tailImAt_base_spec (base := base) (by decide : (0x0d, .jcc 5 0x1a, 2) ∈ tailSpec)
  have i1a := tailImAt_base_spec (base := base) (by decide : (0x1a, .movI 7 3, 5) ∈ tailSpec)
  have i1f := tailImAt_base_spec (base := base) (by decide : (0x1f, .movI 0 60, 5) ∈ tailSpec)
  have i24 := tailImAt_base_spec (base := base) (by decide : (0x24, .syscall, 2) ∈ tailSpec)
  simp only [shiftIns, shift_tgt_lt hb (by decide : 0x12 < 143),
    shift_tgt_lt hb (by decide : 0x1a < 143)] at i00 i03 i05 i08 i0a i0d i1a i1f i24
  have h9 : X86Int.run (tailImAt base) 9 (.ok (tailInit base s0 a b c)) = .exit 3 := by
    rw [run_step_at (by simp [tailInit, TMRVoter.voterInit]) i00]
    simp only [X86Int.exec, lift_some, bind, Option.bind, pure, St.reg, tailInit, TMRVoter.voterInit,
      TMRVoter.voterRegs, List.getElem?_cons_zero, List.getElem?_cons_succ, Option.getD_some]
    rw [run_step_at (by dsimp) i03]
    simp [X86Int.exec, X86Int.cond, aluF_sub_zf_ne ha hbv hab, lift_some]
    rw [run_step_at (by dsimp) i05]
    simp only [X86Int.exec, lift_some, bind, Option.bind, pure, St.reg,
      List.getElem?_cons_zero, List.getElem?_cons_succ, Option.getD_some]
    rw [run_step_at (by dsimp) i08]
    simp [X86Int.exec, X86Int.cond, aluF_sub_zf_ne ha hc hac, lift_some]
    rw [run_step_at (by dsimp) i0a]
    simp only [X86Int.exec, lift_some, bind, Option.bind, pure, St.reg,
      List.getElem?_cons_zero, List.getElem?_cons_succ, Option.getD_some]
    rw [run_step_at (by dsimp) i0d]
    simp [X86Int.exec, X86Int.cond, aluF_sub_zf_ne hbv hc hbc, lift_some]
    rw [run_step_at (by dsimp) i1a]
    simp only [X86Int.exec, St.set, List.set]
    rw [run_step_at (by dsimp) i1f]
    simp only [X86Int.exec, St.set, List.set]
    rw [run_step_at (by dsimp) i24]
    simp [X86Int.exec, St.reg, X86Env.syscall, mod_W_60, mod_W_3, X86Int.run]
  intro n hn
  exact run_ge_exit h9 hn


theorem tail_arch_accept {base : Nat} (hb : base_ok base)
    (s0 : St) {a b c : Nat} (ha : a < X86Env.W) (hbv : b < X86Env.W) (hc : c < X86Env.W)
    (hv : TMR.vote a b c = some ACCEPT) :
    ∀ n, 20 ≤ n → X86Int.run (tailImAt base) n (.ok (tailInit base s0 a b c)) = .exit 0 := by
  obtain ⟨k, fl', hk, hrun0, _⟩ := arch_vote_stage hb s0 ha hbv hc hv
  have hacc0 : 0 < ACCEPT := by decide
  have haccW : ACCEPT < X86Env.W := by decide
  have hzf : (X86Int.aluF .add (0xa533168f5aa5c36a % X86Env.W) ACCEPT).2.zf = some true := by
    have hm : (0xa533168f5aa5c36a : Nat) % X86Env.W = (X86Env.W - ACCEPT) % X86Env.W := by decide
    rw [hm]; exact (aluF_add_compl_zf hacc0 haccW haccW).1.mpr rfl
  have i26 := tailImAt_base_spec (base := base) (by decide : (0x26, .movI 7 5, 5) ∈ tailSpec)
  have i2b := tailImAt_base_spec (base := base) (by decide : (0x2b, .movI 2 0xa533168f5aa5c36a, 10) ∈ tailSpec)
  have i35 := tailImAt_base_spec (base := base) (by decide : (0x35, .alu .add 2 0, 3) ∈ tailSpec)
  have i38 := tailImAt_base_spec (base := base) (by decide : (0x38, .jcc 5 0x59, 2) ∈ tailSpec)
  have i3a := tailImAt_base_spec (base := base) (by decide : (0x3a, .movI 6 0xa533168f5aa5c36a, 10) ∈ tailSpec)
  have i44 := tailImAt_base_spec (base := base) (by decide : (0x44, .alu .add 6 3, 3) ∈ tailSpec)
  have i47 := tailImAt_base_spec (base := base) (by decide : (0x47, .jcc 5 0x59, 2) ∈ tailSpec)
  have i49 := tailImAt_base_spec (base := base) (by decide : (0x49, .movI 0 60, 5) ∈ tailSpec)
  have i4e := tailImAt_base_spec (base := base) (by decide : (0x4e, .lea 7 3 (X86Env.W - 0x96), 7) ∈ tailSpec)
  have i55 := tailImAt_base_spec (base := base) (by decide : (0x55, .syscall, 2) ∈ tailSpec)
  simp only [shiftIns, shift_tgt_lt hb (by decide : 0x59 < 143)] at i26 i2b i35 i38 i3a i44 i47 i49 i4e i55
  have h10 : X86Int.run (tailImAt base) 10
      (.ok { s0 with regs := TMRVoter.voterRegs s0 ACCEPT ACCEPT ACCEPT, rip := base + 0x26, fl := fl' }) =
      .exit 0 := by
    have mod_lea : (ACCEPT + (X86Env.W - 0x96)) % X86Env.W % 256 = 0 := by decide
    rw [run_step_at rfl i26]
    simp only [X86Int.exec, St.set, TMRVoter.voterRegs, List.set]
    rw [run_step_at (by dsimp) i2b]
    simp only [X86Int.exec, St.set, List.set]
    rw [run_step_at (by dsimp) i35]
    simp only [X86Int.exec, lift_some, bind, Option.bind, pure, St.reg, St.set, List.set,
      List.getElem?_cons_zero, List.getElem?_cons_succ, Option.getD_some]
    rw [run_step_at (by dsimp) i38]
    simp [X86Int.exec, X86Int.cond, hzf, lift_some]
    rw [run_step_at (by dsimp) i3a]
    simp only [X86Int.exec, St.set, List.set]
    rw [run_step_at (by dsimp) i44]
    simp only [X86Int.exec, lift_some, bind, Option.bind, pure, St.reg, St.set, List.set,
      List.getElem?_cons_zero, List.getElem?_cons_succ, Option.getD_some]
    rw [run_step_at (by dsimp) i47]
    simp [X86Int.exec, X86Int.cond, hzf, lift_some]
    rw [run_step_at (by dsimp) i49]
    simp only [X86Int.exec, St.set, List.set]
    rw [run_step_at (by dsimp) i4e]
    simp only [X86Int.exec, lift_some, bind, Option.bind, pure, St.reg, St.set, List.set,
      List.getElem?_cons_zero, List.getElem?_cons_succ, Option.getD_some, X86Int.ea]
    rw [run_step_at (by dsimp) i55]
    simp [X86Int.exec, St.reg, X86Env.syscall, mod_W_60, mod_lea, X86Int.run]
  have htot : X86Int.run (tailImAt base) (k + 10) (.ok (tailInit base s0 a b c)) = .exit 0 := by
    rw [run_add, hrun0, h10]
  intro n hn
  exact run_ge_exit htot (by omega)

private theorem tail_arch_guard1_fail {base : Nat} (hb : base_ok base)
    (s0 : St) (fl' : Flags) {v : Nat} (hvW : v < X86Env.W) (hne : v ≠ ACCEPT) :
    X86Int.run (tailImAt base) 4
      (.ok { s0 with regs := TMRVoter.voterRegs s0 v v v, rip := base + 0x26, fl := fl' }) =
      .ok { s0 with
            regs := ((TMRVoter.voterRegs s0 v v v).set 7 (some 5)).set 2 (some (X86Int.aluF .add (0xa533168f5aa5c36a % X86Env.W) v).1),
            rip := base + 0x59,
            fl := (X86Int.aluF .add (0xa533168f5aa5c36a % X86Env.W) v).2 } := by
  have hacc0 : 0 < ACCEPT := by decide
  have haccW : ACCEPT < X86Env.W := by decide
  have hzf : (X86Int.aluF .add (0xa533168f5aa5c36a % X86Env.W) v).2.zf = some false := by
    have hm : (0xa533168f5aa5c36a : Nat) % X86Env.W = (X86Env.W - ACCEPT) % X86Env.W := by decide
    rw [hm]; exact (aluF_add_compl_zf hacc0 haccW hvW).2.mpr hne
  have i26 := tailImAt_base_spec (base := base) (by decide : (0x26, .movI 7 5, 5) ∈ tailSpec)
  have i2b := tailImAt_base_spec (base := base) (by decide : (0x2b, .movI 2 0xa533168f5aa5c36a, 10) ∈ tailSpec)
  have i35 := tailImAt_base_spec (base := base) (by decide : (0x35, .alu .add 2 0, 3) ∈ tailSpec)
  have i38 := tailImAt_base_spec (base := base) (by decide : (0x38, .jcc 5 0x59, 2) ∈ tailSpec)
  simp only [shiftIns, shift_tgt_lt hb (by decide : 0x59 < 143)] at i26 i2b i35 i38
  rw [run_step_at rfl i26]
  simp only [X86Int.exec, St.set]
  rw [run_step_at (by dsimp) i2b]
  simp only [X86Int.exec, St.set]
  rw [run_step_at (by dsimp) i35]
  simp only [X86Int.exec, lift_some, bind, Option.bind, pure, St.reg, St.set, List.set,
    TMRVoter.voterRegs, List.getElem?_cons_zero, List.getElem?_cons_succ, Option.getD_some]
  rw [run_step_at (by dsimp) i38]
  simp [X86Int.exec, X86Int.cond, hzf, lift_some, X86Int.run, mod_W_5]

theorem tail_arch_overflow {base : Nat} (hb : base_ok base)
    (s0 : St) {a b c : Nat} (ha : a < X86Env.W) (hbv : b < X86Env.W) (hc : c < X86Env.W)
    (hv : TMR.vote a b c = some OVERFLOW) :
    ∀ n, 22 ≤ n → X86Int.run (tailImAt base) n (.ok (tailInit base s0 a b c)) = .exit 6 := by
  obtain ⟨k, fl', hk, hrun0, _⟩ := arch_vote_stage hb s0 ha hbv hc hv
  have hov0 : 0 < OVERFLOW := by decide
  have hovW : OVERFLOW < X86Env.W := by decide
  have hne : OVERFLOW ≠ ACCEPT := by decide
  have hzf : (X86Int.aluF .add (0xf452f0e1cc3c3c5b % X86Env.W) OVERFLOW).2.zf = some true := by
    have hm : (0xf452f0e1cc3c3c5b : Nat) % X86Env.W = (X86Env.W - OVERFLOW) % X86Env.W := by decide
    rw [hm]; exact (aluF_add_compl_zf hov0 hovW hovW).1.mpr rfl
  have hg1 := tail_arch_guard1_fail hb s0 fl' hovW hne
  have i59 := tailImAt_base_spec (base := base) (by decide : (0x59, .movI 7 5, 5) ∈ tailSpec)
  have i5e := tailImAt_base_spec (base := base) (by decide : (0x5e, .movI 2 0xf452f0e1cc3c3c5b, 10) ∈ tailSpec)
  have i68 := tailImAt_base_spec (base := base) (by decide : (0x68, .alu .add 2 0, 3) ∈ tailSpec)
  have i6b := tailImAt_base_spec (base := base) (by decide : (0x6b, .jcc 5 0x7e, 2) ∈ tailSpec)
  have i6d := tailImAt_base_spec (base := base) (by decide : (0x6d, .movI 7 6, 5) ∈ tailSpec)
  have i72 := tailImAt_base_spec (base := base) (by decide : (0x72, .movI 0 60, 5) ∈ tailSpec)
  have i77 := tailImAt_base_spec (base := base) (by decide : (0x77, .movI 7 6, 5) ∈ tailSpec)
  have i7c := tailImAt_base_spec (base := base) (by decide : (0x7c, .syscall, 2) ∈ tailSpec)
  simp only [shiftIns, shift_tgt_lt hb (by decide : 0x7e < 143)] at i59 i5e i68 i6b i6d i72 i77 i7c
  have h8 : X86Int.run (tailImAt base) 8
      (.ok { s0 with
             regs := ((TMRVoter.voterRegs s0 OVERFLOW OVERFLOW OVERFLOW).set 7 (some 5)).set 2 (some (X86Int.aluF .add (0xa533168f5aa5c36a % X86Env.W) OVERFLOW).1),
             rip := base + 0x59,
             fl := (X86Int.aluF .add (0xa533168f5aa5c36a % X86Env.W) OVERFLOW).2 }) = .exit 6 := by
    rw [run_step_at rfl i59]
    simp only [X86Int.exec, St.set]
    rw [run_step_at (by dsimp) i5e]
    simp only [X86Int.exec, St.set]
    rw [run_step_at (by dsimp) i68]
    simp only [X86Int.exec, lift_some, bind, Option.bind, pure, St.reg, St.set, List.set,
      TMRVoter.voterRegs, List.getElem?_cons_zero, List.getElem?_cons_succ, Option.getD_some]
    rw [run_step_at (by dsimp) i6b]
    simp [X86Int.exec, X86Int.cond, hzf, lift_some]
    rw [run_step_at (by dsimp) i6d]
    simp only [X86Int.exec, St.set]
    rw [run_step_at (by dsimp) i72]
    simp only [X86Int.exec, St.set]
    rw [run_step_at (by dsimp) i77]
    simp only [X86Int.exec, St.set]
    rw [run_step_at (by dsimp) i7c]
    simp [X86Int.exec, St.reg, X86Env.syscall, mod_W_60, mod_W_6, X86Int.run]
  have htot : X86Int.run (tailImAt base) (k + (4 + 8)) (.ok (tailInit base s0 a b c)) = .exit 6 := by
    rw [run_add, hrun0, run_add, hg1, h8]
  intro n hn
  exact run_ge_exit htot (by omega)

theorem tail_arch_reject {base : Nat} (hb : base_ok base)
    (s0 : St) {a b c v : Nat} (ha : a < X86Env.W) (hbv : b < X86Env.W) (hc : c < X86Env.W)
    (hv : TMR.vote a b c = some v) (hneA : v ≠ ACCEPT) (hneO : v ≠ OVERFLOW) :
    ∀ n, 22 ≤ n → X86Int.run (tailImAt base) n (.ok (tailInit base s0 a b c)) = .exit 5 := by
  obtain ⟨k, fl', hk, hrun0, hvW⟩ := arch_vote_stage hb s0 ha hbv hc hv
  have hov0 : 0 < OVERFLOW := by decide
  have hovW : OVERFLOW < X86Env.W := by decide
  have hzf : (X86Int.aluF .add (0xf452f0e1cc3c3c5b % X86Env.W) v).2.zf = some false := by
    have hm : (0xf452f0e1cc3c3c5b : Nat) % X86Env.W = (X86Env.W - OVERFLOW) % X86Env.W := by decide
    rw [hm]; exact (aluF_add_compl_zf hov0 hovW hvW).2.mpr hneO
  have hg1 := tail_arch_guard1_fail hb s0 fl' hvW hneA
  have i59 := tailImAt_base_spec (base := base) (by decide : (0x59, .movI 7 5, 5) ∈ tailSpec)
  have i5e := tailImAt_base_spec (base := base) (by decide : (0x5e, .movI 2 0xf452f0e1cc3c3c5b, 10) ∈ tailSpec)
  have i68 := tailImAt_base_spec (base := base) (by decide : (0x68, .alu .add 2 0, 3) ∈ tailSpec)
  have i6b := tailImAt_base_spec (base := base) (by decide : (0x6b, .jcc 5 0x7e, 2) ∈ tailSpec)
  have i7e := tailImAt_base_spec (base := base) (by decide : (0x7e, .movI 7 5, 5) ∈ tailSpec)
  have i83 := tailImAt_base_spec (base := base) (by decide : (0x83, .movI 0 60, 5) ∈ tailSpec)
  have i88 := tailImAt_base_spec (base := base) (by decide : (0x88, .movI 7 5, 5) ∈ tailSpec)
  have i8d := tailImAt_base_spec (base := base) (by decide : (0x8d, .syscall, 2) ∈ tailSpec)
  simp only [shiftIns, shift_tgt_lt hb (by decide : 0x7e < 143)] at i59 i5e i68 i6b i7e i83 i88 i8d
  have h8 : X86Int.run (tailImAt base) 8
      (.ok { s0 with
             regs := ((TMRVoter.voterRegs s0 v v v).set 7 (some 5)).set 2 (some (X86Int.aluF .add (0xa533168f5aa5c36a % X86Env.W) v).1),
             rip := base + 0x59,
             fl := (X86Int.aluF .add (0xa533168f5aa5c36a % X86Env.W) v).2 }) = .exit 5 := by
    rw [run_step_at rfl i59]
    simp only [X86Int.exec, St.set]
    rw [run_step_at (by dsimp) i5e]
    simp only [X86Int.exec, St.set]
    rw [run_step_at (by dsimp) i68]
    simp only [X86Int.exec, lift_some, bind, Option.bind, pure, St.reg, St.set, List.set,
      TMRVoter.voterRegs, List.getElem?_cons_zero, List.getElem?_cons_succ, Option.getD_some]
    rw [run_step_at (by dsimp) i6b]
    simp [X86Int.exec, X86Int.cond, hzf, lift_some]
    rw [run_step_at (by dsimp) i7e]
    simp only [X86Int.exec, St.set]
    rw [run_step_at (by dsimp) i83]
    simp only [X86Int.exec, St.set]
    rw [run_step_at (by dsimp) i88]
    simp only [X86Int.exec, St.set]
    rw [run_step_at (by dsimp) i8d]
    simp [X86Int.exec, St.reg, X86Env.syscall, mod_W_60, mod_W_5, X86Int.run]
  have htot : X86Int.run (tailImAt base) (k + (4 + 8)) (.ok (tailInit base s0 a b c)) = .exit 5 := by
    rw [run_add, hrun0, run_add, hg1, h8]
  intro n hn
  exact run_ge_exit htot (by omega)


/-! ### Part 2: Post-fault safety invariant `TailSafe` -/

def valNotAcc (w : Nat) : Prop := w < X86Env.W ∧ w ≠ ACCEPT
def lowNotAcc (w : Nat) : Prop := w < X86Env.W ∧ w % 256 ≠ ACCEPT % 256

def SafeInvalid (v : Nat) : Prop :=
  v < X86Env.W ∧ v ≠ ACCEPT ∧ v % 256 ≠ ACCEPT % 256 ∧
  (2 * v) % X86Env.W ≠ 0 ∧ (2 * v) % X86Env.W ≠ ACCEPT ∧ (2 * v) % X86Env.W % 256 ≠ ACCEPT % 256 ∧
  ∀ b < 64, Nat.xor v (2 ^ b) < X86Env.W ∧ Nat.xor v (2 ^ b) ≠ ACCEPT ∧ (Nat.xor v (2 ^ b)) % 256 ≠ ACCEPT % 256

theorem SafeInvalid_reject : SafeInvalid REJECT := by
  refine ⟨by decide, by decide, by decide, by decide, by decide, by decide, ?_⟩
  have h : (List.range 64).all (fun b =>
      decide (Nat.xor REJECT (2 ^ b) < X86Env.W) &&
      decide (Nat.xor REJECT (2 ^ b) ≠ ACCEPT) &&
      decide ((Nat.xor REJECT (2 ^ b)) % 256 ≠ ACCEPT % 256)) = true := by
    decide +kernel
  intro b hb
  have hmem := List.all_eq_true.1 h b (List.mem_range.2 hb)
  simpa [and_assoc] using hmem

theorem SafeInvalid_overflow : SafeInvalid OVERFLOW := by
  refine ⟨by decide, by decide, by decide, by decide, by decide, by decide, ?_⟩
  have h : (List.range 64).all (fun b =>
      decide (Nat.xor OVERFLOW (2 ^ b) < X86Env.W) &&
      decide (Nat.xor OVERFLOW (2 ^ b) ≠ ACCEPT) &&
      decide ((Nat.xor OVERFLOW (2 ^ b)) % 256 ≠ ACCEPT % 256)) = true := by
    decide +kernel
  intro b hb
  have hmem := List.all_eq_true.1 h b (List.mem_range.2 hb)
  simpa [and_assoc] using hmem

def TailStackSafe (base : Nat) (s : St) : Prop :=
  ∀ sp, s.reg 4 = some sp → ∀ t, X86Int.rd64 s.mem sp = some t → ¬ (base ≤ t ∧ t < base + 143)

def hasValNotAcc (s : St) (r : Nat) : Prop :=
  ∃ w, s.reg r = some w ∧ valNotAcc w

def hasLowNotAcc (s : St) (r : Nat) : Prop :=
  ∃ w, s.reg r = some w ∧ lowNotAcc w

def hasExitNonZero (s : St) : Prop :=
  ∀ a1, s.reg 7 = some a1 → a1 % 256 ≠ 0

def TailSafeOff (base : Nat) (s : St) (off : Nat) : Prop :=
  match off with
  | 0x00 | 0x03 | 0x05 | 0x08 =>
      hasValNotAcc s 0 ∧ hasValNotAcc s 3 ∧ hasValNotAcc s 1
  | 0x0a | 0x0d =>
      hasValNotAcc s 0 ∧ hasValNotAcc s 3
  | 0x0f => hasValNotAcc s 3
  | 0x10 => True
  | 0x12 => hasValNotAcc s 0
  | 0x13 => True
  | 0x14 | 0x2e | 0x3d => TailStackSafe base s
  | 0x15 | 0x16 | 0x18 => hasValNotAcc s 0 ∨ hasValNotAcc s 3
  | 0x1a => hasValNotAcc s 3
  | 0x1f | 0x24 => hasExitNonZero s ∧ hasValNotAcc s 3
  | 0x26 | 0x2b => hasValNotAcc s 0 ∨ hasValNotAcc s 3
  | 0x35 =>
      (s.reg 2 = some ((X86Env.W - ACCEPT) % X86Env.W) ∧ hasValNotAcc s 0) ∨ hasValNotAcc s 3
  | 0x38 => s.fl.zf ≠ some true ∨ hasValNotAcc s 3
  | 0x3a => hasValNotAcc s 3
  | 0x44 => s.reg 6 = some ((X86Env.W - ACCEPT) % X86Env.W) ∧ hasValNotAcc s 3
  | 0x47 => s.fl.zf ≠ some true
  | 0x49 | 0x4e => hasLowNotAcc s 3
  | 0x50 | 0x55 | 0x7c | 0x8d => hasExitNonZero s
  | _ => True

def TailSafe (base : Nat) (s : St) : Prop :=
  s.regs.length = 16 ∧
  (base ≤ s.rip → s.rip < base + 143 → TailSafeOff base s (s.rip - base))

def tailExtraSpec : List (Nat × Ins × Nat) :=
  [ (0x10, .mov32 0 3, 2)
  , (0x13, .mov32 3 0, 2)
  , (0x14, .ret, 1)
  , (0x16, .mov32 1 0, 2)
  , (0x2c, .movI 2 0x5aa5c36a, 5)
  , (0x2d, .pushI (X86Env.W - 61), 2)
  , (0x2e, .ret, 1)
  , (0x30, .pop 2, 1)
  , (0x3b, .movI 6 0x5aa5c36a, 5)
  , (0x3c, .pushI (X86Env.W - 61), 2)
  , (0x3d, .ret, 1)
  , (0x3f, .pop 2, 1)
  , (0x50, .movI 3 0xffffff6a, 5)
  , (0x51, .pushI (X86Env.W - 1), 2)
  , (0x5f, .movI 2 0xcc3c3c5b, 5)
  , (0x60, .pop 3, 1)
  , (0x66, .push 2, 1)
  ]

def tailAllSpec : List (Nat × Ins × Nat) := tailSpec ++ tailExtraSpec

theorem tailAllSpec_ok :
    tailAllSpec.all (fun e => tailImAt 0 e.1 == some (e.2.1, e.2.2)) = true := by
  decide +kernel

theorem tailAllOffs_cover :
    (List.range 143).all (fun off =>
      (tailImAt 0 off).isNone || tailAllSpec.any (fun e => e.1 == off)) = true := by
  decide +kernel

theorem tailImAt_base_all {base off : Nat} {ins : Ins} {len : Nat}
    (h : (off, ins, len) ∈ tailAllSpec) :
    tailImAt base (base + off) = some (shiftIns base ins, len) := by
  have heq := List.all_eq_true.1 tailAllSpec_ok (off, ins, len) h
  rw [tailImAt_eq, (by simpa [beq_iff_eq] using heq : tailImAt 0 off = some (ins, len))]
  rfl

private theorem reg_set_eq {s : St} (hlen : s.regs.length = 16) {r : Nat} (hr : r < 16) (v : Option Nat) :
    (s.set r v).reg r = v := by
  simp only [St.reg, St.set, List.getElem?_set_self (by omega : r < s.regs.length), Option.getD_some]

private theorem reg_set_ne {s : St} {r i : Nat} (hne : i ≠ r) (v : Option Nat) :
    (s.set i v).reg r = s.reg r := by
  simp only [St.reg, St.set, List.getElem?_set_ne hne]

private theorem reg_set_mk {s : St} (hlen : s.regs.length = 16) {r : Nat} (hr : r < 16)
    (v : Option Nat) (rip : Nat) (fl : Flags) (mem : Mem) (k : Kern) :
    ({ regs := (s.set r v).regs, rip := rip, fl := fl, mem := mem, k := k } : St).reg r = v := by
  simp only [St.reg, St.set, List.getElem?_set_self (by omega : r < s.regs.length), Option.getD_some]

private theorem reg_set_mk_ne {s : St} {r i : Nat} (hne : r ≠ i)
    (v : Option Nat) (rip : Nat) (fl : Flags) (mem : Mem) (k : Kern) :
    ({ regs := (s.set r v).regs, rip := rip, fl := fl, mem := mem, k := k } : St).reg i = s.reg i := by
  simp only [St.reg, St.set, List.getElem?_set_ne hne]

private theorem set_length {s : St} (hlen : s.regs.length = 16) (r : Nat) (v : Option Nat) :
    (s.set r v).regs.length = 16 := by
  simp [St.set, hlen]

private theorem valNotAcc_mod32 (x : Nat) : valNotAcc (x % 2 ^ 32) :=
  ⟨Nat.lt_trans (Nat.mod_lt _ (by decide)) (by decide : 2 ^ 32 < X86Env.W),
   Nat.ne_of_lt (Nat.lt_trans (Nat.mod_lt _ (by decide)) (by decide : 2 ^ 32 < ACCEPT))⟩

private theorem lea_exit_nz {w : Nat} (h : lowNotAcc w) :
    ((w + (X86Env.W - 0x96)) % X86Env.W) % 256 ≠ 0 := by
  rcases h with ⟨hw, hlow⟩
  have h150 : ACCEPT % 256 = 0x96 := by decide
  rw [h150] at hlow
  have hdvd : 256 ∣ X86Env.W := by decide
  have hmod : ((w + (X86Env.W - 0x96)) % X86Env.W) % 256 = (w + (X86Env.W - 0x96)) % 256 := Nat.mod_mod_of_dvd _ hdvd
  rw [hmod]
  have hW : X86Env.W - 0x96 = 256 * (2 ^ 56 - 1) + 0x6a := by decide
  omega

private theorem cond5_false_of_zf_ne {f : Flags} (h : f.zf ≠ some true) :
    X86Int.cond f 5 ≠ some false := by
  rcases f with ⟨cf, zf, sf, of⟩
  cases zf with
  | none =>
    intro hfl
    have heq : X86Int.cond ⟨cf, none, sf, of⟩ 5 = none := rfl
    rw [heq] at hfl; cases hfl
  | some b =>
    cases b with
    | false =>
      intro hfl
      have heq : X86Int.cond ⟨cf, some false, sf, of⟩ 5 = some true := rfl
      rw [heq] at hfl; cases hfl
    | true => exact absurd rfl h

private theorem syscall_exit_only {k : Kern} {m : Mem} {nr a1 a2 a3 a4 a5 a6 c : Nat}
    (h : X86Env.syscall k m nr a1 a2 a3 a4 a5 a6 = some (.exit c)) :
    c = a1 % 256 := by
  by_cases h60 : nr = 60
  · subst h60; simp [X86Env.syscall, pure] at h; exact h.symm
  · exfalso
    unfold X86Env.syscall at h
    simp only [bind, Option.bind, pure] at h
    split at h
    · split at h
      · cases hm : m.wrN a2 (List.take (k.count a3 k.stdin.length).1 k.stdin) <;> simp [hm] at h
      · split at h
        · cases hk : k.fds[a1 - 3]? with
          | none => simp [hk] at h
          | some o =>
            cases o with
            | none => simp [hk] at h
            | some fd =>
              cases hm : m.wrN a2 (List.take (k.count a3 (fd.data.length - fd.pos)).1 (List.drop fd.pos fd.data)) <;>
                simp [hk, hm] at h
        · cases h
    · split at h
      · cases hm : m.rdN a2 (k.count a3 a3).1 <;> simp [hm] at h
      · cases h
    · split at h
      · cases h
      · cases hc : m.cstr a1 4096 with
        | none => simp [hc] at h
        | some nm => cases hl : X86Env.lookup k.files nm <;> simp [hc, hl] at h
    · split at h <;> cases h
    · split at h
      · cases h
      · split at h
        · cases h
        · cases hp : X86Env.place m k.next.1 (X86Env.pages a2) with
          | none => simp [hp] at h
          | some opt => cases opt <;> simp [hp] at h
    · cases hf : m.find a1 0 with
      | none => simp [hf] at h
      | some ir =>
        simp only [hf] at h
        split at h <;> cases h
    · split at h
      · cases h
      · cases hf : m.find a1 0 with
        | none => simp [hf] at h
        | some ir =>
          simp only [hf] at h
          split at h
          · cases h
          · split at h
            · cases h
            · cases hp : X86Env.place (List.eraseIdx m ir.1) k.next.1 (X86Env.pages a3) with
              | none => simp [hp] at h
              | some opt => cases opt <;> simp [hp] at h
    · exact h60 rfl
    · cases h

private theorem exec_syscall_safe {base off : Nat} {s : St}
    (hsrip : s.rip = base + off) (hlen : s.regs.length = 16)
    (hexit : hasExitNonZero s)
    (hnext : ∀ s', s'.regs.length = 16 → s'.rip = base + (off + 2) →
      s'.reg 3 = s.reg 3 → TailSafe base s') :
    match X86Int.exec .syscall 2 s with
    | .ok s' => TailSafe base s'
    | .exit c => c ≠ 0
    | .stuck => True := by
  dsimp only [X86Int.exec]
  cases h0 : s.reg 0 with
  | none => trivial
  | some nr =>
    cases h7 : s.reg 7 with
    | none => trivial
    | some a1 =>
      cases h6 : s.reg 6 with
      | none => trivial
      | some a2 =>
        cases h2 : s.reg 2 with
        | none => trivial
        | some a3 =>
          cases h10 : s.reg 10 with
          | none => trivial
          | some a4 =>
            cases h8 : s.reg 8 with
            | none => trivial
            | some a5 =>
              cases h9 : s.reg 9 with
              | none => trivial
              | some a6 =>
                dsimp only
                cases hsys : X86Env.syscall s.k s.mem nr a1 a2 a3 a4 a5 a6 with
                | none => trivial
                | some r =>
                  cases r with
                  | exit c =>
                    have hc := syscall_exit_only hsys
                    rw [hc]
                    exact hexit a1 h7
                  | ret rax k' m' =>
                    refine hnext _ ?_ (by simp [hsrip]; omega) ?_
                    · dsimp only; simp only [St.set, List.length_set, hlen]
                    · change (((s.set 0 (some rax)).set 1 (some (s.rip + 2))).set 11 none).reg 3 = s.reg 3
                      rw [reg_set_ne (by decide), reg_set_ne (by decide), reg_set_ne (by decide)]

theorem TailSafe_step {base : Nat} (hb : base_ok base) {s : St} (hs : TailSafe base s) :
    match X86Int.step (tailImAt base) s with
    | .ok s' => TailSafe base s'
    | .exit c => c ≠ 0
    | .stuck => True := by
  rcases hs with ⟨hlen, hsafe⟩
  by_cases hlo : s.rip < base
  · rw [X86Int.step, tailImAt_lt hlo]; exact trivial
  · have hge_base : base ≤ s.rip := Nat.le_of_not_lt hlo
    by_cases hhi : base + 143 ≤ s.rip
    · rw [X86Int.step, tailImAt_ge hhi]; exact trivial
    · have hlt_top : s.rip < base + 143 := Nat.lt_of_not_le hhi
      have hspec := hsafe hge_base hlt_top
      have hsrip : s.rip = base + (s.rip - base) := (Nat.add_sub_cancel' hge_base).symm
      generalize hs_off : s.rip - base = off at hspec hsrip
      have hoff : off < 143 := by omega
      clear hs_off
      have hcov := List.all_eq_true.1 tailAllOffs_cover off (List.mem_range.2 hoff)
      simp only [Bool.or_eq_true, List.any_eq_true] at hcov
      rcases hcov with hnone | ⟨e, hmem, heq⟩
      · have h0none : tailImAt 0 off = none := Option.isNone_iff_eq_none.1 hnone
        have hnoneb : tailImAt base s.rip = none := by
          rw [hsrip, tailImAt_eq, h0none]; rfl
        rw [X86Int.step, hnoneb]; exact trivial
      · have hoffeq : e.1 = off := beq_iff_eq.1 heq
        rw [← hoffeq] at hspec hsrip
        have hi := tailImAt_base_all (base := base) hmem
        simp only [tailAllSpec, tailSpec, tailExtraSpec, List.mem_append, List.mem_cons,
          List.not_mem_nil, or_false] at hmem
        rcases hmem with
          (rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl |
           rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl |
           rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl) |
          (rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl |
           rfl | rfl | rfl | rfl | rfl)
        all_goals
          rw [X86Int.step, hsrip, hi]
          dsimp only [shiftIns, X86Int.exec, lift, bind, Option.bind, pure]
        · -- 0x00: cmp 0 3 (3) -> 0x03
          rcases hspec with ⟨⟨w0, hw0, _⟩, ⟨w3, hw3, _⟩, ⟨w1, hw1, _⟩⟩
          simp only [hw0, hw3, TailSafe, hlen, true_and]
          intro _ _; rw [show s.rip + 3 - base = 3 by omega]
          exact ⟨⟨w0, hw0, ‹_›⟩, ⟨w3, hw3, ‹_›⟩, ⟨w1, hw1, ‹_›⟩⟩
        · -- 0x03: jcc 4 0x12 (2) -> 0x12 or 0x05
          rcases hspec with ⟨h0, h3, h1⟩
          cases hfl : X86Int.cond s.fl 4 with
          | none => simp
          | some c =>
            simp only [shift_tgt_lt hb (by decide : 0x12 < 143)]
            cases c <;> simp only [Bool.false_eq_true, ↓reduceIte, TailSafe, hlen, true_and]
            · intro _ _; rw [show s.rip + 2 - base = 5 by omega]; exact ⟨h0, h3, h1⟩
            · intro _ _; rw [show base + 18 - base = 18 by omega]; exact h0
        · -- 0x05: cmp 0 1 (3) -> 0x08
          rcases hspec with ⟨⟨w0, hw0, _⟩, ⟨w3, hw3, _⟩, ⟨w1, hw1, _⟩⟩
          simp only [hw0, hw1, TailSafe, hlen, true_and]
          intro _ _; rw [show s.rip + 3 - base = 8 by omega]
          exact ⟨⟨w0, hw0, ‹_›⟩, ⟨w3, hw3, ‹_›⟩, ⟨w1, hw1, ‹_›⟩⟩
        · -- 0x08: jcc 4 0x12 (2) -> 0x12 or 0x0a
          rcases hspec with ⟨h0, h3, h1⟩
          cases hfl : X86Int.cond s.fl 4 with
          | none => simp
          | some c =>
            simp only [shift_tgt_lt hb (by decide : 0x12 < 143)]
            cases c <;> simp only [Bool.false_eq_true, ↓reduceIte, TailSafe, hlen, true_and]
            · intro _ _; rw [show s.rip + 2 - base = 10 by omega]; exact ⟨h0, h3⟩
            · intro _ _; rw [show base + 18 - base = 18 by omega]; exact h0
        · -- 0x0a: cmp 3 1 (3) -> 0x0d
          rcases hspec with ⟨⟨w0, hw0, _⟩, ⟨w3, hw3, _⟩⟩
          cases h1 : s.reg 1 with
          | none => simp [hw3]
          | some w1 =>
            simp only [hw3, TailSafe, hlen, true_and]
            intro _ _; rw [show s.rip + 3 - base = 13 by omega]
            exact ⟨⟨w0, hw0, ‹_›⟩, ⟨w3, hw3, ‹_›⟩⟩
        · -- 0x0d: jcc 5 0x1a (2) -> 0x1a or 0x0f
          rcases hspec with ⟨h0, h3⟩
          cases hfl : X86Int.cond s.fl 5 with
          | none => simp
          | some c =>
            simp only [shift_tgt_lt hb (by decide : 0x1a < 143)]
            cases c <;> simp only [Bool.false_eq_true, ↓reduceIte, TailSafe, hlen, true_and]
            · intro _ _; rw [show s.rip + 2 - base = 15 by omega]; exact h3
            · intro _ _; rw [show base + 26 - base = 26 by omega]; exact h3
        · -- 0x0f: mov 0 3 (3) -> 0x12
          rcases hspec with ⟨w3, hw3, hv3⟩
          simp only [hw3, TailSafe, set_length hlen, true_and]
          intro _ _; rw [show s.rip + 3 - base = 18 by omega]
          exact ⟨w3, by rw [reg_set_mk hlen (by decide)], hv3⟩
        · -- 0x12: mov 3 0 (3) -> 0x15
          rcases hspec with ⟨w0, hw0, hv0⟩
          simp only [hw0, TailSafe, set_length hlen, true_and]
          intro _ _; rw [show s.rip + 3 - base = 21 by omega]
          exact Or.inl ⟨w0, by rw [reg_set_mk_ne (by decide), hw0], hv0⟩
        · -- 0x15: mov 1 0 (3) -> 0x18
          cases h0 : s.reg 0 with
          | none => simp
          | some w0 =>
            simp only [TailSafe, set_length hlen, true_and]
            intro _ _; rw [show s.rip + 3 - base = 24 by omega]
            rcases hspec with ⟨w, hw, hv⟩ | ⟨w, hw, hv⟩
            · exact Or.inl ⟨w, by rw [reg_set_mk_ne (by decide), hw], hv⟩
            · exact Or.inr ⟨w, by rw [reg_set_mk_ne (by decide), hw], hv⟩
        · -- 0x18: jmp 0x26 (2) -> 0x26
          simp only [shift_tgt_lt hb (by decide : 0x26 < 143), TailSafe, hlen, true_and]
          intro _ _; rw [show base + 38 - base = 38 by omega]; exact hspec
        · -- 0x1a: movI 7 3 (5) -> 0x1f
          simp only [TailSafe, set_length hlen, true_and]
          intro _ _; rw [show s.rip + 5 - base = 31 by omega]
          refine ⟨fun a1 ha1 => ?_, ?_⟩
          · rw [reg_set_mk hlen (by decide)] at ha1; cases ha1; decide
          · rcases hspec with ⟨w3, hw3, hv3⟩; exact ⟨w3, by rw [reg_set_mk_ne (by decide), hw3], hv3⟩
        · -- 0x1f: movI 0 60 (5) -> 0x24
          simp only [TailSafe, set_length hlen, true_and]
          intro _ _; rw [show s.rip + 5 - base = 36 by omega]
          refine ⟨fun a1 ha1 => ?_, ?_⟩
          · rw [reg_set_mk_ne (by decide)] at ha1; exact hspec.1 a1 ha1
          · rcases hspec.2 with ⟨w3, hw3, hv3⟩; exact ⟨w3, by rw [reg_set_mk_ne (by decide), hw3], hv3⟩
        · -- 0x24: syscall (2) -> 0x26
          exact exec_syscall_safe hsrip hlen hspec.1 fun s' hlen' hrip' h3' =>
            ⟨hlen', fun _ _ => by
              rw [show s'.rip - base = 38 by omega]
              rcases hspec.2 with ⟨w3, hw3, hv3⟩
              exact Or.inr ⟨w3, by rw [h3', hw3], hv3⟩⟩
        · -- 0x26: movI 7 5 (5) -> 0x2b
          simp only [TailSafe, set_length hlen, true_and]
          intro _ _; rw [show s.rip + 5 - base = 43 by omega]
          rcases hspec with ⟨w, hw, hv⟩ | ⟨w, hw, hv⟩
          · exact Or.inl ⟨w, by rw [reg_set_mk_ne (by decide), hw], hv⟩
          · exact Or.inr ⟨w, by rw [reg_set_mk_ne (by decide), hw], hv⟩
        · -- 0x2b: movI 2 -ACCEPT (10) -> 0x35
          simp only [TailSafe, set_length hlen, true_and]
          intro _ _; rw [show s.rip + 10 - base = 53 by omega]
          rcases hspec with ⟨w, hw, hv⟩ | ⟨w, hw, hv⟩
          · refine Or.inl ⟨?_, ⟨w, by rw [reg_set_mk_ne (by decide), hw], hv⟩⟩
            rw [reg_set_mk hlen (by decide)]; decide
          · exact Or.inr ⟨w, by rw [reg_set_mk_ne (by decide), hw], hv⟩
        · -- 0x35: alu .add 2 0 (3) -> 0x38
          rcases hspec with ⟨hw2eq, ⟨w0, hw0, hv0⟩⟩ | ⟨w3, hw3, hv3⟩
          · simp only [hw2eq, hw0, TailSafe, set_length hlen, true_and]
            intro _ _; rw [show s.rip + 3 - base = 56 by omega]
            left
            have hacc0 : 0 < ACCEPT := by decide
            have haccW : ACCEPT < X86Env.W := by decide
            have hzf : (X86Int.aluF .add ((X86Env.W - ACCEPT) % X86Env.W) w0).2.zf = some false :=
              (aluF_add_compl_zf hacc0 haccW hv0.1).2.mpr hv0.2
            rw [hzf]; decide
          · cases h2 : s.reg 2 with
            | none => simp
            | some w2 =>
              cases h0 : s.reg 0 with
              | none => simp
              | some w0 =>
                simp only [TailSafe, set_length hlen, true_and]
                intro _ _; rw [show s.rip + 3 - base = 56 by omega]
                right; exact ⟨w3, by rw [reg_set_mk_ne (by decide), hw3], hv3⟩
        · -- 0x38: jcc 5 0x59 (2) -> 0x59 or 0x3a
          cases hfl : X86Int.cond s.fl 5 with
          | none => simp
          | some c =>
            simp only [shift_tgt_lt hb (by decide : 0x59 < 143)]
            cases c <;> simp only [Bool.false_eq_true, ↓reduceIte, TailSafe, hlen, true_and]
            · intro _ _; rw [show s.rip + 2 - base = 58 by omega]
              rcases hspec with hzf | h3
              · exfalso; exact cond5_false_of_zf_ne hzf hfl
              · exact h3
            · intro _ _; rw [show base + 89 - base = 89 by omega]; trivial
        · -- 0x3a: movI 6 -ACCEPT (10) -> 0x44
          simp only [TailSafe, set_length hlen, true_and]
          intro _ _; rw [show s.rip + 10 - base = 68 by omega]
          rcases hspec with ⟨w3, hw3, hv3⟩
          refine ⟨?_, ⟨w3, by rw [reg_set_mk_ne (by decide), hw3], hv3⟩⟩
          rw [reg_set_mk hlen (by decide)]; decide
        · -- 0x44: alu .add 6 3 (3) -> 0x47
          rcases hspec with ⟨hw6eq, ⟨w3, hw3, hv3⟩⟩
          simp only [hw6eq, hw3, TailSafe, set_length hlen, true_and]
          intro _ _; rw [show s.rip + 3 - base = 71 by omega]
          have hacc0 : 0 < ACCEPT := by decide
          have haccW : ACCEPT < X86Env.W := by decide
          have hzf : (X86Int.aluF .add ((X86Env.W - ACCEPT) % X86Env.W) w3).2.zf = some false :=
            (aluF_add_compl_zf hacc0 haccW hv3.1).2.mpr hv3.2
          simp only [TailSafeOff]
          rw [hzf]; decide
        · -- 0x47: jcc 5 0x59 (2) -> 0x59 or 0x49
          cases hfl : X86Int.cond s.fl 5 with
          | none => simp
          | some c =>
            simp only [shift_tgt_lt hb (by decide : 0x59 < 143)]
            cases c <;> simp only [Bool.false_eq_true, ↓reduceIte, TailSafe, hlen, true_and]
            · intro _ _; exfalso; exact cond5_false_of_zf_ne hspec hfl
            · intro _ _; rw [show base + 89 - base = 89 by omega]; trivial
        · -- 0x49: movI 0 60 (5) -> 0x4e
          simp only [TailSafe, set_length hlen, true_and]
          intro _ _; rw [show s.rip + 5 - base = 78 by omega]
          rcases hspec with ⟨w3, hw3, hv3⟩
          exact ⟨w3, by rw [reg_set_mk_ne (by decide), hw3], hv3⟩
        · -- 0x4e: lea 7 3 (X86Env.W - 0x96) (7) -> 0x55
          rcases hspec with ⟨w3, hw3, hv3⟩
          simp only [hw3, TailSafe, set_length hlen, true_and]
          intro _ _; rw [show s.rip + 7 - base = 85 by omega]
          intro a1 ha1
          rw [reg_set_mk hlen (by decide)] at ha1
          injection ha1 with ha1
          rw [← ha1]
          exact lea_exit_nz hv3
        · -- 0x55: syscall (2) -> 0x57
          exact exec_syscall_safe hsrip hlen hspec fun s' hlen' hrip' _ =>
            ⟨hlen', fun _ _ => by rw [show s'.rip - base = 87 by omega]; trivial⟩
        · -- 0x57: jmp 0x7e (2)
          simp only [shift_tgt_lt hb (by decide : 0x7e < 143), TailSafe, hlen, true_and]
          intro _ _; rw [show base + 126 - base = 126 by omega]; trivial
        · -- 0x59: movI 7 5 (5) -> 0x5e
          simp only [TailSafe, set_length hlen, true_and]
          intro _ _; rw [show s.rip + 5 - base = 94 by omega]; trivial
        · -- 0x5e: movI 2 -OVERFLOW (10) -> 0x68
          simp only [TailSafe, set_length hlen, true_and]
          intro _ _; rw [show s.rip + 10 - base = 104 by omega]; trivial
        · -- 0x68: alu .add 2 0 (3) -> 0x6b
          cases h2 : s.reg 2 with
          | none => simp
          | some v2 =>
            cases h0 : s.reg 0 with
            | none => simp
            | some v0 =>
              simp only [TailSafe, set_length hlen, true_and]
              intro _ _; rw [show s.rip + 3 - base = 107 by omega]; trivial
        · -- 0x6b: jcc 5 0x7e (2)
          cases hfl : X86Int.cond s.fl 5 with
          | none => simp
          | some c =>
            simp only [shift_tgt_lt hb (by decide : 0x7e < 143)]
            cases c <;> simp only [Bool.false_eq_true, ↓reduceIte, TailSafe, hlen, true_and]
            · intro _ _; rw [show s.rip + 2 - base = 109 by omega]; trivial
            · intro _ _; rw [show base + 126 - base = 126 by omega]; trivial
        · -- 0x6d: movI 7 6 (5) -> 0x72
          simp only [TailSafe, set_length hlen, true_and]
          intro _ _; rw [show s.rip + 5 - base = 114 by omega]; trivial
        · -- 0x72: movI 0 60 (5) -> 0x77
          simp only [TailSafe, set_length hlen, true_and]
          intro _ _; rw [show s.rip + 5 - base = 119 by omega]; trivial
        · -- 0x77: movI 7 6 (5) -> 0x7c
          simp only [TailSafe, set_length hlen, true_and]
          intro _ _; rw [show s.rip + 5 - base = 124 by omega]
          intro a1 ha1; rw [reg_set_mk hlen (by decide)] at ha1; injection ha1 with ha1; rw [← ha1]; decide
        · -- 0x7c: syscall (2) -> 0x7e
          exact exec_syscall_safe hsrip hlen hspec fun s' hlen' hrip' _ =>
            ⟨hlen', fun _ _ => by rw [show s'.rip - base = 126 by omega]; trivial⟩
        · -- 0x7e: movI 7 5 (5) -> 0x83
          simp only [TailSafe, set_length hlen, true_and]
          intro _ _; rw [show s.rip + 5 - base = 131 by omega]; trivial
        · -- 0x83: movI 0 60 (5) -> 0x88
          simp only [TailSafe, set_length hlen, true_and]
          intro _ _; rw [show s.rip + 5 - base = 136 by omega]; trivial
        · -- 0x88: movI 7 5 (5) -> 0x8d
          simp only [TailSafe, set_length hlen, true_and]
          intro _ _; rw [show s.rip + 5 - base = 141 by omega]
          intro a1 ha1; rw [reg_set_mk hlen (by decide)] at ha1; injection ha1 with ha1; rw [← ha1]; decide
        · -- 0x8d: syscall (2) -> 0x8f (≥ 143)
          exact exec_syscall_safe hsrip hlen hspec fun s' hlen' hrip' _ =>
            ⟨hlen', fun _ hlt => by exfalso; omega⟩
        · -- 0x10: mov32 0 3 (2) -> 0x12
          cases h3 : s.reg 3 with
          | none => simp
          | some w3 =>
            simp only [TailSafe, set_length hlen, true_and]
            intro _ _; rw [show s.rip + 2 - base = 18 by omega]
            exact ⟨w3 % 2 ^ 32, by rw [reg_set_mk hlen (by decide)], valNotAcc_mod32 w3⟩
        · -- 0x13: mov32 3 0 (2) -> 0x15
          cases h0 : s.reg 0 with
          | none => simp
          | some w0 =>
            simp only [TailSafe, set_length hlen, true_and]
            intro _ _; rw [show s.rip + 2 - base = 21 by omega]
            exact Or.inr ⟨w0 % 2 ^ 32, by rw [reg_set_mk hlen (by decide)], valNotAcc_mod32 w0⟩
        · -- 0x14: ret (1)
          cases h4 : s.reg 4 with
          | none => simp
          | some sp =>
            cases hrd : X86Int.rd64 s.mem sp with
            | none => simp [hrd]
            | some t =>
              simp only [hrd, TailSafe, set_length hlen, true_and]
              intro h1 h2; exfalso; exact hspec sp h4 t hrd ⟨h1, h2⟩
        · -- 0x16: mov32 1 0 (2) -> 0x18
          cases h0 : s.reg 0 with
          | none => simp
          | some w0 =>
            simp only [TailSafe, set_length hlen, true_and]
            intro _ _; rw [show s.rip + 2 - base = 24 by omega]
            rcases hspec with ⟨w, hw, hv⟩ | ⟨w, hw, hv⟩
            · exact Or.inl ⟨w, by rw [reg_set_mk_ne (by decide), hw], hv⟩
            · exact Or.inr ⟨w, by rw [reg_set_mk_ne (by decide), hw], hv⟩
        · -- 0x2c: movI 2 (5) -> 0x31 (none)
          simp only [TailSafe, set_length hlen, true_and]
          intro _ _; rw [show s.rip + 5 - base = 49 by omega]; trivial
        · -- 0x2d: pushI (2) -> 0x2f (none)
          cases h4 : s.reg 4 with
          | none => simp
          | some sp =>
            dsimp only
            cases X86Int.wr64 s.mem ((sp + X86Env.W - 8) % X86Env.W) ((X86Env.W - 61) % X86Env.W) with
            | none => exact trivial
            | some m' =>
              simp only [TailSafe, St.set, List.length_set, hlen, true_and]
              intro _ _; rw [show s.rip + 2 - base = 47 by omega]; trivial
        · -- 0x2e: ret (1)
          cases h4 : s.reg 4 with
          | none => simp
          | some sp =>
            cases hrd : X86Int.rd64 s.mem sp with
            | none => simp [hrd]
            | some t =>
              simp only [hrd, TailSafe, set_length hlen, true_and]
              intro h1 h2; exfalso; exact hspec sp h4 t hrd ⟨h1, h2⟩
        · -- 0x30: pop 2 (1) -> 0x31 (none)
          cases h4 : s.reg 4 with
          | none => simp
          | some sp =>
            cases hrd : X86Int.rd64 s.mem sp with
            | none => simp [hrd]
            | some _ =>
              simp only [hrd, TailSafe, St.set, List.length_set, hlen, true_and]
              intro _ _; rw [show s.rip + 1 - base = 49 by omega]; trivial
        · -- 0x3b: movI 6 (5) -> 0x40 (none)
          simp only [TailSafe, set_length hlen, true_and]
          intro _ _; rw [show s.rip + 5 - base = 64 by omega]; trivial
        · -- 0x3c: pushI (2) -> 0x3e (none)
          cases h4 : s.reg 4 with
          | none => simp
          | some sp =>
            dsimp only
            cases X86Int.wr64 s.mem ((sp + X86Env.W - 8) % X86Env.W) ((X86Env.W - 61) % X86Env.W) with
            | none => exact trivial
            | some m' =>
              simp only [TailSafe, St.set, List.length_set, hlen, true_and]
              intro _ _; rw [show s.rip + 2 - base = 62 by omega]; trivial
        · -- 0x3d: ret (1)
          cases h4 : s.reg 4 with
          | none => simp
          | some sp =>
            cases hrd : X86Int.rd64 s.mem sp with
            | none => simp [hrd]
            | some t =>
              simp only [hrd, TailSafe, set_length hlen, true_and]
              intro h1 h2; exfalso; exact hspec sp h4 t hrd ⟨h1, h2⟩
        · -- 0x3f: pop 2 (1) -> 0x40 (none)
          cases h4 : s.reg 4 with
          | none => simp
          | some sp =>
            cases hrd : X86Int.rd64 s.mem sp with
            | none => simp [hrd]
            | some _ =>
              simp only [hrd, TailSafe, St.set, List.length_set, hlen, true_and]
              intro _ _; rw [show s.rip + 1 - base = 64 by omega]; trivial
        · -- 0x50: movI 3 (5) -> 0x55
          simp only [TailSafe, set_length hlen, true_and]
          intro _ _; rw [show s.rip + 5 - base = 85 by omega]
          intro a1 ha1; rw [reg_set_mk_ne (by decide)] at ha1; exact hspec a1 ha1
        · -- 0x51: pushI (2) -> 0x53 (none)
          cases h4 : s.reg 4 with
          | none => simp
          | some sp =>
            dsimp only
            cases X86Int.wr64 s.mem ((sp + X86Env.W - 8) % X86Env.W) ((X86Env.W - 1) % X86Env.W) with
            | none => exact trivial
            | some m' =>
              simp only [TailSafe, St.set, List.length_set, hlen, true_and]
              intro _ _; rw [show s.rip + 2 - base = 83 by omega]; trivial
        · -- 0x5f: movI 2 (5) -> 0x64 (none)
          simp only [TailSafe, set_length hlen, true_and]
          intro _ _; rw [show s.rip + 5 - base = 100 by omega]; trivial
        · -- 0x60: pop 3 (1) -> 0x61 (none)
          cases h4 : s.reg 4 with
          | none => simp
          | some sp =>
            cases hrd : X86Int.rd64 s.mem sp with
            | none => simp [hrd]
            | some _ =>
              simp only [hrd, TailSafe, St.set, List.length_set, hlen, true_and]
              intro _ _; rw [show s.rip + 1 - base = 97 by omega]; trivial
        · -- 0x66: push 2 (1) -> 0x67 (none)
          cases h2 : s.reg 2 with
          | none => simp
          | some v2 =>
            cases h4 : s.reg 4 with
            | none => simp
            | some sp =>
              dsimp only
              cases X86Int.wr64 s.mem ((sp + X86Env.W - 8) % X86Env.W) v2 with
              | none => exact trivial
              | some m' =>
                simp only [TailSafe, St.set, List.length_set, hlen, true_and]
                intro _ _; rw [show s.rip + 1 - base = 103 by omega]; trivial

theorem TailSafe_run {base : Nat} (hb : base_ok base) :
    ∀ (m : Nat) {s : St}, TailSafe base s → X86Int.run (tailImAt base) m (.ok s) ≠ .exit 0
  | 0, _, _ => by simp [X86Int.run]
  | m + 1, s, hs => by
    have hstep := TailSafe_step hb hs
    simp only [X86Int.run]
    cases hs' : X86Int.step (tailImAt base) s with
    | stuck => rw [run_stuck]; intro h; cases h
    | exit c =>
      rw [hs'] at hstep
      rw [run_exit]
      intro h; cases h; exact hstep rfl
    | ok s' =>
      rw [hs'] at hstep
      exact TailSafe_run hb m hstep

/-! ### Part 3: Pre-fault trace invariant and single-fault immunity -/

private theorem tailOff_trivial_of_not_mem {base : Nat} {s : St} {t : Nat}
    (h : t ∉ nonTrivialTailOffs) :
    TailSafeOff base s t := by
  dsimp only [TailSafeOff]
  split <;> first | exact trivial | (exfalso; apply h; simp [nonTrivialTailOffs])

private theorem tail_isNone_trivial_spec :
    (List.range 143).all (fun t =>
      !(tailImAt 0 t).isNone ||
      t ∈ [0x14, 0x16, 0x2e, 0x3d, 0x50] ||
      decide (t ∉ nonTrivialTailOffs)) = true := by
  decide +kernel

private theorem tail_isNone_TailSafeOff {base : Nat} {s : St} {t : Nat}
    (ht : t < 143) (hnone : (tailImAt 0 t).isNone = true)
    (hval3 : hasValNotAcc s 3) (hst : TailStackSafe base s)
    (h50 : t = 0x50 → hasExitNonZero s) :
    TailSafeOff base s t := by
  have hmem := List.all_eq_true.1 tail_isNone_trivial_spec t (List.mem_range.2 ht)
  simp only [Bool.or_eq_true, Bool.not_eq_true', hnone, Bool.true_eq_false, false_or,
    decide_eq_true_eq, List.mem_cons, List.not_mem_nil, or_false] at hmem
  rcases hmem with (rfl | rfl | rfl | rfl | rfl) | hnot
  · exact hst
  · exact Or.inr hval3
  · exact hst
  · exact hst
  · exact h50 rfl
  · exact tailOff_trivial_of_not_mem hnot

private theorem tail_isNone_TailSafeOff_guarded {base : Nat} {s : St} {t : Nat}
    (ht : t < 143) (hnone : (tailImAt 0 t).isNone = true)
    (h16 : t = 0x16 → hasValNotAcc s 3)
    (hret : t ∈ [0x14, 0x2e, 0x3d] → TailStackSafe base s)
    (h50 : t = 0x50 → hasExitNonZero s) :
    TailSafeOff base s t := by
  have hmem := List.all_eq_true.1 tail_isNone_trivial_spec t (List.mem_range.2 ht)
  simp only [Bool.or_eq_true, Bool.not_eq_true', hnone, Bool.true_eq_false, false_or,
    decide_eq_true_eq, List.mem_cons, List.not_mem_nil, or_false] at hmem
  rcases hmem with (rfl | rfl | rfl | rfl | rfl) | hnot
  · exact hret (by simp)
  · exact Or.inr (h16 rfl)
  · exact hret (by simp)
  · exact hret (by simp)
  · exact h50 rfl
  · exact tailOff_trivial_of_not_mem hnot

private theorem tail_isNone_generic {base : Nat} {s : St} {t : Nat}
    (ht : t < 143) (hnone : (tailImAt 0 t).isNone = true)
    (hnot5 : t ∉ [0x14, 0x16, 0x2e, 0x3d, 0x50]) :
    TailSafeOff base s t := by
  have hmem := List.all_eq_true.1 tail_isNone_trivial_spec t (List.mem_range.2 ht)
  simp only [Bool.or_eq_true, Bool.not_eq_true', hnone, Bool.true_eq_false, false_or,
    decide_eq_true_eq] at hmem
  rcases hmem with h5 | hnot
  · exact absurd h5 hnot5
  · exact tailOff_trivial_of_not_mem hnot

private theorem nxSafeGeneric_TailSafe {base : Nat} {s : St} {t : Nat}
    (hlen : s.regs.length = 16) (hsrip : s.rip = base + t)
    (hnx : nxSafeGeneric t = true) :
    TailSafe base s := by
  refine ⟨hlen, fun _ hlt => ?_⟩
  rw [hsrip, Nat.add_sub_cancel_left]
  simp only [nxSafeGeneric, Bool.or_eq_true, Bool.and_eq_true, decide_eq_true_eq] at hnx
  rcases hnx with hge | ⟨hnone, hnot⟩
  · exfalso; omega
  · exact tail_isNone_generic (by omega) hnone hnot

private theorem reg_flip_at {s : St} (hlen : s.regs.length = 16) {r : Nat} (hr : r < 16) (b : Nat) :
    (s.flipReg r b).reg r = (s.reg r).map (fun v => Nat.xor v (2 ^ b)) :=
  reg_set_eq hlen hr _

private theorem reg_flip_ne {s : St} {r i : Nat} (hne : r ≠ i) (b : Nat) :
    (s.flipReg r b).reg i = s.reg i :=
  reg_set_ne hne _

private theorem reg_flip_length {s : St} (hlen : s.regs.length = 16) (r b : Nat) :
    (s.flipReg r b).regs.length = 16 :=
  set_length hlen r _

private theorem xor5_mod256_nz :
    (List.range 64).all (fun b => decide (Nat.xor 5 (2 ^ b) % 256 ≠ 0)) = true := by
  decide +kernel

private theorem xor6_mod256_nz :
    (List.range 64).all (fun b => decide (Nat.xor 6 (2 ^ b) % 256 ≠ 0)) = true := by
  decide +kernel

private theorem valNotAcc_60 : valNotAcc 60 := ⟨by decide, by decide⟩
private theorem valNotAcc_0 : valNotAcc 0 := ⟨by decide, by decide⟩

private theorem SafeInvalid_xor {v b : Nat} (hv : SafeInvalid v) (hb : b < 64) :
    valNotAcc (Nat.xor v (2 ^ b)) :=
  ⟨(hv.2.2.2.2.2.2 b hb).1, (hv.2.2.2.2.2.2 b hb).2.1⟩

private theorem SafeInvalid_double {v : Nat} (hv : SafeInvalid v) :
    valNotAcc ((v + v) % X86Env.W) := by
  rw [show v + v = 2 * v by omega]
  exact ⟨Nat.mod_lt _ (by decide), hv.2.2.2.2.1⟩

private theorem syscall60_no_ret {k : Kern} {m : Mem} {a1 a2 a3 a4 a5 a6 rax : Nat} {k' : Kern} {m' : Mem}
    (h : X86Env.syscall k m 60 a1 a2 a3 a4 a5 a6 = some (.ret rax k' m')) : False := by
  simp [X86Env.syscall, pure] at h

def PreFaultOff (v : Nat) (s : St) (off : Nat) : Prop :=
  match off with
  | 0x00 | 0x03 | 0x12 | 0x15 | 0x18 | 0x26 | 0x2b | 0x35 | 0x38 | 0x59 | 0x5e | 0x68 | 0x6b =>
      s.reg 0 = some v ∧ s.reg 1 = some v ∧ s.reg 3 = some v ∧
      (off = 0x03 → s.fl.zf = some true) ∧
      (off = 0x35 → s.reg 2 = some ((X86Env.W - ACCEPT) % X86Env.W)) ∧
      (off = 0x38 → s.fl.zf = some false) ∧
      (0x2b ≤ off → s.reg 7 = some 5)
  | 0x6d | 0x7e | 0x83 =>
      s.reg 7 = some 5 ∧ s.reg 0 = some v ∧ s.reg 1 = some v ∧ s.reg 3 = some v
  | 0x72 =>
      s.reg 7 = some 6 ∧ s.reg 0 = some v ∧ s.reg 1 = some v ∧ s.reg 3 = some v
  | 0x77 | 0x7c =>
      s.reg 7 = some 6 ∧ s.reg 0 = some 60 ∧ s.reg 1 = some v ∧ s.reg 3 = some v
  | 0x88 | 0x8d =>
      s.reg 7 = some 5 ∧ s.reg 0 = some 60 ∧ s.reg 1 = some v ∧ s.reg 3 = some v
  | _ => False

def PreFault (base v : Nat) (s : St) : Prop :=
  SafeInvalid v ∧
  s.regs.length = 16 ∧
  TailStackSafe base s ∧
  base ≤ s.rip ∧ s.rip < base + 143 ∧
  PreFaultOff v s (s.rip - base)

private theorem preFaultOff_mem {v : Nat} {s : St} {off : Nat} (h : PreFaultOff v s off) :
    off ∈ preFaultOffs := by
  dsimp only [PreFaultOff] at h
  split at h
  · simp [preFaultOffs]
  · simp [preFaultOffs]
  · simp [preFaultOffs]
  · simp [preFaultOffs]
  · simp [preFaultOffs]
  · simp [preFaultOffs]
  · simp [preFaultOffs]
  · simp [preFaultOffs]
  · simp [preFaultOffs]
  · simp [preFaultOffs]
  · simp [preFaultOffs]
  · simp [preFaultOffs]
  · simp [preFaultOffs]
  · simp [preFaultOffs]
  · simp [preFaultOffs]
  · simp [preFaultOffs]
  · simp [preFaultOffs]
  · simp [preFaultOffs]
  · simp [preFaultOffs]
  · simp [preFaultOffs]
  · simp [preFaultOffs]
  · exact False.elim h

private theorem PreFault_init {base : Nat} (_hb : base_ok base) (s0 : St)
    (hst : TailStackSafe base s0) {v : Nat} (hv : SafeInvalid v) :
    PreFault base v (tailInit base s0 v v v) := by
  refine ⟨hv, rfl, ?_, Nat.le_refl _, ?_, ?_⟩
  · intro sp hsp t hrd; exact hst sp hsp t hrd
  · simp [tailInit]
  · rw [show (tailInit base s0 v v v).rip - base = 0 by simp [tailInit]]
    refine ⟨rfl, rfl, rfl, fun h => by omega, fun h => by omega, fun h => by omega, fun h => by omega⟩

private theorem PreFault_regs0137 {base v : Nat} {s : St} (h : PreFault base v s) :
    (s.reg 0 = some v ∨ s.reg 0 = some 60) ∧
    s.reg 1 = some v ∧
    s.reg 3 = some v ∧
    (0x2b ≤ s.rip - base → s.reg 7 = some 5 ∨ s.reg 7 = some 6) := by
  rcases h with ⟨hv, _, _, _, _, hpo⟩
  dsimp only [PreFaultOff] at hpo
  split at hpo
  all_goals first
    | exact ⟨Or.inl hpo.1, hpo.2.1, hpo.2.2.1, fun hle => Or.inl (hpo.2.2.2.2.2.2 hle)⟩
    | exact ⟨Or.inl hpo.2.1, hpo.2.2.1, hpo.2.2.2, fun _ => Or.inl hpo.1⟩
    | exact ⟨Or.inl hpo.2.1, hpo.2.2.1, hpo.2.2.2, fun _ => Or.inr hpo.1⟩
    | exact ⟨Or.inr hpo.2.1, hpo.2.2.1, hpo.2.2.2, fun _ => Or.inr hpo.1⟩
    | exact ⟨Or.inr hpo.2.1, hpo.2.2.1, hpo.2.2.2, fun _ => Or.inl hpo.1⟩
    | exact False.elim hpo

private theorem PreFault_val0 {base v : Nat} {s : St} (h : PreFault base v s) : hasValNotAcc s 0 := by
  rcases PreFault_regs0137 h with ⟨h0 | h0, _, _, _⟩
  · exact ⟨v, h0, h.1.1, h.1.2.1⟩
  · exact ⟨60, h0, valNotAcc_60⟩

private theorem PreFault_val3 {base v : Nat} {s : St} (h : PreFault base v s) : hasValNotAcc s 3 :=
  ⟨v, (PreFault_regs0137 h).2.2.1, h.1.1, h.1.2.1⟩

private theorem PreFault_low3 {base v : Nat} {s : St} (h : PreFault base v s) : hasLowNotAcc s 3 :=
  ⟨v, (PreFault_regs0137 h).2.2.1, h.1.1, h.1.2.2.1⟩

private theorem PreFault_exit7 {base v : Nat} {s : St} (h : PreFault base v s)
    (hle : 0x2b ≤ s.rip - base) : hasExitNonZero s := by
  intro a1 ha1
  rcases (PreFault_regs0137 h).2.2.2 hle with h7 | h7 <;> rw [h7] at ha1 <;> cases ha1 <;> decide

private theorem PreFault_to_TailSafe {base v : Nat} {s : St} (h : PreFault base v s) :
    TailSafe base s := by
  refine ⟨h.2.1, fun _ _ => ?_⟩
  have hpo := h.2.2.2.2.2
  have hv := h.1
  have hmem := preFaultOff_mem hpo
  simp only [preFaultOffs, List.mem_cons, List.not_mem_nil, or_false] at hmem
  generalize hoff : s.rip - base = off at hpo hmem ⊢
  rcases hmem with
    (rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl |
     rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl)
  · exact ⟨⟨v, hpo.1, hv.1, hv.2.1⟩, ⟨v, hpo.2.2.1, hv.1, hv.2.1⟩, ⟨v, hpo.2.1, hv.1, hv.2.1⟩⟩
  · exact ⟨⟨v, hpo.1, hv.1, hv.2.1⟩, ⟨v, hpo.2.2.1, hv.1, hv.2.1⟩, ⟨v, hpo.2.1, hv.1, hv.2.1⟩⟩
  · exact ⟨v, hpo.1, hv.1, hv.2.1⟩
  · exact Or.inl ⟨v, hpo.1, hv.1, hv.2.1⟩
  · exact Or.inl ⟨v, hpo.1, hv.1, hv.2.1⟩
  · exact Or.inl ⟨v, hpo.1, hv.1, hv.2.1⟩
  · exact Or.inl ⟨v, hpo.1, hv.1, hv.2.1⟩
  · exact Or.inr ⟨v, hpo.2.2.1, hv.1, hv.2.1⟩
  · exact Or.inr ⟨v, hpo.2.2.1, hv.1, hv.2.1⟩
  · trivial
  · trivial
  · trivial
  · trivial
  · trivial
  · trivial
  · trivial
  · exact fun a1 ha1 => by rw [hpo.1] at ha1; cases ha1; decide
  · trivial
  · trivial
  · trivial
  · exact fun a1 ha1 => by rw [hpo.1] at ha1; cases ha1; decide

private theorem TailSafe_flipReg_of_PreFault {base v : Nat} {s : St} (hs : PreFault base v s)
    {r b : Nat} (hr : r < 16) (hb64 : b < 64) :
    TailSafe base (s.flipReg r b) := by
  have hv := hs.1
  have hlen := hs.2.1
  have hpo := hs.2.2.2.2.2
  refine ⟨reg_flip_length hlen r b, fun _ _ => ?_⟩
  rw [show (s.flipReg r b).rip = s.rip from rfl]
  have flipV : ∀ i, s.reg i = some v → hasValNotAcc (s.flipReg r b) i := fun i hi => by
    by_cases heq : r = i
    · subst heq
      exact ⟨Nat.xor v (2 ^ b), by rw [reg_flip_at hlen hr b, hi]; rfl, SafeInvalid_xor hv hb64⟩
    · exact ⟨v, by rw [reg_flip_ne heq b, hi], hv.1, hv.2.1⟩
  have flipExit : ∀ c, c = 5 ∨ c = 6 → s.reg 7 = some c → hasExitNonZero (s.flipReg r b) := by
    intro c hc h7 a1 ha1
    by_cases heq : r = 7
    · subst heq
      rw [reg_flip_at hlen hr b, h7] at ha1
      injection ha1 with ha1; rw [← ha1]
      rcases hc with rfl | rfl
      · simpa [decide_eq_true_eq] using List.all_eq_true.1 xor5_mod256_nz b (List.mem_range.2 hb64)
      · simpa [decide_eq_true_eq] using List.all_eq_true.1 xor6_mod256_nz b (List.mem_range.2 hb64)
    · rw [reg_flip_ne heq b, h7] at ha1
      injection ha1 with ha1; rw [← ha1]
      rcases hc with rfl | rfl <;> decide
  have hmem := preFaultOff_mem hpo
  simp only [preFaultOffs, List.mem_cons, List.not_mem_nil, or_false] at hmem
  generalize hoff : s.rip - base = off at hpo hmem ⊢
  rcases hmem with
    (rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl |
     rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl)
  · exact ⟨flipV 0 hpo.1, flipV 3 hpo.2.2.1, flipV 1 hpo.2.1⟩
  · exact ⟨flipV 0 hpo.1, flipV 3 hpo.2.2.1, flipV 1 hpo.2.1⟩
  · exact flipV 0 hpo.1
  · exact Or.inl (flipV 0 hpo.1)
  · exact Or.inl (flipV 0 hpo.1)
  · exact Or.inl (flipV 0 hpo.1)
  · exact Or.inl (flipV 0 hpo.1)
  · exact Or.inr (flipV 3 hpo.2.2.1)
  · exact Or.inr (flipV 3 hpo.2.2.1)
  · trivial
  · trivial
  · trivial
  · trivial
  · trivial
  · trivial
  · trivial
  · exact flipExit 6 (Or.inr rfl) hpo.1
  · trivial
  · trivial
  · trivial
  · exact flipExit 5 (Or.inl rfl) hpo.1

private theorem PreFault_regFlips_safe {base v : Nat} {s s' : St}
    (hs : PreFault base v s) (hreg : s' ∈ X86Int.regFlips s) :
    TailSafe base s' := by
  simp only [X86Int.regFlips, List.mem_map, List.mem_range] at hreg
  rcases hreg with ⟨j, hj, rfl⟩
  exact TailSafe_flipReg_of_PreFault hs (by omega) (by omega)

private theorem PreFault_memFlips_safe {base v : Nat} {s s' : St}
    (hs : PreFault base v s) (hmem : s' ∈ X86Int.memFlips s) :
    TailSafe base s' := by
  simp only [X86Int.memFlips, List.mem_flatMap, List.mem_map] at hmem
  rcases hmem with ⟨i, _, j, _, b, _, rfl⟩
  have hts := PreFault_to_TailSafe hs
  have hpo := hs.2.2.2.2.2
  refine ⟨hs.2.1, fun hlo hhi => ?_⟩
  have hoff := hts.2 hlo hhi
  change TailSafeOff base (s.flipMem i j b) (s.rip - base)
  have hmem_pre := preFaultOff_mem hpo
  simp only [preFaultOffs, List.mem_cons, List.not_mem_nil, or_false] at hmem_pre
  generalize hoff_eq : s.rip - base = off at hpo hoff hmem_pre ⊢
  rcases hmem_pre with
    (rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl |
     rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl)
  all_goals exact hoff

private theorem PreFault_flagFlips_safe {base v : Nat} {s s' : St}
    (hs : PreFault base v s) (hfl : s' ∈ X86Int.flagFlips s) :
    TailSafe base s' := by
  have hts := PreFault_to_TailSafe hs
  have hval3 := PreFault_val3 hs
  have hpo := hs.2.2.2.2.2
  have hmem_aux : ∀ f, s' = { s with fl := f } → TailSafe base s' := by
    rintro f rfl
    refine ⟨hs.2.1, fun hlo hhi => ?_⟩
    have hoff := hts.2 hlo hhi
    dsimp only at hlo hhi ⊢
    have hmem_pre := preFaultOff_mem hpo
    simp only [preFaultOffs, List.mem_cons, List.not_mem_nil, or_false] at hmem_pre
    generalize hoff_eq : s.rip - base = off at hpo hoff hmem_pre ⊢
    rcases hmem_pre with
      (rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl |
       rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl)
    · exact hoff
    · exact hoff
    · exact hoff
    · exact hoff
    · exact hoff
    · exact hoff
    · exact hoff
    · exact hoff
    · exact Or.inr hval3
    · trivial
    · trivial
    · trivial
    · trivial
    · trivial
    · trivial
    · trivial
    · exact hoff
    · trivial
    · trivial
    · trivial
    · exact hoff
  simp only [X86Int.flagFlips, List.mem_append, List.mem_map, Option.mem_toList] at hfl
  rcases hfl with (((⟨_, _, rfl⟩ | ⟨_, _, rfl⟩) | ⟨_, _, rfl⟩) | ⟨_, _, rfl⟩)
  · exact hmem_aux _ rfl
  · exact hmem_aux _ rfl
  · exact hmem_aux _ rfl
  · exact hmem_aux _ rfl

theorem PreFault_step {base v : Nat} (hb : base_ok base) {s : St} (hs : PreFault base v s) :
    match X86Int.step (tailImAt base) s with
    | .ok s' => PreFault base v s'
    | .exit c => c ≠ 0
    | .stuck => True := by
  rcases hs with ⟨hv, hlen, hst, hlo, hhi, hpo⟩
  have hmem_pre : s.rip - base ∈ preFaultOffs := preFaultOff_mem hpo
  have hsrip : s.rip = base + (s.rip - base) := (Nat.add_sub_cancel' hlo).symm
  simp only [preFaultOffs, List.mem_cons, List.not_mem_nil, or_false] at hmem_pre
  rcases hmem_pre with
    (heq | heq | heq | heq | heq | heq | heq | heq | heq | heq | heq |
     heq | heq | heq | heq | heq | heq | heq | heq | heq | heq)
  all_goals
    rw [heq] at hpo hsrip
  · -- 0x00: cmp 0 3 -> 0x03
    rw [X86Int.step, hsrip, tailImAt_base_spec (by decide : (0x00, .cmp 0 3, 3) ∈ tailSpec)]
    dsimp only [shiftIns, X86Int.exec, lift, bind, Option.bind, pure]
    simp only [hpo.1, hpo.2.2.1]
    refine ⟨hv, hlen, fun sp hsp => hst sp hsp, by dsimp only; omega, by dsimp only; omega, ?_⟩
    rw [show s.rip + 3 - base = 3 by omega]
    exact ⟨hpo.1, hpo.2.1, hpo.2.2.1, fun _ => aluF_sub_zf_eq v,
      fun h => by omega, fun h => by omega, fun h => by omega⟩
  · -- 0x03: jcc 4 0x12 -> 0x12
    rw [X86Int.step, hsrip, tailImAt_base_spec (by decide : (0x03, .jcc 4 0x12, 2) ∈ tailSpec)]
    dsimp only [shiftIns, X86Int.exec, lift, bind, Option.bind, pure]
    have hfl : X86Int.cond s.fl 4 = some true := by
      rcases s with ⟨_, _, ⟨cf, zf, sf, of⟩, _, _⟩
      have hzf := hpo.2.2.2.1 rfl; dsimp only at hzf; subst hzf; rfl
    simp only [hfl, ↓reduceIte, shift_tgt_lt hb (by decide : 0x12 < 143)]
    refine ⟨hv, hlen, fun sp hsp => hst sp hsp, by dsimp only; omega, by dsimp only; omega, ?_⟩
    rw [show base + 0x12 - base = 0x12 by omega]
    exact ⟨hpo.1, hpo.2.1, hpo.2.2.1, fun h => by omega, fun h => by omega, fun h => by omega, fun h => by omega⟩
  · -- 0x12: mov 3 0 -> 0x15
    rw [X86Int.step, hsrip, tailImAt_base_spec (by decide : (0x12, .mov 3 0, 3) ∈ tailSpec)]
    dsimp only [shiftIns, X86Int.exec, lift, bind, Option.bind, pure]
    simp only [hpo.1]
    refine ⟨hv, set_length hlen _ _, fun sp hsp => by rw [reg_set_mk_ne (by decide)] at hsp; exact hst sp hsp, by dsimp only; omega, by dsimp only; omega, ?_⟩
    rw [show s.rip + 3 - base = 0x15 by omega]
    refine ⟨?_, ?_, ?_, fun h => by omega, fun h => by omega, fun h => by omega, fun h => by omega⟩
    · rw [reg_set_mk_ne (by decide), hpo.1]
    · rw [reg_set_mk_ne (by decide), hpo.2.1]
    · rw [reg_set_mk hlen (by decide)]
  · -- 0x15: mov 1 0 -> 0x18
    rw [X86Int.step, hsrip, tailImAt_base_spec (by decide : (0x15, .mov 1 0, 3) ∈ tailSpec)]
    dsimp only [shiftIns, X86Int.exec, lift, bind, Option.bind, pure]
    simp only [hpo.1]
    refine ⟨hv, set_length hlen _ _, fun sp hsp => by rw [reg_set_mk_ne (by decide)] at hsp; exact hst sp hsp, by dsimp only; omega, by dsimp only; omega, ?_⟩
    rw [show s.rip + 3 - base = 0x18 by omega]
    refine ⟨?_, ?_, ?_, fun h => by omega, fun h => by omega, fun h => by omega, fun h => by omega⟩
    · rw [reg_set_mk_ne (by decide), hpo.1]
    · rw [reg_set_mk hlen (by decide)]
    · rw [reg_set_mk_ne (by decide), hpo.2.2.1]
  · -- 0x18: jmp 0x26 -> 0x26
    rw [X86Int.step, hsrip, tailImAt_base_spec (by decide : (0x18, .jmp 0x26, 2) ∈ tailSpec)]
    dsimp only [shiftIns, X86Int.exec, lift, bind, Option.bind, pure]
    simp only [shift_tgt_lt hb (by decide : 0x26 < 143)]
    refine ⟨hv, hlen, fun sp hsp => hst sp hsp, by dsimp only; omega, by dsimp only; omega, ?_⟩
    rw [show base + 0x26 - base = 0x26 by omega]
    exact ⟨hpo.1, hpo.2.1, hpo.2.2.1, fun h => by omega, fun h => by omega, fun h => by omega, fun h => by omega⟩
  · -- 0x26: movI 7 5 -> 0x2b
    rw [X86Int.step, hsrip, tailImAt_base_spec (by decide : (0x26, .movI 7 5, 5) ∈ tailSpec)]
    dsimp only [shiftIns, X86Int.exec]
    refine ⟨hv, set_length hlen _ _, fun sp hsp => by rw [reg_set_mk_ne (by decide)] at hsp; exact hst sp hsp, by dsimp only; omega, by dsimp only; omega, ?_⟩
    rw [show s.rip + 5 - base = 0x2b by omega]
    refine ⟨?_, ?_, ?_, fun h => by omega, fun h => by omega, fun h => by omega, fun _ => ?_⟩
    · rw [reg_set_mk_ne (by decide), hpo.1]
    · rw [reg_set_mk_ne (by decide), hpo.2.1]
    · rw [reg_set_mk_ne (by decide), hpo.2.2.1]
    · rw [reg_set_mk hlen (by decide)]; decide
  · -- 0x2b: movI 2 -ACCEPT -> 0x35
    rw [X86Int.step, hsrip, tailImAt_base_spec (by decide : (0x2b, .movI 2 0xa533168f5aa5c36a, 10) ∈ tailSpec)]
    dsimp only [shiftIns, X86Int.exec]
    refine ⟨hv, set_length hlen _ _, fun sp hsp => by rw [reg_set_mk_ne (by decide)] at hsp; exact hst sp hsp, by dsimp only; omega, by dsimp only; omega, ?_⟩
    rw [show s.rip + 10 - base = 0x35 by omega]
    refine ⟨?_, ?_, ?_, fun h => by omega, fun _ => ?_, fun h => by omega, fun _ => ?_⟩
    · rw [reg_set_mk_ne (by decide), hpo.1]
    · rw [reg_set_mk_ne (by decide), hpo.2.1]
    · rw [reg_set_mk_ne (by decide), hpo.2.2.1]
    · rw [reg_set_mk hlen (by decide)]; decide
    · rw [reg_set_mk_ne (by decide), hpo.2.2.2.2.2.2 (by decide)]
  · -- 0x35: alu .add 2 0 -> 0x38
    rw [X86Int.step, hsrip, tailImAt_base_spec (by decide : (0x35, .alu .add 2 0, 3) ∈ tailSpec)]
    dsimp only [shiftIns, X86Int.exec, lift, bind, Option.bind, pure]
    have hw2 := hpo.2.2.2.2.1 rfl
    have hw7 := hpo.2.2.2.2.2.2 (by decide)
    simp only [hw2, hpo.1]
    refine ⟨hv, set_length hlen _ _, fun sp hsp => by rw [reg_set_mk_ne (by decide)] at hsp; exact hst sp hsp, by dsimp only; omega, by dsimp only; omega, ?_⟩
    rw [show s.rip + 3 - base = 0x38 by omega]
    have hacc0 : 0 < ACCEPT := by decide
    have haccW : ACCEPT < X86Env.W := by decide
    refine ⟨?_, ?_, ?_, fun h => by omega, fun h => by omega, fun _ => ?_, fun _ => ?_⟩
    · rw [reg_set_mk_ne (by decide), hpo.1]
    · rw [reg_set_mk_ne (by decide), hpo.2.1]
    · rw [reg_set_mk_ne (by decide), hpo.2.2.1]
    · exact (aluF_add_compl_zf hacc0 haccW hv.1).2.mpr hv.2.1
    · rw [reg_set_mk_ne (by decide), hw7]
  · -- 0x38: jcc 5 0x59 -> 0x59
    rw [X86Int.step, hsrip, tailImAt_base_spec (by decide : (0x38, .jcc 5 0x59, 2) ∈ tailSpec)]
    dsimp only [shiftIns, X86Int.exec, lift, bind, Option.bind, pure]
    have hw7 := hpo.2.2.2.2.2.2 (by decide)
    have hfl : X86Int.cond s.fl 5 = some true := by
      rcases s with ⟨_, _, ⟨cf, zf, sf, of⟩, _, _⟩
      have hzf := hpo.2.2.2.2.2.1 rfl; dsimp only at hzf; subst hzf; rfl
    simp only [hfl, ↓reduceIte, shift_tgt_lt hb (by decide : 0x59 < 143)]
    refine ⟨hv, hlen, fun sp hsp => hst sp hsp, by dsimp only; omega, by dsimp only; omega, ?_⟩
    rw [show base + 0x59 - base = 0x59 by omega]
    exact ⟨hpo.1, hpo.2.1, hpo.2.2.1, fun h => by omega, fun h => by omega, fun h => by omega, fun _ => hw7⟩
  · -- 0x59: movI 7 5 -> 0x5e
    rw [X86Int.step, hsrip, tailImAt_base_spec (by decide : (0x59, .movI 7 5, 5) ∈ tailSpec)]
    dsimp only [shiftIns, X86Int.exec]
    refine ⟨hv, set_length hlen _ _, fun sp hsp => by rw [reg_set_mk_ne (by decide)] at hsp; exact hst sp hsp, by dsimp only; omega, by dsimp only; omega, ?_⟩
    rw [show s.rip + 5 - base = 0x5e by omega]
    refine ⟨?_, ?_, ?_, fun h => by omega, fun h => by omega, fun h => by omega, fun _ => ?_⟩
    · rw [reg_set_mk_ne (by decide), hpo.1]
    · rw [reg_set_mk_ne (by decide), hpo.2.1]
    · rw [reg_set_mk_ne (by decide), hpo.2.2.1]
    · rw [reg_set_mk hlen (by decide)]; decide
  · -- 0x5e: movI 2 -OVERFLOW -> 0x68
    rw [X86Int.step, hsrip, tailImAt_base_spec (by decide : (0x5e, .movI 2 0xf452f0e1cc3c3c5b, 10) ∈ tailSpec)]
    dsimp only [shiftIns, X86Int.exec]
    have hw7 := hpo.2.2.2.2.2.2 (by decide)
    refine ⟨hv, set_length hlen _ _, fun sp hsp => by rw [reg_set_mk_ne (by decide)] at hsp; exact hst sp hsp, by dsimp only; omega, by dsimp only; omega, ?_⟩
    rw [show s.rip + 10 - base = 0x68 by omega]
    refine ⟨?_, ?_, ?_, fun h => by omega, fun h => by omega, fun h => by omega, fun _ => ?_⟩
    · rw [reg_set_mk_ne (by decide), hpo.1]
    · rw [reg_set_mk_ne (by decide), hpo.2.1]
    · rw [reg_set_mk_ne (by decide), hpo.2.2.1]
    · rw [reg_set_mk_ne (by decide), hw7]
  · -- 0x68: alu .add 2 0 -> 0x6b
    rw [X86Int.step, hsrip, tailImAt_base_spec (by decide : (0x68, .alu .add 2 0, 3) ∈ tailSpec)]
    dsimp only [shiftIns, X86Int.exec, lift, bind, Option.bind, pure]
    have hw7 := hpo.2.2.2.2.2.2 (by decide)
    cases h2 : s.reg 2 with
    | none => simp
    | some v2 =>
      simp only [hpo.1]
      refine ⟨hv, set_length hlen _ _, fun sp hsp => by rw [reg_set_mk_ne (by decide)] at hsp; exact hst sp hsp, by dsimp only; omega, by dsimp only; omega, ?_⟩
      rw [show s.rip + 3 - base = 0x6b by omega]
      refine ⟨?_, ?_, ?_, fun h => by omega, fun h => by omega, fun h => by omega, fun _ => ?_⟩
      · rw [reg_set_mk_ne (by decide), hpo.1]
      · rw [reg_set_mk_ne (by decide), hpo.2.1]
      · rw [reg_set_mk_ne (by decide), hpo.2.2.1]
      · rw [reg_set_mk_ne (by decide), hw7]
  · -- 0x6b: jcc 5 0x7e -> 0x7e or 0x6d
    rw [X86Int.step, hsrip, tailImAt_base_spec (by decide : (0x6b, .jcc 5 0x7e, 2) ∈ tailSpec)]
    dsimp only [shiftIns, X86Int.exec, lift, bind, Option.bind, pure]
    have hw7 := hpo.2.2.2.2.2.2 (by decide)
    cases hfl : X86Int.cond s.fl 5 with
    | none => simp
    | some c =>
      simp only [shift_tgt_lt hb (by decide : 0x7e < 143)]
      cases c with
      | false =>
        simp only [Bool.false_eq_true, ↓reduceIte]
        refine ⟨hv, hlen, fun sp hsp => hst sp hsp, by dsimp only; omega, by dsimp only; omega, ?_⟩
        rw [show s.rip + 2 - base = 0x6d by omega]
        exact ⟨hw7, hpo.1, hpo.2.1, hpo.2.2.1⟩
      | true =>
        simp only [↓reduceIte]
        refine ⟨hv, hlen, fun sp hsp => hst sp hsp, by dsimp only; omega, by dsimp only; omega, ?_⟩
        rw [show base + 0x7e - base = 0x7e by omega]
        exact ⟨hw7, hpo.1, hpo.2.1, hpo.2.2.1⟩
  · -- 0x6d: movI 7 6 -> 0x72
    rw [X86Int.step, hsrip, tailImAt_base_spec (by decide : (0x6d, .movI 7 6, 5) ∈ tailSpec)]
    dsimp only [shiftIns, X86Int.exec]
    refine ⟨hv, set_length hlen _ _, fun sp hsp => by rw [reg_set_mk_ne (by decide)] at hsp; exact hst sp hsp, by dsimp only; omega, by dsimp only; omega, ?_⟩
    rw [show s.rip + 5 - base = 0x72 by omega]
    refine ⟨?_, ?_, ?_, ?_⟩
    · rw [reg_set_mk hlen (by decide)]; decide
    · rw [reg_set_mk_ne (by decide), hpo.2.1]
    · rw [reg_set_mk_ne (by decide), hpo.2.2.1]
    · rw [reg_set_mk_ne (by decide), hpo.2.2.2]
  · -- 0x72: movI 0 60 -> 0x77
    rw [X86Int.step, hsrip, tailImAt_base_spec (by decide : (0x72, .movI 0 60, 5) ∈ tailSpec)]
    dsimp only [shiftIns, X86Int.exec]
    refine ⟨hv, set_length hlen _ _, fun sp hsp => by rw [reg_set_mk_ne (by decide)] at hsp; exact hst sp hsp, by dsimp only; omega, by dsimp only; omega, ?_⟩
    rw [show s.rip + 5 - base = 0x77 by omega]
    refine ⟨?_, ?_, ?_, ?_⟩
    · rw [reg_set_mk_ne (by decide), hpo.1]
    · rw [reg_set_mk hlen (by decide)]; decide
    · rw [reg_set_mk_ne (by decide), hpo.2.2.1]
    · rw [reg_set_mk_ne (by decide), hpo.2.2.2]
  · -- 0x77: movI 7 6 -> 0x7c
    rw [X86Int.step, hsrip, tailImAt_base_spec (by decide : (0x77, .movI 7 6, 5) ∈ tailSpec)]
    dsimp only [shiftIns, X86Int.exec]
    refine ⟨hv, set_length hlen _ _, fun sp hsp => by rw [reg_set_mk_ne (by decide)] at hsp; exact hst sp hsp, by dsimp only; omega, by dsimp only; omega, ?_⟩
    rw [show s.rip + 5 - base = 0x7c by omega]
    refine ⟨?_, ?_, ?_, ?_⟩
    · rw [reg_set_mk hlen (by decide)]; decide
    · rw [reg_set_mk_ne (by decide), hpo.2.1]
    · rw [reg_set_mk_ne (by decide), hpo.2.2.1]
    · rw [reg_set_mk_ne (by decide), hpo.2.2.2]
  · -- 0x7c: syscall -> exit 6 or stuck
    rw [X86Int.step, hsrip, tailImAt_base_spec (by decide : (0x7c, .syscall, 2) ∈ tailSpec)]
    dsimp only [shiftIns, X86Int.exec, lift, bind, Option.bind, pure]
    cases h2 : s.reg 2 with
    | none => simp
    | some _ => cases h6 : s.reg 6 with
      | none => simp
      | some _ => cases h10 : s.reg 10 with
        | none => simp
        | some _ => cases h8 : s.reg 8 with
          | none => simp
          | some _ => cases h9 : s.reg 9 with
            | none => simp
            | some _ =>
              simp only [hpo.2.1, hpo.1]
              cases hsys : X86Env.syscall s.k s.mem 60 6 _ _ _ _ _ with
              | none => simp
              | some r => cases r with
                | exit c => simp; rw [syscall_exit_only hsys]; decide
                | ret => exfalso; exact syscall60_no_ret hsys
  · -- 0x7e: movI 7 5 -> 0x83
    rw [X86Int.step, hsrip, tailImAt_base_spec (by decide : (0x7e, .movI 7 5, 5) ∈ tailSpec)]
    dsimp only [shiftIns, X86Int.exec]
    refine ⟨hv, set_length hlen _ _, fun sp hsp => by rw [reg_set_mk_ne (by decide)] at hsp; exact hst sp hsp, by dsimp only; omega, by dsimp only; omega, ?_⟩
    rw [show s.rip + 5 - base = 0x83 by omega]
    refine ⟨?_, ?_, ?_, ?_⟩
    · rw [reg_set_mk hlen (by decide)]; decide
    · rw [reg_set_mk_ne (by decide), hpo.2.1]
    · rw [reg_set_mk_ne (by decide), hpo.2.2.1]
    · rw [reg_set_mk_ne (by decide), hpo.2.2.2]
  · -- 0x83: movI 0 60 -> 0x88
    rw [X86Int.step, hsrip, tailImAt_base_spec (by decide : (0x83, .movI 0 60, 5) ∈ tailSpec)]
    dsimp only [shiftIns, X86Int.exec]
    refine ⟨hv, set_length hlen _ _, fun sp hsp => by rw [reg_set_mk_ne (by decide)] at hsp; exact hst sp hsp, by dsimp only; omega, by dsimp only; omega, ?_⟩
    rw [show s.rip + 5 - base = 0x88 by omega]
    refine ⟨?_, ?_, ?_, ?_⟩
    · rw [reg_set_mk_ne (by decide), hpo.1]
    · rw [reg_set_mk hlen (by decide)]; decide
    · rw [reg_set_mk_ne (by decide), hpo.2.2.1]
    · rw [reg_set_mk_ne (by decide), hpo.2.2.2]
  · -- 0x88: movI 7 5 -> 0x8d
    rw [X86Int.step, hsrip, tailImAt_base_spec (by decide : (0x88, .movI 7 5, 5) ∈ tailSpec)]
    dsimp only [shiftIns, X86Int.exec]
    refine ⟨hv, set_length hlen _ _, fun sp hsp => by rw [reg_set_mk_ne (by decide)] at hsp; exact hst sp hsp, by dsimp only; omega, by dsimp only; omega, ?_⟩
    rw [show s.rip + 5 - base = 0x8d by omega]
    refine ⟨?_, ?_, ?_, ?_⟩
    · rw [reg_set_mk hlen (by decide)]; decide
    · rw [reg_set_mk_ne (by decide), hpo.2.1]
    · rw [reg_set_mk_ne (by decide), hpo.2.2.1]
    · rw [reg_set_mk_ne (by decide), hpo.2.2.2]
  · -- 0x8d: syscall -> exit 5 or stuck
    rw [X86Int.step, hsrip, tailImAt_base_spec (by decide : (0x8d, .syscall, 2) ∈ tailSpec)]
    dsimp only [shiftIns, X86Int.exec, lift, bind, Option.bind, pure]
    cases h2 : s.reg 2 with
    | none => simp
    | some _ => cases h6 : s.reg 6 with
      | none => simp
      | some _ => cases h10 : s.reg 10 with
        | none => simp
        | some _ => cases h8 : s.reg 8 with
          | none => simp
          | some _ => cases h9 : s.reg 9 with
            | none => simp
            | some _ =>
              simp only [hpo.2.1, hpo.1]
              cases hsys : X86Env.syscall s.k s.mem 60 5 _ _ _ _ _ with
              | none => simp
              | some r => cases r with
                | exit c => simp; rw [syscall_exit_only hsys]; decide
                | ret => exfalso; exact syscall60_no_ret hsys

private theorem jumpTarget_TailSafe {base off t : Nat} (hb : base_ok base) {s : St}
    (hlen : s.regs.length = 16) (hrange : t < 2 ^ 31 ∨ (X86Env.W - 256 ≤ t ∧ t < X86Env.W))
    (hallow : jumpTargetAllowedFromPre off t = true)
    (hv0 : hasValNotAcc s 0) (hv3 : hasValNotAcc s 3) (hl3 : hasLowNotAcc s 3)
    (hst : t ∈ [0x14, 0x2e, 0x3d] → TailStackSafe base s)
    (h50 : 0x2b ≤ off → hasExitNonZero s) :
    TailSafe base { s with rip := (base + t) % X86Env.W } := by
  refine ⟨hlen, fun hlo hhi => ?_⟩
  rcases hrange with hlt | hneg
  · have hm : (base + t) % X86Env.W = base + t := Nat.mod_eq_of_lt (by have := hb.2; omega)
    dsimp only at hlo hhi ⊢
    rw [hm] at hlo hhi ⊢
    rw [Nat.add_sub_cancel_left]
    have ht143 : t < 143 := by omega
    simp only [jumpTargetAllowedFromPre, show ¬ 143 ≤ t by omega, ↓reduceIte,
      Bool.and_eq_true, Bool.or_eq_true, List.mem_cons, List.not_mem_nil, or_false,
      decide_eq_true_eq] at hallow
    rcases hallow with ⟨h50ok, ht_cases⟩
    have hge2b : t = 0x50 ∨ t = 0x55 → 0x2b ≤ off := by
      intro h; rcases h50ok with hnot | hle
      · exfalso; rcases h with rfl | rfl <;> simp at hnot
      · exact hle
    rcases ht_cases with (rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl) | hnone
    · exact ⟨hv0, hv3⟩
    · exact hv0
    · exact hst (by simp)
    · exact Or.inl hv0
    · exact hl3
    · trivial
    · exact h50 (hge2b (Or.inr rfl))
    · trivial
    · trivial
    · trivial
    · trivial
    · exact tail_isNone_TailSafeOff_guarded ht143 hnone (fun _ => hv3) hst (fun h => by subst h; exact h50 (hge2b (Or.inl rfl)))
  · have hlt_base : (base + t) % X86Env.W < base := by
      have hb0 := hb.1
      have hm : base + t = X86Env.W + (base + t - X86Env.W) := by omega
      have hsub : base + t - X86Env.W < base := by omega
      have hltW : base + t - X86Env.W < X86Env.W := by have := hb.2; omega
      rw [hm, Nat.add_mod_left, Nat.mod_eq_of_lt hltW]
      exact hsub
    dsimp only at hlo
    exfalso; omega

theorem PreFault_fetch_safe {base v : Nat} (hb : base_ok base) {s : St}
    (hs : PreFault base v s) {alt : Option (Ins × Nat)}
    (halt : alt ∈ tailAltsAt base s.rip) :
    match X86Int.execO s alt with
    | .ok s' => TailSafe base s'
    | .exit c => c ≠ 0
    | .stuck => True := by
  have hs_orig := hs
  rcases hs with ⟨hv, hlen, hst, hlo, hhi, hpo⟩
  have hmem_pre := preFaultOff_mem hpo
  have hsrip : s.rip = base + (s.rip - base) := (Nat.add_sub_cancel' hlo).symm
  generalize hoff_eq : s.rip - base = off at hpo hsrip hmem_pre
  rw [hsrip, tailAltsAt_eq] at halt
  simp only [List.mem_map] at halt
  rcases halt with ⟨alt0, halt0, rfl⟩
  have hpre_ok := List.all_eq_true.1
    (List.all_eq_true.1 tailAlts_preFault_safe off hmem_pre) alt0 halt0
  cases alt0 with
  | none => exact trivial
  | some p =>
    rcases p with ⟨ins0, len⟩
    dsimp only [Option.map_some, X86Int.execO]
    dsimp only [altSafeFromPre] at hpre_ok
    cases hjr : altJumpRangeOk ins0 with
    | false => simp [hjr] at hpre_ok
    | true =>
      simp only [hjr, Bool.not_true, Bool.false_eq_true, ↓reduceIte] at hpre_ok
      have hregs0137 := PreFault_regs0137 hs_orig
      have hv0 := PreFault_val0 hs_orig
      have hv3 := PreFault_val3 hs_orig
      have hl3 := PreFault_low3 hs_orig
      have h50 : 0x2b ≤ off → hasExitNonZero s := fun hle => by
        apply PreFault_exit7 hs_orig; rw [hoff_eq]; exact hle
      have hrange : ∀ t, (t < 2 ^ 31 || (X86Env.W - 256 ≤ t && t < X86Env.W)) = true →
          t < 2 ^ 31 ∨ (X86Env.W - 256 ≤ t ∧ t < X86Env.W) := by
        intro t ht
        simp only [Bool.or_eq_true, Bool.and_eq_true, decide_eq_true_eq] at ht
        exact ht
      cases ins0 with
      | jmp t =>
        dsimp only at hpre_ok
        simp only [altJumpRangeOk] at hjr
        exact jumpTarget_TailSafe hb hlen (hrange t hjr) hpre_ok hv0 hv3 hl3
          (fun _ => hst) h50
      | call t =>
        dsimp only at hpre_ok
        simp only [altJumpRangeOk] at hjr
        simp only [Bool.and_eq_true, Bool.or_eq_true, decide_eq_true_eq] at hpre_ok
        dsimp only [shiftIns, X86Int.exec, lift, bind, Option.bind, pure]
        cases h4 : s.reg 4 with
        | none => exact trivial
        | some sp =>
          dsimp only
          cases X86Int.wr64 s.mem ((sp + X86Env.W - 8) % X86Env.W) (s.rip + len) with
          | none => exact trivial
          | some m' =>
            dsimp only
            refine jumpTarget_TailSafe (s := { s.set 4 (some ((sp + X86Env.W - 8) % X86Env.W)) with mem := m' })
              hb (by dsimp only; exact set_length hlen _ _) (hrange t hjr) hpre_ok.1 ?_ ?_ ?_ ?_ ?_
            · rcases hv0 with ⟨w, hw, hvw⟩; exact ⟨w, by rw [reg_set_mk_ne (by decide), hw], hvw⟩
            · rcases hv3 with ⟨w, hw, hvw⟩; exact ⟨w, by rw [reg_set_mk_ne (by decide), hw], hvw⟩
            · rcases hl3 with ⟨w, hw, hlw⟩; exact ⟨w, by rw [reg_set_mk_ne (by decide), hw], hlw⟩
            · intro hmem3
              exfalso
              rcases hpre_ok.2 with h143 | hnot
              · simp only [List.mem_cons, List.not_mem_nil, or_false] at hmem3
                omega
              · exact absurd hmem3 (by simp only [List.mem_cons, List.not_mem_nil, or_false] at hnot ⊢; exact hnot)
            · intro hle a1 ha1; rw [reg_set_mk_ne (by decide)] at ha1; exact h50 hle a1 ha1
      | jcc cc t =>
        dsimp only at hpre_ok
        simp only [altJumpRangeOk] at hjr
        simp only [Bool.and_eq_true, Bool.or_eq_true, decide_eq_true_eq] at hpre_ok
        dsimp only [shiftIns, X86Int.exec, lift, bind, Option.bind, pure]
        cases hfl : X86Int.cond s.fl cc with
        | none => exact trivial
        | some c =>
          cases c with
          | false =>
            simp only [Bool.false_eq_true, ↓reduceIte]
            refine ⟨hlen, fun _ hhi' => ?_⟩
            dsimp only at hhi' ⊢
            have hhi_off : off + len < 143 := by rw [hsrip] at hhi'; omega
            rw [hsrip, show base + off + len - base = off + len by omega]
            rcases hpre_ok.2 with (hge | ⟨hnone, h50nx⟩) | h3
            · exfalso; omega
            · refine tail_isNone_TailSafeOff hhi_off hnone hv3 hst (fun heq => by
                apply h50
                rcases h50nx with hne | hle
                · exfalso; simp [heq] at hne
                · exact hle)
            · simp only [List.mem_cons, List.not_mem_nil, or_false] at h3
              rcases h3 with heq | heq | heq <;> rw [heq]
              · exact ⟨hv0, hv3, ⟨v, hregs0137.2.1, hv.1, hv.2.1⟩⟩
              · exact hv3
              · trivial
          | true =>
            simp only [↓reduceIte]
            exact jumpTarget_TailSafe hb hlen (hrange t hjr) hpre_ok.1 hv0 hv3 hl3
              (fun _ => hst) h50
      | ret => dsimp only at hpre_ok; cases hpre_ok
      | jmpR _ => dsimp only at hpre_ok; cases hpre_ok
      | jtab _ _ _ => dsimp only at hpre_ok; cases hpre_ok
      | syscall => dsimp only at hpre_ok; cases hpre_ok
      | ld _ _ _ => dsimp only at hpre_ok; cases hpre_ok
      | ld32 _ _ _ => dsimp only at hpre_ok; cases hpre_ok
      | ld8 _ _ _ => dsimp only at hpre_ok; cases hpre_ok
      | st8 _ _ _ => dsimp only at hpre_ok; cases hpre_ok
      | push r =>
        dsimp only at hpre_ok
        dsimp only [shiftIns, X86Int.exec, lift, bind, Option.bind, pure]
        cases hr : s.reg r with
        | none => exact trivial
        | some rv =>
          dsimp only
          cases h4 : s.reg 4 with
          | none => exact trivial
          | some sp =>
            dsimp only
            cases X86Int.wr64 s.mem ((sp + X86Env.W - 8) % X86Env.W) rv with
            | none => exact trivial
            | some m' =>
              exact nxSafeGeneric_TailSafe (set_length hlen _ _) (by dsimp only; rw [hsrip]; omega) hpre_ok
      | pushI imm =>
        dsimp only at hpre_ok
        dsimp only [shiftIns, X86Int.exec, lift, bind, Option.bind, pure]
        cases h4 : s.reg 4 with
        | none => exact trivial
        | some sp =>
          dsimp only
          cases X86Int.wr64 s.mem ((sp + X86Env.W - 8) % X86Env.W) (imm % X86Env.W) with
          | none => exact trivial
          | some m' =>
            exact nxSafeGeneric_TailSafe (set_length hlen _ _) (by dsimp only; rw [hsrip]; omega) hpre_ok
      | pop d =>
        dsimp only at hpre_ok
        simp only [Bool.and_eq_true, Bool.or_eq_true, decide_eq_true_eq, List.mem_cons,
          List.not_mem_nil, or_false] at hpre_ok
        rcases hpre_ok with ⟨hd, hnx⟩
        have hd0 : d = 0 := beq_iff_eq.1 hd
        subst hd0
        dsimp only [shiftIns, X86Int.exec, lift, bind, Option.bind, pure]
        cases h4 : s.reg 4 with
        | none => exact trivial
        | some sp =>
          dsimp only
          cases hrd : X86Int.rd64 s.mem sp with
          | none => exact trivial
          | some w =>
            dsimp only
            rcases hnx with hgen | hspec
            · exact nxSafeGeneric_TailSafe (set_length (set_length hlen _ _) _ _)
                (by dsimp only; rw [hsrip]; omega) hgen
            · refine ⟨by simp [St.set, List.length_set, hlen], fun _ hhi' => ?_⟩
              dsimp only at hhi' ⊢
              rw [hsrip, show base + off + len - base = off + len by omega]
              rcases hspec with heq | heq | heq | heq <;> rw [heq]
              · trivial
              · exact Or.inr ⟨v, (by rw [reg_set_mk_ne (by decide), reg_set_ne (by decide), hregs0137.2.2.1]), hv.1, hv.2.1⟩
              · trivial
              · trivial
      | st b disp r =>
        dsimp only at hpre_ok
        dsimp only [shiftIns, X86Int.exec, lift, bind, Option.bind, pure]
        cases hb' : s.reg b with
        | none => exact trivial
        | some ab =>
          dsimp only
          cases hr' : s.reg r with
          | none => exact trivial
          | some vr =>
            dsimp only
            cases X86Int.wr64 s.mem (X86Int.ea ab disp) vr with
            | none => exact trivial
            | some m' =>
              simp only [Bool.or_eq_true] at hpre_ok
              rcases hpre_ok with hgen | h16
              · exact nxSafeGeneric_TailSafe hlen (by dsimp only; rw [hsrip]; omega) hgen
              · refine ⟨hlen, fun _ _ => ?_⟩
                dsimp only; rw [hsrip, show base + off + len - base = off + len by omega, beq_iff_eq.1 h16]
                exact Or.inr hv3
      | cmp a b =>
        dsimp only at hpre_ok
        dsimp only [shiftIns, X86Int.exec, lift, bind, Option.bind, pure]
        cases s.reg a with
        | none => exact trivial
        | some xa =>
          dsimp only
          cases s.reg b with
          | none => exact trivial
          | some yb =>
            simp only [Bool.or_eq_true, Bool.and_eq_true, decide_eq_true_eq,
              List.mem_cons, List.not_mem_nil, or_false] at hpre_ok
            refine ⟨hlen, fun _ hhi' => ?_⟩
            dsimp only at hhi' ⊢
            have hhi_off : off + len < 143 := by rw [hsrip] at hhi'; omega
            rw [hsrip, show base + off + len - base = off + len by omega]
            rcases hpre_ok with (hge | ⟨hnone, h50ok⟩) | (h03 | h38 | h6b)
            · exfalso; omega
            · exact tail_isNone_TailSafeOff hhi_off hnone hv3 hst (fun heq => by
                apply h50; rcases h50ok with hne | hle
                · exfalso; simp [heq] at hne
                · exact hle)
            · rw [h03]; exact ⟨hv0, hv3, ⟨v, hregs0137.2.1, hv.1, hv.2.1⟩⟩
            · rw [h38]; exact Or.inr hv3
            · rw [h6b]; trivial
      | cmpI a imm =>
        dsimp only at hpre_ok
        dsimp only [shiftIns, X86Int.exec, lift, bind, Option.bind, pure]
        cases s.reg a with
        | none => exact trivial
        | some xa =>
          simp only [Bool.or_eq_true, Bool.and_eq_true, decide_eq_true_eq,
            List.mem_cons, List.not_mem_nil, or_false] at hpre_ok
          refine ⟨hlen, fun _ hhi' => ?_⟩
          dsimp only at hhi' ⊢
          have hhi_off : off + len < 143 := by rw [hsrip] at hhi'; omega
          rw [hsrip, show base + off + len - base = off + len by omega]
          rcases hpre_ok with (hge | ⟨hnone, h50ok⟩) | (h03 | h38 | h6b)
          · exfalso; omega
          · exact tail_isNone_TailSafeOff hhi_off hnone hv3 hst (fun heq => by
              apply h50; rcases h50ok with hne | hle
              · exfalso; simp [heq] at hne
              · exact hle)
          · rw [h03]; exact ⟨hv0, hv3, ⟨v, hregs0137.2.1, hv.1, hv.2.1⟩⟩
          · rw [h38]; exact Or.inr hv3
          · rw [h6b]; trivial
      | test a b =>
        dsimp only at hpre_ok
        dsimp only [shiftIns, X86Int.exec, lift, bind, Option.bind, pure]
        cases s.reg a with
        | none => exact trivial
        | some xa =>
          dsimp only
          cases s.reg b with
          | none => exact trivial
          | some yb =>
            simp only [Bool.or_eq_true, Bool.and_eq_true, decide_eq_true_eq,
              List.mem_cons, List.not_mem_nil, or_false] at hpre_ok
            refine ⟨hlen, fun _ hhi' => ?_⟩
            dsimp only at hhi' ⊢
            have hhi_off : off + len < 143 := by rw [hsrip] at hhi'; omega
            rw [hsrip, show base + off + len - base = off + len by omega]
            rcases hpre_ok with (hge | ⟨hnone, h50ok⟩) | (h03 | h38 | h6b)
            · exfalso; omega
            · exact tail_isNone_TailSafeOff hhi_off hnone hv3 hst (fun heq => by
                apply h50; rcases h50ok with hne | hle
                · exfalso; simp [heq] at hne
                · exact hle)
            · rw [h03]; exact ⟨hv0, hv3, ⟨v, hregs0137.2.1, hv.1, hv.2.1⟩⟩
            · rw [h38]; exact Or.inr hv3
            · rw [h6b]; trivial
      | movI d imm =>
        dsimp only at hpre_ok
        dsimp only [shiftIns, X86Int.exec]
        refine ⟨set_length hlen _ _, fun _ hhi' => ?_⟩
        dsimp only at hhi' ⊢
        have hhi_off : off + len < 143 := by rw [hsrip] at hhi'; omega
        rw [hsrip, show base + off + len - base = off + len by omega]
        split at hpre_ok
        · next harm =>
          simp only [Bool.or_eq_true, Bool.and_eq_true, decide_eq_true_eq] at harm
          rcases harm with hge | ⟨⟨hnone, h50ok⟩, hmem5⟩
          · exfalso; omega
          · refine tail_isNone_TailSafeOff_guarded hhi_off hnone
              (fun h16 => by
                rcases hmem5 with hnot | ⟨⟨hd3, _⟩, _⟩
                · exfalso; apply hnot; rw [h16]; simp
                · exact ⟨v, (by rw [reg_set_mk_ne (by intro h; simp [h] at hd3), hregs0137.2.2.1]), hv.1, hv.2.1⟩)
              (fun hret => by
                rcases hmem5 with hnot | ⟨⟨_, hd4⟩, _⟩
                · exfalso; apply hnot
                  simp only [List.mem_cons, List.not_mem_nil, or_false] at hret ⊢
                  rcases hret with h | h | h <;> simp [h]
                · intro sp hsp t hrd
                  rw [reg_set_mk_ne (by intro h; simp [h] at hd4)] at hsp
                  exact hst sp hsp t hrd)
              (fun heq => by
                rcases hmem5 with hnot | ⟨_, hd7⟩
                · exfalso; apply hnot; rw [heq]; simp
                · have hge2b : 0x2b ≤ off := by
                    rcases h50ok with hne | hle
                    · exfalso; simp [heq] at hne
                    · exact hle
                  intro a1 ha1
                  rw [reg_set_mk_ne (by intro h; simp [h] at hd7)] at ha1
                  exact h50 hge2b a1 ha1)
        · split at hpre_ok
          · next h00 =>
            simp only [beq_iff_eq, Bool.and_eq_true, bne_iff_ne] at hpre_ok
            rcases hpre_ok with ⟨⟨hnx, hd0⟩, hd3⟩
            rw [hnx]
            have h0eq : s.reg 0 = some v := by rw [beq_iff_eq.1 h00] at hpo; exact hpo.1
            exact ⟨⟨v, (by rw [reg_set_mk_ne hd0, h0eq]), hv.1, hv.2.1⟩,
              ⟨v, (by rw [reg_set_mk_ne hd3, hregs0137.2.2.1]), hv.1, hv.2.1⟩⟩
          · split at hpre_ok
            · next h26 =>
              have hnx : off + len = 0x2b := by simpa using hpre_ok
              rw [hnx]
              by_cases hd0 : d = 0
              · exact Or.inr ⟨v, (by rw [reg_set_mk_ne (by omega), hregs0137.2.2.1]), hv.1, hv.2.1⟩
              · have h0eq : s.reg 0 = some v := by rw [beq_iff_eq.1 h26] at hpo; exact hpo.1
                exact Or.inl ⟨v, (by rw [reg_set_mk_ne hd0, h0eq]), hv.1, hv.2.1⟩
            · split at hpre_ok
              · simp only [Bool.and_eq_true, beq_iff_eq, Bool.or_eq_true, bne_iff_ne, decide_eq_true_eq] at hpre_ok
                rcases hpre_ok with ⟨⟨hnx, _⟩, hd3⟩
                rw [hnx]
                by_cases h3 : d = 3
                · subst h3
                  have himm : imm % X86Env.W ≠ ACCEPT := by rcases hd3 with h | h; exact absurd rfl h; exact h
                  exact Or.inr ⟨imm % X86Env.W, (by rw [reg_set_mk hlen (by decide)]), Nat.mod_lt _ (by decide), himm⟩
                · exact Or.inr ⟨v, (by rw [reg_set_mk_ne h3, hregs0137.2.2.1]), hv.1, hv.2.1⟩
              · split at hpre_ok
                · next h77 =>
                  simp only [Bool.and_eq_true, beq_iff_eq, Bool.or_eq_true, bne_iff_ne] at hpre_ok
                  rw [hpre_ok.1]
                  intro a1 ha1
                  by_cases h7 : d = 7
                  · subst h7
                    rw [reg_set_mk hlen (by decide)] at ha1
                    injection ha1 with ha1; rw [← ha1]
                    have hmod : (imm % X86Env.W) % 256 = imm % 256 := Nat.mod_mod_of_dvd _ (by decide)
                    rw [hmod]
                    rcases hpre_ok.2 with h | h; exact absurd rfl h; exact h
                  · rw [reg_set_mk_ne h7] at ha1
                    exact h50 (by have := beq_iff_eq.1 h77; omega) a1 ha1
                · split at hpre_ok
                  · next h88 =>
                    simp only [Bool.and_eq_true, beq_iff_eq, Bool.or_eq_true, bne_iff_ne] at hpre_ok
                    rw [hpre_ok.1]
                    intro a1 ha1
                    by_cases h7 : d = 7
                    · subst h7
                      rw [reg_set_mk hlen (by decide)] at ha1
                      injection ha1 with ha1; rw [← ha1]
                      have hmod : (imm % X86Env.W) % 256 = imm % 256 := Nat.mod_mod_of_dvd _ (by decide)
                      rw [hmod]
                      rcases hpre_ok.2 with h | h; exact absurd rfl h; exact h
                    · rw [reg_set_mk_ne h7] at ha1
                      exact h50 (by have := beq_iff_eq.1 h88; omega) a1 ha1
                  · simp only [Bool.and_eq_true, List.mem_cons, List.not_mem_nil, or_false,
                      Bool.or_eq_true, bne_iff_ne, decide_eq_true_eq] at hpre_ok
                    rcases hpre_ok with ⟨⟨hoff6, hd7⟩, hnot28⟩
                    have hge2b : 0x2b ≤ off := by
                      rcases hoff6 with rfl | rfl | rfl | rfl | rfl | rfl <;> omega
                    have hexit : off + len = 0x7c ∨ off + len = 0x8d →
                        hasExitNonZero { s.set d (some (imm % X86Env.W)) with rip := base + off + len } := by
                      intro hnx_exit a1 ha1
                      by_cases h7 : d = 7
                      · exfalso
                        rcases hd7 with hd7 | ⟨⟨_, hne7c⟩, hne8d⟩
                        · exact hd7 h7
                        · rcases hnx_exit with h | h
                          · exact hne7c h
                          · exact hne8d h
                      · rw [reg_set_mk_ne h7] at ha1
                        exact h50 hge2b a1 ha1
                    by_cases h7c : off + len = 0x7c
                    · rw [h7c]; exact hexit (Or.inl h7c)
                    · by_cases h8d : off + len = 0x8d
                      · rw [h8d]; exact hexit (Or.inr h8d)
                      · apply tailOff_trivial_of_not_mem
                        intro hmem_nt
                        simp only [nonTrivialTailOffs, List.mem_cons, List.not_mem_nil, or_false] at hmem_nt
                        rcases hmem_nt with h | h | h | h | h | h | h | h | h | h | h | h | h | h | h | h | h | h | h | h | h | h | h | h | h | h | h | h | h | h
                        all_goals first | exact h7c h | exact h8d h | (apply hnot28; rw [h]; simp)
      | mov d s1 =>
        dsimp only at hpre_ok
        dsimp only [shiftIns, X86Int.exec, lift, bind, Option.bind, pure]
        cases hs1 : s.reg s1 with
        | none => exact trivial
        | some vs =>
          dsimp only
          split at hpre_ok
          · exact nxSafeGeneric_TailSafe (t := off + len) (set_length hlen _ _) (by dsimp only; rw [hsrip]; omega) (by assumption)
          · split at hpre_ok
            · next h12 =>
              simp only [Bool.and_eq_true, beq_iff_eq, Bool.or_eq_true, bne_iff_ne] at hpre_ok
              rcases hpre_ok with ⟨hnx, hcond⟩
              refine ⟨set_length hlen _ _, fun _ hhi' => ?_⟩
              dsimp only at hhi' ⊢
              rw [hsrip, show base + off + len - base = off + len by omega, hnx]
              have h0eq : s.reg 0 = some v := by rw [beq_iff_eq.1 h12] at hpo; exact hpo.1
              rcases hcond with hd0 | hs3
              · exact Or.inl ⟨v, (by rw [reg_set_mk_ne hd0, h0eq]), hv.1, hv.2.1⟩
              · subst hs3
                by_cases hd0 : d = 0
                · exact Or.inl ⟨vs, (by rw [hd0, reg_set_mk hlen (by decide)]), by rw [hregs0137.2.2.1] at hs1; cases hs1; exact ⟨hv.1, hv.2.1⟩⟩
                · exact Or.inl ⟨v, (by rw [reg_set_mk_ne hd0, h0eq]), hv.1, hv.2.1⟩
            · split at hpre_ok
              · next h15 =>
                simp only [Bool.and_eq_true, beq_iff_eq, Bool.or_eq_true, bne_iff_ne, List.mem_cons,
                  List.not_mem_nil, or_false, decide_eq_true_eq] at hpre_ok
                rcases hpre_ok with ⟨hnx, ⟨hd0, hd3⟩ | hs013⟩
                · refine ⟨set_length hlen _ _, fun _ hhi' => ?_⟩
                  dsimp only at hhi' ⊢
                  rw [hsrip, show base + off + len - base = off + len by omega, hnx]
                  exact Or.inr ⟨v, (by rw [reg_set_mk_ne hd3, hregs0137.2.2.1]), hv.1, hv.2.1⟩
                · refine ⟨set_length hlen _ _, fun _ hhi' => ?_⟩
                  dsimp only at hhi' ⊢
                  rw [hsrip, show base + off + len - base = off + len by omega, hnx]
                  have hvs : valNotAcc vs := by
                    rcases hs013 with heq | heq | heq
                    · rw [beq_iff_eq.1 h15] at hpo; rw [heq, hpo.1] at hs1; cases hs1; exact ⟨hv.1, hv.2.1⟩
                    · rw [heq, hregs0137.2.1] at hs1; cases hs1; exact ⟨hv.1, hv.2.1⟩
                    · rw [heq, hregs0137.2.2.1] at hs1; cases hs1; exact ⟨hv.1, hv.2.1⟩
                  by_cases hd3 : d = 3
                  · exact Or.inr ⟨vs, (by rw [hd3, reg_set_mk hlen (by decide)]), hvs⟩
                  · exact Or.inr ⟨v, (by rw [reg_set_mk_ne hd3, hregs0137.2.2.1]), hv.1, hv.2.1⟩
              · cases hpre_ok
      | mov32 d s1 =>
        dsimp only at hpre_ok
        dsimp only [shiftIns, X86Int.exec, lift, bind, Option.bind, pure]
        cases hs1 : s.reg s1 with
        | none => exact trivial
        | some vs =>
          dsimp only
          split at hpre_ok
          · exact nxSafeGeneric_TailSafe (t := off + len) (set_length hlen _ _) (by dsimp only; rw [hsrip]; omega) (by assumption)
          · split at hpre_ok
            · next h12 =>
              simp only [Bool.and_eq_true, beq_iff_eq, Bool.or_eq_true, bne_iff_ne] at hpre_ok
              rcases hpre_ok with ⟨hnx, hd0 | hs3⟩
              · refine ⟨set_length hlen _ _, fun _ hhi' => ?_⟩
                dsimp only at hhi' ⊢
                rw [hsrip, show base + off + len - base = off + len by omega, hnx]
                have h0eq : s.reg 0 = some v := by rw [beq_iff_eq.1 h12] at hpo; exact hpo.1
                exact Or.inl ⟨v, (by rw [reg_set_mk_ne hd0, h0eq]), hv.1, hv.2.1⟩
              · subst hs3
                refine ⟨set_length hlen _ _, fun _ hhi' => ?_⟩
                dsimp only at hhi' ⊢
                rw [hsrip, show base + off + len - base = off + len by omega, hnx]
                by_cases hd0 : d = 0
                · exact Or.inl ⟨vs % 2 ^ 32, (by rw [hd0, reg_set_mk hlen (by decide)]), valNotAcc_mod32 vs⟩
                · have h0eq : s.reg 0 = some v := by rw [beq_iff_eq.1 h12] at hpo; exact hpo.1
                  exact Or.inl ⟨v, (by rw [reg_set_mk_ne hd0, h0eq]), hv.1, hv.2.1⟩
            · split at hpre_ok
              · simp only [Bool.and_eq_true, beq_iff_eq, bne_iff_ne] at hpre_ok
                rcases hpre_ok with ⟨⟨hnx, hd0⟩, hd3⟩
                refine ⟨set_length hlen _ _, fun _ hhi' => ?_⟩
                dsimp only at hhi' ⊢
                rw [hsrip, show base + off + len - base = off + len by omega, hnx]
                exact Or.inr ⟨v, (by rw [reg_set_mk_ne hd3, hregs0137.2.2.1]), hv.1, hv.2.1⟩
              · cases hpre_ok
      | alu op d s1 =>
        dsimp only at hpre_ok
        dsimp only [shiftIns, X86Int.exec, lift, bind, Option.bind, pure]
        cases hd' : s.reg d with
        | none => exact trivial
        | some ad =>
          dsimp only
          cases hs1' : s.reg s1 with
          | none => exact trivial
          | some bs1 =>
            dsimp only
            split at hpre_ok
            · exact nxSafeGeneric_TailSafe (t := off + len) (set_length hlen _ _) (by dsimp only; rw [hsrip]; omega) (by assumption)
            · split at hpre_ok
              · next h00 =>
                simp only [Bool.and_eq_true, beq_iff_eq, Bool.or_eq_true] at hpre_ok
                rcases hpre_ok with ⟨⟨⟨hnx, rfl⟩, rfl⟩, hop⟩
                have h0eq : s.reg 0 = some v := by rw [beq_iff_eq.1 h00] at hpo; exact hpo.1
                rw [h0eq] at hd'; injection hd' with hd_v; subst hd_v
                rw [hregs0137.2.2.1] at hs1'; injection hs1' with hs_v; subst hs_v
                refine ⟨set_length hlen _ _, fun _ hhi' => ?_⟩
                dsimp only at hhi' ⊢
                rw [hsrip, show base + off + len - base = off + len by omega, hnx]
                have hres0 : (X86Int.aluF op v v).1 = 0 := by
                  rcases hop with rfl | rfl
                  · simp [X86Int.aluF, Nat.xor_self]
                  · simp [X86Int.aluF, Nat.add_sub_cancel_left]
                refine ⟨⟨0, (by rw [reg_set_mk hlen (by decide), hres0]), valNotAcc_0⟩,
                  ⟨v, (by rw [reg_set_mk_ne (by decide), hregs0137.2.2.1]), hv.1, hv.2.1⟩,
                  ⟨v, (by rw [reg_set_mk_ne (by decide), hregs0137.2.1]), hv.1, hv.2.1⟩⟩
              · split at hpre_ok
                · next h12 =>
                  simp only [Bool.and_eq_true, beq_iff_eq, bne_iff_ne] at hpre_ok
                  rcases hpre_ok with ⟨hnx, hd0⟩
                  refine ⟨set_length hlen _ _, fun _ hhi' => ?_⟩
                  dsimp only at hhi' ⊢
                  rw [hsrip, show base + off + len - base = off + len by omega, hnx]
                  have h0eq : s.reg 0 = some v := by rw [beq_iff_eq.1 h12] at hpo; exact hpo.1
                  exact Or.inl ⟨v, (by rw [reg_set_mk_ne hd0, h0eq]), hv.1, hv.2.1⟩
                · split at hpre_ok
                  · simp only [Bool.and_eq_true, beq_iff_eq, bne_iff_ne] at hpre_ok
                    rcases hpre_ok with ⟨⟨hnx, hd0⟩, hd3⟩
                    refine ⟨set_length hlen _ _, fun _ hhi' => ?_⟩
                    dsimp only at hhi' ⊢
                    rw [hsrip, show base + off + len - base = off + len by omega, hnx]
                    exact Or.inr ⟨v, (by rw [reg_set_mk_ne hd3, hregs0137.2.2.1]), hv.1, hv.2.1⟩
                  · split at hpre_ok
                    · next h35 =>
                      simp only [Bool.and_eq_true, beq_iff_eq, Bool.or_eq_true, bne_iff_ne] at hpre_ok
                      rcases hpre_ok with ⟨hnx, (⟨hd3, _⟩ | ⟨⟨rfl, rfl⟩, rfl⟩) | ⟨⟨rfl, rfl⟩, rfl⟩⟩
                      · refine ⟨set_length hlen _ _, fun _ hhi' => ?_⟩
                        dsimp only at hhi' ⊢
                        rw [hsrip, show base + off + len - base = off + len by omega, hnx]
                        exact Or.inr ⟨v, (by rw [reg_set_mk_ne hd3, hregs0137.2.2.1]), hv.1, hv.2.1⟩
                      · refine ⟨set_length hlen _ _, fun _ hhi' => ?_⟩
                        dsimp only at hhi' ⊢
                        rw [hsrip, show base + off + len - base = off + len by omega, hnx]
                        have h0eq : s.reg 0 = some v := by rw [beq_iff_eq.1 h35] at hpo; exact hpo.1
                        rw [hregs0137.2.2.1] at hd'; injection hd' with hd_v; subst hd_v
                        rw [h0eq] at hs1'; injection hs1' with hs_v; subst hs_v
                        exact Or.inr ⟨(v + v) % X86Env.W, (by rw [reg_set_mk hlen (by decide)]; rfl), SafeInvalid_double hv⟩
                      · refine ⟨set_length hlen _ _, fun _ hhi' => ?_⟩
                        dsimp only at hhi' ⊢
                        rw [hsrip, show base + off + len - base = off + len by omega, hnx]
                        exact Or.inr ⟨v, (by rw [reg_set_mk_ne (by decide), hregs0137.2.2.1]), hv.1, hv.2.1⟩
                    · split at hpre_ok
                      · have hnx : off + len = 0x6b := by simpa using hpre_ok
                        refine ⟨set_length hlen _ _, fun _ _ => ?_⟩
                        dsimp only; rw [hsrip, show base + off + len - base = off + len by omega, hnx]; trivial
                      · cases hpre_ok
      | aluI op d imm =>
        dsimp only at hpre_ok
        dsimp only [shiftIns, X86Int.exec, lift, bind, Option.bind, pure]
        cases s.reg d with
        | none => exact trivial
        | some ad =>
          dsimp only
          simp only [Bool.or_eq_true, decide_eq_true_eq] at hpre_ok
          rcases hpre_ok with hgen | hnot
          · exact nxSafeGeneric_TailSafe (set_length hlen _ _) (by dsimp only; rw [hsrip]; omega) hgen
          · refine ⟨set_length hlen _ _, fun _ _ => ?_⟩
            dsimp only; rw [hsrip, show base + off + len - base = off + len by omega]
            exact tailOff_trivial_of_not_mem hnot
      | shlI d c =>
        dsimp only at hpre_ok
        dsimp only [shiftIns, X86Int.exec, lift, bind, Option.bind, pure]
        cases s.reg d with
        | none => exact trivial
        | some ad =>
          dsimp only
          simp only [Bool.or_eq_true, decide_eq_true_eq] at hpre_ok
          rcases hpre_ok with hgen | hnot
          · exact nxSafeGeneric_TailSafe (set_length hlen _ _) (by dsimp only; rw [hsrip]; omega) hgen
          · refine ⟨set_length hlen _ _, fun _ _ => ?_⟩
            dsimp only; rw [hsrip, show base + off + len - base = off + len by omega]
            exact tailOff_trivial_of_not_mem hnot
      | shrI d c =>
        dsimp only at hpre_ok
        dsimp only [shiftIns, X86Int.exec, lift, bind, Option.bind, pure]
        cases s.reg d with
        | none => exact trivial
        | some ad =>
          dsimp only
          simp only [Bool.or_eq_true, decide_eq_true_eq] at hpre_ok
          rcases hpre_ok with hgen | hnot
          · exact nxSafeGeneric_TailSafe (set_length hlen _ _) (by dsimp only; rw [hsrip]; omega) hgen
          · refine ⟨set_length hlen _ _, fun _ _ => ?_⟩
            dsimp only; rw [hsrip, show base + off + len - base = off + len by omega]
            exact tailOff_trivial_of_not_mem hnot
      | lea d b1 disp =>
        dsimp only at hpre_ok
        dsimp only [shiftIns, X86Int.exec, lift, bind, Option.bind, pure]
        cases s.reg b1 with
        | none => exact trivial
        | some ab =>
          dsimp only
          simp only [Bool.or_eq_true, decide_eq_true_eq] at hpre_ok
          rcases hpre_ok with hgen | hnot
          · exact nxSafeGeneric_TailSafe (set_length hlen _ _) (by dsimp only; rw [hsrip]; omega) hgen
          · refine ⟨set_length hlen _ _, fun _ _ => ?_⟩
            dsimp only; rw [hsrip, show base + off + len - base = off + len by omega]
            exact tailOff_trivial_of_not_mem hnot

/-! ### Elementary physical faults, `X86Int.pstep`/`prun`, and 3-pass + TMRVoter single-fault theorems -/

/-- One architectural step or one elementary physical fault (`fetch`, `reg`, `mem`, `flag`). -/
inductive ElemFaultStep (base : Nat) (s : St) : Out → Prop where
  | clean : ElemFaultStep base s (X86Int.step (tailImAt base) s)
  | fetch {alt : Option (Ins × Nat)} :
      alt ∈ tailAltsAt base s.rip → ElemFaultStep base s (X86Int.execO s alt)
  | reg {s1 s2 : St} :
      X86Int.step (tailImAt base) s = .ok s1 → s2 ∈ X86Int.regFlips s1 →
      ElemFaultStep base s (.ok s2)
  | mem {s1 s2 : St} :
      X86Int.step (tailImAt base) s = .ok s1 → s2 ∈ X86Int.memFlips s1 →
      ElemFaultStep base s (.ok s2)
  | flag {s1 s2 : St} :
      X86Int.step (tailImAt base) s = .ok s1 → s2 ∈ X86Int.flagFlips s1 →
      ElemFaultStep base s (.ok s2)

def OutSafe (base : Nat) : Out → Prop
  | .ok s' => TailSafe base s'
  | .exit c => c ≠ 0
  | .stuck => True

theorem PreFault_elemFaultStep_safe {base v : Nat} (hb : base_ok base) {s : St}
    (hs : PreFault base v s) {o : Out} (hstep : ElemFaultStep base s o) :
    OutSafe base o := by
  cases hstep with
  | clean =>
    have hc := PreFault_step hb hs
    generalize ho : X86Int.step (tailImAt base) s = o0 at hc ⊢
    cases o0 with
    | ok s' => exact PreFault_to_TailSafe hc
    | exit c => exact hc
    | stuck => exact trivial
  | fetch halt =>
    have hf := PreFault_fetch_safe hb hs halt
    cases h : X86Int.execO s _ with
    | ok s' => rw [h] at hf; exact hf
    | exit c => rw [h] at hf; exact hf
    | stuck => exact trivial
  | reg hclean hreg =>
    have hc := PreFault_step hb hs
    rw [hclean] at hc
    exact PreFault_regFlips_safe hc hreg
  | mem hclean hmem =>
    have hc := PreFault_step hb hs
    rw [hclean] at hc
    exact PreFault_memFlips_safe hc hmem
  | flag hclean hfl =>
    have hc := PreFault_step hb hs
    rw [hclean] at hc
    exact PreFault_flagFlips_safe hc hfl

/-- Arbitrary `n`-step run with at most 1 elementary physical fault total across all steps. -/
inductive OneFaultRun (base : Nat) : Bool → Nat → Out → Out → Prop where
  | zero (faulted : Bool) (o : Out) : OneFaultRun base faulted 0 o o
  | step_exit (faulted : Bool) (n c : Nat) :
      OneFaultRun base faulted (n + 1) (.exit c) (.exit c)
  | step_stuck (faulted : Bool) (n : Nat) :
      OneFaultRun base faulted (n + 1) .stuck .stuck
  | step_clean (faulted : Bool) (n : Nat) (s : St) (o : Out) :
      OneFaultRun base faulted n (X86Int.step (tailImAt base) s) o →
      OneFaultRun base faulted (n + 1) (.ok s) o
  | step_fault (n : Nat) (s : St) (o1 o2 : Out) :
      ElemFaultStep base s o1 →
      OneFaultRun base true n o1 o2 →
      OneFaultRun base false (n + 1) (.ok s) o2

private theorem TailSafe_OneFaultRun_clean {base : Nat} (hb : base_ok base) :
    ∀ {f : Bool} {n : Nat} {o1 o2 : Out},
      f = true → OneFaultRun base f n o1 o2 → OutSafe base o1 → o2 ≠ .exit 0 := by
  intro f n o1 o2 hf hrun hs1
  induction hrun with
  | zero f0 o =>
    cases o with
    | ok _ => intro h; cases h
    | exit c => intro h; cases h; exact hs1 rfl
    | stuck => intro h; cases h
  | step_exit => intro h; cases h; exact hs1 rfl
  | step_stuck => intro h; cases h
  | step_clean faulted m s' o _ ih =>
    apply ih hf
    have hstep := TailSafe_step hb hs1
    generalize ho : X86Int.step (tailImAt base) s' = o0 at hstep ⊢
    cases o0 <;> exact hstep
  | step_fault => cases hf

theorem PreFault_OneFaultRun {base v : Nat} (hb : base_ok base) :
    ∀ {faulted : Bool} {n : Nat} {o0 o : Out},
      OneFaultRun base faulted n o0 o → ∀ s, o0 = .ok s → PreFault base v s → o ≠ .exit 0 := by
  intro faulted n o0 o hrun
  induction hrun with
  | zero _ _ => intro s h _; cases h; intro h1; cases h1
  | step_exit => intro s h; cases h
  | step_stuck => intro s h; cases h
  | step_clean faulted m s' o hnext ih =>
    intro s ho hs
    cases ho
    have hc := PreFault_step hb hs
    cases hst : X86Int.step (tailImAt base) s' with
    | ok s1 => rw [hst] at hc; exact ih s1 hst hc
    | exit c =>
      rw [hst] at hc hnext
      have : o = .exit c := by
        cases hnext with
        | zero => rfl
        | step_exit => rfl
      rw [this]; intro h; cases h; exact hc rfl
    | stuck =>
      rw [hst] at hnext
      have : o = .stuck := by
        cases hnext with
        | zero => rfl
        | step_stuck => rfl
      rw [this]; intro h; cases h
  | step_fault m s' o1 o2 helem hrest =>
    intro s ho hs
    cases ho
    exact TailSafe_OneFaultRun_clean hb rfl hrest (PreFault_elemFaultStep_safe hb hs helem)

theorem tail_oneFaultRun_no_false_accept {base v : Nat} (hb : base_ok base) (s0 : St)
    (hst : TailStackSafe base s0) (hv : SafeInvalid v)
    {n : Nat} {o : Out} (hrun : OneFaultRun base false n (.ok (tailInit base s0 v v v)) o) :
    o ≠ .exit 0 :=
  PreFault_OneFaultRun hb hrun _ rfl (PreFault_init hb s0 hst hv)

/-! ### Direct `X86Int.pstep` / `X86Int.prun` distribution bridge (`SDC_accept_invalid = 0`) -/

def exit0Prob : PDist.PDF Out → Rat
  | [] => 0
  | ((o, _), w) :: rest => (match o with | .exit 0 => w | _ => 0) + exit0Prob rest

theorem exit0Prob_eq_zero_of_forall {d : PDist.PDF Out}
    (h0 : ∀ x ∈ d, 0 ≤ x.2) (hne : ∀ x ∈ d, 0 < x.2 → x.1.1 ≠ .exit 0) :
    exit0Prob d = 0 := by
  induction d with
  | nil => rfl
  | cons x rest ih =>
    rcases x with ⟨⟨o, t⟩, w⟩
    have hw0 : 0 ≤ w := h0 ((o, t), w) (List.mem_cons_self ..)
    have ih' := ih (fun y hy => h0 y (List.mem_cons_of_mem _ hy))
      (fun y hy => hne y (List.mem_cons_of_mem _ hy))
    simp only [exit0Prob, ih', Rat.add_zero]
    by_cases hw_pos : w ≤ 0
    · have hw_eq : w = 0 := Rat.le_antisymm hw_pos hw0
      rw [hw_eq]
      cases o with
      | ok _ => rfl
      | stuck => rfl
      | exit c => cases c <;> rfl
    · have hlt : 0 < w := Rat.not_le.1 hw_pos
      have ho : o ≠ .exit 0 := hne ((o, t), w) (List.mem_cons_self ..) hlt
      cases o with
      | ok _ => rfl
      | stuck => rfl
      | exit c => cases c with
        | zero => exact absurd rfl ho
        | succ _ => rfl

def singleMechanism (r : X86Int.Rates) : Prop :=
  (r.reg = 0 ∧ r.mem = 0 ∧ r.flag = 0) ∨
  (r.fetch = 0 ∧ r.mem = 0 ∧ r.flag = 0) ∨
  (r.fetch = 0 ∧ r.reg = 0 ∧ r.flag = 0) ∨
  (r.fetch = 0 ∧ r.reg = 0 ∧ r.mem = 0)

private theorem rat_pos_mul_left {a b : Rat} (ha : 0 ≤ a) (_hb : 0 ≤ b) (h : 0 < a * b) : 0 < a := by
  by_cases hle : a ≤ 0
  · exfalso; have := Rat.le_antisymm hle ha; rw [this, Rat.zero_mul] at h; exact Rat.lt_irrefl h
  · exact Rat.not_le.1 hle

private theorem rat_pos_mul_right {a b : Rat} (_ha : 0 ≤ a) (hb : 0 ≤ b) (h : 0 < a * b) : 0 < b := by
  by_cases hle : b ≤ 0
  · exfalso; have := Rat.le_antisymm hle hb; rw [this, Rat.mul_zero] at h; exact Rat.lt_irrefl h
  · exact Rat.not_le.1 hle

private theorem mem_pureF_iff {α : Type} {a : α} {y : (α × Bool) × Rat} :
    y ∈ PDist.pureF a ↔ y = ((a, false), 1) := by
  simp [PDist.pureF]

private theorem mem_unif_pos {β : Type} {p : Rat} (hp : PDist.Rate p) {c : β} {xs : List β}
    {y : (β × Bool) × Rat} (hy : y ∈ X86Int.unif p c xs) (hpos : 0 < y.2) :
    y.1 = (c, false) ∨ (0 < p ∧ y.1.2 = true ∧ y.1.1 ∈ xs) := by
  cases xs with
  | nil =>
    simp only [X86Int.unif, PDist.pureF, List.mem_cons, List.not_mem_nil, or_false] at hy
    subst hy; exact Or.inl rfl
  | cons x r =>
    simp only [X86Int.unif, List.mem_cons, List.mem_map] at hy
    rcases hy with rfl | ⟨z, hz, rfl⟩
    · exact Or.inl rfl
    · refine Or.inr ⟨?_, rfl, ?_⟩
      · by_cases hp0 : p ≤ 0
        · exfalso
          have hp_eq : p = 0 := Rat.le_antisymm hp0 hp.1
          rw [hp_eq] at hpos
          simp [Rat.div_def] at hpos
        · exact Rat.not_le.1 hp0
      · simpa using hz

private theorem mem_relabel_pos {α β : Type} {g : α → β} {d : PDist.PDF α}
    {y : (β × Bool) × Rat} (hy : y ∈ X86Int.relabel g d) :
    ∃ x ∈ d, y = ((g x.1.1, x.1.2), x.2) := by
  simp only [X86Int.relabel, List.mem_map] at hy
  rcases hy with ⟨x, hx, rfl⟩
  exact ⟨x, hx, rfl⟩

private theorem mem_bindF_pos {α β : Type} {d : PDist.PDF α} {f : α → PDist.PDF β}
    (hd : ∀ x ∈ d, 0 ≤ x.2) (hf : ∀ a z, z ∈ f a → 0 ≤ z.2) {y : (β × Bool) × Rat}
    (hy : y ∈ PDist.bindF d f) (hpos : 0 < y.2) :
    ∃ x ∈ d, ∃ z ∈ f x.1.1,
      0 < x.2 ∧ 0 < z.2 ∧ y = ((z.1.1, x.1.2 || z.1.2), x.2 * z.2) := by
  induction d with
  | nil => simp [PDist.bindF] at hy
  | cons x rest ih =>
    rcases x with ⟨⟨a, tx⟩, wx⟩
    have hwx : 0 ≤ wx := hd ((a, tx), wx) (List.mem_cons_self ..)
    simp only [PDist.bindF, List.mem_append, PDist.tagW, List.mem_map] at hy
    rcases hy with ⟨z, hz, rfl⟩ | hrest
    · have hwz : 0 ≤ z.2 := hf a z hz
      exact ⟨((a, tx), wx), List.mem_cons_self .., z, hz,
        rat_pos_mul_left hwx hwz hpos, rat_pos_mul_right hwx hwz hpos, rfl⟩
    · rcases ih (fun u hu => hd u (List.mem_cons_of_mem _ hu)) hrest with ⟨u, hu, z, hz, hwu, hwz, heq⟩
      exact ⟨u, List.mem_cons_of_mem _ hu, z, hz, hwu, hwz, heq⟩

private theorem pstep_none_zero_pos {alts : X86Int.Alts} {im : Nat → Option (Ins × Nat)}
    {s : St} {y : (Out × Bool) × Rat}
    (hy : y ∈ X86Int.pstep alts .none 0 im s) (hpos : 0 < y.2) :
    y.1.1 = X86Int.step im s := by
  have h0 : PDist.Rate (0 : Rat) := ⟨Rat.le_refl, by decide⟩
  have hf_ok := X86Int.fetched_ok h0 alts im s
  rcases mem_bindF_pos hf_ok.1 (fun o _ ho => (X86Int.retire_ok .none o).1 _ ho) hy hpos with
    ⟨⟨⟨of, tf⟩, wf⟩, hof, ⟨⟨or_, tr⟩, wr⟩, hor, hwf, _, rfl⟩
  dsimp only at hor hwf ⊢
  have heq_f : of = X86Int.step im s := by
    rw [X86Int.step]
    cases him : im s.rip with
    | none =>
      rw [him] at hof; rw [mem_pureF_iff] at hof; exact congrArg (·.1.1) hof
    | some p =>
      rcases p with ⟨ins, len⟩
      rw [him] at hof
      rcases mem_unif_pos h0 hof hwf with h1 | ⟨hlt, _⟩
      · exact congrArg (·.1) h1
      · exact absurd hlt (Rat.lt_irrefl)
  rw [heq_f] at hor
  cases hstep : X86Int.step im s with
  | stuck =>
    rw [hstep] at hor; rw [mem_pureF_iff] at hor; exact congrArg (·.1.1) hor
  | exit c =>
    rw [hstep] at hor; rw [mem_pureF_iff] at hor; exact congrArg (·.1.1) hor
  | ok s1 =>
    rw [hstep] at hor
    rcases mem_relabel_pos hor with ⟨⟨⟨sr, tr'⟩, wr'⟩, hhit, hrel⟩
    simp only [X86Int.Fault.none, mem_pureF_iff] at hhit
    cases hhit; cases hrel; rfl

private theorem prun_none_zero_pos {alts : X86Int.Alts} {im : Nat → Option (Ins × Nat)} :
    ∀ n o {y : (Out × Bool) × Rat},
      y ∈ X86Int.prun alts .none 0 im n o → 0 < y.2 → y.1.1 = X86Int.run im n o
  | 0, o, y, hy, _ => by
    rw [X86Int.prun, mem_pureF_iff] at hy; subst hy; rfl
  | n + 1, .exit c, y, hy, _ => by
    dsimp only [X86Int.prun] at hy; rw [mem_pureF_iff] at hy; subst hy; rfl
  | n + 1, .stuck, y, hy, _ => by
    dsimp only [X86Int.prun] at hy; rw [mem_pureF_iff] at hy; subst hy; rfl
  | n + 1, .ok s, y, hy, hpos => by
    have h0 : PDist.Rate (0 : Rat) := ⟨Rat.le_refl, by decide⟩
    rcases mem_bindF_pos (X86Int.pstep_ok alts .none h0 im s).1
      (fun o _ ho => (X86Int.prun_ok alts .none h0 im n o).1 _ ho) hy hpos with
      ⟨x, hx, z, hz, hwx, hwz, rfl⟩
    have hx_eq := pstep_none_zero_pos hx hwx
    have hz_eq := prun_none_zero_pos n x.1.1 hz hwz
    rw [hx_eq] at hz_eq
    exact hz_eq

private theorem pstep_singleMechanism_elem {base : Nat} {r : X86Int.Rates} (hr : r.ok)
    (hsing : singleMechanism r) {s : St} {y : (Out × Bool) × Rat}
    (hy : y ∈ X86Int.pstep (tailAltsAt base) (r.state hr) r.fetch (tailImAt base) s)
    (hpos : 0 < y.2) :
    ElemFaultStep base s y.1.1 := by
  have hf_ok := X86Int.fetched_ok hr.1 (tailAltsAt base) (tailImAt base) s
  rcases mem_bindF_pos hf_ok.1 (fun o _ ho => (X86Int.retire_ok (r.state hr) o).1 _ ho) hy hpos with
    ⟨⟨⟨of, tf⟩, wf⟩, hof, ⟨⟨or_, tr⟩, wr⟩, hor, hwf, hwr, rfl⟩
  dsimp only at hof hor hwf hwr ⊢
  have hof_elem : of = X86Int.step (tailImAt base) s ∨
      (0 < r.fetch ∧ ∃ alt ∈ tailAltsAt base s.rip, of = X86Int.execO s alt) := by
    cases him : tailImAt base s.rip with
    | none =>
      rw [him] at hof; rw [mem_pureF_iff] at hof
      have hof' : of = .stuck := congrArg (·.1.1) hof
      exact Or.inl (by rw [hof']; simp [X86Int.step, him])
    | some p =>
      rcases p with ⟨ins, len⟩
      rw [him] at hof
      rcases mem_unif_pos hr.1 hof hwf with h1 | ⟨hpf_pos, _, hmap⟩
      · have h1' : of = X86Int.execO s (some (ins, len)) := congrArg (·.1) h1
        exact Or.inl (by rw [h1']; simp [X86Int.step, him, X86Int.execO])
      · simp only [List.mem_map] at hmap
        rcases hmap with ⟨alt, halt, halto⟩
        exact Or.inr ⟨hpf_pos, alt, halt, halto.symm⟩
  cases of with
  | stuck =>
    rw [mem_pureF_iff] at hor; cases hor
    rcases hof_elem with heq | ⟨_, alt, halt, halto⟩
    · rw [heq]; exact ElemFaultStep.clean
    · rw [halto]; exact ElemFaultStep.fetch halt
  | exit c =>
    rw [mem_pureF_iff] at hor; cases hor
    rcases hof_elem with heq | ⟨_, alt, halt, halto⟩
    · rw [heq]; exact ElemFaultStep.clean
    · rw [halto]; exact ElemFaultStep.fetch halt
  | ok s1 =>
    rcases mem_relabel_pos hor with ⟨⟨⟨sr, tr'⟩, wr'⟩, hhit, hrel⟩
    cases hrel; dsimp only
    dsimp only [X86Int.Rates.state, X86Int.Fault.seq, X86Int.Fault.uniform] at hhit
    have hreg_ok := fun t => X86Int.unif_ok hr.2.1 t (X86Int.regFlips t)
    have hmem_ok := fun t => X86Int.unif_ok hr.2.2.1 t (X86Int.memFlips t)
    have hfl_ok := fun t => X86Int.unif_ok hr.2.2.2 t (X86Int.flagFlips t)
    have hmf_ok : ∀ t, (PDist.bindF (X86Int.unif r.mem t (X86Int.memFlips t))
        (fun x => X86Int.unif r.flag x (X86Int.flagFlips x))).ok :=
      fun t => PDist.bindF_ok (hmem_ok t) hfl_ok
    rcases mem_bindF_pos (hreg_ok s1).1 (fun a z hz => (hmf_ok a).1 z hz) hhit hwr with
      ⟨⟨⟨s2, t2⟩, w2⟩, hs2, ⟨⟨s4, t4⟩, w4⟩, hs4, hw2, hw4, heq4⟩
    rcases mem_bindF_pos (hmem_ok s2).1 (fun a z hz => (hfl_ok a).1 z hz) hs4 hw4 with
      ⟨⟨⟨s3, t3⟩, w3⟩, hs3, ⟨⟨s4', t4'⟩, w4'⟩, hs4', hw3, hw4', heq4'⟩
    cases heq4; cases heq4'; dsimp only at hs2 hs3 hs4' hw2 hw3 hw4' ⊢
    have hs2_cases := mem_unif_pos hr.2.1 hs2 hw2
    have hs3_cases := mem_unif_pos hr.2.2.1 hs3 hw3
    have hs4_cases := mem_unif_pos hr.2.2.2 hs4' hw4'
    rcases hsing with ⟨hreg0, hmem0, hfl0⟩ | ⟨hft0, hmem0, hfl0⟩ | ⟨hft0, hreg0, hfl0⟩ | ⟨hft0, hreg0, hmem0⟩
    · have hs2_eq : s2 = s1 := by
        rcases hs2_cases with h | ⟨hlt, _⟩; exact congrArg (·.1) h; rw [hreg0] at hlt; exact absurd hlt Rat.lt_irrefl
      have hs3_eq : s3 = s2 := by
        rcases hs3_cases with h | ⟨hlt, _⟩; exact congrArg (·.1) h; rw [hmem0] at hlt; exact absurd hlt Rat.lt_irrefl
      have hs4_eq : s4' = s3 := by
        rcases hs4_cases with h | ⟨hlt, _⟩; exact congrArg (·.1) h; rw [hfl0] at hlt; exact absurd hlt Rat.lt_irrefl
      rw [hs4_eq, hs3_eq, hs2_eq]
      rcases hof_elem with heq | ⟨_, alt, halt, halto⟩
      · rw [heq]; exact ElemFaultStep.clean
      · rw [halto]; exact ElemFaultStep.fetch halt
    · have hclean1 : X86Int.step (tailImAt base) s = .ok s1 := by
        rcases hof_elem with heq | ⟨hlt, _⟩; exact heq.symm; rw [hft0] at hlt; exact absurd hlt Rat.lt_irrefl
      have hs3_eq : s3 = s2 := by
        rcases hs3_cases with h | ⟨hlt, _⟩; exact congrArg (·.1) h; rw [hmem0] at hlt; exact absurd hlt Rat.lt_irrefl
      have hs4_eq : s4' = s3 := by
        rcases hs4_cases with h | ⟨hlt, _⟩; exact congrArg (·.1) h; rw [hfl0] at hlt; exact absurd hlt Rat.lt_irrefl
      rw [hs4_eq, hs3_eq]
      rcases hs2_cases with h | ⟨_, _, hmem_reg⟩
      · have hs2_eq : s2 = s1 := congrArg (·.1) h
        rw [hs2_eq, ← hclean1]; exact ElemFaultStep.clean
      · exact ElemFaultStep.reg hclean1 hmem_reg
    · have hclean1 : X86Int.step (tailImAt base) s = .ok s1 := by
        rcases hof_elem with heq | ⟨hlt, _⟩; exact heq.symm; rw [hft0] at hlt; exact absurd hlt Rat.lt_irrefl
      have hs2_eq : s2 = s1 := by
        rcases hs2_cases with h | ⟨hlt, _⟩; exact congrArg (·.1) h; rw [hreg0] at hlt; exact absurd hlt Rat.lt_irrefl
      have hs4_eq : s4' = s3 := by
        rcases hs4_cases with h | ⟨hlt, _⟩; exact congrArg (·.1) h; rw [hfl0] at hlt; exact absurd hlt Rat.lt_irrefl
      rw [hs4_eq]; rw [hs2_eq] at hs3_cases
      rcases hs3_cases with h | ⟨_, _, hmem_m⟩
      · have hs3_eq : s3 = s1 := congrArg (·.1) h
        rw [hs3_eq, ← hclean1]; exact ElemFaultStep.clean
      · exact ElemFaultStep.mem hclean1 hmem_m
    · have hclean1 : X86Int.step (tailImAt base) s = .ok s1 := by
        rcases hof_elem with heq | ⟨hlt, _⟩; exact heq.symm; rw [hft0] at hlt; exact absurd hlt Rat.lt_irrefl
      have hs2_eq : s2 = s1 := by
        rcases hs2_cases with h | ⟨hlt, _⟩; exact congrArg (·.1) h; rw [hreg0] at hlt; exact absurd hlt Rat.lt_irrefl
      have hs3_eq : s3 = s2 := by
        rcases hs3_cases with h | ⟨hlt, _⟩; exact congrArg (·.1) h; rw [hmem0] at hlt; exact absurd hlt Rat.lt_irrefl
      rw [hs3_eq, hs2_eq] at hs4_cases
      rcases hs4_cases with h | ⟨_, _, hmem_fl⟩
      · have hs4_eq : s4' = s1 := congrArg (·.1) h
        rw [hs4_eq, ← hclean1]; exact ElemFaultStep.clean
      · exact ElemFaultStep.flag hclean1 hmem_fl

private theorem PreFault_run {base v : Nat} (hb : base_ok base) :
    ∀ (k : Nat) {s : St}, PreFault base v s →
      match X86Int.run (tailImAt base) k (.ok s) with
      | .ok s' => PreFault base v s'
      | .exit c => c ≠ 0
      | .stuck => True
  | 0, s, hs => hs
  | k + 1, s, hs => by
    have hc := PreFault_step hb hs
    simp only [X86Int.run]
    cases hst : X86Int.step (tailImAt base) s with
    | stuck => rw [run_stuck]; exact trivial
    | exit c => rw [hst] at hc; rw [run_exit]; exact hc
    | ok s1 => rw [hst] at hc; exact PreFault_run hb k hc

/-- Physical execution of `tailCode` over `X86Int.pstep` and `X86Int.prun` with one faulty step at step `k`. -/
def tailPRunWithFaultAt (base : Nat) (r : X86Int.Rates) (hr : r.ok) (k m : Nat) (o : Out) : PDist.PDF Out :=
  PDist.bindF (X86Int.prun (tailAltsAt base) .none 0 (tailImAt base) k o) fun o1 =>
    PDist.bindF
      (match o1 with
       | .ok s1 => X86Int.pstep (tailAltsAt base) (r.state hr) r.fetch (tailImAt base) s1
       | o' => PDist.pureF o')
      (X86Int.prun (tailAltsAt base) .none 0 (tailImAt base) m)

theorem tailPRunWithFaultAt_ok (base : Nat) (r : X86Int.Rates) (hr : r.ok) (k m : Nat) (o : Out) :
    (tailPRunWithFaultAt base r hr k m o).ok := by
  have h0 : PDist.Rate (0 : Rat) := ⟨Rat.le_refl, by decide⟩
  refine PDist.bindF_ok (X86Int.prun_ok (tailAltsAt base) .none h0 (tailImAt base) k o) ?_
  intro o1
  refine PDist.bindF_ok ?_ (X86Int.prun_ok (tailAltsAt base) .none h0 (tailImAt base) m)
  cases o1 with
  | ok s1 => exact X86Int.pstep_ok (tailAltsAt base) (r.state hr) hr.1 (tailImAt base) s1
  | exit c => exact PDist.pureF_ok _
  | stuck => exact PDist.pureF_ok _

theorem tailPRunWithFaultAt_pos_ne_exit0 {base v : Nat} (hb : base_ok base) (s0 : St)
    (hst : TailStackSafe base s0) (hv : SafeInvalid v)
    (r : X86Int.Rates) (hr : r.ok) (hsing : singleMechanism r) (k m : Nat)
    {y : (Out × Bool) × Rat}
    (hy : y ∈ tailPRunWithFaultAt base r hr k m (.ok (tailInit base s0 v v v)))
    (hpos : 0 < y.2) :
    y.1.1 ≠ .exit 0 := by
  have h0 : PDist.Rate (0 : Rat) := ⟨Rat.le_refl, by decide⟩
  have hmid_ok : ∀ o1 : Out, (PDist.bindF
      (match o1 with
       | .ok s1 => X86Int.pstep (tailAltsAt base) (r.state hr) r.fetch (tailImAt base) s1
       | o' => PDist.pureF o')
      (X86Int.prun (tailAltsAt base) .none 0 (tailImAt base) m)).ok := fun o1 =>
    PDist.bindF_ok (by cases o1; exact X86Int.pstep_ok _ _ hr.1 _ _; exact PDist.pureF_ok _; exact PDist.pureF_ok _)
      (X86Int.prun_ok _ .none h0 _ m)
  rcases mem_bindF_pos (X86Int.prun_ok (tailAltsAt base) .none h0 (tailImAt base) k _).1
    (fun o1 z hz => (hmid_ok o1).1 z hz) hy hpos with
    ⟨x1, hx1, z2, hz2, hwx1, hwz2, rfl⟩
  have hx1_eq := prun_none_zero_pos k (.ok (tailInit base s0 v v v)) hx1 hwx1
  have hpf_k := PreFault_run hb k (PreFault_init hb s0 hst hv)
  rcases mem_bindF_pos (by cases x1.1.1; exact (X86Int.pstep_ok _ _ hr.1 _ _).1; exact (PDist.pureF_ok _).1; exact (PDist.pureF_ok _).1)
    (fun o2 z hz => (X86Int.prun_ok (tailAltsAt base) .none h0 (tailImAt base) m o2).1 z hz) hz2 hwz2 with
    ⟨x2, hx2, z3, hz3, hwx2, hwz3, heq3⟩
  have hz3_eq := prun_none_zero_pos m x2.1.1 hz3 hwz3
  have helem_safe : match x2.1.1 with
      | .ok s' => TailSafe base s'
      | .exit c => c ≠ 0
      | .stuck => True := by
    rw [hx1_eq] at hx2
    cases hrun_k : X86Int.run (tailImAt base) k (.ok (tailInit base s0 v v v)) with
    | stuck =>
      rw [hrun_k, mem_pureF_iff] at hx2; rw [(congrArg (·.1.1) hx2)]; exact trivial
    | exit c =>
      rw [hrun_k] at hpf_k hx2; rw [mem_pureF_iff] at hx2; rw [(congrArg (·.1.1) hx2)]; exact hpf_k
    | ok s1 =>
      rw [hrun_k] at hpf_k hx2
      exact PreFault_elemFaultStep_safe hb hpf_k (pstep_singleMechanism_elem hr hsing hx2 hwx2)
  rw [heq3]; dsimp only; rw [hz3_eq]
  cases hx2o : x2.1.1 with
  | stuck => rw [hx2o] at helem_safe; rw [run_stuck]; exact fun h => by cases h
  | exit c => rw [hx2o] at helem_safe; rw [run_exit]; intro h; cases h; exact helem_safe rfl
  | ok s' => rw [hx2o] at helem_safe; exact TailSafe_run hb m helem_safe

theorem tailPRunWithFaultAt_exit0Prob_zero {base v : Nat} (hb : base_ok base) (s0 : St)
    (hst : TailStackSafe base s0) (hv : SafeInvalid v)
    (r : X86Int.Rates) (hr : r.ok) (hsing : singleMechanism r) (k m : Nat) :
    exit0Prob (tailPRunWithFaultAt base r hr k m (.ok (tailInit base s0 v v v))) = 0 :=
  exit0Prob_eq_zero_of_forall (tailPRunWithFaultAt_ok base r hr k m _).1
    (fun _ hy hpos => tailPRunWithFaultAt_pos_ne_exit0 hb s0 hst hv r hr hsing k m hy hpos)

/-! ### 3-pass + `TMRVoter` + dual-`ACCEPT`-guard single-fault composition -/

def NoPassFaultTriple (w : Nat) (t : Nat × Nat × Nat) : Prop :=
  t.1 = w ∧ t.2.1 = w ∧ t.2.2 = w

def SinglePassFaultTriple (w : Nat) (t : Nat × Nat × Nat) : Prop :=
  t.1 < X86Env.W ∧ t.2.1 < X86Env.W ∧ t.2.2 < X86Env.W ∧
  ((t.1 = w ∧ t.2.1 = w) ∨ (t.1 = w ∧ t.2.2 = w) ∨ (t.2.1 = w ∧ t.2.2 = w))

theorem SinglePassFaultTriple_vote {w : Nat} {a b c : Nat}
    (h : SinglePassFaultTriple w (a, b, c)) :
    TMR.vote a b c = some w := by
  dsimp only [SinglePassFaultTriple] at h
  rcases h with ⟨_, _, _, ⟨rfl, rfl⟩ | ⟨rfl, rfl⟩ | ⟨hb, hc⟩⟩
  · unfold TMR.vote; simp
  · unfold TMR.vote; simp
  · unfold TMR.vote; rw [hb, hc]; by_cases haw : a = w
    · subst haw; simp
    · simp [haw]

private theorem run_never_exit0_of_eventually_nz {im} {s : St} {N c : Nat}
    (hc : c ≠ 0) (hevent : X86Int.run im N (.ok s) = .exit c) :
    ∀ n, X86Int.run im n (.ok s) ≠ .exit 0 := by
  intro n hn
  by_cases hle : n ≤ N
  · have := run_ge_exit hn hle; rw [hevent] at this; cases this; exact hc rfl
  · have := run_ge_exit hevent (Nat.le_of_not_le hle); rw [hn] at this; cases this; exact hc rfl

theorem invalidWord_threePass_no_single_fault_false_accept {base w : Nat} (hb : base_ok base)
    (s0 : St) (hst : TailStackSafe base s0) (hw : SafeInvalid w) :
    (∀ {a b c : Nat}, SinglePassFaultTriple w (a, b, c) →
      ∀ n, X86Int.run (tailImAt base) n (.ok (tailInit base s0 a b c)) ≠ .exit 0) ∧
    (∀ {n : Nat} {o : Out},
      OneFaultRun base false n (.ok (tailInit base s0 w w w)) o → o ≠ .exit 0) ∧
    (∀ (r : X86Int.Rates) (hr : r.ok), singleMechanism r → ∀ k m,
      exit0Prob (tailPRunWithFaultAt base r hr k m (.ok (tailInit base s0 w w w))) = 0 ∧
      ∀ y ∈ tailPRunWithFaultAt base r hr k m (.ok (tailInit base s0 w w w)),
        0 < y.2 → y.1.1 ≠ .exit 0) := by
  refine ⟨?_, ?_, ?_⟩
  · intro a b c htri n
    have ha := htri.1
    have hbv := htri.2.1
    have hc := htri.2.2.1
    have hvote := SinglePassFaultTriple_vote htri
    by_cases how : w = OVERFLOW
    · subst how
      exact run_never_exit0_of_eventually_nz (by decide)
        (tail_arch_overflow hb s0 ha hbv hc hvote 22 (Nat.le_refl _)) n
    · exact run_never_exit0_of_eventually_nz (by decide)
        (tail_arch_reject hb s0 ha hbv hc hvote hw.2.1 how 22 (Nat.le_refl _)) n
  · intro n o hrun
    exact tail_oneFaultRun_no_false_accept hb s0 hst hw hrun
  · intro r hr hsing k m
    exact ⟨tailPRunWithFaultAt_exit0Prob_zero hb s0 hst hw r hr hsing k m,
      fun y hy hpos => tailPRunWithFaultAt_pos_ne_exit0 hb s0 hst hw r hr hsing k m hy hpos⟩

end TailProofs

/-! ### Ground `decide +kernel` witnesses for valid and invalid `SPR1` images -/

theorem checkFlat_examples_valid :
    checkFlat Sporos.echo.enc = some true ∧
    checkFlat Sporos.Examples.hello.enc = some true ∧
    checkFlat Sporos.Examples.upper.enc = some true ∧
    checkFlat Sporos.Examples.fib.enc = some true := by
  decide +kernel

theorem checkFlat_echo_valid : checkFlat Sporos.echo.enc = some true := checkFlat_examples_valid.1
theorem checkFlat_hello_valid : checkFlat Sporos.Examples.hello.enc = some true := checkFlat_examples_valid.2.1
theorem checkFlat_upper_valid : checkFlat Sporos.Examples.upper.enc = some true := checkFlat_examples_valid.2.2.1
theorem checkFlat_fib_valid : checkFlat Sporos.Examples.fib.enc = some true := checkFlat_examples_valid.2.2.2

theorem checkFlat_refusal_witnesses :
    checkFlat [] = some false ∧
    checkFlat [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] = some false ∧
    checkFlat (Sporos.Prog.enc { words := [{ ins := 1, outs := 0, locals := 0, code := [.ret], ann := [] }], data := [], main := 0 }) = some false ∧
    checkFlat (Sporos.Prog.enc { words := [{ ins := 0, outs := 0, locals := 0, code := [.prim .add, .ret], ann := [] }], data := [], main := 0 }) = some false ∧
    checkFlat (Sporos.Prog.enc { words := [{ ins := 0, outs := 0, locals := 0, code := [.ret], ann := [] }], data := [], main := 1 }) = some false ∧
    checkFlat (Sporos.echo.enc ++ [0]) = some false ∧
    checkFlat (Sporos.echo.enc.take 16) = some false ∧
    checkFlat
      [ 0x53, 0x50, 0x52, 0x31, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0
      , 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0xff ] = some false ∧
    checkFlat (Sporos.Prog.enc { words := [{ ins := 0, outs := 0, locals := 0, code := [.locget 0, .ret], ann := [] }], data := [], main := 0 }) = some false ∧
    checkFlat (Sporos.Prog.enc { words := [{ ins := 0, outs := 0, locals := 0, code := [.jmp 0], ann := [] }], data := [], main := 0 }) = some false ∧
    checkFlat [256] = none := by
  decide +kernel

theorem checkFlat_empty_refused : checkFlat [] = some false := checkFlat_refusal_witnesses.1
theorem checkFlat_badMagic_refused :
    checkFlat [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] = some false := checkFlat_refusal_witnesses.2.1
theorem checkFlat_badMainIns_refused :
    checkFlat (Sporos.Prog.enc { words := [{ ins := 1, outs := 0, locals := 0, code := [.ret], ann := [] }], data := [], main := 0 }) =
      some false := checkFlat_refusal_witnesses.2.2.1
theorem checkFlat_underflow_refused :
    checkFlat (Sporos.Prog.enc { words := [{ ins := 0, outs := 0, locals := 0, code := [.prim .add, .ret], ann := [] }], data := [], main := 0 }) =
      some false := checkFlat_refusal_witnesses.2.2.2.1
theorem checkFlat_badMain_refused :
    checkFlat (Sporos.Prog.enc { words := [{ ins := 0, outs := 0, locals := 0, code := [.ret], ann := [] }], data := [], main := 1 }) =
      some false := checkFlat_refusal_witnesses.2.2.2.2.1
theorem checkFlat_trailing_refused :
    checkFlat (Sporos.echo.enc ++ [0]) = some false := checkFlat_refusal_witnesses.2.2.2.2.2.1
theorem checkFlat_truncated_refused :
    checkFlat (Sporos.echo.enc.take 16) = some false := checkFlat_refusal_witnesses.2.2.2.2.2.2.1
theorem checkFlat_badOpcode_refused :
    checkFlat
      [ 0x53, 0x50, 0x52, 0x31, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0
      , 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0xff ] = some false :=
  checkFlat_refusal_witnesses.2.2.2.2.2.2.2.1
theorem checkFlat_badLocal_refused :
    checkFlat (Sporos.Prog.enc { words := [{ ins := 0, outs := 0, locals := 0, code := [.locget 0, .ret], ann := [] }], data := [], main := 0 }) =
      some false := checkFlat_refusal_witnesses.2.2.2.2.2.2.2.2.1
theorem checkFlat_unannotatedJump_refused :
    checkFlat (Sporos.Prog.enc { words := [{ ins := 0, outs := 0, locals := 0, code := [.jmp 0], ann := [] }], data := [], main := 0 }) =
      some false := checkFlat_refusal_witnesses.2.2.2.2.2.2.2.2.2.1
theorem checkFlat_nonbyte_none : checkFlat [256] = none :=
  checkFlat_refusal_witnesses.2.2.2.2.2.2.2.2.2.2

/-! ### Symbolic valid-byte `HMAX` overflow witness -/

private def hmaxPref : List Nat :=
  [ 0x53, 0x50, 0x52, 0x31
  , 0, 0, 0, 0
  , 1, 0, 0, 0
  , 0, 0, 0, 0
  , 0, 0, 0, 0
  , 0xff, 0xff, 0xff, 0xff
  , 0, 0, 0, 0
  , 0x01, 0x00, 0x01, 0x00
  , 0, 0, 0, 0 ]

private def call0Bytes : Nat → List Nat
  | 0 => []
  | k + 1 => 0x8b :: 0 :: 0 :: 0 :: 0 :: call0Bytes k

private theorem hmaxPref_len : hmaxPref.length = 36 := rfl

private theorem call0Bytes_len : ∀ k, (call0Bytes k).length = 5 * k
  | 0 => rfl
  | k + 1 => by simp only [call0Bytes, List.length_cons, call0Bytes_len k]; omega

private theorem call0Bytes_bytes : ∀ k b, b ∈ call0Bytes k → b < 256
  | 0, _, h => by cases h
  | k + 1, b, h => by
    simp only [call0Bytes, List.mem_cons] at h
    rcases h with rfl | rfl | rfl | rfl | rfl | h
    · omega
    · omega
    · omega
    · omega
    · omega
    · exact call0Bytes_bytes k b h

private theorem hmaxWitnessBytes_bytes (N : Nat) : ∀ b ∈ hmaxPref ++ call0Bytes N, b < 256 := by
  intro b hb
  simp only [List.mem_append] at hb
  rcases hb with h1 | h2
  · exact of_decide_eq_true (List.all_eq_true.1 (by decide : hmaxPref.all (· < 256) = true) b h1)
  · exact call0Bytes_bytes N b h2

private theorem hmax_rdU32_lt (N : Nat) {pos : Nat} (h : pos + 4 ≤ 36) :
    rdU32 (hmaxPref ++ call0Bytes N) pos = rdU32 hmaxPref pos := by
  simp only [rdU32,
    List.getElem?_append_left (by simp only [hmaxPref_len]; omega : pos < hmaxPref.length),
    List.getElem?_append_left (by simp only [hmaxPref_len]; omega : pos + 1 < hmaxPref.length),
    List.getElem?_append_left (by simp only [hmaxPref_len]; omega : pos + 2 < hmaxPref.length),
    List.getElem?_append_left (by simp only [hmaxPref_len]; omega : pos + 3 < hmaxPref.length)]

private theorem hmax_call0_at : ∀ k j, j < k →
    (call0Bytes k)[5 * j]? = some 0x8b ∧
    (call0Bytes k)[5 * j + 1]? = some 0 ∧
    (call0Bytes k)[5 * j + 2]? = some 0 ∧
    (call0Bytes k)[5 * j + 3]? = some 0 ∧
    (call0Bytes k)[5 * j + 4]? = some 0
  | 0, _, h => by cases h
  | k + 1, 0, _ => by simp [call0Bytes]
  | k + 1, j + 1, h => by
    have ih := hmax_call0_at k j (Nat.lt_of_succ_lt_succ h)
    have h5 : ∀ r, (call0Bytes (k + 1))[5 * (j + 1) + r]? = (call0Bytes k)[5 * j + r]? := by
      intro r
      have hidx : 5 * (j + 1) + r = (5 * j + r) + 1 + 1 + 1 + 1 + 1 := by omega
      rw [hidx]
      simp only [call0Bytes, List.getElem?_cons_succ]
    refine ⟨?_, ?_, ?_, ?_, ?_⟩
    · rw [← Nat.add_zero (5 * (j + 1)), h5 0, Nat.add_zero]; exact ih.1
    · rw [h5 1]; exact ih.2.1
    · rw [h5 2]; exact ih.2.2.1
    · rw [h5 3]; exact ih.2.2.2.1
    · rw [h5 4]; exact ih.2.2.2.2

private theorem hmax_instr_at (N : Nat) {j : Nat} (hj : j < N) :
    (hmaxPref ++ call0Bytes N)[36 + 5 * j]? = some 0x8b ∧
    rdU32 (hmaxPref ++ call0Bytes N) (36 + 5 * j + 1) = some 0 := by
  have hat := hmax_call0_at N j hj
  have ha0 : (hmaxPref ++ call0Bytes N)[36 + 5 * j]? = (call0Bytes N)[5 * j]? := by
    rw [List.getElem?_append_right (by simp only [hmaxPref_len]; omega), hmaxPref_len]
    simp [Nat.add_sub_cancel_left]
  have ha1 : (hmaxPref ++ call0Bytes N)[36 + 5 * j + 1]? = (call0Bytes N)[5 * j + 1]? := by
    rw [List.getElem?_append_right (by simp only [hmaxPref_len]; omega), hmaxPref_len]
    have hidx : 36 + 5 * j + 1 - 36 = 5 * j + 1 := by omega
    rw [hidx]
  have ha2 : (hmaxPref ++ call0Bytes N)[36 + 5 * j + 1 + 1]? = (call0Bytes N)[5 * j + 2]? := by
    rw [List.getElem?_append_right (by simp only [hmaxPref_len]; omega), hmaxPref_len]
    have hidx : 36 + 5 * j + 1 + 1 - 36 = 5 * j + 2 := by omega
    rw [hidx]
  have ha3 : (hmaxPref ++ call0Bytes N)[36 + 5 * j + 1 + 2]? = (call0Bytes N)[5 * j + 3]? := by
    rw [List.getElem?_append_right (by simp only [hmaxPref_len]; omega), hmaxPref_len]
    have hidx : 36 + 5 * j + 1 + 2 - 36 = 5 * j + 3 := by omega
    rw [hidx]
  have ha4 : (hmaxPref ++ call0Bytes N)[36 + 5 * j + 1 + 3]? = (call0Bytes N)[5 * j + 4]? := by
    rw [List.getElem?_append_right (by simp only [hmaxPref_len]; omega), hmaxPref_len]
    have hidx : 36 + 5 * j + 1 + 3 - 36 = 5 * j + 4 := by omega
    rw [hidx]
  refine ⟨by rw [ha0, hat.1], ?_⟩
  simp only [rdU32, ha1, ha2, ha3, ha4, hat.2.1, hat.2.2.1, hat.2.2.2.1, hat.2.2.2.2]
  rfl

private theorem hmax_skipCode (N : Nat) : ∀ rem i, rem + i = N →
    skipCode (hmaxPref ++ call0Bytes N) rem (36 + 5 * i) = some (36 + 5 * N)
  | 0, i, h => by
    have hi : i = N := by omega
    subst hi; rfl
  | r + 1, i, h => by
    have hi : i < N := by omega
    have hat := hmax_instr_at N hi
    have hlen_bs : (hmaxPref ++ call0Bytes N).length = 36 + 5 * N := by
      simp only [List.length_append, hmaxPref_len, call0Bytes_len]
    have hlen_8b : instrLen 0x8b = some 5 := by decide
    have hstep : skipInstr (hmaxPref ++ call0Bytes N) (36 + 5 * i) = some (36 + 5 * (i + 1)) := by
      simp only [skipInstr, hat.1, hlen_8b, bind, Option.bind]
      rw [if_pos (by rw [hlen_bs]; omega)]
      simp [Nat.mul_add, Nat.add_assoc]
    simp only [skipCode, hstep, bind, Option.bind]
    exact hmax_skipCode N r (i + 1) (by omega)

private theorem hmax_readWordAt0 (N : Nat) (hN : N = 65537) :
    readWordAt (hmaxPref ++ call0Bytes N) 16 =
      some ⟨0, 2 ^ 32 - 1, 0, N, 0, 36, 36 + 5 * N, 36 + 5 * N⟩ := by
  have hlen_bs : (hmaxPref ++ call0Bytes N).length = 36 + 5 * N := by
    simp only [List.length_append, hmaxPref_len, call0Bytes_len]
  have h16 : rdU32 (hmaxPref ++ call0Bytes N) 16 = some 0 := by rw [hmax_rdU32_lt N (by omega)]; decide
  have h20 : rdU32 (hmaxPref ++ call0Bytes N) 20 = some (2 ^ 32 - 1) := by rw [hmax_rdU32_lt N (by omega)]; decide
  have h24 : rdU32 (hmaxPref ++ call0Bytes N) 24 = some 0 := by rw [hmax_rdU32_lt N (by omega)]; decide
  have h28 : rdU32 (hmaxPref ++ call0Bytes N) 28 = some N := by
    rw [hmax_rdU32_lt N (by omega), hN]; decide
  have h32 : rdU32 (hmaxPref ++ call0Bytes N) 32 = some 0 := by rw [hmax_rdU32_lt N (by omega)]; decide
  have hskip := hmax_skipCode N N 0 (by omega)
  rw [Nat.mul_zero, Nat.add_zero] at hskip
  simp only [readWordAt, show 16 + 20 ≤ (hmaxPref ++ call0Bytes N).length by rw [hlen_bs]; omega,
    ↓reduceIte, bind, Option.bind, h16, h20, h24, h28, h32, hskip]
  rw [if_pos (by rw [hlen_bs]; omega)]
  simp

private theorem hmax_parse (N : Nat) (hN : N = 65537) :
    parse (hmaxPref ++ call0Bytes N) = some (0, 1, 16) := by
  have hlen_bs : (hmaxPref ++ call0Bytes N).length = 36 + 5 * N := by
    simp only [List.length_append, hmaxPref_len, call0Bytes_len]
  have h0 : rdU32 (hmaxPref ++ call0Bytes N) 0 = some MAGIC := by rw [hmax_rdU32_lt N (by omega)]; decide
  have h4 : rdU32 (hmaxPref ++ call0Bytes N) 4 = some 0 := by rw [hmax_rdU32_lt N (by omega)]; decide
  have h8 : rdU32 (hmaxPref ++ call0Bytes N) 8 = some 1 := by rw [hmax_rdU32_lt N (by omega)]; decide
  have h12 : rdU32 (hmaxPref ++ call0Bytes N) 12 = some 0 := by rw [hmax_rdU32_lt N (by omega)]; decide
  have h28 : rdU32 (hmaxPref ++ call0Bytes N) 28 = some N := by
    rw [hmax_rdU32_lt N (by omega), hN]; decide
  have h32 : rdU32 (hmaxPref ++ call0Bytes N) 32 = some 0 := by rw [hmax_rdU32_lt N (by omega)]; decide
  have hskip := hmax_skipCode N N 0 (by omega)
  rw [Nat.mul_zero, Nat.add_zero] at hskip
  simp only [parse, show ¬ (hmaxPref ++ call0Bytes N).length < 16 by rw [hlen_bs]; omega,
    ↓reduceIte, bind, Option.bind, h0, h4, h8, h12,
    show (MAGIC != MAGIC) = false by decide, Bool.false_eq_true, skipData, skipWords,
    show 16 + 20 ≤ (hmaxPref ++ call0Bytes N).length by rw [hlen_bs]; omega,
    h28, h32, hskip]
  rw [if_pos (by rw [hlen_bs]; omega)]
  simp [hlen_bs]

private theorem hmax_walk_overflow (N : Nat) (hN : N = 65537) : ∀ rem i curD,
    rem + i = N → 1 ≤ rem → curD = i * (2 ^ 32 - 1) → curD < HMAX →
    walk (hmaxPref ++ call0Bytes N) 1 16 (2 ^ 32 - 1) 0 N (36 + 5 * N) 0 rem i (36 + 5 * i) curD 0 = .overflow
  | 0, _, _, _, hge, _, _ => by exfalso; omega
  | r + 1, i, curD, hsum, _, hcur, hlt => by
    have hi : i < N := by omega
    have hat := hmax_instr_at N hi
    have hlen_bs : (hmaxPref ++ call0Bytes N).length = 36 + 5 * N := by
      simp only [List.length_append, hmaxPref_len, call0Bytes_len]
    have hprim8b : primEff 0x8b = none := by decide
    have hlen8b : instrLen 0x8b = some 5 := by decide
    have hstepF :
        hstepF (hmaxPref ++ call0Bytes N) 1 16 (2 ^ 32 - 1) 0 (36 + 5 * N) 0 curD 0 0x8b 0 =
          if curD + (2 ^ 32 - 1) ≥ HMAX then .overflow else .next (curD + (2 ^ 32 - 1)) 0 := by
      unfold hstepF
      rw [hprim8b]
      dsimp only
      simp only [show (0x8b == 0x80) = false by decide,
        show (0x8b == 0x81) = false by decide,
        show (0x8b == 0x82) = false by decide,
        show (0x8b == 0x83) = false by decide,
        show (0x8b == 0x84) = false by decide,
        show (0x8b == 0x85) = false by decide,
        show (0x8b == 0x86) = false by decide,
        show (0x8b == 0x87) = false by decide,
        show (0x8b == 0x88) = false by decide,
        show (0x8b == 0x89) = false by decide,
        show (0x8b == 0x8a) = false by decide,
        show (0x8b == 0x8b) = true by decide,
        Bool.false_or, Bool.false_eq_true, ↓reduceIte,
        show ¬ (0 ≥ 1) by omega, wordAt, hmax_readWordAt0 N hN,
        show ¬ (curD < 0) by omega, Nat.sub_zero]
    have hw_eq : walk (hmaxPref ++ call0Bytes N) 1 16 (2 ^ 32 - 1) 0 N (36 + 5 * N) 0
        (r + 1) i (36 + 5 * i) curD 0 =
        if curD + (2 ^ 32 - 1) ≥ HMAX then .overflow
        else if i + 1 ≥ N then .reject
        else walk (hmaxPref ++ call0Bytes N) 1 16 (2 ^ 32 - 1) 0 N (36 + 5 * N) 0
          r (i + 1) (36 + 5 * i + 5) (curD + (2 ^ 32 - 1)) 0 := by
      have hbd : decide (36 + 5 * i + 5 ≤ (hmaxPref ++ call0Bytes N).length) = true :=
        decide_eq_true (by rw [hlen_bs]; omega)
      rw [walk]
      simp only [hat.1, hat.2, hlen8b, hbd, Bool.not_true, Bool.false_eq_true, ↓reduceIte,
        show (5 == 5) = true by decide, lookup, lookupLoop, Option.getD_none]
      rw [hstepF]
      by_cases hov : curD + (2 ^ 32 - 1) ≥ HMAX
      · simp [hov]
      · simp only [hov, ↓reduceIte]
    rw [hw_eq]
    by_cases hov : curD + (2 ^ 32 - 1) ≥ HMAX
    · simp [hov]
    · simp only [hov, ↓reduceIte]
      have hi1_lt : i + 1 < N := by
        by_cases hge : i + 1 < N
        · exact hge
        · exfalso
          have heq : i = 65536 := by omega
          subst heq
          have hc : curD = 65536 * (2 ^ 32 - 1) := hcur
          unfold HMAX at hov
          omega
      simp only [show ¬ (i + 1 ≥ N) by omega, ↓reduceIte]
      have hr1 : 1 ≤ r := by omega
      have hidx : 36 + 5 * i + 5 = 36 + 5 * (i + 1) := by omega
      rw [hidx]
      exact hmax_walk_overflow N hN r (i + 1) (curD + (2 ^ 32 - 1)) (by omega) hr1
        (by rw [hcur, Nat.add_mul, Nat.one_mul]) (by omega)

private theorem checkFlat_hmax_core (N : Nat) (hN : N = 65537) :
    (∀ b ∈ hmaxPref ++ call0Bytes N, b < 256) ∧
      (hmaxPref ++ call0Bytes N).length < W ∧
      checkFlat (hmaxPref ++ call0Bytes N) = none := by
  have hlen_bs : (hmaxPref ++ call0Bytes N).length = 36 + 5 * N := by
    simp only [List.length_append, hmaxPref_len, call0Bytes_len]
  have hbytes := hmaxWitnessBytes_bytes N
  refine ⟨hbytes, by rw [hlen_bs]; unfold W; omega, ?_⟩
  have hall : (hmaxPref ++ call0Bytes N).all (· < 256) = true :=
    List.all_eq_true.2 fun b hb => decide_eq_true (hbytes b hb)
  have hwalk := hmax_walk_overflow N hN N 0 0 (by omega) (by omega) rfl (by unfold HMAX; omega)
  rw [Nat.mul_zero, Nat.add_zero] at hwalk
  have hres : checkFlatRes (hmaxPref ++ call0Bytes N) = some .overflow := by
    simp only [checkFlatRes, hall, Bool.not_true, Bool.false_or,
      show ¬ ((hmaxPref ++ call0Bytes N).length ≥ W) by rw [hlen_bs]; unfold W; omega,
      decide_false, Bool.false_eq_true, ↓reduceIte, hmax_parse N hN,
      show ¬ (0 ≥ 1) by omega, wordAt, hmax_readWordAt0 N hN,
      show (0 != 0) = false by decide, checkWords,
      beq_false_of_ne (by omega : N ≠ 0), lookup, lookupLoop,
      hwalk]
  simp only [checkFlat, hres, bind, Option.bind]

theorem checkFlat_hmax_overflow_witness :
    ∃ bs, (∀ b ∈ bs, b < 256) ∧ bs.length < W ∧ checkFlat bs = none :=
  ⟨hmaxPref ++ call0Bytes 65537, checkFlat_hmax_core 65537 rfl⟩

/-! ### Top-level 3-pass + `TMRVoter` + dual-`ACCEPT`-guard theorem on `checkFlat` -/

/-- Canonical 64-bit word output of one flat verifier pass. -/
def checkFlatWord (bs : List Nat) : Nat :=
  match checkFlat bs with
  | some true => ACCEPT
  | some false => REJECT
  | none => OVERFLOW

theorem checkFlatWord_lt_W (bs : List Nat) : checkFlatWord bs < X86Env.W := by
  unfold checkFlatWord
  cases checkFlat bs with
  | none => decide
  | some b => cases b <;> decide

theorem checkFlatWord_safeInvalid_of_ne_true {bs : List Nat} (h : checkFlat bs ≠ some true) :
    SafeInvalid (checkFlatWord bs) := by
  unfold checkFlatWord
  cases hcf : checkFlat bs with
  | none => exact SafeInvalid_overflow
  | some b => cases b with
    | false => exact SafeInvalid_reject
    | true => exact absurd hcf h

theorem threePass_valid_accepts {bs : List Nat} (h : checkFlat bs = some true)
    {base : Nat} (hb : base_ok base) (s0 : X86Int.St) :
    ∀ n, 20 ≤ n →
      X86Int.run (tailImAt base) n
        (.ok (tailInit base s0 (checkFlatWord bs) (checkFlatWord bs) (checkFlatWord bs))) = .exit 0 := by
  have hw : checkFlatWord bs = ACCEPT := by simp [checkFlatWord, h]
  rw [hw]
  exact tail_arch_accept hb s0 (by decide) (by decide) (by decide) (by decide)

theorem threePass_no_single_fault_false_accept {bs : List Nat}
    (hbad : ¬ (∃ pr, Sporos.decode bs = some pr ∧ Sporos.checkProg pr = true))
    {base : Nat} (hb : base_ok base) (s0 : X86Int.St) (hst : TailStackSafe base s0) :
    let w := checkFlatWord bs
    (∀ {a b c : Nat}, SinglePassFaultTriple w (a, b, c) →
      ∀ n, X86Int.run (tailImAt base) n (.ok (tailInit base s0 a b c)) ≠ .exit 0) ∧
    (∀ {n : Nat} {o : X86Int.Out},
      OneFaultRun base false n (.ok (tailInit base s0 w w w)) o → o ≠ .exit 0) ∧
    (∀ (r : X86Int.Rates) (hr : r.ok), singleMechanism r → ∀ k m,
      exit0Prob (tailPRunWithFaultAt base r hr k m (.ok (tailInit base s0 w w w))) = 0 ∧
      ∀ y ∈ tailPRunWithFaultAt base r hr k m (.ok (tailInit base s0 w w w)),
        0 < y.2 → y.1.1 ≠ .exit 0) := by
  have hne : checkFlat bs ≠ some true := fun h => hbad (checkFlat_sound h)
  exact invalidWord_threePass_no_single_fault_false_accept hb s0 hst
    (checkFlatWord_safeInvalid_of_ne_true hne)

theorem threePass_single_fault_exit0_safe {bs : List Nat} {base a b c n : Nat}
    (hb : base_ok base) (s0 : X86Int.St) (hst : TailStackSafe base s0)
    (htri : SinglePassFaultTriple (checkFlatWord bs) (a, b, c))
    (hexit : X86Int.run (tailImAt base) n (.ok (tailInit base s0 a b c)) = .exit 0) :
    ∃ pr, Sporos.decode bs = some pr ∧ Sporos.checkProg pr = true ∧
      (∀ ph (s : Sporos.St), Sporos.Inv pr s → Sporos.Safe pr (Sporos.step pr ph s)) ∧
      (∀ ph msize inp k e,
        Sporos.run pr ph k (Sporos.init pr msize inp) = .fail e → e.static = false) := by
  have hcf : checkFlat bs = some true := by
    cases h : checkFlat bs with
    | none =>
      exfalso
      exact (invalidWord_threePass_no_single_fault_false_accept hb s0 hst
        (checkFlatWord_safeInvalid_of_ne_true (by simp [h]))).1 htri n hexit
    | some ok =>
      cases ok with
      | false =>
        exfalso
        exact (invalidWord_threePass_no_single_fault_false_accept hb s0 hst
          (checkFlatWord_safeInvalid_of_ne_true (by simp [h]))).1 htri n hexit
      | true => rfl
  obtain ⟨pr, hdec, hchk⟩ := checkFlat_sound hcf
  exact ⟨pr, hdec, hchk, fun ph s hi => Sporos.step_safe pr ph hchk s hi,
    fun ph msize inp k e hr => Sporos.run_safe pr ph hchk msize inp k e hr⟩

end SporosCheckFlat


