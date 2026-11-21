/-
Flat-byte specification of `spr1_check_arm64`, equivalence with
`Sporos.decode` and `Sporos.checkProg`, and single-fault non-false-accept
immunity over the canonical AArch64 44-byte `TMRVoterArm64` voter + dual-`ACCEPT`
guard (`AArch64Phys.pstep` / `AArch64Phys.prun`).
-/
import Sporos.Bytecode
import Sporos.Check
import Sporos.SafeStep
import Sporos.Run
import Sporos.Examples
import Bootstrap.PDist
import Bootstrap.TMR
import Bootstrap.TMRExposure
import Bootstrap.AArch64Env
import Bootstrap.AArch64Phys
import Bootstrap.TMRVoterArm64

namespace SporosCheckArm64Flat

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
  simp only [hn57, ↓reduceDIte] at hins
  split at hins <;> first | omega | cases hins

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
  simp only [hn57, ↓reduceDIte] at hins
  split at hins <;> first | omega | cases hins

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

/-! ### Tail voter + dual-`ACCEPT`-guard (`tailCode`, 192 bytes = 48 words) and single-fault immunity -/

def OVERFLOW : Nat := 0x0bad0f1e33c3c3a5

/-- Canonical 48-word (192-byte) AArch64 tail (`[vote, core_end)`):
words `0..10` equal `TMRVoterArm64.voterWords` with word 7 (`b +0`) replaced by
`0x14000004` (`b +0x10` to `guard`); words `11..47` run two independent `ACCEPT`
checks on `x19` and `x20`, compute `x0 := x20 - 0x96` before `svc #0`, else
dispatch `OVERFLOW` to `exit(6)` and all other values to `exit(5)` via a
dual-register `movz x0`/`movz x2`/`orr x0, x0, x2` sequence (popcount = 2,
immune to single-bit `hw`, `adrp`, and `reg` upsets). -/
def tailWords : List Nat :=
  [ 0xeb14027f  --  0 (0x00): cmp  x19, x20
  , 0xfa551266  --  1 (0x04): ccmp x19, x21, #6, ne
  , 0x9a940273  --  2 (0x08): csel x19, x19, x20, eq
  , 0xfa551286  --  3 (0x0c): ccmp x20, x21, #6, ne
  , 0x54000081  --  4 (0x10): b.ne +0x10 (-> 8)
  , 0xaa1303f4  --  5 (0x14): mov  x20, x19
  , 0xaa1303f5  --  6 (0x18): mov  x21, x19
  , 0x14000004  --  7 (0x1c): b    +0x10 (-> 11 guard)
  , 0xd2800060  --  8 (0x20): movz x0, #3
  , 0xd2800ba8  --  9 (0x24): movz x8, #93
  , 0xd4000001  -- 10 (0x28): svc  #0
  , 0xd28000a0  -- 11 (0x2c): movz x0, #5
  , 0xd2800008  -- 12 (0x30): movz x8, #0
  , 0xd2800002  -- 13 (0x34): movz x2, #0
  , 0xd2800007  -- 14 (0x38): movz x7, #0
  , 0xd28792c1  -- 15 (0x3c): movz x1, #0x3c96
  , 0xf2b4ab41  -- 16 (0x40): movk x1, #0xa55a, lsl #16
  , 0xf2dd2e01  -- 17 (0x44): movk x1, #0xe970, lsl #32
  , 0xf2eb5981  -- 18 (0x48): movk x1, #0x5acc, lsl #48
  , 0xeb01027f  -- 19 (0x4c): cmp  x19, x1
  , 0x54000181  -- 20 (0x50): b.ne +0x30 (-> 32 not_acc)
  , 0xd2800002  -- 21 (0x54): movz x2, #0
  , 0xd28792c6  -- 22 (0x58): movz x6, #0x3c96
  , 0xf2b4ab46  -- 23 (0x5c): movk x6, #0xa55a, lsl #16
  , 0xf2dd2e06  -- 24 (0x60): movk x6, #0xe970, lsl #32
  , 0xf2eb5986  -- 25 (0x64): movk x6, #0x5acc, lsl #48
  , 0xeb06029f  -- 26 (0x68): cmp  x20, x6
  , 0x540000a1  -- 27 (0x6c): b.ne +0x14 (-> 32 not_acc)
  , 0xd2800ba8  -- 28 (0x70): movz x8, #93
  , 0xd1025a80  -- 29 (0x74): sub  x0, x20, #150
  , 0xd4000001  -- 30 (0x78): svc  #0
  , 0x1400000c  -- 31 (0x7c): b    +0x30 (-> 43 rej_exit)
  , 0xd29874a1  -- 32 (0x80): movz x1, #0xc3a5
  , 0xf2a67861  -- 33 (0x84): movk x1, #0x33c3, lsl #16
  , 0xf2c1e3c1  -- 34 (0x88): movk x1, #0x0f1e, lsl #32
  , 0xf2e175a1  -- 35 (0x8c): movk x1, #0x0bad, lsl #48
  , 0xeb01027f  -- 36 (0x90): cmp  x19, x1
  , 0x540000c1  -- 37 (0x94): b.ne +0x18 (-> 43 rej_exit)
  , 0xd28000c0  -- 38 (0x98): movz x0, #6
  , 0xd28000c2  -- 39 (0x9c): movz x2, #6
  , 0xaa020000  -- 40 (0xa0): orr  x0, x0, x2
  , 0xd2800ba8  -- 41 (0xa4): movz x8, #93
  , 0xd4000001  -- 42 (0xa8): svc  #0
  , 0xd28000a0  -- 43 (0xac): movz x0, #5
  , 0xd28000a2  -- 44 (0xb0): movz x2, #5
  , 0xaa020000  -- 45 (0xb4): orr  x0, x0, x2
  , 0xd2800ba8  -- 46 (0xb8): movz x8, #93
  , 0xd4000001  -- 47 (0xbc): svc  #0
  ]

def tailCode : List Nat := AArch64Phys.wordsToBytes tailWords

theorem tailWords_length : tailWords.length = 48 := by decide +kernel
theorem tailCode_length  : tailCode.length = 192 := by decide +kernel

theorem tailWords_voter_prefix :
    tailWords.take 11 = TMRVoterArm64.voterWords.set 7 0x14000004 := by
  decide +kernel

end FlatSpec

section TailProofs

open AArch64Env (Mem Kern)
open AArch64Phys (Ins St Flags Out Img Alts Rates Fault lift step run pstep prun exec execO unif relabel regFlips memFlips nzcvFlips decodeWord imageOf imageAlts altsAt altsWord altsAt_eq_altsWord wordBytes wordsToBytes addSubOp msb pow2 shiftVal LogicOp ShiftKind MemSize CSelOp)

def tailImAt (base : Nat) : Img := imageOf tailCode base
def tailAltsAt (base : Nat) : Alts := imageAlts tailCode base

def base_ok (base : Nat) : Prop :=
  base % 4 = 0 ∧ 256 ≤ base ∧ base + 2 ^ 32 ≤ AArch64Env.W

def tailInit (base : Nat) (s : St) (a b c : Nat) : St :=
  { s with regs := TMRVoterArm64.voterRegs s a b c, pc := base, fl := Flags.undef }

def valNotAcc (v : Nat) : Prop :=
  v < AArch64Env.W ∧ v ≠ ACCEPT ∧ v % 256 ≠ ACCEPT % 256

def valNotAccB (v : Nat) : Bool :=
  v < AArch64Env.W && v != ACCEPT && v % 256 != ACCEPT % 256

theorem valNotAccB_sound {v : Nat} (h : valNotAccB v = true) : valNotAcc v := by
  unfold valNotAccB at h
  simp only [Bool.and_eq_true, decide_eq_true_eq, bne_iff_ne] at h
  exact ⟨h.1.1, h.1.2, h.2⟩

def SafeInvalid (w : Nat) : Prop :=
  w < AArch64Env.W ∧ w ≠ ACCEPT ∧ w % 256 ≠ ACCEPT % 256 ∧
  (List.range 64).all (fun b => Nat.xor w (2 ^ b) != ACCEPT) = true ∧
  (List.range 8).all (fun b => Nat.xor (w % 256) (2 ^ b) != ACCEPT % 256) = true ∧
  valNotAcc ((w + 1) % AArch64Env.W) ∧
  valNotAcc (AArch64Env.W - 1 - w) ∧
  valNotAcc ((AArch64Env.W - w) % AArch64Env.W) ∧
  valNotAcc (w % 2 ^ 32) ∧
  ([ShiftKind.lsl, .lsr, .asr].all fun sh =>
    [0, 1, 2, 4, 8, 16, 32].all fun amt =>
      valNotAccB (shiftVal sh 64 w amt) && valNotAccB (shiftVal sh 32 (w % pow2 32) amt)) = true

theorem SafeInvalid.valNotAcc {w : Nat} (h : SafeInvalid w) : valNotAcc w :=
  ⟨h.1, h.2.1, h.2.2.1⟩

def SafeInvalidB (w : Nat) : Bool :=
  w < AArch64Env.W && w != ACCEPT && w % 256 != ACCEPT % 256 &&
  (List.range 64).all (fun b => Nat.xor w (2 ^ b) != ACCEPT) &&
  (List.range 8).all (fun b => Nat.xor (w % 256) (2 ^ b) != ACCEPT % 256) &&
  valNotAccB ((w + 1) % AArch64Env.W) &&
  valNotAccB (AArch64Env.W - 1 - w) &&
  valNotAccB ((AArch64Env.W - w) % AArch64Env.W) &&
  valNotAccB (w % 2 ^ 32) &&
  ([ShiftKind.lsl, .lsr, .asr].all fun sh =>
    [0, 1, 2, 4, 8, 16, 32].all fun amt =>
      valNotAccB (shiftVal sh 64 w amt) && valNotAccB (shiftVal sh 32 (w % pow2 32) amt))

theorem SafeInvalidB_sound {w : Nat} (h : SafeInvalidB w = true) : SafeInvalid w := by
  unfold SafeInvalidB at h
  simp only [Bool.and_eq_true, decide_eq_true_eq, bne_iff_ne] at h
  rcases h with ⟨⟨⟨⟨⟨⟨⟨⟨⟨h1, h2⟩, h3⟩, h4⟩, h5⟩, h6⟩, h7⟩, h8⟩, h9⟩, h10⟩
  have h4' : (List.range 64).all (fun b => Nat.xor w (2 ^ b) != ACCEPT) = true := by
    simp [List.all_eq_true] at h4 ⊢; exact h4
  have h5' : (List.range 8).all (fun b => Nat.xor (w % 256) (2 ^ b) != ACCEPT % 256) = true := by
    simp [List.all_eq_true] at h5 ⊢; exact h5
  have h10' : ([ShiftKind.lsl, .lsr, .asr].all fun sh =>
      [0, 1, 2, 4, 8, 16, 32].all fun amt =>
        valNotAccB (shiftVal sh 64 w amt) && valNotAccB (shiftVal sh 32 (w % pow2 32) amt)) = true := by
    simpa [Bool.and_eq_true] using h10
  exact ⟨h1, h2, h3, h4', h5', valNotAccB_sound h6, valNotAccB_sound h7,
    valNotAccB_sound h8, valNotAccB_sound h9, h10'⟩

theorem safeInvalid_REJECT : SafeInvalid REJECT := SafeInvalidB_sound (by decide +kernel)
theorem safeInvalid_OVERFLOW : SafeInvalid OVERFLOW := SafeInvalidB_sound (by decide +kernel)
theorem safeInvalid_zero : SafeInvalid 0 := SafeInvalidB_sound (by decide +kernel)

theorem SafeInvalid_shift {w : Nat} (hw : SafeInvalid w) {sh : ShiftKind}
    (hsh : sh ∈ [ShiftKind.lsl, .lsr, .asr]) {amt : Nat} (hamt : amt ∈ [0, 1, 2, 4, 8, 16, 32]) :
    valNotAcc (shiftVal sh 64 w amt) ∧ valNotAcc (shiftVal sh 32 (w % pow2 32) amt) := by
  have hall := List.all_eq_true.1 hw.2.2.2.2.2.2.2.2.2 sh hsh
  have hall2 := List.all_eq_true.1 hall amt hamt
  simp only [Bool.and_eq_true] at hall2
  exact ⟨valNotAccB_sound hall2.1, valNotAccB_sound hall2.2⟩

theorem valNotAcc_lor_odd {x w : Nat} (hw : w % 2 = 1) (hx : x < AArch64Env.W) (hww : w < AArch64Env.W) :
    valNotAcc (Nat.lor x w) := by
  have hlt : Nat.lor x w < AArch64Env.W := Nat.bitwise_lt_two_pow hx hww
  have hbit0 : (Nat.lor x w).testBit 0 = true := by
    rw [show Nat.lor x w = x ||| w from rfl, Nat.testBit_or,
      show w.testBit 0 = true by simpa [Nat.testBit_zero] using hw]
    simp
  have hodd : Nat.lor x w % 2 = 1 := by
    have ht : (Nat.lor x w).testBit 0 = true := hbit0
    simpa [Nat.testBit_zero] using ht
  refine ⟨hlt, ?_, ?_⟩
  · intro heq; rw [heq] at hodd; revert hodd; decide
  · intro hmod; have : (Nat.lor x w % 256) % 2 = Nat.lor x w % 2 := by omega
    rw [hmod, show ACCEPT % 256 = 150 by decide] at this; omega

theorem lor_mod256_ne0 {a b : Nat} (hne : a % 256 ≠ 0 ∨ b % 256 ≠ 0) :
    Nat.lor a b % 256 ≠ 0 := by
  intro hz
  have hall : ∀ i < 8, a.testBit i = false ∧ b.testBit i = false := by
    intro i hi
    have ht : (Nat.lor a b % 2 ^ 8).testBit i = false := by
      rw [show (2 ^ 8 : Nat) = 256 from rfl, hz]; simp
    rw [Nat.testBit_mod_two_pow, show decide (i < 8) = true from decide_eq_true hi, Bool.true_and,
      show Nat.lor a b = a ||| b from rfl, Nat.testBit_or] at ht
    cases hta : a.testBit i <;> cases htb : b.testBit i <;> simp_all
  have ha0 : a % 256 = 0 := by
    apply Nat.eq_of_testBit_eq; intro i
    rw [show (256 : Nat) = 2 ^ 8 from rfl, Nat.testBit_mod_two_pow]
    by_cases hi : i < 8
    · simp [hi, (hall i hi).1]
    · simp [hi]
  have hb0 : b % 256 = 0 := by
    apply Nat.eq_of_testBit_eq; intro i
    rw [show (256 : Nat) = 2 ^ 8 from rfl, Nat.testBit_mod_two_pow]
    by_cases hi : i < 8
    · simp [hi, (hall i hi).2]
    · simp [hi]
  rcases hne with ha | hb
  · exact ha ha0
  · exact hb hb0

private theorem xor_pow2_lt {a b n : Nat} (ha : a < 2 ^ n) (hb : b < 2 ^ n) :
    Nat.xor a b < 2 ^ n :=
  Nat.xor_lt_two_pow ha hb

private theorem mod_256_xor_ge_8 (w b : Nat) (hb : 8 ≤ b) :
    Nat.xor w (2 ^ b) % 256 = w % 256 := by
  have hd : 256 ∣ 2 ^ b := by
    have : 2 ^ b = 2 ^ 8 * 2 ^ (b - 8) := by rw [← Nat.pow_add]; congr 1; omega
    rw [this]; exact Nat.dvd_mul_right 256 _
  have h1 : Nat.xor w (2 ^ b) % 2 ^ 8 = Nat.xor (w % 2 ^ 8) (2 ^ b % 2 ^ 8) :=
    Nat.xor_mod_two_pow
  rw [show (2 ^ 8 : Nat) = 256 from rfl] at h1
  have hm0 : 2 ^ b % 256 = 0 := Nat.mod_eq_zero_of_dvd hd
  rw [hm0] at h1
  exact h1.trans (Nat.xor_zero _)

theorem SafeInvalid_xor_bit {w b : Nat} (hw : SafeInvalid w) (hb : b < 64) :
    valNotAcc (Nat.xor w (2 ^ b)) := by
  have hpow : 2 ^ b < 2 ^ 64 := Nat.pow_lt_pow_right (by decide) hb
  refine ⟨xor_pow2_lt hw.1 hpow, ?_, ?_⟩
  · have h64 := List.all_eq_true.1 hw.2.2.2.1 b (List.mem_range.2 hb)
    simpa using h64
  · by_cases h8 : b < 8
    · have hmod : Nat.xor w (2 ^ b) % 256 = Nat.xor (w % 256) (2 ^ b) := by
        have h1 : Nat.xor w (2 ^ b) % 2 ^ 8 = Nat.xor (w % 2 ^ 8) (2 ^ b % 2 ^ 8) :=
          Nat.xor_mod_two_pow
        have hlt : 2 ^ b < 256 := Nat.pow_lt_pow_right (by decide) h8
        simpa [Nat.mod_eq_of_lt hlt] using h1
      rw [hmod]
      have hl := List.all_eq_true.1 hw.2.2.2.2.1 b (List.mem_range.2 h8)
      simpa using hl
    · rw [mod_256_xor_ge_8 w b (Nat.le_of_not_lt h8)]
      exact hw.2.2.1

def VoterScratchRegs : List Nat := [3, 4, 5, 16, 17, 18, 22, 23, 27, 28, 29]

def VoterScratchClean (s : St) : Prop :=
  ∀ r ∈ VoterScratchRegs, ∀ v, s.reg r = some v → valNotAcc v

def tailIns : List Ins :=
  [ .addSubReg true true 64 31 19 20 .lsl 0   -- 0: cmp x19, x20
  , .ccmpReg true 64 19 21 6 1               -- 1: ccmp x19, x21, #6, ne
  , .csel .csel 64 19 19 20 0                -- 2: csel x19, x19, x20, eq
  , .ccmpReg true 64 20 21 6 1               -- 3: ccmp x20, x21, #6, ne
  , .bcond 1 16                              -- 4: b.ne +0x10 (-> 8)
  , .logicReg .orr false 64 20 31 19 .lsl 0  -- 5: mov x20, x19
  , .logicReg .orr false 64 21 31 19 .lsl 0  -- 6: mov x21, x19
  , .b false 16                              -- 7: b +0x10 (-> 11 guard)
  , .movWide 2 64 0 3 0                      -- 8: movz x0, #3
  , .movWide 2 64 8 93 0                     -- 9: movz x8, #93
  , .svc 0                                   -- 10: svc #0
  , .movWide 2 64 0 5 0                      -- 11: movz x0, #5
  , .movWide 2 64 8 0 0                      -- 12: movz x8, #0
  , .movWide 2 64 2 0 0                      -- 13: movz x2, #0
  , .movWide 2 64 7 0 0                      -- 14: movz x7, #0
  , .movWide 2 64 1 0x3c96 0                 -- 15: movz x1, #0x3c96
  , .movWide 3 64 1 0xa55a 1                 -- 16: movk x1, #0xa55a, lsl #16
  , .movWide 3 64 1 0xe970 2                 -- 17: movk x1, #0xe970, lsl #32
  , .movWide 3 64 1 0x5acc 3                 -- 18: movk x1, #0x5acc, lsl #48
  , .addSubReg true true 64 31 19 1 .lsl 0   -- 19: cmp x19, x1
  , .bcond 1 48                              -- 20: b.ne +0x30 (-> 32)
  , .movWide 2 64 2 0 0                      -- 21: movz x2, #0
  , .movWide 2 64 6 0x3c96 0                 -- 22: movz x6, #0x3c96
  , .movWide 3 64 6 0xa55a 1                 -- 23: movk x6, #0xa55a, lsl #16
  , .movWide 3 64 6 0xe970 2                 -- 24: movk x6, #0xe970, lsl #32
  , .movWide 3 64 6 0x5acc 3                 -- 25: movk x6, #0x5acc, lsl #48
  , .addSubReg true true 64 31 20 6 .lsl 0   -- 26: cmp x20, x6
  , .bcond 1 20                              -- 27: b.ne +0x14 (-> 32)
  , .movWide 2 64 8 93 0                     -- 28: movz x8, #93
  , .addSubImm true false 64 0 20 150        -- 29: sub x0, x20, #150
  , .svc 0                                   -- 30: svc #0
  , .b false 48                              -- 31: b +0x30 (-> 43)
  , .movWide 2 64 1 0xc3a5 0                 -- 32: movz x1, #0xc3a5
  , .movWide 3 64 1 0x33c3 1                 -- 33: movk x1, #0x33c3, lsl #16
  , .movWide 3 64 1 0x0f1e 2                 -- 34: movk x1, #0x0f1e, lsl #32
  , .movWide 3 64 1 0x0bad 3                 -- 35: movk x1, #0x0bad, lsl #48
  , .addSubReg true true 64 31 19 1 .lsl 0   -- 36: cmp x19, x1
  , .bcond 1 24                              -- 37: b.ne +0x18 (-> 43)
  , .movWide 2 64 0 6 0                      -- 38: movz x0, #6
  , .movWide 2 64 2 6 0                      -- 39: movz x2, #6
  , .logicReg .orr false 64 0 0 2 .lsl 0     -- 40: orr x0, x0, x2
  , .movWide 2 64 8 93 0                     -- 41: movz x8, #93
  , .svc 0                                   -- 42: svc #0
  , .movWide 2 64 0 5 0                      -- 43: movz x0, #5
  , .movWide 2 64 2 5 0                      -- 44: movz x2, #5
  , .logicReg .orr false 64 0 0 2 .lsl 0     -- 45: orr x0, x0, x2
  , .movWide 2 64 8 93 0                     -- 46: movz x8, #93
  , .svc 0                                   -- 47: svc #0
  ]

def tailStepAt (k : Nat) : Option (Ins × Nat) :=
  (tailWords[k]?).bind fun w => (decodeWord w).map (·, 4)

private theorem tailStepAt_eq_tailIns :
    (List.range 48).all (fun k => tailStepAt k == (tailIns[k]?).map (·, 4)) = true := by
  decide +kernel

private theorem tailImAt_0_all :
    (List.range 48).all (fun k => tailImAt 0 (4 * k) == tailStepAt k) = true := by
  decide +kernel

private theorem tailAltsAt_0_all :
    (List.range 48).all
      (fun k => tailAltsAt 0 (4 * k) == (tailWords[k]?.map altsWord).getD []) = true := by
  decide +kernel

private theorem tailImAt_base_zero {base k : Nat} (hb : base % 4 = 0) (hk : k < 48) :
    tailImAt base (base + 4 * k) = tailImAt 0 (4 * k) := by
  unfold tailImAt imageOf
  have hcl : tailCode.length = 192 := tailCode_length
  have hc1 : (base ≤ base + 4 * k && base + 4 * k + 4 ≤ base + tailCode.length &&
      base % 4 == 0 && (base + 4 * k) % 4 == 0) = true := by
    rw [hcl]
    simp only [Bool.and_eq_true, decide_eq_true_eq, beq_iff_eq]
    exact ⟨⟨⟨by omega, by omega⟩, hb⟩, by omega⟩
  have hc0 : (0 ≤ 4 * k && 4 * k + 4 ≤ 0 + tailCode.length &&
      0 % 4 == 0 && (4 * k) % 4 == 0) = true := by
    rw [hcl]
    simp only [Bool.and_eq_true, decide_eq_true_eq, beq_iff_eq]
    exact ⟨⟨⟨by omega, by omega⟩, trivial⟩, by omega⟩
  have hfn : (fun i => tailCode[base + 4 * k - base + i]?) = (fun i => tailCode[4 * k - 0 + i]?) := by
    funext i; rw [show base + 4 * k - base + i = 4 * k - 0 + i by omega]
  rw [if_pos hc1, if_pos hc0, hfn]

private theorem tailAltsAt_base_zero {base k : Nat} (hb : base % 4 = 0) (hk : k < 48) :
    tailAltsAt base (base + 4 * k) = tailAltsAt 0 (4 * k) := by
  unfold tailAltsAt imageAlts
  have hcl : tailCode.length = 192 := tailCode_length
  have hc1 : (base ≤ base + 4 * k && base + 4 * k + 4 ≤ base + tailCode.length &&
      base % 4 == 0 && (base + 4 * k) % 4 == 0) = true := by
    rw [hcl]
    simp only [Bool.and_eq_true, decide_eq_true_eq, beq_iff_eq]
    exact ⟨⟨⟨by omega, by omega⟩, hb⟩, by omega⟩
  have hc0 : (0 ≤ 4 * k && 4 * k + 4 ≤ 0 + tailCode.length &&
      0 % 4 == 0 && (4 * k) % 4 == 0) = true := by
    rw [hcl]
    simp only [Bool.and_eq_true, decide_eq_true_eq, beq_iff_eq]
    exact ⟨⟨⟨by omega, by omega⟩, trivial⟩, by omega⟩
  have hfn : (fun i => tailCode[base + 4 * k - base + i]?) = (fun i => tailCode[4 * k - 0 + i]?) := by
    funext i; rw [show base + 4 * k - base + i = 4 * k - 0 + i by omega]
  rw [if_pos hc1, if_pos hc0, hfn]

theorem tailImAt_word {base k : Nat} (hb : base % 4 = 0) (hk : k < 48) :
    tailImAt base (base + 4 * k) = tailStepAt k := by
  rw [tailImAt_base_zero hb hk]
  exact beq_iff_eq.1 (List.all_eq_true.1 tailImAt_0_all k (List.mem_range.2 hk))

theorem tailImAt_ins {base k : Nat} (hb : base % 4 = 0) (hk : k < 48) :
    tailImAt base (base + 4 * k) = (tailIns[k]?).map (·, 4) :=
  (tailImAt_word hb hk).trans (beq_iff_eq.1 (List.all_eq_true.1 tailStepAt_eq_tailIns k (List.mem_range.2 hk)))

theorem tailAltsAt_word {base k : Nat} (hb : base % 4 = 0) (hk : k < 48) :
    tailAltsAt base (base + 4 * k) = (tailWords[k]?.map altsWord).getD [] := by
  rw [tailAltsAt_base_zero hb hk]
  exact beq_iff_eq.1 (List.all_eq_true.1 tailAltsAt_0_all k (List.mem_range.2 hk))

theorem tailImAt_out {base a : Nat}
    (hout : a < base ∨ base + 192 ≤ a + 3 ∨ (a - base) % 4 ≠ 0) :
    tailImAt base a = none := by
  unfold tailImAt imageOf
  rw [tailCode_length]
  by_cases hc : (base ≤ a && a + 4 ≤ base + 192 && base % 4 == 0 && a % 4 == 0) = true
  · exfalso
    simp only [Bool.and_eq_true, decide_eq_true_eq, beq_iff_eq] at hc
    rcases hc with ⟨⟨⟨hle, hhi⟩, hb⟩, ha⟩
    rcases hout with h1 | h2 | h3
    · omega
    · omega
    · exact h3 (by omega)
  · rw [if_neg hc]

private theorem lift_some (s : St) : lift (some s) = .ok s := rfl

private theorem run_exit (im : Img) (c : Nat) : ∀ n, run im n (.exit c) = .exit c
  | 0 => rfl
  | _ + 1 => rfl

private theorem run_stuck (im : Img) : ∀ n, run im n .stuck = .stuck
  | 0 => rfl
  | _ + 1 => rfl

private theorem run_add (im : Img) :
    ∀ m n o, run im (m + n) o = run im n (run im m o)
  | 0, n, o => by rw [Nat.zero_add]; rfl
  | m + 1, n, .ok s => by
    rw [Nat.add_right_comm]
    simp only [run, run_add im m n (step im s)]
  | _ + 1, n, .exit c => by simp [run, run_exit]
  | _ + 1, n, .stuck => by simp [run, run_stuck]

theorem run_ge_exit {im : Img} {k n c : Nat} {o : Out}
    (h : run im k o = .exit c) (hk : k ≤ n) : run im n o = .exit c := by
  have : n = k + (n - k) := by omega
  rw [this, run_add, h, run_exit]

private theorem reg_set_eq {s : St} (hlen : s.regs.length = 31) {r : Nat} (hr : r < 31) (v : Option Nat) :
    (s.setReg r v).reg r = v := by
  simp [St.reg, St.setReg, show r < s.regs.length by omega]

private theorem reg_set_ne {s : St} {r1 r2 : Nat} (hne : r1 ≠ r2) (v : Option Nat) :
    (s.setReg r1 v).reg r2 = s.reg r2 := by
  simp [St.reg, St.setReg, List.getElem?_set_ne hne]

private theorem setReg_len {s : St} (hlen : s.regs.length = 31) (r : Nat) (v : Option Nat) :
    (s.setReg r v).regs.length = 31 := by
  simp [St.setReg, List.length_set, hlen]

private theorem wrZR_len {s : St} (hlen : s.regs.length = 31) (w r v : Nat) :
    (s.wrZR w r v).regs.length = 31 := by
  unfold St.wrZR
  split
  · exact hlen
  · exact setReg_len hlen r _

private theorem pow2_pos (w : Nat) : 0 < pow2 w := by
  unfold pow2; exact Nat.pow_pos (by decide)

private theorem shiftVal_lt_pow2 {sh : ShiftKind} {w v amt : Nat} (hnr : sh ≠ .ror) :
    shiftVal sh w v amt < pow2 w := by
  have hxlt : v % pow2 w < pow2 w := Nat.mod_lt _ (pow2_pos w)
  unfold shiftVal AArch64Phys.asrVal
  cases sh with
  | lsl => exact Nat.mod_lt _ (pow2_pos w)
  | lsr => exact Nat.lt_of_le_of_lt (Nat.div_le_self _ _) hxlt
  | asr =>
    dsimp only
    rw [Nat.mod_eq_of_lt hxlt]
    by_cases hz : amt = 0
    · simp [hz, hxlt]
    · simp only [beq_iff_eq, hz, ↓reduceIte]
      by_cases haw : w ≤ amt
      · have hpw : pow2 w ≤ pow2 amt := Nat.pow_le_pow_right (by decide) haw
        rw [Nat.div_eq_of_lt (Nat.lt_of_lt_of_le hxlt hpw)]
        have hppos := pow2_pos w
        have hp2 := pow2_pos (w - amt)
        have hle : pow2 (w - amt) ≤ pow2 w := Nat.pow_le_pow_right (by decide) (by omega)
        split <;> omega
      · have halt : amt < w := by omega
        have hmul : pow2 amt * pow2 (w - amt) = pow2 w := by
          unfold pow2; rw [← Nat.pow_add, Nat.add_sub_cancel' (Nat.le_of_lt halt)]
        have hdiv : (v % pow2 w) / pow2 amt < pow2 (w - amt) :=
          Nat.div_lt_of_lt_mul (by rw [hmul]; exact hxlt)
        have hle : pow2 (w - amt) ≤ pow2 w := Nat.pow_le_pow_right (by decide) (by omega)
        split <;> omega
  | ror => exact absurd rfl hnr

private theorem wrZR_eq {s : St} (hlen : s.regs.length = 31) (w : Nat) {r : Nat} (hr : r < 31) (v : Nat) :
    (s.wrZR w r v).reg r = some (v % pow2 w) := by
  simp [St.wrZR, show r ≠ 31 by omega, reg_set_eq hlen hr]

private theorem wrZR_ne {s : St} (w : Nat) {r1 r2 : Nat} (hne : r1 ≠ r2) (v : Nat) :
    (s.wrZR w r1 v).reg r2 = s.reg r2 := by
  unfold St.wrZR
  split
  · rfl
  · exact reg_set_ne hne _

private theorem wrZR_31_eq {s : St} (w v : Nat) : s.wrZR w 31 v = s := by
  simp [St.wrZR]

private theorem wrZR_regs_pc {s : St} (w r v p : Nat) :
    { s with pc := p }.wrZR w r v = { s.wrZR w r v with pc := p } := by
  unfold St.wrZR
  split <;> rfl

private theorem wrZR_pc_len {s : St} (hlen : s.regs.length = 31) {p : Nat} (w r v : Nat) :
    ({ s with pc := p }.wrZR w r v).regs.length = 31 :=
  wrZR_len (s := { s with pc := p }) hlen w r v

private theorem wrZR_pc_eq {s : St} (hlen : s.regs.length = 31) (w : Nat) {r : Nat} (hr : r < 31)
    (v : Nat) {p : Nat} :
    ({ s with pc := p }.wrZR w r v).reg r = some (v % pow2 w) :=
  wrZR_eq (s := { s with pc := p }) hlen w hr v

private theorem wrZR_pc_ne {s : St} (w : Nat) {r1 r2 : Nat} (hne : r1 ≠ r2) {v p : Nat} :
    ({ s with pc := p }.wrZR w r1 v).reg r2 = s.reg r2 :=
  wrZR_ne (s := { s with pc := p }) w hne v

private theorem wrZR_pc_eq_val (s : St) (w r v : Nat) :
    (s.wrZR w r v).pc = s.pc := by
  unfold St.wrZR; split <;> rfl

private theorem wrSP_pc_eq_val (s : St) (w r v : Nat) :
    (s.wrSP w r v).pc = s.pc := by
  unfold St.wrSP; split <;> rfl

private theorem voterRegs_len (s : St) (a b c : Nat) :
    (TMRVoterArm64.voterRegs s a b c).length = 31 := rfl

private theorem tailInit_pc (base : Nat) (s : St) (a b c : Nat) :
    (tailInit base s a b c).pc = base := rfl

private theorem tailInit_regs_len (base : Nat) (s : St) (a b c : Nat) :
    (tailInit base s a b c).regs.length = 31 := rfl

private theorem mod_W_eq {x : Nat} (hx : x < AArch64Env.W) : x % pow2 64 = x :=
  Nat.mod_eq_of_lt hx

private theorem sub_eq_flags (v : Nat) :
    (addSubOp true 64 v v).2 = TMRVoterArm64.voterFlagsEq := by
  unfold addSubOp TMRVoterArm64.voterFlagsEq
  have hpow : pow2 64 = AArch64Env.W := rfl
  rw [hpow]; simp [msb]

private theorem sub_z_ne {x y : Nat} (hx : x < AArch64Env.W) (hy : y < AArch64Env.W) (hne : x ≠ y) :
    (addSubOp true 64 x y).2.z = some false := by
  unfold addSubOp
  have hpow : pow2 64 = AArch64Env.W := rfl
  rw [hpow]
  have hlt : x + AArch64Env.W - y < 2 * AArch64Env.W := by omega
  have hneW : x + AArch64Env.W - y ≠ AArch64Env.W := by omega
  have hnz : (x + AArch64Env.W - y) % AArch64Env.W ≠ 0 := by
    intro hmod
    obtain ⟨k, hk⟩ := Nat.dvd_of_mod_eq_zero hmod
    rcases k with _ | _ | k
    · simp at hk; omega
    · simp at hk; exact hneW hk
    · have hk2 : 2 ≤ k + 2 := by omega
      have hge : 2 * AArch64Env.W ≤ AArch64Env.W * (k + 2) := by
        simpa [Nat.mul_comm] using Nat.mul_le_mul_left AArch64Env.W hk2
      have hkeq : x + AArch64Env.W - y = AArch64Env.W * (k + 2) := hk
      omega
  simp [hnz]

private theorem add_base_mod_W {base off : Nat} (hb : base_ok base) (ho : off < 192) :
    (base + off) % AArch64Env.W = base + off := by
  rcases hb with ⟨_, _, hhi⟩
  exact Nat.mod_eq_of_lt (by omega)

private theorem step_at_word {base k : Nat} {ins : Ins} {s : St}
    (hb : base % 4 = 0) (hk : k < 48) (hpc : s.pc = base + 4 * k)
    (hi : (tailIns[k]?).map (·, 4) = some (ins, 4)) :
    step (tailImAt base) s = exec ins 4 s := by
  unfold step
  rw [hpc, tailImAt_ins hb hk, hi]

-- Voter 0..7 architectural proof (parametric in `base`)
private theorem tail_step0 {base : Nat} (hb : base_ok base) (s : St) {a b c : Nat}
    (ha : a < AArch64Env.W) (hbv : b < AArch64Env.W) :
    step (tailImAt base) (tailInit base s a b c) =
      .ok { s with regs := TMRVoterArm64.voterRegs s a b c, pc := base + 4, fl := (addSubOp true 64 a b).2 } := by
  rw [step_at_word hb.1 (k := 0) (by decide) (by rfl) (by rfl)]
  simp [exec, lift, bind, Option.bind, pure, St.rdZR, St.wrZR, St.reg, tailInit,
    TMRVoterArm64.voterRegs, shiftVal, show pow2 0 = 1 from rfl, mod_W_eq ha, mod_W_eq hbv,
    add_base_mod_W hb (by decide : 4 < 192)]

private theorem tail_step1 {base : Nat} (hb : base_ok base) (s : St) {a b c : Nat} {fl : Flags}
    (ha : a < AArch64Env.W) (hc : c < AArch64Env.W) (hz : ∃ z, fl.z = some z) :
    step (tailImAt base)
      { s with regs := TMRVoterArm64.voterRegs s a b c, pc := base + 4, fl := fl } =
      .ok { s with
        regs := TMRVoterArm64.voterRegs s a b c, pc := base + 8,
        fl := if fl.z == some true then TMRVoterArm64.voterFlagsEq else (addSubOp true 64 a c).2 } := by
  rcases hz with ⟨z, hz⟩
  rw [step_at_word hb.1 (k := 1) (by decide) (by rfl) (by rfl)]
  have hcond : AArch64Phys.cond fl 1 = some (!z) := by simp [AArch64Phys.cond, hz]
  have hbf6 : AArch64Phys.bf 6 3 1 = 0 ∧ AArch64Phys.bf 6 2 1 = 1 ∧
      AArch64Phys.bf 6 1 1 = 1 ∧ AArch64Phys.bf 6 0 1 = 0 := by decide
  cases z <;>
    simp [exec, hcond, lift, bind, Option.bind, pure, St.rdZR, St.reg, TMRVoterArm64.voterRegs,
      TMRVoterArm64.voterFlagsEq, hbf6, mod_W_eq ha, mod_W_eq hc, hz,
      show (base + 4 + 4) % AArch64Env.W = base + 8 by rw [Nat.add_assoc]; exact add_base_mod_W hb (by decide)]

private theorem tail_step2 {base : Nat} (hb : base_ok base) (s : St) {a b c : Nat} {fl : Flags}
    (ha : a < AArch64Env.W) (hbv : b < AArch64Env.W) (hz : ∃ z, fl.z = some z) :
    step (tailImAt base)
      { s with regs := TMRVoterArm64.voterRegs s a b c, pc := base + 8, fl := fl } =
      .ok { s with
        regs := TMRVoterArm64.voterRegs s (if fl.z == some true then a else b) b c,
        pc := base + 12, fl := fl } := by
  rcases hz with ⟨z, hz⟩
  rw [step_at_word hb.1 (k := 2) (by decide) (by rfl) (by rfl)]
  have hcond : AArch64Phys.cond fl 0 = some z := by simp [AArch64Phys.cond, hz]
  cases z <;>
    simp [exec, hcond, lift, bind, Option.bind, pure, St.rdZR, St.wrZR, St.reg, St.setReg,
      TMRVoterArm64.voterRegs, mod_W_eq ha, mod_W_eq hbv, hz,
      show (base + 8 + 4) % AArch64Env.W = base + 12 by rw [Nat.add_assoc]; exact add_base_mod_W hb (by decide)]

private theorem tail_step3 {base : Nat} (hb : base_ok base) (s : St) {v b c : Nat} {fl : Flags}
    (hbv : b < AArch64Env.W) (hc : c < AArch64Env.W) (hz : ∃ z, fl.z = some z) :
    step (tailImAt base)
      { s with regs := TMRVoterArm64.voterRegs s v b c, pc := base + 12, fl := fl } =
      .ok { s with
        regs := TMRVoterArm64.voterRegs s v b c, pc := base + 16,
        fl := if fl.z == some true then TMRVoterArm64.voterFlagsEq else (addSubOp true 64 b c).2 } := by
  rcases hz with ⟨z, hz⟩
  rw [step_at_word hb.1 (k := 3) (by decide) (by rfl) (by rfl)]
  have hcond : AArch64Phys.cond fl 1 = some (!z) := by simp [AArch64Phys.cond, hz]
  have hbf6 : AArch64Phys.bf 6 3 1 = 0 ∧ AArch64Phys.bf 6 2 1 = 1 ∧
      AArch64Phys.bf 6 1 1 = 1 ∧ AArch64Phys.bf 6 0 1 = 0 := by decide
  cases z <;>
    simp [exec, hcond, lift, bind, Option.bind, pure, St.rdZR, St.reg, TMRVoterArm64.voterRegs,
      TMRVoterArm64.voterFlagsEq, hbf6, mod_W_eq hbv, mod_W_eq hc, hz,
      show (base + 12 + 4) % AArch64Env.W = base + 16 by rw [Nat.add_assoc]; exact add_base_mod_W hb (by decide)]

private theorem tail_step4 {base : Nat} (hb : base_ok base) (s : St) {v b c : Nat} {fl : Flags}
    (hz : ∃ z, fl.z = some z) :
    step (tailImAt base)
      { s with regs := TMRVoterArm64.voterRegs s v b c, pc := base + 16, fl := fl } =
      .ok { s with
        regs := TMRVoterArm64.voterRegs s v b c,
        pc := if fl.z == some true then base + 20 else base + 32, fl := fl } := by
  rcases hz with ⟨z, hz⟩
  rw [step_at_word hb.1 (k := 4) (by decide) (by rfl) (by rfl)]
  have hcond : AArch64Phys.cond fl 1 = some (!z) := by simp [AArch64Phys.cond, hz]
  cases z <;>
    simp [exec, hcond, lift, bind, Option.bind, pure, hz,
      show (base + 16 + 4) % AArch64Env.W = base + 20 by rw [Nat.add_assoc]; exact add_base_mod_W hb (by decide),
      show (base + 16 + 16) % AArch64Env.W = base + 32 by rw [Nat.add_assoc]; exact add_base_mod_W hb (by decide)]

private theorem tail_step5 {base : Nat} (hb : base_ok base) (s : St) {v b c : Nat}
    (hv : v < AArch64Env.W) :
    step (tailImAt base)
      { s with regs := TMRVoterArm64.voterRegs s v b c, pc := base + 20, fl := TMRVoterArm64.voterFlagsEq } =
      .ok { s with regs := TMRVoterArm64.voterRegs s v v c, pc := base + 24, fl := TMRVoterArm64.voterFlagsEq } := by
  rw [step_at_word hb.1 (k := 5) (by decide) (by rfl) (by rfl)]
  have hvlt : v < 18446744073709551616 := hv
  simp [exec, lift, bind, Option.bind, pure, St.rdZR, St.wrZR, St.reg, St.setReg,
    TMRVoterArm64.voterRegs, shiftVal, pow2, hvlt,
    show (base + 20 + 4) % AArch64Env.W = base + 24 by rw [Nat.add_assoc]; exact add_base_mod_W hb (by decide)]

private theorem tail_step6 {base : Nat} (hb : base_ok base) (s : St) {v c : Nat}
    (hv : v < AArch64Env.W) :
    step (tailImAt base)
      { s with regs := TMRVoterArm64.voterRegs s v v c, pc := base + 24, fl := TMRVoterArm64.voterFlagsEq } =
      .ok { s with regs := TMRVoterArm64.voterRegs s v v v, pc := base + 28, fl := TMRVoterArm64.voterFlagsEq } := by
  rw [step_at_word hb.1 (k := 6) (by decide) (by rfl) (by rfl)]
  have hvlt : v < 18446744073709551616 := hv
  simp [exec, lift, bind, Option.bind, pure, St.rdZR, St.wrZR, St.reg, St.setReg,
    TMRVoterArm64.voterRegs, shiftVal, pow2, hvlt,
    show (base + 24 + 4) % AArch64Env.W = base + 28 by rw [Nat.add_assoc]; exact add_base_mod_W hb (by decide)]

private theorem tail_step7 {base : Nat} (hb : base_ok base) (s : St) {v : Nat} :
    step (tailImAt base)
      { s with regs := TMRVoterArm64.voterRegs s v v v, pc := base + 28, fl := TMRVoterArm64.voterFlagsEq } =
      .ok { s with regs := TMRVoterArm64.voterRegs s v v v, pc := base + 44, fl := TMRVoterArm64.voterFlagsEq } := by
  rw [step_at_word hb.1 (k := 7) (by decide) (by rfl) (by rfl)]
  simp [exec,
    show (base + 28 + 16) % AArch64Env.W = base + 44 by rw [Nat.add_assoc]; exact add_base_mod_W hb (by decide)]

theorem tail_arch_majority {base v a b c : Nat} (hb : base_ok base) (s : St)
    (ha : a < AArch64Env.W) (hbv : b < AArch64Env.W) (hc : c < AArch64Env.W)
    (hvote : TMR.vote a b c = some v) :
    run (tailImAt base) 8 (.ok (tailInit base s a b c)) =
      .ok { s with
        regs := TMRVoterArm64.voterRegs s v v v, pc := base + 44,
        fl := TMRVoterArm64.voterFlagsEq } := by
  have hv : v < AArch64Env.W := by
    unfold TMR.vote at hvote
    split at hvote
    · cases hvote; exact ha
    · split at hvote
      · cases hvote; exact hbv
      · cases hvote
  have h1_4 : run (tailImAt base) 5 (.ok (tailInit base s a b c)) =
      .ok { s with
        regs := TMRVoterArm64.voterRegs s v b c, pc := base + 20,
        fl := TMRVoterArm64.voterFlagsEq } := by
    by_cases hab : a = b
    · subst a; have hveq : v = b := by unfold TMR.vote at hvote; simp at hvote; exact hvote.symm
      subst v
      have s0 := tail_step0 hb s (c := c) hbv hbv; rw [sub_eq_flags] at s0
      have s1 := tail_step1 hb s (b := b) (c := c) (fl := TMRVoterArm64.voterFlagsEq) hbv hc ⟨true, rfl⟩
      rw [if_pos (by decide)] at s1
      have s2 := tail_step2 hb s (a := b) (b := b) (c := c) (fl := TMRVoterArm64.voterFlagsEq) hbv hbv ⟨true, rfl⟩
      rw [if_pos (by decide)] at s2
      have s3 := tail_step3 hb s (v := b) (b := b) (c := c) (fl := TMRVoterArm64.voterFlagsEq) hbv hc ⟨true, rfl⟩
      rw [if_pos (by decide)] at s3
      have s4 := tail_step4 hb s (v := b) (b := b) (c := c) (fl := TMRVoterArm64.voterFlagsEq) ⟨true, rfl⟩
      rw [if_pos (by decide)] at s4
      change run (tailImAt base) 4 (step (tailImAt base) _) = _; rw [s0]
      change run (tailImAt base) 3 (step (tailImAt base) _) = _; rw [s1]
      change run (tailImAt base) 2 (step (tailImAt base) _) = _; rw [s2]
      change run (tailImAt base) 1 (step (tailImAt base) _) = _; rw [s3]
      change run (tailImAt base) 0 (step (tailImAt base) _) = _; rw [s4]; rfl
    · have hzab : (addSubOp true 64 a b).2.z = some false := sub_z_ne ha hbv hab
      by_cases hac : a = c
      · subst a; have hveq : v = c := by unfold TMR.vote at hvote; simp at hvote; exact hvote.symm
        subst v
        have s0 := tail_step0 hb s (c := c) hc hbv
        have s1 := tail_step1 hb s (a := c) (b := b) (c := c) (fl := (addSubOp true 64 c b).2) hc hc ⟨false, hzab⟩
        rw [if_neg (by simp [hzab]), sub_eq_flags] at s1
        have s2 := tail_step2 hb s (a := c) (b := b) (c := c) (fl := TMRVoterArm64.voterFlagsEq) hc hbv ⟨true, rfl⟩
        rw [if_pos (by decide)] at s2
        have s3 := tail_step3 hb s (v := c) (b := b) (c := c) (fl := TMRVoterArm64.voterFlagsEq) hbv hc ⟨true, rfl⟩
        rw [if_pos (by decide)] at s3
        have s4 := tail_step4 hb s (v := c) (b := b) (c := c) (fl := TMRVoterArm64.voterFlagsEq) ⟨true, rfl⟩
        rw [if_pos (by decide)] at s4
        change run (tailImAt base) 4 (step (tailImAt base) _) = _; rw [s0]
        change run (tailImAt base) 3 (step (tailImAt base) _) = _; rw [s1]
        change run (tailImAt base) 2 (step (tailImAt base) _) = _; rw [s2]
        change run (tailImAt base) 1 (step (tailImAt base) _) = _; rw [s3]
        change run (tailImAt base) 0 (step (tailImAt base) _) = _; rw [s4]; rfl
      · have hbc : b = c := by unfold TMR.vote at hvote; simp [hab, hac] at hvote; exact hvote.1
        subst b; have hveq : v = c := by unfold TMR.vote at hvote; simp [hab] at hvote; exact hvote.symm
        subst v
        have hzac : (addSubOp true 64 a c).2.z = some false := sub_z_ne ha hc hac
        have s0 := tail_step0 hb s (c := c) ha hc
        have s1 := tail_step1 hb s (a := a) (b := c) (c := c) (fl := (addSubOp true 64 a c).2) ha hc ⟨false, hzab⟩
        rw [if_neg (by simp [hzab])] at s1
        have s2 := tail_step2 hb s (a := a) (b := c) (c := c) (fl := (addSubOp true 64 a c).2) ha hc ⟨false, hzac⟩
        rw [if_neg (by simp [hzac])] at s2
        have s3 := tail_step3 hb s (v := c) (b := c) (c := c) (fl := (addSubOp true 64 a c).2) hc hc ⟨false, hzac⟩
        rw [if_neg (by simp [hzac]), sub_eq_flags] at s3
        have s4 := tail_step4 hb s (v := c) (b := c) (c := c) (fl := TMRVoterArm64.voterFlagsEq) ⟨true, rfl⟩
        rw [if_pos (by decide)] at s4
        change run (tailImAt base) 4 (step (tailImAt base) _) = _; rw [s0]
        change run (tailImAt base) 3 (step (tailImAt base) _) = _; rw [s1]
        change run (tailImAt base) 2 (step (tailImAt base) _) = _; rw [s2]
        change run (tailImAt base) 1 (step (tailImAt base) _) = _; rw [s3]
        change run (tailImAt base) 0 (step (tailImAt base) _) = _; rw [s4]; rfl
  rw [show (8 : Nat) = 5 + 3 from rfl, run_add, h1_4]
  change run (tailImAt base) 2 (step (tailImAt base) _) = _; rw [tail_step5 hb s hv]
  change run (tailImAt base) 1 (step (tailImAt base) _) = _; rw [tail_step6 hb s hv]
  change run (tailImAt base) 0 (step (tailImAt base) _) = _; rw [tail_step7 hb s]; rfl

theorem tail_arch_refuse {base a b c : Nat} (hb : base_ok base) (s : St)
    (ha : a < AArch64Env.W) (hbv : b < AArch64Env.W) (hc : c < AArch64Env.W)
    (hvote : TMR.vote a b c = none) :
    run (tailImAt base) 8 (.ok (tailInit base s a b c)) = .exit 3 := by
  have hne : a ≠ b ∧ a ≠ c ∧ b ≠ c := by
    unfold TMR.vote at hvote; repeat' split at hvote <;> simp_all
  have hzab := sub_z_ne ha hbv hne.1
  have hzac := sub_z_ne ha hc hne.2.1
  have hzbc := sub_z_ne hbv hc hne.2.2
  have s0 := tail_step0 hb s (c := c) ha hbv
  have s1 := tail_step1 hb s (a := a) (b := b) (c := c) (fl := (addSubOp true 64 a b).2) ha hc ⟨false, hzab⟩; simp [hzab] at s1
  have s2 := tail_step2 hb s (a := a) (b := b) (c := c) (fl := (addSubOp true 64 a c).2) ha hbv ⟨false, hzac⟩; simp [hzac] at s2
  have s3 := tail_step3 hb s (v := b) (b := b) (c := c) (fl := (addSubOp true 64 a c).2) hbv hc ⟨false, hzac⟩; simp [hzac] at s3
  have s4 := tail_step4 hb s (v := b) (b := b) (c := c) (fl := (addSubOp true 64 b c).2) ⟨false, hzbc⟩; simp [hzbc] at s4
  have h_exit3 : ∀ xfl xr (hxr : xr.length = 31),
      run (tailImAt base) 3 (.ok { s with regs := xr, pc := base + 32, fl := xfl }) = .exit 3 := by
    intro xfl xr hxr
    let st8 : St := { s with regs := xr, pc := base + 32, fl := xfl }
    have s8 : step (tailImAt base) st8 =
        .ok ({ st8 with pc := base + 36 }.wrZR 64 0 3) := by
      rw [step_at_word hb.1 (k := 8) (by decide) (by rfl) (by rfl)]
      simp [st8, exec, pow2,
        show (base + 32 + 4) % AArch64Env.W = base + 36 by rw [Nat.add_assoc]; exact add_base_mod_W hb (by decide)]
    let st9 : St := ({ st8 with pc := base + 36 }.wrZR 64 0 3)
    have hlen9 : st9.regs.length = 31 := wrZR_pc_len hxr _ _ _
    have hx0_9 : st9.reg 0 = some 3 := by
      simpa [st9, pow2] using wrZR_pc_eq hxr 64 (by decide) 3
    have hpc9 : st9.pc = base + 36 := wrZR_pc_eq_val _ _ _ _
    have s9 : step (tailImAt base) st9 =
        .ok ({ st9 with pc := base + 40 }.wrZR 64 8 93) := by
      rw [step_at_word hb.1 (k := 9) (by decide) hpc9 (by rfl)]
      simp [st9, exec, pow2, hpc9,
        show (base + 36 + 4) % AArch64Env.W = base + 40 by rw [Nat.add_assoc]; exact add_base_mod_W hb (by decide)]
    let st10 : St := ({ st9 with pc := base + 40 }.wrZR 64 8 93)
    have hx0_10 : st10.reg 0 = some 3 :=
      (wrZR_pc_ne 64 (by decide : (8 : Nat) ≠ 0)).trans hx0_9
    have hx8_10 : st10.reg 8 = some 93 := by
      simpa [st10, pow2] using wrZR_pc_eq hlen9 64 (by decide) 93
    have hpc10 : st10.pc = base + 40 := wrZR_pc_eq_val _ _ _ _
    have s10 : step (tailImAt base) st10 = .exit 3 := by
      rw [step_at_word hb.1 (k := 10) (by decide) hpc10 (by rfl)]
      simp [st10, exec, AArch64Phys.execSvc, AArch64Phys.applySys, AArch64Env.syscall, hx0_10, hx8_10]
    change run (tailImAt base) 2 (step (tailImAt base) st8) = .exit 3
    rw [s8]
    change run (tailImAt base) 1 (step (tailImAt base) st9) = .exit 3
    rw [s9]
    change run (tailImAt base) 0 (step (tailImAt base) st10) = .exit 3
    rw [s10]; rfl
  change run (tailImAt base) 7 (step (tailImAt base) (tailInit base s a b c)) = _; rw [s0]
  change run (tailImAt base) 6 (step (tailImAt base) _) = _; rw [s1]
  change run (tailImAt base) 5 (step (tailImAt base) _) = _; rw [s2]
  change run (tailImAt base) 4 (step (tailImAt base) _) = _; rw [s3]
  change run (tailImAt base) 3 (step (tailImAt base) _) = _; rw [s4]
  exact h_exit3 _ _ (voterRegs_len s b b c)

-- Elementary step helpers on `tailImAt base` for architectural and `TailSafe` reasoning.

private theorem step_movz_64 {base k rd imm hw : Nat} (hb : base_ok base) (hk : k < 48)
    (hk4 : 4 * k + 4 < 192)
    (hi : (tailIns[k]?).map (·, 4) = some (.movWide 2 64 rd imm hw, 4))
    (s : St) (hpc : s.pc = base + 4 * k) :
    step (tailImAt base) s =
      .ok ({ s with pc := base + 4 * (k + 1) }.wrZR 64 rd (imm * pow2 (hw * 16))) := by
  rw [step_at_word hb.1 hk hpc hi]
  simp [exec, hpc, show (base + 4 * k + 4) % AArch64Env.W = base + 4 * (k + 1) by
    rw [show base + 4 * k + 4 = base + (4 * k + 4) by omega, add_base_mod_W hb hk4]; omega]

private theorem step_movk_64 {base k rd imm hw old : Nat} (hb : base_ok base) (hk : k < 48)
    (hk4 : 4 * k + 4 < 192)
    (hi : (tailIns[k]?).map (·, 4) = some (.movWide 3 64 rd imm hw, 4))
    (s : St) (hpc : s.pc = base + 4 * k) (hrd : rd < 31) (hold : s.reg rd = some old) :
    step (tailImAt base) s =
      .ok ({ s with pc := base + 4 * (k + 1) }.wrZR 64 rd
        ((old % pow2 64) % pow2 (hw * 16) + imm * pow2 (hw * 16) +
          (old % pow2 64) / pow2 (hw * 16 + 16) * pow2 (hw * 16 + 16))) := by
  rw [step_at_word hb.1 hk hpc hi]
  simp [exec, lift, bind, Option.bind, pure, St.rdZR, show rd ≠ 31 by omega, hold, hpc,
    show (base + 4 * k + 4) % AArch64Env.W = base + 4 * (k + 1) by
      rw [show base + 4 * k + 4 = base + (4 * k + 4) by omega, add_base_mod_W hb hk4]; omega]

private theorem step_movk_64_lt {base k rd imm hw old : Nat} (hb : base_ok base) (hk : k < 48)
    (hk4 : 4 * k + 4 < 192)
    (hi : (tailIns[k]?).map (·, 4) = some (.movWide 3 64 rd imm hw, 4))
    (s : St) (hpc : s.pc = base + 4 * k) (hrd : rd < 31) (hold : s.reg rd = some old)
    (holt : old < AArch64Env.W) :
    step (tailImAt base) s =
      .ok ({ s with pc := base + 4 * (k + 1) }.wrZR 64 rd
        (old % pow2 (hw * 16) + imm * pow2 (hw * 16) +
          old / pow2 (hw * 16 + 16) * pow2 (hw * 16 + 16))) := by
  simpa [mod_W_eq holt] using step_movk_64 hb hk hk4 hi s hpc hrd hold

private theorem step_orr_64 {base k rd rn rm a b : Nat} (hb : base_ok base) (hk : k < 48)
    (hk4 : 4 * k + 4 < 192)
    (hi : (tailIns[k]?).map (·, 4) = some (.logicReg .orr false 64 rd rn rm .lsl 0, 4))
    (s : St) (hpc : s.pc = base + 4 * k) (hrn : rn < 31) (hrm : rm < 31)
    (ha : s.reg rn = some a) (hbv : s.reg rm = some b) :
    step (tailImAt base) s =
      .ok ({ s with pc := base + 4 * (k + 1) }.wrZR 64 rd (Nat.lor (a % pow2 64) (b % pow2 64))) := by
  rw [step_at_word hb.1 hk hpc hi]
  simp [exec, lift, bind, Option.bind, pure, St.rdZR, show rn ≠ 31 by omega,
    show rm ≠ 31 by omega, ha, hbv, shiftVal, pow2, hpc,
    show (base + 4 * k + 4) % AArch64Env.W = base + 4 * (k + 1) by
      rw [show base + 4 * k + 4 = base + (4 * k + 4) by omega, add_base_mod_W hb hk4]; omega]

private theorem step_cmp_64 {base k rn rm a b : Nat} (hb : base_ok base) (hk : k < 48)
    (hk4 : 4 * k + 4 < 192)
    (hi : (tailIns[k]?).map (·, 4) = some (.addSubReg true true 64 31 rn rm .lsl 0, 4))
    (s : St) (hpc : s.pc = base + 4 * k)
    (hrn : rn < 31) (hrm : rm < 31) (ha : s.reg rn = some a) (hbv : s.reg rm = some b) :
    step (tailImAt base) s =
      .ok { s with pc := base + 4 * (k + 1), fl := (addSubOp true 64 (a % pow2 64) (b % pow2 64)).2 } := by
  rw [step_at_word hb.1 hk hpc hi]
  simp [exec, lift, bind, Option.bind, pure, St.rdZR, show rn ≠ 31 by omega,
    show rm ≠ 31 by omega, ha, hbv, shiftVal, pow2, wrZR_31_eq, hpc,
    show (base + 4 * k + 4) % AArch64Env.W = base + 4 * (k + 1) by
      rw [show base + 4 * k + 4 = base + (4 * k + 4) by omega, add_base_mod_W hb hk4]; omega]

private theorem step_b_rel {base k off : Nat} (hb : base_ok base) (hk : k < 48)
    (hi : (tailIns[k]?).map (·, 4) = some (.b false off, 4))
    (ho : 4 * k + off < 192)
    (s : St) (hpc : s.pc = base + 4 * k) :
    step (tailImAt base) s = .ok { s with pc := base + (4 * k + off) } := by
  rw [step_at_word hb.1 hk hpc hi]
  simp [exec, hpc, show (base + 4 * k + off) % AArch64Env.W = base + (4 * k + off) by
    rw [Nat.add_assoc]; exact add_base_mod_W hb ho]

private theorem step_bcond_ne {base k off : Nat} (hb : base_ok base) (hk : k < 48)
    (hk4 : 4 * k + 4 < 192)
    (hi : (tailIns[k]?).map (·, 4) = some (.bcond 1 off, 4))
    (ho : 4 * k + off < 192)
    (s : St) (hpc : s.pc = base + 4 * k) {z : Bool} (hz : s.fl.z = some z) :
    step (tailImAt base) s =
      .ok { s with pc := if z then base + 4 * (k + 1) else base + (4 * k + off) } := by
  rw [step_at_word hb.1 hk hpc hi]
  have hcond : AArch64Phys.cond s.fl 1 = some (!z) := by simp [AArch64Phys.cond, hz]
  cases z <;>
    simp [exec, hcond, lift, bind, Option.bind, pure, hpc,
      show (base + 4 * k + 4) % AArch64Env.W = base + 4 * (k + 1) by
        rw [show base + 4 * k + 4 = base + (4 * k + 4) by omega, add_base_mod_W hb hk4]; omega,
      show (base + 4 * k + off) % AArch64Env.W = base + (4 * k + off) by
        rw [Nat.add_assoc]; exact add_base_mod_W hb ho]

private theorem step_subI_64 {base k rd rn imm a : Nat} (hb : base_ok base) (hk : k < 48)
    (hk4 : 4 * k + 4 < 192)
    (hi : (tailIns[k]?).map (·, 4) = some (.addSubImm true false 64 rd rn imm, 4))
    (s : St) (hpc : s.pc = base + 4 * k) (hrd : rd < 31) (hrn : rn < 31)
    (ha : s.reg rn = some a) (halt : a < AArch64Env.W) :
    step (tailImAt base) s =
      .ok ({ s with pc := base + 4 * (k + 1) }.wrZR 64 rd (addSubOp true 64 a (imm % pow2 64)).1) := by
  rw [step_at_word hb.1 hk hpc hi]
  simp [exec, lift, bind, Option.bind, pure, St.rdSP, St.wrSP, St.wrZR,
    show rn ≠ 31 by omega, show rd ≠ 31 by omega, ha, hpc, mod_W_eq halt,
    show (base + 4 * k + 4) % AArch64Env.W = base + 4 * (k + 1) by
      rw [show base + 4 * k + 4 = base + (4 * k + 4) by omega, add_base_mod_W hb hk4]; omega]

private theorem step_svc_exit {base k x0v : Nat} (hb : base_ok base) (hk : k < 48)
    (hi : (tailIns[k]?).map (·, 4) = some (.svc 0, 4))
    (s : St) (hpc : s.pc = base + 4 * k)
    (hx0 : s.reg 0 = some x0v) (hx8 : s.reg 8 = some 93) :
    step (tailImAt base) s = .exit (x0v % 256) := by
  rw [step_at_word hb.1 hk hpc hi]
  simp [exec, AArch64Phys.execSvc, AArch64Phys.applySys, AArch64Env.syscall, hx0, hx8]

private theorem step_movz_64_val {base k k' rd imm hw val : Nat} (hb : base_ok base) (hk : k < 48)
    (hk4 : 4 * k + 4 < 192) (hk' : k + 1 = k')
    (hi : (tailIns[k]?).map (·, 4) = some (.movWide 2 64 rd imm hw, 4))
    (s : St) (hpc : s.pc = base + 4 * k) (hval : imm * pow2 (hw * 16) = val) :
    step (tailImAt base) s = .ok ({ s with pc := base + 4 * k' }.wrZR 64 rd val) := by
  subst hk' hval; exact step_movz_64 hb hk hk4 hi s hpc

private theorem step_movk_64_val {base k k' rd imm hw old val : Nat} (hb : base_ok base) (hk : k < 48)
    (hk4 : 4 * k + 4 < 192) (hk' : k + 1 = k')
    (hi : (tailIns[k]?).map (·, 4) = some (.movWide 3 64 rd imm hw, 4))
    (s : St) (hpc : s.pc = base + 4 * k) (hrd : rd < 31) (hold : s.reg rd = some old)
    (holt : old < AArch64Env.W)
    (hval : old % pow2 (hw * 16) + imm * pow2 (hw * 16) +
        old / pow2 (hw * 16 + 16) * pow2 (hw * 16 + 16) = val) :
    step (tailImAt base) s = .ok ({ s with pc := base + 4 * k' }.wrZR 64 rd val) := by
  subst hk' hval; exact step_movk_64_lt hb hk hk4 hi s hpc hrd hold holt

private theorem step_orr_64_val {base k k' rd rn rm a b val : Nat} (hb : base_ok base) (hk : k < 48)
    (hk4 : 4 * k + 4 < 192) (hk' : k + 1 = k')
    (hi : (tailIns[k]?).map (·, 4) = some (.logicReg .orr false 64 rd rn rm .lsl 0, 4))
    (s : St) (hpc : s.pc = base + 4 * k) (hrn : rn < 31) (hrm : rm < 31)
    (ha : s.reg rn = some a) (hbv : s.reg rm = some b)
    (hval : Nat.lor (a % pow2 64) (b % pow2 64) = val) :
    step (tailImAt base) s = .ok ({ s with pc := base + 4 * k' }.wrZR 64 rd val) := by
  subst hk' hval; exact step_orr_64 hb hk hk4 hi s hpc hrn hrm ha hbv

private theorem step_cmp_64_k {base k k' rn rm a b : Nat} (hb : base_ok base) (hk : k < 48)
    (hk4 : 4 * k + 4 < 192) (hk' : k + 1 = k')
    (hi : (tailIns[k]?).map (·, 4) = some (.addSubReg true true 64 31 rn rm .lsl 0, 4))
    (s : St) (hpc : s.pc = base + 4 * k)
    (hrn : rn < 31) (hrm : rm < 31) (ha : s.reg rn = some a) (hbv : s.reg rm = some b) :
    step (tailImAt base) s =
      .ok { s with pc := base + 4 * k', fl := (addSubOp true 64 (a % pow2 64) (b % pow2 64)).2 } := by
  subst hk'; exact step_cmp_64 hb hk hk4 hi s hpc hrn hrm ha hbv

private theorem step_subI_64_val {base k k' rd rn imm a val : Nat} (hb : base_ok base) (hk : k < 48)
    (hk4 : 4 * k + 4 < 192) (hk' : k + 1 = k')
    (hi : (tailIns[k]?).map (·, 4) = some (.addSubImm true false 64 rd rn imm, 4))
    (s : St) (hpc : s.pc = base + 4 * k) (hrd : rd < 31) (hrn : rn < 31)
    (ha : s.reg rn = some a) (halt : a < AArch64Env.W)
    (hval : (addSubOp true 64 a (imm % pow2 64)).1 = val) :
    step (tailImAt base) s = .ok ({ s with pc := base + 4 * k' }.wrZR 64 rd val) := by
  subst hk' hval; exact step_subI_64 hb hk hk4 hi s hpc hrd hrn ha halt

-- Architectural guard path from pc = base + 44 (`k = 11`) when x19 = x20 = v < W.
private theorem guard_arch_run {base v : Nat} (hb : base_ok base) (s : St) (hlen : s.regs.length = 31)
    (hv : v < AArch64Env.W) (h19 : s.reg 19 = some v) (h20 : s.reg 20 = some v)
    (hpc : s.pc = base + 44) :
    run (tailImAt base) 21 (.ok s) =
      .exit (if v = ACCEPT then 0 else if v = OVERFLOW then 6 else 5) := by
  -- Step 11..20 (guard prefix + guard 1)
  let t12 := ({ s with pc := base + 4 * 12 }.wrZR 64 0 5)
  have s11 : step (tailImAt base) s = .ok t12 :=
    step_movz_64_val hb (k := 11) (k' := 12) (by decide) (by decide) rfl (by rfl) s hpc (by decide)
  have h12_len : t12.regs.length = 31 := wrZR_pc_len hlen _ _ _
  let t13 := ({ t12 with pc := base + 4 * 13 }.wrZR 64 8 0)
  have s12 : step (tailImAt base) t12 = .ok t13 :=
    step_movz_64_val hb (k := 12) (k' := 13) (by decide) (by decide) rfl (by rfl) t12 rfl (by decide)
  have h13_len : t13.regs.length = 31 := wrZR_pc_len h12_len _ _ _
  let t14 := ({ t13 with pc := base + 4 * 14 }.wrZR 64 2 0)
  have s13 : step (tailImAt base) t13 = .ok t14 :=
    step_movz_64_val hb (k := 13) (k' := 14) (by decide) (by decide) rfl (by rfl) t13 rfl (by decide)
  have h14_len : t14.regs.length = 31 := wrZR_pc_len h13_len _ _ _
  let t15 := ({ t14 with pc := base + 4 * 15 }.wrZR 64 7 0)
  have s14 : step (tailImAt base) t14 = .ok t15 :=
    step_movz_64_val hb (k := 14) (k' := 15) (by decide) (by decide) rfl (by rfl) t14 rfl (by decide)
  have h15_len : t15.regs.length = 31 := wrZR_pc_len h14_len _ _ _
  let t16 := ({ t15 with pc := base + 4 * 16 }.wrZR 64 1 0x3c96)
  have s15 : step (tailImAt base) t15 = .ok t16 :=
    step_movz_64_val hb (k := 15) (k' := 16) (by decide) (by decide) rfl (by rfl) t15 rfl (by decide)
  have h16_len : t16.regs.length = 31 := wrZR_pc_len h15_len _ _ _
  have h16_x1 : t16.reg 1 = some 0x3c96 := by
    simpa [t16, pow2] using wrZR_pc_eq h15_len 64 (by decide : (1 : Nat) < 31) 0x3c96
  let t17 := ({ t16 with pc := base + 4 * 17 }.wrZR 64 1 0xa55a3c96)
  have s16 : step (tailImAt base) t16 = .ok t17 :=
    step_movk_64_val hb (k := 16) (k' := 17) (by decide) (by decide) rfl (by rfl) t16 rfl (by decide) h16_x1 (by decide) (by decide)
  have h17_len : t17.regs.length = 31 := wrZR_pc_len h16_len _ _ _
  have h17_x1 : t17.reg 1 = some 0xa55a3c96 := by
    simpa [t17, pow2] using wrZR_pc_eq h16_len 64 (by decide : (1 : Nat) < 31) 0xa55a3c96
  let t18 := ({ t17 with pc := base + 4 * 18 }.wrZR 64 1 0xe970a55a3c96)
  have s17 : step (tailImAt base) t17 = .ok t18 :=
    step_movk_64_val hb (k := 17) (k' := 18) (by decide) (by decide) rfl (by rfl) t17 rfl (by decide) h17_x1 (by decide) (by decide)
  have h18_len : t18.regs.length = 31 := wrZR_pc_len h17_len _ _ _
  have h18_x1 : t18.reg 1 = some 0xe970a55a3c96 := by
    simpa [t18, pow2] using wrZR_pc_eq h17_len 64 (by decide : (1 : Nat) < 31) 0xe970a55a3c96
  let t19 := ({ t18 with pc := base + 4 * 19 }.wrZR 64 1 ACCEPT)
  have s18 : step (tailImAt base) t18 = .ok t19 :=
    step_movk_64_val hb (k := 18) (k' := 19) (by decide) (by decide) rfl (by rfl) t18 rfl (by decide) h18_x1 (by decide) (by decide)
  have h19_len : t19.regs.length = 31 := wrZR_pc_len h18_len _ _ _
  have h19_x1 : t19.reg 1 = some ACCEPT := by
    simpa [t19, ACCEPT, pow2] using wrZR_pc_eq h18_len 64 (by decide : (1 : Nat) < 31) ACCEPT
  have h19_x19 : t19.reg 19 = some v := by
    dsimp only [t19, t18, t17, t16, t15, t14, t13, t12]
    repeat rw [wrZR_pc_ne _ (by decide)]
    exact h19
  have h19_x20 : t19.reg 20 = some v := by
    dsimp only [t19, t18, t17, t16, t15, t14, t13, t12]
    repeat rw [wrZR_pc_ne _ (by decide)]
    exact h20
  have s19 := step_cmp_64_k hb (k := 19) (k' := 20) (by decide) (by decide) rfl (by rfl) t19 rfl (by decide) (by decide) h19_x19 h19_x1
  have hacc_lt : ACCEPT < AArch64Env.W := by decide
  by_cases heq : v = ACCEPT
  · subst heq
    have hfl20 : (addSubOp true 64 (ACCEPT % pow2 64) (ACCEPT % pow2 64)).2 = TMRVoterArm64.voterFlagsEq := by
      rw [mod_W_eq hacc_lt, sub_eq_flags]
    let t20 : St := { t19 with pc := base + 4 * 20, fl := TMRVoterArm64.voterFlagsEq }
    have h20_len : t20.regs.length = 31 := h19_len
    have s20 := step_bcond_ne hb (k := 20) (by decide) (by decide) (by rfl) (by decide) t20 rfl (z := true) rfl
    rw [if_pos rfl, show 4 * (20 + 1) = 4 * 21 by omega] at s20
    let t22 := ({ t20 with pc := base + 4 * 22 }.wrZR 64 2 0)
    have s21 : step (tailImAt base) { t20 with pc := base + 4 * 21 } = .ok t22 :=
      step_movz_64_val hb (k := 21) (k' := 22) (by decide) (by decide) rfl (by rfl) { t20 with pc := base + 4 * 21 } rfl (by decide)
    have h22_len : t22.regs.length = 31 := wrZR_pc_len h20_len _ _ _
    let t23 := ({ t22 with pc := base + 4 * 23 }.wrZR 64 6 0x3c96)
    have s22 : step (tailImAt base) t22 = .ok t23 :=
      step_movz_64_val hb (k := 22) (k' := 23) (by decide) (by decide) rfl (by rfl) t22 rfl (by decide)
    have h23_len : t23.regs.length = 31 := wrZR_pc_len h22_len _ _ _
    have h23_x6 : t23.reg 6 = some 0x3c96 := by
      simpa [t23, pow2] using wrZR_pc_eq h22_len 64 (by decide : (6 : Nat) < 31) 0x3c96
    let t24 := ({ t23 with pc := base + 4 * 24 }.wrZR 64 6 0xa55a3c96)
    have s23 : step (tailImAt base) t23 = .ok t24 :=
      step_movk_64_val hb (k := 23) (k' := 24) (by decide) (by decide) rfl (by rfl) t23 rfl (by decide) h23_x6 (by decide) (by decide)
    have h24_len : t24.regs.length = 31 := wrZR_pc_len h23_len _ _ _
    have h24_x6 : t24.reg 6 = some 0xa55a3c96 := by
      simpa [t24, pow2] using wrZR_pc_eq h23_len 64 (by decide : (6 : Nat) < 31) 0xa55a3c96
    let t25 := ({ t24 with pc := base + 4 * 25 }.wrZR 64 6 0xe970a55a3c96)
    have s24 : step (tailImAt base) t24 = .ok t25 :=
      step_movk_64_val hb (k := 24) (k' := 25) (by decide) (by decide) rfl (by rfl) t24 rfl (by decide) h24_x6 (by decide) (by decide)
    have h25_len : t25.regs.length = 31 := wrZR_pc_len h24_len _ _ _
    have h25_x6 : t25.reg 6 = some 0xe970a55a3c96 := by
      simpa [t25, pow2] using wrZR_pc_eq h24_len 64 (by decide : (6 : Nat) < 31) 0xe970a55a3c96
    let t26 := ({ t25 with pc := base + 4 * 26 }.wrZR 64 6 ACCEPT)
    have s25 : step (tailImAt base) t25 = .ok t26 :=
      step_movk_64_val hb (k := 25) (k' := 26) (by decide) (by decide) rfl (by rfl) t25 rfl (by decide) h25_x6 (by decide) (by decide)
    have h26_len : t26.regs.length = 31 := wrZR_pc_len h25_len _ _ _
    have h26_x6 : t26.reg 6 = some ACCEPT := by
      simpa [t26, ACCEPT, pow2] using wrZR_pc_eq h25_len 64 (by decide : (6 : Nat) < 31) ACCEPT
    have h26_x20 : t26.reg 20 = some ACCEPT := by
      dsimp only [t26, t25, t24, t23, t22]
      repeat rw [wrZR_pc_ne _ (by decide)]
      exact h19_x20
    have s26 := step_cmp_64_k hb (k := 26) (k' := 27) (by decide) (by decide) rfl (by rfl) t26 rfl (by decide) (by decide) h26_x20 h26_x6
    rw [hfl20] at s26
    let t27 : St := { t26 with pc := base + 4 * 27, fl := TMRVoterArm64.voterFlagsEq }
    have h27_len : t27.regs.length = 31 := h26_len
    have s27 := step_bcond_ne hb (k := 27) (by decide) (by decide) (by rfl) (by decide) t27 rfl (z := true) rfl
    rw [if_pos rfl, show 4 * (27 + 1) = 4 * 28 by omega] at s27
    let t29 := ({ t27 with pc := base + 4 * 29 }.wrZR 64 8 93)
    have s28 : step (tailImAt base) { t27 with pc := base + 4 * 28 } = .ok t29 :=
      step_movz_64_val hb (k := 28) (k' := 29) (by decide) (by decide) rfl (by rfl) { t27 with pc := base + 4 * 28 } rfl (by decide)
    have h29_len : t29.regs.length = 31 := wrZR_pc_len h27_len _ _ _
    have h29_x20 : t29.reg 20 = some ACCEPT := by
      dsimp only [t29]
      rw [wrZR_pc_ne _ (by decide)]
      exact h26_x20
    have h29_x8 : t29.reg 8 = some 93 := by
      simpa [t29, pow2] using wrZR_pc_eq h27_len 64 (by decide : (8 : Nat) < 31) 93
    let t30 := ({ t29 with pc := base + 4 * 30 }.wrZR 64 0 (ACCEPT - 150))
    have s29 : step (tailImAt base) t29 = .ok t30 :=
      step_subI_64_val hb (k := 29) (k' := 30) (by decide) (by decide) rfl (by rfl) t29 rfl (by decide) (by decide) h29_x20 hacc_lt (by decide)
    have h30_x0 : t30.reg 0 = some (ACCEPT - 150) := by
      simpa [t30, ACCEPT, pow2] using wrZR_pc_eq h29_len 64 (by decide : (0 : Nat) < 31) (ACCEPT - 150)
    have h30_x8 : t30.reg 8 = some 93 := by
      dsimp only [t30]
      rw [wrZR_pc_ne _ (by decide)]
      exact h29_x8
    have s30 := step_svc_exit hb (k := 30) (by decide) (by rfl) t30 rfl h30_x0 h30_x8
    have hmod0 : (ACCEPT - 150) % 256 = 0 := by decide
    rw [hmod0] at s30
    simp only [↓reduceIte]
    have h20run : run (tailImAt base) 20 (.ok s) = .exit 0 := by
      change run (tailImAt base) 19 (step (tailImAt base) _) = _; rw [s11]
      change run (tailImAt base) 18 (step (tailImAt base) _) = _; rw [s12]
      change run (tailImAt base) 17 (step (tailImAt base) _) = _; rw [s13]
      change run (tailImAt base) 16 (step (tailImAt base) _) = _; rw [s14]
      change run (tailImAt base) 15 (step (tailImAt base) _) = _; rw [s15]
      change run (tailImAt base) 14 (step (tailImAt base) _) = _; rw [s16]
      change run (tailImAt base) 13 (step (tailImAt base) _) = _; rw [s17]
      change run (tailImAt base) 12 (step (tailImAt base) _) = _; rw [s18]
      change run (tailImAt base) 11 (step (tailImAt base) _) = _; rw [s19, hfl20]
      change run (tailImAt base) 10 (step (tailImAt base) _) = _; rw [s20]
      change run (tailImAt base) 9 (step (tailImAt base) _) = _; rw [s21]
      change run (tailImAt base) 8 (step (tailImAt base) _) = _; rw [s22]
      change run (tailImAt base) 7 (step (tailImAt base) _) = _; rw [s23]
      change run (tailImAt base) 6 (step (tailImAt base) _) = _; rw [s24]
      change run (tailImAt base) 5 (step (tailImAt base) _) = _; rw [s25]
      change run (tailImAt base) 4 (step (tailImAt base) _) = _; rw [s26]
      change run (tailImAt base) 3 (step (tailImAt base) _) = _; rw [s27]
      change run (tailImAt base) 2 (step (tailImAt base) _) = _; rw [s28]
      change run (tailImAt base) 1 (step (tailImAt base) _) = _; rw [s29]
      change run (tailImAt base) 0 (step (tailImAt base) _) = _; rw [s30]; rfl
    exact run_ge_exit h20run (by decide)
  · -- Not ACCEPT -> jumps at 20 to 32 (not_acc)
    have hz_ne : (addSubOp true 64 (v % pow2 64) (ACCEPT % pow2 64)).2.z = some false := by
      rw [mod_W_eq hv, mod_W_eq hacc_lt]; exact sub_z_ne hv hacc_lt heq
    let t20 : St := { t19 with pc := base + 4 * 20, fl := (addSubOp true 64 (v % pow2 64) (ACCEPT % pow2 64)).2 }
    have s20 := step_bcond_ne hb (k := 20) (by decide) (by decide) (by rfl) (by decide) t20 rfl (z := false) hz_ne
    have hnot_pc : base + (4 * 20 + 48) = base + 4 * 32 := by omega
    rw [if_neg (by decide), hnot_pc] at s20
    let t32 : St := { t20 with pc := base + 4 * 32 }
    have h32_len : t32.regs.length = 31 := h19_len
    let t33 := ({ t32 with pc := base + 4 * 33 }.wrZR 64 1 0xc3a5)
    have s32 : step (tailImAt base) t32 = .ok t33 :=
      step_movz_64_val hb (k := 32) (k' := 33) (by decide) (by decide) rfl (by rfl) t32 rfl (by decide)
    have h33_len : t33.regs.length = 31 := wrZR_pc_len h32_len _ _ _
    have h33_x1 : t33.reg 1 = some 0xc3a5 := by
      simpa [t33, pow2] using wrZR_pc_eq h32_len 64 (by decide : (1 : Nat) < 31) 0xc3a5
    let t34 := ({ t33 with pc := base + 4 * 34 }.wrZR 64 1 0x33c3c3a5)
    have s33 : step (tailImAt base) t33 = .ok t34 :=
      step_movk_64_val hb (k := 33) (k' := 34) (by decide) (by decide) rfl (by rfl) t33 rfl (by decide) h33_x1 (by decide) (by decide)
    have h34_len : t34.regs.length = 31 := wrZR_pc_len h33_len _ _ _
    have h34_x1 : t34.reg 1 = some 0x33c3c3a5 := by
      simpa [t34, pow2] using wrZR_pc_eq h33_len 64 (by decide : (1 : Nat) < 31) 0x33c3c3a5
    let t35 := ({ t34 with pc := base + 4 * 35 }.wrZR 64 1 0x0f1e33c3c3a5)
    have s34 : step (tailImAt base) t34 = .ok t35 :=
      step_movk_64_val hb (k := 34) (k' := 35) (by decide) (by decide) rfl (by rfl) t34 rfl (by decide) h34_x1 (by decide) (by decide)
    have h35_len : t35.regs.length = 31 := wrZR_pc_len h34_len _ _ _
    have h35_x1 : t35.reg 1 = some 0x0f1e33c3c3a5 := by
      simpa [t35, pow2] using wrZR_pc_eq h34_len 64 (by decide : (1 : Nat) < 31) 0x0f1e33c3c3a5
    let t36 := ({ t35 with pc := base + 4 * 36 }.wrZR 64 1 OVERFLOW)
    have s35 : step (tailImAt base) t35 = .ok t36 :=
      step_movk_64_val hb (k := 35) (k' := 36) (by decide) (by decide) rfl (by rfl) t35 rfl (by decide) h35_x1 (by decide) (by decide)
    have h36_len : t36.regs.length = 31 := wrZR_pc_len h35_len _ _ _
    have h36_x1 : t36.reg 1 = some OVERFLOW := by
      simpa [t36, OVERFLOW, pow2] using wrZR_pc_eq h35_len 64 (by decide : (1 : Nat) < 31) OVERFLOW
    have h36_x19 : t36.reg 19 = some v := by
      dsimp only [t36, t35, t34, t33, t32]
      repeat rw [wrZR_pc_ne _ (by decide)]
      exact h19_x19
    have s36 := step_cmp_64_k hb (k := 36) (k' := 37) (by decide) (by decide) rfl (by rfl) t36 rfl (by decide) (by decide) h36_x19 h36_x1
    have hovf_lt : OVERFLOW < AArch64Env.W := by decide
    simp only [if_neg heq]
    by_cases hovf : v = OVERFLOW
    · subst hovf
      have hfl37 : (addSubOp true 64 (OVERFLOW % pow2 64) (OVERFLOW % pow2 64)).2 = TMRVoterArm64.voterFlagsEq := by
        rw [mod_W_eq hovf_lt, sub_eq_flags]
      let t37 : St := { t36 with pc := base + 4 * 37, fl := TMRVoterArm64.voterFlagsEq }
      have h37_len : t37.regs.length = 31 := h36_len
      have s37 := step_bcond_ne hb (k := 37) (by decide) (by decide) (by rfl) (by decide) t37 rfl (z := true) rfl
      rw [if_pos rfl, show 4 * (37 + 1) = 4 * 38 by omega] at s37
      let t39 := ({ t37 with pc := base + 4 * 39 }.wrZR 64 0 6)
      have s38 : step (tailImAt base) { t37 with pc := base + 4 * 38 } = .ok t39 :=
        step_movz_64_val hb (k := 38) (k' := 39) (by decide) (by decide) rfl (by rfl) { t37 with pc := base + 4 * 38 } rfl (by decide)
      have h39_len : t39.regs.length = 31 := wrZR_pc_len h37_len _ _ _
      have h39_x0 : t39.reg 0 = some 6 := by
        simpa [t39, pow2] using wrZR_pc_eq h37_len 64 (by decide : (0 : Nat) < 31) 6
      let t40 := ({ t39 with pc := base + 4 * 40 }.wrZR 64 2 6)
      have s39 : step (tailImAt base) t39 = .ok t40 :=
        step_movz_64_val hb (k := 39) (k' := 40) (by decide) (by decide) rfl (by rfl) t39 rfl (by decide)
      have h40_len : t40.regs.length = 31 := wrZR_pc_len h39_len _ _ _
      have h40_x0 : t40.reg 0 = some 6 := by
        dsimp only [t40]; rw [wrZR_pc_ne _ (by decide)]; exact h39_x0
      have h40_x2 : t40.reg 2 = some 6 := by
        simpa [t40, pow2] using wrZR_pc_eq h39_len 64 (by decide : (2 : Nat) < 31) 6
      let t41 := ({ t40 with pc := base + 4 * 41 }.wrZR 64 0 6)
      have s40 : step (tailImAt base) t40 = .ok t41 :=
        step_orr_64_val hb (k := 40) (k' := 41) (by decide) (by decide) rfl (by rfl) t40 rfl (by decide) (by decide) h40_x0 h40_x2 (by decide)
      have h41_len : t41.regs.length = 31 := wrZR_pc_len h40_len _ _ _
      have h41_x0 : t41.reg 0 = some 6 := by
        simpa [t41, pow2] using wrZR_pc_eq h40_len 64 (by decide : (0 : Nat) < 31) 6
      let t42 := ({ t41 with pc := base + 4 * 42 }.wrZR 64 8 93)
      have s41 : step (tailImAt base) t41 = .ok t42 :=
        step_movz_64_val hb (k := 41) (k' := 42) (by decide) (by decide) rfl (by rfl) t41 rfl (by decide)
      have h42_x0 : t42.reg 0 = some 6 := by
        dsimp only [t42]; rw [wrZR_pc_ne _ (by decide)]; exact h41_x0
      have h42_x8 : t42.reg 8 = some 93 := by
        simpa [t42, pow2] using wrZR_pc_eq h41_len 64 (by decide : (8 : Nat) < 31) 93
      have s42 := step_svc_exit hb (k := 42) (by decide) (by rfl) t42 rfl h42_x0 h42_x8
      simp only [↓reduceIte]
      change run (tailImAt base) 20 (step (tailImAt base) _) = _; rw [s11]
      change run (tailImAt base) 19 (step (tailImAt base) _) = _; rw [s12]
      change run (tailImAt base) 18 (step (tailImAt base) _) = _; rw [s13]
      change run (tailImAt base) 17 (step (tailImAt base) _) = _; rw [s14]
      change run (tailImAt base) 16 (step (tailImAt base) _) = _; rw [s15]
      change run (tailImAt base) 15 (step (tailImAt base) _) = _; rw [s16]
      change run (tailImAt base) 14 (step (tailImAt base) _) = _; rw [s17]
      change run (tailImAt base) 13 (step (tailImAt base) _) = _; rw [s18]
      change run (tailImAt base) 12 (step (tailImAt base) _) = _; rw [s19]
      change run (tailImAt base) 11 (step (tailImAt base) _) = _; rw [s20]
      change run (tailImAt base) 10 (step (tailImAt base) _) = _; rw [s32]
      change run (tailImAt base) 9 (step (tailImAt base) _) = _; rw [s33]
      change run (tailImAt base) 8 (step (tailImAt base) _) = _; rw [s34]
      change run (tailImAt base) 7 (step (tailImAt base) _) = _; rw [s35]
      change run (tailImAt base) 6 (step (tailImAt base) _) = _; rw [s36, hfl37]
      change run (tailImAt base) 5 (step (tailImAt base) _) = _; rw [s37]
      change run (tailImAt base) 4 (step (tailImAt base) _) = _; rw [s38]
      change run (tailImAt base) 3 (step (tailImAt base) _) = _; rw [s39]
      change run (tailImAt base) 2 (step (tailImAt base) _) = _; rw [s40]
      change run (tailImAt base) 1 (step (tailImAt base) _) = _; rw [s41]
      change run (tailImAt base) 0 (step (tailImAt base) _) = _; rw [s42]; rfl
    · have hz37 : (addSubOp true 64 (v % pow2 64) (OVERFLOW % pow2 64)).2.z = some false := by
        rw [mod_W_eq hv, mod_W_eq hovf_lt]; exact sub_z_ne hv hovf_lt hovf
      let t37 : St := { t36 with pc := base + 4 * 37, fl := (addSubOp true 64 (v % pow2 64) (OVERFLOW % pow2 64)).2 }
      have h37_len : t37.regs.length = 31 := h36_len
      have s37 := step_bcond_ne hb (k := 37) (by decide) (by decide) (by rfl) (by decide) t37 rfl (z := false) hz37
      have hrej_pc : base + (4 * 37 + 24) = base + 4 * 43 := by omega
      rw [if_neg (by decide), hrej_pc] at s37
      let t43 : St := { t37 with pc := base + 4 * 43 }
      let t44 := ({ t43 with pc := base + 4 * 44 }.wrZR 64 0 5)
      have s43 : step (tailImAt base) t43 = .ok t44 :=
        step_movz_64_val hb (k := 43) (k' := 44) (by decide) (by decide) rfl (by rfl) t43 rfl (by decide)
      have h44_len : t44.regs.length = 31 := wrZR_pc_len h37_len _ _ _
      have h44_x0 : t44.reg 0 = some 5 := by
        simpa [t44, pow2] using wrZR_pc_eq h37_len 64 (by decide : (0 : Nat) < 31) 5
      let t45 := ({ t44 with pc := base + 4 * 45 }.wrZR 64 2 5)
      have s44 : step (tailImAt base) t44 = .ok t45 :=
        step_movz_64_val hb (k := 44) (k' := 45) (by decide) (by decide) rfl (by rfl) t44 rfl (by decide)
      have h45_len : t45.regs.length = 31 := wrZR_pc_len h44_len _ _ _
      have h45_x0 : t45.reg 0 = some 5 := by
        dsimp only [t45]; rw [wrZR_pc_ne _ (by decide)]; exact h44_x0
      have h45_x2 : t45.reg 2 = some 5 := by
        simpa [t45, pow2] using wrZR_pc_eq h44_len 64 (by decide : (2 : Nat) < 31) 5
      let t46 := ({ t45 with pc := base + 4 * 46 }.wrZR 64 0 5)
      have s45 : step (tailImAt base) t45 = .ok t46 :=
        step_orr_64_val hb (k := 45) (k' := 46) (by decide) (by decide) rfl (by rfl) t45 rfl (by decide) (by decide) h45_x0 h45_x2 (by decide)
      have h46_len : t46.regs.length = 31 := wrZR_pc_len h45_len _ _ _
      have h46_x0 : t46.reg 0 = some 5 := by
        simpa [t46, pow2] using wrZR_pc_eq h45_len 64 (by decide : (0 : Nat) < 31) 5
      let t47 := ({ t46 with pc := base + 4 * 47 }.wrZR 64 8 93)
      have s46 : step (tailImAt base) t46 = .ok t47 :=
        step_movz_64_val hb (k := 46) (k' := 47) (by decide) (by decide) rfl (by rfl) t46 rfl (by decide)
      have h47_x0 : t47.reg 0 = some 5 := by
        dsimp only [t47]; rw [wrZR_pc_ne _ (by decide)]; exact h46_x0
      have h47_x8 : t47.reg 8 = some 93 := by
        simpa [t47, pow2] using wrZR_pc_eq h46_len 64 (by decide : (8 : Nat) < 31) 93
      have s47 := step_svc_exit hb (k := 47) (by decide) (by rfl) t47 rfl h47_x0 h47_x8
      simp only [if_neg hovf]
      change run (tailImAt base) 20 (step (tailImAt base) _) = _; rw [s11]
      change run (tailImAt base) 19 (step (tailImAt base) _) = _; rw [s12]
      change run (tailImAt base) 18 (step (tailImAt base) _) = _; rw [s13]
      change run (tailImAt base) 17 (step (tailImAt base) _) = _; rw [s14]
      change run (tailImAt base) 16 (step (tailImAt base) _) = _; rw [s15]
      change run (tailImAt base) 15 (step (tailImAt base) _) = _; rw [s16]
      change run (tailImAt base) 14 (step (tailImAt base) _) = _; rw [s17]
      change run (tailImAt base) 13 (step (tailImAt base) _) = _; rw [s18]
      change run (tailImAt base) 12 (step (tailImAt base) _) = _; rw [s19]
      change run (tailImAt base) 11 (step (tailImAt base) _) = _; rw [s20]
      change run (tailImAt base) 10 (step (tailImAt base) _) = _; rw [s32]
      change run (tailImAt base) 9 (step (tailImAt base) _) = _; rw [s33]
      change run (tailImAt base) 8 (step (tailImAt base) _) = _; rw [s34]
      change run (tailImAt base) 7 (step (tailImAt base) _) = _; rw [s35]
      change run (tailImAt base) 6 (step (tailImAt base) _) = _; rw [s36]
      change run (tailImAt base) 5 (step (tailImAt base) _) = _; rw [s37]
      change run (tailImAt base) 4 (step (tailImAt base) _) = _; rw [s43]
      change run (tailImAt base) 3 (step (tailImAt base) _) = _; rw [s44]
      change run (tailImAt base) 2 (step (tailImAt base) _) = _; rw [s45]
      change run (tailImAt base) 1 (step (tailImAt base) _) = _; rw [s46]
      change run (tailImAt base) 0 (step (tailImAt base) _) = _; rw [s47]; rfl

theorem tail_arch_accept {base a b c : Nat} (hb : base_ok base) (s : St)
    (ha : a < AArch64Env.W) (hbv : b < AArch64Env.W) (hc : c < AArch64Env.W)
    (hvote : TMR.vote a b c = some ACCEPT) :
    run (tailImAt base) 29 (.ok (tailInit base s a b c)) = .exit 0 := by
  have hm := tail_arch_majority hb s ha hbv hc hvote
  rw [show (29 : Nat) = 8 + 21 from rfl, run_add, hm]
  have hg := guard_arch_run (v := ACCEPT) hb
    { s with regs := TMRVoterArm64.voterRegs s ACCEPT ACCEPT ACCEPT, pc := base + 44, fl := TMRVoterArm64.voterFlagsEq }
    (voterRegs_len s ACCEPT ACCEPT ACCEPT) (by decide) rfl rfl rfl
  simpa using hg

theorem tail_arch_overflow {base a b c : Nat} (hb : base_ok base) (s : St)
    (ha : a < AArch64Env.W) (hbv : b < AArch64Env.W) (hc : c < AArch64Env.W)
    (hvote : TMR.vote a b c = some OVERFLOW) :
    run (tailImAt base) 29 (.ok (tailInit base s a b c)) = .exit 6 := by
  have hm := tail_arch_majority hb s ha hbv hc hvote
  rw [show (29 : Nat) = 8 + 21 from rfl, run_add, hm]
  have hg := guard_arch_run (v := OVERFLOW) hb
    { s with regs := TMRVoterArm64.voterRegs s OVERFLOW OVERFLOW OVERFLOW, pc := base + 44, fl := TMRVoterArm64.voterFlagsEq }
    (voterRegs_len s OVERFLOW OVERFLOW OVERFLOW) (by decide) rfl rfl rfl
  simpa [show OVERFLOW ≠ ACCEPT by decide] using hg

theorem tail_arch_reject {base w a b c : Nat} (hb : base_ok base) (s : St)
    (ha : a < AArch64Env.W) (hbv : b < AArch64Env.W) (hc : c < AArch64Env.W)
    (hvote : TMR.vote a b c = some w) (hne_acc : w ≠ ACCEPT) (hne_ovf : w ≠ OVERFLOW) :
    run (tailImAt base) 29 (.ok (tailInit base s a b c)) = .exit 5 := by
  have hw : w < AArch64Env.W := by
    unfold TMR.vote at hvote
    split at hvote
    · cases hvote; exact ha
    · split at hvote
      · cases hvote; exact hbv
      · cases hvote
  have hm := tail_arch_majority hb s ha hbv hc hvote
  rw [show (29 : Nat) = 8 + 21 from rfl, run_add, hm]
  have hg := guard_arch_run (v := w) hb
    { s with regs := TMRVoterArm64.voterRegs s w w w, pc := base + 44, fl := TMRVoterArm64.voterFlagsEq }
    (voterRegs_len s w w w) hw rfl rfl rfl
  simpa [hne_acc, hne_ovf] using hg

/-! ### Exhaustive 4-channel fault audit on `tailWords` (`decide +kernel`) and `TailSafe` / `PreFault` -/

def allEqValNotAcc (s : St) : Prop :=
  ∃ v, s.reg 19 = some v ∧ s.reg 20 = some v ∧ s.reg 21 = some v ∧ valNotAcc v

def twoAgreeOrSplit (s : St) : Prop :=
  ∃ a b c, s.reg 19 = some a ∧ s.reg 20 = some b ∧ s.reg 21 = some c ∧
    a < AArch64Env.W ∧ b < AArch64Env.W ∧ c < AArch64Env.W ∧
    (TMR.vote a b c = none ∨ ∃ v, TMR.vote a b c = some v ∧ valNotAcc v)

def allThreeValNotAcc (s : St) : Prop :=
  ∃ a b c, s.reg 19 = some a ∧ s.reg 20 = some b ∧ s.reg 21 = some c ∧
    valNotAcc a ∧ valNotAcc b ∧ valNotAcc c

private theorem allEq_to_allThree {s : St} (h : allEqValNotAcc s) : allThreeValNotAcc s := by
  rcases h with ⟨v, h19, h20, h21, hv⟩
  exact ⟨v, v, v, h19, h20, h21, hv, hv, hv⟩

def tailSafeAt (s : St) : Nat → Prop
  | 0 => twoAgreeOrSplit s
  | 1 => allEqValNotAcc s ∨ (s.fl.z = some true ∧ allThreeValNotAcc s) ∨
      (∃ a b c, s.reg 19 = some a ∧ s.reg 20 = some b ∧ s.reg 21 = some c ∧
        a < AArch64Env.W ∧ b < AArch64Env.W ∧ c < AArch64Env.W ∧
        (TMR.vote a b c = none ∨ ∃ v, TMR.vote a b c = some v ∧ valNotAcc v) ∧
        s.fl.z = some (a == b))
  | 2 => allEqValNotAcc s ∨ (s.fl.z = some true ∧ allThreeValNotAcc s) ∨
      (∃ a b c, s.reg 19 = some a ∧ s.reg 20 = some b ∧ s.reg 21 = some c ∧
        a < AArch64Env.W ∧ b < AArch64Env.W ∧ c < AArch64Env.W ∧
        ((s.fl.z = some true ∧ valNotAcc a) ∨
         (s.fl.z = some false ∧ (a = c → valNotAcc a) ∧ (b = c → valNotAcc b))))
  | 3 => allEqValNotAcc s ∨ (s.fl.z = some true ∧ allThreeValNotAcc s) ∨
      (∃ v19 b c, s.reg 19 = some v19 ∧ s.reg 20 = some b ∧ s.reg 21 = some c ∧
        b < AArch64Env.W ∧ c < AArch64Env.W ∧
        ((s.fl.z = some true ∧ valNotAcc v19) ∨
         (s.fl.z = some false ∧ (b = c → valNotAcc b ∧ valNotAcc v19))))
  | 4 => allEqValNotAcc s ∨ (s.fl.z = some true ∧ ∃ v19, s.reg 19 = some v19 ∧ valNotAcc v19) ∨ s.fl.z = some false
  | 5 => ∃ v19, s.reg 19 = some v19 ∧ valNotAcc v19
  | 6 | 7 => ∃ v19 v20, s.reg 19 = some v19 ∧ s.reg 20 = some v20 ∧ valNotAcc v19 ∧ valNotAcc v20
  | 8 => True
  | 9 => ∃ v0, s.reg 0 = some v0 ∧ v0 % 256 ≠ 0
  | 10 =>
      ((s.reg 8 = some 93 ∨ s.reg 8 = some 94) → ∃ v0, s.reg 0 = some v0 ∧ v0 % 256 ≠ 0) ∧
      (∀ v8, s.reg 8 = some v8 → v8 ≠ 93 → v8 ≠ 94 →
        (∃ v20, s.reg 20 = some v20 ∧ valNotAcc v20) ∨
        (∃ v19, s.reg 19 = some v19 ∧ valNotAcc v19))
  | 11 | 12 | 13 | 14 | 15 =>
      (∃ v20, s.reg 20 = some v20 ∧ valNotAcc v20) ∨
      (∃ v19, s.reg 19 = some v19 ∧ valNotAcc v19)
  | 16 => (∃ v20, s.reg 20 = some v20 ∧ valNotAcc v20) ∨
      (s.reg 1 = some 0x3c96 ∧ ∃ v19, s.reg 19 = some v19 ∧ valNotAcc v19)
  | 17 => (∃ v20, s.reg 20 = some v20 ∧ valNotAcc v20) ∨
      (s.reg 1 = some 0xa55a3c96 ∧ ∃ v19, s.reg 19 = some v19 ∧ valNotAcc v19)
  | 18 => (∃ v20, s.reg 20 = some v20 ∧ valNotAcc v20) ∨
      (s.reg 1 = some 0xe970a55a3c96 ∧ ∃ v19, s.reg 19 = some v19 ∧ valNotAcc v19)
  | 19 => (∃ v20, s.reg 20 = some v20 ∧ valNotAcc v20) ∨
      (s.reg 1 = some ACCEPT ∧ ∃ v19, s.reg 19 = some v19 ∧ valNotAcc v19)
  | 20 => s.fl.z = some false ∨ (∃ v20, s.reg 20 = some v20 ∧ valNotAcc v20)
  | 21 | 22 | 23 | 24 | 25 | 26 => ∃ v20, s.reg 20 = some v20 ∧ valNotAcc v20
  | 27 => s.fl.z = some false ∨ (∃ v20, s.reg 20 = some v20 ∧ valNotAcc v20)
  | 28 | 29 => ∃ v20, s.reg 20 = some v20 ∧ v20 < AArch64Env.W ∧ v20 % 256 ≠ ACCEPT % 256
  | 30 => (s.reg 8 = some 93 ∨ s.reg 8 = some 94) → ∃ v0, s.reg 0 = some v0 ∧ v0 % 256 ≠ 0
  | 31 | 32 | 33 | 34 | 35 | 36 | 37 | 38 | 39 => True
  | 40 | 45 =>
      (∃ v0, s.reg 0 = some v0 ∧ v0 % 256 ≠ 0) ∨
      (∀ v0, s.reg 0 = some v0 → ∃ v2, s.reg 2 = some v2 ∧ v2 % 256 ≠ 0)
  | 41 | 46 => ∃ v0, s.reg 0 = some v0 ∧ v0 % 256 ≠ 0
  | 42 | 47 => (s.reg 8 = some 93 ∨ s.reg 8 = some 94) → ∃ v0, s.reg 0 = some v0 ∧ v0 % 256 ≠ 0
  | 43 | 44 => True
  | _ => True

def TailSafe (base : Nat) (s : St) : Prop :=
  s.regs.length = 31 ∧
  if _h : base ≤ s.pc ∧ s.pc < base + 192 ∧ (s.pc - base) % 4 = 0 then
    tailSafeAt s ((s.pc - base) / 4)
  else True

def preFaultIndices : List Nat :=
  [0, 1, 2, 3, 4, 5, 6, 7,
   11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
   32, 33, 34, 35, 36, 37,
   38, 39, 40, 41, 42,
   43, 44, 45, 46, 47]

def PreFault (base v : Nat) (s : St) : Prop :=
  SafeInvalid v ∧ v % 2 = 1 ∧ s.regs.length = 31 ∧ VoterScratchClean s ∧
  ∃ k, s.pc = base + 4 * k ∧
    ((k < 8 ∧ s.reg 19 = some v ∧ s.reg 20 = some v ∧ s.reg 21 = some v ∧
        (1 ≤ k ∧ k ≤ 4 → s.fl.z = some true)) ∨
     (11 ≤ k ∧ k ≤ 20 ∧ s.reg 19 = some v ∧ s.reg 20 = some v ∧
       (12 ≤ k → s.reg 0 = some 5) ∧
       (13 ≤ k → s.reg 8 = some 0) ∧
       (k = 16 → s.reg 1 = some 0x3c96) ∧
       (k = 17 → s.reg 1 = some 0xa55a3c96) ∧
       (k = 18 → s.reg 1 = some 0xe970a55a3c96) ∧
       (k = 19 → s.reg 1 = some ACCEPT) ∧
       (k = 20 → s.fl.z = some false)) ∨
     (32 ≤ k ∧ k < 48 ∧ s.reg 19 = some v ∧
       (k = 33 → s.reg 1 = some 0xc3a5) ∧
       (k = 34 → s.reg 1 = some 0x33c3c3a5) ∧
       (k = 35 → s.reg 1 = some 0x0f1e33c3c3a5) ∧
       (k = 36 → s.reg 1 = some OVERFLOW) ∧
       (k = 37 → ∃ z, s.fl.z = some z) ∧
       (k ≤ 38 → s.reg 0 = some 5) ∧
       (39 ≤ k ∧ k ≤ 42 → s.reg 0 = some 6) ∧
       (k = 40 → s.reg 2 = some 6) ∧
       (44 ≤ k ∧ k ≤ 47 → s.reg 0 = some 5) ∧
       (k = 45 → s.reg 2 = some 5) ∧
       (k ≤ 41 ∨ k = 43 ∨ k = 44 ∨ k = 45 → s.reg 8 = some 0) ∧
       (k = 42 → s.reg 8 = some 93) ∧
       (k = 47 → s.reg 8 = some 93)))

private theorem voterRegs_reg_scratch (base : Nat) (s : St) (a b c r : Nat) (hr : r ∈ VoterScratchRegs) :
    (tailInit base s a b c).reg r = s.reg r := by
  simp only [VoterScratchRegs, List.mem_cons, List.not_mem_nil, or_false] at hr
  rcases hr with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl <;>
    simp [tailInit, St.reg, TMRVoterArm64.voterRegs]

theorem PreFault_init {base v : Nat} (_ : base_ok base) (s0 : St)
    (hscr : VoterScratchClean s0) (hv : SafeInvalid v) (hodd : v % 2 = 1) :
    PreFault base v (tailInit base s0 v v v) := by
  refine ⟨hv, hodd, voterRegs_len s0 v v v, ?_, 0, by simp [tailInit],
    Or.inl ⟨by decide, rfl, rfl, rfl, by intro h; omega⟩⟩
  intro r hr x hx
  rw [voterRegs_reg_scratch base s0 v v v r hr] at hx
  exact hscr r hr x hx

theorem PreFault_to_TailSafe {base v : Nat} {s : St} (hp : PreFault base v s) :
    TailSafe base s := by
  rcases hp with ⟨hv, _, hlen, _, k, hpc, hcase⟩
  refine ⟨hlen, ?_⟩
  split
  · have hk_eq : (s.pc - base) / 4 = k := by rw [hpc]; omega
    rw [hk_eq]
    have hnv : valNotAcc v := hv.valNotAcc
    rcases hcase with ⟨hk8, h19, h20, h21, _⟩ | ⟨hk11, hk20, _, h20, _, _, _, _, _, _, _⟩ |
      ⟨hk32, hk48, _, _, _, _, _, _, _, h39_42, _, h44_47, _, _, _⟩
    · rcases k with _ | _ | _ | _ | _ | _ | _ | _ | k
      · exact ⟨v, v, v, h19, h20, h21, hv.1, hv.1, hv.1, Or.inr ⟨v, by unfold TMR.vote; simp, hnv⟩⟩
      · exact Or.inl ⟨v, h19, h20, h21, hnv⟩
      · exact Or.inl ⟨v, h19, h20, h21, hnv⟩
      · exact Or.inl ⟨v, h19, h20, h21, hnv⟩
      · exact Or.inl ⟨v, h19, h20, h21, hnv⟩
      · exact ⟨v, h19, hnv⟩
      · exact ⟨v, v, h19, h20, hnv, hnv⟩
      · exact ⟨v, v, h19, h20, hnv, hnv⟩
      · omega
    · have : k = 11 ∨ k = 12 ∨ k = 13 ∨ k = 14 ∨ k = 15 ∨ k = 16 ∨ k = 17 ∨ k = 18 ∨ k = 19 ∨ k = 20 := by omega
      rcases this with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl
      · exact Or.inl ⟨v, h20, hnv⟩
      · exact Or.inl ⟨v, h20, hnv⟩
      · exact Or.inl ⟨v, h20, hnv⟩
      · exact Or.inl ⟨v, h20, hnv⟩
      · exact Or.inl ⟨v, h20, hnv⟩
      · exact Or.inl ⟨v, h20, hnv⟩
      · exact Or.inl ⟨v, h20, hnv⟩
      · exact Or.inl ⟨v, h20, hnv⟩
      · exact Or.inl ⟨v, h20, hnv⟩
      · exact Or.inr ⟨v, h20, hnv⟩
    · have : k = 32 ∨ k = 33 ∨ k = 34 ∨ k = 35 ∨ k = 36 ∨ k = 37 ∨ k = 38 ∨ k = 39 ∨
             k = 40 ∨ k = 41 ∨ k = 42 ∨ k = 43 ∨ k = 44 ∨ k = 45 ∨ k = 46 ∨ k = 47 := by omega
      rcases this with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl
      · trivial
      · trivial
      · trivial
      · trivial
      · trivial
      · trivial
      · trivial
      · trivial
      · exact Or.inl ⟨6, h39_42 ⟨by decide, by decide⟩, by decide⟩
      · exact ⟨6, h39_42 ⟨by decide, by decide⟩, by decide⟩
      · intro _; exact ⟨6, h39_42 ⟨by decide, by decide⟩, by decide⟩
      · trivial
      · trivial
      · exact Or.inl ⟨5, h44_47 ⟨by decide, by decide⟩, by decide⟩
      · exact ⟨5, h44_47 ⟨by decide, by decide⟩, by decide⟩
      · intro _; exact ⟨5, h44_47 ⟨by decide, by decide⟩, by decide⟩
  · trivial

theorem TailSafe_init_majority {base a b c : Nat} (s : St)
    (ha : a < AArch64Env.W) (hbv : b < AArch64Env.W) (hc : c < AArch64Env.W)
    (hvote : TMR.vote a b c = none ∨ ∃ v, TMR.vote a b c = some v ∧ valNotAcc v) :
    TailSafe base (tailInit base s a b c) := by
  refine ⟨voterRegs_len s a b c, ?_⟩
  split
  · next hbd =>
    have hk0 : ((tailInit base s a b c).pc - base) / 4 = 0 := by simp [tailInit]
    rw [hk0]
    exact ⟨a, b, c, rfl, rfl, rfl, ha, hbv, hc, hvote⟩
  · trivial

private theorem voterScratchClean_wrZR_ne {s : St} (hscr : VoterScratchClean s)
    (w r v : Nat) (hnot : r ∉ VoterScratchRegs) : VoterScratchClean (s.wrZR w r v) := by
  intro q hq u hu
  have hqr : r ≠ q := by intro h; subst h; exact hnot hq
  rw [show (s.wrZR w r v).reg q = s.reg q from wrZR_ne w hqr v] at hu
  exact hscr q hq u hu

private theorem voterScratchClean_pc_wrZR_ne {s : St} (hscr : VoterScratchClean s)
    {p : Nat} (w r v : Nat) (hnot : r ∉ VoterScratchRegs) :
    VoterScratchClean ({ s with pc := p }.wrZR w r v) :=
  voterScratchClean_wrZR_ne (s := { s with pc := p }) hscr w r v hnot

private theorem zero_lor_eq (x : Nat) : Nat.lor 0 x = x := by
  apply Nat.eq_of_testBit_eq; intro i
  rw [show Nat.lor 0 x = 0 ||| x from rfl, Nat.testBit_or]; simp

theorem PreFault_step {base v : Nat} (hb : base_ok base) {s : St} (hp : PreFault base v s) :
    match step (tailImAt base) s with
    | .ok s' => PreFault base v s'
    | .exit c => c ≠ 0
    | .stuck => True := by
  rcases hp with ⟨hv, hodd, hlen, hscr, k, hpc, hcase⟩
  have h19lt : (19 : Nat) < 31 := by decide
  have h20lt : (20 : Nat) < 31 := by decide
  have h21lt : (21 : Nat) < 31 := by decide
  have h0lt : (0 : Nat) < 31 := by decide
  have h1lt : (1 : Nat) < 31 := by decide
  have h2lt : (2 : Nat) < 31 := by decide
  have h8lt : (8 : Nat) < 31 := by decide
  have hmod19 : v % pow2 64 = v := mod_W_eq hv.1
  rcases hcase with ⟨hk8, h19, h20, h21, hfl⟩ | ⟨hk11, hk20, h19, h20, hx0, hx8, h16, h17, h18, h19_1, hfl20⟩ |
    ⟨hk32, hk48, h19, h33, h34, h35, h36, hfl37, h32_38, h39_42, h40_2, h44_47, h45_2, h8z, h42_8, h47_8⟩
  · have : k = 0 ∨ k = 1 ∨ k = 2 ∨ k = 3 ∨ k = 4 ∨ k = 5 ∨ k = 6 ∨ k = 7 := by omega
    rcases this with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl
    · rw [step_at_word hb.1 (by decide) hpc (by rfl)]
      simp [exec, lift, bind, Option.bind, pure, St.rdZR, h19, h20, wrZR_31_eq, shiftVal, pow2,
        sub_eq_flags, hpc,
        show (base + 4 * 0 + 4) % AArch64Env.W = base + 4 * 1 by
          rw [show base + 4 * 0 + 4 = base + 4 by omega]; exact add_base_mod_W hb (by decide)]
      refine ⟨hv, hodd, hlen, hscr, 1, rfl, Or.inl ⟨by decide, h19, h20, h21, fun _ => rfl⟩⟩
    · have hz : s.fl.z = some true := hfl ⟨by decide, by decide⟩
      have hc1 : AArch64Phys.cond s.fl 1 = some false := by simp [AArch64Phys.cond, hz]
      rw [step_at_word hb.1 (by decide) hpc (by rfl)]
      simp [exec, lift, bind, Option.bind, pure, St.rdZR, h19, h21, hc1,
        AArch64Phys.bf, pow2, hpc,
        show (base + 4 * 1 + 4) % AArch64Env.W = base + 4 * 2 by
          rw [show base + 4 * 1 + 4 = base + 8 by omega]; exact add_base_mod_W hb (by decide)]
      exact ⟨hv, hodd, hlen, hscr, 2, rfl, Or.inl ⟨by decide, h19, h20, h21, fun _ => rfl⟩⟩
    · have hz : s.fl.z = some true := hfl ⟨by decide, by decide⟩
      have hc0 : AArch64Phys.cond s.fl 0 = some true := by simp [AArch64Phys.cond, hz]
      rw [step_at_word hb.1 (by decide) hpc (by rfl)]
      simp only [exec, lift, bind, Option.bind, pure, Option.map_some, St.rdZR, h19, h20, hc0, hpc, hmod19,
        show (19 == 31) = false by decide, show (20 == 31) = false by decide, Bool.false_eq_true, ↓reduceIte,
        show (base + 4 * 2 + 4) % AArch64Env.W = base + 4 * 3 by
          rw [show base + 4 * 2 + 4 = base + 12 by omega]; exact add_base_mod_W hb (by decide)]
      refine ⟨hv, hodd, wrZR_pc_len hlen _ _ _, voterScratchClean_pc_wrZR_ne hscr _ _ _ (by decide),
        3, rfl, Or.inl ⟨by decide, wrZR_pc_eq hlen 64 h19lt v |>.trans (by simp [hmod19]),
          by rw [wrZR_pc_ne 64 (by decide)]; exact h20,
          by rw [wrZR_pc_ne 64 (by decide)]; exact h21, fun _ => hz⟩⟩
    · have hz : s.fl.z = some true := hfl ⟨by decide, by decide⟩
      have hc1 : AArch64Phys.cond s.fl 1 = some false := by simp [AArch64Phys.cond, hz]
      rw [step_at_word hb.1 (by decide) hpc (by rfl)]
      simp [exec, lift, bind, Option.bind, pure, St.rdZR, h20, h21, hc1,
        AArch64Phys.bf, pow2, hpc,
        show (base + 4 * 3 + 4) % AArch64Env.W = base + 4 * 4 by
          rw [show base + 4 * 3 + 4 = base + 16 by omega]; exact add_base_mod_W hb (by decide)]
      exact ⟨hv, hodd, hlen, hscr, 4, rfl, Or.inl ⟨by decide, h19, h20, h21, fun _ => rfl⟩⟩
    · have hz : s.fl.z = some true := hfl ⟨by decide, by decide⟩
      have hc1 : AArch64Phys.cond s.fl 1 = some false := by simp [AArch64Phys.cond, hz]
      rw [step_at_word hb.1 (by decide) hpc (by rfl)]
      simp [exec, lift, bind, Option.bind, pure, hc1, hpc,
        show (base + 4 * 4 + 4) % AArch64Env.W = base + 4 * 5 by
          rw [show base + 4 * 4 + 4 = base + 20 by omega]; exact add_base_mod_W hb (by decide)]
      exact ⟨hv, hodd, hlen, hscr, 5, rfl, Or.inl ⟨by decide, h19, h20, h21, fun h => by omega⟩⟩
    · rw [step_at_word hb.1 (by decide) hpc (by rfl)]
      have hlor_v : Nat.lor 0 (v % 2 ^ 64 % 2 ^ 64 * 2 ^ 0 % 2 ^ 64) = v := by
        have hv' : v % 2 ^ 64 = v := mod_W_eq hv.1
        simp only [hv', show (2 ^ 0 : Nat) = 1 from rfl, Nat.mul_one]
        exact zero_lor_eq v
      simp only [exec, lift, bind, Option.bind, pure, Option.map_some, St.rdZR, h19, hpc,
        show (19 == 31) = false by decide, show (31 == 31) = true by decide, Bool.false_eq_true, ↓reduceIte,
        shiftVal, pow2, hlor_v,
        show (base + 4 * 5 + 4) % AArch64Env.W = base + 4 * 6 by
          rw [show base + 4 * 5 + 4 = base + 24 by omega]; exact add_base_mod_W hb (by decide)]
      exact ⟨hv, hodd, wrZR_pc_len hlen _ _ _, voterScratchClean_pc_wrZR_ne hscr _ _ _ (by decide),
        6, rfl, Or.inl ⟨by decide, by rw [wrZR_pc_ne 64 (by decide)]; exact h19,
          by rw [wrZR_pc_eq hlen 64 h20lt v, hmod19],
          by rw [wrZR_pc_ne 64 (by decide)]; exact h21, fun h => by omega⟩⟩
    · rw [step_at_word hb.1 (by decide) hpc (by rfl)]
      have hlor_v : Nat.lor 0 (v % 2 ^ 64 % 2 ^ 64 * 2 ^ 0 % 2 ^ 64) = v := by
        have hv' : v % 2 ^ 64 = v := mod_W_eq hv.1
        simp only [hv', show (2 ^ 0 : Nat) = 1 from rfl, Nat.mul_one]
        exact zero_lor_eq v
      simp only [exec, lift, bind, Option.bind, pure, Option.map_some, St.rdZR, h19, hpc,
        show (19 == 31) = false by decide, show (31 == 31) = true by decide, Bool.false_eq_true, ↓reduceIte,
        shiftVal, pow2, hlor_v,
        show (base + 4 * 6 + 4) % AArch64Env.W = base + 4 * 7 by
          rw [show base + 4 * 6 + 4 = base + 28 by omega]; exact add_base_mod_W hb (by decide)]
      exact ⟨hv, hodd, wrZR_pc_len hlen _ _ _, voterScratchClean_pc_wrZR_ne hscr _ _ _ (by decide),
        7, rfl, Or.inl ⟨by decide, by rw [wrZR_pc_ne 64 (by decide)]; exact h19,
          by rw [wrZR_pc_ne 64 (by decide)]; exact h20,
          by rw [wrZR_pc_eq hlen 64 h21lt v, hmod19], fun h => by omega⟩⟩
    · rw [step_at_word hb.1 (by decide) hpc (by rfl)]
      simp only [exec, hpc,
        show (base + 4 * 7 + 16) % AArch64Env.W = base + 4 * 11 by
          rw [show base + 4 * 7 + 16 = base + 44 by omega]; exact add_base_mod_W hb (by decide)]
      exact ⟨hv, hodd, hlen, hscr, 11, rfl, Or.inr (Or.inl ⟨by decide, by decide, h19, h20,
        fun h => by omega, fun h => by omega, fun h => by omega, fun h => by omega,
        fun h => by omega, fun h => by omega, fun h => by omega⟩)⟩
  · have : k = 11 ∨ k = 12 ∨ k = 13 ∨ k = 14 ∨ k = 15 ∨ k = 16 ∨ k = 17 ∨ k = 18 ∨ k = 19 ∨ k = 20 := by omega
    rcases this with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl
    · rw [step_movz_64 hb (by decide) (by decide) (by rfl) s hpc]
      exact ⟨hv, hodd, wrZR_pc_len hlen _ _ _, voterScratchClean_pc_wrZR_ne hscr _ _ _ (by decide),
        12, rfl, Or.inr (Or.inl ⟨by decide, by decide,
          by rw [wrZR_pc_ne 64 (by decide)]; exact h19, by rw [wrZR_pc_ne 64 (by decide)]; exact h20,
          fun _ => by simpa [pow2] using wrZR_pc_eq hlen 64 h0lt 5, fun h => by omega,
          fun h => by omega, fun h => by omega, fun h => by omega, fun h => by omega, fun h => by omega⟩)⟩
    · rw [step_movz_64 hb (by decide) (by decide) (by rfl) s hpc]
      exact ⟨hv, hodd, wrZR_pc_len hlen _ _ _, voterScratchClean_pc_wrZR_ne hscr _ _ _ (by decide),
        13, rfl, Or.inr (Or.inl ⟨by decide, by decide,
          by rw [wrZR_pc_ne 64 (by decide)]; exact h19, by rw [wrZR_pc_ne 64 (by decide)]; exact h20,
          fun _ => by rw [wrZR_pc_ne 64 (by decide)]; exact hx0 (by decide),
          fun _ => by simpa [pow2] using wrZR_pc_eq hlen 64 h8lt 0,
          fun h => by omega, fun h => by omega, fun h => by omega, fun h => by omega, fun h => by omega⟩)⟩
    · rw [step_movz_64 hb (by decide) (by decide) (by rfl) s hpc]
      exact ⟨hv, hodd, wrZR_pc_len hlen _ _ _, voterScratchClean_pc_wrZR_ne hscr _ _ _ (by decide),
        14, rfl, Or.inr (Or.inl ⟨by decide, by decide,
          by rw [wrZR_pc_ne 64 (by decide)]; exact h19, by rw [wrZR_pc_ne 64 (by decide)]; exact h20,
          fun _ => by rw [wrZR_pc_ne 64 (by decide)]; exact hx0 (by decide),
          fun _ => by rw [wrZR_pc_ne 64 (by decide)]; exact hx8 (by decide),
          fun h => by omega, fun h => by omega, fun h => by omega, fun h => by omega, fun h => by omega⟩)⟩
    · rw [step_movz_64 hb (by decide) (by decide) (by rfl) s hpc]
      exact ⟨hv, hodd, wrZR_pc_len hlen _ _ _, voterScratchClean_pc_wrZR_ne hscr _ _ _ (by decide),
        15, rfl, Or.inr (Or.inl ⟨by decide, by decide,
          by rw [wrZR_pc_ne 64 (by decide)]; exact h19, by rw [wrZR_pc_ne 64 (by decide)]; exact h20,
          fun _ => by rw [wrZR_pc_ne 64 (by decide)]; exact hx0 (by decide),
          fun _ => by rw [wrZR_pc_ne 64 (by decide)]; exact hx8 (by decide),
          fun h => by omega, fun h => by omega, fun h => by omega, fun h => by omega, fun h => by omega⟩)⟩
    · rw [step_movz_64 hb (by decide) (by decide) (by rfl) s hpc]
      exact ⟨hv, hodd, wrZR_pc_len hlen _ _ _, voterScratchClean_pc_wrZR_ne hscr _ _ _ (by decide),
        16, rfl, Or.inr (Or.inl ⟨by decide, by decide,
          by rw [wrZR_pc_ne 64 (by decide)]; exact h19, by rw [wrZR_pc_ne 64 (by decide)]; exact h20,
          fun _ => by rw [wrZR_pc_ne 64 (by decide)]; exact hx0 (by decide),
          fun _ => by rw [wrZR_pc_ne 64 (by decide)]; exact hx8 (by decide),
          fun _ => by simpa [pow2] using wrZR_pc_eq hlen 64 h1lt 0x3c96,
          fun h => by omega, fun h => by omega, fun h => by omega, fun h => by omega⟩)⟩
    · rw [step_movk_64 hb (by decide) (by decide) (by rfl) s hpc h1lt (h16 rfl)]
      exact ⟨hv, hodd, wrZR_pc_len hlen _ _ _, voterScratchClean_pc_wrZR_ne hscr _ _ _ (by decide),
        17, rfl, Or.inr (Or.inl ⟨by decide, by decide,
          by rw [wrZR_pc_ne 64 (by decide)]; exact h19, by rw [wrZR_pc_ne 64 (by decide)]; exact h20,
          fun _ => by rw [wrZR_pc_ne 64 (by decide)]; exact hx0 (by decide),
          fun _ => by rw [wrZR_pc_ne 64 (by decide)]; exact hx8 (by decide),
          fun h => by omega,
          fun _ => by simpa [pow2] using wrZR_pc_eq hlen 64 h1lt _,
          fun h => by omega, fun h => by omega, fun h => by omega⟩)⟩
    · rw [step_movk_64 hb (by decide) (by decide) (by rfl) s hpc h1lt (h17 rfl)]
      exact ⟨hv, hodd, wrZR_pc_len hlen _ _ _, voterScratchClean_pc_wrZR_ne hscr _ _ _ (by decide),
        18, rfl, Or.inr (Or.inl ⟨by decide, by decide,
          by rw [wrZR_pc_ne 64 (by decide)]; exact h19, by rw [wrZR_pc_ne 64 (by decide)]; exact h20,
          fun _ => by rw [wrZR_pc_ne 64 (by decide)]; exact hx0 (by decide),
          fun _ => by rw [wrZR_pc_ne 64 (by decide)]; exact hx8 (by decide),
          fun h => by omega, fun h => by omega,
          fun _ => by simpa [pow2] using wrZR_pc_eq hlen 64 h1lt _,
          fun h => by omega, fun h => by omega⟩)⟩
    · rw [step_movk_64 hb (by decide) (by decide) (by rfl) s hpc h1lt (h18 rfl)]
      exact ⟨hv, hodd, wrZR_pc_len hlen _ _ _, voterScratchClean_pc_wrZR_ne hscr _ _ _ (by decide),
        19, rfl, Or.inr (Or.inl ⟨by decide, by decide,
          by rw [wrZR_pc_ne 64 (by decide)]; exact h19, by rw [wrZR_pc_ne 64 (by decide)]; exact h20,
          fun _ => by rw [wrZR_pc_ne 64 (by decide)]; exact hx0 (by decide),
          fun _ => by rw [wrZR_pc_ne 64 (by decide)]; exact hx8 (by decide),
          fun h => by omega, fun h => by omega, fun h => by omega,
          fun _ => by simpa [ACCEPT, pow2] using wrZR_pc_eq hlen 64 h1lt _,
          fun h => by omega⟩)⟩
    · rw [step_cmp_64 hb (by decide) (by decide) (by rfl) s hpc h19lt h1lt h19 (h19_1 rfl)]
      have hz : (addSubOp true 64 (v % pow2 64) (ACCEPT % pow2 64)).2.z = some false := by
        rw [hmod19, mod_W_eq (by decide : ACCEPT < AArch64Env.W)]
        exact sub_z_ne hv.1 (by decide) hv.2.1
      exact ⟨hv, hodd, hlen, hscr, 20, rfl, Or.inr (Or.inl ⟨by decide, by decide, h19, h20,
        fun _ => hx0 (by decide), fun _ => hx8 (by decide),
        fun h => by omega, fun h => by omega, fun h => by omega, fun h => by omega, fun _ => hz⟩)⟩
    · have hz := hfl20 rfl
      rw [step_bcond_ne hb (by decide) (by decide) (by rfl) (by decide) s hpc hz]
      simp only [show (base + (4 * 20 + 48)) = base + 4 * 32 by omega, Bool.false_eq_true, ↓reduceIte]
      exact ⟨hv, hodd, hlen, hscr, 32, rfl, Or.inr (Or.inr ⟨by decide, by decide, h19,
        fun h => by omega, fun h => by omega, fun h => by omega, fun h => by omega, fun h => by omega,
        fun _ => hx0 (by decide), fun h => by omega, fun h => by omega, fun h => by omega, fun h => by omega,
        fun _ => hx8 (by decide), fun h => by omega, fun h => by omega⟩)⟩
  · have : k = 32 ∨ k = 33 ∨ k = 34 ∨ k = 35 ∨ k = 36 ∨ k = 37 ∨ k = 38 ∨ k = 39 ∨
           k = 40 ∨ k = 41 ∨ k = 42 ∨ k = 43 ∨ k = 44 ∨ k = 45 ∨ k = 46 ∨ k = 47 := by omega
    rcases this with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl
    · rw [step_movz_64 hb (by decide) (by decide) (by rfl) s hpc]
      exact ⟨hv, hodd, wrZR_pc_len hlen _ _ _, voterScratchClean_pc_wrZR_ne hscr _ _ _ (by decide),
        33, rfl, Or.inr (Or.inr ⟨by decide, by decide, by rw [wrZR_pc_ne 64 (by decide)]; exact h19,
          fun _ => by simpa [pow2] using wrZR_pc_eq hlen 64 h1lt 0xc3a5,
          fun h => by omega, fun h => by omega, fun h => by omega, fun h => by omega,
          fun _ => by rw [wrZR_pc_ne 64 (by decide)]; exact h32_38 (by decide),
          fun h => by omega, fun h => by omega, fun h => by omega, fun h => by omega,
          fun _ => by rw [wrZR_pc_ne 64 (by decide)]; exact h8z (Or.inl (by decide)),
          fun h => by omega, fun h => by omega⟩)⟩
    · rw [step_movk_64 hb (by decide) (by decide) (by rfl) s hpc h1lt (h33 rfl)]
      exact ⟨hv, hodd, wrZR_pc_len hlen _ _ _, voterScratchClean_pc_wrZR_ne hscr _ _ _ (by decide),
        34, rfl, Or.inr (Or.inr ⟨by decide, by decide, by rw [wrZR_pc_ne 64 (by decide)]; exact h19,
          fun h => by omega,
          fun _ => by simpa [pow2] using wrZR_pc_eq hlen 64 h1lt _,
          fun h => by omega, fun h => by omega, fun h => by omega,
          fun _ => by rw [wrZR_pc_ne 64 (by decide)]; exact h32_38 (by decide),
          fun h => by omega, fun h => by omega, fun h => by omega, fun h => by omega,
          fun _ => by rw [wrZR_pc_ne 64 (by decide)]; exact h8z (Or.inl (by decide)),
          fun h => by omega, fun h => by omega⟩)⟩
    · rw [step_movk_64 hb (by decide) (by decide) (by rfl) s hpc h1lt (h34 rfl)]
      exact ⟨hv, hodd, wrZR_pc_len hlen _ _ _, voterScratchClean_pc_wrZR_ne hscr _ _ _ (by decide),
        35, rfl, Or.inr (Or.inr ⟨by decide, by decide, by rw [wrZR_pc_ne 64 (by decide)]; exact h19,
          fun h => by omega, fun h => by omega,
          fun _ => by simpa [pow2] using wrZR_pc_eq hlen 64 h1lt _,
          fun h => by omega, fun h => by omega,
          fun _ => by rw [wrZR_pc_ne 64 (by decide)]; exact h32_38 (by decide),
          fun h => by omega, fun h => by omega, fun h => by omega, fun h => by omega,
          fun _ => by rw [wrZR_pc_ne 64 (by decide)]; exact h8z (Or.inl (by decide)),
          fun h => by omega, fun h => by omega⟩)⟩
    · rw [step_movk_64 hb (by decide) (by decide) (by rfl) s hpc h1lt (h35 rfl)]
      exact ⟨hv, hodd, wrZR_pc_len hlen _ _ _, voterScratchClean_pc_wrZR_ne hscr _ _ _ (by decide),
        36, rfl, Or.inr (Or.inr ⟨by decide, by decide, by rw [wrZR_pc_ne 64 (by decide)]; exact h19,
          fun h => by omega, fun h => by omega, fun h => by omega,
          fun _ => by simpa [OVERFLOW, pow2] using wrZR_pc_eq hlen 64 h1lt _,
          fun h => by omega,
          fun _ => by rw [wrZR_pc_ne 64 (by decide)]; exact h32_38 (by decide),
          fun h => by omega, fun h => by omega, fun h => by omega, fun h => by omega,
          fun _ => by rw [wrZR_pc_ne 64 (by decide)]; exact h8z (Or.inl (by decide)),
          fun h => by omega, fun h => by omega⟩)⟩
    · rw [step_cmp_64 hb (by decide) (by decide) (by rfl) s hpc h19lt h1lt h19 (h36 rfl)]
      exact ⟨hv, hodd, hlen, hscr, 37, rfl, Or.inr (Or.inr ⟨by decide, by decide, h19,
        fun h => by omega, fun h => by omega, fun h => by omega, fun h => by omega,
        fun _ => ⟨_, rfl⟩,
        fun _ => h32_38 (by decide), fun h => by omega, fun h => by omega, fun h => by omega, fun h => by omega,
        fun _ => h8z (Or.inl (by decide)), fun h => by omega, fun h => by omega⟩)⟩
    · rcases hfl37 rfl with ⟨z, hz⟩
      rw [step_bcond_ne hb (by decide) (by decide) (by rfl) (by decide) s hpc hz]
      cases z
      · simp only [show base + (4 * 37 + 24) = base + 4 * 43 by omega, Bool.false_eq_true, ↓reduceIte]
        exact ⟨hv, hodd, hlen, hscr, 43, rfl, Or.inr (Or.inr ⟨by decide, by decide, h19,
          fun h => by omega, fun h => by omega, fun h => by omega, fun h => by omega, fun h => by omega,
          fun h => by omega, fun h => by omega, fun h => by omega, fun h => by omega, fun h => by omega,
          fun _ => h8z (Or.inl (by decide)), fun h => by omega, fun h => by omega⟩)⟩
      · exact ⟨hv, hodd, hlen, hscr, 38, rfl, Or.inr (Or.inr ⟨by decide, by decide, h19,
          fun h => by omega, fun h => by omega, fun h => by omega, fun h => by omega, fun h => by omega,
          fun _ => h32_38 (by decide), fun h => by omega, fun h => by omega, fun h => by omega, fun h => by omega,
          fun _ => h8z (Or.inl (by decide)), fun h => by omega, fun h => by omega⟩)⟩
    · rw [step_movz_64 hb (by decide) (by decide) (by rfl) s hpc]
      exact ⟨hv, hodd, wrZR_pc_len hlen _ _ _, voterScratchClean_pc_wrZR_ne hscr _ _ _ (by decide),
        39, rfl, Or.inr (Or.inr ⟨by decide, by decide, by rw [wrZR_pc_ne 64 (by decide)]; exact h19,
          fun h => by omega, fun h => by omega, fun h => by omega, fun h => by omega, fun h => by omega,
          fun h => by omega, fun _ => by simpa [pow2] using wrZR_pc_eq hlen 64 h0lt 6,
          fun h => by omega, fun h => by omega, fun h => by omega,
          fun _ => by rw [wrZR_pc_ne 64 (by decide)]; exact h8z (Or.inl (by decide)),
          fun h => by omega, fun h => by omega⟩)⟩
    · rw [step_movz_64 hb (by decide) (by decide) (by rfl) s hpc]
      exact ⟨hv, hodd, wrZR_pc_len hlen _ _ _, voterScratchClean_pc_wrZR_ne hscr _ _ _ (by decide),
        40, rfl, Or.inr (Or.inr ⟨by decide, by decide, by rw [wrZR_pc_ne 64 (by decide)]; exact h19,
          fun h => by omega, fun h => by omega, fun h => by omega, fun h => by omega, fun h => by omega,
          fun h => by omega, fun _ => by rw [wrZR_pc_ne 64 (by decide)]; exact h39_42 ⟨by decide, by decide⟩,
          fun _ => by simpa [pow2] using wrZR_pc_eq hlen 64 h2lt 6,
          fun h => by omega, fun h => by omega,
          fun _ => by rw [wrZR_pc_ne 64 (by decide)]; exact h8z (Or.inl (by decide)),
          fun h => by omega, fun h => by omega⟩)⟩
    · rw [step_orr_64 hb (by decide) (by decide) (by rfl) s hpc h0lt h2lt
        (h39_42 ⟨by decide, by decide⟩) (h40_2 rfl)]
      exact ⟨hv, hodd, wrZR_pc_len hlen _ _ _, voterScratchClean_pc_wrZR_ne hscr _ _ _ (by decide),
        41, rfl, Or.inr (Or.inr ⟨by decide, by decide, by rw [wrZR_pc_ne 64 (by decide)]; exact h19,
          fun h => by omega, fun h => by omega, fun h => by omega, fun h => by omega, fun h => by omega,
          fun h => by omega, fun _ => by simpa [pow2] using wrZR_pc_eq hlen 64 h0lt _,
          fun h => by omega, fun h => by omega, fun h => by omega,
          fun _ => by rw [wrZR_pc_ne 64 (by decide)]; exact h8z (Or.inl (by decide)),
          fun h => by omega, fun h => by omega⟩)⟩
    · rw [step_movz_64 hb (by decide) (by decide) (by rfl) s hpc]
      exact ⟨hv, hodd, wrZR_pc_len hlen _ _ _, voterScratchClean_pc_wrZR_ne hscr _ _ _ (by decide),
        42, rfl, Or.inr (Or.inr ⟨by decide, by decide, by rw [wrZR_pc_ne 64 (by decide)]; exact h19,
          fun h => by omega, fun h => by omega, fun h => by omega, fun h => by omega, fun h => by omega,
          fun h => by omega, fun _ => by rw [wrZR_pc_ne 64 (by decide)]; exact h39_42 ⟨by decide, by decide⟩,
          fun h => by omega, fun h => by omega, fun h => by omega,
          fun h => by rcases h with h | h | h | h <;> omega,
          fun _ => by simpa [pow2] using wrZR_pc_eq hlen 64 h8lt 93,
          fun h => by omega⟩)⟩
    · rw [step_svc_exit hb (by decide) (by rfl) s hpc (h39_42 ⟨by decide, by decide⟩) (h42_8 rfl)]
      simp
    · rw [step_movz_64 hb (by decide) (by decide) (by rfl) s hpc]
      exact ⟨hv, hodd, wrZR_pc_len hlen _ _ _, voterScratchClean_pc_wrZR_ne hscr _ _ _ (by decide),
        44, rfl, Or.inr (Or.inr ⟨by decide, by decide, by rw [wrZR_pc_ne 64 (by decide)]; exact h19,
          fun h => by omega, fun h => by omega, fun h => by omega, fun h => by omega, fun h => by omega,
          fun h => by omega, fun h => by omega, fun h => by omega,
          fun _ => by simpa [pow2] using wrZR_pc_eq hlen 64 h0lt 5,
          fun h => by omega,
          fun _ => by rw [wrZR_pc_ne 64 (by decide)]; exact h8z (by decide),
          fun h => by omega, fun h => by omega⟩)⟩
    · rw [step_movz_64 hb (by decide) (by decide) (by rfl) s hpc]
      exact ⟨hv, hodd, wrZR_pc_len hlen _ _ _, voterScratchClean_pc_wrZR_ne hscr _ _ _ (by decide),
        45, rfl, Or.inr (Or.inr ⟨by decide, by decide, by rw [wrZR_pc_ne 64 (by decide)]; exact h19,
          fun h => by omega, fun h => by omega, fun h => by omega, fun h => by omega, fun h => by omega,
          fun h => by omega, fun h => by omega, fun h => by omega,
          fun _ => by rw [wrZR_pc_ne 64 (by decide)]; exact h44_47 ⟨by decide, by decide⟩,
          fun _ => by simpa [pow2] using wrZR_pc_eq hlen 64 h2lt 5,
          fun _ => by rw [wrZR_pc_ne 64 (by decide)]; exact h8z (by decide),
          fun h => by omega, fun h => by omega⟩)⟩
    · rw [step_orr_64 hb (by decide) (by decide) (by rfl) s hpc h0lt h2lt
        (h44_47 ⟨by decide, by decide⟩) (h45_2 rfl)]
      exact ⟨hv, hodd, wrZR_pc_len hlen _ _ _, voterScratchClean_pc_wrZR_ne hscr _ _ _ (by decide),
        46, rfl, Or.inr (Or.inr ⟨by decide, by decide, by rw [wrZR_pc_ne 64 (by decide)]; exact h19,
          fun h => by omega, fun h => by omega, fun h => by omega, fun h => by omega, fun h => by omega,
          fun h => by omega, fun h => by omega, fun h => by omega,
          fun _ => by simpa [pow2] using wrZR_pc_eq hlen 64 h0lt _,
          fun h => by omega, fun h => by rcases h with h | h | h | h <;> omega,
          fun h => by omega, fun h => by omega⟩)⟩
    · rw [step_movz_64 hb (by decide) (by decide) (by rfl) s hpc]
      exact ⟨hv, hodd, wrZR_pc_len hlen _ _ _, voterScratchClean_pc_wrZR_ne hscr _ _ _ (by decide),
        47, rfl, Or.inr (Or.inr ⟨by decide, by decide, by rw [wrZR_pc_ne 64 (by decide)]; exact h19,
          fun h => by omega, fun h => by omega, fun h => by omega, fun h => by omega, fun h => by omega,
          fun h => by omega, fun h => by omega, fun h => by omega,
          fun _ => by rw [wrZR_pc_ne 64 (by decide)]; exact h44_47 ⟨by decide, by decide⟩,
          fun h => by omega, fun h => by rcases h with h | h | h | h <;> omega,
          fun h => by omega,
          fun _ => by simpa [pow2] using wrZR_pc_eq hlen 64 h8lt 93⟩)⟩
    · rw [step_svc_exit hb (by decide) (by rfl) s hpc (h44_47 ⟨by decide, by decide⟩) (h47_8 rfl)]
      simp

private theorem syscall_ne_exit {k : Kern} {m : Mem}
    {nr a1 a2 a3 a4 a5 a6 c : Nat} (hn93 : nr ≠ 93) (hn94 : nr ≠ 94) :
    AArch64Env.syscall k m nr a1 a2 a3 a4 a5 a6 ≠ some (.exit c) := by
  by_cases h56 : nr = 56
  · subst h56; simp only [AArch64Env.syscall]; split
    · simp
    · simp only [bind, Option.bind]
      cases m.cstr a2 4096 with
      | none => simp
      | some nm => cases hlu : AArch64Env.lookup k.files nm <;> simp [hlu]
  by_cases h57 : nr = 57
  · subst h57; simp only [AArch64Env.syscall]; split <;> simp
  by_cases h63 : nr = 63
  · subst h63; simp only [AArch64Env.syscall]; split
    · simp only [bind, Option.bind]; cases m.wrN a2 _ <;> simp
    · split
      · cases k.fds[a1 - 3]? with
        | none => simp
        | some o => cases o with
          | none => simp
          | some _ => simp only [bind, Option.bind]; cases m.wrN a2 _ <;> simp
      · simp
  by_cases h64 : nr = 64
  · subst h64; simp only [AArch64Env.syscall]; split
    · simp only [bind, Option.bind]; cases m.rdN a2 _ <;> simp
    · simp
  by_cases h215 : nr = 215
  · subst h215; simp only [AArch64Env.syscall]
    cases m.find a1 0 with
    | none => simp
    | some _ => split <;> simp
  by_cases h216 : nr = 216
  · subst h216; simp only [AArch64Env.syscall]; split
    · simp
    · simp only [bind, Option.bind]
      cases m.find a1 0 with
      | none => simp
      | some p =>
        rcases p with ⟨i, r⟩; dsimp only
        split
        · simp
        · split
          · simp
          · cases hpl : AArch64Env.place (m.eraseIdx i) k.next.1 (AArch64Env.pages a3) with
            | none => simp
            | some val =>
              cases val <;> simp
  by_cases h222 : nr = 222
  · subst h222; simp only [AArch64Env.syscall]; split
    · simp
    · split
      · simp
      · simp only [bind, Option.bind]
        cases hpl : AArch64Env.place m k.next.1 (AArch64Env.pages a2) with
        | none => simp
        | some val => cases val <;> simp
  simp [AArch64Env.syscall]

private theorem applySys_cases {s1 : St} {res : Option AArch64Env.SysRes}
    (h1len : s1.regs.length = 31) :
    match AArch64Phys.applySys s1 res with
    | .ok s' => s'.pc = s1.pc ∧ s'.regs.length = 31 ∧ ∀ r, 1 ≤ r → s'.reg r = s1.reg r
    | .exit c => res = some (.exit c)
    | .stuck => True := by
  cases res with
  | none => simp [AArch64Phys.applySys]
  | some r => cases r with
    | exit c => simp [AArch64Phys.applySys]
    | ret x k' m' =>
      refine ⟨rfl, setReg_len h1len 0 _, fun q hq => reg_set_ne (r1 := 0) (r2 := q) (Nat.ne_of_lt hq) _⟩

private theorem execSvc_cases {s : St} {nx : Nat} (hlen : s.regs.length = 31) :
    match AArch64Phys.execSvc s nx with
    | .ok s' => s'.pc = nx ∧ s'.regs.length = 31 ∧ ∀ r, 1 ≤ r → s'.reg r = s.reg r
    | .exit c => (s.reg 8 = some 93 ∨ s.reg 8 = some 94) ∧ ∃ v0, s.reg 0 = some v0 ∧ c = v0 % 256
    | .stuck => True := by
  unfold AArch64Phys.execSvc
  cases hx8 : s.reg 8 with
  | none => trivial
  | some nr =>
    by_cases h93 : nr = 93
    · subst h93; cases hx0 : s.reg 0 with
      | none => trivial
      | some a0 => simp [AArch64Phys.applySys, AArch64Env.syscall]
    by_cases h94 : nr = 94
    · subst h94; cases hx0 : s.reg 0 with
      | none => trivial
      | some a0 => simp [AArch64Phys.applySys, AArch64Env.syscall]
    have checkApp : ∀ a1 a2 a3 a4 a5 a6,
        match AArch64Phys.applySys { s with pc := nx }
            (AArch64Env.syscall s.k s.mem nr a1 a2 a3 a4 a5 a6) with
        | .ok s' => s'.pc = nx ∧ s'.regs.length = 31 ∧ ∀ r, 1 ≤ r → s'.reg r = s.reg r
        | .exit _ => False
        | .stuck => True := by
      intro a1 a2 a3 a4 a5 a6
      have ha := applySys_cases (s1 := { s with pc := nx })
        (res := AArch64Env.syscall s.k s.mem nr a1 a2 a3 a4 a5 a6) hlen
      cases hres : AArch64Phys.applySys { s with pc := nx }
          (AArch64Env.syscall s.k s.mem nr a1 a2 a3 a4 a5 a6) with
      | ok s' => rw [hres] at ha; exact ha
      | exit c => rw [hres] at ha; exact absurd ha (syscall_ne_exit h93 h94)
      | stuck => trivial
    by_cases h57 : nr = 57
    · subst h57; simp only; cases s.reg 0 with
      | none => trivial
      | some a0 => simpa using checkApp a0 0 0 0 0 0
    by_cases h215 : nr = 215
    · subst h215; simp only; cases s.reg 0 <;> cases s.reg 1 <;>
        first | trivial | simpa using checkApp _ _ 0 0 0 0
    by_cases h56 : nr = 56
    · subst h56; simp only; cases s.reg 0 <;> cases s.reg 1 <;> cases s.reg 2 <;>
        first | trivial | simpa using checkApp _ _ _ _ 0 0
    by_cases h63 : nr = 63
    · subst h63; simp only; cases s.reg 0 <;> cases s.reg 1 <;> cases s.reg 2 <;>
        first | trivial | simpa using checkApp _ _ _ 0 0 0
    by_cases h64 : nr = 64
    · subst h64; simp only; cases s.reg 0 <;> cases s.reg 1 <;> cases s.reg 2 <;>
        first | trivial | simpa using checkApp _ _ _ 0 0 0
    by_cases h216 : nr = 216
    · subst h216; simp only; cases s.reg 0 <;> cases s.reg 1 <;> cases s.reg 2 <;> cases s.reg 3 <;>
        first | trivial | simpa using checkApp _ _ _ _ 0 0
    by_cases h222 : nr = 222
    · subst h222; simp only; cases s.reg 0 <;> cases s.reg 1 <;> cases s.reg 2 <;>
        cases s.reg 3 <;> cases s.reg 4 <;> cases s.reg 5 <;>
        first | trivial | simpa using checkApp _ _ _ _ _ _
    simp [h93, h94, h57, h215, h56, h63, h64, h216, h222]

private theorem addSub_z_eq_beq {x y : Nat} (hx : x < AArch64Env.W) (hy : y < AArch64Env.W) :
    (addSubOp true 64 x y).2.z = some (x == y) := by
  by_cases h : x = y
  · subst h; rw [sub_eq_flags]; simp [TMRVoterArm64.voterFlagsEq]
  · rw [sub_z_ne hx hy h]; simp [h]

private theorem sub_150_mod256_ne0 {v : Nat} (hne : v % 256 ≠ ACCEPT % 256) :
    (addSubOp true 64 v (150 % pow2 64)).1 % 256 ≠ 0 := by
  have h150 : 150 % pow2 64 = 150 := by decide
  have hacc : ACCEPT % 256 = 150 := by decide
  rw [hacc] at hne
  rw [h150]
  have h256W : 256 ∣ 2 ^ 64 := by decide
  obtain ⟨m, hm⟩ := h256W
  intro hmod
  simp only [addSubOp, pow2, ↓reduceIte] at hmod
  have h1 : (v + 2 ^ 64 - 150) % 256 = 0 := by
    rw [← Nat.mod_mod_of_dvd (v + 2 ^ 64 - 150) ⟨m, hm⟩, hmod]
  have hpow : 2 ^ 64 = 256 * m := hm
  omega

private theorem tailSafe_pc_intro {base : Nat} (_hb : base_ok base) {s' : St} {k' : Nat}
    (hpc' : s'.pc = base + 4 * k') (hlen' : s'.regs.length = 31)
    (htk : k' < 48 → tailSafeAt s' k') : TailSafe base s' := by
  refine ⟨hlen', ?_⟩
  split
  · rw [show (s'.pc - base) / 4 = k' by rw [hpc']; omega]
    exact htk (by omega)
  · trivial

private theorem tailSafe_oob_intro {base : Nat} {s' : St}
    (hpc' : s'.pc = base + 4 * 48) (hlen' : s'.regs.length = 31) : TailSafe base s' := by
  refine ⟨hlen', ?_⟩
  split
  · next hb' => omega
  · trivial

private theorem TailSafe_step_0_10 {base k : Nat} (hb : base_ok base) {s : St}
    (hlen : s.regs.length = 31) (hk : k < 48) (hk11 : k < 11)
    (hpc : s.pc = base + 4 * k) (hsafe : tailSafeAt s k) :
    match step (tailImAt base) s with
    | .ok s' => TailSafe base s'
    | .exit c => c ≠ 0
    | .stuck => True := by
  have h1lt : (1 : Nat) < 31 := by decide
  have h2lt : (2 : Nat) < 31 := by decide
  have h6lt : (6 : Nat) < 31 := by decide
  have h19lt : (19 : Nat) < 31 := by decide
  have h20lt : (20 : Nat) < 31 := by decide
  have h21lt : (21 : Nat) < 31 := by decide
  have hts_pc := @tailSafe_pc_intro base hb
  · rcases (show k = 0 ∨ k = 1 ∨ k = 2 ∨ k = 3 ∨ k = 4 ∨ k = 5 ∨ k = 6 ∨ k = 7 ∨
        k = 8 ∨ k = 9 ∨ k = 10 by omega) with
      hk0 | hk0 | hk0 | hk0 | hk0 | hk0 | hk0 | hk0 | hk0 | hk0 | hk0
    · -- 0
      subst hk0; rcases hsafe with ⟨a, b, c, h19, h20, h21, ha, hbv, hc, hv⟩
      rw [step_at_word hb.1 hk hpc (by rfl)]
      have hb64 : b % 2 ^ 64 = b := mod_W_eq hbv
      have hmb : b % 2 ^ 64 % 2 ^ 64 * 2 ^ 0 % 2 ^ 64 = b := by
        simp only [hb64, show (2 ^ 0 : Nat) = 1 from rfl, Nat.mul_one]
      have hma : a % 2 ^ 64 = a := mod_W_eq ha
      simp only [exec, lift, bind, Option.bind, pure, Option.map_some, St.rdZR, h19, h20, wrZR_31_eq, hpc,
        show (19 == 31) = false by decide, show (20 == 31) = false by decide, Bool.false_eq_true, ↓reduceIte,
        shiftVal, pow2, hma, hmb, add_base_mod_W hb (by decide : 4 < 192)]
      exact hts_pc (k' := 1) (by first | rfl | omega | (dsimp only; omega)) hlen fun _ => Or.inr (Or.inr ⟨a, b, c, h19, h20, h21, ha, hbv, hc, hv,
        addSub_z_eq_beq ha hbv⟩)
    · -- 1
      subst hk0
      rw [step_at_word hb.1 hk hpc (by rfl)]
      rcases hsafe with ⟨v, h19, h20, h21, hnv⟩ | ⟨hz, a, b, c, h19, h20, h21, ha, hbv, hc⟩ |
          ⟨a, b, c, h19, h20, h21, ha, hbv, hc, hv, hzab⟩
      · simp only [exec, lift, bind, Option.bind, pure, Option.map_some, St.rdZR, h19, h21,
          show (19 == 31) = false by decide, show (21 == 31) = false by decide, Bool.false_eq_true, ↓reduceIte]
        cases hc1 : AArch64Phys.cond s.fl 1 with
        | none => simp
        | some b =>
          simp only [hpc, show (base + 4 * 1 + 4) % AArch64Env.W = base + 8 by
            rw [show base + 4 * 1 + 4 = base + 8 by omega]; exact add_base_mod_W hb (by decide)]
          exact hts_pc (k' := 2) (by first | rfl | omega | (dsimp only; omega)) hlen fun _ => Or.inl ⟨v, h19, h20, h21, hnv⟩
      · have hc1 : AArch64Phys.cond s.fl 1 = some false := by simp [AArch64Phys.cond, hz]
        simp only [exec, lift, bind, Option.bind, pure, Option.map_some, St.rdZR, h19, h21, hc1,
          Bool.false_eq_true, ↓reduceIte, hpc,
          show (base + 4 * 1 + 4) % AArch64Env.W = base + 8 by
            rw [show base + 4 * 1 + 4 = base + 8 by omega]; exact add_base_mod_W hb (by decide)]
        exact hts_pc (k' := 2) (by first | rfl | omega | (dsimp only; omega)) hlen fun _ => Or.inr (Or.inl ⟨rfl, a, b, c, h19, h20, h21, ha, hbv, hc⟩)
      · have hc1 : AArch64Phys.cond s.fl 1 = some (!(a == b)) := by simp [AArch64Phys.cond, hzab]
        simp only [exec, lift, bind, Option.bind, pure, Option.map_some, St.rdZR, h19, h21, hc1, hpc,
          show (19 == 31) = false by decide, show (21 == 31) = false by decide, Bool.false_eq_true, ↓reduceIte,
          mod_W_eq ha, mod_W_eq hc,
          show (base + 4 * 1 + 4) % AArch64Env.W = base + 8 by
            rw [show base + 4 * 1 + 4 = base + 8 by omega]; exact add_base_mod_W hb (by decide)]
        by_cases hab : a = b
        · subst hab
          have hna : valNotAcc a := by
            rcases hv with hn | ⟨v, hv1, hv2⟩
            · unfold TMR.vote at hn; simp at hn
            · unfold TMR.vote at hv1; simp at hv1; rw [hv1]; exact hv2
          simp only [beq_self_eq_true, Bool.not_true, Bool.false_eq_true, ↓reduceIte]
          exact hts_pc (k' := 2) (by first | rfl | omega | (dsimp only; omega)) hlen fun _ => Or.inr (Or.inr ⟨a, a, c, h19, h20, h21, ha, ha, hc,
            Or.inl ⟨rfl, hna⟩⟩)
        · have hne_b : (a == b) = false := beq_false_of_ne hab
          simp only [hne_b, Bool.not_false, ↓reduceIte]
          refine hts_pc (k' := 2) (by first | rfl | omega | (dsimp only; omega)) hlen fun _ => ?_
          by_cases hac : a = c
          · subst hac
            have hna : valNotAcc a := by
              rcases hv with hn | ⟨v, hv1, hv2⟩
              · unfold TMR.vote at hn; simp at hn
              · unfold TMR.vote at hv1; simp [hab] at hv1; rw [hv1]; exact hv2
            refine Or.inr (Or.inr ⟨a, b, a, h19, h20, h21, ha, hbv, ha,
              Or.inl ⟨by rw [sub_eq_flags]; rfl, hna⟩⟩)
          · refine Or.inr (Or.inr ⟨a, b, c, h19, h20, h21, ha, hbv, hc,
              Or.inr ⟨sub_z_ne ha hc hac, fun h => absurd h hac, fun hbc => by
                subst hbc
                rcases hv with hn | ⟨v, hv1, hv2⟩
                · unfold TMR.vote at hn; simp [hab] at hn
                · unfold TMR.vote at hv1; simp [hab] at hv1; rw [hv1]; exact hv2⟩⟩)
    · -- 2
      subst hk0
      rw [step_at_word hb.1 hk hpc (by rfl)]
      rcases hsafe with ⟨v, h19, h20, h21, hnv⟩ | ⟨hz, a, b, c, h19, h20, h21, ha, hbv, hc⟩ |
          ⟨a, b, c, h19, h20, h21, ha, hbv, hc, hfl2⟩
      · simp only [exec, lift, bind, Option.bind, pure, Option.map_some, St.rdZR, h19, h20,
          show (19 == 31) = false by decide, show (20 == 31) = false by decide, Bool.false_eq_true, ↓reduceIte]
        cases hc0 : AArch64Phys.cond s.fl 0 with
        | none => simp
        | some condb =>
          cases condb
          · simp only [Bool.false_eq_true, ↓reduceIte, hpc, mod_W_eq hnv.1,
              show (base + 4 * 2 + 4) % AArch64Env.W = base + 12 by
                rw [show base + 4 * 2 + 4 = base + 12 by omega]; exact add_base_mod_W hb (by decide)]
            exact hts_pc (k' := 3) (by first | rfl | omega | (dsimp only; omega)) (wrZR_pc_len hlen _ _ _) fun _ => Or.inl ⟨v,
              wrZR_pc_eq hlen 64 h19lt v |>.trans (by simp [mod_W_eq hnv.1]),
              by rw [wrZR_pc_ne 64 (by decide)]; exact h20, by rw [wrZR_pc_ne 64 (by decide)]; exact h21, hnv⟩
          · simp only [↓reduceIte, hpc, mod_W_eq hnv.1,
              show (base + 4 * 2 + 4) % AArch64Env.W = base + 12 by
                rw [show base + 4 * 2 + 4 = base + 12 by omega]; exact add_base_mod_W hb (by decide)]
            exact hts_pc (k' := 3) (by first | rfl | omega | (dsimp only; omega)) (wrZR_pc_len hlen _ _ _) fun _ => Or.inl ⟨v,
              wrZR_pc_eq hlen 64 h19lt v |>.trans (by simp [mod_W_eq hnv.1]),
              by rw [wrZR_pc_ne 64 (by decide)]; exact h20, by rw [wrZR_pc_ne 64 (by decide)]; exact h21, hnv⟩
      · have hc0 : AArch64Phys.cond s.fl 0 = some true := by simp [AArch64Phys.cond, hz]
        simp only [exec, lift, bind, Option.bind, pure, Option.map_some, St.rdZR, h19, h20, hc0, ↓reduceIte, hpc,
          mod_W_eq ha.1,
          show (base + 4 * 2 + 4) % AArch64Env.W = base + 12 by
            rw [show base + 4 * 2 + 4 = base + 12 by omega]; exact add_base_mod_W hb (by decide)]
        exact hts_pc (k' := 3) (by first | rfl | omega | (dsimp only; omega)) (wrZR_pc_len hlen _ _ _) fun _ => Or.inr (Or.inl ⟨hz, a, b, c,
          wrZR_pc_eq hlen 64 h19lt a |>.trans (by simp [mod_W_eq ha.1]),
          by rw [wrZR_pc_ne 64 (by decide)]; exact h20, by rw [wrZR_pc_ne 64 (by decide)]; exact h21, ha, hbv, hc⟩)
      · rcases hfl2 with ⟨hz, hna⟩ | ⟨hz, _, hbcn⟩
        · have hc0 : AArch64Phys.cond s.fl 0 = some true := by simp [AArch64Phys.cond, hz]
          simp only [exec, lift, bind, Option.bind, pure, Option.map_some, St.rdZR, h19, h20, hc0, ↓reduceIte, hpc,
            mod_W_eq ha,
            show (base + 4 * 2 + 4) % AArch64Env.W = base + 12 by
              rw [show base + 4 * 2 + 4 = base + 12 by omega]; exact add_base_mod_W hb (by decide)]
          exact hts_pc (k' := 3) (by first | rfl | omega | (dsimp only; omega)) (wrZR_pc_len hlen _ _ _) fun _ => Or.inr (Or.inr ⟨a, b, c,
            wrZR_pc_eq hlen 64 h19lt a |>.trans (by simp [mod_W_eq ha]),
            by rw [wrZR_pc_ne 64 (by decide)]; exact h20, by rw [wrZR_pc_ne 64 (by decide)]; exact h21, hbv, hc,
            Or.inl ⟨hz, hna⟩⟩)
        · have hc0 : AArch64Phys.cond s.fl 0 = some false := by simp [AArch64Phys.cond, hz]
          simp only [exec, lift, bind, Option.bind, pure, Option.map_some, St.rdZR, h19, h20, hc0, Bool.false_eq_true, ↓reduceIte, hpc,
            mod_W_eq hbv,
            show (base + 4 * 2 + 4) % AArch64Env.W = base + 12 by
              rw [show base + 4 * 2 + 4 = base + 12 by omega]; exact add_base_mod_W hb (by decide)]
          exact hts_pc (k' := 3) (by first | rfl | omega | (dsimp only; omega)) (wrZR_pc_len hlen _ _ _) fun _ => Or.inr (Or.inr ⟨b, b, c,
            wrZR_pc_eq hlen 64 h19lt b |>.trans (by simp [mod_W_eq hbv]),
            by rw [wrZR_pc_ne 64 (by decide)]; exact h20, by rw [wrZR_pc_ne 64 (by decide)]; exact h21, hbv, hc,
            Or.inr ⟨hz, fun hbc => ⟨hbcn hbc, hbcn hbc⟩⟩⟩)
    · -- 3
      subst hk0
      rw [step_at_word hb.1 hk hpc (by rfl)]
      rcases hsafe with ⟨v, h19, h20, h21, hnv⟩ | ⟨hz, a, b, c, h19, h20, h21, ha, hbv, hc⟩ |
          ⟨v19, b, c, h19, h20, h21, hbv, hc, hfl3⟩
      · simp only [exec, lift, bind, Option.bind, pure, Option.map_some, St.rdZR, h20, h21,
          show (20 == 31) = false by decide, show (21 == 31) = false by decide, Bool.false_eq_true, ↓reduceIte]
        cases hc1 : AArch64Phys.cond s.fl 1 with
        | none => simp
        | some condb =>
          simp only [hpc, show (base + 4 * 3 + 4) % AArch64Env.W = base + 16 by
            rw [show base + 4 * 3 + 4 = base + 16 by omega]; exact add_base_mod_W hb (by decide)]
          exact hts_pc (k' := 4) (by first | rfl | omega | (dsimp only; omega)) hlen fun _ => Or.inl ⟨v, h19, h20, h21, hnv⟩
      · have hc1 : AArch64Phys.cond s.fl 1 = some false := by simp [AArch64Phys.cond, hz]
        simp only [exec, lift, bind, Option.bind, pure, Option.map_some, St.rdZR, h20, h21, hc1,
          Bool.false_eq_true, ↓reduceIte, hpc,
          show (base + 4 * 3 + 4) % AArch64Env.W = base + 16 by
            rw [show base + 4 * 3 + 4 = base + 16 by omega]; exact add_base_mod_W hb (by decide)]
        exact hts_pc (k' := 4) (by first | rfl | omega | (dsimp only; omega)) hlen fun _ => Or.inr (Or.inl ⟨rfl, a, h19, ha⟩)
      · rcases hfl3 with ⟨hz, hnv19⟩ | ⟨hz, hbc⟩
        · have hc1 : AArch64Phys.cond s.fl 1 = some false := by simp [AArch64Phys.cond, hz]
          simp only [exec, lift, bind, Option.bind, pure, Option.map_some, St.rdZR, h20, h21, hc1,
            Bool.false_eq_true, ↓reduceIte, hpc,
            show (base + 4 * 3 + 4) % AArch64Env.W = base + 16 by
              rw [show base + 4 * 3 + 4 = base + 16 by omega]; exact add_base_mod_W hb (by decide)]
          exact hts_pc (k' := 4) (by first | rfl | omega | (dsimp only; omega)) hlen fun _ => Or.inr (Or.inl ⟨rfl, v19, h19, hnv19⟩)
        · have hc1 : AArch64Phys.cond s.fl 1 = some true := by simp [AArch64Phys.cond, hz]
          simp only [exec, lift, bind, Option.bind, pure, Option.map_some, St.rdZR, h20, h21, hc1, ↓reduceIte, hpc,
            show (20 == 31) = false by decide, show (21 == 31) = false by decide, Bool.false_eq_true,
            mod_W_eq hbv, mod_W_eq hc,
            show (base + 4 * 3 + 4) % AArch64Env.W = base + 16 by
              rw [show base + 4 * 3 + 4 = base + 16 by omega]; exact add_base_mod_W hb (by decide)]
          by_cases heqbc : b = c
          · subst heqbc
            rw [sub_eq_flags]
            exact hts_pc (k' := 4) (by first | rfl | omega | (dsimp only; omega)) hlen fun _ => Or.inr (Or.inl ⟨rfl, v19, h19, (hbc rfl).2⟩)
          · exact hts_pc (k' := 4) (by first | rfl | omega | (dsimp only; omega)) hlen fun _ => Or.inr (Or.inr (sub_z_ne hbv hc heqbc))
    · -- 4
      subst hk0
      rw [step_at_word hb.1 hk hpc (by rfl)]
      rcases hsafe with ⟨v, h19, h20, h21, hnv⟩ | ⟨hz, v19, h19, hnv⟩ | hz
      · simp only [exec, lift, bind, Option.bind, pure]
        cases hc1 : AArch64Phys.cond s.fl 1 with
        | none => simp
        | some condb =>
          cases condb
          · simp [hpc, show (base + 4 * 4 + 4) % AArch64Env.W = base + 4 * 5 by
              rw [show base + 4 * 4 + 4 = base + 20 by omega]; exact add_base_mod_W hb (by decide)]
            exact hts_pc (k' := 5) (by first | rfl | omega | (dsimp only; omega)) hlen fun _ => ⟨v, h19, hnv⟩
          · simp [hpc, show (base + 4 * 4 + 16) % AArch64Env.W = base + 4 * 8 by
              rw [show base + 4 * 4 + 16 = base + 32 by omega]; exact add_base_mod_W hb (by decide)]
            exact hts_pc (k' := 8) (by first | rfl | omega | (dsimp only; omega)) hlen fun _ => trivial
      · have hc1 : AArch64Phys.cond s.fl 1 = some false := by simp [AArch64Phys.cond, hz]
        simp [exec, lift, bind, Option.bind, pure, hc1, hpc,
          show (base + 4 * 4 + 4) % AArch64Env.W = base + 4 * 5 by
            rw [show base + 4 * 4 + 4 = base + 20 by omega]; exact add_base_mod_W hb (by decide)]
        exact hts_pc (k' := 5) (by first | rfl | omega | (dsimp only; omega)) hlen fun _ => ⟨v19, h19, hnv⟩
      · have hc1 : AArch64Phys.cond s.fl 1 = some true := by simp [AArch64Phys.cond, hz]
        simp [exec, lift, bind, Option.bind, pure, hc1, hpc,
          show (base + 4 * 4 + 16) % AArch64Env.W = base + 4 * 8 by
            rw [show base + 4 * 4 + 16 = base + 32 by omega]; exact add_base_mod_W hb (by decide)]
        exact hts_pc (k' := 8) (by first | rfl | omega | (dsimp only; omega)) hlen fun _ => trivial
    · -- 5
      subst hk0; rcases hsafe with ⟨v19, h19, hnv⟩
      rw [step_at_word hb.1 hk hpc (by rfl)]
      have hlor : Nat.lor 0 (v19 % 2 ^ 64 % 2 ^ 64 * 2 ^ 0 % 2 ^ 64) = v19 := by
        have h' : v19 % 2 ^ 64 = v19 := mod_W_eq hnv.1
        simp only [h', show (2 ^ 0 : Nat) = 1 from rfl, Nat.mul_one]
        exact zero_lor_eq v19
      simp only [exec, lift, bind, Option.bind, pure, Option.map_some, St.rdZR, h19, hpc,
        show (19 == 31) = false by decide, show (31 == 31) = true by decide, Bool.false_eq_true, ↓reduceIte,
        shiftVal, pow2, hlor,
        show (base + 4 * 5 + 4) % AArch64Env.W = base + 4 * 6 by
          rw [show base + 4 * 5 + 4 = base + 24 by omega]; exact add_base_mod_W hb (by decide)]
      exact hts_pc (k' := 6) (by first | rfl | omega | (dsimp only; omega)) (wrZR_pc_len hlen _ _ _) fun _ => ⟨v19, v19,
        by rw [wrZR_pc_ne 64 (by decide)]; exact h19,
        wrZR_pc_eq hlen 64 h20lt v19 |>.trans (by simp [mod_W_eq hnv.1]), hnv, hnv⟩
    · -- 6
      subst hk0; rcases hsafe with ⟨v19, v20, h19, h20, hnv19, hnv20⟩
      rw [step_at_word hb.1 hk hpc (by rfl)]
      have hlor : Nat.lor 0 (v19 % 2 ^ 64 % 2 ^ 64 * 2 ^ 0 % 2 ^ 64) = v19 := by
        have h' : v19 % 2 ^ 64 = v19 := mod_W_eq hnv19.1
        simp only [h', show (2 ^ 0 : Nat) = 1 from rfl, Nat.mul_one]
        exact zero_lor_eq v19
      simp only [exec, lift, bind, Option.bind, pure, Option.map_some, St.rdZR, h19, hpc,
        show (19 == 31) = false by decide, show (31 == 31) = true by decide, Bool.false_eq_true, ↓reduceIte,
        shiftVal, pow2, hlor,
        show (base + 4 * 6 + 4) % AArch64Env.W = base + 4 * 7 by
          rw [show base + 4 * 6 + 4 = base + 28 by omega]; exact add_base_mod_W hb (by decide)]
      exact hts_pc (k' := 7) (by first | rfl | omega | (dsimp only; omega)) (wrZR_pc_len hlen _ _ _) fun _ => ⟨v19, v20,
        by rw [wrZR_pc_ne 64 (by decide)]; exact h19, by rw [wrZR_pc_ne 64 (by decide)]; exact h20, hnv19, hnv20⟩
    · -- 7
      subst hk0; rcases hsafe with ⟨v19, v20, h19, h20, _, hnv20⟩
      rw [step_b_rel hb hk (by rfl) (by decide) s hpc]
      exact hts_pc (k' := 11) (by first | rfl | omega | (dsimp only; omega)) hlen fun _ => Or.inl ⟨v20, h20, hnv20⟩
    · -- 8
      subst hk0
      rw [step_movz_64 hb hk (by decide) (by rfl) s hpc]
      exact hts_pc rfl (wrZR_pc_len hlen _ _ _) fun _ => ⟨3, by simpa [pow2] using wrZR_pc_eq hlen 64 (by decide) 3, by decide⟩
    · -- 9
      subst hk0; rcases hsafe with ⟨v0, hx0, hnz⟩
      rw [step_movz_64 hb hk (by decide) (by rfl) s hpc]
      refine hts_pc rfl (wrZR_pc_len hlen _ _ _) fun _ => ⟨?_, ?_⟩
      · intro _; exact ⟨v0, by rw [wrZR_pc_ne 64 (by decide)]; exact hx0, hnz⟩
      · intro v8 hv8 hne93 _
        exfalso; apply hne93
        have h : ({ s with pc := base + 4 * (9 + 1) }.wrZR 64 8 (93 * pow2 (0 * 16))).reg 8 =
            some ((93 * pow2 (0 * 16)) % pow2 64) :=
          wrZR_pc_eq hlen 64 (by decide) _
        rw [hv8] at h; simpa [pow2] using h
    · -- 10
      subst hk0; rcases hsafe with ⟨hex, hnoex⟩
      rw [step_at_word hb.1 hk hpc (by rfl)]
      have hsvc := execSvc_cases (s := s) (nx := (base + 4 * 10 + 4) % AArch64Env.W) hlen
      simp only [exec, hpc, bne_self_eq_false, Bool.false_eq_true, ↓reduceIte]
      cases hres : AArch64Phys.execSvc s ((base + 4 * 10 + 4) % AArch64Env.W) with
      | ok s' =>
        rw [hres] at hsvc
        have hnx : (base + 4 * 10 + 4) % AArch64Env.W = base + 4 * 11 := by
          rw [show base + 4 * 10 + 4 = base + 44 by omega]; exact add_base_mod_W hb (by decide)
        refine hts_pc (hsvc.1.trans hnx) hsvc.2.1 fun _ => ?_
        cases hx8 : s.reg 8 with
        | none =>
          exfalso; unfold AArch64Phys.execSvc at hres; rw [hx8] at hres; cases hres
        | some nr =>
          have hne93 : nr ≠ 93 := by
            intro h; subst h; unfold AArch64Phys.execSvc at hres
            rw [hx8] at hres; cases hx0 : s.reg 0 <;> simp [hx0, AArch64Phys.applySys, AArch64Env.syscall] at hres
          have hne94 : nr ≠ 94 := by
            intro h; subst h; unfold AArch64Phys.execSvc at hres
            rw [hx8] at hres; cases hx0 : s.reg 0 <;> simp [hx0, AArch64Phys.applySys, AArch64Env.syscall] at hres
          rcases hnoex nr hx8 hne93 hne94 with ⟨v20, h20, hnv⟩ | ⟨v19, h19, hnv⟩
          · exact Or.inl ⟨v20, (hsvc.2.2 20 (by decide)).trans h20, hnv⟩
          · exact Or.inr ⟨v19, (hsvc.2.2 19 (by decide)).trans h19, hnv⟩
      | exit c =>
        rw [hres] at hsvc
        rcases hex hsvc.1 with ⟨v0, hx0, hnz⟩
        rcases hsvc.2 with ⟨v0', hx0', hc⟩
        rw [hx0] at hx0'; cases hx0'; rw [hc]; exact hnz
      | stuck => trivial

private theorem TailSafe_step_11_20 {base k : Nat} (hb : base_ok base) {s : St}
    (hlen : s.regs.length = 31) (hk : k < 48) (hklo : 11 ≤ k) (hkhi : k < 21)
    (hpc : s.pc = base + 4 * k) (hsafe : tailSafeAt s k) :
    match step (tailImAt base) s with
    | .ok s' => TailSafe base s'
    | .exit c => c ≠ 0
    | .stuck => True := by
  have h1lt : (1 : Nat) < 31 := by decide
  have h19lt : (19 : Nat) < 31 := by decide
  have hts_pc := @tailSafe_pc_intro base hb
  · rcases (show k = 11 ∨ k = 12 ∨ k = 13 ∨ k = 14 ∨ k = 15 ∨ k = 16 ∨ k = 17 ∨ k = 18 ∨
        k = 19 ∨ k = 20 by omega) with
      hk0 | hk0 | hk0 | hk0 | hk0 | hk0 | hk0 | hk0 | hk0 | hk0
    · -- 11
      subst hk0
      rw [step_movz_64 hb hk (by decide) (by rfl) s hpc]
      exact hts_pc rfl (wrZR_pc_len hlen _ _ _) fun _ => by
        rcases hsafe with ⟨v20, h20, hnv⟩ | ⟨v19, h19, hnv⟩
        · exact Or.inl ⟨v20, by rw [wrZR_pc_ne 64 (by decide)]; exact h20, hnv⟩
        · exact Or.inr ⟨v19, by rw [wrZR_pc_ne 64 (by decide)]; exact h19, hnv⟩
    · -- 12
      subst hk0
      rw [step_movz_64 hb hk (by decide) (by rfl) s hpc]
      exact hts_pc rfl (wrZR_pc_len hlen _ _ _) fun _ => by
        rcases hsafe with ⟨v20, h20, hnv⟩ | ⟨v19, h19, hnv⟩
        · exact Or.inl ⟨v20, by rw [wrZR_pc_ne 64 (by decide)]; exact h20, hnv⟩
        · exact Or.inr ⟨v19, by rw [wrZR_pc_ne 64 (by decide)]; exact h19, hnv⟩
    · -- 13
      subst hk0
      rw [step_movz_64 hb hk (by decide) (by rfl) s hpc]
      exact hts_pc rfl (wrZR_pc_len hlen _ _ _) fun _ => by
        rcases hsafe with ⟨v20, h20, hnv⟩ | ⟨v19, h19, hnv⟩
        · exact Or.inl ⟨v20, by rw [wrZR_pc_ne 64 (by decide)]; exact h20, hnv⟩
        · exact Or.inr ⟨v19, by rw [wrZR_pc_ne 64 (by decide)]; exact h19, hnv⟩
    · -- 14
      subst hk0
      rw [step_movz_64 hb hk (by decide) (by rfl) s hpc]
      exact hts_pc rfl (wrZR_pc_len hlen _ _ _) fun _ => by
        rcases hsafe with ⟨v20, h20, hnv⟩ | ⟨v19, h19, hnv⟩
        · exact Or.inl ⟨v20, by rw [wrZR_pc_ne 64 (by decide)]; exact h20, hnv⟩
        · exact Or.inr ⟨v19, by rw [wrZR_pc_ne 64 (by decide)]; exact h19, hnv⟩
    · -- 15
      subst hk0
      rw [step_movz_64 hb hk (by decide) (by rfl) s hpc]
      exact hts_pc rfl (wrZR_pc_len hlen _ _ _) fun _ => by
        rcases hsafe with ⟨v20, h20, hnv⟩ | ⟨v19, h19, hnv⟩
        · exact Or.inl ⟨v20, by rw [wrZR_pc_ne 64 (by decide)]; exact h20, hnv⟩
        · exact Or.inr ⟨by simpa [pow2] using wrZR_pc_eq hlen 64 h1lt (0x3c96 * pow2 (0 * 16)),
            v19, by rw [wrZR_pc_ne 64 (by decide)]; exact h19, hnv⟩
    · -- 16
      subst hk0
      cases h1 : s.reg 1 with
      | none => rw [step_at_word hb.1 hk hpc (by rfl)]; simp [exec, lift, bind, Option.bind, St.rdZR, h1]
      | some old =>
        rw [step_movk_64 hb hk (by decide) (by rfl) s hpc h1lt h1]
        exact hts_pc rfl (wrZR_pc_len hlen _ _ _) fun _ => by
          rcases hsafe with ⟨v20, h20, hnv⟩ | ⟨hx1, v19, h19, hnv⟩
          · exact Or.inl ⟨v20, by rw [wrZR_pc_ne 64 (by decide)]; exact h20, hnv⟩
          · rw [hx1] at h1; cases h1
            exact Or.inr ⟨by simpa [pow2] using wrZR_pc_eq hlen 64 h1lt _,
              v19, by rw [wrZR_pc_ne 64 (by decide)]; exact h19, hnv⟩
    · -- 17
      subst hk0
      cases h1 : s.reg 1 with
      | none => rw [step_at_word hb.1 hk hpc (by rfl)]; simp [exec, lift, bind, Option.bind, St.rdZR, h1]
      | some old =>
        rw [step_movk_64 hb hk (by decide) (by rfl) s hpc h1lt h1]
        exact hts_pc rfl (wrZR_pc_len hlen _ _ _) fun _ => by
          rcases hsafe with ⟨v20, h20, hnv⟩ | ⟨hx1, v19, h19, hnv⟩
          · exact Or.inl ⟨v20, by rw [wrZR_pc_ne 64 (by decide)]; exact h20, hnv⟩
          · rw [hx1] at h1; cases h1
            exact Or.inr ⟨by simpa [pow2] using wrZR_pc_eq hlen 64 h1lt _,
              v19, by rw [wrZR_pc_ne 64 (by decide)]; exact h19, hnv⟩
    · -- 18
      subst hk0
      cases h1 : s.reg 1 with
      | none => rw [step_at_word hb.1 hk hpc (by rfl)]; simp [exec, lift, bind, Option.bind, St.rdZR, h1]
      | some old =>
        rw [step_movk_64 hb hk (by decide) (by rfl) s hpc h1lt h1]
        exact hts_pc rfl (wrZR_pc_len hlen _ _ _) fun _ => by
          rcases hsafe with ⟨v20, h20, hnv⟩ | ⟨hx1, v19, h19, hnv⟩
          · exact Or.inl ⟨v20, by rw [wrZR_pc_ne 64 (by decide)]; exact h20, hnv⟩
          · rw [hx1] at h1; cases h1
            exact Or.inr ⟨by simpa [ACCEPT, pow2] using wrZR_pc_eq hlen 64 h1lt _,
              v19, by rw [wrZR_pc_ne 64 (by decide)]; exact h19, hnv⟩
    · -- 19
      subst hk0
      cases h19 : s.reg 19 with
      | none => rw [step_at_word hb.1 hk hpc (by rfl)]; simp [exec, lift, bind, Option.bind, St.rdZR, h19]
      | some v19 =>
        cases h1 : s.reg 1 with
        | none => rw [step_at_word hb.1 hk hpc (by rfl)]; simp [exec, lift, bind, Option.bind, St.rdZR, h19, h1]
        | some v1 =>
          rw [step_cmp_64 hb hk (by decide) (by rfl) s hpc h19lt h1lt h19 h1]
          exact hts_pc rfl hlen fun _ => by
            rcases hsafe with ⟨v20, h20, hnv⟩ | ⟨hx1, v19', h19', hnv⟩
            · exact Or.inr ⟨v20, h20, hnv⟩
            · rw [hx1] at h1; cases h1; rw [h19'] at h19; cases h19
              rw [mod_W_eq hnv.1, mod_W_eq (by decide : ACCEPT < AArch64Env.W)]
              exact Or.inl (sub_z_ne hnv.1 (by decide) hnv.2.1)
    · -- 20
      subst hk0
      rw [step_at_word hb.1 hk hpc (by rfl)]
      simp only [exec, lift, bind, Option.bind, pure]
      cases hc1 : AArch64Phys.cond s.fl 1 with
      | none => simp
      | some condb =>
        cases condb
        · rcases hsafe with hfl | ⟨v20, h20, hnv⟩
          · exfalso; simp [AArch64Phys.cond, hfl] at hc1
          · simp [hpc, show (base + 4 * 20 + 4) % AArch64Env.W = base + 4 * 21 by
              rw [show base + 4 * 20 + 4 = base + 84 by omega]; exact add_base_mod_W hb (by decide)]
            exact hts_pc (k' := 21) (by first | rfl | omega | (dsimp only; omega)) hlen fun _ => ⟨v20, h20, hnv⟩
        · simp [hpc, show (base + 4 * 20 + 48) % AArch64Env.W = base + 4 * 32 by
            rw [show base + 4 * 20 + 48 = base + 128 by omega]; exact add_base_mod_W hb (by decide)]
          exact hts_pc (k' := 32) (by first | rfl | omega | (dsimp only; omega)) hlen fun _ => trivial

private theorem TailSafe_step_21_26 {base k : Nat} (hb : base_ok base) {s : St}
    (hlen : s.regs.length = 31) (hk : k < 48) (hklo : 21 ≤ k) (hkhi : k < 27)
    (hpc : s.pc = base + 4 * k) (hsafe : tailSafeAt s k) :
    match step (tailImAt base) s with
    | .ok s' => TailSafe base s'
    | .exit c => c ≠ 0
    | .stuck => True := by
  have h6lt : (6 : Nat) < 31 := by decide
  have h20lt : (20 : Nat) < 31 := by decide
  have hts_pc := @tailSafe_pc_intro base hb
  · rcases (show k = 21 ∨ k = 22 ∨ k = 23 ∨ k = 24 ∨ k = 25 ∨ k = 26 by omega) with
      hk0 | hk0 | hk0 | hk0 | hk0 | hk0
    · -- 21
      subst hk0; rcases hsafe with ⟨v20, h20, hnv⟩
      rw [step_movz_64 hb hk (by decide) (by rfl) s hpc]
      exact hts_pc rfl (wrZR_pc_len hlen _ _ _) fun _ => ⟨v20, by rw [wrZR_pc_ne 64 (by decide)]; exact h20, hnv⟩
    · -- 22
      subst hk0; rcases hsafe with ⟨v20, h20, hnv⟩
      rw [step_movz_64 hb hk (by decide) (by rfl) s hpc]
      exact hts_pc rfl (wrZR_pc_len hlen _ _ _) fun _ => ⟨v20, by rw [wrZR_pc_ne 64 (by decide)]; exact h20, hnv⟩
    · -- 23
      subst hk0; rcases hsafe with ⟨v20, h20, hnv⟩
      cases h6 : s.reg 6 with
      | none => rw [step_at_word hb.1 hk hpc (by rfl)]; simp [exec, lift, bind, Option.bind, St.rdZR, h6]
      | some _ =>
        rw [step_movk_64 hb hk (by decide) (by rfl) s hpc h6lt h6]
        exact hts_pc rfl (wrZR_pc_len hlen _ _ _) fun _ => ⟨v20, by rw [wrZR_pc_ne 64 (by decide)]; exact h20, hnv⟩
    · -- 24
      subst hk0; rcases hsafe with ⟨v20, h20, hnv⟩
      cases h6 : s.reg 6 with
      | none => rw [step_at_word hb.1 hk hpc (by rfl)]; simp [exec, lift, bind, Option.bind, St.rdZR, h6]
      | some _ =>
        rw [step_movk_64 hb hk (by decide) (by rfl) s hpc h6lt h6]
        exact hts_pc rfl (wrZR_pc_len hlen _ _ _) fun _ => ⟨v20, by rw [wrZR_pc_ne 64 (by decide)]; exact h20, hnv⟩
    · -- 25
      subst hk0; rcases hsafe with ⟨v20, h20, hnv⟩
      cases h6 : s.reg 6 with
      | none => rw [step_at_word hb.1 hk hpc (by rfl)]; simp [exec, lift, bind, Option.bind, St.rdZR, h6]
      | some _ =>
        rw [step_movk_64 hb hk (by decide) (by rfl) s hpc h6lt h6]
        exact hts_pc rfl (wrZR_pc_len hlen _ _ _) fun _ => ⟨v20, by rw [wrZR_pc_ne 64 (by decide)]; exact h20, hnv⟩
    · -- 26
      subst hk0; rcases hsafe with ⟨v20, h20, hnv⟩
      cases h6 : s.reg 6 with
      | none => rw [step_at_word hb.1 hk hpc (by rfl)]; simp [exec, lift, bind, Option.bind, St.rdZR, h20, h6]
      | some _ =>
        rw [step_cmp_64 hb hk (by decide) (by rfl) s hpc h20lt h6lt h20 h6]
        exact hts_pc rfl hlen fun _ => Or.inr ⟨v20, h20, hnv⟩

private theorem TailSafe_step_27 {base : Nat} (hb : base_ok base) {s : St}
    (hlen : s.regs.length = 31) (hpc : s.pc = base + 4 * 27) (hsafe : tailSafeAt s 27) :
    match step (tailImAt base) s with
    | .ok s' => TailSafe base s'
    | .exit c => c ≠ 0
    | .stuck => True := by
  have hk : (27 : Nat) < 48 := by decide
  have hts_pc := @tailSafe_pc_intro base hb
  rw [step_at_word hb.1 hk hpc (by rfl)]
  simp only [exec, lift, bind, Option.bind, pure]
  cases hc1 : AArch64Phys.cond s.fl 1 with
  | none => simp
  | some condb =>
    cases condb
    · rcases hsafe with hfl | ⟨v20, h20, hnv⟩
      · exfalso; simp [AArch64Phys.cond, hfl] at hc1
      · simp [hpc, show (base + 4 * 27 + 4) % AArch64Env.W = base + 4 * 28 by
          rw [show base + 4 * 27 + 4 = base + 112 by omega]; exact add_base_mod_W hb (by decide)]
        exact hts_pc (k' := 28) (by first | rfl | omega | (dsimp only; omega)) hlen fun _ => ⟨v20, h20, hnv.1, hnv.2.2⟩
    · simp [hpc, show (base + 4 * 27 + 20) % AArch64Env.W = base + 4 * 32 by
        rw [show base + 4 * 27 + 20 = base + 128 by omega]; exact add_base_mod_W hb (by decide)]
      exact hts_pc (k' := 32) (by first | rfl | omega | (dsimp only; omega)) hlen fun _ => trivial

private theorem TailSafe_step_28 {base : Nat} (hb : base_ok base) {s : St}
    (hlen : s.regs.length = 31) (hpc : s.pc = base + 4 * 28) (hsafe : tailSafeAt s 28) :
    match step (tailImAt base) s with
    | .ok s' => TailSafe base s'
    | .exit c => c ≠ 0
    | .stuck => True := by
  have hk : (28 : Nat) < 48 := by decide
  have hts_pc := @tailSafe_pc_intro base hb
  rcases hsafe with ⟨v20, h20, hlt, hmod⟩
  rw [step_movz_64 hb hk (by decide) (by rfl) s hpc]
  exact hts_pc rfl (wrZR_pc_len hlen _ _ _) fun _ => ⟨v20, by rw [wrZR_pc_ne 64 (by decide)]; exact h20, hlt, hmod⟩

private theorem addSubOp_fst_lt_W (isSub : Bool) (a b : Nat) :
    (addSubOp isSub 64 a b).1 < pow2 64 := by
  unfold addSubOp
  split <;> exact Nat.mod_lt _ (Nat.two_pow_pos 64)

private theorem TailSafe_step_29 {base : Nat} (hb : base_ok base) {s : St}
    (hlen : s.regs.length = 31) (hpc : s.pc = base + 4 * 29) (hsafe : tailSafeAt s 29) :
    match step (tailImAt base) s with
    | .ok s' => TailSafe base s'
    | .exit c => c ≠ 0
    | .stuck => True := by
  have hk : (29 : Nat) < 48 := by decide
  have hk4 : 4 * 29 + 4 < 192 := by decide
  have h0lt : (0 : Nat) < 31 := by decide
  have h20lt : (20 : Nat) < 31 := by decide
  have hts_pc := @tailSafe_pc_intro base hb
  rcases hsafe with ⟨v20, h20, hlt, hmod⟩
  let res := (addSubOp true 64 v20 (150 % pow2 64)).1
  have hstep : step (tailImAt base) s = .ok ({ s with pc := base + 4 * 30 }.wrZR 64 0 res) :=
    step_subI_64_val hb hk hk4 rfl (by rfl) s hpc h0lt h20lt h20 hlt rfl
  rw [hstep]
  refine hts_pc (k' := 30) rfl (wrZR_pc_len hlen _ _ _) fun _ _ => ?_
  refine ⟨res, ?_, sub_150_mod256_ne0 hmod⟩
  have h0eq : ({ s with pc := base + 4 * 30 }.wrZR 64 0 res).reg 0 = some (res % pow2 64) :=
    wrZR_pc_eq hlen 64 h0lt res
  have hwlt : res < pow2 64 := addSubOp_fst_lt_W true v20 (150 % pow2 64)
  rw [Nat.mod_eq_of_lt hwlt] at h0eq
  exact h0eq

private theorem TailSafe_step_30 {base : Nat} (hb : base_ok base) {s : St}
    (hlen : s.regs.length = 31) (hpc : s.pc = base + 4 * 30) (hsafe : tailSafeAt s 30) :
    match step (tailImAt base) s with
    | .ok s' => TailSafe base s'
    | .exit c => c ≠ 0
    | .stuck => True := by
  have hk : (30 : Nat) < 48 := by decide
  have hts_pc := @tailSafe_pc_intro base hb
  rw [step_at_word hb.1 hk hpc (by rfl)]
  have hsvc := execSvc_cases (s := s) (nx := (base + 4 * 30 + 4) % AArch64Env.W) hlen
  simp only [exec, hpc, bne_self_eq_false, Bool.false_eq_true, ↓reduceIte]
  cases hres : AArch64Phys.execSvc s ((base + 4 * 30 + 4) % AArch64Env.W) with
  | ok s' =>
    rw [hres] at hsvc
    have hnx : (base + 4 * 30 + 4) % AArch64Env.W = base + 4 * 31 := by
      rw [show base + 4 * 30 + 4 = base + 124 by omega]; exact add_base_mod_W hb (by decide)
    exact hts_pc (hsvc.1.trans hnx) hsvc.2.1 fun _ => trivial
  | exit c =>
    rw [hres] at hsvc
    rcases hsafe hsvc.1 with ⟨v0, hx0, hnz⟩
    rcases hsvc.2 with ⟨v0', hx0', hc⟩
    rw [hx0] at hx0'; cases hx0'; rw [hc]; exact hnz
  | stuck => trivial

private theorem TailSafe_step_31 {base : Nat} (hb : base_ok base) {s : St}
    (hlen : s.regs.length = 31) (hpc : s.pc = base + 4 * 31) :
    match step (tailImAt base) s with
    | .ok s' => TailSafe base s'
    | .exit c => c ≠ 0
    | .stuck => True := by
  have hk : (31 : Nat) < 48 := by decide
  have hts_pc := @tailSafe_pc_intro base hb
  rw [step_b_rel hb hk (by rfl) (by decide) s hpc]
  exact hts_pc (k' := 43) (by first | rfl | omega | (dsimp only; omega)) hlen fun _ => trivial

private theorem TailSafe_step_27_31 {base k : Nat} (hb : base_ok base) {s : St}
    (hlen : s.regs.length = 31) (_hk : k < 48) (hklo : 27 ≤ k) (hkhi : k < 32)
    (hpc : s.pc = base + 4 * k) (hsafe : tailSafeAt s k) :
    match step (tailImAt base) s with
    | .ok s' => TailSafe base s'
    | .exit c => c ≠ 0
    | .stuck => True := by
  rcases (show k = 27 ∨ k = 28 ∨ k = 29 ∨ k = 30 ∨ k = 31 by omega) with
    hk0 | hk0 | hk0 | hk0 | hk0
  · subst hk0; exact TailSafe_step_27 hb hlen hpc hsafe
  · subst hk0; exact TailSafe_step_28 hb hlen hpc hsafe
  · subst hk0; exact TailSafe_step_29 hb hlen hpc hsafe
  · subst hk0; exact TailSafe_step_30 hb hlen hpc hsafe
  · subst hk0; exact TailSafe_step_31 hb hlen hpc

private theorem TailSafe_step_32_47 {base k : Nat} (hb : base_ok base) {s : St}
    (hlen : s.regs.length = 31) (hk : k < 48) (hklo : 32 ≤ k)
    (hpc : s.pc = base + 4 * k) (hsafe : tailSafeAt s k) :
    match step (tailImAt base) s with
    | .ok s' => TailSafe base s'
    | .exit c => c ≠ 0
    | .stuck => True := by
  have h1lt : (1 : Nat) < 31 := by decide
  have h2lt : (2 : Nat) < 31 := by decide
  have h19lt : (19 : Nat) < 31 := by decide
  have hts_pc := @tailSafe_pc_intro base hb
  have hts_oob := @tailSafe_oob_intro base
  · rcases (show k = 32 ∨ k = 33 ∨ k = 34 ∨ k = 35 ∨ k = 36 ∨ k = 37 ∨ k = 38 ∨ k = 39 ∨
        k = 40 ∨ k = 41 ∨ k = 42 ∨ k = 43 ∨ k = 44 ∨ k = 45 ∨ k = 46 ∨ k = 47 by omega) with
      hk0 | hk0 | hk0 | hk0 | hk0 | hk0 | hk0 | hk0 | hk0 | hk0 | hk0 | hk0 | hk0 | hk0 | hk0 | hk0
    · -- 32
      subst hk0
      rw [step_movz_64 hb hk (by decide) (by rfl) s hpc]
      exact hts_pc rfl (wrZR_pc_len hlen _ _ _) fun _ => trivial
    · -- 33
      subst hk0
      cases h1 : s.reg 1 with
      | none => rw [step_at_word hb.1 hk hpc (by rfl)]; simp [exec, lift, bind, Option.bind, St.rdZR, h1]
      | some _ =>
        rw [step_movk_64 hb hk (by decide) (by rfl) s hpc h1lt h1]
        exact hts_pc rfl (wrZR_pc_len hlen _ _ _) fun _ => trivial
    · -- 34
      subst hk0
      cases h1 : s.reg 1 with
      | none => rw [step_at_word hb.1 hk hpc (by rfl)]; simp [exec, lift, bind, Option.bind, St.rdZR, h1]
      | some _ =>
        rw [step_movk_64 hb hk (by decide) (by rfl) s hpc h1lt h1]
        exact hts_pc rfl (wrZR_pc_len hlen _ _ _) fun _ => trivial
    · -- 35
      subst hk0
      cases h1 : s.reg 1 with
      | none => rw [step_at_word hb.1 hk hpc (by rfl)]; simp [exec, lift, bind, Option.bind, St.rdZR, h1]
      | some _ =>
        rw [step_movk_64 hb hk (by decide) (by rfl) s hpc h1lt h1]
        exact hts_pc rfl (wrZR_pc_len hlen _ _ _) fun _ => trivial
    · -- 36
      subst hk0
      cases h19 : s.reg 19 with
      | none => rw [step_at_word hb.1 hk hpc (by rfl)]; simp [exec, lift, bind, Option.bind, St.rdZR, h19]
      | some v19 =>
        cases h1 : s.reg 1 with
        | none => rw [step_at_word hb.1 hk hpc (by rfl)]; simp [exec, lift, bind, Option.bind, St.rdZR, h19, h1]
        | some v1 =>
          rw [step_cmp_64 hb hk (by decide) (by rfl) s hpc h19lt h1lt h19 h1]
          exact hts_pc rfl hlen fun _ => trivial
    · -- 37
      subst hk0
      rw [step_at_word hb.1 hk hpc (by rfl)]
      simp only [exec, lift, bind, Option.bind, pure]
      cases hc1 : AArch64Phys.cond s.fl 1 with
      | none => simp
      | some condb =>
        cases condb
        · simp [hpc, show (base + 4 * 37 + 4) % AArch64Env.W = base + 4 * 38 by
            rw [show base + 4 * 37 + 4 = base + 152 by omega]; exact add_base_mod_W hb (by decide)]
          exact hts_pc (k' := 38) (by first | rfl | omega | (dsimp only; omega)) hlen fun _ => trivial
        · simp [hpc, show (base + 4 * 37 + 24) % AArch64Env.W = base + 4 * 43 by
            rw [show base + 4 * 37 + 24 = base + 172 by omega]; exact add_base_mod_W hb (by decide)]
          exact hts_pc (k' := 43) (by first | rfl | omega | (dsimp only; omega)) hlen fun _ => trivial
    · -- 38
      subst hk0
      rw [step_movz_64 hb hk (by decide) (by rfl) s hpc]
      exact hts_pc rfl (wrZR_pc_len hlen _ _ _) fun _ => trivial
    · -- 39
      subst hk0
      rw [step_movz_64 hb hk (by decide) (by rfl) s hpc]
      exact hts_pc rfl (wrZR_pc_len hlen _ _ _) fun _ => Or.inr fun _ _ =>
        ⟨6, by simpa [pow2] using wrZR_pc_eq hlen 64 h2lt 6, by decide⟩
    · -- 40
      subst hk0
      cases h0 : s.reg 0 with
      | none => rw [step_at_word hb.1 hk hpc (by rfl)]; simp [exec, lift, bind, Option.bind, St.rdZR, h0]
      | some v0 =>
        cases h2 : s.reg 2 with
        | none => rw [step_at_word hb.1 hk hpc (by rfl)]; simp [exec, lift, bind, Option.bind, St.rdZR, h0, h2]
        | some v2 =>
          rw [step_orr_64 hb hk (by decide) (by rfl) s hpc (by decide) h2lt h0 h2]
          refine hts_pc rfl (wrZR_pc_len hlen _ _ _) fun _ => ?_
          have hnz : Nat.lor (v0 % pow2 64) (v2 % pow2 64) % 256 ≠ 0 := by
            apply lor_mod256_ne0
            rcases hsafe with ⟨v0', h0', h0nz⟩ | hfn
            · rw [h0'] at h0; cases h0; left; rw [show pow2 64 = 2 ^ 64 from rfl, Nat.mod_mod_of_dvd _ (by decide)]; exact h0nz
            · rcases hfn v0 h0 with ⟨v2', h2', h2nz⟩; rw [h2'] at h2; cases h2
              right; rw [show pow2 64 = 2 ^ 64 from rfl, Nat.mod_mod_of_dvd _ (by decide)]; exact h2nz
          have hlt64 : Nat.lor (v0 % pow2 64) (v2 % pow2 64) < pow2 64 :=
            Nat.bitwise_lt_two_pow (Nat.mod_lt _ (by decide)) (Nat.mod_lt _ (by decide))
          exact ⟨Nat.lor (v0 % pow2 64) (v2 % pow2 64),
            (wrZR_pc_eq hlen 64 (by decide) _).trans (congrArg some (Nat.mod_eq_of_lt hlt64)), hnz⟩
    · -- 41
      subst hk0; rcases hsafe with ⟨v0, hx0, hnz⟩
      rw [step_movz_64 hb hk (by decide) (by rfl) s hpc]
      exact hts_pc rfl (wrZR_pc_len hlen _ _ _) fun _ _ => ⟨v0, by rw [wrZR_pc_ne 64 (by decide)]; exact hx0, hnz⟩
    · -- 42
      subst hk0
      rw [step_at_word hb.1 hk hpc (by rfl)]
      have hsvc := execSvc_cases (s := s) (nx := (base + 4 * 42 + 4) % AArch64Env.W) hlen
      simp only [exec, hpc, bne_self_eq_false, Bool.false_eq_true, ↓reduceIte]
      cases hres : AArch64Phys.execSvc s ((base + 4 * 42 + 4) % AArch64Env.W) with
      | ok s' =>
        rw [hres] at hsvc
        have hnx : (base + 4 * 42 + 4) % AArch64Env.W = base + 4 * 43 := by
          rw [show base + 4 * 42 + 4 = base + 172 by omega]; exact add_base_mod_W hb (by decide)
        exact hts_pc (hsvc.1.trans hnx) hsvc.2.1 fun _ => trivial
      | exit c =>
        rw [hres] at hsvc
        rcases hsafe hsvc.1 with ⟨v0, hx0, hnz⟩
        rcases hsvc.2 with ⟨v0', hx0', hc⟩
        rw [hx0] at hx0'; cases hx0'; rw [hc]; exact hnz
      | stuck => trivial
    · -- 43
      subst hk0
      rw [step_movz_64 hb hk (by decide) (by rfl) s hpc]
      exact hts_pc rfl (wrZR_pc_len hlen _ _ _) fun _ => trivial
    · -- 44
      subst hk0
      rw [step_movz_64 hb hk (by decide) (by rfl) s hpc]
      exact hts_pc rfl (wrZR_pc_len hlen _ _ _) fun _ => Or.inr fun _ _ =>
        ⟨5, by simpa [pow2] using wrZR_pc_eq hlen 64 h2lt 5, by decide⟩
    · -- 45
      subst hk0
      cases h0 : s.reg 0 with
      | none => rw [step_at_word hb.1 hk hpc (by rfl)]; simp [exec, lift, bind, Option.bind, St.rdZR, h0]
      | some v0 =>
        cases h2 : s.reg 2 with
        | none => rw [step_at_word hb.1 hk hpc (by rfl)]; simp [exec, lift, bind, Option.bind, St.rdZR, h0, h2]
        | some v2 =>
          rw [step_orr_64 hb hk (by decide) (by rfl) s hpc (by decide) h2lt h0 h2]
          refine hts_pc rfl (wrZR_pc_len hlen _ _ _) fun _ => ?_
          have hnz : Nat.lor (v0 % pow2 64) (v2 % pow2 64) % 256 ≠ 0 := by
            apply lor_mod256_ne0
            rcases hsafe with ⟨v0', h0', h0nz⟩ | hfn
            · rw [h0'] at h0; cases h0; left; rw [show pow2 64 = 2 ^ 64 from rfl, Nat.mod_mod_of_dvd _ (by decide)]; exact h0nz
            · rcases hfn v0 h0 with ⟨v2', h2', h2nz⟩; rw [h2'] at h2; cases h2
              right; rw [show pow2 64 = 2 ^ 64 from rfl, Nat.mod_mod_of_dvd _ (by decide)]; exact h2nz
          have hlt64 : Nat.lor (v0 % pow2 64) (v2 % pow2 64) < pow2 64 :=
            Nat.bitwise_lt_two_pow (Nat.mod_lt _ (by decide)) (Nat.mod_lt _ (by decide))
          exact ⟨Nat.lor (v0 % pow2 64) (v2 % pow2 64),
            (wrZR_pc_eq hlen 64 (by decide) _).trans (congrArg some (Nat.mod_eq_of_lt hlt64)), hnz⟩
    · -- 46
      subst hk0; rcases hsafe with ⟨v0, hx0, hnz⟩
      rw [step_movz_64 hb hk (by decide) (by rfl) s hpc]
      exact hts_pc rfl (wrZR_pc_len hlen _ _ _) fun _ _ => ⟨v0, by rw [wrZR_pc_ne 64 (by decide)]; exact hx0, hnz⟩
    · -- 47
      subst hk0
      rw [step_at_word hb.1 hk hpc (by rfl)]
      have hsvc := execSvc_cases (s := s) (nx := (base + 4 * 47 + 4) % AArch64Env.W) hlen
      simp only [exec, hpc, bne_self_eq_false, Bool.false_eq_true, ↓reduceIte]
      cases hres : AArch64Phys.execSvc s ((base + 4 * 47 + 4) % AArch64Env.W) with
      | ok s' =>
        rw [hres] at hsvc
        have hnx : (base + 4 * 47 + 4) % AArch64Env.W = base + 4 * 48 := by
          rw [show base + 4 * 47 + 4 = base + 192 by omega]
          rcases hb with ⟨_, _, hhi⟩; exact Nat.mod_eq_of_lt (by omega)
        exact hts_oob (hsvc.1.trans hnx) hsvc.2.1
      | exit c =>
        rw [hres] at hsvc
        rcases hsafe hsvc.1 with ⟨v0, hx0, hnz⟩
        rcases hsvc.2 with ⟨v0', hx0', hc⟩
        rw [hx0] at hx0'; cases hx0'; rw [hc]; exact hnz
      | stuck => trivial

theorem TailSafe_step {base : Nat} (hb : base_ok base) {s : St} (ht : TailSafe base s) :
    match step (tailImAt base) s with
    | .ok s' => TailSafe base s'
    | .exit c => c ≠ 0
    | .stuck => True := by
  rcases ht with ⟨hlen, hsafe⟩
  by_cases hbounds : base ≤ s.pc ∧ s.pc < base + 192 ∧ (s.pc - base) % 4 = 0
  · rw [dif_pos hbounds] at hsafe
    obtain ⟨hle, hhi, hal⟩ := hbounds
    obtain ⟨k, hk_def⟩ : ∃ k, (s.pc - base) / 4 = k := ⟨_, rfl⟩
    rw [hk_def] at hsafe
    have hk : k < 48 := by omega
    have hpc : s.pc = base + 4 * k := by omega
    by_cases hk11 : k < 11
    · exact TailSafe_step_0_10 hb hlen hk hk11 hpc hsafe
    · by_cases hk21 : k < 21
      · exact TailSafe_step_11_20 hb hlen hk (by omega) hk21 hpc hsafe
      · by_cases hk27 : k < 27
        · exact TailSafe_step_21_26 hb hlen hk (by omega) hk27 hpc hsafe
        · by_cases hk32 : k < 32
          · exact TailSafe_step_27_31 hb hlen hk (by omega) hk32 hpc hsafe
          · exact TailSafe_step_32_47 hb hlen hk (by omega) hpc hsafe
  · rw [step]
    have him : tailImAt base s.pc = none := by
      apply tailImAt_out
      by_cases h1 : s.pc < base
      · exact Or.inl h1
      · by_cases h2 : base + 192 ≤ s.pc + 3
        · exact Or.inr (Or.inl h2)
        · exact Or.inr (Or.inr fun hal => hbounds ⟨by omega, by omega, hal⟩)
    rw [him]; trivial

theorem TailSafe_run {base : Nat} (hb : base_ok base) (n : Nat) {s : St} (ht : TailSafe base s) :
    run (tailImAt base) n (.ok s) ≠ .exit 0 := by
  induction n generalizing s with
  | zero => simp [run]
  | succ n ih =>
    simp only [run]
    have hs := TailSafe_step hb ht
    cases hst : step (tailImAt base) s with
    | ok s' => rw [hst] at hs; exact ih hs
    | exit c => rw [hst] at hs; simp [run_exit, hs]
    | stuck => simp [run_stuck]

/-! #### Exhaustive 1-bit `fetch` check (`decide +kernel`) + `ElemFaultStep` + `OneFaultRun` -/

def oobOffBool (k off : Nat) : Bool :=
  (192 ≤ 4 * k + off && off < 2 ^ 31) || (4 * k < AArch64Env.W - off && AArch64Env.W - 2 ^ 31 ≤ off)

def branchTargetSafe (k tgt : Nat) : Bool :=
  if k < 8 then tgt < 9 || (11 ≤ tgt && tgt ≤ 29) || (31 ≤ tgt && tgt ≤ 39) || tgt == 43 || tgt == 44
  else if k ≤ 20 then 12 ≤ k && ((21 ≤ tgt && tgt ≤ 39) || tgt == 43 || tgt == 44)
  else if k ≤ 37 then 31 ≤ tgt && tgt ≤ 44
  else if k == 42 then tgt == 42 || tgt == 43 || tgt == 44
  else if k == 47 then tgt == 47 || tgt == 43 || tgt == 44
  else false

def insWriteRd : Ins → Option Nat
  | .movWide _ _ rd _ _ => some rd
  | .addSubImm _ setFlags _ rd _ _ => some (if setFlags && rd == 31 then 31 else rd)
  | .addSubReg _ _ _ rd _ _ _ _ => some rd
  | .logicReg _ _ _ rd _ _ _ _ => some rd
  | .bitfield _ _ rd _ _ _ => some rd
  | .adr _ rd _ => some rd
  | .ccmpReg _ _ _ _ _ _ => some 31
  | .csel _ _ rd _ _ _ => some rd
  | _ => none

def tailAltWriteSafe (k rd : Nat) (ins : Ins) : Bool :=
  if k < 8 then
    if rd != 19 && rd != 20 && (rd != 21 || k == 5 || k == 6) then true
    else if k == 2 && rd == 19 then
      match ins with
      | .movWide kind w _ imm hw =>
          kind == 0 && w == 64 && valNotAccB (AArch64Env.W - 1 - imm * pow2 (hw * 16))
      | .logicReg op inv w _ rn rm sh amt =>
          op == .and && !inv && w == 64 && rn == 19 && rm == 20 && sh == .asr && amt == 0
      | .csel op w _ rn rm _ =>
          ((op == .csel && w == 64) ||
           ((op == .csel || op == .csinc || op == .csinv || op == .csneg) &&
            (w == 64 || (w == 32 && op == .csel)) &&
            (rn == 19 || rn == 20 || rn == 21) && (rm == 19 || rm == 20 || rm == 21))) &&
          (rn == 19 || rn == 20 || rn == 21 || VoterScratchRegs.contains rn) &&
          (rm == 19 || rm == 20 || rm == 21 || VoterScratchRegs.contains rm)
      | _ => false
    else if (k == 5 || k == 6) && rd == 20 then
      match ins with
      | .logicReg op inv w _ rn rm sh amt =>
          (w == 64 || w == 32) && sh != .ror && [0, 1, 2, 4, 8, 16, 32].contains amt &&
          ((op == .orr && !inv && rn == 31 && rm == 19) ||
           (op == .orr && !inv && rn == 31 && VoterScratchRegs.contains rm && amt == 0 && w == 64) ||
           (op == .orr && !inv && rn != 31 && rm == 19 && amt == 0) ||
           (op == .orr && inv && rn == 31 && rm == 19 && amt == 0 && w == 64) ||
           ((op == .and || op == .ands) && !inv && rn == 31 && rm == 19 && amt == 0))
      | .addSubReg isSub _ w _ rn rm sh amt =>
          !isSub && (w == 64 || w == 32) && rn == 31 && rm == 19 && sh == .lsl && amt == 0
      | _ => false
    else false
  else if k ≤ 20 then rd != 20
  else if k ≤ 38 || k == 43 then true
  else if k == 39 || k == 44 then
    rd != 0 || (ins == .movWide 2 64 0 6 0) || (ins == .movWide 2 64 0 5 0)
  else if k == 40 || k == 45 then
    rd != 0 ||
    match ins with
    | .addSubReg isSub _ w _ rn rm sh amt =>
        !isSub && (w == 64 || w == 32) && rn == 0 && rm == 2 && sh == .lsl && amt == 0
    | .logicReg op inv w _ rn rm sh amt =>
        (w == 64 || w == 32) && sh != .ror && amt < w &&
        ((op == .orr && !inv && (rn == 0 || (rn != 31 && rm == 2 && sh == .lsl && amt == 0))) ||
         (op == .orr && inv && rn == 0 && rm == 2 && sh == .lsl && amt == 0) ||
         ((op == .and || op == .ands) && !inv && rn == 0 && rm == 2 && sh == .lsl && amt == 0))
    | _ => false
  else if k == 41 || k == 46 then
    rd != 0 || (ins == .movWide 2 64 0 93 0)
  else if k == 42 || k == 47 then
    rd != 0
  else false

def tailAltWordSafe (k : Nat) (ins : Ins) : Bool :=
  match ins with
  | .b _ off =>
      oobOffBool k off || ((4 * k + off) % 4 == 0 && 4 * k + off < 192 && branchTargetSafe k ((4 * k + off) / 4))
  | .bcond cc off =>
      (k == 4 && cc == 1) ||
      oobOffBool k off ||
      ((4 * k + off) % 4 == 0 && 4 * k + off < 192 && branchTargetSafe k ((4 * k + off) / 4))
  | .cbz _ _ _ off | .tbz _ _ _ off =>
      oobOffBool k off ||
      ((4 * k + off) % 4 == 0 && 4 * k + off < 192 && branchTargetSafe k ((4 * k + off) / 4))
  | .svc imm => imm != 0 || k == 42 || k == 47
  | .ldur _ rd _ _ => k < 8 && rd != 19 && rd != 20 && rd != 21
  | _ =>
    match insWriteRd ins with
    | none => false
    | some rd => tailAltWriteSafe k rd ins

private theorem tailAltWordSafe_of_other {k rd : Nat} {ins : Ins}
    (hnb : ∀ l o, ins ≠ .b l o) (hnbc : ∀ c o, ins ≠ .bcond c o)
    (hncbz : ∀ n w r o, ins ≠ .cbz n w r o) (hntbz : ∀ n b r o, ins ≠ .tbz n b r o)
    (hnsvc : ∀ i, ins ≠ .svc i) (hnldur : ∀ sz r rn o, ins ≠ .ldur sz r rn o)
    (hwr : insWriteRd ins = some rd) :
    tailAltWordSafe k ins = tailAltWriteSafe k rd ins := by
  cases ins with
  | b l o => exact absurd rfl (hnb l o)
  | bcond c o => exact absurd rfl (hnbc c o)
  | cbz n w r o => exact absurd rfl (hncbz n w r o)
  | tbz n b r o => exact absurd rfl (hntbz n b r o)
  | svc i => exact absurd rfl (hnsvc i)
  | ldur sz r rn o => exact absurd rfl (hnldur sz r rn o)
  | _ => simp [tailAltWordSafe, hwr]

def tailFetchSafeBool : Bool :=
  preFaultIndices.all fun k =>
    let w := (tailWords[k]?).getD 0
    (List.range 32).all fun b =>
      match decodeWord (Nat.xor w (2 ^ b)) with
      | none => true
      | some ins => tailAltWordSafe k ins

theorem tailFetchSafe_audit : tailFetchSafeBool = true := by decide +kernel

private theorem branch_target_oob_none {base k off : Nat} (hb : base_ok base) (hk : k < 48)
    (ho : oobOffBool k off = true) :
    tailImAt base ((base + 4 * k + off) % AArch64Env.W) = none := by
  apply tailImAt_out
  unfold oobOffBool at ho
  rcases hb with ⟨_, hlo, hhi⟩
  simp only [Bool.or_eq_true, Bool.and_eq_true, decide_eq_true_eq] at ho
  rcases ho with ⟨hle, hlt⟩ | ⟨hlt, hge⟩
  · have hltW : base + 4 * k + off < AArch64Env.W := by omega
    rw [Nat.mod_eq_of_lt hltW]
    exact Or.inr (Or.inl (by omega))
  · let d := AArch64Env.W - off
    have hd1 : 4 * k < d := by omega
    have hd2 : d ≤ 2 ^ 31 := by omega
    have hoff : off = AArch64Env.W - d := by omega
    rw [hoff]
    by_cases hc : base + 4 * k < d
    · have hltW : base + 4 * k + (AArch64Env.W - d) < AArch64Env.W := by omega
      rw [Nat.mod_eq_of_lt hltW]
      exact Or.inr (Or.inl (by unfold AArch64Env.W at *; omega))
    · have hgeW : AArch64Env.W ≤ base + 4 * k + (AArch64Env.W - d) := by omega
      have hlt2W : base + 4 * k + (AArch64Env.W - d) - AArch64Env.W < AArch64Env.W := by omega
      have hmod : (base + 4 * k + (AArch64Env.W - d)) % AArch64Env.W = base + 4 * k - d := by
        rw [show base + 4 * k + (AArch64Env.W - d) = (base + 4 * k - d) + AArch64Env.W by omega,
          Nat.add_mod_right, Nat.mod_eq_of_lt (by omega)]
      rw [hmod]
      exact Or.inl (by omega)

private theorem setReg_len_eq (s : St) (r : Nat) (v : Option Nat) :
    (s.setReg r v).regs.length = s.regs.length := by
  simp [St.setReg]

private theorem wrZR_len_eq (s : St) (w r v : Nat) :
    (s.wrZR w r v).regs.length = s.regs.length := by
  unfold St.wrZR; split
  · rfl
  · exact setReg_len_eq s r _

private theorem wrSP_len_eq (s : St) (w r v : Nat) :
    (s.wrSP w r v).regs.length = s.regs.length := by
  unfold St.wrSP; split
  · rfl
  · exact setReg_len_eq s r _

private theorem wrSP_ne (s : St) (w : Nat) {r1 r2 : Nat} (hne : r1 ≠ r2) (v : Nat) :
    (s.wrSP w r1 v).reg r2 = s.reg r2 := by
  unfold St.wrSP; split
  · rfl
  · exact reg_set_ne hne _

private theorem lift_ok_eq {o : Option St} {s' : St} (h : lift o = .ok s') : o = some s' := by
  cases o with
  | none => cases h
  | some t => cases h; rfl

private theorem exec_preserves_regs_ne {ins : Ins} {len : Nat} {s s' : St} {rd r : Nat}
    (hwr : insWriteRd ins = some rd) (hx : exec ins len s = .ok s') (hne : r ≠ rd) :
    s'.reg r = s.reg r ∧ s'.regs.length = s.regs.length ∧ s'.pc = (s.pc + len) % AArch64Env.W := by
  cases ins with
  | movWide kind w rd' imm hw =>
    simp [insWriteRd] at hwr; subst hwr; simp only [exec] at hx
    rcases (show kind = 0 ∨ kind = 2 ∨ kind = 3 by
      cases kind with
      | zero => exact Or.inl rfl
      | succ n => cases n with
        | zero => simp at hx
        | succ n => cases n with
          | zero => exact Or.inr (Or.inl rfl)
          | succ n => cases n with | zero => exact Or.inr (Or.inr rfl) | succ => simp at hx) with rfl | rfl | rfl
    · cases hx; exact ⟨wrZR_ne (s := _) w hne.symm _, wrZR_len_eq _ _ _ _, wrZR_pc_eq_val _ _ _ _⟩
    · cases hx; exact ⟨wrZR_ne (s := _) w hne.symm _, wrZR_len_eq _ _ _ _, wrZR_pc_eq_val _ _ _ _⟩
    · have ho := lift_ok_eq hx
      simp only [bind, Option.bind, pure] at ho
      cases h1 : s.rdZR w rd' with
      | none => rw [h1] at ho; cases ho
      | some old =>
        rw [h1] at ho; cases ho
        exact ⟨wrZR_ne (s := _) w hne.symm _, wrZR_len_eq _ _ _ _, wrZR_pc_eq_val _ _ _ _⟩
  | addSubImm isSub setFlags w rd' rn imm =>
    simp [insWriteRd] at hwr
    have hx' := lift_ok_eq hx
    simp only [bind, Option.bind, pure] at hx'
    cases h1 : s.rdSP w rn with
    | none => rw [h1] at hx'; cases hx'
    | some a =>
      rw [h1] at hx'
      by_cases hsf : setFlags = true
      · subst hsf; simp only [↓reduceIte] at hx'; cases hx'
        have hrne : rd' ≠ r := by
          intro heq
          by_cases h31 : rd' = 31
          · simp [h31] at hwr; rw [← hwr] at hne; exact hne (by subst h31; exact heq.symm)
          · simp [h31] at hwr; rw [← hwr] at hne; exact hne heq.symm
        exact ⟨wrZR_ne (s := _) w hrne _, wrZR_len_eq _ _ _ _, wrZR_pc_eq_val _ _ _ _⟩
      · have hsf' : setFlags = false := by cases setFlags <;> simp_all
        subst hsf'; simp only [Bool.false_eq_true, ↓reduceIte] at hwr hx'
        simp at hwr; cases hx'; rw [hwr]
        exact ⟨wrSP_ne _ w hne.symm _, wrSP_len_eq _ _ _ _, wrSP_pc_eq_val _ _ _ _⟩
  | addSubReg isSub setFlags w rd' rn rm sh amt =>
    simp [insWriteRd] at hwr; subst hwr
    have hx' := lift_ok_eq hx
    simp only [bind, Option.bind, pure] at hx'
    cases h1 : s.rdZR w rn <;> cases h2 : s.rdZR w rm <;> simp [h1, h2] at hx'
    by_cases hsf : setFlags = true <;> simp [hsf] at hx'
    · cases hx'; exact ⟨wrZR_ne (s := _) w hne.symm _, wrZR_len_eq _ _ _ _, wrZR_pc_eq_val _ _ _ _⟩
    · cases hx'; exact ⟨wrZR_ne (s := _) w hne.symm _, wrZR_len_eq _ _ _ _, wrZR_pc_eq_val _ _ _ _⟩
  | logicReg op inv w rd' rn rm sh amt =>
    simp [insWriteRd] at hwr; subst hwr
    have hx' := lift_ok_eq hx
    simp only [bind, Option.bind, pure] at hx'
    cases h1 : s.rdZR w rn <;> cases h2 : s.rdZR w rm <;> simp [h1, h2] at hx'
    cases op <;> simp only at hx' <;> cases hx' <;> exact ⟨wrZR_ne (s := _) w hne.symm _, wrZR_len_eq _ _ _ _, wrZR_pc_eq_val _ _ _ _⟩
  | bitfield kind w rd' rn immr imms =>
    simp [insWriteRd] at hwr; subst hwr
    have hx' := lift_ok_eq hx
    simp only [bind, Option.bind, pure] at hx'
    cases h1 : s.rdZR w rn <;> simp [h1] at hx'
    cases kind <;> simp at hx'
    · cases hx'; exact ⟨wrZR_ne (s := _) w hne.symm _, wrZR_len_eq _ _ _ _, wrZR_pc_eq_val _ _ _ _⟩
    · cases h2 : s.rdZR w rd' <;> simp [h2] at hx'; cases hx'; exact ⟨wrZR_ne (s := _) w hne.symm _, wrZR_len_eq _ _ _ _, wrZR_pc_eq_val _ _ _ _⟩
    · cases hx'; exact ⟨wrZR_ne (s := _) w hne.symm _, wrZR_len_eq _ _ _ _, wrZR_pc_eq_val _ _ _ _⟩
  | adr isPage rd' imm =>
    simp [insWriteRd] at hwr; subst hwr; cases hx
    exact ⟨wrZR_ne (s := _) 64 hne.symm _, wrZR_len_eq _ _ _ _, wrZR_pc_eq_val _ _ _ _⟩
  | ccmpReg isSub w rn rm nzcv cc =>
    simp [insWriteRd] at hwr; subst hwr
    have hx' := lift_ok_eq hx
    simp only [bind, Option.bind, pure] at hx'
    cases h1 : s.rdZR w rn <;> cases h2 : s.rdZR w rm <;> cases hc : AArch64Phys.cond s.fl cc <;> simp [h1, h2, hc] at hx'
    cases hx'; exact ⟨rfl, rfl, rfl⟩
  | csel op w rd' rn rm cc =>
    simp [insWriteRd] at hwr; subst hwr
    have hx' := lift_ok_eq hx
    simp only [bind, Option.bind, pure] at hx'
    cases h1 : s.rdZR w rn <;> cases h2 : s.rdZR w rm <;> cases hc : AArch64Phys.cond s.fl cc <;> simp [h1, h2, hc] at hx'
    cases hx'; exact ⟨wrZR_ne (s := _) w hne.symm _, wrZR_len_eq _ _ _ _, wrZR_pc_eq_val _ _ _ _⟩
  | ldur _ _ _ _ | ldr _ _ _ _ | str _ _ _ _ | stur _ _ _ _ | ldrReg _ _ _ _ _ _ | strReg _ _ _ _ _ _
  | b _ _ | bcond _ _ | cbz _ _ _ _ | tbz _ _ _ _ | br _ _ | ret _ | svc _ =>
    simp [insWriteRd] at hwr

private theorem exec_ldur_preserves_regs_ne {sz : MemSize} {rt rn off len : Nat} {s s' : St} {r : Nat}
    (hx : exec (.ldur sz rt rn off) len s = .ok s') (hne : r ≠ rt) :
    s'.reg r = s.reg r ∧ s'.regs.length = s.regs.length ∧ s'.pc = (s.pc + len) % AArch64Env.W := by
  have hx' := lift_ok_eq hx
  simp only [bind, Option.bind, pure] at hx'
  cases h1 : s.rdSP 64 rn with
  | none => rw [h1] at hx'; cases hx'
  | some base =>
    rw [h1] at hx'
    by_cases hsp : (rn == 31 && base % 16 != 0) = true
    · simp [hsp] at hx'
    · simp only [hsp, Bool.false_eq_true, ↓reduceIte] at hx'
      cases h2 : AArch64Phys.rdMem s.mem ((base + off) % AArch64Env.W) sz with
      | none => rw [h2] at hx'; cases hx'
      | some v => rw [h2] at hx'; cases hx'; exact ⟨wrZR_ne (s := _) sz.width hne.symm _, wrZR_len_eq _ _ _ _, wrZR_pc_eq_val _ _ _ _⟩

private theorem exec_exit_only_svc {ins : Ins} {len : Nat} {s : St} {c : Nat}
    (hx : exec ins len s = .exit c) : ∃ imm, ins = .svc imm ∧ imm = 0 := by
  cases ins with
  | svc imm =>
    simp only [exec] at hx
    by_cases h : imm = 0
    · exact ⟨imm, rfl, h⟩
    · simp [h] at hx
  | movWide kind w rd _ _ =>
    simp only [exec, lift, bind, Option.bind, pure] at hx
    rcases kind with _ | _ | _ | _ | _ <;> simp at hx
    cases hrz : s.rdZR w rd <;> simp [hrz] at hx
  | _ =>
    simp only [exec, lift, bind, Option.bind, pure] at hx
    first | cases hx | (split at hx <;> simp at hx)

private theorem oobOff_not_bounds {base k off : Nat} (hb : base_ok base) (hk : k < 48)
    (ho : oobOffBool k off = true) :
    ¬ (base ≤ (base + 4 * k + off) % AArch64Env.W ∧
        (base + 4 * k + off) % AArch64Env.W < base + 192 ∧
        ((base + 4 * k + off) % AArch64Env.W - base) % 4 = 0) := by
  intro ⟨hlo, hhi, _⟩
  unfold oobOffBool at ho
  rcases hb with ⟨_, hblo, hbhi⟩
  simp only [Bool.or_eq_true, Bool.and_eq_true, decide_eq_true_eq] at ho
  rcases ho with ⟨hle, hlt⟩ | ⟨hlt, hge⟩
  · have hltW : base + 4 * k + off < AArch64Env.W := by omega
    rw [Nat.mod_eq_of_lt hltW] at hhi; omega
  · obtain ⟨d, hd_def⟩ : ∃ d, AArch64Env.W - off = d := ⟨_, rfl⟩
    have hd1 : 4 * k < d := by rw [← hd_def]; omega
    have hd2 : d ≤ 2 ^ 31 := by rw [← hd_def]; omega
    have hoff : off = AArch64Env.W - d := by rw [← hd_def]; omega
    rw [hoff] at hlo hhi
    by_cases hc : base + 4 * k < d
    · have hltW : base + 4 * k + (AArch64Env.W - d) < AArch64Env.W := by omega
      rw [Nat.mod_eq_of_lt hltW] at hhi
      unfold AArch64Env.W at *; omega
    · have hmod : (base + 4 * k + (AArch64Env.W - d)) % AArch64Env.W = base + 4 * k - d := by
        rw [show base + 4 * k + (AArch64Env.W - d) = (base + 4 * k - d) + AArch64Env.W by omega,
          Nat.add_mod_right, Nat.mod_eq_of_lt (by omega)]
      rw [hmod] at hlo; omega

private theorem tailSafeAt_congr {s1 s2 : St}
    (h0 : s1.reg 0 = s2.reg 0) (h1 : s1.reg 1 = s2.reg 1) (h2 : s1.reg 2 = s2.reg 2)
    (h8 : s1.reg 8 = s2.reg 8) (h19 : s1.reg 19 = s2.reg 19) (h20 : s1.reg 20 = s2.reg 20)
    (h21 : s1.reg 21 = s2.reg 21) (hfl : s1.fl = s2.fl) : ∀ k,
    tailSafeAt s1 k ↔ tailSafeAt s2 k := by
  intro k
  rcases k with _ | _ | _ | _ | _ | _ | _ | _ | _ | _ | _ | _ | _ | _ | _ | _ |
    _ | _ | _ | _ | _ | _ | _ | _ | _ | _ | _ | _ | _ | _ | _ | _ | _ | _ | _ | _ |
    _ | _ | _ | _ | _ | _ | _ | _ | _ | _ | _ | _ | k <;>
    simp [tailSafeAt, allEqValNotAcc, allThreeValNotAcc, twoAgreeOrSplit, h0, h1, h2, h8, h19, h20, h21, hfl]

private theorem tailSafeAt_setReg30 (s : St) (v : Option Nat) : ∀ k,
    tailSafeAt (s.setReg 30 v) k ↔ tailSafeAt s k := by
  have hr : ∀ {r : Nat}, r < 30 → (s.setReg 30 v).reg r = s.reg r := by
    intro r hr; exact reg_set_ne (by omega) _
  exact tailSafeAt_congr (hr (by decide)) (hr (by decide)) (hr (by decide)) (hr (by decide))
    (hr (by decide)) (hr (by decide)) (hr (by decide)) rfl

private theorem branchTargetSafe_sound_k {v k tgt : Nat} {s : St}
    (hv : SafeInvalid v) (hcase :
      (k < 8 ∧ s.reg 19 = some v ∧ s.reg 20 = some v ∧ s.reg 21 = some v ∧
          (1 ≤ k ∧ k ≤ 4 → s.fl.z = some true)) ∨
       (11 ≤ k ∧ k ≤ 20 ∧ s.reg 19 = some v ∧ s.reg 20 = some v ∧
         (12 ≤ k → s.reg 0 = some 5) ∧
         (13 ≤ k → s.reg 8 = some 0) ∧
         (k = 16 → s.reg 1 = some 0x3c96) ∧
         (k = 17 → s.reg 1 = some 0xa55a3c96) ∧
         (k = 18 → s.reg 1 = some 0xe970a55a3c96) ∧
         (k = 19 → s.reg 1 = some ACCEPT) ∧
         (k = 20 → s.fl.z = some false)) ∨
       (32 ≤ k ∧ k < 48 ∧ s.reg 19 = some v ∧
         (k = 33 → s.reg 1 = some 0xc3a5) ∧
         (k = 34 → s.reg 1 = some 0x33c3c3a5) ∧
         (k = 35 → s.reg 1 = some 0x0f1e33c3c3a5) ∧
         (k = 36 → s.reg 1 = some OVERFLOW) ∧
         (k = 37 → ∃ z, s.fl.z = some z) ∧
         (k ≤ 38 → s.reg 0 = some 5) ∧
         (39 ≤ k ∧ k ≤ 42 → s.reg 0 = some 6) ∧
         (k = 40 → s.reg 2 = some 6) ∧
         (44 ≤ k ∧ k ≤ 47 → s.reg 0 = some 5) ∧
         (k = 45 → s.reg 2 = some 5) ∧
         (k ≤ 41 ∨ k = 43 ∨ k = 44 ∨ k = 45 → s.reg 8 = some 0) ∧
         (k = 42 → s.reg 8 = some 93) ∧
         (k = 47 → s.reg 8 = some 93)))
    (hsafe : branchTargetSafe k tgt = true) :
    tailSafeAt s tgt := by
  have hnv := hv.valNotAcc
  unfold branchTargetSafe at hsafe
  rcases hcase with ⟨hk8, h19, h20, h21, _⟩ | ⟨hk11, hk20, h19, h20, h0_5, _, _, _, _, _, _⟩ |
    ⟨hk32, hk48, _, _, _, _, _, _, h32_38, h39_42, _, h44_47, _, _, _, _⟩
  · simp only [show k < 8 from hk8, ↓reduceIte, Bool.or_eq_true, Bool.and_eq_true,
      decide_eq_true_eq, beq_iff_eq] at hsafe
    rcases (show tgt = 0 ∨ tgt = 1 ∨ tgt = 2 ∨ tgt = 3 ∨ tgt = 4 ∨ tgt = 5 ∨ tgt = 6 ∨ tgt = 7 ∨
        tgt = 8 ∨ (11 ≤ tgt ∧ tgt ≤ 29) ∨ (31 ≤ tgt ∧ tgt ≤ 39) ∨ tgt = 43 ∨ tgt = 44 by omega) with
      rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | ⟨ht11, ht29⟩ | ⟨ht31, ht39⟩ | rfl | rfl
    · exact ⟨v, v, v, h19, h20, h21, hv.1, hv.1, hv.1, Or.inr ⟨v, by unfold TMR.vote; simp, hnv⟩⟩
    · exact Or.inl ⟨v, h19, h20, h21, hnv⟩
    · exact Or.inl ⟨v, h19, h20, h21, hnv⟩
    · exact Or.inl ⟨v, h19, h20, h21, hnv⟩
    · exact Or.inl ⟨v, h19, h20, h21, hnv⟩
    · exact ⟨v, h19, hnv⟩
    · exact ⟨v, v, h19, h20, hnv, hnv⟩
    · exact ⟨v, v, h19, h20, hnv, hnv⟩
    · trivial
    · have : tgt = 11 ∨ tgt = 12 ∨ tgt = 13 ∨ tgt = 14 ∨ tgt = 15 ∨ tgt = 16 ∨ tgt = 17 ∨ tgt = 18 ∨
             tgt = 19 ∨ tgt = 20 ∨ tgt = 21 ∨ tgt = 22 ∨ tgt = 23 ∨ tgt = 24 ∨ tgt = 25 ∨ tgt = 26 ∨
             tgt = 27 ∨ tgt = 28 ∨ tgt = 29 := by omega
      rcases this with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl |
          rfl | rfl | rfl | rfl | rfl | rfl | rfl <;>
        first | exact Or.inl ⟨v, h20, hnv⟩ | exact Or.inr ⟨v, h20, hnv⟩ |
          exact ⟨v, h20, hnv⟩ | exact ⟨v, h20, hnv.1, hnv.2.2⟩
    · have : tgt = 31 ∨ tgt = 32 ∨ tgt = 33 ∨ tgt = 34 ∨ tgt = 35 ∨ tgt = 36 ∨ tgt = 37 ∨ tgt = 38 ∨ tgt = 39 := by omega
      rcases this with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl <;> trivial
    · trivial
    · trivial
  · rw [if_neg (by omega), if_pos (by omega)] at hsafe
    simp only [Bool.and_eq_true, Bool.or_eq_true, decide_eq_true_eq, beq_iff_eq] at hsafe
    have : 12 ≤ k := hsafe.1
    rcases (show (21 ≤ tgt ∧ tgt ≤ 39) ∨ tgt = 43 ∨ tgt = 44 by omega) with ⟨ht21, ht39⟩ | rfl | rfl
    · have : tgt = 21 ∨ tgt = 22 ∨ tgt = 23 ∨ tgt = 24 ∨ tgt = 25 ∨ tgt = 26 ∨ tgt = 27 ∨ tgt = 28 ∨ tgt = 29 ∨
             tgt = 30 ∨ tgt = 31 ∨ tgt = 32 ∨ tgt = 33 ∨ tgt = 34 ∨ tgt = 35 ∨ tgt = 36 ∨ tgt = 37 ∨ tgt = 38 ∨ tgt = 39 := by omega
      rcases this with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl |
          rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl <;>
        first | exact ⟨v, h20, hnv⟩ | exact Or.inr ⟨v, h20, hnv⟩ | exact ⟨v, h20, hnv.1, hnv.2.2⟩ |
          trivial | (intro _; exact ⟨5, h0_5 ‹_›, by decide⟩)
    · trivial
    · trivial
  · rw [if_neg (by omega), if_neg (by omega)] at hsafe
    by_cases hk37 : k ≤ 37
    · rw [if_pos hk37] at hsafe
      simp only [Bool.and_eq_true, decide_eq_true_eq] at hsafe
      have : tgt = 31 ∨ tgt = 32 ∨ tgt = 33 ∨ tgt = 34 ∨ tgt = 35 ∨ tgt = 36 ∨ tgt = 37 ∨
             tgt = 38 ∨ tgt = 39 ∨ tgt = 40 ∨ tgt = 41 ∨ tgt = 42 ∨ tgt = 43 ∨ tgt = 44 := by omega
      rcases this with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl <;>
        first | trivial | exact Or.inl ⟨5, h32_38 (by omega), by decide⟩ | exact ⟨5, h32_38 (by omega), by decide⟩ |
          (intro _; exact ⟨5, h32_38 (by omega), by decide⟩)
    · rw [if_neg hk37] at hsafe
      by_cases hk42 : k = 42
      · subst hk42; simp only [beq_self_eq_true, ↓reduceIte, Bool.or_eq_true, beq_iff_eq] at hsafe
        rcases hsafe with (rfl | rfl) | rfl <;>
          first | trivial | (intro _; exact ⟨6, h39_42 ⟨by decide, by decide⟩, by decide⟩)
      · rw [if_neg (by simpa using hk42)] at hsafe
        by_cases hk47 : k = 47
        · subst hk47; simp only [beq_self_eq_true, ↓reduceIte, Bool.or_eq_true, beq_iff_eq] at hsafe
          rcases hsafe with (rfl | rfl) | rfl <;>
            first | trivial | (intro _; exact ⟨5, h44_47 ⟨by decide, by decide⟩, by decide⟩)
        · rw [if_neg (by simpa using hk47)] at hsafe; cases hsafe

private theorem tailSafe_fall_k_succ {base v k : Nat} {s s' : St}
    (hb : base_ok base) (hk48 : k < 48) (hp : PreFault base v s)
    (hs'_len : s'.regs.length = 31) (hs'_pc : s'.pc = (s.pc + 4) % AArch64Env.W)
    (hs'_reg19 : s'.reg 19 = s.reg 19) (hs'_reg20 : s'.reg 20 = s.reg 20)
    (hs'_reg21 : s'.reg 21 = s.reg 21) (hs'_reg0 : 39 ≤ k → k ≤ 47 → s'.reg 0 = s.reg 0)
    (hkpc : s.pc = base + 4 * k) : TailSafe base s' := by
  have hk_add : (base + 4 * k + 4) % AArch64Env.W = base + 4 * (k + 1) := by
    rcases hb with ⟨_, _, hhi⟩
    rw [Nat.mod_eq_of_lt (by omega)]; omega
  have hpc' : s'.pc = base + 4 * (k + 1) := by rw [hs'_pc, hkpc, hk_add]
  rcases hp with ⟨hv, _, _, _, k', hpc_k', hcase⟩
  have hkeq : k' = k := by rw [hkpc] at hpc_k'; omega
  rw [hkeq] at hpc_k' hcase
  have hnv := hv.valNotAcc
  refine ⟨hs'_len, ?_⟩
  split
  · next hbd =>
    rw [show (s'.pc - base) / 4 = k + 1 by rw [hpc']; omega]
    have hklt47 : k < 47 := by omega
    rcases hcase with ⟨hk8, h19, h20, h21, _⟩ | ⟨hk11, hk20, _, h20, _, _, _, _, _, _, _⟩ |
      ⟨hk32, _, _, _, _, _, _, _, _, h39_42, _, h44_47, _, _, _, _⟩
    · have : k = 0 ∨ k = 1 ∨ k = 2 ∨ k = 3 ∨ k = 4 ∨ k = 5 ∨ k = 6 ∨ k = 7 := by omega
      rcases this with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl
      · exact Or.inl ⟨v, hs'_reg19.trans h19, hs'_reg20.trans h20, hs'_reg21.trans h21, hnv⟩
      · exact Or.inl ⟨v, hs'_reg19.trans h19, hs'_reg20.trans h20, hs'_reg21.trans h21, hnv⟩
      · exact Or.inl ⟨v, hs'_reg19.trans h19, hs'_reg20.trans h20, hs'_reg21.trans h21, hnv⟩
      · exact Or.inl ⟨v, hs'_reg19.trans h19, hs'_reg20.trans h20, hs'_reg21.trans h21, hnv⟩
      · exact ⟨v, hs'_reg19.trans h19, hnv⟩
      · exact ⟨v, v, hs'_reg19.trans h19, hs'_reg20.trans h20, hnv, hnv⟩
      · exact ⟨v, v, hs'_reg19.trans h19, hs'_reg20.trans h20, hnv, hnv⟩
      · trivial
    · have : k = 11 ∨ k = 12 ∨ k = 13 ∨ k = 14 ∨ k = 15 ∨ k = 16 ∨ k = 17 ∨ k = 18 ∨ k = 19 ∨ k = 20 := by omega
      rcases this with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl <;>
        first | exact Or.inl ⟨v, hs'_reg20.trans h20, hnv⟩ |
          exact Or.inr ⟨v, hs'_reg20.trans h20, hnv⟩ | exact ⟨v, hs'_reg20.trans h20, hnv⟩
    · have : k = 32 ∨ k = 33 ∨ k = 34 ∨ k = 35 ∨ k = 36 ∨ k = 37 ∨ k = 38 ∨ k = 39 ∨
             k = 40 ∨ k = 41 ∨ k = 42 ∨ k = 43 ∨ k = 44 ∨ k = 45 ∨ k = 46 := by omega
      rcases this with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl <;>
        first | trivial |
          exact Or.inl ⟨6, (hs'_reg0 (by decide) (by decide)).trans (h39_42 ⟨by decide, by decide⟩), by decide⟩ |
          exact ⟨6, (hs'_reg0 (by decide) (by decide)).trans (h39_42 ⟨by decide, by decide⟩), by decide⟩ |
          exact (fun _ => ⟨6, (hs'_reg0 (by decide) (by decide)).trans (h39_42 ⟨by decide, by decide⟩), by decide⟩) |
          exact Or.inl ⟨5, (hs'_reg0 (by decide) (by decide)).trans (h44_47 ⟨by decide, by decide⟩), by decide⟩ |
          exact ⟨5, (hs'_reg0 (by decide) (by decide)).trans (h44_47 ⟨by decide, by decide⟩), by decide⟩ |
          exact (fun _ => ⟨5, (hs'_reg0 (by decide) (by decide)).trans (h44_47 ⟨by decide, by decide⟩), by decide⟩)
  · trivial

private theorem tailSafe_setReg30_upd {base : Nat} {s : St}
    (hts : TailSafe base s) (v : Option Nat) :
    TailSafe base (s.setReg 30 v) := by
  rcases hts with ⟨hlen, hts⟩
  refine ⟨setReg_len hlen 30 v, ?_⟩
  split
  · next hbd =>
    rw [dif_pos (by simpa [St.setReg] using hbd)] at hts
    exact (tailSafeAt_setReg30 s v _).2 hts
  · trivial

private theorem tailAltWordSafe_none {k : Nat} {ins : Ins}
    (hnb : ∀ l o, ins ≠ .b l o) (hnbc : ∀ c o, ins ≠ .bcond c o)
    (hncbz : ∀ n w r o, ins ≠ .cbz n w r o) (hntbz : ∀ n b r o, ins ≠ .tbz n b r o)
    (hnsvc : ∀ i, ins ≠ .svc i) (hnldur : ∀ sz r rn o, ins ≠ .ldur sz r rn o)
    (hwr : insWriteRd ins = none) :
    tailAltWordSafe k ins = false := by
  cases ins with
  | b l o => exact absurd rfl (hnb l o)
  | bcond c o => exact absurd rfl (hnbc c o)
  | cbz n w r o => exact absurd rfl (hncbz n w r o)
  | tbz n b r o => exact absurd rfl (hntbz n b r o)
  | svc i => exact absurd rfl (hnsvc i)
  | ldur sz r rn o => exact absurd rfl (hnldur sz r rn o)
  | _ => simp [tailAltWordSafe, hwr]

private theorem land_self_eq (x : Nat) : Nat.land x x = x := by
  apply Nat.eq_of_testBit_eq
  intro i
  change (x &&& x).testBit i = x.testBit i
  rw [Nat.testBit_and, Bool.and_self]

private def evalCSel (op : CSelOp) (condv : Bool) (w a b : Nat) : Nat :=
  if condv then a
  else match op with
  | .csel => b
  | .csinc => (b + 1) % pow2 w
  | .csinv => pow2 w - 1 - b
  | .csneg => (pow2 w - b) % pow2 w

private def evalLogic (op : LogicOp) (inv : Bool) (sh : ShiftKind) (w amt a b : Nat) : Nat :=
  let bSh := shiftVal sh w b amt
  let bOp := if inv then pow2 w - 1 - bSh else bSh
  match op with
  | .and | .ands => Nat.land a bOp
  | .orr => Nat.lor a bOp
  | .eor => Nat.xor a bOp

private theorem shiftVal_zero {sh : ShiftKind} {w v : Nat} (hsh : sh ≠ .ror) (hv : v < pow2 w) :
    shiftVal sh w v 0 = v := by
  unfold shiftVal; dsimp only
  rw [Nat.mod_eq_of_lt hv]
  cases sh with
  | lsl => rw [show pow2 0 = 1 from rfl, Nat.mul_one, Nat.mod_eq_of_lt hv]
  | lsr => simp [pow2]
  | asr => simp [AArch64Phys.asrVal]; exact Nat.mod_eq_of_lt hv
  | ror => exact absurd rfl hsh

private theorem tailAltWordSafe_sound_other {base v k : Nat} (hb : base_ok base)
    (hk48 : k < 48) {s : St} (hp : PreFault base v s)
    (hpc : s.pc = base + 4 * k) {ins_other : Ins} (hsafe : tailAltWordSafe k ins_other = true)
    (hnb : ∀ l o, ins_other ≠ .b l o) (hnbc : ∀ c o, ins_other ≠ .bcond c o)
    (hncbz : ∀ n w r o, ins_other ≠ .cbz n w r o) (hntbz : ∀ n b r o, ins_other ≠ .tbz n b r o)
    (hnsvc : ∀ i, ins_other ≠ .svc i) (hnldur : ∀ sz r rn o, ins_other ≠ .ldur sz r rn o) :
    match exec ins_other 4 s with
    | .ok s' => TailSafe base s'
    | .exit c => c ≠ 0
    | .stuck => True := by
  have hpre := hp
  rcases hp with ⟨hv, hodd, hlen, hscr, k', hpc', hcase⟩
  have hkeq : k' = k := by rw [hpc] at hpc'; omega
  rw [hkeq] at hpc' hcase
  have hnv := hv.valNotAcc
  have h19lt : (19 : Nat) < 31 := by decide
  have h20lt : (20 : Nat) < 31 := by decide
  have h0lt : (0 : Nat) < 31 := by decide
  have hk_add : (base + 4 * k + 4) % AArch64Env.W = base + 4 * (k + 1) := by
    rcases hb with ⟨_, _, hhi⟩; rw [Nat.mod_eq_of_lt (by omega)]; omega
  cases hwr : insWriteRd ins_other with
  | none =>
    rw [tailAltWordSafe_none hnb hnbc hncbz hntbz hnsvc hnldur hwr] at hsafe
    cases hsafe
  | some rd =>
    rw [tailAltWordSafe_of_other hnb hnbc hncbz hntbz hnsvc hnldur hwr] at hsafe
    cases hx : exec ins_other 4 s with
    | stuck => trivial
    | exit c =>
      exfalso; obtain ⟨imm, heq, _⟩ := exec_exit_only_svc hx
      rw [heq] at hwr; cases hwr
    | ok s' =>
      have hpr19 : ∀ hne : 19 ≠ rd, s'.reg 19 = s.reg 19 ∧ s'.regs.length = s.regs.length ∧ s'.pc = (s.pc + 4) % AArch64Env.W :=
        fun h => exec_preserves_regs_ne hwr hx h
      have hpr20 : ∀ hne : 20 ≠ rd, s'.reg 20 = s.reg 20 := fun h => (exec_preserves_regs_ne hwr hx h).1
      have hpr21 : ∀ hne : 21 ≠ rd, s'.reg 21 = s.reg 21 := fun h => (exec_preserves_regs_ne hwr hx h).1
      have hpr0 : ∀ hne : 0 ≠ rd, s'.reg 0 = s.reg 0 := fun h => (exec_preserves_regs_ne hwr hx h).1
      have hlen_pc : s'.regs.length = s.regs.length ∧ s'.pc = (s.pc + 4) % AArch64Env.W := by
        by_cases h0r : (0 : Nat) = rd
        · exact (exec_preserves_regs_ne hwr hx (r := 1) (by omega)).2
        · exact (exec_preserves_regs_ne hwr hx (r := 0) h0r).2
      have hlen' : s'.regs.length = 31 := hlen_pc.1.trans hlen
      have hpc'_next : s'.pc = base + 4 * (k + 1) := by rw [hlen_pc.2, hpc, hk_add]
      simp only [tailAltWriteSafe] at hsafe
      by_cases hk8 : k < 8
      · rw [if_pos hk8] at hsafe
        rcases hcase with ⟨_, h19, h20, h21, hfl⟩ | ⟨hk11, _⟩ | ⟨hk32, _⟩
        · by_cases hrd_safe : (rd != 19 && rd != 20 && (rd != 21 || k == 5 || k == 6)) = true
          · simp only [Bool.and_eq_true, Bool.or_eq_true, bne_iff_ne, beq_iff_eq] at hrd_safe
            rcases hrd_safe with ⟨⟨hne19, hne20⟩, hne21⟩
            refine ⟨hlen', ?_⟩
            split
            · rw [show (s'.pc - base) / 4 = k + 1 by rw [hpc'_next]; omega]
              have h19' := (hpr19 hne19.symm).1.trans h19
              have h20' := (hpr20 hne20.symm).trans h20
              have : k = 0 ∨ k = 1 ∨ k = 2 ∨ k = 3 ∨ k = 4 ∨ k = 5 ∨ k = 6 ∨ k = 7 := by omega
              rcases this with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl
              · have h21' := (hpr21 (by rcases hne21 with h | h | h <;> first | exact Ne.symm h | omega)).trans h21
                exact Or.inl ⟨v, h19', h20', h21', hnv⟩
              · have h21' := (hpr21 (by rcases hne21 with h | h | h <;> first | exact Ne.symm h | omega)).trans h21
                exact Or.inl ⟨v, h19', h20', h21', hnv⟩
              · have h21' := (hpr21 (by rcases hne21 with h | h | h <;> first | exact Ne.symm h | omega)).trans h21
                exact Or.inl ⟨v, h19', h20', h21', hnv⟩
              · have h21' := (hpr21 (by rcases hne21 with h | h | h <;> first | exact Ne.symm h | omega)).trans h21
                exact Or.inl ⟨v, h19', h20', h21', hnv⟩
              · exact ⟨v, h19', hnv⟩
              · exact ⟨v, v, h19', h20', hnv, hnv⟩
              · exact ⟨v, v, h19', h20', hnv, hnv⟩
              · trivial
            · trivial
          · simp only [Bool.not_eq_true] at hrd_safe
            rw [if_neg (by simp [hrd_safe])] at hsafe
            by_cases hk2 : k = 2 ∧ rd = 19
            · rcases hk2 with ⟨rfl, rfl⟩
              simp only [beq_self_eq_true, Bool.and_self, ↓reduceIte] at hsafe
              have hz := hfl ⟨by decide, by decide⟩
              have h20' := (hpr20 (by decide)).trans h20
              have h21' := (hpr21 (by decide)).trans h21
              refine ⟨hlen', ?_⟩
              rw [dif_pos (by rcases hb with ⟨_, _, hhi⟩; exact ⟨by omega, by omega, by omega⟩)]
              rw [show (s'.pc - base) / 4 = 3 by rw [hpc'_next]; omega]
              cases ins_other with
              | movWide kind w rd' imm hw =>
                simp [insWriteRd] at hwr; subst hwr
                simp only [Bool.and_eq_true, beq_iff_eq] at hsafe
                rcases hsafe with ⟨⟨rfl, rfl⟩, hna⟩
                simp only [exec] at hx; cases hx
                have hval : valNotAcc (AArch64Env.W - 1 - imm * pow2 (hw * 16)) :=
                  valNotAccB_sound hna
                have hlt : AArch64Env.W - 1 - imm * pow2 (hw * 16) < pow2 64 := hval.1
                have hmod_lt : (pow2 64 - 1 - imm * pow2 (hw * 16)) % pow2 64 = AArch64Env.W - 1 - imm * pow2 (hw * 16) :=
                  Nat.mod_eq_of_lt hlt
                exact Or.inr (Or.inl ⟨hz, _, v, v,
                  wrZR_pc_eq hlen 64 h19lt _ |>.trans (congrArg some hmod_lt), h20', h21', hval, hnv, hnv⟩)
              | logicReg op inv w rd' rn rm sh amt =>
                simp [insWriteRd] at hwr; subst hwr
                simp only [Bool.and_eq_true, Bool.not_eq_true', beq_iff_eq] at hsafe
                rcases hsafe with ⟨⟨⟨⟨⟨⟨rfl, rfl⟩, rfl⟩, rfl⟩, rfl⟩, rfl⟩, rfl⟩
                have hvW : v % AArch64Env.W = v := Nat.mod_eq_of_lt hv.1
                simp [exec, lift, bind, Option.bind, pure, St.rdZR, h19, h20, shiftVal, pow2,
                  show (18446744073709551616 : Nat) = AArch64Env.W from rfl, hvW, AArch64Phys.asrVal] at hx
                cases hx
                exact Or.inr (Or.inl ⟨hz, v, v, v,
                  wrZR_pc_eq hlen 64 h19lt _ |>.trans (by simp [show pow2 64 = AArch64Env.W from rfl, hvW]),
                  h20', h21', hnv, hnv, hnv⟩)
              | csel op w rd' rn rm cc =>
                simp [insWriteRd] at hwr; subst hwr
                simp only [Bool.and_eq_true, Bool.or_eq_true, beq_iff_eq] at hsafe
                rcases hsafe with ⟨⟨hmode, hrn_b⟩, hrm_b⟩
                have rd_raw : ∀ r, (r = 19 ∨ r = 20 ∨ r = 21 ∨ r ∈ VoterScratchRegs) → ∀ u, s.rdZR w r = some u →
                    ∃ raw, raw < AArch64Env.W ∧ (raw = v ∨ valNotAcc raw) ∧ u = raw % pow2 w := by
                  intro r hr u hu
                  have hne31 : r ≠ 31 := by
                    rcases hr with rfl | rfl | rfl | hm <;>
                      first | decide | (simp [VoterScratchRegs] at hm; omega)
                  simp only [St.rdZR, show (r == 31) = false by simp [hne31], Bool.false_eq_true, ↓reduceIte] at hu
                  cases hraw : s.reg r with
                  | none => simp [hraw] at hu
                  | some raw =>
                    simp [hraw] at hu; cases hu
                    rcases hr with rfl | rfl | rfl | hm
                    · rw [h19] at hraw; cases hraw; exact ⟨v, hv.1, Or.inl rfl, rfl⟩
                    · rw [h20] at hraw; cases hraw; exact ⟨v, hv.1, Or.inl rfl, rfl⟩
                    · rw [h21] at hraw; cases hraw; exact ⟨v, hv.1, Or.inl rfl, rfl⟩
                    · have hvr := hscr r hm raw hraw; exact ⟨raw, hvr.1, Or.inr hvr, rfl⟩
                have hrn_m : rn = 19 ∨ rn = 20 ∨ rn = 21 ∨ rn ∈ VoterScratchRegs := by
                  simp only [List.contains_eq_mem, decide_eq_true_eq] at hrn_b
                  rcases hrn_b with ((rfl | rfl) | rfl) | hmem
                  · exact Or.inl rfl
                  · exact Or.inr (Or.inl rfl)
                  · exact Or.inr (Or.inr (Or.inl rfl))
                  · exact Or.inr (Or.inr (Or.inr hmem))
                have hrm_m : rm = 19 ∨ rm = 20 ∨ rm = 21 ∨ rm ∈ VoterScratchRegs := by
                  simp only [List.contains_eq_mem, decide_eq_true_eq] at hrm_b
                  rcases hrm_b with ((rfl | rfl) | rfl) | hmem
                  · exact Or.inl rfl
                  · exact Or.inr (Or.inl rfl)
                  · exact Or.inr (Or.inr (Or.inl rfl))
                  · exact Or.inr (Or.inr (Or.inr hmem))
                have hx' := lift_ok_eq hx
                simp only [bind, Option.bind, pure] at hx'
                cases hr1 : s.rdZR w rn with
                | none => simp [hr1] at hx'
                | some a =>
                  cases hr2 : s.rdZR w rm with
                  | none => simp [hr1, hr2] at hx'
                  | some b =>
                    cases hc : AArch64Phys.cond s.fl cc with
                    | none => simp [hr1, hr2, hc] at hx'
                    | some condv =>
                      simp only [hr1, hr2, hc, Option.some.injEq] at hx'
                      change { s with pc := (s.pc + 4) % AArch64Env.W }.wrZR w 19 (evalCSel op condv w a b) = s' at hx'
                      have res_val : evalCSel op condv w a b < pow2 w ∧ valNotAcc (evalCSel op condv w a b) := by
                        rcases rd_raw rn hrn_m a hr1 with ⟨ra, hraW, hra, ha_def⟩
                        rcases rd_raw rm hrm_m b hr2 with ⟨rb, hrbW, hrb, hb_def⟩
                        rcases hmode with ⟨rfl, rfl⟩ | ⟨⟨⟨hopc, hw32⟩, hrnv⟩, hrmv⟩
                        · unfold evalCSel
                          cases condv
                          · refine ⟨?_, ?_⟩
                            · rw [hb_def, show pow2 64 = AArch64Env.W from rfl, Nat.mod_eq_of_lt hrbW]; exact hrbW
                            · rw [hb_def, show pow2 64 = AArch64Env.W from rfl, Nat.mod_eq_of_lt hrbW]
                              rcases hrb with rfl | hna; exact hnv; exact hna
                          · refine ⟨?_, ?_⟩
                            · rw [ha_def, show pow2 64 = AArch64Env.W from rfl, Nat.mod_eq_of_lt hraW]; exact hraW
                            · rw [ha_def, show pow2 64 = AArch64Env.W from rfl, Nat.mod_eq_of_lt hraW]
                              rcases hra with rfl | hna; exact hnv; exact hna
                        · have hv_eq : ∀ {r u}, ((r = 19 ∨ r = 20) ∨ r = 21) → s.rdZR w r = some u → u = v % pow2 w := by
                            intro r u hrr hu
                            have hne31 : r ≠ 31 := by rcases hrr with (rfl | rfl) | rfl <;> decide
                            simp only [St.rdZR, show (r == 31) = false by simp [hne31], Bool.false_eq_true, ↓reduceIte] at hu
                            cases hraw : s.reg r with
                            | none => simp [hraw] at hu
                            | some raw =>
                              simp [hraw] at hu
                              rcases hrr with (rfl | rfl) | rfl
                              · rw [h19] at hraw; cases hraw; exact hu.symm
                              · rw [h20] at hraw; cases hraw; exact hu.symm
                              · rw [h21] at hraw; cases hraw; exact hu.symm
                          have ha_eq := hv_eq hrnv hr1
                          have hb_eq := hv_eq hrmv hr2
                          rw [ha_eq, hb_eq]
                          cases condv <;> simp only [evalCSel, Bool.false_eq_true, ↓reduceIte]
                          · refine ⟨?_, ?_⟩
                            · cases op with
                              | csel => exact Nat.mod_lt _ (pow2_pos w)
                              | csinc => exact Nat.mod_lt _ (pow2_pos w)
                              | csinv =>
                                have hm := Nat.mod_lt v (pow2_pos w)
                                have hp := pow2_pos w
                                dsimp only
                                omega
                              | csneg => exact Nat.mod_lt _ (pow2_pos w)
                            · rcases hw32 with rfl | ⟨rfl, rfl⟩
                              · rw [show pow2 64 = AArch64Env.W from rfl, Nat.mod_eq_of_lt hv.1]
                                cases op with
                                | csel => exact hnv
                                | csinc => exact hv.2.2.2.2.2.1
                                | csinv => exact hv.2.2.2.2.2.2.1
                                | csneg => exact hv.2.2.2.2.2.2.2.1
                              · exact hv.2.2.2.2.2.2.2.2.1
                          · refine ⟨Nat.mod_lt _ (pow2_pos w), ?_⟩
                            rcases hw32 with rfl | ⟨rfl, _⟩
                            · rw [show pow2 64 = AArch64Env.W from rfl, Nat.mod_eq_of_lt hv.1]; exact hnv
                            · exact hv.2.2.2.2.2.2.2.2.1
                      have h19' : s'.reg 19 = some (evalCSel op condv w a b) := by
                        rw [← hx']
                        exact wrZR_pc_eq hlen w h19lt _ |>.trans (congrArg some (Nat.mod_eq_of_lt res_val.1))
                      have hz' : s'.fl.z = some true := by rw [← hx']; exact hz
                      exact Or.inr (Or.inl ⟨hz', _, v, v, h19', h20', h21', res_val.2, hnv, hnv⟩)
              | _ => cases hsafe
            · rw [if_neg (by simp [hk2])] at hsafe
              have h56 : (k = 5 ∨ k = 6) ∧ rd = 20 := by
                by_cases hcond : ((k == 5 || k == 6) && rd == 20) = true
                · simp only [Bool.and_eq_true, Bool.or_eq_true, beq_iff_eq] at hcond; exact hcond
                · rw [if_neg hcond] at hsafe; cases hsafe
              rcases h56 with ⟨hk56, rfl⟩
              rw [if_pos (by rcases hk56 with rfl | rfl <;> simp)] at hsafe
              have h19' := (hpr19 (by decide)).1.trans h19
              have hv20' : ∃ v20, s'.reg 20 = some v20 ∧ valNotAcc v20 := by
                cases ins_other with
                | logicReg op inv w rd' rn rm sh amt =>
                  simp [insWriteRd] at hwr; subst hwr
                  simp only [Bool.and_eq_true, Bool.or_eq_true, Bool.not_eq_true',
                    bne_iff_ne, beq_iff_eq, List.contains_eq_mem, decide_eq_true_eq] at hsafe
                  rcases hsafe with ⟨⟨⟨hw, hror⟩, hamt⟩, hform⟩
                  have hw_mem : w = 64 ∨ w = 32 := hw
                  have hsh_mem : sh ∈ [ShiftKind.lsl, .lsr, .asr] := by
                    cases sh <;> simp_all
                  have hamt_mem : amt ∈ [0, 1, 2, 4, 8, 16, 32] := by
                    simpa using hamt
                  have hx' := lift_ok_eq hx
                  simp only [bind, Option.bind, pure] at hx'
                  cases hr1 : s.rdZR w rn with
                  | none => simp [hr1] at hx'
                  | some a =>
                    cases hr2 : s.rdZR w rm with
                    | none => simp [hr1, hr2] at hx'
                    | some b =>
                      simp only [hr1, hr2] at hx'
                      have hres : ∃ res, evalLogic op inv sh w amt a b = res ∧
                          res < pow2 w ∧ valNotAcc res := by
                        rcases hform with (((⟨⟨⟨⟨rfl, rfl⟩, rfl⟩, rfl⟩⟩ |
                            ⟨⟨⟨⟨⟨⟨rfl, rfl⟩, rfl⟩, hrmm⟩, rfl⟩, rfl⟩⟩) |
                            ⟨⟨⟨⟨⟨rfl, rfl⟩, hrn31⟩, rfl⟩, rfl⟩⟩) |
                            ⟨⟨⟨⟨⟨⟨rfl, rfl⟩, rfl⟩, rfl⟩, rfl⟩, rfl⟩⟩) |
                            ⟨⟨⟨⟨hop, rfl⟩, rfl⟩, rfl⟩, rfl⟩
                        · simp [St.rdZR] at hr1; cases hr1
                          simp [St.rdZR, h19] at hr2; cases hr2
                          refine ⟨shiftVal sh w (v % pow2 w) amt, by simp [evalLogic],
                            shiftVal_lt_pow2 (fun h => by subst h; simp at hror), ?_⟩
                          rcases hw_mem with rfl | rfl
                          · rw [show pow2 64 = AArch64Env.W from rfl, Nat.mod_eq_of_lt hv.1]
                            exact (SafeInvalid_shift hv hsh_mem hamt_mem).1
                          · exact (SafeInvalid_shift hv hsh_mem hamt_mem).2
                        · simp [St.rdZR] at hr1; cases hr1
                          have hne31 : rm ≠ 31 := by simp [VoterScratchRegs] at hrmm; omega
                          simp only [St.rdZR, show (rm == 31) = false by simp [hne31], Bool.false_eq_true, ↓reduceIte] at hr2
                          cases hraw : s.reg rm with
                          | none => simp [hraw] at hr2
                          | some raw =>
                            simp [hraw] at hr2; cases hr2
                            have hvr := hscr rm hrmm raw hraw
                            have hraw_p : raw < pow2 64 := by rw [show pow2 64 = AArch64Env.W from rfl]; exact hvr.1
                            have hsh0 : shiftVal sh 64 (raw % pow2 64) 0 = raw := by
                              rw [Nat.mod_eq_of_lt hraw_p]
                              exact shiftVal_zero hror hraw_p
                            refine ⟨raw, ?_, hraw_p, hvr⟩
                            unfold evalLogic; dsimp only
                            rw [hsh0]; exact zero_lor_eq raw
                        · have ha_lt : a < pow2 w := by
                            by_cases h31 : rn = 31
                            · simp [St.rdZR, h31] at hr1; cases hr1; exact pow2_pos w
                            · simp only [St.rdZR, show (rn == 31) = false by simp [h31], Bool.false_eq_true, ↓reduceIte] at hr1
                              cases hraw : s.reg rn with
                              | none => simp [hraw] at hr1
                              | some _ => simp [hraw] at hr1; cases hr1; exact Nat.mod_lt _ (pow2_pos w)
                          simp [St.rdZR, h19] at hr2; cases hr2
                          have hvmod_lt : v % pow2 w < pow2 w := Nat.mod_lt _ (pow2_pos w)
                          have hpowW : pow2 w ≤ AArch64Env.W := by
                            rcases hw_mem with rfl | rfl <;> decide
                          have hodd_m : (v % pow2 w) % 2 = 1 := by
                            have hd : 2 ∣ pow2 w := by rcases hw_mem with rfl | rfl <;> decide
                            rw [Nat.mod_mod_of_dvd _ hd]; exact hodd
                          have hsh0 : shiftVal sh w (v % pow2 w) 0 = v % pow2 w :=
                            shiftVal_zero hror hvmod_lt
                          refine ⟨Nat.lor a (v % pow2 w), ?_, Nat.bitwise_lt_two_pow ha_lt hvmod_lt,
                            valNotAcc_lor_odd hodd_m (Nat.lt_of_lt_of_le ha_lt hpowW) (Nat.lt_of_lt_of_le hvmod_lt hpowW)⟩
                          unfold evalLogic; dsimp only; rw [hsh0]; rfl
                        · simp [St.rdZR] at hr1; cases hr1
                          simp [St.rdZR, h19] at hr2; cases hr2
                          have hv_p : v < pow2 64 := by rw [show pow2 64 = AArch64Env.W from rfl]; exact hv.1
                          have hsh0 : shiftVal sh 64 (v % pow2 64) 0 = v := by
                            rw [Nat.mod_eq_of_lt hv_p]; exact shiftVal_zero hror hv_p
                          refine ⟨AArch64Env.W - 1 - v, ?_, ?_, hv.2.2.2.2.2.2.1⟩
                          · unfold evalLogic; dsimp only
                            rw [hsh0, show pow2 64 = AArch64Env.W from rfl]
                            exact zero_lor_eq _
                          · rw [show pow2 64 = AArch64Env.W from rfl]; have := hv.1; omega
                        · simp [St.rdZR] at hr1; cases hr1
                          refine ⟨0, ?_, pow2_pos w, safeInvalid_zero.valNotAcc⟩
                          unfold evalLogic; dsimp only; rcases hop with rfl | rfl <;> simp
                      rcases hres with ⟨res, hres, hres_lt, hna⟩
                      have hs'_eq : ∃ fl', s' = { s with pc := (s.pc + 4) % AArch64Env.W, fl := fl' }.wrZR w 20
                          (evalLogic op inv sh w amt a b) := by
                        by_cases hops : op = .ands
                        · subst hops; simp at hx'; cases hx'; unfold evalLogic; exact ⟨_, rfl⟩
                        · simp [hops] at hx'; cases hx'; unfold evalLogic; exact ⟨_, rfl⟩
                      rcases hs'_eq with ⟨fl', hs'_eq⟩
                      have hs'_reg : s'.reg 20 = some res := by
                        rw [hs'_eq]
                        have hmod_lt : evalLogic op inv sh w amt a b % pow2 w = res := by
                          rw [hres]; exact Nat.mod_eq_of_lt hres_lt
                        exact (wrZR_pc_eq (s := { s with fl := fl' }) hlen w h20lt
                          (evalLogic op inv sh w amt a b) (p := (s.pc + 4) % AArch64Env.W)).trans
                          (congrArg some hmod_lt)
                      exact ⟨res, hs'_reg, hna⟩
                | addSubReg isSub setFlags w rd' rn rm sh amt =>
                  simp [insWriteRd] at hwr; subst hwr
                  simp only [Bool.and_eq_true, Bool.or_eq_true, Bool.not_eq_true', beq_iff_eq] at hsafe
                  rcases hsafe with ⟨⟨⟨⟨⟨rfl, hw⟩, rfl⟩, rfl⟩, rfl⟩, rfl⟩
                  simp [exec, lift, bind, Option.bind, pure, St.rdZR, h19, shiftVal, pow2] at hx
                  have hs'_eq : ∃ fl', s' = { s with pc := (s.pc + 4) % AArch64Env.W, fl := fl' }.wrZR w 20
                      (addSubOp false w 0 (v % pow2 w)).1 := by
                    by_cases hsf : setFlags = true <;> simp [hsf] at hx <;> cases hx <;> exact ⟨_, rfl⟩
                  rcases hs'_eq with ⟨fl', hs'_eq⟩
                  have hres_na : valNotAcc (v % pow2 w) := by
                    rcases hw with rfl | rfl
                    · rw [show pow2 64 = AArch64Env.W from rfl, Nat.mod_eq_of_lt hv.1]; exact hnv
                    · exact hv.2.2.2.2.2.2.2.2.1
                  have hlt : v % pow2 w < pow2 w := Nat.mod_lt _ (pow2_pos w)
                  have hadd : (addSubOp false w 0 (v % pow2 w)).1 = v % pow2 w := by
                    unfold addSubOp; simp [Nat.mod_eq_of_lt hlt]
                  rw [hs'_eq, hadd]
                  exact ⟨v % pow2 w,
                    (wrZR_pc_eq (s := { s with fl := fl' }) hlen w h20lt (v % pow2 w)
                      (p := (s.pc + 4) % AArch64Env.W)).trans (congrArg some (Nat.mod_eq_of_lt hlt)),
                    hres_na⟩
                | _ => cases hsafe
              rcases hv20' with ⟨v20, h20', hnv20⟩
              refine ⟨hlen', ?_⟩
              split
              · rw [show (s'.pc - base) / 4 = k + 1 by rw [hpc'_next]; omega]
                rcases hk56 with rfl | rfl <;> exact ⟨v, v20, h19', h20', hnv, hnv20⟩
              · trivial
        · omega
        · omega
      · rw [if_neg hk8] at hsafe
        by_cases hk20 : k ≤ 20
        · rw [if_pos hk20] at hsafe
          simp only [bne_iff_ne] at hsafe
          rcases hcase with ⟨hk8', _⟩ | ⟨hk11, _, _, h20, _⟩ | ⟨hk32, _⟩
          · omega
          · refine ⟨hlen', ?_⟩
            have h20' := (hpr20 hsafe.symm).trans h20
            split
            · rw [show (s'.pc - base) / 4 = k + 1 by rw [hpc'_next]; omega]
              have : k = 11 ∨ k = 12 ∨ k = 13 ∨ k = 14 ∨ k = 15 ∨ k = 16 ∨ k = 17 ∨ k = 18 ∨ k = 19 ∨ k = 20 := by omega
              rcases this with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl <;>
                first | exact Or.inl ⟨v, h20', hnv⟩ | exact Or.inr ⟨v, h20', hnv⟩ | exact ⟨v, h20', hnv⟩
            · trivial
          · omega
        · rw [if_neg hk20] at hsafe
          by_cases hk38 : k ≤ 38 ∨ k = 43
          · refine ⟨hlen', ?_⟩
            split
            · rw [show (s'.pc - base) / 4 = k + 1 by rw [hpc'_next]; omega]
              rcases hk38 with hle | rfl
              · have : k = 32 ∨ k = 33 ∨ k = 34 ∨ k = 35 ∨ k = 36 ∨ k = 37 ∨ k = 38 := by omega
                rcases this with rfl | rfl | rfl | rfl | rfl | rfl | rfl <;> trivial
              · trivial
            · trivial
          · have hkgt : ¬ (k ≤ 38 || k == 43) = true := by simp [hk38]
            rw [if_neg hkgt] at hsafe
            by_cases hk3944 : k = 39 ∨ k = 44
            · have hx0_ok : ∃ v0, s'.reg 0 = some v0 ∧ v0 % 256 ≠ 0 := by
                rcases hcase with ⟨hk8', _⟩ | ⟨hk11, _⟩ | ⟨_, _, _, _, _, _, _, _, _, h39_42, _, h44_47, _, _, _, _⟩
                · omega
                · omega
                · by_cases hne0 : 0 ≠ rd
                  · rcases hk3944 with rfl | rfl
                    · exact ⟨6, (hpr0 hne0).trans (h39_42 ⟨by decide, by decide⟩), by decide⟩
                    · exact ⟨5, (hpr0 hne0).trans (h44_47 ⟨by decide, by decide⟩), by decide⟩
                  · have hrd0 : rd = 0 := by omega
                    subst hrd0
                    have hcond : (k == 39 || k == 44) = true := by rcases hk3944 with rfl | rfl <;> simp
                    rw [if_pos hcond] at hsafe
                    simp only [bne_iff_ne, ne_eq, not_true_eq_false, false_or, Bool.or_eq_true, beq_iff_eq] at hsafe
                    rcases hsafe with rfl | rfl
                    · simp only [exec, pow2] at hx; cases hx
                      exact ⟨6, by simpa [pow2] using wrZR_pc_eq hlen 64 h0lt 6, by decide⟩
                    · simp only [exec, pow2] at hx; cases hx
                      exact ⟨5, by simpa [pow2] using wrZR_pc_eq hlen 64 h0lt 5, by decide⟩
              rcases hx0_ok with ⟨v0, hx0', hnz⟩
              refine ⟨hlen', ?_⟩
              split
              · rw [show (s'.pc - base) / 4 = k + 1 by rw [hpc'_next]; omega]
                rcases hk3944 with rfl | rfl <;> exact Or.inl ⟨v0, hx0', hnz⟩
              · trivial
            · rw [if_neg (by simp [hk3944])] at hsafe
              by_cases hk4045 : k = 40 ∨ k = 45
              · rw [if_pos (by rcases hk4045 with rfl | rfl <;> simp)] at hsafe
                have hx0_ok : ∃ v0, s'.reg 0 = some v0 ∧ v0 % 256 ≠ 0 := by
                  rcases hcase with ⟨hk8', _⟩ | ⟨hk11, _⟩ | ⟨_, _, _, _, _, _, _, _, _, h39_42, h40_2, h44_47, h45_2, _, _, _⟩
                  · omega
                  · omega
                  · by_cases hne0 : 0 ≠ rd
                    · rcases hk4045 with rfl | rfl
                      · exact ⟨6, (hpr0 hne0).trans (h39_42 ⟨by decide, by decide⟩), by decide⟩
                      · exact ⟨5, (hpr0 hne0).trans (h44_47 ⟨by decide, by decide⟩), by decide⟩
                    · have hrd0 : rd = 0 := by omega
                      subst hrd0
                      simp only [Bool.or_eq_true, bne_iff_ne, ne_eq, not_true_eq_false, false_or] at hsafe
                      have hv0_eq : s.reg 0 = some (if k = 40 then 6 else 5) ∧
                          s.reg 2 = some (if k = 40 then 6 else 5) := by
                        rcases hk4045 with rfl | rfl
                        · exact ⟨h39_42 ⟨by decide, by decide⟩, h40_2 rfl⟩
                        · exact ⟨h44_47 ⟨by decide, by decide⟩, h45_2 rfl⟩
                      let v0 := if k = 40 then (6 : Nat) else 5
                      have hv0_lt : v0 < 16 := by rcases hk4045 with rfl | rfl <;> decide
                      have hv0_nz : v0 % 256 ≠ 0 := by rcases hk4045 with rfl | rfl <;> decide
                      have hx0_0 : s.reg 0 = some v0 := hv0_eq.1
                      have hx2_0 : s.reg 2 = some v0 := hv0_eq.2
                      cases ins_other with
                      | addSubReg isSub setFlags w rd' rn rm sh amt =>
                        simp [insWriteRd] at hwr; subst hwr
                        simp only [Bool.and_eq_true, Bool.or_eq_true, Bool.not_eq_true', beq_iff_eq] at hsafe
                        rcases hsafe with ⟨⟨⟨⟨⟨rfl, hw⟩, rfl⟩, rfl⟩, rfl⟩, rfl⟩
                        simp [exec, lift, bind, Option.bind, pure, St.rdZR, hx0_0, hx2_0, shiftVal, pow2] at hx
                        have hs'_eq : ∃ fl', s' = { s with pc := (s.pc + 4) % AArch64Env.W, fl := fl' }.wrZR w 0
                            (addSubOp false w (v0 % pow2 w) (v0 % pow2 w)).1 := by
                          by_cases hsf : setFlags = true <;> simp [hsf] at hx <;> cases hx <;> exact ⟨_, rfl⟩
                        rcases hs'_eq with ⟨fl', hs'_eq⟩
                        have hlt : (addSubOp false w (v0 % pow2 w) (v0 % pow2 w)).1 < pow2 w := by
                          unfold addSubOp; exact Nat.mod_lt _ (pow2_pos w)
                        have hval : (addSubOp false w (v0 % pow2 w) (v0 % pow2 w)).1 = v0 + v0 := by
                          unfold addSubOp; simp only [Bool.false_eq_true, ↓reduceIte]
                          have hv0_w : v0 < pow2 w := by
                            rcases hw with rfl | rfl
                            · exact Nat.lt_trans hv0_lt (by decide)
                            · exact Nat.lt_trans hv0_lt (by decide)
                          have hv0_sum : v0 + v0 < pow2 w := by
                            rcases hw with rfl | rfl
                            · exact Nat.lt_trans (by omega : v0 + v0 < 32) (by decide)
                            · exact Nat.lt_trans (by omega : v0 + v0 < 32) (by decide)
                          rw [Nat.mod_eq_of_lt hv0_w, Nat.mod_eq_of_lt hv0_sum]
                        rw [hs'_eq]
                        have hlt_v0 : v0 + v0 < pow2 w := by rw [← hval]; exact hlt
                        refine ⟨v0 + v0, ?_, ?_⟩
                        · rw [hval]
                          exact (wrZR_pc_eq (s := { s with fl := fl' }) hlen w h0lt (v0 + v0)
                            (p := (s.pc + 4) % AArch64Env.W)).trans (congrArg some (Nat.mod_eq_of_lt hlt_v0))
                        · rcases hk4045 with rfl | rfl <;> decide
                      | logicReg op inv w rd' rn rm sh amt =>
                        simp [insWriteRd] at hwr; subst hwr
                        simp only [Bool.and_eq_true, Bool.or_eq_true, Bool.not_eq_true',
                          decide_eq_true_eq, bne_iff_ne, beq_iff_eq] at hsafe
                        rcases hsafe with ⟨⟨⟨hw, hror⟩, hamt⟩, hform⟩
                        have hx' := lift_ok_eq hx
                        simp only [bind, Option.bind, pure] at hx'
                        cases hr1 : s.rdZR w rn with
                        | none => simp [hr1] at hx'
                        | some a =>
                          cases hr2 : s.rdZR w rm with
                          | none => simp [hr1, hr2] at hx'
                          | some b =>
                            simp only [hr1, hr2] at hx'
                            have hpow_ge : 256 ≤ pow2 w := by rcases hw with rfl | rfl <;> decide
                            have hv0_w : v0 < pow2 w := by omega
                            have hsh_lt : shiftVal sh w b amt < pow2 w :=
                              shiftVal_lt_pow2 (fun h => by subst h; simp at hror)
                            have hres : evalLogic op inv sh w amt a b < pow2 w ∧
                                evalLogic op inv sh w amt a b % 256 ≠ 0 := by
                              rcases hform with ((⟨⟨rfl, rfl⟩, rfl | ⟨⟨⟨hrn31, rfl⟩, rfl⟩, rfl⟩⟩ |
                                  ⟨⟨⟨⟨⟨rfl, rfl⟩, rfl⟩, rfl⟩, rfl⟩, rfl⟩) |
                                  ⟨⟨⟨⟨⟨hop, rfl⟩, rfl⟩, rfl⟩, rfl⟩, rfl⟩)
                              · simp [St.rdZR, hx0_0, Nat.mod_eq_of_lt hv0_w] at hr1; cases hr1
                                unfold evalLogic
                                exact ⟨Nat.bitwise_lt_two_pow hv0_w hsh_lt, lor_mod256_ne0 (Or.inl hv0_nz)⟩
                              · simp [St.rdZR, hx2_0, Nat.mod_eq_of_lt hv0_w] at hr2; cases hr2
                                have ha_lt : a < pow2 w := by
                                  by_cases h31 : rn = 31
                                  · simp [St.rdZR, h31] at hr1; cases hr1; exact pow2_pos w
                                  · simp only [St.rdZR, show (rn == 31) = false by simp [h31], Bool.false_eq_true, ↓reduceIte] at hr1
                                    cases hraw : s.reg rn with
                                    | none => simp [hraw] at hr1
                                    | some _ => simp [hraw] at hr1; cases hr1; exact Nat.mod_lt _ (pow2_pos w)
                                have hsh0 : shiftVal .lsl w v0 0 = v0 := shiftVal_zero (by decide) hv0_w
                                unfold evalLogic; dsimp only; rw [hsh0]
                                exact ⟨Nat.bitwise_lt_two_pow ha_lt hv0_w, lor_mod256_ne0 (Or.inr hv0_nz)⟩
                              · simp [St.rdZR, hx0_0, Nat.mod_eq_of_lt hv0_w] at hr1; cases hr1
                                simp [St.rdZR, hx2_0, Nat.mod_eq_of_lt hv0_w] at hr2; cases hr2
                                have hsh0 : shiftVal .lsl w v0 0 = v0 := shiftVal_zero (by decide) hv0_w
                                unfold evalLogic; dsimp only; rw [hsh0]
                                have hsub_lt : pow2 w - 1 - v0 < pow2 w := by have := hpow_ge; omega
                                exact ⟨Nat.bitwise_lt_two_pow hv0_w hsub_lt, lor_mod256_ne0 (Or.inl hv0_nz)⟩
                              · simp [St.rdZR, hx0_0, Nat.mod_eq_of_lt hv0_w] at hr1; cases hr1
                                simp [St.rdZR, hx2_0, Nat.mod_eq_of_lt hv0_w] at hr2; cases hr2
                                have hsh0 : shiftVal .lsl w v0 0 = v0 := shiftVal_zero (by decide) hv0_w
                                unfold evalLogic; dsimp only; rw [hsh0]
                                rcases hop with rfl | rfl <;>
                                  (simp only [Bool.false_eq_true, ↓reduceIte, land_self_eq]
                                   exact ⟨hv0_w, hv0_nz⟩)
                            have hs'_eq : ∃ fl', s' = { s with pc := (s.pc + 4) % AArch64Env.W, fl := fl' }.wrZR w 0
                                (evalLogic op inv sh w amt a b) := by
                              by_cases hops : op = .ands
                              · subst hops; simp at hx'; cases hx'; unfold evalLogic; exact ⟨_, rfl⟩
                              · simp [hops] at hx'; cases hx'; unfold evalLogic; exact ⟨_, rfl⟩
                            rcases hs'_eq with ⟨fl', hs'_eq⟩
                            have hs'_reg : s'.reg 0 = some (evalLogic op inv sh w amt a b) := by
                              rw [hs'_eq]
                              exact wrZR_pc_eq (s := { s with fl := fl' }) hlen w h0lt
                                (evalLogic op inv sh w amt a b) (p := (s.pc + 4) % AArch64Env.W) |>.trans
                                (congrArg some (Nat.mod_eq_of_lt hres.1))
                            exact ⟨_, hs'_reg, hres.2⟩
                      | _ => cases hsafe
                rcases hx0_ok with ⟨v0, hx0', hnz⟩
                refine ⟨hlen', ?_⟩
                split
                · rw [show (s'.pc - base) / 4 = k + 1 by rw [hpc'_next]; omega]
                  rcases hk4045 with rfl | rfl <;> exact ⟨v0, hx0', hnz⟩
                · trivial
              · rw [if_neg (by simp [hk4045])] at hsafe
                by_cases hk4146 : k = 41 ∨ k = 46
                · rw [if_pos (by rcases hk4146 with rfl | rfl <;> simp)] at hsafe
                  have hx0_ok : ∃ v0, s'.reg 0 = some v0 ∧ v0 % 256 ≠ 0 := by
                    rcases hcase with ⟨hk8', _⟩ | ⟨hk11, _⟩ | ⟨_, _, _, _, _, _, _, _, _, h39_42, _, h44_47, _, _, _, _⟩
                    · omega
                    · omega
                    · by_cases hne0 : 0 ≠ rd
                      · rcases hk4146 with rfl | rfl
                        · exact ⟨6, (hpr0 hne0).trans (h39_42 ⟨by decide, by decide⟩), by decide⟩
                        · exact ⟨5, (hpr0 hne0).trans (h44_47 ⟨by decide, by decide⟩), by decide⟩
                      · have hrd0 : rd = 0 := by omega
                        subst hrd0
                        simp only [Bool.or_eq_true, bne_iff_ne, ne_eq, not_true_eq_false, false_or, beq_iff_eq] at hsafe
                        rw [hsafe] at hx; simp only [exec, pow2] at hx; cases hx
                        exact ⟨93, by simpa [pow2] using wrZR_pc_eq hlen 64 h0lt 93, by decide⟩
                  rcases hx0_ok with ⟨v0, hx0', hnz⟩
                  refine ⟨hlen', ?_⟩
                  split
                  · rw [show (s'.pc - base) / 4 = k + 1 by rw [hpc'_next]; omega]
                    rcases hk4146 with rfl | rfl <;> exact fun _ => ⟨v0, hx0', hnz⟩
                  · trivial
                · rw [if_neg (by simp [hk4146])] at hsafe
                  by_cases hk4247 : k = 42 ∨ k = 47
                  · refine ⟨hlen', ?_⟩
                    split
                    · rw [show (s'.pc - base) / 4 = k + 1 by rw [hpc'_next]; omega]
                      rcases hk4247 with rfl | rfl <;> trivial
                    · trivial
                  · rw [if_neg (by simp [hk4247])] at hsafe; cases hsafe

private theorem tailAltWordSafe_sound {base v k : Nat} (hb : base_ok base)
    (hk48 : k < 48) {s : St} (hp : PreFault base v s)
    (hpc : s.pc = base + 4 * k) {ins : Ins} (hsafe : tailAltWordSafe k ins = true) :
    match exec ins 4 s with
    | .ok s' => TailSafe base s'
    | .exit c => c ≠ 0
    | .stuck => True := by
  have hpre := hp
  rcases hp with ⟨_, _, hlen, _, k', hpc', hcase⟩
  have hkeq : k' = k := by rw [hpc] at hpc'; omega
  rw [hkeq] at hpc' hcase
  have branchCheck : ∀ off, (oobOffBool k off = true ∨
      ((4 * k + off) % 4 = 0 ∧ 4 * k + off < 192 ∧ branchTargetSafe k ((4 * k + off) / 4) = true)) →
      TailSafe base { s with pc := (s.pc + off) % AArch64Env.W } := by
    intro off hbo
    refine ⟨hlen, ?_⟩
    rw [hpc]
    rcases hbo with hoob | ⟨hal, hlt192, hbts⟩
    · split
      · next hbd => exact absurd hbd (oobOff_not_bounds hb hk48 hoob)
      · trivial
    · split
      · have hmod : (base + 4 * k + off) % AArch64Env.W = base + (4 * k + off) := by
          rcases hb with ⟨_, _, hhi⟩; rw [Nat.mod_eq_of_lt (by omega)]; omega
        rw [hmod, show (base + (4 * k + off) - base) / 4 = (4 * k + off) / 4 by omega]
        exact branchTargetSafe_sound_k (hpre.1) hcase hbts
      · trivial
  have hfall : TailSafe base { s with pc := (s.pc + 4) % AArch64Env.W } :=
    tailSafe_fall_k_succ hb hk48 hpre hlen rfl rfl rfl rfl (fun _ _ => rfl) hpc
  by_cases hbi : ∃ link off, ins = .b link off
  · rcases hbi with ⟨link, off, rfl⟩
    simp only [tailAltWordSafe, Bool.or_eq_true, Bool.and_eq_true, decide_eq_true_eq, beq_iff_eq] at hsafe
    simp only [exec]
    have hbo : oobOffBool k off = true ∨
        ((4 * k + off) % 4 = 0 ∧ 4 * k + off < 192 ∧ branchTargetSafe k ((4 * k + off) / 4) = true) := by
      rcases hsafe with ho | ⟨⟨hal, hlt⟩, hbt⟩; exact Or.inl ho; exact Or.inr ⟨hal, hlt, hbt⟩
    have hbc := branchCheck off hbo
    cases link with
    | false => exact hbc
    | true => exact tailSafe_setReg30_upd hbc (some ((s.pc + 4) % AArch64Env.W))
  by_cases hbci : ∃ cc off, ins = .bcond cc off
  · rcases hbci with ⟨cc, off, rfl⟩
    simp only [tailAltWordSafe, Bool.or_eq_true, Bool.and_eq_true, decide_eq_true_eq, beq_iff_eq] at hsafe
    simp only [exec, lift, bind, Option.bind, pure]
    cases hc : AArch64Phys.cond s.fl cc with
    | none => trivial
    | some taken =>
      cases taken
      · simpa [hc] using hfall
      · rcases hsafe with (⟨rfl, rfl⟩ | ho) | ⟨⟨hal, hlt⟩, hbt⟩
        · rcases hcase with ⟨_, _, _, _, hfl⟩ | ⟨hk11, _⟩ | ⟨hk32, _⟩
          · have hz := hfl ⟨by decide, by decide⟩
            simp [AArch64Phys.cond, hz] at hc
          · omega
          · omega
        · simpa [hc] using branchCheck off (Or.inl ho)
        · simpa [hc] using branchCheck off (Or.inr ⟨hal, hlt, hbt⟩)
  by_cases hcbzi : ∃ nz w rt off, ins = .cbz nz w rt off
  · rcases hcbzi with ⟨nz, w, rt, off, rfl⟩
    simp only [tailAltWordSafe, Bool.or_eq_true, Bool.and_eq_true, decide_eq_true_eq, beq_iff_eq] at hsafe
    simp only [exec, lift, bind, Option.bind, pure]
    cases hrd : s.rdZR w rt with
    | none => trivial
    | some u =>
      by_cases htk : (if nz then u != 0 else u == 0) = true
      · simp only [htk, ↓reduceIte]
        rcases hsafe with ho | ⟨⟨hal, hlt⟩, hbt⟩
        · exact branchCheck off (Or.inl ho)
        · exact branchCheck off (Or.inr ⟨hal, hlt, hbt⟩)
      · simp only [htk]; exact hfall
  by_cases htbzi : ∃ nz bit rt off, ins = .tbz nz bit rt off
  · rcases htbzi with ⟨nz, bit, rt, off, rfl⟩
    simp only [tailAltWordSafe, Bool.or_eq_true, Bool.and_eq_true, decide_eq_true_eq, beq_iff_eq] at hsafe
    simp only [exec, lift, bind, Option.bind, pure]
    cases hrd : s.rdZR 64 rt with
    | none => trivial
    | some u =>
      by_cases htk : (if nz then (u / pow2 bit) % 2 == 1 else !((u / pow2 bit) % 2 == 1)) = true
      · simp only [htk, ↓reduceIte]
        rcases hsafe with ho | ⟨⟨hal, hlt⟩, hbt⟩
        · exact branchCheck off (Or.inl ho)
        · exact branchCheck off (Or.inr ⟨hal, hlt, hbt⟩)
      · simp only [htk]; exact hfall
  by_cases hsvci : ∃ imm, ins = .svc imm
  · rcases hsvci with ⟨imm, rfl⟩
    simp only [tailAltWordSafe, Bool.or_eq_true, bne_iff_ne, beq_iff_eq] at hsafe
    simp only [exec]
    by_cases himm : imm = 0
    · rw [if_neg (by simp [himm])]
      rcases hsafe with (hne | rfl) | rfl
      · exact absurd himm hne
      · rcases hcase with ⟨hk8, _⟩ | ⟨hk11, hk20, _⟩ | ⟨_, _, _, _, _, _, _, _, _, h39_42, _, _, _, _, h42_8, _⟩
        · omega
        · omega
        · have h0 := h39_42 ⟨by decide, by decide⟩
          have h8 := h42_8 rfl
          simp [AArch64Phys.execSvc, AArch64Phys.applySys, AArch64Env.syscall, h0, h8]
      · rcases hcase with ⟨hk8, _⟩ | ⟨hk11, hk20, _⟩ | ⟨_, _, _, _, _, _, _, _, _, _, _, h44_47, _, _, _, h47_8⟩
        · omega
        · omega
        · have h0 := h44_47 ⟨by decide, by decide⟩
          have h8 := h47_8 rfl
          simp [AArch64Phys.execSvc, AArch64Phys.applySys, AArch64Env.syscall, h0, h8]
    · have hnb : (imm != 0) = true := by simp [himm]
      rw [if_pos hnb]; trivial
  by_cases hlduri : ∃ sz rd rn off, ins = .ldur sz rd rn off
  · rcases hlduri with ⟨sz, rd, rn, off, rfl⟩
    simp only [tailAltWordSafe, Bool.and_eq_true, decide_eq_true_eq, bne_iff_ne] at hsafe
    rcases hsafe with ⟨⟨⟨hk8, hne19⟩, hne20⟩, hne21⟩
    cases hx : exec (.ldur sz rd rn off) 4 s with
    | stuck => trivial
    | exit c => exact absurd hx (fun h => by obtain ⟨_, hc, _⟩ := exec_exit_only_svc h; cases hc)
    | ok s' =>
      have h19s := exec_ldur_preserves_regs_ne hx hne19.symm
      have h20s := exec_ldur_preserves_regs_ne hx hne20.symm
      have h21s := exec_ldur_preserves_regs_ne hx hne21.symm
      exact tailSafe_fall_k_succ hb hk48 hpre (by rw [h19s.2.1, hlen]) h19s.2.2 h19s.1 h20s.1 h21s.1
        (fun h39 => by omega) hpc
  exact tailAltWordSafe_sound_other hb hk48 hpre hpc hsafe
    (fun l o h => hbi ⟨l, o, h⟩) (fun c o h => hbci ⟨c, o, h⟩)
    (fun n w r o h => hcbzi ⟨n, w, r, o, h⟩) (fun n b r o h => htbzi ⟨n, b, r, o, h⟩)
    (fun i h => hsvci ⟨i, h⟩) (fun sz r rn o h => hlduri ⟨sz, r, rn, o, h⟩)

private theorem xor_5_mod256_ne0 {b : Nat} (_hb : b < 64) :
    Nat.xor 5 (2 ^ b) % 256 ≠ 0 := by
  by_cases h8 : b < 8
  · have hall : (List.range 8).all (fun i => Nat.xor 5 (2 ^ i) % 256 != 0) = true := by decide +kernel
    exact bne_iff_ne.1 (List.all_eq_true.1 hall b (List.mem_range.2 h8))
  · rw [mod_256_xor_ge_8 5 b (Nat.le_of_not_lt h8)]; decide

private theorem xor_6_mod256_ne0 {b : Nat} (_hb : b < 64) :
    Nat.xor 6 (2 ^ b) % 256 ≠ 0 := by
  by_cases h8 : b < 8
  · have hall : (List.range 8).all (fun i => Nat.xor 6 (2 ^ i) % 256 != 0) = true := by decide +kernel
    exact bne_iff_ne.1 (List.all_eq_true.1 hall b (List.mem_range.2 h8))
  · rw [mod_256_xor_ge_8 6 b (Nat.le_of_not_lt h8)]; decide

private theorem PreFault_flipReg_safe {base v : Nat} (_hb : base_ok base) {s : St}
    (hp : PreFault base v s) {i b : Nat} (hi : i < 32) (hbit : b < 64) :
    TailSafe base (s.flipReg i b) := by
  have ht := PreFault_to_TailSafe hp
  rcases ht with ⟨hlen, hts⟩
  rcases hp with ⟨hv, hodd, _, _, k, hpc, hcase⟩
  have hnv := hv.valNotAcc
  have hlen' : (s.flipReg i b).regs.length = 31 := by
    unfold AArch64Phys.St.flipReg
    split <;> first | exact hlen | exact setReg_len hlen i _
  have hreg_ne : ∀ {r}, r ≠ i → (s.flipReg i b).reg r = s.reg r := by
    intro r hne
    unfold AArch64Phys.St.flipReg
    split
    · rfl
    · exact reg_set_ne hne.symm _
  have hreg_eq : ∀ {r}, i = r → r < 31 → ∀ u, s.reg r = some u →
      (s.flipReg i b).reg r = some (Nat.xor u (2 ^ b)) := by
    intro r hir hr u hu
    subst hir
    unfold AArch64Phys.St.flipReg
    rw [if_neg (by simp; omega)]
    rw [reg_set_eq hlen hr _, hu]; rfl
  refine ⟨hlen', ?_⟩
  have hpc' : (s.flipReg i b).pc = s.pc := by unfold AArch64Phys.St.flipReg; split <;> rfl
  have hfl' : (s.flipReg i b).fl = s.fl := by unfold AArch64Phys.St.flipReg; split <;> rfl
  rw [hpc']
  split
  · next hbd =>
    have hk_eq : (s.pc - base) / 4 = k := by rw [hpc]; omega
    rw [hk_eq]
    have hxor := SafeInvalid_xor_bit hv hbit
    have hxor_lt : Nat.xor v (2 ^ b) < AArch64Env.W := hxor.1
    rcases hcase with ⟨hk8, h19, h20, h21, hfl⟩ | ⟨hk11, hk20, _, h20, _, _, _, _, _⟩ |
      ⟨hk32, hk48, _, _, _, _, _, _, _, h39_42, h40_2, h44_47, h45_2, _, _, _⟩
    · have ha19 : ∃ a, (s.flipReg i b).reg 19 = some a ∧ valNotAcc a := by
        by_cases h19i : i = 19
        · exact ⟨Nat.xor v (2 ^ b), hreg_eq h19i (by decide) v h19, hxor⟩
        · exact ⟨v, (hreg_ne (Ne.symm h19i)).trans h19, hnv⟩
      have ha20 : ∃ a, (s.flipReg i b).reg 20 = some a ∧ valNotAcc a := by
        by_cases h20i : i = 20
        · exact ⟨Nat.xor v (2 ^ b), hreg_eq h20i (by decide) v h20, hxor⟩
        · exact ⟨v, (hreg_ne (Ne.symm h20i)).trans h20, hnv⟩
      have ha21 : ∃ a, (s.flipReg i b).reg 21 = some a ∧ valNotAcc a := by
        by_cases h21i : i = 21
        · exact ⟨Nat.xor v (2 ^ b), hreg_eq h21i (by decide) v h21, hxor⟩
        · exact ⟨v, (hreg_ne (Ne.symm h21i)).trans h21, hnv⟩
      rcases ha19 with ⟨a19, hr19, hva19⟩
      rcases ha20 with ⟨a20, hr20, hva20⟩
      rcases ha21 with ⟨a21, hr21, hva21⟩
      have : k = 0 ∨ k = 1 ∨ k = 2 ∨ k = 3 ∨ k = 4 ∨ k = 5 ∨ k = 6 ∨ k = 7 := by omega
      rcases this with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl
      · by_cases h19i : i = 19
        · exact ⟨Nat.xor v (2 ^ b), v, v, hreg_eq h19i (by decide) v h19,
            (hreg_ne (by omega)).trans h20, (hreg_ne (by omega)).trans h21, hxor_lt, hv.1, hv.1,
            Or.inr ⟨v, by unfold TMR.vote; simp, hnv⟩⟩
        · by_cases h20i : i = 20
          · exact ⟨v, Nat.xor v (2 ^ b), v, (hreg_ne (by omega)).trans h19,
              hreg_eq h20i (by decide) v h20, (hreg_ne (by omega)).trans h21, hv.1, hxor_lt, hv.1,
              Or.inr ⟨v, by unfold TMR.vote; simp, hnv⟩⟩
          · by_cases h21i : i = 21
            · exact ⟨v, v, Nat.xor v (2 ^ b), (hreg_ne (by omega)).trans h19,
                (hreg_ne (by omega)).trans h20, hreg_eq h21i (by decide) v h21, hv.1, hv.1, hxor_lt,
                Or.inr ⟨v, by unfold TMR.vote; simp, hnv⟩⟩
            · exact ⟨v, v, v, (hreg_ne (Ne.symm h19i)).trans h19, (hreg_ne (Ne.symm h20i)).trans h20,
                (hreg_ne (Ne.symm h21i)).trans h21, hv.1, hv.1, hv.1, Or.inr ⟨v, by unfold TMR.vote; simp, hnv⟩⟩
      · exact Or.inr (Or.inl ⟨by rw [hfl']; exact hfl ⟨by decide, by decide⟩, a19, a20, a21, hr19, hr20, hr21, hva19, hva20, hva21⟩)
      · exact Or.inr (Or.inl ⟨by rw [hfl']; exact hfl ⟨by decide, by decide⟩, a19, a20, a21, hr19, hr20, hr21, hva19, hva20, hva21⟩)
      · exact Or.inr (Or.inl ⟨by rw [hfl']; exact hfl ⟨by decide, by decide⟩, a19, a20, a21, hr19, hr20, hr21, hva19, hva20, hva21⟩)
      · exact Or.inr (Or.inl ⟨by rw [hfl']; exact hfl ⟨by decide, by decide⟩, a19, hr19, hva19⟩)
      · exact ⟨a19, hr19, hva19⟩
      · exact ⟨a19, a20, hr19, hr20, hva19, hva20⟩
      · exact ⟨a19, a20, hr19, hr20, hva19, hva20⟩
    · have h20' : ∃ v20, (s.flipReg i b).reg 20 = some v20 ∧ valNotAcc v20 := by
        by_cases h20i : i = 20
        · exact ⟨Nat.xor v (2 ^ b), hreg_eq h20i (by decide) v h20, hxor⟩
        · exact ⟨v, (hreg_ne (Ne.symm h20i)).trans h20, hnv⟩
      have : k = 11 ∨ k = 12 ∨ k = 13 ∨ k = 14 ∨ k = 15 ∨ k = 16 ∨ k = 17 ∨ k = 18 ∨ k = 19 ∨ k = 20 := by omega
      rcases this with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl <;>
        first | exact Or.inl h20' | exact Or.inr h20'
    · have h6ok : 39 ≤ k → k ≤ 42 → ∃ v0, (s.flipReg i b).reg 0 = some v0 ∧ v0 % 256 ≠ 0 := by
        intro hk39 hk42
        by_cases h0i : i = 0
        · exact ⟨Nat.xor 6 (2 ^ b), hreg_eq h0i (by decide) 6 (h39_42 ⟨hk39, hk42⟩), xor_6_mod256_ne0 hbit⟩
        · exact ⟨6, (hreg_ne (Ne.symm h0i)).trans (h39_42 ⟨hk39, hk42⟩), by decide⟩
      have h5ok : 44 ≤ k → k ≤ 47 → ∃ v0, (s.flipReg i b).reg 0 = some v0 ∧ v0 % 256 ≠ 0 := by
        intro hk44 hk47
        by_cases h0i : i = 0
        · exact ⟨Nat.xor 5 (2 ^ b), hreg_eq h0i (by decide) 5 (h44_47 ⟨hk44, hk47⟩), xor_5_mod256_ne0 hbit⟩
        · exact ⟨5, (hreg_ne (Ne.symm h0i)).trans (h44_47 ⟨hk44, hk47⟩), by decide⟩
      have : k = 32 ∨ k = 33 ∨ k = 34 ∨ k = 35 ∨ k = 36 ∨ k = 37 ∨ k = 38 ∨ k = 39 ∨
             k = 40 ∨ k = 41 ∨ k = 42 ∨ k = 43 ∨ k = 44 ∨ k = 45 ∨ k = 46 ∨ k = 47 := by omega
      rcases this with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl
      · trivial
      · trivial
      · trivial
      · trivial
      · trivial
      · trivial
      · trivial
      · trivial
      · by_cases h0i : i = 0
        · exact Or.inr fun _ _ => ⟨6, (hreg_ne (by omega)).trans (h40_2 rfl), by decide⟩
        · exact Or.inl (h6ok (by decide) (by decide))
      · exact h6ok (by decide) (by decide)
      · exact fun _ => h6ok (by decide) (by decide)
      · trivial
      · trivial
      · by_cases h0i : i = 0
        · exact Or.inr fun _ _ => ⟨5, (hreg_ne (by omega)).trans (h45_2 rfl), by decide⟩
        · exact Or.inl (h5ok (by decide) (by decide))
      · exact h5ok (by decide) (by decide)
      · exact fun _ => h5ok (by decide) (by decide)
  · trivial

private theorem TailSafe_with_mem {base : Nat} {s : St} (ht : TailSafe base s) (m' : AArch64Env.Mem) :
    TailSafe base { s with mem := m' } := by
  rcases ht with ⟨hlen, hts⟩
  refine ⟨hlen, ?_⟩
  split
  · next hbd =>
    rw [dif_pos hbd] at hts
    exact (tailSafeAt_congr rfl rfl rfl rfl rfl rfl rfl rfl _).2 hts
  · trivial

private theorem PreFault_flipNzcv_safe {base v : Nat} {s s' : St}
    (hp : PreFault base v s) (hs' : s' ∈ nzcvFlips s) : TailSafe base s' := by
  have hmem : ∀ f, s' = { s with fl := f } → s'.regs = s.regs ∧ s'.pc = s.pc := by
    intro f hf; rw [hf]; exact ⟨rfl, rfl⟩
  have hfl_case : ∃ f, s' = { s with fl := f } := by
    unfold AArch64Phys.nzcvFlips at hs'
    simp only [List.mem_append, List.mem_map] at hs'
    rcases hs' with ((⟨_, _, rfl⟩ | ⟨_, _, rfl⟩) | ⟨_, _, rfl⟩) | ⟨_, _, rfl⟩ <;> exact ⟨_, rfl⟩
  rcases hfl_case with ⟨f, rfl⟩
  rcases hp with ⟨hv, _, hlen, _, k, hpc, hcase⟩
  have hnv := hv.valNotAcc
  refine ⟨hlen, ?_⟩
  split
  · next hbd =>
    have hk_eq : (s.pc - base) / 4 = k := by rw [hpc]; omega
    rw [hk_eq]
    rcases hcase with ⟨hk8, h19, h20, h21, _⟩ | ⟨hk11, hk20, _, h20, _, _, _, _, _, _, _⟩ |
      ⟨hk32, _, _, _, _, _, _, _, _, h39_42, _, h44_47, _, _, _, _⟩
    · have : k = 0 ∨ k = 1 ∨ k = 2 ∨ k = 3 ∨ k = 4 ∨ k = 5 ∨ k = 6 ∨ k = 7 := by omega
      rcases this with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl
      · exact ⟨v, v, v, h19, h20, h21, hv.1, hv.1, hv.1, Or.inr ⟨v, by unfold TMR.vote; simp, hnv⟩⟩
      · exact Or.inl ⟨v, h19, h20, h21, hnv⟩
      · exact Or.inl ⟨v, h19, h20, h21, hnv⟩
      · exact Or.inl ⟨v, h19, h20, h21, hnv⟩
      · exact Or.inl ⟨v, h19, h20, h21, hnv⟩
      · exact ⟨v, h19, hnv⟩
      · exact ⟨v, v, h19, h20, hnv, hnv⟩
      · exact ⟨v, v, h19, h20, hnv, hnv⟩
    · have : k = 11 ∨ k = 12 ∨ k = 13 ∨ k = 14 ∨ k = 15 ∨ k = 16 ∨ k = 17 ∨ k = 18 ∨ k = 19 ∨ k = 20 := by omega
      rcases this with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl <;>
        first | exact Or.inl ⟨v, h20, hnv⟩ | exact Or.inr ⟨v, h20, hnv⟩
    · have : k = 32 ∨ k = 33 ∨ k = 34 ∨ k = 35 ∨ k = 36 ∨ k = 37 ∨ k = 38 ∨ k = 39 ∨
             k = 40 ∨ k = 41 ∨ k = 42 ∨ k = 43 ∨ k = 44 ∨ k = 45 ∨ k = 46 ∨ k = 47 := by omega
      rcases this with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl <;>
        first | trivial |
          exact Or.inl ⟨6, h39_42 ⟨by decide, by decide⟩, by decide⟩ |
          exact ⟨6, h39_42 ⟨by decide, by decide⟩, by decide⟩ |
          exact fun _ => ⟨6, h39_42 ⟨by decide, by decide⟩, by decide⟩ |
          exact Or.inl ⟨5, h44_47 ⟨by decide, by decide⟩, by decide⟩ |
          exact ⟨5, h44_47 ⟨by decide, by decide⟩, by decide⟩ |
          exact fun _ => ⟨5, h44_47 ⟨by decide, by decide⟩, by decide⟩
  · trivial

/-- Elementary single-fault step transition (`clean`, `fetch`, `reg`, `mem`, `nzcv`). -/
inductive ElemFaultStep (alts : Alts) (im : Img) (s : St) : Out → Prop
  | clean : ElemFaultStep alts im s (step im s)
  | fetch (ins : Ins) (len : Nat) (wAlt : Option (Ins × Nat)) :
      im s.pc = some (ins, len) → wAlt ∈ alts s.pc →
      ElemFaultStep alts im s (execO s wAlt)
  | reg (s1 : St) (i b : Nat) :
      step im s = .ok s1 → i < 32 → b < 64 →
      ElemFaultStep alts im s (.ok (s1.flipReg i b))
  | mem (s1 : St) (i j b : Nat) :
      step im s = .ok s1 →
      ElemFaultStep alts im s (.ok (s1.flipMem i j b))
  | nzcv (s1 s2 : St) :
      step im s = .ok s1 → s2 ∈ nzcvFlips s1 →
      ElemFaultStep alts im s (.ok s2)

private def OutSafe (base : Nat) : Out → Prop
  | .ok s' => TailSafe base s'
  | .exit c => c ≠ 0
  | .stuck => True

private theorem OutSafe_to_match {base : Nat} {o : Out} (h : OutSafe base o) :
    match o with
    | .ok s' => TailSafe base s'
    | .exit c => c ≠ 0
    | .stuck => True := by
  cases o <;> exact h

private theorem match_to_OutSafe {base : Nat} {o : Out}
    (h : match o with | .ok s' => TailSafe base s' | .exit c => c ≠ 0 | .stuck => True) :
    OutSafe base o := by
  cases o <;> exact h

private theorem PreFault_elemFaultStep_safe_aux {base v : Nat} (hb : base_ok base) {s : St}
    (hp : PreFault base v s) {o' : Out}
    (hst : ElemFaultStep (tailAltsAt base) (tailImAt base) s o') :
    OutSafe base o' := by
  cases hst with
  | clean =>
    have hps := PreFault_step hb hp
    cases hr : step (tailImAt base) s with
    | ok s' => rw [hr] at hps; exact PreFault_to_TailSafe hps
    | exit c => rw [hr] at hps; exact hps
    | stuck => trivial
  | reg s1 i b hstep hi hbit =>
    have hps := PreFault_step hb hp
    rw [hstep] at hps
    exact PreFault_flipReg_safe hb hps hi hbit
  | mem s1 i j b hstep =>
    have hps := PreFault_step hb hp
    rw [hstep] at hps
    exact TailSafe_with_mem (PreFault_to_TailSafe hps) _
  | nzcv s1 s2 hstep hs2 =>
    have hps := PreFault_step hb hp
    rw [hstep] at hps
    exact PreFault_flipNzcv_safe hps hs2
  | fetch ins len wAlt him hw =>
    rcases hp with ⟨hv, hodd, hlen, hscr, k, hpc, hcase⟩
    have hpre : PreFault base v s := ⟨hv, hodd, hlen, hscr, k, hpc, hcase⟩
    have hkin : k ∈ preFaultIndices := by
      rcases hcase with ⟨hk8, _⟩ | ⟨hk11, hk20, _⟩ | ⟨hk32, hk48, _⟩ <;>
        (simp only [preFaultIndices, List.mem_cons, List.not_mem_nil, or_false]; omega)
    have hk48 : k < 48 := by
      rcases hcase with ⟨hk8, _⟩ | ⟨_, hk20, _⟩ | ⟨_, hk48, _⟩ <;> omega
    rw [hpc, tailAltsAt_word hb.1 hk48] at hw
    have hwlist : wAlt ∈ altsWord ((tailWords[k]?).getD 0) := by
      have hlen_tw : k < tailWords.length := by rw [tailWords_length]; exact hk48
      simpa [List.getElem?_eq_getElem hlen_tw] using hw
    unfold altsWord at hwlist
    simp only [List.mem_map, List.mem_range] at hwlist
    rcases hwlist with ⟨b, hb32, rfl⟩
    cases hd : decodeWord (Nat.xor ((tailWords[k]?).getD 0) (2 ^ b)) with
    | none => simp [execO, OutSafe]
    | some ins' =>
      have haudit := List.all_eq_true.1 (List.all_eq_true.1 tailFetchSafe_audit k hkin) b (List.mem_range.2 hb32)
      simp only [hd] at haudit
      have hsnd := match_to_OutSafe (tailAltWordSafe_sound hb hk48 hpre hpc haudit)
      simpa [execO, hd] using hsnd

theorem PreFault_elemFaultStep_safe {base v : Nat} (hb : base_ok base) {s : St}
    (hp : PreFault base v s) {o' : Out}
    (hst : ElemFaultStep (tailAltsAt base) (tailImAt base) s o') :
    match o' with
    | .ok s' => TailSafe base s'
    | .exit c => c ≠ 0
    | .stuck => True := by
  have h := PreFault_elemFaultStep_safe_aux hb hp hst
  revert hst h
  cases o' <;> intro _ h <;> exact h

/-- One-fault run: either fault-free `run`, or a single `ElemFaultStep` at step `i`
followed by fault-free `run` for the remaining `n - 1 - i` steps. -/
inductive OneFaultRun (alts : Alts) (im : Img) : Nat → Out → Out → Prop
  | of_clean (n : Nat) (o : Out) : OneFaultRun alts im n o (run im n o)
  | of_fault (n i : Nat) (o : Out) (si : St) (oi : Out) :
      i < n → run im i o = .ok si → ElemFaultStep alts im si oi →
      OneFaultRun alts im n o (run im (n - 1 - i) oi)

theorem PreFault_OneFaultRun {base v : Nat} (hb : base_ok base) {s : St}
    (hp : PreFault base v s) {n : Nat} {o' : Out}
    (ho : OneFaultRun (tailAltsAt base) (tailImAt base) n (.ok s) o') :
    o' ≠ .exit 0 := by
  cases ho with
  | of_clean => exact TailSafe_run hb n (PreFault_to_TailSafe hp)
  | of_fault i _ si oi hi hri hstep =>
    have hpi : PreFault base v si := by
      clear hstep hi
      induction i generalizing s with
      | zero => simp [run] at hri; cases hri; exact hp
      | succ i ih =>
        simp only [run] at hri
        have hps := PreFault_step hb hp
        cases hs : step (tailImAt base) s with
        | ok s1 => rw [hs] at hps hri; exact ih hps hri
        | exit c => rw [hs] at hri; rw [run_exit] at hri; cases hri
        | stuck => rw [hs] at hri; rw [run_stuck] at hri; cases hri
    have hsaf := PreFault_elemFaultStep_safe hb hpi hstep
    cases oi with
    | ok s' => exact TailSafe_run hb (n - 1 - i) hsaf
    | exit c => simp [run_exit, hsaf]
    | stuck => simp [run_stuck]

private theorem zeroRate_ok : PDist.Rate 0 :=
  ⟨Rat.le_refl, show (0 : Rat) ≤ 1 from by decide⟩

def singleRates (r : Rates) (hr : r.ok) (ch : Fin 4) : { rc : Rates // rc.ok } :=
  match ch with
  | 0 => ⟨⟨r.fetch, 0, 0, 0⟩, hr.1, zeroRate_ok, zeroRate_ok, zeroRate_ok⟩
  | 1 => ⟨⟨0, r.reg, 0, 0⟩, zeroRate_ok, hr.2.1, zeroRate_ok, zeroRate_ok⟩
  | 2 => ⟨⟨0, 0, r.mem, 0⟩, zeroRate_ok, zeroRate_ok, hr.2.2.1, zeroRate_ok⟩
  | 3 => ⟨⟨0, 0, 0, r.nzcv⟩, zeroRate_ok, zeroRate_ok, zeroRate_ok, hr.2.2.2⟩

private def zeroRates : Rates := ⟨0, 0, 0, 0⟩
private theorem zeroRates_ok : zeroRates.ok :=
  ⟨zeroRate_ok, zeroRate_ok, zeroRate_ok, zeroRate_ok⟩

/-- Physical run over `AArch64Phys.pstep`: at step `fi`, channel `ch` has rate from `r`;
all other steps run with zero fault rates. -/
def prunWithFaultAt (alts : Alts) (r : Rates) (hr : r.ok) (ch : Fin 4) (im : Img)
    (fi : Option Nat) : Nat → Out → PDist.PDF Out
  | 0, o => PDist.pureF o
  | n + 1, .ok s =>
      let (stepPDF, nextFi) := match fi with
        | some 0 =>
            let sr := singleRates r hr ch
            (pstep alts (sr.1.state sr.2) sr.1.fetch im s, none)
        | some (k + 1) =>
            (pstep alts (zeroRates.state zeroRates_ok) 0 im s, some k)
        | none =>
            (pstep alts (zeroRates.state zeroRates_ok) 0 im s, none)
      PDist.bindF stepPDF (prunWithFaultAt alts r hr ch im nextFi n)
  | _ + 1, o => PDist.pureF o

private theorem prunWithFaultAt_ok (alts : Alts) (r : Rates) (hr : r.ok) (ch : Fin 4) (im : Img) :
    ∀ fi n o, (prunWithFaultAt alts r hr ch im fi n o).ok := by
  intro fi n
  induction n generalizing fi with
  | zero => intro o; exact PDist.pureF_ok o
  | succ n ih =>
    intro o
    cases o with
    | exit c => exact PDist.pureF_ok (Out.exit c)
    | stuck => exact PDist.pureF_ok Out.stuck
    | ok s =>
      cases fi with
      | none =>
        exact PDist.bindF_ok (AArch64Phys.pstep_ok alts (zeroRates.state zeroRates_ok) zeroRates_ok.1 im s)
          fun _ => ih none _
      | some fk =>
        cases fk with
        | zero =>
          let sr := singleRates r hr ch
          exact PDist.bindF_ok (AArch64Phys.pstep_ok alts (sr.1.state sr.2) sr.2.1 im s)
            fun _ => ih none _
        | succ fk' =>
          exact PDist.bindF_ok (AArch64Phys.pstep_ok alts (zeroRates.state zeroRates_ok) zeroRates_ok.1 im s)
            fun _ => ih (some fk') _

def tailPRunWithFaultAt (base : Nat) (r : Rates) (hr : r.ok) (ch : Fin 4)
    (fi n : Nat) (o : Out) : PDist.PDF Out :=
  prunWithFaultAt (tailAltsAt base) r hr ch (tailImAt base) (some fi) n o

private theorem pos_mem_tagW {β : Type} {t : Bool} {w : Rat} {e : PDist.PDF β}
    (hw : 0 ≤ w) (he : ∀ x ∈ e, 0 ≤ x.2) {y : (β × Bool) × Rat} (hy : y ∈ PDist.tagW t w e) (hpos : 0 < y.2) :
    0 < w ∧ ∃ y0 ∈ e, y = ((y0.1.1, t || y0.1.2), w * y0.2) ∧ 0 < y0.2 := by
  unfold PDist.tagW at hy
  simp only [List.mem_map] at hy
  rcases hy with ⟨y0, hy0, rfl⟩
  have hwne : w ≠ 0 := by
    intro h0; subst h0; rw [Rat.zero_mul] at hpos; exact Rat.lt_irrefl hpos
  have hwpos : 0 < w := Rat.lt_of_le_of_ne hw (Ne.symm hwne)
  have hy0ne : y0.2 ≠ 0 := by
    intro h0; rw [h0, Rat.mul_zero] at hpos; exact Rat.lt_irrefl hpos
  have hy0pos : 0 < y0.2 := Rat.lt_of_le_of_ne (he y0 hy0) (Ne.symm hy0ne)
  exact ⟨hwpos, y0, hy0, rfl, hy0pos⟩

private theorem pos_mem_bindF_nonneg {α β : Type} {d : PDist.PDF α}
    (hd : ∀ x ∈ d, 0 ≤ x.2) {k : α → PDist.PDF β} (hk : ∀ a, ∀ x ∈ k a, 0 ≤ x.2)
    {b : β} {tb_out : Bool} {wb_out : Rat}
    (hy : ((b, tb_out), wb_out) ∈ PDist.bindF d k) (hpos : 0 < wb_out) :
    ∃ a ta wa tb wb, ((a, ta), wa) ∈ d ∧ ((b, tb), wb) ∈ k a ∧ 0 < wa ∧ 0 < wb := by
  induction d with
  | nil => simp [PDist.bindF] at hy
  | cons x r ih =>
    obtain ⟨⟨a, ta⟩, wa⟩ := x
    simp only [PDist.bindF, List.mem_append] at hy
    rcases hy with hy1 | hy2
    · obtain ⟨_, y0, hy0, heq, hy0p⟩ := pos_mem_tagW (hd _ List.mem_cons_self) (hk a) hy1 hpos
      obtain ⟨⟨z, tb⟩, wb⟩ := y0
      have hz : b = z := by
        have h1 := congrArg (fun x => x.1.1) heq
        exact h1
      subst hz
      have hwa_ne : wa ≠ 0 := fun h0 => by
        have hwb_eq : wb_out = wa * wb := congrArg Prod.snd heq
        rw [hwb_eq, h0, Rat.zero_mul] at hpos; exact Rat.lt_irrefl hpos
      exact ⟨a, ta, wa, tb, wb, List.mem_cons_self, hy0,
        Rat.lt_of_le_of_ne (hd _ List.mem_cons_self) (Ne.symm hwa_ne), hy0p⟩
    · rcases ih (fun z hz => hd z (List.mem_cons_of_mem _ hz)) hy2 with
        ⟨a', ta', wa', tb', wb', hmem, hka, hwa', hwb'⟩
      exact ⟨a', ta', wa', tb', wb', List.mem_cons_of_mem _ hmem, hka, hwa', hwb'⟩

private theorem unif_mem_cases {β : Type} {p : Rat} (c : β) (xs : List β)
    {x : β} {t : Bool} {w : Rat} (hy : ((x, t), w) ∈ AArch64Phys.unif p c xs) :
    x = c ∨ x ∈ xs := by
  unfold AArch64Phys.unif at hy
  cases xs with
  | nil => simp [PDist.pureF] at hy; left; exact hy.1.1
  | cons z r =>
    simp only [List.mem_cons, List.mem_map] at hy
    rcases hy with hEq | ⟨u, hu, hzeq⟩
    · exact Or.inl (congrArg (fun p => p.1.1) hEq)
    · have hux : u = x := congrArg (fun p => p.1.1) hzeq
      exact Or.inr (hux ▸ List.mem_cons.2 hu)

private theorem unif_zero_pos {β : Type} (c : β) (xs : List β)
    {x : β} {t : Bool} {w : Rat} (hy : ((x, t), w) ∈ AArch64Phys.unif 0 c xs) (hpos : 0 < w) :
    x = c := by
  unfold AArch64Phys.unif at hy
  cases xs with
  | nil => simp [PDist.pureF] at hy; exact hy.1.1
  | cons z r =>
    simp only [List.mem_cons, List.mem_map] at hy
    rcases hy with hEq | ⟨u, _, hzeq⟩
    · exact congrArg (fun p => p.1.1) hEq
    · have hwe : w = (0 : Rat) / (((z :: r).length : Nat) : Rat) := (congrArg Prod.snd hzeq).symm
      rw [hwe, Rat.div_def, Rat.zero_mul] at hpos
      exact absurd hpos Rat.lt_irrefl

private theorem fault_uniform_hit (alt : St → List St) (p : Rat) (hp : PDist.Rate p) (s : St) :
    (AArch64Phys.Fault.uniform alt p hp).hit s = AArch64Phys.unif p s (alt s) := rfl

private theorem fault_seq3_unpack (Freg Fmem Fnz : AArch64Phys.Fault)
    {s1 : St} {o : Out} {tb : Bool} {wb : Rat}
    (hret : ((o, tb), wb) ∈ AArch64Phys.relabel Out.ok ((Freg.seq (Fmem.seq Fnz)).hit s1))
    (hpos : 0 < wb) :
    ∃ s_reg s_mem s_nzcv tr wr tm wm tn wn,
      o = .ok s_nzcv ∧
      ((s_reg, tr), wr) ∈ Freg.hit s1 ∧ 0 < wr ∧
      ((s_mem, tm), wm) ∈ Fmem.hit s_reg ∧ 0 < wm ∧
      ((s_nzcv, tn), wn) ∈ Fnz.hit s_mem ∧ 0 < wn := by
  simp only [AArch64Phys.relabel, List.mem_map] at hret
  rcases hret with ⟨⟨⟨s_f, tf⟩, wf⟩, hmem, heq⟩
  have heq_f : o = .ok s_f := (congrArg (fun x => x.1.1) heq).symm
  have hwf : wf = wb := congrArg Prod.snd heq
  subst heq_f; subst hwf
  have hmem' : ((s_f, tf), wf) ∈ PDist.bindF (Freg.hit s1) (fun sr => (Fmem.seq Fnz).hit sr) := hmem
  obtain ⟨s_reg, tr, wr, tb2, wb2, hrg, hseq2, hwr, hwb2⟩ :=
    pos_mem_bindF_nonneg (Freg.ok s1).1 (fun sr => ((Fmem.seq Fnz).ok sr).1) hmem' hpos
  have hseq2' : ((s_f, tb2), wb2) ∈ PDist.bindF (Fmem.hit s_reg) Fnz.hit := hseq2
  obtain ⟨s_mem, tm, wm, tn, wn, hmm, hnz, hwm, hwn⟩ :=
    pos_mem_bindF_nonneg (Fmem.ok s_reg).1 (fun sm => (Fnz.ok sm).1) hseq2' hwb2
  exact ⟨s_reg, s_mem, s_f, tr, wr, tm, wm, tn, wn, rfl, hrg, hwr, hmm, hwm, hnz, hwn⟩

private theorem pstep_retire_unpack {rc : Rates} (hrc : rc.ok) {s1 : St} {o : Out} {tb : Bool} {wb : Rat}
    (hret : ((o, tb), wb) ∈ AArch64Phys.relabel Out.ok ((rc.state hrc).hit s1)) (hpos : 0 < wb) :
    ∃ s_reg s_mem s_nzcv tr wr tm wm tn wn,
      o = .ok s_nzcv ∧
      ((s_reg, tr), wr) ∈ AArch64Phys.unif rc.reg s1 (regFlips s1) ∧ 0 < wr ∧
      ((s_mem, tm), wm) ∈ AArch64Phys.unif rc.mem s_reg (memFlips s_reg) ∧ 0 < wm ∧
      ((s_nzcv, tn), wn) ∈ AArch64Phys.unif rc.nzcv s_mem (nzcvFlips s_mem) ∧ 0 < wn := by
  have hret' : ((o, tb), wb) ∈ AArch64Phys.relabel Out.ok
      (((AArch64Phys.Fault.uniform regFlips rc.reg hrc.2.1).seq
        ((AArch64Phys.Fault.uniform memFlips rc.mem hrc.2.2.1).seq
          (AArch64Phys.Fault.uniform nzcvFlips rc.nzcv hrc.2.2.2))).hit s1) := hret
  obtain ⟨s_r, s_m, s_n, tr, wr, tm, wm, tn, wn, ho, hrg, hwr, hmm, hwm, hnz, hwn⟩ :=
    fault_seq3_unpack _ _ _ hret' hpos
  rw [fault_uniform_hit] at hrg hmm hnz
  exact ⟨s_r, s_m, s_n, tr, wr, tm, wm, tn, wn, ho, hrg, hwr, hmm, hwm, hnz, hwn⟩

private theorem pstep_zeroRates_pos {alts : Alts} {im : Img} {s : St}
    {y : (Out × Bool) × Rat} (hy : y ∈ pstep alts (zeroRates.state zeroRates_ok) 0 im s)
    (hpos : 0 < y.2) : y.1.1 = step im s := by
  unfold pstep at hy
  have hf_ok := AArch64Phys.fetched_ok zeroRate_ok alts im s
  obtain ⟨⟨o_y, t_y⟩, w_y⟩ := y
  obtain ⟨a, ta, wa, tb, wb, ha, hret, hwa, hwb⟩ :=
    pos_mem_bindF_nonneg hf_ok.1 (fun o => (AArch64Phys.retire_ok (zeroRates.state zeroRates_ok) o).1) hy hpos
  have ha_eq : a = step im s := by
    unfold step
    cases him : im s.pc with
    | none => simp [him, PDist.pureF] at ha; exact ha.1.1
    | some p =>
      obtain ⟨ins, len⟩ := p
      simp only [him] at ha
      exact unif_zero_pos _ _ ha hwa
  cases a with
  | exit c =>
    simp [PDist.pureF] at hret
    rw [← ha_eq]; exact hret.1.1
  | stuck =>
    simp [PDist.pureF] at hret
    rw [← ha_eq]; exact hret.1.1
  | ok s1 =>
    rcases pstep_retire_unpack zeroRates_ok hret hwb with
      ⟨s_r, s_m, s_n, _, wr, _, wm, _, wn, ho, hrg, hwr, hmm, hwm, hnz, hwn⟩
    have hr_eq : s_r = s1 := unif_zero_pos _ _ hrg hwr
    have hm_eq : s_m = s_r := unif_zero_pos _ _ hmm hwm
    have hn_eq : s_n = s_m := unif_zero_pos _ _ hnz hwn
    dsimp only at ho ⊢
    rw [ho, hn_eq, hm_eq, hr_eq, ← ha_eq]

private theorem pstep_singleRate_elem {alts : Alts} {r : Rates} (hr : r.ok) (ch : Fin 4)
    {im : Img} {s : St} {y : (Out × Bool) × Rat}
    (hy : y ∈ pstep alts ((singleRates r hr ch).1.state (singleRates r hr ch).2)
        (singleRates r hr ch).1.fetch im s)
    (hpos : 0 < y.2) : ElemFaultStep alts im s y.1.1 := by
  let sr := singleRates r hr ch
  unfold pstep at hy
  have hf_ok := AArch64Phys.fetched_ok sr.2.1 alts im s
  obtain ⟨⟨o_y, t_y⟩, w_y⟩ := y
  obtain ⟨a, ta, wa, tb, wb, ha, hret, hwa, hwb⟩ :=
    pos_mem_bindF_nonneg hf_ok.1 (fun o => (AArch64Phys.retire_ok (sr.1.state sr.2) o).1) hy hpos
  dsimp only at ⊢
  have hretire_clean : ch ≠ 0 → (∃ s1, a = .ok s1 ∧ step im s = .ok s1) ∨ o_y = step im s := by
    intro hne0
    have hft0 : sr.1.fetch = 0 := by
      rcases ch with ⟨_ | _ | _ | _ | _, hlt⟩
      · exact absurd rfl hne0
      · rfl
      · rfl
      · rfl
      · omega
    have ha_eq : a = step im s := by
      unfold step
      cases him : im s.pc with
      | none => simp [him, PDist.pureF] at ha; exact ha.1.1
      | some p =>
        obtain ⟨ins, len⟩ := p
        simp only [him, hft0] at ha
        exact unif_zero_pos _ _ ha hwa
    cases a with
    | exit c =>
      simp [PDist.pureF] at hret
      exact Or.inr (by rw [← ha_eq]; exact hret.1.1)
    | stuck =>
      simp [PDist.pureF] at hret
      exact Or.inr (by rw [← ha_eq]; exact hret.1.1)
    | ok s1 => exact Or.inl ⟨s1, rfl, ha_eq.symm⟩
  rcases ch with ⟨_ | _ | _ | _ | _, hlt⟩
  · -- ch = 0: fetch only
    have hfa : a = step im s ∨ ∃ ins len wAlt, im s.pc = some (ins, len) ∧ wAlt ∈ alts s.pc ∧ a = execO s wAlt := by
      unfold step
      cases him : im s.pc with
      | none => simp [him, PDist.pureF] at ha; exact Or.inl ha.1.1
      | some p =>
        obtain ⟨ins, len⟩ := p
        simp only [him] at ha
        rcases unif_mem_cases _ _ ha with heq | hmem
        · exact Or.inl heq
        · simp only [List.mem_map] at hmem
          rcases hmem with ⟨wAlt, hw, rfl⟩
          exact Or.inr ⟨ins, len, wAlt, rfl, hw, rfl⟩
    have hy_eq_a : o_y = a := by
      cases a with
      | exit c => simp [PDist.pureF] at hret; exact hret.1.1
      | stuck => simp [PDist.pureF] at hret; exact hret.1.1
      | ok s1 =>
        rcases pstep_retire_unpack sr.2 hret hwb with
          ⟨s_r, s_m, s_n, _, wr, _, wm, _, wn, ho, hrg, hwr, hmm, hwm, hnz, hwn⟩
        have hr_eq := unif_zero_pos s1 _ hrg hwr
        have hm_eq := unif_zero_pos s_r _ hmm hwm
        have hn_eq := unif_zero_pos s_m _ hnz hwn
        rw [ho, hn_eq, hm_eq, hr_eq]
    rw [hy_eq_a]
    rcases hfa with rfl | ⟨ins, len, wAlt, him, hw, rfl⟩
    · exact ElemFaultStep.clean
    · exact ElemFaultStep.fetch ins len wAlt him hw
  · -- ch = 1: reg only
    rcases hretire_clean (by simp) with ⟨s1, rfl, hstep⟩ | heq
    · rcases pstep_retire_unpack sr.2 hret hwb with
        ⟨s_r, s_m, s_n, _, wr, _, wm, _, wn, ho, hrg, hwr, hmm, hwm, hnz, hwn⟩
      have hm_eq := unif_zero_pos s_r _ hmm hwm
      have hn_eq := unif_zero_pos s_m _ hnz hwn
      rw [ho, hn_eq, hm_eq]
      rcases unif_mem_cases s1 _ hrg with rfl | hmem
      · rw [← hstep]; exact ElemFaultStep.clean
      · unfold AArch64Phys.regFlips at hmem
        simp only [List.mem_map, List.mem_range] at hmem
        rcases hmem with ⟨j, hj, rfl⟩
        exact ElemFaultStep.reg s1 (j / 64) (j % 64) hstep (by omega) (by omega)
    · rw [heq]; exact ElemFaultStep.clean
  · -- ch = 2: mem only
    rcases hretire_clean (by simp) with ⟨s1, rfl, hstep⟩ | heq
    · rcases pstep_retire_unpack sr.2 hret hwb with
        ⟨s_r, s_m, s_n, _, wr, _, wm, _, wn, ho, hrg, hwr, hmm, hwm, hnz, hwn⟩
      have hr_eq := unif_zero_pos s1 _ hrg hwr
      have hn_eq := unif_zero_pos s_m _ hnz hwn
      rw [ho, hn_eq]
      rcases unif_mem_cases s_r (memFlips s_r) hmm with heq1 | hmem
      · rw [heq1, hr_eq, ← hstep]; exact ElemFaultStep.clean
      · rw [hr_eq] at hmem
        unfold AArch64Phys.memFlips at hmem
        simp only [List.mem_flatMap, List.mem_map, List.mem_range] at hmem
        rcases hmem with ⟨i, _, j, _, b, _, rfl⟩
        exact ElemFaultStep.mem s1 i j b hstep
    · rw [heq]; exact ElemFaultStep.clean
  · -- ch = 3: nzcv only
    rcases hretire_clean (by simp) with ⟨s1, rfl, hstep⟩ | heq
    · rcases pstep_retire_unpack sr.2 hret hwb with
        ⟨s_r, s_m, s_n, _, wr, _, wm, _, wn, ho, hrg, hwr, hmm, hwm, hnz, hwn⟩
      have hr_eq := unif_zero_pos s1 _ hrg hwr
      have hm_eq := unif_zero_pos s_r _ hmm hwm
      rw [ho]
      rcases unif_mem_cases s_m (nzcvFlips s_m) hnz with heq1 | hmem
      · rw [heq1, hm_eq, hr_eq, ← hstep]; exact ElemFaultStep.clean
      · rw [hm_eq, hr_eq] at hmem
        exact ElemFaultStep.nzcv s1 s_n hstep hmem
    · rw [heq]; exact ElemFaultStep.clean
  · exfalso; omega

private theorem prunWithFault_none_pos {alts : Alts} {r : Rates} {hr : r.ok} {ch : Fin 4}
    {im : Img} : ∀ n o {y : (Out × Bool) × Rat},
    y ∈ prunWithFaultAt alts r hr ch im none n o → 0 < y.2 → y.1.1 = run im n o := by
  intro n
  induction n with
  | zero => intro o y hy _; simp [prunWithFaultAt, PDist.pureF] at hy; rw [hy, run]
  | succ n ih =>
    intro o y hy hpos
    cases o with
    | exit c => simp [prunWithFaultAt, PDist.pureF] at hy; rw [hy, run_exit]
    | stuck => simp [prunWithFaultAt, PDist.pureF] at hy; rw [hy, run_stuck]
    | ok s =>
      simp only [prunWithFaultAt] at hy
      have hp_ok := AArch64Phys.pstep_ok alts (zeroRates.state zeroRates_ok) zeroRates_ok.1 im s
      have hk_ok : ∀ a, ∀ x ∈ prunWithFaultAt alts r hr ch im none n a, 0 ≤ x.2 :=
        fun a => (prunWithFaultAt_ok alts r hr ch im none n a).1
      obtain ⟨a, ta, wa, tb, wb, ha, hrest, hwa, hwb⟩ := pos_mem_bindF_nonneg hp_ok.1 hk_ok hy hpos
      have ha_eq := pstep_zeroRates_pos ha hwa
      change a = step im s at ha_eq
      have hrest_eq := ih a hrest hwb
      change y.1.1 = run im n a at hrest_eq
      rw [hrest_eq, ha_eq]; rfl

private theorem prunWithFault_some_oneFault {alts : Alts} {r : Rates} {hr : r.ok} {ch : Fin 4}
    {im : Img} : ∀ n fi o {y : (Out × Bool) × Rat},
    y ∈ prunWithFaultAt alts r hr ch im (some fi) n o → 0 < y.2 → OneFaultRun alts im n o y.1.1 := by
  intro n
  induction n with
  | zero =>
    intro fi o y hy _; simp [prunWithFaultAt, PDist.pureF] at hy
    rw [hy]; exact OneFaultRun.of_clean 0 o
  | succ n ih =>
    intro fi o y hy hpos
    cases o with
    | exit c =>
      simp [prunWithFaultAt, PDist.pureF] at hy
      rw [hy, ← run_exit (im := im) c (n + 1)]; exact OneFaultRun.of_clean (n + 1) (.exit c)
    | stuck =>
      simp [prunWithFaultAt, PDist.pureF] at hy
      rw [hy, ← run_stuck (im := im) (n + 1)]; exact OneFaultRun.of_clean (n + 1) .stuck
    | ok s =>
      cases fi with
      | zero =>
        simp only [prunWithFaultAt] at hy
        let sr := singleRates r hr ch
        have hp_ok := AArch64Phys.pstep_ok alts (sr.1.state sr.2) sr.2.1 im s
        have hk_ok : ∀ a, ∀ x ∈ prunWithFaultAt alts r hr ch im none n a, 0 ≤ x.2 :=
          fun a => (prunWithFaultAt_ok alts r hr ch im none n a).1
        obtain ⟨a, ta, wa, tb, wb, ha, hrest, hwa, hwb⟩ := pos_mem_bindF_nonneg hp_ok.1 hk_ok hy hpos
        have hel : ElemFaultStep alts im s a := pstep_singleRate_elem hr ch ha hwa
        have hrun := prunWithFault_none_pos n a hrest hwb
        change y.1.1 = run im n a at hrun
        rw [hrun]
        simpa using OneFaultRun.of_fault (n + 1) 0 (.ok s) s a (by omega) rfl hel
      | succ fk =>
        simp only [prunWithFaultAt] at hy
        have hp_ok := AArch64Phys.pstep_ok alts (zeroRates.state zeroRates_ok) zeroRates_ok.1 im s
        have hk_ok : ∀ a, ∀ x ∈ prunWithFaultAt alts r hr ch im (some fk) n a, 0 ≤ x.2 :=
          fun a => (prunWithFaultAt_ok alts r hr ch im (some fk) n a).1
        obtain ⟨a, ta, wa, tb, wb, ha, hrest, hwa, hwb⟩ := pos_mem_bindF_nonneg hp_ok.1 hk_ok hy hpos
        have ha_eq := pstep_zeroRates_pos ha hwa
        change a = step im s at ha_eq
        have hsub := ih fk a hrest hwb
        change OneFaultRun alts im n a y.1.1 at hsub
        rw [ha_eq] at hsub
        generalize hyo : y.1.1 = res at hsub ⊢
        cases hsub with
        | of_clean _ => exact OneFaultRun.of_clean (n + 1) (.ok s)
        | of_fault i _ si oi hi hri hel =>
          have hfin := OneFaultRun.of_fault (n + 1) (i + 1) (.ok s) si oi
            (by omega) (by simpa [run] using hri) hel
          rwa [show n + 1 - 1 - (i + 1) = n - 1 - i by omega] at hfin

theorem tailPRunWithFaultAt_sound {base v : Nat} (hb : base_ok base) {s : St}
    (hp : PreFault base v s) (r : Rates) (hr : r.ok) (ch : Fin 4) (fi n : Nat)
    {y : (Out × Bool) × Rat} (hy : y ∈ tailPRunWithFaultAt base r hr ch fi n (.ok s))
    (hpos : 0 < y.2) : y.1.1 ≠ .exit 0 :=
  PreFault_OneFaultRun hb hp (prunWithFault_some_oneFault n fi (.ok s) hy hpos)

/-- If a verifier pass word satisfies `SafeInvalid` and odd parity, NO zero- or
single-fault trajectory in `[vote, core_end)` (`fetch`, `reg`, `mem`, or `nzcv`)
can return `.exit 0`. -/
theorem invalidWord_threePass_no_single_fault_false_accept {base v : Nat} (hb : base_ok base)
    (s0 : St) (hscr : VoterScratchClean s0) (hv : SafeInvalid v) (hodd : v % 2 = 1) :
    (∀ n o', OneFaultRun (tailAltsAt base) (tailImAt base) n (.ok (tailInit base s0 v v v)) o' →
        o' ≠ .exit 0) ∧
    (∀ (r : Rates) (hr : r.ok) (ch : Fin 4) (fi n : Nat) (y : (Out × Bool) × Rat),
        y ∈ tailPRunWithFaultAt base r hr ch fi n (.ok (tailInit base s0 v v v)) →
        0 < y.2 → y.1.1 ≠ .exit 0) :=
  ⟨fun _ _ ho => PreFault_OneFaultRun hb (PreFault_init hb s0 hscr hv hodd) ho,
   fun r hr ch fi n _ hy hpos => tailPRunWithFaultAt_sound hb (PreFault_init hb s0 hscr hv hodd) r hr ch fi n hy hpos⟩

end TailProofs


/-! ### Ground `decide +kernel` witnesses for valid, overflow, and refused `SPR1` images -/

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

theorem checkFlatWord_lt_W (bs : List Nat) : checkFlatWord bs < AArch64Env.W := by
  unfold checkFlatWord
  cases checkFlat bs with
  | some b => cases b <;> decide
  | none => decide

theorem checkFlatWord_safeInvalid {bs : List Nat} (hne : checkFlat bs ≠ some true) :
    SafeInvalid (checkFlatWord bs) ∧ checkFlatWord bs % 2 = 1 := by
  unfold checkFlatWord
  cases hcf : checkFlat bs with
  | some b =>
    cases b
    · exact ⟨safeInvalid_REJECT, by decide⟩
    · exact absurd hcf hne
  | none => exact ⟨safeInvalid_OVERFLOW, by decide⟩

theorem checkFlatWord_accept_iff (bs : List Nat) :
    checkFlatWord bs = ACCEPT ↔ checkFlat bs = some true := by
  unfold checkFlatWord
  cases hcf : checkFlat bs with
  | some b => cases b <;> simp [show REJECT ≠ ACCEPT by decide]
  | none => simp [show OVERFLOW ≠ ACCEPT by decide]

/-- End-to-end flat theorem connecting `Sporos.decode` + `Sporos.checkProg`
with 3-pass `checkFlatWord` -> 192-byte AArch64 `tailImAt` (architectural AND
single-fault 4-channel physical immunity against false `exit 0`). -/
theorem threePass_no_single_fault_false_accept {base : Nat} (hb : base_ok base)
    (s0 : AArch64Phys.St) (hscr : VoterScratchClean s0)
    (bs : List Nat) (hne : ∀ p, Sporos.decode bs = some p → Sporos.checkProg p ≠ true) :
    (∀ n o', OneFaultRun (tailAltsAt base) (tailImAt base) n
        (.ok (tailInit base s0 (checkFlatWord bs) (checkFlatWord bs) (checkFlatWord bs))) o' →
        o' ≠ .exit 0) ∧
    (∀ (r : AArch64Phys.Rates) (hr : r.ok) (ch : Fin 4) (fi n : Nat) (y : (AArch64Phys.Out × Bool) × Rat),
        y ∈ tailPRunWithFaultAt base r hr ch fi n
          (.ok (tailInit base s0 (checkFlatWord bs) (checkFlatWord bs) (checkFlatWord bs))) →
        0 < y.2 → y.1.1 ≠ .exit 0) := by
  have hcf : checkFlat bs ≠ some true := by
    intro ht
    obtain ⟨p, hd, hok⟩ := checkFlat_sound ht
    exact hne p hd hok
  have ⟨hsafe, hodd⟩ := checkFlatWord_safeInvalid hcf
  exact invalidWord_threePass_no_single_fault_false_accept hb s0 hscr hsafe hodd

end SporosCheckArm64Flat
