/-
SporosCheckBinSpec: the streaming checkFlat algorithm (`checkFlatA`) and all
loop specification functions (`skipDataA`, `skipCodeA`, `skipWordsA`, `lookupA`,
`EffTree`, `hstepA`, `walkA`, `annsOKA`, `checkWordA`, `checkWordsA`), plus
bounds lemmas and exact correspondence with `effT` and `Sporos.Prim.eff`.
-/
import Bootstrap.SporosCheckBinExec
import Sporos.Bytecode

namespace SporosCheckBin

open X86Int

def HMAX : Nat := 2 ^ 48

/-! ### Skipping data, code, and words -/

/-- skipData as the listing does it: `n` items `len:u32 byte*` from `pos`. -/
def skipDataA (bs : List Nat) : Nat → Nat → Option Nat
  | 0, pos => some pos
  | n + 1, pos =>
    if bs.length < pos + 4 then none
    else if bs.length - (pos + 4) < u32 bs pos then none
    else skipDataA bs n (pos + 4 + u32 bs pos)

theorem skipDataA_le {bs : List Nat} : ∀ {n pos pos'}, pos ≤ bs.length → skipDataA bs n pos = some pos' →
    pos' ≤ bs.length
  | 0, pos, pos', hp, h => by simp [skipDataA] at h; omega
  | n + 1, pos, pos', hp, h => by
    simp only [skipDataA] at h
    split at h
    · simp at h
    · split at h
      · simp at h
      · exact skipDataA_le (by omega) h

/-- Length of the instruction with opcode `op`, as skipCode classifies it. -/
def instrLenA (op : Nat) : Option Nat :=
  if op < 57 then some 1
  else if op < 0x80 then none
  else if op = 0x80 then some 9
  else if 0x8c < op then none
  else if op = 0x8c then some 1
  else if op < 0x83 then some 5
  else if op < 0x86 then some 1
  else some 5

theorem instrLenA_pos {op l : Nat} (h : instrLenA op = some l) : 1 ≤ l ∧ l ≤ 9 := by
  unfold instrLenA at h
  repeat' (split at h)
  all_goals first | contradiction | (simp only [Option.some.injEq] at h; rw [← h]; decide)

/-- skipCode: `n` instructions from `pos`; `none` on a bad opcode or a
truncated instruction. -/
def skipCodeA (bs : List Nat) : Nat → Nat → Option Nat
  | 0, pos => some pos
  | n + 1, pos =>
    if h : pos < bs.length then
      match instrLenA bs[pos] with
      | none => none
      | some l => if bs.length < pos + l then none else skipCodeA bs n (pos + l)
    else none

theorem skipCodeA_bounds {bs : List Nat} : ∀ {n pos pos'}, pos ≤ bs.length → skipCodeA bs n pos = some pos' →
    pos ≤ pos' ∧ pos' ≤ bs.length
  | 0, pos, pos', hp, h => by simp [skipCodeA] at h; omega
  | n + 1, pos, pos', hp, h => by
    simp only [skipCodeA] at h
    split at h
    · cases hl : instrLenA bs[pos] with
      | none => simp [hl] at h
      | some l =>
        simp only [hl] at h
        split at h
        · simp at h
        · have ⟨h1, _⟩ := instrLenA_pos hl
          have ⟨ih1, ih2⟩ := skipCodeA_bounds (by omega) h
          omega
    · simp at h

theorem skipCodeA_le {bs : List Nat} {n pos pos' : Nat} (hp : pos ≤ bs.length)
    (h : skipCodeA bs n pos = some pos') : pos' ≤ bs.length :=
  (skipCodeA_bounds hp h).2

/-- skipWords: `n` words (20-byte header, code, 12-byte annotations) from `pos`. -/
def skipWordsA (bs : List Nat) : Nat → Nat → Option Nat
  | 0, pos => some pos
  | n + 1, pos =>
    if bs.length < pos + 20 then none
    else match skipCodeA bs (u32 bs (pos + 12)) (pos + 20) with
      | none => none
      | some p =>
        if bs.length < p + 12 * u32 bs (pos + 16) then none
        else skipWordsA bs n (p + 12 * u32 bs (pos + 16))

theorem skipWordsA_bounds {bs : List Nat} : ∀ {n pos pos'}, pos ≤ bs.length → skipWordsA bs n pos = some pos' →
    pos ≤ pos' ∧ pos' ≤ bs.length
  | 0, pos, pos', hp, h => by simp [skipWordsA] at h; omega
  | n + 1, pos, pos', hp, h => by
    simp only [skipWordsA] at h
    split at h
    · simp at h
    · cases hc : skipCodeA bs (u32 bs (pos + 12)) (pos + 20) with
      | none => simp [hc] at h
      | some p =>
        simp only [hc] at h
        split at h
        · simp at h
        · have hcb := skipCodeA_bounds (by omega) hc
          have ⟨ih1, ih2⟩ := skipWordsA_bounds (by omega) h
          omega

theorem skipWordsA_le {bs : List Nat} {n pos pos' : Nat} (hp : pos ≤ bs.length)
    (h : skipWordsA bs n pos = some pos') : pos' ≤ bs.length :=
  (skipWordsA_bounds hp h).2

theorem skipWordsA_add {bs : List Nat} : ∀ (m n pos : Nat),
    skipWordsA bs (m + n) pos = (skipWordsA bs m pos).bind (skipWordsA bs n)
  | 0, n, pos => by simp [skipWordsA]
  | m + 1, n, pos => by
    rw [Nat.add_right_comm]
    simp only [skipWordsA]
    split
    · rfl
    · cases skipCodeA bs (u32 bs (pos + 12)) (pos + 20) with
      | none => rfl
      | some p =>
        simp only
        split
        · rfl
        · exact skipWordsA_add m n _

theorem skipWordsA_lt_step {bs : List Nat} {w nw pos endPos : Nat}
    (hw : w < nw) (hs : skipWordsA bs nw pos = some endPos) :
    ∃ cw, skipWordsA bs w pos = some cw ∧
      cw + 20 ≤ endPos ∧
      ∃ annOff, skipCodeA bs (u32 bs (cw + 12)) (cw + 20) = some annOff ∧
        cw + 20 ≤ annOff ∧ annOff + 12 * u32 bs (cw + 16) ≤ endPos := by
  have heq : nw = w + (1 + (nw - w - 1)) := by omega
  rw [heq, skipWordsA_add] at hs
  cases h1 : skipWordsA bs w pos with
  | none => simp [h1] at hs
  | some cw =>
    simp only [h1, Option.bind_some, skipWordsA_add] at hs
    cases h2 : skipWordsA bs 1 cw with
    | none => simp [h2] at hs
    | some cw1 =>
      simp only [h2, Option.bind_some] at hs
      have hcw1 : cw1 ≤ endPos := by
        -- if cw ≤ bs.length? Actually skipWordsA 1 never checks pos ≤ bs.length directly unless pos+20 ≤ bs.length!
        simp only [skipWordsA] at h2
        split at h2
        · simp at h2
        · cases hc : skipCodeA bs (u32 bs (cw + 12)) (cw + 20) with
          | none => simp [hc] at h2
          | some p =>
            simp only [hc] at h2
            split at h2
            · simp at h2
            · simp at h2
              have hle := skipWordsA_bounds (by omega) hs
              omega
      simp only [skipWordsA] at h2
      split at h2
      · simp at h2
      · cases hc : skipCodeA bs (u32 bs (cw + 12)) (cw + 20) with
        | none => simp [hc] at h2
        | some annOff =>
          simp only [hc] at h2
          split at h2
          · simp at h2
          · simp at h2
            have hcb := skipCodeA_bounds (by omega) hc
            have hrem := skipWordsA_bounds (by omega) hs
            exact ⟨cw, rfl, by omega, annOff, hc, hcb.1, by omega⟩

theorem skipWordsA_end_zero {bs : List Nat} {pos : Nat} :
    ∀ {n : Nat}, pos ≤ bs.length → skipWordsA bs n pos = some bs.length → (pos = bs.length ↔ n = 0)
  | 0, _, h => by simp [skipWordsA] at h; exact ⟨fun _ => rfl, fun _ => h⟩
  | n + 1, hp, h => by
    have ⟨cw, hcw, hcw20, _, _, _, _⟩ := skipWordsA_lt_step (Nat.zero_lt_succ n) h
    simp only [skipWordsA, Option.some.injEq] at hcw
    rw [← hcw] at hcw20
    exact ⟨fun heq => by omega, fun h0 => by contradiction⟩

/-- lookupAnn: the offset of the first of `k` annotations from `q` whose index is `t`. -/
def lookupA (bs : List Nat) (t : Nat) : Nat → Nat → Option Nat
  | 0, _ => none
  | k + 1, q => if u32 bs q = t then some q else lookupA bs t k (q + 12)

theorem lookupA_bounds {bs : List Nat} {t : Nat} : ∀ {k q q'},
    lookupA bs t k q = some q' → q ≤ q' ∧ q' + 12 ≤ q + 12 * k
  | 0, q, q', h => by simp [lookupA] at h
  | k + 1, q, q', h => by
    simp only [lookupA] at h
    split at h
    · simp at h; subst h; omega
    · have ih := lookupA_bounds h; omega

/-! ### Effect tree for `effT` -/

inductive EffTree where
  | leaf (eIn eOut : Nat)
  | node (k : Nat) (l r : EffTree)
deriving DecidableEq, Repr

def EffTree.eval : EffTree → Nat → Nat × Nat
  | .leaf i o, _ => (i, o)
  | .node k l r, op => if op < k then l.eval op else r.eval op

def EffTree.emit (L tgt off : Nat) : EffTree → Nat × List (Nat × Ins × Nat)
  | .leaf i o =>
    (off + 17, [(L + off, .movI 10 i, 6), (L + (off + 6), .movI 11 o, 6), (L + (off + 12), .jmp tgt, 5)])
  | .node k l r =>
    let (offR, cL) := EffTree.emit L tgt (off + 10) l
    let (offE, cR) := EffTree.emit L tgt offR r
    (offE, [(L + off, .cmpI 15 k, 4), (L + (off + 4), .jcc 3 (L + offR), 6)] ++ cL ++ cR)

def primEffTree : EffTree :=
  .node 34
    (.node 15
      (.node 4
        (.node 2 (.node 1 (.leaf 1 2) (.leaf 1 0)) (.node 3 (.leaf 2 2) (.leaf 2 3)))
        (.node 6 (.node 5 (.leaf 3 3) (.leaf 2 1)) (.node 7 (.leaf 2 3) (.node 14 (.leaf 2 1) (.leaf 1 1)))))
      (.node 29
        (.node 19 (.node 18 (.leaf 2 1) (.leaf 1 1)) (.node 28 (.leaf 2 1) (.leaf 1 1)))
        (.node 31 (.node 30 (.leaf 2 0) (.leaf 1 1)) (.node 32 (.leaf 2 0) (.node 33 (.leaf 1 1) (.leaf 2 0))))))
    (.node 44
      (.node 39
        (.node 36 (.node 35 (.leaf 1 1) (.leaf 2 0)) (.node 38 (.leaf 3 0) (.leaf 3 1)))
        (.node 41 (.node 40 (.leaf 1 1) (.leaf 0 1)) (.node 42 (.leaf 1 0) (.node 43 (.leaf 4 1) (.leaf 2 1)))))
      (.node 48
        (.node 46 (.node 45 (.leaf 0 1) (.leaf 3 0)) (.node 47 (.leaf 2 2) (.leaf 0 1)))
        (.node 53 (.node 51 (.leaf 1 0) (.leaf 0 1)) (.node 54 (.leaf 1 1) (.node 56 (.leaf 1 2) (.leaf 2 1))))))

set_option maxRecDepth 20000 in
theorem primEffTree_emit (L tgt : Nat) : (EffTree.emit L tgt 0 primEffTree).2 = effT L tgt := rfl

theorem prim_all_len : Sporos.Prim.all.length = 57 := by decide +kernel
theorem prim_50_exit : Sporos.Prim.all[50]? = some .exit := by decide +kernel

theorem primEffTree_ok :
    (List.range 57).all (fun op => Sporos.Prim.all[op]?.map Sporos.Prim.eff == some (primEffTree.eval op)) = true := by
  decide +kernel

theorem primEffTree_spec {op : Nat} (h : op < 57) :
    Sporos.Prim.all[op]?.map Sporos.Prim.eff = some (primEffTree.eval op) := by
  have := List.all_eq_true.1 primEffTree_ok op (List.mem_range.2 h)
  simpa using this

/-! ### Streaming `checkFlatA` -/

inductive StepResA where
  | reject
  | overflow
  | fall (d a len : Nat)
  | nofall (len : Nat)
deriving DecidableEq, Repr

inductive CheckResA where
  | ok
  | reject
  | overflow
deriving DecidableEq, Repr

/-- Parse the container (header, data, word boundaries) matching P1. -/
def parseA (bs : List Nat) : Option (Nat × Nat × Nat) :=
  if bs.length < 16 then none
  else if u32 bs 0 ≠ 0x31525053 then none
  else
    let main := u32 bs 4
    let nw := u32 bs 8
    let nd := u32 bs 12
    match skipDataA bs nd 16 with
    | none => none
    | some ws =>
      match skipWordsA bs nw ws with
      | none => none
      | some endPos => if endPos = bs.length then some (main, nw, ws) else none

def hstepA (bs : List Nat) (nw ws outs locals na annOff d a p : Nat) : StepResA :=
  let op := (bs[p]?).getD 255
  if op < 57 then
    if op = 50 then
      if d = 0 then .reject else .nofall 1
    else
      let (eIn, eOut) := primEffTree.eval op
      if d < eIn then .reject
      else
        let d' := d - eIn + eOut
        if HMAX ≤ d' then .overflow else .fall d' a 1
  else if op = 0x80 then
    let d' := d + 1
    if HMAX ≤ d' then .overflow else .fall d' a 9
  else if op = 0x81 then
    let d' := d + 2
    if HMAX ≤ d' then .overflow else .fall d' a 5
  else if op = 0x82 then
    let n := u32 bs (p + 1)
    if n < d then
      let d' := d + 1
      if HMAX ≤ d' then .overflow else .fall d' a 5
    else .reject
  else if op = 0x83 then
    if d = 0 then .reject
    else
      let a' := a + 1
      if HMAX ≤ a' then .overflow else .fall (d - 1) a' 1
  else if op = 0x84 then
    if a = 0 then .reject
    else
      let d' := d + 1
      if HMAX ≤ d' then .overflow else .fall d' (a - 1) 1
  else if op = 0x85 then
    if a = 0 then .reject
    else
      let d' := d + 1
      if HMAX ≤ d' then .overflow else .fall d' a 1
  else if op = 0x86 then
    let i := u32 bs (p + 1)
    if i < locals then
      let d' := d + 1
      if HMAX ≤ d' then .overflow else .fall d' a 5
    else .reject
  else if op = 0x87 then
    if d = 0 then .reject
    else
      let i := u32 bs (p + 1)
      if i < locals then .fall (d - 1) a 5 else .reject
  else if op = 0x88 then
    let t := u32 bs (p + 1)
    match lookupA bs t na annOff with
    | none => .reject
    | some q => if u32 bs (q + 4) = d ∧ u32 bs (q + 8) = a then .nofall 5 else .reject
  else if op = 0x89 ∨ op = 0x8a then
    if d = 0 then .reject
    else
      let d' := d - 1
      let t := u32 bs (p + 1)
      match lookupA bs t na annOff with
      | none => .reject
      | some q => if u32 bs (q + 4) = d' ∧ u32 bs (q + 8) = a then .fall d' a 5 else .reject
  else if op = 0x8b then
    let w := u32 bs (p + 1)
    if w < nw then
      match skipWordsA bs w ws with
      | none => .reject
      | some cw =>
        let cins := u32 bs cw
        let couts := u32 bs (cw + 4)
        if d < cins then .reject
        else
          let d' := d - cins + couts
          if HMAX ≤ d' then .overflow else .fall d' a 5
    else .reject
  else if op = 0x8c then
    if d = outs ∧ a = 0 then .nofall 1 else .reject
  else .reject

def walkA (bs : List Nat) (nw ws outs locals nc na annOff : Nat) :
    Nat → Nat → Nat → Nat → Nat → CheckResA
  | 0, _, _, _, _ => .ok
  | rem + 1, i, p, curD, curA =>
    let h := match lookupA bs i na annOff with
      | some q => (u32 bs (q + 4), u32 bs (q + 8))
      | none => (curD, curA)
    match hstepA bs nw ws outs locals na annOff h.1 h.2 p with
    | .reject => .reject
    | .overflow => .overflow
    | .fall nd na' len =>
      if i + 1 < nc then
        match lookupA bs (i + 1) na annOff with
        | none => walkA bs nw ws outs locals nc na annOff rem (i + 1) (p + len) nd na'
        | some q =>
          if u32 bs (q + 4) = nd ∧ u32 bs (q + 8) = na' then
            walkA bs nw ws outs locals nc na annOff rem (i + 1) (p + len) nd na'
          else .reject
      else .reject
    | .nofall len =>
      let nxt := match lookupA bs (i + 1) na annOff with
        | some q => (u32 bs (q + 4), u32 bs (q + 8))
        | none => (0, 0)
      walkA bs nw ws outs locals nc na annOff rem (i + 1) (p + len) nxt.1 nxt.2

def annsOKA (bs : List Nat) (nc na annOff : Nat) : Nat → Nat → Bool
  | 0, _ => true
  | rem + 1, q =>
    let t := u32 bs q
    if t < nc then
      match lookupA bs t na annOff with
      | none => false
      | some q' =>
        u32 bs (q' + 4) == u32 bs (q + 4) &&
        u32 bs (q' + 8) == u32 bs (q + 8) &&
        annsOKA bs nc na annOff rem (q + 12)
    else false

def checkWordA (bs : List Nat) (nw ws wordOff : Nat) : CheckResA :=
  let ins := u32 bs wordOff
  let outs := u32 bs (wordOff + 4)
  let locals := u32 bs (wordOff + 8)
  let nc := u32 bs (wordOff + 12)
  let na := u32 bs (wordOff + 16)
  if nc = 0 then .reject
  else
    match skipCodeA bs nc (wordOff + 20) with
    | none => .reject
    | some annOff =>
      let initOK := match lookupA bs 0 na annOff with
        | none => true
        | some q => u32 bs (q + 4) == ins && u32 bs (q + 8) == 0
      if !initOK then .reject
      else
        match walkA bs nw ws outs locals nc na annOff nc 0 (wordOff + 20) ins 0 with
        | .reject => .reject
        | .overflow => .overflow
        | .ok =>
          if annsOKA bs nc na annOff na annOff then .ok else .reject

def checkWordsA (bs : List Nat) (nw ws : Nat) : Nat → Nat → CheckResA
  | 0, _ => .ok
  | rem + 1, wordOff =>
    match checkWordA bs nw ws wordOff with
    | .reject => .reject
    | .overflow => .overflow
    | .ok =>
      match skipWordsA bs 1 wordOff with
      | none => .reject
      | some nextOff => checkWordsA bs nw ws rem nextOff

def checkFlatA (bs : List Nat) : Option Bool :=
  if !(bs.all (· < 256) && bs.length < HMAX) then none
  else
    match parseA bs with
    | none => some false
    | some (main, nw, ws) =>
      if !(main < nw) then some false
      else
        match skipWordsA bs main ws with
        | none => some false
        | some mainOff =>
          if u32 bs mainOff ≠ 0 then some false
          else
            match checkWordsA bs nw ws nw ws with
            | .ok => some true
            | .reject => some false
            | .overflow => none

def verdictWord : Option Bool → Nat
  | some true => ACCEPT
  | some false => REJECT
  | none => OVERFLOW

theorem primEffTree_eval_eq (op : Nat) (h : op < 57) :
    primEffTree.eval op =
      let p := Sporos.Prim.all[op]'(by rw [prim_all_len]; exact h)
      p.eff := by
  have hs := primEffTree_spec h
  rw [List.getElem?_eq_getElem (by rw [prim_all_len]; exact h)] at hs
  simp only [Option.map_some, Option.some.injEq] at hs
  exact hs.symm

end SporosCheckBin
