/-
The single-pass stack-effect checker the stage-1 hex implements, and its
equivalence to `checkWord ∘ infer` / `checkProg`.

`walk` mirrors the hex code walk (`sporos-stage1.hex`, from `codewalk`
on): it recurses over the word's instructions carrying only the current
height, never the height table `infer` builds. The reduction that
justifies it:

- at instruction `i`, the height is `ann(i)` if announced, else the
  fall-through height — exactly `infer`'s `h`;
- a fall-through to an announced `i+1` must produce the announced
  height (the only cross-check `checkWord` imposes beyond `hstep`
  succeeding, the unannounced case being definitional in `infer`);
- after a non-fall-through, the next height is `ann(i+1)` or `(0,0)`;
- the annotation clause of `checkWord` says every ann entry equals the
  table there, and at an announced index the table is the first entry,
  so it says every entry is in range and agrees with the first entry
  at its index.

`checkWord1_eq` and `checkProg1_eq` close the equivalence. This is the
specification the per-handler triples compose against; it depends on
nothing x86.
-/
import Sporos.Check
import Sporos.Examples

namespace Sporos

/-- One pass over a word's instructions, carrying the current height.
`cur` is the height `infer` would fall through with; the height at an
instruction is the ann there if any, else `cur`. -/
def walk (pr : Prog) (wd : Word) : List Instr → Nat → Height → Bool
  | [], _, _ => true
  | ins :: rest, i, cur =>
    let h := (wd.ann.lookup i).getD cur
    match hstep pr wd h ins with
    | none => false
    | some none => walk pr wd rest (i + 1) ((wd.ann.lookup (i + 1)).getD ⟨0, 0⟩)
    | some (some h') =>
      match rest with
      | [] => false
      | _ =>
        (match wd.ann.lookup (i + 1) with
         | some a => a == h'
         | none => true) &&
        walk pr wd rest (i + 1) h'

/-- `checkWord`'s per-instruction clause as a simultaneous recursion
over code and heights, the bridge between the `getElem?` form in
`checkWord` and `walk`. -/
def stepsOk (pr : Prog) (wd : Word) : List Instr → List Height → Nat → Bool
  | [], [], _ => true
  | ins :: rest, h :: H, i =>
    match hstep pr wd h ins with
    | none => false
    | some none => stepsOk pr wd rest H (i + 1)
    | some (some h') => (H.head? == some h') && stepsOk pr wd rest H (i + 1)
  | _, _, _ => false

/-- The annotation clause in first-match form. -/
def annOk (wd : Word) : Bool :=
  wd.ann.all fun (t, a) => decide (t < wd.code.length) && (wd.ann.lookup t == some a)

/-- The single-pass word checker. -/
def checkWord1 (pr : Prog) (wd : Word) : Bool :=
  !wd.code.isEmpty &&
  (match wd.ann.lookup 0 with
   | some a => a == ⟨wd.ins, 0⟩
   | none => true) &&
  annOk wd &&
  walk pr wd wd.code 0 ⟨wd.ins, 0⟩

/-- The single-pass program checker. -/
def checkProg1 (pr : Prog) : Bool :=
  (pr.words.all fun wd => checkWord1 pr wd) &&
  (match pr.words[pr.main]? with | some m => m.ins == 0 | none => false)

/-! ### Facts about `infer.go` -/

theorem go_length (pr : Prog) (wd : Word) :
    ∀ (code : List Instr) (i : Nat) (cur : Height),
      (infer.go pr wd code i cur).length = code.length := by
  intro code i cur
  induction code generalizing i cur with
  | nil => rfl
  | cons ins rest ih =>
    rw [infer.go]
    simp only [List.length_cons, ih]

theorem go_head? (pr : Prog) (wd : Word) (y : Instr) (ys : List Instr)
    (i : Nat) (cur : Height) :
    (infer.go pr wd (y :: ys) i cur).head? = some ((wd.ann.lookup i).getD cur) := by
  rw [infer.go]
  rfl

theorem go_getElem?_ann (pr : Prog) (wd : Word) :
    ∀ (code : List Instr) (i cur_t : Nat) (cur : Height) (a : Height),
      cur_t < code.length → wd.ann.lookup (i + cur_t) = some a →
      (infer.go pr wd code i cur)[cur_t]? = some a := by
  intro code i cur_t cur a
  induction code generalizing i cur_t cur with
  | nil => intro h; simp at h
  | cons ins rest ih =>
    intro ht hann
    cases cur_t with
    | zero =>
      have hh : (infer.go pr wd (ins :: rest) i cur).head? =
          (infer.go pr wd (ins :: rest) i cur)[0]? := by
        cases infer.go pr wd (ins :: rest) i cur <;> rfl
      rw [← hh, go_head?]
      simp only [Nat.add_zero] at hann
      rw [hann]; rfl
    | succ t =>
      have e1 : (infer.go pr wd (ins :: rest) i cur)[t + 1]? =
          ((infer.go pr wd (ins :: rest) i cur).tail)[t]? := by
        cases infer.go pr wd (ins :: rest) i cur <;> rfl
      rw [e1]
      have e2 : (infer.go pr wd (ins :: rest) i cur).tail =
          infer.go pr wd rest (i + 1)
            (match hstep pr wd ((wd.ann.lookup i).getD cur) ins with
             | some (some h') => h'
             | _ => (wd.ann.lookup (i + 1)).getD ⟨0, 0⟩) := by
        rw [infer.go]; rfl
      rw [e2]
      apply ih
      · simpa using ht
      · have : i + 1 + t = i + (t + 1) := by omega
        rw [this]; exact hann

/-! ### `walk` is `stepsOk` over `infer`'s table -/

theorem walk_stepsOk (pr : Prog) (wd : Word) :
    ∀ (code : List Instr) (i : Nat) (cur : Height),
      walk pr wd code i cur = stepsOk pr wd code (infer.go pr wd code i cur) i := by
  intro code i cur
  induction code generalizing i cur with
  | nil => rfl
  | cons ins rest ih =>
    simp only [walk, infer.go, stepsOk]
    cases hs : hstep pr wd ((wd.ann.lookup i).getD cur) ins with
    | none => rfl
    | some o =>
      cases o with
      | none => dsimp only; exact ih (i + 1) _
      | some h' =>
        cases rest with
        | nil => dsimp only; simp [infer.go]
        | cons y ys =>
          dsimp only
          rw [go_head?]
          have key : (some ((wd.ann.lookup (i + 1)).getD h') == some h') =
              (match wd.ann.lookup (i + 1) with
               | some a => a == h'
               | none => true) := by
            cases hl : wd.ann.lookup (i + 1) with
            | none => simp [hl]
            | some a => simp [hl]
          rw [key, ih]

/-! ### `stepsOk` is `checkWord`'s per-instruction clause -/

/-- The clause `checkWord` imposes at one index, as a proposition. -/
def clauseAt (pr : Prog) (wd : Word) (code : List Instr) (H : List Height) (j : Nat) : Prop :=
  match code[j]?, H[j]? with
  | some ins, some h =>
    match hstep pr wd h ins with
    | none => False
    | some none => True
    | some (some h') => H[j + 1]? = some h'
  | _, _ => False

theorem clauseAt_zero (pr : Prog) (wd : Word) (ins : Instr) (rest : List Instr)
    (h : Height) (H : List Height) :
    clauseAt pr wd (ins :: rest) (h :: H) 0 =
      (match hstep pr wd h ins with
       | none => False
       | some none => True
       | some (some h') => H.head? = some h') := by
  simp [clauseAt, List.getElem?_cons_succ, List.head?_eq_getElem?]

theorem clauseAt_succ (pr : Prog) (wd : Word) (ins : Instr) (rest : List Instr)
    (h : Height) (H : List Height) (j : Nat) :
    clauseAt pr wd (ins :: rest) (h :: H) (j + 1) = clauseAt pr wd rest H j := by
  simp [clauseAt, List.getElem?_cons_succ]

theorem stepsOk_iff (pr : Prog) (wd : Word) :
    ∀ (code : List Instr) (H : List Height) (i : Nat), H.length = code.length →
      (stepsOk pr wd code H i = true ↔
        ∀ j, j < code.length → clauseAt pr wd code H j) := by
  intro code H i hlen
  induction code generalizing H i with
  | nil =>
    cases H with
    | nil =>
      constructor
      · intro _ j hj; exact absurd hj (by simp)
      · intro _; rfl
    | cons h H' => simp at hlen
  | cons ins rest ih =>
    cases H with
    | nil => simp at hlen
    | cons h H' =>
      have hlen' : H'.length = rest.length := by simpa using hlen
      have ih' := ih (H := H') (i := i + 1) hlen'
      simp only [stepsOk]
      cases hs : hstep pr wd h ins with
      | none =>
        constructor
        · intro hcon; cases hcon
        · intro hr
          have h0 := hr 0 (by simp)
          rw [clauseAt_zero, hs] at h0
          exact absurd h0 (by simp)
      | some o =>
        cases o with
        | none =>
          rw [ih']
          constructor
          · intro hr j hj
            cases j with
            | zero => simp [clauseAt_zero, hs]
            | succ j =>
              rw [clauseAt_succ]
              exact hr j (by simpa using hj)
          · intro hr j hj
            rw [← clauseAt_succ]
            exact hr (j + 1) (by simp; omega)
        | some h' =>
          simp only [hs, Bool.and_eq_true]
          rw [ih']
          constructor
          · rintro ⟨hhead, hr⟩ j hj
            cases j with
            | zero =>
              rw [clauseAt_zero, hs]
              cases hh : H'.head? with
              | none => simp [hh] at hhead
              | some x => simpa [hh] using hhead
            | succ j =>
              rw [clauseAt_succ]
              exact hr j (by simpa using hj)
          · intro hr
            have h0 := hr 0 (by simp)
            rw [clauseAt_zero, hs] at h0
            constructor
            · cases hh : H'.head? with
              | none => rw [hh] at h0; cases h0
              | some x => simpa [hh] using h0
            · intro j hj
              rw [← clauseAt_succ]
              exact hr (j + 1) (by simp; omega)

/-! ### The annotation clause in first-match form -/

theorem lookup_some_of_mem :
    ∀ {l : List (Nat × Height)} {t : Nat} {a : Height}, (t, a) ∈ l →
      ∃ a', l.lookup t = some a' := by
  intro l t a h
  induction l with
  | nil => simp at h
  | cons p l ih =>
    obtain ⟨x, b⟩ := p
    simp only [List.mem_cons] at h
    cases h with
    | inl hh =>
      have htx : t = x := congrArg Prod.fst hh
      exact ⟨b, by simp [List.lookup, htx]⟩
    | inr hm =>
      obtain ⟨a', ha'⟩ := ih hm
      by_cases hx : (t == x) = true
      · exact ⟨b, by simp [List.lookup, hx]⟩
      · exact ⟨a', by simp [List.lookup, hx, ha']⟩

theorem all_congr_mem {α : Type} {l : List α} {f g : α → Bool}
    (h : ∀ x ∈ l, f x = g x) : l.all f = l.all g := by
  induction l with
  | nil => rfl
  | cons y ys ih =>
    simp only [List.all_cons, h y (by simp), ih (fun x hx => h x (by simp [hx]))]

theorem infer_eq_go (pr : Prog) (wd : Word) :
    infer pr wd = infer.go pr wd wd.code 0 ⟨wd.ins, 0⟩ := rfl

theorem ann_eq (pr : Prog) (wd : Word) :
    (wd.ann.all fun (t, a) => t < wd.code.length && (infer pr wd)[t]? == some a) =
      annOk wd := by
  unfold annOk
  apply all_congr_mem
  rintro ⟨t, a⟩ hmem
  by_cases ht : t < wd.code.length
  · obtain ⟨a', hl⟩ := lookup_some_of_mem hmem
    have hH : (infer pr wd)[t]? = some a' := by
      rw [infer_eq_go]
      have := go_getElem?_ann pr wd wd.code 0 t ⟨wd.ins, 0⟩ a' ht (by simpa using hl)
      exact this
    simp [ht, hH, hl]
  · simp [ht]

/-! ### Assembly -/

theorem clauseBool_iff (pr : Prog) (wd : Word) {code : List Instr} {H : List Height}
    (hlenH : H.length = code.length) (j : Nat) :
    (match code[j]?, H[j]? with
     | some ins, some h =>
       match hstep pr wd h ins with
       | none => false
       | some none => true
       | some (some h') => j + 1 < code.length && (H[j + 1]? == some h')
     | _, _ => false) = true ↔ clauseAt pr wd code H j := by
  cases hc : code[j]? with
  | none => simp [clauseAt, hc]
  | some ins =>
    cases hh : H[j]? with
    | none => simp [clauseAt, hc, hh]
    | some h =>
      cases hs : hstep pr wd h ins with
      | none => simp [clauseAt, hc, hh, hs]
      | some o =>
        cases o with
        | none => simp [clauseAt, hc, hh, hs]
        | some h' =>
          simp only [clauseAt, hc, hh, hs, Bool.and_eq_true]
          constructor
          · intro hcon
            cases hq : H[j + 1]? with
            | none => simp [hq] at hcon
            | some x => simpa [hq] using hcon.2
          · intro hcon
            have h1 : j + 1 < H.length := (List.getElem?_eq_some_iff.mp hcon).1
            have hlt : j + 1 < code.length := by omega
            simp [hlt, hcon]

theorem checkWord1_eq (pr : Prog) (wd : Word) :
    checkWord1 pr wd = checkWord pr wd (infer pr wd) := by
  have hlenH : (infer pr wd).length = wd.code.length := by
    rw [infer_eq_go]; exact go_length pr wd _ _ _
  have hrange :
      ((List.range wd.code.length).all (fun i =>
        match wd.code[i]?, (infer pr wd)[i]? with
        | some ins, some h =>
          match hstep pr wd h ins with
          | none => false
          | some none => true
          | some (some h') => i + 1 < wd.code.length && (infer pr wd)[i + 1]? == some h'
        | _, _ => false) = true ↔ walk pr wd wd.code 0 ⟨wd.ins, 0⟩ = true) := by
    rw [walk_stepsOk, ← infer_eq_go, stepsOk_iff pr wd _ _ _ hlenH]
    constructor
    · intro hr j hj
      apply (clauseBool_iff pr wd hlenH j).mp
      rw [List.all_eq_true] at hr
      exact hr j (List.mem_range.mpr hj)
    · intro hr
      rw [List.all_eq_true]
      intro j hj
      rw [List.mem_range] at hj
      exact (clauseBool_iff pr wd hlenH j).mpr (hr j hj)
  apply Bool.eq_iff_iff.mpr
  by_cases he : wd.code = []
  · simp [checkWord1, checkWord, he, infer_eq_go, infer.go]
  · obtain ⟨ins0, rest0, hcode⟩ := List.exists_cons_of_ne_nil he
    have hH0 : (infer pr wd)[0]? = some ((wd.ann.lookup 0).getD ⟨wd.ins, 0⟩) := by
      rw [← List.head?_eq_getElem?, infer_eq_go, hcode, go_head?]
    constructor
    · intro h
      simp only [checkWord1, Bool.and_eq_true] at h
      obtain ⟨⟨⟨hnemp, h0⟩, hann⟩, hwalk⟩ := h
      simp only [checkWord, Bool.and_eq_true]
      refine ⟨⟨⟨⟨?_, hnemp⟩, ?_⟩, ?_⟩, ?_⟩
      · simp [hlenH]
      · cases hl : wd.ann.lookup 0 with
        | none => simp [hH0, hl]
        | some a => rw [hl] at h0; simpa [hH0, hl] using h0
      · exact hrange.mpr hwalk
      · rwa [ann_eq]
    · intro h
      simp only [checkWord, Bool.and_eq_true] at h
      obtain ⟨⟨⟨⟨-, hnemp⟩, h0⟩, hrangev⟩, hann⟩ := h
      simp only [checkWord1, Bool.and_eq_true]
      refine ⟨⟨⟨hnemp, ?_⟩, ?_⟩, ?_⟩
      · cases hl : wd.ann.lookup 0 with
        | none => rfl
        | some a => simpa [hH0, hl] using h0
      · rwa [← ann_eq]
      · exact hrange.mp hrangev

theorem checkProg1_eq (pr : Prog) : checkProg1 pr = checkProg pr := by
  unfold checkProg1 checkProg
  congr 1
  exact all_congr_mem (fun wd _ => checkWord1_eq pr wd)

example : checkProg1 Examples.hello = true := by decide +kernel
example : checkProg1 Examples.upper = true := by decide +kernel
example : checkProg1 Examples.fib = true := by decide +kernel

end Sporos
