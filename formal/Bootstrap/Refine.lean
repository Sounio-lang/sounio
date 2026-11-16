/-
Refinement of machine states, and the monotonicity of `step` under it.

`StLe r o a b`: every register, flag and stack byte that `a` defines, `b` defines
with the same value; rip and layout are equal; `b`'s input is `a`'s followed by
`r`, and `b`'s output is `a`'s followed by `o` (output is kept most recent first).
An abstract state is one with `none` where the program does not care, and with
only the input of one loop iteration.

The model makes every read of an undefined value stuck, so a step of `a` that is
not stuck reads only what `a` defines. Then `b` takes the same step and lands in
a state that refines the one `a` lands in (`step_mono`). This is what lets a
finite computation on abstract states stand for every concrete state above them.

The one step this fails for is a read at the end of `a`'s input when `r` is not
empty: `a` sees end of input, `b` the first byte of `r`. `step_mono` therefore
asks for `r = []` or that the step is not such a read (`readsEOF`).
-/
import Bootstrap.X86Min

namespace X86Min

variable {r o : List Nat}

/-- `a` is defined at most where `b` is, with the same value. -/
def OLe {α : Type} (a b : Option α) : Prop := ∀ v, a = some v → b = some v

theorem OLe.refl {α : Type} (a : Option α) : OLe a a := fun _ h => h
theorem OLe.none {α : Type} (b : Option α) : OLe none b := fun _ h => by cases h

structure FlLe (f g : Flags) : Prop where
  cf : OLe f.cf g.cf
  zf : OLe f.zf g.zf
  sf : OLe f.sf g.sf
  of : OLe f.of g.of

theorem FlLe.refl (f : Flags) : FlLe f f := ⟨OLe.refl _, OLe.refl _, OLe.refl _, OLe.refl _⟩

structure StLe (r o : List Nat) (a b : St) : Prop where
  reg : ∀ i, OLe (a.reg i) (b.reg i)
  rlen : a.regs.length = b.regs.length
  rip : a.rip = b.rip
  fl : FlLe a.fl b.fl
  byte : ∀ x, OLe (a.rdByte x) (b.rdByte x)
  mlen : a.mem.length = b.mem.length
  lo : a.lo = b.lo
  inp : b.inp = a.inp ++ r
  out : b.out = a.out ++ o

/-! ### Reads -/

theorem rd_le {r o : List Nat} {a b : St} (h : StLe r o a b) (w i : Nat) : OLe (a.rd w i) (b.rd w i) := by
  intro v hv
  simp only [St.rd] at *
  cases ha : a.reg i with
  | none => rw [ha] at hv; cases hv
  | some x => rw [ha] at hv; rw [h.reg i x ha]; exact hv

theorem le_mono {f g : Nat → Option Nat} (hfg : ∀ x, OLe (f x) (g x)) :
    ∀ (n off : Nat), OLe (le f off n) (le g off n) := by
  intro n; induction n with
  | zero => intro off v hv; simpa [le] using hv
  | succ n ih =>
    intro off v hv
    simp only [le] at hv ⊢
    cases hf : f off with
    | none => rw [hf] at hv; cases hv
    | some lo =>
      rw [hf] at hv; rw [hfg off lo hf]
      cases hl : le f (off + 1) n with
      | none => rw [hl] at hv; cases hv
      | some hi =>
        rw [hl] at hv; rw [ih (off + 1) hi hl]; exact hv

/-! ### Writes -/

theorem getD_set_reg (l : List (Option Nat)) (i j : Nat) (v : Option Nat) :
    ((l.set i v)[j]?).getD none = if i = j ∧ i < l.length then v else (l[j]?).getD none := by
  rw [List.getElem?_set]
  by_cases hij : i = j
  · subst hij
    by_cases hl : i < l.length
    · simp [hl]
    · simp [hl, List.getElem?_eq_none (Nat.le_of_not_lt hl)]
  · simp [hij]

theorem setReg_le {r o : List Nat} {a b : St} (h : StLe r o a b) (i : Nat) (v : Option Nat) :
    StLe r o (a.setReg i v) (b.setReg i v) where
  reg j := by
    intro x hx
    simp only [St.reg, St.setReg, getD_set_reg] at hx ⊢
    rw [← h.rlen]
    by_cases hc : i = j ∧ i < a.regs.length
    · rw [if_pos hc] at hx ⊢; exact hx
    · rw [if_neg hc] at hx ⊢; exact h.reg j x hx
  rlen := by simp [St.setReg, h.rlen]
  rip := h.rip
  fl := h.fl
  byte := h.byte
  mlen := h.mlen
  lo := h.lo
  inp := h.inp
  out := h.out

/-- Lifting of `StLe` to computations that may fail. -/
def SLe (r o : List Nat) (oa ob : Option St) : Prop := ∀ t, oa = some t → ∃ t', ob = some t' ∧ StLe r o t t'

theorem SLe.some {r o : List Nat} {a b : St} (h : StLe r o a b) : SLe r o (some a) (some b) := by
  intro t ht; cases ht; exact ⟨b, rfl, h⟩

theorem wr_le {r o : List Nat} {a b : St} (h : StLe r o a b) (w i v : Nat) : SLe r o (a.wr w i v) (b.wr w i v) := by
  intro t ht
  simp only [St.wr] at ht ⊢
  by_cases hw : w = 8
  · simp only [hw, beq_self_eq_true, if_true] at ht ⊢
    cases ho : a.reg i with
    | none => rw [ho] at ht; cases ht
    | some old =>
      rw [ho] at ht; rw [h.reg i old ho]; cases ht
      exact ⟨_, rfl, setReg_le h i _⟩
  · have hw' : (w == 8) = false := by simp [hw]
    simp only [hw', Bool.false_eq_true, if_false] at ht ⊢
    cases ht; exact ⟨_, rfl, setReg_le h i _⟩

theorem SLe_bind {α : Type} {ox oy : Option α} {f g : α → Option St}
    (hxy : OLe ox oy) (hfg : ∀ v, SLe r o (f v) (g v)) : SLe r o (ox.bind f) (oy.bind g) := by
  intro t ht
  cases hx : ox with
  | none => rw [hx] at ht; cases ht
  | some v =>
    rw [hx] at ht; rw [hxy v hx]; exact hfg v t ht

theorem SLe_none (ob : Option St) : SLe r o none ob := fun _ h => by cases h

/-! ### Field updates -/

theorem StLe.with_rip {r o : List Nat} {a b : St} (h : StLe r o a b) (x : Nat) :
    StLe r o { a with rip := x } { b with rip := x } :=
  ⟨h.reg, h.rlen, rfl, h.fl, h.byte, h.mlen, h.lo, h.inp, h.out⟩

theorem StLe.with_fl {r o : List Nat} {a b : St} (h : StLe r o a b) (f : Flags) :
    StLe r o { a with fl := f } { b with fl := f } :=
  ⟨h.reg, h.rlen, h.rip, FlLe.refl f, h.byte, h.mlen, h.lo, h.inp, h.out⟩

theorem StLe.with_fl_keep_cf {r o : List Nat} {a b : St} (h : StLe r o a b) (z s v : Option Bool) :
    StLe r o { a with fl := { a.fl with zf := z, sf := s, of := v } }
         { b with fl := { b.fl with zf := z, sf := s, of := v } } :=
  ⟨h.reg, h.rlen, h.rip, ⟨h.fl.cf, OLe.refl _, OLe.refl _, OLe.refl _⟩, h.byte, h.mlen, h.lo, h.inp, h.out⟩

theorem StLe.with_inp {r o : List Nat} {a b : St} (h : StLe r o a b) (l : List Nat) :
    StLe r o { a with inp := l } { b with inp := l ++ r } :=
  ⟨h.reg, h.rlen, h.rip, h.fl, h.byte, h.mlen, h.lo, rfl, h.out⟩

theorem StLe.with_out {r o : List Nat} {a b : St} (h : StLe r o a b) (l : List Nat) :
    StLe r o { a with out := l } { b with out := l ++ o } :=
  ⟨h.reg, h.rlen, h.rip, h.fl, h.byte, h.mlen, h.lo, h.inp, rfl⟩

/-! ### Stack bytes -/

theorem wrByte_le {r o : List Nat} {a b : St} (h : StLe r o a b) (x v : Nat) : SLe r o (a.wrByte x v) (b.wrByte x v) := by
  intro t ht
  simp only [St.wrByte, ← h.lo] at ht ⊢
  by_cases hw : (a.lo ≤ x && x < a.lo + winSize) = true
  · rw [if_pos hw] at ht ⊢; cases ht
    refine ⟨_, rfl, ⟨h.reg, h.rlen, h.rip, h.fl, ?_, by simp [h.mlen], rfl, h.inp, h.out⟩⟩
    intro y u hu
    simp only [St.rdByte, ← h.lo] at hu ⊢
    by_cases hy : (a.lo ≤ y && y < a.lo + winSize) = true
    · rw [if_pos hy] at hu ⊢
      rw [getD_set_reg] at hu ⊢
      rw [← h.mlen]
      by_cases hc : x - a.lo = y - a.lo ∧ x - a.lo < a.mem.length
      · rw [if_pos hc] at hu ⊢; exact hu
      · rw [if_neg hc] at hu ⊢
        have := h.byte y u
        simp only [St.rdByte, ← h.lo, if_pos hy] at this
        exact this hu
    · rw [if_neg hy] at hu; cases hu
  · rw [if_neg hw] at ht; cases ht

theorem wrBytes_le : ∀ (n : Nat) {a b : St} (_ : StLe r o a b) (x v : Nat),
    SLe r o (a.wrBytes x v n) (b.wrBytes x v n) := by
  intro n; induction n with
  | zero => intro a b h x v; exact SLe.some h
  | succ n ih =>
    intro a b h x v t ht
    simp only [St.wrBytes] at ht ⊢
    cases ha : a.wrByte x (v % 256) with
    | none => rw [ha] at ht; cases ht
    | some a1 =>
      rw [ha] at ht
      obtain ⟨b1, hb1, h1⟩ := wrByte_le h x (v % 256) a1 ha
      rw [hb1]; exact ih h1 (x + 1) (v / 256) t ht

/-! ### One step -/

/-- What `b` does is at least what `a` does: the next state up to refinement, or
the same exit with `o` written before `a`'s output. -/
def StepLe (r o : List Nat) (ra rb : Sum St Outcome) : Prop :=
  (∀ t, ra = .inl t → ∃ t', rb = .inl t' ∧ StLe r o t t') ∧
  (∀ st x, ra = .inr (.exit st x) → rb = .inr (.exit st (o.reverse ++ x)))

theorem StepLe.okOf {oa ob : Option St} (p : Nat) (h : SLe r o oa ob) :
    StepLe r o (okOf p oa) (okOf p ob) := by
  constructor
  · intro t ht
    cases ha : oa with
    | none => rw [ha] at ht; cases ht
    | some ta =>
      rw [ha] at ht
      obtain ⟨t', hb, ht'⟩ := h ta ha
      have e : ta = t := by injection ht
      subst e
      exact ⟨t', by rw [hb]; rfl, ht'⟩
  · intro st x hx
    cases ha : oa with
    | none => rw [ha] at hx; cases hx
    | some ta => rw [ha] at hx; cases hx

theorem StepLe.inl {a b : St} (h : StLe r o a b) : StepLe r o (.inl a) (.inl b) := by
  constructor
  · intro t ht; cases ht; exact ⟨b, rfl, h⟩
  · intro _ _ hx; cases hx

theorem StepLe.stuck (p w : Nat) (rb : Sum St Outcome) : StepLe r o (.inr (.stuck p w)) rb := by
  constructor
  · intro _ ht; cases ht
  · intro _ _ hx; cases hx

theorem StepLe.exit (st : Nat) (x : List Nat) :
    StepLe r o (.inr (.exit st x.reverse)) (.inr (.exit st (x ++ o).reverse)) := by
  constructor
  · intro _ ht; cases ht
  · intro st' x' hx; cases hx; rw [List.reverse_append]

theorem OLe_bind {α β : Type} {ox oy : Option α} {f g : α → Option β}
    (hxy : OLe ox oy) (hfg : ∀ v, OLe (f v) (g v)) : OLe (ox.bind f) (oy.bind g) := by
  intro t ht
  cases hx : ox with
  | none => rw [hx] at ht; cases ht
  | some v => rw [hx] at ht; rw [hxy v hx]; exact hfg v t ht

theorem cond_le {f g : Flags} (h : FlLe f g) (cc : Nat) : OLe (cond f cc) (cond g cc) := by
  unfold cond
  generalize (fun base => (pure (if (cc % 2 == 0) = true then base else !base) : Option Bool)) = jp
  generalize cc / 2 = k
  match k with
  | 0 => exact OLe_bind h.of fun _ => OLe.refl _
  | 1 => exact OLe_bind h.cf fun _ => OLe.refl _
  | 2 => exact OLe_bind h.zf fun _ => OLe.refl _
  | 3 => exact OLe_bind h.cf fun _ => OLe_bind h.zf fun _ => OLe.refl _
  | 4 => exact OLe_bind h.sf fun _ => OLe.refl _
  | 5 => exact OLe.none _
  | 6 => exact OLe_bind h.sf fun _ => OLe_bind h.of fun _ => OLe.refl _
  | 7 => exact OLe_bind h.sf fun _ => OLe_bind h.of fun _ => OLe_bind h.zf fun _ => OLe.refl _
  | _ + 8 => exact OLe.none _

theorem SLe_bindS {ox oy : Option St} {f g : St → Option St}
    (hxy : SLe r o ox oy) (hfg : ∀ t t', StLe r o t t' → SLe r o (f t) (g t')) :
    SLe r o (ox.bind f) (oy.bind g) := by
  intro t ht
  cases hx : ox with
  | none => rw [hx] at ht; cases ht
  | some v =>
    rw [hx] at ht
    obtain ⟨v', hv', hvv⟩ := hxy v hx
    rw [hv']; exact hfg v v' hvv t ht

theorem guard_le {p p' q q' : Option Nat} (hp : OLe p p') (hq : OLe q q') (x y : Nat)
    {k k' : Option St} (hk : SLe r o k k') :
    SLe r o (if (p != some x || q != some y) = true then none else k)
        (if (p' != some x || q' != some y) = true then none else k') := by
  intro t ht
  by_cases hpx : p = some x
  · by_cases hqy : q = some y
    · rw [hp x hpx, hq y hqy]; rw [hpx, hqy] at ht; simp at ht ⊢; exact hk t ht
    · simp [hqy] at ht
  · simp [hpx] at ht

/-- The step at `s` is a read of standard input with the input used up. -/
def readsEOF (im : Img) (s : St) : Bool :=
  s.inp.isEmpty && s.reg 0 == some 0 &&
    match im.decodeAt s.rip with
    | some (.syscall, _) => true
    | _ => false

theorem step_mono (im : Img) {a b : St} (h : StLe r o a b) (hg : r = [] ∨ readsEOF im a = false) :
    StepLe r o (step im a) (step im b) := by
  obtain ⟨ra, pa, fa, ma, la, ia, oa⟩ := a
  obtain ⟨rb, pb, fb, mb, lb, ib, ob⟩ := b
  have e1 : pa = pb := h.rip
  have e2 : la = lb := h.lo
  have e3 : ib = ia ++ r := h.inp
  have e4 : ob = oa ++ o := h.out
  subst e1 e2 e3 e4
  unfold step
  cases hd : im.decodeAt pa with
  | none => exact StepLe.stuck _ _ _
  | some p =>
    obtain ⟨ins, len⟩ := p
    have h1 := h.with_rip ((pa + len) % pow2 64)
    cases ins with
    | movImm d i => exact StepLe.okOf _ (wr_le h1 _ _ _)
    | rr op w d src =>
      cases op
      case mov => exact StepLe.okOf _ (SLe_bind (rd_le h _ _) fun _ => wr_le h1 _ _ _)
      all_goals
        apply StepLe.okOf
        dsimp only
        repeat (first
          | exact SLe.some (h1.with_fl _)
          | exact wr_le (h1.with_fl _) _ _ _
          | (refine SLe_bind (rd_le h _ _) fun _ => ?_)
          | (refine SLe_bind (OLe.refl _) fun _ => ?_)
          | split)
    | ri op w d i =>
      apply StepLe.okOf
      refine SLe_bind (rd_le h _ _) fun va => ?_
      dsimp only
      split
      · exact SLe.some (h1.with_fl _)
      · exact wr_le (h1.with_fl _) _ _ _
    | shlI w d c =>
      apply StepLe.okOf
      refine SLe_bind (rd_le h _ _) fun va => ?_
      dsimp only
      split
      · exact wr_le h1 _ _ _
      · exact wr_le (h1.with_fl _) _ _ _
    | inc w d =>
      apply StepLe.okOf
      refine SLe_bind (rd_le h _ _) fun va => ?_
      exact wr_le (h1.with_fl_keep_cf _ _ _) _ _ _
    | push i =>
      exact StepLe.okOf _ (SLe_bind (h.reg 4) fun sp =>
        SLe_bindS (wrBytes_le 8 h1 _ _) fun t t' ht => SLe.some (setReg_le ht _ _))
    | pop d =>
      exact StepLe.okOf _ (SLe_bind (h.reg 4) fun sp =>
        SLe_bind (le_mono h.byte 8 sp) fun v => SLe.some (setReg_le (setReg_le h1 _ _) _ _))
    | syscall =>
      have h2 := setReg_le (setReg_le h1 1 (some ((pa + len) % pow2 64))) 11 none
      dsimp only
      cases hr : St.reg ⟨ra, pa, fa, ma, la, ia, oa⟩ 0 with
      | none => exact StepLe.stuck _ _ _
      | some v =>
        rw [h.reg 0 v hr]
        split
        · rename_i hv
          cases hv
          apply StepLe.okOf
          refine guard_le (h.reg 7) (h.reg 2) 0 1 ?_
          refine SLe_bind (h.reg 6) fun ad => ?_
          cases ia with
          | nil =>
            have hr0 : r = [] := by
              rcases hg with hg | hg
              · exact hg
              · simp [readsEOF, hd, hr] at hg
            subst hr0
            exact SLe.some (setReg_le h2 _ _)
          | cons c rest =>
            exact SLe_bindS (wrByte_le h2 _ _) fun t t' ht => SLe.some (setReg_le (ht.with_inp rest) _ _)
        · apply StepLe.okOf
          refine guard_le (h.reg 7) (h.reg 2) 1 1 ?_
          exact SLe_bind (h.reg 6) fun ad => SLe_bind (h.byte ad) fun c =>
            SLe.some (setReg_le (h2.with_out _) _ _)
        · cases h7 : St.reg ⟨ra, pa, fa, ma, la, ia, oa⟩ 7 with
          | none => exact StepLe.stuck _ _ _
          | some st =>
            rw [h.reg 7 st h7]
            exact StepLe.exit _ _
        · exact StepLe.stuck _ _ _
    | jcc cc rel =>
      dsimp only
      cases hc : cond fa cc with
      | none => exact StepLe.stuck _ _ _
      | some v =>
        rw [cond_le h.fl cc v hc]
        cases v
        · exact StepLe.inl h1
        · exact StepLe.inl (h.with_rip _)
    | jmp rel => exact StepLe.inl (h.with_rip _)
    | ldRsp8 d =>
      apply StepLe.okOf
      refine SLe_bind (h.reg 4) fun sp => SLe_bind (h.byte sp) fun v => wr_le h1 _ _ _
    | stRsp8 src =>
      apply StepLe.okOf
      refine SLe_bind (h.reg 4) fun sp => SLe_bind (rd_le h _ _) fun v => wrByte_le h1 _ _

end X86Min
