import Bootstrap.SporosX1
import Sporos.Check1

/-!
# Sporos stage 1: composition of the checker walk

The per-handler Hoare triples of `Bootstrap.SporosX1` composed into
the single-pass stack-effect walk of `Sporos.Check1`: `CodeSeq`
lays the encoded instructions at the cursor, `WalkInv` is the loop
invariant at `walk_top`, each `hand_*` lemma chains the dispatch
triple with the handler triple, and `fall_finish`/`none_finish_*`
run the tails and rebuild the invariant. `walk_loop` then runs the
whole word to `word_done` whenever `Sporos.Check1.walk` accepts.

Addresses are the concrete stage-1 ones of `SporosX1`; the decode
facts of the real image are bundled as `AllCode` hypotheses. Memory
layout (word table, ann array, code, stack) stays parametric.
-/

open X86Env X86Int X86Dec Sporos

namespace SporosX1

/-- All decode facts of the stage-1 image, at the concrete addresses. -/
structure AllCode (im : Nat → Option (Ins × Nat)) : Prop where
  l32 : L32Code im aL32
  annl : ANNLCode im aANNL aL32
  walk : WalkCode im aWalk aWordDone aPrim aLit aStr aPick aTor aFromr aRfetch
    aLocget aLocset aJmp aJz aCall aRet
  lit : LitCode im aLit aFallTail
  strc : StrCode im aStr aFallTail
  tor : TorCode im aTor aFallTail
  fromr : FromrCode im aFromr aFallTail
  rfetch : RfetchCode im aRfetch aFallTail
  pick : PickCode im aPick aFallTail aL32
  locget : LocgetCode im aLocget aFallTail aL32
  locset : LocsetCode im aLocset aFallTail aL32
  prim : PrimCode im aPrim aFallTail aNoneTail aEff
  ret : RetCode im aRet aNoneTail
  call : CallCode im aCall aFallTail aL32
  jmp : JmpCode im aJmp aNoneTail aL32 aANNL
  jz : JzCode im aJz aFallTail aL32 aANNL
  fall : FallCode im aFallTail aWalk aL32 aANNL
  none : NoneCode im aNoneTail aWalk aL32 aANNL

/-- Immediates that the checker reads as `u32` really are `u32`s. -/
def ImmOk : Instr → Prop
  | .str i | .pick i | .locget i | .locset i | .jmp i | .jz i | .jnz i | .call i =>
    i < 256 ^ 4
  | _ => True

/-- Every height `walk` meets stays far below `W`, so the machine's
modular arithmetic is the semantics'. Mirrors `walk`'s recursion. -/
def hBnd (pr : Prog) (wd : Word) : List Instr → Nat → Height → Prop
  | [], _, _ => True
  | ins :: rest, i, cur =>
    ((wd.ann.lookup i).getD cur).d + 2 < W ∧
    ((wd.ann.lookup i).getD cur).a + 1 < W ∧
    match hstep pr wd ((wd.ann.lookup i).getD cur) ins with
    | some (some h') => h'.d + 2 < W ∧ h'.a + 1 < W ∧ hBnd pr wd rest (i + 1) h'
    | some none => hBnd pr wd rest (i + 1) ((wd.ann.lookup (i + 1)).getD ⟨0, 0⟩)
    | none => True

theorem hBnd_head {pr : Prog} {wd : Word} {ins : Instr} {rest : List Instr}
    {i : Nat} {cur : Height} (hb : hBnd pr wd (ins :: rest) i cur) :
    ((wd.ann.lookup i).getD cur).d + 2 < W ∧
      ((wd.ann.lookup i).getD cur).a + 1 < W := by
  unfold hBnd at hb
  exact ⟨hb.1, hb.2.1⟩

theorem hBnd_some_some {pr : Prog} {wd : Word} {ins : Instr} {rest : List Instr}
    {i : Nat} {cur h' : Height} (hb : hBnd pr wd (ins :: rest) i cur)
    (hs : hstep pr wd ((wd.ann.lookup i).getD cur) ins = some (some h')) :
    h'.d + 2 < W ∧ h'.a + 1 < W ∧ hBnd pr wd rest (i + 1) h' := by
  unfold hBnd at hb
  rw [hs] at hb
  exact hb.2.2

theorem hBnd_some_none {pr : Prog} {wd : Word} {ins : Instr} {rest : List Instr}
    {i : Nat} {cur : Height} (hb : hBnd pr wd (ins :: rest) i cur)
    (hs : hstep pr wd ((wd.ann.lookup i).getD cur) ins = some none) :
    hBnd pr wd rest (i + 1) ((wd.ann.lookup (i + 1)).getD ⟨0, 0⟩) := by
  unfold hBnd at hb
  rw [hs] at hb
  exact hb.2.2

/-- Static layout facts for one word's walk. -/
structure WalkStatic (pr : Prog) (wd : Word) (w p sp tbl : Nat) : Prop where
  clenW : wd.code.length < W
  w56 : w + 56 ≤ W
  p0 : 0 < p
  pann : p + 12 * wd.ann.length < W
  sp32 : 32 ≤ sp
  spW : sp < W
  disjw : sp ≤ w ∨ w + 56 ≤ sp - 32
  disjp : sp ≤ p ∨ p + 12 * wd.ann.length ≤ sp - 32
  disjEff : aEff + 114 ≤ sp - 32 ∨ sp ≤ aEff
  wordsB : ∀ n, n < pr.words.length → tbl + n * 64 + 16 < W ∧
    (sp ≤ tbl + n * 64 ∨ tbl + n * 64 + 16 ≤ sp - 32)

/-- The walk invariant at `walk_top`: remaining code `code` at cursor
`c`, index `i`, machine height `hm` linked to `walk`'s `cur`. -/
structure WalkInv (pr : Prog) (wd : Word) (w p sp tbl : Nat) (s : X86Int.St)
    (code : List Instr) (i c : Nat) (cur hm : Height) : Prop where
  rip : s.rip = aWalk
  r8 : s.reg 8 = some hm.d
  r9 : s.reg 9 = some hm.a
  r6 : s.reg 6 = some c
  r4 : s.reg 4 = some sp
  r7 : s.reg 7 = some w
  r1 : s.reg 1 = some i
  r13 : s.reg 13 = some pr.words.length
  r14 : s.reg 14 = some tbl
  len16 : 16 ≤ s.regs.length
  hlink : hm = (wd.ann.lookup i).getD cur
  csq : CodeSeq s.mem c code
  isum : i + code.length = wd.code.length
  fnc : rd64 s.mem (w + 24) = some wd.code.length
  fnann : rd64 s.mem (w + 32) = some wd.ann.length
  fannp : rd64 s.mem (w + 48) = some p
  floc : rd64 s.mem (w + 16) = some wd.locals
  fouts : rd64 s.mem (w + 8) = some wd.outs
  lay : AnnLayout s wd.ann p
  room : StackRoom s 4
  eff : EffTable s
  words : ∀ n (hn : n < pr.words.length),
    rd64 s.mem (tbl + n * 64) = some (pr.words[n]).ins ∧
    rd64 s.mem (tbl + n * 64 + 8) = some (pr.words[n]).outs
  disjC : c + codeBytes code ≤ sp - 32 ∨ sp ≤ c
  cbound : c + codeBytes code < W
  imm : ∀ x ∈ code, ImmOk x
  hb : hBnd pr wd code i cur

/-- What a handler leaves for its tail, relative to the walk state. -/
def HPost (s s2 : X86Int.St) (h' : Height) (c' sp : Nat) : Prop :=
  s2.reg 8 = some h'.d ∧ s2.reg 9 = some h'.a ∧ s2.reg 6 = some c' ∧
  s2.reg 4 = some sp ∧ s2.reg 1 = s.reg 1 ∧ s2.reg 7 = s.reg 7 ∧
  s2.reg 13 = s.reg 13 ∧ s2.reg 14 = s.reg 14 ∧
  (∀ x m, x + m ≤ sp - 32 ∨ sp ≤ x → s2.mem.rdN x m = s.mem.rdN x m) ∧
  wshape s2.mem = wshape s.mem ∧ s2.regs.length = s.regs.length

theorem rd64_of_frame_of_disjw {m m' : Mem} {w sp off v : Nat}
    (hdisj : sp ≤ w ∨ w + 56 ≤ sp - 32)
    (hfr : ∀ x k, x + k ≤ sp - 32 ∨ sp ≤ x → m'.rdN x k = m.rdN x k)
    (hoff : off + 8 ≤ 56) (hv : rd64 m (w + off) = some v) :
    rd64 m' (w + off) = some v := by
  show (m'.rdN (w + off) 8).map leVal = some v
  rw [hfr (w + off) 8 (by
    cases hdisj with
    | inl h => exact Or.inr (by omega)
    | inr h => exact Or.inl (by omega))]
  exact hv

theorem fall_finish {im : Nat → Option (Ins × Nat)} (cfg : AllCode im)
    {pr : Prog} {wd : Word} {w p sp tbl : Nat} (st : WalkStatic pr wd w p sp tbl)
    {ins : Instr} {rest : List Instr} {i c len : Nat} {cur hm h' : Height}
    {s s1 s2 : X86Int.St} {k1 k2 : Nat}
    (inv : WalkInv pr wd w p sp tbl s (ins :: rest) i c cur hm)
    (hs : hstep pr wd hm ins = some (some h'))
    (hann : wd.ann.lookup (i + 1) = none ∨ wd.ann.lookup (i + 1) = some h')
    (hi1 : i + 1 < wd.code.length)
    (hlen : ins.enc.length = len)
    (hr1 : run im k1 (Out.ok s) = Out.ok s1)
    (hr2 : run im k2 (Out.ok s1) = Out.ok s2)
    (hrip2 : s2.rip = aFallTail)
    (hpost : HPost s s2 h' (c + len) sp) :
    ∃ k s3, run im k (Out.ok s) = Out.ok s3 ∧
      WalkInv pr wd w p sp tbl s3 rest (i + 1) (c + len) h' h' ∧
      (∀ x m, x + m ≤ sp - 32 ∨ sp ≤ x → s3.mem.rdN x m = s.mem.rdN x m) ∧
      wshape s3.mem = wshape s.mem := by
  obtain ⟨hs2r8, hs2r9, hs2r6, hs2r4, hs2r1, hs2r7, hs2r13, hs2r14, hfr2, hsh2,
    hlen2⟩ := hpost
  have hs0 : hstep pr wd ((wd.ann.lookup i).getD cur) ins = some (some h') := by
    rw [← inv.hlink]; exact hs
  obtain ⟨hb'd, hb'a, hbrest⟩ := hBnd_some_some inv.hb hs0
  have annls : ANNLSpec im aANNL aL32 := ANNL_triple cfg.annl cfg.l32 (by decide)
  have falls : FallSpec im aFallTail aWalk aL32 aANNL :=
    Fall_triple cfg.fall cfg.l32 annls (by decide)
  have hs2r7v : s2.reg 7 = some w := by rw [hs2r7]; exact inv.r7
  have hs2r1v : s2.reg 1 = some i := by rw [hs2r1]; exact inv.r1
  have hlen16 : 16 ≤ s2.regs.length := by rw [hlen2]; exact inv.len16
  have lay2 : AnnLayout s2 wd.ann p := AnnLayout_of_frame inv.lay (by
    intro x k hx1 hx2
    apply hfr2
    cases st.disjp with
    | inl h => exact Or.inr (by omega)
    | inr h => exact Or.inl (by omega))
  have room2 : StackRoom s2 4 :=
    StackRoom_of_wshape hsh2 (by rw [hs2r4, inv.r4]) inv.room
  obtain ⟨kt, s3, hrt, hrip3, hs3r1, hs3r8, hs3r9, hs3r6, hs3r4, hgen3, hfr3, hsh3⟩ :=
    falls wd h' (c + len) i w p sp hi1 st.clenW hann st.p0 st.w56 st.pann
      (by omega) (by omega) st.sp32 st.spW st.disjw st.disjp s2 hrip2 hs2r8 hs2r9
      hs2r6 hs2r4 hs2r7v hs2r1v hlen16
      (rd64_of_frame_of_disjw st.disjw hfr2 (by decide) inv.fannp)
      (rd64_of_frame_of_disjw st.disjw hfr2 (by decide) inv.fnann)
      (rd64_of_frame_of_disjw st.disjw hfr2 (by decide) inv.fnc)
      lay2 room2
  have frS : ∀ x m, x + m ≤ sp - 32 ∨ sp ≤ x → s3.mem.rdN x m = s.mem.rdN x m :=
    fun x m hx => (hfr3 x m hx).trans (hfr2 x m hx)
  have ecb : codeBytes (ins :: rest) = len + codeBytes rest := by
    show ins.enc.length + codeBytes rest = len + codeBytes rest
    rw [hlen]
  have hcs : s.mem.rdN c ins.enc.length = some ins.enc ∧
      CodeSeq s.mem (c + ins.enc.length) rest := inv.csq
  have htail : CodeSeq s.mem (c + len) rest := by rw [hlen] at hcs; exact hcs.2
  have csq3 : CodeSeq s3.mem (c + len) rest := CodeSeq_of_frame htail (by
    intro x k hx1 hx2
    apply frS
    have hdisj := inv.disjC
    rw [ecb] at hdisj
    cases hdisj with
    | inl h => exact Or.inl (by omega)
    | inr h => exact Or.inr (by omega))
  refine ⟨k1 + (k2 + kt), s3, by
      rw [run_add im k1 (k2 + kt), hr1, run_add im k2 kt, hr2, hrt],
    ⟨hrip3, hs3r8, hs3r9, hs3r6, hs3r4, ?_, hs3r1, ?_, ?_, ?_, ?_, csq3, ?_,
    rd64_of_frame_of_disjw st.disjw frS (by decide) inv.fnc,
    rd64_of_frame_of_disjw st.disjw frS (by decide) inv.fnann,
    rd64_of_frame_of_disjw st.disjw frS (by decide) inv.fannp,
    rd64_of_frame_of_disjw st.disjw frS (by decide) inv.floc,
    rd64_of_frame_of_disjw st.disjw frS (by decide) inv.fouts,
    ?_, ?_, ?_, ?_, ?_, ?_,
    fun x hx => inv.imm x (List.mem_cons_of_mem ins hx), hbrest⟩, frS,
    hsh3.trans hsh2⟩
  · rw [hgen3 7 (by decide) (by decide) (by decide) (by decide) (by decide)
      (by decide) (by decide) (by decide), hs2r7]; exact inv.r7
  · rw [hgen3 13 (by decide) (by decide) (by decide) (by decide) (by decide)
      (by decide) (by decide) (by decide), hs2r13]; exact inv.r13
  · rw [hgen3 14 (by decide) (by decide) (by decide) (by decide) (by decide)
      (by decide) (by decide) (by decide), hs2r14]; exact inv.r14
  · have e1 := run_regs_length kt hrt
    rw [e1, hlen2]; exact inv.len16
  · cases hann with
    | inl hn => simp [hn]
    | inr hn => simp [hn]
  · have h := inv.isum
    simp only [List.length_cons] at h
    omega
  · exact AnnLayout_of_frame inv.lay (by
      intro x k hx1 hx2
      apply frS
      cases st.disjp with
      | inl h => exact Or.inr (by omega)
      | inr h => exact Or.inl (by omega))
  · exact StackRoom_of_wshape (hsh3.trans hsh2) (by rw [hs3r4, inv.r4]) inv.room
  · intro q
    show (s3.mem.rdN (aEff + 2 * q.code) 2) = some [q.eff.1, q.eff.2]
    rw [frS (aEff + 2 * q.code) 2 (by
      have hq : q.code ≤ 56 := by have h57 := Prim_code_lt q; omega
      cases st.disjEff with
      | inl h => exact Or.inl (by omega)
      | inr h => exact Or.inr (by omega))]
    exact inv.eff q
  · intro n hn
    obtain ⟨h1, h2⟩ := inv.words n hn
    obtain ⟨hb1, hb2⟩ := st.wordsB n hn
    refine ⟨?_, ?_⟩
    · show (s3.mem.rdN (tbl + n * 64) 8).map leVal = some _
      rw [frS _ _ (by
        cases hb2 with
        | inl h => exact Or.inr h
        | inr h => exact Or.inl (by omega))]
      exact h1
    · show (s3.mem.rdN (tbl + n * 64 + 8) 8).map leVal = some _
      rw [frS _ _ (by
        cases hb2 with
        | inl h => exact Or.inr (by omega)
        | inr h => exact Or.inl (by omega))]
      exact h2
  · have hdisj := inv.disjC
    rw [ecb] at hdisj
    cases hdisj with
    | inl h => exact Or.inl (by omega)
    | inr h => exact Or.inr (by omega)
  · have hcb := inv.cbound
    rw [ecb] at hcb
    omega

theorem none_finish_cons {im : Nat → Option (Ins × Nat)} (cfg : AllCode im)
    {pr : Prog} {wd : Word} {w p sp tbl : Nat} (st : WalkStatic pr wd w p sp tbl)
    {ins : Instr} {rest : List Instr} {i c len : Nat} {cur hm : Height}
    {s s1 s2 : X86Int.St} {k1 k2 : Nat}
    (inv : WalkInv pr wd w p sp tbl s (ins :: rest) i c cur hm)
    (hs : hstep pr wd hm ins = some none)
    (hi1 : i + 1 < wd.code.length)
    (hlen : ins.enc.length = len)
    (hr1 : run im k1 (Out.ok s) = Out.ok s1)
    (hr2 : run im k2 (Out.ok s1) = Out.ok s2)
    (hrip2 : s2.rip = aNoneTail)
    (hpost : HPost s s2 hm (c + len) sp) :
    ∃ k s3, run im k (Out.ok s) = Out.ok s3 ∧
      WalkInv pr wd w p sp tbl s3 rest (i + 1) (c + len)
        ((wd.ann.lookup (i + 1)).getD ⟨0, 0⟩)
        ((wd.ann.lookup (i + 1)).getD ⟨0, 0⟩) ∧
      (∀ x m, x + m ≤ sp - 32 ∨ sp ≤ x → s3.mem.rdN x m = s.mem.rdN x m) ∧
      wshape s3.mem = wshape s.mem := by
  obtain ⟨hs2r8, hs2r9, hs2r6, hs2r4, hs2r1, hs2r7, hs2r13, hs2r14, hfr2, hsh2,
    hlen2⟩ := hpost
  have hs0 : hstep pr wd ((wd.ann.lookup i).getD cur) ins = some none := by
    rw [← inv.hlink]; exact hs
  have hbrest := hBnd_some_none inv.hb hs0
  have hhead := hBnd_head inv.hb
  have hge : ((wd.ann.lookup i).getD cur) = hm := inv.hlink.symm
  rw [hge] at hhead
  have annls : ANNLSpec im aANNL aL32 := ANNL_triple cfg.annl cfg.l32 (by decide)
  have nones : NoneSpec im aNoneTail aWalk aL32 aANNL :=
    None_triple cfg.none cfg.l32 annls (by decide)
  have hs2r7v : s2.reg 7 = some w := by rw [hs2r7]; exact inv.r7
  have hs2r1v : s2.reg 1 = some i := by rw [hs2r1]; exact inv.r1
  have hlen16 : 16 ≤ s2.regs.length := by rw [hlen2]; exact inv.len16
  have lay2 : AnnLayout s2 wd.ann p := AnnLayout_of_frame inv.lay (by
    intro x k hx1 hx2
    apply hfr2
    cases st.disjp with
    | inl h => exact Or.inr (by omega)
    | inr h => exact Or.inl (by omega))
  have room2 : StackRoom s2 4 :=
    StackRoom_of_wshape hsh2 (by rw [hs2r4, inv.r4]) inv.room
  obtain ⟨kt, s3, hrt, hrip3, hs3r1, hs3r8, hs3r9, hs3r6, hs3r4, hgen3, hfr3, hsh3⟩ :=
    nones wd hm ((wd.ann.lookup (i + 1)).getD ⟨0, 0⟩) (c + len) i w p sp
      (by have hcl := st.clenW; omega) st.clenW (Or.inr ⟨hi1, rfl⟩) st.p0 st.w56 st.pann
      (by have h1 := hhead.1; omega) (by have h2 := hhead.2; omega) st.sp32 st.spW st.disjw st.disjp s2 hrip2 hs2r8 hs2r9
      hs2r6 hs2r4 hs2r7v hs2r1v hlen16
      (rd64_of_frame_of_disjw st.disjw hfr2 (by decide) inv.fannp)
      (rd64_of_frame_of_disjw st.disjw hfr2 (by decide) inv.fnann)
      (rd64_of_frame_of_disjw st.disjw hfr2 (by decide) inv.fnc)
      lay2 room2
  have frS : ∀ x m, x + m ≤ sp - 32 ∨ sp ≤ x → s3.mem.rdN x m = s.mem.rdN x m :=
    fun x m hx => (hfr3 x m hx).trans (hfr2 x m hx)
  have ecb : codeBytes (ins :: rest) = len + codeBytes rest := by
    show ins.enc.length + codeBytes rest = len + codeBytes rest
    rw [hlen]
  have hcs : s.mem.rdN c ins.enc.length = some ins.enc ∧
      CodeSeq s.mem (c + ins.enc.length) rest := inv.csq
  have htail : CodeSeq s.mem (c + len) rest := by rw [hlen] at hcs; exact hcs.2
  have csq3 : CodeSeq s3.mem (c + len) rest := CodeSeq_of_frame htail (by
    intro x k hx1 hx2
    apply frS
    have hdisj := inv.disjC
    rw [ecb] at hdisj
    cases hdisj with
    | inl h => exact Or.inl (by omega)
    | inr h => exact Or.inr (by omega))
  refine ⟨k1 + (k2 + kt), s3, by
      rw [run_add im k1 (k2 + kt), hr1, run_add im k2 kt, hr2, hrt],
    ⟨hrip3, hs3r8, hs3r9, hs3r6, hs3r4, ?_, hs3r1, ?_, ?_, ?_, ?_, csq3, ?_,
    rd64_of_frame_of_disjw st.disjw frS (by decide) inv.fnc,
    rd64_of_frame_of_disjw st.disjw frS (by decide) inv.fnann,
    rd64_of_frame_of_disjw st.disjw frS (by decide) inv.fannp,
    rd64_of_frame_of_disjw st.disjw frS (by decide) inv.floc,
    rd64_of_frame_of_disjw st.disjw frS (by decide) inv.fouts,
    ?_, ?_, ?_, ?_, ?_, ?_,
    fun x hx => inv.imm x (List.mem_cons_of_mem ins hx), hbrest⟩, frS,
    hsh3.trans hsh2⟩
  · rw [hgen3 7 (by decide) (by decide) (by decide) (by decide) (by decide)
      (by decide) (by decide) (by decide), hs2r7]; exact inv.r7
  · rw [hgen3 13 (by decide) (by decide) (by decide) (by decide) (by decide)
      (by decide) (by decide) (by decide), hs2r13]; exact inv.r13
  · rw [hgen3 14 (by decide) (by decide) (by decide) (by decide) (by decide)
      (by decide) (by decide) (by decide), hs2r14]; exact inv.r14
  · have e1 := run_regs_length kt hrt
    rw [e1, hlen2]; exact inv.len16
  · cases hl : wd.ann.lookup (i + 1) <;> simp [hl]
  · have h := inv.isum
    simp only [List.length_cons] at h
    omega
  · exact AnnLayout_of_frame inv.lay (by
      intro x k hx1 hx2
      apply frS
      cases st.disjp with
      | inl h => exact Or.inr (by omega)
      | inr h => exact Or.inl (by omega))
  · exact StackRoom_of_wshape (hsh3.trans hsh2) (by rw [hs3r4, inv.r4]) inv.room
  · intro q
    show (s3.mem.rdN (aEff + 2 * q.code) 2) = some [q.eff.1, q.eff.2]
    rw [frS (aEff + 2 * q.code) 2 (by
      have hq : q.code ≤ 56 := by have h57 := Prim_code_lt q; omega
      cases st.disjEff with
      | inl h => exact Or.inl (by omega)
      | inr h => exact Or.inr (by omega))]
    exact inv.eff q
  · intro n hn
    obtain ⟨h1, h2⟩ := inv.words n hn
    obtain ⟨hb1, hb2⟩ := st.wordsB n hn
    refine ⟨?_, ?_⟩
    · show (s3.mem.rdN (tbl + n * 64) 8).map leVal = some _
      rw [frS _ _ (by
        cases hb2 with
        | inl h => exact Or.inr h
        | inr h => exact Or.inl (by omega))]
      exact h1
    · show (s3.mem.rdN (tbl + n * 64 + 8) 8).map leVal = some _
      rw [frS _ _ (by
        cases hb2 with
        | inl h => exact Or.inr (by omega)
        | inr h => exact Or.inl (by omega))]
      exact h2
  · have hdisj := inv.disjC
    rw [ecb] at hdisj
    cases hdisj with
    | inl h => exact Or.inl (by omega)
    | inr h => exact Or.inr (by omega)
  · have hcb := inv.cbound
    rw [ecb] at hcb
    omega

theorem none_finish_nil {im : Nat → Option (Ins × Nat)} (cfg : AllCode im)
    {pr : Prog} {wd : Word} {w p sp tbl : Nat} (st : WalkStatic pr wd w p sp tbl)
    {ins : Instr} {i c len : Nat} {cur hm : Height}
    {s s1 s2 : X86Int.St} {k1 k2 : Nat}
    (inv : WalkInv pr wd w p sp tbl s [ins] i c cur hm)
    (hs : hstep pr wd hm ins = some none)
    (hlen : ins.enc.length = len)
    (hr1 : run im k1 (Out.ok s) = Out.ok s1)
    (hr2 : run im k2 (Out.ok s1) = Out.ok s2)
    (hrip2 : s2.rip = aNoneTail)
    (hpost : HPost s s2 hm (c + len) sp) :
    ∃ k s4, run im k (Out.ok s) = Out.ok s4 ∧ s4.rip = aWordDone ∧
      s4.reg 1 = some (i + 1) ∧ s4.reg 6 = some (c + len) ∧
      s4.reg 7 = some w ∧ s4.reg 4 = some sp ∧
      (∀ x m, x + m ≤ sp - 32 ∨ sp ≤ x → s4.mem.rdN x m = s.mem.rdN x m) ∧
      wshape s4.mem = wshape s.mem := by
  obtain ⟨hs2r8, hs2r9, hs2r6, hs2r4, hs2r1, hs2r7, hs2r13, hs2r14, hfr2, hsh2,
    hlen2⟩ := hpost
  have hs0 : hstep pr wd ((wd.ann.lookup i).getD cur) ins = some none := by
    rw [← inv.hlink]; exact hs
  have hhead := hBnd_head inv.hb
  have hge : ((wd.ann.lookup i).getD cur) = hm := inv.hlink.symm
  rw [hge] at hhead
  have hisum : wd.code.length ≤ i + 1 := by
    have h := inv.isum
    simp only [List.length_cons, List.length_nil] at h
    omega
  have hiW : i + 1 < W := by
    have h := inv.isum
    have hcl := st.clenW
    simp only [List.length_cons, List.length_nil] at h
    omega
  have annls : ANNLSpec im aANNL aL32 := ANNL_triple cfg.annl cfg.l32 (by decide)
  have nones : NoneSpec im aNoneTail aWalk aL32 aANNL :=
    None_triple cfg.none cfg.l32 annls (by decide)
  have hs2r7v : s2.reg 7 = some w := by rw [hs2r7]; exact inv.r7
  have hs2r1v : s2.reg 1 = some i := by rw [hs2r1]; exact inv.r1
  have hlen16 : 16 ≤ s2.regs.length := by rw [hlen2]; exact inv.len16
  have lay2 : AnnLayout s2 wd.ann p := AnnLayout_of_frame inv.lay (by
    intro x k hx1 hx2
    apply hfr2
    cases st.disjp with
    | inl h => exact Or.inr (by omega)
    | inr h => exact Or.inl (by omega))
  have room2 : StackRoom s2 4 :=
    StackRoom_of_wshape hsh2 (by rw [hs2r4, inv.r4]) inv.room
  obtain ⟨kt, s3, hrt, hrip3, hs3r1, hs3r8, hs3r9, hs3r6, hs3r4, hgen3, hfr3, hsh3⟩ :=
    nones wd hm hm (c + len) i w p sp hiW st.clenW (Or.inl ⟨hisum, rfl⟩)
      st.p0 st.w56 st.pann (by have h1 := hhead.1; omega) (by have h2 := hhead.2; omega) st.sp32 st.spW st.disjw st.disjp
      s2 hrip2 hs2r8 hs2r9 hs2r6 hs2r4 hs2r7v hs2r1v hlen16
      (rd64_of_frame_of_disjw st.disjw hfr2 (by decide) inv.fannp)
      (rd64_of_frame_of_disjw st.disjw hfr2 (by decide) inv.fnann)
      (rd64_of_frame_of_disjw st.disjw hfr2 (by decide) inv.fnc)
      lay2 room2
  have frS : ∀ x m, x + m ≤ sp - 32 ∨ sp ≤ x → s3.mem.rdN x m = s.mem.rdN x m :=
    fun x m hx => (hfr3 x m hx).trans (hfr2 x m hx)
  have hs3r7v : s3.reg 7 = some w := by
    rw [hgen3 7 (by decide) (by decide) (by decide) (by decide) (by decide)
      (by decide) (by decide) (by decide), hs2r7]; exact inv.r7
  have hlen163 : 16 ≤ s3.regs.length := by
    have e1 := run_regs_length kt hrt
    rw [e1, hlen2]; exact inv.len16
  have hnotlt : ¬ i + 1 < wd.code.length := by omega
  obtain ⟨k4, s4, hr4, hrip4, hr04, hr14, hr64, hr74, hr84, hr94, hr44, hg4, hm4⟩ :=
    WalkDone_triple cfg.walk (by decide) hm (c + len) (i + 1) w sp wd.code.length
      st.w56 hnotlt s3 hrip3 hs3r8 hs3r9 hs3r6 hs3r4 hs3r7v hs3r1 hlen163
      (rd64_of_frame_of_disjw st.disjw frS (by decide) inv.fnc)
  refine ⟨k1 + (k2 + (kt + k4)), s4, by
    rw [run_add im k1 (k2 + (kt + k4)), hr1, run_add im k2 (kt + k4), hr2,
      run_add im kt k4, hrt, hr4], hrip4, hr14, hr64, hr74, hr44, ?_, ?_⟩
  · intro x m hx
    rw [hm4]; exact frS x m hx
  · rw [hm4]; exact hsh3.trans hsh2

theorem hand_lit {im : Nat → Option (Ins × Nat)} (cfg : AllCode im)
    {pr : Prog} {wd : Word} {w p sp tbl : Nat} (st : WalkStatic pr wd w p sp tbl)
    {n : Nat} {rest : List Instr} {i c : Nat} {cur hm : Height} {s : X86Int.St}
    (inv : WalkInv pr wd w p sp tbl s (Instr.lit n :: rest) i c cur hm) :
    ∃ k s2, run im k (Out.ok s) = Out.ok s2 ∧ s2.rip = aFallTail ∧
      HPost s s2 ⟨hm.d + 1, hm.a⟩ (c + 9) sp := by
  have hge : ((wd.ann.lookup i).getD cur) = hm := inv.hlink.symm
  have hbhead := hBnd_head inv.hb
  rw [hge] at hbhead
  have hltlen : i < wd.code.length := by
    have h := inv.isum
    simp only [List.length_cons] at h
    omega
  have hcW : c < W := by
    have h := inv.cbound
    have ecb0 : codeBytes (Instr.lit n :: rest) =
        (Instr.lit n).enc.length + codeBytes rest := rfl
    rw [ecb0] at h
    omega
  have hclen : c + 9 < W := by
    have h := inv.cbound
    have ecb0 : codeBytes (Instr.lit n :: rest) =
        (Instr.lit n).enc.length + codeBytes rest := rfl
    rw [ecb0] at h
    have hlen : (Instr.lit n).enc.length = 9 := rfl
    rw [hlen] at h
    omega
  have hcs : s.mem.rdN c (Instr.lit n).enc.length = some (Instr.lit n).enc ∧
      CodeSeq s.mem (c + (Instr.lit n).enc.length) rest := inv.csq
  have h9 : s.mem.rdN c 9 = some (0x80 :: leBytes n 8) := hcs.1
  have hbyte : s.mem.rd c = some 128 := (rdN_cons h9).1
  have hs : hstep pr wd hm (Instr.lit n) = some (some ⟨hm.d + 1, hm.a⟩) := rfl
  obtain ⟨k1, s1, hr1, hrip1, -, hr11, hr61, hr71, hr81, hr91, hr41, hg1, hm1⟩ :=
    WalkLit_triple cfg.walk (by decide) hm c i w sp wd.code.length st.w56 hltlen
      hcW s inv.rip inv.r8 inv.r9 inv.r6 inv.r4 inv.r7 inv.r1 inv.len16 inv.fnc
      hbyte
  obtain ⟨k2, s2, hr2, hrip2, hfall2, -, hr62, hm2, hg2⟩ :=
    (Lit_tripleF cfg.lit pr wd n) hm c (Option.isSome_iff_exists.mpr ⟨_, hs⟩)
      hbhead.1 hbhead.2 hclen s1 hrip1
      ⟨hr81, hr91, hr61, by rw [hm1]; exact Option.isSome_iff_exists.mpr ⟨_, h9⟩⟩
  obtain ⟨hs2r8, hs2r9⟩ := hfall2 _ hs
  refine ⟨k1 + k2, s2, by rw [run_add im k1 k2, hr1, hr2], hrip2,
    hs2r8, hs2r9, hr62,
    by rw [hg2 4 (by decide) (by decide) (by decide), hg1 4 (by decide)]; exact inv.r4,
    by rw [hg2 1 (by decide) (by decide) (by decide), hg1 1 (by decide)],
    by rw [hg2 7 (by decide) (by decide) (by decide), hg1 7 (by decide)],
    by rw [hg2 13 (by decide) (by decide) (by decide), hg1 13 (by decide)],
    by rw [hg2 14 (by decide) (by decide) (by decide), hg1 14 (by decide)],
    by intro x m _; rw [hm2, hm1],
    by rw [hm2, hm1],
    (run_regs_length k2 hr2).trans (run_regs_length k1 hr1)⟩

theorem hand_tor {im : Nat → Option (Ins × Nat)} (cfg : AllCode im)
    {pr : Prog} {wd : Word} {w p sp tbl : Nat} (st : WalkStatic pr wd w p sp tbl)
    {rest : List Instr} {i c : Nat} {cur hm : Height} {s : X86Int.St}
    (inv : WalkInv pr wd w p sp tbl s (Instr.tor :: rest) i c cur hm)
    (hs : hstep pr wd hm Instr.tor = some (some ⟨hm.d - 1, hm.a + 1⟩)) :
    ∃ k s2, run im k (Out.ok s) = Out.ok s2 ∧ s2.rip = aFallTail ∧
      HPost s s2 ⟨hm.d - 1, hm.a + 1⟩ (c + 1) sp := by
  have hge : ((wd.ann.lookup i).getD cur) = hm := inv.hlink.symm
  have hbhead := hBnd_head inv.hb
  rw [hge] at hbhead
  have hltlen : i < wd.code.length := by
    have h := inv.isum
    simp only [List.length_cons] at h
    omega
  have hcW : c < W := by
    have h := inv.cbound
    have ecb0 : codeBytes (Instr.tor :: rest) =
        Instr.tor.enc.length + codeBytes rest := rfl
    rw [ecb0] at h
    omega
  have hclen : c + 1 < W := by
    have h := inv.cbound
    have ecb0 : codeBytes (Instr.tor :: rest) =
        Instr.tor.enc.length + codeBytes rest := rfl
    rw [ecb0] at h
    have hlen : Instr.tor.enc.length = 1 := rfl
    rw [hlen] at h
    omega
  have hcs : s.mem.rdN c Instr.tor.enc.length = some Instr.tor.enc ∧
      CodeSeq s.mem (c + Instr.tor.enc.length) rest := inv.csq
  have h1b : s.mem.rdN c 1 = some [131] := hcs.1
  have hbyte : s.mem.rd c = some 131 := (rdN_cons h1b).1
  obtain ⟨k1, s1, hr1, hrip1, -, hr11, hr61, hr71, hr81, hr91, hr41, hg1, hm1⟩ :=
    WalkTor_triple cfg.walk (by decide) hm c i w sp wd.code.length st.w56 hltlen
      hcW s inv.rip inv.r8 inv.r9 inv.r6 inv.r4 inv.r7 inv.r1 inv.len16 inv.fnc
      hbyte
  obtain ⟨k2, s2, hr2, hrip2, hfall2, -, hr62, hm2, hg2⟩ :=
    (Tor_tripleF cfg.tor pr wd) hm c (Option.isSome_iff_exists.mpr ⟨_, hs⟩)
      hbhead.1 hbhead.2 hclen s1 hrip1
      ⟨hr81, hr91, hr61, by rw [hm1]; exact Option.isSome_iff_exists.mpr ⟨_, h1b⟩⟩
  obtain ⟨hs2r8, hs2r9⟩ := hfall2 _ hs
  refine ⟨k1 + k2, s2, by rw [run_add im k1 k2, hr1, hr2], hrip2,
    hs2r8, hs2r9, hr62,
    by rw [hg2 4 (by decide) (by decide) (by decide), hg1 4 (by decide)]; exact inv.r4,
    by rw [hg2 1 (by decide) (by decide) (by decide), hg1 1 (by decide)],
    by rw [hg2 7 (by decide) (by decide) (by decide), hg1 7 (by decide)],
    by rw [hg2 13 (by decide) (by decide) (by decide), hg1 13 (by decide)],
    by rw [hg2 14 (by decide) (by decide) (by decide), hg1 14 (by decide)],
    by intro x m _; rw [hm2, hm1],
    by rw [hm2, hm1],
    (run_regs_length k2 hr2).trans (run_regs_length k1 hr1)⟩

theorem c_lt_of_inv {pr : Prog} {wd : Word} {w p sp tbl : Nat} {s : X86Int.St}
    {ins : Instr} {rest : List Instr} {i c : Nat} {cur hm : Height}
    (inv : WalkInv pr wd w p sp tbl s (ins :: rest) i c cur hm) : c < W := by
  have h := inv.cbound
  have ecb0 : codeBytes (ins :: rest) = ins.enc.length + codeBytes rest := rfl
  rw [ecb0] at h
  omega

theorem c_add_len_lt {pr : Prog} {wd : Word} {w p sp tbl : Nat} {s : X86Int.St}
    {ins : Instr} {rest : List Instr} {i c len : Nat} {cur hm : Height}
    (inv : WalkInv pr wd w p sp tbl s (ins :: rest) i c cur hm)
    (hlen : ins.enc.length = len) : c + len < W := by
  have h := inv.cbound
  have ecb0 : codeBytes (ins :: rest) = ins.enc.length + codeBytes rest := rfl
  rw [ecb0, hlen] at h
  omega

theorem i_lt_of_inv {pr : Prog} {wd : Word} {w p sp tbl : Nat} {s : X86Int.St}
    {ins : Instr} {rest : List Instr} {i c : Nat} {cur hm : Height}
    (inv : WalkInv pr wd w p sp tbl s (ins :: rest) i c cur hm) :
    i < wd.code.length := by
  have h := inv.isum
  simp only [List.length_cons] at h
  omega

theorem disjC5_of_inv {pr : Prog} {wd : Word} {w p sp tbl : Nat} {s : X86Int.St}
    {ins : Instr} {rest : List Instr} {i c : Nat} {cur hm : Height}
    (inv : WalkInv pr wd w p sp tbl s (ins :: rest) i c cur hm)
    (hlen : ins.enc.length = 5) :
    (sp ≤ c + 1 ∨ c + 5 ≤ sp - 32) ∧ (sp ≤ c + 1 ∨ c + 5 ≤ sp - 8) := by
  have h := inv.disjC
  have ecb0 : codeBytes (ins :: rest) = ins.enc.length + codeBytes rest := rfl
  rw [ecb0, hlen] at h
  cases h with
  | inl h => exact ⟨Or.inr (by omega), Or.inr (by omega)⟩
  | inr h => exact ⟨Or.inl (by omega), Or.inl (by omega)⟩

theorem hand_str {im : Nat → Option (Ins × Nat)} (cfg : AllCode im)
    {pr : Prog} {wd : Word} {w p sp tbl : Nat} (st : WalkStatic pr wd w p sp tbl)
    {n : Nat} {rest : List Instr} {i c : Nat} {cur hm : Height} {s : X86Int.St}
    (inv : WalkInv pr wd w p sp tbl s (Instr.str n :: rest) i c cur hm) :
    ∃ k s2, run im k (Out.ok s) = Out.ok s2 ∧ s2.rip = aFallTail ∧
      HPost s s2 ⟨hm.d + 2, hm.a⟩ (c + 5) sp := by
  have hbhead : hm.d + 2 < W ∧ hm.a + 1 < W := by
    have h1 := hBnd_head inv.hb
    rwa [inv.hlink.symm] at h1
  have hcs : s.mem.rdN c (Instr.str n).enc.length = some (Instr.str n).enc ∧
      CodeSeq s.mem (c + (Instr.str n).enc.length) rest := inv.csq
  have h5 : s.mem.rdN c 5 = some (0x81 :: leBytes n 4) := hcs.1
  have hbyte : s.mem.rd c = some 129 := (rdN_cons h5).1
  have hs : hstep pr wd hm (Instr.str n) = some (some ⟨hm.d + 2, hm.a⟩) := rfl
  obtain ⟨k1, s1, hr1, hrip1, -, hr11, hr61, hr71, hr81, hr91, hr41, hg1, hm1⟩ :=
    WalkStr_triple cfg.walk (by decide) hm c i w sp wd.code.length st.w56
      (i_lt_of_inv inv) (c_lt_of_inv inv) s inv.rip inv.r8 inv.r9 inv.r6 inv.r4
      inv.r7 inv.r1 inv.len16 inv.fnc hbyte
  obtain ⟨k2, s2, hr2, hrip2, hfall2, -, hr62, hm2, hg2⟩ :=
    (Str_tripleF cfg.strc pr wd n) hm c (Option.isSome_iff_exists.mpr ⟨_, hs⟩)
      hbhead.1 hbhead.2 (c_add_len_lt inv rfl) s1 hrip1
      ⟨hr81, hr91, hr61, by rw [hm1]; exact Option.isSome_iff_exists.mpr ⟨_, h5⟩⟩
  obtain ⟨hs2r8, hs2r9⟩ := hfall2 _ hs
  refine ⟨k1 + k2, s2, by rw [run_add im k1 k2, hr1, hr2], hrip2,
    hs2r8, hs2r9, hr62,
    by rw [hg2 4 (by decide) (by decide) (by decide), hg1 4 (by decide)]; exact inv.r4,
    by rw [hg2 1 (by decide) (by decide) (by decide), hg1 1 (by decide)],
    by rw [hg2 7 (by decide) (by decide) (by decide), hg1 7 (by decide)],
    by rw [hg2 13 (by decide) (by decide) (by decide), hg1 13 (by decide)],
    by rw [hg2 14 (by decide) (by decide) (by decide), hg1 14 (by decide)],
    by intro x m _; rw [hm2, hm1],
    by rw [hm2, hm1],
    (run_regs_length k2 hr2).trans (run_regs_length k1 hr1)⟩

theorem hand_fromr {im : Nat → Option (Ins × Nat)} (cfg : AllCode im)
    {pr : Prog} {wd : Word} {w p sp tbl : Nat} (st : WalkStatic pr wd w p sp tbl)
    {rest : List Instr} {i c : Nat} {cur hm : Height} {s : X86Int.St}
    (inv : WalkInv pr wd w p sp tbl s (Instr.fromr :: rest) i c cur hm)
    (hs : hstep pr wd hm Instr.fromr = some (some ⟨hm.d + 1, hm.a - 1⟩)) :
    ∃ k s2, run im k (Out.ok s) = Out.ok s2 ∧ s2.rip = aFallTail ∧
      HPost s s2 ⟨hm.d + 1, hm.a - 1⟩ (c + 1) sp := by
  have hbhead : hm.d + 2 < W ∧ hm.a + 1 < W := by
    have h1 := hBnd_head inv.hb
    rwa [inv.hlink.symm] at h1
  have hcs : s.mem.rdN c Instr.fromr.enc.length = some Instr.fromr.enc ∧
      CodeSeq s.mem (c + Instr.fromr.enc.length) rest := inv.csq
  have h1b : s.mem.rdN c 1 = some [132] := hcs.1
  have hbyte : s.mem.rd c = some 132 := (rdN_cons h1b).1
  obtain ⟨k1, s1, hr1, hrip1, -, hr11, hr61, hr71, hr81, hr91, hr41, hg1, hm1⟩ :=
    WalkFromr_triple cfg.walk (by decide) hm c i w sp wd.code.length st.w56
      (i_lt_of_inv inv) (c_lt_of_inv inv) s inv.rip inv.r8 inv.r9 inv.r6 inv.r4
      inv.r7 inv.r1 inv.len16 inv.fnc hbyte
  obtain ⟨k2, s2, hr2, hrip2, hfall2, -, hr62, hm2, hg2⟩ :=
    (Fromr_tripleF cfg.fromr pr wd) hm c (Option.isSome_iff_exists.mpr ⟨_, hs⟩)
      hbhead.1 hbhead.2 (c_add_len_lt inv rfl) s1 hrip1
      ⟨hr81, hr91, hr61, by rw [hm1]; exact Option.isSome_iff_exists.mpr ⟨_, h1b⟩⟩
  obtain ⟨hs2r8, hs2r9⟩ := hfall2 _ hs
  refine ⟨k1 + k2, s2, by rw [run_add im k1 k2, hr1, hr2], hrip2,
    hs2r8, hs2r9, hr62,
    by rw [hg2 4 (by decide) (by decide) (by decide), hg1 4 (by decide)]; exact inv.r4,
    by rw [hg2 1 (by decide) (by decide) (by decide), hg1 1 (by decide)],
    by rw [hg2 7 (by decide) (by decide) (by decide), hg1 7 (by decide)],
    by rw [hg2 13 (by decide) (by decide) (by decide), hg1 13 (by decide)],
    by rw [hg2 14 (by decide) (by decide) (by decide), hg1 14 (by decide)],
    by intro x m _; rw [hm2, hm1],
    by rw [hm2, hm1],
    (run_regs_length k2 hr2).trans (run_regs_length k1 hr1)⟩

theorem hand_rfetch {im : Nat → Option (Ins × Nat)} (cfg : AllCode im)
    {pr : Prog} {wd : Word} {w p sp tbl : Nat} (st : WalkStatic pr wd w p sp tbl)
    {rest : List Instr} {i c : Nat} {cur hm : Height} {s : X86Int.St}
    (inv : WalkInv pr wd w p sp tbl s (Instr.rfetch :: rest) i c cur hm)
    (hs : hstep pr wd hm Instr.rfetch = some (some ⟨hm.d + 1, hm.a⟩)) :
    ∃ k s2, run im k (Out.ok s) = Out.ok s2 ∧ s2.rip = aFallTail ∧
      HPost s s2 ⟨hm.d + 1, hm.a⟩ (c + 1) sp := by
  have hbhead : hm.d + 2 < W ∧ hm.a + 1 < W := by
    have h1 := hBnd_head inv.hb
    rwa [inv.hlink.symm] at h1
  have hcs : s.mem.rdN c Instr.rfetch.enc.length = some Instr.rfetch.enc ∧
      CodeSeq s.mem (c + Instr.rfetch.enc.length) rest := inv.csq
  have h1b : s.mem.rdN c 1 = some [133] := hcs.1
  have hbyte : s.mem.rd c = some 133 := (rdN_cons h1b).1
  obtain ⟨k1, s1, hr1, hrip1, -, hr11, hr61, hr71, hr81, hr91, hr41, hg1, hm1⟩ :=
    WalkRfetch_triple cfg.walk (by decide) hm c i w sp wd.code.length st.w56
      (i_lt_of_inv inv) (c_lt_of_inv inv) s inv.rip inv.r8 inv.r9 inv.r6 inv.r4
      inv.r7 inv.r1 inv.len16 inv.fnc hbyte
  obtain ⟨k2, s2, hr2, hrip2, hfall2, -, hr62, hm2, hg2⟩ :=
    (Rfetch_tripleF cfg.rfetch pr wd) hm c (Option.isSome_iff_exists.mpr ⟨_, hs⟩)
      hbhead.1 hbhead.2 (c_add_len_lt inv rfl) s1 hrip1
      ⟨hr81, hr91, hr61, by rw [hm1]; exact Option.isSome_iff_exists.mpr ⟨_, h1b⟩⟩
  obtain ⟨hs2r8, hs2r9⟩ := hfall2 _ hs
  refine ⟨k1 + k2, s2, by rw [run_add im k1 k2, hr1, hr2], hrip2,
    hs2r8, hs2r9, hr62,
    by rw [hg2 4 (by decide) (by decide) (by decide), hg1 4 (by decide)]; exact inv.r4,
    by rw [hg2 1 (by decide) (by decide) (by decide), hg1 1 (by decide)],
    by rw [hg2 7 (by decide) (by decide) (by decide), hg1 7 (by decide)],
    by rw [hg2 13 (by decide) (by decide) (by decide), hg1 13 (by decide)],
    by rw [hg2 14 (by decide) (by decide) (by decide), hg1 14 (by decide)],
    by intro x m _; rw [hm2, hm1],
    by rw [hm2, hm1],
    (run_regs_length k2 hr2).trans (run_regs_length k1 hr1)⟩

theorem hand_pick {im : Nat → Option (Ins × Nat)} (cfg : AllCode im)
    {pr : Prog} {wd : Word} {w p sp tbl : Nat} (st : WalkStatic pr wd w p sp tbl)
    {n : Nat} {rest : List Instr} {i c : Nat} {cur hm : Height} {s : X86Int.St}
    (inv : WalkInv pr wd w p sp tbl s (Instr.pick n :: rest) i c cur hm)
    (hs : hstep pr wd hm (Instr.pick n) = some (some ⟨hm.d + 1, hm.a⟩)) :
    ∃ k s2, run im k (Out.ok s) = Out.ok s2 ∧ s2.rip = aFallTail ∧
      HPost s s2 ⟨hm.d + 1, hm.a⟩ (c + 5) sp := by
  have hbhead : hm.d + 2 < W ∧ hm.a + 1 < W := by
    have h1 := hBnd_head inv.hb
    rwa [inv.hlink.symm] at h1
  have hnlt : n < hm.d := (hstep_pick hs).1
  have himmok : n < 256 ^ 4 := inv.imm _ (List.mem_cons_self)
  have hcs : s.mem.rdN c (Instr.pick n).enc.length = some (Instr.pick n).enc ∧
      CodeSeq s.mem (c + (Instr.pick n).enc.length) rest := inv.csq
  have h5 : s.mem.rdN c 5 = some (0x82 :: leBytes n 4) := hcs.1
  have hbyte : s.mem.rd c = some 130 := (rdN_cons h5).1
  have hw8s : ∀ v, (wr64 s.mem (sp - 8) v).isSome := by
    intro v
    have hh := inv.room sp inv.r4 0 (by decide) v
    rwa [slot_sub (k := 0 + 1) (by have h32 := st.sp32; omega) st.spW] at hh
  obtain ⟨k1, s1, hr1, hrip1, -, hr11, hr61, hr71, hr81, hr91, hr41, hg1, hm1⟩ :=
    WalkPick_triple cfg.walk (by decide) hm c i w sp wd.code.length st.w56
      (i_lt_of_inv inv) (c_lt_of_inv inv) s inv.rip inv.r8 inv.r9 inv.r6 inv.r4
      inv.r7 inv.r1 inv.len16 inv.fnc hbyte
  have himm : (s1.mem.rdN (c + 1) 4).map leVal = some n := by
    rw [hm1]
    obtain ⟨-, himm0⟩ := imm32_of_enc h5
    rwa [Nat.mod_eq_of_lt himmok] at himm0
  have hframeL : ∀ m, wr64 s1.mem (sp - 8) (aPick + 12) = some m →
      m.rdN (c + 1) 4 = s1.mem.rdN (c + 1) 4 := by
    intro m hmwr
    exact rdN_wr64_other hmwr (by
      have h2 := (disjC5_of_inv inv rfl).1
      have h32 := st.sp32
      cases h2 with
      | inl h => exact Or.inl (by omega)
      | inr h => exact Or.inr (by omega))
  obtain ⟨k2, s2, hr2, hrip2, hs2r8, hs2r9, hr62, hs2r4, hfr2, hg2, hsh2⟩ :=
    (Pick_triple cfg.pick cfg.l32 (by decide)) hm c n sp hnlt (by omega)
      (c_add_len_lt inv rfl) (by have h32 := st.sp32; omega) st.spW s1 hrip1
      hr81 hr91 hr61 hr41 himm (by intro v; rw [hm1]; exact hw8s v) hframeL
  refine ⟨k1 + k2, s2, by rw [run_add im k1 k2, hr1, hr2], hrip2,
    hs2r8, hs2r9, hr62, hs2r4,
    by rw [hg2 1 (by decide) (by decide) (by decide) (by decide) (by decide), hg1 1 (by decide)],
    by rw [hg2 7 (by decide) (by decide) (by decide) (by decide) (by decide), hg1 7 (by decide)],
    by rw [hg2 13 (by decide) (by decide) (by decide) (by decide) (by decide), hg1 13 (by decide)],
    by rw [hg2 14 (by decide) (by decide) (by decide) (by decide) (by decide), hg1 14 (by decide)],
    by
      intro x m hx
      have hx8 : x + m ≤ sp - 8 ∨ sp ≤ x := by
        have h32 := st.sp32
        cases hx with
        | inl h => exact Or.inl (by omega)
        | inr h => exact Or.inr h
      rw [hfr2 x m hx8, hm1],
    hsh2.trans (congrArg wshape hm1),
    (run_regs_length k2 hr2).trans (run_regs_length k1 hr1)⟩

theorem hand_locget {im : Nat → Option (Ins × Nat)} (cfg : AllCode im)
    {pr : Prog} {wd : Word} {w p sp tbl : Nat} (st : WalkStatic pr wd w p sp tbl)
    {n : Nat} {rest : List Instr} {i c : Nat} {cur hm : Height} {s : X86Int.St}
    (inv : WalkInv pr wd w p sp tbl s (Instr.locget n :: rest) i c cur hm)
    (hs : hstep pr wd hm (Instr.locget n) = some (some ⟨hm.d + 1, hm.a⟩)) :
    ∃ k s2, run im k (Out.ok s) = Out.ok s2 ∧ s2.rip = aFallTail ∧
      HPost s s2 ⟨hm.d + 1, hm.a⟩ (c + 5) sp := by
  have hbhead : hm.d + 2 < W ∧ hm.a + 1 < W := by
    have h1 := hBnd_head inv.hb
    rwa [inv.hlink.symm] at h1
  have hnlt : n < wd.locals := (hstep_locget hs).1
  have himmok : n < 256 ^ 4 := inv.imm _ (List.mem_cons_self)
  have hcs : s.mem.rdN c (Instr.locget n).enc.length = some (Instr.locget n).enc ∧
      CodeSeq s.mem (c + (Instr.locget n).enc.length) rest := inv.csq
  have h5 : s.mem.rdN c 5 = some (0x86 :: leBytes n 4) := hcs.1
  have hbyte : s.mem.rd c = some 134 := (rdN_cons h5).1
  have hw8s : ∀ v, (wr64 s.mem (sp - 8) v).isSome := by
    intro v
    have hh := inv.room sp inv.r4 0 (by decide) v
    rwa [slot_sub (k := 0 + 1) (by have h32 := st.sp32; omega) st.spW] at hh
  obtain ⟨k1, s1, hr1, hrip1, -, hr11, hr61, hr71, hr81, hr91, hr41, hg1, hm1⟩ :=
    WalkLocget_triple cfg.walk (by decide) hm c i w sp wd.code.length st.w56
      (i_lt_of_inv inv) (c_lt_of_inv inv) s inv.rip inv.r8 inv.r9 inv.r6 inv.r4
      inv.r7 inv.r1 inv.len16 inv.fnc hbyte
  have hlen1 : 16 ≤ s1.regs.length := by
    have e := run_regs_length k1 hr1
    rw [e]; exact inv.len16
  have himm : (s1.mem.rdN (c + 1) 4).map leVal = some n := by
    rw [hm1]
    obtain ⟨-, himm0⟩ := imm32_of_enc h5
    rwa [Nat.mod_eq_of_lt himmok] at himm0
  have hdisjw8 : sp ≤ w + 16 ∨ w + 24 ≤ sp - 8 := by
    have h32 := st.sp32
    cases st.disjw with
    | inl h => exact Or.inl (by omega)
    | inr h => exact Or.inr (by omega)
  obtain ⟨k2, s2, hr2, hrip2, hs2r8, hs2r9, hr62, hs2r4, hg2, hfr2, hsh2⟩ :=
    (Locget_triple cfg.locget cfg.l32 (by decide)) hm c n sp w hnlt (by omega)
      (c_add_len_lt inv rfl) (by have h56 := st.w56; omega)
      (by have h32 := st.sp32; omega) st.spW (disjC5_of_inv inv rfl).2 hdisjw8
      s1 hrip1 hr81 hr91 hr61 hr41 hr71 hlen1 himm
      (by rw [hm1]; exact inv.floc) (by intro v; rw [hm1]; exact hw8s v)
  refine ⟨k1 + k2, s2, by rw [run_add im k1 k2, hr1, hr2], hrip2,
    hs2r8, hs2r9, hr62, hs2r4,
    by rw [hg2 1 (by decide) (by decide) (by decide) (by decide) (by decide) (by decide), hg1 1 (by decide)],
    by rw [hg2 7 (by decide) (by decide) (by decide) (by decide) (by decide) (by decide), hg1 7 (by decide)],
    by rw [hg2 13 (by decide) (by decide) (by decide) (by decide) (by decide) (by decide), hg1 13 (by decide)],
    by rw [hg2 14 (by decide) (by decide) (by decide) (by decide) (by decide) (by decide), hg1 14 (by decide)],
    by
      intro x m hx
      have hx8 : x + m ≤ sp - 8 ∨ sp ≤ x := by
        have h32 := st.sp32
        cases hx with
        | inl h => exact Or.inl (by omega)
        | inr h => exact Or.inr h
      rw [hfr2 x m hx8, hm1],
    hsh2.trans (congrArg wshape hm1),
    (run_regs_length k2 hr2).trans (run_regs_length k1 hr1)⟩

theorem hand_locset {im : Nat → Option (Ins × Nat)} (cfg : AllCode im)
    {pr : Prog} {wd : Word} {w p sp tbl : Nat} (st : WalkStatic pr wd w p sp tbl)
    {n : Nat} {rest : List Instr} {i c : Nat} {cur hm : Height} {s : X86Int.St}
    (inv : WalkInv pr wd w p sp tbl s (Instr.locset n :: rest) i c cur hm)
    (hs : hstep pr wd hm (Instr.locset n) = some (some ⟨hm.d - 1, hm.a⟩)) :
    ∃ k s2, run im k (Out.ok s) = Out.ok s2 ∧ s2.rip = aFallTail ∧
      HPost s s2 ⟨hm.d - 1, hm.a⟩ (c + 5) sp := by
  have hbhead : hm.d + 2 < W ∧ hm.a + 1 < W := by
    have h1 := hBnd_head inv.hb
    rwa [inv.hlink.symm] at h1
  obtain ⟨hd1, hnlt, -⟩ := hstep_locset hs
  have himmok : n < 256 ^ 4 := inv.imm _ (List.mem_cons_self)
  have hcs : s.mem.rdN c (Instr.locset n).enc.length = some (Instr.locset n).enc ∧
      CodeSeq s.mem (c + (Instr.locset n).enc.length) rest := inv.csq
  have h5 : s.mem.rdN c 5 = some (0x87 :: leBytes n 4) := hcs.1
  have hbyte : s.mem.rd c = some 135 := (rdN_cons h5).1
  have hw8s : ∀ v, (wr64 s.mem (sp - 8) v).isSome := by
    intro v
    have hh := inv.room sp inv.r4 0 (by decide) v
    rwa [slot_sub (k := 0 + 1) (by have h32 := st.sp32; omega) st.spW] at hh
  obtain ⟨k1, s1, hr1, hrip1, -, hr11, hr61, hr71, hr81, hr91, hr41, hg1, hm1⟩ :=
    WalkLocset_triple cfg.walk (by decide) hm c i w sp wd.code.length st.w56
      (i_lt_of_inv inv) (c_lt_of_inv inv) s inv.rip inv.r8 inv.r9 inv.r6 inv.r4
      inv.r7 inv.r1 inv.len16 inv.fnc hbyte
  have hlen1 : 16 ≤ s1.regs.length := by
    have e := run_regs_length k1 hr1
    rw [e]; exact inv.len16
  have himm : (s1.mem.rdN (c + 1) 4).map leVal = some n := by
    rw [hm1]
    obtain ⟨-, himm0⟩ := imm32_of_enc h5
    rwa [Nat.mod_eq_of_lt himmok] at himm0
  have hdisjw8 : sp ≤ w + 16 ∨ w + 24 ≤ sp - 8 := by
    have h32 := st.sp32
    cases st.disjw with
    | inl h => exact Or.inl (by omega)
    | inr h => exact Or.inr (by omega)
  obtain ⟨k2, s2, hr2, hrip2, hs2r8, hs2r9, hr62, hs2r4, hg2, hfr2, hsh2⟩ :=
    (Locset_triple cfg.locset cfg.l32 (by decide)) hm c n sp w hd1 hnlt (by omega)
      (c_add_len_lt inv rfl) (by have h56 := st.w56; omega)
      (by have h32 := st.sp32; omega) st.spW (disjC5_of_inv inv rfl).2 hdisjw8
      s1 hrip1 hr81 hr91 hr61 hr41 hr71 hlen1 himm
      (by rw [hm1]; exact inv.floc) (by intro v; rw [hm1]; exact hw8s v)
  refine ⟨k1 + k2, s2, by rw [run_add im k1 k2, hr1, hr2], hrip2,
    hs2r8, hs2r9, hr62, hs2r4,
    by rw [hg2 1 (by decide) (by decide) (by decide) (by decide) (by decide) (by decide), hg1 1 (by decide)],
    by rw [hg2 7 (by decide) (by decide) (by decide) (by decide) (by decide) (by decide), hg1 7 (by decide)],
    by rw [hg2 13 (by decide) (by decide) (by decide) (by decide) (by decide) (by decide), hg1 13 (by decide)],
    by rw [hg2 14 (by decide) (by decide) (by decide) (by decide) (by decide) (by decide), hg1 14 (by decide)],
    by
      intro x m hx
      have hx8 : x + m ≤ sp - 8 ∨ sp ≤ x := by
        have h32 := st.sp32
        cases hx with
        | inl h => exact Or.inl (by omega)
        | inr h => exact Or.inr h
      rw [hfr2 x m hx8, hm1],
    hsh2.trans (congrArg wshape hm1),
    (run_regs_length k2 hr2).trans (run_regs_length k1 hr1)⟩

theorem hand_jz {im : Nat → Option (Ins × Nat)} (cfg : AllCode im)
    {pr : Prog} {wd : Word} {w p sp tbl : Nat} (st : WalkStatic pr wd w p sp tbl)
    {t : Nat} {rest : List Instr} {i c : Nat} {cur hm : Height} {s : X86Int.St}
    (inv : WalkInv pr wd w p sp tbl s (Instr.jz t :: rest) i c cur hm)
    (hs : hstep pr wd hm (Instr.jz t) = some (some ⟨hm.d - 1, hm.a⟩)) :
    ∃ k s2, run im k (Out.ok s) = Out.ok s2 ∧ s2.rip = aFallTail ∧
      HPost s s2 ⟨hm.d - 1, hm.a⟩ (c + 5) sp := by
  have hbhead : hm.d + 2 < W ∧ hm.a + 1 < W := by
    have h1 := hBnd_head inv.hb
    rwa [inv.hlink.symm] at h1
  obtain ⟨hd1, hlk, -⟩ := hstep_jz hs
  have himmok : t < 256 ^ 4 := inv.imm _ List.mem_cons_self
  have hcs : s.mem.rdN c (Instr.jz t).enc.length = some (Instr.jz t).enc ∧
      CodeSeq s.mem (c + (Instr.jz t).enc.length) rest := inv.csq
  have h5 : s.mem.rdN c 5 = some (0x89 :: leBytes t 4) := hcs.1
  have hbyte : s.mem.rd c = some 137 := (rdN_cons h5).1
  obtain ⟨k1, s1, hr1, hrip1, -, hr11, hr61, hr71, hr81, hr91, hr41, hg1, hm1⟩ :=
    WalkJz_triple cfg.walk (by decide) hm c i w sp wd.code.length st.w56
      (i_lt_of_inv inv) (c_lt_of_inv inv) s inv.rip inv.r8 inv.r9 inv.r6 inv.r4
      inv.r7 inv.r1 inv.len16 inv.fnc hbyte
  have hlen1 : 16 ≤ s1.regs.length := by
    have e := run_regs_length k1 hr1
    rw [e]; exact inv.len16
  have himm : (s1.mem.rdN (c + 1) 4).map leVal = some t := by
    rw [hm1]
    obtain ⟨-, himm0⟩ := imm32_of_enc h5
    rwa [Nat.mod_eq_of_lt himmok] at himm0
  have lay1 : AnnLayout s1 wd.ann p := AnnLayout_of_mem hm1 inv.lay
  have room1 : StackRoom s1 4 :=
    StackRoom_of_mem hm1 (by rw [hr41, inv.r4]) inv.room
  have annls : ANNLSpec im aANNL aL32 := ANNL_triple cfg.annl cfg.l32 (by decide)
  obtain ⟨k2, s2, hr2, hrip2, hs2r8, hs2r9, hr62, hs2r4, hs2r1, hg2, hfr2, hsh2⟩ :=
    (Jz_triple cfg.jz cfg.l32 annls (show aJz + 104 < W by decide)) wd hm c t w p sp i hd1 hlk
      (Nat.lt_trans himmok (by decide))
      (by have h1 := i_lt_of_inv inv; have h2 := st.clenW; omega)
      st.p0 st.w56 st.pann (by have h1 := hbhead.1; omega)
      (by have h2 := hbhead.2; omega) (c_add_len_lt inv rfl) st.sp32 st.spW
      st.disjw st.disjp (disjC5_of_inv inv rfl).1 s1 hrip1 hr81 hr91 hr61 hr41
      hr71 hr11 hlen1 himm (by rw [hm1]; exact inv.fannp)
      (by rw [hm1]; exact inv.fnann) lay1 room1
  refine ⟨k1 + k2, s2, by rw [run_add im k1 k2, hr1, hr2], hrip2,
    hs2r8, hs2r9, hr62, hs2r4,
    by rw [hs2r1, inv.r1],
    by rw [hg2 7 (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide), hg1 7 (by decide)],
    by rw [hg2 13 (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide), hg1 13 (by decide)],
    by rw [hg2 14 (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide), hg1 14 (by decide)],
    by intro x m hx; rw [hfr2 x m hx, hm1],
    hsh2.trans (congrArg wshape hm1),
    (run_regs_length k2 hr2).trans (run_regs_length k1 hr1)⟩

theorem hand_jnz {im : Nat → Option (Ins × Nat)} (cfg : AllCode im)
    {pr : Prog} {wd : Word} {w p sp tbl : Nat} (st : WalkStatic pr wd w p sp tbl)
    {t : Nat} {rest : List Instr} {i c : Nat} {cur hm : Height} {s : X86Int.St}
    (inv : WalkInv pr wd w p sp tbl s (Instr.jnz t :: rest) i c cur hm)
    (hs : hstep pr wd hm (Instr.jnz t) = some (some ⟨hm.d - 1, hm.a⟩)) :
    ∃ k s2, run im k (Out.ok s) = Out.ok s2 ∧ s2.rip = aFallTail ∧
      HPost s s2 ⟨hm.d - 1, hm.a⟩ (c + 5) sp := by
  have hbhead : hm.d + 2 < W ∧ hm.a + 1 < W := by
    have h1 := hBnd_head inv.hb
    rwa [inv.hlink.symm] at h1
  obtain ⟨hd1, hlk, -⟩ := hstep_jnz hs
  have himmok : t < 256 ^ 4 := inv.imm _ List.mem_cons_self
  have hcs : s.mem.rdN c (Instr.jnz t).enc.length = some (Instr.jnz t).enc ∧
      CodeSeq s.mem (c + (Instr.jnz t).enc.length) rest := inv.csq
  have h5 : s.mem.rdN c 5 = some (0x8a :: leBytes t 4) := hcs.1
  have hbyte : s.mem.rd c = some 138 := (rdN_cons h5).1
  obtain ⟨k1, s1, hr1, hrip1, -, hr11, hr61, hr71, hr81, hr91, hr41, hg1, hm1⟩ :=
    WalkJnz_triple cfg.walk (by decide) hm c i w sp wd.code.length st.w56
      (i_lt_of_inv inv) (c_lt_of_inv inv) s inv.rip inv.r8 inv.r9 inv.r6 inv.r4
      inv.r7 inv.r1 inv.len16 inv.fnc hbyte
  have hlen1 : 16 ≤ s1.regs.length := by
    have e := run_regs_length k1 hr1
    rw [e]; exact inv.len16
  have himm : (s1.mem.rdN (c + 1) 4).map leVal = some t := by
    rw [hm1]
    obtain ⟨-, himm0⟩ := imm32_of_enc h5
    rwa [Nat.mod_eq_of_lt himmok] at himm0
  have lay1 : AnnLayout s1 wd.ann p := AnnLayout_of_mem hm1 inv.lay
  have room1 : StackRoom s1 4 :=
    StackRoom_of_mem hm1 (by rw [hr41, inv.r4]) inv.room
  have annls : ANNLSpec im aANNL aL32 := ANNL_triple cfg.annl cfg.l32 (by decide)
  obtain ⟨k2, s2, hr2, hrip2, hs2r8, hs2r9, hr62, hs2r4, hs2r1, hg2, hfr2, hsh2⟩ :=
    (Jz_triple cfg.jz cfg.l32 annls (show aJz + 104 < W by decide)) wd hm c t w p sp i hd1 hlk
      (Nat.lt_trans himmok (by decide))
      (by have h1 := i_lt_of_inv inv; have h2 := st.clenW; omega)
      st.p0 st.w56 st.pann (by have h1 := hbhead.1; omega)
      (by have h2 := hbhead.2; omega) (c_add_len_lt inv rfl) st.sp32 st.spW
      st.disjw st.disjp (disjC5_of_inv inv rfl).1 s1 hrip1 hr81 hr91 hr61 hr41
      hr71 hr11 hlen1 himm (by rw [hm1]; exact inv.fannp)
      (by rw [hm1]; exact inv.fnann) lay1 room1
  refine ⟨k1 + k2, s2, by rw [run_add im k1 k2, hr1, hr2], hrip2,
    hs2r8, hs2r9, hr62, hs2r4,
    by rw [hs2r1, inv.r1],
    by rw [hg2 7 (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide), hg1 7 (by decide)],
    by rw [hg2 13 (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide), hg1 13 (by decide)],
    by rw [hg2 14 (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide), hg1 14 (by decide)],
    by intro x m hx; rw [hfr2 x m hx, hm1],
    hsh2.trans (congrArg wshape hm1),
    (run_regs_length k2 hr2).trans (run_regs_length k1 hr1)⟩

theorem hand_jmp {im : Nat → Option (Ins × Nat)} (cfg : AllCode im)
    {pr : Prog} {wd : Word} {w p sp tbl : Nat} (st : WalkStatic pr wd w p sp tbl)
    {t : Nat} {rest : List Instr} {i c : Nat} {cur hm : Height} {s : X86Int.St}
    (inv : WalkInv pr wd w p sp tbl s (Instr.jmp t :: rest) i c cur hm)
    (hs : hstep pr wd hm (Instr.jmp t) = some none) :
    ∃ k s2, run im k (Out.ok s) = Out.ok s2 ∧ s2.rip = aNoneTail ∧
      HPost s s2 hm (c + 5) sp := by
  have hbhead : hm.d + 2 < W ∧ hm.a + 1 < W := by
    have h1 := hBnd_head inv.hb
    rwa [inv.hlink.symm] at h1
  have hlk : wd.ann.lookup t = some hm := hstep_jmp hs
  have himmok : t < 256 ^ 4 := inv.imm _ List.mem_cons_self
  have hcs : s.mem.rdN c (Instr.jmp t).enc.length = some (Instr.jmp t).enc ∧
      CodeSeq s.mem (c + (Instr.jmp t).enc.length) rest := inv.csq
  have h5 : s.mem.rdN c 5 = some (0x88 :: leBytes t 4) := hcs.1
  have hbyte : s.mem.rd c = some 136 := (rdN_cons h5).1
  obtain ⟨k1, s1, hr1, hrip1, -, hr11, hr61, hr71, hr81, hr91, hr41, hg1, hm1⟩ :=
    WalkJmp_triple cfg.walk (by decide) hm c i w sp wd.code.length st.w56
      (i_lt_of_inv inv) (c_lt_of_inv inv) s inv.rip inv.r8 inv.r9 inv.r6 inv.r4
      inv.r7 inv.r1 inv.len16 inv.fnc hbyte
  have hlen1 : 16 ≤ s1.regs.length := by
    have e := run_regs_length k1 hr1
    rw [e]; exact inv.len16
  have himm : (s1.mem.rdN (c + 1) 4).map leVal = some t := by
    rw [hm1]
    obtain ⟨-, himm0⟩ := imm32_of_enc h5
    rwa [Nat.mod_eq_of_lt himmok] at himm0
  have lay1 : AnnLayout s1 wd.ann p := AnnLayout_of_mem hm1 inv.lay
  have room1 : StackRoom s1 4 :=
    StackRoom_of_mem hm1 (by rw [hr41, inv.r4]) inv.room
  have annls : ANNLSpec im aANNL aL32 := ANNL_triple cfg.annl cfg.l32 (by decide)
  obtain ⟨k2, s2, hr2, hrip2, hs2r8, hs2r9, hr62, hs2r4, hs2r1, hg2, hfr2, hsh2⟩ :=
    (Jmp_triple cfg.jmp cfg.l32 annls (show aJmp + 82 < W by decide)) wd hm c t w p sp i hlk
      (Nat.lt_trans himmok (by decide))
      (by have h1 := i_lt_of_inv inv; have h2 := st.clenW; omega)
      st.p0 st.w56 st.pann (by have h1 := hbhead.1; omega)
      (by have h2 := hbhead.2; omega) (c_add_len_lt inv rfl) st.sp32 st.spW
      st.disjw st.disjp (disjC5_of_inv inv rfl).1 s1 hrip1 hr81 hr91 hr61 hr41
      hr71 hr11 hlen1 himm (by rw [hm1]; exact inv.fannp)
      (by rw [hm1]; exact inv.fnann) lay1 room1
  refine ⟨k1 + k2, s2, by rw [run_add im k1 k2, hr1, hr2], hrip2,
    hs2r8, hs2r9, hr62, hs2r4,
    by rw [hs2r1, inv.r1],
    by rw [hg2 7 (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide), hg1 7 (by decide)],
    by rw [hg2 13 (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide), hg1 13 (by decide)],
    by rw [hg2 14 (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide), hg1 14 (by decide)],
    by intro x m hx; rw [hfr2 x m hx, hm1],
    hsh2.trans (congrArg wshape hm1),
    (run_regs_length k2 hr2).trans (run_regs_length k1 hr1)⟩

theorem hand_call {im : Nat → Option (Ins × Nat)} (cfg : AllCode im)
    {pr : Prog} {wd : Word} {w p sp tbl : Nat} (st : WalkStatic pr wd w p sp tbl)
    {n : Nat} {rest : List Instr} {i c : Nat} {cur hm h' : Height} {s : X86Int.St}
    (inv : WalkInv pr wd w p sp tbl s (Instr.call n :: rest) i c cur hm)
    (hs : hstep pr wd hm (Instr.call n) = some (some h')) :
    ∃ k s2, run im k (Out.ok s) = Out.ok s2 ∧ s2.rip = aFallTail ∧
      HPost s s2 h' (c + 5) sp := by
  have hbhead : hm.d + 2 < W ∧ hm.a + 1 < W := by
    have h1 := hBnd_head inv.hb
    rwa [inv.hlink.symm] at h1
  have hs0 : hstep pr wd ((wd.ann.lookup i).getD cur) (Instr.call n) =
      some (some h') := by
    rw [inv.hlink.symm]; exact hs
  obtain ⟨hbd, -, -⟩ := hBnd_some_some inv.hb hs0
  obtain ⟨cw, hcwq, hins, hh'⟩ := hstep_call hs
  have hnlt : n < pr.words.length := (List.getElem?_eq_some_iff.mp hcwq).1
  have hcw : pr.words[n] = cw := (List.getElem?_eq_some_iff.mp hcwq).2
  have himmok : n < 256 ^ 4 := inv.imm _ List.mem_cons_self
  have hbound : hm.d - cw.ins + cw.outs < W := by
    have hbd' : hm.d - cw.ins + cw.outs + 2 < W := by rw [hh'] at hbd; exact hbd
    omega
  have hcs : s.mem.rdN c (Instr.call n).enc.length = some (Instr.call n).enc ∧
      CodeSeq s.mem (c + (Instr.call n).enc.length) rest := inv.csq
  have h5 : s.mem.rdN c 5 = some (0x8b :: leBytes n 4) := hcs.1
  have hbyte : s.mem.rd c = some 139 := (rdN_cons h5).1
  have hw8s : ∀ v, (wr64 s.mem (sp - 8) v).isSome := by
    intro v
    have hh := inv.room sp inv.r4 0 (by decide) v
    rwa [slot_sub (k := 0 + 1) (by have h32 := st.sp32; omega) st.spW] at hh
  obtain ⟨k1, s1, hr1, hrip1, -, hr11, hr61, hr71, hr81, hr91, hr41, hg1, hm1⟩ :=
    WalkCall_triple cfg.walk (by decide) hm c i w sp wd.code.length st.w56
      (i_lt_of_inv inv) (c_lt_of_inv inv) s inv.rip inv.r8 inv.r9 inv.r6 inv.r4
      inv.r7 inv.r1 inv.len16 inv.fnc hbyte
  have hlen1 : 16 ≤ s1.regs.length := by
    have e := run_regs_length k1 hr1
    rw [e]; exact inv.len16
  have himm : (s1.mem.rdN (c + 1) 4).map leVal = some n := by
    rw [hm1]
    obtain ⟨-, himm0⟩ := imm32_of_enc h5
    rwa [Nat.mod_eq_of_lt himmok] at himm0
  have hr13 : s1.reg 13 = some pr.words.length := by
    rw [hg1 13 (by decide)]; exact inv.r13
  have hr14 : s1.reg 14 = some tbl := by
    rw [hg1 14 (by decide)]; exact inv.r14
  have hdisjT : sp ≤ tbl + n * 64 ∨ tbl + n * 64 + 16 ≤ sp - 8 := by
    have h32 := st.sp32
    cases (st.wordsB n hnlt).2 with
    | inl h => exact Or.inl h
    | inr h => exact Or.inr (by omega)
  obtain ⟨k2, s2, hr2, hrip2, hs2r8, hs2r9, hr62, hs2r4, hg2, hfr2, hsh2⟩ :=
    (Call_triple cfg.call cfg.l32 (by decide)) cw hm c n sp tbl pr.words.length
      hnlt hins (by have h1 := hbhead.1; omega) hbound (c_add_len_lt inv rfl)
      (by have h32 := st.sp32; omega) st.spW (st.wordsB n hnlt).1
      (disjC5_of_inv inv rfl).2 hdisjT s1 hrip1 hr81 hr91 hr61 hr41 hr13 hr14
      hlen1 himm
      (by rw [hm1, ← hcw]; exact (inv.words n hnlt).1)
      (by rw [hm1, ← hcw]; exact (inv.words n hnlt).2)
      (by intro v; rw [hm1]; exact hw8s v)
  refine ⟨k1 + k2, s2, by rw [run_add im k1 k2, hr1, hr2], hrip2,
    by rw [hh']; exact hs2r8,
    by rw [hh']; exact hs2r9,
    hr62, hs2r4,
    by rw [hg2 1 (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide), hg1 1 (by decide)],
    by rw [hg2 7 (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide), hg1 7 (by decide)],
    by rw [hg2 13 (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide), hg1 13 (by decide)],
    by rw [hg2 14 (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide), hg1 14 (by decide)],
    by
      intro x m hx
      have hx8 : x + m ≤ sp - 8 ∨ sp ≤ x := by
        have h32 := st.sp32
        cases hx with
        | inl h => exact Or.inl (by omega)
        | inr h => exact Or.inr h
      rw [hfr2 x m hx8, hm1],
    hsh2.trans (congrArg wshape hm1),
    (run_regs_length k2 hr2).trans (run_regs_length k1 hr1)⟩

theorem hand_prim_fall {im : Nat → Option (Ins × Nat)} (cfg : AllCode im)
    {pr : Prog} {wd : Word} {w p sp tbl : Nat} (st : WalkStatic pr wd w p sp tbl)
    {q : Prim} {rest : List Instr} {i c : Nat} {cur hm h' : Height} {s : X86Int.St}
    (inv : WalkInv pr wd w p sp tbl s (Instr.prim q :: rest) i c cur hm)
    (hs : hstep pr wd hm (Instr.prim q) = some (some h')) :
    ∃ k s2, run im k (Out.ok s) = Out.ok s2 ∧ s2.rip = aFallTail ∧
      HPost s s2 h' (c + 1) sp := by
  have hbhead : hm.d + 2 < W ∧ hm.a + 1 < W := by
    have h1 := hBnd_head inv.hb
    rwa [inv.hlink.symm] at h1
  have hs0 : hstep pr wd ((wd.ann.lookup i).getD cur) (Instr.prim q) =
      some (some h') := by
    rw [inv.hlink.symm]; exact hs
  obtain ⟨hbd, -, -⟩ := hBnd_some_some inv.hb hs0
  obtain ⟨hguard, hne, hh'⟩ := hstep_prim_some hs
  have hbound : hm.d - q.eff.1 + q.eff.2 < W := by
    have hbd' : hm.d - q.eff.1 + q.eff.2 + 2 < W := by rw [hh'] at hbd; exact hbd
    omega
  have hcs : s.mem.rdN c (Instr.prim q).enc.length = some (Instr.prim q).enc ∧
      CodeSeq s.mem (c + (Instr.prim q).enc.length) rest := inv.csq
  have h1b : s.mem.rdN c 1 = some [q.code] := hcs.1
  have hbyte : s.mem.rd c = some q.code := (rdN_cons h1b).1
  obtain ⟨k1, s1, hr1, hrip1, hr01, hr11, hr61, hr71, hr81, hr91, hr41, hg1, hm1⟩ :=
    WalkPrim_triple cfg.walk (by decide) hm c i w sp wd.code.length st.w56
      (i_lt_of_inv inv) (c_lt_of_inv inv) q.code (Prim_code_lt q) s inv.rip
      inv.r8 inv.r9 inv.r6 inv.r4 inv.r7 inv.r1 inv.len16 inv.fnc hbyte
  have hlen1 : 16 ≤ s1.regs.length := by
    have e := run_regs_length k1 hr1
    rw [e]; exact inv.len16
  have heffW : aEff + 2 * q.code + 2 ≤ W := by
    have h1 : aEff + 114 ≤ W := by decide
    have h2 : q.code ≤ 56 := by have h57 := Prim_code_lt q; omega
    omega
  obtain ⟨k2, s2, hr2, hrip2, hs2r8, hs2r9, hr62, hm2, hg2⟩ :=
    (PrimFall_triple cfg.prim) q hm c (Prim_code_ne_50 hne) hguard
      (by have h1 := hbhead.1; omega) hbound (c_add_len_lt inv rfl) heffW s1
      hrip1 hr01 hr81 hr91 hr61 hlen1 (by rw [hm1]; exact inv.eff q)
  refine ⟨k1 + k2, s2, by rw [run_add im k1 k2, hr1, hr2], hrip2,
    by rw [hh']; exact hs2r8,
    by rw [hh']; exact hs2r9,
    hr62,
    by rw [hg2 4 (by decide) (by decide) (by decide) (by decide) (by decide), hg1 4 (by decide)]; exact inv.r4,
    by rw [hg2 1 (by decide) (by decide) (by decide) (by decide) (by decide), hg1 1 (by decide)],
    by rw [hg2 7 (by decide) (by decide) (by decide) (by decide) (by decide), hg1 7 (by decide)],
    by rw [hg2 13 (by decide) (by decide) (by decide) (by decide) (by decide), hg1 13 (by decide)],
    by rw [hg2 14 (by decide) (by decide) (by decide) (by decide) (by decide), hg1 14 (by decide)],
    by intro x m _; rw [hm2, hm1],
    by rw [hm2, hm1],
    (run_regs_length k2 hr2).trans (run_regs_length k1 hr1)⟩

theorem hand_prim_exit {im : Nat → Option (Ins × Nat)} (cfg : AllCode im)
    {pr : Prog} {wd : Word} {w p sp tbl : Nat} (st : WalkStatic pr wd w p sp tbl)
    {q : Prim} {rest : List Instr} {i c : Nat} {cur hm : Height} {s : X86Int.St}
    (inv : WalkInv pr wd w p sp tbl s (Instr.prim q :: rest) i c cur hm)
    (hs : hstep pr wd hm (Instr.prim q) = some none) :
    ∃ k s2, run im k (Out.ok s) = Out.ok s2 ∧ s2.rip = aNoneTail ∧
      HPost s s2 hm (c + 1) sp := by
  have hbhead : hm.d + 2 < W ∧ hm.a + 1 < W := by
    have h1 := hBnd_head inv.hb
    rwa [inv.hlink.symm] at h1
  obtain ⟨hguard, hq⟩ := hstep_prim_none hs
  have hcs : s.mem.rdN c (Instr.prim q).enc.length = some (Instr.prim q).enc ∧
      CodeSeq s.mem (c + (Instr.prim q).enc.length) rest := inv.csq
  have h1b : s.mem.rdN c 1 = some [q.code] := hcs.1
  have hbyte : s.mem.rd c = some q.code := (rdN_cons h1b).1
  obtain ⟨k1, s1, hr1, hrip1, hr01, hr11, hr61, hr71, hr81, hr91, hr41, hg1, hm1⟩ :=
    WalkPrim_triple cfg.walk (by decide) hm c i w sp wd.code.length st.w56
      (i_lt_of_inv inv) (c_lt_of_inv inv) q.code (Prim_code_lt q) s inv.rip
      inv.r8 inv.r9 inv.r6 inv.r4 inv.r7 inv.r1 inv.len16 inv.fnc hbyte
  have hlen1 : 16 ≤ s1.regs.length := by
    have e := run_regs_length k1 hr1
    rw [e]; exact inv.len16
  have heffW : aEff + 2 * q.code + 2 ≤ W := by
    have h1 : aEff + 114 ≤ W := by decide
    have h2 : q.code ≤ 56 := by have h57 := Prim_code_lt q; omega
    omega
  have hcode50 : q.code = 50 := by rw [hq]; exact Prim_exit_code
  obtain ⟨k2, s2, hr2, hrip2, hs2r8, hs2r9, hr62, hm2, hg2⟩ :=
    (PrimExit_triple cfg.prim) q hm c hcode50 hguard
      (by have h1 := hbhead.1; omega) (c_add_len_lt inv rfl) heffW s1
      hrip1 hr01 hr81 hr91 hr61 hlen1 (by rw [hm1]; exact inv.eff q)
  refine ⟨k1 + k2, s2, by rw [run_add im k1 k2, hr1, hr2], hrip2,
    hs2r8, hs2r9, hr62,
    by rw [hg2 4 (by decide) (by decide) (by decide) (by decide) (by decide), hg1 4 (by decide)]; exact inv.r4,
    by rw [hg2 1 (by decide) (by decide) (by decide) (by decide) (by decide), hg1 1 (by decide)],
    by rw [hg2 7 (by decide) (by decide) (by decide) (by decide) (by decide), hg1 7 (by decide)],
    by rw [hg2 13 (by decide) (by decide) (by decide) (by decide) (by decide), hg1 13 (by decide)],
    by rw [hg2 14 (by decide) (by decide) (by decide) (by decide) (by decide), hg1 14 (by decide)],
    by intro x m _; rw [hm2, hm1],
    by rw [hm2, hm1],
    (run_regs_length k2 hr2).trans (run_regs_length k1 hr1)⟩

theorem hand_ret {im : Nat → Option (Ins × Nat)} (cfg : AllCode im)
    {pr : Prog} {wd : Word} {w p sp tbl : Nat} (st : WalkStatic pr wd w p sp tbl)
    {rest : List Instr} {i c : Nat} {cur hm : Height} {s : X86Int.St}
    (inv : WalkInv pr wd w p sp tbl s (Instr.ret :: rest) i c cur hm)
    (hs : hstep pr wd hm Instr.ret = some none) :
    ∃ k s2, run im k (Out.ok s) = Out.ok s2 ∧ s2.rip = aNoneTail ∧
      HPost s s2 hm (c + 1) sp := by
  have hbhead : hm.d + 2 < W ∧ hm.a + 1 < W := by
    have h1 := hBnd_head inv.hb
    rwa [inv.hlink.symm] at h1
  have hmeq : hm = ⟨wd.outs, 0⟩ := hstep_ret hs
  have hcs : s.mem.rdN c Instr.ret.enc.length = some Instr.ret.enc ∧
      CodeSeq s.mem (c + Instr.ret.enc.length) rest := inv.csq
  have h1b : s.mem.rdN c 1 = some [140] := hcs.1
  have hbyte : s.mem.rd c = some 140 := (rdN_cons h1b).1
  obtain ⟨k1, s1, hr1, hrip1, -, hr11, hr61, hr71, hr81, hr91, hr41, hg1, hm1⟩ :=
    WalkRet_triple cfg.walk (by decide) hm c i w sp wd.code.length st.w56
      (i_lt_of_inv inv) (c_lt_of_inv inv) s inv.rip inv.r8 inv.r9 inv.r6 inv.r4
      inv.r7 inv.r1 inv.len16 inv.fnc hbyte
  have hr81' : s1.reg 8 = some wd.outs := by
    rw [hr81]
    congr 1
    exact congrArg Height.d hmeq
  have hr91' : s1.reg 9 = some 0 := by
    rw [hr91]
    congr 1
    exact congrArg Height.a hmeq
  have houtW : wd.outs < W := by
    have h1 : hm.d = wd.outs := congrArg Height.d hmeq
    have h2 := hbhead.1
    omega
  obtain ⟨k2, s2, hr2, hrip2, hs2r8, hs2r9, hr62, hm2, hg2⟩ :=
    (Ret_triple cfg.ret) c w houtW (c_add_len_lt inv rfl)
      (by have h56 := st.w56; omega) s1 hrip1 hr81' hr91' hr61 hr71
      (by rw [hm1]; exact inv.fouts)
  refine ⟨k1 + k2, s2, by rw [run_add im k1 k2, hr1, hr2], hrip2,
    by rw [hs2r8]; congr 1; exact (congrArg Height.d hmeq).symm,
    by rw [hs2r9]; congr 1; exact (congrArg Height.a hmeq).symm,
    hr62,
    by rw [hg2 4 (by decide) (by decide), hg1 4 (by decide)]; exact inv.r4,
    by rw [hg2 1 (by decide) (by decide), hg1 1 (by decide)],
    by rw [hg2 7 (by decide) (by decide), hg1 7 (by decide)],
    by rw [hg2 13 (by decide) (by decide), hg1 13 (by decide)],
    by rw [hg2 14 (by decide) (by decide), hg1 14 (by decide)],
    by intro x m _; rw [hm2, hm1],
    by rw [hm2, hm1],
    (run_regs_length k2 hr2).trans (run_regs_length k1 hr1)⟩




theorem fall_finish1 {im : Nat → Option (Ins × Nat)} (cfg : AllCode im)
    {pr : Prog} {wd : Word} {w p sp tbl : Nat} (st : WalkStatic pr wd w p sp tbl)
    {ins : Instr} {rest : List Instr} {i c len : Nat} {cur hm h' : Height}
    {s s2 : X86Int.St} {kh : Nat}
    (inv : WalkInv pr wd w p sp tbl s (ins :: rest) i c cur hm)
    (hs : hstep pr wd hm ins = some (some h'))
    (hann : wd.ann.lookup (i + 1) = none ∨ wd.ann.lookup (i + 1) = some h')
    (hi1 : i + 1 < wd.code.length)
    (hlen : ins.enc.length = len)
    (hr : run im kh (Out.ok s) = Out.ok s2)
    (hrip2 : s2.rip = aFallTail)
    (hpost : HPost s s2 h' (c + len) sp) :
    ∃ k s3, run im k (Out.ok s) = Out.ok s3 ∧
      WalkInv pr wd w p sp tbl s3 rest (i + 1) (c + len) h' h' ∧
      (∀ x m, x + m ≤ sp - 32 ∨ sp ≤ x → s3.mem.rdN x m = s.mem.rdN x m) ∧
      wshape s3.mem = wshape s.mem :=
  fall_finish cfg st inv hs hann hi1 hlen hr (k2 := 0) (s1 := s2) (run_zero im (Out.ok s2))
    hrip2 hpost

theorem none_finish_cons1 {im : Nat → Option (Ins × Nat)} (cfg : AllCode im)
    {pr : Prog} {wd : Word} {w p sp tbl : Nat} (st : WalkStatic pr wd w p sp tbl)
    {ins : Instr} {rest : List Instr} {i c len : Nat} {cur hm : Height}
    {s s2 : X86Int.St} {kh : Nat}
    (inv : WalkInv pr wd w p sp tbl s (ins :: rest) i c cur hm)
    (hs : hstep pr wd hm ins = some none)
    (hi1 : i + 1 < wd.code.length)
    (hlen : ins.enc.length = len)
    (hr : run im kh (Out.ok s) = Out.ok s2)
    (hrip2 : s2.rip = aNoneTail)
    (hpost : HPost s s2 hm (c + len) sp) :
    ∃ k s3, run im k (Out.ok s) = Out.ok s3 ∧
      WalkInv pr wd w p sp tbl s3 rest (i + 1) (c + len)
        ((wd.ann.lookup (i + 1)).getD ⟨0, 0⟩)
        ((wd.ann.lookup (i + 1)).getD ⟨0, 0⟩) ∧
      (∀ x m, x + m ≤ sp - 32 ∨ sp ≤ x → s3.mem.rdN x m = s.mem.rdN x m) ∧
      wshape s3.mem = wshape s.mem :=
  none_finish_cons cfg st inv hs hi1 hlen hr (k2 := 0) (s1 := s2) (run_zero im (Out.ok s2))
    hrip2 hpost

theorem none_finish_nil1 {im : Nat → Option (Ins × Nat)} (cfg : AllCode im)
    {pr : Prog} {wd : Word} {w p sp tbl : Nat} (st : WalkStatic pr wd w p sp tbl)
    {ins : Instr} {i c len : Nat} {cur hm : Height}
    {s s2 : X86Int.St} {kh : Nat}
    (inv : WalkInv pr wd w p sp tbl s [ins] i c cur hm)
    (hs : hstep pr wd hm ins = some none)
    (hlen : ins.enc.length = len)
    (hr : run im kh (Out.ok s) = Out.ok s2)
    (hrip2 : s2.rip = aNoneTail)
    (hpost : HPost s s2 hm (c + len) sp) :
    ∃ k s4, run im k (Out.ok s) = Out.ok s4 ∧ s4.rip = aWordDone ∧
      s4.reg 1 = some (i + 1) ∧ s4.reg 6 = some (c + len) ∧
      s4.reg 7 = some w ∧ s4.reg 4 = some sp ∧
      (∀ x m, x + m ≤ sp - 32 ∨ sp ≤ x → s4.mem.rdN x m = s.mem.rdN x m) ∧
      wshape s4.mem = wshape s.mem :=
  none_finish_nil cfg st inv hs hlen hr (k2 := 0) (s1 := s2) (run_zero im (Out.ok s2)) hrip2
    hpost

/-- The whole walk over one word, in the success direction: whenever
`Sporos.walk` accepts the remaining code, execution from `walk_top`
reaches `word_done` with index and cursor advanced over it. The
induction is on the remaining instruction list — the termination
measure. -/
theorem walk_loop {im : Nat → Option (Ins × Nat)} (cfg : AllCode im)
    {pr : Prog} {wd : Word} {w p sp tbl : Nat}
    (st : WalkStatic pr wd w p sp tbl) :
    ∀ (code : List Instr) (i c : Nat) (cur hm : Height) (s : X86Int.St),
      WalkInv pr wd w p sp tbl s code i c cur hm →
      Sporos.walk pr wd code i cur = true →
      ∃ k s', run im k (Out.ok s) = Out.ok s' ∧ s'.rip = aWordDone ∧
        s'.reg 1 = some (i + code.length) ∧
        s'.reg 6 = some (c + codeBytes code) ∧
        s'.reg 7 = some w ∧ s'.reg 4 = some sp ∧
        (∀ x m, x + m ≤ sp - 32 ∨ sp ≤ x → s'.mem.rdN x m = s.mem.rdN x m) ∧
        wshape s'.mem = wshape s.mem := by
  intro code
  induction code with
  | nil =>
    intro i c cur hm s inv hw
    obtain ⟨k, s', hr, hrip, -, hr1, hr6, hr7, -, -, hr4, -, hmD⟩ :=
      WalkDone_triple cfg.walk (by decide) hm c i w sp wd.code.length st.w56
        (by have h := inv.isum; simp at h; omega) s inv.rip inv.r8 inv.r9
        inv.r6 inv.r4 inv.r7 inv.r1 inv.len16 inv.fnc
    refine ⟨k, s', hr, hrip, ?_, ?_, hr7, hr4, ?_, ?_⟩
    · simpa using hr1
    · have e : c + codeBytes ([] : List Instr) = c := rfl
      rw [e]; exact hr6
    · intro x m _
      rw [hmD]
    · rw [hmD]
  | cons ins rest ih =>
    intro i c cur hm s inv hw
    have hge : ((wd.ann.lookup i).getD cur) = hm := inv.hlink.symm
    cases ins with
    | prim pq =>
      cases hs : hstep pr wd hm (Instr.prim pq) with
      | none =>
        have hs0 : hstep pr wd ((wd.ann.lookup i).getD cur) (Instr.prim pq) =
            none := by
          rw [hge]; exact hs
        simp [Sporos.walk, hs0] at hw
      | some o =>
        cases o with
        | some h' =>
          obtain ⟨kh, s2, hrh, hrip2, hpost⟩ := hand_prim_fall cfg st inv hs
          have hs0 : hstep pr wd ((wd.ann.lookup i).getD cur) (Instr.prim pq) =
              some (some h') := by
            rw [hge]; exact hs
          cases rest with
          | nil => simp [Sporos.walk, hs0] at hw
          | cons y ys =>
            have hw2 : (wd.ann.lookup (i + 1) = none ∨
                  wd.ann.lookup (i + 1) = some h') ∧
                Sporos.walk pr wd (y :: ys) (i + 1) h' = true := by
              simp only [Sporos.walk, hs0, Bool.and_eq_true] at hw
              refine ⟨?_, hw.2⟩
              cases hl : wd.ann.lookup (i + 1) with
              | none => exact Or.inl rfl
              | some a =>
                have hbe : (a == h') = true := by
                  have h1 := hw.1
                  rw [hl] at h1
                  exact h1
                exact Or.inr (by rw [height_eq_of_beq hbe])
            have hi1 : i + 1 < wd.code.length := by
              have h := inv.isum
              simp only [List.length_cons] at h
              omega
            obtain ⟨k3, s3, hr3, inv3, hfr3i, hsh3i⟩ :=
              fall_finish1 cfg st inv hs hw2.1 hi1 rfl hrh hrip2 hpost
            obtain ⟨k4, s4, hr4, hrip4, hr14, hr64, hr74, hr44, hfr4, hsh4⟩ :=
              ih (i + 1) (c + 1) h' h' s3 inv3 hw2.2
            refine ⟨k3 + k4, s4, by rw [run_add im k3 k4, hr3, hr4], hrip4, ?_, ?_,
              hr74, hr44,
              fun x m hx => (hfr4 x m hx).trans (hfr3i x m hx), hsh4.trans hsh3i⟩
            · have e : (i + 1) + (y :: ys).length =
                  i + (Instr.prim pq :: y :: ys).length := by
                simp only [List.length_cons]; omega
              rw [← e]; exact hr14
            · have e : (c + 1) + codeBytes (y :: ys) =
                  c + codeBytes (Instr.prim pq :: y :: ys) := by
                show (c + 1) + codeBytes (y :: ys) =
                  c + (1 + codeBytes (y :: ys))
                omega
              rw [← e]; exact hr64
        | none =>
          obtain ⟨kh, s2, hrh, hrip2, hpost⟩ := hand_prim_exit cfg st inv hs
          have hs0 : hstep pr wd ((wd.ann.lookup i).getD cur) (Instr.prim pq) =
              some none := by
            rw [hge]; exact hs
          cases rest with
          | nil =>
            obtain ⟨k3, s4, hr3, hrip4, hr14, hr64, hr74, hr44, hfr4, hsh4⟩ :=
              none_finish_nil1 cfg st inv hs rfl hrh hrip2 hpost
            refine ⟨k3, s4, hr3, hrip4, ?_, ?_, hr74, hr44, hfr4, hsh4⟩
            · simpa using hr14
            · have e : c + codeBytes [Instr.prim pq] = c + 1 := rfl
              rw [e]; exact hr64
          | cons y ys =>
            have hi1 : i + 1 < wd.code.length := by
              have h := inv.isum
              simp only [List.length_cons] at h
              omega
            have hwrest : Sporos.walk pr wd (y :: ys) (i + 1)
                ((wd.ann.lookup (i + 1)).getD ⟨0, 0⟩) = true := by
              simp only [Sporos.walk, hs0] at hw
              exact hw
            obtain ⟨k3, s3, hr3, inv3, hfr3i, hsh3i⟩ :=
              none_finish_cons1 cfg st inv hs hi1 rfl hrh hrip2 hpost
            obtain ⟨k4, s4, hr4, hrip4, hr14, hr64, hr74, hr44, hfr4, hsh4⟩ :=
              ih (i + 1) (c + 1) _ _ s3 inv3 hwrest
            refine ⟨k3 + k4, s4, by rw [run_add im k3 k4, hr3, hr4], hrip4, ?_, ?_,
              hr74, hr44,
              fun x m hx => (hfr4 x m hx).trans (hfr3i x m hx), hsh4.trans hsh3i⟩
            · have e : (i + 1) + (y :: ys).length =
                  i + (Instr.prim pq :: y :: ys).length := by
                simp only [List.length_cons]; omega
              rw [← e]; exact hr14
            · have e : (c + 1) + codeBytes (y :: ys) =
                  c + codeBytes (Instr.prim pq :: y :: ys) := by
                show (c + 1) + codeBytes (y :: ys) =
                  c + (1 + codeBytes (y :: ys))
                omega
              rw [← e]; exact hr64
    | lit lv =>
      have hs : hstep pr wd hm (Instr.lit lv) =
          some (some ⟨hm.d + 1, hm.a⟩) := rfl
      obtain ⟨kh, s2, hrh, hrip2, hpost⟩ := hand_lit cfg st inv
      have hs0 : hstep pr wd ((wd.ann.lookup i).getD cur) (Instr.lit lv) =
          some (some ⟨hm.d + 1, hm.a⟩) := by
        rw [hge]; exact hs
      cases rest with
      | nil => simp [Sporos.walk, hs0] at hw
      | cons y ys =>
        have hw2 : (wd.ann.lookup (i + 1) = none ∨
              wd.ann.lookup (i + 1) = some ⟨hm.d + 1, hm.a⟩) ∧
            Sporos.walk pr wd (y :: ys) (i + 1) ⟨hm.d + 1, hm.a⟩ = true := by
          simp only [Sporos.walk, hs0, Bool.and_eq_true] at hw
          refine ⟨?_, hw.2⟩
          cases hl : wd.ann.lookup (i + 1) with
          | none => exact Or.inl rfl
          | some a =>
            have hbe : (a == ⟨hm.d + 1, hm.a⟩) = true := by
              have h1 := hw.1
              rw [hl] at h1
              exact h1
            exact Or.inr (by rw [height_eq_of_beq hbe])
        have hi1 : i + 1 < wd.code.length := by
          have h := inv.isum
          simp only [List.length_cons] at h
          omega
        obtain ⟨k3, s3, hr3, inv3, hfr3i, hsh3i⟩ :=
          fall_finish1 cfg st inv hs hw2.1 hi1 rfl hrh hrip2 hpost
        obtain ⟨k4, s4, hr4, hrip4, hr14, hr64, hr74, hr44, hfr4, hsh4⟩ :=
          ih (i + 1) (c + 9) ⟨hm.d + 1, hm.a⟩ ⟨hm.d + 1, hm.a⟩ s3 inv3 hw2.2
        refine ⟨k3 + k4, s4, by rw [run_add im k3 k4, hr3, hr4], hrip4, ?_, ?_,
          hr74, hr44,
          fun x m hx => (hfr4 x m hx).trans (hfr3i x m hx), hsh4.trans hsh3i⟩
        · have e : (i + 1) + (y :: ys).length =
              i + (Instr.lit lv :: y :: ys).length := by
            simp only [List.length_cons]; omega
          rw [← e]; exact hr14
        · have e : (c + 9) + codeBytes (y :: ys) =
              c + codeBytes (Instr.lit lv :: y :: ys) := by
            show (c + 9) + codeBytes (y :: ys) =
              c + (9 + codeBytes (y :: ys))
            omega
          rw [← e]; exact hr64
    | str si =>
      have hs : hstep pr wd hm (Instr.str si) =
          some (some ⟨hm.d + 2, hm.a⟩) := rfl
      obtain ⟨kh, s2, hrh, hrip2, hpost⟩ := hand_str cfg st inv
      have hs0 : hstep pr wd ((wd.ann.lookup i).getD cur) (Instr.str si) =
          some (some ⟨hm.d + 2, hm.a⟩) := by
        rw [hge]; exact hs
      cases rest with
      | nil => simp [Sporos.walk, hs0] at hw
      | cons y ys =>
        have hw2 : (wd.ann.lookup (i + 1) = none ∨
              wd.ann.lookup (i + 1) = some ⟨hm.d + 2, hm.a⟩) ∧
            Sporos.walk pr wd (y :: ys) (i + 1) ⟨hm.d + 2, hm.a⟩ = true := by
          simp only [Sporos.walk, hs0, Bool.and_eq_true] at hw
          refine ⟨?_, hw.2⟩
          cases hl : wd.ann.lookup (i + 1) with
          | none => exact Or.inl rfl
          | some a =>
            have hbe : (a == ⟨hm.d + 2, hm.a⟩) = true := by
              have h1 := hw.1
              rw [hl] at h1
              exact h1
            exact Or.inr (by rw [height_eq_of_beq hbe])
        have hi1 : i + 1 < wd.code.length := by
          have h := inv.isum
          simp only [List.length_cons] at h
          omega
        obtain ⟨k3, s3, hr3, inv3, hfr3i, hsh3i⟩ :=
          fall_finish1 cfg st inv hs hw2.1 hi1 rfl hrh hrip2 hpost
        obtain ⟨k4, s4, hr4, hrip4, hr14, hr64, hr74, hr44, hfr4, hsh4⟩ :=
          ih (i + 1) (c + 5) ⟨hm.d + 2, hm.a⟩ ⟨hm.d + 2, hm.a⟩ s3 inv3 hw2.2
        refine ⟨k3 + k4, s4, by rw [run_add im k3 k4, hr3, hr4], hrip4, ?_, ?_,
          hr74, hr44,
          fun x m hx => (hfr4 x m hx).trans (hfr3i x m hx), hsh4.trans hsh3i⟩
        · have e : (i + 1) + (y :: ys).length =
              i + (Instr.str si :: y :: ys).length := by
            simp only [List.length_cons]; omega
          rw [← e]; exact hr14
        · have e : (c + 5) + codeBytes (y :: ys) =
              c + codeBytes (Instr.str si :: y :: ys) := by
            show (c + 5) + codeBytes (y :: ys) =
              c + (5 + codeBytes (y :: ys))
            omega
          rw [← e]; exact hr64
    | pick pk =>
      cases hs : hstep pr wd hm (Instr.pick pk) with
      | none =>
        have hs0 : hstep pr wd ((wd.ann.lookup i).getD cur) (Instr.pick pk) =
            none := by
          rw [hge]; exact hs
        simp [Sporos.walk, hs0] at hw
      | some o =>
        cases o with
        | none => by_cases hcond : (pk < hm.d) <;> simp [hstep, hcond] at hs
        | some h' =>
          obtain ⟨-, hh'⟩ := hstep_pick hs
          subst hh'
          obtain ⟨kh, s2, hrh, hrip2, hpost⟩ := hand_pick cfg st inv hs
          have hs0 : hstep pr wd ((wd.ann.lookup i).getD cur) (Instr.pick pk) =
              some (some ⟨hm.d + 1, hm.a⟩) := by
            rw [hge]; exact hs
          cases rest with
          | nil => simp [Sporos.walk, hs0] at hw
          | cons y ys =>
            have hw2 : (wd.ann.lookup (i + 1) = none ∨
                  wd.ann.lookup (i + 1) = some ⟨hm.d + 1, hm.a⟩) ∧
                Sporos.walk pr wd (y :: ys) (i + 1) ⟨hm.d + 1, hm.a⟩ = true := by
              simp only [Sporos.walk, hs0, Bool.and_eq_true] at hw
              refine ⟨?_, hw.2⟩
              cases hl : wd.ann.lookup (i + 1) with
              | none => exact Or.inl rfl
              | some a =>
                have hbe : (a == ⟨hm.d + 1, hm.a⟩) = true := by
                  have h1 := hw.1
                  rw [hl] at h1
                  exact h1
                exact Or.inr (by rw [height_eq_of_beq hbe])
            have hi1 : i + 1 < wd.code.length := by
              have h := inv.isum
              simp only [List.length_cons] at h
              omega
            obtain ⟨k3, s3, hr3, inv3, hfr3i, hsh3i⟩ :=
              fall_finish1 cfg st inv hs hw2.1 hi1 rfl hrh hrip2 hpost
            obtain ⟨k4, s4, hr4, hrip4, hr14, hr64, hr74, hr44, hfr4, hsh4⟩ :=
              ih (i + 1) (c + 5) ⟨hm.d + 1, hm.a⟩ ⟨hm.d + 1, hm.a⟩ s3 inv3 hw2.2
            refine ⟨k3 + k4, s4, by rw [run_add im k3 k4, hr3, hr4], hrip4, ?_, ?_,
              hr74, hr44,
              fun x m hx => (hfr4 x m hx).trans (hfr3i x m hx), hsh4.trans hsh3i⟩
            · have e : (i + 1) + (y :: ys).length =
                  i + (Instr.pick pk :: y :: ys).length := by
                simp only [List.length_cons]; omega
              rw [← e]; exact hr14
            · have e : (c + 5) + codeBytes (y :: ys) =
                  c + codeBytes (Instr.pick pk :: y :: ys) := by
                show (c + 5) + codeBytes (y :: ys) =
                  c + (5 + codeBytes (y :: ys))
                omega
              rw [← e]; exact hr64
    | tor =>
      cases hs : hstep pr wd hm (Instr.tor) with
      | none =>
        have hs0 : hstep pr wd ((wd.ann.lookup i).getD cur) (Instr.tor) =
            none := by
          rw [hge]; exact hs
        simp [Sporos.walk, hs0] at hw
      | some o =>
        cases o with
        | none => by_cases hcond : (1 ≤ hm.d) <;> simp [hstep, hcond] at hs
        | some h' =>
          obtain ⟨-, hh'⟩ := hstep_tor hs
          subst hh'
          obtain ⟨kh, s2, hrh, hrip2, hpost⟩ := hand_tor cfg st inv hs
          have hs0 : hstep pr wd ((wd.ann.lookup i).getD cur) (Instr.tor) =
              some (some ⟨hm.d - 1, hm.a + 1⟩) := by
            rw [hge]; exact hs
          cases rest with
          | nil => simp [Sporos.walk, hs0] at hw
          | cons y ys =>
            have hw2 : (wd.ann.lookup (i + 1) = none ∨
                  wd.ann.lookup (i + 1) = some ⟨hm.d - 1, hm.a + 1⟩) ∧
                Sporos.walk pr wd (y :: ys) (i + 1) ⟨hm.d - 1, hm.a + 1⟩ = true := by
              simp only [Sporos.walk, hs0, Bool.and_eq_true] at hw
              refine ⟨?_, hw.2⟩
              cases hl : wd.ann.lookup (i + 1) with
              | none => exact Or.inl rfl
              | some a =>
                have hbe : (a == ⟨hm.d - 1, hm.a + 1⟩) = true := by
                  have h1 := hw.1
                  rw [hl] at h1
                  exact h1
                exact Or.inr (by rw [height_eq_of_beq hbe])
            have hi1 : i + 1 < wd.code.length := by
              have h := inv.isum
              simp only [List.length_cons] at h
              omega
            obtain ⟨k3, s3, hr3, inv3, hfr3i, hsh3i⟩ :=
              fall_finish1 cfg st inv hs hw2.1 hi1 rfl hrh hrip2 hpost
            obtain ⟨k4, s4, hr4, hrip4, hr14, hr64, hr74, hr44, hfr4, hsh4⟩ :=
              ih (i + 1) (c + 1) ⟨hm.d - 1, hm.a + 1⟩ ⟨hm.d - 1, hm.a + 1⟩ s3 inv3 hw2.2
            refine ⟨k3 + k4, s4, by rw [run_add im k3 k4, hr3, hr4], hrip4, ?_, ?_,
              hr74, hr44,
              fun x m hx => (hfr4 x m hx).trans (hfr3i x m hx), hsh4.trans hsh3i⟩
            · have e : (i + 1) + (y :: ys).length =
                  i + (Instr.tor :: y :: ys).length := by
                simp only [List.length_cons]; omega
              rw [← e]; exact hr14
            · have e : (c + 1) + codeBytes (y :: ys) =
                  c + codeBytes (Instr.tor :: y :: ys) := by
                show (c + 1) + codeBytes (y :: ys) =
                  c + (1 + codeBytes (y :: ys))
                omega
              rw [← e]; exact hr64
    | fromr =>
      cases hs : hstep pr wd hm (Instr.fromr) with
      | none =>
        have hs0 : hstep pr wd ((wd.ann.lookup i).getD cur) (Instr.fromr) =
            none := by
          rw [hge]; exact hs
        simp [Sporos.walk, hs0] at hw
      | some o =>
        cases o with
        | none => by_cases hcond : (1 ≤ hm.a) <;> simp [hstep, hcond] at hs
        | some h' =>
          obtain ⟨-, hh'⟩ := hstep_fromr hs
          subst hh'
          obtain ⟨kh, s2, hrh, hrip2, hpost⟩ := hand_fromr cfg st inv hs
          have hs0 : hstep pr wd ((wd.ann.lookup i).getD cur) (Instr.fromr) =
              some (some ⟨hm.d + 1, hm.a - 1⟩) := by
            rw [hge]; exact hs
          cases rest with
          | nil => simp [Sporos.walk, hs0] at hw
          | cons y ys =>
            have hw2 : (wd.ann.lookup (i + 1) = none ∨
                  wd.ann.lookup (i + 1) = some ⟨hm.d + 1, hm.a - 1⟩) ∧
                Sporos.walk pr wd (y :: ys) (i + 1) ⟨hm.d + 1, hm.a - 1⟩ = true := by
              simp only [Sporos.walk, hs0, Bool.and_eq_true] at hw
              refine ⟨?_, hw.2⟩
              cases hl : wd.ann.lookup (i + 1) with
              | none => exact Or.inl rfl
              | some a =>
                have hbe : (a == ⟨hm.d + 1, hm.a - 1⟩) = true := by
                  have h1 := hw.1
                  rw [hl] at h1
                  exact h1
                exact Or.inr (by rw [height_eq_of_beq hbe])
            have hi1 : i + 1 < wd.code.length := by
              have h := inv.isum
              simp only [List.length_cons] at h
              omega
            obtain ⟨k3, s3, hr3, inv3, hfr3i, hsh3i⟩ :=
              fall_finish1 cfg st inv hs hw2.1 hi1 rfl hrh hrip2 hpost
            obtain ⟨k4, s4, hr4, hrip4, hr14, hr64, hr74, hr44, hfr4, hsh4⟩ :=
              ih (i + 1) (c + 1) ⟨hm.d + 1, hm.a - 1⟩ ⟨hm.d + 1, hm.a - 1⟩ s3 inv3 hw2.2
            refine ⟨k3 + k4, s4, by rw [run_add im k3 k4, hr3, hr4], hrip4, ?_, ?_,
              hr74, hr44,
              fun x m hx => (hfr4 x m hx).trans (hfr3i x m hx), hsh4.trans hsh3i⟩
            · have e : (i + 1) + (y :: ys).length =
                  i + (Instr.fromr :: y :: ys).length := by
                simp only [List.length_cons]; omega
              rw [← e]; exact hr14
            · have e : (c + 1) + codeBytes (y :: ys) =
                  c + codeBytes (Instr.fromr :: y :: ys) := by
                show (c + 1) + codeBytes (y :: ys) =
                  c + (1 + codeBytes (y :: ys))
                omega
              rw [← e]; exact hr64
    | rfetch =>
      cases hs : hstep pr wd hm (Instr.rfetch) with
      | none =>
        have hs0 : hstep pr wd ((wd.ann.lookup i).getD cur) (Instr.rfetch) =
            none := by
          rw [hge]; exact hs
        simp [Sporos.walk, hs0] at hw
      | some o =>
        cases o with
        | none => by_cases hcond : (1 ≤ hm.a) <;> simp [hstep, hcond] at hs
        | some h' =>
          obtain ⟨-, hh'⟩ := hstep_rfetch hs
          subst hh'
          obtain ⟨kh, s2, hrh, hrip2, hpost⟩ := hand_rfetch cfg st inv hs
          have hs0 : hstep pr wd ((wd.ann.lookup i).getD cur) (Instr.rfetch) =
              some (some ⟨hm.d + 1, hm.a⟩) := by
            rw [hge]; exact hs
          cases rest with
          | nil => simp [Sporos.walk, hs0] at hw
          | cons y ys =>
            have hw2 : (wd.ann.lookup (i + 1) = none ∨
                  wd.ann.lookup (i + 1) = some ⟨hm.d + 1, hm.a⟩) ∧
                Sporos.walk pr wd (y :: ys) (i + 1) ⟨hm.d + 1, hm.a⟩ = true := by
              simp only [Sporos.walk, hs0, Bool.and_eq_true] at hw
              refine ⟨?_, hw.2⟩
              cases hl : wd.ann.lookup (i + 1) with
              | none => exact Or.inl rfl
              | some a =>
                have hbe : (a == ⟨hm.d + 1, hm.a⟩) = true := by
                  have h1 := hw.1
                  rw [hl] at h1
                  exact h1
                exact Or.inr (by rw [height_eq_of_beq hbe])
            have hi1 : i + 1 < wd.code.length := by
              have h := inv.isum
              simp only [List.length_cons] at h
              omega
            obtain ⟨k3, s3, hr3, inv3, hfr3i, hsh3i⟩ :=
              fall_finish1 cfg st inv hs hw2.1 hi1 rfl hrh hrip2 hpost
            obtain ⟨k4, s4, hr4, hrip4, hr14, hr64, hr74, hr44, hfr4, hsh4⟩ :=
              ih (i + 1) (c + 1) ⟨hm.d + 1, hm.a⟩ ⟨hm.d + 1, hm.a⟩ s3 inv3 hw2.2
            refine ⟨k3 + k4, s4, by rw [run_add im k3 k4, hr3, hr4], hrip4, ?_, ?_,
              hr74, hr44,
              fun x m hx => (hfr4 x m hx).trans (hfr3i x m hx), hsh4.trans hsh3i⟩
            · have e : (i + 1) + (y :: ys).length =
                  i + (Instr.rfetch :: y :: ys).length := by
                simp only [List.length_cons]; omega
              rw [← e]; exact hr14
            · have e : (c + 1) + codeBytes (y :: ys) =
                  c + codeBytes (Instr.rfetch :: y :: ys) := by
                show (c + 1) + codeBytes (y :: ys) =
                  c + (1 + codeBytes (y :: ys))
                omega
              rw [← e]; exact hr64
    | locget li =>
      cases hs : hstep pr wd hm (Instr.locget li) with
      | none =>
        have hs0 : hstep pr wd ((wd.ann.lookup i).getD cur) (Instr.locget li) =
            none := by
          rw [hge]; exact hs
        simp [Sporos.walk, hs0] at hw
      | some o =>
        cases o with
        | none => by_cases hcond : (li < wd.locals) <;> simp [hstep, hcond] at hs
        | some h' =>
          obtain ⟨-, hh'⟩ := hstep_locget hs
          subst hh'
          obtain ⟨kh, s2, hrh, hrip2, hpost⟩ := hand_locget cfg st inv hs
          have hs0 : hstep pr wd ((wd.ann.lookup i).getD cur) (Instr.locget li) =
              some (some ⟨hm.d + 1, hm.a⟩) := by
            rw [hge]; exact hs
          cases rest with
          | nil => simp [Sporos.walk, hs0] at hw
          | cons y ys =>
            have hw2 : (wd.ann.lookup (i + 1) = none ∨
                  wd.ann.lookup (i + 1) = some ⟨hm.d + 1, hm.a⟩) ∧
                Sporos.walk pr wd (y :: ys) (i + 1) ⟨hm.d + 1, hm.a⟩ = true := by
              simp only [Sporos.walk, hs0, Bool.and_eq_true] at hw
              refine ⟨?_, hw.2⟩
              cases hl : wd.ann.lookup (i + 1) with
              | none => exact Or.inl rfl
              | some a =>
                have hbe : (a == ⟨hm.d + 1, hm.a⟩) = true := by
                  have h1 := hw.1
                  rw [hl] at h1
                  exact h1
                exact Or.inr (by rw [height_eq_of_beq hbe])
            have hi1 : i + 1 < wd.code.length := by
              have h := inv.isum
              simp only [List.length_cons] at h
              omega
            obtain ⟨k3, s3, hr3, inv3, hfr3i, hsh3i⟩ :=
              fall_finish1 cfg st inv hs hw2.1 hi1 rfl hrh hrip2 hpost
            obtain ⟨k4, s4, hr4, hrip4, hr14, hr64, hr74, hr44, hfr4, hsh4⟩ :=
              ih (i + 1) (c + 5) ⟨hm.d + 1, hm.a⟩ ⟨hm.d + 1, hm.a⟩ s3 inv3 hw2.2
            refine ⟨k3 + k4, s4, by rw [run_add im k3 k4, hr3, hr4], hrip4, ?_, ?_,
              hr74, hr44,
              fun x m hx => (hfr4 x m hx).trans (hfr3i x m hx), hsh4.trans hsh3i⟩
            · have e : (i + 1) + (y :: ys).length =
                  i + (Instr.locget li :: y :: ys).length := by
                simp only [List.length_cons]; omega
              rw [← e]; exact hr14
            · have e : (c + 5) + codeBytes (y :: ys) =
                  c + codeBytes (Instr.locget li :: y :: ys) := by
                show (c + 5) + codeBytes (y :: ys) =
                  c + (5 + codeBytes (y :: ys))
                omega
              rw [← e]; exact hr64
    | locset ls =>
      cases hs : hstep pr wd hm (Instr.locset ls) with
      | none =>
        have hs0 : hstep pr wd ((wd.ann.lookup i).getD cur) (Instr.locset ls) =
            none := by
          rw [hge]; exact hs
        simp [Sporos.walk, hs0] at hw
      | some o =>
        cases o with
        | none => by_cases hcond : (1 ≤ hm.d ∧ ls < wd.locals) <;> simp [hstep, hcond] at hs
        | some h' =>
          obtain ⟨-, -, hh'⟩ := hstep_locset hs
          subst hh'
          obtain ⟨kh, s2, hrh, hrip2, hpost⟩ := hand_locset cfg st inv hs
          have hs0 : hstep pr wd ((wd.ann.lookup i).getD cur) (Instr.locset ls) =
              some (some ⟨hm.d - 1, hm.a⟩) := by
            rw [hge]; exact hs
          cases rest with
          | nil => simp [Sporos.walk, hs0] at hw
          | cons y ys =>
            have hw2 : (wd.ann.lookup (i + 1) = none ∨
                  wd.ann.lookup (i + 1) = some ⟨hm.d - 1, hm.a⟩) ∧
                Sporos.walk pr wd (y :: ys) (i + 1) ⟨hm.d - 1, hm.a⟩ = true := by
              simp only [Sporos.walk, hs0, Bool.and_eq_true] at hw
              refine ⟨?_, hw.2⟩
              cases hl : wd.ann.lookup (i + 1) with
              | none => exact Or.inl rfl
              | some a =>
                have hbe : (a == ⟨hm.d - 1, hm.a⟩) = true := by
                  have h1 := hw.1
                  rw [hl] at h1
                  exact h1
                exact Or.inr (by rw [height_eq_of_beq hbe])
            have hi1 : i + 1 < wd.code.length := by
              have h := inv.isum
              simp only [List.length_cons] at h
              omega
            obtain ⟨k3, s3, hr3, inv3, hfr3i, hsh3i⟩ :=
              fall_finish1 cfg st inv hs hw2.1 hi1 rfl hrh hrip2 hpost
            obtain ⟨k4, s4, hr4, hrip4, hr14, hr64, hr74, hr44, hfr4, hsh4⟩ :=
              ih (i + 1) (c + 5) ⟨hm.d - 1, hm.a⟩ ⟨hm.d - 1, hm.a⟩ s3 inv3 hw2.2
            refine ⟨k3 + k4, s4, by rw [run_add im k3 k4, hr3, hr4], hrip4, ?_, ?_,
              hr74, hr44,
              fun x m hx => (hfr4 x m hx).trans (hfr3i x m hx), hsh4.trans hsh3i⟩
            · have e : (i + 1) + (y :: ys).length =
                  i + (Instr.locset ls :: y :: ys).length := by
                simp only [List.length_cons]; omega
              rw [← e]; exact hr14
            · have e : (c + 5) + codeBytes (y :: ys) =
                  c + codeBytes (Instr.locset ls :: y :: ys) := by
                show (c + 5) + codeBytes (y :: ys) =
                  c + (5 + codeBytes (y :: ys))
                omega
              rw [← e]; exact hr64
    | jmp jt =>
      cases hs : hstep pr wd hm (Instr.jmp jt) with
      | none =>
        have hs0 : hstep pr wd ((wd.ann.lookup i).getD cur) (Instr.jmp jt) =
            none := by
          rw [hge]; exact hs
        simp [Sporos.walk, hs0] at hw
      | some o =>
        cases o with
        | some h' => by_cases hcond : (wd.ann.lookup jt = some hm) <;>
            simp [hstep, hcond] at hs
        | none =>
          obtain ⟨kh, s2, hrh, hrip2, hpost⟩ := hand_jmp cfg st inv hs
          have hs0 : hstep pr wd ((wd.ann.lookup i).getD cur) (Instr.jmp jt) =
              some none := by
            rw [hge]; exact hs
          cases rest with
          | nil =>
            obtain ⟨k3, s4, hr3, hrip4, hr14, hr64, hr74, hr44, hfr4, hsh4⟩ :=
              none_finish_nil1 cfg st inv hs rfl hrh hrip2 hpost
            refine ⟨k3, s4, hr3, hrip4, ?_, ?_, hr74, hr44, hfr4, hsh4⟩
            · simpa using hr14
            · have e : c + codeBytes [Instr.jmp jt] = c + 5 := rfl
              rw [e]; exact hr64
          | cons y ys =>
            have hi1 : i + 1 < wd.code.length := by
              have h := inv.isum
              simp only [List.length_cons] at h
              omega
            have hwrest : Sporos.walk pr wd (y :: ys) (i + 1)
                ((wd.ann.lookup (i + 1)).getD ⟨0, 0⟩) = true := by
              simp only [Sporos.walk, hs0] at hw
              exact hw
            obtain ⟨k3, s3, hr3, inv3, hfr3i, hsh3i⟩ :=
              none_finish_cons1 cfg st inv hs hi1 rfl hrh hrip2 hpost
            obtain ⟨k4, s4, hr4, hrip4, hr14, hr64, hr74, hr44, hfr4, hsh4⟩ :=
              ih (i + 1) (c + 5) _ _ s3 inv3 hwrest
            refine ⟨k3 + k4, s4, by rw [run_add im k3 k4, hr3, hr4], hrip4, ?_, ?_,
              hr74, hr44,
              fun x m hx => (hfr4 x m hx).trans (hfr3i x m hx), hsh4.trans hsh3i⟩
            · have e : (i + 1) + (y :: ys).length =
                  i + (Instr.jmp jt :: y :: ys).length := by
                simp only [List.length_cons]; omega
              rw [← e]; exact hr14
            · have e : (c + 5) + codeBytes (y :: ys) =
                  c + codeBytes (Instr.jmp jt :: y :: ys) := by
                show (c + 5) + codeBytes (y :: ys) =
                  c + (5 + codeBytes (y :: ys))
                omega
              rw [← e]; exact hr64
    | jz zt =>
      cases hs : hstep pr wd hm (Instr.jz zt) with
      | none =>
        have hs0 : hstep pr wd ((wd.ann.lookup i).getD cur) (Instr.jz zt) =
            none := by
          rw [hge]; exact hs
        simp [Sporos.walk, hs0] at hw
      | some o =>
        cases o with
        | none => by_cases hcond : (1 ≤ hm.d ∧ wd.ann.lookup zt = some ⟨hm.d - 1, hm.a⟩) <;> simp [hstep, hcond] at hs
        | some h' =>
          obtain ⟨-, -, hh'⟩ := hstep_jz hs
          subst hh'
          obtain ⟨kh, s2, hrh, hrip2, hpost⟩ := hand_jz cfg st inv hs
          have hs0 : hstep pr wd ((wd.ann.lookup i).getD cur) (Instr.jz zt) =
              some (some ⟨hm.d - 1, hm.a⟩) := by
            rw [hge]; exact hs
          cases rest with
          | nil => simp [Sporos.walk, hs0] at hw
          | cons y ys =>
            have hw2 : (wd.ann.lookup (i + 1) = none ∨
                  wd.ann.lookup (i + 1) = some ⟨hm.d - 1, hm.a⟩) ∧
                Sporos.walk pr wd (y :: ys) (i + 1) ⟨hm.d - 1, hm.a⟩ = true := by
              simp only [Sporos.walk, hs0, Bool.and_eq_true] at hw
              refine ⟨?_, hw.2⟩
              cases hl : wd.ann.lookup (i + 1) with
              | none => exact Or.inl rfl
              | some a =>
                have hbe : (a == ⟨hm.d - 1, hm.a⟩) = true := by
                  have h1 := hw.1
                  rw [hl] at h1
                  exact h1
                exact Or.inr (by rw [height_eq_of_beq hbe])
            have hi1 : i + 1 < wd.code.length := by
              have h := inv.isum
              simp only [List.length_cons] at h
              omega
            obtain ⟨k3, s3, hr3, inv3, hfr3i, hsh3i⟩ :=
              fall_finish1 cfg st inv hs hw2.1 hi1 rfl hrh hrip2 hpost
            obtain ⟨k4, s4, hr4, hrip4, hr14, hr64, hr74, hr44, hfr4, hsh4⟩ :=
              ih (i + 1) (c + 5) ⟨hm.d - 1, hm.a⟩ ⟨hm.d - 1, hm.a⟩ s3 inv3 hw2.2
            refine ⟨k3 + k4, s4, by rw [run_add im k3 k4, hr3, hr4], hrip4, ?_, ?_,
              hr74, hr44,
              fun x m hx => (hfr4 x m hx).trans (hfr3i x m hx), hsh4.trans hsh3i⟩
            · have e : (i + 1) + (y :: ys).length =
                  i + (Instr.jz zt :: y :: ys).length := by
                simp only [List.length_cons]; omega
              rw [← e]; exact hr14
            · have e : (c + 5) + codeBytes (y :: ys) =
                  c + codeBytes (Instr.jz zt :: y :: ys) := by
                show (c + 5) + codeBytes (y :: ys) =
                  c + (5 + codeBytes (y :: ys))
                omega
              rw [← e]; exact hr64
    | jnz nt =>
      cases hs : hstep pr wd hm (Instr.jnz nt) with
      | none =>
        have hs0 : hstep pr wd ((wd.ann.lookup i).getD cur) (Instr.jnz nt) =
            none := by
          rw [hge]; exact hs
        simp [Sporos.walk, hs0] at hw
      | some o =>
        cases o with
        | none => by_cases hcond : (1 ≤ hm.d ∧ wd.ann.lookup nt = some ⟨hm.d - 1, hm.a⟩) <;> simp [hstep, hcond] at hs
        | some h' =>
          obtain ⟨-, -, hh'⟩ := hstep_jnz hs
          subst hh'
          obtain ⟨kh, s2, hrh, hrip2, hpost⟩ := hand_jnz cfg st inv hs
          have hs0 : hstep pr wd ((wd.ann.lookup i).getD cur) (Instr.jnz nt) =
              some (some ⟨hm.d - 1, hm.a⟩) := by
            rw [hge]; exact hs
          cases rest with
          | nil => simp [Sporos.walk, hs0] at hw
          | cons y ys =>
            have hw2 : (wd.ann.lookup (i + 1) = none ∨
                  wd.ann.lookup (i + 1) = some ⟨hm.d - 1, hm.a⟩) ∧
                Sporos.walk pr wd (y :: ys) (i + 1) ⟨hm.d - 1, hm.a⟩ = true := by
              simp only [Sporos.walk, hs0, Bool.and_eq_true] at hw
              refine ⟨?_, hw.2⟩
              cases hl : wd.ann.lookup (i + 1) with
              | none => exact Or.inl rfl
              | some a =>
                have hbe : (a == ⟨hm.d - 1, hm.a⟩) = true := by
                  have h1 := hw.1
                  rw [hl] at h1
                  exact h1
                exact Or.inr (by rw [height_eq_of_beq hbe])
            have hi1 : i + 1 < wd.code.length := by
              have h := inv.isum
              simp only [List.length_cons] at h
              omega
            obtain ⟨k3, s3, hr3, inv3, hfr3i, hsh3i⟩ :=
              fall_finish1 cfg st inv hs hw2.1 hi1 rfl hrh hrip2 hpost
            obtain ⟨k4, s4, hr4, hrip4, hr14, hr64, hr74, hr44, hfr4, hsh4⟩ :=
              ih (i + 1) (c + 5) ⟨hm.d - 1, hm.a⟩ ⟨hm.d - 1, hm.a⟩ s3 inv3 hw2.2
            refine ⟨k3 + k4, s4, by rw [run_add im k3 k4, hr3, hr4], hrip4, ?_, ?_,
              hr74, hr44,
              fun x m hx => (hfr4 x m hx).trans (hfr3i x m hx), hsh4.trans hsh3i⟩
            · have e : (i + 1) + (y :: ys).length =
                  i + (Instr.jnz nt :: y :: ys).length := by
                simp only [List.length_cons]; omega
              rw [← e]; exact hr14
            · have e : (c + 5) + codeBytes (y :: ys) =
                  c + codeBytes (Instr.jnz nt :: y :: ys) := by
                show (c + 5) + codeBytes (y :: ys) =
                  c + (5 + codeBytes (y :: ys))
                omega
              rw [← e]; exact hr64
    | call cwidx =>
      cases hs : hstep pr wd hm (Instr.call cwidx) with
      | none =>
        have hs0 : hstep pr wd ((wd.ann.lookup i).getD cur) (Instr.call cwidx) =
            none := by
          rw [hge]; exact hs
        simp [Sporos.walk, hs0] at hw
      | some o =>
        cases o with
        | none =>
          simp only [hstep] at hs
          cases hc : pr.words[cwidx]? with
          | none => rw [hc] at hs; cases hs
          | some cw =>
            rw [hc] at hs
            by_cases hg : cw.ins ≤ hm.d <;> simp [hg] at hs
        | some h' =>
          obtain ⟨kh, s2, hrh, hrip2, hpost⟩ := hand_call cfg st inv hs
          have hs0 : hstep pr wd ((wd.ann.lookup i).getD cur) (Instr.call cwidx) =
              some (some h') := by
            rw [hge]; exact hs
          cases rest with
          | nil => simp [Sporos.walk, hs0] at hw
          | cons y ys =>
            have hw2 : (wd.ann.lookup (i + 1) = none ∨
                  wd.ann.lookup (i + 1) = some h') ∧
                Sporos.walk pr wd (y :: ys) (i + 1) h' = true := by
              simp only [Sporos.walk, hs0, Bool.and_eq_true] at hw
              refine ⟨?_, hw.2⟩
              cases hl : wd.ann.lookup (i + 1) with
              | none => exact Or.inl rfl
              | some a =>
                have hbe : (a == h') = true := by
                  have h1 := hw.1
                  rw [hl] at h1
                  exact h1
                exact Or.inr (by rw [height_eq_of_beq hbe])
            have hi1 : i + 1 < wd.code.length := by
              have h := inv.isum
              simp only [List.length_cons] at h
              omega
            obtain ⟨k3, s3, hr3, inv3, hfr3i, hsh3i⟩ :=
              fall_finish1 cfg st inv hs hw2.1 hi1 rfl hrh hrip2 hpost
            obtain ⟨k4, s4, hr4, hrip4, hr14, hr64, hr74, hr44, hfr4, hsh4⟩ :=
              ih (i + 1) (c + 5) h' h' s3 inv3 hw2.2
            refine ⟨k3 + k4, s4, by rw [run_add im k3 k4, hr3, hr4], hrip4, ?_, ?_,
              hr74, hr44,
              fun x m hx => (hfr4 x m hx).trans (hfr3i x m hx), hsh4.trans hsh3i⟩
            · have e : (i + 1) + (y :: ys).length =
                  i + (Instr.call cwidx :: y :: ys).length := by
                simp only [List.length_cons]; omega
              rw [← e]; exact hr14
            · have e : (c + 5) + codeBytes (y :: ys) =
                  c + codeBytes (Instr.call cwidx :: y :: ys) := by
                show (c + 5) + codeBytes (y :: ys) =
                  c + (5 + codeBytes (y :: ys))
                omega
              rw [← e]; exact hr64
    | ret =>
      cases hs : hstep pr wd hm Instr.ret with
      | none =>
        have hs0 : hstep pr wd ((wd.ann.lookup i).getD cur) (Instr.ret) =
            none := by
          rw [hge]; exact hs
        simp [Sporos.walk, hs0] at hw
      | some o =>
        cases o with
        | some h' => by_cases hcond : (hm = ⟨wd.outs, 0⟩) <;>
            simp [hstep, hcond] at hs
        | none =>
          obtain ⟨kh, s2, hrh, hrip2, hpost⟩ := hand_ret cfg st inv hs
          have hs0 : hstep pr wd ((wd.ann.lookup i).getD cur) (Instr.ret) =
              some none := by
            rw [hge]; exact hs
          cases rest with
          | nil =>
            obtain ⟨k3, s4, hr3, hrip4, hr14, hr64, hr74, hr44, hfr4, hsh4⟩ :=
              none_finish_nil1 cfg st inv hs rfl hrh hrip2 hpost
            refine ⟨k3, s4, hr3, hrip4, ?_, ?_, hr74, hr44, hfr4, hsh4⟩
            · simpa using hr14
            · have e : c + codeBytes [Instr.ret] = c + 1 := rfl
              rw [e]; exact hr64
          | cons y ys =>
            have hi1 : i + 1 < wd.code.length := by
              have h := inv.isum
              simp only [List.length_cons] at h
              omega
            have hwrest : Sporos.walk pr wd (y :: ys) (i + 1)
                ((wd.ann.lookup (i + 1)).getD ⟨0, 0⟩) = true := by
              simp only [Sporos.walk, hs0] at hw
              exact hw
            obtain ⟨k3, s3, hr3, inv3, hfr3i, hsh3i⟩ :=
              none_finish_cons1 cfg st inv hs hi1 rfl hrh hrip2 hpost
            obtain ⟨k4, s4, hr4, hrip4, hr14, hr64, hr74, hr44, hfr4, hsh4⟩ :=
              ih (i + 1) (c + 1) _ _ s3 inv3 hwrest
            refine ⟨k3 + k4, s4, by rw [run_add im k3 k4, hr3, hr4], hrip4, ?_, ?_,
              hr74, hr44,
              fun x m hx => (hfr4 x m hx).trans (hfr3i x m hx), hsh4.trans hsh3i⟩
            · have e : (i + 1) + (y :: ys).length =
                  i + (Instr.ret :: y :: ys).length := by
                simp only [List.length_cons]; omega
              rw [← e]; exact hr14
            · have e : (c + 1) + codeBytes (y :: ys) =
                  c + codeBytes (Instr.ret :: y :: ys) := by
                show (c + 1) + codeBytes (y :: ys) =
                  c + (1 + codeBytes (y :: ys))
                omega
              rw [← e]; exact hr64


end SporosX1
