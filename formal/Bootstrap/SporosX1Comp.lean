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



/-! ### Around the walk: `check_word`, `annval`, `codewalk`, `word_done` -/

def aCheckWord : Nat := base + 0x433
def aAnnval : Nat := base + 0x453
def aCodewalk : Nat := base + 0x4d5

/-- Decode facts for the code around the walk. -/
structure OuterCode (im : Nat → Option (Ins × Nat)) : Prop where
  wd0 : im aWordDone = some (.aluI .add 7 64, 4)
  wd1 : im (aWordDone + 4) = some (.aluI .sub 5 1, 4)
  wd2 : im (aWordDone + 8) = some (.jmp aCheckWord, 5)
  ac0 : im aAccept = some (.pushI 60, 2)
  ac1 : im (aAccept + 2) = some (.pop 0, 1)
  ac2 : im (aAccept + 3) = some (.pushI 0, 2)
  ac3 : im (aAccept + 5) = some (.pop 7, 1)
  ac4 : im (aAccept + 6) = some (.syscall, 2)
  cw0 : im aCheckWord = some (.test 5 5, 3)
  cw1 : im (aCheckWord + 3) = some (.jcc 5 (aCheckWord + 10), 2)
  cw2 : im (aCheckWord + 5) = some (.jmp aAccept, 5)
  cw3 : im (aCheckWord + 10) = some (.ld 0 7 24, 4)
  cw4 : im (aCheckWord + 14) = some (.test 0 0, 3)
  cw5 : im (aCheckWord + 17) = some (.jcc 5 (aCheckWord + 24), 2)
  cw6 : im (aCheckWord + 24) = some (.ld 6 7 48, 4)
  cw7 : im (aCheckWord + 28) = some (.ld 1 7 32, 4)

theorem step_pushI {im : Nat → Option (Ins × Nat)} {s : X86Int.St}
    {imm len sp : Nat} {m : Mem}
    (hi : im s.rip = some (.pushI imm, len)) (hsp : s.reg 4 = some sp)
    (hm : wr64 s.mem ((sp + W - 8) % W) (imm % W) = some m) :
    X86Int.step im s =
      .ok { s.set 4 (some ((sp + W - 8) % W)) with mem := m, rip := s.rip + len } := by
  simp only [X86Int.step, hi]
  exact exec_pushI hsp hm

theorem step_syscall_exit {im : Nat → Option (Ins × Nat)} {s : X86Int.St}
    {len code a2 a3 a4 a5 a6 : Nat}
    (hi : im s.rip = some (.syscall, len)) (h0 : s.reg 0 = some 60)
    (h7 : s.reg 7 = some code) (h6 : s.reg 6 = some a2)
    (h2 : s.reg 2 = some a3) (h10 : s.reg 10 = some a4)
    (h8 : s.reg 8 = some a5) (h9 : s.reg 9 = some a6) :
    X86Int.step im s = .exit (code % 256) := by
  have hsc : X86Env.syscall s.k s.mem 60 code a2 a3 a4 a5 a6 =
      some (X86Env.SysRes.exit (code % 256)) := rfl
  simp only [X86Int.step, exec, hi, h0, h7, h6, h2, h10, h8, h9, hsc]

/-- `word_done`: advance the entry and decrement the word count. -/
theorem WordDone_triple {im : Nat → Option (Ins × Nat)} (oc : OuterCode im) :
    ∀ (w n : Nat), 1 ≤ n → n < W → w + 64 < W →
    ∀ s, s.rip = aWordDone → s.reg 7 = some w → s.reg 5 = some n →
      ∃ k s', run im k (Out.ok s) = Out.ok s' ∧ s'.rip = aCheckWord ∧
        s'.reg 7 = some (w + 64) ∧ s'.reg 5 = some (n - 1) ∧
        (∀ q, q ≠ 5 → q ≠ 7 → s'.reg q = s.reg q) ∧ s'.mem = s.mem ∧
        s'.regs.length = s.regs.length := by
  intro w n hn1 hnW hw64 s hrip hr7 hr5
  have h64W : 64 < W := by unfold W; decide
  have h1W : 1 < W := by unfold W; decide
  have e1 : X86Int.step im s = .ok { s.set 7 (some (aluF .add w (64 % W)).1) with
      fl := (aluF .add w (64 % W)).2, rip := s.rip + 4 } :=
    step_aluI_add (by rw [hrip]; exact oc.wd0) hr7
  obtain ⟨a1, ha1⟩ : ∃ u : X86Int.St, u = { s.set 7 (some (aluF .add w (64 % W)).1)
    with fl := (aluF .add w (64 % W)).2, rip := s.rip + 4 } := ⟨_, rfl⟩
  have ha1rip : a1.rip = aWordDone + 4 := by
    rw [ha1]; show s.rip + 4 = aWordDone + 4; rw [hrip]
  have hadd : (aluF .add w (64 % W)).1 = w + 64 := aluI_add_val h64W hw64
  have ha1r7 : a1.reg 7 = some (w + 64) := by
    rw [ha1]
    show (s.set 7 (some (aluF .add w (64 % W)).1)).reg 7 = some (w + 64)
    rw [reg_set_self (reg_some_len hr7), hadd]
  have ha1r5 : a1.reg 5 = some n := by
    rw [ha1]
    show (s.set 7 (some (aluF .add w (64 % W)).1)).reg 5 = some n
    rw [reg_set_other (by decide)]; exact hr5
  have ha1mem : a1.mem = s.mem := by
    rw [ha1]
    show (s.set 7 (some (aluF .add w (64 % W)).1)).mem = s.mem
    rw [mem_set]
  have ga1 : ∀ q, q ≠ 7 → a1.reg q = s.reg q := by
    intro q hq
    rw [ha1]
    show (s.set 7 (some (aluF .add w (64 % W)).1)).reg q = s.reg q
    rw [reg_set_other hq]
  have e2 : X86Int.step im a1 = .ok { a1.set 5 (some (aluF .sub n (1 % W)).1) with
      fl := (aluF .sub n (1 % W)).2, rip := a1.rip + 4 } :=
    step_aluI_sub (by rw [ha1rip]; exact oc.wd1) ha1r5
  obtain ⟨a2, ha2⟩ : ∃ u : X86Int.St, u = { a1.set 5 (some (aluF .sub n (1 % W)).1)
    with fl := (aluF .sub n (1 % W)).2, rip := a1.rip + 4 } := ⟨_, rfl⟩
  have ha2rip : a2.rip = aWordDone + 8 := by
    rw [ha2]; show a1.rip + 4 = aWordDone + 8; rw [ha1rip]
  have hsub : (aluF .sub n (1 % W)).1 = n - 1 := aluI_sub_val h1W hn1 hnW
  have ha2r5 : a2.reg 5 = some (n - 1) := by
    rw [ha2]
    show (a1.set 5 (some (aluF .sub n (1 % W)).1)).reg 5 = some (n - 1)
    rw [reg_set_self (by
      have h5len : 5 < s.regs.length := reg_some_len hr5
      have e : a1.regs.length = s.regs.length := by
        rw [ha1]; exact List.length_set
      omega), hsub]
  have ha2r7 : a2.reg 7 = some (w + 64) := by
    rw [ha2]
    show (a1.set 5 (some (aluF .sub n (1 % W)).1)).reg 7 = some (w + 64)
    rw [reg_set_other (by decide)]; exact ha1r7
  have ha2mem : a2.mem = s.mem := by
    rw [ha2]
    show (a1.set 5 (some (aluF .sub n (1 % W)).1)).mem = s.mem
    rw [mem_set]; exact ha1mem
  have ga2 : ∀ q, q ≠ 5 → q ≠ 7 → a2.reg q = s.reg q := by
    intro q h5 h7
    rw [ha2]
    show (a1.set 5 (some (aluF .sub n (1 % W)).1)).reg q = s.reg q
    rw [reg_set_other h5]; exact ga1 q h7
  have e3 : X86Int.step im a2 = .ok { a2 with rip := aCheckWord } :=
    step_jmp (by rw [ha2rip]; exact oc.wd2)
  obtain ⟨a3, ha3⟩ : ∃ u : X86Int.St, u = { a2 with rip := aCheckWord } := ⟨_, rfl⟩
  have r1 : run im 1 (Out.ok s) = Out.ok a1 := by
    rw [run_succ_ok im 0 s, e1, ← ha1, run_zero]
  have r2 : run im 1 (Out.ok a1) = Out.ok a2 := by
    rw [run_succ_ok im 0 a1, e2, ← ha2, run_zero]
  have r3 : run im 1 (Out.ok a2) = Out.ok a3 := by
    rw [run_succ_ok im 0 a2, e3, ← ha3, run_zero]
  refine ⟨1 + 1 + 1, a3, by
      rw [run_add im 1 2, r1, run_add im 1 1, r2, r3], by rw [ha3], ?_, ?_, ?_, ?_,
    ?_⟩
  · rw [ha3]; show ({ a2 with rip := aCheckWord }).reg 7 = some (w + 64)
    rw [reg_with_rip]; exact ha2r7
  · rw [ha3]; show ({ a2 with rip := aCheckWord }).reg 5 = some (n - 1)
    rw [reg_with_rip]; exact ha2r5
  · intro q h5 h7
    rw [ha3]; show ({ a2 with rip := aCheckWord }).reg q = s.reg q
    rw [reg_with_rip]; exact ga2 q h5 h7
  · rw [ha3]; show ({ a2 with rip := aCheckWord }).mem = s.mem
    rw [mem_with_rip]; exact ha2mem
  · have e1 : a3.regs.length = a2.regs.length := by rw [ha3]
    have e2 : a2.regs.length = a1.regs.length := by
      rw [ha2]; exact List.length_set
    have e3 : a1.regs.length = s.regs.length := by
      rw [ha1]; exact List.length_set
    rw [e1, e2, e3]

/-- `accept`: exit with status 0. -/
theorem Accept_exit {im : Nat → Option (Ins × Nat)} (oc : OuterCode im) :
    ∀ (sp a2 a3 a4 a5 a6 : Nat) (m1 m2 : Mem), 32 ≤ sp → sp < W →
    ∀ s, s.rip = aAccept → s.reg 4 = some sp →
      s.reg 6 = some a2 → s.reg 2 = some a3 → s.reg 10 = some a4 →
      s.reg 8 = some a5 → s.reg 9 = some a6 →
      wr64 s.mem ((sp + W - 8) % W) 60 = some m1 →
      wr64 m1 ((sp + W - 8) % W) 0 = some m2 →
      ∃ k, run im k (Out.ok s) = Out.exit 0 := by
  intro sp a2 a3 a4 a5 a6 m1 m2 hsp32 hspW s hrip hr4 hr6 hr2 hr10 hr8 hr9 hm1 hm2
  have hslot : (sp + W - 8) % W = sp - 8 := slot_sub (k := 1) (by omega) hspW
  have hm1' : wr64 s.mem ((sp + W - 8) % W) (60 % W) = some m1 := hm1
  have hr60 : rd64 m1 ((sp + W - 8) % W) = some 60 := rd64_wr64 hm1'
  have hr0 : rd64 m2 ((sp + W - 8) % W) = some 0 := rd64_wr64 hm2
  have e1 : X86Int.step im s =
      .ok { s.set 4 (some ((sp + W - 8) % W)) with mem := m1, rip := s.rip + 2 } :=
    step_pushI (by rw [hrip]; exact oc.ac0) hr4 hm1'
  obtain ⟨a1, ha1⟩ : ∃ u : X86Int.St, u =
    { s.set 4 (some ((sp + W - 8) % W)) with mem := m1, rip := s.rip + 2 } := ⟨_, rfl⟩
  have ha1rip : a1.rip = aAccept + 2 := by
    rw [ha1]; show s.rip + 2 = aAccept + 2; rw [hrip]
  have ha1r4 : a1.reg 4 = some ((sp + W - 8) % W) := by
    rw [ha1]
    show (s.set 4 (some ((sp + W - 8) % W))).reg 4 = some ((sp + W - 8) % W)
    rw [reg_set_self (reg_some_len hr4)]
  have ha1mem : a1.mem = m1 := by rw [ha1]
  have e2 : X86Int.step im a1 =
      .ok { (a1.set 4 (some (((sp + W - 8) % W + 8) % W))).set 0 (some 60) with
        rip := a1.rip + 1 } :=
    step_pop (by rw [ha1rip]; exact oc.ac1) ha1r4 (by rw [ha1mem]; exact hr60)
  obtain ⟨a2s, ha2⟩ : ∃ u : X86Int.St, u =
    { (a1.set 4 (some (((sp + W - 8) % W + 8) % W))).set 0 (some 60) with
      rip := a1.rip + 1 } := ⟨_, rfl⟩
  have hsp8 : ((sp + W - 8) % W + 8) % W = sp := by
    rw [hslot]
    have h1 : sp - 8 + 8 = sp := by omega
    rw [h1]; exact Nat.mod_eq_of_lt hspW
  have ha2rip : a2s.rip = aAccept + 3 := by
    rw [ha2]; show a1.rip + 1 = aAccept + 3; rw [ha1rip]
  have ha2r4 : a2s.reg 4 = some sp := by
    rw [ha2]
    show ((a1.set 4 (some (((sp + W - 8) % W + 8) % W))).set 0 (some 60)).reg 4 =
      some sp
    rw [reg_set_other (by decide), reg_set_self (by
      have e : a1.regs.length = s.regs.length := by
        rw [ha1]; exact List.length_set
      have h4len : 4 < s.regs.length := reg_some_len hr4
      omega), hsp8]
  have ha2r0 : a2s.reg 0 = some 60 := by
    rw [ha2]
    show ((a1.set 4 (some (((sp + W - 8) % W + 8) % W))).set 0 (some 60)).reg 0 =
      some 60
    rw [reg_set_self (by
      have e1 : (a1.set 4 (some (((sp + W - 8) % W + 8) % W))).regs.length =
          a1.regs.length := List.length_set
      have e2 : a1.regs.length = s.regs.length := by
        rw [ha1]; exact List.length_set
      have h9len : 9 < s.regs.length := reg_some_len hr9
      omega)]
  have ha2mem : a2s.mem = m1 := by
    rw [ha2]
    show ((a1.set 4 (some (((sp + W - 8) % W + 8) % W))).set 0 (some 60)).mem = m1
    rw [mem_set, mem_set]; exact ha1mem
  have ga2 : ∀ q, q ≠ 0 → q ≠ 4 → a2s.reg q = s.reg q := by
    intro q h0 h4
    rw [ha2]
    show ((a1.set 4 (some (((sp + W - 8) % W + 8) % W))).set 0 (some 60)).reg q =
      s.reg q
    rw [reg_set_other h0, reg_set_other h4]
    rw [ha1]
    show (s.set 4 (some ((sp + W - 8) % W))).reg q = s.reg q
    rw [reg_set_other h4]
  have hm2s : wr64 a2s.mem ((sp + W - 8) % W) (0 % W) = some m2 := by
    rw [ha2mem]; exact hm2
  have e3 : X86Int.step im a2s =
      .ok { a2s.set 4 (some ((sp + W - 8) % W)) with mem := m2, rip := a2s.rip + 2 } :=
    step_pushI (by rw [ha2rip]; exact oc.ac2) ha2r4 hm2s
  obtain ⟨a3, ha3⟩ : ∃ u : X86Int.St, u =
    { a2s.set 4 (some ((sp + W - 8) % W)) with mem := m2, rip := a2s.rip + 2 } :=
    ⟨_, rfl⟩
  have ha3rip : a3.rip = aAccept + 5 := by
    rw [ha3]; show a2s.rip + 2 = aAccept + 5; rw [ha2rip]
  have ha3r4 : a3.reg 4 = some ((sp + W - 8) % W) := by
    rw [ha3]
    show (a2s.set 4 (some ((sp + W - 8) % W))).reg 4 = some ((sp + W - 8) % W)
    rw [reg_set_self (by
      have e : a2s.regs.length = s.regs.length := by
        have e1 : a2s.regs.length = a1.regs.length := by
          rw [ha2]
          exact (List.length_set).trans (List.length_set)
        have e2 : a1.regs.length = s.regs.length := by
          rw [ha1]; exact List.length_set
        rw [e1, e2]
      have h4len : 4 < s.regs.length := reg_some_len hr4
      omega)]
  have ha3mem : a3.mem = m2 := by rw [ha3]
  have ga3 : ∀ q, q ≠ 4 → a3.reg q = a2s.reg q := by
    intro q h4
    rw [ha3]
    show (a2s.set 4 (some ((sp + W - 8) % W))).reg q = a2s.reg q
    rw [reg_set_other h4]
  have e4 : X86Int.step im a3 =
      .ok { (a3.set 4 (some (((sp + W - 8) % W + 8) % W))).set 7 (some 0) with
        rip := a3.rip + 1 } :=
    step_pop (by rw [ha3rip]; exact oc.ac3) ha3r4 (by rw [ha3mem]; exact hr0)
  obtain ⟨a4, ha4⟩ : ∃ u : X86Int.St, u =
    { (a3.set 4 (some (((sp + W - 8) % W + 8) % W))).set 7 (some 0) with
      rip := a3.rip + 1 } := ⟨_, rfl⟩
  have ha4rip : a4.rip = aAccept + 6 := by
    rw [ha4]; show a3.rip + 1 = aAccept + 6; rw [ha3rip]
  have ha4len : (a3.set 4 (some (((sp + W - 8) % W + 8) % W))).regs.length =
      s.regs.length := by
    have e0 : (a3.set 4 (some (((sp + W - 8) % W + 8) % W))).regs.length =
        a3.regs.length := List.length_set
    have e1 : a3.regs.length = a2s.regs.length := by
      rw [ha3]; exact List.length_set
    have e2 : a2s.regs.length = a1.regs.length := by
      rw [ha2]; exact (List.length_set).trans (List.length_set)
    have e3 : a1.regs.length = s.regs.length := by
      rw [ha1]; exact List.length_set
    rw [e0, e1, e2, e3]
  have ha4r0 : a4.reg 0 = some 60 := by
    rw [ha4]
    show ((a3.set 4 (some (((sp + W - 8) % W + 8) % W))).set 7 (some 0)).reg 0 =
      some 60
    rw [reg_set_other (by decide), reg_set_other (by decide), ga3 0 (by decide)]
    exact ha2r0
  have ha4r7 : a4.reg 7 = some 0 := by
    rw [ha4]
    show ((a3.set 4 (some (((sp + W - 8) % W + 8) % W))).set 7 (some 0)).reg 7 =
      some 0
    rw [reg_set_self (by
      have h9len : 9 < s.regs.length := reg_some_len hr9
      omega)]
  have ha4r : ∀ q, q ≠ 0 → q ≠ 4 → q ≠ 7 → a4.reg q = s.reg q := by
    intro q h0 h4 h7
    rw [ha4]
    show ((a3.set 4 (some (((sp + W - 8) % W + 8) % W))).set 7 (some 0)).reg q =
      s.reg q
    rw [reg_set_other h7, reg_set_other h4, ga3 q h4]
    exact ga2 q h0 h4
  have e5 : X86Int.step im a4 = .exit (0 % 256) :=
    step_syscall_exit (by rw [ha4rip]; exact oc.ac4) ha4r0 ha4r7
      (by rw [ha4r 6 (by decide) (by decide) (by decide)]; exact hr6)
      (by rw [ha4r 2 (by decide) (by decide) (by decide)]; exact hr2)
      (by rw [ha4r 10 (by decide) (by decide) (by decide)]; exact hr10)
      (by rw [ha4r 8 (by decide) (by decide) (by decide)]; exact hr8)
      (by rw [ha4r 9 (by decide) (by decide) (by decide)]; exact hr9)
  have r1 : run im 1 (Out.ok s) = Out.ok a1 := by
    rw [run_succ_ok im 0 s, e1, ← ha1, run_zero]
  have r2 : run im 1 (Out.ok a1) = Out.ok a2s := by
    rw [run_succ_ok im 0 a1, e2, ← ha2, run_zero]
  have r3 : run im 1 (Out.ok a2s) = Out.ok a3 := by
    rw [run_succ_ok im 0 a2s, e3, ← ha3, run_zero]
  have r4 : run im 1 (Out.ok a3) = Out.ok a4 := by
    rw [run_succ_ok im 0 a3, e4, ← ha4, run_zero]
  have r5 : run im 1 (Out.ok a4) = Out.exit 0 := by
    rw [run_succ_ok im 0 a4, e5, run_zero]
  exact ⟨1 + 1 + 1 + 1 + 1, by
    rw [run_add im 1 4, r1, run_add im 1 3, r2, run_add im 1 2, r3, run_add im 1 1,
      r4, r5]⟩

/-- `check_word` with no words left reaches `accept`. -/
theorem CheckWord_done {im : Nat → Option (Ins × Nat)} (oc : OuterCode im) :
    ∀ s, s.rip = aCheckWord → s.reg 5 = some 0 →
      ∃ k s', run im k (Out.ok s) = Out.ok s' ∧ s'.rip = aAccept ∧
        (∀ q, s'.reg q = s.reg q) ∧ s'.mem = s.mem ∧
        s'.regs.length = s.regs.length := by
  intro s hrip hr5
  have e1 : X86Int.step im s =
      .ok { s with fl := (aluF .and 0 0).2, rip := s.rip + 3 } :=
    step_test (by rw [hrip]; exact oc.cw0) hr5 hr5
  obtain ⟨a1, ha1⟩ : ∃ u : X86Int.St, u =
    { s with fl := (aluF .and 0 0).2, rip := s.rip + 3 } := ⟨_, rfl⟩
  have ha1rip : a1.rip = aCheckWord + 3 := by
    rw [ha1]; show s.rip + 3 = aCheckWord + 3; rw [hrip]
  have ha1fl : X86Int.cond a1.fl 5 = some false := by
    have hz : a1.fl.zf = some true := by
      rw [ha1]
      show (aluF .and 0 0).2.zf = some true
      rw [aluF_zf, aluF_and_fst_self]
      exact rfl
    have := cond_cc5 hz
    simpa using this
  have e2 : X86Int.step im a1 = .ok { a1 with rip := aCheckWord + 5 } := by
    have hh := step_jcc (im := im) (s := a1) (by rw [ha1rip]; exact oc.cw1) ha1fl
    simpa [ha1rip] using hh
  obtain ⟨a2, ha2⟩ : ∃ u : X86Int.St, u = { a1 with rip := aCheckWord + 5 } :=
    ⟨_, rfl⟩
  have ha2rip : a2.rip = aCheckWord + 5 := by rw [ha2]
  have e3 : X86Int.step im a2 = .ok { a2 with rip := aAccept } :=
    step_jmp (by rw [ha2rip]; exact oc.cw2)
  obtain ⟨a3, ha3⟩ : ∃ u : X86Int.St, u = { a2 with rip := aAccept } := ⟨_, rfl⟩
  have r1 : run im 1 (Out.ok s) = Out.ok a1 := by
    rw [run_succ_ok im 0 s, e1, ← ha1, run_zero]
  have r2 : run im 1 (Out.ok a1) = Out.ok a2 := by
    rw [run_succ_ok im 0 a1, e2, ← ha2, run_zero]
  have r3 : run im 1 (Out.ok a2) = Out.ok a3 := by
    rw [run_succ_ok im 0 a2, e3, ← ha3, run_zero]
  refine ⟨1 + 1 + 1, a3, by rw [run_add im 1 2, r1, run_add im 1 1, r2, r3],
    by rw [ha3], ?_, ?_, ?_⟩
  · intro q
    rw [ha3]; show ({ a2 with rip := aAccept }).reg q = s.reg q
    rw [reg_with_rip]
    rw [ha2]; show ({ a1 with rip := aCheckWord + 5 }).reg q = s.reg q
    rw [reg_with_rip]
    rw [ha1]; exact rfl
  · rw [ha3]; show ({ a2 with rip := aAccept }).mem = s.mem
    rw [mem_with_rip]
    rw [ha2]; show ({ a1 with rip := aCheckWord + 5 }).mem = s.mem
    rw [mem_with_rip]
    rw [ha1]
  · rw [ha3, ha2, ha1]

/-- `check_word` with a word left and nonempty code reaches `annval`,
with the ann scan set up. -/
theorem CheckWord_head {im : Nat → Option (Ins × Nat)} (oc : OuterCode im) :
    ∀ (w n nc nann annp : Nat), w + 56 ≤ W → n ≠ 0 → nc ≠ 0 →
    ∀ s, s.rip = aCheckWord → s.reg 5 = some n → s.reg 7 = some w →
      rd64 s.mem (w + 24) = some nc → rd64 s.mem (w + 48) = some annp →
      rd64 s.mem (w + 32) = some nann →
      ∃ k s', run im k (Out.ok s) = Out.ok s' ∧ s'.rip = aAnnval ∧
        s'.reg 6 = some annp ∧ s'.reg 1 = some nann ∧ s'.reg 0 = some nc ∧
        s'.reg 5 = some n ∧ s'.reg 7 = some w ∧
        (∀ q, q ≠ 0 → q ≠ 1 → q ≠ 6 → s'.reg q = s.reg q) ∧ s'.mem = s.mem ∧
        s'.regs.length = s.regs.length := by
  intro w n nc nann annp hw56 hn hnc s hrip hr5 hr7 hfnc hfannp hfnann
  have e1 : X86Int.step im s =
      .ok { s with fl := (aluF .and n n).2, rip := s.rip + 3 } :=
    step_test (by rw [hrip]; exact oc.cw0) hr5 hr5
  obtain ⟨a1, ha1⟩ : ∃ u : X86Int.St, u =
    { s with fl := (aluF .and n n).2, rip := s.rip + 3 } := ⟨_, rfl⟩
  have ha1rip : a1.rip = aCheckWord + 3 := by
    rw [ha1]; show s.rip + 3 = aCheckWord + 3; rw [hrip]
  have ha1fl : X86Int.cond a1.fl 5 = some true := by
    have hz : a1.fl.zf = some false := by
      rw [ha1]
      show (aluF .and n n).2.zf = some false
      rw [aluF_zf, aluF_and_fst_self]
      simp [hn]
    have := cond_cc5 hz
    simpa using this
  have e2 : X86Int.step im a1 = .ok { a1 with rip := aCheckWord + 10 } := by
    have hh := step_jcc (im := im) (s := a1) (by rw [ha1rip]; exact oc.cw1) ha1fl
    simpa [ha1rip] using hh
  obtain ⟨a2, ha2⟩ : ∃ u : X86Int.St, u = { a1 with rip := aCheckWord + 10 } :=
    ⟨_, rfl⟩
  have ha2rip : a2.rip = aCheckWord + 10 := by rw [ha2]
  have ha2r7 : a2.reg 7 = some w := by
    rw [ha2]; show ({ a1 with rip := aCheckWord + 10 }).reg 7 = some w
    rw [reg_with_rip]
    rw [ha1]; show s.reg 7 = some w
    exact hr7
  have ha2r5 : a2.reg 5 = some n := by
    rw [ha2]; show ({ a1 with rip := aCheckWord + 10 }).reg 5 = some n
    rw [reg_with_rip]
    rw [ha1]; show s.reg 5 = some n
    exact hr5
  have ha2mem : a2.mem = s.mem := by
    rw [ha2]; show ({ a1 with rip := aCheckWord + 10 }).mem = s.mem
    rw [mem_with_rip]
    rw [ha1]
  have ga2 : ∀ q, a2.reg q = s.reg q := by
    intro q
    rw [ha2]; show ({ a1 with rip := aCheckWord + 10 }).reg q = s.reg q
    rw [reg_with_rip]
    rw [ha1]; exact rfl
  have hea24 : ea w 24 = w + 24 := Nat.mod_eq_of_lt (by omega)
  have e3 : X86Int.step im a2 = .ok { a2.set 0 (some nc) with rip := a2.rip + 4 } :=
    step_ld (by rw [ha2rip]; exact oc.cw3) ha2r7 (by rw [hea24, ha2mem]; exact hfnc)
  obtain ⟨a3, ha3⟩ : ∃ u : X86Int.St, u =
    { a2.set 0 (some nc) with rip := a2.rip + 4 } := ⟨_, rfl⟩
  have ha3rip : a3.rip = aCheckWord + 14 := by
    rw [ha3]; show a2.rip + 4 = aCheckWord + 14; rw [ha2rip]
  have ha3r0 : a3.reg 0 = some nc := by
    rw [ha3]
    show (a2.set 0 (some nc)).reg 0 = some nc
    rw [reg_set_self (by
      have e : a2.regs.length = s.regs.length := by rw [ha2, ha1]
      have h5len : 5 < s.regs.length := reg_some_len hr5
      omega)]
  have ha3r7 : a3.reg 7 = some w := by
    rw [ha3]
    show (a2.set 0 (some nc)).reg 7 = some w
    rw [reg_set_other (by decide)]; exact ha2r7
  have ha3r5 : a3.reg 5 = some n := by
    rw [ha3]
    show (a2.set 0 (some nc)).reg 5 = some n
    rw [reg_set_other (by decide)]; exact ha2r5
  have ha3mem : a3.mem = s.mem := by
    rw [ha3]
    show (a2.set 0 (some nc)).mem = s.mem
    rw [mem_set]; exact ha2mem
  have ga3 : ∀ q, q ≠ 0 → a3.reg q = s.reg q := by
    intro q hq
    rw [ha3]
    show (a2.set 0 (some nc)).reg q = s.reg q
    rw [reg_set_other hq]; exact ga2 q
  have e4 : X86Int.step im a3 =
      .ok { a3 with fl := (aluF .and nc nc).2, rip := a3.rip + 3 } :=
    step_test (by rw [ha3rip]; exact oc.cw4) ha3r0 ha3r0
  obtain ⟨a4, ha4⟩ : ∃ u : X86Int.St, u =
    { a3 with fl := (aluF .and nc nc).2, rip := a3.rip + 3 } := ⟨_, rfl⟩
  have ha4rip : a4.rip = aCheckWord + 17 := by
    rw [ha4]; show a3.rip + 3 = aCheckWord + 17; rw [ha3rip]
  have ha4fl : X86Int.cond a4.fl 5 = some true := by
    have hz : a4.fl.zf = some false := by
      rw [ha4]
      show (aluF .and nc nc).2.zf = some false
      rw [aluF_zf, aluF_and_fst_self]
      simp [hnc]
    have := cond_cc5 hz
    simpa using this
  have ha4r0 : a4.reg 0 = some nc := by
    rw [ha4]; show a3.reg 0 = some nc; exact ha3r0
  have ha4r7 : a4.reg 7 = some w := by
    rw [ha4]; show a3.reg 7 = some w; exact ha3r7
  have ha4r5 : a4.reg 5 = some n := by
    rw [ha4]; show a3.reg 5 = some n; exact ha3r5
  have ha4mem : a4.mem = s.mem := by
    rw [ha4]; show a3.mem = s.mem; exact ha3mem
  have ga4 : ∀ q, q ≠ 0 → a4.reg q = s.reg q := by
    intro q hq
    rw [ha4]; show a3.reg q = s.reg q; exact ga3 q hq
  have e5 : X86Int.step im a4 = .ok { a4 with rip := aCheckWord + 24 } := by
    have hh := step_jcc (im := im) (s := a4) (by rw [ha4rip]; exact oc.cw5) ha4fl
    simpa [ha4rip] using hh
  obtain ⟨a5, ha5⟩ : ∃ u : X86Int.St, u = { a4 with rip := aCheckWord + 24 } :=
    ⟨_, rfl⟩
  have ha5rip : a5.rip = aCheckWord + 24 := by rw [ha5]
  have ha5r0 : a5.reg 0 = some nc := by
    rw [ha5]; show ({ a4 with rip := aCheckWord + 24 }).reg 0 = some nc
    rw [reg_with_rip]; exact ha4r0
  have ha5r7 : a5.reg 7 = some w := by
    rw [ha5]; show ({ a4 with rip := aCheckWord + 24 }).reg 7 = some w
    rw [reg_with_rip]; exact ha4r7
  have ha5r5 : a5.reg 5 = some n := by
    rw [ha5]; show ({ a4 with rip := aCheckWord + 24 }).reg 5 = some n
    rw [reg_with_rip]; exact ha4r5
  have ha5mem : a5.mem = s.mem := by
    rw [ha5]; show ({ a4 with rip := aCheckWord + 24 }).mem = s.mem
    rw [mem_with_rip]; exact ha4mem
  have ga5 : ∀ q, q ≠ 0 → a5.reg q = s.reg q := by
    intro q hq
    rw [ha5]; show ({ a4 with rip := aCheckWord + 24 }).reg q = s.reg q
    rw [reg_with_rip]; exact ga4 q hq
  have hea48 : ea w 48 = w + 48 := Nat.mod_eq_of_lt (by omega)
  have e6 : X86Int.step im a5 =
      .ok { a5.set 6 (some annp) with rip := a5.rip + 4 } :=
    step_ld (by rw [ha5rip]; exact oc.cw6) ha5r7
      (by rw [hea48, ha5mem]; exact hfannp)
  obtain ⟨a6, ha6⟩ : ∃ u : X86Int.St, u =
    { a5.set 6 (some annp) with rip := a5.rip + 4 } := ⟨_, rfl⟩
  have ha6rip : a6.rip = aCheckWord + 28 := by
    rw [ha6]; show a5.rip + 4 = aCheckWord + 28; rw [ha5rip]
  have ha6r6 : a6.reg 6 = some annp := by
    rw [ha6]
    show (a5.set 6 (some annp)).reg 6 = some annp
    rw [reg_set_self (by
      have e : a5.regs.length = s.regs.length := by
        rw [ha5, ha4, ha3, ha2, ha1]
        exact List.length_set
      have h7len : 7 < s.regs.length := reg_some_len hr7
      omega)]
  have ha6r0 : a6.reg 0 = some nc := by
    rw [ha6]
    show (a5.set 6 (some annp)).reg 0 = some nc
    rw [reg_set_other (by decide)]; exact ha5r0
  have ha6r7 : a6.reg 7 = some w := by
    rw [ha6]
    show (a5.set 6 (some annp)).reg 7 = some w
    rw [reg_set_other (by decide)]; exact ha5r7
  have ha6r5 : a6.reg 5 = some n := by
    rw [ha6]
    show (a5.set 6 (some annp)).reg 5 = some n
    rw [reg_set_other (by decide)]; exact ha5r5
  have ha6mem : a6.mem = s.mem := by
    rw [ha6]
    show (a5.set 6 (some annp)).mem = s.mem
    rw [mem_set]; exact ha5mem
  have ga6 : ∀ q, q ≠ 0 → q ≠ 6 → a6.reg q = s.reg q := by
    intro q h0 h6
    rw [ha6]
    show (a5.set 6 (some annp)).reg q = s.reg q
    rw [reg_set_other h6]; exact ga5 q h0
  have hea32 : ea w 32 = w + 32 := Nat.mod_eq_of_lt (by omega)
  have e7 : X86Int.step im a6 =
      .ok { a6.set 1 (some nann) with rip := a6.rip + 4 } :=
    step_ld (by rw [ha6rip]; exact oc.cw7) ha6r7
      (by rw [hea32, ha6mem]; exact hfnann)
  obtain ⟨a7, ha7⟩ : ∃ u : X86Int.St, u =
    { a6.set 1 (some nann) with rip := a6.rip + 4 } := ⟨_, rfl⟩
  have r1 : run im 1 (Out.ok s) = Out.ok a1 := by
    rw [run_succ_ok im 0 s, e1, ← ha1, run_zero]
  have r2 : run im 1 (Out.ok a1) = Out.ok a2 := by
    rw [run_succ_ok im 0 a1, e2, ← ha2, run_zero]
  have r3 : run im 1 (Out.ok a2) = Out.ok a3 := by
    rw [run_succ_ok im 0 a2, e3, ← ha3, run_zero]
  have r4 : run im 1 (Out.ok a3) = Out.ok a4 := by
    rw [run_succ_ok im 0 a3, e4, ← ha4, run_zero]
  have r5 : run im 1 (Out.ok a4) = Out.ok a5 := by
    rw [run_succ_ok im 0 a4, e5, ← ha5, run_zero]
  have r6 : run im 1 (Out.ok a5) = Out.ok a6 := by
    rw [run_succ_ok im 0 a5, e6, ← ha6, run_zero]
  have r7 : run im 1 (Out.ok a6) = Out.ok a7 := by
    rw [run_succ_ok im 0 a6, e7, ← ha7, run_zero]
  refine ⟨1 + 1 + 1 + 1 + 1 + 1 + 1, a7, by
      rw [run_add im 1 6, r1, run_add im 1 5, r2, run_add im 1 4, r3,
        run_add im 1 3, r4, run_add im 1 2, r5, run_add im 1 1, r6, r7],
    by rw [ha7]; show a6.rip + 4 = aAnnval; rw [ha6rip]; rfl, ?_, ?_, ?_, ?_, ?_,
    ?_, ?_, ?_⟩
  · rw [ha7]; show (a6.set 1 (some nann)).reg 6 = some annp
    rw [reg_set_other (by decide)]; exact ha6r6
  · rw [ha7]; show (a6.set 1 (some nann)).reg 1 = some nann
    rw [reg_set_self (by
      have e : a6.regs.length = s.regs.length := by
        have e1 : a6.regs.length = a5.regs.length := by
          rw [ha6]; exact List.length_set
        have e2 : a5.regs.length = s.regs.length := by
          rw [ha5, ha4, ha3, ha2, ha1]
          exact List.length_set
        rw [e1, e2]
      have h7len : 7 < s.regs.length := reg_some_len hr7
      omega)]
  · rw [ha7]; show (a6.set 1 (some nann)).reg 0 = some nc
    rw [reg_set_other (by decide)]; exact ha6r0
  · rw [ha7]; show (a6.set 1 (some nann)).reg 5 = some n
    rw [reg_set_other (by decide)]; exact ha6r5
  · rw [ha7]; show (a6.set 1 (some nann)).reg 7 = some w
    rw [reg_set_other (by decide)]; exact ha6r7
  · intro q h0 h1 h6
    rw [ha7]; show (a6.set 1 (some nann)).reg q = s.reg q
    rw [reg_set_other h1]; exact ga6 q h0 h6
  · rw [ha7]; show (a6.set 1 (some nann)).mem = s.mem
    rw [mem_set]; exact ha6mem
  · have e1 : a7.regs.length = a6.regs.length := by
      rw [ha7]; exact List.length_set
    have e2 : a6.regs.length = a5.regs.length := by
      rw [ha6]; exact List.length_set
    have e3 : a5.regs.length = s.regs.length := by
      rw [ha5, ha4, ha3, ha2, ha1]; exact List.length_set
    rw [e1, e2, e3]


theorem StackRoom_weaken {s : X86Int.St} {k : Nat} (h : StackRoom s (k + 1)) :
    StackRoom s k := fun sp hsp i hi v => h sp hsp i (by omega) v

/-- A `call L32` site, packaged: returns the `u32` at `rdx` in `rax`,
restores `rsp`, writes only the return slot. -/
theorem CallL32 {im : Nat → Option (Ins × Nat)} {l32 : Nat}
    (hl : L32Code im l32) (hret : im (l32 + 2) = some (.ret, 1))
    {site a v sp : Nat} (hsite : im site = some (.call l32, 5))
    (haW : a < W) (hsp8 : 8 ≤ sp) (hspW : sp < W) (hsite5 : site + 5 < W)
    (hdisj : a + 4 ≤ sp - 8 ∨ sp ≤ a) :
    ∀ s, s.rip = site → s.reg 2 = some a → s.reg 4 = some sp →
      (s.mem.rdN a 4).map leVal = some v → StackRoom s 1 →
      ∃ k s', run im k (Out.ok s) = Out.ok s' ∧ s'.rip = site + 5 ∧
        s'.reg 0 = some v ∧ s'.reg 4 = some sp ∧
        (∀ q, q ≠ 0 → q ≠ 4 → s'.reg q = s.reg q) ∧
        s'.regs.length = s.regs.length ∧
        (∀ x n, x + n ≤ sp - 8 ∨ sp ≤ x → s'.mem.rdN x n = s.mem.rdN x n) ∧
        wshape s'.mem = wshape s.mem := by
  intro s hrip hr2 hr4 hrd hroom
  have hslot8 : (sp + W - 8) % W = sp - 8 := by
    rw [show sp + W - 8 = sp + W - 8 * 1 by omega, slot_sub (by omega) hspW]
  have hw : (wr64 s.mem ((sp + W - 8 * (0 + 1)) % W) (site + 5)).isSome :=
    hroom sp hr4 0 (by decide) (site + 5)
  obtain ⟨m1, hm1s⟩ := Option.isSome_iff_exists.mp hw
  have hm1 : wr64 s.mem ((sp + W - 8) % W) (s.rip + 5) = some m1 := by
    rw [hrip]
    have e : sp + W - 8 = sp + W - 8 * (0 + 1) := by omega
    rw [e]; exact hm1s
  have e1 : X86Int.step im s = .ok { s.set 4 (some ((sp + W - 8) % W)) with
      mem := m1, rip := l32 } :=
    step_call (by rw [hrip]; exact hsite) hr4 hm1
  obtain ⟨a1, ha1⟩ : ∃ u : X86Int.St, u = { s.set 4 (some ((sp + W - 8) % W)) with
    mem := m1, rip := l32 } := ⟨_, rfl⟩
  have ha1rip : a1.rip = l32 := by rw [ha1]
  have ha1r4 : a1.reg 4 = some (sp - 8) := by
    rw [ha1]
    show (s.set 4 (some ((sp + W - 8) % W))).reg 4 = some (sp - 8)
    rw [reg_set_self (by
      have h4len : 4 < s.regs.length := reg_some_len hr4
      omega), hslot8]
  have ha1r2 : a1.reg 2 = some a := by
    rw [ha1]
    show (s.set 4 (some ((sp + W - 8) % W))).reg 2 = some a
    rw [reg_set_other (by decide)]; exact hr2
  have ha1mem : a1.mem = m1 := by rw [ha1]
  have ga1 : ∀ q, q ≠ 4 → a1.reg q = s.reg q := by
    intro q hq
    rw [ha1]
    show (s.set 4 (some ((sp + W - 8) % W))).reg q = s.reg q
    rw [reg_set_other hq]
  have ha1len : a1.regs.length = s.regs.length := by
    rw [ha1]; exact List.length_set
  have hm1' : wr64 s.mem (sp - 8) (site + 5) = some m1 := by
    have e : sp - 8 = (sp + W - 8 * (0 + 1)) % W :=
      (slot_sub (k := 0 + 1) (by omega) hspW).symm
    rw [e]; exact hm1s
  have hrdm1 : (m1.rdN a 4).map leVal = some v := by
    rw [rdN_wr64_other hm1' (by omega)]
    exact hrd
  have hrdsome : (a1.mem.rdN a 4).isSome := by
    rw [ha1mem]
    cases h : m1.rdN a 4 with
    | none => simp [h] at hrdm1
    | some bs => rfl
  obtain ⟨n8, a2, hrun8, ha2rip, ha2r0, -, ha2mem, ha2other, ha2len⟩ :=
    L32_triple hl a haW a1 ha1rip ⟨ha1r2, hrdsome⟩
  have ha2mem' : a2.mem = m1 := by rw [ha2mem]; exact ha1mem
  have ha2r4 : a2.reg 4 = some (sp - 8) := by
    rw [ha2other 4 (by decide)]; exact ha1r4
  have hrd64 : rd64 a2.mem (sp - 8) = some (site + 5) := by
    have h1 := rd64_wr64 hm1'
    rw [Nat.mod_eq_of_lt hsite5] at h1
    rw [ha2mem']; exact h1
  have e2 : X86Int.step im a2 = .ok { a2.set 4 (some ((sp - 8 + 8) % W)) with
      rip := site + 5 } :=
    step_ret (by rw [ha2rip]; exact hret) ha2r4 hrd64
  obtain ⟨a3, ha3⟩ : ∃ u : X86Int.St, u = { a2.set 4 (some ((sp - 8 + 8) % W)) with
    rip := site + 5 } := ⟨_, rfl⟩
  have r1 : run im 1 (Out.ok s) = Out.ok a1 := by
    rw [run_succ_ok im 0 s, e1, ← ha1, run_zero]
  have r3 : run im 1 (Out.ok a2) = Out.ok a3 := by
    rw [run_succ_ok im 0 a2, e2, ← ha3, run_zero]
  have hm1' : wr64 s.mem (sp - 8) (site + 5) = some m1 := by
    have e : sp - 8 = (sp + W - 8 * (0 + 1)) % W :=
      (slot_sub (k := 0 + 1) (by omega) hspW).symm
    rw [e]; exact hm1s
  refine ⟨1 + (n8 + 1), a3, by rw [run_add im 1 (n8 + 1), r1, run_add im n8 1,
    hrun8, r3], by rw [ha3], ?_, ?_, ?_, ?_, ?_, ?_⟩
  · rw [ha3]
    show ({ a2.set 4 (some ((sp - 8 + 8) % W)) with rip := site + 5 }).reg 0 =
      some v
    rw [reg_with_rip, reg_set_other (by decide), ha2r0, ha1mem]
    exact hrdm1
  · rw [ha3]
    show ({ a2.set 4 (some ((sp - 8 + 8) % W)) with rip := site + 5 }).reg 4 =
      some sp
    rw [reg_with_rip, reg_set_self (by
      have h4len : 4 < a2.regs.length := by
        rw [ha2len, ha1len]; exact reg_some_len hr4
      omega), slot_add8 hsp8 hspW]
  · intro q hq0 hq4
    rw [ha3]
    show ({ a2.set 4 (some ((sp - 8 + 8) % W)) with rip := site + 5 }).reg q =
      s.reg q
    rw [reg_with_rip, reg_set_other hq4, ha2other q hq0]
    exact ga1 q hq4
  · rw [ha3]
    show (a2.set 4 (some ((sp - 8 + 8) % W))).regs.length = s.regs.length
    exact List.length_set.trans (ha2len.trans ha1len)
  · intro x n hx
    rw [ha3]
    show ({ a2.set 4 (some ((sp - 8 + 8) % W)) with rip := site + 5 }).mem.rdN x n =
      s.mem.rdN x n
    rw [mem_with_rip, mem_set, ha2mem']
    exact rdN_wr64_other hm1' (by omega)
  · rw [ha3]
    show wshape ({ a2.set 4 (some ((sp - 8 + 8) % W)) with
      rip := site + 5 }).mem = wshape s.mem
    rw [mem_with_rip, mem_set, ha2mem']
    exact wrN_wshape _ _ _ hm1'

/-- A `call ANNL` site, packaged from `ANNL_triple` plus its `ret`. -/
theorem CallANNL {im : Nat → Option (Ins × Nat)} {entry l32 : Nat}
    (hc : ANNLCode im entry l32) (hl : L32Code im l32)
    (hentry : entry + 50 < W) {site : Nat}
    (hsite : im site = some (.call entry, 5)) (hsite5 : site + 5 < W)
    {wd : Sporos.Word} {t w p sp cv : Nat}
    (htW : t < W) (hp0 : 0 < p) (hw56 : w + 56 ≤ W)
    (hbound : p + 12 * wd.ann.length < W) (hsp32 : 32 ≤ sp) (hspW : sp < W) :
    ∀ s, s.rip = site → s.reg 12 = some t → s.reg 7 = some w →
      s.reg 4 = some sp → s.reg 1 = some cv →
      rd64 s.mem (w + 48) = some p → rd64 s.mem (w + 32) = some wd.ann.length →
      AnnLayout s wd.ann p → StackRoom s 4 →
      (sp ≤ w ∨ w + 56 ≤ sp - 32) →
      (sp ≤ p ∨ p + 12 * wd.ann.length ≤ sp - 32) →
      ∃ k s', run im k (Out.ok s) = Out.ok s' ∧ s'.rip = site + 5 ∧
        AnnPost wd.ann t s' ∧ s'.reg 1 = some (cv % W) ∧ s'.reg 4 = some sp ∧
        s'.reg 12 = some t ∧ s'.reg 7 = some w ∧ AnnLayout s' wd.ann p ∧
        (∀ q, q ≠ 0 → q ≠ 1 → q ≠ 2 → q ≠ 4 → s'.reg q = s.reg q) ∧
        (∀ x n, x + n ≤ sp - 32 ∨ sp ≤ x → s'.mem.rdN x n = s.mem.rdN x n) ∧
        (∃ e, s'.reg 0 = some e) ∧
        (∀ e, s'.reg 0 = some e → e ≠ 0 →
          p ≤ e ∧ e + 12 ≤ p + 12 * wd.ann.length) ∧
        wshape s'.mem = wshape s.mem ∧ s'.regs.length = s.regs.length := by
  intro s hrip hr12 hr7 hr4 hr1 hfp hfn hlay hroom hdisjw hdisjp
  have hslot8 : (sp + W - 8) % W = sp - 8 := by
    rw [show sp + W - 8 = sp + W - 8 * 1 by omega, slot_sub (by omega) hspW]
  have hw : (wr64 s.mem ((sp + W - 8 * (0 + 1)) % W) (site + 5)).isSome :=
    hroom sp hr4 0 (by decide) (site + 5)
  obtain ⟨m1, hm1s⟩ := Option.isSome_iff_exists.mp hw
  have hm1 : wr64 s.mem ((sp + W - 8) % W) (s.rip + 5) = some m1 := by
    rw [hrip]
    have e : sp + W - 8 = sp + W - 8 * (0 + 1) := by omega
    rw [e]; exact hm1s
  have hm1' : wr64 s.mem (sp - 8) (site + 5) = some m1 := by
    have e : sp - 8 = (sp + W - 8 * (0 + 1)) % W :=
      (slot_sub (k := 0 + 1) (by omega) hspW).symm
    rw [e]; exact hm1s
  have e1 : X86Int.step im s = .ok { s.set 4 (some ((sp + W - 8) % W)) with
      mem := m1, rip := entry } :=
    step_call (by rw [hrip]; exact hsite) hr4 hm1
  obtain ⟨a1, ha1⟩ : ∃ u : X86Int.St, u = { s.set 4 (some ((sp + W - 8) % W)) with
    mem := m1, rip := entry } := ⟨_, rfl⟩
  have ha1rip : a1.rip = entry := by rw [ha1]
  have ha1r4 : a1.reg 4 = some (sp - 8) := by
    rw [ha1]
    show (s.set 4 (some ((sp + W - 8) % W))).reg 4 = some (sp - 8)
    rw [reg_set_self (reg_some_len hr4), hslot8]
  have ga1 : ∀ q, q ≠ 4 → a1.reg q = s.reg q := by
    intro q hq
    rw [ha1]
    show (s.set 4 (some ((sp + W - 8) % W))).reg q = s.reg q
    rw [reg_set_other hq]
  have ha1r12 : a1.reg 12 = some t := by rw [ga1 12 (by decide)]; exact hr12
  have ha1r7 : a1.reg 7 = some w := by rw [ga1 7 (by decide)]; exact hr7
  have ha1r1 : a1.reg 1 = some cv := by rw [ga1 1 (by decide)]; exact hr1
  have ha1mem : a1.mem = m1 := by rw [ha1]
  have ha1len : a1.regs.length = s.regs.length := by
    rw [ha1]; exact List.length_set
  have hfp1 : rd64 m1 (w + 48) = some p := by
    rw [rd64_wr64_other hm1' (by omega)]
    exact hfp
  have hfn1 : rd64 m1 (w + 32) = some wd.ann.length := by
    rw [rd64_wr64_other hm1' (by omega)]
    exact hfn
  have lay1 : AnnLayout a1 wd.ann p := by
    have h1 : AnnLayout { s with mem := m1 } wd.ann p :=
      AnnLayout_wr64 hlay hm1' (by omega)
    exact AnnLayout_of_mem (by rw [ha1]) h1
  have hroom1 : StackRoom a1 3 := by
    intro sp' hsp' i hi v
    have hsp'e : sp' = sp - 8 := by
      rw [ha1r4] at hsp'
      exact Option.some.inj hsp'.symm
    subst hsp'e
    have e : ((sp - 8) + W - 8 * (i + 1)) % W = sp - 8 * (i + 2) := by
      rw [slot_sub (by omega) (by omega)]
      omega
    rw [e, ha1mem]
    apply wr64_some_of_wr64 hm1'
    have hh := hroom sp hr4 (i + 1) (by omega) v
    have e2 : (sp + W - 8 * (i + 1 + 1)) % W = sp - 8 * (i + 2) :=
      slot_sub (by omega) hspW
    rwa [e2] at hh
  obtain ⟨nA, a2, hrunA, hripA, hpost, ha2r1, ha2r4, ha2r12, ha2r7, hlay2,
    hgenA, hframeA, hex0, hloc, hshA⟩ :=
    ANNL_triple hc hl hentry wd t w p (sp - 8) cv htW hp0 hw56 hbound
      (by omega) (by omega) a1 ha1rip ha1r12 ha1r7 ha1r4 ha1r1
      (by rw [ha1mem]; exact hfp1) (by rw [ha1mem]; exact hfn1)
      lay1 hroom1 (by omega) (by omega)
  have hslotv : rd64 a2.mem (sp - 8) = some (site + 5) := by
    have hm1v : rd64 m1 (sp - 8) = some ((site + 5) % W) := rd64_wr64 hm1'
    rw [Nat.mod_eq_of_lt hsite5] at hm1v
    unfold rd64
    rw [hframeA (sp - 8) 8 (Or.inr (by omega)), ha1mem]
    exact hm1v
  have e2 : X86Int.step im a2 = .ok { a2.set 4 (some ((sp - 8 + 8) % W)) with
      rip := site + 5 } := by
    cases hripA with
    | inl hh => exact step_ret (by rw [hh]; exact hc.dret_none) ha2r4 hslotv
    | inr hh => exact step_ret (by rw [hh]; exact hc.dret_found) ha2r4 hslotv
  obtain ⟨a3, ha3⟩ : ∃ u : X86Int.St, u = { a2.set 4 (some ((sp - 8 + 8) % W)) with
    rip := site + 5 } := ⟨_, rfl⟩
  have r1 : run im 1 (Out.ok s) = Out.ok a1 := by
    rw [run_succ_ok im 0 s, e1, ← ha1, run_zero]
  have r3 : run im 1 (Out.ok a2) = Out.ok a3 := by
    rw [run_succ_ok im 0 a2, e2, ← ha3, run_zero]
  have ha3mem : a3.mem = a2.mem := by rw [ha3]; exact rfl
  have ha3r0 : ∀ e, a3.reg 0 = some e ↔ a2.reg 0 = some e := by
    intro e
    rw [ha3]
    show ({ a2.set 4 (some ((sp - 8 + 8) % W)) with rip := site + 5 }).reg 0 =
        some e ↔ a2.reg 0 = some e
    rw [reg_with_rip, reg_set_other (by decide)]
  have ga3 : ∀ q, q ≠ 4 → a3.reg q = a2.reg q := by
    intro q hq
    rw [ha3]
    show ({ a2.set 4 (some ((sp - 8 + 8) % W)) with rip := site + 5 }).reg q =
      a2.reg q
    rw [reg_with_rip, reg_set_other hq]
  refine ⟨1 + (nA + 1), a3, by rw [run_add im 1 (nA + 1), r1, run_add im nA 1,
    hrunA, r3], by rw [ha3], ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · intro e he
    have he2 : a2.reg 0 = some e := (ha3r0 e).mp he
    cases hpost e he2 with
    | inl hb => exact Or.inl hb
    | inr hb =>
      obtain ⟨d, a, hne, hat, hlk⟩ := hb
      exact Or.inr ⟨d, a, hne, AnnAt_of_mem ha3mem hat, hlk⟩
  · rw [ga3 1 (by decide)]; exact ha2r1
  · rw [ha3]
    show ({ a2.set 4 (some ((sp - 8 + 8) % W)) with rip := site + 5 }).reg 4 =
      some sp
    rw [reg_with_rip, reg_set_self (by
      have h4len : 4 < a2.regs.length := reg_some_len ha2r4
      omega), slot_add8 (by omega) hspW]
  · rw [ga3 12 (by decide)]; exact ha2r12
  · rw [ga3 7 (by decide)]; exact ha2r7
  · exact AnnLayout_of_mem ha3mem hlay2
  · intro q h0 h1 h2 h4
    rw [ga3 q h4, hgenA q h0 h1 h2 h4]
    exact ga1 q h4
  · intro x n hx
    rw [ha3mem, hframeA x n (by omega), ha1mem]
    exact rdN_wr64_other hm1' (by omega)
  · obtain ⟨e, he⟩ := hex0
    exact ⟨e, (ha3r0 e).mpr he⟩
  · intro e he hne
    exact hloc e ((ha3r0 e).mp he) hne
  · rw [ha3mem, hshA, ha1mem]
    exact wrN_wshape _ _ _ hm1'
  · rw [ha3]
    show (a2.set 4 (some ((sp - 8 + 8) % W))).regs.length = s.regs.length
    exact List.length_set.trans ((run_regs_length nA hrunA).trans ha1len)


end SporosX1
