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
  wshape s2.mem = wshape s.mem ∧ s2.regs.length = s.regs.length ∧
  s2.reg 5 = s.reg 5

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
      wshape s3.mem = wshape s.mem ∧ s3.reg 5 = s.reg 5 := by
  obtain ⟨hs2r8, hs2r9, hs2r6, hs2r4, hs2r1, hs2r7, hs2r13, hs2r14, hfr2, hsh2,
    hlen2, hs2r5⟩ := hpost
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
    hsh3.trans hsh2, by rw [hgen3 5 (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)]; exact hs2r5⟩
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
      wshape s3.mem = wshape s.mem ∧ s3.reg 5 = s.reg 5 := by
  obtain ⟨hs2r8, hs2r9, hs2r6, hs2r4, hs2r1, hs2r7, hs2r13, hs2r14, hfr2, hsh2,
    hlen2, hs2r5⟩ := hpost
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
    hsh3.trans hsh2, by rw [hgen3 5 (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)]; exact hs2r5⟩
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
      wshape s4.mem = wshape s.mem ∧ s4.reg 5 = s.reg 5 ∧ s4.reg 13 = s.reg 13 ∧
      s4.reg 14 = s.reg 14 := by
  obtain ⟨hs2r8, hs2r9, hs2r6, hs2r4, hs2r1, hs2r7, hs2r13, hs2r14, hfr2, hsh2,
    hlen2, hs2r5⟩ := hpost
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
      run_add im kt k4, hrt, hr4], hrip4, hr14, hr64, hr74, hr44, ?_, ?_, ?_,
    ?_, ?_⟩
  · intro x m hx
    rw [hm4]; exact frS x m hx
  · rw [hm4]; exact hsh3.trans hsh2
  · rw [hg4 5 (by decide), hgen3 5 (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)]; exact hs2r5
  · rw [hg4 13 (by decide), hgen3 13 (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide), hs2r13]
  · rw [hg4 14 (by decide), hgen3 14 (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide), hs2r14]

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
    (run_regs_length k2 hr2).trans (run_regs_length k1 hr1),
    by rw [hg2 5 (by decide) (by decide) (by decide), hg1 5 (by decide)]⟩

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
    (run_regs_length k2 hr2).trans (run_regs_length k1 hr1),
    by rw [hg2 5 (by decide) (by decide) (by decide), hg1 5 (by decide)]⟩

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
    (run_regs_length k2 hr2).trans (run_regs_length k1 hr1),
    by rw [hg2 5 (by decide) (by decide) (by decide), hg1 5 (by decide)]⟩

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
    (run_regs_length k2 hr2).trans (run_regs_length k1 hr1),
    by rw [hg2 5 (by decide) (by decide) (by decide), hg1 5 (by decide)]⟩

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
    (run_regs_length k2 hr2).trans (run_regs_length k1 hr1),
    by rw [hg2 5 (by decide) (by decide) (by decide), hg1 5 (by decide)]⟩

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
    (run_regs_length k2 hr2).trans (run_regs_length k1 hr1),
    by rw [hg2 5 (by decide) (by decide) (by decide) (by decide) (by decide), hg1 5 (by decide)]⟩

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
    (run_regs_length k2 hr2).trans (run_regs_length k1 hr1),
    by rw [hg2 5 (by decide) (by decide) (by decide) (by decide) (by decide) (by decide), hg1 5 (by decide)]⟩

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
    (run_regs_length k2 hr2).trans (run_regs_length k1 hr1),
    by rw [hg2 5 (by decide) (by decide) (by decide) (by decide) (by decide) (by decide), hg1 5 (by decide)]⟩

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
    (run_regs_length k2 hr2).trans (run_regs_length k1 hr1),
    by rw [hg2 5 (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide), hg1 5 (by decide)]⟩

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
    (run_regs_length k2 hr2).trans (run_regs_length k1 hr1),
    by rw [hg2 5 (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide), hg1 5 (by decide)]⟩

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
    (run_regs_length k2 hr2).trans (run_regs_length k1 hr1),
    by rw [hg2 5 (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide), hg1 5 (by decide)]⟩

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
    (run_regs_length k2 hr2).trans (run_regs_length k1 hr1),
    by rw [hg2 5 (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide), hg1 5 (by decide)]⟩

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
    (run_regs_length k2 hr2).trans (run_regs_length k1 hr1),
    by rw [hg2 5 (by decide) (by decide) (by decide) (by decide) (by decide), hg1 5 (by decide)]⟩

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
    (run_regs_length k2 hr2).trans (run_regs_length k1 hr1),
    by rw [hg2 5 (by decide) (by decide) (by decide) (by decide) (by decide), hg1 5 (by decide)]⟩

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
    (run_regs_length k2 hr2).trans (run_regs_length k1 hr1),
    by rw [hg2 5 (by decide) (by decide), hg1 5 (by decide)]⟩




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
      wshape s3.mem = wshape s.mem ∧ s3.reg 5 = s.reg 5 :=
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
      wshape s3.mem = wshape s.mem ∧ s3.reg 5 = s.reg 5 :=
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
      wshape s4.mem = wshape s.mem ∧ s4.reg 5 = s.reg 5 ∧ s4.reg 13 = s.reg 13 ∧
      s4.reg 14 = s.reg 14 :=
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
        wshape s'.mem = wshape s.mem ∧ s'.reg 5 = s.reg 5 ∧
        s'.reg 13 = s.reg 13 ∧ s'.reg 14 = s.reg 14 := by
  intro code
  induction code with
  | nil =>
    intro i c cur hm s inv hw
    obtain ⟨k, s', hr, hrip, -, hr1, hr6, hr7, -, -, hr4, hgD, hmD⟩ :=
      WalkDone_triple cfg.walk (by decide) hm c i w sp wd.code.length st.w56
        (by have h := inv.isum; simp at h; omega) s inv.rip inv.r8 inv.r9
        inv.r6 inv.r4 inv.r7 inv.r1 inv.len16 inv.fnc
    refine ⟨k, s', hr, hrip, ?_, ?_, hr7, hr4, ?_, ?_, hgD 5 (by decide), hgD 13 (by decide), hgD 14 (by decide)⟩
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
            obtain ⟨k3, s3, hr3, inv3, hfr3i, hsh3i, hr53⟩ :=
              fall_finish1 cfg st inv hs hw2.1 hi1 rfl hrh hrip2 hpost
            obtain ⟨k4, s4, hr4, hrip4, hr14, hr64, hr74, hr44, hfr4, hsh4, hr54, hr134, hr144⟩ :=
              ih (i + 1) (c + 1) h' h' s3 inv3 hw2.2
            refine ⟨k3 + k4, s4, by rw [run_add im k3 k4, hr3, hr4], hrip4, ?_, ?_,
              hr74, hr44,
              fun x m hx => (hfr4 x m hx).trans (hfr3i x m hx), hsh4.trans hsh3i, hr54.trans hr53, by rw [hr134, inv3.r13, inv.r13], by rw [hr144, inv3.r14, inv.r14]⟩
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
            obtain ⟨k3, s4, hr3, hrip4, hr14, hr64, hr74, hr44, hfr4, hsh4, hr54, hr134, hr144⟩ :=
              none_finish_nil1 cfg st inv hs rfl hrh hrip2 hpost
            refine ⟨k3, s4, hr3, hrip4, ?_, ?_, hr74, hr44, hfr4, hsh4, hr54, hr134, hr144⟩
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
            obtain ⟨k3, s3, hr3, inv3, hfr3i, hsh3i, hr53⟩ :=
              none_finish_cons1 cfg st inv hs hi1 rfl hrh hrip2 hpost
            obtain ⟨k4, s4, hr4, hrip4, hr14, hr64, hr74, hr44, hfr4, hsh4, hr54, hr134, hr144⟩ :=
              ih (i + 1) (c + 1) _ _ s3 inv3 hwrest
            refine ⟨k3 + k4, s4, by rw [run_add im k3 k4, hr3, hr4], hrip4, ?_, ?_,
              hr74, hr44,
              fun x m hx => (hfr4 x m hx).trans (hfr3i x m hx), hsh4.trans hsh3i, hr54.trans hr53, by rw [hr134, inv3.r13, inv.r13], by rw [hr144, inv3.r14, inv.r14]⟩
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
        obtain ⟨k3, s3, hr3, inv3, hfr3i, hsh3i, hr53⟩ :=
          fall_finish1 cfg st inv hs hw2.1 hi1 rfl hrh hrip2 hpost
        obtain ⟨k4, s4, hr4, hrip4, hr14, hr64, hr74, hr44, hfr4, hsh4, hr54, hr134, hr144⟩ :=
          ih (i + 1) (c + 9) ⟨hm.d + 1, hm.a⟩ ⟨hm.d + 1, hm.a⟩ s3 inv3 hw2.2
        refine ⟨k3 + k4, s4, by rw [run_add im k3 k4, hr3, hr4], hrip4, ?_, ?_,
          hr74, hr44,
          fun x m hx => (hfr4 x m hx).trans (hfr3i x m hx), hsh4.trans hsh3i, hr54.trans hr53, by rw [hr134, inv3.r13, inv.r13], by rw [hr144, inv3.r14, inv.r14]⟩
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
        obtain ⟨k3, s3, hr3, inv3, hfr3i, hsh3i, hr53⟩ :=
          fall_finish1 cfg st inv hs hw2.1 hi1 rfl hrh hrip2 hpost
        obtain ⟨k4, s4, hr4, hrip4, hr14, hr64, hr74, hr44, hfr4, hsh4, hr54, hr134, hr144⟩ :=
          ih (i + 1) (c + 5) ⟨hm.d + 2, hm.a⟩ ⟨hm.d + 2, hm.a⟩ s3 inv3 hw2.2
        refine ⟨k3 + k4, s4, by rw [run_add im k3 k4, hr3, hr4], hrip4, ?_, ?_,
          hr74, hr44,
          fun x m hx => (hfr4 x m hx).trans (hfr3i x m hx), hsh4.trans hsh3i, hr54.trans hr53, by rw [hr134, inv3.r13, inv.r13], by rw [hr144, inv3.r14, inv.r14]⟩
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
            obtain ⟨k3, s3, hr3, inv3, hfr3i, hsh3i, hr53⟩ :=
              fall_finish1 cfg st inv hs hw2.1 hi1 rfl hrh hrip2 hpost
            obtain ⟨k4, s4, hr4, hrip4, hr14, hr64, hr74, hr44, hfr4, hsh4, hr54, hr134, hr144⟩ :=
              ih (i + 1) (c + 5) ⟨hm.d + 1, hm.a⟩ ⟨hm.d + 1, hm.a⟩ s3 inv3 hw2.2
            refine ⟨k3 + k4, s4, by rw [run_add im k3 k4, hr3, hr4], hrip4, ?_, ?_,
              hr74, hr44,
              fun x m hx => (hfr4 x m hx).trans (hfr3i x m hx), hsh4.trans hsh3i, hr54.trans hr53, by rw [hr134, inv3.r13, inv.r13], by rw [hr144, inv3.r14, inv.r14]⟩
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
            obtain ⟨k3, s3, hr3, inv3, hfr3i, hsh3i, hr53⟩ :=
              fall_finish1 cfg st inv hs hw2.1 hi1 rfl hrh hrip2 hpost
            obtain ⟨k4, s4, hr4, hrip4, hr14, hr64, hr74, hr44, hfr4, hsh4, hr54, hr134, hr144⟩ :=
              ih (i + 1) (c + 1) ⟨hm.d - 1, hm.a + 1⟩ ⟨hm.d - 1, hm.a + 1⟩ s3 inv3 hw2.2
            refine ⟨k3 + k4, s4, by rw [run_add im k3 k4, hr3, hr4], hrip4, ?_, ?_,
              hr74, hr44,
              fun x m hx => (hfr4 x m hx).trans (hfr3i x m hx), hsh4.trans hsh3i, hr54.trans hr53, by rw [hr134, inv3.r13, inv.r13], by rw [hr144, inv3.r14, inv.r14]⟩
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
            obtain ⟨k3, s3, hr3, inv3, hfr3i, hsh3i, hr53⟩ :=
              fall_finish1 cfg st inv hs hw2.1 hi1 rfl hrh hrip2 hpost
            obtain ⟨k4, s4, hr4, hrip4, hr14, hr64, hr74, hr44, hfr4, hsh4, hr54, hr134, hr144⟩ :=
              ih (i + 1) (c + 1) ⟨hm.d + 1, hm.a - 1⟩ ⟨hm.d + 1, hm.a - 1⟩ s3 inv3 hw2.2
            refine ⟨k3 + k4, s4, by rw [run_add im k3 k4, hr3, hr4], hrip4, ?_, ?_,
              hr74, hr44,
              fun x m hx => (hfr4 x m hx).trans (hfr3i x m hx), hsh4.trans hsh3i, hr54.trans hr53, by rw [hr134, inv3.r13, inv.r13], by rw [hr144, inv3.r14, inv.r14]⟩
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
            obtain ⟨k3, s3, hr3, inv3, hfr3i, hsh3i, hr53⟩ :=
              fall_finish1 cfg st inv hs hw2.1 hi1 rfl hrh hrip2 hpost
            obtain ⟨k4, s4, hr4, hrip4, hr14, hr64, hr74, hr44, hfr4, hsh4, hr54, hr134, hr144⟩ :=
              ih (i + 1) (c + 1) ⟨hm.d + 1, hm.a⟩ ⟨hm.d + 1, hm.a⟩ s3 inv3 hw2.2
            refine ⟨k3 + k4, s4, by rw [run_add im k3 k4, hr3, hr4], hrip4, ?_, ?_,
              hr74, hr44,
              fun x m hx => (hfr4 x m hx).trans (hfr3i x m hx), hsh4.trans hsh3i, hr54.trans hr53, by rw [hr134, inv3.r13, inv.r13], by rw [hr144, inv3.r14, inv.r14]⟩
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
            obtain ⟨k3, s3, hr3, inv3, hfr3i, hsh3i, hr53⟩ :=
              fall_finish1 cfg st inv hs hw2.1 hi1 rfl hrh hrip2 hpost
            obtain ⟨k4, s4, hr4, hrip4, hr14, hr64, hr74, hr44, hfr4, hsh4, hr54, hr134, hr144⟩ :=
              ih (i + 1) (c + 5) ⟨hm.d + 1, hm.a⟩ ⟨hm.d + 1, hm.a⟩ s3 inv3 hw2.2
            refine ⟨k3 + k4, s4, by rw [run_add im k3 k4, hr3, hr4], hrip4, ?_, ?_,
              hr74, hr44,
              fun x m hx => (hfr4 x m hx).trans (hfr3i x m hx), hsh4.trans hsh3i, hr54.trans hr53, by rw [hr134, inv3.r13, inv.r13], by rw [hr144, inv3.r14, inv.r14]⟩
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
            obtain ⟨k3, s3, hr3, inv3, hfr3i, hsh3i, hr53⟩ :=
              fall_finish1 cfg st inv hs hw2.1 hi1 rfl hrh hrip2 hpost
            obtain ⟨k4, s4, hr4, hrip4, hr14, hr64, hr74, hr44, hfr4, hsh4, hr54, hr134, hr144⟩ :=
              ih (i + 1) (c + 5) ⟨hm.d - 1, hm.a⟩ ⟨hm.d - 1, hm.a⟩ s3 inv3 hw2.2
            refine ⟨k3 + k4, s4, by rw [run_add im k3 k4, hr3, hr4], hrip4, ?_, ?_,
              hr74, hr44,
              fun x m hx => (hfr4 x m hx).trans (hfr3i x m hx), hsh4.trans hsh3i, hr54.trans hr53, by rw [hr134, inv3.r13, inv.r13], by rw [hr144, inv3.r14, inv.r14]⟩
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
            obtain ⟨k3, s4, hr3, hrip4, hr14, hr64, hr74, hr44, hfr4, hsh4, hr54, hr134, hr144⟩ :=
              none_finish_nil1 cfg st inv hs rfl hrh hrip2 hpost
            refine ⟨k3, s4, hr3, hrip4, ?_, ?_, hr74, hr44, hfr4, hsh4, hr54, hr134, hr144⟩
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
            obtain ⟨k3, s3, hr3, inv3, hfr3i, hsh3i, hr53⟩ :=
              none_finish_cons1 cfg st inv hs hi1 rfl hrh hrip2 hpost
            obtain ⟨k4, s4, hr4, hrip4, hr14, hr64, hr74, hr44, hfr4, hsh4, hr54, hr134, hr144⟩ :=
              ih (i + 1) (c + 5) _ _ s3 inv3 hwrest
            refine ⟨k3 + k4, s4, by rw [run_add im k3 k4, hr3, hr4], hrip4, ?_, ?_,
              hr74, hr44,
              fun x m hx => (hfr4 x m hx).trans (hfr3i x m hx), hsh4.trans hsh3i, hr54.trans hr53, by rw [hr134, inv3.r13, inv.r13], by rw [hr144, inv3.r14, inv.r14]⟩
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
            obtain ⟨k3, s3, hr3, inv3, hfr3i, hsh3i, hr53⟩ :=
              fall_finish1 cfg st inv hs hw2.1 hi1 rfl hrh hrip2 hpost
            obtain ⟨k4, s4, hr4, hrip4, hr14, hr64, hr74, hr44, hfr4, hsh4, hr54, hr134, hr144⟩ :=
              ih (i + 1) (c + 5) ⟨hm.d - 1, hm.a⟩ ⟨hm.d - 1, hm.a⟩ s3 inv3 hw2.2
            refine ⟨k3 + k4, s4, by rw [run_add im k3 k4, hr3, hr4], hrip4, ?_, ?_,
              hr74, hr44,
              fun x m hx => (hfr4 x m hx).trans (hfr3i x m hx), hsh4.trans hsh3i, hr54.trans hr53, by rw [hr134, inv3.r13, inv.r13], by rw [hr144, inv3.r14, inv.r14]⟩
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
            obtain ⟨k3, s3, hr3, inv3, hfr3i, hsh3i, hr53⟩ :=
              fall_finish1 cfg st inv hs hw2.1 hi1 rfl hrh hrip2 hpost
            obtain ⟨k4, s4, hr4, hrip4, hr14, hr64, hr74, hr44, hfr4, hsh4, hr54, hr134, hr144⟩ :=
              ih (i + 1) (c + 5) ⟨hm.d - 1, hm.a⟩ ⟨hm.d - 1, hm.a⟩ s3 inv3 hw2.2
            refine ⟨k3 + k4, s4, by rw [run_add im k3 k4, hr3, hr4], hrip4, ?_, ?_,
              hr74, hr44,
              fun x m hx => (hfr4 x m hx).trans (hfr3i x m hx), hsh4.trans hsh3i, hr54.trans hr53, by rw [hr134, inv3.r13, inv.r13], by rw [hr144, inv3.r14, inv.r14]⟩
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
            obtain ⟨k3, s3, hr3, inv3, hfr3i, hsh3i, hr53⟩ :=
              fall_finish1 cfg st inv hs hw2.1 hi1 rfl hrh hrip2 hpost
            obtain ⟨k4, s4, hr4, hrip4, hr14, hr64, hr74, hr44, hfr4, hsh4, hr54, hr134, hr144⟩ :=
              ih (i + 1) (c + 5) h' h' s3 inv3 hw2.2
            refine ⟨k3 + k4, s4, by rw [run_add im k3 k4, hr3, hr4], hrip4, ?_, ?_,
              hr74, hr44,
              fun x m hx => (hfr4 x m hx).trans (hfr3i x m hx), hsh4.trans hsh3i, hr54.trans hr53, by rw [hr134, inv3.r13, inv.r13], by rw [hr144, inv3.r14, inv.r14]⟩
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
            obtain ⟨k3, s4, hr3, hrip4, hr14, hr64, hr74, hr44, hfr4, hsh4, hr54, hr134, hr144⟩ :=
              none_finish_nil1 cfg st inv hs rfl hrh hrip2 hpost
            refine ⟨k3, s4, hr3, hrip4, ?_, ?_, hr74, hr44, hfr4, hsh4, hr54, hr134, hr144⟩
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
            obtain ⟨k3, s3, hr3, inv3, hfr3i, hsh3i, hr53⟩ :=
              none_finish_cons1 cfg st inv hs hi1 rfl hrh hrip2 hpost
            obtain ⟨k4, s4, hr4, hrip4, hr14, hr64, hr74, hr44, hfr4, hsh4, hr54, hr134, hr144⟩ :=
              ih (i + 1) (c + 1) _ _ s3 inv3 hwrest
            refine ⟨k3 + k4, s4, by rw [run_add im k3 k4, hr3, hr4], hrip4, ?_, ?_,
              hr74, hr44,
              fun x m hx => (hfr4 x m hx).trans (hfr3i x m hx), hsh4.trans hsh3i, hr54.trans hr53, by rw [hr134, inv3.r13, inv.r13], by rw [hr144, inv3.r14, inv.r14]⟩
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

def aCheckStart : Nat := base + 0x42d
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
  cs0 : im aCheckStart = some (.mov 7 14, 3)
  cs1 : im (aCheckStart + 3) = some (.mov 5 13, 3)
  av0 : im aAnnval = some (.test 1 1, 3)
  av1 : im (aAnnval + 3) = some (.jcc 5 (aAnnval + 10), 2)
  av2 : im (aAnnval + 5) = some (.jmp aCodewalk, 5)
  av3 : im (aAnnval + 10) = some (.push 1, 1)
  av4 : im (aAnnval + 11) = some (.push 6, 1)
  av5 : im (aAnnval + 12) = some (.mov 2 6, 3)
  av6 : im (aAnnval + 15) = some (.call aL32, 5)
  av7 : im (aAnnval + 20) = some (.mov 12 0, 3)
  av8 : im (aAnnval + 23) = some (.ld 2 7 24, 4)
  av9 : im (aAnnval + 27) = some (.cmp 0 2, 3)
  av10 : im (aAnnval + 30) = some (.jcc 2 (aAnnval + 37), 2)
  av11 : im (aAnnval + 37) = some (.call aANNL, 5)
  av12 : im (aAnnval + 42) = some (.push 0, 1)
  av13 : im (aAnnval + 43) = some (.mov 2 0, 3)
  av14 : im (aAnnval + 46) = some (.aluI .add 2 4, 4)
  av15 : im (aAnnval + 50) = some (.call aL32, 5)
  av16 : im (aAnnval + 55) = some (.mov 8 0, 3)
  av17 : im (aAnnval + 58) = some (.pop 2, 1)
  av18 : im (aAnnval + 59) = some (.aluI .add 2 8, 4)
  av19 : im (aAnnval + 63) = some (.call aL32, 5)
  av20 : im (aAnnval + 68) = some (.mov 9 0, 3)
  av21 : im (aAnnval + 71) = some (.pop 6, 1)
  av22 : im (aAnnval + 72) = some (.mov 2 6, 3)
  av23 : im (aAnnval + 75) = some (.aluI .add 2 4, 4)
  av24 : im (aAnnval + 79) = some (.call aL32, 5)
  av25 : im (aAnnval + 84) = some (.cmp 0 8, 3)
  av26 : im (aAnnval + 87) = some (.jcc 4 (aAnnval + 94), 2)
  av27 : im (aAnnval + 94) = some (.mov 2 6, 3)
  av28 : im (aAnnval + 97) = some (.aluI .add 2 8, 4)
  av29 : im (aAnnval + 101) = some (.call aL32, 5)
  av30 : im (aAnnval + 106) = some (.cmp 0 9, 3)
  av31 : im (aAnnval + 109) = some (.jcc 4 (aAnnval + 116), 2)
  av32 : im (aAnnval + 116) = some (.aluI .add 6 12, 4)
  av33 : im (aAnnval + 120) = some (.pop 1, 1)
  av34 : im (aAnnval + 121) = some (.aluI .sub 1 1, 4)
  av35 : im (aAnnval + 125) = some (.jmp aAnnval, 5)
  ck0 : im aCodewalk = some (.ld 6 7 40, 4)
  ck1 : im (aCodewalk + 4) = some (.alu .xor 1 1, 2)
  ck2 : im (aCodewalk + 6) = some (.ld 8 7 0, 3)
  ck3 : im (aCodewalk + 9) = some (.alu .xor 9 9, 3)
  ck4 : im (aCodewalk + 12) = some (.alu .xor 12 12, 3)
  ck5 : im (aCodewalk + 15) = some (.call aANNL, 5)
  ck6 : im (aCodewalk + 20) = some (.test 0 0, 3)
  ck7 : im (aCodewalk + 23) = some (.jcc 4 aWalk, 2)
  ck8 : im (aCodewalk + 25) = some (.push 0, 1)
  ck9 : im (aCodewalk + 26) = some (.mov 2 0, 3)
  ck10 : im (aCodewalk + 29) = some (.aluI .add 2 4, 4)
  ck11 : im (aCodewalk + 33) = some (.call aL32, 5)
  ck12 : im (aCodewalk + 38) = some (.cmp 0 8, 3)
  ck13 : im (aCodewalk + 41) = some (.jcc 4 (aCodewalk + 48), 2)
  ck14 : im (aCodewalk + 48) = some (.pop 2, 1)
  ck15 : im (aCodewalk + 49) = some (.aluI .add 2 8, 4)
  ck16 : im (aCodewalk + 53) = some (.call aL32, 5)
  ck17 : im (aCodewalk + 58) = some (.cmp 0 9, 3)
  ck18 : im (aCodewalk + 61) = some (.jcc 4 aWalk, 2)

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


/-- Stack room at a lowered `rsp`, transported through a `wshape`
preserving segment from six slots at the original `rsp`. -/
theorem StackRoom_shift {s u : X86Int.St} {sp L k : Nat}
    (hroom : StackRoom s 6) (hs4 : s.reg 4 = some sp)
    (hu4 : u.reg 4 = some (sp - L)) (hwsh : wshape u.mem = wshape s.mem)
    (hL : ∃ m, L = 8 * m ∧ L + 8 * k ≤ 48) (hsp : 48 ≤ sp) (hspW : sp < W) :
    StackRoom u k := by
  obtain ⟨m, rfl, hm⟩ := hL
  intro sp' hsp' i hi v
  have hsp'e : sp' = sp - 8 * m := by
    rw [hu4] at hsp'
    exact Option.some.inj hsp'.symm
  subst hsp'e
  have e : ((sp - 8 * m) + W - 8 * (i + 1)) % W = sp - 8 * (m + i + 1) := by
    rw [slot_sub (by omega) (by omega)]
    omega
  rw [e]
  have hh := hroom sp hs4 (m + i) (by omega) v
  have e2 : (sp + W - 8 * (m + i + 1)) % W = sp - 8 * (m + i + 1) :=
    slot_sub (by omega) hspW
  rw [e2] at hh
  exact wrN_some_of_wshape _ s.mem u.mem hwsh _ hh

/-- Static bounds and disjointness for the ann validation of one word. -/
def AnnStatic (wd : Sporos.Word) (w p sp : Nat) : Prop :=
  w + 56 ≤ W ∧ 0 < p ∧ p + 12 * wd.ann.length < W ∧ 48 ≤ sp ∧ sp < W ∧
  wd.code.length < W ∧ (∀ x ∈ wd.ann, x.2.d < W ∧ x.2.a < W) ∧
  (sp ≤ w ∨ w + 56 ≤ sp - 48) ∧ (sp ≤ p ∨ p + 12 * wd.ann.length ≤ sp - 48)

/-- The `annval` loop state: `rest` ann entries remain to validate. -/
def AnnSt (wd : Sporos.Word) (w p sp : Nat) (rest : List (Nat × Height))
    (s : X86Int.St) : Prop :=
  s.rip = aAnnval ∧ s.reg 7 = some w ∧
  s.reg 6 = some (p + 12 * (wd.ann.length - rest.length)) ∧
  s.reg 1 = some rest.length ∧ s.reg 4 = some sp ∧
  rd64 s.mem (w + 24) = some wd.code.length ∧
  rd64 s.mem (w + 32) = some wd.ann.length ∧
  rd64 s.mem (w + 48) = some p ∧
  AnnLayout s wd.ann p ∧ 12 < s.regs.length ∧ StackRoom s 6

/-- `annval` with no entries left falls into `codewalk`. -/
theorem annval_done {im : Nat → Option (Ins × Nat)} (oc : OuterCode im)
    {wd : Sporos.Word} {w p sp : Nat} :
    ∀ s, AnnSt wd w p sp [] s →
      ∃ k s', run im k (Out.ok s) = Out.ok s' ∧ s'.rip = aCodewalk ∧
        s'.mem = s.mem ∧ (∀ q, s'.reg q = s.reg q) ∧
        s'.regs.length = s.regs.length := by
  intro s hs
  obtain ⟨hrip, -, -, hr1, -, -, -, -, -, -, -⟩ := hs
  have hr1' : s.reg 1 = some 0 := hr1
  have e1 : X86Int.step im s =
      .ok { s with fl := (aluF .and 0 0).2, rip := s.rip + 3 } :=
    step_test (by rw [hrip]; exact oc.av0) hr1' hr1'
  obtain ⟨a1, ha1⟩ : ∃ u : X86Int.St, u =
    { s with fl := (aluF .and 0 0).2, rip := s.rip + 3 } := ⟨_, rfl⟩
  have ha1rip : a1.rip = aAnnval + 3 := by
    rw [ha1]; show s.rip + 3 = aAnnval + 3; rw [hrip]
  have ha1fl : X86Int.cond a1.fl 5 = some false := by
    have hz : a1.fl.zf = some true := by
      rw [ha1]
      show (aluF .and 0 0).2.zf = some true
      rw [aluF_zf, aluF_and_fst_self]
      exact rfl
    have := cond_cc5 hz
    simpa using this
  have e2 : X86Int.step im a1 = .ok { a1 with rip := aAnnval + 5 } := by
    have hh := step_jcc (im := im) (s := a1) (by rw [ha1rip]; exact oc.av1)
      ha1fl
    simpa [ha1rip] using hh
  obtain ⟨a2, ha2⟩ : ∃ u : X86Int.St, u = { a1 with rip := aAnnval + 5 } :=
    ⟨_, rfl⟩
  have ha2rip : a2.rip = aAnnval + 5 := by rw [ha2]
  have e3 : X86Int.step im a2 = .ok { a2 with rip := aCodewalk } :=
    step_jmp (by rw [ha2rip]; exact oc.av2)
  obtain ⟨a3, ha3⟩ : ∃ u : X86Int.St, u = { a2 with rip := aCodewalk } :=
    ⟨_, rfl⟩
  have r1 : run im 1 (Out.ok s) = Out.ok a1 := by
    rw [run_succ_ok im 0 s, e1, ← ha1, run_zero]
  have r2 : run im 1 (Out.ok a1) = Out.ok a2 := by
    rw [run_succ_ok im 0 a1, e2, ← ha2, run_zero]
  have r3 : run im 1 (Out.ok a2) = Out.ok a3 := by
    rw [run_succ_ok im 0 a2, e3, ← ha3, run_zero]
  refine ⟨1 + 1 + 1, a3, by rw [run_add im 1 2, r1, run_add im 1 1, r2, r3],
    by rw [ha3], ?_, ?_, ?_⟩
  · rw [ha3, ha2, ha1]
  · intro q
    rw [ha3, ha2, ha1]
    exact rfl
  · rw [ha3, ha2, ha1]

set_option maxHeartbeats 3200000 in
/-- One `annval` iteration: entry `(t, h)` is in range and agrees with
the first entry at `t`, and the loop invariant is restored for the
remaining entries. -/
theorem annval_step {im : Nat → Option (Ins × Nat)}
    (ac : AllCode im) (oc : OuterCode im)
    {wd : Sporos.Word} {w p sp : Nat} (st : AnnStatic wd w p sp)
    {t : Nat} {h : Height} {rest' : List (Nat × Height)}
    (hlen : rest'.length + 1 ≤ wd.ann.length)
    (htlt : t < wd.code.length) (hlk : wd.ann.lookup t = some h) :
    ∀ s, AnnSt wd w p sp ((t, h) :: rest') s →
      AnnAt s (p + 12 * (wd.ann.length - (rest'.length + 1))) t h.d h.a →
      t < W →
      ∃ k s', run im k (Out.ok s) = Out.ok s' ∧ AnnSt wd w p sp rest' s' ∧
        (∀ q, q ≠ 0 → q ≠ 1 → q ≠ 2 → q ≠ 4 → q ≠ 6 → q ≠ 8 → q ≠ 9 →
          q ≠ 12 → s'.reg q = s.reg q) ∧
        (∀ x n, x + n ≤ sp - 48 ∨ sp ≤ x → s'.mem.rdN x n = s.mem.rdN x n) ∧
        wshape s'.mem = wshape s.mem ∧ s'.regs.length = s.regs.length := by
  obtain ⟨hw56, hp0, hpb, hsp48, hspW, hncW, hhW, hdisjw, hdisjp⟩ := st
  intro s hs hcur htW
  obtain ⟨hrip, hr7, hr6, hr1, hr4, hf24, hf32, hf48, hlay, h12len,
    hroom⟩ := hs
  have hnrem : rest'.length + 1 ≠ 0 := by omega
  have hcurb : p + 12 * (wd.ann.length - (rest'.length + 1)) + 12 ≤
      p + 12 * wd.ann.length := by
    have : wd.ann.length - (rest'.length + 1) + 1 ≤ wd.ann.length := by omega
    omega
  have hannW : wd.ann.length < W := by omega
  obtain ⟨hdW, haW⟩ : h.d < W ∧ h.a < W := by
    have hmem : (t, h) ∈ wd.ann := by
      have := lookup_mem hlk
      exact this
    exact hhW (t, h) hmem
  have hslot8 : (sp + W - 8) % W = sp - 8 := by
    rw [show sp + W - 8 = sp + W - 8 * 1 by omega, slot_sub (by omega) hspW]
  -- on-demand transports from a frame against `s`
  have layOf : ∀ u : X86Int.St, (∀ x n, x + n ≤ sp - 48 ∨ sp ≤ x →
      u.mem.rdN x n = s.mem.rdN x n) → AnnLayout u wd.ann p :=
    fun u hu => AnnLayout_of_frame hlay (fun x k hx1 hx2 => hu x k (by omega))
  have atOf : ∀ u : X86Int.St, (∀ x n, x + n ≤ sp - 48 ∨ sp ≤ x →
      u.mem.rdN x n = s.mem.rdN x n) →
      AnnAt u (p + 12 * (wd.ann.length - (rest'.length + 1))) t h.d h.a :=
    fun u hu => ⟨by
        rw [hu (p + 12 * (wd.ann.length - (rest'.length + 1))) 4 (by omega)];
        exact hcur.1,
      by
        rw [hu (p + 12 * (wd.ann.length - (rest'.length + 1)) + 4) 4
          (by omega)];
        exact hcur.2.1,
      by
        rw [hu (p + 12 * (wd.ann.length - (rest'.length + 1)) + 8) 4
          (by omega)];
        exact hcur.2.2⟩
  have fldOf : ∀ u : X86Int.St, (∀ x n, x + n ≤ sp - 48 ∨ sp ≤ x →
      u.mem.rdN x n = s.mem.rdN x n) →
      rd64 u.mem (w + 24) = some wd.code.length ∧
      rd64 u.mem (w + 32) = some wd.ann.length ∧
      rd64 u.mem (w + 48) = some p :=
    fun u hu => ⟨by unfold rd64; rw [hu (w + 24) 8 (by omega)]; exact hf24,
      by unfold rd64; rw [hu (w + 32) 8 (by omega)]; exact hf32,
      by unfold rd64; rw [hu (w + 48) 8 (by omega)]; exact hf48⟩
  have hfr0 : ∀ x n, x + n ≤ sp - 48 ∨ sp ≤ x → s.mem.rdN x n = s.mem.rdN x n :=
    fun x n _ => rfl
  have hw0 : wshape s.mem = wshape s.mem := rfl
  have hlen0 : s.regs.length = s.regs.length := rfl
  have gb0 : ∀ q, q ≠ 0 → q ≠ 1 → q ≠ 2 → q ≠ 4 → q ≠ 6 → q ≠ 8 → q ≠ 9 →
      q ≠ 12 → s.reg q = s.reg q :=
    fun q _ _ _ _ _ _ _ _ => rfl
  -- test rcx, rcx
  have hr1' : s.reg 1 = some (rest'.length + 1) := hr1
  have e1 : X86Int.step im s =
      .ok { s with fl := (aluF .and (rest'.length + 1) (rest'.length + 1)).2, rip := s.rip + 3 } :=
    step_test (by rw [hrip]; exact oc.av0) hr1' hr1'
  obtain ⟨b1, hb1⟩ : ∃ u : X86Int.St, u =
    { s with fl := (aluF .and (rest'.length + 1) (rest'.length + 1)).2, rip := s.rip + 3 } := ⟨_, rfl⟩
  have hb1rip : b1.rip = aAnnval + 3 := by
    rw [hb1]; show s.rip + 3 = aAnnval + 3; rw [hrip]
  have hb1fl : X86Int.cond b1.fl 5 = some true := by
    have hz : b1.fl.zf = some false := by
      rw [hb1]
      show (aluF .and (rest'.length + 1) (rest'.length + 1)).2.zf = some false
      rw [aluF_zf, aluF_and_fst_self]
      simp [hnrem]
    have := cond_cc5 hz
    simpa using this
  have gb1 : ∀ q, b1.reg q = s.reg q := fun q => by rw [hb1]; exact rfl
  have hb1mem : b1.mem = s.mem := by rw [hb1]
  -- jnz taken
  have e2 : X86Int.step im b1 = .ok { b1 with rip := aAnnval + 10 } := by
    have hh := step_jcc (im := im) (s := b1) (by rw [hb1rip]; exact oc.av1)
      hb1fl
    simpa using hh
  obtain ⟨b2, hb2⟩ : ∃ u : X86Int.St, u = { b1 with rip := aAnnval + 10 } :=
    ⟨_, rfl⟩
  have hb2rip : b2.rip = aAnnval + 10 := by rw [hb2]
  have gb2 : ∀ q, b2.reg q = s.reg q := fun q => by
    rw [hb2]; show b1.reg q = s.reg q; exact gb1 q
  have hb2mem : b2.mem = s.mem := by rw [hb2]; exact hb1mem
  have hb2r4 : b2.reg 4 = some sp := by rw [gb2 4]; exact hr4
  have hb2r1 : b2.reg 1 = some (rest'.length + 1) := by
    rw [gb2 1]; exact hr1'
  have hlen2 : b2.regs.length = s.regs.length := by rw [hb2, hb1]
  have hb2r6 : b2.reg 6 =
      some (p + 12 * (wd.ann.length - (rest'.length + 1))) := by
    rw [gb2 6]; exact hr6
  -- push rcx
  have hw8 : ∀ v, (wr64 s.mem (sp - 8) v).isSome := by
    intro v
    have h := hroom sp hr4 0 (by decide) v
    have e : (sp + W - 8 * (0 + 1)) % W = sp - 8 :=
      slot_sub (by omega) hspW
    rw [e] at h; exact h
  have hw16s : ∀ v, (wr64 s.mem (sp - 16) v).isSome := by
    intro v
    have h := hroom sp hr4 1 (by decide) v
    have e : (sp + W - 8 * (1 + 1)) % W = sp - 16 :=
      slot_sub (by omega) hspW
    rw [e] at h; exact h
  obtain ⟨m3, hm3s⟩ := Option.isSome_iff_exists.mp (hw8 (rest'.length + 1))
  have hm3 : wr64 b2.mem ((sp + W - 8) % W) (rest'.length + 1) = some m3 := by
    rw [hb2mem, hslot8]; exact hm3s
  have e3 : X86Int.step im b2 = .ok { b2.set 4 (some ((sp + W - 8) % W)) with
      mem := m3, rip := b2.rip + 1 } :=
    step_push (by rw [hb2rip]; exact oc.av3) hb2r1 hb2r4 hm3
  obtain ⟨b3, hb3⟩ : ∃ u : X86Int.St, u =
    { b2.set 4 (some ((sp + W - 8) % W)) with mem := m3, rip := b2.rip + 1 } :=
    ⟨_, rfl⟩
  have hb3rip : b3.rip = aAnnval + 11 := by
    rw [hb3]; show b2.rip + 1 = aAnnval + 11; rw [hb2rip]
  have hb3r4 : b3.reg 4 = some (sp - 8) := by
    rw [hb3]
    show (b2.set 4 (some ((sp + W - 8) % W))).reg 4 = some (sp - 8)
    rw [reg_set_self (by omega), hslot8]
  have gb3 : ∀ q, q ≠ 4 → b3.reg q = s.reg q := by
    intro q hq
    rw [hb3]
    show (b2.set 4 (some ((sp + W - 8) % W))).reg q = s.reg q
    rw [reg_set_other hq]; exact gb2 q
  have hb3mem : b3.mem = m3 := by rw [hb3]
  have hfr3 : ∀ x n, x + n ≤ sp - 48 ∨ sp ≤ x →
      b3.mem.rdN x n = s.mem.rdN x n := by
    intro x n hx
    rw [hb3mem]
    exact rdN_wr64_other hm3s (by omega)
  have hw3 : wshape b3.mem = wshape s.mem := by
    rw [hb3mem]; exact wrN_wshape _ _ _ hm3s
  have hlen3 : b3.regs.length = s.regs.length := by
    rw [hb3]
    show (b2.set 4 (some ((sp + W - 8) % W))).regs.length = s.regs.length
    exact List.length_set.trans hlen2
  have hslot8_3 : rd64 b3.mem (sp - 8) = some (rest'.length + 1) := by
    have h1 := rd64_wr64 hm3s
    rw [Nat.mod_eq_of_lt (by omega)] at h1
    rw [hb3mem]; exact h1
  -- push rsi
  obtain ⟨m4, hm4s⟩ := Option.isSome_iff_exists.mp
    (wr64_some_of_wr64 hm3s
      (hw16s (p + 12 * (wd.ann.length - (rest'.length + 1)))))
  have hsp16 : ((sp - 8) + W - 8) % W = sp - 16 := by
    rw [show (sp - 8) + W - 8 = (sp - 8) + W - 8 * 1 by omega,
      slot_sub (by omega) (by omega)]
    omega
  have hm4 : wr64 b3.mem (((sp - 8) + W - 8) % W)
      (p + 12 * (wd.ann.length - (rest'.length + 1))) = some m4 := by
    rw [hb3mem, hsp16]; exact hm4s
  have hb3r6 : b3.reg 6 =
      some (p + 12 * (wd.ann.length - (rest'.length + 1))) := by
    rw [gb3 6 (by decide)]; exact hr6
  have e4 : X86Int.step im b3 = .ok { b3.set 4 (some (((sp - 8) + W - 8) % W))
      with mem := m4, rip := b3.rip + 1 } :=
    step_push (by rw [hb3rip]; exact oc.av4) hb3r6 hb3r4 hm4
  obtain ⟨b4, hb4⟩ : ∃ u : X86Int.St, u =
    { b3.set 4 (some (((sp - 8) + W - 8) % W)) with mem := m4, rip := b3.rip + 1 } := ⟨_, rfl⟩
  have hb4rip : b4.rip = aAnnval + 12 := by
    rw [hb4]; show b3.rip + 1 = aAnnval + 12; rw [hb3rip]
  have hb4r4 : b4.reg 4 = some (sp - 16) := by
    rw [hb4]
    show (b3.set 4 (some (((sp - 8) + W - 8) % W))).reg 4 = some (sp - 16)
    rw [reg_set_self (by omega), hsp16]
  have gb4 : ∀ q, q ≠ 4 → b4.reg q = s.reg q := by
    intro q hq
    rw [hb4]
    show (b3.set 4 (some (((sp - 8) + W - 8) % W))).reg q = s.reg q
    rw [reg_set_other hq]; exact gb3 q hq
  have hb4mem : b4.mem = m4 := by rw [hb4]
  have hfr4 : ∀ x n, x + n ≤ sp - 48 ∨ sp ≤ x →
      b4.mem.rdN x n = s.mem.rdN x n := by
    intro x n hx
    rw [hb4mem, rdN_wr64_other hm4s (by omega), ← hb3mem]
    exact hfr3 x n hx
  have hw4 : wshape b4.mem = wshape s.mem := by
    rw [hb4mem]
    exact (wrN_wshape _ _ _ hm4s).trans (by rw [← hb3mem]; exact hw3)
  have hlen4 : b4.regs.length = s.regs.length := by
    rw [hb4]
    show (b3.set 4 (some (((sp - 8) + W - 8) % W))).regs.length =
      s.regs.length
    exact List.length_set.trans hlen3
  have hslot16_4 : rd64 b4.mem (sp - 16) =
      some (p + 12 * (wd.ann.length - (rest'.length + 1))) := by
    have h1 := rd64_wr64 hm4s
    rw [Nat.mod_eq_of_lt (by omega)] at h1
    rw [hb4mem]; exact h1
  have hslot8_4 : rd64 b4.mem (sp - 8) = some (rest'.length + 1) := by
    rw [hb4mem, rd64_wr64_other hm4s (by omega)]
    rw [← hb3mem]; exact hslot8_3
  -- mov rdx, rsi
  have hb4r6 : b4.reg 6 =
      some (p + 12 * (wd.ann.length - (rest'.length + 1))) := by
    rw [gb4 6 (by decide)]; exact hr6
  have e5 : X86Int.step im b4 = .ok { b4.set 2
      (some (p + 12 * (wd.ann.length - (rest'.length + 1)))) with
      rip := b4.rip + 3 } :=
    step_mov (by rw [hb4rip]; exact oc.av5) hb4r6
  obtain ⟨b5, hb5⟩ : ∃ u : X86Int.St, u = { b4.set 2
      (some (p + 12 * (wd.ann.length - (rest'.length + 1)))) with
      rip := b4.rip + 3 } := ⟨_, rfl⟩
  have hb5rip : b5.rip = aAnnval + 15 := by
    rw [hb5]; show b4.rip + 3 = aAnnval + 15; rw [hb4rip]
  have hb5r2 : b5.reg 2 =
      some (p + 12 * (wd.ann.length - (rest'.length + 1))) := by
    rw [hb5]
    show (b4.set 2 (some (p + 12 * (wd.ann.length - (rest'.length + 1))))).reg 2 = _
    rw [reg_set_self (by omega)]
  have gb5 : ∀ q, q ≠ 2 → b5.reg q = b4.reg q := by
    intro q hq
    rw [hb5]
    show (b4.set 2 _).reg q = b4.reg q
    rw [reg_set_other hq]
  have hb5mem : b5.mem = b4.mem := by rw [hb5]; exact rfl
  have hb5r4 : b5.reg 4 = some (sp - 16) := by
    rw [gb5 4 (by decide)]; exact hb4r4
  have hlen5 : b5.regs.length = s.regs.length := by
    rw [hb5]
    show (b4.set 2 _).regs.length = s.regs.length
    exact List.length_set.trans hlen4
  have hfr5 : ∀ x n, x + n ≤ sp - 48 ∨ sp ≤ x →
      b5.mem.rdN x n = s.mem.rdN x n := by
    intro x n hx; rw [hb5mem]; exact hfr4 x n hx
  have hw5 : wshape b5.mem = wshape s.mem := by rw [hb5mem]; exact hw4
  -- call L32: rax = this entry's index t
  have hb5r1 : b5.reg 1 = some (rest'.length + 1) := by
    rw [gb5 1 (by decide), gb4 1 (by decide)]; exact hr1'
  have hrd5 : (b5.mem.rdN (p + 12 * (wd.ann.length - (rest'.length + 1))) 4).map
      leVal = some t := (atOf b5 hfr5).1
  obtain ⟨k6, b6, hrun6, hb6rip, hb6r0, hb6r4, gb6, hlen6, hfr6seg, hw6seg⟩ :=
    CallL32 ac.l32 ac.annl.dret oc.av6 (by omega) (by omega) (by omega)
      (by decide) (by omega) b5 hb5rip hb5r2 hb5r4 hrd5
      (StackRoom_shift hroom hr4 hb5r4 hw5 ⟨2, rfl, by omega⟩ hsp48 hspW)
  have hlen6s : b6.regs.length = s.regs.length := hlen6.trans hlen5
  have hfr6 : ∀ x n, x + n ≤ sp - 48 ∨ sp ≤ x →
      b6.mem.rdN x n = s.mem.rdN x n := by
    intro x n hx
    exact (hfr6seg x n (by omega)).trans (hfr5 x n hx)
  have hw6 : wshape b6.mem = wshape s.mem := hw6seg.trans hw5
  have hb6r7 : b6.reg 7 = some w := by
    rw [gb6 7 (by decide) (by decide), gb5 7 (by decide), gb4 7 (by decide)]
    exact hr7
  have hb6r1 : b6.reg 1 = some (rest'.length + 1) := by
    rw [gb6 1 (by decide) (by decide)]; exact hb5r1
  have hb6r6 : b6.reg 6 =
      some (p + 12 * (wd.ann.length - (rest'.length + 1))) := by
    rw [gb6 6 (by decide) (by decide), gb5 6 (by decide)]; exact hb4r6
  have hslot16_6 : rd64 b6.mem (sp - 16) =
      some (p + 12 * (wd.ann.length - (rest'.length + 1))) := by
    unfold rd64
    rw [hfr6seg (sp - 16) 8 (by omega), hb5mem]
    exact hslot16_4
  have hslot8_6 : rd64 b6.mem (sp - 8) = some (rest'.length + 1) := by
    unfold rd64
    rw [hfr6seg (sp - 8) 8 (by omega), hb5mem]
    exact hslot8_4
  -- mov r12, rax
  have e7 : X86Int.step im b6 = .ok { b6.set 12 (some t) with
      rip := b6.rip + 3 } :=
    step_mov (by rw [hb6rip]; exact oc.av7) hb6r0
  obtain ⟨b7, hb7⟩ : ∃ u : X86Int.St, u = { b6.set 12 (some t) with
    rip := b6.rip + 3 } := ⟨_, rfl⟩
  have hb7rip : b7.rip = aAnnval + 23 := by
    rw [hb7]; show b6.rip + 3 = aAnnval + 23; rw [hb6rip]
  have hb7r12 : b7.reg 12 = some t := by
    rw [hb7]
    show (b6.set 12 (some t)).reg 12 = some t
    rw [reg_set_self (by omega)]
  have gb7 : ∀ q, q ≠ 12 → b7.reg q = b6.reg q := by
    intro q hq
    rw [hb7]
    show (b6.set 12 (some t)).reg q = b6.reg q
    rw [reg_set_other hq]
  have hb7mem : b7.mem = b6.mem := by rw [hb7]; exact rfl
  have hlen7 : b7.regs.length = s.regs.length := by
    rw [hb7]
    show (b6.set 12 (some t)).regs.length = s.regs.length
    exact List.length_set.trans hlen6s
  have hfr7 : ∀ x n, x + n ≤ sp - 48 ∨ sp ≤ x →
      b7.mem.rdN x n = s.mem.rdN x n := by
    intro x n hx; rw [hb7mem]; exact hfr6 x n hx
  have hw7 : wshape b7.mem = wshape s.mem := by rw [hb7mem]; exact hw6
  -- mov rdx, [rdi+24]
  have hb7r7 : b7.reg 7 = some w := by
    rw [gb7 7 (by decide)]; exact hb6r7
  have hea24 : ea w 24 = w + 24 := Nat.mod_eq_of_lt (by omega)
  have e8 : X86Int.step im b7 = .ok { b7.set 2 (some wd.code.length) with
      rip := b7.rip + 4 } :=
    step_ld (by rw [hb7rip]; exact oc.av8) hb7r7
      (by rw [hea24]; exact (fldOf b7 hfr7).1)
  obtain ⟨b8, hb8⟩ : ∃ u : X86Int.St, u = { b7.set 2 (some wd.code.length) with
    rip := b7.rip + 4 } := ⟨_, rfl⟩
  have hb8rip : b8.rip = aAnnval + 27 := by
    rw [hb8]; show b7.rip + 4 = aAnnval + 27; rw [hb7rip]
  have gb8 : ∀ q, q ≠ 2 → b8.reg q = b7.reg q := by
    intro q hq
    rw [hb8]
    show (b7.set 2 (some wd.code.length)).reg q = b7.reg q
    rw [reg_set_other hq]
  have hb8r2 : b8.reg 2 = some wd.code.length := by
    rw [hb8]
    show (b7.set 2 (some wd.code.length)).reg 2 = some wd.code.length
    rw [reg_set_self (by omega)]
  have hb8r0 : b8.reg 0 = some t := by
    rw [gb8 0 (by decide), gb7 0 (by decide)]; exact hb6r0
  have hb8mem : b8.mem = b7.mem := by rw [hb8]; exact rfl
  have hlen8 : b8.regs.length = s.regs.length := by
    rw [hb8]
    show (b7.set 2 (some wd.code.length)).regs.length = s.regs.length
    exact List.length_set.trans hlen7
  have hfr8 : ∀ x n, x + n ≤ sp - 48 ∨ sp ≤ x →
      b8.mem.rdN x n = s.mem.rdN x n := by
    intro x n hx; rw [hb8mem]; exact hfr7 x n hx
  have hw8 : wshape b8.mem = wshape s.mem := by rw [hb8mem]; exact hw7
  -- cmp rax, rdx ; jb taken
  have e9 : X86Int.step im b8 =
      .ok { b8 with fl := (aluF .sub t wd.code.length).2, rip := b8.rip + 3 } :=
    step_cmp (by rw [hb8rip]; exact oc.av9) hb8r0 hb8r2
  obtain ⟨b9, hb9⟩ : ∃ u : X86Int.St, u =
    { b8 with fl := (aluF .sub t wd.code.length).2, rip := b8.rip + 3 } :=
    ⟨_, rfl⟩
  have hb9rip : b9.rip = aAnnval + 30 := by
    rw [hb9]; show b8.rip + 3 = aAnnval + 30; rw [hb8rip]
  have hb9fl : X86Int.cond b9.fl 2 = some true := by
    have hcf : b9.fl.cf = some true := by
      rw [hb9]
      show (aluF .sub t wd.code.length).2.cf = some true
      exact aluF_sub_cf_true htlt
    exact cond_cc2 hcf
  have gb9 : ∀ q, b9.reg q = b8.reg q := fun q => by rw [hb9]; exact rfl
  have hb9mem : b9.mem = b8.mem := by rw [hb9]
  have e10 : X86Int.step im b9 = .ok { b9 with rip := aAnnval + 37 } := by
    have hh := step_jcc (im := im) (s := b9) (by rw [hb9rip]; exact oc.av10)
      hb9fl
    simpa using hh
  obtain ⟨b10, hb10⟩ : ∃ u : X86Int.St, u = { b9 with rip := aAnnval + 37 } :=
    ⟨_, rfl⟩
  have hb10rip : b10.rip = aAnnval + 37 := by rw [hb10]
  have gb10 : ∀ q, b10.reg q = b8.reg q := fun q => by
    rw [hb10]; show b9.reg q = b8.reg q; exact gb9 q
  have hb10mem : b10.mem = b8.mem := by rw [hb10]; exact hb9mem
  have hfr10 : ∀ x n, x + n ≤ sp - 48 ∨ sp ≤ x →
      b10.mem.rdN x n = s.mem.rdN x n := by
    intro x n hx; rw [hb10mem]; exact hfr8 x n hx
  have hw10 : wshape b10.mem = wshape s.mem := by rw [hb10mem]; exact hw8
  have hlen10 : b10.regs.length = s.regs.length := by rw [hb10, hb9]; exact hlen8
  have hb10r12 : b10.reg 12 = some t := by
    rw [gb10 12, gb8 12 (by decide)]; exact hb7r12
  have hb10r7 : b10.reg 7 = some w := by
    rw [gb10 7, gb8 7 (by decide)]; exact hb7r7
  have hb10r4 : b10.reg 4 = some (sp - 16) := by
    rw [gb10 4, gb8 4 (by decide), gb7 4 (by decide)]; exact hb6r4
  have hb10r1 : b10.reg 1 = some (rest'.length + 1) := by
    rw [gb10 1, gb8 1 (by decide), gb7 1 (by decide)]; exact hb6r1
  have hb10r6 : b10.reg 6 =
      some (p + 12 * (wd.ann.length - (rest'.length + 1))) := by
    rw [gb10 6, gb8 6 (by decide), gb7 6 (by decide)]; exact hb6r6
  have hslot16_10 : rd64 b10.mem (sp - 16) =
      some (p + 12 * (wd.ann.length - (rest'.length + 1))) := by
    rw [hb10mem, hb8mem, hb7mem]; exact hslot16_6
  have hslot8_10 : rd64 b10.mem (sp - 8) = some (rest'.length + 1) := by
    rw [hb10mem, hb8mem, hb7mem]; exact hslot8_6
  -- call ANNL: rax = first entry at t
  obtain ⟨k11, b11, hrun11, hb11rip, hpost, hb11r1, hb11r4, hb11r12, hb11r7,
    hlay11, gb11, hfr11seg, hex0, hloc11, hw11seg, hlen11⟩ :=
    CallANNL ac.annl ac.l32 (by decide) oc.av11 (by decide) htW hp0 hw56
      hpb (by omega) (by omega) b10 hb10rip hb10r12 hb10r7 hb10r4 hb10r1
      (fldOf b10 hfr10).2.2 (fldOf b10 hfr10).2.1 (layOf b10 hfr10)
      (StackRoom_shift hroom hr4 hb10r4 hw10 ⟨2, rfl, by omega⟩ hsp48 hspW)
      (by omega) (by omega)
  have hlen11s : b11.regs.length = s.regs.length := hlen11.trans hlen10
  have hfr11 : ∀ x n, x + n ≤ sp - 48 ∨ sp ≤ x →
      b11.mem.rdN x n = s.mem.rdN x n := by
    intro x n hx
    exact (hfr11seg x n (by omega)).trans (hfr10 x n hx)
  have hw11 : wshape b11.mem = wshape s.mem := hw11seg.trans hw10
  have hslot16_11 : rd64 b11.mem (sp - 16) =
      some (p + 12 * (wd.ann.length - (rest'.length + 1))) := by
    unfold rd64
    rw [hfr11seg (sp - 16) 8 (by omega)]
    exact hslot16_10
  have hslot8_11 : rd64 b11.mem (sp - 8) = some (rest'.length + 1) := by
    unfold rd64
    rw [hfr11seg (sp - 8) 8 (by omega)]
    exact hslot8_10
  obtain ⟨e, hb11r0⟩ := hex0
  obtain ⟨he0, hat11, hd1, ha1⟩ : e ≠ 0 ∧ AnnAt b11 e t h.d h.a ∧
      p ≤ e ∧ e + 12 ≤ p + 12 * wd.ann.length := by
    cases hpost e hb11r0 with
    | inl hb =>
      rw [hb.2] at hlk
      simp at hlk
    | inr hb =>
      obtain ⟨d1, a1, hne, hat, hlk1⟩ := hb
      have hdeq : d1 = h.d := by
        have hhh : (⟨d1, a1⟩ : Height) = h :=
          Option.some.inj (hlk1.symm.trans hlk)
        exact congrArg Height.d hhh
      have haeq : a1 = h.a := by
        have hhh : (⟨d1, a1⟩ : Height) = h :=
          Option.some.inj (hlk1.symm.trans hlk)
        exact congrArg Height.a hhh
      obtain ⟨hl1, hl2⟩ := hloc11 e hb11r0 hne
      exact ⟨hne, by rwa [hdeq, haeq] at hat, hl1, hl2⟩
  have hadd : ∀ (x k : Nat), x + k < W → (aluF .add x (k % W)).1 = x + k := by
    intro x k hk
    rw [aluF_add_fst]
    have hkW : k % W = k := Nat.mod_eq_of_lt (by omega)
    rw [hkW]
    exact Nat.mod_eq_of_lt hk
  have hp24 : ((sp - 24) + 8) % W = sp - 16 := by
    rw [show sp - 24 + 8 = sp - 16 by omega]
    exact Nat.mod_eq_of_lt (by omega)
  have hp16 : ((sp - 16) + 8) % W = sp - 8 := by
    rw [show sp - 16 + 8 = sp - 8 by omega]
    exact Nat.mod_eq_of_lt (by omega)
  -- push rax (first entry)
  have hroom11 : StackRoom b11 1 :=
    StackRoom_shift hroom hr4 hb11r4 hw11 ⟨2, rfl, by omega⟩ hsp48 hspW
  obtain ⟨m12, hm12s⟩ := Option.isSome_iff_exists.mp
    (hroom11 (sp - 16) hb11r4 0 (by decide) e)
  have hsp24 : ((sp - 16) + W - 8) % W = sp - 24 := by
    rw [show (sp - 16) + W - 8 = (sp - 16) + W - 8 * 1 by omega,
      slot_sub (by omega) (by omega)]
    omega
  have hm12 : wr64 b11.mem (sp - 24) e = some m12 := by
    have e' : (sp - 16 + W - 8 * (0 + 1)) % W = sp - 24 :=
      slot_sub (by omega) (by omega)
    rw [e'] at hm12s; exact hm12s
  have hm12p : wr64 b11.mem (((sp - 16) + W - 8) % W) e = some m12 := by
    rw [hsp24]; exact hm12
  have e12 : X86Int.step im b11 = .ok { b11.set 4 (some (((sp - 16) + W - 8) % W))
      with mem := m12, rip := b11.rip + 1 } :=
    step_push (by rw [hb11rip]; exact oc.av12) hb11r0 hb11r4 hm12p
  obtain ⟨b12, hb12⟩ : ∃ u : X86Int.St, u =
    { b11.set 4 (some (((sp - 16) + W - 8) % W)) with mem := m12, rip := b11.rip + 1 } := ⟨_, rfl⟩
  have hb12rip : b12.rip = aAnnval + 43 := by
    rw [hb12]; show b11.rip + 1 = aAnnval + 43; rw [hb11rip]
  have hb12r4 : b12.reg 4 = some (sp - 24) := by
    rw [hb12]
    show (b11.set 4 (some (((sp - 16) + W - 8) % W))).reg 4 = some (sp - 24)
    rw [reg_set_self (by omega), hsp24]
  have gb12 : ∀ q, q ≠ 4 → b12.reg q = b11.reg q := by
    intro q hq
    rw [hb12]
    show (b11.set 4 (some (((sp - 16) + W - 8) % W))).reg q = b11.reg q
    rw [reg_set_other hq]
  have hb12mem : b12.mem = m12 := by rw [hb12]
  have hfr12 : ∀ x n, x + n ≤ sp - 48 ∨ sp ≤ x →
      b12.mem.rdN x n = s.mem.rdN x n := by
    intro x n hx
    rw [hb12mem, rdN_wr64_other hm12 (by omega)]
    exact hfr11 x n hx
  have hw12 : wshape b12.mem = wshape s.mem := by
    rw [hb12mem]
    exact (wrN_wshape _ _ _ hm12).trans hw11
  have hlen12 : b12.regs.length = s.regs.length := by
    rw [hb12]
    show (b11.set 4 (some (((sp - 16) + W - 8) % W))).regs.length =
      s.regs.length
    exact List.length_set.trans hlen11s
  have hslot24_12 : rd64 b12.mem (sp - 24) = some e := by
    have h1 := rd64_wr64 hm12
    rw [Nat.mod_eq_of_lt (by omega)] at h1
    rw [hb12mem]; exact h1
  have hslot16_12 : rd64 b12.mem (sp - 16) =
      some (p + 12 * (wd.ann.length - (rest'.length + 1))) := by
    rw [hb12mem, rd64_wr64_other hm12 (by omega)]
    exact hslot16_11
  have hslot8_12 : rd64 b12.mem (sp - 8) = some (rest'.length + 1) := by
    rw [hb12mem, rd64_wr64_other hm12 (by omega)]
    exact hslot8_11
  have hat12 : AnnAt b12 e t h.d h.a :=
    ⟨by rw [hb12mem, rdN_wr64_other hm12 (by omega)]; exact hat11.1,
     by rw [hb12mem, rdN_wr64_other hm12 (by omega)]; exact hat11.2.1,
     by rw [hb12mem, rdN_wr64_other hm12 (by omega)]; exact hat11.2.2⟩
  -- mov rdx, rax ; add rdx, 4
  have hb12r0 : b12.reg 0 = some e := by
    rw [gb12 0 (by decide)]; exact hb11r0
  have e13 : X86Int.step im b12 = .ok { b12.set 2 (some e) with
      rip := b12.rip + 3 } :=
    step_mov (by rw [hb12rip]; exact oc.av13) hb12r0
  obtain ⟨b13, hb13⟩ : ∃ u : X86Int.St, u = { b12.set 2 (some e) with
    rip := b12.rip + 3 } := ⟨_, rfl⟩
  have hb13rip : b13.rip = aAnnval + 46 := by
    rw [hb13]; show b12.rip + 3 = aAnnval + 46; rw [hb12rip]
  have gb13 : ∀ q, q ≠ 2 → b13.reg q = b12.reg q := by
    intro q hq
    rw [hb13]
    show (b12.set 2 (some e)).reg q = b12.reg q
    rw [reg_set_other hq]
  have hb13r2 : b13.reg 2 = some e := by
    rw [hb13]
    show (b12.set 2 (some e)).reg 2 = some e
    rw [reg_set_self (by omega)]
  have hb13mem : b13.mem = b12.mem := by rw [hb13]; exact rfl
  have e14 : X86Int.step im b13 = .ok { b13.set 2
      (some (aluF .add e (4 % W)).1) with fl := (aluF .add e (4 % W)).2, rip := b13.rip + 4 } :=
    step_aluI_add (by rw [hb13rip]; exact oc.av14) hb13r2
  obtain ⟨b14, hb14⟩ : ∃ u : X86Int.St, u = { b13.set 2
      (some (aluF .add e (4 % W)).1) with fl := (aluF .add e (4 % W)).2, rip := b13.rip + 4 } := ⟨_, rfl⟩
  have hb14rip : b14.rip = aAnnval + 50 := by
    rw [hb14]; show b13.rip + 4 = aAnnval + 50; rw [hb13rip]
  have hb14r2 : b14.reg 2 = some (e + 4) := by
    rw [hb14]
    show (b13.set 2 (some (aluF .add e (4 % W)).1)).reg 2 = some (e + 4)
    rw [reg_set_self (by
      have : b13.regs.length = s.regs.length := by
        rw [hb13]
        show (b12.set 2 (some e)).regs.length = s.regs.length
        exact List.length_set.trans hlen12
      omega), hadd e 4 (by omega)]
  have gb14 : ∀ q, q ≠ 2 → b14.reg q = b13.reg q := by
    intro q hq
    rw [hb14]
    show (b13.set 2 (some (aluF .add e (4 % W)).1)).reg q = b13.reg q
    rw [reg_set_other hq]
  have hb14mem : b14.mem = b12.mem := by
    rw [hb14]; show b13.mem = b12.mem; exact hb13mem
  have hb14r4 : b14.reg 4 = some (sp - 24) := by
    rw [gb14 4 (by decide), gb13 4 (by decide)]; exact hb12r4
  have hfr14 : ∀ x n, x + n ≤ sp - 48 ∨ sp ≤ x →
      b14.mem.rdN x n = s.mem.rdN x n := by
    intro x n hx; rw [hb14mem]; exact hfr12 x n hx
  have hw14 : wshape b14.mem = wshape s.mem := by rw [hb14mem]; exact hw12
  have hlen14 : b14.regs.length = s.regs.length := by
    rw [hb14]
    show (b13.set 2 (some (aluF .add e (4 % W)).1)).regs.length =
      s.regs.length
    refine List.length_set.trans ?_
    rw [hb13]
    show (b12.set 2 (some e)).regs.length = s.regs.length
    exact List.length_set.trans hlen12
  -- call L32: rax = first d
  obtain ⟨k15, b15, hrun15, hb15rip, hb15r0, hb15r4, gb15, hlen15, hfr15seg,
    hw15seg⟩ :=
    CallL32 ac.l32 ac.annl.dret oc.av15 (by omega) (by omega) (by omega)
      (by decide) (by omega) b14 hb14rip hb14r2 hb14r4
      (by rw [hb14mem]; exact hat12.2.1)
      (StackRoom_shift hroom hr4 hb14r4 hw14 ⟨3, rfl, by omega⟩ hsp48 hspW)
  have hlen15s : b15.regs.length = s.regs.length := hlen15.trans hlen14
  have hfr15 : ∀ x n, x + n ≤ sp - 48 ∨ sp ≤ x →
      b15.mem.rdN x n = s.mem.rdN x n := by
    intro x n hx
    exact (hfr15seg x n (by omega)).trans (hfr14 x n hx)
  have hw15 : wshape b15.mem = wshape s.mem := hw15seg.trans hw14
  have hslot24_15 : rd64 b15.mem (sp - 24) = some e := by
    unfold rd64
    rw [hfr15seg (sp - 24) 8 (by omega), hb14mem]
    exact hslot24_12
  have hslot16_15 : rd64 b15.mem (sp - 16) =
      some (p + 12 * (wd.ann.length - (rest'.length + 1))) := by
    unfold rd64
    rw [hfr15seg (sp - 16) 8 (by omega), hb14mem]
    exact hslot16_12
  have hslot8_15 : rd64 b15.mem (sp - 8) = some (rest'.length + 1) := by
    unfold rd64
    rw [hfr15seg (sp - 8) 8 (by omega), hb14mem]
    exact hslot8_12
  have hat15 : AnnAt b15 e t h.d h.a :=
    ⟨by rw [hfr15seg e 4 (by omega), hb14mem]; exact hat12.1,
     by rw [hfr15seg (e + 4) 4 (by omega), hb14mem]; exact hat12.2.1,
     by rw [hfr15seg (e + 8) 4 (by omega), hb14mem]; exact hat12.2.2⟩
  -- mov r8, rax
  have e16 : X86Int.step im b15 = .ok { b15.set 8 (some h.d) with
      rip := b15.rip + 3 } :=
    step_mov (by rw [hb15rip]; exact oc.av16) hb15r0
  obtain ⟨b16, hb16⟩ : ∃ u : X86Int.St, u = { b15.set 8 (some h.d) with
    rip := b15.rip + 3 } := ⟨_, rfl⟩
  have hb16rip : b16.rip = aAnnval + 58 := by
    rw [hb16]; show b15.rip + 3 = aAnnval + 58; rw [hb15rip]
  have hb16r8 : b16.reg 8 = some h.d := by
    rw [hb16]
    show (b15.set 8 (some h.d)).reg 8 = some h.d
    rw [reg_set_self (by omega)]
  have gb16 : ∀ q, q ≠ 8 → b16.reg q = b15.reg q := by
    intro q hq
    rw [hb16]
    show (b15.set 8 (some h.d)).reg q = b15.reg q
    rw [reg_set_other hq]
  have hb16mem : b16.mem = b15.mem := by rw [hb16]; exact rfl
  have hb16r4 : b16.reg 4 = some (sp - 24) := by
    rw [gb16 4 (by decide)]; exact hb15r4
  have hlen16 : b16.regs.length = s.regs.length := by
    rw [hb16]
    show (b15.set 8 (some h.d)).regs.length = s.regs.length
    exact List.length_set.trans hlen15s
  -- pop rdx
  have e17 : X86Int.step im b16 = .ok
      { (b16.set 4 (some ((sp - 24 + 8) % W))).set 2 (some e) with
        rip := b16.rip + 1 } :=
    step_pop (by rw [hb16rip]; exact oc.av17) hb16r4
      (by rw [hb16mem]; exact hslot24_15)
  obtain ⟨b17, hb17⟩ : ∃ u : X86Int.St, u =
    { (b16.set 4 (some ((sp - 24 + 8) % W))).set 2 (some e) with
      rip := b16.rip + 1 } := ⟨_, rfl⟩
  have hb17rip : b17.rip = aAnnval + 59 := by
    rw [hb17]; show b16.rip + 1 = aAnnval + 59; rw [hb16rip]
  have hb17r4 : b17.reg 4 = some (sp - 16) := by
    rw [hb17]
    show ((b16.set 4 (some ((sp - 24 + 8) % W))).set 2 (some e)).reg 4 =
      some (sp - 16)
    rw [reg_set_other (by decide), reg_set_self (by omega), hp24]
  have hb17r2 : b17.reg 2 = some e := by
    rw [hb17]
    show ((b16.set 4 (some ((sp - 24 + 8) % W))).set 2 (some e)).reg 2 =
      some e
    rw [reg_set_self (by
      have e : (b16.set 4 (some ((sp - 24 + 8) % W))).regs.length =
          s.regs.length := List.length_set.trans hlen16
      omega)]
  have gb17 : ∀ q, q ≠ 2 → q ≠ 4 → b17.reg q = b16.reg q := by
    intro q h2 h4
    rw [hb17]
    show ((b16.set 4 (some ((sp - 24 + 8) % W))).set 2 (some e)).reg q =
      b16.reg q
    rw [reg_set_other h2, reg_set_other h4]
  have hb17mem : b17.mem = b16.mem := by rw [hb17]; exact rfl
  have hlen17 : b17.regs.length = s.regs.length := by
    rw [hb17]
    show ((b16.set 4 (some ((sp - 24 + 8) % W))).set 2 (some e)).regs.length =
      s.regs.length
    exact List.length_set.trans (List.length_set.trans hlen16)
  have hb17r8 : b17.reg 8 = some h.d := by
    rw [gb17 8 (by decide) (by decide)]; exact hb16r8
  -- add rdx, 8
  have e18 : X86Int.step im b17 = .ok { b17.set 2
      (some (aluF .add e (8 % W)).1) with fl := (aluF .add e (8 % W)).2, rip := b17.rip + 4 } :=
    step_aluI_add (by rw [hb17rip]; exact oc.av18) hb17r2
  obtain ⟨b18, hb18⟩ : ∃ u : X86Int.St, u = { b17.set 2
      (some (aluF .add e (8 % W)).1) with fl := (aluF .add e (8 % W)).2, rip := b17.rip + 4 } := ⟨_, rfl⟩
  have hb18rip : b18.rip = aAnnval + 63 := by
    rw [hb18]; show b17.rip + 4 = aAnnval + 63; rw [hb17rip]
  have hb18r2 : b18.reg 2 = some (e + 8) := by
    rw [hb18]
    show (b17.set 2 (some (aluF .add e (8 % W)).1)).reg 2 = some (e + 8)
    rw [reg_set_self (by omega), hadd e 8 (by omega)]
  have gb18 : ∀ q, q ≠ 2 → b18.reg q = b17.reg q := by
    intro q hq
    rw [hb18]
    show (b17.set 2 (some (aluF .add e (8 % W)).1)).reg q = b17.reg q
    rw [reg_set_other hq]
  have hb18mem : b18.mem = b15.mem := by
    rw [hb18]; show b17.mem = b15.mem; rw [hb17mem]; exact hb16mem
  have hb18r4 : b18.reg 4 = some (sp - 16) := by
    rw [gb18 4 (by decide)]; exact hb17r4
  have hfr18 : ∀ x n, x + n ≤ sp - 48 ∨ sp ≤ x →
      b18.mem.rdN x n = s.mem.rdN x n := by
    intro x n hx; rw [hb18mem]; exact hfr15 x n hx
  have hw18 : wshape b18.mem = wshape s.mem := by rw [hb18mem]; exact hw15
  have hlen18 : b18.regs.length = s.regs.length := by
    rw [hb18]
    show (b17.set 2 (some (aluF .add e (8 % W)).1)).regs.length =
      s.regs.length
    exact List.length_set.trans hlen17
  have hb18r8 : b18.reg 8 = some h.d := by
    rw [gb18 8 (by decide)]; exact hb17r8
  -- call L32: rax = first a
  obtain ⟨k19, b19, hrun19, hb19rip, hb19r0, hb19r4, gb19, hlen19, hfr19seg,
    hw19seg⟩ :=
    CallL32 ac.l32 ac.annl.dret oc.av19 (by omega) (by omega) (by omega)
      (by decide) (by omega) b18 hb18rip hb18r2 hb18r4
      (by rw [hb18mem]; exact hat15.2.2)
      (StackRoom_shift hroom hr4 hb18r4 hw18 ⟨2, rfl, by omega⟩ hsp48 hspW)
  have hlen19s : b19.regs.length = s.regs.length := hlen19.trans hlen18
  have hfr19 : ∀ x n, x + n ≤ sp - 48 ∨ sp ≤ x →
      b19.mem.rdN x n = s.mem.rdN x n := by
    intro x n hx
    exact (hfr19seg x n (by omega)).trans (hfr18 x n hx)
  have hw19 : wshape b19.mem = wshape s.mem := hw19seg.trans hw18
  have hb19r8 : b19.reg 8 = some h.d := by
    rw [gb19 8 (by decide) (by decide)]; exact hb18r8
  have hslot16_19 : rd64 b19.mem (sp - 16) =
      some (p + 12 * (wd.ann.length - (rest'.length + 1))) := by
    unfold rd64
    rw [hfr19seg (sp - 16) 8 (by omega), hb18mem]
    exact hslot16_15
  have hslot8_19 : rd64 b19.mem (sp - 8) = some (rest'.length + 1) := by
    unfold rd64
    rw [hfr19seg (sp - 8) 8 (by omega), hb18mem]
    exact hslot8_15
  -- mov r9, rax
  have e20 : X86Int.step im b19 = .ok { b19.set 9 (some h.a) with
      rip := b19.rip + 3 } :=
    step_mov (by rw [hb19rip]; exact oc.av20) hb19r0
  obtain ⟨b20, hb20⟩ : ∃ u : X86Int.St, u = { b19.set 9 (some h.a) with
    rip := b19.rip + 3 } := ⟨_, rfl⟩
  have hb20rip : b20.rip = aAnnval + 71 := by
    rw [hb20]; show b19.rip + 3 = aAnnval + 71; rw [hb19rip]
  have hb20r9 : b20.reg 9 = some h.a := by
    rw [hb20]
    show (b19.set 9 (some h.a)).reg 9 = some h.a
    rw [reg_set_self (by omega)]
  have gb20 : ∀ q, q ≠ 9 → b20.reg q = b19.reg q := by
    intro q hq
    rw [hb20]
    show (b19.set 9 (some h.a)).reg q = b19.reg q
    rw [reg_set_other hq]
  have hb20mem : b20.mem = b19.mem := by rw [hb20]; exact rfl
  have hb20r4 : b20.reg 4 = some (sp - 16) := by
    rw [gb20 4 (by decide)]; exact hb19r4
  have hb20r8 : b20.reg 8 = some h.d := by
    rw [gb20 8 (by decide)]; exact hb19r8
  have hlen20 : b20.regs.length = s.regs.length := by
    rw [hb20]
    show (b19.set 9 (some h.a)).regs.length = s.regs.length
    exact List.length_set.trans hlen19s
  -- pop rsi
  have e21 : X86Int.step im b20 = .ok
      { (b20.set 4 (some ((sp - 16 + 8) % W))).set 6
          (some (p + 12 * (wd.ann.length - (rest'.length + 1)))) with
        rip := b20.rip + 1 } :=
    step_pop (by rw [hb20rip]; exact oc.av21) hb20r4
      (by rw [hb20mem]; exact hslot16_19)
  obtain ⟨b21, hb21⟩ : ∃ u : X86Int.St, u =
    { (b20.set 4 (some ((sp - 16 + 8) % W))).set 6
        (some (p + 12 * (wd.ann.length - (rest'.length + 1)))) with
      rip := b20.rip + 1 } := ⟨_, rfl⟩
  have hb21rip : b21.rip = aAnnval + 72 := by
    rw [hb21]; show b20.rip + 1 = aAnnval + 72; rw [hb20rip]
  have hb21r4 : b21.reg 4 = some (sp - 8) := by
    rw [hb21]
    show ((b20.set 4 (some ((sp - 16 + 8) % W))).set 6 _).reg 4 =
      some (sp - 8)
    rw [reg_set_other (by decide), reg_set_self (by omega), hp16]
  have hb21r6 : b21.reg 6 =
      some (p + 12 * (wd.ann.length - (rest'.length + 1))) := by
    rw [hb21]
    show ((b20.set 4 (some ((sp - 16 + 8) % W))).set 6 _).reg 6 = _
    rw [reg_set_self (by
      have e : (b20.set 4 (some ((sp - 16 + 8) % W))).regs.length =
          s.regs.length := List.length_set.trans hlen20
      omega)]
  have gb21 : ∀ q, q ≠ 6 → q ≠ 4 → b21.reg q = b20.reg q := by
    intro q h6 h4
    rw [hb21]
    show ((b20.set 4 (some ((sp - 16 + 8) % W))).set 6 _).reg q = b20.reg q
    rw [reg_set_other h6, reg_set_other h4]
  have hb21mem : b21.mem = b20.mem := by rw [hb21]; exact rfl
  have hlen21 : b21.regs.length = s.regs.length := by
    rw [hb21]
    show ((b20.set 4 (some ((sp - 16 + 8) % W))).set 6 _).regs.length =
      s.regs.length
    exact List.length_set.trans (List.length_set.trans hlen20)
  have hb21r8 : b21.reg 8 = some h.d := by
    rw [gb21 8 (by decide) (by decide)]; exact hb20r8
  have hb21r9 : b21.reg 9 = some h.a := by
    rw [gb21 9 (by decide) (by decide)]; exact hb20r9
  have hfr21 : ∀ x n, x + n ≤ sp - 48 ∨ sp ≤ x →
      b21.mem.rdN x n = s.mem.rdN x n := by
    intro x n hx; rw [hb21mem, hb20mem]; exact hfr19 x n hx
  have hw21 : wshape b21.mem = wshape s.mem := by
    rw [hb21mem, hb20mem]; exact hw19
  have hslot8_21 : rd64 b21.mem (sp - 8) = some (rest'.length + 1) := by
    rw [hb21mem, hb20mem]; exact hslot8_19
  -- mov rdx, rsi ; add rdx, 4
  have e22 : X86Int.step im b21 = .ok { b21.set 2
      (some (p + 12 * (wd.ann.length - (rest'.length + 1)))) with
      rip := b21.rip + 3 } :=
    step_mov (by rw [hb21rip]; exact oc.av22) hb21r6
  obtain ⟨b22, hb22⟩ : ∃ u : X86Int.St, u = { b21.set 2
      (some (p + 12 * (wd.ann.length - (rest'.length + 1)))) with
      rip := b21.rip + 3 } := ⟨_, rfl⟩
  have hb22rip : b22.rip = aAnnval + 75 := by
    rw [hb22]; show b21.rip + 3 = aAnnval + 75; rw [hb21rip]
  have gb22 : ∀ q, q ≠ 2 → b22.reg q = b21.reg q := by
    intro q hq
    rw [hb22]
    show (b21.set 2 _).reg q = b21.reg q
    rw [reg_set_other hq]
  have hb22r2 : b22.reg 2 =
      some (p + 12 * (wd.ann.length - (rest'.length + 1))) := by
    rw [hb22]
    show (b21.set 2 _).reg 2 = _
    rw [reg_set_self (by omega)]
  have hb22mem : b22.mem = b21.mem := by rw [hb22]; exact rfl
  have e23 : X86Int.step im b22 = .ok { b22.set 2
      (some (aluF .add (p + 12 * (wd.ann.length - (rest'.length + 1)))
        (4 % W)).1) with
      fl := (aluF .add (p + 12 * (wd.ann.length - (rest'.length + 1)))
        (4 % W)).2, rip := b22.rip + 4 } :=
    step_aluI_add (by rw [hb22rip]; exact oc.av23) hb22r2
  obtain ⟨b23, hb23⟩ : ∃ u : X86Int.St, u = { b22.set 2
      (some (aluF .add (p + 12 * (wd.ann.length - (rest'.length + 1)))
        (4 % W)).1) with
      fl := (aluF .add (p + 12 * (wd.ann.length - (rest'.length + 1)))
        (4 % W)).2, rip := b22.rip + 4 } := ⟨_, rfl⟩
  have hb23rip : b23.rip = aAnnval + 79 := by
    rw [hb23]; show b22.rip + 4 = aAnnval + 79; rw [hb22rip]
  have hb23r2 : b23.reg 2 =
      some (p + 12 * (wd.ann.length - (rest'.length + 1)) + 4) := by
    rw [hb23]
    show (b22.set 2 _).reg 2 = _
    rw [reg_set_self (by
      have : b22.regs.length = s.regs.length := by
        rw [hb22]
        show (b21.set 2 _).regs.length = s.regs.length
        exact List.length_set.trans hlen21
      omega), hadd _ 4 (by omega)]
  have gb23 : ∀ q, q ≠ 2 → b23.reg q = b22.reg q := by
    intro q hq
    rw [hb23]
    show (b22.set 2 _).reg q = b22.reg q
    rw [reg_set_other hq]
  have hb23mem : b23.mem = b21.mem := by
    rw [hb23]; show b22.mem = b21.mem; exact hb22mem
  have hb23r4 : b23.reg 4 = some (sp - 8) := by
    rw [gb23 4 (by decide), gb22 4 (by decide)]; exact hb21r4
  have hb23r8 : b23.reg 8 = some h.d := by
    rw [gb23 8 (by decide), gb22 8 (by decide)]; exact hb21r8
  have hb23r9 : b23.reg 9 = some h.a := by
    rw [gb23 9 (by decide), gb22 9 (by decide)]; exact hb21r9
  have hb23r6 : b23.reg 6 =
      some (p + 12 * (wd.ann.length - (rest'.length + 1))) := by
    rw [gb23 6 (by decide), gb22 6 (by decide)]; exact hb21r6
  have hfr23 : ∀ x n, x + n ≤ sp - 48 ∨ sp ≤ x →
      b23.mem.rdN x n = s.mem.rdN x n := by
    intro x n hx; rw [hb23mem]; exact hfr21 x n hx
  have hw23 : wshape b23.mem = wshape s.mem := by rw [hb23mem]; exact hw21
  have hlen23 : b23.regs.length = s.regs.length := by
    rw [hb23]
    show (b22.set 2 _).regs.length = s.regs.length
    refine List.length_set.trans ?_
    rw [hb22]
    show (b21.set 2 _).regs.length = s.regs.length
    exact List.length_set.trans hlen21
  -- call L32: rax = this d
  obtain ⟨k24, b24, hrun24, hb24rip, hb24r0, hb24r4, gb24, hlen24, hfr24seg,
    hw24seg⟩ :=
    CallL32 ac.l32 ac.annl.dret oc.av24 (by omega) (by omega) (by omega)
      (by decide) (by omega) b23 hb23rip hb23r2 hb23r4
      (by rw [hb23mem]; exact (atOf b21 hfr21).2.1)
      (StackRoom_shift hroom hr4 hb23r4 hw23 ⟨1, by omega, by omega⟩ hsp48
        hspW)
  have hlen24s : b24.regs.length = s.regs.length := hlen24.trans hlen23
  have hfr24 : ∀ x n, x + n ≤ sp - 48 ∨ sp ≤ x →
      b24.mem.rdN x n = s.mem.rdN x n := by
    intro x n hx
    exact (hfr24seg x n (by omega)).trans (hfr23 x n hx)
  have hw24 : wshape b24.mem = wshape s.mem := hw24seg.trans hw23
  have hb24r8 : b24.reg 8 = some h.d := by
    rw [gb24 8 (by decide) (by decide)]; exact hb23r8
  have hb24r9 : b24.reg 9 = some h.a := by
    rw [gb24 9 (by decide) (by decide)]; exact hb23r9
  have hb24r6 : b24.reg 6 =
      some (p + 12 * (wd.ann.length - (rest'.length + 1))) := by
    rw [gb24 6 (by decide) (by decide)]; exact hb23r6
  have hslot8_24 : rd64 b24.mem (sp - 8) = some (rest'.length + 1) := by
    unfold rd64
    rw [hfr24seg (sp - 8) 8 (by omega), hb23mem]
    exact hslot8_21
  -- cmp rax, r8 ; jz taken
  have e25 : X86Int.step im b24 =
      .ok { b24 with fl := (aluF .sub h.d h.d).2, rip := b24.rip + 3 } :=
    step_cmp (by rw [hb24rip]; exact oc.av25) hb24r0 hb24r8
  obtain ⟨b25, hb25⟩ : ∃ u : X86Int.St, u =
    { b24 with fl := (aluF .sub h.d h.d).2, rip := b24.rip + 3 } := ⟨_, rfl⟩
  have hb25rip : b25.rip = aAnnval + 87 := by
    rw [hb25]; show b24.rip + 3 = aAnnval + 87; rw [hb24rip]
  have hb25fl : X86Int.cond b25.fl 4 = some true := by
    have hz : b25.fl.zf = some true := by
      rw [hb25]
      show (aluF .sub h.d h.d).2.zf = some true
      rw [aluF_zf, aluF_sub_fst, subW_eq_zero hdW hdW]; simp
    exact cond_cc4 hz
  have gb25 : ∀ q, b25.reg q = b24.reg q := fun q => by rw [hb25]; exact rfl
  have hb25mem : b25.mem = b24.mem := by rw [hb25]
  have e26 : X86Int.step im b25 = .ok { b25 with rip := aAnnval + 94 } := by
    have hh := step_jcc (im := im) (s := b25) (by rw [hb25rip]; exact oc.av26)
      hb25fl
    simpa using hh
  obtain ⟨b26, hb26⟩ : ∃ u : X86Int.St, u = { b25 with rip := aAnnval + 94 } :=
    ⟨_, rfl⟩
  have hb26rip : b26.rip = aAnnval + 94 := by rw [hb26]
  have gb26 : ∀ q, b26.reg q = b24.reg q := fun q => by
    rw [hb26]; show b25.reg q = b24.reg q; exact gb25 q
  have hb26mem : b26.mem = b24.mem := by rw [hb26]; exact hb25mem
  have hb26r6 : b26.reg 6 =
      some (p + 12 * (wd.ann.length - (rest'.length + 1))) := by
    rw [gb26 6]; exact hb24r6
  have hb26r4 : b26.reg 4 = some (sp - 8) := by rw [gb26 4]; exact hb24r4
  have hb26r9 : b26.reg 9 = some h.a := by rw [gb26 9]; exact hb24r9
  -- mov rdx, rsi ; add rdx, 8
  have e27 : X86Int.step im b26 = .ok { b26.set 2
      (some (p + 12 * (wd.ann.length - (rest'.length + 1)))) with
      rip := b26.rip + 3 } :=
    step_mov (by rw [hb26rip]; exact oc.av27) hb26r6
  obtain ⟨b27, hb27⟩ : ∃ u : X86Int.St, u = { b26.set 2
      (some (p + 12 * (wd.ann.length - (rest'.length + 1)))) with
      rip := b26.rip + 3 } := ⟨_, rfl⟩
  have hb27rip : b27.rip = aAnnval + 97 := by
    rw [hb27]; show b26.rip + 3 = aAnnval + 97; rw [hb26rip]
  have gb27 : ∀ q, q ≠ 2 → b27.reg q = b26.reg q := by
    intro q hq
    rw [hb27]
    show (b26.set 2 _).reg q = b26.reg q
    rw [reg_set_other hq]
  have hb27r2 : b27.reg 2 =
      some (p + 12 * (wd.ann.length - (rest'.length + 1))) := by
    rw [hb27]
    show (b26.set 2 _).reg 2 = _
    rw [reg_set_self (by
      have : b26.regs.length = s.regs.length := by
        rw [hb26, hb25]; exact hlen24s
      omega)]
  have hb27mem : b27.mem = b26.mem := by rw [hb27]; exact rfl
  have e28 : X86Int.step im b27 = .ok { b27.set 2
      (some (aluF .add (p + 12 * (wd.ann.length - (rest'.length + 1)))
        (8 % W)).1) with
      fl := (aluF .add (p + 12 * (wd.ann.length - (rest'.length + 1)))
        (8 % W)).2, rip := b27.rip + 4 } :=
    step_aluI_add (by rw [hb27rip]; exact oc.av28) hb27r2
  obtain ⟨b28, hb28⟩ : ∃ u : X86Int.St, u = { b27.set 2
      (some (aluF .add (p + 12 * (wd.ann.length - (rest'.length + 1)))
        (8 % W)).1) with
      fl := (aluF .add (p + 12 * (wd.ann.length - (rest'.length + 1)))
        (8 % W)).2, rip := b27.rip + 4 } := ⟨_, rfl⟩
  have hb28rip : b28.rip = aAnnval + 101 := by
    rw [hb28]; show b27.rip + 4 = aAnnval + 101; rw [hb27rip]
  have hb28r2 : b28.reg 2 =
      some (p + 12 * (wd.ann.length - (rest'.length + 1)) + 8) := by
    rw [hb28]
    show (b27.set 2 _).reg 2 = _
    rw [reg_set_self (by
      have : b27.regs.length = s.regs.length := by
        rw [hb27]
        show (b26.set 2 _).regs.length = s.regs.length
        refine List.length_set.trans ?_
        rw [hb26, hb25]; exact hlen24s
      omega), hadd _ 8 (by omega)]
  have gb28 : ∀ q, q ≠ 2 → b28.reg q = b27.reg q := by
    intro q hq
    rw [hb28]
    show (b27.set 2 _).reg q = b27.reg q
    rw [reg_set_other hq]
  have hb28mem : b28.mem = b24.mem := by
    rw [hb28]; show b27.mem = b24.mem; rw [hb27mem]; exact hb26mem
  have hb28r4 : b28.reg 4 = some (sp - 8) := by
    rw [gb28 4 (by decide), gb27 4 (by decide)]; exact hb26r4
  have hb28r9 : b28.reg 9 = some h.a := by
    rw [gb28 9 (by decide), gb27 9 (by decide)]; exact hb26r9
  have hb28r6 : b28.reg 6 =
      some (p + 12 * (wd.ann.length - (rest'.length + 1))) := by
    rw [gb28 6 (by decide), gb27 6 (by decide)]; exact hb26r6
  have hfr28 : ∀ x n, x + n ≤ sp - 48 ∨ sp ≤ x →
      b28.mem.rdN x n = s.mem.rdN x n := by
    intro x n hx; rw [hb28mem]; exact hfr24 x n hx
  have hw28 : wshape b28.mem = wshape s.mem := by rw [hb28mem]; exact hw24
  have hlen28 : b28.regs.length = s.regs.length := by
    rw [hb28]
    show (b27.set 2 _).regs.length = s.regs.length
    refine List.length_set.trans ?_
    rw [hb27]
    show (b26.set 2 _).regs.length = s.regs.length
    refine List.length_set.trans ?_
    rw [hb26, hb25]; exact hlen24s
  -- call L32: rax = this a
  obtain ⟨k29, b29, hrun29, hb29rip, hb29r0, hb29r4, gb29, hlen29, hfr29seg,
    hw29seg⟩ :=
    CallL32 ac.l32 ac.annl.dret oc.av29 (by omega) (by omega) (by omega)
      (by decide) (by omega) b28 hb28rip hb28r2 hb28r4
      (by rw [hb28mem, hb26mem] at *; exact (atOf b24 hfr24).2.2)
      (StackRoom_shift hroom hr4 hb28r4 hw28 ⟨1, by omega, by omega⟩ hsp48
        hspW)
  have hlen29s : b29.regs.length = s.regs.length := hlen29.trans hlen28
  have hfr29 : ∀ x n, x + n ≤ sp - 48 ∨ sp ≤ x →
      b29.mem.rdN x n = s.mem.rdN x n := by
    intro x n hx
    exact (hfr29seg x n (by omega)).trans (hfr28 x n hx)
  have hw29 : wshape b29.mem = wshape s.mem := hw29seg.trans hw28
  have hb29r9 : b29.reg 9 = some h.a := by
    rw [gb29 9 (by decide) (by decide)]; exact hb28r9
  have hb29r6 : b29.reg 6 =
      some (p + 12 * (wd.ann.length - (rest'.length + 1))) := by
    rw [gb29 6 (by decide) (by decide)]; exact hb28r6
  have hslot8_29 : rd64 b29.mem (sp - 8) = some (rest'.length + 1) := by
    unfold rd64
    rw [hfr29seg (sp - 8) 8 (by omega), hb28mem]
    exact hslot8_24
  -- cmp rax, r9 ; jz taken
  have e30 : X86Int.step im b29 =
      .ok { b29 with fl := (aluF .sub h.a h.a).2, rip := b29.rip + 3 } :=
    step_cmp (by rw [hb29rip]; exact oc.av30) hb29r0 hb29r9
  obtain ⟨b30, hb30⟩ : ∃ u : X86Int.St, u =
    { b29 with fl := (aluF .sub h.a h.a).2, rip := b29.rip + 3 } := ⟨_, rfl⟩
  have hb30rip : b30.rip = aAnnval + 109 := by
    rw [hb30]; show b29.rip + 3 = aAnnval + 109; rw [hb29rip]
  have hb30fl : X86Int.cond b30.fl 4 = some true := by
    have hz : b30.fl.zf = some true := by
      rw [hb30]
      show (aluF .sub h.a h.a).2.zf = some true
      rw [aluF_zf, aluF_sub_fst, subW_eq_zero haW haW]; simp
    exact cond_cc4 hz
  have gb30 : ∀ q, b30.reg q = b29.reg q := fun q => by rw [hb30]; exact rfl
  have hb30mem : b30.mem = b29.mem := by rw [hb30]
  have e31 : X86Int.step im b30 = .ok { b30 with rip := aAnnval + 116 } := by
    have hh := step_jcc (im := im) (s := b30) (by rw [hb30rip]; exact oc.av31)
      hb30fl
    simpa using hh
  obtain ⟨b31, hb31⟩ : ∃ u : X86Int.St, u = { b30 with rip := aAnnval + 116 } :=
    ⟨_, rfl⟩
  have hb31rip : b31.rip = aAnnval + 116 := by rw [hb31]
  have gb31 : ∀ q, b31.reg q = b29.reg q := fun q => by
    rw [hb31]; show b30.reg q = b29.reg q; exact gb30 q
  have hb31mem : b31.mem = b29.mem := by rw [hb31]; exact hb30mem
  have hb31r6 : b31.reg 6 =
      some (p + 12 * (wd.ann.length - (rest'.length + 1))) := by
    rw [gb31 6]; exact hb29r6
  have hb31r4 : b31.reg 4 = some (sp - 8) := by rw [gb31 4]; exact hb29r4
  -- add rsi, 12
  have e32 : X86Int.step im b31 = .ok { b31.set 6
      (some (aluF .add (p + 12 * (wd.ann.length - (rest'.length + 1)))
        (12 % W)).1) with
      fl := (aluF .add (p + 12 * (wd.ann.length - (rest'.length + 1)))
        (12 % W)).2, rip := b31.rip + 4 } :=
    step_aluI_add (by rw [hb31rip]; exact oc.av32) hb31r6
  obtain ⟨b32, hb32⟩ : ∃ u : X86Int.St, u = { b31.set 6
      (some (aluF .add (p + 12 * (wd.ann.length - (rest'.length + 1)))
        (12 % W)).1) with
      fl := (aluF .add (p + 12 * (wd.ann.length - (rest'.length + 1)))
        (12 % W)).2, rip := b31.rip + 4 } := ⟨_, rfl⟩
  have hb32rip : b32.rip = aAnnval + 120 := by
    rw [hb32]; show b31.rip + 4 = aAnnval + 120; rw [hb31rip]
  have hb32r6 : b32.reg 6 =
      some (p + 12 * (wd.ann.length - (rest'.length + 1)) + 12) := by
    rw [hb32]
    show (b31.set 6 _).reg 6 = _
    rw [reg_set_self (show 6 < b31.regs.length from
      Nat.lt_of_le_of_lt (show (6 : Nat) ≤ 12 by decide) (by
        have e : b31.regs.length = s.regs.length := by
          rw [hb31, hb30]; exact hlen29s
        rw [e]; exact h12len)),
      hadd _ 12 (Nat.lt_of_le_of_lt hcurb hpb)]
  have gb32 : ∀ q, q ≠ 6 → b32.reg q = b31.reg q := by
    intro q hq
    rw [hb32]
    show (b31.set 6 _).reg q = b31.reg q
    rw [reg_set_other hq]
  have hb32mem : b32.mem = b29.mem := by
    rw [hb32]; show b31.mem = b29.mem; exact hb31mem
  have hb32r4 : b32.reg 4 = some (sp - 8) := by
    rw [gb32 4 (by decide)]; exact hb31r4
  have hlen32 : b32.regs.length = s.regs.length := by
    rw [hb32]
    show (b31.set 6 _).regs.length = s.regs.length
    refine List.length_set.trans ?_
    rw [hb31, hb30]; exact hlen29s
  -- pop rcx
  have e33 : X86Int.step im b32 = .ok
      { (b32.set 4 (some ((sp - 8 + 8) % W))).set 1
          (some (rest'.length + 1)) with rip := b32.rip + 1 } :=
    step_pop (by rw [hb32rip]; exact oc.av33) hb32r4
      (by rw [hb32mem]; exact hslot8_29)
  obtain ⟨b33, hb33⟩ : ∃ u : X86Int.St, u =
    { (b32.set 4 (some ((sp - 8 + 8) % W))).set 1 (some (rest'.length + 1)) with
      rip := b32.rip + 1 } := ⟨_, rfl⟩
  have hb33rip : b33.rip = aAnnval + 121 := by
    rw [hb33]; show b32.rip + 1 = aAnnval + 121; rw [hb32rip]
  have hb33r4 : b33.reg 4 = some sp := by
    rw [hb33]
    show ((b32.set 4 (some ((sp - 8 + 8) % W))).set 1 _).reg 4 = some sp
    rw [reg_set_other (by decide),
      reg_set_self (show 4 < b32.regs.length from
        Nat.lt_of_le_of_lt (show (4 : Nat) ≤ 12 by decide) (by rw [hlen32]; exact h12len)),
      slot_add8 (by omega) hspW]
  have hb33r1 : b33.reg 1 = some (rest'.length + 1) := by
    rw [hb33]
    show ((b32.set 4 (some ((sp - 8 + 8) % W))).set 1 _).reg 1 = _
    rw [reg_set_self (show 1 < (b32.set 4 (some ((sp - 8 + 8) % W))).regs.length
      from Nat.lt_of_le_of_lt (show (1 : Nat) ≤ 12 by decide) (by
        have e : (b32.set 4 (some ((sp - 8 + 8) % W))).regs.length =
            s.regs.length := List.length_set.trans hlen32
        rw [e]; exact h12len))]
  have gb33 : ∀ q, q ≠ 1 → q ≠ 4 → b33.reg q = b32.reg q := by
    intro q h1 h4
    rw [hb33]
    show ((b32.set 4 (some ((sp - 8 + 8) % W))).set 1 _).reg q = b32.reg q
    rw [reg_set_other h1, reg_set_other h4]
  have hb33mem : b33.mem = b32.mem := by rw [hb33]; exact rfl
  have hlen33 : b33.regs.length = s.regs.length := by
    rw [hb33]
    show ((b32.set 4 (some ((sp - 8 + 8) % W))).set 1 _).regs.length =
      s.regs.length
    exact List.length_set.trans (List.length_set.trans hlen32)
  have hb33r6 : b33.reg 6 =
      some (p + 12 * (wd.ann.length - (rest'.length + 1)) + 12) := by
    rw [gb33 6 (by decide) (by decide)]; exact hb32r6
  -- sub rcx, 1
  have e34 : X86Int.step im b33 = .ok { b33.set 1
      (some (aluF .sub (rest'.length + 1) (1 % W)).1) with
      fl := (aluF .sub (rest'.length + 1) (1 % W)).2, rip := b33.rip + 4 } :=
    step_aluI_sub (by rw [hb33rip]; exact oc.av34) hb33r1
  obtain ⟨b34, hb34⟩ : ∃ u : X86Int.St, u = { b33.set 1
      (some (aluF .sub (rest'.length + 1) (1 % W)).1) with
      fl := (aluF .sub (rest'.length + 1) (1 % W)).2, rip := b33.rip + 4 } :=
    ⟨_, rfl⟩
  have hb34rip : b34.rip = aAnnval + 125 := by
    rw [hb34]; show b33.rip + 4 = aAnnval + 125; rw [hb33rip]
  have hb34r1 : b34.reg 1 = some rest'.length := by
    rw [hb34]
    show (b33.set 1 (some (aluF .sub (rest'.length + 1) (1 % W)).1)).reg 1 =
      some rest'.length
    have hsub1 : rest'.length + 1 + W - 1 = rest'.length + W := by
      rw [Nat.add_right_comm, Nat.add_sub_cancel]
    rw [reg_set_self (show 1 < b33.regs.length from
        Nat.lt_of_le_of_lt (show (1 : Nat) ≤ 12 by decide) (by rw [hlen33]; exact h12len)),
      aluF_sub_fst,
      show (1 : Nat) % W = 1 from Nat.mod_eq_of_lt (by unfold W; decide),
      hsub1, Nat.add_mod_right,
      Nat.mod_eq_of_lt (Nat.lt_of_le_of_lt (Nat.le_of_succ_le hlen) hannW)]
  have gb34 : ∀ q, q ≠ 1 → b34.reg q = b33.reg q := by
    intro q hq
    rw [hb34]
    show (b33.set 1 _).reg q = b33.reg q
    rw [reg_set_other hq]
  have hb34mem : b34.mem = b29.mem := by
    rw [hb34]; show b33.mem = b29.mem; rw [hb33mem]; exact hb32mem
  have hlen34 : b34.regs.length = s.regs.length := by
    rw [hb34]
    show (b33.set 1 _).regs.length = s.regs.length
    exact List.length_set.trans hlen33
  -- jmp annval
  have e35 : X86Int.step im b34 = .ok { b34 with rip := aAnnval } :=
    step_jmp (by rw [hb34rip]; exact oc.av35)
  obtain ⟨b35, hb35⟩ : ∃ u : X86Int.St, u = { b34 with rip := aAnnval } :=
    ⟨_, rfl⟩
  have hb35rip : b35.rip = aAnnval := by rw [hb35]
  have gb35 : ∀ q, b35.reg q = b34.reg q := fun q => by rw [hb35]; exact rfl
  have hb35mem : b35.mem = b29.mem := by rw [hb35]; exact hb34mem
  have hfr35 : ∀ x n, x + n ≤ sp - 48 ∨ sp ≤ x →
      b35.mem.rdN x n = s.mem.rdN x n := by
    intro x n hx; rw [hb35mem]; exact hfr29 x n hx
  have hw35 : wshape b35.mem = wshape s.mem := by rw [hb35mem]; exact hw29
  have hlen35 : b35.regs.length = s.regs.length := by
    rw [hb35]; exact hlen34

  have rr1 : run im 1 (Out.ok s) = Out.ok b1 := by
    rw [run_succ_ok im 0 s, e1, ← hb1, run_zero]
  have rr2 : run im 1 (Out.ok b1) = Out.ok b2 := by
    rw [run_succ_ok im 0 b1, e2, ← hb2, run_zero]
  have rr3 : run im 1 (Out.ok b2) = Out.ok b3 := by
    rw [run_succ_ok im 0 b2, e3, ← hb3, run_zero]
  have rr4 : run im 1 (Out.ok b3) = Out.ok b4 := by
    rw [run_succ_ok im 0 b3, e4, ← hb4, run_zero]
  have rr5 : run im 1 (Out.ok b4) = Out.ok b5 := by
    rw [run_succ_ok im 0 b4, e5, ← hb5, run_zero]
  have rr7 : run im 1 (Out.ok b6) = Out.ok b7 := by
    rw [run_succ_ok im 0 b6, e7, ← hb7, run_zero]
  have rr8 : run im 1 (Out.ok b7) = Out.ok b8 := by
    rw [run_succ_ok im 0 b7, e8, ← hb8, run_zero]
  have rr9 : run im 1 (Out.ok b8) = Out.ok b9 := by
    rw [run_succ_ok im 0 b8, e9, ← hb9, run_zero]
  have rr10 : run im 1 (Out.ok b9) = Out.ok b10 := by
    rw [run_succ_ok im 0 b9, e10, ← hb10, run_zero]
  have rr12 : run im 1 (Out.ok b11) = Out.ok b12 := by
    rw [run_succ_ok im 0 b11, e12, ← hb12, run_zero]
  have rr13 : run im 1 (Out.ok b12) = Out.ok b13 := by
    rw [run_succ_ok im 0 b12, e13, ← hb13, run_zero]
  have rr14 : run im 1 (Out.ok b13) = Out.ok b14 := by
    rw [run_succ_ok im 0 b13, e14, ← hb14, run_zero]
  have rr16 : run im 1 (Out.ok b15) = Out.ok b16 := by
    rw [run_succ_ok im 0 b15, e16, ← hb16, run_zero]
  have rr17 : run im 1 (Out.ok b16) = Out.ok b17 := by
    rw [run_succ_ok im 0 b16, e17, ← hb17, run_zero]
  have rr18 : run im 1 (Out.ok b17) = Out.ok b18 := by
    rw [run_succ_ok im 0 b17, e18, ← hb18, run_zero]
  have rr20 : run im 1 (Out.ok b19) = Out.ok b20 := by
    rw [run_succ_ok im 0 b19, e20, ← hb20, run_zero]
  have rr21 : run im 1 (Out.ok b20) = Out.ok b21 := by
    rw [run_succ_ok im 0 b20, e21, ← hb21, run_zero]
  have rr22 : run im 1 (Out.ok b21) = Out.ok b22 := by
    rw [run_succ_ok im 0 b21, e22, ← hb22, run_zero]
  have rr23 : run im 1 (Out.ok b22) = Out.ok b23 := by
    rw [run_succ_ok im 0 b22, e23, ← hb23, run_zero]
  have rr25 : run im 1 (Out.ok b24) = Out.ok b25 := by
    rw [run_succ_ok im 0 b24, e25, ← hb25, run_zero]
  have rr26 : run im 1 (Out.ok b25) = Out.ok b26 := by
    rw [run_succ_ok im 0 b25, e26, ← hb26, run_zero]
  have rr27 : run im 1 (Out.ok b26) = Out.ok b27 := by
    rw [run_succ_ok im 0 b26, e27, ← hb27, run_zero]
  have rr28 : run im 1 (Out.ok b27) = Out.ok b28 := by
    rw [run_succ_ok im 0 b27, e28, ← hb28, run_zero]
  have rr30 : run im 1 (Out.ok b29) = Out.ok b30 := by
    rw [run_succ_ok im 0 b29, e30, ← hb30, run_zero]
  have rr31 : run im 1 (Out.ok b30) = Out.ok b31 := by
    rw [run_succ_ok im 0 b30, e31, ← hb31, run_zero]
  have rr32 : run im 1 (Out.ok b31) = Out.ok b32 := by
    rw [run_succ_ok im 0 b31, e32, ← hb32, run_zero]
  have rr33 : run im 1 (Out.ok b32) = Out.ok b33 := by
    rw [run_succ_ok im 0 b32, e33, ← hb33, run_zero]
  have rr34 : run im 1 (Out.ok b33) = Out.ok b34 := by
    rw [run_succ_ok im 0 b33, e34, ← hb34, run_zero]
  have rr35 : run im 1 (Out.ok b34) = Out.ok b35 := by
    rw [run_succ_ok im 0 b34, e35, ← hb35, run_zero]
  have hb35r4 : b35.reg 4 = some sp := by
    rw [gb35 4, gb34 4 (by decide)]; exact hb33r4
  have hb35r7 : b35.reg 7 = some w := by
    rw [gb35 7, gb34 7 (by decide), gb33 7 (by decide) (by decide),
      gb32 7 (by decide), gb31 7, gb29 7 (by decide) (by decide),
      gb28 7 (by decide), gb27 7 (by decide), gb26 7,
      gb24 7 (by decide) (by decide), gb23 7 (by decide),
      gb22 7 (by decide), gb21 7 (by decide) (by decide),
      gb20 7 (by decide), gb19 7 (by decide) (by decide),
      gb18 7 (by decide), gb17 7 (by decide) (by decide),
      gb16 7 (by decide), gb15 7 (by decide) (by decide),
      gb14 7 (by decide), gb13 7 (by decide), gb12 7 (by decide)]
    exact hb11r7
  refine ⟨1 + (1 + (1 + (1 + (1 + (k6 + (1 + (1 + (1 + (1 + (k11 + (1 + (1 + (1 + (k15 + (1 + (1 + (1 + (k19 + (1 + (1 + (1 + (1 + (k24 + (1 + (1 + (1 + (1 + (k29 + (1 + (1 + (1 + (1 + (1 + (1)))))))))))))))))))))))))))))))))), b35, ?_, ?_, ?_, hfr35, hw35, hlen35⟩
  · rw [run_add im 1 _, rr1, run_add im 1 _, rr2, run_add im 1 _, rr3, run_add im 1 _, rr4, run_add im 1 _, rr5, run_add im k6 _, hrun6, run_add im 1 _, rr7, run_add im 1 _, rr8, run_add im 1 _, rr9, run_add im 1 _, rr10, run_add im k11 _, hrun11, run_add im 1 _, rr12, run_add im 1 _, rr13, run_add im 1 _, rr14, run_add im k15 _, hrun15, run_add im 1 _, rr16, run_add im 1 _, rr17, run_add im 1 _, rr18, run_add im k19 _, hrun19, run_add im 1 _, rr20, run_add im 1 _, rr21, run_add im 1 _, rr22, run_add im 1 _, rr23, run_add im k24 _, hrun24, run_add im 1 _, rr25, run_add im 1 _, rr26, run_add im 1 _, rr27, run_add im 1 _, rr28, run_add im k29 _, hrun29, run_add im 1 _, rr30, run_add im 1 _, rr31, run_add im 1 _, rr32, run_add im 1 _, rr33, run_add im 1 _, rr34, rr35]
  · refine ⟨hb35rip, hb35r7, ?_, ?_, hb35r4, (fldOf b35 hfr35).1,
      (fldOf b35 hfr35).2.1, (fldOf b35 hfr35).2.2, layOf b35 hfr35,
      (by rw [hlen35]; exact h12len), ?_⟩
    · rw [gb35 6, gb34 6 (by decide), hb33r6]
      have e1 : wd.ann.length - rest'.length =
          (wd.ann.length - (rest'.length + 1)) + 1 := by
        rw [← Nat.sub_sub]
        exact (Nat.sub_add_cancel (Nat.sub_pos_of_lt hlen)).symm
      rw [e1, Nat.mul_add, Nat.mul_one, Nat.add_assoc]
    · rw [gb35 1]; exact hb34r1
    · exact StackRoom_of_wshape hw35 (by rw [hb35r4, hr4]) hroom
  · intro q h0 h1 h2 h4 h6 h8 h9 h12
    rw [gb35 q, gb34 q h1, gb33 q h1 h4, gb32 q h6, gb31 q, gb29 q h0 h4,
      gb28 q h2, gb27 q h2, gb26 q, gb24 q h0 h4, gb23 q h2, gb22 q h2,
      gb21 q h6 h4, gb20 q h9, gb19 q h0 h4, gb18 q h2, gb17 q h2 h4,
      gb16 q h8, gb15 q h0 h4, gb14 q h2, gb13 q h2, gb12 q h4,
      gb11 q h0 h1 h2 h4, gb10 q, gb8 q h2, gb7 q h12, gb6 q h0 h4,
      gb5 q h2]
    exact gb4 q h4

/-- The whole `annval` loop: if `annOk wd` holds, validation of every
remaining entry succeeds and control reaches `codewalk` with the ann
cursor at the end of the array. Induction on the remaining entries. -/
theorem annval_loop {im : Nat → Option (Ins × Nat)}
    (ac : AllCode im) (oc : OuterCode im)
    {wd : Sporos.Word} {w p sp : Nat} (st : AnnStatic wd w p sp)
    (hann : Sporos.annOk wd = true) :
    ∀ (rest : List (Nat × Height)) (pre : List (Nat × Height)),
      wd.ann = pre ++ rest → ∀ s, AnnSt wd w p sp rest s →
      ∃ k s', run im k (Out.ok s) = Out.ok s' ∧ s'.rip = aCodewalk ∧
        s'.reg 7 = some w ∧ s'.reg 4 = some sp ∧
        s'.reg 6 = some (p + 12 * wd.ann.length) ∧ s'.reg 1 = some 0 ∧
        rd64 s'.mem (w + 24) = some wd.code.length ∧
        rd64 s'.mem (w + 32) = some wd.ann.length ∧
        rd64 s'.mem (w + 48) = some p ∧
        AnnLayout s' wd.ann p ∧ StackRoom s' 6 ∧ 12 < s'.regs.length ∧
        (∀ q, q ≠ 0 → q ≠ 1 → q ≠ 2 → q ≠ 4 → q ≠ 6 → q ≠ 8 → q ≠ 9 →
          q ≠ 12 → s'.reg q = s.reg q) ∧
        (∀ x n, x + n ≤ sp - 48 ∨ sp ≤ x → s'.mem.rdN x n = s.mem.rdN x n) ∧
        wshape s'.mem = wshape s.mem ∧ s'.regs.length = s.regs.length := by
  intro rest
  induction rest with
  | nil =>
    intro pre happ s hs
    obtain ⟨k, s', hrun, hrip', hmem', hreg', hlen'⟩ := annval_done oc s hs
    obtain ⟨hrip, hr7, hr6, hr1, hr4, hf24, hf32, hf48, hlay, h12len,
      hroom⟩ := hs
    have hr6' : s'.reg 6 = some (p + 12 * wd.ann.length) := by
      rw [hreg' 6, hr6]
      simp
    refine ⟨k, s', hrun, hrip', by rw [hreg' 7]; exact hr7,
      by rw [hreg' 4]; exact hr4, hr6', by rw [hreg' 1]; exact hr1,
      by rw [hmem']; exact hf24, by rw [hmem']; exact hf32,
      by rw [hmem']; exact hf48, AnnLayout_of_mem hmem' hlay,
      StackRoom_of_mem hmem' (by rw [hreg' 4]) hroom,
      by omega, ?_, ?_, by rw [hmem'], hlen'⟩
    · intro q _ _ _ _ _ _ _ _
      exact hreg' q
    · intro x n _
      rw [hmem']
  | cons th rest' ih =>
    obtain ⟨t, h⟩ := th
    intro pre happ s hs
    have hs0 := hs
    obtain ⟨hrip, hr7, hr6, hr1, hr4, hf24, hf32, hf48, hlay, h12len,
      hroom⟩ := hs
    have hlenann : wd.ann.length = pre.length + (rest'.length + 1) := by
      rw [happ, List.length_append, List.length_cons]
    have hmem : (t, h) ∈ wd.ann := by
      rw [happ]
      exact List.mem_append_right _ List.mem_cons_self
    have hpt : (decide (t < wd.code.length) &&
        (wd.ann.lookup t == some h)) = true :=
      List.all_eq_true.mp hann (t, h) hmem
    obtain ⟨hpt1, hpt2⟩ : decide (t < wd.code.length) = true ∧
        (wd.ann.lookup t == some h) = true := by simpa using hpt
    have htlt : t < wd.code.length := of_decide_eq_true hpt1
    have hlk : wd.ann.lookup t = some h := eq_of_beq hpt2
    have hi0 : pre.length < wd.ann.length := by omega
    obtain ⟨hat, htW⟩ := hlay pre.length hi0
    have hget? : wd.ann[pre.length]? = some (t, h) := by
      rw [happ, List.getElem?_append_right (by simp)]
      simp
    have hget : wd.ann[pre.length]'hi0 = (t, h) :=
      Option.some.inj ((List.getElem?_eq_getElem hi0).symm.trans hget?)
    have hcur : AnnAt s (p + 12 * (wd.ann.length - (rest'.length + 1)))
        t h.d h.a := by
      have e : p + 12 * (wd.ann.length - (rest'.length + 1)) =
          p + 12 * pre.length := by
        congr 1
        congr 1
        omega
      rw [e]
      rw [hget] at hat
      exact hat
    have htW' : t < W := by
      rw [hget] at htW
      exact htW
    obtain ⟨k1, s1, hrun1, hs1, gfr1, hfr1, hw1, hlen1⟩ :=
      annval_step ac oc st (by omega) htlt hlk s hs0 hcur htW'
    have happ' : wd.ann = (pre ++ [(t, h)]) ++ rest' := by
      rw [happ, List.append_assoc, List.singleton_append]
    obtain ⟨k2, s2, hrun2, hrip2, hr72, hr42, hr62, hr12, hf242, hf322,
      hf482, hlay2, hroom2, h12len2, gfr2, hfr2, hw2, hlen2⟩ :=
      ih (pre ++ [(t, h)]) happ' s1 hs1
    refine ⟨k1 + k2, s2, by rw [run_add im k1 k2, hrun1, hrun2], hrip2, hr72,
      hr42, hr62, hr12, hf242, hf322, hf482, hlay2, hroom2, h12len2, ?_, ?_,
      hw2.trans hw1, hlen2.trans hlen1⟩
    · intro q h0 h1 h2 h4 h6 h8 h9 h12
      exact (gfr2 q h0 h1 h2 h4 h6 h8 h9 h12).trans
        (gfr1 q h0 h1 h2 h4 h6 h8 h9 h12)
    · intro x n hx
      exact (hfr2 x n hx).trans (hfr1 x n hx)


/-- `codewalk`: set the cursor and initial height, check an ann entry
at 0 against `(ins, 0)`, and land at `walk_top` in the walk invariant
for the whole word. Both annotation cases. -/
theorem codewalk_entry {im : Nat → Option (Ins × Nat)}
    (ac : AllCode im) (oc : OuterCode im)
    {pr : Prog} {wd : Sporos.Word} {w p sp tbl cp : Nat}
    (st : WalkStatic pr wd w p sp tbl) (hsp48 : 48 ≤ sp)
    (h0 : (match wd.ann.lookup 0 with
      | some a => a == ⟨wd.ins, 0⟩
      | none => true) = true)
    (hinsW : wd.ins < W) :
    ∀ s, s.rip = aCodewalk → s.reg 7 = some w → s.reg 4 = some sp →
      s.reg 1 = some 0 → (∃ v, s.reg 9 = some v) →
      (∃ v, s.reg 12 = some v) →
      s.reg 13 = some pr.words.length → s.reg 14 = some tbl →
      16 ≤ s.regs.length →
      rd64 s.mem w = some wd.ins → rd64 s.mem (w + 8) = some wd.outs →
      rd64 s.mem (w + 16) = some wd.locals →
      rd64 s.mem (w + 24) = some wd.code.length →
      rd64 s.mem (w + 32) = some wd.ann.length →
      rd64 s.mem (w + 40) = some cp → rd64 s.mem (w + 48) = some p →
      AnnLayout s wd.ann p → StackRoom s 6 → EffTable s →
      CodeSeq s.mem cp wd.code →
      (∀ n (hn : n < pr.words.length),
        rd64 s.mem (tbl + n * 64) = some (pr.words[n]).ins ∧
        rd64 s.mem (tbl + n * 64 + 8) = some (pr.words[n]).outs) →
      (cp + codeBytes wd.code ≤ sp - 32 ∨ sp ≤ cp) →
      cp + codeBytes wd.code < W →
      (∀ x ∈ wd.code, ImmOk x) → hBnd pr wd wd.code 0 ⟨wd.ins, 0⟩ →
      ∃ k s', run im k (Out.ok s) = Out.ok s' ∧
        WalkInv pr wd w p sp tbl s' wd.code 0 cp ⟨wd.ins, 0⟩ ⟨wd.ins, 0⟩ ∧
        (∀ q, q ≠ 0 → q ≠ 1 → q ≠ 2 → q ≠ 4 → q ≠ 6 → q ≠ 8 → q ≠ 9 →
          q ≠ 12 → s'.reg q = s.reg q) ∧
        (∀ x n, x + n ≤ sp - 32 ∨ sp ≤ x → s'.mem.rdN x n = s.mem.rdN x n) ∧
        wshape s'.mem = wshape s.mem ∧ s'.regs.length = s.regs.length := by
  intro s hrip hr7 hr4 hr1 h9ex h12ex hr13 hr14 h16 hf0 hf8 hf16 hf24 hf32
    hf40 hf48 hlay hroom heff hcsq hwords hdisjC hcbound himm hhb
  have hdisjw := st.disjw
  have hdisjp0 := st.disjp
  have hpann := st.pann
  have hw56 := st.w56
  have hspW := st.spW
  -- assembling the walk invariant from a frame against `s`
  have mkInv : ∀ u : X86Int.St, u.rip = aWalk → u.reg 8 = some wd.ins →
      u.reg 9 = some 0 → u.reg 6 = some cp → u.reg 4 = some sp →
      u.reg 7 = some w → u.reg 1 = some 0 →
      u.reg 13 = some pr.words.length → u.reg 14 = some tbl →
      u.regs.length = s.regs.length →
      (∀ x n, x + n ≤ sp - 32 ∨ sp ≤ x → u.mem.rdN x n = s.mem.rdN x n) →
      wshape u.mem = wshape s.mem →
      WalkInv pr wd w p sp tbl u wd.code 0 cp ⟨wd.ins, 0⟩ ⟨wd.ins, 0⟩ := by
    intro u hurip hu8 hu9 hu6 hu4 hu7 hu1 hu13 hu14 hulen hufr huw
    have hlink : (⟨wd.ins, 0⟩ : Height) =
        (wd.ann.lookup 0).getD ⟨wd.ins, 0⟩ := by
      cases hl : wd.ann.lookup 0 with
      | none => rfl
      | some a =>
        have hbe : (a == ⟨wd.ins, 0⟩) = true := by
          rw [hl] at h0
          exact h0
        exact (eq_of_beq hbe).symm
    refine ⟨hurip, hu8, hu9, hu6, hu4, hu7, hu1, hu13, hu14,
      by rw [hulen]; exact h16, hlink, ?_, Nat.zero_add _,
      rd64_of_frame_of_disjw st.disjw hufr (by decide) hf24,
      rd64_of_frame_of_disjw st.disjw hufr (by decide) hf32,
      rd64_of_frame_of_disjw st.disjw hufr (by decide) hf48,
      rd64_of_frame_of_disjw st.disjw hufr (by decide) hf16,
      rd64_of_frame_of_disjw st.disjw hufr (by decide) hf8,
      AnnLayout_of_frame hlay (fun x k _ _ => hufr x k (by omega)),
      StackRoom_of_wshape huw (by rw [hu4, hr4])
        (StackRoom_weaken (StackRoom_weaken hroom)), ?_, ?_, hdisjC,
      hcbound, himm, hhb⟩
    · exact CodeSeq_of_frame hcsq (fun x k _ _ => hufr x k (by omega))
    · intro pp
      rw [hufr (aEff + 2 * pp.code) 2 (by
        have hc := Prim_code_lt pp
        cases st.disjEff with
        | inl hh => exact Or.inl (by omega)
        | inr hh => exact Or.inr (by omega))]
      exact heff pp
    · intro n hn
      obtain ⟨hb1, hb2⟩ := st.wordsB n hn
      refine ⟨?_, ?_⟩
      · show (u.mem.rdN (tbl + n * 64) 8).map leVal = _
        rw [hufr (tbl + n * 64) 8 (by omega)]
        exact (hwords n hn).1
      · show (u.mem.rdN (tbl + n * 64 + 8) 8).map leVal = _
        rw [hufr (tbl + n * 64 + 8) 8 (by omega)]
        exact (hwords n hn).2
  -- mov rsi, [rdi+40]
  have hea40 : ea w 40 = w + 40 := Nat.mod_eq_of_lt (by omega)
  have e1 : X86Int.step im s = .ok { s.set 6 (some cp) with rip := s.rip + 4 } :=
    step_ld (by rw [hrip]; exact oc.ck0) hr7 (by rw [hea40]; exact hf40)
  obtain ⟨b1, hb1⟩ : ∃ u : X86Int.St, u = { s.set 6 (some cp) with
    rip := s.rip + 4 } := ⟨_, rfl⟩
  have hb1rip : b1.rip = aCodewalk + 4 := by
    rw [hb1]; show s.rip + 4 = aCodewalk + 4; rw [hrip]
  have gb1 : ∀ q, q ≠ 6 → b1.reg q = s.reg q := by
    intro q hq
    rw [hb1]
    show (s.set 6 (some cp)).reg q = s.reg q
    rw [reg_set_other hq]
  have hb1r6 : b1.reg 6 = some cp := by
    rw [hb1]
    show (s.set 6 (some cp)).reg 6 = some cp
    rw [reg_set_self (by omega)]
  have hb1mem : b1.mem = s.mem := by rw [hb1]; exact rfl
  have hlen1 : b1.regs.length = s.regs.length := by
    rw [hb1]
    show (s.set 6 (some cp)).regs.length = s.regs.length
    exact List.length_set
  -- xor ecx, ecx
  have hb1r1 : b1.reg 1 = some 0 := by rw [gb1 1 (by decide)]; exact hr1
  have e2 : X86Int.step im b1 = .ok { b1.set 1 (some (aluF .xor 0 0).1) with
      fl := (aluF .xor 0 0).2, rip := b1.rip + 2 } :=
    step_alu_xor (by rw [hb1rip]; exact oc.ck1) hb1r1 hb1r1
  obtain ⟨b2, hb2⟩ : ∃ u : X86Int.St, u = { b1.set 1 (some (aluF .xor 0 0).1)
      with fl := (aluF .xor 0 0).2, rip := b1.rip + 2 } := ⟨_, rfl⟩
  have hb2rip : b2.rip = aCodewalk + 6 := by
    rw [hb2]; show b1.rip + 2 = aCodewalk + 6; rw [hb1rip]
  have gb2 : ∀ q, q ≠ 1 → b2.reg q = b1.reg q := by
    intro q hq
    rw [hb2]
    show (b1.set 1 (some (aluF .xor 0 0).1)).reg q = b1.reg q
    rw [reg_set_other hq]
  have hb2r1 : b2.reg 1 = some 0 := by
    rw [hb2]
    show (b1.set 1 (some (aluF .xor 0 0).1)).reg 1 = some 0
    rw [reg_set_self (by omega), aluF_xor_fst_self]
  have hb2mem : b2.mem = s.mem := by
    rw [hb2]; show b1.mem = s.mem; exact hb1mem
  have hlen2 : b2.regs.length = s.regs.length := by
    rw [hb2]
    show (b1.set 1 (some (aluF .xor 0 0).1)).regs.length = s.regs.length
    exact List.length_set.trans hlen1
  -- mov r8, [rdi]
  have hb2r7 : b2.reg 7 = some w := by
    rw [gb2 7 (by decide), gb1 7 (by decide)]; exact hr7
  have hea0 : ea w 0 = w := by
    unfold ea
    rw [Nat.add_zero]
    exact Nat.mod_eq_of_lt (by omega)
  have e3 : X86Int.step im b2 = .ok { b2.set 8 (some wd.ins) with
      rip := b2.rip + 3 } :=
    step_ld (by rw [hb2rip]; exact oc.ck2) hb2r7
      (by rw [hea0, hb2mem]; exact hf0)
  obtain ⟨b3, hb3⟩ : ∃ u : X86Int.St, u = { b2.set 8 (some wd.ins) with
    rip := b2.rip + 3 } := ⟨_, rfl⟩
  have hb3rip : b3.rip = aCodewalk + 9 := by
    rw [hb3]; show b2.rip + 3 = aCodewalk + 9; rw [hb2rip]
  have gb3 : ∀ q, q ≠ 8 → b3.reg q = b2.reg q := by
    intro q hq
    rw [hb3]
    show (b2.set 8 (some wd.ins)).reg q = b2.reg q
    rw [reg_set_other hq]
  have hb3r8 : b3.reg 8 = some wd.ins := by
    rw [hb3]
    show (b2.set 8 (some wd.ins)).reg 8 = some wd.ins
    rw [reg_set_self (by omega)]
  have hb3mem : b3.mem = s.mem := by
    rw [hb3]; show b2.mem = s.mem; exact hb2mem
  have hlen3 : b3.regs.length = s.regs.length := by
    rw [hb3]
    show (b2.set 8 (some wd.ins)).regs.length = s.regs.length
    exact List.length_set.trans hlen2
  -- xor r9d, r9d
  obtain ⟨v9, hb3r9⟩ : ∃ v, b3.reg 9 = some v := by
    obtain ⟨v, hv⟩ := h9ex
    refine ⟨v, ?_⟩
    rw [gb3 9 (by decide), gb2 9 (by decide), gb1 9 (by decide)]
    exact hv
  have e4 : X86Int.step im b3 = .ok { b3.set 9 (some (aluF .xor v9 v9).1) with
      fl := (aluF .xor v9 v9).2, rip := b3.rip + 3 } :=
    step_alu_xor (by rw [hb3rip]; exact oc.ck3) hb3r9 hb3r9
  obtain ⟨b4, hb4⟩ : ∃ u : X86Int.St, u = { b3.set 9 (some (aluF .xor v9 v9).1)
      with fl := (aluF .xor v9 v9).2, rip := b3.rip + 3 } := ⟨_, rfl⟩
  have hb4rip : b4.rip = aCodewalk + 12 := by
    rw [hb4]; show b3.rip + 3 = aCodewalk + 12; rw [hb3rip]
  have gb4 : ∀ q, q ≠ 9 → b4.reg q = b3.reg q := by
    intro q hq
    rw [hb4]
    show (b3.set 9 (some (aluF .xor v9 v9).1)).reg q = b3.reg q
    rw [reg_set_other hq]
  have hb4r9 : b4.reg 9 = some 0 := by
    rw [hb4]
    show (b3.set 9 (some (aluF .xor v9 v9).1)).reg 9 = some 0
    rw [reg_set_self (by omega), aluF_xor_fst_self]
  have hb4mem : b4.mem = s.mem := by
    rw [hb4]; show b3.mem = s.mem; exact hb3mem
  have hlen4 : b4.regs.length = s.regs.length := by
    rw [hb4]
    show (b3.set 9 (some (aluF .xor v9 v9).1)).regs.length = s.regs.length
    exact List.length_set.trans hlen3
  -- xor r12d, r12d
  obtain ⟨v12, hb4r12⟩ : ∃ v, b4.reg 12 = some v := by
    obtain ⟨v, hv⟩ := h12ex
    exact ⟨v, by rw [gb4 12 (by decide), gb3 12 (by decide), gb2 12 (by decide),
      gb1 12 (by decide)]; exact hv⟩
  have e5 : X86Int.step im b4 = .ok { b4.set 12 (some (aluF .xor v12 v12).1)
      with fl := (aluF .xor v12 v12).2, rip := b4.rip + 3 } :=
    step_alu_xor (by rw [hb4rip]; exact oc.ck4) hb4r12 hb4r12
  obtain ⟨b5, hb5⟩ : ∃ u : X86Int.St, u =
    { b4.set 12 (some (aluF .xor v12 v12).1) with
      fl := (aluF .xor v12 v12).2, rip := b4.rip + 3 } := ⟨_, rfl⟩
  have hb5rip : b5.rip = aCodewalk + 15 := by
    rw [hb5]; show b4.rip + 3 = aCodewalk + 15; rw [hb4rip]
  have gb5 : ∀ q, q ≠ 12 → b5.reg q = b4.reg q := by
    intro q hq
    rw [hb5]
    show (b4.set 12 (some (aluF .xor v12 v12).1)).reg q = b4.reg q
    rw [reg_set_other hq]
  have hb5r12 : b5.reg 12 = some 0 := by
    rw [hb5]
    show (b4.set 12 (some (aluF .xor v12 v12).1)).reg 12 = some 0
    rw [reg_set_self (by omega), aluF_xor_fst_self]
  have hb5mem : b5.mem = s.mem := by
    rw [hb5]; show b4.mem = s.mem; exact hb4mem
  have hlen5 : b5.regs.length = s.regs.length := by
    rw [hb5]
    show (b4.set 12 (some (aluF .xor v12 v12).1)).regs.length =
      s.regs.length
    exact List.length_set.trans hlen4
  have hb5r7 : b5.reg 7 = some w := by
    rw [gb5 7 (by decide), gb4 7 (by decide), gb3 7 (by decide)]; exact hb2r7
  have hb5r4 : b5.reg 4 = some sp := by
    rw [gb5 4 (by decide), gb4 4 (by decide), gb3 4 (by decide),
      gb2 4 (by decide), gb1 4 (by decide)]; exact hr4
  have hb5r1 : b5.reg 1 = some 0 := by
    rw [gb5 1 (by decide), gb4 1 (by decide), gb3 1 (by decide)]; exact hb2r1
  have hb5r8 : b5.reg 8 = some wd.ins := by
    rw [gb5 8 (by decide), gb4 8 (by decide)]; exact hb3r8
  have hb5r9 : b5.reg 9 = some 0 := by
    rw [gb5 9 (by decide)]; exact hb4r9
  have hb5r6 : b5.reg 6 = some cp := by
    rw [gb5 6 (by decide), gb4 6 (by decide), gb3 6 (by decide),
      gb2 6 (by decide)]; exact hb1r6
  have hb5r13 : b5.reg 13 = some pr.words.length := by
    rw [gb5 13 (by decide), gb4 13 (by decide), gb3 13 (by decide),
      gb2 13 (by decide), gb1 13 (by decide)]; exact hr13
  have hb5r14 : b5.reg 14 = some tbl := by
    rw [gb5 14 (by decide), gb4 14 (by decide), gb3 14 (by decide),
      gb2 14 (by decide), gb1 14 (by decide)]; exact hr14
  -- call ANNL at 0
  obtain ⟨k6, b6, hrun6, hb6rip, hpost, hb6r1, hb6r4, hb6r12, hb6r7, hlay6,
    gb6, hfr6seg, hex0, hloc6, hw6seg, hlen6⟩ :=
    CallANNL ac.annl ac.l32 (by decide) oc.ck5 (by decide) (by decide) st.p0
      st.w56 st.pann (by omega) st.spW b5 hb5rip hb5r12 hb5r7 hb5r4 hb5r1
      (by rw [hb5mem]; exact hf48) (by rw [hb5mem]; exact hf32)
      (AnnLayout_of_mem hb5mem hlay)
      (StackRoom_of_mem hb5mem (by rw [hb5r4, hr4])
        (StackRoom_weaken (StackRoom_weaken hroom)))
      st.disjw st.disjp
  have hfr6 : ∀ x n, x + n ≤ sp - 32 ∨ sp ≤ x →
      b6.mem.rdN x n = s.mem.rdN x n := by
    intro x n hx
    rw [hfr6seg x n hx, hb5mem]
  have hw6 : wshape b6.mem = wshape s.mem := by rw [hw6seg, hb5mem]
  have hlen6s : b6.regs.length = s.regs.length := hlen6.trans hlen5
  have hb6r8 : b6.reg 8 = some wd.ins := by
    rw [gb6 8 (by decide) (by decide) (by decide) (by decide)]; exact hb5r8
  have hb6r9 : b6.reg 9 = some 0 := by
    rw [gb6 9 (by decide) (by decide) (by decide) (by decide)]; exact hb5r9
  have hb6r6 : b6.reg 6 = some cp := by
    rw [gb6 6 (by decide) (by decide) (by decide) (by decide)]; exact hb5r6
  have hb6r13 : b6.reg 13 = some pr.words.length := by
    rw [gb6 13 (by decide) (by decide) (by decide) (by decide)]; exact hb5r13
  have hb6r14 : b6.reg 14 = some tbl := by
    rw [gb6 14 (by decide) (by decide) (by decide) (by decide)]; exact hb5r14
  have hb6r1' : b6.reg 1 = some 0 := hb6r1
  obtain ⟨e, hb6r0⟩ := hex0
  have rr1 : run im 1 (Out.ok s) = Out.ok b1 := by
    rw [run_succ_ok im 0 s, e1, ← hb1, run_zero]
  have rr2 : run im 1 (Out.ok b1) = Out.ok b2 := by
    rw [run_succ_ok im 0 b1, e2, ← hb2, run_zero]
  have rr3 : run im 1 (Out.ok b2) = Out.ok b3 := by
    rw [run_succ_ok im 0 b2, e3, ← hb3, run_zero]
  have rr4 : run im 1 (Out.ok b3) = Out.ok b4 := by
    rw [run_succ_ok im 0 b3, e4, ← hb4, run_zero]
  have rr5 : run im 1 (Out.ok b4) = Out.ok b5 := by
    rw [run_succ_ok im 0 b4, e5, ← hb5, run_zero]
  have hrun06 : run im (1 + (1 + (1 + (1 + (1 + k6))))) (Out.ok s) =
      Out.ok b6 := by
    rw [run_add im 1 _, rr1, run_add im 1 _, rr2, run_add im 1 _, rr3,
      run_add im 1 _, rr4, run_add im 1 _, rr5, hrun6]
  have gfr06 : ∀ q, q ≠ 0 → q ≠ 1 → q ≠ 2 → q ≠ 4 → q ≠ 6 → q ≠ 8 → q ≠ 9 →
      q ≠ 12 → b6.reg q = s.reg q := by
    intro q h0 h1 h2 h4 h6 h8 h9 h12
    rw [gb6 q h0 h1 h2 h4, gb5 q h12, gb4 q h9, gb3 q h8, gb2 q h1,
      gb1 q h6]
  cases hl0 : wd.ann.lookup 0 with
  | none =>
    have he0 : e = 0 := by
      cases hpost e hb6r0 with
      | inl hb => exact hb.1
      | inr hb =>
        obtain ⟨d1, a1, -, -, hlk1⟩ := hb
        rw [hl0] at hlk1
        simp at hlk1
    -- test rax, rax ; jz walk_top
    have hb6r0' : b6.reg 0 = some 0 := by rw [hb6r0, he0]
    have e7 : X86Int.step im b6 =
        .ok { b6 with fl := (aluF .and 0 0).2, rip := b6.rip + 3 } :=
      step_test (by rw [hb6rip]; exact oc.ck6) hb6r0' hb6r0'
    obtain ⟨b7, hb7⟩ : ∃ u : X86Int.St, u =
      { b6 with fl := (aluF .and 0 0).2, rip := b6.rip + 3 } := ⟨_, rfl⟩
    have hb7rip : b7.rip = aCodewalk + 23 := by
      rw [hb7]; show b6.rip + 3 = aCodewalk + 23; rw [hb6rip]
    have hb7fl : X86Int.cond b7.fl 4 = some true := by
      have hz : b7.fl.zf = some true := by
        rw [hb7]
        show (aluF .and 0 0).2.zf = some true
        rw [aluF_zf, aluF_and_fst_self]
        exact rfl
      exact cond_cc4 hz
    have gb7 : ∀ q, b7.reg q = b6.reg q := fun q => by rw [hb7]; exact rfl
    have hb7mem : b7.mem = b6.mem := by rw [hb7]
    have e8 : X86Int.step im b7 = .ok { b7 with rip := aWalk } := by
      have hh := step_jcc (im := im) (s := b7) (by rw [hb7rip]; exact oc.ck7)
        hb7fl
      simpa using hh
    obtain ⟨b8, hb8⟩ : ∃ u : X86Int.St, u = { b7 with rip := aWalk } :=
      ⟨_, rfl⟩
    have rr7 : run im 1 (Out.ok b6) = Out.ok b7 := by
      rw [run_succ_ok im 0 b6, e7, ← hb7, run_zero]
    have rr8 : run im 1 (Out.ok b7) = Out.ok b8 := by
      rw [run_succ_ok im 0 b7, e8, ← hb8, run_zero]
    have hb8r : ∀ q, b8.reg q = b6.reg q := fun q => by
      rw [hb8]; show b7.reg q = b6.reg q; exact gb7 q
    have hb8mem : b8.mem = b6.mem := by rw [hb8]; exact hb7mem
    have hfr8 : ∀ x n, x + n ≤ sp - 32 ∨ sp ≤ x →
        b8.mem.rdN x n = s.mem.rdN x n := by
      intro x n hx; rw [hb8mem]; exact hfr6 x n hx
    refine ⟨1 + (1 + (1 + (1 + (1 + k6)))) + (1 + 1), b8,
      by rw [run_add im _ (1 + 1), hrun06, run_add im 1 1, rr7, rr8],
      mkInv b8 (by rw [hb8]) (by rw [hb8r 8]; exact hb6r8)
        (by rw [hb8r 9]; exact hb6r9) (by rw [hb8r 6]; exact hb6r6)
        (by rw [hb8r 4]; exact hb6r4) (by rw [hb8r 7]; exact hb6r7)
        (by rw [hb8r 1]; exact hb6r1') (by rw [hb8r 13]; exact hb6r13)
        (by rw [hb8r 14]; exact hb6r14)
        (by rw [hb8, hb7]; exact hlen6s) hfr8 (by rw [hb8mem]; exact hw6),
      ?_, hfr8, by rw [hb8mem]; exact hw6, by rw [hb8, hb7]; exact hlen6s⟩
    intro q h0 h1 h2 h4 h6 h8 h9 h12
    rw [hb8r q]
    exact gfr06 q h0 h1 h2 h4 h6 h8 h9 h12
  | some a =>
    have haa : a = ⟨wd.ins, 0⟩ := by
      have hbe : (a == ⟨wd.ins, 0⟩) = true := by
        rw [hl0] at h0
        exact h0
      exact eq_of_beq hbe
    obtain ⟨he0, hat6⟩ : e ≠ 0 ∧ AnnAt b6 e 0 wd.ins 0 := by
      cases hpost e hb6r0 with
      | inl hb =>
        rw [hb.2] at hl0
        simp at hl0
      | inr hb =>
        obtain ⟨d1, a1, hne, hat, hlk1⟩ := hb
        have hdeq : d1 = wd.ins := by
          have hhh : (⟨d1, a1⟩ : Height) = ⟨wd.ins, 0⟩ :=
            Option.some.inj (hlk1.symm.trans (hl0.trans (by rw [haa])))
          exact congrArg Height.d hhh
        have haeq : a1 = 0 := by
          have hhh : (⟨d1, a1⟩ : Height) = ⟨wd.ins, 0⟩ :=
            Option.some.inj (hlk1.symm.trans (hl0.trans (by rw [haa])))
          exact congrArg Height.a hhh
        exact ⟨hne, by rwa [hdeq, haeq] at hat⟩
    obtain ⟨hel, heh⟩ := hloc6 e hb6r0 he0
    -- test rax, rax ; jz not taken
    have e7 : X86Int.step im b6 =
        .ok { b6 with fl := (aluF .and e e).2, rip := b6.rip + 3 } :=
      step_test (by rw [hb6rip]; exact oc.ck6) hb6r0 hb6r0
    obtain ⟨b7, hb7⟩ : ∃ u : X86Int.St, u =
      { b6 with fl := (aluF .and e e).2, rip := b6.rip + 3 } := ⟨_, rfl⟩
    have hb7rip : b7.rip = aCodewalk + 23 := by
      rw [hb7]; show b6.rip + 3 = aCodewalk + 23; rw [hb6rip]
    have hb7fl : X86Int.cond b7.fl 4 = some false := by
      have hz : b7.fl.zf = some false := by
        rw [hb7]
        show (aluF .and e e).2.zf = some false
        rw [aluF_zf, aluF_and_fst_self]
        simp [he0]
      exact cond_cc4 hz
    have gb7 : ∀ q, b7.reg q = b6.reg q := fun q => by rw [hb7]; exact rfl
    have hb7mem : b7.mem = b6.mem := by rw [hb7]
    have e8 : X86Int.step im b7 = .ok { b7 with rip := aCodewalk + 25 } := by
      have hh := step_jcc (im := im) (s := b7) (by rw [hb7rip]; exact oc.ck7)
        hb7fl
      simpa [hb7rip] using hh
    obtain ⟨b8, hb8⟩ : ∃ u : X86Int.St, u = { b7 with rip := aCodewalk + 25 } :=
      ⟨_, rfl⟩
    have hb8rip : b8.rip = aCodewalk + 25 := by rw [hb8]
    have gb8 : ∀ q, b8.reg q = b6.reg q := fun q => by
      rw [hb8]; show b7.reg q = b6.reg q; exact gb7 q
    have hb8mem : b8.mem = b6.mem := by rw [hb8]; exact hb7mem
    have hb8r0 : b8.reg 0 = some e := by rw [gb8 0]; exact hb6r0
    have hb8r4 : b8.reg 4 = some sp := by rw [gb8 4]; exact hb6r4
    have hlen8 : b8.regs.length = s.regs.length := by
      rw [hb8, hb7]; exact hlen6s
    have hfr8 : ∀ x n, x + n ≤ sp - 32 ∨ sp ≤ x →
        b8.mem.rdN x n = s.mem.rdN x n := by
      intro x n hx; rw [hb8mem]; exact hfr6 x n hx
    have hw8 : wshape b8.mem = wshape s.mem := by rw [hb8mem]; exact hw6
    have rr7 : run im 1 (Out.ok b6) = Out.ok b7 := by
      rw [run_succ_ok im 0 b6, e7, ← hb7, run_zero]
    have rr8 : run im 1 (Out.ok b7) = Out.ok b8 := by
      rw [run_succ_ok im 0 b7, e8, ← hb8, run_zero]
    have hdisjp := st.disjp
    have hadd : ∀ (x k : Nat), x + k < W →
        (aluF .add x (k % W)).1 = x + k := by
      intro x k hk
      rw [aluF_add_fst]
      have hkW : k % W = k := Nat.mod_eq_of_lt (by omega)
      rw [hkW]
      exact Nat.mod_eq_of_lt hk
    have hslot8 : (sp + W - 8) % W = sp - 8 := by
      rw [show sp + W - 8 = sp + W - 8 * 1 by omega, slot_sub (by omega) hspW]
    have hroom8 : StackRoom b8 1 :=
      StackRoom_shift (L := 0) hroom hr4 hb8r4 hw8 ⟨0, rfl, by omega⟩ hsp48 hspW
    obtain ⟨m9, hm9s⟩ := Option.isSome_iff_exists.mp (hroom8 sp hb8r4 0
      (by decide) e)
    have hm9 : wr64 b8.mem (sp - 8) e = some m9 := by
      have e' : (sp + W - 8 * (0 + 1)) % W = sp - 8 :=
        slot_sub (by omega) hspW
      rw [e'] at hm9s; exact hm9s
    have hm9p : wr64 b8.mem ((sp + W - 8) % W) e = some m9 := by
      rw [hslot8]; exact hm9
    have e9 : X86Int.step im b8 = .ok { b8.set 4 (some ((sp + W - 8) % W)) with
        mem := m9, rip := b8.rip + 1 } :=
      step_push (by rw [hb8rip]; exact oc.ck8) hb8r0 hb8r4 hm9p
    obtain ⟨b9, hb9⟩ : ∃ u : X86Int.St, u =
      { b8.set 4 (some ((sp + W - 8) % W)) with mem := m9, rip := b8.rip + 1 } :=
      ⟨_, rfl⟩
    have hb9rip : b9.rip = aCodewalk + 26 := by
      rw [hb9]; show b8.rip + 1 = aCodewalk + 26; rw [hb8rip]
    have hb9r4 : b9.reg 4 = some (sp - 8) := by
      rw [hb9]
      show (b8.set 4 (some ((sp + W - 8) % W))).reg 4 = some (sp - 8)
      rw [reg_set_self (by omega), hslot8]
    have gb9 : ∀ q, q ≠ 4 → b9.reg q = b8.reg q := by
      intro q hq
      rw [hb9]
      show (b8.set 4 (some ((sp + W - 8) % W))).reg q = b8.reg q
      rw [reg_set_other hq]
    have hb9mem : b9.mem = m9 := by rw [hb9]
    have hfr9 : ∀ x n, x + n ≤ sp - 32 ∨ sp ≤ x →
        b9.mem.rdN x n = s.mem.rdN x n := by
      intro x n hx
      rw [hb9mem, rdN_wr64_other hm9 (by omega)]
      exact hfr8 x n hx
    have hw9 : wshape b9.mem = wshape s.mem := by
      rw [hb9mem]
      exact (wrN_wshape _ _ _ hm9).trans hw8
    have hlen9 : b9.regs.length = s.regs.length := by
      rw [hb9]
      show (b8.set 4 (some ((sp + W - 8) % W))).regs.length = s.regs.length
      exact List.length_set.trans hlen8
    have hslot9 : rd64 b9.mem (sp - 8) = some e := by
      have h1 := rd64_wr64 hm9
      rw [Nat.mod_eq_of_lt (by omega)] at h1
      rw [hb9mem]; exact h1
    have hat9 : AnnAt b9 e 0 wd.ins 0 :=
      ⟨by rw [hb9mem, rdN_wr64_other hm9 (by omega), hb8mem]; exact hat6.1,
       by rw [hb9mem, rdN_wr64_other hm9 (by omega), hb8mem]; exact hat6.2.1,
       by rw [hb9mem, rdN_wr64_other hm9 (by omega), hb8mem]; exact hat6.2.2⟩
    have rr9 : run im 1 (Out.ok b8) = Out.ok b9 := by
      rw [run_succ_ok im 0 b8, e9, ← hb9, run_zero]
    -- mov rdx, rax ; add rdx, 4
    have hb9r0 : b9.reg 0 = some e := by rw [gb9 0 (by decide)]; exact hb8r0
    have e10 : X86Int.step im b9 = .ok { b9.set 2 (some e) with
        rip := b9.rip + 3 } :=
      step_mov (by rw [hb9rip]; exact oc.ck9) hb9r0
    obtain ⟨b10, hb10⟩ : ∃ u : X86Int.St, u = { b9.set 2 (some e) with
      rip := b9.rip + 3 } := ⟨_, rfl⟩
    have hb10rip : b10.rip = aCodewalk + 29 := by
      rw [hb10]; show b9.rip + 3 = aCodewalk + 29; rw [hb9rip]
    have gb10 : ∀ q, q ≠ 2 → b10.reg q = b9.reg q := by
      intro q hq
      rw [hb10]
      show (b9.set 2 (some e)).reg q = b9.reg q
      rw [reg_set_other hq]
    have hb10r2 : b10.reg 2 = some e := by
      rw [hb10]
      show (b9.set 2 (some e)).reg 2 = some e
      rw [reg_set_self (by omega)]
    have hb10mem : b10.mem = b9.mem := by rw [hb10]; exact rfl
    have rr10 : run im 1 (Out.ok b9) = Out.ok b10 := by
      rw [run_succ_ok im 0 b9, e10, ← hb10, run_zero]
    have e11 : X86Int.step im b10 = .ok { b10.set 2
        (some (aluF .add e (4 % W)).1) with fl := (aluF .add e (4 % W)).2, rip := b10.rip + 4 } :=
      step_aluI_add (by rw [hb10rip]; exact oc.ck10) hb10r2
    obtain ⟨b11, hb11⟩ : ∃ u : X86Int.St, u = { b10.set 2
        (some (aluF .add e (4 % W)).1) with fl := (aluF .add e (4 % W)).2, rip := b10.rip + 4 } := ⟨_, rfl⟩
    have hb11rip : b11.rip = aCodewalk + 33 := by
      rw [hb11]; show b10.rip + 4 = aCodewalk + 33; rw [hb10rip]
    have hb11r2 : b11.reg 2 = some (e + 4) := by
      rw [hb11]
      show (b10.set 2 (some (aluF .add e (4 % W)).1)).reg 2 = some (e + 4)
      rw [reg_set_self (by
        have : b10.regs.length = s.regs.length := by
          rw [hb10]
          show (b9.set 2 (some e)).regs.length = s.regs.length
          exact List.length_set.trans hlen9
        omega), hadd e 4 (by omega)]
    have gb11 : ∀ q, q ≠ 2 → b11.reg q = b10.reg q := by
      intro q hq
      rw [hb11]
      show (b10.set 2 (some (aluF .add e (4 % W)).1)).reg q = b10.reg q
      rw [reg_set_other hq]
    have hb11mem : b11.mem = b9.mem := by
      rw [hb11]; show b10.mem = b9.mem; exact hb10mem
    have hb11r4 : b11.reg 4 = some (sp - 8) := by
      rw [gb11 4 (by decide), gb10 4 (by decide)]; exact hb9r4
    have hfr11 : ∀ x n, x + n ≤ sp - 32 ∨ sp ≤ x →
        b11.mem.rdN x n = s.mem.rdN x n := by
      intro x n hx; rw [hb11mem]; exact hfr9 x n hx
    have hw11 : wshape b11.mem = wshape s.mem := by rw [hb11mem]; exact hw9
    have hlen11 : b11.regs.length = s.regs.length := by
      rw [hb11]
      show (b10.set 2 (some (aluF .add e (4 % W)).1)).regs.length =
        s.regs.length
      refine List.length_set.trans ?_
      rw [hb10]
      show (b9.set 2 (some e)).regs.length = s.regs.length
      exact List.length_set.trans hlen9
    have rr11 : run im 1 (Out.ok b10) = Out.ok b11 := by
      rw [run_succ_ok im 0 b10, e11, ← hb11, run_zero]
    -- call L32: rax = first d = ins
    obtain ⟨k12, b12, hrun12, hb12rip, hb12r0, hb12r4, gb12, hlen12, hfr12seg,
      hw12seg⟩ :=
      CallL32 ac.l32 ac.annl.dret oc.ck11 (by omega) (by omega) (by omega)
        (by decide) (by omega) b11 hb11rip hb11r2 hb11r4
        (by rw [hb11mem]; exact hat9.2.1)
        (StackRoom_shift hroom hr4 hb11r4 hw11 ⟨1, by omega, by omega⟩ hsp48
          hspW)
    have hfr12 : ∀ x n, x + n ≤ sp - 32 ∨ sp ≤ x →
        b12.mem.rdN x n = s.mem.rdN x n := by
      intro x n hx
      exact (hfr12seg x n (by omega)).trans (hfr11 x n hx)
    have hw12 : wshape b12.mem = wshape s.mem := hw12seg.trans hw11
    have hlen12s : b12.regs.length = s.regs.length := hlen12.trans hlen11
    have hslot12 : rd64 b12.mem (sp - 8) = some e := by
      unfold rd64
      rw [hfr12seg (sp - 8) 8 (by omega), hb11mem]
      exact hslot9
    have hat12 : AnnAt b12 e 0 wd.ins 0 :=
      ⟨by rw [hfr12seg e 4 (by omega), hb11mem]; exact hat9.1,
       by rw [hfr12seg (e + 4) 4 (by omega), hb11mem]; exact hat9.2.1,
       by rw [hfr12seg (e + 8) 4 (by omega), hb11mem]; exact hat9.2.2⟩
    have hb12r8 : b12.reg 8 = some wd.ins := by
      rw [gb12 8 (by decide) (by decide), gb11 8 (by decide),
        gb10 8 (by decide), gb9 8 (by decide), gb8 8]; exact hb6r8
    -- cmp rax, r8 ; jz taken
    have e13 : X86Int.step im b12 =
        .ok { b12 with fl := (aluF .sub wd.ins wd.ins).2, rip := b12.rip + 3 } :=
      step_cmp (by rw [hb12rip]; exact oc.ck12) hb12r0 hb12r8
    obtain ⟨b13, hb13⟩ : ∃ u : X86Int.St, u =
      { b12 with fl := (aluF .sub wd.ins wd.ins).2, rip := b12.rip + 3 } :=
      ⟨_, rfl⟩
    have hb13rip : b13.rip = aCodewalk + 41 := by
      rw [hb13]; show b12.rip + 3 = aCodewalk + 41; rw [hb12rip]
    have hb13fl : X86Int.cond b13.fl 4 = some true := by
      have hz : b13.fl.zf = some true := by
        rw [hb13]
        show (aluF .sub wd.ins wd.ins).2.zf = some true
        rw [aluF_zf, aluF_sub_fst, subW_eq_zero hinsW hinsW]; simp
      exact cond_cc4 hz
    have gb13 : ∀ q, b13.reg q = b12.reg q := fun q => by
      rw [hb13]; exact rfl
    have hb13mem : b13.mem = b12.mem := by rw [hb13]
    have rr13 : run im 1 (Out.ok b12) = Out.ok b13 := by
      rw [run_succ_ok im 0 b12, e13, ← hb13, run_zero]
    have e14 : X86Int.step im b13 = .ok { b13 with rip := aCodewalk + 48 } := by
      have hh := step_jcc (im := im) (s := b13)
        (by rw [hb13rip]; exact oc.ck13) hb13fl
      simpa using hh
    obtain ⟨b14, hb14⟩ : ∃ u : X86Int.St, u =
      { b13 with rip := aCodewalk + 48 } := ⟨_, rfl⟩
    have hb14rip : b14.rip = aCodewalk + 48 := by rw [hb14]
    have gb14 : ∀ q, b14.reg q = b12.reg q := fun q => by
      rw [hb14]; show b13.reg q = b12.reg q; exact gb13 q
    have hb14mem : b14.mem = b12.mem := by rw [hb14]; exact hb13mem
    have hb14r4 : b14.reg 4 = some (sp - 8) := by rw [gb14 4]; exact hb12r4
    have rr14 : run im 1 (Out.ok b13) = Out.ok b14 := by
      rw [run_succ_ok im 0 b13, e14, ← hb14, run_zero]
    -- pop rdx
    have hp8 : ((sp - 8) + 8) % W = sp := slot_add8 (by omega) hspW
    have e15 : X86Int.step im b14 = .ok
        { (b14.set 4 (some ((sp - 8 + 8) % W))).set 2 (some e) with
          rip := b14.rip + 1 } :=
      step_pop (by rw [hb14rip]; exact oc.ck14) hb14r4
        (by rw [hb14mem]; exact hslot12)
    obtain ⟨b15, hb15⟩ : ∃ u : X86Int.St, u =
      { (b14.set 4 (some ((sp - 8 + 8) % W))).set 2 (some e) with
        rip := b14.rip + 1 } := ⟨_, rfl⟩
    have hb15rip : b15.rip = aCodewalk + 49 := by
      rw [hb15]; show b14.rip + 1 = aCodewalk + 49; rw [hb14rip]
    have hb15r4 : b15.reg 4 = some sp := by
      rw [hb15]
      show ((b14.set 4 (some ((sp - 8 + 8) % W))).set 2 (some e)).reg 4 = some sp
      rw [reg_set_other (by decide),
        reg_set_self (show 4 < b14.regs.length from by
          have e : b14.regs.length = s.regs.length := by
            rw [hb14, hb13]; exact hlen12s
          omega), hp8]
    have hb15r2 : b15.reg 2 = some e := by
      rw [hb15]
      show ((b14.set 4 (some ((sp - 8 + 8) % W))).set 2 (some e)).reg 2 = some e
      rw [reg_set_self (by
        have ee : (b14.set 4 (some ((sp - 8 + 8) % W))).regs.length =
            s.regs.length := List.length_set.trans (by
              rw [hb14, hb13]; exact hlen12s)
        omega)]
    have gb15 : ∀ q, q ≠ 2 → q ≠ 4 → b15.reg q = b14.reg q := by
      intro q h2 h4
      rw [hb15]
      show ((b14.set 4 (some ((sp - 8 + 8) % W))).set 2 (some e)).reg q =
        b14.reg q
      rw [reg_set_other h2, reg_set_other h4]
    have hb15mem : b15.mem = b14.mem := by rw [hb15]; exact rfl
    have hlen15 : b15.regs.length = s.regs.length := by
      rw [hb15]
      show ((b14.set 4 (some ((sp - 8 + 8) % W))).set 2 (some e)).regs.length =
        s.regs.length
      refine List.length_set.trans (List.length_set.trans ?_)
      rw [hb14, hb13]; exact hlen12s
    have rr15 : run im 1 (Out.ok b14) = Out.ok b15 := by
      rw [run_succ_ok im 0 b14, e15, ← hb15, run_zero]
    -- add rdx, 8
    have e16 : X86Int.step im b15 = .ok { b15.set 2
        (some (aluF .add e (8 % W)).1) with fl := (aluF .add e (8 % W)).2, rip := b15.rip + 4 } :=
      step_aluI_add (by rw [hb15rip]; exact oc.ck15) hb15r2
    obtain ⟨b16, hb16⟩ : ∃ u : X86Int.St, u = { b15.set 2
        (some (aluF .add e (8 % W)).1) with fl := (aluF .add e (8 % W)).2, rip := b15.rip + 4 } := ⟨_, rfl⟩
    have hb16rip : b16.rip = aCodewalk + 53 := by
      rw [hb16]; show b15.rip + 4 = aCodewalk + 53; rw [hb15rip]
    have hb16r2 : b16.reg 2 = some (e + 8) := by
      rw [hb16]
      show (b15.set 2 (some (aluF .add e (8 % W)).1)).reg 2 = some (e + 8)
      rw [reg_set_self (by omega), hadd e 8 (by omega)]
    have gb16 : ∀ q, q ≠ 2 → b16.reg q = b15.reg q := by
      intro q hq
      rw [hb16]
      show (b15.set 2 (some (aluF .add e (8 % W)).1)).reg q = b15.reg q
      rw [reg_set_other hq]
    have hb16mem : b16.mem = b12.mem := by
      rw [hb16]; show b15.mem = b12.mem; rw [hb15mem]; exact hb14mem
    have hb16r4 : b16.reg 4 = some sp := by
      rw [gb16 4 (by decide)]; exact hb15r4
    have hfr16 : ∀ x n, x + n ≤ sp - 32 ∨ sp ≤ x →
        b16.mem.rdN x n = s.mem.rdN x n := by
      intro x n hx; rw [hb16mem]; exact hfr12 x n hx
    have hw16 : wshape b16.mem = wshape s.mem := by rw [hb16mem]; exact hw12
    have hlen16 : b16.regs.length = s.regs.length := by
      rw [hb16]
      show (b15.set 2 (some (aluF .add e (8 % W)).1)).regs.length =
        s.regs.length
      exact List.length_set.trans hlen15
    have hb16r9 : b16.reg 9 = some 0 := by
      rw [gb16 9 (by decide), gb15 9 (by decide) (by decide), gb14 9,
        gb12 9 (by decide) (by decide), gb11 9 (by decide),
        gb10 9 (by decide), gb9 9 (by decide), gb8 9]; exact hb6r9
    have rr16 : run im 1 (Out.ok b15) = Out.ok b16 := by
      rw [run_succ_ok im 0 b15, e16, ← hb16, run_zero]
    -- call L32: rax = first a = 0
    obtain ⟨k17, b17, hrun17, hb17rip, hb17r0, hb17r4, gb17, hlen17, hfr17seg,
      hw17seg⟩ :=
      CallL32 ac.l32 ac.annl.dret oc.ck16 (by omega) (by omega) (by omega)
        (by decide) (by omega) b16 hb16rip hb16r2 hb16r4
        (by rw [hb16mem]; exact hat12.2.2)
        (StackRoom_shift (L := 0) hroom hr4 hb16r4 hw16 ⟨0, rfl, by omega⟩ hsp48 hspW)
    have hfr17 : ∀ x n, x + n ≤ sp - 32 ∨ sp ≤ x →
        b17.mem.rdN x n = s.mem.rdN x n := by
      intro x n hx
      exact (hfr17seg x n (by omega)).trans (hfr16 x n hx)
    have hw17 : wshape b17.mem = wshape s.mem := hw17seg.trans hw16
    have hlen17s : b17.regs.length = s.regs.length := hlen17.trans hlen16
    have hb17r9 : b17.reg 9 = some 0 := by
      rw [gb17 9 (by decide) (by decide)]; exact hb16r9
    have hb17r : ∀ q, q ≠ 0 → q ≠ 1 → q ≠ 2 → q ≠ 4 → q ≠ 6 → q ≠ 8 →
        q ≠ 9 → q ≠ 12 → b17.reg q = s.reg q := by
      intro q h0 h1 h2 h4 h6 h8 h9 h12
      rw [gb17 q h0 h4, gb16 q h2, gb15 q h2 h4, gb14 q,
        gb12 q h0 h4, gb11 q h2, gb10 q h2, gb9 q h4, gb8 q]
      exact gfr06 q h0 h1 h2 h4 h6 h8 h9 h12
    have hb17r8 : b17.reg 8 = some wd.ins := by
      rw [gb17 8 (by decide) (by decide), gb16 8 (by decide),
        gb15 8 (by decide) (by decide), gb14 8]; exact hb12r8
    have hb17r6 : b17.reg 6 = some cp := by
      rw [gb17 6 (by decide) (by decide), gb16 6 (by decide),
        gb15 6 (by decide) (by decide), gb14 6, gb12 6 (by decide) (by decide),
        gb11 6 (by decide), gb10 6 (by decide), gb9 6 (by decide), gb8 6]
      exact hb6r6
    have hb17r7 : b17.reg 7 = some w := hb17r 7 (by decide) (by decide)
      (by decide) (by decide) (by decide) (by decide) (by decide)
      (by decide) ▸ hr7
    have hb17r13 : b17.reg 13 = some pr.words.length := hb17r 13 (by decide)
      (by decide) (by decide) (by decide) (by decide) (by decide)
      (by decide) (by decide) ▸ hr13
    have hb17r14 : b17.reg 14 = some tbl := hb17r 14 (by decide) (by decide)
      (by decide) (by decide) (by decide) (by decide) (by decide)
      (by decide) ▸ hr14
    have hb17r1 : b17.reg 1 = some 0 := by
      rw [gb17 1 (by decide) (by decide), gb16 1 (by decide),
        gb15 1 (by decide) (by decide), gb14 1, gb12 1 (by decide) (by decide),
        gb11 1 (by decide), gb10 1 (by decide), gb9 1 (by decide), gb8 1]
      exact hb6r1'
    -- cmp rax, r9 ; jz walk_top
    have e18 : X86Int.step im b17 =
        .ok { b17 with fl := (aluF .sub 0 0).2, rip := b17.rip + 3 } :=
      step_cmp (by rw [hb17rip]; exact oc.ck17) hb17r0 hb17r9
    obtain ⟨b18, hb18⟩ : ∃ u : X86Int.St, u =
      { b17 with fl := (aluF .sub 0 0).2, rip := b17.rip + 3 } := ⟨_, rfl⟩
    have hb18rip : b18.rip = aCodewalk + 61 := by
      rw [hb18]; show b17.rip + 3 = aCodewalk + 61; rw [hb17rip]
    have hb18fl : X86Int.cond b18.fl 4 = some true := by
      have hz : b18.fl.zf = some true := by
        rw [hb18]
        show (aluF .sub 0 0).2.zf = some true
        rw [aluF_zf, aluF_sub_fst, subW_eq_zero (by decide) (by decide)]; simp
      exact cond_cc4 hz
    have gb18 : ∀ q, b18.reg q = b17.reg q := fun q => by
      rw [hb18]; exact rfl
    have hb18mem : b18.mem = b17.mem := by rw [hb18]
    have rr18 : run im 1 (Out.ok b17) = Out.ok b18 := by
      rw [run_succ_ok im 0 b17, e18, ← hb18, run_zero]
    have e19 : X86Int.step im b18 = .ok { b18 with rip := aWalk } := by
      have hh := step_jcc (im := im) (s := b18)
        (by rw [hb18rip]; exact oc.ck18) hb18fl
      simpa using hh
    obtain ⟨b19, hb19⟩ : ∃ u : X86Int.St, u = { b18 with rip := aWalk } :=
      ⟨_, rfl⟩
    have rr19 : run im 1 (Out.ok b18) = Out.ok b19 := by
      rw [run_succ_ok im 0 b18, e19, ← hb19, run_zero]
    have hb19r : ∀ q, b19.reg q = b17.reg q := fun q => by
      rw [hb19]; show b18.reg q = b17.reg q; exact gb18 q
    have hb19mem : b19.mem = b17.mem := by rw [hb19]; exact hb18mem
    have hfr19 : ∀ x n, x + n ≤ sp - 32 ∨ sp ≤ x →
        b19.mem.rdN x n = s.mem.rdN x n := by
      intro x n hx; rw [hb19mem]; exact hfr17 x n hx
    have hlen19 : b19.regs.length = s.regs.length := by
      rw [hb19, hb18]; exact hlen17s
    refine ⟨(1 + (1 + (1 + (1 + (1 + k6))))) + (1 + (1 + (1 + (1 + (1 + (k12 + (1 + (1 + (1 + (1 + (k17 + (1 + (1))))))))))))), b19,
      by rw [run_add im _ _, hrun06, run_add im 1 _, rr7, run_add im 1 _, rr8,
        run_add im 1 _, rr9, run_add im 1 _, rr10, run_add im 1 _, rr11,
        run_add im k12 _, hrun12, run_add im 1 _, rr13, run_add im 1 _, rr14,
        run_add im 1 _, rr15, run_add im 1 _, rr16, run_add im k17 _, hrun17,
        run_add im 1 1, rr18, rr19],
      mkInv b19 (by rw [hb19]) (by rw [hb19r 8]; exact hb17r8)
        (by rw [hb19r 9]; exact hb17r9) (by rw [hb19r 6]; exact hb17r6)
        (by rw [hb19r 4]; exact hb17r4) (by rw [hb19r 7]; exact hb17r7)
        (by rw [hb19r 1]; exact hb17r1) (by rw [hb19r 13]; exact hb17r13)
        (by rw [hb19r 14]; exact hb17r14) hlen19 hfr19
        (by rw [hb19mem]; exact hw17),
      ?_, hfr19, by rw [hb19mem]; exact hw17, hlen19⟩
    intro q h0 h1 h2 h4 h6 h8 h9 h12
    rw [hb19r q]
    exact hb17r q h0 h1 h2 h4 h6 h8 h9 h12


/-- One architectural step rewrites the register file by at most two
`some`-valued sets, or is a syscall (two `some` sets and `r11`
clobbered to `none`). -/
theorem exec_regs_form {ins : Ins} {len : Nat} {s s' : X86Int.St}
    (h : exec ins len s = .ok s') :
    s'.regs = s.regs ∨ (∃ d v, s'.regs = s.regs.set d (some v)) ∨
    (∃ d1 v1 d2 v2, s'.regs = (s.regs.set d1 (some v1)).set d2 (some v2)) ∨
    ∃ d1 v1 d2 v2,
      s'.regs = ((s.regs.set d1 (some v1)).set d2 (some v2)).set 11 none := by
  cases ins <;> simp only [exec] at h
  all_goals
    first
      | (injection h with h; cases h)
      | (have h1 := lift_ok h
         obtain ⟨a, -, h1⟩ := bind_some h1
         cases h1)
      | (have h1 := lift_ok h
         obtain ⟨a, -, h1⟩ := bind_some h1
         obtain ⟨b, -, h1⟩ := bind_some h1
         cases h1)
      | (have h1 := lift_ok h
         obtain ⟨a, -, h1⟩ := bind_some h1
         obtain ⟨b, -, h1⟩ := bind_some h1
         obtain ⟨c, -, h1⟩ := bind_some h1
         cases h1)
      | (split at h <;> (try split at h) <;>
          (first
            | cases h
            | (injection h with hh; cases hh)))
  all_goals
    first
      | exact Or.inl rfl
      | exact Or.inr (Or.inl ⟨_, _, rfl⟩)
      | exact Or.inr (Or.inr (Or.inl ⟨_, _, _, _, rfl⟩))
      | exact Or.inr (Or.inr (Or.inr ⟨_, _, _, _, rfl⟩))

theorem reg_set_some_or {s : X86Int.St} {d v i : Nat} :
    (s.set d (some v)).reg i = s.reg i ∨ (s.set d (some v)).reg i = some v := by
  by_cases h1 : i = d
  · subst h1
    by_cases h2 : i < s.regs.length
    · exact Or.inr (reg_set_self h2)
    · left
      have h3 : s.regs[i]? = none :=
        List.getElem?_eq_none (by omega)
      have h4 : (s.regs.set i (some v))[i]? = none :=
        List.getElem?_eq_none (by rw [List.length_set]; omega)
      show (s.set i (some v)).reg i = s.reg i
      rw [St.reg, St.reg, St.set, h4, h3]
  · exact Or.inl (reg_set_other h1)

theorem reg_of_regs {s s' : X86Int.St} {i : Nat} (h : s'.regs = s.regs) :
    s'.reg i = s.reg i := by
  rw [St.reg, St.reg, h]

theorem exec_reg_ne11 {ins : Ins} {len : Nat} {s s' : X86Int.St}
    (h : exec ins len s = .ok s') {i : Nat} (hi : i ≠ 11) :
    s'.reg i = s.reg i ∨ ∃ v, s'.reg i = some v := by
  have key : ∀ (u : X86Int.St) (d v : Nat),
      (u.set d (some v)).reg i = u.reg i ∨ ∃ x, (u.set d (some v)).reg i = some x := by
    intro u d v
    cases reg_set_some_or (s := u) (d := d) (v := v) (i := i) with
    | inl hh => exact Or.inl hh
    | inr hh => exact Or.inr ⟨v, hh⟩
  have cvt : ∀ (u : X86Int.St), s'.regs = u.regs → s'.reg i = u.reg i :=
    fun u hu => reg_of_regs hu
  cases exec_regs_form h with
  | inl hr => exact Or.inl (reg_of_regs hr)
  | inr hr =>
    cases hr with
    | inl hr =>
      obtain ⟨d, v, hv⟩ := hr
      rw [cvt (s.set d (some v)) hv]
      exact key s d v
    | inr hr =>
      cases hr with
      | inl hr =>
        obtain ⟨d1, v1, d2, v2, hv⟩ := hr
        rw [cvt ((s.set d1 (some v1)).set d2 (some v2)) hv]
        cases key (s.set d1 (some v1)) d2 v2 with
        | inl hh =>
          rw [hh]
          exact key s d1 v1
        | inr hh => exact Or.inr hh
      | inr hr =>
        obtain ⟨d1, v1, d2, v2, hv⟩ := hr
        rw [cvt (((s.set d1 (some v1)).set d2 (some v2)).set 11 none) hv]
        have hne : i ≠ 11 := hi
        rw [reg_set_other hne]
        cases key (s.set d1 (some v1)) d2 v2 with
        | inl hh =>
          rw [hh]
          exact key s d1 v1
        | inr hh => exact Or.inr hh

theorem step_reg_ne11 {im : Nat → Option (Ins × Nat)} {s s' : X86Int.St}
    (h : X86Int.step im s = .ok s') {i : Nat} (hi : i ≠ 11) :
    s'.reg i = s.reg i ∨ ∃ v, s'.reg i = some v := by
  unfold X86Int.step at h
  cases him : im s.rip with
  | none => rw [him] at h; cases h
  | some p =>
    obtain ⟨ins, len⟩ := p
    rw [him] at h
    exact exec_reg_ne11 h hi

/-- Every register except `r11` (clobbered by `syscall`), once
defined, stays defined under any run: every step either preserves it
or writes a `some`. -/
theorem run_reg_ne11_some {im : Nat → Option (Ins × Nat)} :
    ∀ (k : Nat) {s s' : X86Int.St} {i : Nat}, i ≠ 11 →
      run im k (Out.ok s) = Out.ok s' →
      (∃ v, s.reg i = some v) → ∃ v, s'.reg i = some v := by
  intro k
  induction k with
  | zero =>
    intro s s' i _ hr hv
    rw [run_zero] at hr
    cases hr
    exact hv
  | succ k ih =>
    intro s s' i hi hr hv
    have h1 : run im (k + 1) (Out.ok s) = run im k (X86Int.step im s) := rfl
    rw [h1] at hr
    cases hst : X86Int.step im s with
    | ok s1 =>
      rw [hst] at hr
      apply ih hi hr
      cases step_reg_ne11 hst hi with
      | inl hh =>
        obtain ⟨v, hv'⟩ := hv
        exact ⟨v, by rw [hh]; exact hv'⟩
      | inr hh => exact hh
    | exit n =>
      rw [hst] at hr
      cases k <;> simp [X86Int.run] at hr
    | stuck =>
      rw [hst] at hr
      cases k <;> simp [X86Int.run] at hr

/-- Registers `0..15` except `r11` are all defined. -/
def RegsOk (s : X86Int.St) : Prop :=
  ∀ i, i < 16 → i ≠ 11 → ∃ v, s.reg i = some v

theorem run_regs_ok {im : Nat → Option (Ins × Nat)} {k : Nat}
    {s s' : X86Int.St} (hr : run im k (Out.ok s) = Out.ok s')
    (h : RegsOk s) : RegsOk s' :=
  fun i hi16 hi11 => run_reg_ne11_some k hi11 hr (h i hi16 hi11)

/-- `check_start`: cursor to the first entry, count to `rbp`. -/
theorem CheckStart_triple {im : Nat → Option (Ins × Nat)}
    (oc : OuterCode im) {nw tbl : Nat} :
    ∀ s, s.rip = aCheckStart → s.reg 14 = some tbl → s.reg 13 = some nw →
      ∃ k s', run im k (Out.ok s) = Out.ok s' ∧ s'.rip = aCheckWord ∧
        s'.reg 7 = some tbl ∧ s'.reg 5 = some nw ∧
        (∀ q, q ≠ 5 → q ≠ 7 → s'.reg q = s.reg q) ∧ s'.mem = s.mem ∧
        s'.regs.length = s.regs.length := by
  intro s hrip hr14 hr13
  have e1 : X86Int.step im s = .ok { s.set 7 (some tbl) with
      rip := s.rip + 3 } :=
    step_mov (by rw [hrip]; exact oc.cs0) hr14
  obtain ⟨a1, ha1⟩ : ∃ u : X86Int.St, u = { s.set 7 (some tbl) with
    rip := s.rip + 3 } := ⟨_, rfl⟩
  have ha1rip : a1.rip = aCheckStart + 3 := by
    rw [ha1]; show s.rip + 3 = aCheckStart + 3; rw [hrip]
  have ga1 : ∀ q, q ≠ 7 → a1.reg q = s.reg q := by
    intro q hq
    rw [ha1]
    show (s.set 7 (some tbl)).reg q = s.reg q
    rw [reg_set_other hq]
  have ha1r7 : a1.reg 7 = some tbl := by
    rw [ha1]
    show (s.set 7 (some tbl)).reg 7 = some tbl
    rw [reg_set_self (by
      have : 13 < s.regs.length := reg_some_len hr13
      omega)]
  have e2 : X86Int.step im a1 = .ok { a1.set 5 (some nw) with
      rip := a1.rip + 3 } :=
    step_mov (by rw [ha1rip]; exact oc.cs1)
      (by rw [ga1 13 (by decide)]; exact hr13)
  obtain ⟨a2, ha2⟩ : ∃ u : X86Int.St, u = { a1.set 5 (some nw) with
    rip := a1.rip + 3 } := ⟨_, rfl⟩
  have rr1 : run im 1 (Out.ok s) = Out.ok a1 := by
    rw [run_succ_ok im 0 s, e1, ← ha1, run_zero]
  have rr2 : run im 1 (Out.ok a1) = Out.ok a2 := by
    rw [run_succ_ok im 0 a1, e2, ← ha2, run_zero]
  refine ⟨1 + 1, a2, by rw [run_add im 1 1, rr1, rr2], ?_, ?_, ?_, ?_, ?_, ?_⟩
  · rw [ha2]; show a1.rip + 3 = aCheckWord; rw [ha1rip]; rfl
  · rw [ha2]
    show (a1.set 5 (some nw)).reg 7 = some tbl
    rw [reg_set_other (by decide)]; exact ha1r7
  · rw [ha2]
    show (a1.set 5 (some nw)).reg 5 = some nw
    rw [reg_set_self (by
      have e : a1.regs.length = s.regs.length := by
        rw [ha1]
        show (s.set 7 (some tbl)).regs.length = s.regs.length
        exact List.length_set
      have : 13 < s.regs.length := reg_some_len hr13
      omega)]
  · intro q h5 h7
    rw [ha2]
    show (a1.set 5 (some nw)).reg q = s.reg q
    rw [reg_set_other h5]; exact ga1 q h7
  · rw [ha2, ha1]; exact rfl
  · rw [ha2, ha1]
    exact List.length_set.trans List.length_set

/-- Everything the checker needs about one table entry besides its
bytes: statics for the walk and the ann validation. -/
def WordOk (pr : Prog) (wd : Sporos.Word) (w p cp sp tbl : Nat) : Prop :=
  WalkStatic pr wd w p sp tbl ∧ wd.ins < W ∧
  (∀ x ∈ wd.ann, x.2.d < W ∧ x.2.a < W) ∧
  (∀ x ∈ wd.code, ImmOk x) ∧ hBnd pr wd wd.code 0 ⟨wd.ins, 0⟩ ∧
  (cp + codeBytes wd.code ≤ sp - 48 ∨ sp ≤ cp) ∧
  cp + codeBytes wd.code < W ∧
  (sp ≤ w ∨ w + 56 ≤ sp - 48) ∧ (sp ≤ p ∨ p + 12 * wd.ann.length ≤ sp - 48)

/-- The decoded words table: fields, code and ann arrays per entry. -/
def WordTab (pr : Prog) (tbl sp : Nat) (s : X86Int.St) : Prop :=
  ∀ j (hj : j < pr.words.length), ∃ cp p,
    rd64 s.mem (tbl + 64 * j) = some (pr.words[j]).ins ∧
    rd64 s.mem (tbl + 64 * j + 8) = some (pr.words[j]).outs ∧
    rd64 s.mem (tbl + 64 * j + 16) = some (pr.words[j]).locals ∧
    rd64 s.mem (tbl + 64 * j + 24) = some (pr.words[j]).code.length ∧
    rd64 s.mem (tbl + 64 * j + 32) = some (pr.words[j]).ann.length ∧
    rd64 s.mem (tbl + 64 * j + 40) = some cp ∧
    rd64 s.mem (tbl + 64 * j + 48) = some p ∧
    CodeSeq s.mem cp (pr.words[j]).code ∧ AnnLayout s (pr.words[j]).ann p ∧
    WordOk pr (pr.words[j]) (tbl + 64 * j) p cp sp tbl

theorem WordTab_of_frame {pr : Prog} {tbl sp : Nat} {s s' : X86Int.St}
    (ht : WordTab pr tbl sp s)
    (htab : ∀ j, j < pr.words.length →
      sp ≤ tbl + 64 * j ∨ tbl + 64 * j + 64 ≤ sp - 48)
    (hfr : ∀ x n, x + n ≤ sp - 48 ∨ sp ≤ x → s'.mem.rdN x n = s.mem.rdN x n) :
    WordTab pr tbl sp s' := by
  intro j hj
  obtain ⟨cp, p, hf0, hf8, hf16, hf24, hf32, hf40, hf48, hcsq, hlay,
    hok⟩ := ht j hj
  have hd := htab j hj
  have fld : ∀ (off v : Nat), off + 8 ≤ 64 →
      rd64 s.mem (tbl + 64 * j + off) = some v →
      rd64 s'.mem (tbl + 64 * j + off) = some v := by
    intro off v hoff hv
    unfold rd64
    rw [hfr (tbl + 64 * j + off) 8 (by omega)]
    exact hv
  refine ⟨cp, p, fld 0 _ (by omega) (by simpa using hf0),
    fld 8 _ (by omega) hf8, fld 16 _ (by omega) hf16,
    fld 24 _ (by omega) hf24, fld 32 _ (by omega) hf32,
    fld 40 _ (by omega) hf40, fld 48 _ (by omega) hf48, ?_, ?_, hok⟩
  · exact CodeSeq_of_frame hcsq (fun x k _ _ => hfr x k (by
      have hdc := hok.2.2.2.2.2.1
      omega))
  · exact AnnLayout_of_frame hlay (fun x k _ _ => hfr x k (by
      have hdp := hok.2.2.2.2.2.2.2.2
      omega))

/-- The outer `check_word` loop state: `rest` words remain. -/
def OuterSt (pr : Prog) (tbl sp : Nat) (rest : List Sporos.Word)
    (s : X86Int.St) : Prop :=
  s.rip = aCheckWord ∧ s.reg 5 = some rest.length ∧
  s.reg 7 = some (tbl + 64 * (pr.words.length - rest.length)) ∧
  s.reg 13 = some pr.words.length ∧ s.reg 14 = some tbl ∧
  s.reg 4 = some sp ∧ 16 ≤ s.regs.length ∧ RegsOk s ∧
  StackRoom s 6 ∧ EffTable s ∧ WordTab pr tbl sp s

/-- One word of the outer loop: from `check_word` on `wd :: rest`
back to `check_word` on `rest`, when `checkWord1` accepts `wd`. -/
theorem word_step {im : Nat → Option (Ins × Nat)}
    (ac : AllCode im) (oc : OuterCode im)
    {pr : Prog} {tbl sp : Nat} (hsp48 : 48 ≤ sp) (hspW : sp < W)
    (htblW : tbl + 64 * pr.words.length < W)
    (htab : ∀ j, j < pr.words.length →
      sp ≤ tbl + 64 * j ∨ tbl + 64 * j + 64 ≤ sp - 48)
    (hdisjEff48 : aEff + 114 ≤ sp - 48 ∨ sp ≤ aEff)
    (hchk : ∀ wd ∈ pr.words, Sporos.checkWord1 pr wd = true) :
    ∀ (wd : Sporos.Word) (rest : List Sporos.Word)
      (pre : List Sporos.Word), pr.words = pre ++ (wd :: rest) →
      ∀ s, OuterSt pr tbl sp (wd :: rest) s →
      ∃ k s', run im k (Out.ok s) = Out.ok s' ∧ OuterSt pr tbl sp rest s' := by
  intro wd rest pre happ s hs
  obtain ⟨hrip, hr5, hr7, hr13, hr14, hr4, h16, hregs, hroom, heff,
    htabS⟩ := hs
  have hlenann : pr.words.length = pre.length + (rest.length + 1) := by
    rw [happ, List.length_append, List.length_cons]
  have hj0 : pre.length < pr.words.length := by omega
  have hget? : pr.words[pre.length]? = some wd := by
    rw [happ, List.getElem?_append_right (by simp)]
    simp
  have hget : pr.words[pre.length]'hj0 = wd :=
    Option.some.inj ((List.getElem?_eq_getElem hj0).symm.trans hget?)
  obtain ⟨cp, p, hf0, hf8, hf16, hf24, hf32, hf40, hf48, hcsq, hlay,
    hok⟩ := htabS pre.length hj0
  rw [hget] at hf0 hf8 hf16 hf24 hf32 hcsq hlay hok
  obtain ⟨st, hinsW, hhW, himm, hhb, hdisjC48, hcbound, hdisjw48,
    hdisjp48⟩ := hok
  have hr7w : s.reg 7 = some (tbl + 64 * pre.length) := by
    have hw : tbl + 64 * (pr.words.length - (wd :: rest).length) =
        tbl + 64 * pre.length := by
      rw [List.length_cons]
      congr 1
      congr 1
      omega
    rw [hw] at hr7
    exact hr7
  have hmem : wd ∈ pr.words := by
    rw [happ]
    exact List.mem_append_right _ List.mem_cons_self
  have hwd := hchk wd hmem
  have hwd2 := hwd
  simp only [Sporos.checkWord1, Bool.and_eq_true] at hwd2
  have hne := hwd2.1.1.1
  have h0clause := hwd2.1.1.2
  have hann := hwd2.1.2
  have hwalk := hwd2.2
  have hnc : wd.code.length ≠ 0 := by
    intro hz
    have hc : wd.code = [] := List.eq_nil_of_length_eq_zero hz
    rw [hc] at hne
    simp at hne
  have hlenW : pr.words.length < W := by
    have h1 : pr.words.length ≤ 64 * pr.words.length :=
      Nat.le_mul_of_pos_left _ (by decide)
    omega
  -- check_word head: to annval
  obtain ⟨k1, s1, hrun1, hs1rip, hs1r6, hs1r1, -, hs1r5, hs1r7, gb1, hs1mem,
    hlen1⟩ :=
    CheckWord_head oc (tbl + 64 * pre.length) (rest.length + 1)
      wd.code.length wd.ann.length p st.w56 (by omega) hnc s hrip hr5 hr7w
      hf24 hf48 hf32
  have hs1r4 : s1.reg 4 = some sp := by
    rw [gb1 4 (by decide) (by decide) (by decide)]; exact hr4
  have hregs1 : RegsOk s1 := run_regs_ok hrun1 hregs
  have hs1r13 : s1.reg 13 = some pr.words.length := by
    rw [gb1 13 (by decide) (by decide) (by decide)]; exact hr13
  have hs1r14 : s1.reg 14 = some tbl := by
    rw [gb1 14 (by decide) (by decide) (by decide)]; exact hr14
  have hannst : AnnSt wd (tbl + 64 * pre.length) p sp wd.ann s1 := by
    refine ⟨hs1rip, hs1r7, ?_, hs1r1, hs1r4,
      by rw [hs1mem]; exact hf24, by rw [hs1mem]; exact hf32,
      by rw [hs1mem]; exact hf48, AnnLayout_of_mem hs1mem hlay, by omega,
      StackRoom_of_mem hs1mem (by rw [hs1r4, hr4]) hroom⟩
    simpa using hs1r6
  have hannstatic : AnnStatic wd (tbl + 64 * pre.length) p sp :=
    ⟨st.w56, st.p0, st.pann, hsp48, hspW, st.clenW, hhW, hdisjw48, hdisjp48⟩
  -- annval loop: to codewalk
  obtain ⟨k2, s2, hrun2, hs2rip, hs2r7, hs2r4, hs2r6, hs2r1, hf24_2, hf32_2,
    hf48_2, hlay2, hroom2, h12len2, gb2, hfr2, hw2, hlen2⟩ :=
    annval_loop ac oc hannstatic hann wd.ann [] (by simp) s1 hannst
  have hrun12 : run im (k1 + k2) (Out.ok s) = Out.ok s2 := by
    rw [run_add im k1 k2, hrun1, hrun2]
  have hregs2 : RegsOk s2 := run_regs_ok hrun12 hregs
  have hs2r13 : s2.reg 13 = some pr.words.length := by
    rw [gb2 13 (by decide) (by decide) (by decide) (by decide) (by decide)
      (by decide) (by decide) (by decide)]; exact hs1r13
  have hs2r14 : s2.reg 14 = some tbl := by
    rw [gb2 14 (by decide) (by decide) (by decide) (by decide) (by decide)
      (by decide) (by decide) (by decide)]; exact hs1r14
  have hs2r5 : s2.reg 5 = some (rest.length + 1) := by
    rw [gb2 5 (by decide) (by decide) (by decide) (by decide) (by decide)
      (by decide) (by decide) (by decide)]; exact hs1r5
  have hfr02 : ∀ x n, x + n ≤ sp - 48 ∨ sp ≤ x →
      s2.mem.rdN x n = s.mem.rdN x n := by
    intro x n hx
    rw [hfr2 x n hx, hs1mem]
  have hfld2 : ∀ (off v : Nat), off + 8 ≤ 64 →
      rd64 s.mem (tbl + 64 * pre.length + off) = some v →
      rd64 s2.mem (tbl + 64 * pre.length + off) = some v := by
    intro off v hoff hv
    unfold rd64
    rw [hfr02 (tbl + 64 * pre.length + off) 8 (by
      have hd := htab pre.length hj0
      omega)]
    exact hv
  have heff2 : EffTable s2 := by
    intro q
    rw [hfr02 (aEff + 2 * q.code) 2 (by
      have hc := Prim_code_lt q
      cases hdisjEff48 with
      | inl hh => exact Or.inl (by omega)
      | inr hh => exact Or.inr (by omega))]
    exact heff q
  have hcsq2 : CodeSeq s2.mem cp wd.code :=
    CodeSeq_of_frame hcsq (fun x k _ _ => hfr02 x k (by omega))
  have hwords2 : ∀ n (hn : n < pr.words.length),
      rd64 s2.mem (tbl + n * 64) = some (pr.words[n]).ins ∧
      rd64 s2.mem (tbl + n * 64 + 8) = some (pr.words[n]).outs := by
    intro n hn
    obtain ⟨cpn, pn, g0, g8, -, -, -, -, -, -, -, -⟩ := htabS n hn
    have hd := htab n hn
    refine ⟨?_, ?_⟩
    · have e : tbl + n * 64 = tbl + 64 * n := by rw [Nat.mul_comm]
      rw [e]
      unfold rd64
      rw [hfr02 (tbl + 64 * n) 8 (by omega)]
      exact g0
    · have e : tbl + n * 64 + 8 = tbl + 64 * n + 8 := by rw [Nat.mul_comm]
      rw [e]
      unfold rd64
      rw [hfr02 (tbl + 64 * n + 8) 8 (by omega)]
      exact g8
  have hdisjC32 : cp + codeBytes wd.code ≤ sp - 32 ∨ sp ≤ cp := by
    cases hdisjC48 with
    | inl hh => exact Or.inl (by omega)
    | inr hh => exact Or.inr hh
  -- codewalk: to walk_top in WalkInv
  obtain ⟨k3, s3, hrun3, hinv3, gb3, hfr3, hw3, hlen3⟩ :=
    codewalk_entry ac oc st hsp48 h0clause hinsW s2 hs2rip hs2r7 hs2r4
      hs2r1 (hregs2 9 (by decide) (by decide))
      (hregs2 12 (by decide) (by decide)) hs2r13 hs2r14 (by omega)
      (hfld2 0 _ (by omega) (by simpa using hf0))
      (hfld2 8 _ (by omega) hf8) (hfld2 16 _ (by omega) hf16) hf24_2 hf32_2
      (hfld2 40 _ (by omega) hf40) hf48_2 hlay2 hroom2 heff2 hcsq2 hwords2
      hdisjC32 hcbound himm hhb
  have hs3r5 : s3.reg 5 = some (rest.length + 1) := by
    rw [gb3 5 (by decide) (by decide) (by decide) (by decide) (by decide)
      (by decide) (by decide) (by decide)]; exact hs2r5
  have hfr03 : ∀ x n, x + n ≤ sp - 48 ∨ sp ≤ x →
      s3.mem.rdN x n = s.mem.rdN x n := by
    intro x n hx
    exact (hfr3 x n (by omega)).trans (hfr02 x n hx)
  have hw02 : wshape s2.mem = wshape s.mem := by rw [hw2, hs1mem]
  have hlen02 : s2.regs.length = s.regs.length := hlen2.trans hlen1
  have hw03 : wshape s3.mem = wshape s.mem := hw3.trans hw02
  have hlen03 : s3.regs.length = s.regs.length := hlen3.trans hlen02
  -- the walk itself
  obtain ⟨k4, s4, hrun4, hs4rip, hs4r1, hs4r6, hs4r7, hs4r4, hfr4, hw4,
    hs4r5, hs4r13, hs4r14⟩ :=
    walk_loop ac st wd.code 0 cp ⟨wd.ins, 0⟩ ⟨wd.ins, 0⟩ s3 hinv3 hwalk
  have hlen4 : s4.regs.length = s.regs.length :=
    (run_regs_length k4 hrun4).trans hlen03
  have hfr04 : ∀ x n, x + n ≤ sp - 48 ∨ sp ≤ x →
      s4.mem.rdN x n = s.mem.rdN x n := by
    intro x n hx
    exact (hfr4 x n (by omega)).trans (hfr03 x n hx)
  have hw04 : wshape s4.mem = wshape s.mem := hw4.trans hw03
  have hs4r5' : s4.reg 5 = some (rest.length + 1) := by
    rw [hs4r5]; exact hs3r5
  have hs4r13' : s4.reg 13 = some pr.words.length := by
    rw [hs4r13]; exact hinv3.r13
  have hs4r14' : s4.reg 14 = some tbl := by
    rw [hs4r14]; exact hinv3.r14
  -- word_done: advance and count down
  obtain ⟨k5, s5, hrun5, hs5rip, hs5r7, hs5r5, gb5, hs5mem, hlen5⟩ :=
    WordDone_triple oc (tbl + 64 * pre.length) (rest.length + 1) (by omega)
      (by omega)
      (show tbl + 64 * pre.length + 64 < W by
        have : 64 * pre.length + 64 = 64 * (pre.length + 1) := by
          rw [Nat.mul_add, Nat.mul_one]
        omega)
      s4 hs4rip hs4r7 hs4r5'
  have hrunT : run im (k1 + k2 + k3 + k4 + k5) (Out.ok s) = Out.ok s5 := by
    rw [run_add im (k1 + k2 + k3 + k4) k5,
      run_add im (k1 + k2 + k3) k4, run_add im (k1 + k2) k3,
      run_add im k1 k2, hrun1, hrun2, hrun3, hrun4, hrun5]
  have hfrT : ∀ x n, x + n ≤ sp - 48 ∨ sp ≤ x →
      s5.mem.rdN x n = s.mem.rdN x n := by
    intro x n hx
    rw [hs5mem]; exact hfr04 x n hx
  have hwT : wshape s5.mem = wshape s.mem := by rw [hs5mem]; exact hw04
  have hlenT : s5.regs.length = s.regs.length := hlen5.trans hlen4
  have hs5r4 : s5.reg 4 = some sp := by
    rw [gb5 4 (by decide) (by decide)]; exact hs4r4
  have hs5r13 : s5.reg 13 = some pr.words.length := by
    rw [gb5 13 (by decide) (by decide)]; exact hs4r13'
  have hs5r14 : s5.reg 14 = some tbl := by
    rw [gb5 14 (by decide) (by decide)]; exact hs4r14'
  have hs5r5' : s5.reg 5 = some rest.length := by
    rw [hs5r5]
    congr 1
  have hs5r7' : s5.reg 7 =
      some (tbl + 64 * (pr.words.length - rest.length)) := by
    rw [hs5r7]
    congr 1
    have e : pr.words.length - rest.length = pre.length + 1 := by omega
    rw [e, Nat.mul_add, Nat.mul_one, Nat.add_assoc]
  have heff5 : EffTable s5 := by
    intro q
    rw [hfrT (aEff + 2 * q.code) 2 (by
      have hc := Prim_code_lt q
      cases hdisjEff48 with
      | inl hh => exact Or.inl (by omega)
      | inr hh => exact Or.inr (by omega))]
    exact heff q
  refine ⟨k1 + k2 + k3 + k4 + k5, s5, hrunT, hs5rip, hs5r5', hs5r7', hs5r13,
    hs5r14, hs5r4, by omega, run_regs_ok hrunT hregs,
    StackRoom_of_wshape hwT (by rw [hs5r4, hr4]) hroom, heff5,
    WordTab_of_frame htabS htab hfrT⟩

/-- The whole outer loop plus `accept`: from `check_word` with
`rest` words left, the run exits 0. Induction on the remaining
words. -/
theorem words_loop {im : Nat → Option (Ins × Nat)}
    (ac : AllCode im) (oc : OuterCode im)
    {pr : Prog} {tbl sp : Nat} (hsp48 : 48 ≤ sp) (hspW : sp < W)
    (htblW : tbl + 64 * pr.words.length < W)
    (htab : ∀ j, j < pr.words.length →
      sp ≤ tbl + 64 * j ∨ tbl + 64 * j + 64 ≤ sp - 48)
    (hdisjEff48 : aEff + 114 ≤ sp - 48 ∨ sp ≤ aEff)
    (hchk : ∀ wd ∈ pr.words, Sporos.checkWord1 pr wd = true) :
    ∀ (rest : List Sporos.Word) (pre : List Sporos.Word),
      pr.words = pre ++ rest → ∀ s, OuterSt pr tbl sp rest s →
      ∃ k, run im k (Out.ok s) = Out.exit 0 := by
  intro rest
  induction rest with
  | nil =>
    intro pre _ s hs
    obtain ⟨hrip, hr5, -, -, -, hr4, -, hregs, hroom, -, -⟩ := hs
    have hr5' : s.reg 5 = some 0 := hr5
    obtain ⟨k1, s1, hrun1, hs1rip, gs1, hs1mem, hlen1⟩ :=
      CheckWord_done oc s hrip hr5'
    have hs1r4 : s1.reg 4 = some sp := by
      rw [gs1 4]; exact hr4
    have hregs1 : RegsOk s1 := run_regs_ok hrun1 hregs
    have hroom1 : StackRoom s1 6 :=
      StackRoom_of_mem hs1mem (by rw [hs1r4, hr4]) hroom
    obtain ⟨v6, hv6⟩ := hregs1 6 (by decide) (by decide)
    obtain ⟨v2, hv2⟩ := hregs1 2 (by decide) (by decide)
    obtain ⟨v10, hv10⟩ := hregs1 10 (by decide) (by decide)
    obtain ⟨v8, hv8⟩ := hregs1 8 (by decide) (by decide)
    obtain ⟨v9, hv9⟩ := hregs1 9 (by decide) (by decide)
    obtain ⟨m1, hm1s⟩ := Option.isSome_iff_exists.mp
      (hroom1 sp hs1r4 0 (by decide) 60)
    obtain ⟨m2, hm2s⟩ := Option.isSome_iff_exists.mp
      (wr64_some_of_wr64 hm1s (hroom1 sp hs1r4 0 (by decide) 0))
    obtain ⟨k2, hrun2⟩ := Accept_exit oc sp v6 v2 v10 v8 v9 m1 m2 (by omega)
      hspW s1 hs1rip hs1r4 hv6 hv2 hv10 hv8 hv9 hm1s hm2s
    exact ⟨k1 + k2, by
      rw [run_add im k1 k2, hrun1]
      exact hrun2⟩
  | cons wd rest' ih =>
    intro pre happ s hs
    obtain ⟨k1, s1, hrun1, hs1⟩ :=
      word_step ac oc hsp48 hspW htblW htab hdisjEff48 hchk wd rest' pre
        happ s hs
    have happ' : pr.words = (pre ++ [wd]) ++ rest' := by
      rw [happ, List.append_assoc, List.singleton_append]
    obtain ⟨k2, hrun2⟩ := ih (pre ++ [wd]) happ' s1 hs1
    exact ⟨k1 + k2, by
      rw [run_add im k1 k2, hrun1]
      exact hrun2⟩

/-- Accept direction of the stage-1 checker at `check_word`: a
program with `checkProg1 pr = true`, in the table/memory layout the
outer invariant describes, exits 0. -/
theorem checker_accepts {im : Nat → Option (Ins × Nat)}
    (ac : AllCode im) (oc : OuterCode im)
    {pr : Prog} {tbl sp : Nat} (hsp48 : 48 ≤ sp) (hspW : sp < W)
    (htblW : tbl + 64 * pr.words.length < W)
    (htab : ∀ j, j < pr.words.length →
      sp ≤ tbl + 64 * j ∨ tbl + 64 * j + 64 ≤ sp - 48)
    (hdisjEff48 : aEff + 114 ≤ sp - 48 ∨ sp ≤ aEff)
    (hchk : Sporos.checkProg1 pr = true) :
    ∀ s, OuterSt pr tbl sp pr.words s →
      ∃ k, run im k (Out.ok s) = Out.exit 0 := by
  have hall : ∀ wd ∈ pr.words, Sporos.checkWord1 pr wd = true := by
    have h2 : (pr.words.all fun wd => Sporos.checkWord1 pr wd) = true := by
      have hh := hchk
      simp only [Sporos.checkProg1, Bool.and_eq_true] at hh
      exact hh.1
    exact fun wd hmem => List.all_eq_true.mp h2 wd hmem
  exact fun s hs =>
    words_loop ac oc hsp48 hspW htblW htab hdisjEff48 hall pr.words []
      (by simp) s hs


/-- Accept direction against `Sporos.checkProg`, via the single-pass
equivalence `checkProg1_eq`. -/
theorem checker_accepts_checkProg {im : Nat → Option (Ins × Nat)}
    (ac : AllCode im) (oc : OuterCode im)
    {pr : Prog} {tbl sp : Nat} (hsp48 : 48 ≤ sp) (hspW : sp < W)
    (htblW : tbl + 64 * pr.words.length < W)
    (htab : ∀ j, j < pr.words.length →
      sp ≤ tbl + 64 * j ∨ tbl + 64 * j + 64 ≤ sp - 48)
    (hdisjEff48 : aEff + 114 ≤ sp - 48 ∨ sp ≤ aEff)
    (hchk : Sporos.checkProg pr = true) :
    ∀ s, OuterSt pr tbl sp pr.words s →
      ∃ k, run im k (Out.ok s) = Out.exit 0 :=
  checker_accepts ac oc hsp48 hspW htblW htab hdisjEff48
    (by rw [Sporos.checkProg1_eq]; exact hchk)

/-- From `check_start`: cursor and count are set from the table
registers, then the outer loop runs to the accept exit. -/
theorem checker_accepts_from_start {im : Nat → Option (Ins × Nat)}
    (ac : AllCode im) (oc : OuterCode im)
    {pr : Prog} {tbl sp : Nat} (hsp48 : 48 ≤ sp) (hspW : sp < W)
    (htblW : tbl + 64 * pr.words.length < W)
    (htab : ∀ j, j < pr.words.length →
      sp ≤ tbl + 64 * j ∨ tbl + 64 * j + 64 ≤ sp - 48)
    (hdisjEff48 : aEff + 114 ≤ sp - 48 ∨ sp ≤ aEff)
    (hchk : Sporos.checkProg pr = true) :
    ∀ s, s.rip = aCheckStart → s.reg 14 = some tbl →
      s.reg 13 = some pr.words.length → s.reg 4 = some sp →
      16 ≤ s.regs.length → RegsOk s → StackRoom s 6 → EffTable s →
      WordTab pr tbl sp s →
      ∃ k, run im k (Out.ok s) = Out.exit 0 := by
  intro s hrip hr14 hr13 hr4 h16 hregs hroom heff htabS
  obtain ⟨k1, s1, hrun1, hs1rip, hs1r7, hs1r5, gb1, hs1mem, hlen1⟩ :=
    CheckStart_triple oc s hrip hr14 hr13
  have hs1 : OuterSt pr tbl sp pr.words s1 := by
    refine ⟨hs1rip, hs1r5, by simpa using hs1r7,
      by rw [gb1 13 (by decide) (by decide)]; exact hr13,
      by rw [gb1 14 (by decide) (by decide)]; exact hr14,
      by rw [gb1 4 (by decide) (by decide)]; exact hr4, by omega,
      run_regs_ok hrun1 hregs,
      StackRoom_of_mem hs1mem (by
        rw [gb1 4 (by decide) (by decide)]) hroom,
      by
        intro q
        rw [hs1mem]; exact heff q,
      by
        apply WordTab_of_frame htabS htab
        intro x n _
        rw [hs1mem]⟩
  obtain ⟨k2, hrun2⟩ := checker_accepts_checkProg ac oc hsp48 hspW htblW htab
    hdisjEff48 hchk s1 hs1
  exact ⟨k1 + k2, by
    rw [run_add im k1 k2, hrun1]
    exact hrun2⟩



/-! ### Refusal direction: status exits and first failure branches -/

/-- Decode facts for a five-instruction status-exit block. -/
structure ExitCode (im : Nat → Option (Ins × Nat)) (entry code : Nat) : Prop where
  d0 : im entry = some (.pushI 60, 2)
  d1 : im (entry + 2) = some (.pop 0, 1)
  d2 : im (entry + 3) = some (.pushI code, 2)
  d3 : im (entry + 5) = some (.pop 7, 1)
  d4 : im (entry + 6) = some (.syscall, 2)

/-- A status-exit block: `exit(code)` for any status below `W`. -/
theorem Status_exit {im : Nat → Option (Ins × Nat)} {entry code : Nat}
    (fc : ExitCode im entry code) (hcode : code < W) :
    ∀ (sp a2 a3 a4 a5 a6 : Nat) (m1 m2 : Mem), 32 ≤ sp → sp < W →
    ∀ s, s.rip = entry → s.reg 4 = some sp →
      s.reg 6 = some a2 → s.reg 2 = some a3 → s.reg 10 = some a4 →
      s.reg 8 = some a5 → s.reg 9 = some a6 →
      wr64 s.mem ((sp + W - 8) % W) 60 = some m1 →
      wr64 m1 ((sp + W - 8) % W) (code % W) = some m2 →
      ∃ k, run im k (Out.ok s) = Out.exit (code % 256) := by
  intro sp a2 a3 a4 a5 a6 m1 m2 hsp32 hspW s hrip hr4 hr6 hr2 hr10 hr8 hr9 hm1 hm2
  have hslot : (sp + W - 8) % W = sp - 8 := slot_sub (k := 1) (by omega) hspW
  have hm1' : wr64 s.mem ((sp + W - 8) % W) (60 % W) = some m1 := hm1
  have hr60 : rd64 m1 ((sp + W - 8) % W) = some 60 := rd64_wr64 hm1'
  have hcodemod : code % W = code := Nat.mod_eq_of_lt hcode
  have hm2' : wr64 m1 ((sp + W - 8) % W) (code % W) = some m2 := hm2
  have hr0 : rd64 m2 ((sp + W - 8) % W) = some code := by
    have h := rd64_wr64 hm2
    have e : (code % W) % W = code := by
      rw [Nat.mod_mod]
      exact hcodemod
    exact e ▸ h
  have e1 : X86Int.step im s =
      .ok { s.set 4 (some ((sp + W - 8) % W)) with mem := m1, rip := s.rip + 2 } :=
    step_pushI (by rw [hrip]; exact fc.d0) hr4 hm1'
  obtain ⟨a1, ha1⟩ : ∃ u : X86Int.St, u =
    { s.set 4 (some ((sp + W - 8) % W)) with mem := m1, rip := s.rip + 2 } := ⟨_, rfl⟩
  have ha1rip : a1.rip = entry + 2 := by
    rw [ha1]; show s.rip + 2 = entry + 2; rw [hrip]
  have ha1r4 : a1.reg 4 = some ((sp + W - 8) % W) := by
    rw [ha1]
    show (s.set 4 (some ((sp + W - 8) % W))).reg 4 = some ((sp + W - 8) % W)
    rw [reg_set_self (reg_some_len hr4)]
  have ha1mem : a1.mem = m1 := by rw [ha1]
  have e2 : X86Int.step im a1 =
      .ok { (a1.set 4 (some (((sp + W - 8) % W + 8) % W))).set 0 (some 60) with
        rip := a1.rip + 1 } :=
    step_pop (by rw [ha1rip]; exact fc.d1) ha1r4 (by rw [ha1mem]; exact hr60)
  obtain ⟨a2s, ha2⟩ : ∃ u : X86Int.St, u =
    { (a1.set 4 (some (((sp + W - 8) % W + 8) % W))).set 0 (some 60) with
      rip := a1.rip + 1 } := ⟨_, rfl⟩
  have hsp8 : ((sp + W - 8) % W + 8) % W = sp := by
    rw [hslot]
    have h1 : sp - 8 + 8 = sp := by omega
    rw [h1]; exact Nat.mod_eq_of_lt hspW
  have ha2rip : a2s.rip = entry + 3 := by
    rw [ha2]; show a1.rip + 1 = entry + 3; rw [ha1rip]
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
  have hm2s : wr64 a2s.mem ((sp + W - 8) % W) (code % W) = some m2 := by
    rw [ha2mem]; exact hm2'
  have e3 : X86Int.step im a2s =
      .ok { a2s.set 4 (some ((sp + W - 8) % W)) with mem := m2, rip := a2s.rip + 2 } :=
    step_pushI (by rw [ha2rip]; exact fc.d2) ha2r4 hm2s
  obtain ⟨a3, ha3⟩ : ∃ u : X86Int.St, u =
    { a2s.set 4 (some ((sp + W - 8) % W)) with mem := m2, rip := a2s.rip + 2 } :=
    ⟨_, rfl⟩
  have ha3rip : a3.rip = entry + 5 := by
    rw [ha3]; show a2s.rip + 2 = entry + 5; rw [ha2rip]
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
      .ok { (a3.set 4 (some (((sp + W - 8) % W + 8) % W))).set 7 (some code) with
        rip := a3.rip + 1 } :=
    step_pop (by rw [ha3rip]; exact fc.d3) ha3r4 (by rw [ha3mem]; exact hr0)
  obtain ⟨a4, ha4⟩ : ∃ u : X86Int.St, u =
    { (a3.set 4 (some (((sp + W - 8) % W + 8) % W))).set 7 (some code) with
      rip := a3.rip + 1 } := ⟨_, rfl⟩
  have ha4rip : a4.rip = entry + 6 := by
    rw [ha4]; show a3.rip + 1 = entry + 6; rw [ha3rip]
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
    show ((a3.set 4 (some (((sp + W - 8) % W + 8) % W))).set 7 (some code)).reg 0 =
      some 60
    rw [reg_set_other (by decide), reg_set_other (by decide), ga3 0 (by decide)]
    exact ha2r0
  have ha4r7 : a4.reg 7 = some code := by
    rw [ha4]
    show ((a3.set 4 (some (((sp + W - 8) % W + 8) % W))).set 7 (some code)).reg 7 =
      some code
    rw [reg_set_self (by
      have h9len : 9 < s.regs.length := reg_some_len hr9
      omega)]
  have ha4r : ∀ q, q ≠ 0 → q ≠ 4 → q ≠ 7 → a4.reg q = s.reg q := by
    intro q h0 h4 h7
    rw [ha4]
    show ((a3.set 4 (some (((sp + W - 8) % W + 8) % W))).set 7 (some code)).reg q =
      s.reg q
    rw [reg_set_other h7, reg_set_other h4, ga3 q h4]
    exact ga2 q h0 h4
  have e5 : X86Int.step im a4 = .exit (code % 256) :=
    step_syscall_exit (by rw [ha4rip]; exact fc.d4) ha4r0 ha4r7
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
  have r5 : run im 1 (Out.ok a4) = Out.exit (code % 256) := by
    rw [run_succ_ok im 0 a4, e5, run_zero]
  exact ⟨1 + 1 + 1 + 1 + 1, by
    rw [run_add im 1 4, r1, run_add im 1 3, r2, run_add im 1 2, r3, run_add im 1 1,
      r4, r5]⟩


/-- More stack room gives less. -/
theorem StackRoom_mono {s : X86Int.St} {k k' : Nat}
    (h : StackRoom s k) (hk : k' ≤ k) : StackRoom s k' :=
  fun sp hsp i hi v => h sp hsp i (by omega) v

/-- A status-exit block from `RegsOk` and one slot of stack room. -/
theorem Status_exit_of_room {im : Nat → Option (Ins × Nat)}
    {entry code sp : Nat} (fc : ExitCode im entry code) (hcode : code < W)
    (hsp32 : 32 ≤ sp) (hspW : sp < W) :
    ∀ s, s.rip = entry → s.reg 4 = some sp → RegsOk s → StackRoom s 1 →
      ∃ k, run im k (Out.ok s) = Out.exit (code % 256) := by
  intro s hrip hr4 hregs hroom
  obtain ⟨v6, hv6⟩ := hregs 6 (by decide) (by decide)
  obtain ⟨v2, hv2⟩ := hregs 2 (by decide) (by decide)
  obtain ⟨v10, hv10⟩ := hregs 10 (by decide) (by decide)
  obtain ⟨v8, hv8⟩ := hregs 8 (by decide) (by decide)
  obtain ⟨v9, hv9⟩ := hregs 9 (by decide) (by decide)
  obtain ⟨m1, hm1s⟩ := Option.isSome_iff_exists.mp
    (hroom sp hr4 0 (by decide) 60)
  obtain ⟨m2, hm2s⟩ := Option.isSome_iff_exists.mp
    (wr64_some_of_wr64 hm1s (hroom sp hr4 0 (by decide) (code % W)))
  exact Status_exit fc hcode sp v6 v2 v10 v8 v9 m1 m2 (by omega) hspW s
    hrip hr4 hv6 hv2 hv10 hv8 hv9 hm1s hm2s

/-- Decode facts only the refusal direction needs: the `fail1` exit
block, and the failure jumps the accept proofs jump over. -/
structure RefuseCode (im : Nat → Option (Ins × Nat)) : Prop where
  fail1 : ExitCode im aFail1 1
  cwEmpty : im (aCheckWord + 19) = some (.jmp aFail1, 5)
  avRange : im (aAnnval + 32) = some (.jmp aFail1, 5)

/-- Any state that reaches `fail1` with the registers defined and
stack room exits 1. -/
theorem exit1_of_fail1 {im : Nat → Option (Ins × Nat)}
    (rc : RefuseCode im) {sp : Nat} (hsp32 : 32 ≤ sp) (hspW : sp < W) :
    ∀ s, s.rip = aFail1 → s.reg 4 = some sp → RegsOk s → StackRoom s 1 →
      ∃ k, run im k (Out.ok s) = Out.exit 1 := by
  intro s hrip hr4 hregs hroom
  obtain ⟨k, hk⟩ := Status_exit_of_room rc.fail1 (by unfold W; decide)
    (by omega) hspW s hrip hr4 hregs hroom
  exact ⟨k, by simpa using hk⟩

/-- `check_word` refuses a word with no instructions: the zero
length branches to `fail1`, which exits 1. -/
theorem CheckWord_empty_exit1 {im : Nat → Option (Ins × Nat)}
    (oc : OuterCode im) (rc : RefuseCode im) :
    ∀ (w n sp : Nat), w + 56 ≤ W → n ≠ 0 → 48 ≤ sp → sp < W →
    ∀ s, s.rip = aCheckWord → s.reg 5 = some n → s.reg 7 = some w →
      s.reg 4 = some sp → rd64 s.mem (w + 24) = some 0 →
      RegsOk s → StackRoom s 6 →
      ∃ k, run im k (Out.ok s) = Out.exit 1 := by
  intro w n sp hw56 hn hsp48 hspW s hrip hr5 hr7 hr4 hfnc hregs hroom
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
  have e3 : X86Int.step im a2 = .ok { a2.set 0 (some 0) with rip := a2.rip + 4 } :=
    step_ld (by rw [ha2rip]; exact oc.cw3) ha2r7 (by rw [hea24, ha2mem]; exact hfnc)
  obtain ⟨a3, ha3⟩ : ∃ u : X86Int.St, u =
    { a2.set 0 (some 0) with rip := a2.rip + 4 } := ⟨_, rfl⟩
  have ha3rip : a3.rip = aCheckWord + 14 := by
    rw [ha3]; show a2.rip + 4 = aCheckWord + 14; rw [ha2rip]
  have ha3r0 : a3.reg 0 = some 0 := by
    rw [ha3]
    show (a2.set 0 (some 0)).reg 0 = some 0
    rw [reg_set_self (by
      have e : a2.regs.length = s.regs.length := by rw [ha2, ha1]
      have h5len : 5 < s.regs.length := reg_some_len hr5
      omega)]
  have ha3r7 : a3.reg 7 = some w := by
    rw [ha3]
    show (a2.set 0 (some 0)).reg 7 = some w
    rw [reg_set_other (by decide)]; exact ha2r7
  have ha3r5 : a3.reg 5 = some n := by
    rw [ha3]
    show (a2.set 0 (some 0)).reg 5 = some n
    rw [reg_set_other (by decide)]; exact ha2r5
  have ha3mem : a3.mem = s.mem := by
    rw [ha3]
    show (a2.set 0 (some 0)).mem = s.mem
    rw [mem_set]; exact ha2mem
  have ga3 : ∀ q, q ≠ 0 → a3.reg q = s.reg q := by
    intro q hq
    rw [ha3]
    show (a2.set 0 (some 0)).reg q = s.reg q
    rw [reg_set_other hq]; exact ga2 q
  have e4 : X86Int.step im a3 =
      .ok { a3 with fl := (aluF .and 0 0).2, rip := a3.rip + 3 } :=
    step_test (by rw [ha3rip]; exact oc.cw4) ha3r0 ha3r0
  obtain ⟨a4, ha4⟩ : ∃ u : X86Int.St, u =
    { a3 with fl := (aluF .and 0 0).2, rip := a3.rip + 3 } := ⟨_, rfl⟩
  have ha4rip : a4.rip = aCheckWord + 17 := by
    rw [ha4]; show a3.rip + 3 = aCheckWord + 17; rw [ha3rip]
  have ha4fl : X86Int.cond a4.fl 5 = some false := by
    have hz : a4.fl.zf = some true := by
      rw [ha4]
      show (aluF .and 0 0).2.zf = some true
      rw [aluF_zf, aluF_and_fst_self]
      exact rfl
    have := cond_cc5 hz
    simpa using this
  have ha4r0 : a4.reg 0 = some 0 := by
    rw [ha4]; show a3.reg 0 = some 0; exact ha3r0
  have ha4r7 : a4.reg 7 = some w := by
    rw [ha4]; show a3.reg 7 = some w; exact ha3r7
  have ha4r5 : a4.reg 5 = some n := by
    rw [ha4]; show a3.reg 5 = some n; exact ha3r5
  have ha4mem : a4.mem = s.mem := by
    rw [ha4]; show a3.mem = s.mem; exact ha3mem
  have ga4 : ∀ q, q ≠ 0 → a4.reg q = s.reg q := by
    intro q hq
    rw [ha4]; show a3.reg q = s.reg q; exact ga3 q hq
  have e5 : X86Int.step im a4 = .ok { a4 with rip := aCheckWord + 19 } := by
    have hh := step_jcc (im := im) (s := a4) (by rw [ha4rip]; exact oc.cw5) ha4fl
    simpa [ha4rip] using hh
  obtain ⟨a5, ha5⟩ : ∃ u : X86Int.St, u = { a4 with rip := aCheckWord + 19 } :=
    ⟨_, rfl⟩
  have ha5rip : a5.rip = aCheckWord + 19 := by rw [ha5]
  have e6 : X86Int.step im a5 = .ok { a5 with rip := aFail1 } :=
    step_jmp (by rw [ha5rip]; exact rc.cwEmpty)
  obtain ⟨a6, ha6⟩ : ∃ u : X86Int.St, u = { a5 with rip := aFail1 } := ⟨_, rfl⟩
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
  have hrun6 : run im 6 (Out.ok s) = Out.ok a6 := by
    rw [run_add im 1 5, r1, run_add im 1 4, r2, run_add im 1 3, r3,
      run_add im 1 2, r4, run_add im 1 1, r5, r6]
  have ha6rip : a6.rip = aFail1 := by rw [ha6]
  have ha6mem : a6.mem = s.mem := by
    rw [ha6, ha5]; exact ha4mem
  have ha6r4 : a6.reg 4 = some sp := by
    rw [ha6]; show ({ a5 with rip := aFail1 }).reg 4 = some sp
    rw [reg_with_rip]
    rw [ha5]; show ({ a4 with rip := aCheckWord + 19 }).reg 4 = some sp
    rw [reg_with_rip, ga4 4 (by decide)]; exact hr4
  obtain ⟨k7, hk7⟩ := exit1_of_fail1 rc (by omega) hspW a6 ha6rip ha6r4
    (run_regs_ok hrun6 hregs)
    (StackRoom_mono (StackRoom_of_mem ha6mem (by rw [ha6r4, hr4]) hroom) (by decide))
  exact ⟨6 + k7, by rw [run_add im 6 k7, hrun6]; exact hk7⟩

set_option maxHeartbeats 3200000 in
/-- One `annval` refusal: an entry whose index is at or beyond the
code length branches to `fail1` and exits 1. -/
theorem annval_range_exit1 {im : Nat → Option (Ins × Nat)}
    (ac : AllCode im) (oc : OuterCode im) (rc : RefuseCode im)
    {wd : Sporos.Word} {w p sp : Nat} (st : AnnStatic wd w p sp)
    {t : Nat} {h : Height} {rest' : List (Nat × Height)}
    (hlen : rest'.length + 1 ≤ wd.ann.length)
    (hge : wd.code.length ≤ t) (hmem : (t, h) ∈ wd.ann) :
    ∀ s, AnnSt wd w p sp ((t, h) :: rest') s →
      AnnAt s (p + 12 * (wd.ann.length - (rest'.length + 1))) t h.d h.a →
      t < W → RegsOk s →
      ∃ k, run im k (Out.ok s) = Out.exit 1 := by
  obtain ⟨hw56, hp0, hpb, hsp48, hspW, hncW, hhW, hdisjw, hdisjp⟩ := st
  intro s hs hcur htW hregs
  obtain ⟨hrip, hr7, hr6, hr1, hr4, hf24, hf32, hf48, hlay, h12len,
    hroom⟩ := hs
  have hnrem : rest'.length + 1 ≠ 0 := by omega
  have hcurb : p + 12 * (wd.ann.length - (rest'.length + 1)) + 12 ≤
      p + 12 * wd.ann.length := by
    have : wd.ann.length - (rest'.length + 1) + 1 ≤ wd.ann.length := by omega
    omega
  have hannW : wd.ann.length < W := by omega
  obtain ⟨hdW, haW⟩ : h.d < W ∧ h.a < W := hhW (t, h) hmem
  have hslot8 : (sp + W - 8) % W = sp - 8 := by
    rw [show sp + W - 8 = sp + W - 8 * 1 by omega, slot_sub (by omega) hspW]
  -- on-demand transports from a frame against `s`
  have layOf : ∀ u : X86Int.St, (∀ x n, x + n ≤ sp - 48 ∨ sp ≤ x →
      u.mem.rdN x n = s.mem.rdN x n) → AnnLayout u wd.ann p :=
    fun u hu => AnnLayout_of_frame hlay (fun x k hx1 hx2 => hu x k (by omega))
  have atOf : ∀ u : X86Int.St, (∀ x n, x + n ≤ sp - 48 ∨ sp ≤ x →
      u.mem.rdN x n = s.mem.rdN x n) →
      AnnAt u (p + 12 * (wd.ann.length - (rest'.length + 1))) t h.d h.a :=
    fun u hu => ⟨by
        rw [hu (p + 12 * (wd.ann.length - (rest'.length + 1))) 4 (by omega)];
        exact hcur.1,
      by
        rw [hu (p + 12 * (wd.ann.length - (rest'.length + 1)) + 4) 4
          (by omega)];
        exact hcur.2.1,
      by
        rw [hu (p + 12 * (wd.ann.length - (rest'.length + 1)) + 8) 4
          (by omega)];
        exact hcur.2.2⟩
  have fldOf : ∀ u : X86Int.St, (∀ x n, x + n ≤ sp - 48 ∨ sp ≤ x →
      u.mem.rdN x n = s.mem.rdN x n) →
      rd64 u.mem (w + 24) = some wd.code.length ∧
      rd64 u.mem (w + 32) = some wd.ann.length ∧
      rd64 u.mem (w + 48) = some p :=
    fun u hu => ⟨by unfold rd64; rw [hu (w + 24) 8 (by omega)]; exact hf24,
      by unfold rd64; rw [hu (w + 32) 8 (by omega)]; exact hf32,
      by unfold rd64; rw [hu (w + 48) 8 (by omega)]; exact hf48⟩
  have hfr0 : ∀ x n, x + n ≤ sp - 48 ∨ sp ≤ x → s.mem.rdN x n = s.mem.rdN x n :=
    fun x n _ => rfl
  have hw0 : wshape s.mem = wshape s.mem := rfl
  have hlen0 : s.regs.length = s.regs.length := rfl
  have gb0 : ∀ q, q ≠ 0 → q ≠ 1 → q ≠ 2 → q ≠ 4 → q ≠ 6 → q ≠ 8 → q ≠ 9 →
      q ≠ 12 → s.reg q = s.reg q :=
    fun q _ _ _ _ _ _ _ _ => rfl
  -- test rcx, rcx
  have hr1' : s.reg 1 = some (rest'.length + 1) := hr1
  have e1 : X86Int.step im s =
      .ok { s with fl := (aluF .and (rest'.length + 1) (rest'.length + 1)).2, rip := s.rip + 3 } :=
    step_test (by rw [hrip]; exact oc.av0) hr1' hr1'
  obtain ⟨b1, hb1⟩ : ∃ u : X86Int.St, u =
    { s with fl := (aluF .and (rest'.length + 1) (rest'.length + 1)).2, rip := s.rip + 3 } := ⟨_, rfl⟩
  have hb1rip : b1.rip = aAnnval + 3 := by
    rw [hb1]; show s.rip + 3 = aAnnval + 3; rw [hrip]
  have hb1fl : X86Int.cond b1.fl 5 = some true := by
    have hz : b1.fl.zf = some false := by
      rw [hb1]
      show (aluF .and (rest'.length + 1) (rest'.length + 1)).2.zf = some false
      rw [aluF_zf, aluF_and_fst_self]
      simp
    have := cond_cc5 hz
    simpa using this
  have gb1 : ∀ q, b1.reg q = s.reg q := fun q => by rw [hb1]; exact rfl
  have hb1mem : b1.mem = s.mem := by rw [hb1]
  -- jnz taken
  have e2 : X86Int.step im b1 = .ok { b1 with rip := aAnnval + 10 } := by
    have hh := step_jcc (im := im) (s := b1) (by rw [hb1rip]; exact oc.av1)
      hb1fl
    simpa using hh
  obtain ⟨b2, hb2⟩ : ∃ u : X86Int.St, u = { b1 with rip := aAnnval + 10 } :=
    ⟨_, rfl⟩
  have hb2rip : b2.rip = aAnnval + 10 := by rw [hb2]
  have gb2 : ∀ q, b2.reg q = s.reg q := fun q => by
    rw [hb2]; show b1.reg q = s.reg q; exact gb1 q
  have hb2mem : b2.mem = s.mem := by rw [hb2]; exact hb1mem
  have hb2r4 : b2.reg 4 = some sp := by rw [gb2 4]; exact hr4
  have hb2r1 : b2.reg 1 = some (rest'.length + 1) := by
    rw [gb2 1]; exact hr1'
  have hlen2 : b2.regs.length = s.regs.length := by rw [hb2, hb1]
  have hb2r6 : b2.reg 6 =
      some (p + 12 * (wd.ann.length - (rest'.length + 1))) := by
    rw [gb2 6]; exact hr6
  -- push rcx
  have hw8 : ∀ v, (wr64 s.mem (sp - 8) v).isSome := by
    intro v
    have h := hroom sp hr4 0 (by decide) v
    have e : (sp + W - 8 * (0 + 1)) % W = sp - 8 :=
      slot_sub (by omega) hspW
    rw [e] at h; exact h
  have hw16s : ∀ v, (wr64 s.mem (sp - 16) v).isSome := by
    intro v
    have h := hroom sp hr4 1 (by decide) v
    have e : (sp + W - 8 * (1 + 1)) % W = sp - 16 :=
      slot_sub (by omega) hspW
    rw [e] at h; exact h
  obtain ⟨m3, hm3s⟩ := Option.isSome_iff_exists.mp (hw8 (rest'.length + 1))
  have hm3 : wr64 b2.mem ((sp + W - 8) % W) (rest'.length + 1) = some m3 := by
    rw [hb2mem, hslot8]; exact hm3s
  have e3 : X86Int.step im b2 = .ok { b2.set 4 (some ((sp + W - 8) % W)) with
      mem := m3, rip := b2.rip + 1 } :=
    step_push (by rw [hb2rip]; exact oc.av3) hb2r1 hb2r4 hm3
  obtain ⟨b3, hb3⟩ : ∃ u : X86Int.St, u =
    { b2.set 4 (some ((sp + W - 8) % W)) with mem := m3, rip := b2.rip + 1 } :=
    ⟨_, rfl⟩
  have hb3rip : b3.rip = aAnnval + 11 := by
    rw [hb3]; show b2.rip + 1 = aAnnval + 11; rw [hb2rip]
  have hb3r4 : b3.reg 4 = some (sp - 8) := by
    rw [hb3]
    show (b2.set 4 (some ((sp + W - 8) % W))).reg 4 = some (sp - 8)
    rw [reg_set_self (by omega), hslot8]
  have gb3 : ∀ q, q ≠ 4 → b3.reg q = s.reg q := by
    intro q hq
    rw [hb3]
    show (b2.set 4 (some ((sp + W - 8) % W))).reg q = s.reg q
    rw [reg_set_other hq]; exact gb2 q
  have hb3mem : b3.mem = m3 := by rw [hb3]
  have hfr3 : ∀ x n, x + n ≤ sp - 48 ∨ sp ≤ x →
      b3.mem.rdN x n = s.mem.rdN x n := by
    intro x n hx
    rw [hb3mem]
    exact rdN_wr64_other hm3s (by omega)
  have hw3 : wshape b3.mem = wshape s.mem := by
    rw [hb3mem]; exact wrN_wshape _ _ _ hm3s
  have hlen3 : b3.regs.length = s.regs.length := by
    rw [hb3]
    show (b2.set 4 (some ((sp + W - 8) % W))).regs.length = s.regs.length
    exact List.length_set.trans hlen2
  have hslot8_3 : rd64 b3.mem (sp - 8) = some (rest'.length + 1) := by
    have h1 := rd64_wr64 hm3s
    rw [Nat.mod_eq_of_lt (by omega)] at h1
    rw [hb3mem]; exact h1
  -- push rsi
  obtain ⟨m4, hm4s⟩ := Option.isSome_iff_exists.mp
    (wr64_some_of_wr64 hm3s
      (hw16s (p + 12 * (wd.ann.length - (rest'.length + 1)))))
  have hsp16 : ((sp - 8) + W - 8) % W = sp - 16 := by
    rw [show (sp - 8) + W - 8 = (sp - 8) + W - 8 * 1 by omega,
      slot_sub (by omega) (by omega)]
    omega
  have hm4 : wr64 b3.mem (((sp - 8) + W - 8) % W)
      (p + 12 * (wd.ann.length - (rest'.length + 1))) = some m4 := by
    rw [hb3mem, hsp16]; exact hm4s
  have hb3r6 : b3.reg 6 =
      some (p + 12 * (wd.ann.length - (rest'.length + 1))) := by
    rw [gb3 6 (by decide)]; exact hr6
  have e4 : X86Int.step im b3 = .ok { b3.set 4 (some (((sp - 8) + W - 8) % W))
      with mem := m4, rip := b3.rip + 1 } :=
    step_push (by rw [hb3rip]; exact oc.av4) hb3r6 hb3r4 hm4
  obtain ⟨b4, hb4⟩ : ∃ u : X86Int.St, u =
    { b3.set 4 (some (((sp - 8) + W - 8) % W)) with mem := m4, rip := b3.rip + 1 } := ⟨_, rfl⟩
  have hb4rip : b4.rip = aAnnval + 12 := by
    rw [hb4]; show b3.rip + 1 = aAnnval + 12; rw [hb3rip]
  have hb4r4 : b4.reg 4 = some (sp - 16) := by
    rw [hb4]
    show (b3.set 4 (some (((sp - 8) + W - 8) % W))).reg 4 = some (sp - 16)
    rw [reg_set_self (by omega), hsp16]
  have gb4 : ∀ q, q ≠ 4 → b4.reg q = s.reg q := by
    intro q hq
    rw [hb4]
    show (b3.set 4 (some (((sp - 8) + W - 8) % W))).reg q = s.reg q
    rw [reg_set_other hq]; exact gb3 q hq
  have hb4mem : b4.mem = m4 := by rw [hb4]
  have hfr4 : ∀ x n, x + n ≤ sp - 48 ∨ sp ≤ x →
      b4.mem.rdN x n = s.mem.rdN x n := by
    intro x n hx
    rw [hb4mem, rdN_wr64_other hm4s (by omega), ← hb3mem]
    exact hfr3 x n hx
  have hw4 : wshape b4.mem = wshape s.mem := by
    rw [hb4mem]
    exact (wrN_wshape _ _ _ hm4s).trans (by rw [← hb3mem]; exact hw3)
  have hlen4 : b4.regs.length = s.regs.length := by
    rw [hb4]
    show (b3.set 4 (some (((sp - 8) + W - 8) % W))).regs.length =
      s.regs.length
    exact List.length_set.trans hlen3
  have hslot16_4 : rd64 b4.mem (sp - 16) =
      some (p + 12 * (wd.ann.length - (rest'.length + 1))) := by
    have h1 := rd64_wr64 hm4s
    rw [Nat.mod_eq_of_lt (by omega)] at h1
    rw [hb4mem]; exact h1
  have hslot8_4 : rd64 b4.mem (sp - 8) = some (rest'.length + 1) := by
    rw [hb4mem, rd64_wr64_other hm4s (by omega)]
    rw [← hb3mem]; exact hslot8_3
  -- mov rdx, rsi
  have hb4r6 : b4.reg 6 =
      some (p + 12 * (wd.ann.length - (rest'.length + 1))) := by
    rw [gb4 6 (by decide)]; exact hr6
  have e5 : X86Int.step im b4 = .ok { b4.set 2
      (some (p + 12 * (wd.ann.length - (rest'.length + 1)))) with
      rip := b4.rip + 3 } :=
    step_mov (by rw [hb4rip]; exact oc.av5) hb4r6
  obtain ⟨b5, hb5⟩ : ∃ u : X86Int.St, u = { b4.set 2
      (some (p + 12 * (wd.ann.length - (rest'.length + 1)))) with
      rip := b4.rip + 3 } := ⟨_, rfl⟩
  have hb5rip : b5.rip = aAnnval + 15 := by
    rw [hb5]; show b4.rip + 3 = aAnnval + 15; rw [hb4rip]
  have hb5r2 : b5.reg 2 =
      some (p + 12 * (wd.ann.length - (rest'.length + 1))) := by
    rw [hb5]
    show (b4.set 2 (some (p + 12 * (wd.ann.length - (rest'.length + 1))))).reg 2 = _
    rw [reg_set_self (by omega)]
  have gb5 : ∀ q, q ≠ 2 → b5.reg q = b4.reg q := by
    intro q hq
    rw [hb5]
    show (b4.set 2 _).reg q = b4.reg q
    rw [reg_set_other hq]
  have hb5mem : b5.mem = b4.mem := by rw [hb5]; exact rfl
  have hb5r4 : b5.reg 4 = some (sp - 16) := by
    rw [gb5 4 (by decide)]; exact hb4r4
  have hlen5 : b5.regs.length = s.regs.length := by
    rw [hb5]
    show (b4.set 2 _).regs.length = s.regs.length
    exact List.length_set.trans hlen4
  have hfr5 : ∀ x n, x + n ≤ sp - 48 ∨ sp ≤ x →
      b5.mem.rdN x n = s.mem.rdN x n := by
    intro x n hx; rw [hb5mem]; exact hfr4 x n hx
  have hw5 : wshape b5.mem = wshape s.mem := by rw [hb5mem]; exact hw4
  -- call L32: rax = this entry's index t
  have hb5r1 : b5.reg 1 = some (rest'.length + 1) := by
    rw [gb5 1 (by decide), gb4 1 (by decide)]; exact hr1'
  have hrd5 : (b5.mem.rdN (p + 12 * (wd.ann.length - (rest'.length + 1))) 4).map
      leVal = some t := (atOf b5 hfr5).1
  obtain ⟨k6, b6, hrun6, hb6rip, hb6r0, hb6r4, gb6, hlen6, hfr6seg, hw6seg⟩ :=
    CallL32 ac.l32 ac.annl.dret oc.av6 (by omega) (by omega) (by omega)
      (by decide) (by omega) b5 hb5rip hb5r2 hb5r4 hrd5
      (StackRoom_shift hroom hr4 hb5r4 hw5 ⟨2, rfl, by omega⟩ hsp48 hspW)
  have hlen6s : b6.regs.length = s.regs.length := hlen6.trans hlen5
  have hfr6 : ∀ x n, x + n ≤ sp - 48 ∨ sp ≤ x →
      b6.mem.rdN x n = s.mem.rdN x n := by
    intro x n hx
    exact (hfr6seg x n (by omega)).trans (hfr5 x n hx)
  have hw6 : wshape b6.mem = wshape s.mem := hw6seg.trans hw5
  have hb6r7 : b6.reg 7 = some w := by
    rw [gb6 7 (by decide) (by decide), gb5 7 (by decide), gb4 7 (by decide)]
    exact hr7
  have hb6r1 : b6.reg 1 = some (rest'.length + 1) := by
    rw [gb6 1 (by decide) (by decide)]; exact hb5r1
  have hb6r6 : b6.reg 6 =
      some (p + 12 * (wd.ann.length - (rest'.length + 1))) := by
    rw [gb6 6 (by decide) (by decide), gb5 6 (by decide)]; exact hb4r6
  have hslot16_6 : rd64 b6.mem (sp - 16) =
      some (p + 12 * (wd.ann.length - (rest'.length + 1))) := by
    unfold rd64
    rw [hfr6seg (sp - 16) 8 (by omega), hb5mem]
    exact hslot16_4
  have hslot8_6 : rd64 b6.mem (sp - 8) = some (rest'.length + 1) := by
    unfold rd64
    rw [hfr6seg (sp - 8) 8 (by omega), hb5mem]
    exact hslot8_4
  -- mov r12, rax
  have e7 : X86Int.step im b6 = .ok { b6.set 12 (some t) with
      rip := b6.rip + 3 } :=
    step_mov (by rw [hb6rip]; exact oc.av7) hb6r0
  obtain ⟨b7, hb7⟩ : ∃ u : X86Int.St, u = { b6.set 12 (some t) with
    rip := b6.rip + 3 } := ⟨_, rfl⟩
  have hb7rip : b7.rip = aAnnval + 23 := by
    rw [hb7]; show b6.rip + 3 = aAnnval + 23; rw [hb6rip]
  have hb7r12 : b7.reg 12 = some t := by
    rw [hb7]
    show (b6.set 12 (some t)).reg 12 = some t
    rw [reg_set_self (by omega)]
  have gb7 : ∀ q, q ≠ 12 → b7.reg q = b6.reg q := by
    intro q hq
    rw [hb7]
    show (b6.set 12 (some t)).reg q = b6.reg q
    rw [reg_set_other hq]
  have hb7mem : b7.mem = b6.mem := by rw [hb7]; exact rfl
  have hlen7 : b7.regs.length = s.regs.length := by
    rw [hb7]
    show (b6.set 12 (some t)).regs.length = s.regs.length
    exact List.length_set.trans hlen6s
  have hfr7 : ∀ x n, x + n ≤ sp - 48 ∨ sp ≤ x →
      b7.mem.rdN x n = s.mem.rdN x n := by
    intro x n hx; rw [hb7mem]; exact hfr6 x n hx
  have hw7 : wshape b7.mem = wshape s.mem := by rw [hb7mem]; exact hw6
  -- mov rdx, [rdi+24]
  have hb7r7 : b7.reg 7 = some w := by
    rw [gb7 7 (by decide)]; exact hb6r7
  have hea24 : ea w 24 = w + 24 := Nat.mod_eq_of_lt (by omega)
  have e8 : X86Int.step im b7 = .ok { b7.set 2 (some wd.code.length) with
      rip := b7.rip + 4 } :=
    step_ld (by rw [hb7rip]; exact oc.av8) hb7r7
      (by rw [hea24]; exact (fldOf b7 hfr7).1)
  obtain ⟨b8, hb8⟩ : ∃ u : X86Int.St, u = { b7.set 2 (some wd.code.length) with
    rip := b7.rip + 4 } := ⟨_, rfl⟩
  have hb8rip : b8.rip = aAnnval + 27 := by
    rw [hb8]; show b7.rip + 4 = aAnnval + 27; rw [hb7rip]
  have gb8 : ∀ q, q ≠ 2 → b8.reg q = b7.reg q := by
    intro q hq
    rw [hb8]
    show (b7.set 2 (some wd.code.length)).reg q = b7.reg q
    rw [reg_set_other hq]
  have hb8r2 : b8.reg 2 = some wd.code.length := by
    rw [hb8]
    show (b7.set 2 (some wd.code.length)).reg 2 = some wd.code.length
    rw [reg_set_self (by omega)]
  have hb8r0 : b8.reg 0 = some t := by
    rw [gb8 0 (by decide), gb7 0 (by decide)]; exact hb6r0
  have hb8mem : b8.mem = b7.mem := by rw [hb8]; exact rfl
  have hlen8 : b8.regs.length = s.regs.length := by
    rw [hb8]
    show (b7.set 2 (some wd.code.length)).regs.length = s.regs.length
    exact List.length_set.trans hlen7
  have hfr8 : ∀ x n, x + n ≤ sp - 48 ∨ sp ≤ x →
      b8.mem.rdN x n = s.mem.rdN x n := by
    intro x n hx; rw [hb8mem]; exact hfr7 x n hx
  have hw8 : wshape b8.mem = wshape s.mem := by rw [hb8mem]; exact hw7
  -- cmp rax, rdx ; jb taken
  have e9 : X86Int.step im b8 =
      .ok { b8 with fl := (aluF .sub t wd.code.length).2, rip := b8.rip + 3 } :=
    step_cmp (by rw [hb8rip]; exact oc.av9) hb8r0 hb8r2
  obtain ⟨b9, hb9⟩ : ∃ u : X86Int.St, u =
    { b8 with fl := (aluF .sub t wd.code.length).2, rip := b8.rip + 3 } :=
    ⟨_, rfl⟩
  have hb9rip : b9.rip = aAnnval + 30 := by
    rw [hb9]; show b8.rip + 3 = aAnnval + 30; rw [hb8rip]
  have hb9fl : X86Int.cond b9.fl 2 = some false := by
    have hcf : b9.fl.cf = some false := by
      rw [hb9]
      show (aluF .sub t wd.code.length).2.cf = some false
      exact aluF_sub_cf (by omega)
    exact cond_cc2 hcf
  have gb9 : ∀ q, b9.reg q = b8.reg q := fun q => by rw [hb9]; exact rfl
  have hb9mem : b9.mem = b8.mem := by rw [hb9]
  have e10 : X86Int.step im b9 = .ok { b9 with rip := aAnnval + 32 } := by
    have hh := step_jcc (im := im) (s := b9) (by rw [hb9rip]; exact oc.av10)
      hb9fl
    simpa [hb9rip] using hh
  obtain ⟨b10, hb10⟩ : ∃ u : X86Int.St, u = { b9 with rip := aAnnval + 32 } :=
    ⟨_, rfl⟩
  have hb10rip : b10.rip = aAnnval + 32 := by rw [hb10]
  have e11 : X86Int.step im b10 = .ok { b10 with rip := aFail1 } :=
    step_jmp (by rw [hb10rip]; exact rc.avRange)
  obtain ⟨b11, hb11⟩ : ∃ u : X86Int.St, u = { b10 with rip := aFail1 } := ⟨_, rfl⟩
  have q1 : run im 1 (Out.ok s) = Out.ok b1 := by
    rw [run_succ_ok im 0 s, e1, ← hb1, run_zero]
  have q2 : run im 1 (Out.ok b1) = Out.ok b2 := by
    rw [run_succ_ok im 0 b1, e2, ← hb2, run_zero]
  have q3 : run im 1 (Out.ok b2) = Out.ok b3 := by
    rw [run_succ_ok im 0 b2, e3, ← hb3, run_zero]
  have q4 : run im 1 (Out.ok b3) = Out.ok b4 := by
    rw [run_succ_ok im 0 b3, e4, ← hb4, run_zero]
  have q5 : run im 1 (Out.ok b4) = Out.ok b5 := by
    rw [run_succ_ok im 0 b4, e5, ← hb5, run_zero]
  have q7 : run im 1 (Out.ok b6) = Out.ok b7 := by
    rw [run_succ_ok im 0 b6, e7, ← hb7, run_zero]
  have q8 : run im 1 (Out.ok b7) = Out.ok b8 := by
    rw [run_succ_ok im 0 b7, e8, ← hb8, run_zero]
  have q9 : run im 1 (Out.ok b8) = Out.ok b9 := by
    rw [run_succ_ok im 0 b8, e9, ← hb9, run_zero]
  have q10 : run im 1 (Out.ok b9) = Out.ok b10 := by
    rw [run_succ_ok im 0 b9, e10, ← hb10, run_zero]
  have q11 : run im 1 (Out.ok b10) = Out.ok b11 := by
    rw [run_succ_ok im 0 b10, e11, ← hb11, run_zero]
  have hrunT : run im (5 + (k6 + 5)) (Out.ok s) = Out.ok b11 := by
    rw [run_add im 5 (k6 + 5),
      run_add im 1 4, q1, run_add im 1 3, q2, run_add im 1 2, q3,
        run_add im 1 1, q4, q5,
      run_add im k6 5, hrun6,
      run_add im 1 4, q7, run_add im 1 3, q8, run_add im 1 2, q9,
        run_add im 1 1, q10, q11]
  have hb11rip : b11.rip = aFail1 := by rw [hb11]
  have hb11r4 : b11.reg 4 = some (sp - 16) := by
    rw [hb11]; show ({ b10 with rip := aFail1 }).reg 4 = some (sp - 16)
    rw [reg_with_rip]
    rw [hb10]; show ({ b9 with rip := aAnnval + 32 }).reg 4 = some (sp - 16)
    rw [reg_with_rip, gb9 4, gb8 4 (by decide), gb7 4 (by decide)]
    exact hb6r4
  have hw11 : wshape b11.mem = wshape s.mem := by
    rw [hb11, hb10, hb9mem]
    exact hw8
  obtain ⟨k12, hk12⟩ := exit1_of_fail1 rc (by omega) (by omega) b11 hb11rip
    hb11r4 (run_regs_ok hrunT hregs)
    (StackRoom_shift hroom hr4 hb11r4 hw11 ⟨2, rfl, by omega⟩ hsp48 hspW)
  exact ⟨5 + (k6 + 5) + k12, by
    rw [run_add im (5 + (k6 + 5)) k12, hrunT]; exact hk12⟩

/-- Consume a prefix of good annotation entries, leaving the
`annval` state at the remaining target. -/
theorem annval_prefix {im : Nat → Option (Ins × Nat)}
    (ac : AllCode im) (oc : OuterCode im)
    {wd : Sporos.Word} {w p sp : Nat} (st : AnnStatic wd w p sp) :
    ∀ (good target seen : List (Nat × Height)),
      wd.ann = seen ++ (good ++ target) →
      (∀ x ∈ good, x.1 < wd.code.length ∧ wd.ann.lookup x.1 = some x.2) →
      ∀ s, AnnSt wd w p sp (good ++ target) s →
      ∃ k s', run im k (Out.ok s) = Out.ok s' ∧ AnnSt wd w p sp target s' ∧
        (∀ q, q ≠ 0 → q ≠ 1 → q ≠ 2 → q ≠ 4 → q ≠ 6 → q ≠ 8 → q ≠ 9 →
          q ≠ 12 → s'.reg q = s.reg q) ∧
        (∀ x n, x + n ≤ sp - 48 ∨ sp ≤ x → s'.mem.rdN x n = s.mem.rdN x n) ∧
        wshape s'.mem = wshape s.mem ∧ s'.regs.length = s.regs.length := by
  intro good
  induction good with
  | nil =>
    intro target seen happ hgood s hs
    exact ⟨0, s, rfl, hs, fun q _ _ _ _ _ _ _ _ => rfl,
      fun x n _ => rfl, rfl, rfl⟩
  | cons th good ih =>
    obtain ⟨t, h⟩ := th
    intro target seen happ hgood s hs
    have hs0 : AnnSt wd w p sp ((t, h) :: (good ++ target)) s := hs
    obtain ⟨hrip, hr7, hr6, hr1, hr4, hf24, hf32, hf48, hlay, h12len,
      hroom⟩ := hs
    have hlenann : wd.ann.length =
        seen.length + ((good ++ target).length + 1) := by
      have hgt : (good ++ target).length = good.length + target.length :=
        List.length_append
      have hlen : wd.ann.length =
          seen.length + (good.length + target.length + 1) := by
        simp [happ, List.length_append]
      omega
    have hhead : t < wd.code.length ∧ wd.ann.lookup t = some h :=
      hgood (t, h) List.mem_cons_self
    have hi0 : seen.length < wd.ann.length := by omega
    obtain ⟨hat, htW⟩ := hlay seen.length hi0
    have hget? : wd.ann[seen.length]? = some (t, h) := by
      rw [happ, List.getElem?_append_right (by simp)]
      simp
    have hget : wd.ann[seen.length]'hi0 = (t, h) :=
      Option.some.inj ((List.getElem?_eq_getElem hi0).symm.trans hget?)
    have hcur : AnnAt s
        (p + 12 * (wd.ann.length - ((good ++ target).length + 1))) t h.d h.a := by
      have e : p + 12 * (wd.ann.length - ((good ++ target).length + 1)) =
          p + 12 * seen.length := by
        congr 1
        congr 1
        omega
      rw [e]
      rw [hget] at hat
      exact hat
    have htW' : t < W := by
      rw [hget] at htW
      exact htW
    obtain ⟨k1, s1, hrun1, hs1, gfr1, hfr1, hw1, hlen1⟩ :=
      annval_step ac oc st (by omega) hhead.1 hhead.2 s hs0 hcur htW'
    have happ' : wd.ann = (seen ++ [(t, h)]) ++ (good ++ target) := by
      simpa [List.append_assoc] using happ
    obtain ⟨k2, s2, hrun2, hs2, gfr2, hfr2, hw2, hlen2⟩ :=
      ih target (seen ++ [(t, h)]) happ'
        (fun x hx => hgood x (List.mem_cons_of_mem _ hx)) s1 hs1
    exact ⟨k1 + k2, s2, by rw [run_add im k1 k2, hrun1, hrun2], hs2,
      fun q h0 h1 h2 h4 h6 h8 h9 h12 =>
        (gfr2 q h0 h1 h2 h4 h6 h8 h9 h12).trans
          (gfr1 q h0 h1 h2 h4 h6 h8 h9 h12),
      fun x n hx => (hfr2 x n hx).trans (hfr1 x n hx),
      hw2.trans hw1, hlen2.trans hlen1⟩

/-- The `annval` loop refuses the first out-of-range entry after a
prefix of good entries. -/
theorem annval_range_loop_exit1 {im : Nat → Option (Ins × Nat)}
    (ac : AllCode im) (oc : OuterCode im) (rc : RefuseCode im)
    {wd : Sporos.Word} {w p sp : Nat} (st : AnnStatic wd w p sp)
    {good tail : List (Nat × Height)} {t : Nat} {h : Height}
    (happ : wd.ann = good ++ (t, h) :: tail)
    (hgood : ∀ x ∈ good, x.1 < wd.code.length ∧
      wd.ann.lookup x.1 = some x.2)
    (hge : wd.code.length ≤ t) :
    ∀ s, AnnSt wd w p sp wd.ann s → RegsOk s →
      ∃ k, run im k (Out.ok s) = Out.exit 1 := by
  intro s hs hregs
  have hs' : AnnSt wd w p sp (good ++ (t, h) :: tail) s := by
    rwa [← happ]
  obtain ⟨k1, s1, hrun1, hs1, gfr1, hfr1, hw1, hlen1⟩ :=
    annval_prefix ac oc st good ((t, h) :: tail) [] (by simpa using happ)
      hgood s hs'
  have hs1c := hs1
  obtain ⟨hrip, hr7, hr6, hr1, hr4, hf24, hf32, hf48, hlay, h12len,
    hroom⟩ := hs1
  have hlenann : wd.ann.length = good.length + (tail.length + 1) := by
    rw [happ, List.length_append, List.length_cons]
  have hi0 : good.length < wd.ann.length := by omega
  obtain ⟨hat, htW⟩ := hlay good.length hi0
  have hget? : wd.ann[good.length]? = some (t, h) := by
    rw [happ, List.getElem?_append_right (by simp)]
    simp
  have hget : wd.ann[good.length]'hi0 = (t, h) :=
    Option.some.inj ((List.getElem?_eq_getElem hi0).symm.trans hget?)
  have hcur : AnnAt s1
      (p + 12 * (wd.ann.length - (tail.length + 1))) t h.d h.a := by
    have e : p + 12 * (wd.ann.length - (tail.length + 1)) =
        p + 12 * good.length := by
      congr 1
      congr 1
      omega
    rw [e]
    rw [hget] at hat
    exact hat
  have htW' : t < W := by
    rw [hget] at htW
    exact htW
  have hmem : (t, h) ∈ wd.ann := by
    rw [happ]
    exact List.mem_append_right _ List.mem_cons_self
  obtain ⟨k2, hk2⟩ := annval_range_exit1 ac oc rc st (by omega) hge hmem s1
    hs1c hcur htW' (run_regs_ok hrun1 hregs)
  exact ⟨k1 + k2, by rw [run_add im k1 k2, hrun1]; exact hk2⟩

end SporosX1
