/-
SporosCheckBin: end-to-end machine-code verification of `spr1_check`
(`bootstrap/model/spr1_check`, modeled as `SporosCheckBin.im` and `spr1Alts`).

Composes `SporosCheckBinTail`, `SporosCheckBinFault`, and `SporosCheckFlat` to
prove:
1. `checkFlatA_eq_checkFlat_of_lt_HMAX`: exact agreement between the streaming
   checker specification `checkFlatA` and `SporosCheckFlat.checkFlat` on all
   byte slices of length `< 2 ^ 48`.
2. `spr1_bin_sound`, `spr1_bin_refuse_sound`, and `spr1_bin_complete`:
   universal soundness, refusal soundness, and completeness from ELF entry
   `initState` against `Sporos.decode` and `Sporos.checkProg`.
3. `spr1_bin_no_single_fault_false_accept`: universal single-disturbance
   immunity (`reg`, `mem`, `flag` across Passes 1–3 and tail, plus `fetch` via
   `X86Dec.altsAt` in the voter/guard tail) from any `TripledState`, together
   with the explicit non-vacuity witness `mkTripledState_exists`.
4. Ground `decide +kernel` and `spr1_run_flat_canon` witnesses on
   `Sporos.Examples.{echo,hello,upper,fib}` and invalid/refused inputs.
-/
import Bootstrap.SporosCheckBinTail
import Bootstrap.SporosCheckBinFault
import Bootstrap.SporosCheckFlat

namespace SporosCheckBin

open X86Env X86Int

/-! ### Exact equivalence of `checkFlatA` and `SporosCheckFlat.checkFlat` below `HMAX` -/

private theorem u32_eq_rdU32 (bs : List Nat) {pos : Nat} (h : pos + 4 ≤ bs.length) :
    SporosCheckFlat.rdU32 bs pos = some (u32 bs pos) := by
  have h0 : pos < bs.length := by omega
  have h1 : pos + 1 < bs.length := by omega
  have h2 : pos + 2 < bs.length := by omega
  have h3 : pos + 3 < bs.length := by omega
  have hg0 : bs[pos]? = some bs[pos] := List.getElem?_eq_getElem h0
  have hg1 : bs[pos + 1]? = some bs[pos + 1] := List.getElem?_eq_getElem h1
  have hg2 : bs[pos + 2]? = some bs[pos + 2] := List.getElem?_eq_getElem h2
  have hg3 : bs[pos + 3]? = some bs[pos + 3] := List.getElem?_eq_getElem h3
  have hdt : (bs.drop pos).take 4 = [bs[pos], bs[pos + 1], bs[pos + 2], bs[pos + 3]] := by
    rw [List.drop_eq_getElem_cons h0, List.drop_eq_getElem_cons h1,
        List.drop_eq_getElem_cons h2, List.drop_eq_getElem_cons h3]
    rfl
  simp only [SporosCheckFlat.rdU32, bind, Option.bind, pure, hg0, hg1, hg2, hg3, u32, hdt, leVal]
  congr 1; omega

private theorem rdU32_none_of_lt (bs : List Nat) {pos : Nat} (h : bs.length < pos + 4) :
    SporosCheckFlat.rdU32 bs pos = none := by
  cases hr : SporosCheckFlat.rdU32 bs pos with
  | none => rfl
  | some v => exact False.elim (by have := SporosCheckFlat.rdU32_bounds hr; omega)

private theorem skipDataA_eq_skipData (bs : List Nat) :
    ∀ n pos, skipDataA bs n pos = SporosCheckFlat.skipData bs n pos
  | 0, pos => rfl
  | n + 1, pos => by
    simp only [skipDataA, SporosCheckFlat.skipData, bind, Option.bind]
    by_cases h4 : bs.length < pos + 4
    · rw [if_pos h4, rdU32_none_of_lt bs h4]
    · have h4' : pos + 4 ≤ bs.length := by omega
      rw [if_neg h4, u32_eq_rdU32 bs h4']
      dsimp only
      by_cases hl : bs.length - (pos + 4) < u32 bs pos
      · rw [if_pos hl, if_neg (by omega)]
      · rw [if_neg hl, if_pos (by omega)]
        exact skipDataA_eq_skipData bs n (pos + 4 + u32 bs pos)

private theorem instrLenA_eq_instrLen_small :
    (List.range 256).all (fun op => instrLenA op == SporosCheckFlat.instrLen op) = true := by
  decide +kernel

private theorem instrLenA_eq_instrLen (op : Nat) :
    instrLenA op = SporosCheckFlat.instrLen op := by
  by_cases h : op < 256
  · have ha := List.all_eq_true.1 instrLenA_eq_instrLen_small op (List.mem_range.2 h)
    exact beq_iff_eq.1 ha
  · unfold instrLenA SporosCheckFlat.instrLen
    rw [if_neg (by omega), if_neg (by omega), if_neg (by omega), if_pos (by omega)]
    simp only [Bool.or_eq_true, beq_iff_eq]
    rw [if_neg (by omega), if_neg (by omega), if_neg (by omega), if_neg (by omega)]

private theorem skipCodeA_eq_skipCode (bs : List Nat) :
    ∀ n pos, skipCodeA bs n pos = SporosCheckFlat.skipCode bs n pos
  | 0, pos => rfl
  | n + 1, pos => by
    simp only [skipCodeA, SporosCheckFlat.skipCode, SporosCheckFlat.skipInstr, bind, Option.bind]
    by_cases hp : pos < bs.length
    · rw [dif_pos hp, List.getElem?_eq_getElem hp]
      dsimp only
      rw [instrLenA_eq_instrLen bs[pos]]
      cases hl : SporosCheckFlat.instrLen bs[pos] with
      | none => rfl
      | some l =>
        dsimp only
        by_cases hpl : bs.length < pos + l
        · rw [if_pos hpl, if_neg (by omega)]
        · rw [if_neg hpl, if_pos (by omega)]
          exact skipCodeA_eq_skipCode bs n (pos + l)
    · rw [dif_neg hp, List.getElem?_eq_none (by omega)]

private theorem skipWordsA_eq_skipWords (bs : List Nat) :
    ∀ n pos, skipWordsA bs n pos = SporosCheckFlat.skipWords bs n pos
  | 0, pos => rfl
  | n + 1, pos => by
    simp only [skipWordsA, SporosCheckFlat.skipWords, bind, Option.bind]
    by_cases h20 : bs.length < pos + 20
    · rw [if_pos h20, if_neg (by omega)]
    · have h20' : pos + 20 ≤ bs.length := by omega
      rw [if_neg h20, if_pos h20', u32_eq_rdU32 bs (by omega), u32_eq_rdU32 bs (by omega)]
      dsimp only
      rw [skipCodeA_eq_skipCode bs (u32 bs (pos + 12)) (pos + 20)]
      cases hc : SporosCheckFlat.skipCode bs (u32 bs (pos + 12)) (pos + 20) with
      | none => rfl
      | some p =>
        dsimp only
        have hm : 12 * u32 bs (pos + 16) = u32 bs (pos + 16) * 12 := Nat.mul_comm _ _
        rw [hm]
        by_cases hp : bs.length < p + u32 bs (pos + 16) * 12
        · rw [if_pos hp, if_neg (by omega)]
        · rw [if_neg hp, if_pos (by omega)]
          exact skipWordsA_eq_skipWords bs n (p + u32 bs (pos + 16) * 12)

private theorem parseA_eq_parse (bs : List Nat) :
    parseA bs = SporosCheckFlat.parse bs := by
  simp only [parseA, SporosCheckFlat.parse, bind, Option.bind]
  by_cases h16 : bs.length < 16
  · rw [if_pos h16, if_pos h16]
  · rw [if_neg h16, if_neg h16, u32_eq_rdU32 bs (by omega), u32_eq_rdU32 bs (by omega),
      u32_eq_rdU32 bs (by omega), u32_eq_rdU32 bs (by omega)]
    dsimp only
    have hmagic : SporosCheckFlat.MAGIC = 0x31525053 := rfl
    rw [hmagic]
    by_cases hm : u32 bs 0 = 0x31525053
    · have hf1 : (u32 bs 0 ≠ 0x31525053) = False := by simp [hm]
      have hf2 : (u32 bs 0 != 0x31525053) = false := by simp [hm]
      rw [if_neg (by simp [hm]), if_neg (by simp [hm])]
      rw [skipDataA_eq_skipData]
      cases hsd : SporosCheckFlat.skipData bs (u32 bs 12) 16 with
      | none => rfl
      | some ws =>
        dsimp only
        rw [skipWordsA_eq_skipWords]
        cases hsw : SporosCheckFlat.skipWords bs (u32 bs 8) ws with
        | none => rfl
        | some ep =>
          dsimp only
          by_cases hep : ep = bs.length
          · simp [hep]
          · simp [hep]
    · rw [if_pos hm, if_pos (by simp [hm])]

private theorem lookupA_eq_lookup (bs : List Nat) (t : Nat) :
    ∀ k q, q + 12 * k ≤ bs.length →
      SporosCheckFlat.lookup bs q k t =
        (lookupA bs t k q).map (fun q' => (u32 bs (q' + 4), u32 bs (q' + 8)))
  | 0, _, _ => rfl
  | k + 1, q, hq => by
    simp only [SporosCheckFlat.lookup, SporosCheckFlat.lookupLoop, lookupA, bind, Option.bind, pure]
    rw [u32_eq_rdU32 bs (by omega)]
    dsimp only
    by_cases heq : u32 bs q = t
    · simp only [heq, ↓reduceIte, beq_self_eq_true]
      rw [u32_eq_rdU32 bs (by omega), u32_eq_rdU32 bs (by omega)]
      rfl
    · have hne : (u32 bs q == t) = false := beq_false_of_ne heq
      rw [if_neg (by simp [hne]), if_neg heq]
      exact lookupA_eq_lookup bs t k (q + 12) (by omega)

private theorem readWordAt_eq_step (bs : List Nat) {cw nextOff : Nat}
    (h1 : skipWordsA bs 1 cw = some nextOff) :
    ∃ wm annOff,
      skipCodeA bs (u32 bs (cw + 12)) (cw + 20) = some annOff ∧
      SporosCheckFlat.readWordAt bs cw = some wm ∧
      wm.ins = u32 bs cw ∧ wm.outs = u32 bs (cw + 4) ∧ wm.locals = u32 bs (cw + 8) ∧
      wm.nc = u32 bs (cw + 12) ∧ wm.na = u32 bs (cw + 16) ∧
      wm.codeOff = cw + 20 ∧ wm.annOff = annOff ∧ wm.nextOff = nextOff ∧
      wm.annOff + 12 * wm.na ≤ bs.length := by
  simp only [skipWordsA] at h1
  split at h1
  · simp at h1
  · rename_i hn20
    have h20 : cw + 20 ≤ bs.length := by omega
    cases hc : skipCodeA bs (u32 bs (cw + 12)) (cw + 20) with
    | none => simp [hc] at h1
    | some annOff =>
      simp only [hc] at h1
      split at h1
      · simp at h1
      · rename_i hna
        simp only [Option.some.injEq] at h1
        have h12a : annOff + 12 * u32 bs (cw + 16) ≤ bs.length := by omega
        have hr0 := u32_eq_rdU32 bs (pos := cw) (by omega)
        have hr4 := u32_eq_rdU32 bs (pos := cw + 4) (by omega)
        have hr8 := u32_eq_rdU32 bs (pos := cw + 8) (by omega)
        have hr12 := u32_eq_rdU32 bs (pos := cw + 12) (by omega)
        have hr16 := u32_eq_rdU32 bs (pos := cw + 16) (by omega)
        have hread : SporosCheckFlat.readWordAt bs cw =
            some ⟨u32 bs cw, u32 bs (cw + 4), u32 bs (cw + 8), u32 bs (cw + 12), u32 bs (cw + 16),
              cw + 20, annOff, annOff + u32 bs (cw + 16) * 12⟩ := by
          simp only [SporosCheckFlat.readWordAt, if_pos h20, bind, Option.bind,
            hr0, hr4, hr8, hr12, hr16, ← skipCodeA_eq_skipCode, hc]
          rw [if_pos (by rw [Nat.mul_comm] at h12a; exact h12a)]
        refine ⟨_, annOff, rfl, hread, rfl, rfl, rfl, rfl, rfl, rfl, rfl, ?_, h12a⟩
        rw [← h1, Nat.mul_comm]

private theorem wordAt_eq_skipWordsA (bs : List Nat) :
    ∀ (w nw ws endPos : Nat), w < nw → skipWordsA bs nw ws = some endPos →
      ∃ cw wm, skipWordsA bs w ws = some cw ∧
        SporosCheckFlat.wordAt bs ws w = some wm ∧
        wm.ins = u32 bs cw ∧ wm.outs = u32 bs (cw + 4) ∧ wm.locals = u32 bs (cw + 8) ∧
        wm.nc = u32 bs (cw + 12) ∧ wm.na = u32 bs (cw + 16) ∧
        wm.codeOff = cw + 20 ∧ skipCodeA bs wm.nc wm.codeOff = some wm.annOff ∧
        wm.annOff + 12 * wm.na ≤ bs.length ∧ skipWordsA bs 1 cw = some wm.nextOff
  | 0, nw, ws, endPos, hw, hsw => by
    have heq : nw = 1 + (nw - 1) := by omega
    rw [heq, skipWordsA_add] at hsw
    cases h1 : skipWordsA bs 1 ws with
    | none => simp [h1] at hsw
    | some nextOff =>
      obtain ⟨wm, annOff, hc, hread, h0, h4, h8, h12, h16, hco, hao, hno, hna⟩ := readWordAt_eq_step bs h1
      refine ⟨ws, wm, rfl, hread, h0, h4, h8, h12, h16, hco, ?_, ?_, ?_⟩
      · rw [h12, hco, hao]; exact hc
      · exact hna
      · rw [hno]; exact h1
  | w + 1, nw, ws, endPos, hw, hsw => by
    have heq : nw = 1 + (nw - 1) := by omega
    rw [heq, skipWordsA_add] at hsw
    cases h1 : skipWordsA bs 1 ws with
    | none => simp [h1] at hsw
    | some nextOff =>
      simp only [h1, Option.bind_some] at hsw
      obtain ⟨wm0, _, _, hread0, _, _, _, _, _, _, _, hno0, _⟩ := readWordAt_eq_step bs h1
      have h1eq : nextOff = wm0.nextOff := hno0.symm
      subst h1eq
      obtain ⟨cw, wm, hcw, hwa, hi, ho, hl, hnc, hna, hco, hsc, hna_le, hstep⟩ :=
        wordAt_eq_skipWordsA bs w (nw - 1) wm0.nextOff endPos (by omega) hsw
      refine ⟨cw, wm, ?_, ?_, hi, ho, hl, hnc, hna, hco, hsc, hna_le, hstep⟩
      · rw [show w + 1 = 1 + w by omega, skipWordsA_add, h1]; exact hcw
      · simp only [SporosCheckFlat.wordAt, hread0, bind, Option.bind, hwa]

private theorem primEff_eq_primEffTree_small :
    (List.range 57).all (fun op => SporosCheckFlat.primEff op == some (primEffTree.eval op)) = true := by
  decide +kernel

private theorem primEff_eq_primEffTree {op : Nat} (h : op < 57) :
    SporosCheckFlat.primEff op = some (primEffTree.eval op) :=
  beq_iff_eq.1 (List.all_eq_true.1 primEff_eq_primEffTree_small op (List.mem_range.2 h))

private def CheckResA.toRes : CheckResA → SporosCheckFlat.CheckRes
  | .ok => .ok
  | .reject => .reject
  | .overflow => .overflow

private theorem hstepA_eq_hstepF {bs : List Nat} {nw ws endPos outs locals na annOff hd ha p op len : Nat}
    (hsw : skipWordsA bs nw ws = some endPos)
    (hna : annOff + 12 * na ≤ bs.length)
    (hop : bs[p]? = some op)
    (hlen : instrLenA op = some len)
    (hplen : p + len ≤ bs.length) :
    let imm := if len = 5 then u32 bs (p + 1) else 0
    match SporosCheckFlat.hstepF bs nw ws outs locals annOff na hd ha op imm,
          hstepA bs nw ws outs locals na annOff hd ha p with
    | .overflow, .overflow => True
    | .reject, .reject => True
    | .noFall, .nofall l => l = len
    | .next d' a', .fall nd na' l => d' = nd ∧ a' = na' ∧ l = len
    | _, _ => False := by
  have hop_d : (bs[p]?).getD 255 = op := by rw [hop]; rfl
  have hHMAX : SporosCheckFlat.HMAX = HMAX := rfl
  dsimp only
  simp only [SporosCheckFlat.hstepF, hstepA, hop_d]
  by_cases h57 : op < 57
  · have hlen1 : len = 1 := by
      unfold instrLenA at hlen; rw [if_pos h57] at hlen; cases hlen; rfl
    rw [primEff_eq_primEffTree h57]
    dsimp only
    rw [if_pos h57]
    by_cases h50 : op = 50
    · subst h50
      have heval : primEffTree.eval 50 = (1, 0) := by decide
      rw [heval]
      dsimp only
      by_cases hd0 : hd = 0
      · simp [hd0]
      · simp [hd0, hlen1]
    · have hne50 : (op == 50) = false := beq_false_of_ne h50
      cases hev : primEffTree.eval op with
      | mk eIn eOut =>
        simp only [if_neg h50, hne50]
        by_cases hlt : hd < eIn
        · simp [hlt]
        · rw [if_neg hlt, hHMAX]
          by_cases hov : HMAX ≤ hd - eIn + eOut
          · simp [hlt, hov]
          · simp [hlt, hov, hlen1]
  · rw [SporosCheckFlat.primEff_none_of_ge h57, if_neg h57]
    dsimp only
    unfold instrLenA at hlen
    rw [if_neg h57] at hlen
    by_cases h80 : op < 0x80
    · rw [if_pos h80] at hlen; cases hlen
    · rw [if_neg h80] at hlen
      by_cases heq80 : op = 0x80
      · subst heq80
        rw [if_pos rfl] at hlen; cases hlen
        simp only [beq_self_eq_true, ↓reduceIte, hHMAX]
        by_cases hov : HMAX ≤ hd + 1
        · simp [hov]
        · simp [hov]
      · rw [if_neg heq80] at hlen
        have hne80 : (op == 0x80) = false := beq_false_of_ne heq80
        by_cases h8c : 0x8c < op
        · rw [if_pos h8c] at hlen; cases hlen
        · rw [if_neg h8c] at hlen
          have hle8c : op ≤ 0x8c := by omega
          have hge81 : 0x81 ≤ op := by omega
          have hcases : op = 0x81 ∨ op = 0x82 ∨ op = 0x83 ∨ op = 0x84 ∨ op = 0x85 ∨
              op = 0x86 ∨ op = 0x87 ∨ op = 0x88 ∨ op = 0x89 ∨ op = 0x8a ∨ op = 0x8b ∨ op = 0x8c := by omega
          rcases hcases with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl
          · -- 0x81
            simp at hlen; subst hlen
            simp only [hHMAX]; by_cases hov : HMAX ≤ hd + 2 <;> simp [hov]
          · -- 0x82
            simp at hlen; subst hlen
            by_cases hm : u32 bs (p + 1) < hd
            · have hm' : ¬ (u32 bs (p + 1) ≥ hd) := by omega
              by_cases hov : HMAX ≤ hd + 1 <;> simp [hm, hm', hHMAX, hov]
            · simp [hm, (by omega : u32 bs (p + 1) ≥ hd)]
          · -- 0x83
            simp at hlen; subst hlen
            by_cases hd0 : hd = 0
            · simp [hd0]
            · have hdpos : ¬ (hd < 1) := by omega
              by_cases hov : HMAX ≤ ha + 1 <;> simp [hd0, hdpos, hHMAX, hov]
          · -- 0x84
            simp at hlen; subst hlen
            by_cases ha0 : ha = 0
            · simp [ha0]
            · have hapos : ¬ (ha < 1) := by omega
              by_cases hov : HMAX ≤ hd + 1 <;> simp [ha0, hapos, hHMAX, hov]
          · -- 0x85
            simp at hlen; subst hlen
            by_cases ha0 : ha = 0
            · simp [ha0]
            · have hapos : ¬ (ha < 1) := by omega
              by_cases hov : HMAX ≤ hd + 1 <;> simp [ha0, hapos, hHMAX, hov]
          · -- 0x86
            simp at hlen; subst hlen
            by_cases hi : u32 bs (p + 1) < locals
            · have hi' : ¬ (u32 bs (p + 1) ≥ locals) := by omega
              by_cases hov : HMAX ≤ hd + 1 <;> simp [hi, hi', hHMAX, hov]
            · simp [hi, (by omega : u32 bs (p + 1) ≥ locals)]
          · -- 0x87
            simp at hlen; subst hlen
            by_cases hok : hd = 0 ∨ locals ≤ u32 bs (p + 1)
            · rcases hok with rfl | hlo
              · simp
              · by_cases hd0 : hd = 0
                · simp [hd0]
                · simp [hd0, hlo, show ¬ (u32 bs (p + 1) < locals) by omega]
            · have hdpos : hd ≠ 0 := fun h => hok (Or.inl h)
              have hlt : u32 bs (p + 1) < locals := by omega
              have hle : ¬ (locals ≤ u32 bs (p + 1)) := by omega
              simp [hdpos, hlt, hle]
          · -- 0x88
            simp at hlen; subst hlen
            simp [lookupA_eq_lookup bs (u32 bs (p + 1)) na annOff hna]
            cases lookupA bs (u32 bs (p + 1)) na annOff with
            | none => simp
            | some q =>
              simp only [Option.map_some]
              by_cases heq : u32 bs (q + 4) = hd ∧ u32 bs (q + 8) = ha
              · simp [heq]
              · simp [heq]
          · -- 0x89
            simp at hlen; subst hlen
            by_cases hd0 : hd = 0
            · simp [hd0]
            · simp [hd0, lookupA_eq_lookup bs (u32 bs (p + 1)) na annOff hna]
              cases lookupA bs (u32 bs (p + 1)) na annOff with
              | none => simp
              | some q =>
                simp only [Option.map_some]
                by_cases heq : u32 bs (q + 4) = hd - 1 ∧ u32 bs (q + 8) = ha
                · simp [heq]
                · simp [heq]
          · -- 0x8a
            simp at hlen; subst hlen
            by_cases hd0 : hd = 0
            · simp [hd0]
            · simp [hd0, lookupA_eq_lookup bs (u32 bs (p + 1)) na annOff hna]
              cases lookupA bs (u32 bs (p + 1)) na annOff with
              | none => simp
              | some q =>
                simp only [Option.map_some]
                by_cases heq : u32 bs (q + 4) = hd - 1 ∧ u32 bs (q + 8) = ha
                · simp [heq]
                · simp [heq]
          · -- 0x8b
            simp at hlen; subst hlen
            by_cases hw : u32 bs (p + 1) < nw
            · obtain ⟨cw, wm, hcw, hwa, hi, ho, _⟩ := wordAt_eq_skipWordsA bs (u32 bs (p + 1)) nw ws endPos hw hsw
              simp [hw, show ¬ (nw ≤ u32 bs (p + 1)) by omega, hcw, hwa, hi, ho]
              by_cases hdc : hd < u32 bs cw
              · simp [hdc]
              · by_cases hov : HMAX ≤ hd - u32 bs cw + u32 bs (cw + 4)
                · simp [hdc, hHMAX, hov]
                · simp [hdc, hHMAX, hov]
            · simp [hw, (by omega : u32 bs (p + 1) ≥ nw)]
          · -- 0x8c
            simp at hlen; subst hlen
            by_cases heq : hd = outs ∧ ha = 0
            · simp [heq]
            · simp [heq]

private theorem walkA_eq_walk {bs : List Nat} {nw ws endPos outs locals nc na annOff : Nat}
    (hsw : skipWordsA bs nw ws = some endPos)
    (hna : annOff + 12 * na ≤ bs.length) :
    ∀ rem i p curD curA curD' curA',
      rem + i = nc → skipCodeA bs rem p = some annOff → p ≤ bs.length →
      (SporosCheckFlat.lookup bs annOff na i = none → curD' = curD ∧ curA' = curA) →
      (walkA bs nw ws outs locals nc na annOff rem i p curD' curA').toRes =
        SporosCheckFlat.walk bs nw ws outs locals nc annOff na rem i p curD curA
  | 0, _, _, _, _, _, _, _, _, _, _ => rfl
  | rem + 1, i, p, curD, curA, curD', curA', hsum, hsc, hp, hcur => by
    simp only [skipCodeA] at hsc
    split at hsc
    · rename_i hplt
      cases hl : instrLenA bs[p] with
      | none => rw [hl] at hsc; cases hsc
      | some len =>
        rw [hl] at hsc
        dsimp only at hsc
        split at hsc
        · cases hsc
        · rename_i hno_ov
          have hplen : p + len ≤ bs.length := by omega
          have hop : bs[p]? = some bs[p] := List.getElem?_eq_getElem hplt
          have hlen' : SporosCheckFlat.instrLen bs[p] = some len := by rw [← instrLenA_eq_instrLen]; exact hl
          have himm : (if len == 5 then SporosCheckFlat.rdU32 bs (p + 1) else some 0) =
              some (if len = 5 then u32 bs (p + 1) else 0) := by
            by_cases h5 : len = 5
            · subst h5; simp only [beq_self_eq_true, ↓reduceIte]; exact u32_eq_rdU32 bs (by omega)
            · have hb5 : (len == 5) = false := beq_false_of_ne h5
              simp [hb5, h5]
          have hcond : (!decide (p + len ≤ bs.length)) = false := by simp [hplen]
          simp only [SporosCheckFlat.walk, walkA, hop, hlen', hcond, Bool.false_eq_true, ↓reduceIte, himm]
          have hlk := lookupA_eq_lookup bs i na annOff hna
          have hbody : ∀ d0 a0 : Nat,
              (match hstepA bs nw ws outs locals na annOff d0 a0 p with
              | .reject => CheckResA.reject
              | .overflow => CheckResA.overflow
              | .fall nd na' l =>
                if i + 1 < nc then
                  match lookupA bs (i + 1) na annOff with
                  | none => walkA bs nw ws outs locals nc na annOff rem (i + 1) (p + l) nd na'
                  | some q =>
                    if u32 bs (q + 4) = nd ∧ u32 bs (q + 8) = na' then
                      walkA bs nw ws outs locals nc na annOff rem (i + 1) (p + l) nd na'
                    else CheckResA.reject
                else CheckResA.reject
              | .nofall l =>
                let nxt := match lookupA bs (i + 1) na annOff with
                  | some q => (u32 bs (q + 4), u32 bs (q + 8))
                  | none => (0, 0)
                walkA bs nw ws outs locals nc na annOff rem (i + 1) (p + l) nxt.1 nxt.2).toRes =
              match SporosCheckFlat.hstepF bs nw ws outs locals annOff na d0 a0 bs[p]
                  (if len = 5 then u32 bs (p + 1) else 0) with
              | .overflow => SporosCheckFlat.CheckRes.overflow
              | .reject => SporosCheckFlat.CheckRes.reject
              | .noFall => SporosCheckFlat.walk bs nw ws outs locals nc annOff na rem (i + 1) (p + len) 0 0
              | .next nd na' =>
                if i + 1 ≥ nc then SporosCheckFlat.CheckRes.reject
                else
                  if (match SporosCheckFlat.lookup bs annOff na (i + 1) with
                      | none => true | some (ad, aa) => ad == nd && aa == na') = true then
                    SporosCheckFlat.walk bs nw ws outs locals nc annOff na rem (i + 1) (p + len) nd na'
                  else SporosCheckFlat.CheckRes.reject := by
            intro d0 a0
            have hs := hstepA_eq_hstepF (outs := outs) (locals := locals) (hd := d0) (ha := a0)
              hsw hna hop hl hplen
            dsimp only at hs
            cases hsf : SporosCheckFlat.hstepF bs nw ws outs locals annOff na d0 a0 bs[p]
                (if len = 5 then u32 bs (p + 1) else 0) with
            | overflow =>
              rw [hsf] at hs
              cases hsa : hstepA bs nw ws outs locals na annOff d0 a0 p <;> rw [hsa] at hs <;> first | (cases hs; done) | rfl
            | reject =>
              rw [hsf] at hs
              cases hsa : hstepA bs nw ws outs locals na annOff d0 a0 p <;> rw [hsa] at hs <;> first | (cases hs; done) | rfl
            | noFall =>
              rw [hsf] at hs
              cases hsa : hstepA bs nw ws outs locals na annOff d0 a0 p <;> rw [hsa] at hs <;> try (cases hs; done)
              rw [hs]
              dsimp only
              refine walkA_eq_walk hsw hna rem (i + 1) (p + len) 0 0 _ _ (by omega) hsc hplen ?_
              intro hn
              rw [lookupA_eq_lookup bs (i + 1) na annOff hna] at hn
              cases hln : lookupA bs (i + 1) na annOff with
              | none => simp
              | some _ => rw [hln] at hn; cases hn
            | next nd na' =>
              rw [hsf] at hs
              cases hsa : hstepA bs nw ws outs locals na annOff d0 a0 p <;> rw [hsa] at hs <;> try (cases hs; done)
              rcases hs with ⟨rfl, rfl, hlen_eq⟩
              rw [hlen_eq]
              dsimp only
              by_cases hic : i + 1 < nc
              · rw [if_neg (by omega : ¬ (i + 1 ≥ nc)), if_pos hic]
                have hlk1 := lookupA_eq_lookup bs (i + 1) na annOff hna
                rw [hlk1]
                cases hq1 : lookupA bs (i + 1) na annOff with
                | none =>
                  rw [Option.map_none]
                  dsimp only; rw [if_pos rfl]
                  exact walkA_eq_walk hsw hna rem (i + 1) (p + len) nd na' nd na' (by omega) hsc hplen (fun _ => ⟨rfl, rfl⟩)
                | some q1 =>
                  rw [Option.map_some]
                  dsimp only
                  by_cases hmatch : u32 bs (q1 + 4) = nd ∧ u32 bs (q1 + 8) = na'
                  · have hbm : (u32 bs (q1 + 4) == nd && u32 bs (q1 + 8) == na') = true := by
                      simp [hmatch.1, hmatch.2]
                    rw [if_pos hbm, if_pos hmatch]
                    exact walkA_eq_walk hsw hna rem (i + 1) (p + len) nd na' nd na' (by omega) hsc hplen (fun _ => ⟨rfl, rfl⟩)
                  · have hbf : (u32 bs (q1 + 4) == nd && u32 bs (q1 + 8) == na') = false := by
                      simp only [Bool.and_eq_false_iff, beq_eq_false_iff_ne]
                      by_cases h1 : u32 bs (q1 + 4) = nd
                      · exact Or.inr (fun h2 => hmatch ⟨h1, h2⟩)
                      · exact Or.inl h1
                    have hbmf : ¬ ((u32 bs (q1 + 4) == nd && u32 bs (q1 + 8) == na') = true) := by simp [hbf]
                    simp only [if_neg hbmf, if_neg hmatch, CheckResA.toRes]
              · simp only [if_pos (by omega : i + 1 ≥ nc), if_neg hic, CheckResA.toRes]
          cases hlkA : lookupA bs i na annOff with
          | none =>
            have hfn : SporosCheckFlat.lookup bs annOff na i = none := by rw [hlk, hlkA]; rfl
            have ⟨h1, h2⟩ := hcur hfn
            rw [hlk, hlkA, h1, h2]
            exact hbody curD curA
          | some q =>
            rw [hlk, hlkA]
            exact hbody (u32 bs (q + 4)) (u32 bs (q + 8))
    · cases hsc

private theorem annsOKA_eq_annsOK {bs : List Nat} {nc na annOff : Nat}
    (hna : annOff + 12 * na ≤ bs.length) :
    ∀ k off, off + 12 * k ≤ annOff + 12 * na →
      annsOKA bs nc na annOff k off = SporosCheckFlat.annsOK bs nc annOff na k off
  | 0, _, _ => rfl
  | k + 1, off, hoff => by
    simp only [annsOKA, SporosCheckFlat.annsOK]
    rw [u32_eq_rdU32 bs (by omega), u32_eq_rdU32 bs (by omega), u32_eq_rdU32 bs (by omega)]
    dsimp only
    by_cases hlt : u32 bs off < nc
    · rw [if_pos hlt, lookupA_eq_lookup bs (u32 bs off) na annOff hna]
      cases hq : lookupA bs (u32 bs off) na annOff with
      | none => simp [hlt]
      | some q =>
        have hlt_b : decide (u32 bs off < nc) = true := decide_eq_true hlt
        have hcmp : ((Option.map (fun q' => (u32 bs (q' + 4), u32 bs (q' + 8))) (some q)) ==
            some (u32 bs (off + 4), u32 bs (off + 8))) =
            ((u32 bs (q + 4) == u32 bs (off + 4)) && (u32 bs (q + 8) == u32 bs (off + 8))) := rfl
        rw [hlt_b, Bool.true_and, hcmp, annsOKA_eq_annsOK hna k (off + 12) (by omega)]
    · have hlt_f : decide (u32 bs off < nc) = false := decide_eq_false hlt
      rw [if_neg hlt, hlt_f]
      simp

private theorem checkWordsA_eq_checkWords {bs : List Nat} {nw ws endPos : Nat}
    (hsw : skipWordsA bs nw ws = some endPos) :
    ∀ rem wordOff, skipWordsA bs rem wordOff = some endPos →
      (checkWordsA bs nw ws rem wordOff).toRes = SporosCheckFlat.checkWords bs nw ws rem wordOff
  | 0, _, _ => rfl
  | rem + 1, wordOff, hrem => by
    rw [show rem + 1 = 1 + rem by omega, skipWordsA_add] at hrem
    cases h1 : skipWordsA bs 1 wordOff with
    | none => simp [h1] at hrem
    | some nextOff =>
      simp only [h1, Option.bind_some] at hrem
      obtain ⟨wm, annOff, hsc, hread, hi, ho, hl, hnc, hna, hco, hao, hno, hna_le⟩ :=
        readWordAt_eq_step bs h1
      simp only [checkWordsA, checkWordA, SporosCheckFlat.checkWords, hread]
      by_cases hnc0 : u32 bs (wordOff + 12) = 0
      · have hb : (wm.nc == 0) = true := by rw [hnc, hnc0]; decide
        simp [hnc0, hb]; rfl
      · have hb : (wm.nc == 0) = false := beq_false_of_ne (by rw [hnc]; exact hnc0)
        rw [if_neg hnc0, if_neg (by simp [hb])]
        have hlk0 := lookupA_eq_lookup bs 0 wm.na wm.annOff hna_le
        rw [hsc, ← ho, ← hl, ← hnc, ← hna, ← hao, ← hco, ← hi]
        dsimp only
        suffices htail :
            (match
                (match walkA bs nw ws wm.outs wm.locals wm.nc wm.na wm.annOff
                    wm.nc 0 wm.codeOff wm.ins 0 with
                  | .reject => CheckResA.reject
                  | .overflow => CheckResA.overflow
                  | .ok =>
                    if annsOKA bs wm.nc wm.na wm.annOff wm.na wm.annOff = true then CheckResA.ok
                    else CheckResA.reject) with
              | .reject => CheckResA.reject
              | .overflow => CheckResA.overflow
              | .ok =>
                match skipWordsA bs 1 wordOff with
                | none => CheckResA.reject
                | some nextOff => checkWordsA bs nw ws rem nextOff).toRes =
            (match SporosCheckFlat.walk bs nw ws wm.outs wm.locals wm.nc wm.annOff wm.na wm.nc 0
                wm.codeOff wm.ins 0 with
              | .overflow => SporosCheckFlat.CheckRes.overflow
              | .reject => SporosCheckFlat.CheckRes.reject
              | .ok =>
                if SporosCheckFlat.annsOK bs wm.nc wm.annOff wm.na wm.na wm.annOff = true then
                  SporosCheckFlat.checkWords bs nw ws rem wm.nextOff
                else SporosCheckFlat.CheckRes.reject) by
          cases hlk0A : lookupA bs 0 wm.na wm.annOff with
          | none =>
            rw [hlk0, hlk0A, Option.map_none]
            simp only [Bool.not_true, Bool.false_eq_true, ↓reduceIte]
            exact htail
          | some q =>
            rw [hlk0, hlk0A, Option.map_some]
            dsimp only
            by_cases hq0 : u32 bs (q + 4) = wm.ins ∧ u32 bs (q + 8) = 0
            · have hqt : (u32 bs (q + 4) == wm.ins && u32 bs (q + 8) == 0) = true := by
                simp [hq0.1, hq0.2]
              simp only [hqt, Bool.not_true, Bool.false_eq_true, ↓reduceIte]
              exact htail
            · have hqf : (u32 bs (q + 4) == wm.ins && u32 bs (q + 8) == 0) = false := by
                simp only [Bool.and_eq_false_iff, beq_eq_false_iff_ne]
                by_cases h1' : u32 bs (q + 4) = wm.ins
                · exact Or.inr (fun h2' => hq0 ⟨h1', h2'⟩)
                · exact Or.inl h1'
              simp only [hqf, Bool.not_false, ↓reduceIte, CheckResA.toRes]
        have hwalk := walkA_eq_walk (nw := nw) (ws := ws) (outs := wm.outs) (locals := wm.locals)
            (nc := wm.nc) (na := wm.na) (annOff := wm.annOff) hsw hna_le
            wm.nc 0 wm.codeOff wm.ins 0 wm.ins 0 (by omega)
            (by rw [hnc, hco, hao]; exact hsc)
            (by have h20 : wordOff + 20 ≤ bs.length := by
                  simp only [skipWordsA] at h1
                  split at h1 <;> first | cases h1 | omega
                rw [hco]; exact h20)
            (fun _ => ⟨rfl, rfl⟩)
        cases hwa_f : SporosCheckFlat.walk bs nw ws wm.outs wm.locals wm.nc wm.annOff wm.na wm.nc 0 wm.codeOff wm.ins 0 with
        | overflow =>
          rw [hwa_f] at hwalk
          cases hwa_a : walkA bs nw ws wm.outs wm.locals wm.nc wm.na wm.annOff wm.nc 0 wm.codeOff wm.ins 0 <;>
            rw [hwa_a] at hwalk <;> cases hwalk <;> rfl
        | reject =>
          rw [hwa_f] at hwalk
          cases hwa_a : walkA bs nw ws wm.outs wm.locals wm.nc wm.na wm.annOff wm.nc 0 wm.codeOff wm.ins 0 <;>
            rw [hwa_a] at hwalk <;> cases hwalk <;> rfl
        | ok =>
          rw [hwa_f] at hwalk
          cases hwa_a : walkA bs nw ws wm.outs wm.locals wm.nc wm.na wm.annOff wm.nc 0 wm.codeOff wm.ins 0 <;>
            rw [hwa_a] at hwalk <;> try (cases hwalk; done)
          dsimp only
          rw [annsOKA_eq_annsOK hna_le wm.na wm.annOff (by omega)]
          cases hanns : SporosCheckFlat.annsOK bs wm.nc wm.annOff wm.na wm.na wm.annOff
          · simp [CheckResA.toRes]
          · simp only [↓reduceIte, h1, hno]
            exact checkWordsA_eq_checkWords hsw rem nextOff hrem

theorem checkFlatA_eq_checkFlat_of_lt_HMAX {bs : List Nat}
    (hby : ∀ b ∈ bs, b < 256) (hsm : bs.length < HMAX) :
    checkFlatA bs = SporosCheckFlat.checkFlat bs := by
  have hba : bs.all (· < 256) = true := List.all_eq_true.2 fun b hb => decide_eq_true (hby b hb)
  have hltW : bs.length < SporosCheckFlat.W := by unfold HMAX SporosCheckFlat.W at *; omega
  simp only [checkFlatA, SporosCheckFlat.checkFlat, SporosCheckFlat.checkFlatRes, bind, Option.bind]
  rw [if_neg (by simp [hba, hsm]), if_neg (by simp [hba]; omega)]
  dsimp only
  rw [parseA_eq_parse]
  cases hp : SporosCheckFlat.parse bs with
  | none => rfl
  | some parsed =>
    rcases parsed with ⟨main, nw, ws⟩
    dsimp only
    by_cases hmn : main < nw
    · rw [if_neg (by simp [hmn]), if_neg (by omega)]
      have hsw : skipWordsA bs nw ws = some bs.length := by
        simp only [SporosCheckFlat.parse, bind, Option.bind] at hp
        split at hp
        · cases hp
        · cases h0 : SporosCheckFlat.rdU32 bs 0 with
          | none => simp [h0] at hp
          | some v0 =>
            simp only [h0] at hp
            split at hp
            · cases hp
            · cases h4 : SporosCheckFlat.rdU32 bs 4 with
              | none => simp [h4] at hp
              | some v4 =>
                simp only [h4] at hp
                cases h8 : SporosCheckFlat.rdU32 bs 8 with
                | none => simp [h8] at hp
                | some v8 =>
                  simp only [h8] at hp
                  cases h12 : SporosCheckFlat.rdU32 bs 12 with
                  | none => simp [h12] at hp
                  | some v12 =>
                    simp only [h12] at hp
                    cases hsd : SporosCheckFlat.skipData bs v12 16 with
                    | none => simp [hsd] at hp
                    | some ws0 =>
                      simp only [hsd] at hp
                      cases hsw' : SporosCheckFlat.skipWords bs v8 ws0 with
                      | none => simp [hsw'] at hp
                      | some ep =>
                        simp only [hsw'] at hp
                        split at hp
                        · rename_i hep; injection hp with htu
                          simp only [Prod.mk.injEq] at htu
                          obtain ⟨_, h1, h2⟩ := htu
                          rw [← h1, ← h2, skipWordsA_eq_skipWords, hsw', beq_iff_eq.1 hep]
                        · cases hp
      obtain ⟨mw, mwm, hmw, hwa, hi, _⟩ := wordAt_eq_skipWordsA bs main nw ws bs.length hmn hsw
      rw [hmw, hwa]
      dsimp only
      by_cases hins0 : u32 bs mw = 0
      · have hne0 : (mwm.ins != 0) = false := by simp [hi, hins0]
        rw [if_neg (by simp [hins0]), if_neg (by simp [hne0])]
        have hcw := checkWordsA_eq_checkWords hsw nw ws hsw
        cases hcwf : SporosCheckFlat.checkWords bs nw ws nw ws with
        | ok =>
          rw [hcwf] at hcw
          cases hcwa : checkWordsA bs nw ws nw ws <;> rw [hcwa] at hcw <;> cases hcw <;> rfl
        | reject =>
          rw [hcwf] at hcw
          cases hcwa : checkWordsA bs nw ws nw ws <;> rw [hcwa] at hcw <;> cases hcw <;> rfl
        | overflow =>
          rw [hcwf] at hcw
          cases hcwa : checkWordsA bs nw ws nw ws <;> rw [hcwa] at hcw <;> cases hcw <;> rfl
      · have hne0 : (mwm.ins != 0) = true := bne_iff_ne.2 (by rw [hi]; exact hins0)
        rw [if_pos hins0, if_pos hne0]
    · rw [if_pos (by simp [hmn]), if_pos (by omega)]

/-! ### Top-level ELF binary soundness, refusal soundness, completeness, and single-disturbance immunity -/

theorem checkFlatA_lt_HMAX_of_some {bs : List Nat} {b : Bool} (h : checkFlatA bs = some b) :
    (∀ x ∈ bs, x < 256) ∧ bs.length < HMAX := by
  unfold checkFlatA at h
  by_cases hok : (bs.all (· < 256) && decide (bs.length < HMAX)) = true
  · simp only [Bool.and_eq_true, List.all_eq_true, decide_eq_true_eq] at hok
    exact hok
  · exfalso
    have hneg : (!(bs.all (· < 256) && decide (bs.length < HMAX))) = true := by
      cases hb : (bs.all (· < 256) && decide (bs.length < HMAX)) <;> simp_all
    rw [if_pos hneg] at h
    cases h

theorem spr1_bin_sound {k : Kern} (hby : ∀ x ∈ k.stdin, x < 256) :
    ∀ {n c : Nat}, run im n (.ok (initState k)) = .exit c →
      (c = 0 → ∃ pr, Sporos.decode k.stdin = some pr ∧ Sporos.checkProg pr = true) ∧
      c ≠ 3 ∧ (c = 0 ∨ c = 5 ∨ c = 6) := by
  intro n c hrun
  have hu := spr1_run_flat_uncond hby hrun
  refine ⟨fun hc0 => ?_, hu.2.2.1, hu.2.2.2⟩
  have hcfA := hu.1 hc0
  have ⟨hby', hsm⟩ := checkFlatA_lt_HMAX_of_some hcfA
  rw [checkFlatA_eq_checkFlat_of_lt_HMAX hby' hsm] at hcfA
  exact SporosCheckFlat.checkFlat_sound hcfA

theorem spr1_bin_refuse_sound {k : Kern} (hby : ∀ x ∈ k.stdin, x < 256)
    {n : Nat} (hrun : run im n (.ok (initState k)) = .exit 5) :
    ∀ pr, Sporos.decode k.stdin = some pr → Sporos.checkProg pr = false := by
  have hcfA := (spr1_run_flat_uncond hby hrun).2.1 rfl
  have ⟨hby', hsm⟩ := checkFlatA_lt_HMAX_of_some hcfA
  rw [checkFlatA_eq_checkFlat_of_lt_HMAX hby' hsm] at hcfA
  exact SporosCheckFlat.checkFlat_refuse_sound hcfA

theorem spr1_bin_complete_good {k : Kern} {pr : Sporos.Prog}
    (hdec : Sporos.decode k.stdin = some pr)
    (hchk : Sporos.checkProg pr = true)
    (hfit : SporosCheckFlat.Fits48 k.stdin pr)
    (hsm : k.stdin.length < HMAX)
    (hor : GoodInitOrac (shape initMem) k.stdin.length k.oracle) :
    Ends (initState k) (· = .exit 0) := by
  have hcf := SporosCheckFlat.checkFlat_complete hdec hchk hfit
  rw [← checkFlatA_eq_checkFlat_of_lt_HMAX hfit.1 hsm] at hcf
  have hrun := spr1_run_flat_good hfit.1 hor
  rwa [hcf, flatExit] at hrun

theorem spr1_bin_complete {bs : List Nat} {pr : Sporos.Prog}
    (hdec : Sporos.decode bs = some pr)
    (hchk : Sporos.checkProg pr = true)
    (hfit : SporosCheckFlat.Fits48 bs pr)
    (h44 : bs.length < 2 ^ 44) :
    Ends (initState ⟨[], [], bs, [], canonOracle bs⟩) (· = .exit 0) :=
  spr1_bin_complete_good hdec hchk hfit (by dsimp; unfold HMAX; omega) (canonOracle_good h44)

theorem spr1_bin_no_single_fault_false_accept {bs : List Nat} {s0 : St}
    (htri : TripledState bs s0)
    (hsm : bs.length < HMAX)
    (hbad : ¬ (∃ pr, Sporos.decode bs = some pr ∧ Sporos.checkProg pr = true)) :
    (∀ k m {sk sk1 s2 : St},
      run im k (.ok s0) = .ok sk → step im sk = .ok sk1 →
      (s2 ∈ regFlips sk1 ∨ s2 ∈ memFlips sk1 ∨ s2 ∈ flagFlips sk1) →
      run im m (.ok s2) ≠ .exit 0) ∧
    (∀ (r : Rates) (hr : r.ok), SporosCheckFlat.singleMechanism r → r.fetch = 0 → ∀ k m,
      SporosCheckFlat.exit0Prob (pRunWithFaultAt r hr k m (.ok s0)) = 0) ∧
    (∃ N s_vote, run im N (.ok s0) = .ok s_vote ∧
      (∀ {n : Nat} {o : Out}, BinOneFaultRun false n (.ok s_vote) o → o ≠ .exit 0) ∧
      (∀ (r : Rates) (hr : r.ok), SporosCheckFlat.singleMechanism r → ∀ k m,
        SporosCheckFlat.exit0Prob (pRunWithFaultAt r hr k m (.ok s_vote)) = 0)) := by
  have hcf_bad : checkFlatA bs ≠ some true := by
    intro htrue
    rw [checkFlatA_eq_checkFlat_of_lt_HMAX htri.bytes hsm] at htrue
    exact hbad (SporosCheckFlat.checkFlat_sound htrue)
  obtain ⟨N, s_vote, hrun, _, h1f, hpr⟩ := tripledState_tail_fourChannel_no_exit0 htri hcf_bad
  exact ⟨tripledState_oneStateFault_no_exit0 htri hcf_bad,
    tripledState_prun_noFetch_exit0Prob_zero htri hcf_bad,
    ⟨N, s_vote, hrun, h1f, hpr⟩⟩

/-! ### Ground `decide +kernel` witnesses and example executables -/

theorem checkFlatA_echo_valid : checkFlatA Sporos.echo.enc = some true := by decide +kernel
theorem checkFlatA_hello_valid : checkFlatA Sporos.Examples.hello.enc = some true := by decide +kernel
theorem checkFlatA_upper_valid : checkFlatA Sporos.Examples.upper.enc = some true := by decide +kernel
theorem checkFlatA_fib_valid : checkFlatA Sporos.Examples.fib.enc = some true := by decide +kernel

theorem spr1_bin_echo_accepts :
    Ends (initState ⟨[], [], Sporos.echo.enc, [], canonOracle Sporos.echo.enc⟩) (· = .exit 0) := by
  have hrun := spr1_run_flat_canon (bs := Sporos.echo.enc) (by decide +kernel) (by decide +kernel)
  rwa [checkFlatA_echo_valid, flatExit] at hrun

theorem spr1_bin_hello_accepts :
    Ends (initState ⟨[], [], Sporos.Examples.hello.enc, [], canonOracle Sporos.Examples.hello.enc⟩) (· = .exit 0) := by
  have hrun := spr1_run_flat_canon (bs := Sporos.Examples.hello.enc) (by decide +kernel) (by decide +kernel)
  rwa [checkFlatA_hello_valid, flatExit] at hrun

theorem spr1_bin_upper_accepts :
    Ends (initState ⟨[], [], Sporos.Examples.upper.enc, [], canonOracle Sporos.Examples.upper.enc⟩) (· = .exit 0) := by
  have hrun := spr1_run_flat_canon (bs := Sporos.Examples.upper.enc) (by decide +kernel) (by decide +kernel)
  rwa [checkFlatA_upper_valid, flatExit] at hrun

theorem spr1_bin_fib_accepts :
    Ends (initState ⟨[], [], Sporos.Examples.fib.enc, [], canonOracle Sporos.Examples.fib.enc⟩) (· = .exit 0) := by
  have hrun := spr1_run_flat_canon (bs := Sporos.Examples.fib.enc) (by decide +kernel) (by decide +kernel)
  rwa [checkFlatA_fib_valid, flatExit] at hrun

theorem checkFlatA_refusal_witnesses :
    checkFlatA [] = some false ∧
    checkFlatA [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] = some false ∧
    checkFlatA [0x53, 0x50, 0x52, 0x31, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
      1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0x32] = some false ∧
    checkFlatA [0x53, 0x50, 0x52, 0x31, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0x01] = some false := by
  decide +kernel

theorem spr1_bin_empty_refused :
    Ends (initState ⟨[], [], [], [], canonOracle []⟩) (· = .exit 5) := by
  have hrun := spr1_run_flat_canon (bs := []) (by decide +kernel) (by decide +kernel)
  rwa [checkFlatA_refusal_witnesses.1, flatExit] at hrun

end SporosCheckBin
