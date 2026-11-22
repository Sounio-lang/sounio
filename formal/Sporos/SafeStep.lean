import Sporos.Safe

namespace Sporos

theorem safe_next {pr : Prog} {s' : St} (h : Inv pr s') : Safe pr (.next s') := h

theorem step_safe (pr : Prog) (ph : Phys) (hp : checkProg pr = true) (s : St) (hi : Inv pr s) :
    Safe pr (step pr ph s) := by
  unfold Inv at hi
  cases hrs : s.rs with
  | nil => rw [hrs] at hi; exact hi.elim
  | cons f rest =>
    rw [hrs] at hi
    obtain ⟨⟨wd, h, hw, hh, hpc, hd, ha, hloc⟩, hch⟩ := hi
    have hcw := checkProg_word hp hw
    have hc : wd.code[f.pc]? = some wd.code[f.pc] := List.getElem?_eq_getElem hpc
    have hfetch : fetch pr f.word f.pc = some wd.code[f.pc] := by simp [St.setTop, fetch, hw, hc]
    obtain ⟨r, hr, hnext⟩ := checkWord_at hcw hc hh
    unfold step
    rw [hrs]; dsimp only
    rw [hfetch]; dsimp only
    generalize wd.code[f.pc] = ins at hr hc
    have fall : ∀ {h' : Height} {s' : St} (g : Frame), r = some h' → s'.rs = g :: rest →
        g.word = f.word → g.base = f.base → g.abase = f.abase → g.locs.length = wd.locals → g.pc = f.pc + 1 →
        s'.ds.length = f.base + h'.d → s'.aux.length = f.abase + h'.a → Safe pr (.next s') := by
      intro h' s' g e hrs' hgw hgb hga hgl hgp hd' ha'
      obtain ⟨hlt, hh'⟩ := hnext h' e
      exact safe_next <| inv_move hrs' hw hgw hgb hga hgl (hgp ▸ hlt) (hgp ▸ hh') hd' ha' hch
    cases ins with
    | prim p =>
      dsimp only
      simp only [hstep] at hr
      by_cases hk : p.eff.1 ≤ h.d
      · rw [if_pos hk] at hr
        have hlen : (s.ds.take p.eff.1).length = p.eff.1 := by simp [St.setTop, List.length_take]; omega
        rw [if_neg (by omega)]
        split
        · rename_i res s' he
          obtain ⟨hres, hrs', hax, hds⟩ := execPrim_ok hlen he
          have hne : p ≠ .exit := by intro e; subst e; exact execPrim_exit he
          rw [if_neg hne] at hr; cases hr
          rw [St.setTop_eq (by rw [hrs'])]
          exact fall _ rfl rfl rfl rfl rfl hloc rfl
            (by simp [St.setTop, List.length_append, hres, List.length_drop, hd]; omega) (by simp [St.setTop, hax, ha])
        · trivial
        · rename_i e he; exact execPrim_fail hlen he
      · rw [if_neg hk] at hr; cases hr
    | lit n =>
      dsimp only
      simp only [hstep] at hr; cases hr
      exact fall _ rfl rfl rfl rfl rfl hloc rfl (by simp [St.setTop, hd]; omega) (by simp [St.setTop, ha])
    | str i =>
      dsimp only
      simp only [hstep] at hr; cases hr
      split
      · trivial
      · exact fall _ rfl rfl rfl rfl rfl hloc rfl (by simp [St.setTop, hd]; omega) (by simp [St.setTop, ha])
    | pick n =>
      dsimp only
      simp only [hstep] at hr
      by_cases hn : n < h.d
      · rw [if_pos hn] at hr; cases hr
        have : s.ds[n]? = some s.ds[n] := List.getElem?_eq_getElem (by omega)
        rw [this]
        exact fall _ rfl rfl rfl rfl rfl hloc rfl (by simp [St.setTop, hd]; omega) (by simp [St.setTop, ha])
      · rw [if_neg hn] at hr; cases hr
    | tor =>
      dsimp only
      simp only [hstep] at hr
      by_cases hn : 1 ≤ h.d
      · rw [if_pos hn] at hr; cases hr
        cases hds : s.ds with
        | nil => rw [hds] at hd; simp at hd; omega
        | cons v r =>
          dsimp only
          exact fall _ rfl rfl rfl rfl rfl hloc rfl (by rw [hds] at hd; simp [St.setTop] at hd ⊢; omega)
            (by simp [St.setTop, ha]; omega)
      · rw [if_neg hn] at hr; cases hr
    | fromr =>
      dsimp only
      simp only [hstep] at hr
      by_cases hn : 1 ≤ h.a
      · rw [if_pos hn] at hr; cases hr
        cases hax : s.aux with
        | nil => rw [hax] at ha; simp at ha; omega
        | cons v r =>
          dsimp only
          exact fall _ rfl rfl rfl rfl rfl hloc rfl (by simp [St.setTop, hd]; omega)
            (by rw [hax] at ha; simp [St.setTop] at ha ⊢; omega)
      · rw [if_neg hn] at hr; cases hr
    | rfetch =>
      dsimp only
      simp only [hstep] at hr
      by_cases hn : 1 ≤ h.a
      · rw [if_pos hn] at hr; cases hr
        cases hax : s.aux with
        | nil => rw [hax] at ha; simp at ha; omega
        | cons v r =>
          dsimp only
          exact fall _ rfl rfl rfl rfl rfl hloc rfl (by simp [St.setTop, hd]; omega) (by rw [hax] at ha; simpa [St.setTop] using ha)
      · rw [if_neg hn] at hr; cases hr
    | locget i =>
      dsimp only
      simp only [hstep] at hr
      by_cases hn : i < wd.locals
      · rw [if_pos hn] at hr; cases hr
        have : f.locs[i]? = some f.locs[i] := List.getElem?_eq_getElem (by omega)
        rw [this]
        exact fall _ rfl rfl rfl rfl rfl hloc rfl (by simp [St.setTop, hd]; omega) (by simp [St.setTop, ha])
      · rw [if_neg hn] at hr; cases hr
    | locset i =>
      dsimp only
      simp only [hstep] at hr
      by_cases hn : 1 ≤ h.d ∧ i < wd.locals
      · rw [if_pos hn] at hr; cases hr
        cases hds : s.ds with
        | nil => rw [hds] at hd; simp at hd; omega
        | cons v r =>
          dsimp only
          rw [if_pos (by omega)]
          exact fall _ rfl rfl rfl rfl rfl (by simp [St.setTop, hloc]) rfl (by rw [hds] at hd; simp [St.setTop] at hd ⊢; omega)
            (by simp [St.setTop, ha])
      · rw [if_neg hn] at hr; cases hr
    | jmp t =>
      dsimp only
      simp only [hstep] at hr
      split at hr
      · rename_i hl
        obtain ⟨hlt, hht⟩ := checkWord_ann hcw hl
        exact safe_next <| inv_move rfl hw rfl rfl rfl hloc hlt hht hd ha hch
      · cases hr
    | jz t =>
      dsimp only
      simp only [hstep] at hr
      split at hr
      · rename_i hn
        cases hr
        obtain ⟨hlt, hht⟩ := checkWord_ann hcw hn.2
        cases hds : s.ds with
        | nil => rw [hds] at hd; simp at hd; omega
        | cons c r =>
          dsimp only
          have hd' : r.length = f.base + (h.d - 1) := by rw [hds] at hd; simp at hd; omega
          split
          · exact safe_next <| inv_move rfl hw rfl rfl rfl hloc hlt hht hd' (by simp [St.setTop, ha]) hch
          · exact fall _ rfl rfl rfl rfl rfl hloc rfl hd' (by simp [St.setTop, ha])
      · cases hr
    | jnz t =>
      dsimp only
      simp only [hstep] at hr
      split at hr
      · rename_i hn
        cases hr
        obtain ⟨hlt, hht⟩ := checkWord_ann hcw hn.2
        cases hds : s.ds with
        | nil => rw [hds] at hd; simp at hd; omega
        | cons c r =>
          dsimp only
          have hd' : r.length = f.base + (h.d - 1) := by rw [hds] at hd; simp at hd; omega
          split
          · exact fall _ rfl rfl rfl rfl rfl hloc rfl hd' (by simp [St.setTop, ha])
          · exact safe_next <| inv_move rfl hw rfl rfl rfl hloc hlt hht hd' (by simp [St.setTop, ha]) hch
      · cases hr
    | call w =>
      dsimp only
      simp only [hstep] at hr
      cases hcwd : pr.words[w]? with
      | none => rw [hcwd] at hr; cases hr
      | some c =>
        rw [hcwd] at hr; dsimp only at hr
        split at hr
        · rename_i hin
          cases hr
          obtain ⟨hlt, hh'⟩ := hnext _ rfl
          have hcc := checkProg_word hp hcwd
          obtain ⟨hpos, h0⟩ := checkWord_entry hcc
          dsimp only
          rw [if_neg (by omega)]
          apply safe_next
          unfold Inv
          refine ⟨⟨c, ⟨c.ins, 0⟩, hcwd, h0, hpos,
            by simp; omega, by simp, by simp⟩, ⟨⟨c, hcwd, ?_⟩, chain_congr (u := f) rfl rfl rfl hch⟩⟩
          exact ⟨wd, _, hw, hh', hlt, by simp; omega, by simp [St.setTop, ha], hloc⟩
        · cases hr
    | ret =>
      dsimp only
      simp only [hstep] at hr
      split at hr
      · rename_i heq
        cases hr
        cases rest with
        | nil => trivial
        | cons g r' =>
          dsimp only
          obtain ⟨⟨wu, hwu, hg⟩, hc'⟩ := hch
          rw [hw] at hwu; cases hwu
          apply safe_next
          unfold Inv
          refine ⟨?_, hc'⟩
          rw [heq] at hd ha
          simp at hd ha
          rw [hd, ha]; exact hg
      · cases hr

end Sporos
