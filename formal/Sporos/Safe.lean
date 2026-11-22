/-
Soundness of the Sporos stack-effect checker (`step_safe`).
-/
import Sporos.Check

namespace Sporos

/-! ### What a checked word gives -/

theorem checkWord_len {pr : Prog} {wd : Word} {H : List Height} (h : checkWord pr wd H = true) :
    H.length = wd.code.length := by
  simp only [checkWord, Bool.and_eq_true, beq_iff_eq] at h; exact h.1.1.1.1

theorem checkWord_entry {pr : Prog} {wd : Word} {H : List Height} (h : checkWord pr wd H = true) :
    0 < wd.code.length ∧ H[0]? = some ⟨wd.ins, 0⟩ := by
  simp only [checkWord, Bool.and_eq_true, beq_iff_eq, Bool.not_eq_true', List.isEmpty_eq_false_iff] at h
  obtain ⟨⟨⟨⟨_, hne⟩, h0⟩, _⟩, _⟩ := h
  exact ⟨List.length_pos_iff.mpr hne, h0⟩

theorem checkWord_at {pr : Prog} {wd : Word} {H : List Height} (h : checkWord pr wd H = true)
    {i : Nat} {ins : Instr} {hi : Height} (hc : wd.code[i]? = some ins) (hh : H[i]? = some hi) :
    ∃ r, hstep pr wd hi ins = some r ∧
      ∀ h', r = some h' → i + 1 < wd.code.length ∧ H[i + 1]? = some h' := by
  simp only [checkWord, Bool.and_eq_true] at h
  have hall := h.1.2
  have hlt : i < wd.code.length := by
    rcases List.getElem?_eq_some_iff.mp hc with ⟨hl, _⟩; exact hl
  have := List.all_eq_true.mp hall i (List.mem_range.mpr hlt)
  rw [hc, hh] at this
  dsimp only at this
  cases hs : hstep pr wd hi ins with
  | none => rw [hs] at this; cases this
  | some r =>
    refine ⟨r, rfl, fun h' e => ?_⟩
    subst e
    rw [hs] at this
    simp only [Bool.and_eq_true, decide_eq_true_eq, beq_iff_eq] at this
    exact this

theorem lookup_mem {l : List (Nat × Height)} {t : Nat} {a : Height} (h : l.lookup t = some a) :
    (t, a) ∈ l := by
  induction l with
  | nil => simp [List.lookup] at h
  | cons x l ih =>
    obtain ⟨k, v⟩ := x
    simp only [List.lookup] at h
    by_cases e : t = k
    · subst e; simp at h; subst h; simp
    · have : (t == k) = false := by simp [e]
      rw [this] at h; exact List.mem_cons_of_mem _ (ih h)

theorem checkWord_ann {pr : Prog} {wd : Word} {H : List Height} (h : checkWord pr wd H = true)
    {t : Nat} {a : Height} (hl : wd.ann.lookup t = some a) :
    t < wd.code.length ∧ H[t]? = some a := by
  simp only [checkWord, Bool.and_eq_true] at h
  have := List.all_eq_true.mp h.2 (t, a) (lookup_mem hl)
  simp only [Bool.and_eq_true, decide_eq_true_eq, beq_iff_eq] at this
  exact this

/-! ### Primitives -/

theorem St.st_fields {s s' : St} {a n v : Nat} {p : Prov} (h : s.st a n v p = some s') :
    s'.rs = s.rs ∧ s'.aux = s.aux ∧ s'.ds = s.ds := by
  unfold St.st at h; split at h
  · cases h; exact ⟨rfl, rfl, rfl⟩
  · cases h

theorem execPrim_ok {ph : Phys} {p : Prim} {args : List Val} {s : St} {o : Origin} {res : List Val} {s' : St}
    (hl : args.length = p.eff.1) (h : execPrim ph p args s o = .ok res s') :
    res.length = p.eff.2 ∧ s'.rs = s.rs ∧ s'.aux = s.aux ∧ s'.ds = s.ds := by
  cases p <;> simp only [Prim.eff] at hl <;>
    rcases args with _ | ⟨a, _ | ⟨b, _ | ⟨c, _ | ⟨d, _ | ⟨e, _⟩⟩⟩⟩⟩ <;> simp at hl <;>
    simp only [execPrim] at h <;> (repeat' split at h) <;>
    first
      | (cases h; simp [Prim.eff]; done)
      | (cases h; done)
      | (cases h; exact ⟨rfl, St.st_fields (by assumption)⟩)

theorem execPrim_fail {ph : Phys} {p : Prim} {args : List Val} {s : St} {o : Origin} {e : Err}
    (hl : args.length = p.eff.1) (h : execPrim ph p args s o = .fail e) : e.static = false := by
  cases p <;> simp only [Prim.eff] at hl <;>
    rcases args with _ | ⟨a, _ | ⟨b, _ | ⟨c, _ | ⟨d, _ | ⟨e, _⟩⟩⟩⟩⟩ <;> simp at hl <;>
    simp only [execPrim] at h <;> (repeat' split at h) <;> (first | (cases h; rfl) | cases h)

theorem execPrim_exit {ph : Phys} {args : List Val} {s : St} {o : Origin} {res : List Val} {s' : St} :
    execPrim ph .exit args s o ≠ .ok res s' := by
  intro h
  rcases args with _ | ⟨a, _ | ⟨b, _⟩⟩ <;> simp only [execPrim] at h <;> cases h

end Sporos

/-! ### One step -/

namespace Sporos

/-- What `step_safe` promises of a step. -/
def Safe (pr : Prog) : Res → Prop
  | .next s' => Inv pr s'
  | .halt _ _ => True
  | .fail e => e.static = false

theorem St.setTop_eq {s : St} {f g : Frame} {rest : List Frame} (h : s.rs = f :: rest) :
    s.setTop g = { s with rs := g :: rest } := by
  unfold St.setTop; rw [h]

theorem chain_congr {pr : Prog} {u u' : Frame} {rest : List Frame} (hw : u'.word = u.word)
    (hb : u'.base = u.base) (ha : u'.abase = u.abase) (h : Chain pr u rest) : Chain pr u' rest := by
  cases rest with
  | nil => trivial
  | cons g r =>
    obtain ⟨⟨wu, hwu, hg⟩, hc⟩ := h
    refine ⟨⟨wu, by rw [hw]; exact hwu, by rw [hb, ha]; exact hg⟩, hc⟩

theorem checkProg_word {pr : Prog} (hp : checkProg pr = true) {w : Nat} {wd : Word}
    (hw : pr.words[w]? = some wd) : checkWord pr wd (tbl pr w) = true := by
  simp only [checkProg, Bool.and_eq_true] at hp
  have := List.all_eq_true.mp hp.1 wd (List.mem_of_getElem? hw)
  simp only [tbl, hw]; exact this

/-- Same word, a new index whose declared height matches the stacks. -/
theorem inv_move {pr : Prog} {s' : St} {f g : Frame} {rest : List Frame} {wd : Word} {h' : Height}
    (hrs : s'.rs = g :: rest) (hw : pr.words[f.word]? = some wd) (hgw : g.word = f.word)
    (hgb : g.base = f.base) (hga : g.abase = f.abase) (hgl : g.locs.length = wd.locals)
    (hlt : g.pc < wd.code.length) (hh : (tbl pr f.word)[g.pc]? = some h')
    (hd : s'.ds.length = f.base + h'.d) (ha : s'.aux.length = f.abase + h'.a)
    (hc : Chain pr f rest) : Inv pr s' := by
  unfold Inv; rw [hrs]
  refine ⟨⟨wd, h', by rw [hgw]; exact hw, by rw [hgw]; exact hh, hlt, by rw [hd, hgb], by rw [ha, hga], hgl⟩,
    chain_congr hgw hgb hga hc⟩

end Sporos
