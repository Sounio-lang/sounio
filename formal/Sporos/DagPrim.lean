/-
The DAG semantics simulates the list semantics, part 2: primitives that do not
allocate. `sim_prim` covers all but seven by one tactic; `sim_tnew`, `sim_tget`,
`sim_putb`, `sim_putn` and `sim_mcmp` are proved one by one. `copy` and `tput`
allocate nodes and are proved separately.
-/
import Sporos.DagSim
namespace Sporos

def DPR.abs : DPR → PR
  | .ok res t' => .ok (res.map fun r => ⟨r.1, r.2.den (leaves t'.ar)⟩) (α t')
  | .halt c => .halt c
  | .fail e => .fail e

theorem α_pc (t : DSt) : (α t).pc = leaves t.ar t.pc := rfl

theorem sim_prim (ph : Phys) (p : Prim) (args : List DVal) (t : DSt) (o : Origin) (qn : Nat)
    (hl : args.length = p.eff.1)
    (hp : p ≠ .copy ∧ p ≠ .tput ∧ p ≠ .tnew ∧ p ≠ .tget ∧ p ≠ .putb ∧ p ≠ .putn ∧ p ≠ .mcmp)
    (hqd : leaves t.ar qn = (qPlan args t.pc o).den (leaves t.ar)) :
    execPrim ph p (args.map (αv t.ar)) (α t) o = (execPrimD ph p args t o qn).abs := by
  have hq := hqd.trans (qPlan_den t.ar args t.pc o)
  cases p <;> simp only [Prim.eff] at hl <;>
    rcases args with _ | ⟨a, _ | ⟨b, _ | ⟨c, _ | ⟨d, _ | ⟨e, _⟩⟩⟩⟩⟩ <;> (try simp at hl) <;> (try simp at hp) <;>
    simp only [List.map, αv] at hq <;>
    simp only [execPrim, execPrimD, List.map, αv, α_pc] <;>
    (first
      | rfl
      | (rw [← hq, α_st]; rcases t.st _ _ _ _ with _ | s' <;> rfl)
      | (rw [α_ld]; rcases t.ld _ _ with _ | ⟨v, pm⟩ <;>
          simp only [Option.map, DPR.abs, PE.den, qPlan_den, List.map, αv, α_pc] <;> rfl)
      | (simp only [DPR.abs, qPlan_den, List.map, αv, α]; rfl)
      | (split <;> rename_i hc <;> (try simp only [α] at hc) <;>
          simp only [hc, DPR.abs, if_true, if_false, qPlan_den, List.map, αv, α] <;> rfl)
      | (simp only [← hq, DPR.abs, α, List.map, List.map_append, List.map_reverse, List.map_map,
          List.length_map, Function.comp_def]; rfl)
      | (split <;> rename_i hc <;> (try simp only [α, List.length_map] at hc) <;>
          simp only [hc, ← hq, DPR.abs, dite_true, dite_false, if_true, if_false, qPlan_den, PE.den,
            foldPlan2_den, List.map, αv, α, List.length_map, List.getElem_map, tfind_map] <;>
          first
            | rfl
            | (congr 2; funext x; split <;> rfl)
            | (rcases tfind' _ _ with _ | v <;> rfl)))



theorem sim_tnew (ph : Phys) (t : DSt) (o : Origin) (qn : Nat) :
    execPrim ph .tnew ([].map (αv t.ar)) (α t) o = (execPrimD ph .tnew [] t o qn).abs := by
  simp only [execPrim, execPrimD, DPR.abs, List.map, qPlan_den]
  simp only [α, List.length_map, List.map_append, List.map]

theorem sim_putb (ph : Phys) (b : DVal) (t : DSt) (o : Origin) (qn : Nat)
    (hqd : leaves t.ar qn = (qPlan [b] t.pc o).den (leaves t.ar)) :
    execPrim ph .putb ([b].map (αv t.ar)) (α t) o = (execPrimD ph .putb [b] t o qn).abs := by
  rw [qPlan_den] at hqd
  simp only [List.map] at hqd
  simp only [execPrim, execPrimD, DPR.abs, List.map]
  simp only [α, List.map, ← hqd]
  rfl

theorem sim_putn (ph : Phys) (b : DVal) (t : DSt) (o : Origin) (qn : Nat)
    (hqd : leaves t.ar qn = (qPlan [b] t.pc o).den (leaves t.ar)) :
    execPrim ph .putn ([b].map (αv t.ar)) (α t) o = (execPrimD ph .putn [b] t o qn).abs := by
  rw [qPlan_den] at hqd
  simp only [List.map] at hqd
  simp only [execPrim, execPrimD, DPR.abs, List.map]
  simp only [α, List.map_append, List.map_reverse, List.map_map, Function.comp_def, ← hqd]
  rfl

theorem sim_tget (ph : Phys) (k tb : DVal) (t : DSt) (o : Origin) (qn : Nat) :
    execPrim ph .tget ([k, tb].map (αv t.ar)) (α t) o = (execPrimD ph .tget [k, tb] t o qn).abs := by
  simp only [execPrim, execPrimD, List.map]
  have hlen : (α t).tabs.length = t.tabs.length := by simp [α]
  by_cases h : tb.w < t.tabs.length
  · have h' : (αv t.ar tb).w < (α t).tabs.length := by rw [hlen]; exact h
    rw [dif_pos h', dif_pos h]
    have hg : (α t).tabs[(αv t.ar tb).w]'h' = t.tabs[tb.w].map fun kv => (kv.1, αv t.ar kv.2) := by
      simp [α, αv]
    rw [hg, tfind_map]
    show _ = DPR.abs (match tfind' t.tabs[tb.w] (αv t.ar k).w with
      | some v => DPR.ok [(wrap 1, qPlan [k, tb] t.pc o), (v.w, (PE.node v.n).app (qPlan [k, tb] t.pc o))] t
      | none => DPR.ok [(wrap 0, qPlan [k, tb] t.pc o), (wrap 0, qPlan [k, tb] t.pc o)] t)
    rcases tfind' t.tabs[tb.w] (αv t.ar k).w with _ | v
    · simp only [Option.map, DPR.abs, List.map, qPlan_den]; rfl
    · simp only [Option.map, DPR.abs, List.map, qPlan_den, PE.den]; rfl
  · have h' : ¬ (αv t.ar tb).w < (α t).tabs.length := by rw [hlen]; exact h
    rw [dif_neg h', dif_neg h]; rfl

theorem sim_mcmp (ph : Phys) (n y x : DVal) (t : DSt) (o : Origin) (qn : Nat) :
    execPrim ph .mcmp ([n, y, x].map (αv t.ar)) (α t) o = (execPrimD ph .mcmp [n, y, x] t o qn).abs := by
  simp only [execPrim, execPrimD, List.map]
  by_cases h : x.w + n.w ≤ t.msize ∧ y.w + n.w ≤ t.msize
  · have h' : (αv t.ar x).w + (αv t.ar n).w ≤ (α t).msize ∧ (αv t.ar y).w + (αv t.ar n).w ≤ (α t).msize := h
    rw [if_pos h', if_pos h]
    simp only [DPR.abs, List.map, PE.den, qPlan_den, foldPlan2_den]
    rfl
  · have h' : ¬ ((αv t.ar x).w + (αv t.ar n).w ≤ (α t).msize ∧ (αv t.ar y).w + (αv t.ar n).w ≤ (α t).msize) := h
    rw [if_neg h', if_neg h]; rfl

end Sporos
