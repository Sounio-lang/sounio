/-
The DAG semantics simulates the list semantics, part 3: `copy` and `tput`, the
two primitives that allocate nodes, and `α_ext`: growing the arena does not
change what a valid state stands for.
-/
import Sporos.DagPrim

namespace Sporos

theorem α_ext {t : DSt} (hv : Valid t) {ar' : Arena} (he : Ext t.ar ar') :
    α { t with ar := ar' } = α t := by
  obtain ⟨hg, hds, hax, hrs, hmem, htab, hout, hpc⟩ := hv
  simp only [α]
  congr 1
  · exact map_αv_ext hg he hds
  · exact map_αv_ext hg he hax
  · apply List.map_congr_left; intro g hg'
    simp only [αf]; rw [map_αv_ext hg he (hrs g hg')]
  · funext x; exact αv_ext hg he (hmem x)
  · apply List.map_congr_left; intro tb htb
    apply List.map_congr_left; intro kv hkv
    rw [αv_ext hg he (htab tb htb kv hkv)]
  · apply List.map_congr_left; intro bn hbn
    rw [hg.leaves_ext he (hout bn hbn)]
  · exact hg.leaves_ext he hpc

theorem sim_tput (ph : Phys) (v k tb : DVal) (t : DSt) (o : Origin) (qn : Nat) (hv : Valid t)
    (hvn : v.n < t.ar.length) (hq : qn < t.ar.length)
    (hqd : leaves t.ar qn = (qPlan [v, k, tb] t.pc o).den (leaves t.ar)) :
    execPrim ph .tput ([v, k, tb].map (αv t.ar)) (α t) o = (execPrimD ph .tput [v, k, tb] t o qn).abs := by
  rw [qPlan_den] at hqd
  simp only [List.map] at hqd
  simp only [execPrim, execPrimD, List.map]
  have hlen : (α t).tabs.length = t.tabs.length := by simp [α]
  by_cases h : tb.w < t.tabs.length
  · have h' : (αv t.ar tb).w < (α t).tabs.length := by rw [hlen]; exact h
    rw [dif_pos h', dif_pos h]
    simp only [DPR.abs, List.map]
    have he : Ext t.ar (t.ar ++ [DNode.join v.n qn]) := ⟨_, rfl⟩
    have hnew : leaves (t.ar ++ [DNode.join v.n qn]) t.ar.length = (αv t.ar v).p ++ leaves t.ar qn :=
      leaves_join hv.1.1 hvn hq
    have hα := α_ext hv he
    congr 1
    simp only [α] at hα ⊢
    simp only [St.mk.injEq] at hα
    obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11⟩ := hα
    simp only [St.mk.injEq, List.map_set, h1, h2, h3, h4, h7, h10, h11, true_and, and_true]
    congr 1
    simp only [List.map_cons, List.getElem_map]
    congr 1
    · simp only [αv, hnew, hqd]
    · apply List.map_congr_left; intro kv hkv
      rw [αv_ext hv.1 he (hv.2.2.2.2.2.1 _ (List.getElem_mem h) kv hkv)]
  · have h' : ¬ (αv t.ar tb).w < (α t).tabs.length := by rw [hlen]; exact h
    rw [dif_neg h', dif_neg h]; rfl

/-- `n` joins, each of node `f i` with `qn`, pushed in order. -/
theorem leaves_joins {ar : Arena} (hw : AWF ar) (f : Nat → Nat) (qn : Nat)
    (hf : ∀ i, f i < ar.length) (hq : qn < ar.length) :
    ∀ n, AWF (ar ++ (List.range n).map (fun i => DNode.join (f i) qn)) ∧
      ∀ i < n, leaves (ar ++ (List.range n).map (fun i => DNode.join (f i) qn)) (ar.length + i) =
        leaves ar (f i) ++ leaves ar qn := by
  intro n
  induction n with
  | zero => exact ⟨by simpa using hw, fun i h => absurd h (Nat.not_lt_zero _)⟩
  | succ n ih =>
    obtain ⟨hw', hl⟩ := ih
    have heq : ar ++ (List.range (n + 1)).map (fun i => DNode.join (f i) qn) =
        (ar ++ (List.range n).map (fun i => DNode.join (f i) qn)) ++ [DNode.join (f n) qn] := by
      simp [List.range_succ, List.map_append]
    have hlen : (ar ++ (List.range n).map (fun i => DNode.join (f i) qn)).length = ar.length + n := by simp
    have hfa : f n < (ar ++ (List.range n).map (fun i => DNode.join (f i) qn)).length := by
      rw [hlen]; exact Nat.lt_of_lt_of_le (hf n) (Nat.le_add_right _ _)
    have hqa : qn < (ar ++ (List.range n).map (fun i => DNode.join (f i) qn)).length := by
      rw [hlen]; exact Nat.lt_of_lt_of_le hq (Nat.le_add_right _ _)
    rw [heq]
    refine ⟨hw'.push_join hfa hqa, fun i hi => ?_⟩
    by_cases hin : i < n
    · rw [leaves_append hw' _ (by rw [hlen]; omega), hl i hin]
    · have : i = n := by omega
      subst this
      rw [← hlen, leaves_join hw' hfa hqa, leaves_append hw _ (hf i), leaves_append hw _ hq]

theorem sim_copy (ph : Phys) (n dst src : DVal) (t : DSt) (o : Origin) (qn : Nat) (hv : Valid t)
    (hq : qn < t.ar.length)
    (hqd : leaves t.ar qn = (qPlan [n, dst, src] t.pc o).den (leaves t.ar)) :
    execPrim ph .copy ([n, dst, src].map (αv t.ar)) (α t) o = (execPrimD ph .copy [n, dst, src] t o qn).abs := by
  rw [qPlan_den] at hqd
  simp only [List.map] at hqd
  simp only [execPrim, execPrimD, List.map]
  by_cases h : src.w + n.w ≤ t.msize ∧ dst.w + n.w ≤ t.msize
  · have h' : (αv t.ar src).w + (αv t.ar n).w ≤ (α t).msize ∧ (αv t.ar dst).w + (αv t.ar n).w ≤ (α t).msize := h
    rw [if_pos h', if_pos h]
    simp only [DPR.abs, List.map]
    obtain ⟨hw', hl⟩ := leaves_joins hv.1.1 (fun i => (t.mem (src.w + i)).n) qn
      (fun i => hv.2.2.2.2.1 _) hq n.w
    have he : Ext t.ar (t.ar ++ (List.range n.w).map fun i => DNode.join (t.mem (src.w + i)).n qn) := ⟨_, rfl⟩
    have hα := α_ext hv he
    congr 1
    simp only [α] at hα ⊢
    simp only [St.mk.injEq] at hα
    obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11⟩ := hα
    simp only [St.mk.injEq, h1, h2, h3, h7, h10, h11, true_and, and_true]
    funext x
    by_cases hx : dst.w ≤ x ∧ x < dst.w + n.w
    · simp only [hx, and_self, if_true, αv]
      simp only [αv] at hqd
      rw [hl (x - dst.w) (by omega), ← hqd]
    · have hx' : ¬ ((αv t.ar dst).w ≤ x ∧ x < (αv t.ar dst).w + (αv t.ar n).w) := hx
      rw [if_neg hx', if_neg hx]
      exact (αv_ext hv.1 he (hv.2.2.2.2.1 x)).symm
  · have h' : ¬ ((αv t.ar src).w + (αv t.ar n).w ≤ (α t).msize ∧ (αv t.ar dst).w + (αv t.ar n).w ≤ (α t).msize) := h
    rw [if_neg h', if_neg h]; rfl

end Sporos
