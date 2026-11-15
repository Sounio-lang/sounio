/-
The DAG semantics simulates the list semantics, part 4: the step and the run.

`dstep_sim`: from a valid DAG state, one DAG step stands for one list step of the
state it stands for, and leaves a valid state. `drun_init`: from the initial
states, `n` DAG steps and `n` list steps give the same result, for every program,
input and `n`, with each output byte's provenance read back from the arena.
-/
import Sporos.DagAlloc
import Sporos.Run

namespace Sporos

theorem Valid.ext {t : DSt} (hv : Valid t) {ar' : Arena} (hg : Good ar') (he : Ext t.ar ar') :
    Valid { t with ar := ar' } := by
  obtain ⟨_, hds, hax, hrs, hmem, htab, hout, hpc⟩ := hv
  have hl := he.len
  exact ⟨hg, fun d h => Nat.lt_of_lt_of_le (hds d h) hl, fun d h => Nat.lt_of_lt_of_le (hax d h) hl,
    fun g hg d h => Nat.lt_of_lt_of_le (hrs g hg d h) hl, fun x => Nat.lt_of_lt_of_le (hmem x) hl,
    fun tb h kv hk => Nat.lt_of_lt_of_le (htab tb h kv hk) hl, fun bn h => Nat.lt_of_lt_of_le (hout bn h) hl,
    Nat.lt_of_lt_of_le hpc hl⟩

theorem buildAll_spec : ∀ (res : List (Nat × PE)) (ar : Arena), Good ar → (∀ r ∈ res, r.2.ok ar.length) →
    Good (buildAll res ar).2 ∧ Ext ar (buildAll res ar).2 ∧
      (∀ d ∈ (buildAll res ar).1, d.n < (buildAll res ar).2.length) ∧
      (buildAll res ar).1.map (αv (buildAll res ar).2) = res.map (fun r => ⟨r.1, r.2.den (leaves ar)⟩)
  | [], ar, hg, _ => ⟨hg, Ext.refl _, by simp [buildAll], rfl⟩
  | (w, e) :: r, ar, hg, hok => by
    have he : e.ok ar.length := hok (w, e) (by simp)
    obtain ⟨g1, e1, l1, d1⟩ := build_spec e ar hg he
    have hr : ∀ x ∈ r, x.2.ok (build e ar).2.length := fun x hx => PE.ok_mono (hok x (by simp [hx])) e1.len
    obtain ⟨g2, e2, l2, d2⟩ := buildAll_spec r (build e ar).2 g1 hr
    refine ⟨g2, Ext.trans e1 e2, ?_, ?_⟩
    · intro d hd
      simp only [buildAll, List.mem_cons] at hd
      rcases hd with rfl | hd
      · exact Nat.lt_of_lt_of_le l1 e2.len
      · exact l2 d hd
    · simp only [buildAll, List.map_cons]
      rw [d2]
      congr 1
      · simp only [αv]; rw [g1.leaves_ext e2 l1, d1]
      · apply List.map_congr_left; intro x hx
        rw [PE.den_ext hg e1 (hok x (by simp [hx]))]

theorem α_setTop (t : DSt) (f : DFrame) : α (t.setTop f) = (α t).setTop (αf t.ar f) := by
  unfold DSt.setTop St.setTop
  rcases h : t.rs with _ | ⟨g, r⟩ <;> simp [α, h]

theorem Valid.setTop {t : DSt} (hv : Valid t) {f : DFrame} (hf : ∀ d ∈ f.locs, d.n < t.ar.length) :
    Valid (t.setTop f) := by
  unfold DSt.setTop
  rcases h : t.rs with _ | ⟨g, r⟩
  · exact hv
  · obtain ⟨hg, hds, hax, hrs, hmem, htab, hout, hpc⟩ := hv
    refine ⟨hg, hds, hax, ?_, hmem, htab, hout, hpc⟩
    intro g' hg' d hd
    simp only [List.mem_cons] at hg'
    rcases hg' with rfl | hg'
    · exact hf d hd
    · exact hrs g' (by rw [h]; simp [hg']) d hd

/-- Pushing a value: the arena grows by the plan, the state it stands for gains the value. -/
theorem push_spec {t : DSt} (hv : Valid t) (w : Nat) {e : PE} (he : e.ok t.ar.length) :
    Valid (t.push w e) ∧ α (t.push w e) = { α t with ds := ⟨w, e.den (leaves t.ar)⟩ :: (α t).ds } := by
  obtain ⟨g1, e1, l1, d1⟩ := build_spec e t.ar hv.1 he
  have hv1 := hv.ext g1 e1
  refine ⟨?_, ?_⟩
  · obtain ⟨hg, hds, hax, hrs, hmem, htab, hout, hpc⟩ := hv1
    refine ⟨hg, ?_, hax, hrs, hmem, htab, hout, hpc⟩
    intro d hd
    simp only [DSt.push, List.mem_cons] at hd
    rcases hd with rfl | hd
    · exact l1
    · exact hds d hd
  · have hα := α_ext hv e1
    have : t.push w e = { { t with ar := (build e t.ar).2 } with ds := ⟨w, (build e t.ar).1⟩ :: t.ds } := rfl
    rw [this, ← hα]
    simp only [α, List.map_cons, αv, d1]

/-! ### What a primitive keeps valid -/

theorem tagPlan_ok {pc L : Nat} (o : Origin) {v : DVal} (hp : pc < L) (hv : v.n < L) :
    (tagPlan pc o v).ok L := ⟨⟨hv, hp⟩, trivial⟩

theorem foldPlan_ok {L : Nat} (f : Nat → DVal) (hf : ∀ i, (f i).n < L) (l : List Nat) :
    (l.foldr (fun i acc => PE.app (.node (f i).n) acc) .nil).ok L := by
  induction l with
  | nil => trivial
  | cons i r ih => exact ⟨hf i, ih⟩

theorem foldPlan2_ok {L : Nat} (f g : Nat → DVal) (hf : ∀ i, (f i).n < L) (hg : ∀ i, (g i).n < L)
    (l : List Nat) :
    (l.foldr (fun i acc => PE.app (.app (.node (f i).n) (.node (g i).n)) acc) .nil).ok L := by
  induction l with
  | nil => trivial
  | cons i r ih => exact ⟨⟨hf i, hg i⟩, ih⟩

theorem Valid.mem {t : DSt} (hv : Valid t) {m : Nat → DVal} (hm : ∀ x, (m x).n < t.ar.length) :
    Valid { t with mem := m } := by
  obtain ⟨hg, hds, hax, hrs, _, htab, hout, hpc⟩ := hv
  exact ⟨hg, hds, hax, hrs, hm, htab, hout, hpc⟩

theorem tfind'_mem : ∀ (l : List (Nat × DVal)) (k : Nat) {v : DVal}, tfind' l k = some v → ∃ kv ∈ l, kv.2 = v
  | [], _, _, h => by simp [tfind'] at h
  | (k', v') :: r, k, v, h => by
    simp only [tfind'] at h
    split at h
    · exact ⟨(k', v'), by simp, Option.some.inj h⟩
    · obtain ⟨kv, hkv, he⟩ := tfind'_mem r k h
      exact ⟨kv, by simp [hkv], he⟩

/-- The conclusion of `prim_valid`. -/
def PV (t s' : DSt) (res : List (Nat × PE)) : Prop :=
  Valid s' ∧ Ext t.ar s'.ar ∧ (∀ r ∈ res, r.2.ok s'.ar.length) ∧
    s'.ds = t.ds ∧ s'.aux = t.aux ∧ s'.rs = t.rs ∧ s'.pc = t.pc ∧ s'.step = t.step

theorem DSt.ld_ok {t : DSt} (hv : Valid t) {a n v : Nat} {pm : PE} (h : t.ld a n = some (v, pm)) :
    pm.ok t.ar.length := by
  unfold DSt.ld at h
  split at h
  · simp only [Option.some.injEq, Prod.mk.injEq] at h
    obtain ⟨-, rfl⟩ := h
    exact foldPlan_ok _ (fun i => hv.2.2.2.2.1 _) _
  · contradiction

theorem DSt.st_valid {t : DSt} (hv : Valid t) {a n v qn : Nat} (hq : qn < t.ar.length) {s' : DSt}
    (h : t.st a n v qn = some s') : PV t s' [] := by
  unfold DSt.st at h
  split at h
  · simp only [Option.some.injEq] at h
    subst h
    refine ⟨hv.mem ?_, Ext.refl _, by simp, rfl, rfl, rfl, rfl, rfl⟩
    intro x; dsimp only; split
    · exact hq
    · exact hv.2.2.2.2.1 x
  · contradiction

theorem Good.append_joins {ar : Arena} (hg : Good ar) (f : Nat → Nat) (qn : Nat)
    (hf : ∀ i, f i < ar.length) (hq : qn < ar.length) (n : Nat) :
    Good (ar ++ (List.range n).map (fun i => DNode.join (f i) qn)) := by
  refine ⟨(leaves_joins hg.1 f qn hf hq n).1, ?_⟩
  have : 0 < ar.length := by have := hg.2; rcases ar with _ | _ <;> simp_all
  rw [List.getElem?_append_left this]; exact hg.2

theorem valid_copy {t : DSt} (hv : Valid t) {qn : Nat} (hq : qn < t.ar.length) (a b c : DVal) :
    PV t { t with
      mem := fun x =>
        if b.w ≤ x ∧ x < b.w + a.w then { w := (t.mem (c.w + (x - b.w))).w, n := t.ar.length + (x - b.w) }
        else t.mem x,
      ar := t.ar ++ (List.range a.w).map (fun i => DNode.join (t.mem (c.w + i)).n qn) } [] := by
  have hg := hv.1.append_joins (fun i => (t.mem (c.w + i)).n) qn (fun _ => hv.2.2.2.2.1 _) hq a.w
  have he : Ext t.ar (t.ar ++ (List.range a.w).map (fun i => DNode.join (t.mem (c.w + i)).n qn)) := ⟨_, rfl⟩
  have hv1 := hv.ext hg he
  refine ⟨hv1.mem ?_, he, by simp, rfl, rfl, rfl, rfl, rfl⟩
  intro x; dsimp only; split
  · simp only [List.length_append, List.length_map, List.length_range]; omega
  · exact Nat.lt_of_lt_of_le (hv.2.2.2.2.1 x) he.len

theorem valid_tput {t : DSt} (hv : Valid t) {qn : Nat} (hq : qn < t.ar.length) (v k tb : DVal)
    (hvn : v.n < t.ar.length) (h : tb.w < t.tabs.length) :
    PV t { t with ar := t.ar ++ [.join v.n qn],
                  tabs := t.tabs.set tb.w ((k.w, ⟨v.w, t.ar.length⟩) :: t.tabs[tb.w]) } [] := by
  have hg : Good (t.ar ++ [.join v.n qn]) := by
    refine ⟨hv.1.1.push_join hvn hq, ?_⟩
    have : 0 < t.ar.length := by have := hv.1.2; rcases h : t.ar with _ | _ <;> simp_all
    rw [List.getElem?_append_left this]; exact hv.1.2
  have he : Ext t.ar (t.ar ++ [.join v.n qn]) := ⟨_, rfl⟩
  obtain ⟨_, hds, hax, hrs, hmem, htab, hout, hpc⟩ := hv.ext hg he
  refine ⟨⟨hg, hds, hax, hrs, hmem, ?_, hout, hpc⟩, he, by simp, rfl, rfl, rfl, rfl, rfl⟩
  intro tb' htb kv hkv
  dsimp only at htb ⊢
  rcases List.mem_or_eq_of_mem_set htb with htb | rfl
  · exact htab tb' htb kv hkv
  · simp only [List.mem_cons] at hkv
    rcases hkv with rfl | hkv
    · simp
    · exact htab _ (List.getElem_mem h) kv hkv

theorem prim_valid (ph : Phys) (p : Prim) (args : List DVal) (t : DSt) (o : Origin) (qn : Nat)
    (hv : Valid t) (hq : qn < t.ar.length) (ha : ∀ d ∈ args, d.n < t.ar.length)
    {res : List (Nat × PE)} {s' : DSt} (h : execPrimD ph p args t o qn = .ok res s') : PV t s' res := by
  have hqo := qPlan_ok ha hv.2.2.2.2.2.2.2 o
  have hpc := hv.2.2.2.2.2.2.2
  cases p <;>
    rcases args with _ | ⟨a, _ | ⟨b, _ | ⟨c, _ | ⟨d, _ | ⟨e, _⟩⟩⟩⟩⟩ <;>
    simp only [execPrimD] at h <;> (repeat' split at h) <;> (try contradiction) <;>
    (try (simp only [DPR.ok.injEq] at h; obtain ⟨rfl, rfl⟩ := h)) <;>
    first
      | exact DSt.st_valid hv hq (by assumption)
      | (refine ⟨hv, Ext.refl _, ?_, rfl, rfl, rfl, rfl, rfl⟩
         intro r hr
         simp only [List.mem_cons, List.not_mem_nil, or_false] at hr <;>
         rcases hr with rfl | rfl | rfl <;> dsimp only <;>
           first
             | exact hqo
             | exact tagPlan_ok o hpc (ha _ (by simp))
             | exact ⟨hqo, DSt.ld_ok hv (by assumption)⟩
             | exact ⟨hqo, foldPlan2_ok _ _ (fun _ => hv.2.2.2.2.1 _) (fun _ => hv.2.2.2.2.1 _) _⟩
             | exact ⟨hqo, foldPlan_ok _ (fun _ => hv.2.2.2.2.1 _) _⟩
             | exact ⟨hqo, trivial⟩
             | exact ⟨hpc, trivial⟩
             | (refine ⟨?_, hqo⟩
                obtain ⟨kv, hkv, rfl⟩ := tfind'_mem _ _ (by assumption)
                exact hv.2.2.2.2.2.1 _ (List.getElem_mem _) kv hkv))
      | exact valid_copy hv hq _ _ _
      | exact valid_tput hv hq _ _ _ (ha _ (by simp)) (by assumption)
      | (refine ⟨hv.mem ?_, Ext.refl _, by simp, rfl, rfl, rfl, rfl, rfl⟩
         intro x; dsimp only; split
         · exact hq
         · exact hv.2.2.2.2.1 x)
      | (obtain ⟨hg, hds, hax, hrs, hmem, htab, hout, hpc'⟩ := hv
         refine ⟨⟨hg, hds, hax, hrs, hmem, ?_, hout, hpc'⟩, Ext.refl _, ?_, rfl, rfl, rfl, rfl, rfl⟩
         · intro tb htb kv hkv
           simp only [List.mem_append, List.mem_cons, List.not_mem_nil, or_false] at htb
           rcases htb with htb | rfl
           · exact htab tb htb kv hkv
           · simp at hkv
         · intro r hr
           simp only [List.mem_cons, List.not_mem_nil, or_false] at hr
           subst hr; exact hqo)
      | (obtain ⟨hg, hds, hax, hrs, hmem, htab, hout, hpc'⟩ := hv
         refine ⟨⟨hg, hds, hax, hrs, hmem, htab, ?_, hpc'⟩, Ext.refl _, by simp, rfl, rfl, rfl, rfl, rfl⟩
         intro bn hbn
         simp only [List.mem_cons, List.mem_append, List.mem_reverse, List.mem_map, List.mem_range] at hbn
         first
           | (rcases hbn with rfl | hbn
              · exact hq
              · exact hout bn hbn)
           | (rcases hbn with ⟨_, _, rfl⟩ | hbn
              · exact hq
              · exact hout bn hbn))
theorem sim_prim_all (ph : Phys) (p : Prim) (args : List DVal) (t : DSt) (o : Origin) (qn : Nat)
    (hv : Valid t) (hq : qn < t.ar.length) (ha : ∀ d ∈ args, d.n < t.ar.length)
    (hl : args.length = p.eff.1)
    (hqd : leaves t.ar qn = (qPlan args t.pc o).den (leaves t.ar)) :
    execPrim ph p (args.map (αv t.ar)) (α t) o = (execPrimD ph p args t o qn).abs := by
  cases p
  case copy => rcases args with _ | ⟨a, _ | ⟨b, _ | ⟨c, _ | ⟨d, _ | ⟨e, _⟩⟩⟩⟩⟩ <;> simp only [Prim.eff] at hl <;> (try simp at hl) <;> exact sim_copy ph _ _ _ t o qn hv hq hqd
  case tput => rcases args with _ | ⟨a, _ | ⟨b, _ | ⟨c, _ | ⟨d, _ | ⟨e, _⟩⟩⟩⟩⟩ <;> simp only [Prim.eff] at hl <;> (try simp at hl) <;> exact sim_tput ph _ _ _ t o qn hv (ha _ (by simp)) hq hqd
  case tnew => rcases args with _ | ⟨a, _ | ⟨b, _ | ⟨c, _ | ⟨d, _ | ⟨e, _⟩⟩⟩⟩⟩ <;> simp only [Prim.eff] at hl <;> (try simp at hl) <;> exact sim_tnew ph t o qn
  case tget => rcases args with _ | ⟨a, _ | ⟨b, _ | ⟨c, _ | ⟨d, _ | ⟨e, _⟩⟩⟩⟩⟩ <;> simp only [Prim.eff] at hl <;> (try simp at hl) <;> exact sim_tget ph _ _ t o qn
  case putb => rcases args with _ | ⟨a, _ | ⟨b, _ | ⟨c, _ | ⟨d, _ | ⟨e, _⟩⟩⟩⟩⟩ <;> simp only [Prim.eff] at hl <;> (try simp at hl) <;> exact sim_putb ph _ t o qn hqd
  case putn => rcases args with _ | ⟨a, _ | ⟨b, _ | ⟨c, _ | ⟨d, _ | ⟨e, _⟩⟩⟩⟩⟩ <;> simp only [Prim.eff] at hl <;> (try simp at hl) <;> exact sim_putn ph _ t o qn hqd
  case mcmp => rcases args with _ | ⟨a, _ | ⟨b, _ | ⟨c, _ | ⟨d, _ | ⟨e, _⟩⟩⟩⟩⟩ <;> simp only [Prim.eff] at hl <;> (try simp at hl) <;> exact sim_mcmp ph _ _ _ t o qn
  case streq => rcases args with _ | ⟨a, _ | ⟨b, _ | ⟨c, _ | ⟨d, _ | ⟨e, _⟩⟩⟩⟩⟩ <;> simp only [Prim.eff] at hl <;> (try simp at hl) <;> exact sim_streq ph _ _ _ _ t o qn
  case strhash => rcases args with _ | ⟨a, _ | ⟨b, _ | ⟨c, _ | ⟨d, _ | ⟨e, _⟩⟩⟩⟩⟩ <;> simp only [Prim.eff] at hl <;> (try simp at hl) <;> exact sim_strhash ph _ _ t o qn
  all_goals exact sim_prim ph _ args t o qn hl (by simp) hqd
def DRes.abs : DRes → Res
  | .next t => .next (α t)
  | .halt c out ar => .halt c (out.map fun bn => (bn.1, leaves ar bn.2))
  | .fail e => .fail e

theorem αf_ext {a b : Arena} (ha : Good a) (h : Ext a b) {g : DFrame} (hg : ∀ d ∈ g.locs, d.n < a.length) :
    αf b g = αf a g := by
  simp only [αf]; rw [map_αv_ext ha h hg]

theorem adv_next {X : DSt} (hX : Valid X) (g : DFrame) (hg : ∀ d ∈ g.locs, d.n < X.ar.length) :
    (DRes.next (X.setTop g)).abs = .next ((α X).setTop (αf X.ar g)) ∧
      ∀ t', DRes.next (X.setTop g) = .next t' → Valid t' :=
  ⟨by simp only [DRes.abs, α_setTop], fun t' h => by cases h; exact hX.setTop hg⟩

theorem push_adv {X : DSt} (hX : Valid X) (w : Nat) {e : PE} (he : e.ok X.ar.length) (g : DFrame)
    (hg : ∀ d ∈ g.locs, d.n < X.ar.length) :
    (DRes.next ((X.push w e).setTop g)).abs =
        .next ({ α X with ds := ⟨w, e.den (leaves X.ar)⟩ :: (α X).ds }.setTop (αf X.ar g)) ∧
      ∀ t', DRes.next ((X.push w e).setTop g) = .next t' → Valid t' := by
  obtain ⟨hv2, hα2⟩ := push_spec hX w he
  have hext := (build_spec e X.ar hX.1 he).2.1
  obtain ⟨h1, h2⟩ := adv_next hv2 g (fun d hd => Nat.lt_of_lt_of_le (hg d hd) hext.len)
  refine ⟨?_, h2⟩
  rw [h1, hα2]
  have : (X.push w e).ar = (build e X.ar).2 := rfl
  rw [this, αf_ext hX.1 hext hg]

theorem dstep_sim (pr : Prog) (ph : Phys) (t : DSt) (hv : Valid t) :
    (dstep pr ph t).abs = step pr ph (α t) ∧ ∀ t', dstep pr ph t = .next t' → Valid t' := by
  obtain ⟨ds, aux, rs, mem, msize, hp, tabs, inp, ipos, out, pc, stp, ar⟩ := t
  rcases rs with _ | ⟨f, rest⟩
  · exact ⟨rfl, fun _ h => by simp [dstep] at h⟩
  · have hf : ∀ d ∈ f.locs, d.n < ar.length := hv.2.2.2.1 f (by simp)
    have hα : α ⟨ds, aux, f :: rest, mem, msize, hp, tabs, inp, ipos, out, pc, stp, ar⟩ =
        ⟨ds.map (αv ar), aux.map (αv ar), αf ar f :: rest.map (αf ar), fun x => αv ar (mem x), msize, hp,
          tabs.map (List.map fun kv => (kv.1, αv ar kv.2)), inp, ipos, out.map (fun bn => (bn.1, leaves ar bn.2)),
          leaves ar pc, stp⟩ := rfl
    rw [hα]
    simp only [dstep, step, αf]
    rcases hfe : fetch pr f.word f.pc with _ | ins
    · exact ⟨rfl, fun _ h => by simp at h⟩
    · cases ins <;> dsimp only
      case lit n =>
        have hv1 : Valid ⟨ds, aux, f :: rest, mem, msize, hp, tabs, inp, ipos, out, pc, stp + 1, ar⟩ := hv
        refine push_adv hv1 _ ?_ { f with pc := f.pc + 1 } hf
        exact ⟨hv.2.2.2.2.2.2.2, trivial⟩
      case pick n =>
        have hv1 : Valid ⟨ds, aux, f :: rest, mem, msize, hp, tabs, inp, ipos, out, pc, stp + 1, ar⟩ := hv
        rw [List.getElem?_map]
        rcases hn : ds[n]? with _ | v
        · exact ⟨rfl, fun _ h => by simp at h⟩
        · have hvn : v.n < ar.length := hv.2.1 v (List.mem_of_getElem? hn)
          refine push_adv hv1 _ ?_ { f with pc := f.pc + 1 } hf
          exact tagPlan_ok _ hv.2.2.2.2.2.2.2 hvn
      case fromr =>
        rcases aux with _ | ⟨v, r⟩
        · exact ⟨rfl, fun _ h => by simp at h⟩
        · have hv1 : Valid ⟨ds, r, f :: rest, mem, msize, hp, tabs, inp, ipos, out, pc, stp + 1, ar⟩ :=
            ⟨hv.1, hv.2.1, fun d hd => hv.2.2.1 d (by simp [hd]), hv.2.2.2⟩
          have hvn : v.n < ar.length := hv.2.2.1 v (by simp)
          refine push_adv hv1 _ ?_ { f with pc := f.pc + 1 } hf
          exact tagPlan_ok _ hv.2.2.2.2.2.2.2 hvn
      case rfetch =>
        rcases haux : aux with _ | ⟨v, r⟩
        · exact ⟨rfl, fun _ h => by simp at h⟩
        · have hv1 : Valid ⟨ds, v :: r, f :: rest, mem, msize, hp, tabs, inp, ipos, out, pc, stp + 1, ar⟩ :=
            haux ▸ hv
          have hvn : v.n < ar.length := hv.2.2.1 v (by simp [haux])
          refine push_adv hv1 _ ?_ { f with pc := f.pc + 1 } hf
          exact tagPlan_ok _ hv.2.2.2.2.2.2.2 hvn
      case locget i =>
        have hv1 : Valid ⟨ds, aux, f :: rest, mem, msize, hp, tabs, inp, ipos, out, pc, stp + 1, ar⟩ := hv
        rw [List.getElem?_map]
        rcases hn : f.locs[i]? with _ | v
        · exact ⟨rfl, fun _ h => by simp at h⟩
        · have hvn : v.n < ar.length := hf v (List.mem_of_getElem? hn)
          refine push_adv hv1 _ ?_ { f with pc := f.pc + 1 } hf
          exact tagPlan_ok _ hv.2.2.2.2.2.2.2 hvn
      case jmp tg =>
        have hv1 : Valid ⟨ds, aux, f :: rest, mem, msize, hp, tabs, inp, ipos, out, pc, stp + 1, ar⟩ := hv
        exact adv_next hv1 { f with pc := tg } hf
      case str i =>
        rcases hd : pr.data[i]? with _ | d
        · exact ⟨rfl, fun _ h => by simp at h⟩
        · have hv1 : Valid ⟨ds, aux, f :: rest, mem, msize, hp, tabs, inp, ipos, out, pc, stp + 1, ar⟩ := hv
          have hP : ((PE.node pc).app (PE.leaf ⟨none, f.word, f.pc, ph.core stp⟩)).ok ar.length :=
            ⟨hv.2.2.2.2.2.2.2, trivial⟩
          obtain ⟨hv2, hα2⟩ := push_spec hv1 (((pr.data.take i).map List.length).foldl (· + ·) 0) hP
          have hext := (build_spec _ ar hv.1 hP).2.1
          obtain ⟨h1, h2⟩ := push_adv hv2 d.length (PE.ok_mono hP hext.len) { f with pc := f.pc + 1 }
            (fun d hd => Nat.lt_of_lt_of_le (hf d hd) hext.len)
          refine ⟨?_, h2⟩
          rw [h1, hα2]
          have : (DSt.push ⟨ds, aux, f :: rest, mem, msize, hp, tabs, inp, ipos, out, pc, stp + 1, ar⟩
              (((pr.data.take i).map List.length).foldl (· + ·) 0)
              ((PE.node pc).app (PE.leaf ⟨none, f.word, f.pc, ph.core stp⟩))).ar =
              (build ((PE.node pc).app (PE.leaf ⟨none, f.word, f.pc, ph.core stp⟩)) ar).2 := rfl
          rw [this, PE.den_ext hv.1 hext hP, αf_ext hv.1 hext (g := { f with pc := f.pc + 1 }) hf]
          rfl
      case tor =>
        rcases ds with _ | ⟨v, r⟩
        · exact ⟨rfl, fun _ h => by simp at h⟩
        · have hvn : v.n < ar.length := hv.2.1 v (by simp)
          have hP := tagPlan_ok ⟨none, f.word, f.pc, ph.core stp⟩ hv.2.2.2.2.2.2.2 hvn
          obtain ⟨g1, e1, l1, d1⟩ := build_spec _ ar hv.1 hP
          have hY1 : Valid ⟨r, aux, f :: rest, mem, msize, hp, tabs, inp, ipos, out, pc, stp + 1, ar⟩ :=
            ⟨hv.1, fun d hd => hv.2.1 d (by simp [hd]), hv.2.2⟩
          have hα1 := α_ext hY1 e1
          have hY := hY1.ext g1 e1
          have hY' : Valid ⟨r, ⟨v.w, (build (tagPlan pc ⟨none, f.word, f.pc, ph.core stp⟩ v) ar).1⟩ :: aux,
              f :: rest, mem, msize, hp, tabs, inp, ipos, out, pc, stp + 1,
              (build (tagPlan pc ⟨none, f.word, f.pc, ph.core stp⟩ v) ar).2⟩ := by
            obtain ⟨a1, a2, a3, a4⟩ := hY
            refine ⟨a1, a2, ?_, a4⟩
            intro d hd; simp only [List.mem_cons] at hd
            rcases hd with rfl | hd
            · exact l1
            · exact a3 d hd
          obtain ⟨h1, h2⟩ := adv_next hY' { f with pc := f.pc + 1 } (fun d hd => Nat.lt_of_lt_of_le (hf d hd) e1.len)
          refine ⟨?_, h2⟩
          rw [h1]
          have : α ⟨r, ⟨v.w, (build (tagPlan pc ⟨none, f.word, f.pc, ph.core stp⟩ v) ar).1⟩ :: aux,
              f :: rest, mem, msize, hp, tabs, inp, ipos, out, pc, stp + 1,
              (build (tagPlan pc ⟨none, f.word, f.pc, ph.core stp⟩ v) ar).2⟩ =
              { α ⟨r, aux, f :: rest, mem, msize, hp, tabs, inp, ipos, out, pc, stp + 1, (build (tagPlan pc ⟨none, f.word, f.pc, ph.core stp⟩ v) ar).2⟩ with
                aux := ⟨v.w, leaves (build (tagPlan pc ⟨none, f.word, f.pc, ph.core stp⟩ v) ar).2
                  (build (tagPlan pc ⟨none, f.word, f.pc, ph.core stp⟩ v) ar).1⟩ ::
                  aux.map (αv (build (tagPlan pc ⟨none, f.word, f.pc, ph.core stp⟩ v) ar).2) } := rfl
          rw [this, d1]
          have hax := map_αv_ext hv.1 e1 hv.2.2.1
          simp only at hα1
          rw [hα1, hax, αf_ext hv.1 e1 (g := { f with pc := f.pc + 1 }) hf]
          rfl
      case locset i =>
        rcases ds with _ | ⟨v, r⟩
        · exact ⟨rfl, fun _ h => by simp at h⟩
        · simp only [List.length_map]
          by_cases hi : i < f.locs.length
          · simp only [hi, if_true]
            have hvn : v.n < ar.length := hv.2.1 v (by simp)
            have hP := tagPlan_ok ⟨none, f.word, f.pc, ph.core stp⟩ hv.2.2.2.2.2.2.2 hvn
            obtain ⟨g1, e1, l1, d1⟩ := build_spec _ ar hv.1 hP
            have hY1 : Valid ⟨r, aux, f :: rest, mem, msize, hp, tabs, inp, ipos, out, pc, stp + 1, ar⟩ :=
              ⟨hv.1, fun d hd => hv.2.1 d (by simp [hd]), hv.2.2⟩
            have hα1 := α_ext hY1 e1
            have hY := hY1.ext g1 e1
            have hloc : ∀ d ∈ (f.locs.set i ⟨v.w, (build (tagPlan pc ⟨none, f.word, f.pc, ph.core stp⟩ v) ar).1⟩), d.n < (build (tagPlan pc ⟨none, f.word, f.pc, ph.core stp⟩ v) ar).2.length := by
              intro d hd
              rcases List.mem_or_eq_of_mem_set hd with hd | rfl
              · exact Nat.lt_of_lt_of_le (hf d hd) e1.len
              · exact l1
            obtain ⟨h1, h2⟩ := adv_next hY { f with pc := f.pc + 1, locs := f.locs.set i ⟨v.w, (build (tagPlan pc ⟨none, f.word, f.pc, ph.core stp⟩ v) ar).1⟩ } hloc
            refine ⟨?_, h2⟩
            rw [h1]
            simp only at hα1
            rw [hα1]
            simp only [αf, List.map_set, αv, d1, map_αv_ext hv.1 e1 hf]
            rfl
          · simp only [hi, if_false]; exact ⟨rfl, fun _ h => by simp at h⟩
      case jz tg =>
        rcases ds with _ | ⟨c, r⟩
        · exact ⟨rfl, fun _ h => by simp at h⟩
        · have hcn : c.n < ar.length := hv.2.1 c (by simp)
          have hP : ((PE.node pc).app (PE.node c.n)).app (PE.leaf ⟨none, f.word, f.pc, ph.core stp⟩) |>.ok ar.length := ⟨⟨hv.2.2.2.2.2.2.2, hcn⟩, trivial⟩
          obtain ⟨g1, e1, l1, d1⟩ := build_spec _ ar hv.1 hP
          have hY1 : Valid ⟨r, aux, f :: rest, mem, msize, hp, tabs, inp, ipos, out, pc, stp + 1, ar⟩ :=
            ⟨hv.1, fun d hd => hv.2.1 d (by simp [hd]), hv.2.2⟩
          have hα1 := α_ext hY1 e1
          have hY := hY1.ext g1 e1
          have hY' : Valid ⟨r, aux, f :: rest, mem, msize, hp, tabs, inp, ipos, out,
              (build (((PE.node pc).app (PE.node c.n)).app (PE.leaf ⟨none, f.word, f.pc, ph.core stp⟩)) ar).1, stp + 1,
              (build (((PE.node pc).app (PE.node c.n)).app (PE.leaf ⟨none, f.word, f.pc, ph.core stp⟩)) ar).2⟩ :=
            ⟨hY.1, hY.2.1, hY.2.2.1, hY.2.2.2.1, hY.2.2.2.2.1, hY.2.2.2.2.2.1, hY.2.2.2.2.2.2.1, l1⟩
          obtain ⟨h1, h2⟩ := adv_next hY' { f with pc := if c.w % M = 0 then tg else f.pc + 1 }
            (fun d hd => Nat.lt_of_lt_of_le (hf d hd) e1.len)
          refine ⟨?_, h2⟩
          rw [h1]
          simp only at hα1
          have : ∀ X : DSt, α X = { α { X with pc := pc } with pc := leaves X.ar X.pc } := fun _ => rfl
          rw [this, hα1, d1]
          simp only [αf, αv, PE.den, map_αv_ext hv.1 e1 hf]
          rfl
      case jnz tg =>
        rcases ds with _ | ⟨c, r⟩
        · exact ⟨rfl, fun _ h => by simp at h⟩
        · have hcn : c.n < ar.length := hv.2.1 c (by simp)
          have hP : ((PE.node pc).app (PE.node c.n)).app (PE.leaf ⟨none, f.word, f.pc, ph.core stp⟩) |>.ok ar.length := ⟨⟨hv.2.2.2.2.2.2.2, hcn⟩, trivial⟩
          obtain ⟨g1, e1, l1, d1⟩ := build_spec _ ar hv.1 hP
          have hY1 : Valid ⟨r, aux, f :: rest, mem, msize, hp, tabs, inp, ipos, out, pc, stp + 1, ar⟩ :=
            ⟨hv.1, fun d hd => hv.2.1 d (by simp [hd]), hv.2.2⟩
          have hα1 := α_ext hY1 e1
          have hY := hY1.ext g1 e1
          have hY' : Valid ⟨r, aux, f :: rest, mem, msize, hp, tabs, inp, ipos, out,
              (build (((PE.node pc).app (PE.node c.n)).app (PE.leaf ⟨none, f.word, f.pc, ph.core stp⟩)) ar).1, stp + 1,
              (build (((PE.node pc).app (PE.node c.n)).app (PE.leaf ⟨none, f.word, f.pc, ph.core stp⟩)) ar).2⟩ :=
            ⟨hY.1, hY.2.1, hY.2.2.1, hY.2.2.2.1, hY.2.2.2.2.1, hY.2.2.2.2.2.1, hY.2.2.2.2.2.2.1, l1⟩
          obtain ⟨h1, h2⟩ := adv_next hY' { f with pc := if c.w % M = 0 then f.pc + 1 else tg }
            (fun d hd => Nat.lt_of_lt_of_le (hf d hd) e1.len)
          refine ⟨?_, h2⟩
          rw [h1]
          simp only at hα1
          have : ∀ X : DSt, α X = { α { X with pc := pc } with pc := leaves X.ar X.pc } := fun _ => rfl
          rw [this, hα1, d1]
          simp only [αf, αv, PE.den, map_αv_ext hv.1 e1 hf]
          rfl
      case ret =>
        rcases rest with _ | ⟨g, rest'⟩
        · exact ⟨rfl, fun _ h => by simp at h⟩
        · refine ⟨rfl, fun t' h => ?_⟩
          simp only [DRes.next.injEq] at h; subst h
          obtain ⟨a1, a2, a3, a4, a5⟩ := hv
          exact ⟨a1, a2, a3, fun g' hg' d hd => a4 g' (List.mem_cons_of_mem _ hg') d hd, a5⟩
      case call w =>
        rcases hw : pr.words[w]? with _ | wd
        · exact ⟨rfl, fun _ h => by simp at h⟩
        · simp only [List.length_map]
          by_cases hl : ds.length < wd.ins
          · simp only [hl, if_true]; exact ⟨rfl, fun _ h => by simp at h⟩
          · simp only [hl, if_false]
            have hP : ((PE.node pc).app (PE.leaf ⟨none, f.word, f.pc, ph.core stp⟩)).ok ar.length := ⟨hv.2.2.2.2.2.2.2, trivial⟩
            obtain ⟨g1, e1, l1, d1⟩ := build_spec _ ar hv.1 hP
            have hY1 : Valid ⟨ds, aux, f :: rest, mem, msize, hp, tabs, inp, ipos, out, pc, stp + 1, ar⟩ := hv
            have hα1 := α_ext hY1 e1
            have hY := hY1.ext g1 e1
            have hrs : ∀ g ∈ rest, αf (build ((PE.node pc).app (PE.leaf ⟨none, f.word, f.pc, ph.core stp⟩)) ar).2 g = αf ar g :=
              fun g hg => αf_ext hv.1 e1 (hv.2.2.2.1 g (by simp [hg]))
            refine ⟨?_, fun t' h => ?_⟩
            · simp only [DRes.abs]
              simp only at hα1
              have : ∀ X : DSt, ∀ rs', α { X with rs := rs' } = { α X with rs := rs'.map (αf X.ar) } :=
                fun _ _ => rfl
              show Res.next (α { (⟨ds, aux, f :: rest, mem, msize, hp, tabs, inp, ipos, out, pc, stp + 1,
                (build ((PE.node pc).app (PE.leaf ⟨none, f.word, f.pc, ph.core stp⟩)) ar).2⟩ : DSt) with rs := _ }) = _
              rw [this, hα1]
              simp only [List.map, αf, αv, List.map_replicate, d1, PE.den, map_αv_ext hv.1 e1 hf,
                List.map_congr_left hrs]
              rfl
            · simp only [DRes.next.injEq] at h; subst h
              obtain ⟨a1, a2, a3, a4, a5⟩ := hY
              refine ⟨a1, a2, a3, ?_, a5⟩
              intro g hg d hd
              simp only [List.mem_cons] at hg
              rcases hg with rfl | rfl | hg
              · simp only [List.mem_replicate] at hd; rw [hd.2]; exact l1
              · exact Nat.lt_of_lt_of_le (hf d hd) e1.len
              · exact a4 g (by simp [hg]) d hd
      case prim p =>
        simp only [List.length_map]
        by_cases hk : ds.length < p.eff.1
        · simp only [hk, if_true]; exact ⟨rfl, fun _ h => by simp at h⟩
        · simp only [hk, if_false]
          have hv1 : Valid ⟨ds, aux, f :: rest, mem, msize, hp, tabs, inp, ipos, out, pc, stp + 1, ar⟩ := hv
          have ha : ∀ d ∈ ds.take p.eff.1, d.n < ar.length := fun d hd => hv.2.1 d (List.mem_of_mem_take hd)
          have hP := qPlan_ok ha hv.2.2.2.2.2.2.2 ⟨none, f.word, f.pc, ph.core stp⟩
          obtain ⟨g1, e1, l1, d1⟩ := build_spec _ ar hv.1 hP
          have hT1 : Valid (⟨ds, aux, f :: rest, mem, msize, hp, tabs, inp, ipos, out, pc, stp + 1, (build (qPlan (ds.take p.eff.1) pc ⟨none, f.word, f.pc, ph.core stp⟩) ar).2⟩ : DSt) := hv1.ext g1 e1
          have hαT : α (⟨ds, aux, f :: rest, mem, msize, hp, tabs, inp, ipos, out, pc, stp + 1, (build (qPlan (ds.take p.eff.1) pc ⟨none, f.word, f.pc, ph.core stp⟩) ar).2⟩ : DSt) = α ⟨ds, aux, f :: rest, mem, msize, hp, tabs, inp, ipos, out, pc, stp + 1, ar⟩ :=
            α_ext hv1 e1
          have hargs : (ds.map (αv ar)).take p.eff.1 = (ds.take p.eff.1).map (αv (build (qPlan (ds.take p.eff.1) pc ⟨none, f.word, f.pc, ph.core stp⟩) ar).2) := by
            rw [← List.map_take, map_αv_ext hv.1 e1 ha]
          have hl : (ds.take p.eff.1).length = p.eff.1 := List.length_take_of_le (by omega)
          have hqd : leaves (build (qPlan (ds.take p.eff.1) pc ⟨none, f.word, f.pc, ph.core stp⟩) ar).2 (build (qPlan (ds.take p.eff.1) pc ⟨none, f.word, f.pc, ph.core stp⟩) ar).1 = (qPlan (ds.take p.eff.1) pc ⟨none, f.word, f.pc, ph.core stp⟩).den (leaves (build (qPlan (ds.take p.eff.1) pc ⟨none, f.word, f.pc, ph.core stp⟩) ar).2) := by
            rw [d1, PE.den_ext hv.1 e1 hP]
          have hsim := sim_prim_all ph p (ds.take p.eff.1) (⟨ds, aux, f :: rest, mem, msize, hp, tabs, inp, ipos, out, pc, stp + 1, (build (qPlan (ds.take p.eff.1) pc ⟨none, f.word, f.pc, ph.core stp⟩) ar).2⟩ : DSt) ⟨none, f.word, f.pc, ph.core stp⟩ (build (qPlan (ds.take p.eff.1) pc ⟨none, f.word, f.pc, ph.core stp⟩) ar).1 hT1 l1
            (fun d hd => Nat.lt_of_lt_of_le (ha d hd) e1.len) hl hqd
          have hlit : α ⟨ds, aux, f :: rest, mem, msize, hp, tabs, inp, ipos, out, pc, stp + 1, ar⟩ =
            ⟨ds.map (αv ar), aux.map (αv ar), ⟨f.word, f.pc, f.locs.map (αv ar), f.base, f.abase⟩ :: rest.map (αf ar),
              fun x => αv ar (mem x), msize, hp,
              tabs.map (List.map fun kv => (kv.1, αv ar kv.2)), inp, ipos, out.map (fun bn => (bn.1, leaves ar bn.2)),
              leaves ar pc, stp + 1⟩ := rfl
          rw [hargs, ← hlit, ← hαT, hsim]
          rcases hr : execPrimD ph p (ds.take p.eff.1) (⟨ds, aux, f :: rest, mem, msize, hp, tabs, inp, ipos, out, pc, stp + 1, (build (qPlan (ds.take p.eff.1) pc ⟨none, f.word, f.pc, ph.core stp⟩) ar).2⟩ : DSt) ⟨none, f.word, f.pc, ph.core stp⟩ (build (qPlan (ds.take p.eff.1) pc ⟨none, f.word, f.pc, ph.core stp⟩) ar).1 with ⟨res, s'⟩ | c | e
          · obtain ⟨hs'v, hext', hres, hds', haux', hrs', hpc', hstp'⟩ :=
              prim_valid ph p _ _ _ _ hT1 l1 (fun d hd => Nat.lt_of_lt_of_le (ha d hd) e1.len) hr
            obtain ⟨g2, e2, l2, d2⟩ := buildAll_spec res s'.ar hs'v.1 hres
            have eA : Ext ar (buildAll res s'.ar).2 := Ext.trans (Ext.trans e1 hext') e2
            have hD : ∀ d ∈ ds.drop p.eff.1, d.n < ar.length := fun d hd => hv.2.1 d (List.mem_of_mem_drop hd)
            have hZ : Valid { s' with ar := (buildAll res s'.ar).2, ds := (buildAll res s'.ar).1 ++ ds.drop p.eff.1 } := by
              obtain ⟨a1, a2, a3⟩ := hs'v.ext g2 e2
              refine ⟨a1, ?_, a3⟩
              intro d hd
              rcases List.mem_append.1 hd with hd | hd
              · exact l2 d hd
              · exact Nat.lt_of_lt_of_le (hD d hd) eA.len
            have hfr : ∀ d ∈ ({ f with pc := f.pc + 1 } : DFrame).locs, d.n < (buildAll res s'.ar).2.length :=
              fun d hd => Nat.lt_of_lt_of_le (hf d hd) eA.len
            obtain ⟨h1, h2⟩ := adv_next hZ { f with pc := f.pc + 1 } hfr
            refine ⟨?_, h2⟩
            rw [h1]
            simp only [DPR.abs]
            have hαZ : α { s' with ar := (buildAll res s'.ar).2, ds := (buildAll res s'.ar).1 ++ ds.drop p.eff.1 } =
                { α { s' with ar := (buildAll res s'.ar).2 } with
                  ds := ((buildAll res s'.ar).1 ++ ds.drop p.eff.1).map (αv (buildAll res s'.ar).2) } := rfl
            rw [hαZ, α_ext hs'v e2, List.map_append, d2, map_αv_ext hv.1 eA hD, αf_ext hv.1 eA (g := { f with pc := f.pc + 1 }) hf, ← List.map_drop]
            rfl
          · exact ⟨rfl, fun _ h => by simp at h⟩
          · exact ⟨rfl, fun _ h => by simp at h⟩

/-! ### Whole runs -/

/-- `n` DAG steps, or the end of the run. -/
def drun (pr : Prog) (ph : Phys) : Nat → DSt → DRes
  | 0, s => .next s
  | n + 1, s =>
    match dstep pr ph s with
    | .next t => drun pr ph n t
    | r => r

theorem drun_sim (pr : Prog) (ph : Phys) :
    ∀ (n : Nat) (t : DSt), Valid t → (drun pr ph n t).abs = run pr ph n (α t) := by
  intro n
  induction n with
  | zero => intro t _; rfl
  | succ n ih =>
    intro t hv
    obtain ⟨hs, hn⟩ := dstep_sim pr ph t hv
    simp only [drun, run]
    rw [← hs]
    rcases h : dstep pr ph t with t' | ⟨c, o, a⟩ | e
    · simp only [DRes.abs]; exact ih t' (hn t' h)
    · rfl
    · rfl

/-- The DAG machine's initial state: one `empty` node, every provenance empty. -/
def dinit (pr : Prog) (msize : Nat) (inp : List Nat) : DSt :=
  let locals := match pr.words[pr.main]? with | some m => m.locals | none => 0
  { ds := [], aux := [], rs := [⟨pr.main, 0, List.replicate locals ⟨0, 0⟩, 0, 0⟩],
    mem := fun a => ⟨(loadData pr.data a).w, 0⟩, msize := msize, hp := pr.data.flatten.length, tabs := [],
    inp := inp, ipos := 0, out := [], pc := 0, step := 0, ar := [.empty] }

theorem dinit_valid (pr : Prog) (msize : Nat) (inp : List Nat) : Valid (dinit pr msize inp) := by
  refine ⟨⟨?_, rfl⟩, by simp [dinit], by simp [dinit], ?_, fun _ => by simp [dinit], by simp [dinit],
    by simp [dinit], by simp [dinit]⟩
  · intro i a b h
    rcases i with _ | i <;> simp [dinit] at h
  · intro g hg d hd
    simp only [dinit, List.mem_cons, List.not_mem_nil, or_false] at hg
    subst hg
    simp only [List.mem_replicate] at hd
    rw [hd.2]; simp [dinit]

theorem α_dinit (pr : Prog) (msize : Nat) (inp : List Nat) : α (dinit pr msize inp) = init pr msize inp := by
  have h0 : leaves [DNode.empty] 0 = [] := leaves_empty rfl
  simp only [α, dinit, init, αf, αv, List.map, List.map_replicate, h0]
  congr 1
  funext a
  simp only [loadData]
  split <;> rfl

/-- The DAG machine computes exactly what the list machine computes, on every
program, input and number of steps: same words, same halting code, and each
output byte's provenance, read back from the arena, is the list one. -/
theorem drun_init (pr : Prog) (ph : Phys) (msize : Nat) (inp : List Nat) (n : Nat) :
    (drun pr ph n (dinit pr msize inp)).abs = run pr ph n (init pr msize inp) := by
  rw [drun_sim pr ph n _ (dinit_valid pr msize inp), α_dinit]

end Sporos
