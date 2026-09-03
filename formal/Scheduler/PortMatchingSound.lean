/-!
# Sounio.Scheduler.PortMatchingSound

Pure Lean 4 stdlib.  No `Mathlib` imports.  No `sorry`.  No `Classical`
(this file's theorems sit on the same `[propext, Quot.sound]` axiom
footprint as `CriticalPathLowerBound`).

Companion proof obligation for the per-cycle greedy port-matching loop
in `cgplan_phase_d_list_schedule` (`self-hosted/native/codegen_plan.sio`).

Within one cycle, the inner `while` loop of Phase D repeatedly invokes
`cgplan_global_pick_best_pp_n` (ready instruction whose port mask has a
free bit) followed by `cgplan_pick_port_idx` (greedy port chooser).  The
chosen port is OR-ed into `busy_mask` before the next iteration:

      let port = cgplan_pick_port_idx(b_mask, busy_mask)
      busy_mask = busy_mask | (1 << port)

`cgplan_pick_port_idx` returns `best_port` only from indices `0..7` and
only when `(1 << best_port) & busy_mask == 0` (it scans `free = mask &
~busy_mask`).  Those two operational guarantees are the **abstract
spec** of one greedy pick.

This file proves, on the abstract set-of-issued-ports model:

  `port_matching_sound : LegalCycle xs → xs.Nodup ∧ xs.length ≤ K`

i.e. **(a)** the ports assigned within a single cycle are pairwise
distinct (no two issued instructions share a port), and **(b)** at most
`K = 8` instructions are issued per cycle (cardinality bound).

The work here is the pigeonhole: a duplicate-free list of `Nat`s whose
every element is `< K` cannot exceed length `K`.  Proved by induction on
`K`, case-splitting on whether the top bound appears in the list and
stripping it via `List.erase`.

The bit-mask ↔ set-of-ports bridge — proving that the Sounio code's
`busy_mask : i64` faithfully tracks the set of ports issued so far — is
a separate operational claim (paper-side, alongside the Hoare-classical
split for the rest of the scheduler).
-/

namespace Sounio.Scheduler.PortMatching

/-- Number of execution ports in the Skylake-class model.  Matches the
    fixed 0..7 range over which `cgplan_pick_port_idx` enumerates and
    the size of the `CGPLAN_PICK_PORT_PRESS` array. -/
def K : Nat := 8

/-- One cycle's issue, as the ordered list of port indices assigned by
    successive calls to `cgplan_pick_port_idx`.  The Sounio code's
    `busy_mask` is reflected here by the set of ports already in the
    prefix. -/
abbrev IssueSeq := List Nat

/-- Legality of one greedy pick at a busy-state represented by the
    already-issued ports `issued`:
      * `p < K` — only indices `0..7` are ever returned,
      * `p ∉ issued` — the chosen port was free in `busy_mask`. -/
def LegalPick (issued : IssueSeq) (p : Nat) : Prop :=
  p < K ∧ p ∉ issued

/-- A whole-cycle issue: every prefix's next pick is legal w.r.t. the
    set of ports already chosen.  Mirrors the inner `while` loop of
    `cgplan_phase_d_list_schedule`. -/
inductive LegalCycle : IssueSeq → Prop where
  | nil  : LegalCycle []
  | cons (p : Nat) (rest : IssueSeq) :
      LegalPick rest p → LegalCycle rest → LegalCycle (p :: rest)

/-- Every port in a legal cycle is in `[0, K)`. -/
theorem LegalCycle.mem_lt_K {xs : IssueSeq} (h : LegalCycle xs) :
    ∀ p ∈ xs, p < K := by
  induction h with
  | nil => intro p hp; cases hp
  | cons q rest hpick _ ih =>
    intro p hp
    cases List.mem_cons.mp hp with
    | inl heq => exact heq ▸ hpick.1
    | inr hin => exact ih p hin

/-- A legal cycle assigns no port twice.  Direct from the `LegalPick`
    `p ∉ issued` clause: that is exactly the head case of `List.Nodup`'s
    `Pairwise (· ≠ ·)` definition. -/
theorem LegalCycle.nodup {xs : IssueSeq} (h : LegalCycle xs) : xs.Nodup := by
  induction h with
  | nil => exact List.Pairwise.nil
  | cons q rest hpick _ ih =>
    refine List.Pairwise.cons ?_ ih
    intro a ha heq
    exact hpick.2 (heq ▸ ha)

/-- Auxiliary: `Nodup` of a cons unpacks to head-fresh + tail-Nodup.
    Avoids `simp`/`rcases` so the axiom footprint stays minimal. -/
private theorem nodup_cons_iff {a : Nat} {t : List Nat} :
    (a :: t).Nodup ↔ a ∉ t ∧ t.Nodup := by
  constructor
  · intro h
    cases h with
    | cons hhead htail =>
      refine ⟨?_, htail⟩
      intro hmem
      exact (hhead a hmem) rfl
  · intro ⟨hfresh, htnod⟩
    refine List.Pairwise.cons ?_ htnod
    intro b hb heq
    exact hfresh (heq ▸ hb)

/-- If `l` has no duplicates, then `List.erase l x` removes the sole
    copy of `x` and `x` no longer appears.  Pure structural induction on
    `l`; no `Classical`. -/
private theorem not_mem_erase_of_nodup :
    ∀ {l : List Nat}, l.Nodup → ∀ x, x ∉ l.erase x := by
  intro l hl x
  induction l with
  | nil =>
    intro h
    -- `([] : List Nat).erase x = []`, so membership is impossible.
    cases h
  | cons a t ih =>
    have ⟨ha_not_in_t, htnod⟩ := nodup_cons_iff.mp hl
    if hax : x = a then
      subst hax
      rw [List.erase_cons_head]
      exact ha_not_in_t
    else
      have hba : ¬(a == x) = true := by
        intro hab
        exact hax (LawfulBEq.eq_of_beq hab).symm
      rw [List.erase_cons_tail hba]
      intro hmem
      cases List.mem_cons.mp hmem with
      | inl heq => exact hax heq
      | inr hin => exact ih htnod hin

/-- Local replacement for `List.length_erase_of_mem` that doesn't drag
    in `Classical.choice` from `mathlib`-style developments — pure
    structural induction on the list. -/
private theorem length_eq_erase_succ :
    ∀ {l : List Nat} {x : Nat}, x ∈ l → l.length = (l.erase x).length + 1 := by
  intro l x hx
  induction l with
  | nil => cases hx
  | cons a t ih =>
    if hax : x = a then
      subst hax
      rw [List.erase_cons_head]
      rfl
    else
      have hba : ¬(a == x) = true := by
        intro hab
        exact hax (LawfulBEq.eq_of_beq hab).symm
      rw [List.erase_cons_tail hba]
      have hx' : x ∈ t := by
        cases List.mem_cons.mp hx with
        | inl heq => exact absurd heq hax
        | inr hin => exact hin
      have ih' : t.length = (t.erase x).length + 1 := ih hx'
      -- `(a :: t).length` and `(a :: t.erase x).length` are both
      -- `_.length + 1` by computation; rewrite once and we're done.
      show t.length + 1 = (t.erase x).length + 1 + 1
      rw [ih']

/-- **Pigeonhole.**  A duplicate-free list of `Nat`s whose every element
    is strictly below `K` has length at most `K`.  Pure stdlib induction
    on `K`; the inductive step strips a single occurrence of the top
    bound `n` (when present) using `List.erase` and applies IH. -/
theorem nodup_lt_length_le :
    ∀ (K : Nat) (xs : List Nat),
      xs.Nodup → (∀ x ∈ xs, x < K) → xs.length ≤ K := by
  intro K
  induction K with
  | zero =>
    intro xs _ hlt
    cases xs with
    | nil => exact Nat.le_refl 0
    | cons a _ =>
      exact absurd (hlt a List.mem_cons_self) (Nat.not_lt_zero _)
  | succ n ih =>
    intro xs hnod hlt
    if hin : n ∈ xs then
      -- strip the unique occurrence of `n`; rest has all entries < n
      have hnod' : (xs.erase n).Nodup := hnod.erase n
      have hlt' : ∀ x ∈ xs.erase n, x < n := by
        intro x hx
        have hx_in : x ∈ xs := List.mem_of_mem_erase hx
        have hx_ne : x ≠ n := by
          intro heq
          subst heq
          exact (not_mem_erase_of_nodup hnod x) hx
        have hx_lt : x < n + 1 := hlt x hx_in
        omega
      have hlen' : (xs.erase n).length ≤ n := ih (xs.erase n) hnod' hlt'
      have hlen_eq : xs.length = (xs.erase n).length + 1 :=
        length_eq_erase_succ hin
      omega
    else
      -- n ∉ xs, so every element is ≤ n and ≠ n, hence < n
      have hlt' : ∀ x ∈ xs, x < n := by
        intro x hx
        have hx_lt : x < n + 1 := hlt x hx
        have hx_ne : x ≠ n := fun heq => hin (heq ▸ hx)
        omega
      exact Nat.le_succ_of_le (ih xs hnod hlt')

/-- **Theorem (PortMatchingSound).**  In any single cycle, the inner
    greedy port-matching loop of `cgplan_phase_d_list_schedule` assigns
    pairwise distinct ports (`xs.Nodup`) and issues at most `K = 8`
    instructions (`xs.length ≤ K`). -/
theorem port_matching_sound {xs : IssueSeq} (h : LegalCycle xs) :
    xs.Nodup ∧ xs.length ≤ K := by
  refine ⟨h.nodup, ?_⟩
  exact nodup_lt_length_le K xs h.nodup h.mem_lt_K

/-- **Corollary.**  The per-cycle issue width never exceeds `K`. -/
theorem cycle_width_le_K {xs : IssueSeq} (h : LegalCycle xs) :
    xs.length ≤ K :=
  (port_matching_sound h).2

/-- **Corollary.**  No two instructions issued in the same cycle land on
    the same execution port. -/
theorem ports_pairwise_distinct {xs : IssueSeq} (h : LegalCycle xs) :
    xs.Nodup :=
  (port_matching_sound h).1

end Sounio.Scheduler.PortMatching
