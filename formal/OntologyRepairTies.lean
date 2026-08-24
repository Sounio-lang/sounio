/-
# Sounio.OntologyRepairTies — greedy ≡ fold repair under ARBITRARY confidences

Round-5 extension of `OntologyRepairEquivalence.lean` (round 2). That file's
main equivalence `repair_iff_greedy` assumed **distinct confidences** (`hinj`,
`hdist`), excluding ties — the documented residual gap of frontier 1
(`artifacts/ontology-frontiers/epistemic-alignment-repair/FRONTIER.md`). This
file closes the gap: confidences may be arbitrary, and ties are broken
deterministically by the mapping **id**.

## The tie-break (and a note on the briefing formula)

The priority is the lexicographic order on `(conf, id)` with ties resolved by
id order. The round-5 briefing phrase "ties broken by `id m > id m'`" is a
slip: both binding behavioural specifications — the `.sio` greedy rule
`if conf[i] >= conf[j] { drop j }` on id-ordered pairs `(i, j)` with `i < j`,
and the required Fin 6 instance ("m0 vs m1 tie-break keeps the lower id") —
say the **lower** id wins a tie. We therefore define:

  `outranks conf id m m'  :=  conf m' < conf m ∨ (conf m' = conf m ∧ id m < id m')`

i.e. lexicographic on (confidence descending, id ascending). For an id-ordered
pair `(a, b)` with `id a < id b`, dropping the lexicographically weaker
endpoint is *exactly* the `.sio` rule (`greedyStep_prio_eq_sio`).

## What is proved here

1. `prio` — an injective Nat encoding of the lexicographic priority:
   `prio m = conf m * (I + 1) + (I - id m)` for `id m ≤ I`. Lower id ⇒ higher
   `prio` on a confidence tie (`prio_lt_of_tie_lower_id`); higher confidence ⇒
   higher `prio` regardless of ids (`prio_lt_of_conf_lt`).
2. `outranks_total` / `outranks_antisymm` — the lexicographic priority is
   **total and antisymmetric on mappings with distinct ids**. Hence the
   "distinct confidences" hypothesis of `repair_iff_greedy` can be replaced by
   "distinct priorities", which always holds for distinct ids
   (`prio_injective_on`, via `prio_inj`).
3. `repair_iff_greedy_ties` — **main equivalence for arbitrary conf**:
   candidates sorted by decreasing lexicographic priority (`outranks`
   pairwise), nodup, distinct bounded ids, symmetric cluster-graph conflicts,
   covering pair list ⇒ `m ∈ repair C [] ms ↔ m ∈ ms ∧ m ∉ greedyDrop C (prio
   conf id I) [] ps`.
4. `greedyStep_prio_eq_sio` — the `prio`-driven greedy step on an id-ordered
   pair is definitionally the `.sio` `conf[i] >= conf[j]` drop-`j` rule.
5. `greedyDrop_deterministic` — **greedy determinism**: equal inputs give
   equal dropped sets (induction on the pair list).
6. §6: concrete `Fin 6` instance WITH A TIE — m0 (0.50) and m1 (0.50) tie,
   both conflict with m2 (0.30), and the tie pair (m0, m1) itself conflicts;
   m3 (0.95) vs m4 (0.80); m5 isolated. Both algorithms are evaluated by
   `native_decide` and keep exactly `{m0, m3, m5}`.

## Which parent lemmas needed priority-generalized copies

**None.** All of `repair_iff_greedy`'s distinctness hypotheses are *derived*
for `prio`:

- `hinj` (injectivity of the priority on candidates) follows from
  `prio_injective_on` — distinct ids always give distinct priorities;
- `hdist` (conflicting pairs have distinct priorities) follows from `hinj`
  plus two new side hypotheses that are automatic in every realistic
  instance: conflicts only relate candidates (`hCms`) and nothing conflicts
  with itself (`hirr`).

The parent theorem `repair_iff_greedy` is therefore applied verbatim with
`conf := prio conf id I`; only the genuinely new arithmetic of the encoding
(`mul_add_lt_mul_add`, `prio_inj`) is proved here.

Self-contained except for `OntologyRepairEquivalence` (transitively
`OntologyAlignmentRepair`). No Mathlib. Zero sorry. No new axioms.
-/

import OntologyRepairEquivalence

namespace Sounio.OntologyAlignmentRepair

variable {α : Type}

-- ---------------------------------------------------------------------------
-- §1. The lexicographic priority and its Nat encoding
-- ---------------------------------------------------------------------------

/-- **Lexicographic priority**: `m` outranks `m'` iff it has strictly higher
    confidence, or equal confidence and strictly lower id (ties are resolved
    by list/id order, matching the `.sio` `conf[i] >= conf[j]` drop-`j` rule
    on id-ordered pairs). -/
def outranks (conf id : α → Nat) (m m' : α) : Prop :=
  conf m' < conf m ∨ (conf m' = conf m ∧ id m < id m')

/-- The lexicographic priority is decidable (needed by `decide` on the
    concrete instances). -/
instance outranksDecidable {conf id : α → Nat} {a b : α} :
    Decidable (outranks conf id a b) :=
  inferInstanceAs (Decidable (conf b < conf a ∨ (conf b = conf a ∧ id a < id b)))

/-- Nat encoding of the priority: for ids bounded by `I`, `prio` orders
    exactly as `outranks` (higher `prio` = higher priority). On a confidence
    tie the *lower* id gets the *higher* `prio`. -/
def prio (conf id : α → Nat) (I : Nat) (m : α) : Nat :=
  conf m * (I + 1) + (I - id m)

/-- The id-slack is always strictly below the multiplier (truncated
    subtraction needs no bound for this direction). -/
theorem sub_lt_succ (I n : Nat) : I - n < I + 1 :=
  Nat.lt_of_le_of_lt (Nat.sub_le I n) (Nat.lt_succ_self I)

/-- **Base-`k` order lemma**: if the low digit `r` is below the base `k`, a
    smaller high digit always means a smaller number, whatever the other low
    digit `s` is. (This is the only nonlinear-arithmetic fact the encoding
    needs; it is proved by `x*k + r < x*k + k = (x+1)*k ≤ y*k ≤ y*k + s`.) -/
theorem mul_add_lt_mul_add {x y r s k : Nat} (hr : r < k) (hxy : x < y) :
    x * k + r < y * k + s := by
  have h1 : x * k + r < x * k + k := Nat.add_lt_add_left hr _
  have h2 : x * k + k = (x + 1) * k := by rw [Nat.add_mul, Nat.one_mul]
  have h3 : (x + 1) * k ≤ y * k := Nat.mul_le_mul hxy (Nat.le_refl k)
  omega

/-- Higher confidence always gives higher priority, regardless of ids. -/
theorem prio_lt_of_conf_lt {conf id : α → Nat} {I : Nat} {a b : α}
    (h : conf a < conf b) : prio conf id I a < prio conf id I b :=
  mul_add_lt_mul_add (sub_lt_succ I (id a)) h

/-- On a confidence tie, the lower id gets the higher priority
    (needs the winning id to be within the bound, else the truncated
    subtraction collapses both slacks to 0). -/
theorem prio_lt_of_tie_lower_id {conf id : α → Nat} {I : Nat} {a b : α}
    (hc : conf a = conf b) (hid : id a < id b) (hb : id b ≤ I) :
    prio conf id I b < prio conf id I a := by
  unfold prio
  rw [hc]
  omega

/-- **Injectivity of the encoding on bounded ids**: equal priorities force
    equal confidences and equal ids. -/
theorem prio_inj {conf id : α → Nat} {I : Nat} {a b : α}
    (ha : id a ≤ I) (hb : id b ≤ I)
    (h : prio conf id I a = prio conf id I b) : conf a = conf b ∧ id a = id b := by
  obtain hlt | heq | hgt := Nat.lt_trichotomy (conf a) (conf b)
  · exact absurd h (Nat.ne_of_lt (prio_lt_of_conf_lt hlt))
  · refine ⟨heq, ?_⟩
    have h'' : conf b * (I + 1) + (I - id a) = conf b * (I + 1) + (I - id b) := by
      have h3 := h
      unfold prio at h3
      rw [heq] at h3
      exact h3
    have h2 : I - id a = I - id b := Nat.add_left_cancel h''
    omega
  · exact absurd h.symm (Nat.ne_of_lt (prio_lt_of_conf_lt hgt))

-- ---------------------------------------------------------------------------
-- §2. Totality/antisymmetry; distinct ids ⇒ distinct priorities
-- ---------------------------------------------------------------------------

/-- **Totality on distinct ids**: any two mappings with different ids are
    comparable — exactly one outranks the other. This is what makes the
    tie-break a *deterministic* total priority. -/
theorem outranks_total {conf id : α → Nat} {a b : α} (hne : id a ≠ id b) :
    outranks conf id a b ∨ outranks conf id b a := by
  unfold outranks
  omega

/-- **Antisymmetry**: `m` outranks `m'` and `m'` outranks `m` is impossible. -/
theorem outranks_antisymm {conf id : α → Nat} {a b : α}
    (h : outranks conf id a b) : ¬ outranks conf id b a := by
  unfold outranks at h ⊢
  intro h'
  obtain h1 | ⟨h1, h2⟩ := h <;> obtain h3 | ⟨h3, h4⟩ := h' <;> omega

/-- **Distinct priorities for free**: on a candidate list with distinct ids
    bounded by `I`, the encoded priorities are all distinct. The
    "distinct confidences" hypothesis of `repair_iff_greedy` is thus replaced
    by "distinct priorities", which needs no assumption on `conf` at all. -/
theorem prio_injective_on {conf id : α → Nat} {I : Nat} {ms : List α}
    (hid : ∀ a ∈ ms, ∀ b ∈ ms, id a = id b → a = b)
    (hidB : ∀ a ∈ ms, id a ≤ I) :
    ∀ a ∈ ms, ∀ b ∈ ms, prio conf id I a = prio conf id I b → a = b := by
  intro a ha b hb h
  obtain ⟨-, hid'⟩ := prio_inj (hidB a ha) (hidB b hb) h
  exact hid a ha b hb hid'

/-- `outranks` implies the `prio` order (decreasing priority processing). -/
theorem prio_le_of_outranks {conf id : α → Nat} {I : Nat} {a b : α}
    (_ha : id a ≤ I) (hb : id b ≤ I)
    (h : outranks conf id a b) : prio conf id I b ≤ prio conf id I a := by
  obtain hlt | ⟨hc, hid⟩ := h
  · exact Nat.le_of_lt (prio_lt_of_conf_lt hlt)
  · exact Nat.le_of_lt (prio_lt_of_tie_lower_id hc.symm hid hb)

/-- Sortedness by decreasing lexicographic priority implies sortedness by
    decreasing encoded priority (the form the parent theorems consume). -/
theorem pairwise_prio_of_outranks {conf id : α → Nat} {I : Nat} :
    ∀ {ms : List α}, (∀ a ∈ ms, id a ≤ I) →
      ms.Pairwise (outranks conf id) →
      ms.Pairwise fun a b => prio conf id I b ≤ prio conf id I a := by
  intro ms
  induction ms with
  | nil => intro _ _; exact List.Pairwise.nil
  | cons x xs ihx =>
      intro hidB h
      obtain ⟨hhead, htail⟩ := List.pairwise_cons.mp h
      apply List.pairwise_cons.mpr
      refine ⟨?_, ihx (fun a ha => hidB a (List.mem_cons.mpr (Or.inr ha))) htail⟩
      intro b hb
      exact prio_le_of_outranks (hidB x (List.mem_cons.mpr (Or.inl rfl)))
        (hidB b (List.mem_cons.mpr (Or.inr hb))) (hhead b hb)

-- ---------------------------------------------------------------------------
-- §3. The tie-broken greedy matches the .sio rule
-- ---------------------------------------------------------------------------

/-- On an id-ordered pair `(a, b)` (`id a < id b`), the `prio`-driven greedy
    step is **definitionally** the `.sio` rule
    `if conf[i] >= conf[j] { drop j } else { drop i }`: the second component
    is dropped exactly when its confidence is ≤ the first's (ties included). -/
theorem greedyStep_prio_eq_sio [DecidableEq α] {C : α → α → Bool} {conf id : α → Nat}
    {I : Nat} {d : List α} {a b : α}
    (hbI : id b ≤ I) (hab : id a < id b) :
    greedyStep C (prio conf id I) d (a, b) =
      if a ∈ d ∨ b ∈ d then d
      else if C a b then (if conf b ≤ conf a then b :: d else a :: d)
      else d := by
  unfold greedyStep
  by_cases hd : a ∈ d ∨ b ∈ d
  · simp [hd]
  · by_cases hC : C a b = true
    · by_cases hle : conf b ≤ conf a
      · have hnot : ¬ prio conf id I a < prio conf id I b := by
          obtain hc | hc := Nat.eq_or_lt_of_le hle
          · exact Nat.not_lt_of_ge (Nat.le_of_lt (prio_lt_of_tie_lower_id hc.symm hab hbI))
          · exact Nat.not_lt_of_ge (Nat.le_of_lt (prio_lt_of_conf_lt hc))
        simp [hd, hC, hnot, hle]
      · have hlt : conf a < conf b := Nat.lt_of_not_le hle
        have hprio : prio conf id I a < prio conf id I b := prio_lt_of_conf_lt hlt
        simp [hd, hC, hprio, hle]
    · simp [hd, hC]

-- ---------------------------------------------------------------------------
-- §4. Main equivalence for arbitrary confidences
-- ---------------------------------------------------------------------------

/-- **Main equivalence with ties**: for *arbitrary* confidences, a candidate
    list sorted by decreasing lexicographic priority (`outranks` pairwise)
    with distinct ids bounded by `I`, a symmetric cluster-graph conflict
    relation that only relates distinct candidates, and a pair list covering
    all conflicting candidate pairs, the pairwise drop-weaker greedy driven by
    `prio` (equivalently: the `.sio` tie-break on id-ordered pairs, cf.
    `greedyStep_prio_eq_sio`) keeps exactly the priority fold's kept set.

    This is `repair_iff_greedy` with `conf := prio conf id I`; the distinctness
    hypotheses `hinj`/`hdist` are derived, not assumed. -/
theorem repair_iff_greedy_ties [DecidableEq α] {C : α → α → Bool} {conf id : α → Nat}
    {I : Nat} {ms : List α} {ps : List (α × α)}
    (hsorted : ms.Pairwise (outranks conf id))
    (hnodup : ms.Nodup)
    (hid : ∀ a ∈ ms, ∀ b ∈ ms, id a = id b → a = b)
    (hidB : ∀ a ∈ ms, id a ≤ I)
    (hCms : ∀ a b, C a b = true → a ∈ ms ∧ b ∈ ms)
    (hirr : ∀ a, C a a = false)
    (hsym : ∀ a b, C a b = C b a)
    (hclus : ∀ a b c, C a b = true → C b c = true → a ≠ c → C a c = true)
    (hcov : ∀ a b, a ∈ ms → b ∈ ms → C a b = true →
      (a, b) ∈ ps ∨ (b, a) ∈ ps)
    (hps : ∀ p ∈ ps, p.1 ∈ ms ∧ p.2 ∈ ms) :
    ∀ m, m ∈ repair C [] ms ↔
      (m ∈ ms ∧ m ∉ greedyDrop C (prio conf id I) [] ps) := by
  have hsorted' : ms.Pairwise fun a b => prio conf id I b ≤ prio conf id I a :=
    pairwise_prio_of_outranks hidB hsorted
  have hinj' : ∀ a ∈ ms, ∀ b ∈ ms, prio conf id I a = prio conf id I b → a = b :=
    prio_injective_on hid hidB
  have hdist' : ∀ a b, C a b = true → prio conf id I a ≠ prio conf id I b := by
    intro a b hC heq
    obtain ⟨ha, hb⟩ := hCms a b hC
    obtain ⟨-, hid'⟩ := prio_inj (hidB a ha) (hidB b hb) heq
    have hab : a = b := hid a ha b hb hid'
    subst hab
    rw [hirr a] at hC
    exact Bool.noConfusion hC
  exact repair_iff_greedy hsorted' hnodup hinj' hsym hclus hdist' hcov hps

-- ---------------------------------------------------------------------------
-- §5. Greedy determinism
-- ---------------------------------------------------------------------------

/-- **Greedy determinism**: the tie-broken greedy is a function of its input —
    starting from equal dropped sets and scanning the same pair list, two runs
    end with equal dropped sets (hence equal kept sets). Induction on the pair
    list; each step is deterministic because `greedyStep` is a function and
    the tie-break is fixed by the id order (`outranks_total`). -/
theorem greedyDrop_deterministic [DecidableEq α] {C : α → α → Bool} {conf id : α → Nat}
    {I : Nat} :
    ∀ {ps : List (α × α)} {d₁ d₂ : List α}, d₁ = d₂ →
      greedyDrop C (prio conf id I) d₁ ps = greedyDrop C (prio conf id I) d₂ ps := by
  intro ps
  induction ps with
  | nil => intro d₁ d₂ h; exact h
  | cons p ps ih =>
      intro d₁ d₂ h
      show greedyDrop C (prio conf id I) (greedyStep C (prio conf id I) d₁ p) ps =
        greedyDrop C (prio conf id I) (greedyStep C (prio conf id I) d₂ p) ps
      exact ih (by rw [h])

/-- Two runs of the tie-broken greedy on the same input agree. -/
theorem greedyDrop_run_agrees [DecidableEq α] {C : α → α → Bool} {conf id : α → Nat}
    {I : Nat} {ps : List (α × α)} :
    greedyDrop C (prio conf id I) [] ps = greedyDrop C (prio conf id I) [] ps :=
  greedyDrop_deterministic rfl

-- ---------------------------------------------------------------------------
-- §6. Concrete instance WITH A TIE (Fin 6)
-- ---------------------------------------------------------------------------

/-- Confidences in hundredths: m0 = 0.50 and m1 = 0.50 **tie**; m2 = 0.30
    conflicts with both; m3 = 0.95 conflicts with m4 = 0.80; m5 = 0.10 is
    conflict-free. Deliberately NOT all distinct. -/
def ex6Conf : Fin 6 → Nat :=
  fun m => [50, 50, 30, 95, 80, 10].getD m.val 0

/-- Ids are the mapping indices themselves. -/
def ex6Id : Fin 6 → Nat := fun m => m.val

/-- Conflict relation: the tie clique {0,1,2} — m0 vs m1 (the tie pair
    itself conflicts), m0 vs m2, m1 vs m2 — plus the disjoint edge {3,4}.
    A disjoint union of cliques (cluster graph). -/
def ex6C : Fin 6 → Fin 6 → Bool := fun a b =>
  (a.val == 0 && b.val == 1) || (a.val == 1 && b.val == 0) ||
  (a.val == 0 && b.val == 2) || (a.val == 2 && b.val == 0) ||
  (a.val == 1 && b.val == 2) || (a.val == 2 && b.val == 1) ||
  (a.val == 3 && b.val == 4) || (a.val == 4 && b.val == 3)

/-- Candidates in decreasing lexicographic priority order:
    m3 (0.95), m4 (0.80), m0 (0.50, id 0 — wins the tie), m1 (0.50, id 1),
    m2 (0.30), m5 (0.10). -/
def ex6Ms : List (Fin 6) := [3, 4, 0, 1, 2, 5]

/-- The conflicting pairs scanned by the `.sio` greedy (id order). -/
def ex6Pairs : List (Fin 6 × Fin 6) := [(0, 1), (0, 2), (1, 2), (3, 4)]

/-- On the tie, m0 outranks m1 because of the lower id. -/
theorem ex6_outranks_tie : outranks ex6Conf ex6Id (0 : Fin 6) (1 : Fin 6) := by
  decide

/-- The encoding agrees: m1's priority is strictly below m0's. -/
theorem ex6_prio_tie : prio ex6Conf ex6Id 5 (1 : Fin 6) < prio ex6Conf ex6Id 5 (0 : Fin 6) := by
  decide

/-- The fold keeps m3 and m0 (the tie winner) and m5. -/
theorem ex6_repair : repair ex6C [] ex6Ms = [5, 0, 3] := by
  native_decide

/-- The tie-broken greedy drops m1 (loses the tie to m0 by higher id),
    m2 (0.30 < 0.50) and m4 (0.80 < 0.95). -/
theorem ex6_greedyDrop :
    greedyDrop ex6C (prio ex6Conf ex6Id 5) [] ex6Pairs = [4, 2, 1] := by
  native_decide

/-- **Instance of the tie-broken equivalence**, obtained by instantiating the
    general theorem; every hypothesis is discharged by `decide`. Note that
    `ex6Conf 0 = ex6Conf 1` — the round-2 theorem could not be instantiated
    here at all. -/
theorem ex6_repair_iff_greedy_ties :
    ∀ m : Fin 6, m ∈ repair ex6C [] ex6Ms ↔
      (m ∈ ex6Ms ∧ m ∉ greedyDrop ex6C (prio ex6Conf ex6Id 5) [] ex6Pairs) :=
  repair_iff_greedy_ties
    (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)
    (by decide) (by decide)
    (fun a b _ _ h => by
      have hd : ∀ a b : Fin 6, ex6C a b = true →
          (a, b) ∈ ex6Pairs ∨ (b, a) ∈ ex6Pairs := by decide
      exact hd a b h)
    (by decide)

/-- Sanity check of the same fact by exhaustive evaluation through BOTH
    algorithms: fold kept = greedy kept = {m0, m3, m5}. -/
theorem ex6_repair_iff_greedy_check :
    ∀ m : Fin 6, m ∈ repair ex6C [] ex6Ms ↔
      (m ∈ ex6Ms ∧ m ∉ greedyDrop ex6C (prio ex6Conf ex6Id 5) [] ex6Pairs) := by
  native_decide

/-- Determinism on the instance: two runs of the tie-broken greedy agree. -/
theorem ex6_two_runs_agree :
    greedyDrop ex6C (prio ex6Conf ex6Id 5) [] ex6Pairs =
      greedyDrop ex6C (prio ex6Conf ex6Id 5) [] ex6Pairs :=
  greedyDrop_run_agrees

end Sounio.OntologyAlignmentRepair
