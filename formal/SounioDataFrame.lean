/-!
# Sounio.DataFrame — Formal Structural Invariants of DataFrame Verbs

Formal verification of the row-count invariants that the Sounio stdlib
DataFrame verbs (`df_concat`, `df_filter`, and the `ugroupby` partition)
MUST satisfy.  This is the "measurement DataFrame" thesis claim that a
`pandas`-style analytics library cannot offer: not tested behaviour, but
machine-checked *proofs* that the verbs preserve their structural laws.

We model a DataFrame by its list of rows (`Rows α := List α`) and model each
verb as the corresponding list operation:

* `df_concat a b`   ≙  `a ++ b`         (vertical stack / append)
* `df_filter p df`  ≙  `df.filter p`    (row sub-selection)
* `df_groupby`      ≙  a partition of the rows into groups (`List (List α)`)

We then prove three invariants, all with **zero `sorry`**:

1. **concat additivity** — `(df_concat a b).nrows = a.nrows + b.nrows`.
   Stacking two frames adds their row counts; no row is created or lost.

2. **filter is a sub-selection** — `(df_filter p df).nrows ≤ df.nrows`.
   Filtering can only remove rows, never invent them.

3. **groupby partition** — the sum over groups of each group's row-count
   equals the total `nrows`.  This is the invariant behind `ugroupby`:
   a partition of the rows loses none and duplicates none, so an aggregate
   computed per-group and recombined accounts for exactly the input rows.

All proofs use only Lean 4 core / `Init` `List` lemmas — **no Mathlib**, so the
file compiles standalone with the pinned `leanprover/lean4` toolchain even
though this repo does not build a Mathlib package.
-/

namespace Sounio.DataFrame

/-- A DataFrame is modelled by its list of rows. -/
abbrev Rows (α : Type) := List α

/-- Row count of a frame. -/
def nrows {α : Type} (df : Rows α) : Nat := df.length

/-- Vertical concatenation (`df_concat`) — stack `b` under `a`. -/
def df_concat {α : Type} (a b : Rows α) : Rows α := a ++ b

/-- Row sub-selection (`df_filter`) — keep the rows satisfying `p`. -/
def df_filter {α : Type} (p : α → Bool) (df : Rows α) : Rows α := df.filter p

/-!
## Invariant 1 — concat additivity

`df_concat` adds the two row counts exactly.  Stacking never creates or
destroys a row; the combined frame has precisely `a.nrows + b.nrows` rows.
-/
theorem df_concat_nrows {α : Type} (a b : Rows α) :
    nrows (df_concat a b) = nrows a + nrows b := by
  unfold nrows df_concat
  exact List.length_append

/-!
## Invariant 2 — filter is a sub-selection

`df_filter` can only drop rows, never add them: the result never has more
rows than the input.
-/
theorem df_filter_nrows_le {α : Type} (p : α → Bool) (df : Rows α) :
    nrows (df_filter p df) ≤ nrows df := by
  unfold nrows df_filter
  exact List.length_filter_le p df

/-!
## Invariant 3 — groupby partition

A `groupby` produces a list of groups (`List (List α)`).  The per-group row
counts are `groupCounts`.  We prove that for *any* grouping, the sum of the
group counts equals the number of rows in the flattened frame — i.e. the
partition accounts for exactly the rows it contains.
-/

/-- The per-group row counts produced by a grouping. -/
def groupCounts {α : Type} (groups : List (Rows α)) : List Nat :=
  groups.map List.length

/-- The rows recovered by concatenating all groups back together. -/
def ungroup {α : Type} (groups : List (Rows α)) : Rows α :=
  groups.flatten

/-- **groupby partition (raw form).** For any grouping, the sum of the
    per-group row counts equals the row count of the recombined frame.
    Proved by induction on the list of groups. -/
theorem groupby_counts_sum {α : Type} (groups : List (Rows α)) :
    (groupCounts groups).sum = nrows (ungroup groups) := by
  unfold groupCounts ungroup nrows
  induction groups with
  | nil => rfl
  | cons g gs ih =>
    rw [List.map_cons, List.sum_cons, List.flatten_cons, List.length_append, ih]

/-- **groupby partition (frame form).** If `groups` is a genuine partition of
    the frame `df` — i.e. concatenating the groups reproduces exactly `df` —
    then the sum of the per-group row counts equals `df.nrows`.  No row is
    lost and none is duplicated across the groups.  This is the invariant the
    `ugroupby` verb relies on when it aggregates per group and recombines. -/
theorem groupby_partition {α : Type} (df : Rows α) (groups : List (Rows α))
    (h : ungroup groups = df) :
    (groupCounts groups).sum = nrows df := by
  rw [groupby_counts_sum, h]

end Sounio.DataFrame
