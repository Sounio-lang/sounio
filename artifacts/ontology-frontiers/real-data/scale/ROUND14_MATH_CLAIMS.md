# Round-14 EL+ closure optimization — equivalence claims under audit

File: artifacts/ontology-frontiers/real-data/scale/go_full_elplus_driver.sio
(round 13 -> round 14).  Context: EL+ role-aware boolean closure over the
GO go-plus ontology (H = 38,245 classes, 92 roles, 57,824 sub axioms,
18,791 existential restrictions, 55 disjoint pairs, 107 roleSub axioms,
60 roleComp chains).  The driver computes, in order: (1) atomic ancestor
closure, (2) role-edge fixpoint F[r][c] under three rule families
(parent propagation F[r][c] |= F[r][p] for stated parents p of c;
roleSub F[s][c] |= F[r][c] for r <= s in the role-hierarchy closure;
roleComp F[r3][c] |= union over f in F[r1][c] of F[r2][f] for chains
r1 o r2 <= r3), (3) disjointness conflict counting, (4) two ablation
re-runs (no-roleComp, no-roleSub).  All printed numbers are gated
against a python bitmask mirror; ALL PASS requires exact equality.

Four equivalence claims need scrutiny.  For each: is the transformation
provably output-identical (not just fixpoint-equivalent), and is the
proof sketch sound?

## Claim 1 — sparse ancestor rows == anc bitmask rows

Old: anc[c] (bitmask over H+1 ids) = {c} ∪ {H} ∪ ⋃_{p parent of c} anc[p],
computed in a topological order of the sub-DAG (Kahn; completeness of
the order is checked, nord == H else FAIL).

New: a sorted-list row A[c] = [c, H] merged (sorted union with dedup)
with A[p] for each stated parent p, same topological order.
atomic_edges := Σ_c (|A[c]| - 1), replacing Σ_c (popcount(anc[c]) - 1).

Proof sketch: by induction on the topological position, A[c] as a set
equals anc[c] as a set (same recurrence, same base, union is union;
sorted merge with dedup computes set union).  Therefore |A[c]| =
popcount(anc[c]) and the edge counts agree.

## Claim 2 — expand restricted to seed-bearing roles is a no-op change

The expand sweep does, for each class c in topological order, each
stated parent p, each role r: F[r][c] |= F[r][p].  Claim: for a role r
with NO existential seed (no restriction ∃r.f on any class), every
F[r][·] row is empty throughout the sweep, so every check/merge for
that r is a no-op, and skipping r entirely leaves the state entering
the role fixpoint bit-identical.  Proof sketch: induction on topo
position — F[r][c] can only receive content from F[r][p] (expand) or a
seed (none for r); all empty.  Roles without seeds still acquire edges
later via roleSub/roleComp inside the fixpoint, which operates on the
non-empty-cell list and the dirty queue, not on the expand loop; the
monotone rule set is unchanged, so the computed least fixpoint (and the
printed round counts 4/2/4 under the same chaotic-iteration schedule)
are unchanged.

## Claim 3 — chain-relevant cell list for comp_scan is a no-op change

comp_scan iterates the non-empty cell list; for each cell (r1, c) it
iterates the chains with that r1 (CSR koff/klist).  Cells whose role is
not any chain's r1 have an empty CSR range: the inner loop body never
executes for them.  Claim: pre-filtering the scan list to cells with
koff[r1+1] > koff[r1] (collected at the same moment a cell is first
marked non-empty, snapshot length taken at scan start, same as before)
yields the identical stream of (cell, chain) pair processings and thus
identical results.  Instrumented check: 1,313,528 pair checks and
341,635 reprocessings, identical counts to the unfiltered round-13
driver; final ne_n = 216,783 and arena_n = 5,041,814 bit-identical.

## Claim 4 — group-level partner masks and conflict sum == per-class version

Setup: endpoints are the classes appearing in the 55 disjoint pairs
(≤140 distinct, indexed 1..nep).  epm[c] = 128-bit mask of endpoints
that are ancestors of c (incl. c itself, since c ∈ anc[c]).  pm[c] =
mask of endpoints partnered (via some disjoint pair) with at least one
endpoint in epm[c].  Conflict count:
conf = Σ_{c : pm[c]≠0} ( |{ c' : epm[c'] ∩ pm[c] ≠ ∅ }| - [epm[c] ∩ pm[c] ≠ ∅] ).

New: group classes by epm value (218 distinct non-zero values g with
multiplicity mult[g]; classes with epm = 0 contribute nothing since
pm = 0 for them).  pm is computed per group: pmg[g] = f(gv[g]) where f
is the partner-map — valid because pm[c] depends on c only through
epm[c].  Then
conf = Σ_{gc : pmg[gc]≠0} mult[gc] · ( Σ_{g : gv[g] ∩ pmg[gc] ≠ ∅} mult[g] - [gv[gc] ∩ pmg[gc] ≠ ∅] ).

Proof sketch: inner per-class sum S(c) = Σ_{groups g overlapping pm[c]}
mult[g] counts exactly the classes c' with epm[c'] ∩ pm[c] ≠ ∅ (every
class with non-zero epm is in exactly one group).  The diagonal
correction subtracts 1 exactly when c itself is among them, i.e. when
epm[c] ∩ pm[c] ≠ ∅; since epm[c] = gv[gc] for c in group gc, the
correction is group-uniform, and multiplying by mult[gc] applies it per
member.  Hence the group-level double sum equals the per-class sum.

Also under this claim: epm built by walking sparse ancestor rows
(setting bit ep_id[a]-1 for each ancestor a that is an endpoint) is
equivalent to testing each endpoint's bit in the anc bitmask, given
Claim 1 (same ancestor sets).

## Empirical corroboration

All mirror-gated numbers are bit-identical to round 13: atomic edges
395,939; role edges 2,135,207; conflicts 792,814,846; ablations
1,883,813 / 597,305; rounds 4/2/4; ne_n 216,783; arena_n 5,041,814.
The question is whether the PROOFS (not just this dataset's numbers)
hold — in particular any edge case: duplicate stated sub axioms (c,p)
appearing twice; a class that is its own ancestor; disjoint pairs where
both endpoints coincide or where an endpoint equals the top id H;
roleComp chains with r1 == r2 or r3 == r1; a seed role that is also
inactive; self-overlapping groups in Claim 4's diagonal correction.

## LLM-offload verdict (2026-08-05, `bin/llm-offload -t math-review -p xai`, Grok 4.5)

All four claims [OK]: sorted merge+dedup = set union on a checked topo
order (Claim 1); seedless roles receive nothing in expand, so the state
entering the fixpoint is bit-identical and the LFP/round counts are
unchanged (Claim 2); cells with an empty chain CSR range contribute zero
(cell, chain) iterations, so the filtered scan yields the identical pair
stream (Claim 3); pm depends on c only via epm[c], the diagonal flag is
uniform per group, and self-overlap / (e,e) pairs / endpoint = H are
preserved (Claim 4).  "No edge case in the audit list breaks
output-identity.  Empirical bit-identity is corroboration only; the four
proofs are already sufficient."  Logged in .claude/llm_offload_log.md.
