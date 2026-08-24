# Round 13 math claims for review — optimized EL+ closure on full GO go-plus

Context: artifacts/ontology-frontiers round 12 ran a bitmask reduction of the
Lean-verified EL+ 8-rule saturation (crStep / closeSatF of
formal/OntologyELPlusClosureComplete.lean) on the full GO go-plus ontology
(H = 38,245 classes, NR = 92 roles, 57,824 subClassOf, 18,791 existential
restrictions C <= ex r.F, 55 disjoint pairs, 107 roleSub, 60 roleComp chains).
Round 13 rewrites the fixpoint ENGINE for speed (3m43s -> 4.7s wall) and
claims the OUTPUT IS UNCHANGED. Three claims need review.

## Claim 1 (semi-naive roleComp version skipping is complete)
The fixpoint operator on role-edge filler sets F[r][c] (monotone, inflationary,
finite) is iterated as Gauss-Seidel chaotic rounds: a roleSub worklist drain
(single-input rule, standard worklist completeness), then a roleComp scan.
For chain k = (r1, r2, r3) and cell (r1, c) the scan recomputes
acc = union over f in F[r1][c] of F[r2][f] and merges it into F[r3][c].
Optimization: a global version counter gver is bumped on every row change;
ver[cell] records the version of cell's last change; rmax[r] the latest
version in role r; lpv[k,c] the gver value after the last processing of
(k, c). The pair (k, c) is REPROCESSED iff ver[(r1,c)] > lpv[k,c], or
rmax[r2] > lpv[k,c] AND some f currently in F[r1][c] has ver[(r2,f)] > lpv[k,c].
Claim: skipping otherwise is exact — if neither the cell's row nor any of its
current fillers' r2-rows changed since lpv[k,c], then acc is identical to the
acc computed at the last processing, whose merge into F[r3][c] already
happened (sets only grow), so the rule output is already contained in the
current state. Hence the iteration reaches the same least fixpoint as the
naive full-scan fixpoint. (Note the subtlety handled: lpv[k,c] must be set to
gver AFTER processing, and the row must be re-read per chain, because a chain
with r3 == r1 can grow the row mid-scan; caching a stale row and then setting
lpv loses the new fillers' contributions — observed empirically as a 902-edge
leak in the python prototype, fixed by re-reading, after which the prototype
agrees exactly with the naive mirror in all three configurations:
full 2,135,207 / no-roleComp 1,883,813 / no-roleSub 597,305 edges, rounds
4/2/4 identical to the naive schedule.)

## Claim 2 (sparse sorted-list rows preserve semantics)
The bitmask cube F rows (598 words each) are replaced by sorted linked
segments of filler ids in an arena with geometric reallocation; set union is
a linear merge with dedup. Claim: merge-with-dedup of two sorted lists
computes exactly the set union, so every rule application adds exactly the
same fillers as the bitmask OR; no filler is lost or spuriously added.

## Claim 3 (grouped conflict counting is exact)
conflict(c1, c2) for ordered pairs c1 != c2 iff pm[c1] & epm[c2] != 0, where
epm[c] = set of disjointness endpoints that are ancestors of c (<= 70
endpoints, 2 words), pm[c] = union of partner masks of c's endpoints. All
38,245 classes are actors, so the naive count is 1.46G pair checks.
Optimization: group classes by their epm value (218 distinct values with
multiplicities mult[g]); then
  conf = sum over c1 with pm[c1] != 0 of
         (sum over groups g with (gv[g] & pm[c1]) != 0 of mult[g])
         - (1 if (epm[c1] & pm[c1]) != 0 else 0).
Claim: this equals the naive count. The inner sum counts exactly the c2 with
pm[c1] & epm[c2] != 0 (grouped by value), and the subtraction removes exactly
the diagonal pair (c1, c1) when it was counted. Python cross-check on the
real data: naive 792,814,846 == grouped 792,814,846.
