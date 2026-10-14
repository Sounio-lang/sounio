# Round 12 math claims for review — EL+ role-aware closure on FULL GO (go-plus)

Context: rounds 9-11 of artifacts/ontology-frontiers built a role-aware EL+
boolean closure (mirror of the Lean-verified 8-rule saturation, crStep /
closeSatF of formal/OntologyELPlusClosureComplete.lean) and ran it on an
OAEI Anatomy TBox (1 role, no role axioms) and a 204-class GO slice (8
roles).  Round 12 scales to the FULL GO go-plus ontology (GO-only policy):
H = 38,245 classes, 57,824 subClassOf, 18,791 existential restrictions
C <= ex r.F with named GO fillers, 55 disjoint pairs, NR = 92 roles
(55 used + 37 RO-closure derived-edge targets), 107 roleSub, 60 roleComp.

The dense boolean encoding (S matrix over U = (H+1)*(NR+1) = 3.56M interned
concepts) is infeasible at this scale, so round 12 uses a bitmask reduction
and claims it computes the SAME least fixpoint on this data profile.

## Claim 1 (profile, carried from round 11, re-verified computationally)
For a TBox with no conjunctions and no superclass-side restrictions
(ex r.F <= C), roles add no ATOMIC subsumptions and no ATOMIC conflicts:
the atom-atom projection of the full 8-rule fixpoint equals the transitive
closure of the stated subClassOf edges.  (go-plus has 0 superclass-side
restrictions; intersection restrictions are dropped by the syntactic
extractor and counted: 91 restr_shape + 14,898 other anonymous subClassOf.)

## Claim 2 (bitmask reduction exactness)
With anc[c] = stated-sub ancestor set (reflexive, incl. top), F[r][c] =
filler set of derived role edges (r, c, .) with atom source c:
  (a) seeding F[r][c] |= anc[f] for each stated (c, r, f) folds stoR and
      Rmono: seeds are ancestor-closed, and unions of ancestor-closed sets
      are ancestor-closed, so every F[r][c] stays Rmono-closed forever —
      no separate Rmono pass is needed;
  (b) one pass F[r][c] |= F[r][p] over parents p in any topological order
      of the (acyclic) sub DAG folds transitivity+stoR: after the pass,
      F[r][c] = union over ancestors a of c of seed[r][a];
  (c) iterating roleSub (F[s][c] |= F[r][c] for r <=* s) and roleComp
      (F[r3][c] |= union over f in F[r1][c] of F[r2][f]) to a fixpoint
      yields exactly the atom-sourced edge set of the general 8-rule
      fixpoint;
  (d) the existential targets of an atom row c are exactly its role-edge
      fillers (stoR/RtoS bijection), and atomic conflicts computed over
      anc[] equal the role-aware conflicts (by Claim 1).

## Claim 3 (validation ladder)
The reduction was validated against the general set-based fixpoint of
gen_elplus_data.py on the round-11 slice (204 classes, 8 roles) with
EXACT agreement of atom-level numbers (role edges 3380 = ex targets 3380,
atomic edges 1051) in three configurations: full, no-roleComp (2550),
no-roleSub (2617).  The full-GO numbers below are from the bitmask mirror
cross-checked against an independent Sounio implementation of the same
reduction.

## Full-GO results (bitmask mirror, python)
  atomic closure edges (reflexive, excl. top): 395,939
  atom-source role edges = existential targets: 2,135,093
  dirty-fixpoint rounds: 5
  atomic conflicts (ordered pairs): 792,814,846
  ablation no-roleComp: 1,883,813 role edges (roleComp contributes 251,280)
  ablation no-roleSub: 597,284 role edges (roleSub contributes 1,537,809 —
  the dominant family on full GO, the opposite of the 204-class slice
  where roleComp dominated)

## Questions for the reviewer
1. Is the Rmono-closure invariant argument (2a) sound, including its
   interaction with roleComp (composition of closed sets stays closed)?
2. Is the single topological pass (2b) a correct fold of
   transitivity+stoR given parents-before-children processing?
3. Does the dirty-fixpoint (2c) compute the least fixpoint of
   roleSub+roleComp from the seeded state (monotone rules, finite
   universe)?
4. Is the scope limitation honest: existential-SOURCED edge sets (and
   universe-level S-cell totals) are NOT computed — they require a
   composition fixpoint per (r, f) existential concept — and atom-level
   statistics suffice for the round-11 claims?

---

## Reviewer output (bin/llm-offload -t math-review -p xai, 2026-08-04)

[WRONG] Claim 2a wording: "seeding folds stoR and Rmono" -> seeding folds
Rmono only (filler up-closure); stoR is source-side, folded by the topo
pass (2b).  Correction applied to driver header + README.

[OK] Claim 2a (Rmono invariant + roleComp interaction)
[OK] Claim 2b (single topo pass = transitivity + stoR)
[OK] Claim 2c (dirty fixpoint = least fixpoint of roleSub+roleComp)
[OK] Claim 2d (atom-row ex-targets = role-edge fillers; conflicts via anc)
[OK] Claim 3 (validation ladder)
[OK] Arithmetic of full-GO ablations
[OK] Q1-Q3, Q4 (scope limitation honest)
# Round 12b math claims — direction-2 roleComp leak + corrected hybrid fixpoint

Follow-up to the round-12 review (which passed the bitmask reduction).
While cross-validating the Sounio implementation against the python mirror
on the FULL GO (H = 38,245 classes, NR = 92 roles, 60 composition chains),
the two implementations disagreed by exactly 7,200 role edges in the full
fixpoint while agreeing exactly on both ablations (roleSub-only and
roleComp-only) and on the round-11 slice.

## Claim 1 (the leak)
A dirty-cell worklist keyed on the FIRST components of the composition
rule is INCOMPLETE for roleComp.  The rule
  edge(r1, c, f) ∧ edge(r2, f, e) ⇒ edge(r3, c, e)
has two inputs; a new edge (r2, f, e) must re-fire the rule for every c
with f ∈ F[r1][c], but those (r1, c) cells are not in the dirty set when
only (r2, f) is marked.  Missing these "direction-2" derivations makes the
computed relation strictly smaller than the least fixpoint, and the leak
is iteration-order dependent (phase order: 2,135,093; sweep order:
2,127,893 — both below the true value).

## Claim 2 (the corrected scheme is the least fixpoint)
Hybrid round: (i) roleSub applied via a dirty worklist drained to
quiescence — COMPLETE because roleSub (edge(r,c,f) ∧ r ⊑* s ⇒ edge(s,c,f))
has a single relational input, so firing exactly on changed cells is
semi-naive evaluation of that rule; (ii) roleComp applied as a FULL SCAN
over all cells with non-empty F[r1] — the naive rule application.  The
hybrid round is Gauss-Seidel chaotic iteration of a monotone operator on a
finite powerset lattice, hence converges to the unique least fixpoint,
equal to the naive fixpoint and (on this profile) to the atom-sourced
edge set of the general 8-rule EL+ system.

## Claim 3 (evidence)
  (a) The corrected python mirror still agrees EXACTLY with the general
      set-based 8-rule fixpoint on the round-11 slice in all three
      configurations (full 3,380 / no-roleComp 2,550 / no-roleSub 2,617).
  (b) Corrected full-GO totals: full = 2,135,207 (= phase-order leaky
      + 114, sweep-order leaky + 7,314); roleSub-only ablation unchanged
      at 1,883,813 (consistent with the roleSub worklist being complete);
      roleComp-only ablation 597,305 (leaky worklists had agreed on the
      wrong 597,284 — a 21-edge direction-2 leak even without roleSub).
  (c) The Sounio driver implements the same hybrid scheme.

## Questions
1. Is the completeness argument for the roleSub worklist (single-input
   rule, semi-naive) correct?
2. Is Gauss-Seidel chaotic iteration of the monotone role operator
   guaranteed to reach the least fixpoint (no fairness assumption needed
   beyond "every rule is applied in every round")?
3. Is (b)'s pattern — roleSub-only unchanged, roleComp-only slightly
   under, full under by order-dependent amounts — consistent with the
   direction-2 leak diagnosis and nothing else?

---

## Reviewer output 12b (bin/llm-offload -t math-review -p xai, 2026-08-04)

[OK] Claim 1: dirty-worklist keyed only on first components is incomplete
for roleComp (direction-2 firings lost; result ⊆ LFP, order-dependent).
[OK] Claim 2 / Q1: roleSub dirty-worklist is complete (semi-naive,
single relational body atom).
[OK] Claim 2 / Q2: hybrid rounds reach the unique LFP (finite powerset
lattice, monotone inflationary operators, fair schedule).
[OK] Claim 2: hybrid LFP = naive LFP.
[OK] Claim 3(a-c) internal arithmetic and ablation pattern consistent
with the direction-2 leak and nothing else.
