# Verified Epistemic Ontology Engineering in Sounio

**Technical note — ontology-frontiers research line, rounds 1–6**
**Date:** 2026-08-02 · **Branch:** `research/zd-fiber-antisymmetry-lemma-20260731`
**Artifacts:** `artifacts/ontology-frontiers/`, `formal/Ontology*.lean`,
`formal/ClaimStatusInterval.lean`, `stdlib/ontology/{closure,repair,evolve}.sio`,
`examples/ontology_pipeline_demo.sio`

## 1. Summary

Biomedical ontologies (SNOMED CT, FMA, NCI, UMLS) are aligned, repaired, and
evolved at a scale where manual auditing is impossible — yet the confidence
values attached to mappings, the conflict oracles used by repair tools, and
the consistency of published ontology versions all rest on heuristic
conventions without machine-checked contracts. This research line attacks
three open problems mined from the recent literature and delivers, for each
one: an executable Sounio prototype with runtime-verified invariants, a
Lean 4 formalization with mechanized proofs (no `sorry`, no new axioms),
and — as of round 6 — a reusable standard-library packaging of the verified
algorithms plus an end-to-end demo.

## 2. The three open problems

**P1 — Epistemic alignment repair.** Ontology alignment tools emit mappings
with heuristic confidences that serve as removal costs during repair, with no
formal notion of how much confidence a *retained* entailment deserves.
Jiménez-Ruiz, Cuenca Grau and Horrocks showed that UMLS sources contain
logically detectable errors and proposed confidence-based repair
(doi:`10.1186/2041-1480-2-s1-s2`); Solimando, Jiménez-Ruiz and Guerrini
minimized conservativity violations treating mappings as weights, not
epistemic quantities (doi:`10.1007/s10115-016-0983-3`); and the state of the
art still leaves ~17% of correct mappings undiscovered on the OAEI Anatomy
track (Rovai 2026, doi:`10.48550/arxiv.2605.09184`), so validation and repair
remain open.

**P2 — Verifiable epistemic status of knowledge-graph claims.** Recent work
frames epistemic infrastructure as open: belief graphs with reasoning zones
(doi:`10.48550/arxiv.2602.15353`), epistemic infrastructure for
organizational AI beyond retrieval (doi:`10.48550/arxiv.2601.21116`),
uncertainty-driven evidence selection for RAG
(doi:`10.48550/arxiv.2604.11759`), and tracking the epistemic status of
architectural decisions (doi:`10.48550/arxiv.2603.28444`) — none with a
machine-checkable propagation semantics.

**P3 — A priori consistent ontology evolution.** Consistency checking is
typically *a posteriori*: inconsistencies are detected after a version is
published. Bayoudhi, Sassi and Jaziri proposed a priori consistency for
multiversion OWL 2 DL ontologies but without a mechanized chain invariant
(doi:`10.1111/exsy.12355`); the *consistency principle* of
Jiménez-Ruiz et al. (doi:`10.1186/2041-1480-2-s1-s2`) states that integrating
well-established ontologies should never introduce logical inconsistency.

A fourth, cross-cutting problem emerged in round 4: the conflict oracle used
by P1/P3 was *hardcoded*. Grounding it in the logic of the ontology itself
(mini-EL: subsumption + disjointness) is what makes the repair guarantees
meaningful — exactly the position of `10.1186/2041-1480-2-s1-s2` and
`10.1111/exsy.12355` that matching decisions must be anchored in logical
structure.

## 3. What is verified (five rounds of proofs)

All formalizations are Lean 4 without Mathlib; confidence values are exact
per-mil naturals where arithmetic is involved. Highlights per round:

- **Round 1 — core repair, confidence propagation, guarded evolution.**
  `mem_repair_nil` / `pairwise_repair_nil`: the retained set after repair is
  conflict-free. `repair_witness_nil`: every dropped mapping has a
  *maximality witness* — a retained conflicting mapping of at least its
  confidence. `chainConf_ge`: weakest-link chains preserve confidence
  thresholds of arbitrary length. `dsNum_ge_max`: Dempster–Shafer fusion
  never drops below the best individual source. Consistent chain invariants
  (`mem_versions_consistent`, `applyEdit_reject`): every reachable version of
  a guarded evolution chain is consistent — a priori, by induction — and
  rejecting an edit preserves the previous version exactly.
- **Round 2 — equivalence, counterexample, removal, intervals.**
  `repair_iff_greedy`: the pairwise drop-weaker greedy of the prototype
  computes exactly the priority-fold repair — under distinct confidences and
  a *cluster hypothesis* (conflict graph = disjoint union of cliques). The
  mechanization revealed the original "distinct confidences suffice" claim to
  be **false in general**: `cx_equivalence_fails` certifies by
  `native_decide` a 3-vertex conflict path where greedy and fold diverge —
  the cluster hypothesis is *necessary*. `repair_retry`: after removing the
  unique conflicting partner, a rejected axiom is accepted and consistency
  holds. Interval confidences `[lo, hi]` preserve validity, contain pointwise
  results, and preserve thresholds on the `lo` side.
- **Round 3 — multi-partner minimal repair.** `decide_optimal`: the binary
  admit/reject decision by epistemic mass (sum of partner confidences) is
  optimal between the two options; `admit_succeeds` / `reject_consistent`:
  both branches preserve consistency; the removal set (all partners) is
  uniquely minimal.
- **Round 4 — EL-grounded conflict oracle.** `incoherent_empty`: every class
  marked incoherent by the closure is empty in every model of the TBox.
  `oracle_sound`: derived conflict pairs cannot hold simultaneously in any
  model — the conflict oracle is now a theorem, not an input.
- **Round 5 — verified closure, deterministic ties.** `subB_iff_subDer`: the
  Boolean fixpoint closure computed by the prototype coincides with the
  inductive deductive system (soundness *and* completeness) in full
  generality. `conflictB_iff`: the Boolean conflict oracle *is* the semantic
  derived-conflict relation, both directions — closing round 4's honest gap.
  `repair_iff_greedy_ties`: the greedy≡fold equivalence extended to
  *arbitrary* confidences via lexicographic (confidence, id) priority, with
  `greedyStep_prio_eq_sio` proving the prototype's tie-break step is
  definitionally the prioritized one, and `greedyDrop_deterministic` proving
  determinism.

Every round passed the repository's mandatory external math review
(`bin/llm-offload -t math-review -p xai`: PASS; logs in
`.claude/llm_offload_log.md`).

## 4. Round 6 — reusable packaging (this note's contribution)

The verified algorithms are now packaged as standard-library modules with
documented contracts (`stdlib/ontology/`):

- **`closure.sio`** — `closure_init`, `closure_add_edge`,
  `subsumption_closure` (fixpoint while-loop over a flattened 64×64 Boolean
  matrix), `closure_reaches`, and `derive_conflicts` (the EL-grounded oracle
  of round 4–5 over up to 256 candidate mappings).
- **`repair.sio`** — `greedy_repair` (drop-weaker with the proved
  (confidence, id) tie-break), plus the machine-checkable invariants as
  functions: `repair_is_conflict_free`, `repair_all_witnessed`,
  `repair_count_kept`.
- **`evolve.sio`** — `guarded_add` (apply iff no conflict with the active
  version; rejection preserves the version exactly), `remove_axiom`,
  `version_consistent`.

`examples/ontology_pipeline_demo.sio` runs the whole pipeline end to end:
an 8-class mini-EL TBox (lymphokine ⊑ protein ⊑ molecule; heart ⊑ organ;
two disjointness pairs) → closure (12 edges) → *derived* conflicts for the
shared 5-mapping instance ({m0–m1, m2–m3}) → greedy repair keeping
{m0, m2, m4} with conflict-free + witness invariants checked → a guarded
version chain (add 1, 2, 3; add 4 rejected; remove 2; re-add 4 accepted;
final {1, 3, 4}) with consistency asserted after *every* edit. It prints
`ALL PASS` under `./bin/souc run`.

## 5. Compiler limitations encountered

Documented with minimal verified reproductions in
`artifacts/ontology-frontiers/compiler-repros/REPORT.md`:

1. `where` refinement clauses do not parse in the current Madaros parser —
   contracts are enforced as runtime assertions instead.
2. Arrays of structs and arrays without splat initialization segfault at
   runtime — all prototypes and modules use parallel primitive arrays with
   splat initialization (`var a: [f64; N] = [0.0; N]`).
3. **(New, round 6.)** On the imported-module native lane, the *qualified*
   import form (`use ontology::closure; closure::f(...)`) miscompiles:
   cross-module calls lose `&!` mutations, misread arrays, and can segfault —
   even a scalar `(i64, i64) -> i64` call across modules faults. The *named*
   form (`use ontology::closure::{f, ...}`) compiles and runs correctly
   (mutations propagate, reads are correct). All packaged modules and the
   demo therefore use named imports exclusively. This is consistent with the
   documented open residuals of the imported-module native path
   (`docs/compiler/KNOWN_LIMITATIONS.md`, "D3 exclusive-ref / memory-wall
   chains"); the single-file lane is unaffected (all 8 round 1–5 prototypes
   pass the CI gate `scripts/ci/ontology_frontiers_gate.sh`).

## 6. Honest limitations

- The conflict oracle is logically grounded only for the **mini-EL fragment**
  (subsumption + disjointness). Full EL++ (conjunction, existential
  restrictions, completion-rule classification) is future work.
- The greedy≡fold equivalence holds under the **cluster hypothesis**; the
  mechanized counterexample shows it fails on general conflict graphs.
- Scale is untested beyond prototypes: 8 classes, 5–7 mappings/axioms. Real
  deployments (SNOMED CT 300k+ entities) are out of scope; the
  formalizations cover the combinatorial core, not the systems engineering.
- The propagation algebra (min for derivation, Dempster–Shafer for fusion)
  is one choice among several; probabilistic or fuzzy alternatives are
  unexplored.
- Confidence intervals `[lo, hi]` are verified; a full second-order
  **p-box / GUM** treatment remains open (connects to the repository's GUM
  track).
- The bridge between the proved Lean rules and the `.sio` prototypes is by
  inspection — the prototypes are executable witnesses, not extracted code.

## 7. Reproduce

```bash
./bin/souc check stdlib/ontology/closure.sio   # check: OK
./bin/souc check stdlib/ontology/repair.sio    # check: OK
./bin/souc check stdlib/ontology/evolve.sio    # check: OK
./bin/souc run examples/ontology_pipeline_demo.sio   # prints ALL PASS
bash scripts/ci/ontology_frontiers_gate.sh     # re-verifies the 8 prototypes
cd formal && lake build                        # 10 verified Lean roots
```
