<!-- docs:meta
topic_id: repo.docs.papers.ontology-frontiers-technical-note-2026-08-05
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.papers.ontology-frontiers-technical-note-2026-08-05
-->

# A Verified EL⁺ Reasoning Stack in a Self-Hosted Compiler: Mechanized Metatheory, Real-Ontology Scale, and a 3.3× Edge over ELK

**Consolidated technical note — ontology-frontiers research line, rounds 1–14**
**Date:** 2026-08-05 · **Branch:** `research/zd-fiber-antisymmetry-lemma-20260731`
**Artifacts:** `formal/Ontology*.lean`, `formal/ClaimStatusInterval.lean`,
`stdlib/ontology/`, `examples/ontology*/`, `examples/clinical/ddi_elplus_demo.sio`,
`examples/epistemic/traceability_elplus_demo.sio`,
`artifacts/ontology-frontiers/`

## Abstract

We report a fourteen-round research line that takes description-logic
ontology engineering in the Sounio ecosystem from three open problems mined
out of the recent literature to a **machine-checked, executable EL⁺ reasoning
stack that outperforms the reference ELK reasoner on full-scale data**. On
the formal side, Lean 4 mechanizations (no Mathlib, zero `sorry`, no new
axioms) deliver: soundness of a nine-rule EL⁺ deduction system against a
Tarski semantics; soundness *and* completeness of the boolean role-aware
saturation closure via a constructive fixpoint and a canonical-model truth
lemma; query-side normalization with mechanized conservativity of
definitional extensions, lifting completeness to arbitrary concepts; and a
conflict oracle that is a theorem rather than an input. On the executable
side, the executable mirror of that closure — an 8-rule completion-rule
fixpoint over a seed of stated/reflexive/⊤ facts, compiled by the
self-hosted Sounio compiler — closes the **full GO go-plus ontology under
the repository's GO-only extraction policy** (38,245 classes, 92 roles,
2,135,207 derived role edges, 792,814,846 derived atomic conflicts
(ordered pairs)) in
**4.7 s wall, 3.3× faster than the recorded ELK 0.6.0 reference figure
(15.4 s)**, with every output figure bit-identical to an independent Python
mirror. Applications
demonstrated on top of the verified engine: epistemic alignment repair on
the real OAEI Anatomy track, a role-aware drug–drug interaction smoke demo,
a SNOMED CT mini-closure adapter, and VIM3 metrological traceability
expressed as EL⁺ role composition. Fourteen executable prototypes are
continuously gated; every mathematical round passed mandatory external
LLM-offload math review. We are explicit about what is *not* shown:
super-class-side existential restrictions, logical definitions
(intersections), and cross-namespace fillers are outside the extracted
fragment; full SNOMED CT scale and pharmacological validation of the DDI
demo remain open.

## 1. The research line

The line grew in three phases:

- **Rounds 1–6 — epistemic ontology engineering, grounded.** Alignment
  repair with typed confidence, verifiable claim status, and a priori
  consistent evolution, each with a Sounio prototype and a Lean 4
  formalization; then grounding of the *conflict oracle* in mini-EL
  semantics, verified closure, deterministic tie-breaking, and validation on
  the real OAEI 2016 Anatomy track.
- **Rounds 7–9 — EL⁺ metatheory.** The fragment SNOMED CT actually uses
  (concepts `atom | ⊤ | ⊓ | ∃r.C`; axioms `sub | disj | roleSub | roleComp`),
  a role-aware saturation engine proved sound, then complete over the
  saturation universe (canonical model), then complete for *all* concepts
  (query-side normalization + conservativity).
- **Rounds 9–14 — scale and applications.** The verified engine in the
  real-data repair pipeline, on role-rich GO/RO, on full GO go-plus, a
  worklist-leak bug found and fixed, a ~47× self-speedup (3 m 43 s → 4.7 s)
  that moves the engine from behind ELK to ahead of it, multi-ontology
  scale (GO cones + CL + UBERON), and four application integrations
  (repair, DDI, SNOMED, metrological traceability).

Round-by-round ledger (commits on this branch):

| Round | Contribution | Commit |
|---|---|---|
| 1–2 | 3 verified frontiers + gap-closure swarm | `54cef93d7e` |
| 3 | Multi-partner minimal repair + CI gate | `1568589167` |
| 4 | EL-grounded conflict oracle | `3ae13e1249` |
| 5 | Verified closure + deterministic ties | `79764b60ac` |
| 6 | Real OAEI validation + stdlib packaging | `f144266b41` |
| 7 | EL⁺ formalization, full Anatomy scale, miscompile audit | `f4bf420834` |
| 7b | Role composition `r∘s⊑t`; Slurm N=100k confirmation | `f471600433` |
| 7c | Role-aware closure soundness | `8e24ad044e` |
| 8 | Completeness of the role-aware closure | `049c36fdc7` |
| 8b | Executable EL⁺ closure demo; complete-universe note | `14bef4ca93` |
| 9 | Normalization + conservativity (all-concepts completeness) | `62f16a646c` |
| 9 | Role-aware closure in the real-data pipeline (same round, data lane) | (round-9 artifacts) |
| 10 | Role-aware closure in the repair pipeline | `83017c71cf` |
| 11 | Role-rich real ontology (GO/RO slice) | `28bb8da3df` |
| 12 | Full GO go-plus | `a978f2f113` |
| 13 | Optimization (3.3× vs ELK), DDI, multi-ontology | `56ea5df30b` |
| 13b | SNOMED adapter demo + `close()` repair | `3e4441051f`, `2e67714f0a` |
| 14 | Metrological traceability as role composition | `fa70e298e8` |

## 2. Formal results (Lean 4)

All formalizations are Lean 4 **without Mathlib**, with **zero `sorry` and
no new axioms**; confidence arithmetic is exact per-mil naturals. Every
round passed the repository's mandatory external math review
(`bin/llm-offload -t math-review -p xai`: PASS; logs in
`.claude/llm_offload_log.md` and `agent_logs/*offload*.md`).

### 2.1 Soundness — mini-EL and EL⁺

- `formal/OntologyELReasoner.lean` — mini description logic (`sub`/`disj`),
  Tarski semantics, inductive subsumption closure. **`incoherent_empty`**:
  every class marked incoherent by the closure is empty in every model of
  the TBox. **`oracle_sound`**: derived conflict pairs cannot hold
  simultaneously in any model.
- `formal/OntologyELPlus.lean` (~530 lines) — the EL⁺ fragment: concepts
  `atom | ⊤ | C ⊓ D | ∃r.C`, axioms `sub | disj | roleSub | roleComp`,
  role-interpretation semantics, a **nine-rule** deduction system
  (`ofAxiom`, `refl`, `trans`, `conjIntro`, the two `conjElim`s counted as
  one family, `exMono`, `exRoleSub`, `exComp`, `topRule`) with
  **`der_sound`** by induction, **`incoherentP_empty`**, and
  **`oracle_sound_P`**. The executable engine mirrors this as **8
  completion rules** (transitivity, ⊓-elim/intro, stoR/RtoS, Rmono,
  roleSub, roleComp) over a seed that already covers `ofAxiom`, `refl`,
  and `topRule`. Worked `Fin 8 × Fin 3` SNOMED-style instance
  (Pneumonia ⊑ ∃RoleGroup.(Lung ⊓ Inflammation) ⊑ ∃RoleGroup.Organ),
  including `der_pneumonia_rg_organ_via_comp` through the composition axiom
  `DirectSite ∘ PartOf ⊑ RoleGroup`.

### 2.2 Soundness of the computed closures

- `formal/OntologyELClosureVerified.lean` (~700 lines) —
  **`subB_iff_subDer`**: the Boolean fixpoint closure coincides with the
  inductive deductive system in full generality (soundness via an iteration
  invariant; completeness via walk linearization with an n+1 iteration
  bound). **`conflictB_iff`**: the Boolean conflict oracle *is* the semantic
  derived-conflict relation, both directions.
- `formal/OntologyELPlusClosureVerified.lean` — the role-aware saturation
  engine `seedS` / `crStep` / `closeSat` over the full concept language.
  **`roleSubB_iff`**: the role-hierarchy closure is exactly the deductive
  role hierarchy. **`subBPlus_sound`** / **`conflictBPlus_sound`**: every
  computed answer is derivable (`SatJustified` invariant, preserved by each
  completion rule).

### 2.3 Completeness — fixpoint, canonical model, truth lemma

- `formal/OntologyELPlusClosureComplete.lean` — the round-7 engine iterated
  a fixed six rounds; this file closes the gap:
  - **Constructive fixpoint**: `closeSatF` stabilises within
    `satFuel = |allPairs| + |allTriples|` rounds (pigeonhole over the finite
    fact set; `find_stable` proved by induction, no classical choice), and
    is closed under every completion rule.
  - **Canonical model**: domain = the saturation universe; atoms and roles
    read off the closure. `canon_satisfies` (the model satisfies the
    subsumption/role part of the TBox) and the **truth lemma**
    (`truth_lemma`: truth in the canonical model ↔ membership in the
    closure, by structural induction on concepts).
  - **`subBPlusC_iff` / `conflictBPlusC_iff`**: the computed Boolean answers
    are *exactly* the deductive closure over the saturation universe —
    soundness and completeness of the engine the `.sio` code mirrors.

### 2.4 Normalization and conservativity — all-concepts completeness

- `formal/OntologyELPlusNormalization.lean` — answers the open question of
  `docs/research/ontology_elplus_complete_universe_open_question_2026-08-03.md`
  via query-side fresh-name flattening (in the spirit of
  Baader–Brandt–Lutz, *Pushing the EL Envelope*):
  - **`der_collapse`**: every derivation over the definitional extension
    collapses to a derivation over the original TBox — the syntactic form of
    **conservativity of definitional extensions**.
  - **`der_normTBox_iff`**: `Der (normTBox t C D) A_C A_D ↔ Der t C D`;
    composed with `subBPlusC_iff` this yields **all-concepts completeness**:
    the closure decides `C ⊑ D` for arbitrary `C, D`, with no universe
    membership side condition.

### 2.5 The repair/evolution cluster (rounds 1–5)

- `OntologyAlignmentRepair.lean`: `mem_repair_nil`, `pairwise_repair_nil`
  (retained set conflict-free), `repair_witness_nil` (maximality witnesses).
- `OntologyRepairEquivalence.lean`: **`repair_iff_greedy`** (pairwise
  drop-weaker greedy ≡ priority-fold) under distinct confidences and a
  **cluster hypothesis** — and the mechanized counterexample
  **`cx_equivalence_fails`** showing the hypothesis is *necessary* (the
  original "distinct confidences suffice" claim is false in general).
- `OntologyMinimalRepair.lean`: `decide_optimal` (epistemic-mass decision
  optimal), both branches consistency-preserving, unique-minimal removal.
- `OntologyRepairTies.lean`: equivalence extended to arbitrary confidences
  via lexicographic (confidence, id) priority; `greedyStep_prio_eq_sio`;
  `greedyDrop_deterministic`.
- `OntologyEvolution.lean` / `OntologyEvolutionRepair.lean`:
  `mem_versions_consistent` (a priori chain invariant), `applyEdit_reject`,
  `repair_retry`.
- `OntologyClaimStatus.lean` / `ClaimStatusInterval.lean`:
  weakest-link chain theorems, `dsNum_ge_max` (Dempster–Shafer fusion never
  drops below the best source), interval confidences `[lo, hi]`.

## 3. Executable results

All drivers run under `./bin/souc` (Madaros engine) and print `ALL PASS`;
14 prototypes are gated by `scripts/ci/ontology_frontiers_gate.sh` (14/14
OK) plus `scripts/ci/ontology_multi_ontology_gate.sh` (2/2 OK). Every
large-scale figure is cross-checked against an independent Python bitmask
mirror that aborts on any divergence.

### 3.1 Real data: OAEI Anatomy (rounds 6–7, 9–10)

- Extraction of human.owl (3,304 classes) / mouse.owl (2,744) with the
  official reference alignment: 1,961 classes under the round-6 ancestral
  cap, 2,266 subsumptions, 17 disjointness pairs, 6,638 candidate mappings
  (lexical Jaccard; P=0.187 / R=0.817 against the reference).
- **368 derived conflicts (unordered pairs) — zero among reference
  mappings.** Epistemic repair discards 246 mappings (mean confidence 0.41
  vs 0.55 retained) and **only 3 of 1,238 reference mappings**: the repair
  preferentially removes non-reference mappings.
- Round 7 removed the cap: full 3,304-class TBox, **21,859 closure edges**,
  output byte-identical (the cap was lossless). Round 9 added the role
  layer: 21,761 atom-source role edges (72,089 total), 103,863 S-cells over
  the interned universe of 9,915 concepts, **736 ordered conflicts = the
  same 368 unordered pairs — byte-identical to the atomic closure**,
  confirming the *profile theorem*
  computationally (Anatomy has one active role, no conjunctions, no
  roleSub/roleComp, so roles extend subsumption only into existential
  targets).

### 3.2 Scale envelopes (round 7, incl. Slurm)

| Probe | N | Closure edges | Result |
|---|---|---|---|
| Sparse star | 10,000,000 | 19,999,999 | ALL PASS, 3.5 s |
| Sparse star | 100,000 | 199,999 | ALL PASS, 10.8 s workspace / ~13 s **Slurm `cpu-ops` job 8563+** |
| Sparse chain | 30,000 | 450,015,000 | ALL PASS, 8.9 s |
| Dense N² | 50,000 | — | ALL PASS, 46.7 s (7.5 GB bools) |

The ~24k-statement-per-compilation wall is the real ceiling; the dense N²
N=100k bisection remains a documented Slurm gate (not run, per repo rule).

### 3.3 Full GO go-plus (rounds 11–12)

Extraction policy: GO-only (namespace-restricted fillers, `owl:deprecated`
excluded), role set RO-closed. Full GO: **H = 38,245 classes, NR = 92
roles**, 57,824 sub, 18,791 existential restrictions, 55 disjoint pairs,
**107 roleSub, 60 roleComp**.

- **395,939** atomic closure edges (GO-only, reflexive, no ⊤ column).
- **2,135,207** atom-source role edges = existential targets revealed by
  roles (5.4× the atomic closure).
- **792,814,846** atomic conflicts (ordered pairs), independently confirmed
  by per-disjointness-pair counting (the three big GO disjointness cones —
  molecular_function 10,041 × biological_process 24,129 ×
  cellular_component 4,075 — account for ~763M).
- Hybrid fixpoint converges in **4 rounds**. Ablations: without roleComp
  1,883,813 role edges (roleComp contributes 251,394); without roleSub
  597,305 (**roleSub contributes 1,537,902 — 72% of role edges**; the deep
  RO hierarchy dominates at full scale, inverting the slice-level picture
  where roleComp dominated).
- **Bug found and fixed (12b/12c):** the first worklist design was
  *incomplete for roleComp* — when `F[r2][f]` gains an edge, chains
  `r1∘r2⊑r3` must re-fire for every `c` with `f ∈ F[r1][c]` ("direction 2"),
  and those cells are not in the dirty set. Two nominally correct iteration
  orders diverged by 7,200 edges (2,135,093 vs 2,127,893), exposing the
  leak; the hybrid roleSub-worklist + roleComp-full-scan scheme is
  Gauss–Seidel chaotic iteration of the same monotone operator and reaches
  the same least fixpoint. A second 21-edge cascade leak (dirty-set swap)
  was fixed the same round.

### 3.4 Optimization and the ELK baseline (round 13)

Round 12 ran in **3 m 43 s wall** (`souc run`, compile + run) — an order of
magnitude slower than the recorded ELK 0.6.0 reference figure (~15.4 s on
the same full-GO classification task; ≈14× by wall time, recorded as
"13.6×" in the round-13 results doc). Round 13 rewrote the fixpoint
engine:

| | compile | run | wall (`souc run`) | vs ELK 15.4 s |
|---|---|---|---|---|
| Round 12 (dense bitmask cube, 17.9 GB BSS) | ~3 s | 204 s | **3 m 43 s**¹ | ≈14× slower |
| Round 13 (sparse sorted-list rows, ~0.9 GB BSS) | ~2.5 s | **2.2 s** | **4.7 s** | **3.3× faster** |

¹ The round-12 wall (223 s) exceeds compile+run (~207 s): the difference is
compiler-wrapper and harness overhead inside `souc run`, as recorded in
`OPTIMIZATION_RESULTS.md`.

Techniques: sparse sorted-list rows in a 256 MB arena (only 216,783 of the
~3.74M padded `(role, class)` cells — 5.8% — are ever non-empty; rows
average ~10 fillers); a
non-empty-cell list driving expand/count/clear; a single-queue roleSub
worklist; **version-skipped semi-naive roleComp** (re-fire only if the
cell's row or a current filler's r2-row changed — the complete semi-naive
form of the round-12b rule); grouped conflict counting (218 distinct
endpoint-ancestor masks: 1.46G → 8.3M iterations). Every output figure is
**bit-identical** to the round-12 mirror, including both ablations and the
4/2/4 round counts. Soundness/completeness are unchanged: every merge is an
exact set union and the skipping machinery only avoids recomputing rule
outputs whose inputs are unchanged. Prototyped and validated in
`analyze_sparse.py` / `analyze_opt.py` / `analyze_conf.py`; full write-up in
`artifacts/ontology-frontiers/real-data/scale/OPTIMIZATION_RESULTS.md`.

### 3.5 Multi-ontology scale (round 13)

Same engine, five targets (`artifacts/ontology-frontiers/multi-ontology/`):

| Target | H | NR | sub | exsub | disj | roleSub | roleComp | atomic edges | role edges | conflicts | rounds |
|---|---|---|---|---|---|---|---|---|---|---|---|
| GO:0008150 BP | 24,129 | 32 | 40,863 | 12,597 | 28 | 36 | 31 | 298,203 | 1,480,543 | 21,144,668 | 4 |
| GO:0005575 CC | 4,075 | 7 | 4,693 | 2,158 | 21 | 6 | 6 | 23,943 | 105,887 | 8,621,578 | 4 |
| GO:0003674 MF | 10,041 | 28 | 12,268 | 433 | 3 | 37 | 26 | 73,793 | 45,685 | 4,522 | 3 |
| CL | 3,335 | 29 | 4,664 | 477 | 35 | 29 | 14 | 37,926 | 146,188 | 1,071,098 | 5 |
| UBERON | 14,975 | 128 | 19,607 | 17,080 | 589 | 87 | 36 | 150,515 | 2,343,535 | 25,001,610 | 7 |

- The three GO cones **partition** full GO exactly; their atomic edges sum
  to **395,939 — exactly the round-12 total**, and the grouped conflict
  counter over cone masks reproduces **792,814,846** by a second,
  independent algorithm. The cross-cone share is **763,044,078 (96.24%)** —
  exactly `2×(10,041×24,129 + 10,041×4,075 + 24,129×4,075)`, the ordered
  pairs across the three pairwise-disjoint root cones.
- The cone role-edge sum (1,632,115) falls 503,092 short of the full-GO
  total because 3,603 stated restrictions cross cones — measured, not
  assumed; the decomposition identity is deliberately *not* asserted there.
- **roleSub dominates every target** (46–77% of role edges). The
  role/stated amplification varies 17× across targets: CL 306×, UBERON
  137×, GO BP 118×, GO MF 105×, GO CC 49×.

## 4. Applications on the verified engine

- **Alignment repair (rounds 6, 10).** `stdlib/ontology/elplus.sio` exports
  `elplus_derive_conflicts` (conflict derivation over the closed role-aware
  matrix, stride-compatible with the round-6 repair module), and all three
  repair drivers now compute conflicts with the verified EL⁺ engine instead
  of a hardcoded oracle. On the Anatomy profile the repair output is
  provably unchanged (profile theorem); on the mini TBox a genuinely
  *role-derived* conflict (`heart ⊑ ∃part_of.Organ`,
  `∃part_of.Organ ⊥ DrugClass`) appears that the atomic closure cannot see.
- **Drug–drug interactions (round 13).**
  `examples/clinical/ddi_elplus_demo.sio` bridges the ChEBI grounding of
  `stdlib/chemistry/ontology.sio` with `elplus_fixpoint` +
  `elplus_derive_conflicts`: a mini pharmacological TBox with role hierarchy
  and composition chains derives DDI flags for a patient panel instead of a
  hardcoded interaction table. **Claims split (per offload review):**
  *proven* — logical fidelity (the closure coincides with EL⁺ entailment,
  by `subBPlusC_iff` / `conflictBPlusC_iff`); *not established* —
  pharmacological adequacy, drug-list completeness, or any FP/FN rate
  against DrugBank/Lexicomp/FDA labels. A derived conflict is a *potential
  pharmacokinetic interaction flag*, not a clinical contraindication.
- **SNOMED adapter (round 13b).** `stdlib/ontology/biomedical/snomed.sio`:
  the `SNOMEDElplus` struct interns SNOMED relationship triples into the
  dense concept table and exposes O(1) subsumption/role-target queries. A
  cross-module miscompile (`ELPLUS_MAXC` resolving to 8 instead of 64 inside
  an impl-method frame, making the fixpoint a no-op) was root-caused and
  fixed by inlining the full 8-rule fixpoint in `close()`; the acceptance
  test `tests/stdlib/ontology/test_snomed_elplus_adapter.sio` now passes.
  `examples/ontology/biomedical/snomed_elplus_demo.sio` shows the classic
  Pericarditis/Endocardium ⇒ "heart finding" inference through
  `finding_site ⊑ part_of` and `part_of ∘ part_of ⊑ part_of`.
- **Metrological traceability (round 14).**
  `examples/epistemic/traceability_elplus_demo.sio` expresses the VIM3
  (JCGM 200:2012, 2.41) "unbroken chain of calibrations" declaratively:
  roles `calibratedAgainst`, `traceableTo`, `contributesUncertainty`; role
  axioms `calibratedAgainst ∘ calibratedAgainst ⊑ traceableTo`,
  `calibratedAgainst ⊑ traceableTo`, and
  `traceableTo ∘ traceableTo ⊑ traceableTo`. The query "is result R
  traceable to the SI?" becomes `Measurement ⊑ ∃traceableTo.SIPrimary`,
  answered in O(1) after one 3-round fixpoint — with N-link chains (a
  5-edge `FieldDevice` path, no 4-link cap), monotonicity through
  `SIPrimary ⊑ ReferenceStandard`, and negative checks (a generic
  `ReferenceStandard` is not SI-traceable; `contributesUncertainty` does not
  compose into `traceableTo`). This replaces the imperative 4-link walk of
  `stdlib/epistemic/traceability.sio` with a verified declarative
  derivation.

## 5. Baseline: ELK comparison

ELK 0.6.0 is the reference EL⁺⁺ classifier. On full GO go-plus:

| System | Time | Memory footprint | Notes |
|---|---|---|---|
| ELK 0.6.0 | ~15.4 s | — | reference figure for the same ontology |
| Sounio round 12 | 3 m 43 s | 17.9 GB BSS | dense bitmask cube, naive worklist |
| Sounio round 13 | **4.7 s wall (2.2 s run)** | ~0.9 GB BSS | sparse rows + semi-naive evaluation |

The Sounio engine is **3.3× faster** than the recorded ELK figure on wall
time and uses no parallelism — the wins are algorithmic (work only on
non-empty cells, only on changed inputs). Honest caveats: the comparison is
on the extracted GO-only fragment (the Sounio engine's scope), not the full
OWL 2 EL profile ELK handles (it computes no equivalent-class/intersection
reasoning, no super-class-side restrictions); ELK's 15.4 s covers a richer
inference task. The ELK figure itself is the reference number recorded in
the round-12/13 artifacts — the hardware and thread configuration of that
reference run is not controlled by this repository (ELK is multi-threaded
by default; the Sounio run is single-threaded), so treat it as an
order-of-magnitude reference point rather than a controlled benchmark. The
claim is deliberately narrow: *on this fragment and this data*, the
verified-mirror engine beats the reference implementation's recorded time.

## 6. By-products: compiler forensics

The research line doubled as a stress test of the self-hosted compiler.
Documented in `artifacts/ontology-frontiers/compiler-repros/`,
`docs/audit/QUALIFIED_IMPORT_MISCOMPILE_2026-08-02.md`, and
`docs/compiler/KNOWN_LIMITATIONS.md`:

- **Qualified-import miscompile (P5), root-caused and fixed.**
  `self-hosted/ir/lower.sio:15698-15717` mangled `m::f` → `m_f` while
  imported functions were registered as `f`; the linker fabricated a
  bodiless stub and calls fell into it silently. Fixed in `5dc8ca2570`.
- The ~24k-statement wall per compilation (single- and multi-module);
  arrays of structs and unsplat arrays segfault; module-level scalar and
  partially-written array initializers read back garbage; `&&`/`||` do not
  short-circuit the RHS array read; module-level arrays passed across `&!`
  module boundaries miscompile (P5 family). All current drivers carry
  explicit workarounds.

## 7. Open research

1. **Super-class-side existential restrictions** (`∃r.F ⊑ C`) are outside
   both the extraction and the 8-rule engine. GO go-plus has provably zero
   (probed); CL/UBERON each contain one, shown inert under the
   namespace-only policy — but atom-level completeness against full OWL
   semantics is *not* guaranteed there (math-review correction, round 13).
2. **Logical definitions (intersections/equivalentClass)** are counted but
   not extracted (GO: 14,898 anonymous + 91 restriction-shape); the
   normalization file provides the metatheory for conjunctions, but the
   data path does not yet feed them in.
3. **Cross-namespace fillers** (CHEBI/CL/UBERON inside GO; ~85k
   restrictions) are excluded by policy — a multi-namespace interning pass
   is the natural next scale step.
4. **Existential-source statistics** (role edges with `∃r.f` sources, full
   S over the interned universe) are not computed at scale.
5. **Full SNOMED CT** (~300k+ entities) not attempted; the dense N=100k
   Slurm gate remains open.
6. **Second-order uncertainty**: interval confidences are verified; a full
   p-box/GUM treatment remains open (connects to the repository's GUM
   track).
7. **Repair metatheory**: greedy≡fold holds only under the cluster
   hypothesis; minimal repair with multiple conflicting partners is
   decision-optimal but the general minimal-removal problem is open.
8. **Bridge by inspection**: the `.sio` engines are executable mirrors of
   the Lean rules, not extracted code; mirror cross-checks and CI gates
   stand in for a formal extraction guarantee.
9. **DDI validation**: pharmacological adequacy against
   DrugBank/Lexicomp/FDA labels is future clinical work, explicitly out of
   scope of the current claims.

## 8. Reproduce

```bash
cd /workspace/sounio
bash scripts/ci/ontology_frontiers_gate.sh          # 14 prototypes, ALL PASS
bash scripts/ci/ontology_multi_ontology_gate.sh     # GO cones + CL + UBERON
./bin/souc run artifacts/ontology-frontiers/real-data/scale/go_full_elplus_driver.sio   # ~3.6 s (round 14)
./bin/souc run examples/epistemic/traceability_elplus_demo.sio
./bin/souc run examples/clinical/ddi_elplus_demo.sio        # lean_single lane
cd formal && lake build                                     # 14 verified ontology roots
```

## 9. References

**Lean formalizations** (`formal/`): `OntologyAlignmentRepair.lean`,
`OntologyClaimStatus.lean`, `OntologyEvolution.lean`,
`OntologyRepairEquivalence.lean`, `OntologyEvolutionRepair.lean`,
`ClaimStatusInterval.lean`, `OntologyMinimalRepair.lean`,
`OntologyELReasoner.lean`, `OntologyELClosureVerified.lean`,
`OntologyRepairTies.lean`, `OntologyELPlus.lean`,
`OntologyELPlusClosureVerified.lean`, `OntologyELPlusClosureComplete.lean`,
`OntologyELPlusNormalization.lean`.

**Executable stack**: `stdlib/ontology/{closure,repair,evolve,elplus}.sio`,
`stdlib/ontology/biomedical/snomed.sio`,
`examples/ontology_pipeline_demo.sio`,
`examples/ontology_elplus_closure_demo.sio`,
`examples/ontology/biomedical/snomed_elplus_demo.sio`,
`examples/clinical/ddi_elplus_demo.sio`,
`examples/epistemic/traceability_elplus_demo.sio`,
`tests/stdlib/ontology/test_snomed_elplus_adapter.sio`.

**Benchmarks and data**: `artifacts/ontology-frontiers/real-data/`
(OAEI 2016 Anatomy; GO go-plus + RO; scale probes),
`artifacts/ontology-frontiers/real-data/scale/{SCALE_RESULTS,OPTIMIZATION_RESULTS,ROUND13_MATH_CLAIMS}.md`,
`artifacts/ontology-frontiers/multi-ontology/RESULTS.md`,
`artifacts/ontology-frontiers/README.md` (round-by-round ledger),
`docs/research/ontology_elplus_complete_universe_open_question_2026-08-03.md`,
`docs/research/elplus_applications_2026-08-05.md`.

**Literature anchors**: Jiménez-Ruiz et al. 2011
(doi:`10.1186/2041-1480-2-s1-s2`); Solimando et al. 2016
(doi:`10.1007/s10115-016-0983-3`); Bayoudhi et al. 2018
(doi:`10.1111/exsy.12355`); Rovai 2026 (doi:`10.48550/arxiv.2605.09184`);
Baader–Brandt–Lutz, *Pushing the EL Envelope*; JCGM 200:2012 (VIM3, 2.41);
Kazakov, Krötzsch, Simančík, *The Incredible ELK* (ELK 0.6.0 baseline).

**Reviews**: mandatory LLM-offload math review per round
(`bin/llm-offload -t math-review -p xai`), all PASS; logs in
`.claude/llm_offload_log.md`, `agent_logs/go_elplus_offload_2026-08-04.md`,
`agent_logs/go_full_elplus_offload_2026-08-04.md`,
`agent_logs/multi_ontology_offload_2026-08-05.md`,
`artifacts/ontology-frontiers/LEAN_MATH_REVIEW_XAI.md`.
