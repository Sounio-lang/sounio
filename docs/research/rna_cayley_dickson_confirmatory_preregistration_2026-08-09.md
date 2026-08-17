<!-- docs:meta
topic_id: repo.docs.research.rna-cayley-dickson-confirmatory-preregistration-2026-08-09
authority: repo_only
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.rna-cayley-dickson-confirmatory-preregistration-2026-08-09
-->

# RNA Cayley-Dickson Tree-Fold Confirmatory Protocol

Status: **pre-registered design; no confirmatory result yet**

Date frozen: 2026-08-09

Concept: `SOUNIO-RNA-CD-INDUCTIVE-BIAS`

## Research question and registered hypotheses

The primary hypothesis is a **structural-hierarchy correspondence**:

> The Cayley-Dickson level used as a fixed composition law is matched to the
> topological complexity of the RNA secondary-structure relation: octonion
> composition is preferentially useful for strictly nested relations, whereas
> sedenion composition is preferentially useful when the relation contains
> independently crossing pairing classes.

This is a finite-data inductive-bias prediction, not an assertion that RNA
molecules literally are Cayley-Dickson numbers or that an algebraic
multiplication table is a molecular mechanism.

The registered hypotheses are:

- **H1, nested correspondence.** On structures with no crossing base-pair
  relation, `OctTree-CD8` has a positive held-out advantage over a matched
  associative 8-dimensional control.
- **H2, crossing correspondence.** On structures containing crossing pairs,
  `SedenTree-CD16` has a positive held-out advantage over a matched associative
  16-dimensional control.
- **H3, hierarchy interaction.** The residual advantage of `SedenTree-CD16`
  over its dimension-matched associative control is larger in the `crossing`
  stratum than in the `nested` stratum, relative to the corresponding
  `OctTree-CD8` residual. A monotone trend across the registered
  crossing-complexity tuple is secondary, not the H3 decision variable.
- **M1, mechanistic sub-hypothesis.** Any OctTree advantage is associated with
  contexts in which reassociation changes the composition. This is secondary.
  OctTree versus Clifford is not an isolated manipulation of associativity,
  because the algebras differ in more than that identity.

H1-H3 can fail. Such failure is evidence against the corresponding registered
prediction under this protocol. It is not permission to replace the hierarchy
hypothesis retrospectively with a different task.

## Exploratory evidence being frozen

The KIMI exploratory lane produced a real but non-confirmatory signal:

- the local Rfam-derived corpus contains 108,072 three-line records from 4,225
  represented families;
- in one sampled `L=128` run, OctTree-8 scored `0.871` and the elementwise
  RealTree-8 control scored `0.551`, a difference of 0.320 accuracy;
- in the same task family, GRU-8 reached `0.998`;
- an affine low-rank tree called `MatrixTree` remained at `0.500`, but it is not
  a free bilinear control and its failure is not evidence that associativity is
  insufficient;
- the experiments were single-run prototypes, sampled train and test from the
  same family pool, did not fully seed PyTorch, and selected the best test
  accuracy;
- the expected result JSON files were not written successfully, so the numbers
  survive in session logs rather than hash-closed receipts.

The frozen input `rfam_structures.fasta` has SHA-256
`050ab71872e2291e424b21bc2af72082c48a78046aa18d422840d98fa9e55eb9`.
An audit before confirmation found 19,692 of 108,072 derived dot-bracket
structures to be unbalanced. The previous extractor removed sequence-specific
alignment gaps without jointly removing the opposite endpoint of a consensus
base pair. Consequently, that derived FASTA is an exploratory input only and
must not be used as the confirmatory cohort.

The complete untracked KIMI package was preserved before this lane was opened:

- custody archive SHA-256:
  `471e78d35782876a64e8a770c1d05b97d6d67c959657fbc641507a2030385988`;
- custody manifest: 36 files;
- source Git commit:
  `4f8419002b22fbb053e335457c7df389a3f8b978`.

These observations generate the hypothesis. They are not part of the
confirmatory estimate.

## Relationship to the KIMI C0-C5 grid

The KIMI C0-C5 grid is preserved as an auxiliary mechanistic experiment. It
tests whether OctTree beats a dense associative Clifford product on a
real-versus-corrupted classifier whose input was reduced to `(`, `)`, and `.`.
It does not include SedenTree, a pseudoknot stratum, full WUSS relations, or an
algebra-by-structural-complexity interaction.

Therefore:

- `OctTree ~= CliffordTree` in that grid means only that octonion-specific
  non-associativity was not separated from dense associative coupling on that
  task;
- chance performance on the balance-preserving sibling-swap task means that
  this corruption was not distinguishable by the registered models;
- neither observation is a test or a refutation of H2 or H3;
- all partial and final C0-C5 results remain reportable and hash-closed, but
  none enters the confirmatory hierarchy estimands below.

This amendment is frozen after seeing partial C0-C5 outcomes but before any
Tier I or Tier II hierarchy outcome. C0-C5 results cannot tune the new task,
thresholds, split, or model-selection budget.

## Two-tier design and primary estimands

### Tier I — structure-only hierarchy test

Tier I tests H1-H3 directly. Input is the canonical structural relation emitted
from complete WUSS parsing: unpaired state, paired endpoints, pairing class,
and a pre-registered relation mask. It contains no nucleotide identity, family
or clan identifier, accession, or sequence-derived feature.

The mask removes both endpoints of each selected pair together. The task is to
reconstruct the hidden pair relation from the remaining structural context.
Mask rate, candidate-pair construction, span matching, and all random streams
are frozen before training and are identical across models. The primary
per-record score is hidden-pair F1; the primary aggregate is the macro-mean over
held-out clan/family grouping units.

Every accepted record belongs to exactly one pre-outcome stratum:

1. `nested`: no two retained base pairs cross;
2. `crossing`: at least one retained crossing pair;
3. `excluded`: a stable reason code explains why the record is unavailable.

For pairs `(i,j)` and `(k,l)` with `i < j` and `k < l`, a crossing is present
when `i < k < j < l` or `k < i < l < j`. The crossing graph has one vertex
for every retained base pair and one undirected edge for every crossing. Its
metrics are defined exactly as follows:

- `crossing_relation_count`: number of graph edges, i.e. unordered crossing
  relations between two base pairs;
- `crossing_pair_count`: number of non-isolated graph vertices;
- `crossing_component_count`: connected components in the subgraph induced by
  non-isolated vertices;
- `maximum_crossing_degree`: maximum number of other base pairs crossed by any
  one base pair, or zero when the graph has no edge.

Crossing complexity is frozen as:

```text
(crossing_relation_count, crossing_pair_count,
 crossing_component_count, maximum_crossing_degree)
```

For structural stratum `s`, define the dimension-matched residual advantages:

```text
B_8,s  = macro_F1(OctTree-CD8, s) - macro_F1(AssocTree-8, s)
B_16,s = macro_F1(SedenTree-CD16, s) - macro_F1(AssocTree-16, s)
```

The primary hierarchy interaction is:

```text
I_hierarchy = (B_16,crossing - B_16,nested)
            - (B_8,crossing  - B_8,nested)
```

This is the stratum interaction of Cayley-Dickson-minus-associative residuals.
The dimension-matched associative comparator is partialled inside each `B`; the
contrast does not claim to remove every optimisation or cross-dimensional
confound. The secondary crossover contrasts are:

```text
Q_nested   = B_8,nested   - B_16,nested
Q_crossing = B_16,crossing - B_8,crossing
```

By construction, `I_hierarchy = Q_nested + Q_crossing`. The two `Q`
contrasts decompose the primary interaction; they are not independent
confirmations and are never counted as additional replications.

The registered direction is `I_hierarchy > 0`, with `Q_nested > 0` and
`Q_crossing > 0` as secondary evidence that the preferred residual bias changes
with structural class. No single model is required to win every stratum in raw
accuracy.

### Tier II — biological transfer

Tier II asks whether a bias surviving Tier I transfers to sequence-to-pair
prediction. Input is RNA nucleotide sequence only; target is the canonical pair
relation. Pair F1 remains macro-averaged by held-out group and is reported
separately for `nested` and `crossing` records.

Tier II is secondary. A positive sequence-to-pair result can support biological
transfer in the same stratum, but it cannot retrospectively rescue a failed
Tier I hierarchy interaction.

Pair precision and recall, F1 by pair distance and crossing complexity,
calibration, runtime, stability, and non-convergence are secondary outcomes.
The legacy real-versus-corrupted classifier remains diagnostic only.

## Cohort production

The raw versioned Rfam Stockholm seed file is the source. Its whole-file hash,
Rfam release identifier, extraction source hash, and family-to-clan metadata
must be recorded before a split is produced.

For every alignment block, the canonical extractor must:

1. parse the complete WUSS consensus pairing relation, including `()`, `<>`,
   `[]`, `{}`, and lettered pseudoknot classes;
2. map each aligned sequence column to an ungapped sequence position;
3. retain a consensus pair only when both endpoints contain nucleotides for
   that sequence;
4. drop both endpoints together when either endpoint is a gap;
5. preserve crossing pairs as separate bracket classes rather than flattening
   them into nested parentheses;
6. emit an exclusion with a stable reason code for malformed WUSS, length
   disagreement, duplicate identity, unsupported residue, or missing metadata;
7. prove in the emitted ledger that every retained pair is reciprocal, in
   bounds, non-self, and unique;
8. derive the `nested` or `crossing` stratum and the complete crossing-complexity
   tuple from the retained pair relation rather than from bracket glyphs alone;
9. require at least one retained pair for Tier I. A record with no pair that can
   be masked is emitted to the exclusion ledger as `NO_MASKABLE_PAIR`, never as
   a zero-pair member of the `nested` estimand.

Each accepted record has a stable `record_id`, family, clan when present,
sequence hash, canonical pair-list hash, structural stratum, crossing-complexity
tuple, length, GC count, pair count, and normalisation version. Families without
an Rfam clan form explicit singleton groups `family:<accession>`; this preserves
group disjointness without claiming an invented biological clan.

Corrupted or augmented descendants carry `parent_record_id` and must inherit
the split of their parent. Splitting a parent and descendant across partitions
is a hard failure.

## Split contract

The primary analysis uses five outer folds assigned by a versioned deterministic
hash of `group_id` and a frozen salt. A `group_id` is `clan:<accession>` when a
clan exists and `family:<accession>` otherwise.

For outer fold `k`:

- fold `k` is test;
- fold `(k + 1) mod 5` is validation;
- the remaining folds are training.

No family, clan, duplicate-sequence cluster, parent record, or augmented child
may cross partitions. Near-neighbour identity is audited and reported after the
group split; it is never used to move individual records across group
boundaries.

Before any fit, Sounio emits and Julia independently verifies a fold-by-stratum
availability table. Every held-out fold must contain at least five independent
groups in each of `nested` and `crossing`, and its corresponding training pool
must contain at least twenty independent groups in each stratum. A missing
cell, an undefined stratum metric, or failure of either minimum is `REFUSE`, not
an invitation to merge folds or move records after inspecting outcomes.

## Models and capacity matching

All models use identical tokenisation, relation mask, canonical tree/decomposition
topology, decoder, loss, optimiser family, update budget, early-stopping rule,
and hyperparameter-search budget unless the difference is the registered
ablation.

Required models:

1. `OctTree-CD8`;
2. `AssocTree-8`, a frozen dense associative 8-dimensional product;
3. `RandomFanoTree-8`, with a frozen table/sign randomisation;
4. `RealTree-8`, the historical elementwise control;
5. `QuaternionTree-CD4`, the lower Cayley-Dickson control;
6. `SedenTree-CD16`;
7. `AssocTree-16`, a frozen dense associative 16-dimensional product;
8. `RandomCayleyTree-16`, with a frozen 16-dimensional table/sign
   randomisation;
9. `TwoBranchOctTree-8x2`, with the same 16-scalar state budget and a frozen
   fusion rule;
10. `RealTree-16`;
11. capacity-matched learned bilinear and bidirectional GRU references.

The primary contrasts use the fixed `AssocTree-8` and `AssocTree-16`; a best
control is not selected after observing outer-test performance. Random-table,
lower-algebra, two-branch, learned, and GRU models are robustness or predictive
references.

Trainable parameter count must be within 10% inside each registered contrast.
For 8-versus-16 comparisons, scalar-state budget, decoder capacity, update
budget, estimated FLOPs, effective batch size, wall time, and failures are all
reported; parameter count alone cannot explain a hierarchy effect.

`SedenTree-CD16` must emit non-finite state and gradient counts, gradient-norm
distribution, loss divergence, and pre-specified zero-divisor diagnostics.
Instability is a result, never a silently repaired condition.

## Training and deterministic execution

There are five outer folds and ten pre-specified seeds per model. Seeds govern
initialisation, minibatch order, augmentation, random control tables, and all
accelerator libraries. Deterministic-algorithm mode is mandatory. A requested
deterministic operation that the runtime cannot provide yields `REFUSE`; it
does not silently fall back.

Checkpoint and hyperparameter selection use only the inner validation data.
Every launched seed, including divergence and non-convergence, remains in the
receipt. The test score is never used to select an epoch or configuration.

## Negative controls

The following controls are pre-specified:

- labels permuted within length and GC strata;
- nucleotide order shuffled while preserving mono- and dinucleotide counts in
  Tier II;
- pair masks permuted within pair-span and structural-stratum bins in Tier I;
- consistent leaf-order permutation;
- correct versus randomised Fano table;
- frozen random versus learned structure constants;
- parent-aware corruption controls that cannot cross a split;
- nearest-neighbour audit between train and test.

Any above-chance performance on permuted labels, any parent leakage, or any
hash mismatch invalidates the affected run.

## Statistical analysis

The grouping unit, not the individual sequence or training seed, is the primary
unit of uncertainty. For each fold and seed, metrics are retained per record and
aggregated per family/group. Every model contrast is paired within group, fold,
mask, and seed.

The registered point estimate is the equal-group-weighted mean of paired
group-level contrasts, with fold and seed retained as paired repeated measures.
The 95% confidence interval is a two-sided BCa hierarchical bootstrap interval
over groups, records within groups, and seeds, applied to that same estimator.
A paired permutation analysis is the confirmatory sensitivity check. A
gatekeeping family controls multiplicity:
`I_hierarchy` is tested first; H1 and H2 are then tested with Holm correction.
`Q_nested`, `Q_crossing`, complexity gradients, and mechanistic analyses are
secondary and Holm-corrected. Every fold/seed estimate is published; no
best-seed summary is permitted.

## Decision rule

Full support for the registered hierarchy requires all of the following on
Tier I held-out groups:

- `B_8,nested >= 0.02`, with a 95% lower confidence bound above zero;
- `B_16,crossing >= 0.02`, with a 95% lower confidence bound above zero;
- `I_hierarchy >= 0.02`, with a 95% lower confidence bound above zero;
- each primary effect has the registered sign in at least four of five outer
  folds and is not concentrated in one seed or grouping unit;
- random-table, leakage, mask, capacity, update-budget, stability, and
  deterministic-execution gates pass;
- independent Sounio and Julia artifacts agree byte for byte.

The `0.02` minimum applies to the registered point estimate. The separate
confidence requirement is that the lower bound of its two-sided 95% BCa
interval is above zero; the lower bound is not required to exceed `0.02`.

Results are classified without collapsing distinct hypotheses:

- `HIERARCHY_SUPPORTED`: H1, H2, and H3 all pass;
- `LOCAL_CD_BIASES_ONLY`: H1 or H2 passes but the interaction does not;
- `NESTED_ONLY` or `CROSSING_ONLY`: exactly one registered stratum passes;
- `NO_CONFIRMATORY_EVIDENCE`: the confidence intervals include no registered
  minimum effect;
- `DIRECTION_CONTRADICTED`: `I_hierarchy` is negative with its upper 95% bound
  below zero;
- `REFUSE`: data, split, stability, execution, or independent-validation gates
  fail.

An OctTree-Clifford tie in the KIMI classifier can demote only M1 for that task.
It cannot set any of the H1-H3 outcomes. The exploratory observations remain
preserved regardless of outcome.

## Language and receipt boundary

Sounio is the canonical scientific producer. It owns Stockholm/WUSS
normalisation, complete pair extraction, nested/crossing assignment, crossing
complexity, exclusions, group/split assignment, mask generation, run contracts,
canonical predictions, and receipts. The training runtime is an explicitly
bounded accelerator worker consuming Sounio-produced inputs; it is not the
authority for cohort or metric semantics.

Julia independently reparses the raw Stockholm/WUSS input, reconstructs the
pair mapping, structural stratum, crossing complexity, masks, and split,
verifies raw predictions, recomputes all metrics and uncertainty analyses, and
compares canonical serialisations byte for byte. Julia never becomes a
production fallback. Missing Sounio or Julia, a dirty release source, an absent
manifest, or a hash divergence fails closed.

## Claim boundary

If Tier I passes H1-H3, the strongest permitted claim is:

> Under this frozen Rfam version and clan/family-held-out protocol, the
> registered octonion tree law was preferentially predictive for strictly
> nested RNA relations and the registered sedenion tree law was preferentially
> predictive for relations containing crossings, relative to matched
> dimension-specific controls.

If Tier II also passes in the corresponding strata, the additional permitted
claim is:

> The registered structural bias transferred to the sequence-to-pair task in
> the same held-out structural strata.

The following remain forbidden:

- “RNA is a Cayley-Dickson algebra”;
- “the Cayley-Dickson hierarchy corresponds universally to RNA hierarchy”;
- “non-associativity is biologically necessary”;
- “zero divisors explain pseudoknot biology”;
- any clinical, causal, or all-RNA generalisation;
- any confirmatory claim derived from the legacy classifier, a single seed,
  the failed affine MatrixTree, or RF00008/RF00050 alone.
