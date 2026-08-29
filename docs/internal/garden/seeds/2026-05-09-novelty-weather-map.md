<!-- docs:meta
topic_id: repo.docs.internal.garden.seeds.2026-05-09-novelty-weather-map
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.internal.garden.seeds.2026-05-09-novelty-weather-map
-->

# Novelty Weather Map

> **Status**: Garden seed | **Last validated**: 2026-05-09 | **Source**: live session capture plus literature triangulation

## Butterfly

> I want help on new ideas, novel knowledge...deep research

The Garden should not only preserve ideas after they crystallize. It should also
help create pressure systems: places where existing Sounio artifacts, outside
literature, and hostile reviewer questions collide hard enough to make new
knowledge possible.

This seed is a weather map. It marks where the air feels charged, where the
storm is probably only metaphor, and where a careful executable bridge could
turn a butterfly into evidence.

## External Anchors Checked

This pass used a lightweight literature triangulation on 2026-05-09. It did not
try to prove priority. It looked for nearby load-bearing ancestors and reviewer
counterexamples.

- Bornholt, Mytkowicz, and McKinley introduced `Uncertain<T>` as a programming
  abstraction for probabilistic uncertain data, explicitly motivated by bugs
  from treating estimates as ordinary floats and booleans:
  <https://www.cs.utexas.edu/~mckinley/papers/uncertainty-asplos-2014.pdf>.
- The BIPM/JCGM GUM suite remains the normative measurement-uncertainty anchor,
  including JCGM 100, JCGM 101, and JCGM 102:
  <https://www.bipm.org/en/web/guest/publications/guides>.
- JCGM 102 extends the GUM framework to multivariate output quantities,
  covariance between outputs, and Monte Carlo propagation for cases where
  linearization is questionable:
  <https://www.bipm.org/documents/20126/2071204/JCGM_102_2011_E.pdf>.
- Refinement types already provide the broad PL ancestor for attaching logical
  predicates to types:
  <https://arxiv.org/abs/2010.07763>.
- Koka and related work are the main effect-system neighbors for algebraic
  effects and row-polymorphic effect typing:
  <https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/koka-effects-2013.pdf>.
- Network meta-analysis already has mature inconsistency machinery, especially
  design-by-treatment interaction, node splitting, and loop inconsistency:
  <https://pmc.ncbi.nlm.nih.gov/articles/PMC4946625/>.
- Octonion associators are an established mathematical object; the nearby
  novelty cannot be "associators exist":
  <https://arxiv.org/abs/1509.07718>.
- FDA PBPK guidance makes PBPK an explicit regulatory-reporting surface:
  <https://www.fda.gov/regulatory-information/search-fda-guidance-documents/physiologically-based-pharmacokinetic-analyses-format-and-content-guidance-industry>.
- Mamba and selective SSMs make associative scan a major modern sequence-model
  reference point:
  <https://arxiv.org/abs/2312.00752>.

## Constellations

### 1. Metrological Compilation

**Bold thesis.** The strongest Sounio-shaped claim is not "uncertainty
propagation in code"; it is "measurement uncertainty becomes a compiler
obligation." The language is interesting where a program cannot silently move
from `Knowledge<T>` to `T` without entering an effect, guard, proof, or audit
surface.

**Prior-art gravity.** `Uncertain<T>` owns the ancestor idea that uncertainty
should change types and conditionals. GUM owns the measurement standard.
Refinement types own predicate-bearing types. Koka owns a mature effect-system
neighborhood.

**Hostile reviewer.** "This is just GUM plus refinements plus effects, with a
new wrapper type."

**Falsification path.** Find one irreducible example where all prior tools either
accept silent uncertainty loss or require runtime/social discipline, while
Sounio rejects or routes the loss through a typed effect boundary.

**First executable bridge.** A tiny comparison corpus: one Sounio program, one
`Uncertain<T>`-style sketch, one GUM spreadsheet-style sketch, and one
refinement-only sketch, all attempting the same confidence-gated collapse.

**Evidence state.** `Hypothesis`. Parts are executable in the repo, but the
comparative novelty floor still needs a deliberately hostile artifact.

### 2. Multivariate GUM-Through-ODE

**Bold thesis.** PBPK is not only a pharmacometric application; it is a stress
test for whether a compiler can carry covariance through a dynamical system
without letting uncertainty become parallel paperwork.

**Prior-art gravity.** JCGM 102 already covers multivariate outputs and
covariances. PBPK uncertainty and variability are old problems. FDA guidance
already standardizes PBPK report shape.

**Hostile reviewer.** "This is standard sensitivity/covariance propagation
inside an ODE. The compiler is incidental."

**Falsification path.** Show a case where omitting a covariance term changes an
uncertainty budget enough to alter an acceptance gate, then show the compiler
surface makes that omission structurally visible.

**First executable bridge.** The current K-AXI/PBPK/GUM lane can become a
minimal "covariance omission witness": CPU analytic truth, GPU kernel result,
and a deliberately broken no-cross-covariance variant that fails the budget
claim.

**Evidence state.** `Executable` for the local proof lane; `Hypothesis` for the
broader research claim.

### 3. Associators For Evidence Synthesis

**Bold thesis.** Network meta-analysis inconsistency can be reframed as an
associator problem: not just whether A-vs-B plus B-vs-C agrees with A-vs-C, but
whether evidence composition changes when the same treatments are parenthesized
through different study designs.

**Prior-art gravity.** Design-by-treatment interaction, node splitting, and loop
inconsistency already model inconsistency directly. Octonion associators already
measure failure of associativity.

**Hostile reviewer.** "You renamed loop inconsistency with octonions."

**Falsification path.** Construct synthetic evidence networks where the
associator detects a design-sensitive inconsistency missed by a standard local
test, and networks where it collapses to a reparameterization and loses.

**First executable bridge.** A Python or Sounio synthetic NMA generator with
known ground truth, standard inconsistency metrics, and an associator score. The
Garden should forbid paper claims until this benchmark exists.

**Evidence state.** `Garden` to `Hypothesis`. A quick literature search did not
surface an obvious "octonion associator NMA" predecessor, but that is not a
priority proof.

### 4. Non-Associative Meaning

**Bold thesis.** O-SSM and conversational O-SSM are strongest when they stop
saying "octonions are interesting embeddings" and instead say: "some meanings
are parenthesization-sensitive, and models that erase parenthesization erase the
phenomenon."

**Prior-art gravity.** Mamba and modern SSMs are built around efficient sequence
processing; octonion-valued neural networks already exist; conversation
modeling has many non-algebraic traditions.

**Hostile reviewer.** "This is a decorative hypercomplex latent space."

**Falsification path.** Build paired tasks where the token multiset and pairwise
relations are identical, but parenthesization differs; require the model to
distinguish the two. If octonion associator magnitude does not track the known
parenthesization label, the thesis weakens.

**First executable bridge.** A synthetic conversation benchmark with consistent,
contradictory, persona-switch, and repair trajectories. Score whether associator
norm predicts annotated rupture better than a same-parameter associative
baseline.

**Evidence state.** `Hypothesis`. Repo-local demos exist nearby, but the
conversation claim needs a benchmark before it becomes external-facing.

### 5. Clinical Temporality Without Clinical Claims

**Bold thesis.** The clinical-facing idea is not a prescription algorithm. It is
a formal language for trajectories where patient state, drug kinetics,
measurement uncertainty, and narrative time are all first-class and cannot be
collapsed into a single point estimate without leaving a trace.

**Prior-art gravity.** PBPK, clinical pharmacometrics, measurement uncertainty,
and psychiatry time-perception research are all established fields.

**Hostile reviewer.** "This is poetic clinical decision support without clinical
validation."

**Falsification path.** Keep the first study non-clinical or retrospective:
choose a public dataset or simulated patient trajectory, pre-register the
temporal features, and test whether the proposed representation predicts a
held-out trajectory property better than ordinary covariates.

**First executable bridge.** A Garden-to-protocol handoff that forbids treatment
recommendations, names a single retrospective endpoint, and routes any clinical
draft through the offload policy before promotion.

**Evidence state.** `Garden`. It can become `Hypothesis` only after the endpoint,
dataset, and no-advice boundary are explicit.

### 6. GPU Epistemic Execution

**Bold thesis.** GPU kernels should not merely accelerate scientific computing;
they should execute measurement-aware programs where uncertainty budgets,
provenance, and acceptance gates survive device lowering.

**Prior-art gravity.** GPU compilers and adaptive dispatch are mature. GUM and
Monte Carlo UQ are mature. The novelty pressure is their fused execution
contract, not any one component.

**Hostile reviewer.** "This is a GPU benchmark with extra metadata."

**Falsification path.** Show that uncertainty metadata changes the chosen path
or acceptance decision, not only the reported output. Then compare against a
metadata-erasing kernel that produces the same point estimate but fails the
budget.

**First executable bridge.** Turn the Phase Y PBPK/GUM gate into a paired
point-estimate-vs-budget witness with exact digest checks and a small ISO-style
budget artifact.

**Evidence state.** `Executable` for the gate shape; `Hypothesis` for the
general compiler research claim.

## Research Method

For future deep-research passes, use this sequence:

1. Start from a Garden phrase, not a venue.
2. Find the nearest mature prior art that could kill the idea.
3. State the strongest hostile reviewer objection in one sentence.
4. Design the smallest falsification artifact.
5. Only then decide whether the idea belongs in the Garden, a prototype, a
   paper, a dissertation chapter, or nowhere.

## What This Is Not

- Not a priority claim.
- Not a literature review.
- Not medical guidance.
- Not a proof of novelty.
- Not a public-facing research statement.
- Not permission to merge Garden language into papers without external review.

## Next Executable Bridge

Build a small `docs/internal/garden/index.md` that tracks seeds by evidence
state, nearest artifact, external anchors, and promotion blockers. The index
should make it easy to ask: "Which butterfly is ready to become executable?"

The first candidate to promote is **Metrological Compilation**, because it has a
clean hostile reviewer test: one small program where uncertainty collapse is
allowed, rejected, or effect-routed across competing systems.
