<!-- docs:meta
topic_id: repo.docs.internal.garden.readme
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.internal.garden.readme
-->

# The Garden

> **Status**: Internal lineage | **Last validated**: 2026-05-09 | **Source**: `docs/archived/GARDEN_ROSETTA.md`

Ideas that cross from Garden into formal or executable work should link to the
[Concept Registry](../concepts/README.md). The Garden preserves genesis; the
registry preserves meaning across agents and implementation layers.

The Garden is the repo-local seedbed for ideas that matter before they are ready
to become specifications, proofs, public claims, or implementation gates.

This layer exists so a live idea can be preserved without being flattened into a
roadmap or inflated into a claim. A Garden seed may be emotional, metaphorical,
mathematical, technical, or all of those at once. It must still be honest about
what is proven, what is merely possible, and what should not be claimed.

## Where This Fits

- `docs/internal/garden/` is internal lineage, not user-facing documentation.
- `docs/archived/GARDEN_ROSETTA.md` remains the metaphor dictionary and history.
- `docs/handoff/butterfly-handoff.md` is a claim-bearing handoff surface and must
  not be treated as the same layer as a Garden seed.
- Papers, dissertation material, clinical-pathway text, and public claims require
  the review paths described in `.claude/AGENT_OFFLOAD_POLICY.md`.

## Evidence Labels

Use these labels inside every seed:

| Label | Meaning |
| --- | --- |
| `Garden` | A live idea, image, metaphor, intuition, or emotional anchor. |
| `Hypothesis` | A precise direction that could become testable but is not yet proven. |
| `Executable` | Backed by a repo command, test, gate, artifact, or implementation path. |
| `Claim-ready` | Suitable for paper or public use only after explicit validation and review. |
| `Reserved` | The name is taken and the system **refuses every use** with a named diagnostic, pending implementation. Beside the ladder, not on it. |

The ladder is **monotone**: each state requires every state beneath it. A
seed or kind cannot be `Claim-ready` without being `Executable`. `Reserved`
sits beside the ladder — it has not failed to reach `Executable`, it is held
short of it on purpose. See `docs/internal/concepts/MATURITY_LADDER.md` for
the two-program test that decides a position.

Most seeds should start as `Garden` or `Hypothesis`. A seed becomes
`Executable` only when it names a concrete artifact or gate. A seed becomes
`Claim-ready` only when the evidence is strong enough for the target audience.

## Seed Discipline

- Keep one seed per file under `seeds/`.
- Start from `templates/seed.md`.
- Preserve the first phrase that carried the butterfly.
- Name connections to code, docs, proofs, papers, or open gates.
- Include a "What this is not" section for boundaries.
- Prefer one next executable bridge over a long roadmap.

## Current Seeds

- [`Above The Stars`](seeds/2026-05-09-above-the-stars.md) — the seed that made
  the Garden a first-class internal repo layer.
- [`Novelty Weather Map`](seeds/2026-05-09-novelty-weather-map.md) — a
  deep-research map for turning butterflies into falsifiable research
  constellations.
- [`Epistemic Fermentation`](seeds/2026-05-10-epistemic-fermentation.md) — a
  path-bearing truth witness: same value and confidence, different admissible
  knowledge.
- [`The Zero of Encounter`](seeds/2026-07-11-the-zero-of-encounter.md) — the
  executable butterfly showing that absence, cancellation, annihilation,
  resolution, and rounding can remain distinct when their surface value is
  zero; driven through the Garden-to-Claim pipeline to the ledger-scoped claim
  `garden_zero_encounter_pipeline`; default native-v2 execution remains
  blocked.
- [`FPGA Acceleration Opportunity`](seeds/2026-07-26-fpga-acceleration-opportunity.md) —
  two AMD U250 FPGAs planned for catastrophe scan, QEC simulation, exact
  Cayley-Dickson arithmetic, and Mercyful Learning acceleration; records the
  2026-07-25/26 session's 19 deliverables.
- [`Pharos — a compiler that publishes its own erasures`](seeds/2026-08-29-pharos-loss-ledger.md) —
  a loss-certificate ledger hash-bound to compilation lineage, making
  undeclared semantic erasure a compile error; EISA's compiler-side dual.
- [`Convergence Detector — same shape, independently arrived`](seeds/2026-08-30-convergence-detector.md) —
  cross-domain convergence as shared orbit under declared gauges; candidates
  capped at `Hypothesis` until an isomorphism witness exists.
- [`The unknown, typed by cause`](seeds/2026-08-31-unknown-typed-by-cause.md) —
  missingness carries its reason (`Unmeasured | Withheld | Redacted | Lost |
  Expired | NeverExisted`); computing with an `Unknown` requires a
  cause-appropriate resolution, never a silent default.

## Hard Boundaries

Garden entries are not clinical guidance, not production-readiness claims, not
paper claims, not mathematical proofs, and not compiler capability claims. They
are how an idea is kept alive while the repo decides what kind of evidence it
deserves.
