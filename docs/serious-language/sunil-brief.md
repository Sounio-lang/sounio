<!-- docs:meta
topic_id: repo.docs.serious-language.sunil-brief
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.serious-language.sunil-brief
-->

# Sunil Brief

> **Status**: Research readiness | **Operational check**: 2026-05-11 | **Source**: readiness ledger and compiler maturity snapshot

Use this as the short frame for a senior technical conversation.

## One-Minute Positioning

Sounio is a research programming language for epistemic and scientific computing. Its distinctive idea is that uncertainty, provenance, observation, and scientific evidence should be visible to the compiler instead of living only in libraries or comments.

The serious part is not that every subsystem is finished. The serious part is that Sounio already has a checked compiler entry point, self-hosting evidence, native compile/run behavior, epistemic/GUM gates, formal artifacts, and a repo culture that can tie claims to commands.

## What To Show

Show this path first:

```bash
./bin/souc --version
./bin/souc info
./bin/souc check examples/hello.sio
./bin/souc run examples/hello.sio
```

Then show:

- one epistemic/GUM example;
- one compile-fail/effect or observation-boundary example;
- the readiness ledger;
- a generated paper bundle with logs.

## What To Say Clearly

- Sounio is not a finished production language.
- Linux x86-64 is the primary live-demo lane.
- GPU, hypercomplex, ontology, and clinical surfaces are research or validated-research lanes with explicit evidence boundaries.
- Some Lean modules are no-sorry/no-axiom; others retain `sorry` or explicit axioms and must be reported honestly.
- The official public entry point remains `./bin/souc`, which routes to Madaros; the checked binary still depends on `lean_single.sio`, and the modular compiler tree is not yet the binary source of truth.

## What To Ask Sunil For

- Does the PL contribution read as a language design contribution or as a domain-science tool?
- Which claim would a skeptical POPL/ICFP reviewer attack first?
- Is the paper stronger as "epistemic types and effects" or as "evidence-bearing compiler architecture"?
- Which formal result must be tightened before submission?
- Which demo should be cut because it creates credibility risk?

## Sunil Feedback Record

Fill this immediately after the conversation:

| Field | Notes |
|---|---|
| Date discussed | TBD |
| PL contribution framing | TBD |
| Most attackable claim | TBD |
| Stronger thesis | TBD |
| Formal result to tighten | TBD |
| Demo or claim to cut | TBD |
| Action items | TBD |

## Best Honest Thesis

Sounio's near-term conference thesis should be:

> A self-hosted research language can make epistemic evidence, observation boundaries, and scientific uncertainty part of the compiler contract; Sounio is an existence proof with a validated core and an explicit maturity ledger.
