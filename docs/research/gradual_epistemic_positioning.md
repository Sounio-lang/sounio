<!-- docs:meta
topic_id: repo.docs.research.gradual-epistemic-positioning
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.gradual-epistemic-positioning
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Gradual Epistemic Compiler: Research Positioning Memo

This memo sets the conservative external positioning for Sounio's epistemic language/compiler work.
It is intentionally narrower than the manifesto and stricter than the front-door README.
The target reader is a PL researcher who will ask two questions first:

1. What is actually new here?
2. Which nearby literatures already cover parts of the idea?

## Core Claim

**One-sentence claim**

Sounio is a compiler-centric language design that makes knowledge, confidence, provenance, observation boundaries, and measurement uncertainty explicit language semantics, while allowing partial adoption through gradual confidence and optional refinement-backed checking.

**Abstract-sized claim**

Sounio should be positioned not as the first language to handle uncertainty, provenance, or gradual typing in isolation, but as a synthesis of those lines of work in a single compiler-oriented design. Gradual typing contributes the idea that partially known programs should remain admissible while retaining a path toward stronger guarantees; refinement and liquid-type work contributes proof-oriented elimination of checks; provenance systems contribute typed treatment of derivation metadata; probabilistic and uncertain-data systems contribute explicit reasoning about uncertainty. Sounio's distinct claim is that these concerns are unified at the language/compiler boundary: values can carry confidence and provenance, observation is an explicit semantic event, uncertainty propagates through ordinary code, and the compiler itself can use those signals to decide when direct code generation is justified and when guarded execution or additional proof is still required.

## Nearest-Neighbor Matrix

| Line of work | Central problem | Primary semantic unit | What Sounio reuses | What Sounio adds |
| --- | --- | --- | --- | --- |
| Gradual typing | Admitting partially typed programs while preserving sound interaction with more precise regions | Dynamic type, casts, consistency, precision | Partial adoption, staged precision, guarded boundaries | Replaces pure known/unknown staging with graded confidence and compiler-facing admissibility gates |
| Refinement / Liquid types | Proving value-level invariants and eliminating dynamic checks | Predicates over base types plus SMT-backed checking | Static discharge of bounds and confidence obligations | Uses refinements as an auxiliary proof engine, not as the main carrier of epistemic meaning |
| Language-integrated provenance | Tracking origin and derivation of data with language-level guarantees | Provenance metadata embedded in query/language terms | Typed derivation and lineage as first-class metadata | Generalizes provenance beyond query translation into ordinary compiled values plus observation/confidence semantics |
| Probabilistic / uncertain programming | Modeling distributions, inference, and uncertain values in executable programs | Random variables, distributions, inference controllers, uncertain values | Explicit uncertainty as part of the programming model | Focuses on typed knowledge state, admissibility, and observation boundaries rather than inference as the primary runtime model |

Representative external anchors:

- Gradual typing and cast semantics: Siek and Taha (2006), Siek et al. (2015), Siek and Chen (2021), Ye and Oliveira (2024)
- Refinement / Liquid types: Rondon, Kawaguchi, and Jhala (2008)
- Provenance as a language feature: Fehrenbach and Cheney (2018)
- Probabilistic programming overview: van de Meent, Paige, Yang, and Wood (2021)

## Primacy Program

If Sounio wants to pursue primacy, it should do so through a staged claim ladder rather than by pushing the paper surface past the available proof.

**Manifesto claim**

Sounio aims to establish the Gradual Epistemic Compiler as a new design center for programming languages.

**Internal research hypothesis**

Sounio may be the first system at the intersection of:

- gradual admissibility for partially known programs
- typed observation boundaries
- provenance-bearing values
- measurement-aware uncertainty propagation
- compiler-visible confidence that affects lowering decisions

**Paper-safe claim**

Sounio is a compiler-centric synthesis of graduality, epistemic values, provenance, and uncertainty-aware checking.

**Upgrade conditions for a future primacy claim**

1. Define the necessary criteria for what counts as a Gradual Epistemic Compiler.
2. Survey at least 10--15 serious neighbor systems across the four comparison axes.
3. Show that no surveyed system satisfies the full criterion set simultaneously.
4. Tie each criterion to concrete repo evidence rather than aspiration.
5. Separate implementation evidence from future-theory aspirations in every comparison row.

## What We Are Not Claiming

- We are not claiming that Sounio is the first system with uncertainty-aware values.
- We are not claiming that Sounio is the first system with provenance-aware types or metadata.
- We are not claiming that Sounio replaces the gradual typing literature with a new universal semantics.
- We are not claiming that refinement types are novel in Sounio; they are a supporting mechanism.
- We are not claiming that probabilistic programming and epistemic compilation solve the same problem.
- We are not claiming that the current implementation already proves the full meta-theory suggested by the long-term design.

## Repo-Backed Evidence

The positioning above is only defensible if it is anchored in concrete repo behavior.
The current minimum evidence set is:

1. **Confidence-gated admission**
   - `tests/run-pass/vancomycin_propagation.sio`
   - `tests/compile-fail/med/vancomycin_low_conf_refusal.sio`
   - This is the cleanest evidence that confidence thresholds are treated as admissibility conditions rather than comments or logging.

2. **Observation as a semantic boundary**
   - `tests/run-pass/observe_with_effect.sio`
   - `tests/run-pass/algebra_observe_synthesis.sio`
   - These show that `Unobserved<T>` is not just a wrapper type; crossing an observation boundary requires `with Observe`.

3. **Uncertainty propagation with provenance**
   - `tests/run-pass/med/vancomycin_full_propagation.sio`
   - `docs/research/vancomycin-uncertainty.md`
   - These show propagation and provenance on a domain workload rather than a toy arithmetic example.

## Open Theory vs Current Claim

The following are active theory/program goals and must stay out of the current novelty claim:

- An epistemic analogue of the gradual guarantee stated and proved end-to-end
- A full meta-theory for observation, confidence subsumption, and guarded lowering
- A precise formal account of how refinement proofs interact with epistemic confidence
- A completed bridge from language-level epistemic semantics to all runtime representations used in the compiler and standard library

Those belong to the future-work section or a companion theory paper, not the present external positioning.
