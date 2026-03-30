<!-- docs:meta
topic_id: repo.docs.decisions.adr-003-wrapper-provenance-preserved
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.decisions.adr-003-wrapper-provenance-preserved
-->

# ADR-003: Wrapper Provenance Preserved

**Status**: accepted
**Date**: 2026-03-30

## Context

The rebuilt direct-driver ontology checker path currently collapses all fixtures
(good and bad) to the same coarse result: witness=0, verdict=ok. This means the
direct path cannot yet distinguish a valid program from an invalid one on the
tiny truth frontier. The wrapper path — which combines rebuilt direct output with
fallback compile evidence — preserves provenance and converts
rebuilt/fallback disagreement into UNKNOWN rather than false-OK.

The tiny truth frontier (5 fixtures) shows: 2 good files provisionally trusted
via rebuilt_direct, 3 bad files still collapsed to wrong verdict. The wrapper
catches this disagreement.

## Decision

Wrapper provenance stays operational until direct-driver truth is restored on
the target surface.

- Wrapper is NOT a temporary hack to be removed ASAP. It is the
  provenance-preserving layer that prevents false confidence.
- Direct-driver semantic work is paused until execution truth (ADR-002 layer 3)
  is stable on the target surface.
- Removing the wrapper requires: direct driver correctly rejects all bad
  fixtures in the tiny frontier AND correctly accepts all good fixtures,
  without fallback assistance.

## Consequences

- No pressure to "simplify" by removing the wrapper prematurely.
- Gate scripts must test both paths and flag disagreement.
- The wrapper's UNKNOWN verdict is strictly more honest than the direct driver's
  false-OK — prefer honesty over apparent simplicity.
- Removing the wrapper becomes a gate-gated milestone, not a cleanup task.

## Grounded in

- Truth frontier matrix: `docs/architecture/truth-frontier.md`
- 3/5 bad fixtures: ontology_subclass_reject, ontology_type_mismatch,
  acquisition_reason_requires_plan — all false-OK on direct path
- Wrapper disagreement detection: operational in current validation scripts
