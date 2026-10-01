# Current checks and research handoff

Worktree: /workspace/.wt/zd-two-mode-main-20260906
Branch: research/zd-two-mode-main-20260906
Base HEAD: 0c6514a0cd7327260b7962016770651c52e090b8
Concept-ID for this lane: SOUNIO-ZD-COUNT-RECONSTRUCTION.
The old worktree has no docs/internal/concepts/SEMANTIC_LANE_CONTRACT.md;
the exact research-only write set was registered through sounio-coord.
The interpretation is principal C(e_W) components only.

## Implemented

- NOTE.md: full count-injectivity argument, executable inverse specification,
  weighted-origin proof, omission examples and sharp scalar witness.
- BIBLIOGRAPHY.md: explicit correspondence to the established components,
  theorem-level comparison, bounded search record and novelty decision.
- decoder.py: exact inverse from counts, with native/auxiliary channel distinction.
- test_decoder.py: inverse rejection and independent native algebra controls.
- REVIEW.md: every external review finding and its adjudication.

No existing Lean source was modified. The 30 current Sounio Lean sources
match the source copies of the previously rebuilt dependency cache; see
[evidence/source-check.json](evidence/source-check.json).
The present run reused those compiled dependencies rather than rebuilding them.

## Executed validation

1. Remote command: python3 research/zd-count-reconstruction/test_decoder.py.
   PASS: 374 catalogue round trips through depth 6; complete small invalid-input
   boxes; exact large-integer examples at depth 64; native X-channel checks.
   Independently computed 62 native graphs via sparse basis multiplication,
   including all W at depths 0, 1 and 2, seven labels at depth 3, and the two
   scalar witnesses at depth 5. See [tests.json](evidence/tests.json).
   The direct construction has no dependency on the graph-count recurrence.

2. Lean 4.33.0 command, executed remotely with the verified existing cache:
       LEAN_PATH=<verified-cache> lean CountChecks.lean
   Exit 0. The three dimension-32 edge/triangle examples were checked against
   actual_native_counts. The five requested inherited theorem closures list
   only propext, Classical.choice and Quot.sound; no sorryAx.
   See [probe](evidence/CountChecks.lean) and [output](evidence/lean-check.log).
   This is a current import/probe check; the full dependency rebuild belongs
   to the earlier residual-search reproduction.

3. Reviewer fan-out: xai/grok-4.6, Qwen 3 235B and Mistral Large.
   Qwen and Mistral returned reviews; xAI produced no response with a 120s
   configured limit. See REVIEW.md and the provider texts in evidence/.
   The final changes after review clarify predecessor depth, divisibility,
   positive-triangle signs and the source's zero-divisor exceptions.
   No mathematical statement changed in adjudication.

An initial Python run caught an incorrect expected scalar literal in the
new test (1,006,560). Direct multiplication and ordinary integer arithmetic
give 1,006,776; the expected value was corrected and the complete check passed.
The note uses the correct value throughout.

## Evidence boundaries

The finite controls validate the new implementation and the examples.
They are not a substitute for the arbitrary-depth proof.
The correspondence to the printed literature is an explicit mathematical
argument from the definitions, not a new Lean formalization.
No exhaustive forward-citation census or publication-priority proof is claimed.
There is no unresolved mathematical counterexample from the checks/reviews.
The unavailable xAI leg remains unavailable, not “approved.”

The source comparison rejects object-level and general recursive-classification
novelty. It retains the weighted two-class bound as a scoped candidate for a
short reconstruction/compression note. No larger search or residual-separation
claim is needed for this result.
