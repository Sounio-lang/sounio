<!-- docs:meta
topic_id: repo.docs.research.llm-collaboration-codex-backlog
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.llm-collaboration-codex-backlog
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Codex memo — implementation backlog (tracking)

Source: `docs/research/llm_collaboration_codex_response.md` (questions 1, 10, 11, 12). **Not implemented** — checklist only.

## Recommended order (from memo)

1. Bidirectional late-fusion prototype in plain Sounio (octonion main loop; sedenions guarded only if needed).
2. Associator telemetry and calibration dataset.
3. Quaternion chunk-summary experiment for partial parallelism.
4. Sedenion auxiliary lane with zero-divisor margin guard.

## Q1 — bidirectional O-SSM

- Add `examples/ossm_bidirectional_v0.sio` with separate `A_f`, `A_b`, `B_f`, `B_b`; fusion readout-only in v0.
- Reuse full-BPTT lane from `examples/ossm_fullbp_v2.sio` before new compiler primitives.
- After prototype: consider IR ops such as `oct_mul`, `oct_associator_norm`.

## Q10 — Moufang-aware partial scan

- Add `oct_project_fano_line` (stdlib or intrinsic candidate).
- Add `tests/run-pass/ossm_chunk_summary_quat.sio`.
- Measure endpoint error, associator residual, wall-clock vs sequential rollout.

## Q11 — associator as reliability signal

- Synthetic conversational benchmark: consistent / contradictory / persona-switch / novelty cases.
- Joint logging: `||A_t||`, entropy, epistemic confidence.
- Fit reliability curve before “hallucination detector” wording.

## Q12 — zero-divisor safety (sedenions)

- Add `sed_zero_divisor_margin` (library or intrinsic candidate).
- Tests around sedenion primitive pair structure from existing sedenion work.
- Policy: below threshold — project, clip, or explicit forget-gate; no silent continuation in main recurrent lane.
