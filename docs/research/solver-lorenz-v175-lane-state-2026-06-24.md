<!-- docs:meta
topic_id: repo.docs.research.solver-lorenz-v175-lane-state-2026-06-24
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.solver-lorenz-v175-lane-state-2026-06-24
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Solver-Lorenz v175 — Lane State (consolidation)

Date: 2026-06-24
Branch: `codex/solver-lorenz-v175` · Worktree: `/tmp/sounio-solver-lorenz-v175`
Owner: opencode (glm-5.2) — handed off from prior agent

This is the consolidation/state note for the `solver-lorenz-v175` lane. It is a
map for parallel agents working on disjoint spines, not a public claim.

## Purpose + hard boundary

The lane builds a **local solver/proof-checker router** as layered readiness/
receipt "bridges": each layer binds already-reviewed anchors to solver metadata
via a weighted-sum fingerprint (mod `1_000_000_000`) and exposes a local
receipt. The boundary is **strict and programmatic** — every layer keeps:

- `native_i256_evidence_mask = 0`
- `imported_runtime_evidence_mask = 0`
- `public_claim_mask = 0`
- `formal_theorem_ready = 0`

This is **not** portfolio/global wiring, not a cryptographic proof, not a Lorenz
integrator theorem, not finite-cover/boundary-gluing/flowpipe certification,
not native i256, and not imported/native runtime evidence.

## Open blocker

`BLK-20260623-madaros-mm-seed-segfault` — the Madaros imported/native lowering
segfaults at `lower_array: seed_begin` for ANY multimodule program. Effect on
this lane: every `_imported.sio` smoke is **frontend/typecheck evidence only**
(a registered known-failure, exit 139). `_tiny.sio` self-contained tests run
natively (green). The blocker is owned by the madaros native-lowering lane.

## Anatomy of a layer (the bridge pattern)

Each layer = 4 artifacts, all mandatory:

1. `stdlib/.../<name>.sio` — module with:
   - a `..._mask` fn (bit-accumulation, exact-match bits),
   - a weighted-sum fingerprint fn (`with Div`, `% limb_base`, exact-match guards returning `0 - 1`),
   - a `..._receipt_check` fn returning a fixed artifact fp,
   - (optional) a `..._audit_fingerprint` fn.
2. `tests/run-pass/<name>_tiny.sio` — `//@ run-pass`, self-contained, asserts all
   constants + recomputes the fingerprint; returns `0`.
3. `tests/run-pass/<name>_imported.sio` — `//@ run-pass` + `//@ known-failure: ...
   exits 139 at runtime`, `use`s the module(s), asserts returns + negatives.
4. `docs/research/<name>.md` — formula, anchors, lineage, boundary, claim boundary.

Idioms: `var mask = 0` / `mask = mask + N`; reject with `return 0 - 1`; constants
are decimal-limb receipt ids; `checker_family_id = 73`, `checker_kind_id = 118`,
`target_integer_width = 256`, `limb_base = 1_000_000_000`.

Mandatory per layer: `bin/llm-offload -t math-review -p xai` (task-style; usually
`NO MATHEMATICAL CONTENT`) + a focused `bin/llm-offload --raw <prompt> xai` that
VERIFIES the arithmetic + zero-masks/no-overclaim. Log in `.claude/llm_offload_log.md`.

## Verified status lineage — trajectory2 bounded bridge chain

| Status | Layer | Key fp |
|---|---|---|
| 117 | replay-verifier receipt (certificate bridge predecessor) | — |
| 118 | `lorenz_i256_trajectory2_certificate_bounded_bridge` | cert artifact `918274650` |
| 119 | `lorenz_i256_trajectory2_manifest_bounded_bridge` | manifest_fp `294601254`, artifact `846135279` |
| 120 | `lorenz_i256_trajectory2_acceptance_bounded_bridge` | acceptance_mask `31`, readiness_fp `9371853` |
| 121 | `lorenz_i256_trajectory2_readiness_dispatch_envelope` | dispatch_mask `63`, envelope_fp `9387383`, artifact `682914503` |

Lineage chain of readiness receipts: `9371853 -> 9387383`.

## Spine inventory (not all individually re-verified; structural)

- **trajectory2 bounded bridge** — 117→121 (verified above).
- **child1–child4 cover** — per-child: `validation_core`, `discharge_preflight`,
  `obligation_seed`, `local_flowpipe_preflight`; child4 adds boundary-face
  preflights (face0–4), boundary-gluing, finite-cover-promotion-guard,
  replay-manifest/verifier, solver-profile bridge. Statuses ~80–90.
- **microkernels** — `sat_rup_microkernel`, `smt_farkas_microkernel`,
  `pb_highwidth_microkernel` (stdlib/theorem) + tiny/imported tests.
- **solver/proof shared** — `solver_proof_profile` (status 88,
  accepted_profile_mask 15), `proof_checker_family_matrix`,
  `proof_checker_microkernel_suite`.
- **safety/ABI** — `runtime_abi_gate_blocker`, `private_evidence_envelope`,
  `imported_runtime_lift_contract`, `kernel_replay_evidence_router`.
- **scope** — `erdos_scope`, `div_witness`, `portfolio` (stub).

## Done vs not-done

Done: ~50 receipt/readiness layers across the spines, each with green tiny +
frontend-only imported smoke + logged offload.
Not done (gated, NOT lane-owned to fix): imported/native runtime evidence
(BLK-20260623), native i256, formal theorem proofs, finite-cover/boundary-gluing/
global-flowpipe certification, public/global portfolio promotion.

## Recipe: add a layer (for parallel agents)

1. Read the predecessor module in this worktree to extract its anchor fps.
2. Pick a new, disjoint module name; choose distinct receipt fps; document them.
3. Author module + `_tiny` + `_imported` + research note per the anatomy above.
4. Gates: `bin/souc check <module>`; `bin/souc check <imported>`; `bin/souc run
   <tiny>` (exit 0); `scripts/run_sio_test_suite.sh <filter>` (Pass 1, Known 1).
5. Offload: task-style math-review + focused raw; both logged. Boundary masks zero.
6. Do NOT `git add`/`git commit` — the lane owner integrates.
