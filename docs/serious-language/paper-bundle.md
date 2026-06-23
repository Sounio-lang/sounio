<!-- docs:meta
topic_id: repo.docs.serious-language.paper-bundle
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.serious-language.paper-bundle
-->

# Serious-Language Paper Bundle

> **Status**: Research readiness | **Operational check**: 2026-05-11 | **Source**: bundle script and readiness ledger

The paper bundle is the proof object for conference conversations. It should make the paper's claims auditable without asking a reviewer to trust repository sprawl.

## Bundle Contents

Each generated bundle should contain:

- `RESULTS.md`: commands, exit codes, and summary.
- `manifest.json`: commit, branch, host, timestamp, selected compiler metadata, binary hashes, tool versions, and claim coverage.
- `logs/`: raw command logs.
- `serious-conformance/`: bounded conformance summary when full mode runs.
- a copy or link to `docs/serious-language/readiness-ledger.md`.
- a link to `docs/serious-language/real-world-defensibility.md`.
- a spec/evidence drift log from `scripts/ci/serious_language_spec_drift_gate.sh`, including the bounded conformance summary used to validate cited conformance cases.
- a copy or link to the paper draft under review.
- offload review logs or pointers in `.claude/llm_offload_log.md` before submission.

## Environment

Smoke mode requires the checked repository artifact, Bash, Git, and Python 3. Full mode may also require the Lean `lake` toolchain and any runtime dependencies used by optional gates.

The bundle records:

- `SOUC_BIN`;
- `SOUNIO_STDLIB_PATH`, pinned to this checkout's `stdlib` unless `SOUNIO_SERIOUS_LANGUAGE_STDLIB_PATH` is set;
- SHA256 and byte size for the wrapper and selected compiler binary;
- `git`, `bash`, `python3`, and `lake` versions;
- relative log paths;
- claim-to-log coverage for the main readiness claims.

Default command:

```bash
bash scripts/paper/build_serious_language_bundle.sh
```

Heavier command:

```bash
SOUNIO_SERIOUS_LANGUAGE_FULL=1 bash scripts/paper/build_serious_language_bundle.sh
```

Full mode includes the bounded serious-language conformance gate, passes the same pinned compiler and stdlib paths into that sub-gate, and writes its summary under `serious-conformance/`.

## Required Claim Checks

Before any paper or slide deck says a claim is stable:

1. The claim appears in `readiness-ledger.md`.
2. Any spec-backed claim appears in `spec-evidence-matrix.v1.tsv` with appropriate status. The readiness ledger governs public wording; the matrix governs spec-area evidence status.
3. Any public PL claim closes through `public-claim-registry.v1.tsv` and any repo doc surface carrying the claim is covered by `doc-claim-surface.v1.tsv`.
4. Any high-value public PL claim has an exact line anchor in `claim-line-annotations.v1.tsv`.
5. The generated bundle includes a command or artifact supporting it.
6. Failures are either fixed or explicitly downgraded in wording.
7. External-facing prose has at least two-provider `bin/llm-offload` review.

## Paper Structure

Recommended PL paper framing:

1. Problem: scientific programs discard uncertainty, provenance, and observation boundaries.
2. Design: Sounio makes epistemic evidence part of the language contract.
3. Implementation: checked self-hosted compiler path plus native execution.
4. Evidence: compiler gates, epistemic/GUM gates, formal surfaces, and artifact bundle.
5. Limits: unfinished general production ecosystem, scaffolded surfaces, research GPU/hypercomplex lanes.

Do not build the paper around broad claims such as "production-ready language" or "complete GPU stack." Build it around the stronger, narrower claim:

> Sounio demonstrates that epistemic/scientific evidence can be made compiler-visible and reproducibly checked in a self-hosted research language.

## Acceptance Criteria

A bundle is conference-ready only when:

- the smoke bundle completes with no required-step failures;
- the spec/evidence drift gate passes;
- any full-lane failures are documented in `RESULTS.md`;
- paper claims are downgraded to match the ledger;
- Lean proof status includes explicit `sorry` and `axiom` accounting;
- clinical, math, and external-facing artifacts have required offload review;
- the final narrative has a visible "what is not claimed" section.

## What This Paper Does Not Claim

Use this template in every external paper or slide deck:

- Sounio is not yet a production-ready general-purpose language.
- The checked Linux x86-64 lane is the default live-demo lane, invoked as `./bin/souc` (official route, Madaros-backed).
- GPU/PTX support is not a broad runtime or performance claim unless a generated bundle includes the exact hardware run.
- Hypercomplex neural-network work is not a complete training framework unless the cited gate proves that specific surface.
- Ontology work does not imply federation-scale ontology support.
- Formal Lean support does not mean every module is axiom-free or sorry-free; the bundle must report exact status.
- The modular compiler tree is not yet the checked binary source of truth; `lean_single.sio` remains part of the current source-of-truth story.

## Current Submission Blockers

These are blockers for a top-tier PL submission, not blockers for an internal or exploratory talk:

- binary provenance needs to be cited from the generated bundle's hashes and self-host gate logs;
- `lean_single.sio` versus modular compiler parity needs a conformance or parity gate before any source-swap claim;
- the language spec needs broader conformance coverage before it is treated as normative;
- full-suite failures must be fixed or downgraded before saying the whole suite is green;
- Lean build status must be generated on a host with `lake` before formal claims are submitted.
