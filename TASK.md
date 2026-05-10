# Lane 8b — Multi-Drug Confidence Aggregation (dissertation contribution #2 extension)

**Owner:** Codex (offload). Reviewer: Claude B.
**Branch:** `coord/lane-8b-multi-drug` off `origin/main` (91d48adb).
**Worktree:** `/workspace/sounio-lane-8b-multi-drug`.
**Companion contract:** `.claude/PARALLEL_BLOCKER_CONTRACT.md`.

## Why

Phase J (`scripts/ci/kretikos_kaxi_phase_j_gate.sh`, dissertation contribution #2, commit `cbe6716e`) gates **per-kernel** confidence with `--min-conf`. The dissertation needs a **study-wide** attestation across multiple drugs (rapamycin + haloperidol + a third). Without an aggregation rule, three per-drug confidences (e.g. 612, 587, 503) cannot be combined into a single team-confidence claim a regulator would accept.

This lane defines **three aggregation rules** as compile-time selectable strategies:
- `worst_case` — min over drugs (most conservative; sufficient for go/no-go)
- `rss` — root-sum-square of complements: 1000 − sqrt(Σ (1000−cᵢ)²) (standard ISO)
- `cov_weighted` — covariance-aware: 1000 − sqrt(eᵀ Σ e) where eᵢ = 1000−cᵢ and Σ is the inter-drug correlation matrix (off-diagonal user-supplied; diagonal=1)

Each rule emits a PTX kernel that takes N drug confidences as inputs and produces one aggregate confidence the existing `--min-conf` flag can gate against.

## CLAIM (announce in `artifacts/omega/agent_handoff.log.md` before edit)

NEW files only:

- `stdlib/darwin_pbpk/aggregate_confidence.sio` — three pure-Sounio aggregator functions returning `Knowledge<f64>` aggregate.
- `scripts/ci/kretikos_kaxi_phase_j_aggregate_gate.sh` — new gate parallel to phase_j_gate.sh.
- `tests/golden/multi_drug_conf/worst_case.ptx` (NEW dir; **NOT under `tests/golden/kaxi_ptx/`** — that path is owned by Lane 1).
- `tests/golden/multi_drug_conf/rss.ptx`
- `tests/golden/multi_drug_conf/cov_weighted.ptx`
- `tests/run-pass/multi_drug_aggregate_test.sio`

**Disjoint check:**
- Lane 1 owns `tests/golden/kaxi_ptx/**` — we use `tests/golden/multi_drug_conf/**` instead.
- Lane 2 owns `dissertation_pbpk_suite_gate.sh` and `validation/**`, `release/**` — none touched.
- Lane 4 (nv2-hardening) does not own these files.
- Read-only consult: `scripts/ci/kretikos_kaxi_phase_j_gate.sh`, `tests/golden/kaxi_ptx/f64_epistemic_gate/conf_*.ptx` — pattern reference only.

## Acceptance gate

`bash scripts/ci/kretikos_kaxi_phase_j_aggregate_gate.sh` rc=0. Must verify:

1. `bin/souc check stdlib/darwin_pbpk/aggregate_confidence.sio` rc=0.
2. `bin/souc compile tests/run-pass/multi_drug_aggregate_test.sio -o /tmp/agg && /tmp/agg` final line `PASS multi_drug_aggregate`.
3. For inputs `(c1=612, c2=587, c3=503)` the aggregators produce **deterministic** outputs (encode the expected integers in the test):
   - worst_case → 503
   - rss → integer round of (1000 − sqrt(388² + 413² + 497²))
   - cov_weighted with Σ = I → equals `rss` exactly (sanity check)
4. `--min-conf 600` rejects worst_case (503<600) and rss output, while `--min-conf 400` passes both. Encode this in the gate script.
5. Three golden PTX files match bytewise (`diff -q`) the kernels emitted by `bin/kretikos kaxi-emit` against the three aggregator functions.

## Sounio dialect pins

Read first: `docs/guide/SOUNIO_QUICK_START.md`, `docs/guide/SOUNIO_GOTCHAS.md`, `sounio_llm_training.md`. Same five-mistake checklist as Lane 8a (no semicolons, fixed arrays, `&!`, `effect io`, no impl/derives at the level used by lean_single).

For the aggregator return type use `Knowledge<f64>` (existing in stdlib) — read confidence integer 0..1000 from `.confidence` field.

## Reuse (read-only)

- `scripts/ci/kretikos_kaxi_phase_j_gate.sh` — gate skeleton: how it invokes `bin/kretikos`, where it pulls golden, how `--min-conf` is asserted.
- `tests/golden/kaxi_ptx/f64_epistemic_gate/conf_pass_min500.ptx` — pattern.
- `stdlib/darwin_pbpk/bbb/bbb_gum.sio::iso_budget_row` — confidence accumulation pattern.
- `bin/kretikos` — `kaxi-emit` subcommand for PTX golden generation.

## Procedure

1. CLAIM record (lane=8b) in `artifacts/omega/agent_handoff.log.md`.
2. Implement aggregator (3 fns, ~80 LoC). Unit test inline.
3. Generate golden PTX via `bin/kretikos kaxi-emit` against each aggregator. Commit goldens.
4. Write gate script that re-emits goldens and `diff -q`s them; runs the e2e test.
5. Commit prefix `[lane-8b]`. PR title `[lane-8b] multi-drug confidence aggregation (worst-case / RSS / cov-weighted)`.

## Blocker shape

`BLK-20260510-lane8b-<slug>` per the contract. Most likely class if it surfaces: `compiler-semantics` (PTX divergence) or `evidence-gap` (golden capture flake).
