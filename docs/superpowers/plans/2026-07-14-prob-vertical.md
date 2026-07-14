<!-- docs:meta
topic_id: repo.docs.superpowers.plans.2026-07-14-prob-vertical
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.superpowers.plans.2026-07-14-prob-vertical
-->

# Prob (distributions) Hardening — Implementation Plan

> Execute task-by-task; compile-and-run is the gate. **Build with `SOUNIO_SOUC_ENGINE=lean_single`** (Madaros native can't link the ~210-fn graph — see docs/audit/MADAROS_NATIVE_MULTIMODULE_SCALE_2026-07-14.md); `chmod +x` the output.

**Goal:** Make `stdlib/prob/distributions.sio` run-proven + documented (import idiom + lean_single build). No compiler changes.

**Ground rules:** never touch `self-hosted/`/`bootstrap/`; no change to existing `pub` signatures; additive; EN-UK; atomic commits; no AI attribution.

Spec: `docs/superpowers/specs/2026-07-14-prob-vertical-design.md`.

## Task 1 — Header note + escalate native-scale bug
- [ ] Add usage note to `stdlib/prob/distributions.sio` header: `use prob::distributions::*`; native compile needs `SOUNIO_SOUC_ENGINE=lean_single` + `chmod +x` (Madaros native scale limit, ref audit); print floats with `print`/`println` not `print_f64`.
- [ ] Commit the audit `docs/audit/MADAROS_NATIVE_MULTIMODULE_SCALE_2026-07-14.md` (already written) with the header change.
- [ ] `souc check stdlib/prob/distributions.sio` green.

## Task 2 — Run-proof driver
- [ ] `tests/stdlib/prob/test_prob_stdlib.sio` (inline in main, wildcard import). Assert: normal pdf(0)=0.398942, cdf(0)=0.5, cdf(1.96)≈0.975; exponential_mean(2)=0.5, exponential_cdf(ln2,1)=0.5; uniform_mean(0,10)=5, uniform_quantile(0.5,0,10)=5; poisson_variance(3)=3. Then `PROB_STDLIB_OK`.
- [ ] `SOUNIO_SOUC_ENGINE=lean_single souc compile … -o out && chmod +x out && ./out` → `PROB_STDLIB_OK`, exit 0. No tolerance-retrofit. Commit.

## Task 3 — Consumer example
- [ ] `examples/prob/distribution_report.sio` — print a small distribution report (normal cdf at a few points, exponential mean, etc.). Compile (lean_single) + run, exit 0. Commit.

## Task 4 — Gate
- [ ] `scripts/prob_gate.sh` — `souc check` distributions.sio; then with `SOUNIO_SOUC_ENGINE=lean_single`, compile+chmod+run driver (grep `PROB_STDLIB_OK`) and example; end `PROB_GATE_OK`. Run it. Commit.

## Task 5 — Math-review + PR
- [ ] `bin/llm-offload -t math-review -p xai` on the distribution identities; append to `.claude/llm_offload_log.md`.
- [ ] `node scripts/docs/sync_governance_metadata.mjs`; commit governance + docs.
- [ ] File the Madaros native-scale issue to CODEX intake (GitHub issue).
- [ ] Push; PR to `main`; ensure `Contracts`/`CI Decision` green; merge.
