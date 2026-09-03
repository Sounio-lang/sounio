<!-- docs:meta
topic_id: website.docs.compiler.scientific-features
authority: dual
audience: contributors
last_validated: 2026-03-07
validated_by: A4
source_of_truth: docs/governance/topic-registry.v1.json#website.docs.compiler.scientific-features
-->

# Sounio Scientific Features Architecture

Scientific computing in Sounio is a system-level story, not a single module. The current picture combines epistemic typing, stdlib science lanes, hyper-execution coverage, hypercomplex math, GPU work, and domain-specific fixtures.

## Strongest current evidence

Committed gate artifacts currently show:

- stdlib science pipeline: `2/2` required lanes passing (`fmri`, `darwin_pbpk`)
- stdlib hyper execution: `7/7` required lanes passing (`nn`, `onn`, `qnn`, `snn`, `spnn`, `quantnn`, `math`)
- stdlib reliability totals: `251 pass / 0 fail / 0 skip / 251 total`
- science runtime regressions are still tracked separately and currently show `0` failures under soft local enforcement

The strongest public science proof points are therefore the passing fixtures and committed artifacts, not the mere presence of an ambitious source tree.

## Source map

Tests and artifacts:

- `tests/stdlib/fmri/`
- `tests/stdlib/darwin_pbpk/`
- `tests/stdlib/nn/`
- `tests/stdlib/onn/`
- `tests/stdlib/qnn/`
- `tests/stdlib/snn/`
- `tests/stdlib/spnn/`
- `tests/stdlib/quantnn/`
- `tests/stdlib/math/`
- `artifacts/stdlib/stdlib_science_pipeline_status.v1.json`
- `artifacts/stdlib/stdlib_hyper_execution_status.v1.json`
- `artifacts/stdlib/stdlib_reliability_status.v1.json`

Implementation-oriented directories:

- `self-hosted/hypercomplex/`
- `self-hosted/gpu/`
- `self-hosted/tensor/`
- `self-hosted/distributed/`

## Reproducing the gate-backed picture

```bash
bash scripts/stdlib_hyper_execution_gate.sh
STDLIB_RUNTIME_REGRESSION_STRICT=1 bash scripts/stdlib_science_pipeline_gate.sh
bash scripts/stdlib_reliability_gate.sh
```

## Documentation rules

- Lead with passing lanes and validated fixtures.
- Treat disabled, stubbed, or artifact-disabled scientific subsystems as source inventory or roadmap context.
- When discussing advanced GPU or backend-assisted scientific paths, distinguish clearly between source-tree implementation work and checked-artifact behavior.
