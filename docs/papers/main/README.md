<!-- docs:meta
topic_id: repo.docs.papers.main.readme
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.papers.main.readme
-->

# Sounio Papers

This directory contains both the legacy TeX preprint source and the JOSS submission source.

## Abstract

We present Sounio, a systems programming language with native support for
*epistemic types*—type-level representations of uncertain values that
automatically propagate measurement uncertainty through computations.

## Building

```bash
make        # Build PDF
make arxiv  # Create arXiv submission tarball
make clean  # Remove build artifacts
./reproduce.sh  # Build + benchmark logs + reference checks
```

## Files

- `paper.md` — JOSS paper source
- `paper.bib` — JOSS bibliography
- `sounio-epistemic-types.tex` — TeX preprint source
- `sounio-ieee-cise.tex` — IEEE CiSE manuscript source
- `references.bib` — TeX preprint bibliography
- `Makefile` — TeX build script
- `reproduce.sh` — end-to-end reproducibility script

## Executable Science Evidence Notes

For manuscript claims that reference runnable fMRI/PBPK pipelines, use the
repository gates and artifacts as the source of truth:

```bash
OMEGA_GPU_RUNTIME_GATE_MODE=required bash scripts/omega/omega_gpu_runtime_attest_gate.sh
bash scripts/stdlib_hyper_execution_gate.sh
STDLIB_RUNTIME_REGRESSION_STRICT=1 bash scripts/stdlib_science_pipeline_gate.sh
STDLIB_RUNTIME_REGRESSION_STRICT=1 bash scripts/stdlib_reliability_gate.sh
bash scripts/omega/omega_gpu_runtime_attest_gate.sh
bash scripts/stdlib_hyper_execution_gate.sh
bash scripts/stdlib_science_pipeline_gate.sh
bash scripts/stdlib_reliability_gate.sh
```

Primary evidence artifacts:
- `artifacts/omega/gpu_runtime_attest_gate.v1.json`
- `artifacts/stdlib/stdlib_hyper_execution_status.v1.json`
- `artifacts/stdlib/stdlib_science_pipeline_status.v1.json`
- `artifacts/stdlib/stdlib_reliability_status.v1.json`
- `tests/fixtures/fmri/fixture_manifest.v1.json`
- `tests/fixtures/fmri/pipeline_golden.v1.json`
- `tests/fixtures/hyper/pipeline_golden.v1.json`

Runtime policy note:
- science status JSON includes `runtime_regressions`, `runtime_regression_enforcement`, `runtime_regression_summary`, and `runtime_provenance`
- runtime probe sources are committed in `tests/stdlib/runtime_regression/` for reproducible paper evidence
- local gate mode is soft telemetry for runtime regressions; required CI full gate uses `STDLIB_RUNTIME_REGRESSION_STRICT=1`
- strict mode is fail-closed; runtime probes must pass for strict CI success
- hyper execution lane is fail-closed with no-ignore policy in required hyper tests
- GPU runtime attestation uses `OMEGA_GPU_RUNTIME_GATE_MODE=required` in CI and records pass|fail|not_run with blocker metadata in the artifact
- canonical pinned `souc` version is sourced from `scripts/omega/omega_resolve_souc_bin.sh` unless explicitly overridden via `SOUNIO_SOUC_VERSION`

## Target Venues

### Legacy Preprint
- **Primary**: arXiv cs.PL (Programming Languages)
- **Secondary**: OOPSLA, PLDI, or Software X journal

### Research Track (18-Month Roadmap)

#### Paper 1: Epistemic Types for Scientific Computing
- **Directory:** `epistemic-types/`
- **Target:** PLDI 2027 or ICFP 2027
- **Status:** Formalization complete (Month 1-2 ✓)
- **Contribution:** First type system with GUM-compliant uncertainty propagation

#### Paper 2: Causal Programming with do-Calculus Types
- **Directory:** `causal-types/`
- **Target:** PLDI 2027 or UAI 2027
- **Status:** Planned for Month 7-8
- **Contribution:** Compile-time causal identifiability verification

#### Paper 3: Quaternionic Neural Networks with Epistemic Uncertainty
- **Directory:** `qnn-epistemic/`
- **Target:** NeurIPS 2027 or ICML 2028
- **Status:** Planned for Month 15-16
- **Contribution:** Type-safe epistemic neural networks

## Citation

```bibtex
@article{chiuratto2026sounio,
  author = {Chiuratto Agourakis, Demetrios},
  title = {Sounio: Epistemic Types for Scientific Computing with Native Uncertainty Quantification},
  journal = {arXiv preprint},
  year = {2026},
  doi = {10.5281/zenodo.18404188}
}
```

## License

CC BY 4.0
