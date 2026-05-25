<!-- docs:meta
topic_id: repo.frontdoor.docs-index
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.frontdoor.docs-index
-->

# Sounio Documentation

Welcome to the Sounio documentation index. Public website summaries live under `website/src/content`, while repo-native deep dives, evidence packs, and implementation notes live here under `docs/`, `paper/`, `tests/`, and `examples/`.

If you only read one current-state document first, read `guide/MINIMUM_VIABLE_SOUNIO.md`. It is the conservative contract for what is actually validated by repository gates and committed artifacts.

## For Users

### Getting Started
- [Getting Started](guide/getting-started.md)
- [Installation Guide](../INSTALL.md)
- [Tutorial](guide/tutorial.md)
- [Minimum Viable Sounio](guide/MINIMUM_VIABLE_SOUNIO.md)
- [Programming Guide](guide/programming.md)

### Reference
- [Standard Library Reference](stdlib/STDLIB_REFERENCE.md)
- [Standard Library Organization](stdlib/STDLIB_MODULE_ORGANIZATION.md)
- [Knowledge Reference](reference/KNOWLEDGE_REFERENCE.md)

## For Contributors

Start with these current-state maps before diving into older deep reports:

- [Codebase Overview](codebase_overview.md)
- [Compiler Architecture Overview](compiler/COMPILER_ARCHITECTURE_OVERVIEW.md)
- [Self-Hosted Compiler](implementation/SELF_HOSTED_COMPILER.md)
- [Tooling Summary](implementation/TOOLING_SUMMARY.md)
- [Developer Workflow](contributor-guide/DEVELOPER_WORKFLOW.md)
- [Foundry/Slurm Handoff](ops/foundry_slurm_handoff.md)

### Additional internals and design notes
- [Technical Report](compiler/TECHNICAL_REPORT.md)
- [Effect Dispatch](compiler/EFFECT_DISPATCH_INTEGRATION.md)
- [Epistemic Backend](compiler/EPISTEMIC_BACKEND_GUIDE.md)
- [GPU Kernels](compiler/GPU_KERNELS.md)
- [Known Limitations](compiler/KNOWN_LIMITATIONS.md)
- [LLVM Codegen](architecture/LLVM_CODEGEN.md)
- [GPU Runtime](features/GPU_RUNTIME.md)
- [Async Runtime](architecture/ASYNC_RUNTIME.md)
- [Roadmap](archived/COMPILER_ROADMAP.md)
- [Paper Artifact Packaging Spec](implementation/PAPER_ARTIFACT_PACKAGING_SPEC.md)
- [Website Design System & UI Components](../website/README.md)

### Governance
- [Docs Authority Matrix](governance/DOCS_AUTHORITY_MATRIX.md)
- [Docs Acceptance Report](governance/DOCS_ACCEPTANCE_REPORT.md)

## Specification

- [Language Specification](spec/LANGUAGE_SPECIFICATION.md)

## Important reading rule

Some historical architecture reports still describe earlier Rust-centric layouts or experimental backend states. When a deep report disagrees with the current codebase:

- trust the checked artifact for user-facing capability claims
- trust `self-hosted/` for the current implementation map
- trust committed gate artifacts for reliability claims
