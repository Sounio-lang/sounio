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
- [Getting Started](guide/getting-started.md) — **canonical entry point**
- [Minimum Viable Sounio](guide/MINIMUM_VIABLE_SOUNIO.md) — conservative "what works today" contract
- [Tutorial](guide/tutorial.md) — step-by-step learning guide
- [Cookbook](COOKBOOK.md) — task-oriented recipes
- [LLM Programming Guide](guide/LLM_PROGRAMMING_GUIDE.md) — definitive syntax for LLMs
- [Gotchas](guide/SOUNIO_GOTCHAS.md) — common mistakes and anti-patterns
- [Migration Guide](MIGRATION_GUIDE.md) — upgrading between versions
- [Installation Guide](../INSTALL.md)

### Reference
- [Standard Library Reference](stdlib/STDLIB_REFERENCE.md)
- [Standard Library Organization](stdlib/STDLIB_MODULE_ORGANIZATION.md)
- [Standard Library API Reference](stdlib/STDLIB_API_REFERENCE.md)
- [Knowledge Reference](reference/KNOWLEDGE_REFERENCE.md)
- [Migration Guide](MIGRATION_GUIDE.md)

## For Contributors

Start with these current-state maps:

- [Codebase Overview](codebase_overview.md)
- [Compiler Architecture Overview](compiler/COMPILER_ARCHITECTURE_OVERVIEW.md)
- [Self-Hosted Compiler](internal/implementation/SELF_HOSTED_COMPILER.md)
- [Tooling Summary](internal/implementation/TOOLING_SUMMARY.md)
- [Developer Workflow](contributor-guide/DEVELOPER_WORKFLOW.md)
- [Conventions](CONVENTIONS.md)

### Compiler internals and design notes
- [Technical Report](compiler/TECHNICAL_REPORT.md)
- [Effect Dispatch](compiler/EFFECT_DISPATCH_INTEGRATION.md)
- [Epistemic Backend](compiler/EPISTEMIC_BACKEND_GUIDE.md)
- [GPU Kernels](compiler/GPU_KERNELS.md)
- [Known Limitations](compiler/KNOWN_LIMITATIONS.md)
- [LLVM Codegen](architecture/LLVM_CODEGEN.md)
- [GPU Runtime](features/GPU_RUNTIME.md)
- [Async Runtime](architecture/ASYNC_RUNTIME.md)
- [Roadmap](architecture/COMPILER_ROADMAP.md)
- [Paper Artifact Packaging Spec](internal/implementation/PAPER_ARTIFACT_PACKAGING_SPEC.md)

### Internal (process artifacts, not user-facing)
Sprint reports, agent handoff logs, and implementation status docs live in [docs/internal/](internal/). They are not listed here.

### Governance
- [Docs Conventions](governance/DOCS_CONVENTIONS.md)

## Specification

- [Language Specification](spec/LANGUAGE_SPECIFICATION.md)

## Important reading rule

Some historical architecture reports still describe earlier Rust-centric layouts or experimental backend states. When a deep report disagrees with the current codebase:

- trust the checked artifact for user-facing capability claims
- trust `self-hosted/` for the current implementation map
- trust committed gate artifacts for reliability claims
