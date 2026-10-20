<!-- docs:meta
topic_id: repo.frontdoor.docs-index
authority: repo_only
audience: users
last_validated: unvalidated
validated_by: none
source_of_truth: docs/governance/topic-registry.v1.json#repo.frontdoor.docs-index
-->

# Sounio Documentation

**Ελληνικά:** [εγκατάσταση, εφαρμογές στη χημεία και την τεχνητή νοημοσύνη,
ανάπτυξη, LLM και συνεισφορά](el/README.md).

Welcome to the Sounio documentation index. Public website summaries live under `website/src/content`, while repo-native deep dives, evidence packs, and implementation notes live here under `docs/`, `paper/`, `tests/`, and `examples/`.

For a first program, start with [`guide/getting-started.md`](guide/getting-started.md), then use the [`tutorial`](guide/tutorial.md) for the core syntax. The longer [`programming guide`](guide/programming.md) is organized by feature, but explicitly marks sections that are not part of the checked user surface. For current limits, consult [`compiler/KNOWN_LIMITATIONS.md`](compiler/KNOWN_LIMITATIONS.md) and the linked tests; [`TOUR.md`](../TOUR.md) is a dated measurement snapshot, not a current release claim.

## Recommended reading path

| Goal | Start here |
|---|---|
| Install and write a first program | [Getting started](guide/getting-started.md) |
| Learn the small-program syntax | [Tutorial: basic syntax](guide/tutorial.md#2-basic-syntax) |
| Look up a feature and its status | [Programming guide](guide/programming.md) — follow its status table and cited evidence |
| Study precise language rules | [Language specification](spec/LANGUAGE_SPECIFICATION.md) — chapters, grammar, and rule-level source/witness catalogs |
| Generate Sounio with a language model | [LLM guide](llm-guide/README.md) — English/Portuguese manuals, complete examples, library discovery, and diagnostic repair |
| Understand parser implementation | [Lexer and parser architecture](compiler/LEXER_PARSER_ARCHITECTURE.md) |
| Check current limits | [Known limitations](compiler/KNOWN_LIMITATIONS.md) and the named tests |
| Understand measured examples | [Measured tour](../TOUR.md), with its 2026-10-04 snapshot date |

Apple Silicon has a separate current-source native ARM64 route; it is not the committed Linux prebuilt. See the [macOS ARM64 guide](compiler/MADAROS_MACOS_ARM64.md) for setup and the limits of the measured route.

## For Users

### Getting Started
- [Getting Started](guide/getting-started.md)
- [Installation Guide](../INSTALL.md)
- [Tutorial](guide/tutorial.md)
- [Minimum Viable Sounio](guide/MINIMUM_VIABLE_SOUNIO.md)
- [Programming Guide](guide/programming.md)

### Reference
- [Programming with language models](llm-guide/README.md) · [English](guide/LLM_PROGRAMMING_GUIDE.md) · [Português](guide/LLM_PROGRAMMING_GUIDE_PT.md)
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

### Additional internals and design notes
- [Technical Report](compiler/TECHNICAL_REPORT.md)
- [Effect Dispatch](compiler/EFFECT_DISPATCH_INTEGRATION.md)
- [Epistemic Backend](compiler/EPISTEMIC_BACKEND_GUIDE.md)
- [GPU Kernels](compiler/GPU_KERNELS.md)
- [Known Limitations](compiler/KNOWN_LIMITATIONS.md)
- [LLVM Codegen](architecture/LLVM_CODEGEN.md)
- [GPU Runtime](features/GPU_RUNTIME.md)
- [Async Runtime](architecture/ASYNC_RUNTIME.md)
- Roadmap: `archived/COMPILER_ROADMAP.md` is archived lineage, not the current contract. Use [TOUR.md](../TOUR.md) and [Known Limitations](compiler/KNOWN_LIMITATIONS.md).
- [Paper Artifact Packaging Spec](implementation/PAPER_ARTIFACT_PACKAGING_SPEC.md)
- [Website Design System & UI Components](../website/README.md)

### Governance
- [Docs Authority Matrix](governance/DOCS_AUTHORITY_MATRIX.md)
- [Docs Acceptance Report](governance/DOCS_ACCEPTANCE_REPORT.md)

## Specification

- [Language Specification](spec/LANGUAGE_SPECIFICATION.md) — October 2026 research edition, with chapters on syntax, types, effects, scientific values, numerics, memory, targets, libraries, ontology, quantum, and conformance.
- [Rule and witness catalogs](spec/2026-10/edition.json) — source identities, stable rule identifiers, evidence paths, and remaining decisions.
- [Conformance and diagnostics](spec/2026-10/09-conformance-diagnostics-and-compatibility.md) — actual test-runner contracts and execution receipt requirements.
- [Historical S17 study](spec/S17_CONFORMANCE_SUITE.md) — dated engine/test-suite measurements and earlier open questions.

For practical syntax, start with the [tutorial](guide/tutorial.md). For precise rules, use the specification chapter and its evidence catalog. The [public-claim registry](serious-language/public-claim-registry.v1.tsv) retains the implementation-support boundaries; a source observation or a linked test is not a fresh execution receipt.

The specification identifies the standing of each rule and separates current
source observations from approved contracts and unresolved choices. Its
structural validator checks references and coverage; compiler and formal-model
claims use their respective executable gates.

## Important reading rule

Some historical architecture reports still describe earlier Rust-centric layouts or experimental backend states. When a deep report disagrees with the current codebase:

- trust the checked artifact for user-facing capability claims
- trust `self-hosted/` for the current implementation map
- trust committed gate artifacts for reliability claims
