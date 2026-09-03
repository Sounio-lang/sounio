## Description

<!-- Describe your changes in detail. What problem or issue does this solve? -->

## Type of Change

<!-- Mark with an `x` all that apply -->

- [ ] Bug fix (non-breaking change that fixes a compiler or runtime bug)
- [ ] New feature (non-breaking change that adds a language or stdlib capability)
- [ ] Breaking change (fix or feature that would cause existing Sounio code to fail compilation)
- [ ] Refactoring (internal compiler or website optimization, no functional changes)
- [ ] Documentation update (translations, expansions, or spec syncs)
- [ ] Test improvement (adding compile-fail gates or golden snapshot validations)
- [ ] Performance improvement (reduction of bootstrap build time or codegen optimization)

## Related Issues

<!-- Link any related issues here, e.g., "Fixes #123" or "Related to #456" -->

## Traceability

<!-- Map this PR to release-readiness tracking when relevant -->

- Traceability Matrix ID(s): <!-- e.g., COMP-001, WEB-001, ONT-042 -->
- Evidence Log(s): <!-- e.g., artifacts/diagnostic/... -->

## Release Scope

<!-- Mark with an `x` all that apply -->

- [ ] Compiler (self-hosted / codegen)
- [ ] Stdlib (modules, units, epistemic types, ontology)
- [ ] Docs
- [ ] Website
- [ ] CI / Workflow contracts
- [ ] Non-release-blocking change

<!-- Sounio automated intake parses this section. Keep scope and traceability populated. -->

---

## Quality Checklist

<!-- Mark with an `x` all that apply. Sounio's PR intake enforcement requires these to be verified. -->

- [ ] **Semicolon Check**: I have verified there are absolutely no semicolons at the end of Sounio statements.
- [ ] **Sounio Syntax Standards**: I used `var` instead of `let mut`, and `&!` instead of `&mut`.
- [ ] **Algebraic Effects declared**: Every modified function declares its active side-effects correctly (`with IO, Mut, Div, Panic`, etc.).
- [ ] **No Rust Macro syntax**: I used standard `println()` and `assert()` instead of `println!` or `assert!`.
- [ ] **Mathematical Operators**: I wrote negative numbers using explicit math (`0 - x` instead of `-x`).
- [ ] **Bit Shifts**: Shift operands are explicitly cast or typed as `u8` (e.g. `x >> 4u8`).
- [ ] **Compile and Type-Check**: I verified my code compiles and type-checks successfully using `./bin/souc check <file>`.
- [ ] **Test Suite passes**: I have run `bash scripts/run_sio_test_suite.sh` and verified that the entire test suite is green.
- [ ] **Documentation Registry Sync**: I ran `node scripts/docs/check_docs_registry.mjs` and ensured the topic registry is consistent. If I edited documentation, I updated the metadata with `node scripts/docs/sync_governance_metadata.mjs`.
- [ ] **LLM-Offload Policy Compliance**: If this PR touches math derivations, Lean 4 proofs, or clinical pathways, I ran the mandatory `bin/llm-offload` audits and appended logs to `.claude/llm_offload_log.md`.
- [ ] **Apache-2.0 License**: I understand that my contributions will be licensed under the Apache License, Version 2.0.
