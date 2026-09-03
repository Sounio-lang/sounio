<!-- docs:meta
topic_id: repo.docs.serious-language.readiness-ledger
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.serious-language.readiness-ledger
-->

# Serious-Language Readiness Ledger

> **Superseded 2026-08-15**: the current critical-path assessment for launch
> readiness is [`CAMINHO_CRITICO_CORTADO_2026-08-14.md`](CAMINHO_CRITICO_CORTADO_2026-08-14.md),
> not this ledger. This file's per-claim table is still useful as a reference
> for wording individual claims, but its overall "current status" framing and
> the specific compiler-binary-source claim below are stale (this ledger
> predates `bin/souc` routing to Madaros by default; see the correction inline).

> **Status**: Research readiness | **Operational check**: 2026-05-11 | **Source**: checked commands and current repository docs

This ledger maps public-facing claims to evidence and safe wording. Use it before talks, papers, abstracts, or website updates.

## Claim Levels

| Level | Meaning | Public wording rule |
|---|---|---|
| `stable` | Narrow behavior is checked and appropriate for a demo or paper appendix. | State the exact surface and command. |
| `validated research` | Evidence exists, but scope is research-lane or environment-dependent. | Name the gate and boundary. |
| `prototype` | Useful implementation exists, but not enough to sell as reliable. | Describe as prototype or experimental. |
| `scaffold` | Files or APIs exist, but callable behavior is not established. | Do not present as a working feature. |
| `stale/conflicting` | Documentation or claims disagree across repo surfaces. | Refresh before citing externally. |

## Core Language And Compiler Claims

| Claim | Level | Current evidence | Safe wording |
|---|---|---|---|
| Checked compiler entry point | `stable` | `./bin/souc --version`; `./bin/souc info`; `docs/guide/MINIMUM_VIABLE_SOUNIO.md` | "The checked entry point is `bin/souc`; on Linux x86-64 in this checkout it selects `souc-linux-x86_64` with `check`, `compile`, `build`, `run`, `info`, and `version` compatibility commands." |
| Binary attestation | `stable` | `scripts/paper/build_serious_language_bundle.sh` manifest hashes | "A generated paper bundle records the wrapper path, selected compiler binary, SHA256 hashes, byte sizes, host, branch, and commit for the evidence run." |
| Bounded conformance spine | `stable` | `scripts/ci/serious_language_conformance_gate.sh`; `tests/conformance/manifest.v1.tsv`; `docs/serious-language/conformance-spine.md` | "A small claim-indexed conformance corpus checks selected core, effect, module, generic, ownership, and epistemic/GUM behavior. It is not a complete language specification suite." |
| Spec/evidence drift gate | `prototype` | `docs/serious-language/spec-evidence-matrix.v1.tsv`; `scripts/ci/serious_language_spec_drift_gate.sh`; generated bundle `spec-drift/RESULTS.md` | "Executable spec claims are tracked in a v1 spec/evidence matrix. The drift gate requires tracked executable rows to cite live repo evidence and requires cited conformance cases to pass." |
| Public claim closure gate | `prototype` | `docs/serious-language/public-claim-registry.v1.tsv`; `docs/serious-language/doc-claim-surface.v1.tsv`; `docs/serious-language/claim-line-annotations.v1.tsv`; `scripts/ci/serious_language_claim_closure_gate.sh`; generated bundle `claim-closure/RESULTS.md` | "Public PL claims are registry-backed. A repo doc claim must either cite closed evidence or remain explicitly downgraded, internal, or historical." |
| Self-hosted compiler | `validated research` | `scripts/ci/selfhost_host_gate.sh`; CI `native-selfhost-*` jobs; `docs/compiler/KNOWN_LIMITATIONS.md` | "Sounio has a self-hosted checked compiler path with host gates; it is a serious research compiler, not a finished general-purpose production toolchain." |
| Binary source of truth | `stable` | `./bin/souc --version` prints `Madaros`; CLAUDE.md §4; `.github/workflows/ci.yml`'s `madaros-witness-gate` (PR #1739) builds and tests Madaros from `self-hosted/compiler/main.sio` directly | "`bin/souc` routes to Madaros, the modular self-hosted compiler built from `self-hosted/compiler/main.sio` (`make build-madaros`), not from `lean_single.sio`. `lean_single.sio` remains the bootstrap seed and the fixed-point-verified ELF (`make build`), but is no longer the default user-facing engine; force it with `SOUNIO_SOUC_ENGINE=lean_single`." (STALE CLAIM CORRECTED 2026-08-15 — see superseding note above) |
| Linux x86-64 native compile/run | `stable` | `bin/souc info`; `examples/hello.sio`; `scripts/run_sio_test_suite.sh` | "Linux x86-64 is the primary serious lane for live compiler demonstrations." |
| macOS artifact support | `validated research` | CI `native-selfhost-macos-arm64`; `docs/guide/MINIMUM_VIABLE_SOUNIO.md` | "macOS has checked artifact lanes, but Apple support should not be described as JIT or native-v2 parity." |
| Windows target | `prototype` | `docs/compiler/KNOWN_LIMITATIONS.md` platform table | "The repository contains PE/COFF target support, but no public Windows binary should be promised from this checkout." |
| Sedenion native-v2 import smoke | `validated research` | `scripts/ci/madaros_sedenion_native_v2_gate.sh`; `tests/run-pass/sedenion_import_native_v2_smoke.sio`; `docs/compiler/KNOWN_LIMITATIONS.md` zero-event section | "Madaros native-v2 executes the sedenion zero-divisor import smoke after array-ref CD helpers." |
| qd128_core / qd_mul native-v2 | `validated research` | `scripts/ci/madaros_qd128_core_native_v2_gate.sh`; `scripts/ci/madaros_qd128_mul_native_v2_gate.sh`; #1652 | "Madaros executes qd128_core constructors and full qd_mul after the [f64;9] nine-sum pack. Combined sedenion+eisa zero-provenance remains fail-closed." |
| Combined zero-provenance (sedenion+eisa) | `validated research` (fail-closed) | `scripts/ci/madaros_zero_provenance_failclosed_gate.sh`; `docs/handoff/BLK-20260805-p0b-zero-provenance.md` | "Combined sedenion+eisa::core_v2 zero-provenance is classified fail-closed under stock Madaros (thin-link rc=12, no segfault) with lean_single oracle PASS. Not Madaros-green." |
| Compact zero-provenance (sedenion+local) | `validated research` | `scripts/ci/madaros_zero_provenance_native_v2_gate.sh`; `tests/run-pass/zero_provenance_native_v2_smoke.sio` | "Madaros executes a compact zero-provenance smoke (sedenion annihilated zero + local f64 kinds). This is not the eisa::core_v2 combined-import claim." |
| Fixed-array OLS under Madaros | `validated research` | `scripts/ci/madaros_ols_fixed_e2e_gate.sh`; `stdlib/stats/ols_fixed.sio` | "Fixed-array OLS (`stats::ols_fixed` + cooks + shapiro) is green under Madaros, with a positive `stats::validation` control." |
| stats::validation under Madaros | `validated research` | `scripts/ci/madaros_validation_import_gate.sh`; `stdlib/stats/validation.sio` | "stats::validation uses fixed `[f64; 256]`+`n` (no open-slice `.len()` / growable `.push()`). Do not claim open-slice array methods under Madaros import." |
| Direct-driver support cohort | `validated research` | `scripts/ci/sounio_direct_driver_support_gate.sh`; `tests/selfhost-driver-output/*.sio`; `docs/architecture/compiler-maturity-blueprint.md` M9 frontier | "The bounded direct-driver support cohort compiles and executes 24/24 small selfhost driver-output fixtures. This is not large-surface execution, semantic authority, ontology negative-truth restoration, or wrapper replacement." |
| Large-surface direct-driver closure | `prototype` | `docs/architecture/compiler-maturity-blueprint.md` M9 frontier; `docs/architecture/truth-frontier.md` | "Large-surface execution is still a compiler-maturity frontier; failures must be classified, not hidden." |

## Language Feature Claims

| Claim | Level | Current evidence | Safe wording |
|---|---|---|---|
| Core syntax, functions, structs, control flow | `stable` | `docs/guide/LLM_PROGRAMMING_GUIDE.md`; `tests/run-pass/` | "The core language surface supports ordinary small programs and is covered by run-pass tests." |
| Effects | `stable` | `docs/compiler/KNOWN_LIMITATIONS.md`; compile-fail tests | "Effects are a central checked surface; demos should use small examples with expected diagnostics." |
| Epistemic `Knowledge` and GUM propagation | `validated research` | `docs/guide/MINIMUM_VIABLE_SOUNIO.md`; `scripts/ci/package_pbpk_gum_gate.sh`; stdlib science gates | "The epistemic/GUM core is one of Sounio's strongest validated research surfaces." |
| Refinement and SMT | `prototype` | `docs/compiler/KNOWN_LIMITATIONS.md` beta table | "Refinement support exists for common/static cases; complex predicates may fall back or remain beta." |
| Ownership and borrowing | `validated research` | `docs/compiler/KNOWN_LIMITATIONS.md`; run/compile-fail coverage | "Ownership and borrowing are implemented for the checked examples; avoid broad Rust-equivalence claims." |
| Traits and generics | `prototype` | `docs/compiler/KNOWN_LIMITATIONS.md` turbofish/generic notes | "Generics and trait syntax exist in constrained forms; they are not a full mature trait ecosystem." |
| Modules/imports | `validated research` | `docs/guide/MINIMUM_VIABLE_SOUNIO.md`; module resolver docs | "Imports work for active module surfaces; not every path under `stdlib/` is callable." |
| Standard library support surface | `validated research` | `scripts/ci/sounio_stdlib_surface_support_gate.sh`; `scripts/ci/package_pbpk_gum_gate.sh`; `scripts/stdlib/scan_stdlib.sh` | "Sounio has a bounded stdlib support contract: checked inventory plus package-backed epistemic/GUM/units/formats/io/PBPK workflows. It is not broad all-file stdlib callability, hyper/science pipeline closure, API stability, or external-runtime/clinical validation." |
| Formatter, REPL, LSP | `validated research` | `scripts/ci/sounio_editor_tooling_support_gate.sh`; `tools/lsp/test_smoke.sh`; G5a/G5b gates | "Sounio has SOTA-preview editor tooling: checked formatter, file-backed REPL, preview LSP over stdio, and VS Code/Helix/Neovim wiring. It is not mature IDE support; pure-Sounio LSP rebuild, semantic-token delta, notebooks, AI assistant integration, marketplace polish, and unopened-file workspace indexing remain out of scope." |

## Scientific, Formal, And Research Claims

| Claim | Level | Current evidence | Safe wording |
|---|---|---|---|
| Formal Lean corpus | `validated research` | `formal/lean4/`; CI `lean-proofs`; bundle `lean-sorry-audit` and `lean-build` logs | "Sounio has a substantial Lean proof surface. Generated bundles must report exact `sorry`/`axiom` counts and whether `lake build` ran on that host." |
| 168 theorem / Cayley-Dickson algebra | `validated research` | `formal/lean4/SounioCayleyDickson.lean`; `formal/lean4/SounioZeroDivisorBridge.lean`; papers | "The algebraic counting work is a serious research result with computational/formal artifacts; avoid turning it into unsupported biological or EEG claims." |
| GPU/PTX | `validated research` | GPU CI, `scripts/ci/native_v2_epistemic_accel_spine_gate.sh`, local GPU gates when available | "PTX/GPU work is a compiler/research backend lane with checked fixtures; general GPU runtime or performance claims require a named run and hardware." |
| Hypercomplex neural networks | `prototype` | README honest status; stdlib hyper gates; research artifacts | "Hypercomplex ML is an active research/prototype area; do not imply a complete training stack unless the specific gate is cited." |
| Ontology | `validated research` | `scripts/ci/run_ontology_validation.sh --mode rebuilt ontology`; README honest status | "Ontology support has validated local surfaces; federation-scale ontology claims are not supported." |
| Clinical/pharmacology artifacts | `validated research` | `scripts/ci/package_pbpk_gum_gate.sh`; clinical paper docs; offload policy | "Clinical-pathway claims require strict scope, no PHI, external review, and explicit validation gates." |

## Ecosystem And Presentation Claims

| Claim | Level | Current evidence | Safe wording |
|---|---|---|---|
| Installation | `prototype` | `INSTALL.md`; `docs/guide/installation.md`; checked artifacts | "For conferences, use the checked repo artifact path instead of promising broad package-manager installation." |
| Package manager | `validated research` | `scripts/ci/sounio_package_support_gate.sh`; `docs/compiler/PACKAGE_IMPORT_RESOLUTION.md` | "Local package manifests, imports, package-tool smoke, and local registry design are checked; no public registry launch should be promised." |
| Website/docs | `prototype` | CI website job; docs registry checks | "Docs are extensive but must be filtered through the readiness ledger before external reuse." |
| Paper bundle | `prototype` | `docs/serious-language/paper-bundle.md`; `scripts/paper/build_serious_language_bundle.sh` | "The paper bundle exists as a reproducibility scaffold; each generated bundle must be reviewed before submission." |

## Current Red Flags

- The repo contains honest-status docs and older ambitious docs side by side.
- `README.md`, `docs/guide/MINIMUM_VIABLE_SOUNIO.md`, and `docs/compiler/KNOWN_LIMITATIONS.md` do not always use the same confidence vocabulary.
- Formal claims must separate no-sorry/no-axiom modules from modules with explicit axioms or remaining `sorry`.
- GPU and hypercomplex work must be tied to exact hardware, command, and artifact outputs.
- External-facing papers need `bin/llm-offload` review before submission.
- Public PL claims must be present in `docs/serious-language/public-claim-registry.v1.tsv` or kept out of public support wording; high-value public claims should also appear in `docs/serious-language/claim-line-annotations.v1.tsv`.

Operationally, the default external entry point is `bin/souc` (official) with Madaros as the active engine; teams should reference this route first for conferences, onboarding, and reproducibility scripts.
