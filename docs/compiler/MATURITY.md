<!-- docs:meta
topic_id: repo.docs.compiler.maturity
authority: repo_only
audience: contributors
last_validated: 2026-03-07
validated_by: A4
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.compiler.maturity
-->

# Maturity Tiers

**Authoritative source for maturity tiering is `docs/serious-language/public-claim-registry.v1.tsv`.**
This file is reconciled to that registry. If they disagree, the registry wins.
Open language/compiler defects are tracked separately in
`docs/compiler/KNOWN_LIMITATIONS.md` (ledger); the historical narrative lives in
`docs/audit/KNOWN_LIMITATIONS_HISTORY_2026-09.md`.

Last reconciled: 2026-08-24 against `origin/main` at `7ecec1088158db3a92983cb3c96f9fc52f5ed19e`, then re-measured 2026-08-27 on rebase against `origin/main` at `055825a3f9`, and again 2026-08-29 on merge against `origin/main` at `64db7167f8` (only the stdlib file count moved: 1604 -> 1608). Moved out of `KNOWN_LIMITATIONS.md` on 2026-09-11 (KL-0) and re-measured against `origin/main` at `eb7c7350e` (stdlib file count 1608 -> 1615, entrypoints unchanged at 178). Engine- and artifact-specific rows name their scope; an unqualified claim must hold for both Madaros and `lean_single`.

The token contracts checked by `scripts/ci/sounio_stdlib_surface_support_gate.sh`,
`scripts/ci/sounio_direct_driver_support_gate.sh` and
`scripts/ci/sounio_package_support_gate.sh` read this file.

Tiers below mirror the public-claim registry's `claim_level`/`closure_status` columns. "Production" is reserved for rows the registry calls `stable / closed`. Anything the registry marks `prototype` or `stale_conflicting` is tiered accordingly here — even if the feature works on small fixtures.

### Production (registry: stable / closed)

| Component | Status | Notes |
|-----------|--------|-------|
| Lexer/Parser/AST | Production | logos-based, error recovery, comprehensive |
| Type Checker (core) | Production | Bidirectional inference, generic monomorphization (1–2 params), unification |
| Effects System | Production | 9+ effects (IO, Mut, Panic, Div, Alloc, Session, Observe, Audit, Hypothesis; +GPU, Deterministic). Strict E035 subset check at call sites. |
| HIR + HLIR | Production | SSA generation, async transform |
| SIR | Production | Domain-specific IR, epistemic passes |
| Native Backend (Linux x86-64) | Production | Registry: `platform.linux_x86_64 = stable`. Direct x86-64 ELF emission + epistemic runtime + continuations. (There is no Cranelift/JIT backend — the retired Rust Cranelift runner is gone; the default compiler is self-hosted Madaros.) |
| Interpreter | Production | Full eval, 100+ builtins |
| Module System (single-file import unit) | Production | 2-pass resolver, imports, hierarchical namespaces |
| CLI commands `check`/`compile`/`build`/`run`/`info`/`--version` | Production | Wired through `bin/souc`. **Exit-code contract repaired 2026-05-27 — typecheck failures now exit non-zero (G2).** |
| snn/ (sedenion NN) | Production | Training, backward, similarity, 8 scoring functions |

### Validated Research (registry: validated_research / closed-with-named-gate)

| Component | Status | Notes |
|-----------|--------|-------|
| Native Backend (macOS arm64 / x86_64) | Validated research | Registry: `platform.macos = validated_research`. Mach-O cross-compile lane; no Apple JIT, no native-v2 parity. |
| Self-hosted Compiler (single-file path) | Validated research | `lean_single.sio` self-hosts; gen2==gen3 fixed point. Registry: `selfhost = validated_research`. |
| LLVM Codegen | Validated research | LLVM 18 bridge `self-hosted/llvm/souc_emit_llvm.c` wired but disabled in the checked artifact; `--backend llvm` needs a feature-flag rebuild. Previously over-claimed as Production. |
| Refinement Types + SMT | Validated research | Static engine handles constants, condition narrowing, monotonicity; complex predicates fall back to runtime assertions with W040 diagnostic. |
| Module imports across files | Validated research | Registry: `modules.imports = validated_research`. |
| Ownership / borrowing | Validated research | Registry: `ownership.borrowing = validated_research`; exclusive/shared receiver fixtures are covered, but this is not a claim of Rust-equivalent ownership semantics. |
| Editor tooling preview | Validated research | Registry: `tooling.editor = validated_research/closed`. `scripts/ci/sounio_editor_tooling_support_gate.sh` proves public `bin/souc format`/`fmt`, file-backed `bin/souc repl`, preview `bin/souc lsp --stdio`, G5a/G5b, bash LSP smoke, initialize capability smoke, and VS Code/Helix/Neovim static wiring. This is a SOTA-preview support contract, not mature IDE support. |
| LSP pure-Sounio server rebuild | Prototype blocker | `self-hosted/lsp/server.sio` currently fails to rebuild under the active Madaros path; the checked preview LSP route is `tools/lsp/sounio-lsp.sh` via `bin/souc lsp --stdio`. Do not claim the pure-Sounio LSP rebuild until `tools/lsp/test_protocol.sh` or an equivalent gate is green. |
| GPU PTX backend | Validated research | Registry: `gpu.ptx = validated_research`. Named gate covers L4 fixtures; out-of-fixture behavior is research. |
| 168 / Cayley-Dickson algebra | Validated research | Registry: `algebra.168 = validated_research`. Algebraic/formal artifacts only — no biological or EEG advantage claims. |
| Ontology subsystem | Validated research | Registry: `ontology = validated_research`. Rebuilt ontology validation surfaces only. |
| Epistemic Types — `Knowledge<T>` / GUM | Validated research | Single-file emit and named GUM gates are validated surfaces. Engine-specific epsilon semantics and payload bounds remain open (`KNOWN_LIMITATIONS.md` rows KL-5 and KL-1); clinical use must cite the exact engine and gate. |

Editor-tooling details:

- Formatter: `souc format <file>` / `souc fmt` dispatches to
  `tools/fmt/sounio-fmt.sh`. Phase 1 is token-level and idempotent for the
  G5a corpus; it does not claim AST round-trip formatting or full style
  configuration.
- REPL: `souc repl` dispatches to `tools/repl.sh`, a file-backed eval loop
  that accumulates definitions and runs expressions through the active `souc`
  wrapper. A fully Sounio-native eval loop remains deferred until process-spawn
  primitives are available.
- LSP: `souc lsp --stdio` dispatches to the preview `tools/lsp/sounio-lsp.sh`
  server. It is smoke-tested for JSON-RPC framing, compiler-backed diagnostics,
  hover/definition roundtrips, multi-document isolation, timeouts, and failure
  diagnostics. No pure-Sounio LSP rebuild under current Madaros is claimed or
  demonstrated; that server source remains a separate rebuild blocker.

### Bounded validated research surfaces and prototypes

| Component | Registry row | Honest status |
|-----------|--------------|---------------|
| **Standard library support surface** | `stdlib.surface = validated_research` | Claim only the bounded support contract checked by `scripts/ci/sounio_stdlib_surface_support_gate.sh`: current inventory has 1615 `.sio` files, 0 disabled files, 0 stub-only `mod.sio` files, and 178 active module entrypoints; package-backed epistemic/GUM, units, formats, io-primitives, canonical PETAB, and PBPK/GUM workflows pass through `scripts/ci/package_pbpk_gum_gate.sh`. **NOT PROVED:** broad all-file stdlib callability, `scripts/ci/stdlib_evolution_gate.sh`, hyper native lanes, fMRI/PBPK science pipeline, external runtime dependencies, cryptographic security, clinical/regulatory validity, or API stability beyond the checked gate. |
| **Package manager / registry** | `tooling.package = validated_research` | Local `~/.sounio/registry/` only. No public registry. Local package manifests, local package imports, and `tools/sounio-pkg/sounio-pkg` build/check/test smoke are covered by `scripts/ci/sounio_package_support_gate.sh`. |
| **Generic structs/functions/traits** | `generics.* = prototype` | Multi-type-param generic functions work (incl. 3+ params, verified); do not claim a mature trait ecosystem. Trait bounds are parsed but not enforced at call sites. No trait objects. |
| **Closures** | `closures.lambdas = stale_conflicting` | Non-capturing/direct/HOF forms and some captured forms have passing fixtures, but the registry is deliberately downgraded. Captured closures used as first-class values and cross-engine parity remain unresolved; `closure_linear.sio` is still ignored. Do not promote this row until the registry and native gates are reconciled. |
| **Units of measure** | `units.measure = prototype` | Fixture-backed prototype surface. |
| **Refinement types (general)** | `refinement.types = prototype` | Beta/prototype; runtime fallback dominates non-trivial predicates. |
| **Hypercomplex NN (broad)** | `hypercomplex.nn = prototype` | Research/prototype unless a named gate covers the exact behavior. |
| **Direct-driver support cohort** | `direct_driver.support = validated_research` | Claim only the bounded support cohort checked by `scripts/ci/sounio_direct_driver_support_gate.sh`: 24/24 `tests/selfhost-driver-output/*.sio` fixtures compile to ELF and execute with expected stdout/exit. **NOT PROVED:** large-surface direct-driver execution, ontology-sized semantic truth, wrapper-provenance replacement, native-v2 driver self-compile/fixed-point closure, direct-driver negative-truth restoration, or broad production readiness. |
| **Direct-driver execution at scale** | `direct_driver = prototype` | Large-surface direct-driver execution remains a maturity frontier. The bounded support cohort above does not promote direct-driver semantic authority on ontology-sized or compiler-sized surfaces. |
| **Windows target** | `platform.windows = prototype` | PE/COFF lane wired; not stable. |
| **`binary.source` (modular self-hosted tree)** | `validated_research` | The checked x86-64 Madaros prebuilt is built from `self-hosted/compiler/main.sio` and covered by the named Madaros source-to-ELF/full gates. This claim applies to `bin/madaros-linux-x86_64`, not the legacy `bin/souc-linux-x86_64`; `lean_single.sio` remains the bootstrap seed and escape hatch. |

### Strict Numerical Regressions and Mathematical Rigor Policy

Numerical regressions in pharmacokinetics, GUM uncertainty propagation, and clinical pathways are strictly monitored. It is an absolute policy that regression tolerances must never be loosened or "afrouxadas" through artificial modifications simply to make tests pass. If a physical model test or mathematical/clinical verification fails, the underlying compiler code or the physical model itself must be fixed honestly.

### Experimental and external-dependency modules

The current stdlib inventory contains **0 disabled files** and **0 stub-only
`mod.sio` files**; the previous disabled/stub list was stale. That inventory is
not a completeness or security proof:

- `stdlib/gpu/` requires a CUDA runtime for execution.
- `stdlib/crypto/` is active, but no cryptographic-security claim follows from
  source presence or the bounded stdlib support gate.
- `stdlib/compress/` has external-runtime lanes for libz/libzstd; arbitrary
  dynamic linking remains limited (`KNOWN_LIMITATIONS.md` row KL-14).
- `stdlib/ffi/` contains active helpers, subject to the allowlisted,
  engine-specific FFI boundary (`KNOWN_LIMITATIONS.md` row KL-14).
- `stdlib/autodiff/` and `stdlib/interop/` have active source surfaces, but broad
  end-to-end validation is not established by the inventory gate.

### Recently Activated Modules

- `stdlib/text/format.sio` - `format_int(i64) → string`, `format_f64(f64) → string` (4 decimal places); uses str_concat+str_slice, no heap. Smoke test: `tests/run-pass/stdlib_time_basic.sio`.
- `stdlib/text/case.sio` - char/string case conversion (uppercase, lowercase, titlecase, snake_case, camelCase, PascalCase, kebab-case); pure Sounio, no FFI.
- `stdlib/text/unicode.sio` - Unicode character classification (alphabetic, numeric, whitespace, punctuation, control, ASCII variants); pure Sounio.
- `stdlib/time/duration.sio` - `Duration` struct with nanosecond precision; arithmetic: dur_add, dur_sub, dur_from_millis, dur_to_millis; pure Sounio, no FFI.
- `stdlib/time/datetime.sio` - `DateTime` struct with full calendar arithmetic (leap year, days-in-month, unix epoch roundtrip, year rollover); pure Sounio, no FFI. Smoke test: `tests/run-pass/stdlib_time_basic.sio`.
- `stdlib/time/instant.sio` - Monotonic clock via `clock_gettime` syscall; uses integer FFI (now working).
- `stdlib/os/process.sio` - getpid/getppid/exit/abort via extern "C" stubs (integer FFI now works)
- `stdlib/mem/` - heap_alloc/heap_free (malloc/free stubs), arena bump allocator, box/rc/arc wrappers — all active
- `stdlib/sync/mutex.sio` - pthread_mutex_{init,lock,trylock,unlock,destroy} via extern "C" stubs
- `stdlib/prob/` - Beta, Normal, MCMC, random distributions (4 modules activated)
- `stdlib/onn/` - Octonion neural network: activation, attention, conv, linear, loss, normalization, optimizer, training (8 modules)
- `stdlib/ontology/` - LOINC, biomedical module, namespaces (3 modules)
- `stdlib/compress/deflate.sio` - stored-block DEFLATE only (RFC 1951 BTYPE=00, no compression); gzip/zstd modules still require integer FFI
- `stdlib/heliobiology/units.sio` - space weather units
- `stdlib/ode/tsit5_multicomp.sio` - multi-compartment adaptive Tsit5 solver
- `stdlib/medlang/` - full MedLang DSL (lexer, parser, AST, codegen, PK models, population, dosing) — all active

### Optional External Dependencies

| Feature | Dependency | Effect if Missing |
|---------|------------|-------------------|
| `--features llvm` | LLVM 18 (`libLLVM-18.so`) | `--backend llvm` and `--emit-llvm` active; install `llvm-18-dev` + `clang-18` |
| `--features smt` | Z3 + cmake | Without Z3: static engine handles constants/narrowing/monotonicity; QF_LIA Fourier-Motzkin tier (`smt_qflia.sio`) sits between static analysis and runtime fallback; complex predicates beyond FM fall back to runtime checks with W040 |
| `--features gpu` | CUDA toolkit | GPU codegen works, execution requires runtime |

### Platform Support

- **Linux x86-64**: Primary supported platform (default)
- **Linux aarch64**: Supported via `--target aarch64-linux`
- **macOS x86-64**: Mach-O backend (2,512 lines) wired; cross-compile via `--target x86_64-macos`
- **macOS ARM64**: Mach-O ARM64 backend wired; cross-compile via `--target aarch64-macos`
- **Windows x86-64**: PE/COFF backend (3,508 lines) wired; cross-compile via `--target x86_64-windows`. No pre-built .exe shipped in this checkout.

Cross-compiled binaries must be executed on the target OS. The compiler runs on Linux and emits the correct binary format for each target.
