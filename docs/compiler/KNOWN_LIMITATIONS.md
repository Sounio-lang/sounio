<!-- docs:meta
topic_id: repo.docs.compiler.known-limitations
authority: repo_only
audience: contributors
last_validated: 2026-03-07
validated_by: A4
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.compiler.known-limitations
-->

# Known Language Limitations

This document tracks limitations in the Sounio language implementation.
Updated February 2026 after full-project audit.

## Maturity Tiers

### Production (ship with confidence)

| Component | Status | Notes |
|-----------|--------|-------|
| Lexer/Parser/AST | Production | logos-based, error recovery, comprehensive |
| Type Checker (core) | Production | Bidirectional inference, generics, unification |
| Epistemic Types | Production | GUM uncertainty, confidence propagation, provenance |
| Effects System | Production | 9 effects (IO, Mut, Alloc, Panic, Async, GPU, Prob, Div, Observe) |
| HIR + HLIR | Production | SSA generation, async transform |
| SIR | Production | Domain-specific IR, epistemic passes |
| Ownership/Borrowing | Production | Method receiver type is now looked up from the declared signature (`scan_fnsig_param_type`). Exclusive `&!Self` receivers enforce borrow-conflict checks and ephemeral borrow tracking; shared `&Self` receivers perform read-only access checks. No heuristic string matching. |
| Native Backend | Production | ELF/Mach-O/PE, epistemic runtime, continuations; cross-compile via `--target` |
| Cranelift Codegen | Production | Full implementation, effect handlers |
| LLVM Codegen | Production | LLVM 18 wired, `--backend llvm` or `--emit-llvm`; bridge: `self-hosted/llvm/souc_emit_llvm.c` |
| Interpreter | Production | Full eval, 100+ builtins |
| Module System | Production | 2-pass resolver, imports, hierarchical namespaces |
| CLI | Production | check/build/run/repl/format/doc |
| Formatter | Production | AST-based, all constructs, diff mode |
| snn/ (sedenion NN) | Production | Training, backward, similarity, 8 scoring functions |

### Beta (works for common patterns, edge cases exist)

| Component | Status | Limitations |
|-----------|--------|-------------|
| LLVM Codegen | Production | Moved to Production — see above |
| Refinement Types + SMT | Beta | Static engine (no Z3) handles constants, condition narrowing, monotonicity; complex predicates fall back to runtime assertions with W040 diagnostic |
| LSP | Beta | Cross-file navigation now uses module resolver symbol index (Section 27 of lsp/goto_def.sio); cross-module hover and qualified completions wired via module resolver bridge |
| REPL | Beta | 21 commands, JIT, epistemic badges; :type/:econf/:hist + multi-line input added |
| Self-hosted Compiler | Beta | Phases 1.3–1.6 + Async 1-3 + generics complete (2026-04-20). Pattern matching: if-let, while-let, or-patterns (`A \| B => body`), struct destructuring. Async: `spawn { }`/`.await`, `channel::<T>()`, `sleep(ms).await`, `join(h1, h2)` — all 11 async tests PASS. Generic monomorphization: 1–2 type params. SRET: all struct sizes including 8+ fields verified. x86-64 and ARM64. |
| Ontology | Beta | 10K terms, subsumption, distance |
| Package Manager | Beta | Local registry active (`~/.sounio/registry/`), `souc publish/search/list` commands; no public registry |

### Known Bugs

*No active known bugs.* All previously listed bugs have been fixed in `self-hosted/compiler/lean_single.sio` and are live in the current `bin/souc-native` binary (rebuilt 2026-04-20).

### Fixed in Self-Hosted Compiler — All Bugs Closed

**`extern "C"` integer FFI return register** (fixed): `strip_extern_blocks()` now emits Sounio stub functions (OS syscalls for integer-returning `getpid`/`getppid`, `heap_alloc`/`heap_free` for `malloc`/`free`, `__native_*_f64` intrinsics for math). Stubs use Sounio's internal calling convention (RAX), bypassing the XMM0/RAX confusion entirely. Unblocks `stdlib/os/`, `stdlib/mem/`, `stdlib/sync/`. Regression test: `tests/run-pass/ffi_integer_return.sio`.

**Observation boundary coverage** (fixed): `Observe` now enforced for comparison, IO-arg, FFI-arg, and pattern-match scrutinee in both x86-64 and ARM64 codepaths. Self-hosted compiler and multi-file checker are now aligned. Test: `tests/compile-fail/observe_io_boundary.sio`.

### Fixed in Self-Hosted Compiler (live in current binary)

The following bugs were fixed in `lean_single.sio` and are active in the current `bin/souc-native` (rebuilt 2026-04-20):

**Mixed-Hyper optimizer metadata** (fixed): When a function mixes Hyper algebras (2+ distinct algebra kinds in its type signature), `checker_infer_fn_hyper_algebra` now computes the most-restrictive algebra kind (intersection of rule sets) instead of bailing with -1. `ocp_configure_small_context` applies the appropriate conservative reassoc strategy for that kind: free(0) for Real/Complex/Quaternion, fano_selective(2) for Octonion, blocked(1) for Sedenion/Clifford. Additionally, when a function's `hyper_algebra_kind` is -1 (tag lost at lowering) but the compilation unit has a single unambiguous algebra declaration, `ocp_infer_algebra_from_table` re-infers the kind from the registry entry so homogeneous helper functions benefit from algebra-specific reassociation. Also fixed: Octonion (kind=3) incorrectly defaulted to strategy=1 (blocked) in the fallback path; now correctly uses strategy=2 (fano_selective). Multi-algebra intersection remains a TODO (`// TODO: mixed algebra intersection` in `ocp_infer_algebra_from_table`).

**`&![T; N]` mutable ref mutation — bare array index** (fixed): When passing a bare array variable by `&!` reference, mutations via `arr[i] = v` (bare index, without explicit deref) are now correctly written back through the pointer for all element sizes. Root cause: the parameter registration in the codegen did not set `VAR_ESIZ` for `&![T; N]` fixed-size array ref parameters, so the element stride defaulted to 8 regardless of the actual element type. For `&![i64; N]` this happened to work (stride-8 is correct), but for `&![i8; N]` the stride was wrong, causing memory corruption. Fix: after `var_add` registers the parameter slot, a new branch detects `SCAN_TY == 10` with inner type `8` and sets `VAR_ESIZ = arr_hash_esiz(ref_hash_inner_hash(SCAN_TY_HASH))`. Regression test: `tests/run-pass/array_mut_ref_bare.sio`.

**Implicit `var`/`let` with `i32` type** (fixed): Integer literal narrowing now allows `var x: i32 = 5` without "expected I32, found I64" errors. Literals are compatible with annotated smaller integer types (i32, i8).

**`Option::None` type inference** (fixed): Bidirectional type inference now propagates the expected type for enum variant paths. `let x: Option<i32> = Option::None` correctly infers `Option<i32>`.

**Unit type declarations** (fixed): The resolver now registers `unit` declarations as `SymUnit` (was incorrectly using `SymTypeAlias`).

**String methods** (fixed): `.as_bytes()` returns the string as a byte array (works). `.len()` on `string` now emits a runtime null-terminated byte count (x86-64 and ARM64); previously the condition missed `EXPR_TY == 3` and leaked the string pointer as the length. Regression test: `tests/run-pass/string_len.sio`.

**Turbofish + generic monomorphization** (working): Single and dual type-parameter generic functions are monomorphised and execute correctly. `func::<T>(args)` and `func::<T, U>(args)` are fully supported — the `<TPARAMS>` section is stripped from the specialised token copy, both type parameters are substituted, and the specialised function is compiled as an ordinary function. Limitation: 3+ type parameters are not yet tracked (infrastructure covers 2 params; extend `GEN_FN_TP2_S/E` and `MONO_TY2_S/E` to add a third).

**Range slice half-open syntax** (fixed): `&arr[..n]` (start omitted, defaults to 0) now correctly compiles. Previously `compile_primary()` consumed the `..` token as an unrecognised primary, causing both the range-check and base-check to fail. Fix: detect `..`/`..=` at the start of the slice index and emit start=0 directly.

**String `.as_bytes()`** (fixed): `.as_bytes()` on a `string` is now a recognised builtin — it passes through as a no-op (string pointer unchanged, type stays `string`), making `&bytes[..n]` range slices work on the result. Previously the method fell through to field-access dispatch, producing type 0 and causing the slice borrow to segfault.

**Trait definitions** (added): `trait Name { fn method(); ... }` syntax is now parsed and trait definitions are collected into the `TraitRegistry`. Builtin trait implementations (Copy, Drop, Eq, Ord, Hash, Add, Sub, Mul, Div, Display, Debug) are pre-registered for primitive types.

**`&string[..n]` slice borrow** (fixed): String variables are now accepted as slice borrow bases in `&bytes[..n]`. Element size is 1 byte, runtime length is computed via `strlen`. Result type is `&[i8]`. Previously produced "slice borrow requires array or slice base" warning and a null-pointer segfault.

**Borrow release at call boundaries** (fixed): Borrows taken for function call arguments are now unconditionally released after the call returns, fixing false positive errors on consecutive calls borrowing the same variable.

**`(*ptr).field = value` store through explicit deref** (fixed): Explicit pointer dereference field assignment (`(*c).field = v` where `c: &! S`) was silently a no-op in the JIT — mutations were lost. The LHS deref-then-field store path was only recognising raw pointer type (ty==11) and rejecting `&!T` exclusive references (ty==10). Fix: both type 10 (`&!T`) and type 11 (`*T`) are now accepted; inner type and field offset lookup uses the shared `ptr_hash_inner_ty`/`ptr_hash_inner_hash` helpers which work identically for both. Test: `tests/run-pass/explicit_deref_field.sio`.

**Ownership state machine** (wired): The `OwnContext` ownership tracker (2836 lines, 72+ functions) is now integrated into the `Checker` — linear variable registration, ownership transfer on use, and linear-at-end checking at function exit.

**Effect propagation** (verified): Call-site effect checking (`check_callee_effects`) validates that callee effects are a subset of the caller's declared effects, reporting E035 on violations.

### Pruned/Experimental Modules

The following stdlib modules are stubs or incomplete:

- `stdlib/gpu/` - requires CUDA runtime (behind `--features gpu`)
- `stdlib/crypto/` - pure-Sounio sha256/hmac/rng are active; random.sio.disabled and hash.sio.disabled require additional algorithm work
- `stdlib/compress/` - gzip.sio requires libz at link time; zstd.sio requires libzstd at link time (external runtime libraries, not an FFI limitation)
- `stdlib/ffi/` - stub
- `stdlib/autodiff/` - framework only
- `stdlib/interop/` - stub
- `stdlib/text/*.sio.disabled`, `stdlib/time/*.sio.disabled` - old Rust-style stubs (use `u32`/`u64`/closures/`for..in`); superseded by pure-Sounio rewrites already active as `.sio` files

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

---

## Single-source build path (`lean_single.sio`)

**Status:** active constraint. Not a bug; a maturity-stage reality that contributors must know about before editing type-system logic.

### What the situation actually is

The shipped compiler binary (`bin/souc-linux-x86_64`, consumed by the `bin/souc` launcher) is produced today from a **single self-hosted source file**:

- `self-hosted/compiler/lean_single.sio`

The modular directory layout most readers expect —

- `self-hosted/lexer/`
- `self-hosted/parser/`
- `self-hosted/check/`
- `self-hosted/types/`
- `self-hosted/ir/`
- `self-hosted/native/`

— does exist, is kept in sync by hand, and describes the architectural decomposition we aim to bootstrap from. **It is not yet the source the binary is built from.** The 2-stage bootstrap recipe below uses `lean_single.sio` exclusively:

```bash
./bin/souc-linux-x86_64 self-hosted/compiler/lean_single.sio /tmp/souc-stage1
/tmp/souc-stage1 self-hosted/compiler/lean_single.sio /tmp/souc-stage2
cp /tmp/souc-stage2 bin/souc-linux-x86_64
```

### Implication for contributors

Any change to the type system, effects table, error codes, or surface syntax **must be made in `lean_single.sio`** to reach the binary. Changes made only to the modular tree are silently absent from the shipped compiler, even if the repo builds green and the tests pass against the stale binary.

Examples of this pattern in recent history:

- 2026-04-20 — surgical type gates (`ExactlyPrivate`, `Editable`, `CapabilityGated`) and error codes `E201`–`E203` added to `lean_single.sio`; modular files updated in parallel.
- 2026-04-29 — extended surgical type gates (`Composable`, `Audited`, `Revivable`, `Interpretable`), new effect bit-flags (`Witness=32768`, `Temporal=65536`, `Learn=131072`), and error codes `E204`–`E207` added to `lean_single.sio`; 2-stage bootstrap executed; `bin/souc-linux-x86_64` rebuilt.

### Risk of silent divergence

Because the two universes are kept in lock-step by discipline rather than by a test, a change that touches only one side can pass CI without any signal. Until an operational-parity harness lands under `tests/parity/`, reviewers of a PR that modifies type-system logic should explicitly confirm that `lean_single.sio` was touched and that a 2-stage bootstrap was run.

### Planned resolution

1. **Parity harness (`tests/parity/`, planned near-term).** For a fixed set of `.sio` programs drawn from `examples/` and `tests/compile-fail/`, compile via both paths and diff the stdout/stderr and exit codes (not the binaries — timestamps and symbol ordering make binary-equality unreliable). Divergence flips CI red.
2. **Source swap (roadmap, long term).** Rebuild `bin/souc-linux-x86_64` from the modular tree and retire `lean_single.sio`. This is a multi-week refactor and is not a Wave 9 target.

Until both land, treat `lean_single.sio` as the source of truth for the binary and treat the modular tree as the maintained future target.

---

## Syntax Limitations - All Resolved

This section documents previously-resolved limitations for historical context.

## Syntax - All Resolved

### Module System
- **Status**: Resolved (v0.99.0)
- **Resolution**: Full `module`/`use` support with file-based module loading and hierarchical namespace resolution.

### Visibility Modifiers
- **Status**: Resolved (v0.99.0)
- **Resolution**: `pub` visibility supported and enforced across module boundaries.

### Logical Operators
- **Status**: Resolved (v0.66.0)
- **Resolution**: `&&` and `||` implemented with short-circuit evaluation and boolean type checking.
```sio
if a > 0 && b > 0 { ... }
if is_empty || is_null { ... }
```

### Documentation Comments
- **Status**: Resolved (v0.99.0)
- **Resolution**: `///` outer docs and `//!` inner docs are parsed and preserved through AST → HIR.

### Numeric Literals
- **Status**: Resolved (v0.99.0)
- **Resolution**: Scientific notation supported in the lexer (e.g., `1e10`, `1.5e-3`).

## Type System - All Resolved

### Type Aliases
- **Status**: Resolved (v0.99.0)
- **Resolution**: `type` aliases are supported, including generic aliases; aliases expand transparently during type checking.
```sio
type Vec2 = (f64, f64)
```

### Unit Definitions
- **Status**: Resolved (v0.99.0)
- **Resolution**: User-defined units are supported and integrate with unit checking.
```sio
unit kg;
unit mg = 0.001 * kg;
unit velocity = m / s;
```

## Reserved Keywords

The following identifiers are reserved and used by the language:
- `var` - mutable binding
- `effect` - effect declaration
- `type` - type alias definition
- `module` - module declaration
- `use` - module import
- `pub` - public visibility modifier
- `unit` - unit definition

## Scoping Behavior - All Resolved

### Variable Shadowing
- **Status**: Resolved (v0.99.0)
- **Resolution**: Shadowing works correctly across nested scopes.

### Forward Declarations
- **Status**: Resolved (v0.99.0)
- **Resolution**: 2-pass resolver enables forward references and mutual recursion.

## Feature Resolution Summary

All previously planned features are implemented as of v0.99.0:

| Feature | Resolved In | Resolution |
|---------|------------|------------|
| Module system | v0.99.0 | File-based module loading with `module`/`use` |
| `&&` / `\|\|` operators | v0.66.0 | Short-circuit logical operators |
| `pub` visibility | v0.99.0 | Visibility enforcement across modules |
| Scientific notation | v0.99.0 | Lexer supports `1e10`, `1.5e-3` |
| Type aliases | v0.99.0 | `type Name = Type;` with generics |
| Doc comments | v0.99.0 | `///` + `//!` parsed and preserved |
| Variable shadowing | v0.99.0 | Correct scoping rules |
| Forward declarations | v0.99.0 | 2-pass resolver |
| Unit definitions | v0.99.0 | User-defined units + checking |

## Hessian AD Capabilities and Architectural Limits (β⁷)

`hessian_of(expr, j, k)` computes ∂²expr/∂xⱼ∂xₖ via second-order forward-mode AD.

### What Works

- **8 function inputs** (channels 0–7): indices 0–7 from `measure()` calls, 36 upper-triangular pairs
- **Arithmetic**: `+`, `−`, `*`, `/` propagate full Hessian and first-order sensitivities
- **Transcendentals (unary)**: `sqrt`, `exp`, `ln`/`log`, `sin`, `cos`, `tan`, `atan`, `tanh`, `asin`, `acos` — full chain rule f′ and f″ in all 8 channels
- **Two-arg builtins**: `atan2(y,x)` and `pow(x,y)` — full Hessian propagation for channels 0–3 and 10 pairs

### Architectural Limitations (Tier 4 — Not Planned for Near-Term)

- **Inter-procedural**: Hessian shadows do not cross user-defined function call boundaries. Workaround: inline the computation.
- **Loop accumulation**: Hessian state resets between loop iterations; only the final body is live.
- **Branch merging**: `if/else` branches do not merge Hessian state (no phi nodes for shadow slots).
- **Channels 4–7 in transcendentals**: Transcendental chain rule only propagates channels 0–3. Channels 4–7 are zero for transcendental outputs even if the input has active sensitivity there.
- **Two-arg builtins (channels 4–7)**: `atan2`/`pow` handlers propagate channels 0–3 only.

### Channel-at-`.value` semantics (resolves former "Butterfly #3")

Phase 5 re-evaluation: the MEAS_KNOW_IDX counter at `lean_single.sio:393` is incremented on every `.value` access to a Knowledge variable.  Channels are assigned **at `.value` extraction time, not at `measure()` time**.  A Knowledge struct at rest has no channel identity; it acquires one only when the user extracts `.value`.

This means the KAS-1 pattern (extract `.value` first, do scalar arithmetic) is **not a workaround** for a compiler limitation — it is the direct expression of the channel-assignment semantics.  Formalised in `formal/ChannelAssignmentSemantics.lean` (Phase 5 Lean file).

`compile_knowledge_muldiv_x86` at `lean_single.sio:5766` correctly does not touch `MEAS_KNOW_IDX`; Knowledge multiplication is channel-silent.  Attempting `hessian_of((k1 * k2).value, 0, 1)` asks for `∂²/∂x_0∂x_1` of a one-input function (the single `.value` access seeds only channel 0); the result is zero by correctness of the channel-at-`.value` model, not by any bug.

**The KAS-1 pattern (formalised in `formal/KnowledgeArithmeticSoundness.lean` + `formal/ChannelAssignmentSemantics.lean`)** expresses a multi-input Hessian function directly under the channel-at-`.value` semantics:

```sio
// Two-input Hessian function f(x, y) = x * y:
let k1: Knowledge<f64> = measure(2.0, uncertainty: 0.1)
let k2: Knowledge<f64> = measure(3.0, uncertainty: 0.1)
let x = k1.value          // seeds channel 0 with 1.0, channel 1 with 0.0
let y = k2.value          // seeds channel 1 with 1.0, channel 0 with 0.0
let z = x * y             // scalar; shadows propagate via product rule
let j: [f64; 8] = [sensitivity_of(z, 0), sensitivity_of(z, 1), ...]
let h: [f64; 36] = [hessian_of(z, 0, 0), hessian_of(z, 0, 1), ...]
let v2 = gum_second_order_variance(j, h, &sigma)
```

Phase 5 attempted to "close the butterfly" at the compiler level (commit reverted — `self-hosted/compiler/lean_single.sio` unchanged).  The attempt added 44 cross-function shadow-bridging globals and product-rule emission inside `compile_knowledge_muldiv_x86`.  It correctly set `EXPR_SSHADOW` before the function returned, but the downstream `.value` access re-seeded channel 0 via MEAS_KNOW_IDX — overwriting the propagated shadow.  The lesson: under channel-at-`.value` semantics, there is no butterfly to close.  `tests/run-pass/knowledge_kas1_policy.sio` remains as a demonstration of the two paths; the "butterfly" path correctly returns zero under the model.

## Native linking / FFI (open)

- **Dynamic `.so` linking / GOT–PLT**: The native toolchain emits statically linked executables on the default bring-up path. Relocation metadata records `R_X86_64_PLT32` for ET_REL objects (`self-hosted/native/reloc.sio`), but wiring arbitrary shared-library symbols end-to-end (libc-style `-lfoo`, full GOT/PLT for external calls) remains future work. FFI tests that require `libzstd` stay behind `//@ ignore` until dynamic link and stable stubs land consistently.
- **`*const u8` in callee position**: Passing `expr as *const u8` can still produce arity/type mismatch diagnostics versus `*mut u8` in some FFI-heavy shapes; stdlib wrappers prefer `*mut u8` until call lowering treats `*const T` and `*mut T` uniformly at the invocation site. See the header comment in `tests/stdlib/compress/test_zstd_e2e.sio`.

## Reporting Issues

If you encounter any new issues, please report them at:
https://github.com/sounio-lang/sounio/issues
