# Decision Record: Rustless Cutover (Sounio-First Toolchain)

**Date**: 2026-02-13
**Status**: Direction set, execution phased
**Scope**: Toolchain architecture, CI gates, bootstrap strategy

## Context

Sounio is self-hosting. Today, Rust still appears in the critical path as:

- the `souc` CLI binary (parsing args, loading stdlib, orchestrating compilation)
- the Rust bytecode VM that executes self-hosted compiler output
- Rust integration tests used as an oracle/runner for complex programs (historic decision)

We want to **go big**: minimize and then remove Rust from the correctness and execution path,
so the toolchain is *Sounio-first* and Rust becomes (at most) a temporary Stage 0 bootstrap.

This aligns with:

- driver-first self-hosted compilation (`bootstrap::driver`)
- bootstrap attestation (`stdlib/compiler/bootstrap/verify.sio`)
- the need to validate nontrivial numeric workloads (hypercomplex + ML kernels) without Rust runners

## Decision

### D1) Rust is no longer allowed as the *validation oracle* for self-hosted correctness

**Decision**: We will not rely on Rust interpreter execution to validate self-hosted compilation correctness.

**Supersedes**: the spirit of “Decision 9: Rust integration test for self-hosted execution” in
`.claude/decisions.md` (kept for history, but treated as a transitional workaround).

**Replacement**:

- Validation happens via **self-hosted execution** (Sounio VM / self-hosted VM / native codegen),
  with explicit test suites written in Sounio.

### D2) Rust becomes Stage 0 bootstrap only (thin launcher), with a planned exit

**Decision**: Rust, if present, must be a minimal launcher whose only responsibilities are:

- loading the embedded stdlib/compiler modules
- invoking `bootstrap::driver` entrypoints
- executing the resulting artifacts (temporarily via a VM)

**Exit strategy**: replace the Rust launcher with a tiny non-Rust loader (C/Zig) OR a Sounio-compiled native binary.

### D3) The correctness target is “attested self-compilation”, not “matches Rust”

**Decision**: The north-star correctness criterion is a multi-stage self-compilation chain:

- Stage 0: trusted bootstrap (temporary)
- Stage 1: self-hosted compiler built by Stage 0
- Stage 2: self-hosted compiler built by Stage 1
- Verified: Stage 1 vs Stage 2 cross-validation + semantic gates

This is the intent of `stdlib/compiler/bootstrap/verify.sio` and will be wired to real compilation outputs.

## Integration With Existing Plans

### Native codegen self-hosted plan (`.claude/decisions.md`)

This rustless decision does not change the native codegen design; it changes **how we validate and cut over**:

- The self-hosted VM hypercomplex tower (Quat/Oct/Sed + ML kernels) becomes an executable spec and gate.
- Native codegen must match the VM semantics for these ops (or provide equivalent runtime calls).

See also: `.claude/decisions/2026-02-13-selfhost-vm-hypercomplex.md`.

### Effects backend integration (`.claude/effect_backend_integration_plan.md`)

That plan currently assumes a Rust runtime library (object/static lib) for effect dispatch.
Rustless cutover requires we eventually:

- compile the runtime layer from Sounio (or a tiny C runtime), and
- remove “Rust runtime crate” from the long-term dependency graph.

## Execution Plan (Phased)

### Phase R0: Remove Rust-as-oracle (now)

- Keep `souc` as a launcher, but stop using Rust interpreter as “truth”.
- CI gates should run self-hosted test suites through the self-hosted pipeline (strict mode).

### Phase R1: Make driver-first compilation handle real programs (self-hosted directory/packages)

- Ensure directory compilation (multi-file) is the default route for internal suites.
- Expand driver boundaries so that the self-hosted compiler can compile its own test corpus without special-casing.

### Phase R2: Replace Rust bytecode execution with a Sounio-side executor

Options:

- execute bytecode using a Sounio-implemented Bytecode VM (compiled by native codegen), or
- bypass bytecode and execute a lower-level IR VM / native output directly.

### Phase R3: Replace Rust launcher

End state options:

- `poseidon` (tiny C/Zig runner) loads the verified compiler artifact + runs it
- or `sounio` native binary is produced by the self-hosted native backend and becomes the canonical compiler.

## Acceptance Criteria

- No CI “correctness” gate depends on Rust interpreter execution.
- Self-hosted compilation of the self-hosted compiler and its suites succeeds in strict mode.
- Hypercomplex + ML forward kernels run under self-hosted execution as part of the test corpus.
- A documented bootstrap chain exists (Stage 0/1/2) with measurable parity checks.
