<!-- docs:meta
topic_id: repo.docs.features.platform-support
authority: repo_only
audience: users
last_validated: 2026-04-22
validated_by: Codex
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.features.platform-support
-->

# Platform Support

This document describes the platform contract for the current repository checkout.
It is intentionally conservative: support means a checked artifact, a repo workflow,
or a CI gate proves the lane.

## Supported Host Lanes

### Tier 1: First-Class Supported Hosts

These hosts have checked compiler artifacts and explicit CI self-host gates.

| Host OS | Architecture | Status | Contract |
|---------|--------------|--------|----------|
| Linux | x86_64 | ✅ | Checked self-hosted compiler artifact, self-host fixed-point gate, full suite lane |
| macOS | arm64 | ✅ | Checked self-hosted compiler artifact, host-native self-host gate, Mach-O ARM64 output lane |

### Tier 2: Supported But Not Yet Fully Gated

These lanes are part of the active contract, but do not yet have the same CI
coverage as the Tier 1 hosts above.

| Host OS | Architecture | Status | Contract |
|---------|--------------|--------|----------|
| macOS | x86_64 | ⚠️ | Checked self-hosted compiler artifact and Mach-O output support; no dedicated CI lane yet |
| Linux | aarch64 | ⚠️ | Target triple is accepted and compile-only output is covered; no checked host artifact in this checkout |

### Not In The Current Contract

- Windows host support
- 32-bit architectures
- Big-endian hosts
- JIT parity on macOS
- Native-v2 parity as the baseline Apple implementation

## Compiler Artifact Lanes

The repository currently exposes three distinct compiler lanes. Do not treat them
as interchangeable.

### 1. Checked self-hosted launcher (`bin/souc`)

This is the default repo-local entrypoint for contributors.

- Host-aware wrapper around checked self-hosted artifacts
- Supports Linux `x86_64`, macOS `arm64`, and macOS `x86_64`
- Provides compatibility commands:
  - `check`
  - `compile`
  - `build`
  - `run`
  - `info`
  - `--version`
- Also supports the raw self-hosted compiler interface:
  - `bin/souc <source.sio> <output> [--target <triple>]`

### 2. Omega/pinned release lane

This lane is still important for richer Linux workflows and pinned-release
resolution, especially for workflows that depend on capabilities outside the
checked self-hosted launcher contract.

- Resolver: `scripts/omega/omega_resolve_souc_bin.sh`
- Primary use: pinned Linux release binaries, GPU/JIT related flows, stricter provenance workflows
- Not the baseline for Apple support in this checkout

### 3. Native-v2 preview lane

This is an active implementation track, but it is not the support baseline for
Apple parity.

- Some ARM64 lowering remains preview-grade
- Apple support in the current contract is delivered through the current
  self-hosted Mach-O path, not through native-v2 completion

## Native Output Targets

The self-hosted compiler accepts these target triples in the current checkout:

| Target triple | Output format | Status |
|---------------|---------------|--------|
| `x86_64-linux` | ELF x86_64 | ✅ |
| `aarch64-linux` | ELF ARM64 | ✅ compile-only coverage |
| `x86_64-macos` | Mach-O x86_64 | ✅ |
| `aarch64-macos` | Mach-O arm64 | ✅ |

Cross-target outputs must be executed on the matching target OS and architecture.

## What CI Proves

### Linux `x86_64`

The main CI lane proves:

- host-native compiler bootstrap
- self-host fixed point
- representative runtime smokes
- full test suite execution
- additional native-v2 feature gate coverage

### macOS `arm64`

The Apple self-host gate proves:

- host-native compiler bootstrap on Apple Silicon
- self-host fixed point on Apple Silicon
- host-native runtime smokes for representative compiled programs
- compile-only cross-target smoke for the secondary macOS target lane

### What CI Does Not Yet Prove

- macOS `x86_64` as a dedicated host lane
- Linux `aarch64` as a dedicated host lane
- Apple JIT parity
- native-v2 as the canonical Apple implementation

## Recommended Usage

For repository work from a checkout:

```bash
export SOUC_BIN="$(pwd)/bin/souc"
export SOUNIO_STDLIB_PATH="$(pwd)/stdlib"

"$SOUC_BIN" --version
"$SOUC_BIN" info
"$SOUC_BIN" check examples/hello.sio
"$SOUC_BIN" compile examples/hello.sio -o /tmp/hello.out
"$SOUC_BIN" run self-hosted/compiler/native_print_f64_smoke.sio
```

For explicit target output:

```bash
"$SOUC_BIN" compile examples/hello.sio -o /tmp/hello-macos --target aarch64-macos
"$SOUC_BIN" compile examples/hello.sio -o /tmp/hello-linux-arm --target aarch64-linux
```

## Current Limitations

### Apple-specific

- Apple Silicon is a first-class supported host for the checked self-hosted
  compiler lane, but JIT is not part of that parity contract.
- macOS support currently ships as separate `arm64` and `x86_64` artifacts.
  Universal-binary packaging is not yet part of the release contract.
- The Apple lane is anchored to the current self-hosted Mach-O path, not to
  native-v2 completion.

### General

- Some repo workflows still depend on richer omega/pinned binaries rather than
  the checked self-hosted launcher.
- Not every historical script or archived doc describes the current host-aware
  launcher contract.
- Cross-target compile support is broader than host-native execution support.

## Bottom Line

If you want the safest statement for the current repository:

- Linux `x86_64` and macOS `arm64` are the real supported hosts
- macOS support is delivered through the checked self-hosted Mach-O compiler lane
- JIT parity is not included in the Apple support contract
- native-v2 remains an important future convergence path, but it is not the
  current Apple baseline
