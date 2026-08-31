<!-- docs:meta
topic_id: repo.docs.implementation.bootstrap-seed-policy
authority: historical
audience: maintainers
last_validated: 2026-03-07
validated_by: A7
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.implementation.bootstrap-seed-policy
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Bootstrap Seed Policy (R1)

This document defines the transition seed artifact used to bootstrap the
self-hosted compiler without runtime Rust compilation of the root suite.

## R2 Cutover Update (No-Rust Contracts)

This section described signed bundle/state bootstrap commands:

> **These commands do not exist.** `bootstrap` and `opt` were subcommands of the
> Rust `souc` (`crates/souc/src/main.rs`), removed on 2026-02-26 by
> `79acc192e1 [cutover] Remove Rust crates -- compiler is self-hosted`. The
> Madaros compiler shipped as `bin/souc` has no `bootstrap` or `opt` verb --
> `souc --help` does not list them. They do fail loudly rather than silently:
> `souc bootstrap verify --bundle bootstrap` exits **1**, because `souc` falls
> back to treating an unrecognised argument as a source filename and
> reports `error: at bootstrap:0:0 - could not read input file`. The diagnostic
> names a missing file rather than a missing subcommand, so it misdirects, but
> it is a real refusal and no script relying on it can pass silently. (Note that
> `souc --help` is not a complete inventory of what the binary dispatches --
> `souc ontology`, for instance, is a real, working subcommand that the help
> text omits. Absence from `--help` is not by itself proof a verb is gone;
> running it is.) The signed
> data these verbs operated on is still in the tree
> (`bootstrap/artifacts/manifest.v2.json`, `bootstrap/policies/policy.v1.json`),
> but nothing in the shipped compiler reads it, and every in-repo wrapper that
> still calls these verbs (`scripts/selfhost/selfhost_independence_gate.sh`,
> `scripts/selfhost/selfhost_cycle_gate.sh`,
> `scripts/omega/omega_strict_closure_gate.sh`,
> `scripts/bootstrap/bootstrap_verify_artifacts.sh`,
> `scripts/omega/omega_prepare_policy_smoke.sh`,
> `scripts/omega/omega_policy_status.sh`,
> `scripts/bootstrap/omega_canonical_policy_sign.sh`,
> `scripts/dev/diverse_double_compile_check.sh`) is dead for the same reason.
> The bootstrap chain that is actually exercised today is
> `scripts/ci/bootstrap_chain_gate.sh`, which drives `bootstrap/stage0.c` ->
> `bootstrap/boot1.sio` directly without any `souc` subcommand, alongside
> `scripts/bootstrap/bootstrap_full_gate.sh`. Seed rebuilds go through
> `scripts/bootstrap/build_bootstrap_seed.sh` and
> `scripts/dev/refresh_lean_seed.sh`. The optimization-policy lane has no
> surviving entrypoint at all. The list below is retained for lineage only.

- `souc bootstrap verify --bundle <dir>`
- `souc bootstrap init --bundle <dir> --state <dir>`
- `souc bootstrap cycle --state <dir>`

The signed bundle contract is defined by `bootstrap/artifacts/manifest.v2.json`
(`schema = "sounio.bootstrap.manifest.v2"`), with Ed25519 signatures on:

- each artifact (`<artifact>.sig`)
- the manifest (`manifest.v2.json.sig`)

Optimization policy contract is defined by
`bootstrap/policies/policy.v1.json`
(`schema = "sounio.optimization.policy.v1"`). The file is still in the tree, but
the CLI that promoted and evaluated it is gone with the Rust crates (see the
note above); these four invocations are retained for lineage only:

- `souc opt policy train --corpus <path> --output <file>`
- `souc opt policy eval --policy <file>`
- `souc opt policy promote --policy <file> --output <file>`
- `souc opt policy status --policy <file>`

Performance release gating contract is defined by
`benchmarks/independence/contract.v1.json`
(`schema = "sounio.independence.contract.v1"`).

Legacy transition env contracts are removed and now hard-error with migration
guidance:

- `SOUNIO_SELFHOST_PIPELINE`
- `SOUNIO_RUST_GHOST`
- `SOUNIO_SELFHOST_NO_RUST_FALLBACK`
- `SOUNIO_SELFHOST_NO_RUST_HARNESS`
- `SOUNIO_SELFHOST_DRIVER_REQUIRE_OUTPUT`

## Artifact

- Default path: `bootstrap/seeds/sounio-bootstrap-linux-x86_64.sio.bin`
- Sidecars:
  - checksum: `bootstrap/seeds/sounio-bootstrap-linux-x86_64.sio.bin.sha256`
  - signature marker: `bootstrap/seeds/sounio-bootstrap-linux-x86_64.sio.bin.sig`

## Binary Format (`v1`)

Header (20 bytes, little-endian):

1. Magic: 8 bytes (`SNSDSEED`)
2. Version: `u16` (`1`)
3. Reserved: `u16` (`0`)
4. Payload length: `u64`

Payload:

- Serialized `Vec<Bytecode>` bytes compatible with `crate::vm::serialize`.

## Integrity and Signature Policy

Loader validation:

1. Read seed file.
2. Verify SHA-256 against `.sha256` sidecar.
3. Verify `.sig` sidecar contains:
   - marker `SOUNIO-SEED-SIG-V1`
   - `key=<trusted-key>`
   - `sha256=<digest>` matching the seed SHA-256.
4. Decode header and payload.
5. Enforce:
   - `reserved == 0` in seed header
   - payload length <= max cache payload budget (`DIR_BYTECODE_CACHE_MAX_BYTES`)

Trusted key source:

- env: `SOUNIO_BOOTSTRAP_SEED_TRUSTED_KEY`
- default: `sounio-dev`

This is a transition policy: the `.sig` file is an attestation envelope with
digest binding. A future phase can replace it with asymmetric signature
verification without changing the seed payload format.

## Failure Behavior

When seed enforcement is enabled (`SOUNIO_BOOTSTRAP_SEED_ENFORCE=1`), loader
fails hard on:

- missing seed
- checksum mismatch/parse failure
- signature marker/key/digest mismatch
- invalid header/magic/version/reserved/length
- payload exceeds max seed payload budget
- payload decode failure

When enforcement is disabled, the loader warns and falls back to dynamic
self-host compilation paths.

## Wrapper Behavior (Seed-Enforced)

For `souc run self-hosted/`, seed-enforced wrapper mode activates when either:

- `SOUNIO_BOOTSTRAP_SEED_ENFORCE=1`.

In seed-enforced wrapper mode:

1. Wrapper resolves `self-hosted/main.sio`.
2. Wrapper emits a deterministic preflight skip marker:
   `SELFHOST=run schema=v1 event=selfhost_preflight status=skipped ... reason=seed_enforced`
3. Wrapper delegates directly to seed loading (`SELFHOST=seed ...`) without
   Rust preflight AST parsing of the self-hosted suite.

## Build Script

Generate/update seed artifact with:

```bash
bash scripts/bootstrap/build_bootstrap_seed.sh
```

This script:

1. builds `souc`
2. compiles the self-hosted bootstrap kernel profile via `SOUNIO_SELFHOST_BOOTSTRAP_MANIFEST`
   (default: `bootstrap/selfhost-kernel.manifest`)
3. forces self-hosted directory cache emission (`.sounio_bytecode.sobc`)
4. wraps it into seed binary format v1
5. emits `.sha256` and `.sig` sidecars

Relevant environment overrides:

- `BOOTSTRAP_KERNEL_MANIFEST_PATH`: explicit manifest path used by the seed build script.
- `SOUNIO_SELFHOST_BOOTSTRAP_MANIFEST`: alternate manifest override; if set, it is used as
  the default for `BOOTSTRAP_KERNEL_MANIFEST_PATH`.
- `SOUNIO_BOOTSTRAP_SEED_TRUSTED_KEY`: key label emitted into `.sig` and required by loader.

---

## §4 Diverse Double-Compilation Guarantee

### Threat model

Ken Thompson's *Reflections on Trusting Trust* (1984) demonstrated that a
compiler can be trojaned to silently inject backdoors into any binary it
produces — including the next generation of itself — while showing clean source
code to human reviewers. A SHA-256 checksum or code review cannot detect a
backdoor that lives only in a compiled artifact.

**Diverse Double-Compilation (DDC)** defeats this attack: if a backdoor was
planted by host compiler A, compiling `souc` with a *different* host compiler B
will produce a binary without the backdoor. When both binaries are then run on
the **same** Sounio source, their outputs must be **byte-identical**. Divergence
is an unambiguous signal of compromise (or a determinism bug, which is itself a
defect worth fixing).

Reference: Wheeler, D. (2009). *Fully Countering Trusting Trust through
Diverse Double-Compilation*. PhD dissertation, George Mason University.

### Guarantee

Before any tagged release of Sounio, the following command **must exit 0**:

```bash
bash scripts/dev/diverse_double_compile_check.sh
```

This builds `souc` under two independent host toolchains:

| Variant | Linker       | `CARGO_TARGET_DIR`  |
|---------|--------------|---------------------|
| GCC     | GCC 13.3.0   | `target/ddc-gcc/`   |
| Clang   | Clang 18.1.3 | `target/ddc-clang/` |

Both binaries are executed on the same minimal Sounio reference program:

```sio
fn main() with IO {
    print("DDC-REF-PASS\n")
    print_int(6 * 7)
    print("\n")
}
```

If their combined stdout+stderr outputs are byte-identical the script prints
`DDC_CHECK_PASS` and exits 0. Any divergence causes an immediate `DDC_CHECK_FAIL`
with a line-level diff and exit code 1.

### Relationship to the bootstrap seed

The bootstrap seed (`bootstrap/seeds/sounio-bootstrap-linux-x86_64.sio.bin`)
is itself produced by `souc`. DDC provides an independent evidence trail that
the `souc` binary used to generate the seed was not trojaned: if both the
GCC-built and Clang-built `souc` produce identical seed artifacts, neither
compiler could have silently modified the Sounio code.

### Running the check

```bash
# Full check (two cargo builds, ~5–10 min):
bash scripts/dev/diverse_double_compile_check.sh

# Skip rebuild if binaries already exist:
bash scripts/dev/diverse_double_compile_check.sh --skip-build

# Use a custom reference program:
bash scripts/dev/diverse_double_compile_check.sh --ref-program path/to/program.sio

# Rust integration test (requires pre-built DDC binaries):
cargo test -p souc --test ddc_check -- --include-ignored
```

### Environment

| Variable             | Default           | Purpose                                   |
|----------------------|-------------------|-------------------------------------------|
| `GCC_TARGET_DIR`     | `target/ddc-gcc`  | Output dir for GCC-linked build           |
| `CLANG_TARGET_DIR`   | `target/ddc-clang`| Output dir for Clang-linked build         |
| `SOUNIO_STDLIB_PATH` | `./stdlib`        | Stdlib path passed to both binaries       |
| `WORK_DIR`           | `/tmp/sounio-ddc` | Working directory for temp files and logs |
