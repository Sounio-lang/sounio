# Bootstrap Seed Policy (R1)

This document defines the transition seed artifact used to bootstrap the
self-hosted compiler without runtime Rust compilation of the root suite.

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

- `SOUNIO_BOOTSTRAP_SEED_ENFORCE=1`, or
- transition legacy path is explicitly requested:
  `SOUNIO_SELFHOST_PIPELINE=rust` + `SOUNIO_RUST_GHOST=1`.

In seed-enforced wrapper mode:

1. Wrapper resolves `self-hosted/main.sio`.
2. Wrapper emits a deterministic preflight skip marker:
   `SELFHOST=run schema=v1 event=selfhost_preflight status=skipped ... reason=seed_enforced`
3. Wrapper delegates directly to seed loading (`SELFHOST=seed ...`) without
   Rust preflight AST parsing of the self-hosted suite.

## Build Script

Generate/update seed artifact with:

```bash
bash scripts/build_bootstrap_seed.sh
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
