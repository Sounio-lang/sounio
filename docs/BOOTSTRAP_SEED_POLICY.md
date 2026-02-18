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
   - digest matching the seed SHA-256.
4. Decode header and payload.

This is a transition policy: the `.sig` file is an attestation envelope with
digest binding. A future phase can replace it with asymmetric signature
verification without changing the seed payload format.

## Failure Behavior

When seed enforcement is enabled (`SOUNIO_BOOTSTRAP_SEED_ENFORCE=1`), loader
fails hard on:

- missing seed
- checksum mismatch/parse failure
- signature marker/digest mismatch
- invalid header/magic/version/length
- payload decode failure

When enforcement is disabled, the loader warns and falls back to dynamic
self-host compilation paths.

## Build Script

Generate/update seed artifact with:

```bash
bash scripts/build_bootstrap_seed.sh
```

This script:

1. builds `souc`
2. forces self-hosted directory cache emission (`.sounio_bytecode.sobc`)
3. wraps it into seed binary format v1
4. emits `.sha256` and `.sig` sidecars
