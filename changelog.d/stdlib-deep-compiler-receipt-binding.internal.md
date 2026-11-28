`scripts/ci/stdlib_deep_compiler_receipt.sh` now binds the compiler it uses to its build inputs and verifies that receipt before running the compiler. The inputs bound are:
- the source revision;
- the closure hashes, and each tracked symlink's target;
- the engine and target;
- the bootstrap path and hash;
- the toolchain, including the build-environment fingerprint the Madaros build cache keys on.

No user-visible change.

`TOOLCHAIN.txt` now names the producer of the compiler bytes, not the
runner that wrote the receipt. On a Madaros build-cache hit those differ.
`scripts/dev/madaros-cache.sh` stores the producer manifest with each seed
and Madaros entry and writes it beside the output as `<out>.toolchain`.
Entries stored without a manifest are rebuilt. The orchestration script's
bytes are now a field of both cache keys (`seed-v3`, `madaros-v3`), so a
prefix-restored entry from another script version is a miss. The receipt
refuses a compiler whose manifest names another build script or `SOUNIO_*`
environment.
