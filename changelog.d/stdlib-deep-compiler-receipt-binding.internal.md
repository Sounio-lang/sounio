`scripts/ci/stdlib_deep_compiler_receipt.sh` now binds the compiler it uses to its build inputs and verifies that receipt before running the compiler. The inputs bound are:
- the source revision;
- the closure hashes, and each tracked symlink's target;
- the engine and target;
- the bootstrap path and hash;
- the toolchain, including the build-environment fingerprint the Madaros build cache keys on.

No user-visible change.
