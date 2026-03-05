## Summary
- add strict Sprint1 gate orchestration for critical compiler bug-fix checks
- add dedicated int_to_string perf gate with JIT-first probing and structured blockers
- extend scripts/omega/omega_resolve_souc_bin.sh with SOUNIO_SOUC_VARIANT=jit pinned asset resolution
- resolve pinned JIT souc in selfhost-regression workflow and enforce required-pass mode

## Release artifact
- published signed JIT asset release: v0.100.3-jit.1
- assets: souc-linux-x86_64-jit, .sha256, .sig

## Notes
- lane is fail-closed when JIT runner is missing or unusable
- resolver now uses atomic downloads to avoid ETXTBSY cache-write races
