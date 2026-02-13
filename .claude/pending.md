# Pending Work & Open Questions

## In Progress

### Phase 4: Cleanup and Extraction (45% complete)

- [x] Extract SOIR library (`crates/soir/`) - COMPLETE
- [ ] Extract Poseidon VM wrapper (`crates/poseidon-vm/`)
- [ ] Clean up self-hosted code (add doc comments)
- [ ] Add property-based tests and fuzzing
- [ ] Benchmark and optimize critical paths

See: `.claude/phase4_progress.md` for detailed status

## Next (Rustless Cutover)

- [x] Add a CI gate that runs the self-hosted suite via self-hosted pipeline in strict mode (no Rust oracle)
  - **DONE**: Phase 3 - Added `rustless-e2e` CI job with 10 comprehensive tests
- [ ] Wire `stdlib/compiler/bootstrap/verify.sio` to real compilation outputs (Stage 1/Stage 2)
  - **IN PROGRESS**: Phase 1 (verification pipeline integration)
- [x] Decide the non-Rust "Stage 0" runner target (C vs Zig vs Sounio-native), and document the choice
  - **DECIDED**: C-based VM (poseidon) for universal portability
- [x] Define the canonical artifact boundary (bytecode vs IR vs ELF) for Stage 1/2 comparisons
  - **DECIDED**: IR (IrModule) with SOIR serialization for verification

## Blocked

<!-- Items waiting on external input or dependencies -->

## Open Questions

<!-- Unresolved design questions needing discussion -->

1. Should the rustless end-state runner execute:
   - bytecode (keep SOBC stable), or
   - self-host IR directly, or
   - native ELF only?
2. Do we want hypercomplex builtins to remain VM-builtins long-term, or migrate into stdlib + compiler intrinsics?

## Deferred

<!-- Explicitly postponed items with rationale -->
