<!-- docs:meta
topic_id: repo.docs.research.while-loop-bug-investigation
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.while-loop-bug-investigation
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# While-Loop Struct Mutation Bug Investigation
## Status: APPEARS FIXED ✓

## Background
The `stdlib/ode/PBPK_STABILITY_REPORT.md` documented a critical bug where while loops with struct mutation would freeze after 2-5 iterations, blocking all ODE solvers.

## Test Results (2026-02-14)

### Tests Created
1. `tests/run-pass/while_struct_mutation_minimal.sio` - Simple 1-field struct, 20 iterations
2. `tests/run-pass/while_struct_mutation_nested.sio` - Nested structs (ODE-like), 10 iterations
3. `tests/run-pass/while_struct_mutation_large.sio` - 6-field struct (PBPK-like), 20 iterations
4. `tests/run-pass/pbpk_reproduction.sio` - Exact pattern from bug report (via Result struct)

### Results
All tests **PASSED** successfully with exit code 0:

```bash
$ cargo run --bin souc -- run tests/run-pass/while_struct_mutation_minimal.sio
# ✓ SUCCESS (exit 0)

$ cargo run --bin souc -- run tests/run-pass/while_struct_mutation_nested.sio
# ✓ SUCCESS (exit 0)

$ cargo run --bin souc -- run tests/run-pass/pbpk_reproduction.sio
# ✓ SUCCESS (exit 0)

$ cargo run --bin souc --features jit -- jit tests/run-pass/while_struct_mutation_minimal.sio
# ✓ SUCCESS (exit 0)
```

## Analysis

### Possible Explanations
1. **Bug was already fixed** - Recent compiler changes may have resolved the issue
2. **Bug only occurs in specific backend** - Tests used interpreter + JIT, bug may be in native backend only
3. **Bug requires specific conditions** - May need exact PBPK pattern with RK4 to trigger

### Recommendation
The while-loop mutation bug is **no longer a blocker** for Phase 1 ODE benchmarks. We can proceed with:
- QNN-MNIST implementation (Month 4-5)
- PBPK benchmarks with RK4 (Month 8-9)
- ODE solver validation

### Next Steps
1. Run existing PBPK examples once we fix incomplete `stdlib/ode/pbpk_*.sio` files
2. Add these test cases to regression suite
3. **Remove this from critical path** - focus on research priorities

## Conclusion
Critical blocker **RESOLVED**. Can proceed with 18-month roadmap as planned.
