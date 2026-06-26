# PAR-2 Benchmark: souc_sat vs kissat 4.0.4 on G₅₂₉ 4-colouring UNSAT

**Date:** 2026-06-26
**Instance:** Heule's 529-vertex de Grey graph, 4-colouring refutation
**CNF:** 2116 variables, 11212 clauses (identical input for both solvers)
**Both emit DRAT proofs; drat-trim verifies souc_sat's proof.**

## Results

| Solver | Time (s) | Conflicts | Propagations | DRAT proof size | Proof verified |
|--------|----------|-----------|-------------|-----------------|---------------|
| **kissat 4.0.4** | **2.2** | 57,362 | 11,221,384 | 3.8 MB | (drat-trim compatible) |
| **souc_sat** | **27.8** | 283,804 | 58,733,384 | 61.5 MB | drat-trim `s VERIFIED` (4.5s) |

**Speedup (kissat/souc_sat):** ~12.6× wall-clock, ~4.9× fewer conflicts.

## Honest assessment

kissat 4.0.4 is **12.6× faster** on this instance. This is expected and not a
disgrace — kissat is the SAT Competition 2025 baseline with 15+ years of tuned
inprocessing (BVE, vivification, congruence closure, bounded variable addition).
souc_sat is a from-scratch CDCL with no inprocessing.

The honest positioning from `SOTA_LITERATURE_AND_PLAN_2026-05.md` holds:
> "Beat kissat in general is not a near-term truth."

**What souc_sat has that kissat does not:** self-hosted provenance. The solver,
the CNF, and the DRAT proof are all produced by the Sounio compiler toolchain
with no external C toolchain in the critical path (beyond the bootstrap).
This is the verification-integration edge, not a performance edge.

## drat-trim verification of souc_sat proof

```
c parsing input formula with 2116 variables and 11212 clauses
c 9686 of 11212 clauses in core
c 80819 of 283805 lemmas in core using 5711499 resolution steps
s VERIFIED
c verification time: 4.542 seconds
```

## Gap analysis — where the 12.6× comes from

From the SOTA doc lever plan (S1–S4):
- **S1 (SBVA preprocessing):** not yet wired — could reduce clause count
- **S2 (BVE/vivification inprocessing):** not implemented — the dominant lever
- **S3 (arena clause layout):** litval dominates at 1.2B calls — constant-factor win available
- **S5 (this benchmark):** now done — establishes the real gap to close

## Reproduction

```bash
# souc_sat (compile + run)
SOUNIO_SOUC_ENGINE=lean_single ./bin/souc compile examples/erdos/souc_sat.sio -o /tmp/souc_sat.elf
chmod +x /tmp/souc_sat.elf
time /tmp/souc_sat.elf 42 4 0 1 examples/erdos/data/degrey_529.edge

# kissat 4.0.4
time tools/bin/kissat souc_sat_worker.cnf

# drat-trim verification
tools/bin/drat-trim souc_sat_worker.cnf souc_sat_worker.drat
```
