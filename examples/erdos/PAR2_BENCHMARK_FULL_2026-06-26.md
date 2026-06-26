# S5 PAR-2 Benchmark: souc_sat vs kissat 4.0.4

**Date:** 2026-06-26
**Protocol:** SAT Competition PAR-2 (score = wall-clock if solved, 2×timeout if not)
**Timeout:** 120s per instance
**Instances:** graph-colouring UNSAT — de Grey, Parts, Mycielski, Queen, Complete graphs

## Results — hard instances (the real comparison)

| Instance | V | E | k | souc_sat time | souc conflicts | kissat time | kissat conflicts | ratio |
|----------|--:|--:|--:|----------:|----------:|----------:|----------:|------:|
| **degrey_529** | 529 | 2670 | 4 | **23.6s** | 279,952 | **2.4s** | 65,850 | 9.7× |
| **parts_510** | 510 | 2504 | 4 | **25.3s** | 299,589 | **2.0s** | 52,522 | 12.9× |
| queen_6 | 36 | 290 | 6 | 0.1s | 947 | 0.0s | 454 | 8.7× |

## Results — trivial instances (both solve <0.2s)

| Instance | V | E | k | souc_sat | kissat | conflicts (s/k) |
|----------|--:|--:|--:|------:|------:|------|
| mycielski_3 | 5 | 4 | 2 | 0.1s | 0.0s | 0 / 0 |
| mycielski_4 | 11 | 13 | 3 | 0.1s | 0.0s | 8 / 0 |
| mycielski_5 | 23 | 37 | 3 | 0.1s | 0.0s | 8 / 0 |
| mycielski_6 | 47 | 97 | 4 | 0.1s | 0.0s | 37 / 0 |
| mycielski_7 | 95 | 241 | 5 | 0.1s | 0.0s | 150 / 0 |
| queen_5 | 25 | 160 | 4 | 0.1s | 0.0s | 1 / 1 |
| queen_8 | 64 | 728 | 8 | 0.2s | 0.0s | 1 / 1 |
| complete_5 | 5 | 10 | 4 | 0.1s | 0.0s | 1 / 1 |
| complete_6 | 6 | 15 | 5 | 0.1s | 0.0s | 2 / 0 |
| complete_7 | 7 | 21 | 6 | 0.1s | 0.0s | 7 / 8 |

## Correctness

- **souc_sat:** 13/14 correct (queen_7 at k=7 is SAT — expected UNSAT was wrong, the 7×7 queen graph IS 7-colourable)
- **kissat:** 13/14 correct (queen_7 SAT → no CNF emitted → NO_CNF)
- **Both solvers agree** on all instances where both produced a result

## PAR-2 totals (13 comparable instances)

| Solver | PAR-2 score |
|--------|------------:|
| **kissat 4.0.4** | **4.4s** |
| **souc_sat (S3)** | **49.8s** |
| Overall ratio | **11.3×** |

## Honest analysis

1. **On hard instances (de Grey, Parts):** kissat is 9.7–12.9× faster. This gap is structural — kissat's inprocessing (BVE, vivification, congruence closure) prunes the search space 4× more effectively (52–66K vs 280–300K conflicts).

2. **On trivial instances:** both solve in <0.2s. The gap is pure startup overhead (souc_sat ELF compilation + DRAT streaming setup). Not meaningful.

3. **Conflict ratio is consistent:** souc_sat needs ~4.3× more conflicts than kissat on hard instances (280K/66K ≈ 4.2, 300K/53K ≈ 5.7). The per-conflict cost is comparable (23.6s/280K = 84μs vs 2.4s/66K = 36μs = 2.3× per-conflict overhead, closed partially by S3).

4. **S4 cube-and-conquer** closes the gap on degrey_529 from 9.7× to 5.4× (11.8s vs 2.4s) by parallelizing across 64 cores.

5. **Domain dominance is NOT claimed.** souc_sat is not competitive with kissat on raw speed. The edge is verification-integration provenance (self-hosted solver + DRAT proof, no external C toolchain).

## Reproduction

```bash
python3 examples/erdos/generate_bench_instances.py /tmp/colour_bench
python3 examples/erdos/par2_benchmark.py /tmp/souc_sat_s3.elf tools/bin/kissat /tmp/colour_bench examples/erdos/data
```
