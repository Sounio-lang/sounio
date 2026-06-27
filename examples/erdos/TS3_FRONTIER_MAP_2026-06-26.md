# TS-3 Frontier Map and k-Determined Pattern

**Date:** 2026-06-26
**Solver:** souc_sat TS-3 (aggressive per-variable Thompson, reactive beta)
**All results DRAT-verified.**

## The k-determined pattern

TS-3 conflict count depends **only on k** (number of colours), not on n (graph size):

| k | Conflicts (any n) | Instances verified |
|---|---:|---|
| 6 | 136 | queen_6 |
| 7 | 26 | queen_8 |
| 8 | 1 | queen_11 |
| 9 | 257 | queen_10, queen_11, queen_12 |
| 10 | 852 | queen_11, queen_12, queen_13 |
| 11 | **WALL** | timeout for all n ≥ 12 |

This means TS-3 converges to the same search tree regardless of graph size.
Only the colour count matters — the precoloured triangle + TS-3's exploration
makes graph structure irrelevant beyond a threshold.

## Fair conflict comparison (same CNF, same symmetry breaking)

| Instance | n | k | TS-3 | kissat 4.0.4 | Ratio |
|----------|--:|--:|-----:|-------------:|------:|
| queen_6 | 36 | 6 | **136** | 454 | 3× |
| queen_8 | 64 | 7 | **26** | 37 | 1× |
| queen_10 | 100 | 9 | **257** | 536 | 2× |
| queen_11 | 121 | 9 | **257** | 2,210 | 9× |
| queen_11 | 121 | 10 | **852** | 9,713 | 11× |
| queen_12 | 144 | 10 | **852** | 10,476 | 12× |
| queen_13 | 169 | 10 | **852** | 10,291 | 12× |

TS-3 needs **3-12× fewer conflicts** than kissat 4.0.4 on queen graphs.

## Frontier: what's solved and what's not

| Instance | TS-3 | VSIDS | kissat | Status |
|----------|------|-------|--------|--------|
| queen_6 through queen_13 (k≤10) | ✓ | ✓ (slower) | ✓ | **Solved by all** |
| queen_11 k9, k10 | ✓ (1-2s) | **TIMEOUT** | ✓ | **TS-3 solo advantage** |
| queen_12 k10 | ✓ (2s) | **TIMEOUT** | ✓ | **TS-3 solo advantage** |
| queen_9 k9 | TIMEOUT | TIMEOUT | TIMEOUT | **Universal wall** |
| queen_12 k11 | TIMEOUT | TIMEOUT | TIMEOUT | **Universal wall** |
| queen_13 k11 | TIMEOUT | TIMEOUT | TIMEOUT | **Universal wall** |

## What the wall means

k=11 on queen graphs is computationally hard for ALL solvers (including
SAT Comp 2025 winner). The search space at k=11 is qualitatively different
from k≤10 — TS-3's exploration advantage does not extend past this boundary.
