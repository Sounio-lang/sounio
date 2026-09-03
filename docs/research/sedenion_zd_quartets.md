<!-- docs:meta
topic_id: repo.docs.research.sedenion-zd-quartets
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.sedenion-zd-quartets
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# The 42 support-quartets of the sedenion ZD geometry (executed exactly)

**One line.** The 168 unordered sedenion zero-divisor pairs group by support-union into exactly **42
quartets**, each a 4-set with **2 lower `{1..7}` + 2 upper `{8..15}` indices**, each hosting exactly
**4 pairs** (`42 × 4 = 168`). This is the second factorization of the ZD `168` — alongside `7 × 24`
(the fibers, `sedenion_zd_fibers.md`) — executed exactly and verified on three independent legs.

## Setup

Each zero-divisor pair `(a, b)` of two-support primitives `a = e_alo ± e_ahi`, `b = e_blo ± e_bhi`
has a **support union** `{alo, ahi, blo, bhi}`. Grouping the 168 pairs by this 4-set gives the
"support quartets" of the zero-divisor geometry report (`SEDENION_ZERO_DIVISOR_GEOMETRY_REPORT.md`).

## Result

| Quantity | Value |
|---|---|
| unordered zero-divisor pairs | 168 |
| distinct support-quartets | **42** |
| pairs per quartet | exactly **4** (`42 × 4 = 168`) |
| shape of each quartet | **2 lower + 2 upper** indices |

Every quartet is a 4-element index set with exactly two indices in `{1..7}` and two in `{8..15}`, and
each such quartet hosts exactly four of the 168 pairs. (Each quartet also spans exactly two of the
seven `L = lo⊕hi` fibers, tying the `42 × 4` factorization to the `7 × 24` one.)

## Honest scope

The `42` support-quartets and the `42 × 4 = 168` factorization are from the Python geometry report.
This brick's delta: it **executes** the quartet grouping in Sounio (decidable ℤ-equality), emits the
**42 specific quartet bitmasks**, and cross-verifies them on three legs. All three legs transcribe the
same Cayley–Dickson sign law (implementation-agreement); Lean `native_decide` is the independent
checker.

## Certification

- **Executed in Sounio:** `tests/run-pass/sedenion_zd_quartets.sio` (self-contained, no #637).
  Verdict `QUARTETS OK` (`PAIRS 168 / QUARTETS 42 / BAD_SIZE 0 / BAD_COUNT 0`).
- **Cross-toolchain:** `scripts/ci/sedenion_zd_quartets_gate.sh` — souc summary vs the Python oracle
  (which emits the 42 specific quartet bitmasks). Registered in CI (Contracts); under `bin/souc` and stage2.
- **Lean:** `formal/lean4/SounioSedenionQuartets.lean` `native_decide`-proves `pairs_168`,
  `quartets_42`, and `quartets_structure` (every quartet is a 2-lower/2-upper 4-set hosting 4 pairs).
  `lake build` green; verified by the Lean Proofs CI job.

## Reproduce

```bash
SOUNIO_STDLIB_PATH=$PWD/stdlib ./bin/souc run tests/run-pass/sedenion_zd_quartets.sio
python3 scripts/research/sedenion_zd_quartets_oracle.py
bash scripts/ci/sedenion_zd_quartets_gate.sh
(cd formal/lean4 && lake build SounioSedenionQuartets)
```
