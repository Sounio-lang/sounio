# CS6 V7-B target-23 depth-4 sibling cover

**Status:** pre-execution frozen protocol. No cover result is available in this
revision.

## Plain question

The previous ladder found that the lower-left descendant of target cell `23`
emits `PROBE_PASS=true` at depth delta `+4`, for both tested carriers. One
passing descendant does not tell us what happens in its 255 siblings.

This experiment evaluates the entire `16 x 16` dyadic partition at that same
depth. Every one of the 256 children is run with both carriers, producing 512
required attempts.

## Verdicts

The result must distinguish three levels:

- `PARENT_COVER_EVALUATED=true` means all 512 required attempts produced a
  classified result with no unknown failures;
- `PARENT_PROBE_COVER_PASS=true` requires `PROBE_PASS=true` in every attempt;
- `PARENT_CERTIFICATE_COVER_PASS=true` requires `CERTIFICATE_PASS=true` in
  every attempt.

A partial collection of passing children is only a localization map. It cannot
establish a parent cover. Even a complete probe cover does not by itself enable
V7-B eligibility. The parent V7-B contract independently requires a nonempty
intersection of the Liouville, C1, C2, and section-resident determinant
enclosures, followed by its frozen winner criterion. This experiment does not
evaluate that joint intersection or select a winner.

## Frozen matrix

```text
PARENT_NODE=U03-0000000006_S04-0000000010
DEPTH_DELTA=4
CHILD_U_DEPTH=7
CHILD_S_DEPTH=8
CHILD_OFFSETS=0..15 x 0..15
CHILD_COUNT=256
CARRIERS=2
ATTEMPT_COUNT=512
```

Heavy execution is routed through Slurm with 32 CPU workers. No FPGA is used.

## Local protocol gate

```bash
bash scripts/ci/cs6_v7b_target23_depth4_cover_gate.sh
```

The gate validates the exact partition, coordinate uniqueness, attempt count,
claim fences, Python syntax, and Slurm job syntax. It does not execute CAPD and
does not claim a scientific result.
