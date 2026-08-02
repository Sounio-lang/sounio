# CS6 V7-B bridge execution checkpoint

**Status:** six-attempt execution checkpoint. This is the first run-facing step
after the V7-B bridge freeze. It does not change the frozen V7-A.1 evidence and
does not claim V7-B eligibility.

## Plain result

The freeze said that 18 downstream rows were still missing. This checkpoint adds
a bounded runner that can ask the next concrete question:

```text
For the same three cells and the two candidate carriers, can the existing
H-PG worker get past C1/C2 and the section-resident crossing?
```

The runner reuses `scripts/research/cs6_hapg_liouville_carrier_ablation_probe.cpp`
as a read-only computation engine. It executes six attempts:

```text
3 cells x 2 candidate carriers = 6 attempts
```

Each attempt is classified as either a complete bridge probe, a known
section-resident crossing negative, a verifier rejection, or an unknown
failure. Unknown failures invalidate the run. A known
`one-step Newton crossing was not available` result is retained as a classified
negative, not treated as V7-B evidence.

The 2026-08-02 local execution result was:

```text
ATTEMPTS_COMPLETED=6
FULL_BRIDGE_PROBE_PASS=0
SECTION_RESIDENT_CROSSING_UNAVAILABLE=2
UNKNOWN_FAILURE=0
V7_B_ELIGIBILITY=false
PROMOTION_ELIGIBLE=false
OPEN_PROBLEM_SOLVED=false
```

In plain terms: all six attempts executed and were classified. The masked target
cell failed at the section-resident crossing for both candidate carriers. The two
neighbor control cells reached the worker summary but still ended with
`PROBE_PASS=false` for both carriers. This narrows the blocker but does not
produce a V7-B bridge.

## Claim boundary

This checkpoint is allowed to say that the six-attempt runner exists, compiles,
runs when CAPD is available, and classifies all attempts. It is not allowed to
say that V7-B is eligible unless all six attempts produce full bridge probes
and a later winner-scoring verifier is implemented.

The current verifier intentionally keeps:

```text
V7_B_WINNER=NONE
PROMOTION_ELIGIBLE=false
OPEN_PROBLEM_SOLVED=false
FPGA_EXECUTION=false
```

## Acceptance

The local gate is:

```bash
bash scripts/ci/cs6_v7b_full_hpg_bridge_execution_gate.sh
```

It writes receipts under
`scripts/research/receipts/cs6_v7b_full_hpg_bridge_execution_v1/`.

## Blocker update

The scientific blocker changes from "no runner exists" to "the frozen V7-B
candidate carriers do not produce a full bridge on the three-cell checkpoint."
For the masked target, the narrower blocker is still the section-resident
crossing. For the two neighboring control cells, the worker reaches the summary
but rejects the full probe.
