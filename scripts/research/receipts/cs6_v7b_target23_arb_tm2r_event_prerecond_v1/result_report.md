# Target-23 pre-QR predictor-centered residual gate

## Objective

Accept a predictor-centered residual event chart on the frozen XLEL critical leaf
using **raw-projection** six-variable symbolic weights, after the residual `rho3`
locus diagnosis proved that post-QR renumbering collapses residual rank to 3.

## Foundry execution

| Field | Value |
|---|---|
| Job | `8763` |
| Node | `gpuorangefs-r770-proxmox` |
| Snapshot | `ce1853d123` |
| Stage | `/orangefs/training/cs6-tm2r-event-prerecond-ce1853d123-20260807T030638Z` |
| Elapsed | `00:07:41` |
| Exit | `0:0` |
| Receipt SHA-256 | `4b615c5632ba9537d639d4fe831c924aff1586a0d4a9db1f2f4efd9c1f1daa3a` |
| Verifier SHA-256 | `ae95a5cffb2cb8faf3b57c9f25708573c6d897a7d3963348023a8efb66fb55c9` |

Prior job `8762` produced the same scientific acceptance but failed verify on
Arb-sum widening of `combined_event_time`; forensics under `foundry_failure_8762/`.

Local independent re-verification and **14/14** negative mutations pass.

## Result

Classification: **`PREDICTOR_CENTERED_PRERECOND_EVENT_ACCEPTED`**.

| Gate | Outcome |
|---|---|
| Implementation checks | **26/26** |
| Accepted residual power | **`2^-12`** |
| Picard / Newton residual | pass |
| Exact section `w=0` | pass |
| Strict negative normal | pass |
| Raw-projection six-variable weights | **pass (ρ₃ positive)** |
| Pure source monomials retained (raw) | **15** |
| Post-QR residual rank (forensic) | **3** |
| Post-QR ρ₃ (forensic) | false |

Powers 18…13 remain `CENTERED_PREDICTOR_ESCAPED`; power 12 is the first accepted
scale under the pre-QR symbolic policy.

## Interpretation

The earlier event-centered refusal was an artifact of measuring symbolic
dependence after QR residual renumbering. With dependence measured on the raw
projected carrier, the residual event chart closes with all six parameters and
the geometric residual gates intact.

This **authorizes residual event-local acceptance only**. It does **not**
certify a complete support transport, h-set covering relation, recurrent graph,
degree/determinant edge, chaos, or open-problem solution.

## Next falsifier

Use the accepted raw-projected residual carrier as the source for the next
complete-transport / composability attack (second section return), preserving
the six-variable dependence and without reintroducing post-QR symbolic
acceptance.

## Claim boundary

No covering relation, recurrent graph, chaos, novelty, priority, or open-problem
solution is claimed. Production and legacy transport paths remain unchanged.
