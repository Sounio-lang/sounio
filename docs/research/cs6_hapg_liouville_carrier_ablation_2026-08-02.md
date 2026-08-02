# CS6 V7-A Liouville carrier ablation bounded result

**Status:** the exact 120-attempt Slurm run completed and both retained audits
passed, but the frozen experiment is invalid rather than GO or scientific
NO-GO. Both alternative carriers produced complete verified receipts on 23 of
24 target cells. On the remaining cell, both alternative attempts terminated
with a later, carrier-independent one-step crossing signature instead of the baseline
tripleton signature. The bound source localizes where that later exception is
thrown, but no Liouville checkpoint was emitted. The pre-result contract
correctly refuses to classify the cell post hoc.

## Question and frozen boundary

V7-A asked whether replacing only the four-dimensional Liouville carrier could
remove the `C0HOTripletonSet` `rQ=[-nan,-nan]` failure without changing the ODE,
section, order, return count, tile, challenge, or the rest of the H-PG worker.
The frozen design contained 40 cells and three carriers:

- 24 cells with the exact parent CAPD-set signature;
- eight verifier-complete controls without a signed chart;
- eight positive H-PG controls;
- `C0HOTripletonSet`, `C0HORect2Set`, and `C0Rect2Set` on every cell.

The contract SHA-256 is
`decf9089e1dc9aae513f48c48a00e1c815a585b6ba7e9cd1c09b0b514fd58481`.
The coordinate manifest SHA-256 is
`df665eceee8a45ea687a9f0bb643fe9fef28c800650482092be113f24fa41fdd`.
The root challenge is
`47d47736b1c9c181f2041982c4ceb3cb3becf79a66b8740d990212d1a19eadc4`.
No retry, carrier fallback, cell substitution, or early stop was allowed.

The GO threshold required 40/40 verified receipts and 24/24 target repairs for
one alternative. An unknown failure makes the run invalid. Partial improvement
cannot select a V7-B winner.

## Execution lineage

| Field | Authoritative value |
| --- | --- |
| Git commit | `988567febfcdf07acd2ad234a69649d36ff5f1e2` |
| Slurm job | `8480` |
| Node | `gpuorangefs-r770-proxmox` |
| UTC interval | `15:12:25` to `15:15:14` |
| Wall time | 169 s |
| Total CPU | 16 min 08.932 s |
| Requested/allocated CPUs | 32 / 120 exclusive |
| Peak batch RSS | 2,078,124 KiB |
| CAPD path | 5.3.0, FILIB, `O0`, CPU prebuilt |
| Worker source SHA-256 | `1b0cee7fdd4df70487af3c9ec516471298c3ae9e5f8e291cee1e8d1adc6f97fa` |
| Worker binary SHA-256 | `bb6434c012e20b7e2974b313a16459b7e32d3a86174a94d55fd4b98e5ba569e7` |
| Result archive SHA-256 | `a0839c73f84b77415f6eccbd989dc3ea85626662529ea2295c455c7c8647fec8` |

Slurm records the job as `FAILED 2:0` because runner exit code 2 is the frozen
encoding for a completed invalid/negative result. This was not an
infrastructure failure: the runner completed all 120 attempts, the retained
verifier returned 0, and the publication wrapper only then created the
canonical archive. Audit or transport failures publish only into quarantine.

The partition allocated one GPU, but the code used no GPU. No U250 or other
FPGA was installed or used.

## Exact outcome

| Carrier | Verified complete | Complete target receipts | Controls preserved | Other | Decision |
| --- | ---: | ---: | ---: | --- | --- |
| `C0HOTripletonSet` | 16/40 | 0/24 | 16/16 | 24 exact `rQ=NaN` failures | `BASELINE_VALID` |
| `C0HORect2Set` | 39/40 | 23/24 | 16/16 | 1 unknown crossing failure | `RUN_INVALID` |
| `C0Rect2Set` | 39/40 | 23/24 | 16/16 | 1 unknown crossing failure | `RUN_INVALID` |

The missing target receipt is the same ordinal-23 cell for both alternatives;
23/24 is explicitly insufficient for the predeclared 24/24 GO criterion.

All 94 complete receipts passed the exact parent verifier plus the V7 binding,
initial-hull reconstruction, finite-endpoint, strict-negative Liouville,
joint-determinant, and reference-invariance checks. Their 8,742 mutations were
all rejected. The 24 baseline negatives were bound to their carrier, attempt,
raw stderr, and exact CAPD signature. Reference-invariance checks passed on all
39 cells that produced the required alternative receipts; the remaining cell
was not checkable because neither alternative emitted a complete receipt.

Both the in-job audit and a second replay from a clean checkout created from the
Git bundle reported:

```text
AUDIT_PASS=true
ATTEMPTS_RECONSTRUCTED=120
VERIFIER_REPLAYS=94
BOUND_NEGATIVES=24
RUN_VALID=false
V7_B_WINNER=NONE
```

## The masked cell

The only incomplete coordinate was
`U03-0000000006_S04-0000000010`, ordinal 23. Its three attempts were adjacent
in the frozen matrix:

| Attempt | Carrier | Exact result |
| ---: | --- | --- |
| 67 | `C0HOTripletonSet` | `CenteredTripletonSet::evalAffineFunctional` empty intersection, `rQ=[-nan,-nan]` |
| 68 | `C0HORect2Set` | `one-step Newton crossing was not available` |
| 69 | `C0Rect2Set` | `one-step Newton crossing was not available` |

This is not evidence that the carrier alternatives reproduced the tripleton
failure. The byte-bound worker deterministically evaluates all C1/C2 returns,
then calls
`liouville_two_return(input, carrier)`, and only later calls the shared
section-resident return. The latter is the only source of the exact one-step
exception in this worker. The exact alternative stderr and bound sequential
source therefore localize the thrown exception after the Liouville call site,
while the baseline stopped inside that call with the tripleton exception. This
is control-flow localization only: without the un-emitted numerical values it
is not evidence of a complete 24th repair or of a valid Liouville result. In
particular, changing the exception does not establish that either alternative
computed a sound Liouville enclosure; that is the explicit soundness gap
V7-A.1 must test.

That control-flow fact localizes the mechanism, but it is not a missing 40th
certificate: the worker emits its ledger only after all computations complete.
There is no serialized, independently replayable Liouville checkpoint for this
cell. Calling it a 24/24 GO after seeing the failure would violate the frozen
contract.

## What was learned

On 23 of 24 fully witnessed targets, both doubleton families produced complete,
finite, determinant-compatible receipts where the baseline produced the exact
tripleton exception; both also preserved all 16 controls. This is the complete
contrast licensed by the ablation. The 24th target remains uncertified
and is not counted as a repair. The Taylor and Hermite-Obreschkoff doubletons
had identical binary outcomes on this matrix; no enclosure-width, tightness, or
propagated-set equivalence is claimed. The result narrows the next diagnostic
to one already known crossing cell without closing it.

This conclusion is conditioned on one predeclared 40-cell matrix and one
execution per attempt. It supplies no replication rate, sensitivity estimate,
or carrier ranking. Repetition under a separately frozen matrix would be
required before generalizing the 23-cell contrast.

The defensible result is **inconclusive partial improvement with a precisely
localized masked blocker**. It is not a carrier proof, full-source proof,
hyperbolicity result, attractor result, solved open problem, or novelty/priority
claim. V7-B remains blocked because the predeclared winner set is empty.

## Next experiment

The next smallest experiment is a separately frozen V7-A.1, not the
228-evaluation depth tomography. Freeze a carrier-only checkpoint before any
downstream section-resident work. It should emit and independently verify the initial
hull, two-return time/image, normal velocities, `ell`, `exp(ell)`, and the
oriented Liouville determinant for the masked cell under all three carriers,
with nearby complete cells as positive controls and carrier/binding mutations.

This new test can distinguish a complete 24th carrier repair from an unobserved
Liouville defect using a single-digit evaluation budget. It may inform the
mechanism prospectively, but it cannot amend, reinterpret, or retroactively turn
job 8480 into a frozen V7-A GO, nor can it nominate a V7-B winner under the V7-A
contract.

## Retention

Compact receipts are retained under
`scripts/research/receipts/cs6_hapg_liouville_carrier_ablation_job_8480_v1/`.
They include the exact 120-row matrix, decisions, summary, contracts, raw-file
indexes, provenance, final `sacct`, audit outputs, and the three masked-cell
stderr records as `repro/A0067-tripleton.stderr.txt.gz` (deterministic gzip of
the exact raw bytes),
`repro/A0068-ho-rect2.stderr.txt`, and `repro/A0069-rect2.stderr.txt`. The
4,577,280-byte canonical archive remains at
`/orangefs/training/cs6-hapg-cover/55c331622b6076ac/v7-carrier-988567feb/result.tar`;
its bytes are not committed, and its digest is retained above and in the
receipt manifest. The committed material is an audit transcript plus a compact
result subset, not a self-contained replay package. Full third-party replay of
all 94 raw receipts still depends on access to that archive and the anchored
prebuilt. The CPU resource figures above are provenance, not a claim of
efficiency or necessity; no CPU/GPU/FPGA performance comparison was made.
