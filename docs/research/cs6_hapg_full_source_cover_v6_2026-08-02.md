# CS6 H-APG full-source-cover v6 bounded result

**Status:** the fresh KAT prerequisite passed, the fresh adaptive run completed,
and the frozen 255-node experiment returned a valid bounded negative result.
It did not produce an H-APG cover.

## Question and boundary

The v6 experiment asked whether the existing H-PG to fixed-chart H-APG
certificate pipeline could cover the frozen dyadic parameter square with at
most 255 nodes and eight breadth-first waves. The test was frozen at Git commit
`55c331622b6076ac688d446a15012c7046b4280a` under contract SHA-256
`874c3d7f455880aea9a76b58bb03afebf32f8f54fc31a90950fb9882c9456c8d`.

The answer is **no under this budget and protocol**. This is not a proof that no
H-APG cover exists. It is not evidence for a full-source carrier,
hyperbolicity, a chaotic attractor, or a solved open problem.

## Execution lineage

| Stage | Slurm job | State | UTC interval | Elapsed | Config SHA-256 |
| --- | ---: | --- | --- | ---: | --- |
| KAT | 8469 | `COMPLETED 0:0` | 09:36:03 to 09:37:09 | 66 s | `fde9020fa26a...` |
| adaptive | 8470 | `COMPLETED 0:0` | 09:42:12 to 09:45:29 | 197 s | `bcec9b471fff...` |

Both stages ran on `gpuorangefs-r770-proxmox` with one task, 32 requested CPUs,
and 120 CPUs allocated by the exclusive partition. The code path was CPU-only
CAPD 5.3/FILIB at `O0`. The partition allocated a GPU, but the experiment did
not use it. No U250 or other FPGA was used.

The KAT archive SHA-256 is `347047e5b5da326f5fc57871c39e5bfae62b2a358ec9ff351121b98c4786a402`.
The adaptive archive SHA-256 is `30b063c746338cdc641e36f9420fdb9b1c01c1d758d7987fc880561adcb721ac`.
The large archive bytes remain on OrangeFS and are not committed. Their compact
configs, final `sacct` rows, content indexes, summaries, contracts, logs, and
hash sidecars are retained under
`scripts/research/receipts/cs6_hapg_full_source_cover_v6_jobs_8469_8470_v1/`.

The authoritative KAT prerequisite certificate is the copy embedded in the
adaptive result, SHA-256 `7b0edca4d83dc5464a18ff67a68a7f8d23635dcc58ea60a1fbaae9e970be338f`.
It binds KAT job 8469 to adaptive job 8470. A staging-only preflight certificate
with dummy `ADAPTIVE_JOB_ID=1` was deliberately excluded from retained evidence.
Final completion claims use `sacct`; transport job records captured while jobs
were running are not completion authorities.

## KAT prerequisite

The fresh KAT evaluated 53 predeclared coordinates. The root preserved its
expected unresolved class. All 52 non-root leaves produced verified H-PG signed
charts and entered H-APG:

| Measurement | Result |
| --- | ---: |
| H-PG verifier replays | 52/52 |
| H-APG verifier replays | 52/52 |
| H-APG certified | 48 |
| H-APG uncertified | 4 |
| APG rescues | 20 |
| H-PG mutations rejected | 4108/4108 |
| H-APG mutations rejected | 5824/5824 |

This proves that the executable certificate chain still works at the retained
coordinates. It does not make those coordinates a cover. The 52 non-root KAT
cells have individual U and S depths from 8 through 16 and total depths from 16
through 30. Their raw area sum is only
`16525/268435456`, approximately `6.156e-5`, and they contain 96 strict
dyadic-containment pairs. They are not an antichain, so even that raw sum is not
a union-area certificate.

## Adaptive result

The adaptive stage evaluated the complete frozen budget:

```text
wave populations: 1, 2, 4, 8, 16, 32, 64, 128
tree nodes:        255
max total depth:   7
```

No node produced an H-PG signed chart, so no node was eligible for H-APG.

| H-PG status | Count |
| --- | ---: |
| `H_PG_INTERVAL_DOMAIN` | 1 |
| `H_PG_CROSSING` | 18 |
| `H_PG_INVALID_NO_SIGNED_CHART` | 54 |
| `H_PG_CAPD_SET` | 182 |

The controller made 85 S splits and 42 U splits. All 128 wave-7 cells ended as
`UNRESOLVED/WAVE_LIMIT`. Therefore:

```text
certified terminals = 0
unresolved terminals = 128
accepted area = 0/1
unresolved area = 1/1
LOCAL_COMPLETE_HAPG_COVER = false
```

The independent local aggregation was byte-identical to the Slurm aggregation,
rejected 17/17 aggregate mutations, and confirmed the tree and wave chain.
`FRESH_REPLAY_COMPLETE=true` is vacuous here: there were zero certified
terminals to replay.

## What the negative located

The outcome separates protocol success from numerical reach. Transport,
lineage, KAT binding, worker execution, verifier mutation resistance, tree
closure, and aggregation all passed. The first scientific obstruction is H-PG
signed-chart eligibility, not H-APG itself.

The retained 255-row raw H-PG corpus identifies the dominant operational
signature behind the 182 `H_PG_CAPD_SET` rows as
`CenteredTripletonSet::evalAffineFunctional` rejecting an empty intersection
after `rQ=[-nan,-nan]`. The C1 and C2 returns precede this call in the worker.
The exception arises in the separate four-dimensional Liouville cross-check,
where `liouville_two_return` constructs a `C0HOTripletonSet` with exactly zero
radii for the auxiliary `w` and `ell` coordinates. It is therefore an
operational set-representation failure, not a rigorous exclusion of the
underlying cell. The raw stderr corpus is already retained at
`scripts/research/receipts/cs6_hapg_full_source_cover_v3_abort_8453_v1/hpg-full255-stderr.jsonl`
with SHA-256 `c0f7700824d35eb86fb566c310188f4e5c0b5ed5006fe09091495665a1e0b6be`.

Of the 18 crossing rows, 16 report that a one-step Newton crossing was
unavailable and two have an interval transversality enclosure that crosses
zero. The 54 structured `H_PG_INVALID_NO_SIGNED_CHART` outcomes mean that the
verifier completed but did not produce the required signed chart; they are not
proofs that no mathematical chart exists.

The KAT/adaptive depth gap is decisive for the next experiment: the shallow
adaptive tree stops at total depth 7, while the first successful retained KAT
cell is at total depth 16. Location and depth are confounded, so simply spending
an exponentially larger full-tree budget would not isolate the cause.

## v7 novelty window

The negative suggests two separable causes: a tripleton-specific degeneracy in
the Liouville cross-check and depth starvation from an uninformed breadth-first
schedule. V7 should distinguish them before any larger cover run.

### V7-A: Liouville carrier ablation

Freeze 40 cells before execution:

- 24 `H_PG_CAPD_SET` cells, stratified across depths `U2/S3`, `U3/S3`, and
  `U3/S4`;
- eight verifier-complete cells without a signed chart;
- eight positive KAT controls.

Evaluate the same ODE, section, order, tile, input hash, and challenge with
three Liouville carriers: the current `C0HOTripletonSet`, `C0HORect2Set`, and
`C0Rect2Set`. The maximum is 120 H-PG evaluations. No variant is a fallback:
every result is labeled by carrier and retained independently.

The go condition is that one alternative completes all 24 baseline NaN cells,
preserves all eight positive controls, emits only finite interval state, and
keeps determinant enclosures sign-compatible with the existing C1/C2
cross-checks. Reproducing the failure on all 24 cells, or contradicting a
control sign, is a no-go. Partial improvement is inconclusive and cannot enter
the cover pipeline.

### V7-B: depth-onset tomography

Only after V7-A, freeze eight KAT anchors: the four uncertified `U8/S8` cells
and four certified cells representing the `U12/S12`, `U13/S14`, `U14/S13`,
and `U16/S12` regimes. Evaluate deterministic ancestor chains from total depth
7 to each endpoint with the baseline and at most one V7-A winner selected by a
predeclared control-preservation and width rule. Record the first depth with a
finite return, H-PG probe pass, four signed charts, H-APG attempt, and H-APG
certificate. Excluding the already measured depth-7 starting cells, the eight
chains contain
`4*(16-7) + (24-7) + (27-7) + (27-7) + (28-7) = 114` new cells. Baseline plus
one winning carrier therefore costs at most 228 evaluations, rather than
materializing a full tree of 131,071 nodes merely to reach `U8/S8`.

The go condition is endpoint reproduction on 8/8 chains plus a repeatable
failure-to-chart transition under refinement. Endpoint drift, contradictory
signs, or depth-independent failure is a no-go.

### V7-C: failure-typed scheduler

Only A and B can justify a 511-node scheduler comparison. The control remains
the frozen alternating S/U policy. The candidate may use a carrier result and
one-step, deterministic failure-class scores to choose a split, with every
unexpanded branch still counted as unresolved. A useful bounded signal requires
at least eight H-PG-eligible leaves across four top-level regions. Zero H-APG
attempts or concentration in one branch is a no-go.

The defensible novelty window is a carrier ensemble coupled to a typed,
receipt-auditable scheduler for validated Poincare cocycles. This is a research
direction, not a priority or novelty claim.

Promotion, open-problem, attractor, hyperbolicity, execution-attestation, and
FPGA claims remain false.
