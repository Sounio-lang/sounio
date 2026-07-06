<!-- docs:meta
topic_id: repo.docs.research.sedenion-quartet-fiber-incidence
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.sedenion-quartet-fiber-incidence
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# The quartet↔fiber incidence of the sedenion ZD geometry: 42 quartets = 2·K₇ (executed exactly)

**One line.** The two factorizations of the sedenion zero-divisor `168` — `7 fibers × 24 edges` and
`42 quartets × 4 pairs` — interlock exactly: each support-quartet's 4 pairs split across exactly **2
fibers**, and the **42 quartets, viewed as edges on the 7 fibers, form the doubled complete graph
`2·K₇`** — every one of the `C(7,2) = 21` fiber-pairs is joined by exactly 2 quartets, and every
fiber has incidence-degree 12. Executed exactly and verified on three independent legs.

## The two factorizations, interlocked

- `sedenion_zd_fibers.md`: the 84 participating primitives split into 7 fibers by `L = lo⊕hi ∈ {9..15}`;
  annihilation is intra-fiber, `168 = 7 × 24`.
- `sedenion_zd_quartets.md`: the 168 pairs group by support-union into 42 quartets, `168 = 42 × 4`.

This note is the bridge: a quartet's 4 pairs are **not** all in one fiber — they split `2 + 2` across
two fibers. Treating each quartet as the edge `{L₁, L₂}` it joins gives a multigraph on the 7 fibers.

## Result

| Quantity | Value |
|---|---|
| zero-divisor pairs | 168 |
| fibers a quartet spans | exactly **2** |
| distinct fiber-pairs used | **21 = C(7,2)** (all of them) |
| quartets per fiber-pair | exactly **2** (`2 × 21 = 42`) |
| incidence-degree of each fiber | **12** |

So the 42 quartets are exactly `2·K₇` on the 7 fibers: the complete graph on the seven fibers, each
edge doubled. The zero-divisor geometry is a **doubled-K₇ of 12-vertex fibers**, all bounded by the
`e8` seam.

## Honest scope

The individual factorizations (`7 × 24`, `42 × 4`) are from the Python geometry report. The **`2·K₇`
incidence** relating them — every fiber-pair joined by exactly 2 quartets — is executed and verified
here on three legs (souc, Python oracle, Lean `native_decide`). All three transcribe the same sign
law; Lean is the independent checker.

## Certification

- **Executed in Sounio:** `tests/run-pass/sedenion_quartet_fiber_incidence.sio` (self-contained, no #637).
  Verdict `INCIDENCE OK` (`PAIRS 168 / FIBERPAIRS 21 / BAD_FIBERS 0 / BAD_PAIRCT 0 / BAD_DEG 0`).
- **Cross-toolchain:** `scripts/ci/sedenion_quartet_fiber_incidence_gate.sh` — souc vs the Python oracle
  (which emits the 21 fiber-pair records). Registered in CI (Contracts); under `bin/souc` and stage2.
- **Lean:** `formal/lean4/SounioSedenionIncidence.lean` `native_decide`-proves `pairs_168`,
  `each_quartet_spans_2`, `fiberpairs_21`, `two_per_fiberpair`. `lake build` green; Lean Proofs CI job.

## Reproduce

```bash
SOUNIO_STDLIB_PATH=$PWD/stdlib ./bin/souc run tests/run-pass/sedenion_quartet_fiber_incidence.sio
python3 scripts/research/sedenion_quartet_fiber_incidence_oracle.py
bash scripts/ci/sedenion_quartet_fiber_incidence_gate.sh
(cd formal/lean4 && lake build SounioSedenionIncidence)
```
