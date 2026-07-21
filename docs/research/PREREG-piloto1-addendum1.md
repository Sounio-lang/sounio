<!-- docs:meta
topic_id: repo.docs.research.prereg-piloto1-addendum1
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.prereg-piloto1-addendum1
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Addendum 1 to PREREG-piloto1 — pre-A decision + first-pass outcome + mandated legibility work

**Date:** 2026-07-20. **Status:** addendum to [PREREG-piloto1-semantic-barriers.md](PREREG-piloto1-semantic-barriers.md).

The §1 decision below was fixed **before A's number was known** (B negative already in hand; A's PMI still
computing on `cpuops-t560`, SLURM job 7706). It is recorded to remove the post-hoc freedom in interpreting
A≠B. The §2 outcome and §3 mandated work were written immediately after A completed.

---

## 1. Pre-A decision on A≠B (locked before the number): **(b), bound — "raises, establishes nothing"**

Aggregating B across generators averages over distinct individual semantic manifolds and would wash out any
person-specific geometry, so A-positive / B-negative is the *expected* pattern under a person-specific
hypothesis, not an anomaly. Therefore option (b), bound so it is not an escape hatch:

- **A-positive RAISES the person-specific hypothesis and establishes nothing.** A single long series cannot
  separate "person-specific geometry" from "this-dreamer idiosyncrasy" (the A-vs-B confound one level up).
  "A-positive" requires the barrier to survive **every** control B passed: **k-stable** Δ(c)<0 (not the
  k=5-only artifact), beating the trivial jump control, beating the permuted null at empirical p<0.05 with
  c+k-sweep correction, **and** above the sensitivity floor δ\* from the positive control (§3.1).
- **Establishment requires pre-specified replication: 6 distinct-dreamer long single-subject DreamBank
  series, n≥1000** — Barb Sanders + 5 of {Izzy 4329, Kenneth 2022, Emma 1221, Elizabeth 1707, Norman 1235,
  Pegasus 1093} (English; excludes Barb #2 = same dreamer, von Uslar = German). **Criterion: barrier present
  in ≥5/6 AND absent in ≥3 matched cross-generator aggregates of comparable sentence-count.** Fewer, or
  present also in aggregates → the person-specific claim fails. (7 qualifying series exist, so N=6 is a real
  bar, not an impossible one.)
- Until that replication passes, A-positive is a hypothesis-generating single case, worth nothing for either
  the general or the person-specific claim.

## 2. First-pass outcome (GPT-2 PMI, jump control only) — negative, but **not yet legible**

Both samples negative; the decision in §1 did **not** trigger (no A-positive).

| | k=5 | k=10 | k=20 | max contiguous below-null run |
|---|---|---|---|---|
| A — Barb Sanders (46 471 nós) PMI | +0.000 | −0.001 | +0.000 | **0.35 %** |
| A jump (trivial) | −0.000 | −0.001 | −0.001 | 0.10 % |
| B — Norms (7 886 nós) PMI | −0.027 | −0.002 | −0.001 | 2.9 % |
| B jump (trivial) | −0.002 | −0.003 | −0.007 | 2.7 % |

`ρ_obs` sits inside the permuted-null envelope across the whole sweep, for every k, in both samples. B's
2.9 % is (i) indistinguishable from its own trivial control (2.7 %) and (ii) an order-of-magnitude
k-dependent (Δ −0.027 at k=5 vs −0.001 at k=20) — the pre-registered graph-construction-artifact falsifier,
self-refuting. A is flat-negative with no k=5 artifact at all.

**This points at the pre-registered strong falsifier** — all sublevels connected, in both individual and
aggregate corpora → the necessary/gratuitous-suffering distinction (§A.2/§A.3 of Mercyful Learning) has no
empirical referent in real semantic fields under this field, not just in 𝕊. **But the negative is
illegible until §3.** "No barrier" and "a design that could not detect a barrier" produce this same table;
and GPT-2 is a weak estimator whose noise could itself wash out a barrier. The result is **not declared**
until the two items below are done.

## 3. Mandated work before the negative is legible (blocks any final claim)

### 3.1 Positive control / sensitivity floor — MANDATORY (also closes the O-SSM injected-structure debt)
Inject a synthetic barrier of known magnitude: choose a graph cut splitting the mutual-kNN graph into two
large components, raise `s` by δ on one side so `{s≤c}` is disconnected by construction, and verify the
union-find + permuted-null pipeline flags it; sweep δ down to the detection threshold **δ\***. The negative
is legible only as: *no barrier, and the design would have detected one of magnitude ≥ δ\*.* This is
simultaneously the injected-structure recovery demonstration open since the O-SSM work — one task, two debts.

### 3.2 Stronger LM — MANDATORY (scope of the negative depends on it)
The first pass used **GPT-2 (124M)**. A weak causal LM yields a noisy PMI field, and noise destroys barrier,
so the negative is currently conditioned on the estimator, not the geometry. Re-run PMI with a modern LM —
**Qwen2.5-Coder-1.5B**, already resident in the BEAGLE shared HF cache — and report both. Model + version go
in the record: primary first pass = `gpt2` (transformers 5.14.1); robustness = `Qwen2.5-Coder-1.5B`.

### 3.3 Ollivier–Ricci curvature field + empirical p-values
The pre-reg specified three fields and the falsifier reads "for the three definitions"; the curvature field
(the one with independent empirical precedent in this program) must be computed and tabled, or this addendum
amends the pre-reg to two fields with reason. And the empirical p(c) with c+k-sweep correction, pre-registered,
must be tabled (weighs less than §3.1 for a negative, but was registered).

## 4. Infra note (not a lab default)
Bringing torch onto the bare cluster nodes required bootstrapping pip over an **SSL-verification bypass**
(unverified context / `--trusted-host`), because the nodes' system CA bundle is broken. This is a one-off to
unblock this pilot, **not** a standing practice — replace with an internal PyPI index or a vendored wheel
before it becomes a lab habit. (Within the installed env, `certifi` restores normal verification, which is
why model downloads worked without the bypass.)
