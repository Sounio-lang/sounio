<!-- docs:meta
topic_id: repo.docs.research.self-falsifying-compilation-line-r14-2026-07-27
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.self-falsifying-compilation-line-r14-2026-07-27
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Self-falsifying compilation R14 — turning the instrument around: what this corpus computes, it checks

**Date:** 2026-07-27
**Orthography:** EN-UK
**Status:** `EXECUTABLE` — `VACUITY_REFUTED__CORPUS_CHECKS_WHAT_IT_COMPUTES`
**Parents:** `self_falsifying_compilation_line_r13_2026-07-27.md` (the instrument, and the run this rung corrects), `self_falsifying_compilation_line_r12_2026-07-27.md` (why the code measure was abandoned)
**Harness:** `scripts/research/self_falsifying_compilation_line_r14_contract.py` (+ `scripts/research/r14/`)
**Gate:** `scripts/ci/self_falsifying_compilation_line_r14_gate.sh`

---

## 0. The hypothesis this rung went to test, and lost

R13 perturbed the shared mathematical object to find contracts that **die
together**. R14 turns the same instrument the other way: for each contract, of
everything it computes, how much does its stated conclusion depend on?

The hypothesis was **vacuity**. Several contracts compute the Cayley–Dickson
tower to levels 5–10 — one queries 260 610 distinct basis products at level 9
alone. A conclusion resting only on the cheap low levels would make that
expensive high-level work decoration, and the claim would be broader than its
evidence.

**It is refuted.** The corpus checks what it computes, at every level, up to the
1024-dimensional algebra. Reporting it because a pre-registered hypothesis
losing is a result, and because the refutation is a non-trivial positive fact
about the corpus that nobody had measured.

---

## 1. Result

| | |
|---|---:|
| contracts traced | 31 |
| deepest level queried | **10** (1024-dim) |
| perturbation cells | **536** |
| verdict changed | **407** |
| crashed (conclusion unobtainable) | **117** |
| survived | **12** |
| missing (timeout / lost output) | **0** |

Verdict: `SELF_FALSIFYING_R14_VERDICT VACUITY_REFUTED__CORPUS_CHECKS_WHAT_IT_COMPUTES`.

By level, and the high levels are the point:

| level | verdict changed | crashed | survived |
|---:|---:|---:|---:|
| 2 | 24 | 0 | 0 |
| 3 | 160 | 8 | 8 |
| 4 | 73 | 53 | 2 |
| 5 | 48 | 16 | 0 |
| 6 | 40 | 16 | 0 |
| 7 | 24 | 16 | 0 |
| 8 | 22 | 8 | 2 |
| **9** | **8** | **0** | **0** |
| **10** | **8** | **0** | **0** |

Levels 9 and 10 are **pure verdict changes, no crashes**: the deepest
computations in this corpus are load-bearing in the controlled sense, not merely
fragile.

### 1.1 The measure needs three outcomes, not two

- **verdict changed** — the contract computed a different answer and reported
  it. Clean load-bearing.
- **crashed** — the perturbation made the contract raise (matmul shape
  mismatch, index out of bounds, negative shift, `KeyError` on a histogram bin,
  SVD non-convergence). The conclusion can no longer be established, so it is a
  dependence; but it is not a *check* noticing anything.
- **missing** — timeout or lost output file. **No information.** Scoring this as
  a kill inflates load-bearing, and the first version of the R13 analysis did
  exactly that.

**Four contracts are ALL-CRASH** — `cd_tower_nullity_histogram_law`,
`routon_zd`, `functor_f_ord3_quotient_fill`, `functor_f_ord3_ternary_anatomy`
(at level 4). For these the measure reports **fragility, not
conclusion-dependence**, and their contribution to the headline is qualified
accordingly. `cd_tower_nullity_histogram_law` is the extreme case: all 40 of its
cells are crashes, so it is useless as a test bed — every perturbation, target
or control, raises `KeyError`.

---

## 2. The twelve survivors, and RULE 2 doing its job

A level where every sampled flip survives is **not** a vacuity finding. Two
different things produce it:

**(a)** the level is queried but not checked — vacuity, a defect;
**(b)** the checked quantity is genuinely **invariant** under a single sign flip
— mathematics, not a defect.

The analysis fixed this distinction *before* seeing the numbers, and it decided
the one candidate. `functor_f_ord3_ternary_anatomy` at level 3: **all 10 single
pair flips survive; all 4 in-range element row-flips kill** (the 2 out-of-range
ones correctly do not). So the level *is* checked, and what it checks does not
see a single sign — case (b), resolved from data already on disk, no new
compute.

That is the rule earning its keep: without it this rung would have shipped
"a vacuous level found in the corpus", which is false.

### 2.1 One survivor left standing, and three explanations refuted

`cd_tower_zd_fiber_signed_localization` and `..._spectral_classifier` both
survive exactly one perturbation: **`L8_64_192`** — flipping the sign of
e₆₄·e₁₉₂ at level 8. Reproduced three times independently (load-bearing battery,
diagonal probe, form probe), unique among ~28 distinct level-8 perturbations
tested on that contract.

Three hypotheses were tested and **all three failed**:

1. *Shared blind spot between two contracts.* **No** — their R6 structural
   similarity is **1.000**. They are near-copies, so this is one observation,
   not two.
2. *Doubling diagonal.* 192 = 128 + 64, so e₁₉₂ = (0, e₆₄) is the
   Cayley–Dickson double of e₆₄. Tested the whole diagonal {(k, 128+k)}:
   **1 of 10 survives** — only (64, 192). Controls: 0 of 10.
3. *The arithmetic form (h/2, h + h/2), h = 2^(bits−1).* Tested at every level
   the contract spans: **kills at levels 4, 5, 6, 7; survives only at 8.** So it
   is not the shape of the pair — it is the level.

Level 8 is the boundary of these contracts' own claim
(`ZD_FIBER_SPECTRUM_COMPLETE_INVARIANT_N_LE_8`).

**Nothing further is claimed.** This may be a blind spot, or it may be correct:
if the flip yields an *isomorphic* annihilation graph, an unchanged spectrum is
exactly what a complete invariant should give, and insensitivity is a success.
Distinguishing those needs an isomorphism test this rung did not run. Recorded
as a located, reproducible anomaly with its explanations eliminated — not as a
finding.

---

## 3. What this rung corrects in R13

The call trace showed that two of the three contracts R13 excluded for "no
baseline verdict" **do emit verdicts**:
`ZD_FIBER_SPECTRUM_COMPLETE_INVARIANT_N_LE_8` (86 s) and
`ZD_FIBER_SPECTRAL_FORALL_N_STRONG_EVIDENCE_NOT_CLOSED` (19 s). They hit the
600 s cap under 96-way contention, and the harness could not tell a timeout from
a missing token.

Both are `cd_sigma` — the scarce derivation family — so the loss fell entirely
on the thin side of R13's comparison. Re-run at 6-way concurrency with a 2 400 s
cap, identical battery, control inert, 25 kills each:

| | R13 as shipped | corrected |
|---|---:|---:|
| usable contracts | 28 | **30** |
| `cd_sigma` contracts | 2 | **4** |
| identical-fate pairs below 0.90 | 15 | **21** |

R13's finding is **strengthened, not weakened**, and its spec and data are
updated in place with §5.2 recording why. Two lessons kept:

1. **A crash is a kill; a timeout is missing data.** The 21 pairs hold under
   either convention — checked, not assumed.
2. **Concurrency is a measurement parameter.** The same battery at 96 workers
   and at 6 workers yields different corpora. Any result of this shape must
   report the worker count.

---

## 4. What this is NOT

- **Not proof that no level is vacuous.** 8 samples against 64 770 queried pairs
  at level 8 is a thin probe. "No non-load-bearing pair found in k of n" is
  reported with the denominator attached, never as "the level is checked
  exhaustively".
- **Not a claim about crash-dominated contracts.** For the four ALL-CRASH
  contracts the measure does not separate *checking* from *breaking*.
- **Not an explanation of `L8_64_192`.** §2.1.
- **Not a compiler change.** Still Python-only.

---

## 5. Reproduce

```bash
python3 scripts/research/self_falsifying_compilation_line_r14_contract.py
# expect: C1 30/30 inert, C2 536 cells (407/117/12/0), C3 the single
#         all-survive level resolved as invariance,
#         SELF_FALSIFYING_R14_VERDICT VACUITY_REFUTED__CORPUS_CHECKS_WHAT_IT_COMPUTES

bash scripts/ci/self_falsifying_compilation_line_r14_gate.sh
```

Recorded evidence is under `scripts/research/r14/`. Regenerating it needs a
large machine: the call trace and the 596-run battery were produced on a
128-core node (`trace.py`, `loadbearing.py`), and the focused probes
(`diag.py`, `hhalf.py`) took ~33 s per run at level 8.

### 5.1 Two infrastructure faults, both recorded

**No `procps` in the runner image.** `pkill -f loadbearing.py 2>/dev/null`
silently did nothing — there is no `pkill`, and the redirect hid it. A second
battery started alongside the first: 112 concurrent probes on a 100-core budget,
load average 765. The launcher now kills by walking `/proc`. *A cleanup step
whose failure is invisible is not a cleanup step.*

**Unpinned BLAS threads.** Each worker's numpy opened a thread pool sized to the
visible core count. `OMP_NUM_THREADS=1` and friends are now set explicitly;
parallelism comes from the worker pool alone.

Neither changed a result, but both changed how long the truth took to arrive,
and the first one nearly cost the two contracts §3 recovers.

---

## 6. AI disclosure

Tracer, batteries, probes, analysis, gate and spec drafted under human direction
(2026-07-27). All figures are machine-computed from
`scripts/research/r14/call_trace.json` and `loadbearing.json`. The vacuity
hypothesis and the escalation rule that resolved the survivors were fixed before
the corrected battery ran; the hypothesis lost. No clinical content.
GAIDeT-ICMJE 2025.
