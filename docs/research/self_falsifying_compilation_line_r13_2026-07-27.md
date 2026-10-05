<!-- docs:meta
topic_id: repo.docs.research.self-falsifying-compilation-line-r13-2026-07-27
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.self-falsifying-compilation-line-r13-2026-07-27
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Self-falsifying compilation R13 — the counterexample R12 said would need constructing, found in the corpus instead

**Date:** 2026-07-27
**Orthography:** EN-UK
**Status:** `EXECUTABLE` — `STRUCTURAL_INDEPENDENCE_DOES_NOT_IMPLY_INDEPENDENT_FATE`
**Parents:** `self_falsifying_compilation_line_r12_2026-07-27.md` (narrowed C6 to one-sided *by transfer* from someone else's study; this rung measures it here), `self_falsifying_compilation_line_r6_2026-07-26.md` (the measure under test), `self_falsifying_compilation_line_r9_2026-07-26.md` (route independence, the thing that actually worked)
**Harness:** `scripts/research/self_falsifying_compilation_line_r13_contract.py` (+ `scripts/research/r13/`)
**Gate:** `scripts/ci/self_falsifying_compilation_line_r13_gate.sh`

---

## 1. Result

> **Twenty-one pairs of this repository's contracts have byte-for-byte identical
> responses to all 36 targeted perturbations of the shared mathematical object,
> while their R6 structural similarity is 0.479–0.594 — far below the 0.90
> threshold at which R6 declares them independent evidence of one another.**

Verdict: `SELF_FALSIFYING_R13_VERDICT STRUCTURAL_INDEPENDENCE_DOES_NOT_IMPLY_INDEPENDENT_FATE`.

| | |
|---|---:|
| contracts defining a CD-sign function | 31 |
| usable (emit a verdict token) | 30 |
| perturbations + baseline + control | 36 + 1 + 1 |
| probe runs | **1 254** |
| informative mutants (kill 10–90 %) | **24** (pre-registered floor 8) |
| distinct kill patterns | **6** |
| identical-fate pairs below R6's threshold | **21**, all cross-derivation |

And the direction is the opposite of the one R6's inference needs:

| pairs | n | mean kill-set agreement |
|---|---:|---:|
| R6 says **INDEPENDENT** (sim < 0.90) | 101 | **0.565** |
| R6 says **SHARED** (sim ≥ 0.90) | 334 | 0.513 |

Gap **−0.052**. A measure that predicted shared evidential fate would show a
large *positive* gap. This one is slightly negative.

---

## 2. What was perturbed, and why it is not tautological

The corpus contains two derivations of the Cayley–Dickson sign: `cds`
(iterative, 26 contracts) and `cd_sigma` (recursive, 5) — **disjoint**, no
contract carries both, R6 similarity **0.507**. R6 therefore classifies every
cross-derivation pair as independent evidence.

The obvious experiment — mutate the *source* of `cds` — is worthless: it can
only reach `cds` users, so it would re-derive R6's structural partition by
construction. Instead the perturbation targets the **shared mathematical
object**:

> flip the CD sign on base pair (a, b) at level L

Both functions take `(a, b, bits)` and return ±1, so the identical conceptual
perturbation crosses both derivations. The battery is graduated — single base
pairs, then all products involving one basis element, then whole levels, then a
catastrophic anchor — at octonion (L3) and sedenion (L4) level.

**The contracts do respond differently**, which is what makes the agreement
meaningful rather than trivial: low-index L4 flips kill all 28, while
sedenion-only units (indices 8–15) split the corpus roughly in half. Six
distinct kill patterns emerge. The finding is that this partition **crosses the
derivation boundary**: both `cd_sigma` contracts land in a sensitivity class
already occupied by `cds` contracts.

- `cd_tower_zd_fiber_signed_localization`, `..._spectral_classifier` and
  `..._spectral_forall_n_progress` (25 kills each) — identical to **3** `cds`
  contracts apiece, structural similarity **0.479–0.558**
- `rupture_r4_fano_field` (9 kills) — identical to **12** `cds` contracts,
  structural similarity **0.512–0.594**

**What this shows, stated no more strongly than it is:** evidential fate is
fixed by *which proposition you assert about the shared object*, not by *which
code you wrote to compute it*. R6 measures the code.

---

## 3. What this is NOT

- **Not a demonstration of shared misinterpretation.** Co-sensitivity is
  sensitivity to the same perturbations; a shared misinterpretation is a
  specific wrong belief held in common. The first is a measurable proxy for the
  second, not the second.
- **Not a correlation result.** Pearson r between structural similarity and kill
  agreement is +0.056, but structural similarity here is nearly degenerate
  (q1 = median = 0.929), so r is computed over almost no variance in x and
  carries little. **The load is borne by the existence claim in §1**, which
  needs no distributional assumption: these 15 pairs exist and are identical.
- **Not high resolution.** 6 kill patterns over 28 contracts. Jaccard is
  effectively asking which of six buckets a contract is in.
- **Not broad on the scarce side.** 4 `cd_sigma` contracts, against 26 `cds`.
  (The first run had 2 — see §5.2.)
- **Not a compiler change.** Still Python-only. R12 withdrew the compiler rule;
  this rung is the local evidence for that withdrawal, not a reversal of it.

---

## 4. What it settles, and what it leaves open

R12 narrowed C6 to one-sided by **transfer** — CodeBLEU is a richer measure than
R6's, it was refuted at 224 × 12, so the poorer measure cannot do better. That
argument is sound but it is an argument about someone else's population.

R13 measures it **here**, on this corpus's own internal checks, and the
one-sided reading survives contact: structural similarity still reliably detects
*shared* derivation (the copy-paste finding of R6 stands — 343 pairs, one `cds`
table), and demonstrably fails in the *independent* direction.

R12 also expected this would require a hand-built corpus of 12 implementation
pairs, and flagged the single-author confound that would come with it. It did
not: the counterexamples were already in the repository, written for other
reasons, over months. **Fifteen of them.**

**Open, and now sharper.** R9's successful audits used route independence — `L_x`
rank-deficiency, "a route the corpus's own predicates never take". Co-sensitivity
is a *negative* test: it detects shared fate, so it can refuse a corroborator.
It cannot certify one, because insensitivity to a perturbation battery is not
evidence of an independent route. A compile-time obligation could use it to say
**no**; nothing here lets it say **yes**.

---

## 5. Reproduce

```bash
python3 scripts/research/self_falsifying_compilation_line_r13_contract.py
# expect: C1 28/28 inert, C2 24/36 informative, C3 15 pairs,
#         SELF_FALSIFYING_R13_VERDICT STRUCTURAL_INDEPENDENCE_DOES_NOT_IMPLY_INDEPENDENT_FATE

bash scripts/ci/self_falsifying_compilation_line_r13_gate.sh
```

The contract reads the recorded battery (`scripts/research/r13/`). Regenerating
it means 1 178 contract executions — 1 221 s on 128 cores, and far longer on a
workstation. `scripts/research/r13/battery.py` is the runner; every probe forks a
child that writes JSON and leaves via `os._exit(0)` (R11 §3).

### 5.1 The first battery was contaminated, and by the instrument

The first run reported two contracts killed by a perturbation that is
mathematically impossible: a sign flip on a base pair with index ≥ 8 **at level
3**, where only indices 0–7 exist. That reads as a striking corpus finding.

It was the harness. The wrapper was written as
`def cds(a, b, bits={target}, ...)`, which **overrode the contract's own
default**. Contracts declaring `def cds(a, b, bits=4)` and calling
`cds(k ^ j, j)` were silently switched from sedenion to octonion arithmetic, and
duly changed verdict. Nothing was wrong with the corpus; the instrument had
changed the question.

Two consequences, both kept:

1. The wrapper no longer touches the signature — it forwards `*args` and
   recovers the effective `bits` from the original's `__defaults__`.
2. **A null-wrap control was added**: identical wrapper machinery, condition that
   can never fire. It is `C1`, it runs before anything else, and it fails the
   rung outright rather than being one number among many. One column per
   contract would have caught this automatically instead of it needing to be
   noticed.

The corrected battery ran **7× slower** than the contaminated one — direct
confirmation of the diagnosis, since the bug had been silently running
octonion-sized (8-element) computations in place of sedenion-sized (16-element)
ones.

This is the sixth self-catch on this line, and the second in two rungs where the
guard failed its own negative test before the corpus was ever read.

### 5.2 The first run undercounted, and the cause was again the harness

R13 first reported **15** pairs over **28** usable contracts, excluding three for
"no baseline verdict". The R14 call trace showed two of those three DO emit
verdicts — `ZD_FIBER_SPECTRUM_COMPLETE_INVARIANT_N_LE_8` in 86 s and
`ZD_FIBER_SPECTRAL_FORALL_N_STRONG_EVIDENCE_NOT_CLOSED` in 19 s. They had not
been silent; they hit the 600 s cap under 96-way contention, and the harness
recorded a timeout indistinguishably from a missing token.

Both are `cd_sigma`, the scarce derivation family, so the loss fell entirely on
the side of the comparison that had least data: the run went out with 2 of them
instead of 4. Re-run at 6-way concurrency with a 2 400 s cap, identical
36-mutant battery, null-wrap control inert for both, 25 kills each. Merged into
`battery_results.json` so there is one source of truth, and the numbers above
are the merged ones: **30 usable, 21 pairs**.

Two changes follow, both kept:

1. **A crash is a kill; a timeout is missing data.** The first analysis scored
   any error as a kill, which would count a lost run as evidence. The 21 pairs
   are identical under either convention — checked, not assumed — but the
   conflation was real and is now fixed in the contract.
2. **Concurrency is a measurement parameter, not an implementation detail.**
   The same battery at 96 workers and at 6 workers gives different corpora. Any
   result of this shape has to report the worker count, and a timeout has to be
   reported as missing rather than folded into a verdict.

---

## 6. AI disclosure

Battery, probe, analysis, gate and spec drafted under human direction
(2026-07-27). All figures are machine-computed from
`scripts/research/r13/battery_results.json`, produced by 1 178 recorded contract
executions on a 128-core node. The discrimination floor and the analysis rules
were written before the corrected battery ran. §5.1 records an instrument fault
that was hit, not anticipated. No clinical content. GAIDeT-ICMJE 2025.
