<!-- docs:meta
topic_id: repo.docs.research.ocssm-preregistration-20260615
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.ocssm-preregistration-20260615
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# O-CSSM Pre-Registration — Confirmatory Test Plan

**Status:** DRAFT, locked for review 2026-05-01. Final hash-lock target: end
of week 11 of the homology-functor push (≈ 2026-06-15). Not yet deposited.
This document supersedes `ocssm_preregistration_v0.md`; v0 retained for
lineage.

**Principal investigator:** Demetrios Chiuratto Agourakis (ORCID 0009-0001-8671-8878).

**Branch / commit lineage at draft time:**

| Item | Commit | Path |
|---|---|---|
| F1 algorithm | `751b018e` | `tools/ocssm/f1_moufang.sio` |
| F2 algorithm | `796c36ed` | `tools/ocssm/f2_conjugation.sio` |
| SWDA corpus loader (M3a) | `5a97a111` | `tools/ocssm/swda_loader.py`, `tests/fixtures/f2_swda.bin` |
| Lean naturality (M4) | `b8b3692b` | `formal/lean4/SounioNaturalityG2.lean` |
| Sounio embed | `662f5414` | `stdlib/dialogue/embed.sio` |
| Trajectory | `fd50eb3a` | `stdlib/dialogue/trajectory.sio` |
| `NaturalityG2` effect | `50ee76b2` | `lean_single.sio:18841` (bit 20) |
| sqrt precision fix | `cd9d7217` | `lean_single.sio:29056` |

After wk-11 hash-lock, any deviation from this plan becomes registered
exploratory per §6 of `ocssm_preprint_skeleton.md`.

---

## 1. Hypotheses

The homology functor `F : C_dial → C_𝕆` is asserted to commute with the
following three structural operations on dyadic dialogue. Each
hypothesis is paired with a single confirmatory test statistic and a
pre-defined threshold.

### H1 — Alternativity (F1, Moufang)

For every repetition triple `(u_i = u_j, v)` in a dialogue trajectory,
the embedded associator vanishes:

> `‖[e(u_i), e(u_i), e(v)]‖ / (‖e(u_i)‖² · ‖e(v)‖) ≤ τ_alt`

with **`τ_alt = 1e-12`** (squared form: τ_alt² ≤ 1e-24, as implemented
in `tools/ocssm/f1_moufang.sio:f1_tau_alt_sq`).

### H2 — Conjugation under speaker reversal (F2)

For every σ-pair `(u, v)` of consecutive opposite-speaker turns, the
embedded conjugate matches:

> ratio = mean_random ‖e(b) − conj(e(a))‖² / mean_paired ‖e(b) − conj(e(a))‖²

PASS criteria: **ratio ≥ 5.0** AND **mean_paired ≤ 0.10**.
Implemented in `tools/ocssm/f2_conjugation.sio:f2_ratio` +
`f2_paired_dist_sq`. Synthetic-baseline gate at `f2_conjugation_synthetic.sio`
PASSes with `ratio = 25.5`, `d_paired² = 0.0807` on the M3a fixture.

### H3 — DEFERRED to v2 (sedenion zero-divisor / affect dissociation)

Per the closed-door ABIDE result (`project_g2_bridge.md`), the
sedenion-ZD ↔ affect-dissociation correspondence is removed from this
preregistration. v1 carries OCSSM affirmations (i) and (ii) only.

---

## 2. Sample

### Primary corpus

**Switchboard Dialog Act Corpus (SWDA)** — public, dyadic, speaker-labeled,
surface repair markers. Loader: `tools/ocssm/swda_loader.py`. Format
specification: 16-byte header (magic, version, count), 40 bytes per
utterance (5 × i64: speaker, hash, affect, rupture, dim).

### Pilot annotation (rupture markers)

80 SWDA conversations, two annotators, target Cohen's κ ≥ 0.65 on
rupture markers. **Kill-switch:** if κ < 0.65 by end of pilot, retreat
to F1-only (intra-speaker repetition is annotator-free) and pre-register
null on F2.

### Held-out split

Fix at end of wk 9 (start of M3 corpus run): random 20% held-out fold,
seed pinned in this file under §4.3. Replication on held-out is the
M5 gate.

---

## 3. Decision rule

| Outcome | Disposition |
|---|---|
| H1 PASS ∧ H2 PASS | Confirmatory result. Submit Neuron-class. §6.5 row "(iii) cai" claimed openly. |
| H1 PASS ∧ H2 FAIL | Registered partial null on F2; F1 stands. Submit *Phil Sci* / *Cog Sci* registered-report track. |
| H1 FAIL ∧ H2 PASS | Registered partial null on F1; F2 stands. Same registered-report track. |
| H1 FAIL ∧ H2 FAIL | Registered full null. Publishable, dissertation Chapter 7 anchors on null. |
| Annotator κ < 0.65 | F2 dropped. F1-only confirmatory. F2 deferred to v2 + Alexander Street + IRB. |
| Held-out (M5) outside original 95% CI | Mark inconsistent; publish both. Defense narrative shifts to "discovered limitation." |

All four outcomes are publishable. None constitute a HARK ("hypothesizing
after results known") liability because all are pre-registered here.

---

## 4. Implementation invariants

These are bound by commit hashes above and not subject to change after
hash-lock without registered-deviation.

### 4.1 Embedding

`embed_into : i64 → [f64; 8]` from `stdlib/dialogue/embed.sio`. 1024-row
Halton table (bases 2,3,5,7,11,13,17,19) + Box–Muller pairs +
unit-normalisation onto S^7. Determinism, unit norm, and spread are
gated at `tests/run-pass/embed_octonion.sio`.

### 4.2 Octonion arithmetic

`stdlib/algebra/octonion.sio`. Cayley–Dickson sign tensor at level 3.
`oct_mul`, `oct_associator` are the primitive operations. The Lean
mirror is `formal/lean4/SounioNaturalityG2.lean` (M4, decide-closed
on the discrete G₂ skeleton + canonical basis embed).

### 4.3 Random seed

The held-out split seed will be added in this file's §6.1 *before*
wk-9 corpus run, drawn from `/dev/random` and committed as a single
i64 hex. Until then, the seed is `RESERVED — TO BE WRITTEN`.

### 4.4 Numerical precision contract

All algebra is computed inside Sounio. No Python in algebra paths.
Python is permitted only in `tools/ocssm/swda_loader.py` for CSV
ingest. The sqrt precision fix at `cd9d7217` is part of the
implementation invariants — replicators must use ≥ commit `cd9d7217`
of the Sounio compiler.

---

## 5. Deviation protocol

After hash-lock:

- A **clarification** that does not change a threshold or invariant
  is a comment-only edit; no registered-deviation needed.
- A **threshold change**, **commit-hash bump**, or **scope change**
  is a deviation. Append to `docs/research/ocssm_preregistration_deviations.md`
  with date, deviation, reason. The original threshold remains the
  confirmatory bar; the deviated threshold becomes a registered
  exploratory readout.
- A **kill-switch trigger** (κ < 0.65, M5 inconsistency) follows §3
  decision rule and does not count as deviation.

---

## 6. Hash lock

### 6.1 Random seed (RESERVED)

`SEED = TO BE FILLED BEFORE WK-9 CORPUS RUN`

### 6.2 Self-hash

The SHA-256 hash of this file (after §6.1 is filled) will be computed,
posted to OSF, and added below as `LOCKED_HASH`. The OSF deposit URL
will be the canonical pre-registration receipt.

`LOCKED_HASH = TO BE COMPUTED ON WK-11 LOCK`

`OSF_URL = TO BE FILLED ON WK-11 LOCK`

---

## 7. Honest residuals (for transparency)

These are gaps the preregistration acknowledges; they do not bear on
the H1/H2 decision rules but reviewers should know:

1. **Lean closure on continuous embed.** `SounioNaturalityG2.lean`
   decides naturality on the *canonical basis embed* (`basisEmbed`),
   not on the runtime Halton–Box–Muller embed. The latter is a
   low-discrepancy approximation of the former across a 1024-row span;
   the empirical bound is checked at runtime by `embed_octonion.sio`,
   not in Lean. Any reviewer requirement for Lean closure on the
   *runtime* embed would require either Mathlib measure theory or a
   1024-row table dump in Lean — out of v1 scope.
2. **Annotator pool size.** N=2 is the minimum for κ; planned single
   round, no adjudication. Inter-annotator disagreements are reported
   raw, not adjudicated, to preserve the κ as a clean signal.
3. **Algebraic uniqueness.** F1's contrast against a random sign-tensor
   control (per preprint §6.1, contrast ≥ 10³) is *not* in this
   confirmatory plan; it is a separate exploratory analysis to be run
   on the same corpus and reported as exploratory regardless of result.

---

*This pre-registration draft was prepared by Claude Opus 4.7 (1M context)
under direction of D. C. Agourakis. It awaits human PI sign-off,
random-seed inscription, and SHA-256 deposit.*
