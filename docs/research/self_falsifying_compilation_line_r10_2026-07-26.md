<!-- docs:meta
topic_id: repo.docs.research.self-falsifying-compilation-line-r10-2026-07-26
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.self-falsifying-compilation-line-r10-2026-07-26
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Self-falsifying compilation R10 — latent corroboration discovery: the procedure works, and it found nothing new

**Date:** 2026-07-26
**Orthography:** EN-UK
**Status:** `EXECUTABLE` — `LATENT_CORROBORATION_FOUND`
**Parents:** `self_falsifying_compilation_line_r8_2026-07-26.md` (found the `cds`/`cd_sigma` corroboration by hand), `self_falsifying_compilation_line_r6_2026-07-26.md` (the independence threshold this reuses)
**Harness:** `scripts/research/self_falsifying_compilation_line_r10_contract.py`
**Gate:** `scripts/ci/self_falsifying_compilation_line_r10_gate.sh`

---

## 0. The question past every neighbour

Three literature searches mapped the solid ground, and every neighbour stops at
the same place. `build.rs` executes checks. Clone detection measures
duplication. The repeatability/replicability taxonomy names why an independent
re-implementation beats a re-run. N-version programming studies whether **two**
implementations are independent — pairwise, by construction, because someone
commissioned them separately.

None asks the corpus-level question: **how many independent things does this
body of code already know?**

R8 answered a fragment by hand: `cd_sigma`, recursive and sitting unused in
three contracts, is an independent derivation of the same object as the
iterative `cds`, and nobody had ever compared them. A **latent corroboration** —
evidence the corpus already owned and had never cashed.

R10 automates the search. A pair that is **structurally independent** (below
R6's clone threshold) yet **behaviourally identical** (agrees on every probe
input) is a latent corroboration.

---

## 1. Result

> **The procedure works: from nothing but the source, it rediscovers the
> `cds`/`cd_sigma` corroboration R8 found by hand. Within its probeable slice it
> finds no others — and that slice turned out to be one function wearing 31
> faces.**

| | |
|---|---:|
| probeable functions found | 31 |
| **distinct behaviours among them** | **1** |
| behaviour classes with more than one derivation (**real corroborations**) | **1** |
| **newly discovered corroborations** | **0** |
| cross-derivation pairs | 130 — *copies, not corroborations* |

Verdict: `SELF_FALSIFYING_R10_VERDICT LATENT_CORROBORATION_FOUND`.

**The token is mechanically correct and overstates the result, so the result is
stated here.** A latent corroboration exists and the search found it — but it
was already known from R8, and the search discovered nothing new. The criteria
were fixed before running and the token is not being retro-fitted; §1.1 and §2
carry the meaning.

### 1.1 The 130 is a trap, and it is this rung's own inflated metric

A first version of this harness reported **130 latent corroborations**. That
counts *pairs of copies*: 24 copies of one derivation against 7 copies of
another produce ~168 "independent pairs" while representing **exactly one**
corroboration.

Same error class as R8's four-versus-three derivations, and caught the same
way — by asking what the unit is. The honest unit is the **behaviour class with
more than one derivation**. There is one. The pair count is kept in the output,
explicitly labelled as counting copies, because deleting it would hide how
easily the inflated number arises.

---

## 2. What this actually establishes

**A validated discovery procedure.** Given only source, with no knowledge of
which functions are supposed to compute what, it finds the corroboration a human
found by reading. That is worth having: it is the mechanism by which a project
can *cash* evidence it already owns, and R8's finding suggests such evidence goes
unnoticed by default.

**A metric nobody computes: corroboration depth.** For each behaviour the corpus
computes, how many structurally distinct derivations of it does the corpus
contain? Depth 1 — one implementation, no internal corroboration at all — is the
default state and, before this rung, an unmeasured one. Counting files gives the
opposite impression: 31 implementations look like breadth and are depth 2.

**And a negative, within a narrow scope.** No new latent corroboration exists
among the probeable functions.

---

## 3. The coverage limitation, which is severe

The probe accepts functions taking 2–3 positional integers and returning a
scalar. **Every one of the 31 it found computes the same thing** — the
Cayley–Dickson sign — because that is the only kernel in this corpus with that
signature.

So the negative result covers almost nothing. Untouched by this search:

- array-valued kernels (`omul`, `mul`, `o`, `cd_mul`) — same signature family,
  but the probe rejects non-scalar returns;
- set- and dict-valued kernels (`expected_labels`, `missing_diagonal`,
  `compute_fibers`, `p_add`/`p_sub`);
- anything needing structured or float inputs (`cusp_wells`);
- anything requiring its module's helpers to run (`zd_line` — the R9 boundary).

Extending the probe to array and set returns is mechanical and is the obvious
next step. Until then, "no new corroborations" means "none among scalar-valued
integer functions", which is a much smaller claim than it sounds.

---

## 4. What this is NOT

- **Not a discovery.** Zero new corroborations. The one found was known.
- **Not a corpus-wide audit.** §3: one signature family, one behaviour.
- **Not a proof of equivalence.** Behavioural identity is over a finite probe
  grid (levels 3–5, all index pairs). Two functions agreeing there can differ
  elsewhere; this finds *candidates* for corroboration, and confirming one is
  then a separate job (R7/R8 did it properly for `cds`/`cd_sigma`).
- **Not a compiler change.** The whole R6–R10 arc is Python-only.

---

## 5. Reproduce

```bash
python3 scripts/research/self_falsifying_compilation_line_r10_contract.py
# expect: 31 probeable functions, 1 distinct behaviour, 1 real corroboration,
#         0 newly discovered,
#         SELF_FALSIFYING_R10_VERDICT LATENT_CORROBORATION_FOUND

bash scripts/ci/self_falsifying_compilation_line_r10_gate.sh
# expect: SELF_FALSIFYING_COMPILATION_LINE_R10_GATE_OK
```

Functions are compiled in isolation by AST; the surrounding modules are never
imported.

---

## 6. AI disclosure

Harness, gate and spec drafted under human direction (2026-07-26). All figures
are machine-computed and re-runnable. The inflated pair metric is retained in
the output, labelled, rather than removed. No clinical content.
GAIDeT-ICMJE 2025.
