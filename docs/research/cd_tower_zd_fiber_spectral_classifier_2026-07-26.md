<!-- docs:meta
topic_id: repo.docs.research.cd-tower-zd-fiber-spectral-classifier-2026-07-26
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.cd-tower-zd-fiber-spectral-classifier-2026-07-26
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# CD-tower ZD fibers — the adjacency spectrum is a complete geometry invariant, closing the Fano-stratum open half (n ≤ 8)

**Date:** 2026-07-26
**Orthography:** EN-UK
**Status:** `EXECUTABLE` — `ZD_FIBER_SPECTRUM_COMPLETE_INVARIANT_N_LE_8`
**Thread:** the frozen `PSL(2,7)=168` acting on the growing CD zero-divisor fibers (orbit theorem, proven ∀n)
**Harness:** `scripts/research/cd_tower_zd_fiber_spectral_classifier_contract.py`

---

## 0. The result

The orbit theorem (frozen `PSL(2,7)=168` on the `2^{n-1}-1` zero-divisor fibers of `A_n`:
`2^{n-4}` size-7 Fano orbits + `2^{n-4}-1` fixed seams, proven ∀n) leaves a question the group
action **cannot** answer: how many distinct **fiber geometries** (annihilation-graph isomorphism
classes) are there? The naive "distinct orbits ⇒ distinct geometries" was **false** (retracted): the
**parity-collapse law** (nauty-complete `n≤8`) gives `#geometries = 3·2^{n-5} < #orbits = 2^{n-3}-1`,
because even-weight seams collapse onto Fano orbits. The combinatorial colour-refinement (1-WL) /
degree
invariants **over-merge** the near-regular Fano stratum — so the reviewer flagged **odd/Fano-stratum
injectivity** as *"needs spectral, not degrees"* and **OPEN**.

This rung closes it with a concrete computable invariant:

> **The adjacency spectrum of the ZD annihilation graph is a complete invariant of the fiber
> geometry for `n = 6, 7, 8`.** The number of distinct spectra over all `2^{n-1}-1` fibers is exactly
> `3·2^{n-5}` (`= 6, 12, 24`) — the full nauty-complete count — while COLOUR REFINEMENT (1-WL)
> gives only `4, 8, 16`. The spectrum is **strictly finer than colour refinement**: it separates the
> odd/Fano stratum that colour refinement cannot.
>
> ⚠ **SCOPE CORRECTION (2026-08-07, Fable-5 review).** This must always say 1-WL / colour
> refinement, never "Weisfeiler–Leman" unqualified. `wl_signature` in the contract script refines by
> (own colour, multiset of neighbour colours) — that is 1-WL. The spectrum can NEVER strictly refine
> **2**-WL, because 2-WL-equivalent graphs are cospectral; an unqualified claim is not merely
> unsupported but impossible, and a referee kills the hook in one line.

And the spectral bound, paired with the *explicit* even-weight collapse isomorphisms, closes the
open half self-containedly:

> **Corollary (Fano-stratum injectivity, `n≤8`, no nauty).** A pincer — spectral **lower** bound
> `#iso ≥ #spectra = 3·2^{n-5}` and an **upper** bound `#iso ≤ 3·2^{n-5}` using *only* the verified
> even-weight collapse maps `Φ` — meets exactly, so **no odd-weight collapse can occur**: distinct
> odd-weight (Fano) orbits are pairwise **non-isomorphic**. The reviewer's open half is closed for
> `n≤8` without invoking nauty.

---

## 1. Results

| Clause | Result | Reading |
|---|---|---|
| `S1_SPECTRUM_COUNT` | `n=6,7,8`: `#distinct spectra = 6, 12, 24 = 3·2^{n-5}` | matches the nauty-complete geometry count. |
| `S2_WL_UNDERCOUNTS` | `#1-WL = 4, 8, 16 < #spectra`; explicit witness (same colour-refinement class, distinct spectra) | spectrum **strictly finer than colour refinement (1-WL)** — separates the Fano stratum. NOT a claim about 2-WL, which cospectrality forbids. |
| `S3_COMPLETENESS` | pincer: `#iso ≥ #spectra` (spectrum is an iso-invariant) and `#iso ≤ 3·2^{n-5}` (monochromaticity ∀n + even-weight `Φ`, `n≤8`) | `#iso = 3·2^{n-5}` exactly ⇒ **spectrum is complete** (no cospectral non-isomorphic fibers). |
| `S3b_FANO_INJECTIVITY` | bounds meet using only even-weight merges ⇒ no odd-weight collapse | **Fano-stratum injectivity, `n≤8`, self-contained** — the open half, closed. |
| `S4_DEFLATION_GUARD` | `#spectra` (6/12/24) ≫ `(n-4)` possible outer-seam-bit values; spectrum ⊋ degree histogram | the invariant is **not** a function of the core-law datum `b` — genuinely finer. |

Verdict: `CD_TOWER_ZDSPEC_VERDICT ZD_FIBER_SPECTRUM_COMPLETE_INVARIANT_N_LE_8`. §10 review: Grok `[OK]`
on both bounds, the corollary, the "strictly finer" statement, the novelty, and the `n≤8` scope.

---

## 2. Why this is not a deflation (and how it differs from the ord-3 negative)

The recurring risk in this programme is an invariant that is secretly a function of cheaper
structure (the ord-3 chase deflated exactly so: `2·V₃` was CD-doubling of the coordinate space).
Here the guard `S4` is explicit: the spectrum distinguishes `3·2^{n-5}` classes, far more than the
`n-4` values of the outermost seam bit `b` (the **core-law** datum), and strictly more than the
degree histogram. And unlike ord-3 — a self-generated object pinned by known structure — this is a
graph **isomorphism** classification with an **adversarial nauty ground truth** (`n≤8`) that the
spectrum is measured against, not a count it could trivially reproduce (`#spectra ≤ #iso-classes`
always, so equality is a real completeness statement, impossible to fake).

---

## 3. What this is / is NOT

- **Is:** a concrete, computable, complete invariant (the adjacency spectrum / characteristic
  polynomial) for the CD ZD fiber-geometry classification, `n≤8`; and a self-contained proof of the
  Fano-stratum injectivity for `n≤8`.
- **Not** an ∀n result — `∀n` completeness (no cospectral pairs at larger `n`) and `∀n` Fano
  injectivity remain **OPEN** (cospectral graphs are common in general; the honest next target is a
  spectral doubling recursion, in the spirit of the kernel-dimension proof).
- **Not** a claim on the orbit theorem or parity law (both prior, cited); **not** symbolic beyond a
  numerical eigenvalue computation; **not** the Petitot conjecture (`D3`); **not** clinical.

---

## 4. Reproduce

```bash
python3 scripts/research/cd_tower_zd_fiber_spectral_classifier_contract.py
# expect: S1/S2/S4 OK for n=6,7,8; VERDICT ZD_FIBER_SPECTRUM_COMPLETE_INVARIANT_N_LE_8
# (~70s; n=8 builds 127 graphs of 252 vertices)
```

Self-contained: replicates the committed closed-form annihilation-graph construction
(`cd_tower_fiber_geometry_collision.py`) verbatim and adds the adjacency spectrum.

---

## 5. AI disclosure

Probe, contract, and note produced under human direction (2026-07-26), pursuing the genuinely-novel
`PSL(2,7)` orbit-theorem thread (distinct from the ord-3 vein, which closed negative). The foundation
(kernel-dimension spectrum, orbit theorem, parity-collapse law + explicit `Φ`) is prior in-repo work,
independently re-verified here. **New:** the adjacency spectrum realizes the full fiber-geometry
classification (`#spectra = 3·2^{n-5}`, `n=6,7,8`), strictly finer than colour refinement (1-WL), closing the
reviewer's "needs spectral" open half for `n≤8` — with a self-contained pincer proof of Fano-stratum
injectivity. §10 math-review (Grok `[OK]` on all claims; novelty "not standard in de Marrais/Moreno").
Numerical certificate; `∀n` OPEN. No semantic claim, no clinical content. GAIDeT-ICMJE 2025.
