<!-- docs:meta
topic_id: repo.docs.dissertation.status-audit-2026-08-16
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.dissertation.status-audit-2026-08-16
-->

# Dissertation Status Audit — 2026-08-16

**Author:** `minimax-cli3` (read-only audit)
**Lane:** `dissertation-status-audit`
**Defense target:** 2026-09-22 (37 days from today)
**Today's date:** 2026-08-16
**Branch:** `lane/minimax-cli3/20260815` @ `8ce767f33b`

---

## 0. Executive summary — blunt assessment

The master's dissertation is **code-complete but manuscript-pending**. The
two QUALIFICATION_STATUS files (2026-06-13 and 2026-06-23) report 6/6 CI
gates green; the SOLO chapter-prose handoff packet
(`handoff/fo_pk_method_science_package.md`, 336 lines, math-reviewed) was
created on 2026-07-31 and constitutes the only piece of dissertation-grade
prose produced in the last two months. **No chapter file
(`chapter_01.md` … `chapter_06.md`, in PT or EN) exists in the
repository.** The Aug-Sep 2026 window is **not achievable** for a full
manuscript at the quality standard the truth table
(`pbpk_claim_truth_table.md`) and the two qualification status reports
imply. The honest options are: (a) request a deferral, (b) submit a
substantially shorter, narrower-scope dissertation that does not claim
the seven contributions enumerated in `VISAO_GERAL.md`, or (c) bring in
a co-author to write the prose in parallel.

The two issues that loomed in June — the three "unauthored" modules and
the `Knowledge<T>` ε/provenance gap — are both *partially* closed; the
truth table row update on 2026-06-29 and the git history through
2026-08-11 are accurate. These are no longer the binding constraint.
**The binding constraint is that the manuscript itself does not exist.**

---

## 1. Per-chapter table

For each chapter implied by `VISAO_GERAL.md` (a 7-chapter thesis in
PT-BR), the table below records what actually exists in the repository
as of HEAD `8ce767f33b`. The column "First-class prose" requires the
file to read as a draft chapter — introduction, body sections, and
references — not an outline, handoff packet, or template.

| # | Chapter (template title) | First-class prose | Outline / handoff | Nothing | Expected LOC |
|---|---|---|---|---|---|
| 1 | Introdução / Motivation | **none** | none | ✓ | 8–12 |
| 2 | Background & Related Work | **none** | none | ✓ | 15–20 |
| 3 | PBPK14 well-stirred model | **none** | none | ✓ | 10–15 |
| 4 | PBPK28 permeability-limited model | **none** | `docs/dissertation/handoff/chapter_04.md` (258 lines, handoff packet — not prose) | mostly | 18–25 |
| 5 | `Knowledge<T>` epistemic arithmetic | **none** | none | ✓ | 12–18 |
| 6 | Clinical case studies (11 drug arms) | **none** | `docs/dissertation/chapter_clinical_verified_outline.md` (10-section outline, 8.7 kB) | mostly | 25–35 |
| 7 | Lean 4 proof obligations | **none** | none | ✓ | 6–10 |
| 8 | Dissertação viewer demo | **none** | none | ✓ | 4–6 |
| Back | Conclusões / Future work | **none** | none | ✓ | 2–4 |
| Back | References | **none** | none | ✓ | 6–10 |

**Single existing dissertation prose file in the repo:**
`docs/research/bbb_pbpk_dissertation_chapter.md` (472 lines, marked
"historical" in its frontmatter). It is a 2017-era backstory document
on BBB-PBPK and is **not** part of the current outline.

**Single chapter-prose handoff packet produced this cycle:**
`docs/dissertation/handoff/fo_pk_method_science_package.md` (336 lines,
added in commit `dad708ee7d` on 2026-07-31, math-reviewed by xAI + Z.AI).
This is **a methods/results package for Madaros FO oral Css freezes**,
not a chapter draft. It is the closest thing to chapter prose that
exists, and it is named `handoff/` because it is a packet the founder
will paste into the manuscript — the band of prose has not yet been
adapted into chapter shape.

**Other `handoff/` packets (also not chapter prose):**
- `handoff/section_4_10_sobol_hdmr_package.md` (17.1 kB)
- `handoff/psychiatric_pgx_mtor_168_pop_package.md` (51.8 kB)
- `handoff/fo_pk_method_science_package.md` (22.7 kB)
- `handoff/chapter_04.md` (17.0 kB)

These are well-organised source material for §4.10, the psychiatric
PGx/mTOR section, the FO PK methods section, and §4 respectively. They
are not prose.

**Templates / scaffolds:**
- `docs/dissertation/dossier_template.md` (6.4 kB, §1–§10 skeleton).
- `docs/dissertation/audit/discovery_log.md` (13.9 kB, 2026-05-11
  snapshot — superseded by the 2026-06-29 truth-table update).
- `docs/dissertation/audit/gap_report.json` (47.8 kB).

**Result receipts (numerics, not prose):** 28 files in
`docs/dissertation/results/` — these are the gate receipts and
quantitative annexes the chapters would cite. Most are 2–14 kB. They
are evidence, not narrative.

**Read of the situation:** the artefact corpus is broad and in many
places well-organised. The narrative that ties the artefact corpus
into a single thesis does not exist. Recipe: chapter prose of the
length above would be ~115–175 pages of writing. There is, on the
LAN-bound branch, **0 pages** of it.

---

## 2. What the two QUALIFICATION_STATUS files tell us

`docs/dissertation/QUALIFICATION_STATUS_2026-06-13.md` (11.8 kB):

- Branch `fix/dissertation-confidence-gate`.
- 6/6 CI gates PASS.
- 50/53 activos + 3 PENDING.
- Knowledge<T> constructor was "just restored" — a recent migration.
- Defense target 2026-09-22.

`docs/dissertation/QUALIFICATION_STATUS_2026-06-23.md` (4.5 kB):

- Branch `codex/website-living-language-plan`.
- 6/6 CI gates PASS.
- 51/53 activos + 2 PENDING.
- `rapamycin_kaxi_fuse_prior` un-PENDED (the Seq\<T> restoration).
- **NEW ISSUE:** Madaros backend bugs surfaced
  (`println(var)` segfault); workaround applied without rebuilding
  Madaros.

**Delta between the two reports (10 days):**

1. Branch moved from `fix/dissertation-confidence-gate` to
   `codex/website-living-language-plan`. Implications: the
   qualification results were captured on ephemeral branches and may
   not be reproducible from `main` without re-running all six gates.
2. One PENDING was cleared (the K-AXI fusion witness via Seq\<T>
   restoration).
3. Madaros backend bugs surfaced in the interim, and a workaround was
   applied without rebuilding Madaros. This is a **fragile** state:
   the gate may be green because of a workaround that is not durable
   across a clean Madaros build.
4. Both reports are 10–54 days old; the current branch
   `lane/minimax-cli3/20260815` is neither of those branches, and the
   latest merged dissertation-related work on `main` is the FO PK
   oral Css prose handoff (commit `dad708ee7d`, 2026-07-31) and the
   FO PK R5–R12 freezes. Whether the 6/6 gate result still holds on
   `main` at HEAD `8ce767f33b` has **not been re-verified** in this
   audit (read-only constraint); the gate scripts themselves are
   present and runnable (`scripts/ci/dissertation_*.sh`).

---

## 3. The three "unauthored" modules — verified

The 2026-05-11 `audit/discovery_log.md` flagged that
`core/tissue_composition.sio`, `core/peptide_partitioning.sio`, and
`tmdd/qe_approximation.sio` did not exist. The 2026-06-29
`pbpk_claim_truth_table.md` row update says they are now authored.

**Verification on disk at HEAD `8ce767f33b`:**

| Module | Path | LOC | Truth-table claim | Match |
|---|---|---|---|---|
| `tissue_composition.sio` | `stdlib/darwin_pbpk/core/tissue_composition.sio` | 172 | 172 | ✓ |
| `peptide_partitioning.sio` | `stdlib/darwin_pbpk/core/peptide_partitioning.sio` | 138 | 138 | ✓ |
| `qe_approximation.sio` | `stdlib/darwin_pbpk/tmdd/qe_approximation.sio` | 134 | 134 | ✓ |

Authored in commit `13d16b1e96` (2026-05-31, "feat(stdlib/pbpk): add
peptide partitioning, tissue composition, QE approximation modules").
Two further pre-commit PRs (`4e2f55c11a`, `b64e2f0139`) and the
single-LOC PR (`335fe66587`) attest to the split landing.

**Honest caveat from the truth table itself:** these are described as
"standalone/reference modules — no hard use-import wires them into the
active solver yet." In other words, they exist and compile, but the
PBPK28 run path does not actually call them. The June 13/23 gate
results may therefore be green on a §4 that *does not* include their
contributions. The dissertation §4 prose would need to be honest about
this — present them as the validated reference implementations behind
§4.6–§4.8, not as load-bearing solver code.

---

## 4. The `Knowledge<T>` ε/provenance gap — partially closed

Two layers to this gap:

**Layer A — struct-level "all priors sourced as `Knowledge<T>`":**
CLOSED. Three commits in sequence:

- `7295004446` — "feat(pbpk28): declare priors as Knowledge\<T>
  (per-parameter epistemic confidence)."
- `77e8111864` — "refactor(pbpk28): EpPrior28 is single-source
  Knowledge\<T> — delete f64 arrays."
- `2e67d5a9da` — "feat(pbpk28): ep28_run consumes the Knowledge\<T>
  priors directly."

`stdlib/darwin_pbpk/epistemic_pbpk28.sio` is now 813 LOC and contains
35 references to `Knowledge` / `epistemic` — the single-source
construction is in place.

**Layer B — gate-through-kernel enforcement (compile-time):**
**OPEN.** The truth table notes: "ep28_run enforces/propagates the ε
at compile time" — it does **not**. The `EpistemicComplete` confidence
gate is enforced at the construction site (the `Knowledge<T>::new`
call), but reading a `Knowledge<T>` field out of a struct and
threading it through further computation does not trigger the gate
again. The dissertation §5.4 (machine-checked type safety) and §5.5
(E-Measure) work covers *value-carrying* Knowledge by other routes,
but the "every read of `kn[i].value` is checked against the live
confidence bound" claim is not currently a compile-time property.

The truth table has a "Do not say" forbidden wording for the through-
read: in dissertation prose, claim "compile-time confidence at
construction site" and "ε propagated through run-time reads per
calibration" — do **not** claim "compile-time ε propagation through
struct-field reads."

**Implication for the manuscript:** the §5 narrative can describe
Layer A as a real result. Layer B must be described as future work
(the post-defense "inference pass" mentioned in the truth table).

---

## 5. What does the truth table actually claim?

`pbpk_claim_truth_table.md` (34.0 kB, 2026-06-29 version) is the
narrowing-claim ledger. It uses four status vocabularies:

- `repo-backed` — a file path is named; the gate is named.
- `experimental` — the recipe + numbers are presented, but no gate
  pins them.
- `future-work` — out of scope for the defense.
- `unsupported/overclaim` — forbidden wording.
- `unsupported/benchmark-needed` — direction noted, benchmark absent.

By spot count, ~25 rows are `repo-backed`. The three "unauthored"
modules + the K-AXI fusion witness + the Knowledge\<T> struct-level
migration are all `repo-backed` as of June 29. The two remaining
PENDING (`pbpk28_rapamycin_clinical`, `pbpk28_semaglutide_clinical`)
are awaiting literature observed data — pending does not mean
blocking, but it does mean the §6 clinical validation is incomplete
for those two arms.

**Forbidden wording the dissertation must avoid** (from the truth
table's "Do not say" entries):

- "GPU PBPK14 single-kernel" — the single-kernel variant is not the
  shipped artefact.
- "10–100× speedup" — speedup claims require a benchmark that is not
  yet public.
- "compile-time ε propagation through field reads" — see §4 Layer B.

The manuscript will need to use the table's "narrowest claim" wording
for every numeric claim. This is a discipline constraint, not a
freshness constraint — it does not close the gap of "no chapter prose
exists," but it does mean the prose that *does* eventually get
written has a tight claim ledger behind it.

---

## 6. What does the live dissertation viewer actually demonstrate?

**Repo side (verified in this audit):**

- `website/src/lib/pbpk28_core.mjs` — 1098 LOC (the JS port of the
  PBPK28 solver).
- `website/src/lib/pbpk14_core.mjs` — 161 LOC.
- `website/src/components/dissertation/` — 18 React/TSX components
  totalling 2239 LOC:
  - `DissertationViewer.tsx` (499 lines, the host)
  - `Compartments.tsx`, `Stent.tsx`, `SCDepot.tsx`, `TmddPanel.tsx`,
    `PdReadoutPanel.tsx`, `ConfidenceGate.tsx`, `HessianHeatmap.tsx`,
    `GumBudgetBar.tsx`, `TimeScrubber.tsx`, `TourControls.tsx`,
    `DrugSelector.tsx`, `CameraDirector.tsx`, `BloodFlowEdges.tsx`,
    `Silhouette.tsx`, `OrganDetailModal.tsx`, `InfoPopover.tsx`.
- `website/src/components/dissertation/tours.ts` — 196 LOC, 11
  rapamycin/semaglutide mentions in stage-G commits.
- `website/src/components/dissertation/compartments.ts` — 92 LOC.
- `website/src/pages/dissertation/index.astro` — page entry.

Total viewer surface: ~3.8 kLOC of TypeScript + ported solver.

**History (committed 2026-04 to 2026-05):**

- `48abc65d65` — Lane 9 MVP: 3D interactive PBPK14 viewer.
- `79ac12d909` — Stage D: side panels + KaTeX organ modal + time
  scrubber.
- `ac6514a7ce` — Stage E: guided tours, snapshot, reduced-motion.
- `eab5a45308` — Stage G-ε-1: drug A/B selector + unified PBPK hook.
- `4f052e605d` — Stage G-ε-2: TMDD occupancy + PD readout panels.
- `ff3f43fb4d` — Stage G-ε-3: visual release-source swap (Cypher
  ↔ SC depot).
- `169c33f9fa` — Stage G-ε-4: per-drug patient sliders + release
  scale.
- `84463c1f61` — Stage G-ε-5: per-drug tours + D/T keyboard + tagged
  snapshots.
- `890971ea02` — Stage G-ε-6: per-drug Phase J evidence bands.
- `27e40dbb4a` — Stage G-ε-8: drug-aware absolute concentration.
- `c9911d3706` — Stage G-ε-9: drug-aware OrganDetailModal with TMDD
  + PD ODE blocks.
- `c72fb424e0` — "website: clinical funnel, PT defense copy, mobile
  polish" (PT-BR localisation landed; defense copy in place).

**Deployment state:** the repository has the viewer source; the
`ab7bf29dc8` commit ("chore(website): disable Vercel git auto-deploy")
shows **public auto-deploy was disabled at some point**. Whether the
viewer is currently reachable at `souniolang.org/dissertation` is
**not verifiable from inside the repo** — this is an external
dependency. The `ADVISOR_HANDOFF.md` document describes a 6-tour
walkthrough at that URL. If the live site is currently down or at a
different URL, the §8 viewer-demo chapter prose would need to either
re-deploy it or freeze screenshots.

**Honest assessment:** the viewer is the *most* complete surface in
the project. It is also the *most* portable: it is a static
Astro/React site with the solver ported to JS. The risk for the
defense is **not** "the viewer is broken" — it is "the chapter that
narrates the viewer does not exist."

---

## 7. What blocks submission — explicit list

Listed in priority order (most binding first):

1. **No chapter prose exists.** The hard blocker. 0 pages of
   dissertation narrative have been written against the current
   outline. The 37-day window is insufficient for a 7-chapter
   ~120–175 page thesis at the writing quality a master's defense
   demands. **This is the closing-the-gap-while-keeping-claim-truth
   constraint that determines the realistic outcome.**
2. **The 2026-06-13 and 2026-06-23 gate reports are on different
   branches from the current `main` and from each other.** The 6/6
   green status has not been re-verified on the current `main` HEAD
   since the Madaros workaround was applied. A fresh
   `dissertation_*_gate.sh` run on `main` is the first pre-defense
   gate.
3. **Two PENDING clinical cases:** `pbpk28_rapamycin_clinical` and
   `pbpk28_semaglutide_clinical` (per June 23 report). These are
   closed by acquiring literature observed data and matching it to
   the PBPK28 prior. The §6 chapter prose cannot finalise without
   these two rows.
4. **Knowledge\<T> through-kernel gate (Layer B):** not a blocker if
   the prose stays within the truth-table "narrowest claim" wording;
   is a blocker if §5 prose drifts into "the compiler enforces ε
   everywhere."
5. **Three standalone modules (§3) are not wired into the active
   solver.** Same as (4): not a blocker if §4 describes them as
   reference implementations, but a blocker if §4 claims they are
   the load-bearing solver elements.
6. **Madaros backend workaround** (the `println(var)` segfault noted
   on 2026-06-23) — if a clean Madaros rebuild is performed before
   the defense demo, the workaround may need to be re-applied or
   the underlying bug fixed. Risk: silently regressions during the
   demo.
7. **Live viewer deployment** — `chore(website): disable Vercel git
   auto-deploy` is in the history. The defense walkthrough
   (`ADVISOR_HANDOFF.md`) references `souniolang.org/dissertation`.
   If the site is not currently live, §8 needs screenshots or a
   re-deploy.
8. **Translation status:** `c72fb424e0` added PT-BR defense copy; the
   manuscript itself is in PT-BR per `VISAO_GERAL.md`. No EN version
   is in scope per `VISAO_GERAL.md`.

---

## 8. Realistic assessment of the Aug-Sep 2026 window

**Today:** 2026-08-16. **Defense:** 2026-09-22. **Window:** 37 days.

**What 37 days can produce:**

- A 30–40 page "monograph" chapter on §4 (PBPK28) using the existing
  `handoff/chapter_04.md` packet + result receipts. Tight scope.
- §1 Introduction (8–10 pages) from `VISAO_GERAL.md` + truth table.
- §2 Background (15 pages) using existing review-paper literature.
- A §8 Viewer demo chapter (5–8 pages) using screenshots / the live
  site.
- A revised §5 (10 pages) limited to Knowledge\<T> Layer A only.
- A bibliography.

**What 37 days cannot produce at the quality standard the truth table
implies:**

- A 7-chapter ~120–175 page thesis with §3 (PBPK14), §5 Layer B,
  §6 (all 11 drug arms), §7 (Lean 4 obligation walkthrough), §8
  (viewer demo with reproducible A/B), and a properly framed
  introduction / background / conclusion arc.
- Two clinical PBPK28 cases pending observed data (`rapamycin` and
  `semaglutide`) cannot be closed in 37 days if the literature
  acquisition is the bottleneck (it usually is).
- A dissertation defensible against the "what is novel?" question
  for all seven contributions listed in `VISAO_GERAL.md`.

**Three honest options from where things actually stand:**

A. **Defer the defense to Dec 2026 / Mar 2027.** This is the
   scientifically honest path. The artefact is fundamentally done
   (gates green, modules authored, viewer polished, knowledge layer
   partial), but the manuscript is not. A 4–6 month delay buys
   enough time to write the prose properly.

B. **Submit a narrower-scope thesis.** Drop the §4 well-stirred
   PBPK14 model from the contribution list (or relegate it to a
   background appendix). Drop Layer B of Knowledge\<T> to the
   future-work section. Cap §6 to the 9 drug arms with
   repo-backed evidence (drop the two PENDING cases). The resulting
   thesis is ~70–90 pages, writeable in 37 days, and is honest
   about its scope. The reviewer can argue this is "too
   conservative" — but the truth table already supports it.

C. **Bring in a co-author to write the prose in parallel.** This
   carries the highest risk of fabricating paragraphs (the worst
   outcome the user's instruction explicitly forbids) but is the
   only path that preserves the original 7-chapter scope on a
   37-day timeline. A co-author writing **chapters** is not the
   same as a co-author writing **technical content**; the latter
   is fine, the former is exactly what the user's directive
   forbids.

**Recommended path:** (A) deferral, with (B) as a fallback if the
defense date is immovable. The artefact is excellent; the manuscript
is the missing piece; deferral is the way to fix that without
compromising either.

---

## 9. Source-of-truth pointers

- `docs/dissertation/QUALIFICATION_STATUS_2026-06-13.md` — earlier
  gate report (branch `fix/dissertation-confidence-gate`).
- `docs/dissertation/QUALIFICATION_STATUS_2026-06-23.md` — later
  gate report (branch `codex/website-living-language-plan`).
- `docs/dissertation/VISAO_GERAL.md` — schedule + seven contributions.
- `docs/dissertation/ADVISOR_HANDOFF.md` — advisor walkthrough +
  viewer narrative.
- `docs/dissertation/pbpk_claim_truth_table.md` — claim ledger.
- `docs/dissertation/chapter_clinical_verified_outline.md` — §6
  outline.
- `docs/dissertation/handoff/*.md` — 4 handoff packets (methods,
  population, §4.10, §4).
- `docs/dissertation/audit/discovery_log.md` — 2026-05-11 snapshot
  (superseded by the June 29 truth-table update).
- `docs/dissertation/audit/gap_report.json` — 2026-05-21 gap
  classification (also superseded by the truth-table update).
- `docs/dissertation/dossier_template.md` — §1–§10 skeleton.
- `docs/dissertation/results/` — 28 quantitative annexes.
- `docs/research/bbb_pbpk_dissertation_chapter.md` — historical
  2017 chapter (not part of current outline).
- `stdlib/darwin_pbpk/core/tissue_composition.sio` (172 LOC,
  commit `13d16b1e96`).
- `stdlib/darwin_pbpk/core/peptide_partitioning.sio` (138 LOC,
  commit `13d16b1e96`).
- `stdlib/darwin_pbpk/tmdd/qe_approximation.sio` (134 LOC,
  commit `13d16b1e96`).
- `stdlib/darwin_pbpk/epistemic_pbpk28.sio` (813 LOC, Knowledge\<T>
  — commits `7295004446`, `77e8111864`, `2e67d5a9da`).
- `stdlib/darwin_pbpk/epistemic_pbpk28_hessian.sio` (591 LOC).
- `website/src/lib/pbpk28_core.mjs` (1098 LOC) +
  `pbpk14_core.mjs` (161 LOC).
- `website/src/components/dissertation/` — 18 components, 2239 LOC.
- `website/src/pages/dissertation/index.astro` — page entry.
- `scripts/ci/dissertation_*.sh` — 6 CI gate scripts.

---

## 10. What this audit does NOT establish

- It does **not** measure whether the 6/6 gate result still holds on
  `main` at HEAD `8ce767f33b`. The audit is read-only; running the
  gates is a separate lane.
- It does **not** verify whether `souniolang.org/dissertation` is
  currently reachable. That is an external dependency.
- It does **not** evaluate the prose quality of any chapter — there
  are no chapters to evaluate.
- It does **not** propose dissertation prose. Per the user's
  directive, prose authorship is reserved for the founder.

---

## 11. Lane closeout

Lane `dissertation-status-audit`, agent `minimax-cli3`, intent
"honest artefact-vs-manuscript audit ahead of the Aug-Sep defense",
files `docs/dissertation/STATUS_AUDIT_2026-08-16.md`. Audit's only
output is this document.

Recommendation for the lane that picks up after this one: rerun the 6
dissertation CI gates on `main` HEAD and update
`pbpk_claim_truth_table.md` with the 2026-08-16 row. That is the
smallest concrete next step that does not require prose authorship.
