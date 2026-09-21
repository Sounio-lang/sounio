<!-- docs:meta
topic_id: repo.docs.papers.oopsla2027.outline
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.papers.oopsla2027.outline
-->

# Self-Falsifying Compilation — paper skeleton (OOPSLA 2027)

**Status:** `SKELETON` — `PAPER_SKELETON_TOKEN_BOUND__NOVELTY_NARROWED_BY_SEARCH__CI_WIRED`
**Date:** 2026-07-26
**Harness:** `scripts/research/self_falsifying_compilation_line_r5_contract.py`
**Gate:** `scripts/ci/self_falsifying_compilation_line_r5_gate.sh`

> This is a **skeleton**, not a draft: an argument structure in which every
> empirical claim is bound to the rung that measured it. The R5 gate fails if a
> verdict token cited here disagrees with the one its spec declares, so the
> paper cannot drift away from its evidence while prose is written on top of it.
> Chain of custody: **paper → spec** is checked here; **spec → contract** is
> checked by each rung's own gate.
>
> **That chain is now executed.** All **21** of the line's gates are invoked by
> `.github/workflows/ci.yml`; `W4_CI_WIRING` measures it and the token has
> flipped to `__CI_WIRED`. This paragraph said it would have to change when
> that happened, and this is it.
>
> **What wiring cost, and what it immediately found.** 20 gates run per push for
> ~200 s; R6's corpus sweep costs **957 s** — O(pairs × functions²) over a corpus
> grown from 47 files to 61 — so it runs **nightly** rather than being dropped.
> A check too expensive for every push is not thereby exempt from running; it
> runs on a cadence the repo can pay for, and the number is in the workflow.
>
> The compile arms (`SFCL_R2/R17/R20_RUN_COMPILE`) are deliberately **not**
> wired: each needs a purpose-built self-hosted compiler, ~40 min per push.
> Their behaviour receipts are bound to the executor's sha256, so editing the
> executor turns those rungs **red** in CI rather than certifying stale
> behaviour.
>
> **Two gates were already red the first time they ran.** R1's bound-claim count
> had gone stale (15 → 16) the moment R18 added a claim to the manifest, and
> R2's surface clause pinned a decision-variable name that R17 and R20
> legitimately renamed while behaviour stayed identical. Both had been wrong for
> as long as nobody executed them — which is precisely the condition this
> paragraph used to describe.

---

## Working title

*Where Did the Evidence Come From? Compile-Time Claims, Their Limits, and
Measuring Whether a Corpus Corroborates Itself*

## Thesis in one paragraph

In scientific software the correctness of a program is contingent on premises
that live **outside the source**. We build a compiler that treats them as a
compile-time obligation — it runs each claim's check before code generation and
refuses to emit an artifact whose premises no longer hold — and then measure,
against a real scientific codebase's own history of self-correction, whether it
would have caught anything. **It would not**, and the reason is precise: the
failures were claim and check being wrong *together*, which nothing consulting
only the claim's own check can detect. That impossibility is an **antecedent,
not a wall**. Changing it — requiring evidence the claim's author did not
supply, and making the **independence of that evidence machine-checkable** —
turns out to be both cheap and sharp: it showed that **a third of this corpus's
checks are not independent evidence of one another**, collapsed 51 contracts'
shared foundation to essentially one function, and found a second independent
derivation of that function already sitting unused in the repository.

---

## 1. Contributions, each bound to a measured rung

### Part I — the compile-time obligation, and its limit

| # | Contribution | Rung | Verdict token |
|---|---|---|---|
| C1 | An implementation of claim-gated code generation in a self-hosted compiler. **Not a novel capability** (cf. `build.rs`, §8.1); reported because the rest is measured on it. | R0 | `SUBSTRATE_LIVE__CORPUS_BOUND__HISTORICAL_FAILURES_ARE_INTERPRETIVE` |
| C2 | **What it costs to attach such a guard to a real corpus** (R1), and **a limitation of this mechanism removed** (R29). Verification ran on the main source file only, so a refuted claim in an imported module was never checked; it now walks the transitive import closure, so under `--verify-claims` a premise refuted anywhere in that closure blocks the build. Measured past one hop: a claim refuted **two** imports away blocks (`modules=3`), and a diamond visits its shared leaf **once** (`modules=4, pass=4`). Propagation across dependency edges is **not** novel — a failing `build.rs` already fails its dependents (§8.1) — so what is claimed here is the cost measurement and the repair, not the propagation. | R1, R29 | `BOUND_16__MODULE_CLOSURE_PASSES`; `CLOSURE_WALKED__MODULE_CLOSURE_PASSES` |
| C3 | **Verdict-token binding**: bind the build to the *proposition* a check reports, where prior art binds an exit status or a literal output. | R2 | `TOKEN_BINDING_IMPLEMENTED__CATCHES_DRIFT_NOT_MISINTERPRETATION` |
| C4 | *Drift* vs *shared misinterpretation*, with an argument that the latter is out of reach, and a test of what does reach it. | R3 | `FALSIFIERS_NONVACUOUS_ONLY_FOR_CLOSED_FORM_CLAIMS` |
| C5 | A retrospective under a predicate **fixed before the study ran**: a negative result and a degenerate arm, reported as such. | R4 | `RETROSPECTIVE_RUN__SOME_ARM_FIRED` |

### Part II — changing the antecedent

| # | Contribution | Rung | Verdict token |
|---|---|---|---|
| C6 | **Evidential independence as a static property.** Prior art binds *what* a check reports; nothing asks **where the evidence came from**. Measured: **343/1081 contract pairs share a derivation**. **Read in one direction only** — see C12 and §8.6. | R6 | `INDEPENDENCE_CHECKABLE__CORROBORATION_BINDS` |
| C7 | Auditing what the sharing exposes: the shared kernel checked against an independent derivation over 5 440 products. | R7 | `SHARED_KERNEL_CORROBORATED` |
| C8 | **The trusted base of a research corpus**: enumerate shared derivations, rank by blast radius, **collapse wrappers into kernels**, audit what is irreducible. 23 clusters → 12 kernels; 51 contracts rest on essentially one function. | R8 | `TRUSTED_BASE_MAPPED__KERNELS_AGREE` |
| C9 | Completing the audit, and **measuring the method's boundary**: 6 kernels corroborated, 2 with no adjudicator, and a principled reason for each. | R9 | `TRUSTED_BASE_PARTIALLY_AUDITABLE` |
| C10 | **Corroboration depth as a corpus metric**, and a procedure that finds latent corroborations from source alone — validated by rediscovering the one a human found by reading. Depth 1 is the corpus's normal state. | R10 | `LATENT_CORROBORATION_FOUND` |
| C11 | The procedure widened as far as isolation permits: **4 behaviour classes, zero new corroborations.** Wrappers are structurally unprobeable under isolation, and by C8's collapse probing them would add nothing. | R11 | `WIDER_PROBE__NO_NEW_PREEXISTING_CORROBORATION` |
| C12 | **The fourth narrowing, and the one that withdraws a planned contribution.** C6's measure is not new and its central assumption is refuted by a study at 224 problems × 12 models; C6 survives one-sided, and the compiler rule it motivated is withdrawn rather than deferred (§8.6). | R12 | `PRIOR_ART_HAS_ARTEFACT_MEASURE__CLAIM_NARROWS_FOURTH` |
| C13 | **The one-sidedness measured here, not transferred.** Perturb the shared object rather than the code: **21 pairs of this corpus's contracts have identical responses to all 36 perturbations while their structural similarity is 0.479–0.594**, below the threshold at which C6 calls them independent evidence. Pairs C6 calls independent agree *more* (0.565) than pairs it calls shared (0.513). | R13 | `STRUCTURAL_INDEPENDENCE_DOES_NOT_IMPLY_INDEPENDENT_FATE` |
| C14 | **The same instrument turned around, and a pre-registered hypothesis losing.** Of everything these contracts compute — to level 10, 1024 dimensions — how much does the conclusion depend on? **407 verdict changes, 117 crashes, 12 survivors in 536 cells**; levels 9 and 10 are pure verdict changes. Vacuity refuted; the corpus checks what it computes. | R14 | `VACUITY_REFUTED__CORPUS_CHECKS_WHAT_IT_COMPUTES` |
| C15 | **The limit above C3, and its repair.** C3 binds the build to the *proposition* a check reports. That is still blind to anything preserving the proposition's truth: a flip changing **126 of 128 fiber graphs and every spectrum** leaves `#spectra = 24` intact, so the token holds. A token's resolution is bounded by the **invariance group of its proposition**. Repair, verified: **bind the witness, not the predicate**. | R15 | `TOKEN_RESOLUTION_BOUNDED_BY_PROPOSITION_INVARIANCE` |
| C16 | **The invariance group, identified.** The blind spot is not "maps preserving the count" but maps acting **within the blocks** of the classification: the flip preserves the *identical set partition* of fibers (sizes [1,7,7] / [1,1,7,7,7,8] / [1,1,1,1,7,7,7,7,7,7,8,9]) and relabels every block, changing exactly **2 edges per fiber** because the perturbed pair's home fiber is the one the check never examines. **Any claim of the form "there are exactly N equivalence classes" has this blind spot by construction.** | R16 | `INVARIANCE_GROUP_IS_PARTITION_PRESERVING_NOT_MERELY_COUNT_PRESERVING` |


Eleven further rungs — R17, R19–R28 — are results of the same line that this
skeleton does not plan to argue. They are indexed in Appendix A so that a rung
is not quietly dropped; none is a contribution of the paper.

### Methodological results that generalise

- **Behaviour receipts (R2).** The rung's own contract certified the mechanism
  as implemented while the compiler built from that source segfaulted on every
  claim. Certification is now bound to a receipt of an actual run, hashed to the
  source it attests. *The tooling committed the error the paper is about.*
- **Degenerate-predicate detection (R4).** The retrospective's one firing arm
  fired by construction. Reporting that, rather than counting it, is the
  difference between a study and a story.
- **The tool catching its own author (R8).** A draft claimed four independent
  derivations; the measured independence matrix said three — two were the same
  derivation at similarity 0.929. The error came from inferring independence by
  eye, which is exactly what C6 replaces.

---

## 2. Motivation

Scientific code's premises are empirical and external; type systems, contracts
and refinement types describe the *program*, not the world. Failure is silent.
Concretely: a repository where a claim's headline was wrong for three commits
while every check stayed green — and, measured later, while the check *executed*
and emitted exactly the token the claim declared.

## 3. Mechanism (C1)

Claim syntax; `--verify-claims`; placement after type-check, before codegen.
Sandbox: `fork`/`execve` with fixed argv, no shell interpolation, per-gate
timeout, capture via `open`+`dup2` rather than a shell redirect — the naive way
to read a gate's output would trade the no-interpolation property away.

## 4. Verdict-token binding (C3)

Exit code binds a computation; a token binds a proposition. `MISMATCH` and
`ABSENT` both falsify; absent fails **closed**. Every probe gate exits 0, so
exit-code gating cannot account for the result.

## 5. What it cannot catch (C4, C5)

Definitions of *drift* and *shared misinterpretation*; the proposition that no
compile-time procedure whose only evidence is the claim's own check can detect
the latter; executable falsifiers as a partial escape — non-vacuous only for
claims reducing to a closed form, and **not self-starting**.

## 6. Changing the antecedent (C6–C9)

- Independence made checkable. **Import-closure disjointness is vacuous** on a
  corpus of self-contained scripts — the formalisation a reader proposes first,
  reported rather than dropped. Derivation disjointness is what discriminates.
- The **ceiling, stated before the results**: a checkable *lower bound*. Two
  files can share no code and encode the same misunderstanding; R3 demonstrated
  exactly that.
- The trusted base: blast radius, wrapper collapse, and the fact that finding
  the base is what makes the audit affordable — minutes to check one function,
  the whole research programme to re-derive what it supports.
- **The corpus already contained its own independent check**, unused. That is
  the cheapest form the method takes.
- Auditing predictive kernels against ground truth computed a different way
  (rank-deficiency of the left-multiplication matrix). The kernels that *could*
  have been false hold.

## 7. Evaluation

Corpus binding and its bindability criteria, discovered by trying: per-gate time
budget, **hermeticity** (a gate that rewrites tracked files makes builds
non-idempotent), and the existence of a declared token. Module closure: the
limitation at R1, its removal at R29, the depth-2 and diamond probes that take
the claim past one hop, and the arms that bound it — a census showing the reached set
is today unchanged, the opt-in flag, and a mixed green/red import pair reported
as a pass and a failure in one run, which is what rules out a compiler that
simply fails whatever it imports. The
retrospective, with its buckets and its `UNCLASSIFIABLE` never redistributed.
The independence sweep over 1 081 pairs. The trusted-base audit.

## 8. Related work — **load-bearing neighbours CHECKED (2026-07-26); every novelty claim was narrowed as a result**

Searching this section changed the paper **three times**, and the pattern held
every time: the **technique** was already old, and only a narrower **semantics**
survived. Each search cost minutes and removed a claim that would not have
survived review.

### 8.1 The compile-time mechanism is not new — checked

- **Cargo build scripts (`build.rs`)** run arbitrary code before the main
  compilation; if the script fails, the build fails. "An external process runs
  before compilation and can abort it" is standard practice, not a contribution.
- **Snapshot / approval testing** (Jest, Vitest, golden masters) binds a
  declared expected output to actual output and fails on mismatch, gating the
  merge.

**What survives:** `build.rs` binds a check's **exit status**; snapshot testing
binds its **literal output**. Neither binds a **proposition the source
declares** — neither has a place to write *what the check is supposed to
establish* — so neither can detect a check that still succeeds while
establishing something else.

### 8.2 The independence technique is not new either — checked

- **Code clone detection** is decades old. Type-2 clones are fragments identical
  up to identifier and literal names; AST- and token-based detectors (CCFinder,
  NiCad, tree-based methods) find them by exactly the normalisation this
  paper's harness uses.

**What survives:** the technique is clone detection; the **framing is not**.
Clone detection is posed as a *maintainability* problem — duplication is debt.
Here the same measurement answers an *epistemic* question: **do these checks
corroborate each other, or are they one check restated?** Plus two things a
clone report does not give you — the **wrapper collapse** that turns a clone
list into an irreducible base, and **auditing that base against independently
computed ground truth**.

### 8.2b And the epistemic idea is not new either — checked, third narrowing

The reproducibility literature's standard taxonomy already separates **rerunning
the same artifact** from an **independent re-implementation**, and already
treats the latter as the stronger evidence. So "an independent derivation
corroborates more than a re-run" is established vocabulary, not a finding here.
(Consulted: Sinha et al., *Reproducibility, Replicability, and Repeatability*
survey, arXiv 2402.07530 — read in summary only, the PDF did not yield verbatim
definitions, so it is cited as orientation and must be quoted properly before
submission.)

**What survives, and it is now the whole of C6:** that taxonomy classifies
*studies*, by human judgement, after the fact. This work makes it a
**mechanically decidable property of a single codebase's own internal checks** —
which of *my* checks are independent of *each other*, computed without asking
anyone, and used to decide where auditing pays.

**A tension worth stating, since it cuts against standard guidance.**
Reproducibility practice says *share the code so others can re-run it*, which
maximises repeatability. Corroboration wants the opposite: a second check that
shares no derivation with the first. Both are goods; they are not the same good,
and a corpus that maximises one can score zero on the other while looking
healthy. This corpus did: 343 pairs of mutually re-runnable, mutually
non-corroborating checks.

### 8.3 Nearest neighbours in scientific software — checked

- **Continuous analysis** (Beaulieu-Jones & Greene, *Nature Biotechnology* 2017)
  — Docker + CI re-running an analysis whenever code or data change. Runs
  *alongside* the build; does not withhold an artifact; binds a pipeline rather
  than a stated proposition.
- **Executable papers** (*Toward Executable Scientific Publications*, Procedia
  CS 2011) — the document is the artifact; nothing blocks code generation.

### 8.4 Distinct by definition, argued rather than searched

Design-by-contract, refinement types, `constexpr`/`comptime`/staging, certified
compilation and proof-carrying code all describe or prove properties **of the
program**. The premises here are propositions **about the world**. Reproducible
and hermetic builds are the *opposite* design goal.

### 8.5 Residual risk

The searches were targeted, not exhaustive; a systematic pass remains before
submission. Searching for clone detection applied to research software as a
*validity* question surfaced only code-*sharing*-for-reproducibility work and
duplication-as-debt case studies — nothing measuring whether a corpus's own
checks corroborate one another. That is weak evidence of absence, not evidence
of novelty.

**Specifically not yet checked, in priority order:** (i) mutation-testing and
test-suite-independence literature, which asks a structurally similar question
about tests rather than about scientific checks; (iii) whether any
reproducibility-badging scheme already requires demonstrated implementation
independence rather than accepting a declaration.

### 8.6 (ii) N-version programming — CHECKED, and it narrowed C6 a fourth time

This section previously listed N-version programming as *"the most dangerous to
C6 — decades of work on exactly 'are these two implementations independent', and
if it contains a mechanical independence measure, C6 narrows again."* R12 ran
that search. It does, and it did.

**Nogueira, Pattabiraman, Vieira & Campos, arXiv:2607.02808 (2 July 2026)**
measure structural diversity with **CodeBLEU** — lexical n-grams (0.1), weighted
n-grams (0.1), AST similarity (0.4), dataflow similarity (0.4) — across 224
problems, 12 models, 5 languages. Structural diversity is moderate; the
implementations nonetheless fail on the same tests far more often than
independence predicts, three- and five-version ensembles realise only 0.43 and
0.44 of the achievable reliability gain, and manual fault analysis finds that
*even different failure patterns often share root causes*. Knight & Leveson
established the original negative in 1986, attributed to shared interpretation
of the specification.

Separately, **Type-4 (semantic) clone detection** is a mature field defined
around fragments *"functionally similar without being textually similar"* —
explicitly, code *"structurally different enough that a model clone detector may
not find them to be within its structural similarity threshold."* That is C6's
failure mode, named and studied by another community.

**What this does to C6, precisely.** C6's measure consults strictly less
information than CodeBLEU: canonicalised syntax and nothing else, omitting dataflow
and lexical n-grams — 60 % of CodeBLEU by weight. (Less information, not a
sub-computation: the two syntactic measures differ and neither contains the other.) The richer
measure was tested against behavioural failure independence at a scale we cannot
approach and does not predict it; a poorer measure cannot do better on the same
question. C6 therefore survives **one-sided only**: high structural similarity
reliably indicates *shared* evidence (that is Type-2 clone detection, and it
works — the 343 pairs are real), while low structural similarity does **not**
license the inference to independent evidence. The paper must not make that
second inference, and R12's gate guards the concession.

**What this does to the roadmap.** The obvious next contribution — a
`corroborator` claim field where the compiler refuses codegen unless a second
derivation measures as independent — is **withdrawn**, not deferred. Its premise
is refuted, and it was separately pre-registered at 0/3 on the historical
replay.

## 9. Threats

Single repository, single author. `n = 6` classifiable pairs in the
retrospective. Message-matched population with unknown recall. Arm A executed
only where a harness runs standalone. **Independence measured structurally,
which §8.6 establishes is a one-sided test and not a lower bound on evidential
independence** — it detects shared evidence, it does not certify independent
evidence. Predictive kernels audited at levels 4–5 only. The line's own
artefacts excluded from its own measurements, and the self-referential share of
coverage disclosed.

## 10. Conclusion

The compile-time mechanism is buildable and was built. It guards drift, and the
failure that actually damaged this corpus was claim and check being wrong
together — which no compiler reaches. What *does* reach it is not in the
compiler at all: asking **where the evidence came from**, and refusing to count
agreement between checks that share a derivation. That question is cheap to
answer, decidable, and on this corpus it changed the status of a third of the
cross-checks.

C3's own limit is now measured, and then closed in the compiler twice — in both
cases by work indexed in Appendix A rather than argued here. R17: a build whose
check exits 0 and reports exactly the declared proposition is refused when its
grounds have been replaced. R20: it is refused again when the derivation the
claim cites is not in the tree — a failure class none of the earlier checks
can reach, because all of them read what a gate emits and none reads what a claim
depends on. The limit is not the impossibility of §5 but a
**resolution** limit: a token is exactly as fine as the proposition it states,
and propositions about counts are coarse. Sharpened in C16 — the token is blind
to every map acting *within* the classification it counts, and the stronger a
classification theorem is, the coarser its verdict token, because the theorem
states a cardinality while its content is a labelling. The repair — bind a *witness* of the
proposition rather than its truth value — is a strict strengthening of C3 and is
verified rather than proposed (§C15).

Four literature searches narrowed this paper's claims four times, and each
narrowing improved it. The compile-time mechanism went first, then the
independence technique, then the epistemic framing, then — in §8.6 — the
inference the independence measure licenses. What is left is small and survives
contact: **a study-level distinction made into a machine-checkable property of
one codebase's internal evidence, used to find where auditing pays, and read in
one direction only.** Reporting the narrowings rather than the first draft is
the same discipline the paper is about.

What the fourth narrowing leaves open is worth naming, because the arc answered
it once by hand and cannot answer it by machine. The audits that actually caught
things (R7, R9) did not use structural distance: R9's ground truth was
rank-deficiency of the left-multiplication matrix, *a route the corpus's own
predicates never take*. That is independence of the **derivational route**, not
of the **code**. It is what worked here; it is what the N-version literature
still lacks a measure for; and this line cannot compute it either.

R13 turns the fourth narrowing from a transfer into a measurement, and finds one
thing the code measure can be replaced by in one direction. Perturbing the
**shared object** instead of the code — flip the Cayley–Dickson sign on a
targeted base pair — partitions the corpus by *evidential fate*, and that
partition crosses the derivation boundary that structural distance draws.
Co-sensitivity is a **negative test**: it can refuse a corroborator that shares
fate with the claim, which is more than C6 could honestly do after §8.6. It
cannot certify one, because insensitivity to a finite battery is not evidence of
an independent route. A compile-time obligation built on it could say **no**;
nothing in this paper lets it say **yes**.

---


## Appendix A — the rest of the line: bound to gates, not argued here

The rungs below are results of the same line, each with an executable gate and a
declared verdict token that this paper's contract checks on every run. They are
**not** claimed as contributions of this paper: their arguments live in their
rung specs, and a contribution table is a promise about the body. They are listed
because a rung dropped from the index is a result quietly disappeared — including
the three (R22, R23, R25) that are findings about this project's own governance
metadata, which are engineering observations rather than research results and are
marked as such here rather than promoted by placement.

| Result | Rung | Verdict token |
|---|---|---|
| **Witness binding.** A claim may declare a `witness`; the compiler refuses codegen when the gate exits 0 **and** emits exactly the declared token while the evidence establishing it has been replaced — the repair C15 identified. | R17 | `WITNESS_BINDING_IMPLEMENTED__REFUSES_ON_PRESERVED_PROPOSITION` |
| **Half of C16 derived, two explanations killed.** C16's "two edges per fiber" follows from index arithmetic and the exceptional fiber is predicted rather than observed; two candidate accounts of the remaining half were tested and both failed, leaving one equivariance lemma stated precisely enough to be attacked. | R19 | `LOCALITY_DERIVED__EQUIVARIANCE_REDUCED_TO_ONE_LEMMA` |
| **Provenance binding.** A claim may declare `provenance = "<path>"`; the compiler refuses codegen when that **path is absent from the tree**, emitting `CLAIM_PROVENANCE_MISSING`. Audit behind it: 2 155 cited artifacts, 93 absent, 8 on an unmerged branch. | R20 | `PROVENANCE_BINDING_IMPLEMENTED__CITED_DERIVATION_MUST_EXIST` |
| **C16's inference proved, not merely made plausible.** Both relations generating the classification are F₂-linear and fix `h`, so each carries the added edge to the added edge and the partition is preserved; the proof is stated in the rung and completes the lemma R19 isolated. It could not be written until R20 restored the derivation it depends on. | R21 | `EQUIVARIANCE_PROVED__R16_INFERENCE_IS_A_THEOREM` |
| **A green gate that certifies a constant.** `last_validated` is the string `2026-03-07`, hardcoded at two sites in the registry generator and declared identically by **1 063 of 1 063** governed documents — a date fixed in the generator, not read from any validation event, and earlier than this repository's first commit. CI enforces the constant, so a document recording a real validation date fails the gate. | R22 | `VALIDATION_DATE_IS_A_LITERAL__GATE_REJECTS_THE_TRUE_DATE` |
| **The sibling field.** `validated_by` is filled from the topic's owner, and every path under `docs/research/` carries the literal `A6` (309/309). The checker enforces equality, so a document naming its real validator fails the gate: the field answers a directory question under a validation name. | R23 | `VALIDATED_BY_IS_PATH_OWNERSHIP__GATE_REJECTS_TRUE_VALIDATOR` |
| **A binding refused where it would be a rubber stamp.** Of 16 production claims exactly **one** has a derivation its gate does not itself run; for the other 15 a `provenance` field would name a file the gate already fails on if absent. Bound where it carries information and nowhere else, both directions gate-enforced. | R24 | `PROVENANCE_BOUND_WHERE_HONEST__REST_WOULD_BE_HOLLOW` |
| **The third field in the same header.** `authority` is computed as membership of a **three-literal** whitelist, defaulting everything else to `historical`; **317 of 320** research topics are historical, and claiming to be current without whitelist membership fails the gate. The value records a path property, not an assessment. | R25 | `RESEARCH_AUTHORITY_IS_PATH_DEFAULT_HISTORICAL__GATE_REJECTS_CURRENT` |
| **The dangling dependency closed by reconstruction.** The oracle R20 found never committed anywhere — loaded with no fallback by the orbit theorem's verifier, which therefore could not run in any checkout — is rebuilt from that verifier's own proof and checked by running it: it reproduces the predicted orbit structure at n = 4, 5, 6, 7. Of the 168 GL(3,2) maps all admit a sign completion and 21 preserve the signed table. | R26 | `ORACLE_RECONSTRUCTED__ORBIT_VERIFIER_RUNS_IN_TREE` |
| **The literal-field shape found inside the compiler.** By static census of checked-in source: every production claim declares `verdict = Verdict::Alive`, the executor's only read of that field scans the slice for the substring `"archived"`, and the token `Alive` occurs nowhere in the executor. Aliveness is asserted by every claim and tested by none; **1 of 16** binds anything beyond an exit code. | R27 | `CLAIM_LIVENESS_DEFINED__DECLARED_ALIVE_IS_UNCHECKED__1_OF_16_BOUND` |
| **The question that precedes calibration.** The confidence scalar in `0..1000` with its gate at 950 takes **66 distinct values** over 30.6 M expression tokens — graded, not boolean — while **99.9933 %** of the mass sits at exactly 0 or exactly 1000 and only **891 tokens (0.003 %)** fall strictly between 0 and the gate. Whether 950 is calibrated is a question about a population that barely exists. | R28 | `CONFIDENCE_IS_GRADED_IN_PRINCIPLE__BINARY_IN_PRACTICE` |

## Open before this becomes a draft

1. The systematic related-work pass, especially §8.5.
2. Wire the line's gates into CI so the chain of custody stops being
   aspirational.
3. Decide whether the artefact is the compiler, the corpus, or both.
4. Generalisation beyond one repository: these are case-study results and the
   title should say so.
