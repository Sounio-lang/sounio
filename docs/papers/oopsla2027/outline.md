# Self-Falsifying Compilation — paper skeleton (OOPSLA 2027)

**Status:** `SKELETON` — `PAPER_SKELETON_TOKEN_BOUND__NOVELTY_NARROWED_BY_SEARCH__CI_UNWIRED`
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
> **That chain is currently aspirational, and the verdict token says so.** None
> of the line's gates is invoked by any CI workflow — they have only ever been
> run by hand. Until they are wired, `spec → contract` is guarded by a check
> nobody executes. `W4_CI_WIRING` measures this and the token carries it
> (`__CI_UNWIRED`), so the claim cannot be quietly assumed; wiring them flips it
> to `__CI_WIRED` and this paragraph must change with it. `ci.yml` was left
> untouched deliberately — another agent has uncommitted edits to it, and
> staging that file would have swept their work into this line's commits.

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
| C2 | What it costs to attach such a guard to a real corpus, and the wall it hits: claims in **imported modules never execute**. | R1 | `BOUND_15__MODULE_CLOSURE_BLOCKS` |
| C3 | **Verdict-token binding**: bind the build to the *proposition* a check reports, where prior art binds an exit status or a literal output. | R2 | `TOKEN_BINDING_IMPLEMENTED__CATCHES_DRIFT_NOT_MISINTERPRETATION` |
| C4 | *Drift* vs *shared misinterpretation*, with an argument that the latter is out of reach, and a test of what does reach it. | R3 | `FALSIFIERS_NONVACUOUS_ONLY_FOR_CLOSED_FORM_CLAIMS` |
| C5 | A retrospective under a predicate **fixed before the study ran**: a negative result and a degenerate arm, reported as such. | R4 | `RETROSPECTIVE_RUN__SOME_ARM_FIRED` |

### Part II — changing the antecedent

| # | Contribution | Rung | Verdict token |
|---|---|---|---|
| C6 | **Evidential independence as a static property.** Prior art binds *what* a check reports; nothing asks **where the evidence came from**. Measured: **343/1081 contract pairs share a derivation**. | R6 | `INDEPENDENCE_CHECKABLE__CORROBORATION_BINDS` |
| C7 | Auditing what the sharing exposes: the shared kernel checked against an independent derivation over 5 440 products. | R7 | `SHARED_KERNEL_CORROBORATED` |
| C8 | **The trusted base of a research corpus**: enumerate shared derivations, rank by blast radius, **collapse wrappers into kernels**, audit what is irreducible. 23 clusters → 12 kernels; 51 contracts rest on essentially one function. | R8 | `TRUSTED_BASE_MAPPED__KERNELS_AGREE` |
| C9 | Completing the audit, and **measuring the method's boundary**: 6 kernels corroborated, 2 with no adjudicator, and a principled reason for each. | R9 | `TRUSTED_BASE_PARTIALLY_AUDITABLE` |

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
non-idempotent), and the existence of a declared token. Module closure. The
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
about tests rather than about scientific checks; (ii) N-version programming,
where independence of implementations is the entire premise and its measurement
has been studied; (iii) whether any reproducibility-badging scheme already
requires demonstrated implementation independence rather than accepting a
declaration. **(ii) is the most dangerous to C6** — N-version programming has
decades of work on exactly "are these two implementations independent", and if
it contains a mechanical independence measure, C6 narrows again.

## 9. Threats

Single repository, single author. `n = 6` classifiable pairs in the
retrospective. Message-matched population with unknown recall. Arm A executed
only where a harness runs standalone. Independence measured structurally, which
is a lower bound. Predictive kernels audited at levels 4–5 only. The line's own
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

Three literature searches narrowed this paper's claims three times, and each
narrowing improved it. The compile-time mechanism went first, then the
independence technique, then the epistemic framing. What is left is small and
survives contact: **a study-level distinction made into a machine-checkable
property of one codebase's internal evidence, and used to find where auditing
pays.** Reporting the narrowings rather than the first draft is the same
discipline the paper is about.

---

## Open before this becomes a draft

1. The systematic related-work pass, especially §8.5.
2. Wire the line's gates into CI so the chain of custody stops being
   aspirational.
3. Decide whether the artefact is the compiler, the corpus, or both.
4. Generalisation beyond one repository: these are case-study results and the
   title should say so.
