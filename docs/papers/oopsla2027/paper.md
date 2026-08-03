<!-- docs:meta
topic_id: repo.docs.papers.oopsla2027.paper
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.papers.oopsla2027.paper
-->

# Where Did the Evidence Come From? Compile-Time Claims, Their Limits, and Measuring Whether a Corpus Corroborates Itself

**Status:** `DRAFT` — full prose over the R5-bound skeleton (`outline.md`); every
empirical claim below cites the rung that measured it, and the skeleton's R5
gate continues to bind the contribution table to the rung specs.
**Date:** 2026-07-28
**Orthography:** EN-UK
**Target venue:** OOPSLA 2027
**Skeleton:** `docs/papers/oopsla2027/outline.md` (verdict-token-bound;
`scripts/ci/self_falsifying_compilation_line_r5_gate.sh`)
**Evidence:** rungs R0–R15 of the self-falsifying compilation line,
`docs/research/self_falsifying_compilation_line*_2026-07-2[6-8].md`

---

## Abstract

In scientific software, the correctness of a program is contingent on premises
that live outside the source: empirical facts about the world that the program's
author believed when they wrote it. We build, inside a self-hosted compiler, a
mechanism that treats such premises as a compile-time obligation: each claim's
check runs after type-checking and before code generation, and the compiler
refuses to emit an artifact whose premises no longer hold. The mechanism is not
novel — build scripts and snapshot testing are its neighbours — so we say what
is: binding the build to the *proposition* a check reports, via a verdict token
the claim declares and the check must emit, rather than to an exit status or a
literal output.

We then measure the mechanism against the corpus it was built to guard: a real
research repository's own history of self-correction, under a predicate fixed
before the study ran. **On what that history lets grade, it would have caught
nothing.** The failures that
actually damaged the corpus were claim and check being wrong *together* —
authored from the same misunderstanding and agreeing with each other perfectly —
and we show — by a scoping argument, not a deep theorem — that no compile-time
procedure whose only evidence is the claim's
own check can detect that. That impossibility is an antecedent, not a wall.
Changing it — requiring evidence the claim's author did not supply, and making
the independence of that evidence machine-checkable — turns out to be cheap and
sharp: **343 of 1 081 pairs of the corpus's research contracts (31.7 %) share a
derivation and are not independent evidence of one another**; and the corpus's
contracts rest
on essentially one function; and a second, structurally unrelated derivation of
that function was already sitting unused in the repository.

The rest of the paper is about what survived contact with the literature and
with the corpus itself. Four targeted related-work searches narrowed the claims
four times; the last narrowing — an N-version-programming study at 224 problems
× 12 models showing that structural independence does not predict failure
independence — withdrew a planned compiler feature and reduced the independence
measure to a one-sided test. We then measured the one-sidedness locally:
21 pairs of contracts respond identically to all 36 targeted perturbations of
their shared mathematical object while their structural similarity is
0.479–0.594, far below the threshold at which the measure calls them
independent. The paper's methodological contributions are the ones that
generalise: behaviour receipts, degenerate-predicate detection, cardinality
pins, and the discipline of reporting the narrowings rather than the first
draft.

---

## 1. Introduction: contributions, each bound to a measured rung

This paper reports a research line, not a single result. The line runs in
*rungs*: each rung is a spec, a machine-checkable harness, and a CI gate; each
spec declares a verdict token; each gate fails if the token the harness emits
disagrees with the one the spec declares. The skeleton of this paper is itself
a rung (R5): its contribution table cites the verdict token of the rung that
measured each claim, and a gate fails if any cited token drifts from its spec.
The chain of custody is **paper → spec → contract → measurement**, and the
first link is guarded by the same discipline the paper studies.

We state the contributions up front, each bound to the rung that measured it,
so that the reader can check any of them against its evidence before reading
the prose.

### Part I — the compile-time obligation, and its limit

| # | Contribution | Rung | Verdict token |
|---|---|---|---|
| C1 | An implementation of claim-gated code generation in a self-hosted compiler. **Not a novel capability** (cf. `build.rs`, §8.1); reported because everything else is measured on it. | R0 | `SUBSTRATE_LIVE__CORPUS_BOUND__HISTORICAL_FAILURES_ARE_INTERPRETIVE` |
| C2 | **What it costs to attach such a guard to a real corpus** (R1), and **a limitation of this mechanism removed** (R29). Verification ran on the main source file only, so a refuted claim in an imported module was never checked; it now walks the transitive import closure, so under `--verify-claims` a premise refuted anywhere in that closure blocks the build. Measured past one hop: a claim refuted **two** imports away blocks (`modules=3`), and a diamond visits its shared leaf **once** (`modules=4, pass=4`). Propagation across dependency edges is **not** novel — a failing `build.rs` already fails its dependents (§8.1) — so what is claimed here is the cost measurement and the repair, not the propagation. | R1, R29 | `BOUND_16__MODULE_CLOSURE_PASSES`; `CLOSURE_WALKED__MODULE_CLOSURE_PASSES` |
| C3 | **Verdict-token binding**: bind the build to the *proposition* a check reports, where prior art binds an exit status or a literal output. | R2 | `TOKEN_BINDING_IMPLEMENTED__CATCHES_DRIFT_NOT_MISINTERPRETATION` |
| C4 | *Drift* vs *shared misinterpretation*, with an argument that the latter is out of reach, and a test of what does reach it. | R3 | `FALSIFIERS_NONVACUOUS_ONLY_FOR_CLOSED_FORM_CLAIMS` |
| C5 | A retrospective under a predicate **fixed before the study ran**: a negative result and a degenerate arm, reported as such. | R4 | `RETROSPECTIVE_RUN__SOME_ARM_FIRED` |

### Part II — changing the antecedent

| # | Contribution | Rung | Verdict token |
|---|---|---|---|
| C6 | **Evidential independence as a static property.** Prior art binds *what* a check reports; nothing asks **where the evidence came from**. Measured: **343/1 081 contract pairs share a derivation**. **Read in one direction only** — see C12 and §8.6. | R6 | `INDEPENDENCE_CHECKABLE__CORROBORATION_BINDS` |
| C7 | Auditing what the sharing exposes: the shared kernel checked against an independent derivation over 5 440 products. | R7 | `SHARED_KERNEL_CORROBORATED` |
| C8 | **The trusted base of a research corpus**: enumerate shared derivations, rank by blast radius, **collapse wrappers into kernels**, audit what is irreducible. 23 clusters → 12 kernels; 51 contracts rest on essentially one function. | R8 | `TRUSTED_BASE_MAPPED__KERNELS_AGREE` |
| C9 | Completing the audit, and **measuring the method's boundary**: 6 kernels corroborated, 2 with no adjudicator, and a principled reason for each. | R9 | `TRUSTED_BASE_PARTIALLY_AUDITABLE` |
| C10 | **Corroboration depth as a corpus metric**, and a procedure that finds latent corroborations from source alone — validated by rediscovering the one a human found by reading. Depth 1 is the corpus's normal state. | R10 | `LATENT_CORROBORATION_FOUND` |
| C11 | The procedure widened as far as isolation permits: **4 behaviour classes, zero new corroborations.** Wrappers are structurally unprobeable under isolation, and by C8's collapse probing them would add nothing. | R11 | `WIDER_PROBE__NO_NEW_PREEXISTING_CORROBORATION` |
| C12 | **The fourth narrowing, and the one that withdraws a planned contribution.** C6's measure is not new and its central assumption is refuted by a study at 224 problems × 12 models; C6 survives one-sided, and the compiler rule it motivated is withdrawn rather than deferred (§8.6). | R12 | `PRIOR_ART_HAS_ARTEFACT_MEASURE__CLAIM_NARROWS_FOURTH` |
| C13 | **The one-sidedness measured here, not transferred.** Perturb the shared object rather than the code: **21 pairs of this corpus's contracts have identical responses to all 36 perturbations while their structural similarity is 0.479–0.594**, below the threshold at which C6 calls them independent evidence. Pairs C6 calls independent agree *more* (0.565) than pairs it calls shared (0.513). | R13 | `STRUCTURAL_INDEPENDENCE_DOES_NOT_IMPLY_INDEPENDENT_FATE` |
| C14 | **The same instrument turned around, and a pre-registered hypothesis losing.** Of everything these contracts compute — to level 10, 1 024 dimensions — how much does the conclusion depend on? **407 verdict changes, 117 crashes, 12 survivors in 536 cells**; levels 9 and 10 are pure verdict changes. Vacuity refuted; the corpus checks what it computes. | R14 | `VACUITY_REFUTED__CORPUS_CHECKS_WHAT_IT_COMPUTES` |
| C15 | **The limit above C3, and its repair.** C3 binds the build to the *proposition* a check reports. That is still blind to anything preserving the proposition's truth: a flip changing **126 of 128 fiber graphs and every spectrum** leaves `#spectra = 24` intact, so the token holds. A token's resolution is bounded by the **invariance group of its proposition**. Repair, verified: **bind the witness, not the predicate**. | R15 | `TOKEN_RESOLUTION_BOUNDED_BY_PROPOSITION_INVARIANCE` |
| C16 | **The invariance group, identified.** The blind spot is not "maps preserving the count" but maps acting **within the blocks** of the classification: the flip preserves the *identical set partition* of fibers into spectrum-classes and relabels every block, changing exactly **2 edges per fiber**, because the perturbed pair's home fiber is the one the check never examines. **Any claim whose check tests only the *number* of equivalence classes has this blind spot by construction** — it is a property of predicates that project away the witness. | R16 | `INVARIANCE_GROUP_IS_PARTITION_PRESERVING_NOT_MERELY_COUNT_PRESERVING` |



The line contains eleven further rungs — R17, R19–R28 — whose results are
bound to gates but not argued in this paper. They are indexed in Appendix A
so that a rung is not quietly dropped; none is claimed as a contribution here.

### Methodological results that generalise

Three by-products of running the line under its own discipline are, we believe,
the contributions most likely to outlive the setting:

- **Behaviour receipts (R2).** The rung's own contract certified the token
  mechanism as implemented — from source text alone — while the compiler built
  from that source segfaulted on every claim, including claims using none of
  the new machinery. Certification is now bound to a receipt of an actual run,
  hashed to the source it attests: edit the source and the receipt goes stale.
  *The tooling committed the error the paper is about*, and the receipt is the
  repair.
- **Degenerate-predicate detection (R4).** The retrospective's one firing arm
  fired *by construction*: for a correction that changes a token, replaying the
  corrected harness against the old state restates the definition of a
  correction. Reporting that, rather than counting it as a catch, is the
  difference between a study and a story.
- **The tool catching its own author (R8).** A draft of the trusted-base spec
  claimed four independent derivations of the corpus's shared kernel; the
  measured independence matrix said three — two were the same derivation at
  similarity 0.929. The error came from inferring independence by eye, which is
  exactly what C6 replaces. R10's first harness then made the same error in the
  other direction, reporting 130 latent corroborations that were pairs of
  *copies*; the honest unit is the behaviour class with more than one
  derivation, of which there is one. Both numbers are kept in the artefacts,
  labelled, because deleting them would hide how easily they arise.

The thesis of the paper is the arc, stated once: **the compile-time obligation
is buildable, guards a failure mode this corpus does not exhibit, and provably
cannot guard the one it does — but the impossibility has an antecedent, and
changing the antecedent is a cheap, mechanical, and genuinely informative thing
to do.** What the antecedent-change licenses is narrower than we first wrote,
and §8.6 and §7.5 are the narrowing.

---

## 2. Motivation

Scientific code differs from the programs type systems, contracts and
refinement types were built for. Those formalisms describe the *program*: its
states, its invariants, its own behaviour. A scientific program's load-bearing
premises are propositions about the *world* — this algebra has these structure
constants, this census has this value, this spectrum is a complete invariant
up to this level — and they are empirical. They can stop being true without
anyone editing the program: the author re-derives, or a colleague refutes, or
the author notices the derivation was wrong and fixes the prose but not the
code that encodes it. When that happens, nothing in the program notices. The
failure is silent by construction, because the program was never connected to
the premise in the first place.

This is not a hypothetical. The repository this paper is measured on contains,
in its own git history, a claim whose headline was wrong for three commits
while every check stayed green — and, measured later at the offending commit,
while the check *executed* and emitted exactly the token the claim declared
(R0 §2). The check was not broken. It was faithfully verifying the proposition
its author believed, and the proposition was mislabelled: the computation was
right and the interpretation was wrong, and both had been written by the same
person in the same sitting.

Three such self-corrections, audited case by case in R0, fell into two
mechanisms:

1. **Shared misinterpretation** (two cases). Claim and check were authored
   together from the same misunderstanding. At the parent commit — the state
   where the claim was false — spec and harness agreed on the wrong token,
   exactly. No CI gate script changed.
2. **Sub-token error** (one case). The corrected fact was a supporting detail
   the verdict token never encoded. The headline stayed true; a load-bearing
   detail underneath it was false. Token `NO_INVARIANT_FILL` held while the
   group it quantified over was misidentified — the abstract S₄ of order 24,
   where the acting group is in fact (ℤ₂)³ ⋊ S₄ of order 192, itself inside
   the full signed-automorphism group 2³:PSL(2,7) of order 1344.

The sample is `n = 3`, hand-picked because they were known corrections — the
frequency question is R4's, and §5 gives its answer. What `n = 3` establishes
is existence and shape: the failure mode is real, it is not rare in this
corpus, and it has a structure that a compiler either can or cannot reach.
The rest of the paper is the working out of that "can or cannot", first as an
impossibility, then as an antecedent to be changed, then — after the
literature had its say — as a one-sided tool with a measured boundary.

---

## 3. The mechanism (C1)

The substrate is a self-hosted compiler (Sounio) with a native claim syntax.
A claim is a source-level declaration that binds a proposition to an external
check: a hypothesis, the path of an executable gate, and — after R2 — an
optional declared verdict token. Compiled with `--verify-claims`, the compiler
executes each claim's gate after type-checking and before code generation; a
falsified claim aborts the compilation and no artifact is emitted.

The execution environment is deliberately paranoid, and two of its properties
are load-bearing later:

- Gates run via `fork`/`execve` with a fixed argv — no shell, no command
  interpolation. A gate is a path and an argument vector, not a string.
- Gate output is captured via `open` + `dup2` to a file, not via a shell
  redirect — the naive way to read a gate's output would trade the
  no-interpolation property away. Each gate has a per-gate timeout, and a hung
  gate is killed, not waited on.

R0 verified the mechanism end to end on a claim-aware build: all seven
mechanism clauses pass, including that a falsified claim aborts compilation
with no ELF emitted on both compiler lanes, and that the timeout really does
kill a hung gate. **The substrate is live.** We do not claim novelty for it
(§8.1); we claim that the measurements below were taken on a real, working
implementation rather than a design.

R0 also measured what the substrate was guarding at the moment the line
opened: **nothing.** The repository contained 9 native claims across 4 files,
every one a test or CI fixture — 0 in production source — against 295 CI gates
and 40 research contracts. Counting generously, 11 of 295 gates (3.7 %) were
named by any claim at all. The empirical surface of the project was essentially
disconnected from the mechanism built to guard it. That gap, not the
mechanism, is what the line set out to close. (The denominators are a moving
count over tracked files — the corpus grew while the line ran: R0 measured 40
research contracts at its commit; the R6–R16 arc measures 47. We give each
figure with the rung that measured it rather than pretending a single
denominator.)

### 3.1 Binding the corpus (C2)

R1 attached real gates to real claims: 15 native claims in a non-test,
non-fixture source, each bound to a real CI gate, each asserting no more than
its gate establishes. Compiled with `--verify-claims`, all 15 gates run before
codegen (`VERIFY_CLAIMS_OK pass=15`, ELF emitted, ~30 s wall-clock); swapping
one bound gate for an always-failing fixture gives `VERIFY_CLAIMS_FALSIFIED`,
non-zero exit, no ELF. Verified, not assumed.

Binding surfaced four constraints, all measured, all load-bearing for what
follows:

1. **A quarter of the sampled gates cannot be bound at all.** Of 20 gates
   probed, 5 exceed even a 45 s budget; the executor's per-gate budget is 30 s.
   Two of the five are worth pausing on: the falsification ledger's own gate
   and the QEC-prediction gate are too slow to be claim gates.
2. **Verification is serial.** One subprocess at a time; the cost of a build
   is the sum of its claims' gates. Binding all ~300 gates at observed rates
   would put a build in the tens of minutes.
3. **Hermeticity is a bindability criterion, not just speed and colour.** One
   of the first 16 gates bound was green and fast, and rewrote a receipt file
   stamped with the current time and git SHA on every run — binding it made
   every compile dirty the working tree and the build non-idempotent. It was
   unbound, and the harness now refuses to let a known-non-hermetic gate back
   in. If running a claim's check can change the tree, "compile, verify,
   compile again" is not guaranteed to converge.
4. **Most specs cannot be token-bound yet.** At R1, of 269 research specs only
   24 (8.9 %) declared a machine-parseable verdict token; the overwhelming
   majority of the corpus needed a convention change before it could be
   guarded at all.

And binding walked into the wall the mechanism spec had already noted as a
caveat, converting it from a caveat into the line's main engineering obstacle:
**claims in imported modules never execute.** A module carrying a claim bound
to an always-failing gate, imported by a main source that calls into it,
compiled cleanly — `VERIFY_CLAIMS_OK pass=1`, ELF emitted, the program ran — and
the imported false claim was never run. A library whose scientific premise had
been refuted passed silently into every dependent build. Binding therefore
meant hoisting claims out of the libraries they describe into a manifest that
CI compiles; that manifest is R1's deliverable *and* the evidence of the
limitation's cost. 15 of 295 gates is 5.1 % **as measured at R1, 2026-07-26**: the corpus was no longer
at zero, and it was not bound. Both figures have since moved — the manifest
carries 16 claims and the tree 423 gate scripts — which is why this rung's token
counts bindings rather than embedding a denominator (§5), and why the draft's
citation of `BOUND_15__…` was a drift its own contract caught.

R29 removed that limitation. Verification collects the module closure before it
verifies anything and walks every node in it, so under `--verify-claims` a premise
refuted anywhere in the closure blocks the build: the probe that compiled clean at
R1 reports `CLAIM_FAIL` on the imported claim, `VERIFY_CLAIMS_FALSIFIED fail=1`,
and emits no ELF.

**Transitivity is measured, not inferred from the one-hop case.** A two-module
probe cannot distinguish a closure walk from a walk over direct imports, so the
claim is tested past one hop. A chain in which the refuted claim sits **two**
imports away — the importer and the middle module both green — reports
`modules=3` and is blocked by the leaf (`CLAIM_FAIL mcl_chain_leaf_claim_false`,
no ELF); a depth-1 walk would have emitted that ELF. A diamond whose two arms
import one shared leaf reports `modules=4, pass=4`, so a module reachable by more
than one path is verified once rather than counted twice.

**What this is not is a novelty claim, and the related-work position is the
opposite of what it may look like.** Carrying a build failure across a dependency
edge is ordinary: a `build.rs` that exits non-zero fails every crate that depends
on it, and Make and Bazel propagate the same way over their graphs (§8.1). This
mechanism simply *lacked* that property until R29 and now has it. What C2 reports
is therefore the cost of binding a real corpus and the removal of a limitation
peculiar to this implementation — not a capability the prior art is missing. The
novelty of the line lives in C3 and C15, where the build is bound to the
*proposition* and to the *witness* rather than to an exit status — and, in work
this paper indexes but does not argue, in R20's binding of the *provenance*
(Appendix A).

The practical consequence is local and worth stating plainly: a claim can live in
the module whose science it describes instead of being hoisted into a manifest to
be seen at all. The manifest is not thereby obsolete — it is the file CI actually
compiles under `--verify-claims`, so it remains the one place that guarantees a
claim runs on every push regardless of whether anything imports it.

**Three limits, all measured.** The widening is **scope, not corpus**, and the
size of that scope today is a census rather than an inference: outside the
purpose-built probe fixtures, claims exist in exactly three files of this
repository — the manifest and two tests — and no library, compiler module or
stdlib module carries one. The manifest itself has no imports. So the walk
changes the set of claims actually reached by zero at the time of writing; what
it changes is where a claim *may* be put, and the bindability criteria that
decide what can be bound at all are untouched.

It is **opt-in**: without `--verify-claims` no claim in any module executes,
verified as its own arm.

And discrimination is tested directly, because "a refuted import blocks" is also
satisfied by a compiler that simply fails whatever it imports. A single
compilation importing one green-claimed module and one red-claimed module
reports `CLAIM_PASS mcl_green_library_claim` and `CLAIM_FAIL
mcl_library_claim_that_is_false` in the same run, blocks, and emits no ELF. The
green claim is not merely tolerated; it is executed and reported as a pass while
its red neighbour is reported as a failure.

Cost on an import-free source, three trials each against the pre-R29 binary:
10 s, 11 s, 10 s before and 11 s, 11 s, 10 s after, with verdicts unchanged — the
two are indistinguishable at this resolution, which is the strongest statement
three runs support.

---

## 4. Verdict-token binding (C3)

Exit-code gating binds a build artifact to a *computation*: a gate exiting 0
says the check ran. It says nothing about whether the check establishes what
the claim asserts. R2 closes the closeable half of that gap.

A claim may now declare `verdict_token`. The compiler captures the gate's
stdout, extracts the token the gate actually emitted, and refuses to emit code
if the two disagree — or if the gate emitted none at all. `MISMATCH` and
`ABSENT` both falsify; absent fails **closed**. Measured on a compiler built
from the exact executor source, with three probes:

| Probe | Gate behaviour | Result |
|---|---|---|
| pass | exits 0, emits the declared token | `CLAIM_PASS`, ELF emitted |
| drift | **exits 0**, emits a *different* token | `CLAIM_TOKEN_MISMATCH`, exit 1, no ELF |
| absent | **exits 0**, emits no token | `CLAIM_TOKEN_ABSENT`, exit 1, no ELF |

Every probe gate exits 0. That is the point: exit-code gating cannot account
for any of these results, and if any probe had exited non-zero it would prove
nothing about token binding. Claims without a declared token behave exactly as
before.

What it catches is *drift*: a check and a claim that have come to disagree —
the computation still succeeds, but the proposition it reports is no longer
the one the claim declares. The class is real and cheap to prevent, and the
mechanism caught a live instance within the line's first day: when R1's
binding landed, R0's own gate re-measured the tree and went red on its own
accord, because R0's declared token (`CORPUS_UNBOUND`) no longer matched the
measured state (`CORPUS_BOUND`).

What it provably does not catch is the subject of the next section.

### 4.1 The behaviour receipt

R2's own tooling committed the error the line studies. The rung's contract
first certified the mechanism as implemented *from source text alone* — field
present, capture present, extraction present — while the compiler built from
that source segfaulted on every claim, including claims using none of the new
machinery. A contract that certifies "implemented" from source text is
checking the computation, not the proposition.

The repair is now part of the mechanism's evidence chain: the contract refuses
to emit `TOKEN_BINDING_IMPLEMENTED` without a **receipt** recording that all
four probe behaviours were actually observed, hashed to the executor source it
attests. Edit the executor and the receipt goes stale. No run, no claim.

Two further hazards are recorded in R2 because each looked fine: runtime
string building in the capture path segfaulted the compiler (the capture path
is a fixed file, not per-process — a known limitation, unresolved, in a
workspace that runs several agents at once); and assigning a token verdict
back into the executor's outcome variable silently did not stick, so for two
builds a drifted token passed while extraction had been correct from the first
build. The bug was always downstream of where the symptom pointed. Neither is
claimed to be understood beyond what was measured.

---

## 5. What it cannot catch (C4, C5)

### 5.1 Two failure modes, one proposition

**Definition (drift).** A claim declares a proposition *p* and names a check
*g*. *Drift* is the state in which *g* still passes but *p* no longer
describes what *g* establishes, because one side changed without the other.

**Definition (shared misinterpretation).** The state in which *g* passes, *p*
is false, and *p* and *g* were authored together from the same
misunderstanding: *g* faithfully computes something, and *p* mislabels what
that something means.

**Proposition (scope limit of self-falsification).** No compile-time procedure
whose only evidence about *p* is the behaviour of the claim's own check can
detect shared misinterpretation.

*Argument.* The compiler observes *g*'s exit status and output. Under shared
misinterpretation *g* runs and reports exactly as it would if *p* were true —
by construction, since *g* was written to check *p* as its author understood
it. The mislabelling is a relation between *p* and the world, not a property
of *g*'s observable behaviour, so no predicate over that behaviour separates
the two cases. Detecting it requires a derivation of *p* **independent of the
claim**, which is by definition not part of the claim. ∎

This is a scoping argument, not a deep theorem — it is the compile-time
analogue of the familiar fact that a test suite encodes its author's
misunderstanding as faithfully as their understanding (the test oracle
problem, in compile-time dress; §8.4). But it is the honest
boundary, and the corpus's own history is what forced it into view.

### 5.2 Executable falsifiers: the partial escape (C4)

R3 tested the obvious escape — an *executable falsifier*: a check that must
**fail** for the claim to live, authored *against* the claim rather than with
it. The rung was narrowed before any code was written: a falsifier by the same
author is just a gate with inverted polarity and inherits the proposition
above unchanged. The answerable question is whether an *independent* falsifier
exists and fires, for the three audited corrections.

One of three did. The E6-bridge claim — that φ is the complement of the E6
cubic — is refuted by a closed-form identity: for imaginary octonions,
`Re(x·y·z) = −φ(x,y,z)`, which puts φ *inside* the invariant it was claimed to
complement. The falsifier is about forty lines, uses nothing from the claim's
own harness, and fires decisively: max `|Re(xyz) + φ(x,y,z)| = 3.55e-15` over
400 random imaginary triples and all 343 imaginary basis triples. The other
two claims are propositions *about a constructed object* whose construction is
itself the contested work; falsifying them costs as much as the claim, and
taking the construction from the claim forfeits independence.

Two honest limits, stated rather than discovered. First, the split is about
**closed form**, not subject matter: independently falsifiable means
expressible as an identity or a finite check over a re-derivable structure —
one of three in this (hand-picked) sample. Second, the falsifiers were written
*knowing the corrections*. Writing the E6 falsifier requires suspecting φ
might sit inside the cubic rather than beside it — which is the insight whose
absence caused the error. The guard is real but **not self-starting**. What
executable falsifiers add is a place to record an independent refutation once
someone has had the idea — turning a correction into a permanent,
machine-checked obstacle to the same error returning.

### 5.3 The retrospective (C5): a negative result under a fixed predicate

R4 is the empirical half of the proposition. The predicate was fixed in R0 §5
**before the study ran**: for each correction commit *c* with parent *c^*, three
arms are evaluated at *c^* — exit-code gating (arm A), token binding (arm B),
and cross-version replay of the corrected harness against the old state
(arm C) — with five buckets and `UNCLASSIFIABLE` never redistributed. R0 also
stated in advance that arms A and B were known-blind and arm C was the open
question.

The scan found 65 (commit, spec) pairs from 51 message-flagged commits; 20 had
a prior claim; 6 of those were classifiable. Results:

| Bucket | Count |
|---|---:|
| `CAUGHT_A` (exit-code gating) | **0** |
| `CAUGHT_B` (token binding) | **0** |
| `CAUGHT_C` (cross-version replay) | **2** |
| `SILENT` | 4 |
| `UNCLASSIFIABLE` — no verdict token declared at *c^* | 14 |
| `NO_PRIOR_CLAIM` — the commit created the spec | 45 |

Arm A is *executed*, not assumed, for the two objective corrections: the
harness as it stood at *c^* exits 0 and emits exactly the token the spec
declared, while the claim was false. Shared misinterpretation, demonstrated by
execution rather than static comparison.

Arm C's two firings **establish nothing**, and reporting that is a
contribution, not an embarrassment. These harnesses are pure computations;
"run against *c^*" is the same as running them now, and they emit the
corrected token because the correction is what they encode. For a correction
that changes the token, arm C fires *by construction* — it restates the
definition of a correction. R0 had called arm C's outcome "genuinely unknown";
it was, and the answer is that the arm is degenerate for this corpus. The
verdict token records the mechanics (`SOME_ARM_FIRED`) and the spec records
what they mean; that the two disagree is the honest state, and the token is
not retro-fitted, because retro-fitting a token to its narrative is the
failure the line studies.

70 % of the pairs where a prior claim existed could not be graded at all,
because the spec declared no verdict token at *c^* — the corpus's history
predates the convention the predicate needs. An honest denominator matters
more than a large one.

Putting Part I together, on this corpus: the mechanism works; it can be
attached to real science, at a price, and not inside libraries; exit-code
gating would have caught 0 of 6 classifiable corrections; token binding would
have caught 0 of 6; most of the history cannot be graded; and the only
remaining instrument that reaches the real failure mode works for the minority
of claims that reduce to a closed form, and is not self-starting. **The line's
premise is buildable and was built; what it guards against is drift; what
actually damaged this corpus was claim and check being wrong together.** That
is a negative result about the idea's usefulness *here*, obtained without
weakening a single definition along the way — and it is the result the line
was set up to be able to reach.

---

## 6. Changing the antecedent (C6–C9)

### 6.1 The move

The proposition of §5.1 has an antecedent: *whose only evidence about p is the
claim's own check*. R0–R5 treated the conclusion as a wall. It is not a wall;
it is an antecedent, and antecedents can be changed. Require a second,
independent derivation of the proposition, and the impossibility no longer
applies — because the compiler's evidence is no longer only the claim's own
check.

That raises the question the prior art does not ask. `build.rs` binds a build
to a check's exit status; snapshot testing binds it to a check's literal
output; R2 bound it to the proposition a check reports. **None of them asks
where the evidence came from.** R6 makes that a static, mechanically decidable
property: two checks are *derivation-disjoint* if, after canonicalising
identifiers, no function body in one matches a function body in the other at
structural similarity ≥ 0.90 (with bodies below 50 AST nodes excluded as
trivial — there is one natural way to negate all-but-one component, and
structural identity there carries no evidence of reuse).

The ceiling was stated before the results, and it matters more after §8.6:
structural independence is a **checkable lower bound, nothing more**. Two
files can share no code and encode the same misunderstanding, because the
misunderstanding lives in the author's head. R3 demonstrated exactly that: its
falsifier shared no code with the harness it refuted (similarity 0.151) and
still fired only because the author already knew the correction.

### 6.2 A third of the corpus is not independent (C6)

Two formalisations were tried. **Import-closure disjointness** — the one a
reader proposes first — is **vacuous on this corpus** and reported rather than
dropped: these contracts are self-contained scripts that import nothing local,
so the check passes universally. In a corpus of self-contained scripts, a
misunderstanding propagates by copy-paste, not by dependency. **Derivation
disjointness** discriminates: on designed pairs it classifies the independent
falsifier at 0.151 and a copy-paste corroborator at 1.000, surviving renaming.

Then the sweep: over all 47 research contracts, **343 of 1 081 pairs (31.7 %)
share a derivation**. And what they share is almost entirely one function:
`cds`, the Cayley–Dickson sign table, copy-pasted verbatim across the
functor-F and CD-tower contracts, plus `cd_sigma` across the ZD-fiber spectral
contracts. Those contracts look like a family of independent results that
corroborate one another. They are not: they share one multiplication table.
If `cds` encodes an error, every result built on it inherits that error
identically, every gate stays green, and cross-checking between them
establishes nothing at all. That is a *structural* version of the shared
misinterpretation R0 measured behaviourally — and unlike R0's, it is
detectable by machine, today, without knowing which claim is wrong.

(The triviality floor of 50 AST nodes was chosen after seeing two functions;
R6 says so, and the load is carried by the 1 081-pair sweep, not the designed
cases. Fitting a threshold to its own answer and re-testing it on the same
answer is the failure this line exists to catch.)

### 6.3 Auditing the exposed kernel (C7)

Independence checking says where to look; R7 looks. `cds` computes the sign of
`e_i·e_j` by iterative descent over bit positions, carrying a running sign.
The harness re-derives the same structure constants by recursive
Cayley–Dickson doubling on split arrays — `(a,b)(c,d) = (ac − d̄b, da + bc̄)` —
in which no sign table appears at all: the signs fall out of the recursion.
The two routes sit at structural similarity 0.151; they are not variants of
one another.

The independence claimed here is of the **derivational route**, argued from
the mathematics rather than licensed by the distance: one route maintains a
running sign over bit positions; the other never materialises a sign table at
all. The 0.151 is reported in the detecting direction only — after §8.6, a
low similarity cannot *certify* independence, and single authorship means a
shared misconception is not ruled out. What gives the comparison its bite is
the adjudication below: the oracle earns standing against axioms before it is
allowed to say anything about `cds`.

Adjudication was decided before the comparison: the oracle is first checked
against properties that hold for Cayley–Dickson algebras regardless of
implementation — squares, anticommutativity, the unit, octonion alternativity
(to 7.11e-15), and the level-3/level-4 zero-divisor boundary. An oracle
failing those would have no standing, and the harness would report
`UNTESTABLE` rather than blame `cds`. The axioms passed. Then the comparison:
**zero disagreements in 5 440 basis products across levels 3–6**, and zero
disagreements among the six copy-pasted instances of `cds` themselves (they
are the same function).

This is the first result in the line where the machinery *prevented* something
rather than measuring a limitation — not by blocking a build, but by saying
which of 47 contracts' worth of code was load-bearing, so that auditing one
function retired the risk in all of them. The cost asymmetry is the point:
R6's sweep plus this audit is minutes of compute; independently re-deriving
the 343 pairs' results would be the entire research programme over again.

### 6.4 The trusted base (C8)

Two data points are not a method. R8 makes it one, in four steps: enumerate
*every* shared derivation; rank by **blast radius** (how many contracts
inherit it); **collapse wrappers into the kernels they call** (`omul`, `mul`,
`o`, `sign_matrix` all multiply by looking up `cds`; auditing them audits
`cds` plus a loop); audit what is irreducible.

The map: **23 shared clusters across 47 contracts, 86 function instances,
collapsing to 12 irreducible kernels and 11 wrappers.** The top kernels are
`cds` (blast radius 17) and a textual variant of it (9); the top wrappers —
`omul` (10), `mul` (6), `g2auto` (4), `o` (3) — all reduce to the same
function. **The corpus's contracts rest on essentially one object: the
Cayley–Dickson sign table**, reached directly in 26 contract-slots and
transitively through wrappers in ~28 more (R8's own accounting, over a
47-contract corpus; the contribution table keeps the skeleton's "51" as
bound — the two count slots differently, and we report the rung's numbers in
the body).

Then the audit, and the line's favourite catch. A first draft of R8's spec
claimed **four** independent derivations of the sign table. The measured
independence matrix says **three**: the two textual variants of `cds` sit at
similarity 0.929 — above the threshold this line uses to define shared
derivation — one derivation wearing two shirts. The three that survive are
`cds` (iterative, 26 contract-slots), `cd_sigma` (recursive on the doubling
structure, 3 contracts, similarity 0.477–0.507 to `cds`), and the harness
oracle (recursive on split arrays, 0.058–0.107). All three agree on all
5 440 fully comparable basis products.

**`cd_sigma` was already in the repository, structurally unrelated to `cds`,
and the two had never been compared.** The corroboration R8 reports was
sitting unused in the corpus. Independence checking did not create the
evidence; it noticed the evidence was there and that nobody had put the two
side by side. That is the cheapest form the method takes: before writing a new
corroborator, check whether the corpus already contains one.

Read with §7.5's hindsight, this finding is also the paper's own cautionary
example: `cd_sigma`'s structural distance from `cds` did **not** protect it
from sharing evidential fate — R13's perturbation battery later placed
`cd_sigma` contracts in kill patterns already occupied by `cds` contracts.
The route difference is real (recursive doubling versus iterative descent);
independence of fate is a separate, stronger property, and this pair does not
have it. That is exactly the one-sidedness §8.6 formalises, encountered here
before it was known.

### 6.5 Completing the audit, and the method's boundary (C9)

R8 audited the kernel that dominates the blast radius and flagged the
boundary in advance: *a shared kernel encoding a choice rather than a theorem
would have no adjudicator*. R9 finishes the audit and turns that sentence into
a measurement. Each kernel is classified **before** being audited —
`PREDICTIVE` (asserts a structural fact checkable against independent ground
truth; these can be wrong), `ALGEBRAIC` (pinned by laws), `MECHANICAL` (a
regrouping with one possible behaviour), `CONVENTION` (encodes a choice; no
adjudicator exists, and saying so is the result). Ground truth is computed by
**rank-deficiency of the left-multiplication matrix** — *x* is a zero divisor
iff `L_x` is singular — a route the corpus's own predicates never take.

Six kernels corroborated, zero divergences, two with no adjudicator. The two
predictive kernels — the ones that *could* have been false — hold exactly:
`expected_labels` matches the brute-force census at levels 4 and 5 (7 and 22
labels), and `missing_diagonal` — a theorem about how a defect born at one
level propagates upward under doubling — matches the rank-computed census for
all 7 level-4 and all 22 level-5 fibers. Of the two remainder: `zd_line` is
`NO_ADJUDICATOR` because auditing it requires reconstructing its helpers, and
the cheapest way to do that is to copy them from the file under audit —
exactly the failure R6 measures, declined rather than faked; and `chk` is a
mapping artefact (nested, not a module-level kernel), recorded as a small
imprecision in R8's map rather than hidden. **9 of 11 extractable kernels
corroborated.** The token says `PARTIALLY` and is not being changed, because
`FULLY_AUDITED` would be false.

The trusted base of this corpus is now checked against independent evidence
everywhere it can be. What remains uncorroborated is one kernel whose audit
would require redoing the research it encodes — a research task, not a
harness.

---

## 7. Evaluation

### 7.1 Corpus binding and its criteria

The corpus is the authors' own research repository — single-author, and
disclosed as such throughout: 47 research contracts
(pure-Python harnesses over numpy), ~295 CI gates, and their git history. The
bindability criteria were discovered by trying, not designed: a per-gate time
budget (30 s; a quarter of a 20-gate sample exceeds it), **hermeticity** (a
gate that rewrites tracked files makes builds non-idempotent — one of the
first 16 bound gates did exactly that and was unbound), and the existence of a
declared verdict token (8.9 % of specs at R1, 9.3 % — 25 of 270 — at R2; the
convention is days old and the history predates it). Module closure bounded
where claims could live — main source files only — until R29 walked it; under
`--verify-claims` claims now execute anywhere in a build's transitive import
closure, measured to depth 2 and across a shared leaf. The criteria above are
unchanged and still bound *which* gates can be bound at all.

### 7.2 The retrospective

§5.3. 65 pairs, 6 classifiable, arms A and B silent everywhere, arm C
degenerate by construction, `UNCLASSIFIABLE` never redistributed. The
populations: the objective token-change population is 2, which bounds how much
the objective route can ever see in this history; the message-matched
population has unknown recall — a correction whose commit message used none of
the flagged words is invisible. The line's own specs are excluded from both
populations (R0's token moved `UNBOUND → BOUND` because R1 did the binding — a
state change, not a correction — and including it would have manufactured a
third "correction").

### 7.3 The independence sweep and the trusted-base audit

§6. The sweep is O(pairs × functions²) and takes minutes. The audit of the
base: 12 kernels mapped, 3 audited in R7/R8 (the sign table, three
derivations), 6 in R9, 2 with no adjudicator. Every figure in §6 is
machine-computed and re-runnable via each rung's gate.

### 7.4 Latent corroboration discovery (C10, C11)

R8 found the `cds`/`cd_sigma` corroboration by hand — evidence the corpus
already owned and had never cashed. R10 automates the search: a pair that is
**structurally independent** (below R6's threshold) yet **behaviourally
identical** (agrees on every probe input) is a latent corroboration. The
procedure works: from nothing but source, it rediscovers the `cds`/`cd_sigma`
corroboration. It also defines the metric nobody computes: **corroboration
depth** — for each behaviour the corpus computes, how many structurally
distinct derivations of it does the corpus contain?

The honest accounting is the interesting part. A first version of the harness
reported **130 latent corroborations** — pairs of *copies*: 24 copies of one
derivation against 7 copies of another produce ~168 "independent pairs" while
representing exactly one corroboration. Same error class as R8's
four-versus-three, caught the same way, by asking what the unit is. The
honest unit is the behaviour class with more than one derivation. There is
one. The pair count is kept in the output, labelled, because deleting it would
hide how easily the inflated number arises.

R10's negative — no *new* latent corroborations — covered one signature
family (2–3 integers in, scalar out), and every function it accepted computed
the same thing: the probeable slice was one function wearing 31 faces. R11
widened the probe to array-, set-, dict- and float-valued kernels: **35
probeable functions, 4 behaviour classes, 1 pre-existing corroboration, 0
new.** The array kernels turned out to be unreachable *in principle*: `omul`
fails in isolation with `NameError: cds`, because it calls its kernel and the
probe never imports the module. The isolation that makes the probe safe and
independent is exactly what withholds dependencies — so the probe can only
reach self-contained functions, which are precisely R8's irreducible kernels.
Wrappers are structurally unprobeable under isolation, and by R8's own
collapse, probing them would add nothing: a wrapper's evidence is its kernel's
evidence plus a loop. The two results agree, from opposite directions.

Together: **this corpus contains exactly one latent corroboration, and it was
already found. Everything else it computes, it computes once.** Depth 1 is not
an anomaly; it is the corpus's normal state, now measured rather than assumed,
across every behaviour class the probe can reach.

R11 also paid for the widening with five separate tooling failures — a census
routine returning a structure large enough to exhaust memory; runaway
allocation inside the callee killing the process before Python could raise
(exit 120, empty log); a probed function *closing fd 1*, so the work finished
and the report died on the way out; a truncated report through a descriptor
the dup did not cover; and a failure that appears only under output buffering
— green by hand with `-u`, red in CI, identical output. All five have one root
and one answer: **never run foreign code in the process that has to report the
result.** Probing now happens in a child process that writes JSON and leaves
via `os._exit(0)`. That is precisely the decision the claim executor reached
in R2, for gates, for the same reason — this rung ignored its own line's
conclusion and patched around four symptoms before arriving back at it.

### 7.5 Measuring the one-sidedness here (C13)

§8.6 narrows C6 to one-sided *by transfer* — a richer measure failed at scale,
so the poorer one cannot do better. R13 measures the question locally, on this
corpus's own internal checks, with a new instrument: perturb the **shared
mathematical object** rather than the code. Flip the Cayley–Dickson sign on a
targeted base pair — a perturbation both derivations (`cds` iterative,
`cd_sigma` recursive) take as the identical conceptual change, since both have
signature `(a, b, bits) → ±1` — and record which contracts' verdicts die.
Co-sensitivity is a proxy for shared evidential fate. (Mutating the *source*
of `cds` instead would be worthless: it can only reach `cds` users and would
re-derive R6's structural partition by construction.)

The battery: 36 graduated perturbations (single base pairs, then all products
involving one basis element, then whole levels, then a catastrophic anchor) at
octonion and sedenion level, plus a baseline and a null-wrap control;
1 254 probe runs across 30 usable contracts; 24 informative mutants (killing
10–90 % of contracts; the pre-registered floor was 8); 6 distinct kill
patterns.

The result: **21 pairs of contracts have byte-for-byte identical responses to
all 36 perturbations while their R6 structural similarity is 0.479–0.594** —
far below the 0.90 threshold at which R6 declares them independent evidence of
one another. All 21 are cross-derivation pairs: a `cd_sigma` contract and a
`cds` contract, structurally distinct, dying on exactly the same
perturbations. And the aggregate direction is the opposite of the one R6's
inference needs: pairs R6 calls *independent* have mean kill-set agreement
**0.565**; pairs it calls *shared*, **0.513**. A measure that predicted shared
evidential fate would show a large positive gap; this one is slightly
negative.

What this shows, stated no more strongly than it is: **evidential fate is
fixed by which proposition you assert about the shared object, not by which
code you wrote to compute it.** R6 measures the code. The one-sided reading
survives contact — structural similarity still reliably detects *shared*
derivation (the 343 pairs are real; copy-paste detection is Type-2 and it
works) — and demonstrably fails in the independent direction, here, on the
population the claim was about. R12 expected this to require a hand-built
corpus of 12 implementation pairs with a single-author confound; the
counterexamples were already in the repository, written for other reasons,
over months.

Co-sensitivity is a **negative test**: it can refuse a corroborator that
shares fate with the claim, which is more than C6 could honestly do after
§8.6. It cannot certify one, because insensitivity to a finite battery is not
evidence of an independent route. A compile-time obligation built on it could
say **no**; nothing in this paper lets it say **yes**.

Two instrument faults are part of the record, because both nearly became
findings. The first battery reported two contracts killed by a perturbation
that is mathematically impossible (a sign flip on an index that does not exist
at that level): the probe wrapper had overridden the contracts' own default
`bits`, silently switching sedenion arithmetic to octonion. The fix forwards
arguments untouched, and a **null-wrap control** — identical machinery,
condition that can never fire — now runs first and fails the rung outright.
And two contracts were first excluded as "no baseline verdict" when they had
in fact hit a timeout under 96-way contention — both `cd_sigma`, the scarce
side of the comparison. Re-run at 6-way concurrency: **a crash is a kill; a
timeout is missing data**, and concurrency is a measurement parameter, not an
implementation detail. The corrected battery *strengthened* the finding
(15 → 21 pairs).

### 7.6 Turning the instrument around: a pre-registered hypothesis losing (C14)

R13 found contracts that die together. R14 asks the dual question: for each
contract, of everything it computes, how much does its stated conclusion
depend on? The pre-registered hypothesis was **vacuity** — several contracts
compute the Cayley–Dickson tower to levels 5–10 (one queries 260 610 distinct
basis products at level 9 alone), and a conclusion resting only on the cheap
low levels would make that expensive high-level work decoration: the claim
broader than its evidence.

**The hypothesis lost.** 31 contracts traced to level 10 (1 024-dimensional
algebra); 536 perturbation cells: **407 verdict changes, 117 crashes, 12
survivors, 0 missing**. Levels 9 and 10 are *pure verdict changes, no
crashes*: the deepest computations in this corpus are load-bearing in the
controlled sense, not merely fragile. The corpus checks what it computes, at
every level — a non-trivial positive fact about the corpus that nobody had
measured.

The measure needs three outcomes, not two: *verdict changed* (clean
load-bearing), *crashed* (the conclusion can no longer be established — a
dependence, but not a check noticing anything), and *missing* (timeout or lost
output — **no information**; scoring it as a kill inflates load-bearing, and
an earlier analysis did exactly that). Four contracts are ALL-CRASH; for them
the measure reports fragility, not conclusion-dependence, and their
contribution is qualified accordingly.

The twelve survivors are not vacuity findings, and the analysis rule that
decides was fixed before seeing the numbers: a level where every flip
survives is either (a) queried but not checked — vacuity — or (b) checking a
quantity genuinely invariant under a single sign flip — mathematics. The one
candidate resolved as (b) from data already on disk: all 10 single-pair flips
survive while all 4 in-range row-flips kill, so the level *is* checked, and
what it checks does not see a single sign. Without the rule, this rung would
have shipped "a vacuous level found in the corpus", which is false. One
survivor remains unexplained and is recorded as a located, reproducible
anomaly with three explanations tested and refuted — not as a finding
(§7.7 resolves it).

### 7.7 The limit above C3, and its repair (C15)

R14's anomaly: flipping the sign of σ(64, 192) at level 8 leaves the
contract's verdict unchanged. R15 resolves it, and the resolution is not
about that product. **The flip changes 126 of 128 fiber graphs and every one
of their spectra. What it preserves is the *number* of distinct spectra — 24
before, 24 after, while the set of 24 is entirely replaced.** The contract's
claim has the form `#distinct spectra = 3·2^(n−5)`, so its check tests a
cardinality, and a cardinality cannot see a transformation that swaps the
things it counts. It is a family, not an accident: the flip σ(H/2, H + H/2)
preserves the count at every level tested (3, 6, 12, 24 at n = 5–8) while
generic flips at the same levels change it (up to 40) — the controls are what
make it a finding rather than a robustness observation. And it explains why
only level 8 was invisible: a flip aimed at level *k* also perturbs every
deeper level computed through it, so a count-preserving flip at 5, 6 or 7
betrays itself higher up; at level 8 — the boundary of the contract's own
claim — there is nowhere higher to look.

This locates the limit **above** C3, and it is structural rather than
incidental:

> **A verdict token's resolution is bounded by the invariance group of the
> proposition it states.** A claim of the form `#X = N` is invariant under
> everything that preserves |X|. Binding the proposition does not bind the
> witness.

It is not the shared-misinterpretation impossibility of §5.1 — the check is
not wrong, and neither is the claim. It is a resolution limit: the token is
exactly as fine as the proposition, and propositions about counts are coarse.
(Not a refutation of the underlying completeness claim either: a perturbed
sign table is not a Cayley–Dickson algebra; what is measured is the reach of
the *check*, not the truth of the *claim*. And not a cospectral
counterexample: of 128 fibers, zero change adjacency without also changing
spectrum — the spectrum tracks the graph faithfully; only the aggregate is
blind.)

The repair is verified rather than proposed: **bind the token to the witness,
not the predicate** — a hash of the sorted set of spectra instead of its
cardinality. Measured at n = 5, 6 (live) and n = 8 (recorded): in every case
|S| = |S′| while S ≠ S′. A witness-bound token changes; a count-bound token
does not. Two lines in a real contract, and the harness verifies the
discrimination rather than asserting it. The general form: **bind a witness
of the proposition, not its truth value** — a strict strengthening of C3.

### 7.8 Why the repair is the right shape: the group, identified (C16)

R15 exhibited one element of the group a token cannot see and left *why* open.
R16 answers it, and the answer enlarges the group. The flip σ(H/2, H + H/2)
does not merely preserve the count: it preserves the **identical set
partition** of fibers into spectrum-classes — same blocks, same sizes
([1,7,7] at n = 5; [1,1,7,7,7,8] at n = 6; twelve blocks at n = 7) — while
replacing every spectrum that labels them. It changes exactly **2 edges per
fiber**, uniformly, and the reason it can do no more is arithmetic, checked
for n = 5…12 rather than sampled: the flipped pair (h, H + h) satisfies
h XOR (H + h) = H, so its home fiber is L = H — the single fiber the check
never examines. The correct characterisation is therefore not "maps
preserving |X|" but:

> A check testing **|partition|** is blind to every map that acts **within
> blocks** — the partition-preserving maps. Count-preservation is a
> consequence, not the mechanism.

This explains why the witness repair works — the witness is precisely the
labelling the flip destroys while the partition survives — and it locates the
general hazard, scoped precisely: **any claim whose check tests only the
*number* of equivalence classes has this blind spot by construction.** It is
a property of predicates that project away the witness, not of classification
theorems as such — a check bound to the witness from the start has no such
blind spot, which is exactly the repair. But the coarser the token, the wider
the blind spot: a verdict token that states a cardinality while the content
is a labelling is blind to every within-block map, and classification claims
are where tokens of that shape most naturally arise. Stated bounds, kept: the
edge-change and partition measurements are at n = 5, 6, 7, and the step from
uniform local change to preserved classification is inferred, not proved —
the equivariance that would establish it is open, as is whether the
partition-preserving maps have more exhibitable elements.

---

## 8. Related work — every novelty claim here was narrowed by searching

Searching this section changed the paper **four times**, and the pattern held
every time: the *technique* was already old, and only a narrower *semantics*
survived. Each search cost minutes and removed a claim that would not have
survived review. We report the narrowings rather than the first draft, because
that is the discipline the paper is about.

### 8.1 The compile-time mechanism is not new

**Cargo build scripts (`build.rs`)** run arbitrary code before the main
compilation; if the script fails, the build fails. "An external process runs
before compilation and can abort it" is standard practice, not a contribution.
**Snapshot / approval testing** (Jest, Vitest, golden masters) binds a
declared expected output to actual output and fails on mismatch, gating the
merge.

**What survives:** `build.rs` binds a check's **exit status**; snapshot testing
binds its **literal output**. Neither binds a **proposition the source
declares** — neither has a place to write *what the check is supposed to
establish* — so neither can detect a check that still succeeds while
establishing something else. C3 is exactly that remainder, and C15 measures
where even that remainder runs out.

### 8.2 The independence technique is not new either

**Code clone detection** is decades old. Type-2 clones are fragments identical
up to identifier and literal names; AST- and token-based detectors (CCFinder,
NiCad, tree-based methods) find them by exactly the normalisation this paper's
harness uses.

**What survives:** the technique is clone detection; the **framing is not**.
Clone detection is posed as a *maintainability* problem — duplication is debt.
Here the same measurement answers an *epistemic* question: **do these checks
corroborate each other, or are they one check restated?** Plus two things a
clone report does not give you: the **wrapper collapse** that turns a clone
list into an irreducible base (C8), and **auditing that base against
independently computed ground truth** (C7, C9).

### 8.2b And the epistemic idea is not new either — third narrowing

The reproducibility literature's standard taxonomy already separates
**rerunning the same artifact** from an **independent re-implementation**, and
already treats the latter as the stronger evidence (Sinha et al., arXiv
2402.07530 — consulted in summary; to be quoted properly before submission).
"An independent derivation corroborates more than a re-run" is established
vocabulary, not a finding here.

**What survives, and it is now the whole of C6:** that taxonomy classifies
*studies*, by human judgement, after the fact. This work makes it a
**mechanically decidable property of a single codebase's own internal
checks** — which of *my* checks are independent of *each other*, computed
without asking anyone, and used to decide where auditing pays.

A tension worth stating, since it cuts against standard guidance.
Reproducibility practice says *share the code so others can re-run it*, which
maximises repeatability. Corroboration wants the opposite: a second check
that shares no derivation with the first. Both are goods; they are not the
same good, and a corpus that maximises one can score zero on the other while
looking healthy. This corpus did: 343 pairs of mutually re-runnable, mutually
non-corroborating checks.

### 8.3 Nearest neighbours in scientific software

**Continuous analysis** (Beaulieu-Jones & Greene, *Nature Biotechnology* 2017)
re-runs an analysis in Docker + CI whenever code or data change — alongside
the build; it does not withhold an artifact, and it binds a pipeline rather
than a stated proposition. **Executable papers** (*Toward Executable
Scientific Publications*, Procedia CS 2011) make the document the artifact;
nothing blocks code generation.

### 8.4 Distinct by definition, argued rather than searched

Design-by-contract, refinement types, `constexpr`/`comptime`/staging,
certified compilation and proof-carrying code all describe or prove properties
**of the program**. The premises here are propositions **about the world**.
Reproducible and hermetic builds are the *opposite* design goal — this
mechanism makes the build depend on the world on purpose.

Two further neighbours deserve explicit differentiation, because a reviewer
will reach for both. **Metamorphic testing** (Chen et al.) attacks the same
oracle problem as §5.1: when no trustworthy oracle exists, test follow-up
inputs against necessary relations over outputs. It evaluates *one program's*
behaviour under related inputs; it does not ask whether two checks derive
from shared evidence, and its metamorphic relations are themselves authored —
so it inherits the same authorship confound this paper measures rather than
assumes. **Workflow and computational provenance** (W3C PROV, VisTrails and
successors) tracks lineage at the granularity of artifacts and executions —
which outputs were produced from which inputs and code versions. The question
here is orthogonal and finer: given two checks whose provenance is disjoint
(no shared artifacts), do they nonetheless share a *derivation*? Provenance
answers "what produced this"; R6's measure answers "is this the same
computation restated". Neither subsumes the other.

### 8.5 Residual risk

The searches were targeted, not exhaustive; a systematic pass remains before
submission. Searching for clone detection applied to research software as a
*validity* question surfaced only code-*sharing*-for-reproducibility work and
duplication-as-debt case studies — nothing measuring whether a corpus's own
checks corroborate one another. That is weak evidence of absence, not evidence
of novelty. Specifically not yet checked, in priority order: (i) the
mutation-testing and test-suite-independence literature, which asks a
structurally similar question about tests rather than scientific checks;
(iii) whether any reproducibility-badging scheme already requires demonstrated
implementation independence rather than accepting a declaration.

### 8.6 (ii) N-version programming — the fourth narrowing, and a withdrawn contribution

This section previously listed N-version programming as *"the most dangerous
to C6 — decades of work on exactly 'are these two implementations
independent', and if it contains a mechanical independence measure, C6 narrows
again."* R12 ran that search. It does, and it did.

**Nogueira, Pattabiraman, Vieira & Campos (arXiv:2607.02808, 2 July 2026)**
measure structural diversity with **CodeBLEU** — lexical n-grams (0.1),
weighted n-grams (0.1), AST similarity (0.4), dataflow similarity (0.4) —
across 224 problems, 12 models, 5 languages. Structural diversity is moderate
(mean CodeBLEU 0.41–0.58 by language); the implementations nonetheless fail
on the same tests far more often than independence predicts; three- and
five-version ensembles realise only **0.43** and **0.44** of the achievable
reliability gain (below 0.3 within a single model); and manual fault analysis
finds that *even different failure patterns often share root causes*. Knight &
Leveson established the original negative in 1986, attributed to shared
interpretation of the specification. Separately, **Type-4 (semantic) clone
detection** is a mature field defined around fragments *functionally similar
without being textually similar* — explicitly, code *structurally different
enough that a model clone detector may not find them within its structural
similarity threshold*. That is C6's failure mode, named and studied by another
community.

**What this does to C6, precisely.** C6's measure consults strictly less
information than CodeBLEU: canonicalised syntax and nothing else, omitting
dataflow and lexical n-grams — 60 % of CodeBLEU by weight. (Less information,
not a sub-computation: the two syntactic measures differ and neither contains
the other. What is nested is the input, not the algorithm — and the argument
only needs the input.) The richer measure was tested against behavioural
failure independence at a scale we cannot approach and does not predict it; a
poorer measure cannot do better on the same question. C6 therefore survives
**one-sided only**: high structural similarity reliably indicates *shared*
evidence — that is Type-2 clone detection, and it works; the 343 pairs are
real — while low structural similarity does **not** license the inference to
independent evidence. This paper makes no such inference; R12's gate guards
the concession, and C13 measures it locally rather than leaving it as a
transfer.

**What this does to the roadmap.** The obvious next contribution — a
`corroborator` claim field where the compiler refuses codegen unless a second
derivation measures as independent — is **withdrawn**, not deferred. Its
premise is refuted, and it was separately pre-registered at 0/3 on the
historical replay: one sub-token error a corroborator cannot reach, two shared
misinterpretations a corroborator would agree with, and R3's falsifier not
self-starting. The transfer direction is also against us, not for us: if
twelve independently-trained models still fail together, a single author's two
derivations certainly do. R12 stopped at its pre-registered Phase-0 stop
condition and did not build the underpowered replication — a well-audited
null honoured at the same value as a discovery.

**What the fourth narrowing leaves open.** The audits that actually caught
things (C7, C9) did not use structural distance: R9's ground truth was
rank-deficiency of the left-multiplication matrix, *a route the corpus's own
predicates never take*. That is independence of the **derivational route**,
not of the **code**. It is what worked here; it is what the N-version
literature still lacks a measure for; and this line cannot compute it either —
R9 needed judgement to pick `L_x`, and reported `NO_ADJUDICATOR` on `zd_line`
rather than fake one. Stated as the honest frontier, not as a claim: **the
independence that matters is of the derivational route; it is what worked
here; and nobody — including this line — can compute it.**

---

## 9. Threats to validity

- **Single repository, single author, single domain.** These are case-study
  results, and the title should be read that way. The corpus is pure-Python
  mathematical contracts over one algebraic structure; nothing here estimates
  rates for scientific software generally.
- **Small n.** 6 classifiable pairs in the retrospective; the zero for arms A
  and B is "every case we could grade came out silent", not a rate estimate.
  The message-matched population has unknown recall; the objective
  token-change population is 2.
- **Arm A executed only where a harness runs standalone.** For the two
  objective corrections it ran and exited 0 (measured); elsewhere it is
  recorded as not-executed with the reason, contributing no evidence either
  way rather than a silent pass.
- **Independence is measured structurally, which §8.6 establishes is a
  one-sided test and not a lower bound on evidential independence** — it
  detects shared evidence; it does not certify independent evidence. Every
  use of it in this paper is in the detecting direction or is explicitly
  flagged.
- **The triviality floor (50 AST nodes) is post-hoc**; the load is carried by
  the full sweep, and the floor is disclosed rather than buried.
- **Predictive kernels audited at levels 4–5 only**; the census is
  exponential. Random sampling bounds the algebraic kernels; it does not
  exhaust them. Levels ≥ 7 of the sign table, and non-basis products
  throughout, are untested.
- **Perturbation batteries are finite.** R13's 6 kill patterns over 30
  contracts is coarse resolution; R14's 8 samples against 64 770 queried pairs
  at level 8 is a thin probe, reported with its denominator attached.
  Insensitivity to a battery is not evidence of independence (§7.5).
- **The line's own artefacts are excluded from its own measurements**, and the
  self-referential share of its coverage figures is disclosed where it occurs
  (§3.1, §7.2). R11's corroboration search counts the line's own oracle
  separately from pre-existing corpus evidence.
- **Concurrency is a measurement parameter** (§7.5): the same battery at
  different worker counts yields different corpora, and results of this shape
  must report the worker count.

---

## 10. Conclusion

The compile-time mechanism is buildable and was built: claims in source, gates
executed after type-check and before codegen, a verdict-token binding that
refuses to emit an artifact whose declared proposition is no longer the one
its check reports. It guards drift. The failure that actually damaged this
corpus was claim and check being wrong *together* — which no compiler reaches,
for the proved reason that a check authored with its claim reports identically
whether or not the claim is true. The retrospective, run under a predicate
fixed before the study, caught nothing, and one of its arms turned out to fire
by construction; both facts are reported as measured.

What *does* reach that failure is not in the compiler at all: asking **where
the evidence came from**, and refusing to count agreement between checks that
share a derivation. That question is cheap to answer, mechanically decidable
in the shared direction, and on this corpus it changed the status of a third
of the cross-checks — 343 of 1 081 pairs — collapsed 51 contracts' shared
foundation to essentially one function, found that function independently
derivable three ways, found one of those derivations already sitting unused in
the repository, and audited the whole irreducible base against independently
computed ground truth wherever an adjudicator exists. The one latent
corroboration the corpus contained has been found; everything else it
computes, it computes once.

C3's own limit is now measured too, and it is not the impossibility of §5 but
a **resolution** limit: a token is exactly as fine as the proposition it
states, and propositions about counts are coarse — a flip replacing all 24
spectra preserves `#spectra = 24`. R16 sharpens the statement: the token is
blind not merely to count-preserving maps but to every map acting within the
blocks of the classification — a property of predicates that project away the
witness, and one that cardinality-shaped classification claims carry by
construction. The repair — bind a
*witness* of the proposition rather than its truth value — is a strict
strengthening of C3 and is verified, not proposed.

Four literature searches narrowed this paper's claims four times, and each
narrowing improved it: the compile-time mechanism first, then the independence
technique, then the epistemic framing, then the inference the independence
measure licenses — the last one withdrawing a planned compiler feature rather
than deferring it. What is left is small and survives contact: **a
study-level distinction made into a machine-checkable property of one
codebase's internal evidence, used to find where auditing pays, and read in
one direction only.** Reporting the narrowings rather than the first draft is
the same discipline the paper is about.

What remains open is the independence that actually caught things: of the
**derivational route**, not of the code. R13's co-sensitivity instrument gives
a compile-time obligation that can say **no** — refuse a corroborator that
shares evidential fate with the claim. Nothing in this paper, and nothing we
found in the literature, lets one say **yes**.

---

## References

1. Knight, J. C., & Leveson, N. G. (1986). An experimental evaluation of the
   assumption of independence in multiversion programming. *IEEE Transactions
   on Software Engineering*, 12(1).
2. Nogueira, Pattabiraman, Vieira & Campos (2026). A Systematic Methodology
   for Evaluating Failure Independence in LLM-Generated Code.
   arXiv:2607.02808.
3. Ren, S., et al. (2020). CodeBLEU: a Method for Automatic Evaluation of Code
   Synthesis. arXiv:2009.10297.
4. Kamiya, T., Kusumoto, S., & Inoue, K. (2002). CCFinder: A multilinguistic
   token-based code clone detection system for large scale source code.
   *IEEE TSE*, 28(7).
5. Cordy, J. R., & Roy, C. K. (2011). The NiCad Clone Detector. *ICPC*.
6. Sinha, et al. (2024). Reproducibility, Replicability, and Repeatability:
   a survey. arXiv:2402.07530. *(Consulted in summary; to be quoted properly
   before submission.)*
7. Beaulieu-Jones, B. K., & Greene, C. S. (2017). Reproducibility of
   computational workflows is automated using continuous analysis. *Nature
   Biotechnology*, 35.
8. *Toward Executable Scientific Publications*. Procedia Computer Science,
   2011.
9. EnsLLM. arXiv:2503.15838. *(AST-based similarity for N-version ensemble
   selection, pairing CodeBLEU with CrossHair; cited via [2].)*
10. Cargo build scripts (`build.rs`), The Cargo Book.
11. Jest / Vitest snapshot testing documentation.
12. Chen, T. Y., et al. (1998). Metamorphic testing: a new approach for
    generating next test cases. HKUST CS technical report; and Chen, T. Y.,
    et al. (2018). Metamorphic testing: a review of challenges and
    opportunities. *ACM Computing Surveys*, 51(1).
13. Moreau, L., et al. (2013). PROV-DM: The PROV data model. W3C
    Recommendation; and VisTrails provenance literature.

---

## Artefact and reproduction

Every numbered claim is reproducible from the repository:

- Rung specs (each with harness, gate, and declared verdict token):
  `docs/research/self_falsifying_compilation_line*_2026-07-2[6-8].md`.
- Rung contracts: `scripts/research/self_falsifying_compilation_line*_contract.py`.
- Rung gates: `scripts/ci/self_falsifying_compilation_line*_gate.sh`.
- The skeleton's paper-binding gate:
  `scripts/ci/self_falsifying_compilation_line_r5_gate.sh` (fails if any token
  cited in `outline.md`'s contribution table disagrees with its spec).
- Recorded batteries (too large to regenerate in CI): `scripts/research/r13/`,
  `scripts/research/r14/`, `scripts/research/r15/`, with producing scripts
  alongside.


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

## AI disclosure

Spec, harnesses, gates, skeleton and this draft were prepared under human
direction (2026-07-26 to 2026-07-28). All empirical figures are
machine-computed and re-runnable via the cited gates; external-review
offloads for this draft are logged in `.claude/llm_offload_log.md`. No
clinical content. GAIDeT-ICMJE 2025.
