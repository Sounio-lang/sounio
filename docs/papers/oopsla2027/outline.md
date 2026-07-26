# Self-Falsifying Compilation — paper skeleton (OOPSLA 2027)

**Status:** `SKELETON` — `PAPER_SKELETON_TOKEN_BOUND__RELATED_WORK_PARTIALLY_VERIFIED__CI_UNWIRED`
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
> of the line's six gates is invoked by any CI workflow — they have only ever
> been run by hand. Until they are wired, `spec → contract` is guarded by a
> check nobody executes. `W4_CI_WIRING` measures this and the token carries it
> (`__CI_UNWIRED`), so the claim cannot be quietly assumed; wiring them flips it
> to `__CI_WIRED` and this paragraph must change with it. `ci.yml` was left
> untouched deliberately — another agent has uncommitted edits to it, and
> staging that file would have swept their work into this line's commits.

---

## Working title

*Self-Falsifying Compilation: Binding Empirical Premises to Build Artifacts, and
Measuring What That Cannot Catch*

## Thesis in one paragraph

In scientific software the correctness of a program is contingent on premises
that live **outside the source** — a measured constant, a statistical result, a
numerical experiment. Today those premises are recorded in prose and drift
silently. We build a compiler that treats them as a compile-time obligation: it
executes each claim's check after type-check and **before code generation**, and
refuses to emit an artifact whose premises no longer hold. We then do the thing
such papers usually omit: we ask, against a real scientific codebase's own
history of self-correction, whether the mechanism would have caught anything.
It would not — and the reason is precise, general, and worth more than the
mechanism.

---

## 1. Contributions, each bound to a measured rung

| # | Contribution | Evidence | Verdict token |
|---|---|---|---|
| C1 | A compiler that conditions code generation on the execution of external empirical checks, implemented in a self-hosted language. | R0 | `SUBSTRATE_LIVE__CORPUS_BOUND__HISTORICAL_FAILURES_ARE_INTERPRETIVE` |
| C2 | What it costs to attach such a guard to a real scientific corpus, and the wall it hits: claims in **imported modules never execute**. | R1 | `BOUND_15__MODULE_CLOSURE_BLOCKS` |
| C3 | **Verdict-token binding**: bind the build to the *proposition* the check reports, not merely to its exit status. | R2 | `TOKEN_BINDING_IMPLEMENTED__CATCHES_DRIFT_NOT_MISINTERPRETATION` |
| C4 | A distinction — *drift* vs *shared misinterpretation* — with an argument that the latter is out of reach of any self-falsifying scheme, and an empirical test of what does reach it. | R3 | `FALSIFIERS_NONVACUOUS_ONLY_FOR_CLOSED_FORM_CLAIMS` |
| C5 | A retrospective over the corpus's own correction history under a predicate **fixed before the study ran**, reporting a negative result and a degenerate arm. | R4 | `RETROSPECTIVE_RUN__SOME_ARM_FIRED` |

Two methodological results generalise beyond compilers and are, on reflection,
the least obvious parts:

- **Behaviour receipts (from R2).** The rung's own contract certified the
  mechanism as implemented — field present, capture present, both failure
  outcomes present — while the compiler built from that source segfaulted on
  every claim. A contract that reads source text is checking the computation,
  not the proposition. The fix binds certification to a receipt of an actual
  run, hashed to the source it attests. *The tooling committed the error the
  paper is about.*
- **Degenerate-predicate detection (from R4).** The retrospective's one firing
  arm turned out to fire by construction. Reporting that, rather than counting
  it, is the difference between a study and a story.

---

## 2. Motivation

- Scientific code's premises are empirical and external; type systems, contracts
  and refinement types describe the *program*, not the world.
- Failure is silent: nothing in a build tells you a premise stopped holding.
- Sketch the concrete instance: a repository where a claim's headline was wrong
  for three commits while every check stayed green.

## 3. Mechanism

- Claim syntax; `--verify-claims`; placement after type-check, before codegen.
- Sandbox properties: `fork`/`execve` with fixed argv, no shell interpolation,
  per-gate wall-clock timeout. **Capture via `open`+`dup2`, not a shell
  redirect** — the naive way to read a gate's output would trade the
  no-interpolation property away.
- Failure semantics: falsified claim ⇒ non-zero exit, no ELF, on both lanes.

## 4. Verdict-token binding (C3)

- Exit code binds a computation; a token binds a proposition.
- Semantics: declared token vs emitted token; `MISMATCH` and `ABSENT` both
  falsify; absent fails **closed**.
- Probes all **exit 0**, so exit-code gating cannot account for the result.
- Reach: a minority of the corpus can be token-bound at all — the convention
  must exist before it can be enforced.

## 5. What it cannot catch (C4)

- **Definitions.** *Drift*: check and claim disagree after one changed.
  *Shared misinterpretation*: both authored from the same misunderstanding, in
  agreement, and wrong.
- **Proposition.** No compile-time procedure whose only evidence about a
  proposition is the claim's own check can detect shared misinterpretation: the
  check reports identically in both worlds, by construction.
- **Executable falsifiers** as the escape hatch — importing evidence the claim
  does not contain. Non-vacuous, but only where the proposition reduces to a
  closed form, and **not self-starting**: writing the falsifier needs the
  insight whose absence caused the error.

## 6. Evaluation (C2, C5)

- Corpus binding, bindability criteria discovered by trying: per-gate time
  budget, **hermeticity** (a gate that rewrites tracked files makes builds
  non-idempotent), and the existence of a declared token.
- Module closure: an imported module's false claim is invisible; the guard
  cannot live where the science lives.
- The retrospective: predicate fixed in advance, three arms, five buckets,
  `UNCLASSIFIABLE` never redistributed.
- **The central measurement.** At the commit where each audited claim was
  *false*, the harness — executed, not inspected — exited 0 and emitted exactly
  the token the spec declared. Green, self-consistent, and wrong.

## 7. Related work — **PARTIALLY VERIFIED, do not cite beyond §7.1**

### 7.1 Checked (2026-07-26)

- **Continuous analysis** (Beaulieu-Jones & Greene, *Nature Biotechnology* 2017)
  — Docker + CI to re-run a computational analysis whenever code or data change.
  The closest neighbour found. Distinction: it re-runs an analysis *alongside*
  the build and reports; it does not make the **compiler** withhold an artifact,
  and it binds a pipeline rather than a stated proposition.
- **Executable papers** (*Toward Executable Scientific Publications*, Procedia
  CS 2011) — ship data and code so a reader can re-validate. Distinction: the
  document is the artifact; nothing blocks code generation.

### 7.2 Conjectured, **not yet checked** — a full pass is required before submission

Design-by-contract and runtime assertions · refinement types and static analysis
· `constexpr`/`comptime`/staging · build-system test gating · certified
compilation and proof-carrying code · reproducible and hermetic builds (the
*opposite* design goal — this deliberately makes the build depend on the world).

## 8. Threats

Single repository, single author. `n = 6` classifiable pairs in the
retrospective. Message-matched population with unknown recall. Arm A executed
only where a harness runs standalone. The line's own artefacts excluded from its
own measurements, and the self-referential share of coverage disclosed.

## 9. Conclusion

The mechanism is buildable and was built. It guards drift. The failure that
actually damaged this corpus was claim and check being wrong together, which no
compiler reaches. Stating that precisely — with the predicate fixed beforehand,
the degenerate arm reported as degenerate, and the verdict token left disagreeing
with the narrative rather than retro-fitted — is the contribution.

---

## Open before this becomes a draft

1. §7.2 related-work pass, with verified references.
2. Decide whether the artefact is the compiler, the corpus, or both.
3. Generalisation beyond one repository: the results are a case study and should
   be titled as one.
