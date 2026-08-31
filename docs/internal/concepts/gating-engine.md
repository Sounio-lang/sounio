<!-- docs:meta
topic_id: repo.docs.internal.concepts.gating-engine
authority: repo_only
audience: users
last_validated: 2026-08-19
validated_by: claude-1
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.internal.concepts.gating-engine
-->

# Gating Engine

Concept-ID: `SOUNIO-GATING-ENGINE`

Status: **Hypothesis** — founder ruling of 2026-08-19. **It may not be
implemented before the divergence count exists** (in flight); the reason is in
*Sequencing* below.

## Founder Intent

> **Madaros gates the repository. Red is the truth.**

The canonical compiler decides whether the tree is green. A pass on any other
engine is information, never a gate.

## What forced the ruling

Measured 2026-08-19 (PR #1964): the CI **Full Test Suite runs `souc-stage2`,
which is `lean_single`** — the bootstrap seed. `CLAUDE.md` states Madaros is
canonical and that `bin/souc` routes to it. Both could not be true at once.

The consequence was not theoretical. Twenty-one known-failure tags "passed" and
were read as the first-order-variance blocker having fallen. They were all
lean_single results. On a Madaros built from source (Slurm job 10338, r770,
226s, ELF 100553240 B, from `cdea9d7eef`):

    ADD3 = 0.000000     ADD4 = 0.000000     IMP_ADD3 = 0
    IMP_ADD2 = 5        <- the only branch #1939 closed

The arity branch of the defect is fully open on the canonical compiler, and the
suite could not say so, because the suite was not asking it.

`bin/souc` compounds this: it resolves to Madaros **if the modular ELF exists**
and otherwise falls back to lean_single with a stderr notice. The engine a job
exercises therefore depends on an artefact existing in the runner. The wrapper
already carries `_reject_science_without_madaros` — someone guarded the science
surfaces — and the test suite goes around that guard.

## The form the ruling takes

Gating on Madaros outright would stop the merge queue by an unknown amount: the
PBPK suite alone is known to go from 52/53 on lean_single to **19 failures of 53**
on Madaros. The ruling is therefore implemented in the shape the founder already
chose for `Reserved`:

**Madaros gates, with a declared divergence list.**

- The gate runs Madaros. Its verdict is the repository's verdict.
- Every test that fails on Madaros and passes on lean_single is entered in a
  list with an **owner** and a **reason** — the same three-field discipline as
  `reserved-owed`, minus nothing.
- The list is **printed at every run**, oldest first, whether or not the gate
  passes. It is the debt, and it is counted.
- The list may only shrink. **No new entry without a signature**: a test that
  starts diverging must be entered deliberately, never absorbed.
- A malformed entry — missing owner or reason — is **redder than an absent
  one**, for the reason stated throughout this corpus: if declaring badly were
  cheaper than declaring well, everyone would declare badly.

This is not a softening of the ruling. Under it, green means *the canonical
compiler passes*, and what does not pass is **named, counted and owned** —
where today it is invisible behind the wrong engine.

## Sequencing — this may not land yet

The divergence count does not exist. A measurement is in flight: the same corpus
run twice, once on lean_single and once on a from-source Madaros, reporting the
difference set in both directions and classifying the Madaros failures (real
defect / unimplemented feature / harness difference / test assuming lean
behaviour).

**188 divergences and 1,800 are different decisions.** Landing the gate before
the number is known would be the exact error this corpus was written to prevent:
acting on a magnitude nobody measured.

## Required Invariants

- A green must name the engine that produced it. An unqualified pass is a claim
  about `(test, engine, binary vintage)` reported as a claim about the test.
- The divergence list is a debt, not an exemption. An entry means *known and
  owned*, never *acceptable*.
- The inverse set is also recorded: tests failing on lean_single and passing on
  Madaros are known-failure tags held for the wrong reason, and they are
  findings too.
- The fallback must be loud where it matters. A job that intends Madaros and
  silently receives the seed reports nothing about the canonical compiler; the
  stderr notice is not sufficient if nothing reads it.

## Claims Forbidden

- Do not describe this as implemented. CI runs lean_single today.
- Do not read "red is the truth" as licence to disable tests. The divergence
  list requires an owner and a reason per entry; silencing is the failure mode
  it exists to prevent.
- Do not treat a Madaros failure as automatically a compiler defect. The
  classification decides that, and it is not done.
- Do not quote the PBPK 19/53 as the expected scale. It is one suite, measured
  earlier, and is the only divergence figure that exists — not an estimate of
  the whole.

## Related

- `SOUNIO-NO-VERSUS-UNKNOWN` — an unqualified green is ignorance wearing
  knowledge's face
- `MATURITY_LADDER` — `reserved-owed` supplies the owner/date/blocked-on shape
- `SOUNIO-EFFORT-LOCATION` — why the debt is a printed list and not a note
