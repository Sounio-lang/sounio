<!-- docs:meta
topic_id: repo.docs.audit.epsilon-polarity-fork-2026-08-19
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.epsilon-polarity-fork-2026-08-19
-->

---
title: The two engines hold opposite meanings for ε
status: measured
date: 2026-08-19
last_validated: 2026-08-19
engines: Madaros v0.80.0 (default), lean_single
---

# The two engines hold opposite meanings for ε

`tests/compile-fail/vancomycin_low_conf.sio` states its own purpose in its own
comments:

> The compiler must reject assignment to the prescription slot at compile time
> — a hard patient-safety guarantee that runtime checks cannot provide.

It does not hold on the engine `bin/souc` routes to.

| engine | result on the test |
|---|---|
| `lean_single` | `error[P0003]: Type mismatch — expected confidence level not satisfied: Knowledge ε boundary violation at line 27` |
| **Madaros v0.80.0 (default)** | **`check: OK`, rc=0** |

## This is not a missing check

The first reading — *the ε bound was never ported to Madaros* — is **false**, and
is recorded here because it is the reading a grep for `P0003` produces (`3` hits
in `lean_single.sio`, `0` in the Madaros tree). Madaros has the machinery, under
its own name, and it is named for exactly this case:

    self-hosted/check/epistemic.sio:601
    pub fn epsilon_subsumes_call_boundary(provided_eps: f64, required_eps: f64) -> bool

with a live caller at `epistemic.sio:892`. The violation site in the test is
line 27, `prescribe_vancomycin(risky_dose)` — a **call boundary**. The check
exists, is reached, and returns `true`.

## The cause: ε has two polarities

    self-hosted/check/epistemic.sio:595
    pub fn epsilon_subsumes(eps_a: f64, eps_b: f64) -> bool {
        eps_a <= eps_b
    }
    // Get the more confident (lower) epsilon        <- :615

Madaros reads ε as an **error bound**: *lower is better*, and `provided <=
required` subsumes. `parser/types.sio:873` documents the same polarity —
`Knowledge[T, ε < 0.05, ...]`.

The clinical surface reads ε as **confidence**: *higher is better*.

    fn prescribe_vancomycin(dose: Knowledge[f64, ε >= 0.82]) with IO
    let risky_dose: Knowledge[f64, ε=0.40] = ...      // "40% confidence only"

lean_single agrees with the surface — its own diagnostic says *"expected
**confidence** level not satisfied"*, and it refuses `0.40` against `>= 0.82`.

So under Madaros the check computes `0.40 <= 0.82` → subsumes → **accept**. It
is correct for its own semantics. The under-confident prescription compiles
because the two halves of the language disagree about which direction is safe.

**Neither engine is wrong in isolation. There is no recorded decision for either
to be wrong against.**

## Why nothing caught it

Three gates cover this test:

| gate | pins engine | workflow-reachable |
|---|---|---|
| `scripts/clinical_vanco_tdm_e2e_gate.sh` | `SOUNIO_SOUC_ENGINE=lean_single` | **no** |
| `scripts/epistemic_prescription_chain_e2e_gate.sh` | `lean_single` | **no** |
| `scripts/ousadia_epistemic_method_rx_gate.sh` | `lean_single` | **no** |

Three layers, each of which alone would hide the divergence: the polarity fork
itself; gates that pin to the one engine where the surface's polarity holds; and
gates no workflow reaches. `bin/souc` defaults to Madaros.

## It is a family, not a test — five of eight

**Second correction, 2026-08-20.** The count above read *three of six*, then
*three of six* again after a first widening. Both were low. Searching `ε` with
any comparison operator — not just `ε >=` — finds **eight** ε-bounded
`compile-fail` tests, and **five** of them are accepted by Madaros. The two the
earlier searches missed are the two that matter most:

| test | Madaros | lean_single |
|---|---|---|
| `dissertation_pbpk28_overclaim.sio` | **`check: OK`** | refused |
| `knowledge_nonliteral_eps_ungated.sio` | **`check: OK`** | refused |

The first is the dissertation's own over-claim guard: a rapamycin dosing function
demanding 95% confidence, built from a 65%-confidence hepatic-clearance prior.

The second describes itself as a **soundness** guard against *confidence
laundering* — a `Knowledge` built with a **non-literal** ε (a runtime variable)
cannot prove its confidence at compile time and must not satisfy a
`with Epistemic(N)` gate, *"otherwise a low/unknown confidence could be laundered
as certain by binding ε to a variable."* Under the default compiler it satisfies
it.

Two undercounts in a row on the same question is the reason
`scripts/ci/epsilon_engine_parity_gate.sh` now exists: the number was wrong twice
because nothing was watching it. The gate does not choose the polarity. It names
every divergence on every run and refuses to let the count grow.

## The original measurement — three of six

An earlier revision of this document said *three of three*. That was measured on
a census requiring `Knowledge[... ε ...]` on one line, which under-counts. A
wider search on `ε >=` finds **six** `compile-fail` tests, and three of them are
refused. The claim is corrected here rather than quietly widened.

| test | lean_single | Madaros v0.80.0 |
|---|---|---|
| `vancomycin_low_conf.sio` | `error[P0003]` ε boundary violation | **`check: OK`** |
| `epsilon_bound_violation.sio` | `error[P0003]` | **`check: OK`** |
| `covid_2020_knightian_refusal.sio` | `type error line 10: Knightian uncertainty (ε=⊥) cannot satisfy required confidence` | **`check: OK`** |
| `knightian_mixed.sio` | `type error` | `error[E004]` — refused |
| `med/vancomycin_low_conf_refusal.sio` | `error[P0003]` | refused, but **`module failed to parse`** |
| `med/vancomycin_weak_evidence_refusal.sio` | `error[E001]` | refused, but **`module failed to parse`** |

Only `knightian_mixed.sio` is refused by Madaros on a decision. The two `med/`
tests are refused because the module does not parse — a different thing, but
**not** an undetected one, and an earlier revision of this section said it was.
The harness (`scripts/dev/run_sio_test_suite.sh`, which
`scripts/run_sio_test_suite.sh` is a five-line shim to) does read
`//@ error-pattern:` and requires the diagnostic to contain it. Measured: the
Madaros parse-failure output contains neither declared pattern (`ε` and
`StrongEvidence`), so both tests would **fail** the suite under Madaros. They are
green in CI because CI runs `lean_single`, where they refuse with the declared
diagnostics. The correction is recorded because the retracted claim — that a
harness cannot tell a parse failure from a refusal — was more damning than the
truth and was not measured before it was written.

The covid case is worth separating: `epsilon_subsumes_call_boundary` opens with
`if !epsilon_is_valid(provided_eps) { return false }` — an explicit fail-closed
on an invalid ε, which `⊥` should be. It still accepts, so for that test the ⊥
does not reach the check at all. Same outcome, second mechanism.

## The corpus is on the other polarity, 15 to 4

ε-bounded `Knowledge[...]` annotations in versioned `.sio` outside
`archive/`/`bootstrap/`: **12 files**, with operators

| operator | count | polarity |
|---|---:|---|
| `>=` | 15 | confidence — higher is better |
| `=` | 10 | (bare, no direction) |
| `<` | 4 | error — lower is better |

The confidence polarity is the majority reading, and it is the one the compiler
does **not** implement. Its users include `stdlib/darwin_pbpk/core/tissue_composition.sio`,
`stdlib/pbpk/regulatory.sio`, `stdlib/epistemic/klibanoff.sio`, and
`stdlib/epistemic/graded_effects.sio` — the dissertation's own stdlib and the
regulatory surface, not only tests.

## Dissertation exposure: latent, not active

`stdlib/darwin_pbpk/epistemic_pbpk28.sio` writes `epsilon: c[i]` where `c` holds
**confidence** (`0.65`), i.e. the surface polarity. `stdlib/darwin_pbpk/`
contains **zero** `.epsilon` reads, and no ε-bounded call boundary, so no
inverted comparison is currently performed on those values. The exposure is
latent. See `EPSILON_WRITE_READ_MISMATCH_2026-08-19.md` (#2028) for the separate
positional-literal defect on the same field.

## Corpus

Bare `Knowledge {` literals in versioned `.sio` outside `archive/`/`bootstrap/`:
**120 files**. Literals whose first two named fields are parseable: 11, of which
6 are `value, variance` and **2 are `value, epsilon`** — both tests
(`tests/compile-fail/vancomycin_low_conf.sio`,
`tests/run-pass/epsilon_comparison_valid.sio`). No `stdlib/` literal deviates
from the canonical order.

## The cost of each decision, measured

**Option A — ε is confidence (amend Madaros).** Every site lives in **one file**,
`self-hosted/check/epistemic.sio`: the comparison `epsilon_subsumes` (:595), its
three call sites (:611, :648, :668), the boundary caller (:892), the lattice pair
`epsilon_meet`/`epsilon_join` (:615, :624) whose direction flips with it, and
three comments (:30, :588, :614). **~8 sites, one file.** lean_single already
implements this reading and does not change. No corpus file changes.

**Option B — ε is an error bound (rewrite the corpus).** **51 annotation sites
across 22 files**, including `stdlib/darwin_pbpk/core/tissue_composition.sio`,
`stdlib/pbpk/regulatory.sio`, `stdlib/epistemic/klibanoff.sio`,
`stdlib/epistemic/graded_effects.sio`, `tests/run-pass/dissertation_pbpk28_confidence_gate.sio`,
and `tools/test-framework/src/lib.sio` — the test framework itself. Six
`compile-fail` tests are in the set, and each must have its intent re-derived
rather than mechanically flipped, since a refusal test whose comparison reverses
may stop being a refusal. **lean_single changes too**, so option B is the only
one that costs both engines.

The asymmetry is not close: one file against twenty-two, one engine against two.
This is a measurement of cost, not a recommendation — the cheap direction is not
automatically the correct one, and ε as an error bound is the reading most of the
literature on interval and GUM propagation uses.

## What is owed

A **decision**, not a patch: ε is an error bound or a confidence, and one of the
two engines and one of the two surfaces must be changed to match it. Patching
either engine alone silently re-points the other half of the corpus.

Until that decision exists, the honest statement about the vancomycin guarantee
is: **it holds under `SOUNIO_SOUC_ENGINE=lean_single` and does not hold under the
default compiler.**

## Reproduce

    export SOUNIO_STDLIB_PATH=$(pwd)/stdlib
    ./bin/souc check tests/compile-fail/vancomycin_low_conf.sio            # check: OK, rc=0
    SOUNIO_SOUC_ENGINE=lean_single ./bin/souc check \
        tests/compile-fail/vancomycin_low_conf.sio                        # error[P0003]
