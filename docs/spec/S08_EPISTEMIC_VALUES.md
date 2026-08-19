<!-- docs:meta
topic_id: repo.docs.spec.s08-epistemic-values
authority: repo_only
audience: users
last_validated: 2026-08-19
validated_by: claude-1
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.spec.s08-epistemic-values
-->

# §8 — Epistemic Values

Spec-Section: `SOUNIO-SPEC-08`
Frame: `docs/spec/E2E_SPECIFICATION_FRAME.md`

Status: **Hypothesis.** The normative statement below is a founder ruling of
2026-08-19. No conformance test exists yet; the section reaches `Executable`
only when one runs on **both** engines with a negative control
(`SOUNIO-GATING-ENGINE`, `SOUNIO-NO-VERSUS-UNKNOWN`).

## 8.1 Normative

> **`Knowledge<T>` is a type with invariants, not a record with three fields.**

Founder ruling. The distinction is the whole section: a record exposes fields,
and a type exposes **operations that preserve its invariants**. What is written
`k.value` is therefore an *operation on an epistemic value*, not a field access
that happens to be spelled with a dot.

Everything else in this section follows from that sentence rather than being
stipulated beside it.

## 8.2 What is measured today

`origin/main`, 2026-08-19:

    struct Knowledge<T> { value: T, variance: f64, confidence: i64 }

- **Not linear.** Dropping one requires nothing.
- **No provenance field.** `stdlib/epistemic/README.md` draws
  `└── provenance: Provenance`; the struct has no such member.
- `confidence` is an integer **0–1000**, clamped at construction
  (`ep_clamp_conf`, `stdlib/epistemic/knowledge.sio:52`).
- The scale already carries undocumented meaning: `ep_exact` constructs
  `variance: 0.0, confidence: 1000`; `ep_measured` constructs `confidence: 900`.
- `.value` occurs **2,278 times** across `stdlib/` and `examples/`.

So the implementation is a record today, and the ruling is a change of kind, not
a description of the present. That gap is the section's work.

## 8.3 Invariants entailed by the ruling

These follow from 8.1. Each is normative; none is implemented.

1. **An epistemic value is inseparable from its uncertainty.** `value` and
   `variance` are not independently meaningful members. Obtaining one without
   the other is an operation, and the operation is not silent.
2. **Projection is an operation with a rule.** `k.value` yields a value that
   carries the mark of having been separated (`SOUNIO-EPISTEMIC-ERASURE`).
   The mark is inferred and propagates; the programmer never writes it.
3. **Re-attachment requires an act.** Uncertainty is restored only by
   `attest(v, uncertainty:, because:)`, whose floor is a discharged proof
   obligation (`SOUNIO-JUSTIFICATION`). There is no coercion.
4. **Decisions read the invariant, not the number.** `Admissible<T>` requires
   support that has not been degraded without justification
   (`SOUNIO-ADMISSIBILITY`). Deciding is the fifth sink.
5. **`confidence` is bounded and its endpoints mean something.** `0` and `1000`
   are not merely clamps. What they denote is **undefined** — see 8.5.

## 8.4 What the ruling buys

Before it, four registered concepts were four independent decisions, each
separately contestable and separately forgettable. After it they are
**consequences of one definition**:

| concept | becomes |
|---|---|
| `SOUNIO-EPISTEMIC-ERASURE` | the rule of the projection operation |
| `SOUNIO-JUSTIFICATION` | the sole re-entry, with its floor |
| `SOUNIO-PROVENANCE` | a member the invariant requires, not an addition |
| `SOUNIO-ADMISSIBILITY` | a reader of the invariant at the point of action |

A rule can be argued away one at a time. A definition has to be replaced whole.

## 8.5 Undefined — rulings owed

- **The meaning of `confidence = 1000`.** `ep_exact` uses it for an exact value
  with zero variance. Whether it denotes *certainty*, *maximum representable
  confidence*, or *no confidence claim made* is unstated, and the three differ
  where it matters (`SOUNIO-NO-VERSUS-UNKNOWN`).
- **Whether `variance = 0.0` is legitimate.** An exact value has no variance; a
  degraded value reports none. The two currently print identically. Whether the
  type admits a genuine zero, or reserves it, is owed.
- **Where provenance lives.** `SOUNIO-PROVENANCE` rules class-in-the-type and
  instance-as-id; neither exists in the struct.
- **Linearity.** Whether `Knowledge<T>` is affine (dropping is an act) is not
  settled; the erasure ruling addressed projection, not discard.
- **`T`'s obligations.** What a type must satisfy to be carried — whether any
  `T` may be, or only those with defined arithmetic — is unstated.

## 8.6 Conformance

The section is `Executable` when, on **both** engines:

- a programme that constructs, propagates and reads uncertainty through a call
  produces the specified variance, and
- a programme that projects and then reads uncertainty is **refused** with a
  named diagnostic, and
- the negative control shows the refusal firing for the stated reason and not
  from name-ignorance.

It is **not** `Claim-ready` on one engine. The current state is the reason: the
FO matrix gives `ADD3 = 0.000000` on Madaros and passing tests on lean_single
for the same source (#1964).

## Claims Forbidden

- Do not read 8.1 as a description of the implementation. The struct is a
  record today; the ruling is what it must become.
- Do not treat 8.3 as implemented. None of the five invariants is enforced.
- Do not fill 8.5 by inference. Those are rulings owed, and a plausible answer
  written there is the failure this corpus exists to prevent.
- Do not cite the `confidence` endpoints as meaning anything until 8.5 is ruled.
