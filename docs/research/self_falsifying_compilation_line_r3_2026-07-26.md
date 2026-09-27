<!-- docs:meta
topic_id: repo.docs.research.self-falsifying-compilation-line-r3-2026-07-26
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.self-falsifying-compilation-line-r3-2026-07-26
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Self-falsifying compilation R3 — executable falsifiers: non-vacuous, but only where the claim reduces to a closed form

**Date:** 2026-07-26
**Orthography:** EN-UK
**Status:** `EXECUTABLE` — `FALSIFIERS_NONVACUOUS_ONLY_FOR_CLOSED_FORM_CLAIMS`
**Parents:** `self_falsifying_compilation_line_2026-07-26.md` (R0 §3: the scope limit), `self_falsifying_compilation_line_r2_2026-07-26.md` (R2: token binding closed the drift half)
**Harness:** `scripts/research/self_falsifying_compilation_line_r3_contract.py`
**Gate:** `scripts/ci/self_falsifying_compilation_line_r3_gate.sh`

---

## 0. The question, narrowed on purpose

R0 §5 fixed this rung's verdict as `FALSIFIERS_EXECUTABLE__GUARD_{NONVACUOUS,VACUOUS}`.
That framing is circular and was narrowed before any code was written: a
falsifier that must *fail* for a claim to live is just a gate with inverted
polarity, and if the same author writes claim and falsifier, R0 §3's proposition
applies unchanged. Building the mechanism would demonstrate nothing.

The narrowed question, which is answerable:

> For the three audited self-corrections, can an executable falsifier be
> expressed **independently of the claim's own harness**, and would it have
> refuted the proposition asserted at the parent commit?

*Independently* is load-bearing. A falsifier that imports the claim's machinery
inherits the claim's misunderstanding, which is precisely the failure R0
measured.

**Verdict options, fixed before computing** (encoded in the harness's
`main()`): `FALSIFIERS_NONVACUOUS_GENERALLY` · `FALSIFIERS_NONVACUOUS_ONLY_FOR_CLOSED_FORM_CLAIMS` ·
`FALSIFIERS_VACUOUS`.

---

## 1. Result

> **One of three could be falsified independently. It fired decisively, in about
> forty lines. The other two cannot be falsified without rebuilding the
> machinery whose output is in dispute.**

| Falsifier | Parent proposition | Independently expressible? | Outcome |
|---|---|---|---|
| `F2_E6_BRIDGE` | `PHI_IS_G2_SHADOW_OF_E6_CUBIC` — φ is the *complement* / blind-spot of the E6 cubic | **yes** — closed-form identity, ~40 lines, no group theory | **REFUTES**: `max │Re(xyz) + φ(x,y,z)│ = 3.55e-15` over 400 random imaginary triples and all 343 imaginary basis triples |
| `F1_ORD3_MODULE` | `ORD3_MODULE_IS_2xV3`, framed as a fingerprint of the ord-3 operation | **no** | needs the automorphism action, the Fano-line class structure and the sedenion zero-divisor set before the module can even be formed |
| `F3_GROUP_ID` | the ord-3 symmetry-fill group is `S4`, order 24 | **no** | the disputed fact *is* the group's order, so the falsifier cannot take the group's definition from the claim |

Verdict: `SELF_FALSIFYING_R3_VERDICT FALSIFIERS_NONVACUOUS_ONLY_FOR_CLOSED_FORM_CLAIMS`.

### The one that fired

For imaginary octonions, `Re(x y z) = −⟨x y, z⟩ = −φ(x, y, z)`. If that identity
holds, φ **is** the octonion cross-term of the Albert cubic — it sits *inside*
the invariant and therefore cannot be its complement. The identity holds to
`3.55e-15`. The claim asserted at `2b33d7500` was refutable by a two-line
computation, using nothing from its own harness. `ec579a24c` corrected it, but
by re-derivation and review, not by any check going red.

### The cost objection, tested rather than asserted

Declaring F1 and F3 "too expensive" would be an authored judgment dressed as a
measurement. One component of F3 *is* cheaply and independently computable — the
diagonal sign automorphisms of the octonions, the `2^3` factor the correction
names. Computed here from a re-derived multiplication: **8**, as expected.

That corroborates a component and **does not refute** `order 24` on its own:
without the symmetry-fill group's definition one cannot assert those maps lie in
it. So the cost objection survives its own test — the cheap part is cheap, and
the part that would settle the dispute is not.

---

## 2. What this does and does not establish

**Establishes.** Executable falsifiers are **non-vacuous**: for a claim that
reduces to a closed-form identity, an independent falsifier is cheap to write
and refutes decisively. That is strictly more than R2's token binding could do —
token binding would have caught none of the three, because claim and check
agreed (R0 §2).

**Does not establish — and this is the honest limit, stated rather than
discovered.** These falsifiers were written **knowing the corrections**. The
experiment shows a falsifier *exists* and *fires*; it cannot show one would have
been *written* at the time. Writing F2 requires suspecting that φ might sit
inside the cubic rather than beside it — which is the insight whose absence
caused the error. The guard is real but **not self-starting**.

**The shape of the answer to R0 §3.** Shared misinterpretation stays out of
reach of anything the compiler can check by itself. What executable falsifiers
add is a place to *record* an independent refutation once someone has had the
idea — turning a correction into a permanent, machine-checked obstacle to the
same error returning. That is a smaller claim than "falsifiers solve it", and it
is the one the evidence supports.

---

## 3. Coverage: which claims can carry a falsifier at all

The split is not about subject matter but about **closed form**:

- **Falsifiable independently** — propositions expressible as an identity or a
  finite check over a re-derivable structure (`F2`; the sign-automorphism count).
- **Not** — propositions *about a constructed object* whose construction is
  itself the contested work (`F1`, `F3`). Here the falsifier costs as much as
  the claim, and taking the construction from the claim forfeits independence.

One of three in this sample. The sample is the three corrections R0 audited,
chosen because they were known corrections — **not** a random sample, so the
ratio is not an estimate of how much of the corpus is falsifiable.

---

## 4. What this is NOT

- **Not a mechanism.** No compiler change: R3 tests whether the idea is worth
  building before building it. On this evidence a `falsifier` execution mode
  would be worth having and would apply to a minority of claims.
- **Not a frequency estimate.** `n = 3`, hand-picked. See §3.
- **Not independent of the corrections.** See §2.
- **Not a refutation of R0 §3.** The proposition stands; falsifiers work
  *outside* it, by importing evidence the claim does not contain.

---

## 5. Reproduce

```bash
python3 scripts/research/self_falsifying_compilation_line_r3_contract.py
# expect: E1_EXPRESSIBILITY 1/3, E2_WOULD_HAVE_FIRED 1/1,
#         SELF_FALSIFYING_R3_VERDICT FALSIFIERS_NONVACUOUS_ONLY_FOR_CLOSED_FORM_CLAIMS

bash scripts/ci/self_falsifying_compilation_line_r3_gate.sh
# expect: SELF_FALSIFYING_COMPILATION_LINE_R3_GATE_OK
```

Cayley–Dickson multiplication is re-derived inside the harness rather than
imported from any functor-F contract: independence from the claim's own
machinery is the property under test, so reusing that machinery would void the
experiment.

Pure Python 3 + numpy; deterministic (seeded).

---

## 6. AI disclosure

Harness, gate and spec drafted under human direction (2026-07-26). The E6
identity and the sign-automorphism count are machine-computed and re-runnable;
the expressibility judgments for `F1` and `F3` are argued, with `F3`'s partially
tested (§1). No clinical content. GAIDeT-ICMJE 2025.
