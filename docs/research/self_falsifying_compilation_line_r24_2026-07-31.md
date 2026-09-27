<!-- docs:meta
topic_id: repo.docs.research.self-falsifying-compilation-line-r24-2026-07-31
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.self-falsifying-compilation-line-r24-2026-07-31
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Self-falsifying compilation R24 — provenance bound where it is honest; the rest would be hollow

**Date:** 2026-07-31
**Orthography:** EN-UK
**Status:** `EXECUTABLE` — `PROVENANCE_BOUND_WHERE_HONEST__REST_WOULD_BE_HOLLOW`
**Parents:** `self_falsifying_compilation_line_r20_2026-07-28.md` (the provenance mechanism), the R23 commit (`0e3c60f10`, the first bound production claim)
**Harness:** `scripts/research/self_falsifying_compilation_line_r24_contract.py`
**Gate:** `scripts/ci/self_falsifying_compilation_line_r24_gate.sh`

---

## 1. Result

The instruction was: bind provenance to the other production claims. Carried
out literally it would have shipped fifteen hollow checks. Carried out honestly
it binds none of them, and that refusal is the measured finding.

> **Of the 16 production claims, exactly one has a derivation its gate does not
> itself run — `zd_fiber_spectra_count_law_holds`, whose completeness rests on
> the parity-collapse map Φ. It is bound (R23). The other 15 are self-contained
> (10) or infra (5): their gates run the whole derivation, so a `provenance`
> field would name a file the gate already fails on if it is missing. Declaring
> it would duplicate gate existence — the rubber-stamp this line exists to
> refuse.**

Verdict: `SELF_FALSIFYING_R24_VERDICT PROVENANCE_BOUND_WHERE_HONEST__REST_WOULD_BE_HOLLOW`.

| class | count | provenance is… |
|---|---:|---|
| external-derivation | **1** | meaningful — the derivation can go missing while the gate passes (R20's failure class) |
| self-contained | 10 | redundant with gate existence |
| infra | 5 | absent — the gate checks a compiler/build invariant, not a math derivation |

## 2. Why "bind them all" is the wrong reading

A `provenance` field earns its keep only when it names a derivation the gate
does **not** run — so that the artifact can vanish while the check stays green.
That is precisely the situation R20 found and R23 closed: the ZD-fiber witness
gate computes a spectrum count, but the *meaning* of that count as a statement
about geometries rests on Φ, which the gate never invokes. Φ was absent from
this branch while the gate passed; provenance now refuses the build if it goes
missing again.

For a self-contained claim the gate runs the claim's whole derivation. Delete
that script and the gate errors — `CLAIM_FAIL` — before provenance is ever
consulted. So `provenance = "<that script>"` adds nothing a green gate did not
already assert. A green check that asserts nothing is the exact pathology this
line catalogues: R1 a claim bound to no gate, R5 a gate nobody ran, R20 an
oracle never committed, R22 a gate certifying a literal. Adding a sixteenth
would be authoring the disease the line studies.

## 3. What this is NOT

- **Not a claim that the other 15 lack derivations.** They have derivations;
  their gates *run* them. The point is narrower: those derivations are not
  separately citable, because they are not separable from the check.
- **Not a refusal to bind more later.** If a production claim is added whose
  derivation lives outside its gate, `U2` fails until provenance is declared —
  the gate enforces "bound iff bindable" in both directions.
- **Not automatic classification of intent.** `zd_fiber`'s external derivation
  is named (Φ), because that is R23's human judgement; the contract checks the
  judgement held, it does not re-derive which artifact "matters" heuristically.
- **Not a compiler change.**

## 4. Reproduce

```bash
python3 scripts/research/self_falsifying_compilation_line_r24_contract.py
bash scripts/ci/self_falsifying_compilation_line_r24_gate.sh
```

The contract reads the manifest and the gate scripts; it runs none of them. A
gate is "external-derivation" when a script it executes hard-depends
(`exec_module`/`import`) on a repo artifact the gate does not itself run.

## 5. AI disclosure

Classifier, contract, gate and spec drafted under human direction (2026-07-31).
The one external-derivation target is named from R23's judgement and checked,
not inferred. The decision to bind zero of the remaining fifteen is recorded as
the finding, against a literal reading of the instruction. No clinical content.
GAIDeT-ICMJE 2025.
