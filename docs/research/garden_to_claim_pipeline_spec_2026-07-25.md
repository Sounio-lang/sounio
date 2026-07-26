<!-- docs:meta
topic_id: repo.docs.research.garden-to-claim-pipeline-spec-2026-07-25
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.garden-to-claim-pipeline-spec-2026-07-25
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Garden-to-Claim pipeline — driving a Garden seed to a ledger-encoded claim

**Date:** 2026-07-25  
**Status:** `EXECUTABLE`  
**Seed:** `docs/internal/garden/seeds/2026-07-11-the-zero-of-encounter.md`  
**Harness:** `scripts/research/garden_to_claim_pipeline_contract.py`  
**Gate:** `scripts/ci/garden_to_claim_gate.sh`  
**Ledger artifact:** `stdlib/epistemic/zero_encounter_pipeline_claim.sio`

---

## 1. What this is

The Garden README defines four evidence labels — `Garden`, `Hypothesis`,
`Executable`, `Claim-ready` — but until now no seed had a *mechanized* path
between them. This document specifies the **Garden-to-Claim pipeline**: a
repeatable procedure that takes one Garden seed and drives it through all four
layers, ending in a bounded, falsifier-backed claim encoded in the
Falsification Ledger (`docs/research/falsification_ledger_spec_2026-07-25.md`).

The pipeline is instantiated once, on the **Zero of Encounter** seed, whose
executable witnesses (`stdlib/epistemic/zero_event.sio`,
`tests/known_failures/zero_provenance_native_v2_probe.sio`) already exist. The
pipeline adds the missing rungs: a contract that verifies the stage evidence,
a CI gate that composes the witness gates with that contract, and a
ledger-encoded claim whose evidence level cannot overstate what the seed
declares.

---

## 2. Stage mapping

| Stage | Garden label | Artifact required | Zero-of-Encounter instance |
|---|---|---|---|
| S1 | `Garden` | Seed file with butterfly, core idea, boundaries | `docs/internal/garden/seeds/2026-07-11-the-zero-of-encounter.md` |
| S2 | `Hypothesis` | Evidence State table naming a precise, testable direction | Typed zero-provenance taxonomy distinguishes absence from relational, numerical, metrological, and policy zeros |
| S3 | `Executable` | Repo command/test/gate backing the hypothesis | `stdlib/epistemic/zero_event.sio` + `tests/known_failures/zero_provenance_native_v2_probe.sio` + witness gates |
| S4 | `Claim-ready` | Ledger-encoded claim with falsifier, harness, gate, and an evidence level bounded by the seed's own declaration | `garden_zero_encounter_pipeline` at evidence `gate_green` |

Promotion S3 → S4 is the novel rung: the claim may only assert what the
executable layer demonstrates, and the contract (clause **P5**) enforces that
ceiling mechanically.

---

## 3. The encoded claim

```text
claim:     garden_zero_encounter_pipeline
hypothesis: five computations sharing a zero f64 surface value retain
            distinguishable provenance receipts across absent, cancelled,
            annihilated, below-resolution, and rounded construction paths
falsifier: any gate execution in which two distinct provenance paths become
           indistinguishable, or in which a receipt constructor accepts a
           vacuous path
evidence:  gate_green
verdict:   alive
```

This is deliberately narrow. It asserts a theorem-shaped statement about the
repo's own witnesses (`same surface value != same zero provenance`), not a
biological, psychopharmacological, metaphysical, or novelty claim. The sedenion
zero-divisor pair used in the annihilation path is an exact computational fact
about `stdlib/algebra/sedenion.sio`; the pipeline does not extend it to any
interpretation of human encounters.

---

## 4. Contract clauses

| Clause | Statement | Acceptance gate |
|---|---|---|
| **P1_SEED_STRUCTURE** | The seed file exists and carries an Evidence State table with all four Garden labels, a "What This Is Not" boundary section, and an Executable Bridge section. | Structural scan passes. |
| **P2_WITNESSES_EXIST** | Both executable witnesses exist, and the probe still contains its `ZERO_PROVENANCE PASS` marker. | Files and marker present. |
| **P3_GATES_EXECUTABLE** | `zero_event_gate.sh`, `zero_provenance_witness_gate.sh`, and `zero_event_native_v2_matrix.sh` exist and are executable. | Executable bits set. |
| **P4_LEDGER_CLAIM** | `garden_zero_encounter_pipeline` scans as a complete Falsification Ledger claim with valid enum values, and its harness and gate paths resolve. | Ledger schema checks pass. |
| **P5_EVIDENCE_CEILING** | The claim's evidence level is not `claim_ready` unless the seed's Claim-ready row explicitly opens with `Yes`. | No overclaim. |
| **P6_ENGINE_SPLIT_DISCLOSED** | The claim note discloses the `lean_single` execution engine, and the pipeline gate composes both witness gates without hiding the native-v2 frontier. | Disclosure strings present. |

---

## 5. What this is NOT

- **Not a promotion of the seed's metaphors.** The Spinozan bad encounter and
  the clinical butterfly remain `Garden`/`Hypothesis` layer content; only the
  executable proposition reaches the ledger.
- **Not a native-v2 parity claim.** Witness execution is explicit
  `lean_single`; native-v2 parity remains the open frontier tracked by
  `scripts/ci/zero_event_native_v2_matrix.sh`.
- **Not a general pipeline for all seeds.** This is one instantiation, on one
  seed, with one claim. Generalizing requires a second instantiation, not
  speculation.
- **Not a proof system.** The contract checks structure, presence, and honesty
  bounds; it does not verify the underlying mathematics (see the R2 contract
  and the Lean formalization for proof-shaped evidence).
- **Not a clinical artifact.** No patient data, no clinical interpretation
  rule, no treatment claim.

---

## 6. Reproduce

```bash
python3 scripts/research/garden_to_claim_pipeline_contract.py
# expect: P1..P6 PASS, GARDEN_TO_CLAIM_VERDICT P_GREEN

bash scripts/ci/garden_to_claim_gate.sh
# expect: both witness gates PASS, then GARDEN_TO_CLAIM_GATE_OK

bash scripts/ci/falsification_ledger_gate.sh
# expect: FALSIFICATION_LEDGER_GATE_OK (ledger still green with the new claim)
```

Pure Python plus the existing witness gates; no parser change, no new
dependencies.

---

## 7. AI disclosure

Spec, contract, gate, and claim drafted under human direction (2026-07-25).
Math-facing content reviewed via `bin/llm-offload -t math-review` per
`.claude/AGENT_OFFLOAD_POLICY.md`. No clinical content. GAIDeT-ICMJE 2025.
