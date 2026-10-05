<!-- docs:meta
topic_id: repo.docs.audit.epistemic-shape-ratchet-2026-08-19
authority: repo_only
audience: users
last_validated: 2026-08-19
validated_by: grok-cli5
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.epistemic-shape-ratchet-2026-08-19
-->

# Epistemic value-shape ratchet

**Filed:** 2026-08-19 · **Lane:** grok-cli5 · **Status:** gate live, shapes frozen, not unified

## Why

On `origin/main` the noun for the epistemic value denotes structurally
different objects. `label: 0` is `measured` (strongest provenance).
`confidence: 0` is the weakest possible claim. No diagnostic separates
them. Which form is canonical is a founder ruling and is not decided
here. What follows without that ruling is that a further form must not
appear.

This change does **not** alter any existing declaration. The two-line
`stdlib/compiler/epistemic/knowledge.sio` fixture stays.

## What is frozen

`scripts/ci/epistemic_shape_ratchet.tsv` holds the set of field
signatures (`name:type,…`, whitespace collapsed). The gate measures
**shape**, not declaration count.

Measured 2026-08-19 on `origin/main` `ddcacddfaa`. 15 live declarations,
9 shapes.

| Shape | Sites |
|---|---|
| `val:f64,variance:f64,confidence:i64` | `stdlib/epistemic/knowledge.sio`, `examples/real_sounio_capability_demo.sio` |
| `value:f64,variance:f64,label:i64` | four `examples/` files, including `examples/science/darwin_epistemic_pbpk.sio` |
| `provenance:Provenance` | `stdlib/compiler/epistemic/knowledge.sio` (two-line fixture) |
| `inner_type:Box<TypeExpr>,epsilon:Option<Box<EpsilonBound>>,validity:Option<Box<ValidityCondition>>,provenance:Option<Box<AstProvenanceKind>>,proof_constraints:Option<Box<KnowledgeConstraintList>>` | `self-hosted/parser/ast.sio` `KnowledgeTypeInfo` |
| `value:i64,variance:i64,confidence:i64` | `examples/knowledge_native.sio` |
| `value:T,uncertainty:T,confidence:T` | three `tests/native-v2/` files |
| `value:T,uncertainty:f64,confidence:f64,provenance:String` | `examples/test_knowledge_type.sio` |
| `value:T,confidence:BetaConfidence,provenance:ProvenanceNode` | `examples/alphageozero_final.sio` |
| `value:T,variance:f64,confidence:BetaConf,provenance:Prov` | `tests/run-pass/generic_knowledge.sio` |

The dispatch table listed four rows. A full word-boundary scan of exact
`Epistemic` / `Knowledge` / `KnowledgeTypeInfo` found the five extra
shapes above. They are frozen too. A sixth form among the original four,
or a tenth overall, both fail.

## Match rule

Line-anchored `struct Epistemic`, `struct Knowledge`,
`struct KnowledgeTypeInfo` (optional `pub`, optional `<…>`). Word
boundary after the name, so `EpistemicOrderedMap`,
`EpistemicNeuralNetwork`, `KnowledgeARIMA`, and `KnowledgeConstraint`
are not this noun.

## Exclusions

`archive/`, `bootstrap/`, and `self-hosted/bootstrap/` are history (or
the frozen seed), not live code. The gate states this in its header.

## Controls

**Positive.** A seventh declaration with a new shape was written at
`examples/science/seventh_epistemic_ratchet_probe.sio` and removed after
the run:

```
EPISTEMIC_SHAPE_RATCHET_FAIL new_shape file=examples/science/seventh_epistemic_ratchet_probe.sio:1 shape=value:f64,mystery:i64
status=fail
metrics {total=16, passed=15, failed=1, not_run=0}
```

**Negative.** `--self-test` writes a temp file that repeats
`val:f64,variance:f64,confidence:i64`. The gate stays green
(`negative_frozen_shape_allowed=true`). Shape is measured, not count.

`--self-test` is what CI runs on every pull request, so the refuse is
not a one-off.

## Reachability

Wired as `.github/workflows/epistemic-shape-ratchet.yml` (pull_request,
merge_group, push to main). It does not compile anything. `ci.yml` was
under another lane's claim; a dedicated workflow is still a reachable
gate. A gate that never runs is not a gate (#1978).

## Reproduce

```bash
env -u SOUC_BIN -u SOUNIO_SOUC_BIN bash scripts/ci/epistemic_shape_ratchet_gate.sh
env -u SOUC_BIN -u SOUNIO_SOUC_BIN bash scripts/ci/epistemic_shape_ratchet_gate.sh --self-test
```

Expected: `status=pass` and
`metrics {total=15, passed=15, failed=0, not_run=0}` on the scan;
`total=17` on `--self-test`.
