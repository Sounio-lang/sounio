# TypeKind archaeology fixtures (families F + H + ladder controls)

**Protocol:** v3 — position is **derived**, never stored.

## Layout

```
tests/typekind/<kind_slug>/{pass,refuse}.sio   # Claim-ready / Reserved pairs
tests/typekind/<kind_slug>/README.md           # Garden declaration (no fixtures)
tests/typekind/index.tsv                       # kind → paths + expected diagnostic + deepest layer
scripts/ci/typekind_archaeology_gate.sh        # runs pairs; prints derived ladder table
```

## Index columns (no position)

| column | meaning |
|---|---|
| `kind` | TypeKind name |
| `pass_fixture` | program that **must** compile+run (empty ⇒ Garden) |
| `refuse_fixture` | program that **must** be refused with named diagnostic |
| `expected_diagnostic` | e.g. `E001`, `E218` |
| `deepest_named_layer` | deepest layer that still names the kind (parser…codegen) |

## Derived positions (gate output)

| condition | position |
|---|---|
| both fixture paths empty | **Garden** |
| `pass` runs (rc=0) + `refuse` fails with expected diag | **Claim-ready** |
| `pass` fails with expected diag + `refuse` fails with same | **Reserved** |
| `pass` runs + `refuse` also passes | **Executable** + gate **XPASS** (fail) |
| `pass` fails without Reserved pattern | gate **PASS_REGRESSION** (fail) |

A new TypeKind with no fixtures lands in Garden without a human decision.
A refuse fixture that starts passing trips XPASS — same pattern as the XPAS
gate that landed 2026-08-19.

## Scope of this tree

Families **F** (shape/gradient/complexity) and **H** (wide ints / pointers /
slices) plus ladder controls TyI64/TyBool/TyArray. Other families (A causal,
D epistemic, …) own their own indices under other lanes — do not merge without
coordination.

## Convert from prose census

Prior handwritten verdicts map 1:1 onto missing fixtures:

- wrote **Hypothesis** → no constructing program → leave empty (Garden under v3
  executable rule; prose Hypothesis is historical archaeology of internal seeds)
- wrote **Claim-ready** → extract refuse + pass into this tree
- wrote **Reserved** (v2 off-ladder) → both programs refuse with the reserved diag

Run:

```bash
bash scripts/ci/typekind_archaeology_gate.sh
```
