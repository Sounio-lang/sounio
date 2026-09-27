# Knowledge annotation silent drop — 2026-08-19

An unrecognised component in a `Knowledge<…>` / `Knowledge[…]` annotation
is accepted with no diagnostic. The parser advances and leaves provenance
empty. This is NO-VERSUS-UNKNOWN at the site where provenance is alleged.

This receipt does not change the language. It makes the silence audible.

Compiler: Madaros v0.80.0. SHA: `9fdaa5772b`. Measurements ran on Slurm
partition `cpu-ops` (host `cpuops-t560-proxmox`), not on the workspace pod.

## Controls

| Program | Role | Slurm `souc check` |
|---|---|---|
| `tests/audit/knowledge_annotation_unknown_component.sio` — `Knowledge[f64, fn]` | Positive: unrecognised **keyword** hits the else-arm (must not be `Ident`; those become epsilon binders) | **rc=0**, no diagnostic |
| `tests/audit/knowledge_annotation_measured.sio` — `Knowledge[f64, Measured]` | Negative: recognised provenance token | **rc=0** |

If the unknown witness had failed to compile, the defect would not have been
the else-arm described in the dispatch, and this work would have stopped. It
compiled. The silence is the one described.

Angle-form `Knowledge<f64, fn>` also typechecks (rc=0). So do `true` and `IO`.

`Literature` and `Source` are **not** lexer tokens. Written as components they
take the `Ident` → epsilon-binder arm, not the else-arm. A different hole:
they are not skipped, they are misread as ε.

## Provenance is empty at TypeEntry even when the AST filled it

`check_knowledge_type` (`self-hosted/check/epistemic.sio`) builds
`ty_knowledge(inner, eps)` only. The comment in that function states that
`TypeEntry` does not persist validity/provenance. Live check:

| Program | Result |
|---|---|
| `Knowledge[f64, fn] -> Knowledge[f64]` | rc=0 |
| `Knowledge[f64, Measured] -> Knowledge[f64]` | rc=0 |
| `Knowledge[f64, Measured] -> Knowledge[f64, Derived]` | rc=0 |

The negative control therefore shows only that `Measured` is a **named parser
arm**, not that the type carries a non-empty provenance tag. `provenance_from_ast(None)`
returns `PROVENANCE_KIND_DERIVED` (default), then that meta is discarded.
Type-level observation cannot tell “unknown was dropped” from “provenance is
never stored”. The accusation is the **missing diagnostic**, which the
unknown witness still demonstrates.

## Parser versus AST

### Provenance — `AstProvenanceKind`

| AST variant | Parser arm in `parse_knowledge` / `parse_epistemic_wrapper` |
|---|---|
| `AstProvDerived` | `TokenKind::Derived` |
| `AstProvComputed` | `TokenKind::Computed` |
| `AstProvMeasured` | `TokenKind::Measured` |
| `AstProvSource` | **none** (not a token) |
| `AstProvLiterature` | **none** (not a token) |
| `AstProvInput` | **none** (not a token) |

Parser recognises **3**. AST declares **6**. The three unwriteable names
appear once each in `self-hosted/parser/` — the enum declaration.

### Validity — `AstValidityKind`

| AST variant | Parser arm |
|---|---|
| `ValidDuration` | `TokenKind::Valid` |
| `ValidUntilTime` | `TokenKind::ValidUntil` |
| `ValidWhileCond` | `TokenKind::ValidWhile` |

**3 / 3.** Validity does not have the same unwriteable-variant hole.
An unknown *token* in that position still falls through to the provenance
else-arm or the Ident-as-epsilon arm.

## Silent `else` arms

Exact comment `Unknown component — skip`:

| File | Role |
|---|---|
| `self-hosted/parser/types.sio` (~1117) | `parse_knowledge` component loop |
| `self-hosted/bootstrap/bootstrap_v0.sio` (~3934) | bootstrap copy of the same loop |

A second skip **without** that comment lives in
`parse_epistemic_wrapper_type` (`self-hosted/parser/types.sio` ~1387–1391):
`else { p = p.advance() }` after Derived/Computed/Measured. Same trap for
`Intervention` / `Counterfactual` bracket wrappers.

No other `Unknown component` comments exist under `self-hosted/`.

## Versioned `Knowledge` annotations with components

Non-comment lines matching `Knowledge[…]` / `Knowledge<…>` that name a
provenance or validity word, outside `self-hosted/`, `bootstrap/`, `archive/`:

| Word | Files / lines |
|---|---|
| `Derived` + `ValidUntil` | 10 lines, 3 test files (`tests/run-pass/covid_2020_kernel.sio`, `tests/compile-fail/covid_2020_temporal_*.sio`) |
| `Measured`, `Computed`, `Source`, `Literature`, `Input` | **0** |
| Keyword else-arm (`fn`, `true`, `IO`, …) | **0** |

No versioned user program currently writes a component that falls into the
else-arm. The hole is live in the parser and unoccupied in the tree. Epsilon
components (`epsilon`, `ε`) are common and take the Ident arm by design.

## Gate

```text
bash scripts/ci/knowledge_annotation_silent_drop_gate.sh
```

Uses Slurm `srun -p cpu-ops` when `srun` is present
(`KNOWLEDGE_SILENT_DROP_SLURM=0` forces local). Artifact:

`artifacts/audit/knowledge_annotation_silent_drop/status.json`

```text
status=fail
metrics {total=4, passed=3, failed=1, not_run=0}
```

while the unknown witness typechecks. `failed=1` is the accusation. The
gate exits 1 until a named diagnostic refuses that witness. Wiring:
`.github/workflows/knowledge-annotation-silent-drop.yml`
(`continue-on-error: true` so the accusation stays visible without
blocking unrelated merges).

Do not close this by adding `Source` / `Literature` / `Input` keywords.
That is a founder language decision. Close it by refusing the unknown
component with a named diagnostic.
