# TypeKind archaeology — family C (probabilistic / information)

**Protocol v3.** Position is **derived**, never stored.

Assigned kinds: `Distribution` `Sample` `ConditionalDist` `Entropic`
`MutualInfo` `KLBounded` `ELBO` `VariationalFamily` `MarkovChain` `SDE`
`Martingale` `StationaryDist`.

## Why the indexed pass/refuse columns are empty

v1/v2 wrote **Hypothesis**: constructors and `compat.sio` rules exist in
the checker; no program constructs `TypeKind::TyX`. Under v3 a kind
without a pass+refuse pair is **Garden**. That is the conversion, not a
loss: the handwritten Hypothesis was "we did not find a constructing
program." The constructing program is what would fill `pass`.

Do **not** fill the pair with:

| program | what it does today |
|---|---|
| `fn id(x: Kind<f64>) -> Kind<f64> { x }` | rc=0 for Kind **and** for `NoSuchType` |
| `let x: Kind<f64> = 1.0` | E001 expected Kind — same wall as `NoSuchType` |
| `fn coerce(x: Kind<f64>) -> Kind<i64> { x }` | rc=0 — inner args ignored |

Family G measured the same pair on `FairPrediction` / `NoSuchType`. If
that pair were indexed, `NoSuchType` would derive Claim-ready. It is a
label, not a type.

`*.ghost_identity.sio` and `*.ghost_inner.sio` are the two programs the
census already ran. They are **attempts**, not fixtures. When
`ghost_inner` starts failing with a named diagnostic that
`NoSuchType.ghost_inner` does not emit, fill `index.tsv` with a real
constructor as `pass` and that file as `refuse`.

Bounded kinds also keep `*.ghost_bound.sio` (`Kind<f64,2> -> Kind<f64,1>`).
Today rc=0. That would be the refuse if `TyKLBounded` / `TyEntropic` /
`TyELBO` enforced ε.

## Deepest named layer

All twelve die at **checker** (`TypeKind::TyX` in
`self-hosted/check/types.sio`). There is no `TypeExprKind::TypeX` in
the parser, no `HlirTypeX`, nothing in `self-hosted/ir/` or codegen.
Token `sample` exists as a keyword; it does not produce `TySample`.

Index rows use `-` for an empty path (bash `read` collapses consecutive tabs).
`-` + `-` is Garden.

## Run

```bash
bash scripts/ci/typekind_archaeology_c.sh
```

The script prints the derived table. It does not store positions.
It also re-runs each ghost pair against `NoSuchType` and fails if a
kind **diverges** (the type became real or Reserved).
