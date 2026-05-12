# formal/proof_obligations/

This directory hosts **canonical proof-obligation references** for Sounio.
The actual Lean files live in `formal/lean4/` so they are discovered by
`lakefile.lean` and built by `lake build`.

## Index

| Obligation | Lean module |
|---|---|
| Vancomycin dosing safety (M3 milestone) | `formal/lean4/SounioVancomycinDosingSafety.lean` |
| Tacrolimus oral C24h-trough dosing safety | `formal/lean4/SounioTacrolimusDosingSafety.lean` |
| Tacrolimus + sirolimus DDI uncertainty | `formal/lean4/SounioTacrolimusDDI.lean` |

## How to add a new obligation

1. Add the Lean module under `formal/lean4/SounioFoo.lean`.
2. Register it in `formal/lean4/lakefile.lean` as a `lean_lib`.
3. Add a row to the index above.
4. Run `cd formal/lean4 && lake build SounioFoo` to verify.

## Status

This is a *staging* directory; the long-term plan (M4+) is to add a
small extractor that scans Sounio sources for `pub fn ... with ...`
contracts and emits Lean obligations automatically, indexed by
`<file>:<line>`.
