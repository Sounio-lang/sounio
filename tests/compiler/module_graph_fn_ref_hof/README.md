# Imported named-function reference vertical

This fixture proves a narrow full-path claim about an imported named function
used as a value. It deliberately keeps checker evidence separate from lowering
evidence.

During `check`, imported names begin as `TyUnknown` stubs. The positive fixture
uses `double` once through the ordinary in-place checker path and once inside an
`ExprLoop`, which crosses the remaining by-value bridge. With
`SOUNIO_CHECKER_CONTEXTUAL_LOOKUP_TRACE=1`, the gate requires exactly two
`TyUnknown` to `TyFn` resolution receipts.

The private-reference fixture exercises the same two paths and must emit exactly
two E175 diagnostics. This prevents the permissive `TyUnknown` fallback from
masking visibility authorization for function values.

Only after those checker controls pass does the gate build the positive fixture
through the canonical AST-closure full-IR path. The resulting ELF must exit 42
with empty stdout and stderr, while compact/prebundle/fallback markers remain
forbidden.

This does not prove that the checker `TypeEntry` is the identity carrier used by
lowering. Lowering still resolves the named function independently. Capturing
closures, arbitrary closure environments, canonical import bindings, function
value ABI generality, and SOIR round-trip identity remain outside this witness.
