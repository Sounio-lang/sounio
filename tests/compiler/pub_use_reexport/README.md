# lean_single named-function `pub use` contract

This fixture set pins the narrow lean_single fix for GitHub issue #842.

## Supported boundary

The compiler supports direct and one-facade named function imports written as
braced lists:

```sio
use public_leaf::{public_route_value}
pub use public_leaf::{public_route_value}
use public_facade::{public_route_value}
```

The facade forwards only the listed function. Another public function in the
same leaf remains unknown through the facade, and a facade's private `use` does
not expose its imported public function to a consumer.

Types, structs, constants, aliases, globs, renames, and chained re-exports are
outside this contract. Their prior lean_single behavior is intentionally kept.

## Implementation boundary

`resolve_imports` retains the flattened source bundle but records braced named
edges between the main source and loaded source files. Root function lookup is
constrained only when the root has a matching named import:

1. A definition in the directly requested source is visible.
2. A definition one source deeper is visible only through a matching `pub use`.
3. A private edge or a different exported name is not visible.

This is not a general module graph, namespace, or visibility implementation.
The legacy flattened bundle remains in place for imports outside this boundary.

## Fixtures

| Fixture | Expected result |
|---|---|
| `public_consumer.sio` | Compiles and prints `PASS pub_use_named_function_reexport` |
| `not_reexported_consumer.sio` | Rejects `not_reexported_value` as unknown |
| `missing_consumer.sio` | Rejects `missing_route_value` as unknown |
| `private_consumer.sio` | Rejects `private_route_value` as unknown |

Every semantic rejection must exit exactly `1`, contain an anchored `unknown
identifier` diagnostic, and contain the exact line `typecheck: failed`. Signal
exits, fatal diagnostics, and `rc >= 128` are not accepted as semantic evidence.

## Gate

Run the bounded current-source gate:

```bash
bash scripts/ci/lean_single_pub_use_reexport_gate.sh
```

The required receipt is:

```text
scope=named_function_reexports direct_import=PASS missing_symbol=REJECTED facade_forwarding=1 selective_reexport=1 private_import_isolated=1
```

The gate first builds `self-hosted/compiler/lean_single.sio` with the checked-in
lean_single seed, then compiles/runs the positive witnesses and classifies each
negative witness. Its diagnostic classifier also self-tests valid `rc=1`,
signal-style `rc=139`, and fatal-contaminated output.

Concept-ID: `SOUNIO-MODULE-CLOSURE-AUTHORITY`

Resolved blockers:

```text
BLK-20260713-lean-single-import-visibility
BLK-20260713-lean-single-private-import-visibility
```

The evidence level is the named gate receipt and executable facade ELF only; it
does not broaden the modular compiler's separate module-closure claims.
