# stdlib/algebra

Advanced algebraic structures.

## Types
- `Sedenion`: 16-dimensional algebra
- `Ladder`: Ladder algebra
- `Clifford`: Clifford algebra
- `CayleyDickson`: Cayley-Dickson construction
- `Fano`: Fano plane
- `Jordan`: Jordan algebra

## Multiplication convention (canonical) — Convention X: `cd_sigma` / XOR

All hypercomplex products (`oct_mul`, `sed_mul`, and the Cayley–Dickson tower) are canonically defined
by the recursive Cayley–Dickson 2-cocycle sign `cd_sigma` with **XOR indexing**:

```
e_i · e_j = σ(i, j) · e_{i ⊕ j},   σ = cd_sigma(i, j, k) ∈ {+1, −1}
```

where `k` is the tower level (octonion `k=3`, sedenion `k=4`). `cd_sigma` lives in
`stdlib/algebra/octonion.sio` and `stdlib/algebra/cayley_dickson.sio` (byte-identical bodies;
Albuquerque–Majid, Lean-verified). Canonical markers: **`e1·e2 = +e3`**, **`e2·e5 = +e7`**,
`e_i² = −1`, and the canonical sedenion zero-divisor `(e3+e10)·(e6−e15) = 0`.

**Executable oracle:** `tests/run-pass/hypercomplex_convention_crosscheck.sio` asserts the full 8×8
and 16×16 canonical tables. Any new/edited `*_mul` must match it.

### Conformance map (as of 2026-07-14)

| Status | Files |
|---|---|
| ✅ Convention X (canonical) | `algebra/octonion.sio`, `math/octonion.sio`, `nn/octonion.sio`, `onn/lib.sio`; `math/sedenion.sio`, `algebra/sedenion.sio`, `compiler/ast/sedenion_ops.sio`, `math/sedenion64.sio` + `math/cayley_dickson.sio` |
| ⚠️ divergent — migration in progress | `hypercomplex_graph/oct_graph.sio` & `self-hosted/hypercomplex/octonion.sio` (Convention Y, Fano-triples `(1,2,4)…`, `e1·e2=+e4`; 42/64 basis products differ); `snn/base.sio` (sedenion sign table, 56/256 differ) |

The X-files were verified numerically identical to `cd_sigma` (octonion 0/64, sedenion 0/256). The
sedenion CD-split `(ac−d̄b, da+bc̄)` used by the f64 sedenion files equals `cd_sigma(·,·,4)` exactly.
The divergent files are being migrated onto X; see the hypercomplex audit
(`docs/audit/HYPERCOMPLEX_ALGEBRA_AUDIT_2026-07-14.md`). Note: Conventions X and Y are isomorphic
octonion algebras (both 168 non-associative basis triples) under different basis labelings, so
convention-independent quantities (norm, associator-norm) are unaffected by the migration; only
component-level outputs change.
