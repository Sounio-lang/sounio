# Pre-Implementation Survey Results (2026-03-03)

## TypeKind (check/types.sio) — Files Needing Modification

- check/types.sio — add constructors (no dispatch breaks)
- check/compat.sio — types_compatible falls through to false; is_numeric/is_integer/is_float/is_bool predicates exclude new variants
- check/check.sio — field access (2177-2236), index (2272-2353), call (2395), match (2805-2828) dispatch chains silently skip new variants
- ir/serialize.sio — write_type_tag (692-719) falls to tag 0, corrupts round-trips
- check/epistemic.sio — wildcard arm handles gracefully; semantic review needed
- check/hyper.sio — no break, hyper-specific predicates return false

Zero TypeKind references in: hlir/, native/, gpu/, parser/, lexer/, resolve/, io/, effects/, llvm/

## HlirTypeKind (hlir/ir.sio) — Files Needing Modification

- hlir/ir.sio — add variants to enum (line 88) + 2 constructors; 5 predicate chains (415-454) silent-omit
- llvm/type_convert.sio — CRITICAL: 9 dispatch chains (73-610), convert_hlir_type falls to i64, size/alignment/predicates all need new arms
- hlir/lower.sio — no break (uses constructors only)
- llvm/codegen.sio — no break (stores kinds, no dispatch)

## Epsilon/Validation Call Sites

- epsilon_subsumes_call_boundary: def check/epistemic.sio:482, called at :773
- knowledge_call_boundary_compatible: def check/epistemic.sio:764, called at check/check.sio:2577
- knowledge_meta_is_valid: def check/epistemic.sio:460, called at :54
- knowledge_epsilon field: decl types.sio:52, read at check.sio:157,161 compat.sio:108,113 epistemic.sio:402,529,549,684,689
- epsilon_bound field: decl hlir/ir.sio:157, propagated through 16 constructors
- 3 separate ValidationResult structs (darwin_pbpk/metrics, darwin_pbpk/simulation, stats/validation)
- Validated is a RiskLevel enum variant in stdlib/epistemic/policy.sio:123, not a type constructor

## Provenance Fields Inventory (for R3 manifest)

Compiler-internal:
- provenance_id: i64 on HlirType (hlir/ir.sio:158)
- ProvenanceKind enum: 6 variants (check/epistemic.sio:204-211)
- AstProvenanceKind: 6 variants (parser/ast.sio:430-437), only 3 parsed
- provenance_root_l64: Merkle roots in intern.sio:51, epistemic_arena.sio:91, epistemic_ordered_map.sio:89
- merge_proof: [i64; 4] in GPU tensor (epistemic_tensor_core.sio:64)

Stdlib:
- Provenance struct (knowledge.sio:74): source + steps (rich)
- Provenance struct (check/epistemic.sio:214): kind + source_id (flat)
- MerkleDAG + audit_trail (epistemic/merkle.sio:264,357)
- provenance_chain display (plot/epistemic.sio:327)
- to_audit_trail() (interop/medlang.sio:349,375)

Two distinct Provenance struct definitions — compiler vs stdlib.
Merkle salts replicated across 5 collection files; canonical source: hardware/rtl/kaxi/merkle_root_lane.sio
