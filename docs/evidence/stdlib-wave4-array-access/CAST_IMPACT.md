# Declaration-linked cast impact inventory

inventory.cjs is the second, corrected scanner. It preserves line positions while stripping comments and literals, recognizes function signatures with array extents, and recognizes parenthesized and unparenthesized explicit references. It reads tracked stdlib, tests and self-hosted Sounio files from c3c4345, with the three before-library42 files overlaid. The first scanner and both full outputs remain in the durable archive, so the correction does not erase an earlier result.

The second scan reads 8,211 files and records 2,333 raw-pointer casts: 449 in self-hosted, 1,573 in stdlib and 311 in tests. Sixty casts have a unique syntactic link to an explicitly declared local, parameter or global array. One linked declaration is global/BSS. The remaining 2,273 casts are unresolved by this scanner; they include heap addresses, fields, aliases and imports. This is a selection inventory, not a typed rejection count and not evidence that unresolved uses are safe.

The sixty linked uses are:

- 42 local res arrays in BigFrame ops/stats/ML, repaired by c3c4345.
- Six references to [Name;64] cast to pointers to that same array in check.sio. These must remain admitted.
- Six value/reference fixtures from the original BigFrame array diagnosis.
- Three ABI regression witnesses, including global/BSS GBUF:[i64;2] cast to Pair.
- Two array-reference-to-u8-pointer uses in test_gzip_pure.sio.
- One array-reference-to-i8-pointer use in kaxi_feedback.sio for CUDA host readback.

The fixture/BSS/gzip/CUDA uses need explicit migration or an independently specified address operation if a guard is adopted. A source contract must not silently depend on whether a backend placed the array in BSS or represented it through a local managed handle.

DECLARATION_LINKED_IMPACT.json retains the exact declaration and cast line for all sixty uses, plus the counts and limitations. The full 2,333-record IMPACT.json is archived under services/array-access-custody13/input/input/impact02, not expanded into this Git receipt. Aliases, imported types, field projections, nested generics, shadowing and complete BFS resolution remain outside the scanner's guarantee. E398 adoption remains subject to actual checker component qualification and coordination with the existing ref-to-raw owner.
