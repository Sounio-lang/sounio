# Historical extension design notes

This record captures the design before final gate 19769. See FINAL_CANDIDATE.md
for the qualified source, results and remaining promotion boundary.

This document describes the WIP following component commit b778f1bd60.
It is not a production qualification or authorization to replace a seed.

Root approved the x86 sidecar design after reviewing the actual legacy registry:
EN stores a spelling hash, variant count and spelling hashes. It does not carry
a module owner or defining declaration. en_find returns the first hash match;
gl_name_hash is FNV with a37-bit mask over name bytes. The parser can flatten
qualified paths. Arm-count therefore cannot prove nominal coverage.

The new proof binds one EN entry to exact declaration/name/variant token spans.
It refuses any duplicate enum hash/name or conflicting struct hash/name. Its
structural validator balances parentheses, brackets and braces and skips payload
identifiers; it must agree with the existing emitter's registry indices. Unknown
or mismatched legacy metadata remains unproved. This is deliberately narrower
than a new nominal type resolver. Homonyms/aliases whose origins have been lost
cannot be recovered by guessing.

Budgets: at most256 enum records,254 variants,64 nested payload delimiters and
2048 validation token steps per enum. Unknown/exhausted scans conserve. Coverage
is a local array of confirmed variant indices with an exact enum prefix. A
duplicate marks the same cell; a guard does not add coverage. Bool requires
scrutinee Ty4 and literal tokens35/36; integer0/1 cannot prove Bool. Hash0 does
not prove Option/Result identity.

19755 qualified the first86cases and private generation2, including wrong-owner,
homonym, struct conflict, duplicate, guard and payload-identifier negatives.
Real hash-collision control was generated independently by C++23 in19759:
Collision1023412112909222624 and Collision12091073836520776754 both hash to
98110416378. The bounded unsuccessful search19758 and syntax failure19757 remain
preserved as predecessors.

Shadow presence follows the exact existing emitter predicate src>=0; slot0 is
present and negative slots are absent. Three32-bit masks cover69 fields without
signed bit63 overflow. Each join unions only live arms; a missing field in one
arm stores zero if another arm supplies that field, while a field absent from
every live arm remains-1 after restore.

19760 qualifies the actual extracted helper bodies independently of the whole
producer:211output rows cover all69fields, -1/0, boundaries31/32/63/64/68.
C++23 unsigned masks reject132predecessor mismatches and accept the masked
restore. This alone does not qualify the full32-channel/Hessian pipeline.

The full pipeline controls under preparation include pure f64,32channels,
off-diagonal H01, SX channel31, and the existing cap32 behavior. The33rd measure
retains a variance fallback and does not acquire a33rd sensitivity channel.
No new channel capacity is being introduced.

A fresh modern compiler build and replay of the eleven original SIG139 tests
remain mandatory. No lower.sio or AArch64 change is permitted in this lane.
