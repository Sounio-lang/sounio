# Final x86 flow candidate: nominal coverage and shadow presence

Status: **qualified private producer candidate, awaiting root review and a fresh
modern compiler replay**. This extends component b778f1bd60. It does not claim
that the eleven original compiler SIG139 regressions are already repaired.

## Evidence and exact artifacts

Slurm job19769 completed0 in89s using1CPU and8GiB. The final runner built the
baseline and candidate from source with the immutable9cf seed, then privately
recompiled the candidate source and repeated the same fixtures with generation2.

| Arm | C++23 oracle |
| --- | --- |
| Rebuilt baseline, byte-identical to seed9cf | 104cases,45 failures |
| Candidate generation1 | 104/104 |
| Private generation2 | 104/104 |

Selfcompile returned0. The wrong-value and duplicate-output oracle controls
returned4, and missing-output returned3. Source, SDK and seed hashes passed the
before/after checks. The source guard found148 AArch64 function chunks
byte-identical; there was no AArch64 behavioral change or new target claim.

Final source SHA256:
2157ef5891872bdd6cd45f903ac6ddf0432fb151bab11673450ef64662cc37f1

Generation1 producer:
55062670c533c25fcd455af6cefe82f6fbb6f63cf0d6fa2750ac645d1e6547c5

Generation2 producer:
a2da55a3c40e25df6346e0e72b570fc37d6c7315e5d07d9474bcb5354b4f71dc

Both private ELFs are retained under:
 /scratch/sounio-stdlib-wave3-20261027/compiler-resumed/legacy-flow-unified03/results/

No existing binary was replaced. Root owns the next single sealed modern build;
the producer here is not a replacement for bin/souc-lean-single-x86_64.

## Nominal coverage

The legacy EN registry contains spelling hashes and ordinal indices, without a
physical owner. Its old scanner can count payload identifiers as variants.
The new x86 sidecar therefore validates the entire declaration structure and
binds accepted entries to exact enum/variant token spans and bytes. It requires
agreement with the existing emitter registry. A unique nominal declaration is
required; duplicate names, hash collisions and struct/enum conflicts conserve.

Match coverage marks confirmed indices, requiring an exact defining enum prefix.
Duplicates mark one cell; guards add none. A wildcard/binding/OR wildcard supplies
coverage only with a validated arrow token and no guard. The old arm-count rule
is not a proof. Unknown/lost alias identity remains unproved. Bool coverage uses
only scrutinee Ty4 plus literal tokens35/36; integers0/1 are not Bool proof.
Option coverage requires a registered option_hash. Hash0 does not certify
Option/Result; an unconditional wildcard remains a valid coverage proof.

The validator is bounded to256 enum records,254 variants,64 payload delimiters
and2048 token steps per enum. OR-patterns are bounded to8 alternatives.
Truncation, incomplete structure, mismatched registry indices or exhausted
budgets do not gain exhaustiveness. The implementation does not introduce a new
payload ABI or repair an already-inconsistent legacy registry.

The real37-bit collision was generated independently by C++23 in job19759:
Collision1023412112909222624 and Collision12091073836520776754 both hash to
98110416378. The final matrix rejects this collision both between nominal
declarations and between variants in one enum.

## Presence of derivative metadata

The exact existing spill/consumer predicate is slot>=0:0 is present, negatives
are absent. A missing arm stores the zero bit pattern; that is distinct from
claiming a shadow exists at the joined value.

The new three32-bit presence masks cover the variance slot,8 first derivatives,
36 Hessian fields and24 extra sensitivity fields. Only live branches contribute
to their union. Restore preserves-1 if a field is absent on all live paths; if any
live path supplies it, the join slot is used, with zero written by missing arms.
The normal x86 Ty2 if merge now uses the same mechanism as if-let and match.
RAX is preserved by target-code push/pop while spills are emitted. Ty13/f128
does not use this helper; no new f128 qualification is claimed.

Job19760 compares the actual extracted helper bodies with their predecessor.
An independent C++23 unsigned-mask oracle validates211 rows covering all69fields,
-1/0 and boundaries31/32/63/64/68. The predecessor has132 mismatches; the masked
restore has none. Custody job19770 confirms these helper bodies are byte-identical
to the final candidate.

The104-case pipeline matrix includes exact bit checks for:

- pure f64 with no measure;
- value and variance in both live and single-live joins;
- all32 channels and SX channel31 in if, if-let and match;
- off-diagonal H01 for x0*x1 + sum(x2..x31), with each xi=1±0.5:
  value31, variance8, S31=1 and H01=1;
- the33rd input's existing opaque variance fallback0.25, without claiming a33rd
  channel, including a sole live arm opposite a returning tracked arm;
- explicit integer-to-f64 casts in both live arms, while a direct Ty1/Ty2 join
  remains rejected.

This preserves the existing type-join policy. Implicit integer binding promotion
was not assumed: the original test spelling was correctly refused by both
producers, preserved in19761, and the final positive fixtures use explicit casts.
The reference tolerances were not adjusted; these are exact binary64 cases.

## Reproduction and promotion boundary

Run scripts/compiler/legacy_flow_gate.sh inside Slurm with explicit baseline
source, candidate source, immutable seed, self-hosted SDK and a new output
directory. It retains every source/ELF/status/output and runs the independent
C++23 oracle. Original scientific tests and deadlines were not changed.

Predecessors19753,19755,19760,19761 and19764 remain preserved. The full final
archive, input SDK, generated fixture ELFs, compact receipts and SHA are linked
from19769/compact/CUSTODY.sha256. Earlier failed collision searches remain in
scratch;19757 was a C++ syntax error and19758 exhausted its original bounded
name sequence. Neither is counted as a collision qualification.

Root must still rebuild its sealed current modern compiler using the chosen
private producer and replay the eleven original SDK19392 SIG139 inputs, PBPK,
loader and focused current compiler gates. No bootstrap was started by this lane.
There was no lower.sio edit, shared seed replacement, push or main merge.
