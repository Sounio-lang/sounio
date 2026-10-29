# Proposed x86 producer reachability repair — not implemented

The diagnosed failure is in the legacy producer's join metadata, not evidence
that the modern compiler's IR path reproduces the minimal Packet defect.
The fixed implementation must remain outside lower.sio. No compiler source is
changed by this document.

Root's reviewed first-step architecture is a CAN_FALLTHROUGH boolean (or an
equivalent small mask) separate from LAST_STMT_HAS_VALUE/TY/HASH. The boolean
means a path reaches the current syntactic join, not that a value is Unit and
not automatically that the whole function returns.

| Branch A | Branch B | Join result |
| --- | --- | --- |
| reachable value T | reachable value T | value T; preserve corresponding metadata |
| reachable Unit | reachable value T | reject the type mismatch |
| explicit return | reachable value T | preserve T only from the reachable branch |
| explicit return | explicit return | no fallthrough and no join value |
| resolved abort intrinsic | reachable value T | T only if existing resolution proves the intrinsic terminates |
| unknown call | reachable value T | call remains potentially fallthrough; no inferred divergence |

Explicit return sets no-fallthrough independently of HAS_VALUE. A block begins
reachable; sequential composition updates its value/type only on paths that
remain reachable. Parsing/typechecking later unreachable statements must not
erase prior termination or manufacture a join value.

If, if-let and match snapshot branch state separately, including nested
constructs. Join value/type comes only from fallthrough paths. Nonexhaustive or
unproven match coverage retains a possible unmatched fallthrough; an exhaustive
all-return match has none. Do not infer coverage from arm count alone.

Loops conservatively may conclude, including zero iterations. Break/continue
are local transfers to the appropriate loop exit/header and must not escape as
a function-return claim. The first repair need not infer additional aborts or
infinite loops. A user function named panic is not enough: only a previously
resolved terminating intrinsic can justify no-fallthrough.

The implementation needs explicit reset/snapshot/restore at function and
closure boundaries as well as nested blocks. Existing statement-value and
shadow tracking is global and recursive. An inner closure return must not
terminate its enclosing compilation context.

For one live branch, retain that branch's aggregate type/hash and value in RAX,
and its f64/FO/f128 shadow metadata. For two live branches, retain current type
agreement rules and shadow joins. Do not drop existing gradient/variance
bookkeeping for explicit return paths. Unit+value cannot be repaired by OR-ing
HAS_VALUE. The implicit function epilogue must stabilize a reachable aggregate
return into the caller's SRet buffer; a reachable missing value must not silently
return an unwritten buffer.

No decision may rely on the final emitted byte being ret. Existing nop merge
defenses in lean_single.sio already explain why that heuristic is insufficient.

Scope is x86 first. AArch64 has separate duplicated merge/epilogue code around
48k/49k and must not receive an unqualified mirror patch. No cross-engine fix is
claimed until its own specific gate exists.

Before applying a patch:
1. Preserve the seed9cf predecessor and current source/hash.
2. Build a private candidate producer, without replacing seed9cf in place.
3. Run the current Packet matrix: pure if, mixed return branch, explicit control,
   small and280-word payload; keep the four predecessor wrong results.
4. Add reachable Unit+value rejection, Unit+Unit, both-returning branches,
   nested if/match, f64 bits and shadow metadata, closure/function context restore,
   and loop-local break/continue/resolved intrinsic boundaries.
5. Recompile the original representative chemistry/graph and all eleven
   historical failures with identical SDK/tests/1200-second limits, then existing
   compiler semantic suites before claiming integration.
6. Keep new producer, reconstructed compiler raw, focused correctness, complete
   stdlib replay and scientific numerical agreement as separate gate states.

The available evidence is19704 (real compiler state/epilogue),19720 (modern vs
legacy minimal behavior),19723 (independent C++23 field oracle and corruption
controls) and19722 (durable source/raw/seed/trace/matrix custody). No repair,
candidate producer or after-fix pass is claimed yet.
