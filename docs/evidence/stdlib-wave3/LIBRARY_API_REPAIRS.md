# Callback effects and arena repairs

Slurm18162 used fresh a955 compiler7df87 and preserved the three original
test sources. test_calculus_e2e, test_walley_frechet_compose, and
test_arena_e2e failed check before the library changes and passed check,
compile, and native execution afterward. Algorithms, test iterations,
numeric assertions, and tolerances were unchanged.

Calculus callbacks and Knightian binary monotone callbacks now declare
Mut/Div/Panic in their function type, matching their library's existing
effect contract. Pure callbacks remain admissible. The added IO-callback
negative control still fails E009 even when its caller declares IO.
The compiler's effect compatibility was preserved.

Arena allocation now checks nonzero power-of-two alignment, aligns the actual
allocation address including allocator prefixes, checks padding and requested
size by subtraction before advancing counters, and returns a typed null on
failure. Drop uses the same explicit null representation and remains idempotent.
The added native control exercises alignments1..256, invalid0/3, oversized
requests, unchanged counters after refusal, and allocation after drop.
This does not introduce linear Arena ownership or change the allocator ABI.

Receipts, input identities, native ELFs, and before/after logs are retained in
the wave3 library-api-repair packet. Source/test identities and compiler
hash guards passed. No Python numerical implementation was used.

The legacy epistemic_new constructor is a public forwarding alias to ep_new,
including its canonical confidence clamping. The original DP value-iteration
test first passed check but was refused during native compilation for an absent
constructor body. After providing the alias it reached SIGILL before output:
mdp_new indexed nested arrays that had never been allocated. Its constructor now
initializes both existing 256-element arrays explicitly. All Bellman algorithms,
dimensions, test scenarios, assertions, and tolerances remain unchanged.

Slurm 18244 preserved the original compiler-refusal witness and executed the
original DP test successfully. Slurm 18248 repeated its native execution and
verified the complete sealed source and compiler manifests, producing
DP_ORIGINAL_BEFORE_AFTER_OK. The separate new bridge fixture executed but its
independent C++23 oracle rejected an existing EFE logarithm approximation; that
failure is retained and is not part of this DP pass claim.

Slurm 18254 ran the reusable epistemic composition gate before the active
logarithm repair and rejected the native EFE output against C++23 at the fixed
1e-11 threshold. Slurm 18271 ran the same gate after the canonical logarithm
repair: original DP, constructor bit fields/clamping, five deterministic
Monte Carlo rows and two Klibanoff/EFE compositions all passed. A deliberately
zeroed MC variance was rejected by the independent C++23 oracle. Missing Prob
and an I/O callback were rejected with E035 and E009; zero samples failed before
the success marker. All sealed source, compiler and executable manifests passed
final verification. This is focused qualification, not whole-corpus completion.

The propagate callback contracts now admit the existing Mut/Div/Panic callback
effects. Prob remains required for Monte Carlo APIs, including deterministic
seeded sampling. All four Monte Carlo entrypoints explicitly require a positive
sample count. klibanoff_compose_with_active exposes composition of the existing
certainty-equivalent operator and active EFE calculation.
