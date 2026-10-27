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
