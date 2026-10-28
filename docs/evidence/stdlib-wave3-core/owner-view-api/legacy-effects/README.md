# Concrete heap alias effects: static qualification

Four existing scalar aliases mutate their owner fields and called write_i64 or
write_f64, but declared only Alloc: heap_i64_vec_push, heap_i64_vec_free,
heap_f64_vec_push, heap_f64_vec_free. Their signatures now also declare Mut.
No function body, generic container implementation, fixture or expected value
changes in this commit.

The patch was formed from the committed HEAD file, then staged independently
of the existing generic HeapVec draft. Qualified committed file SHA256:
195ccc8b0e2a7758d571c84f25855bde9a55dc30275ec7c3264142994d0f9010.
EFFECTS_ONLY.patch contains exactly the four annotation changes.

Slurm 18990 first exposed fourteen E035 diagnostics when checking
collections::mod through the existing heap_vec dependency. Slurm 19014
checks the corrected library module successfully. Slurm 19020 checks the
same complete production file plus a same-file alias caller: before check1
with E035, after check0. The caller and before/after diagnostics are retained
here. Raw compiler SHA256:
f3ceba1af545a006ab61c66dafde4bbc5eb63c0b1463f63d6ed2f3dd853cb332.

Native qualification is explicitly OPEN: Slurm 19020 after compile1 exposed
the separate template-retention defect (E008, generic get/pop returning i64
for T), rather than an effect error. The original imported method fixture,
unchanged in 18997, also exposes generic/concrete method-name collision.
These failures are preserved, not reclassified as passes or excluded from
the sprint. Receiver/type-identity dispatch and correct retention of the
specializer result are being repaired in the isolated receiver worktree.

This commit qualifies the effect declarations by the actual checker. It does
not claim native HeapVec methods, a generic linear-owner HeapVec, or the view
facades are complete.

