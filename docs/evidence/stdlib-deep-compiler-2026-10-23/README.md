<!-- docs:meta
topic_id: repo.docs.evidence.stdlib-deep-compiler-2026-10-23.readme
authority: repo_only
audience: users
last_validated: unvalidated
validated_by: none
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.evidence.stdlib-deep-compiler-2026-10-23.readme
-->

# Checked linear borrows and lexical callbacks — native receipt

This is a focused compiler gate. Full corpus/CI/merge are separate states.

Main basis: ce3886de895fa669ab325cb66f23fa5133ef9e90.
Slurm16074 rebuilt the modular compiler from frozen source, then typechecked,
compiled and executed both positive programs. It also required exit1 and E175
for bare/qualified private calls and acquisition of a private function value.
Moved-owner and private-field controls were rejected.

Compiler SHA256:
1148eabd3905ec76911d027c578fb8d6b2bd2dde2c186ee3c02c4c3b97733029

Compiler source checksums:
- check/check.sio: b6d01a3772b1902c511560199d564a63ad37c3334834646b65646d82e502316d
- check/borrow.sio: 15dfc8657a2aa8425532d45a5ec867f72cfa732989b070a52974c8f2ea3da85d

The linear-borrow positive was rejected before the fix with E039. The callback
positive was rejected before lexical dispatch with E175. Frozen failures and
build/resource logs are retained under the private scratch sprint run tree.
The final callbacks include a parameter named tan, without enabling an
unimplemented bare native-math builtin.

Reproduction on a Linux native worker:

```sh
bash scripts/ci/build_modular_madaros.sh "$OUT/madaros.elf"
export MADAROS_RAW_BIN="$OUT/madaros.elf" SOUNIO_SOUC_ENGINE=madaros
export SOUNIO_STDLIB_PATH="$PWD/stdlib"
bin/souc check tests/run-pass/stdlib_deep_linear_borrow.sio
bin/souc compile tests/run-pass/stdlib_deep_linear_borrow.sio -o "$OUT/linear.elf"
"$OUT/linear.elf"
bin/souc check tests/run-pass/stdlib_deep_callback_binding.sio
bin/souc compile tests/run-pass/stdlib_deep_callback_binding.sio -o "$OUT/callback.elf"
"$OUT/callback.elf"
# Each private-access negative must reject with E175, not a crash or timeout.
```

The compiler input manifest covers listed bin/scripts/self-hosted/stdlib files
in the frozen packet; it is not a whole-tree or complete-consumption assertion.
The shipped prebuilt ELF was not refreshed by this source change. Tests and
source checksums were read back against the integration worktree after the run.

An independent read-only agent review of the exact staged source found no
remaining blockers. It verified the local/global distinction, private-value
acquisition checks and balanced borrow suppression. This is not maintainer
acceptance or a second execution. Assigned callback aliases/reassignment were
not part of the focused positive fixture.

CI follow-up: callback regressions now declare Panic on their callback types and
are marked requires: madaros. These exercise Madaros lexical/visibility behavior;
the separate stage2 full-suite compiler does not implement the same path.
Slurm job 16096 reran the updated positive (native exit 0 with pass marker) and
all three private-access negatives (exit 1, E175) using the same protected
current-source compiler. This changes test contracts, not compiler source.
The remote CI suite remains a separate pending gate.
