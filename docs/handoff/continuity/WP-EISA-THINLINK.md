<!-- docs:meta
topic_id: repo.docs.handoff.continuity.wp-eisa-thinlink
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.handoff.continuity.wp-eisa-thinlink
-->

# WP-EISA-THINLINK — Madaros multi-module native ELF thin-link write failure (rc=13) [Opus]

## Status of the EISA default-lane track (2026-07-07, post phase-2)
The checker gaps are CLOSED. `test_eisa_isa` / `test_eisa_evm` now **type-check fully clean** and
**lower completely** on the default Madaros engine (all imported-lane E004/E012/E137 cleared by
A7+A9; the transitive cross-module call drop cleared by A14). Verified on a fresh main build with the
EISA dep closure present and adequate vmem:
```
imported_compile: typecheck ok
lower_array: dep_begin 1..4 ; lower_done ; Merged IR: 142 functions
Error: Failed to write native binary to /tmp/isa.elf rc=13
error: multimodule native thin-link compilation failed
```
So the ONLY remaining EISA runtime blocker is the **final native ELF-emit / thin-link stage** returning
**rc=13** for a large (142-fn) merged multi-module program.

## Environment (do this first — a prior agent got a false E137/E004 from a bad env)
`stdlib/eisa`, `stdlib/math/dd64.sio`, `stdlib/math/qd128.sio` are NOT on main — they live in
`/workspace/sounio-eisa`. Copy them into your worktree's stdlib (keep OUT of git; compiler/verification
only). Do NOT copy the eisa worktree's OTHER stdlib files (str::lib etc.) over main's — that reintroduces
the A7 int-width skew (E004) and is a verification artifact, not a real failure.
Build with a big-RAM path: the compile reserves ~24 GB VIRTUAL address space (lean_single Box arena
reserves 8 GB chunks, never frees; RSS ~1.5 GB). Use the RAW binary with `ulimit -v unlimited`:
`SOUNIO_STDLIB_PATH=<repo>/stdlib "<madaros.elf>" tests/stdlib/eisa/test_eisa_isa.sio -o /tmp/isa.elf`
(the `bin/madaros` wrapper caps vmem at 48 GB post-A5, which is enough — but the raw path is simplest for
iteration). All builds go to Slurm (see [[reference_slurm_madaros_offload]]); the 515 GB r740 node is
intermittently offline, so pin modest `--mem` and retry on scheduling errors.

## Where to look
The failure is AFTER `imported_compile: lower_done` — in the native ELF write / thin-link path, NOT the
checker or IR lowering. Candidates: `self-hosted/native/elf.sio`, `elf_bulk.sio`, `reloc.sio`, and the
`module_native_driver` / thin-link emit in `self-hosted/compiler/module_frontend.sio` (the same file that
prints `Native compilation failed: imported_simple_ir_emit_failed` on the compact-IR path before falling
back to the full IR path — trace both paths). `rc=13` is the emit function's return code; find where it
originates (grep for the `Failed to write native binary` string and the rc plumbing). Likely a size/offset
or relocation issue that only trips at 142-fn scale, or an unhandled emit case for a specific EISA construct.

## Constraint
`module_frontend.sio` and the native emit files are in `main.sio`'s import closure; the lean_single SRC
import buffer is **8 MB** and the closure sits at the cliff. Keep edits BYTE-FRUGAL and confirm the madaros
build still succeeds (BUILD_RC=0) as your first recipe step (a prior agent overflowed it with +977 comment
bytes; A11's -113B and A14's +231B both stayed safe).

## Witnesses
- W1: `test_eisa_isa` (with deps + raw binary + `ulimit -v unlimited`) => BUILD rc=0, ELF produced, RUN =>
  the test's expected output (see its header/asserts), no SIGILL/SIGSEGV.
- W2: `test_eisa_evm` => same.
- W3: minimal repro — a synthetic multi-module program that merges to a comparable fn count and hits the
  same `Failed to write native binary rc=13`, to bisect the emit path fast.
- W4 (regression): `cd_exact_generic_i64` still green (`ZD PROVED`… rc=0); madaros BUILD_RC=0; 8-10
  multi-module run-pass tests byte-identical.
- Then B2 (gate refresh): conformance 21/21 + 13-test default-lane suite (see WP-B2), once the emit works.

## Done criteria
`test_eisa_isa`/`test_eisa_evm` build + run PASS on the default lane; no new umbrella reds; PR merged.

## Related residual gaps (not this WP)
- Octonion multi-module native-lowering / memory wall: `algebra_g2_invariants_import` (uses
  `algebra::octonion`) rc=1/139; pre-existing, exposed (not caused) by A14.
- lean_single cannot build 4-module generic closures (`unresolved cd_zero_exact`) — a known lean_single
  limitation; cross-engine parity for cd_exact is via the proven output + `cd_exact_generic_vs_concrete`
  (now real BYTECOMPARE PASS after A14), not a line-diff of a lean_single 4-module build.
