# Native stack-reference closure and Unit-region elision

Follow-up to compiler commit `66797ec4e482e0197d9843ca19a3a63eea16ceb6`
(root adoption `800d4f185c`). Production native issuance remains closed.
No codegen, runtime thunk, lowerer, public API or capacity table changes here.

The native component now includes an i64 entry that creates a scalar stack
reference and passes it to a two-argument helper. The helper writes twice the
input through that reference. The original entry opens/closes a real compiler
region; an ordinary serial wrapper calls the same original helper without
opening a region itself. Clones have separate function identities and retain
the reference/store instructions while removing proven region operations and
omitting their automatic StorePtr barriers.

The first adversarial fixture consumed the incidental RAX result of
`__rgn_exit_scalar` as if it returned its scalar argument. That operation has
an ignored Unit result. The previous proof accepted this IR and its rewrite
invented a scalar return, which the concurrent oracle exposed as exactly
25,000 incorrect serial-entry results while all 150,000 clone results and the
final runtime context were correct. This is a real missing elision precondition
for mutated IR, not evidence that generated source consumes a Unit result.
The failed before artifacts and diagnostic are retained in
`worker-stack-components/BEFORE_*` (Slurm 19304/19308). Slurm 19302 failed the
C++ strict warning gate before running the oracle; it is not execution evidence.

The proof now requires every exit/leave/keep destination to have no reads
anywhere in the complete function, including pooled call arguments. A result
may be absent or present but unused. The clone replaces these Unit operations
with Nop and never fabricates a return value. Enter still produces only a
neutralized token. An observed Unit result fails closure reason 140 before
ELF creation. This intentionally conservative def-use check does not infer
that an earlier read or unreachable read is harmless.

**Slurm 19312 completed 0 in 55 seconds** with the exact production proof SHA256
`80bbd7a34d0e7d6b7465135335987b30d86d261408ccc9fbff59d29e797dc73a`:

* Modern 8-module source typecheck: verdict 0. Legacy producer compile: no
  explicit error/dropped-store diagnostics, nonempty ELF.
* Eleven original machine/ABI checks pass with every writable ELF data/BSS
  segment PROT_NONE. The new i64 clone actually executes the two-argument
  stack-reference helper and its StorePtr.
* Fourteen signal refusals pass, including the original serial wrapper's
  automatic barrier accessing protected runtime data, the original entry's
  explicit region access, and the helper clone being refused by the entry
  allowlist. The previous wrong-word, wrong-ABI, interior-address and original
  address controls remain in place. Three deliberately wrong oracle controls
  are also rejected.
* In a separate phase, writable runtime data is enabled and a context is
  installed using field offsets emitted by the canonical native
  `runtime_context` helpers. One serial thread alternates original entry and
  ordinary serial-wrapper calls (50,000 total) alongside three clone threads
  (150,000 calls: i64 mutation, f64 exact-word identity including NaN/-0, and
  f64 half). All four mismatch counters are zero and the entire final context
  and context-slot pointer equal their initial values. This phase uses a
  writable context; the earlier protected-memory phase proves absence of
  clone runtime-state access.
* The same checks pass again when the Unit operation has a destination that
  is never read. A mutant returning that destination instead is refused with
  `closure=140` and no ELF.
* All source and seed hash guards pass. The before artifacts are preserved;
  the canonical runtime thunk is unchanged.

The test body, driver and C++23 oracle are the three `native_stack_*` fixtures
under `tests/fixtures/wave4_worker_capability/components/`. Append the body only
to a fresh compiler-source snapshot's `worker_proof.sio`, copy the driver to a
`.sio` file in that snapshot, and use the legacy seed's `INPUT OUTPUT.elf`
invocation. The producer takes `OUTPUT.elf MODE`, with modes `good`,
`unused_region_result` and `observed_region_result`. Compile the C++ oracle
with `-std=c++23 -O2 -fno-omit-frame-pointer -Wall -Wextra -Werror -pedantic
-pthread` and pass `OUTPUT.elf EMITTER_LOG`; keep `runtime_thunk.inc` beside it.
The complete captured Slurm recipe and immutable source are preserved at
`/scratch/sounio-stdlib-wave3-20261027/compiler-resumed/worker-stack-native-v4/`
and the adjacent `worker-stack-native-v4-worker.sh`.

This is still an IR/native component with injected checked-source metadata.
It closes the native stack-reference/barrier and mixed serial/clone gap for
this concrete component. Imported/BFS admission, the real canonical normal
kernels, real service pools/threads, cache reproof and whole-stdlib acceptance
remain separate gates. The prior `worker-certified-components` hashes record
the earlier 66797ec4 checkpoint; `worker-stack-components/QUALIFIED_SOURCE.sha256`
records this follow-up's proof and fixtures.
