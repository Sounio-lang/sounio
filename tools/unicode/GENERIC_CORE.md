# Canonical generic Option and Result

The public `core::option::Option<T>` and `core::result::Result<T,E>`
own their payloads. A concrete instantiation is linear when its payload is
linear; copy-only instantiations remain reusable even when linear instances
exist in the same program. Unit is a valid payload.

The existing IntOption/FloatOption and IntResult/FloatResult constructors
and accessors remain unchanged. Generic source requires the qualified generic
enum/import compiler stack; the old committed compiler cannot parse the
declarations. This is a toolchain prerequisite, not a claim that the shipped
compiler has been replaced.

Slurm18716 completed0 in4 seconds on the actual imported public modules.
The unchanged positive fixture checks legacy helpers, independent i64/f64
Option instances, Result success/error, a linear ByteBuffer nested inside
Option<Result<...>>, a linear error payload, Unit success and subsequent reuse
of copy-only instances. Exact bytes65,0,255 survived owner transfer and were
closed once. The older a2c compiler failed the same source before qualification;
its original parse errors and nonzero status are retained.

Qualified raw compiler:
e9ad6cde0328e1dbf249da726fb44a798b0dd575236d5ec4040cd309f558b03e.
Executed positive ELF:
bc08bf9ef90a7a679bbc45161421e215ec9d467dfa375fe8dcb2aca064eeb06e.
Platform ABI112:
de01597868c09725c2c26651d23203043c65fdc04a3e5e96f45dfe4a51d494c4.
Source, compiler, runtime, input archives and ELF passed before/after identity
guards in the positive job. The snapshot uses the exact already-qualified
byte_buffer_budget dependency adopted in the root integration worktree;
the core worktree did not edit the services-owned file.

Slurm18721 completed0 in3 seconds. Both check and compile rejected duplicate
ownership with E039, unconsumed ownership with E040 and a wrong payload with
E009; none produced an ELF. The three retained probes import the public Option
rather than redeclaring a local substitute.

The unchanged returned-borrow positive probe still fails on this compiler.
That diagnostic is recorded separately and is not counted as a passing
negative. Owner-bound Option/Result/tuple/struct borrow propagation, mutable
borrow exclusivity, generic HeapVec typed storage and final integrated
whole-stdlib acceptance remain open dependencies.

## Durable custody for this tranche

Slurm18730 completed0 and archived the complete closed metadata/generic tranche,
including failed attempts and diagnostic crashes, at
`/orangefs/training/sounio-foundry/runs/stdlib-wave3-20261027/core/closed-export07/core-closed-export07.tar.gz`.
Archive SHA256:
dad4c1f94a342525f7ba199baa9abe3332fd12ec812fe62e6d0c63e3719a2afe.
The export retains a complete file-hash list and verifies the original closed
directories after archiving. No earlier failure receipt was deleted or relabeled.
