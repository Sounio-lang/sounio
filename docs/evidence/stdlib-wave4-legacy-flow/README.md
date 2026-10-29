# Legacy x86 producer: join reachability component (19753)

Status: **component-qualified candidate; not a production compiler promotion**.
The existing raw compilers, seed and all scientific regressions remain intact.
The nominal enum/Bool extension and a fresh modern compiler replay of the eleven
original SIG139 regressions are still required before promotion.

The preceding evidence is in ../stdlib-wave4-integrated/ (and the original
wave4_sig139_census fixtures). Jobs19704/19720 established that the legacy
producer omitted the SRet copy when an implicit aggregate return joined a live
value arm with an arm that explicitly returned. The modern raw03 route passed
the same minimal Packet matrix. The source repair here is confined to the x86
legacy producer; lower.sio was not edited.

## Semantics

LAST_STMT_CAN_FALLTHROUGH_X86 is independent of LAST_STMT_HAS_VALUE.
Sequence preserves termination even when later unreachable statements are parsed.
Explicit return, valid loop-local break/continue, and the already-resolved panic
intrinsic do not reach their immediate join. Short-circuit RHS termination does
not eliminate the bypass path. Loops conservatively may conclude or take zero
iterations; break/continue remain local.

Both ordinary functions and closures decide whether to emit their implicit
epilogue using reachability, **not the last machine-code byte**. A reachable
non-Unit tail without a value is rejected. Mixed Unit/value branches have no join
value, so statements may discard them while expression consumers reject them.
The previous defensive nops remain unchanged; no new nop is the semantic fix.

Only live arms participate in value/type joins. if-let/match scalar joins own
runtime shadow slots, preventing a live value from referring to a slot initialized
only in an arm that returned. The component tests binary64 values and GUM
variance; storing all shadow fields is not a claim that every higher derivative
shape has been numerically qualified.

## Exact qualification

Slurm19753 COMPLETED0, 76s, 1CPU/8GiB. The versioned runner is
scripts/compiler/legacy_flow_gate.sh, and requires a Slurm allocation.

- Baseline:68cases,24 failures detected by the independent C++23 oracle.
- Candidate produced by immutable seed9cf:68/68 accepted/rejected as specified.
- Candidate privately recompiles its own source; generation2 repeats68/68.
- Wrong-value and duplicated-output controls return4; missing-output returns3.
- 148 AArch64 function source chunks are byte-identical. No AArch64 behavioral
  qualification is claimed.
- The source, SDK and seed hashes are checked before and after.
- All native fixtures use a5s limit; all fixture compilations use30s. No long
  scientific native test was rerun.

The baseline rebuilt byte-identically to seed9cf:
9cfb6a970d73718b09091331dc514d015e3d4e6f402ee756cdc7f7b0cc55110a.
Candidate source:
20e6dd74865558f82f6a3ef3ee8189d8a7d44ebb0f42779efb97c0d904f3d1fb.
Generation1 ELF:
6b7a74349a8aa1bea2ea5fdbcd3aba78a583586b5a70dd798d84059227d2cbd4.
Generation2 ELF:
f5a0be0ca7583d7f3efa7f15e2d07d4329e56ce3b6eca757b7a0ac30a4f569d3.

68cases cover three-field and280-word Packet returns, nested/early returns,
both-return paths with unreachable trailing statements, closures, explicit
return-if whose argument itself returns, if-let/match, loop-local break/continue,
Unit/value consumption negatives, incomplete/duplicated/guarded match coverage,
and six exact scalar-shadow cases. For the latter C++23 verifies the bit patterns
of2.5 and variance1.0 from the selected2*(1.25±0.5) computation.

19744 is preserved as an operational permission failure after build0; no fixtures
ran there.19747 is preserved as the intermediate failed component, including its
five native failures and rejected selfcompile.19749/19751/19752 are intermediate
qualifications;19753 is the sealed component, including the reachability epilogue
revision requested by root.

Full inputs/SDK, producers, fixture ELFs and all outputs are retained at:
 /scratch/sounio-stdlib-wave3-20261027/compiler-resumed/legacy-flow-final01

The OrangeFS custody archive and its SHA are in19753/compact/CUSTODY.sha256;
custody job19754 completed0. Compact evidence preserves the original outputs,
including trailing whitespace from tools.

## Remaining boundaries

The component's earlier two-tag test uses legacy type metadata and does not
establish general nominal identity. In particular hash0 is not a proof of
Option/Result identity. Nominal enum coverage and Bool handling must be completed
with source identity, structural validation and explicit fail-closed budgets.
Do not infer exhaustiveness from arm-count.

No fresh modern compiler was built with this producer, and none of the eleven
original compiler crashes has yet been replayed with such a compiler. The WIDE
undefined-register diagnostic19724 is separate; its source origin remains open.
There was no push, main merge, shared binary replacement or lower.sio edit.
