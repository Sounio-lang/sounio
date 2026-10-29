# Compiler SIG139 census and implicit aggregate-return loss

Status: read-only diagnosis with immutable source snapshots and private C++23
ptrace tools. No compiler, lowerer, frontend, proof, mathematical formula,
assertion or test deadline is changed. The eleven historical compiler crashes
remain failures in RESULT19392.

## Census and reproduction

RESULT19392 records733 files, with eleven compiler139 cases:
five chemistry, one graph, two NN and three tensor. All eleven reproduce SIGSEGV
with raw03 e0279ff62e97e861415cd8b105b854ae645aa61865b33b1e2b91184d2c28bf54
and the byte-identical original19392 SDK, using a new snapshot for each job and
the original1200-second compile limit. No produced scientific ELF is executed.

The original two chemical captures in raw11332/raw8abe do not constitute raw03
evidence. Raw03 is independently observed here. The initial selector for saved
reporter fields belonged only to raw8abe RIP5884ade; it was not reused by address
analogy to classify raw03.

| Family | Original tests | Raw03 fault | Common stack |
| --- | --- | --- | --- |
| chemistry | chem_network_ode_sys5; kin_sigma_small_variance; kinetics_deep_stdlib; kinetics_epistemic_ensemble; kinetics_gri_mech | SIG11 at588d598, address -1 | reporter588d4be ← fresh_label59b0967 ← lower_match_arm_ref5a65f18 |
| graph/NN/tensor | epistemic_curvature_e2e; nn_primitives_d3; pinn_full_integration_d6; tensor_ops_d1; tensor_autograd_d2; tensor_autograd_d2_hardening | SIG11 at59e51d2, address7fff | user_variant_index_ref59e5163 ← lower_match_arm_ref5a65f18 |

The second family dereferences the invalid enum-variant table pointer at
self+0x68, then its count at+0x8000. The identity is tied to the actual source
shape and caller/disassembly, not merely an address fingerprint.

Jobs19677 and19681 cover the four initial discriminators and the seven siblings.
19677 recorded all four compiler139 outcomes and all source/tool/raw final guards,
then its worker exited127 solely because readelf is absent on the worker.
19681 omits that unavailable read-only tool and completed0. This infrastructure
status is not presented as a compiler result.

## Causal state boundary before either fault

Private tracer19704 uses validated x86 software breakpoints at:
- lower_match_arms_ref entry5a656c7;
- lower_match_arm_ref entry5a65f18;
- its caller return5a65871;
- the common epilogue immediately before5a74009.

All addresses are from raw03 call/frame targets and disassembly. Entry byte
guards verify the prologues/instructions before inserting a breakpoint.
The compiler raw file is never modified; ptrace changes only the private child
process instruction bytes and restores/steps them. Signal and /bin/true controls
qualify the underlying tracer. Catalysis still compiles0 with instrumentation;
the two representative failures retain the same SIG11/RIP as the general tracer.

Both representative cases reach the FIRST match arm with a valid Lowerer:
chemistry current_fn365, graph current_fn15, valid module/current-function/
enum-table pointers and loaded flag1. Just before the common function epilogue,
RAX still points at a valid Lowerer with those fields. R12 points at the caller's
2320-byte SRet destination, which has not received that value.

The common epilogue executes xor eax,eax and returns R12 without a0x122-word
copy. The returned pointer is the correct SRet address but its contents are
unwritten/invalid. The caller's old lo remains valid at that instant; its assignment
copies the bad packet, and only then does the SECOND match arm enter with-1
fields. A wildcard arm reaches fresh_label/reporter9036; an enum arm reaches the
enum lookup. The two crash sites are downstream symptoms of the same missing
implicit-return copy.

Explicit early-return epilogues in this SAME machine function do include
rep movsq with0x122 words. A hardware-watchpoint experiment19695 produced no
write events; it is retained but is not used as independent proof of no writes.
The software boundary capture and full epilogue disassembly establish the fact.

## Minimal producer discriminator

The first18 Packet fixtures (three fields and a280-i64 payload variant; pure
implicit if, conditionally early-returning implicit if, and explicit return-if;
three selected branches) pass in BOTH seed9cf and raw03. This excludes a blanket
claim that all implicit struct returns fail.

The second18-case matrix changes the mixed implicit branch to end
unconditionally in return. Modern raw03 checks/compiles/runs18/18 correctly.
Seed9cf compiles18/18 but four cases fail their unchanged field assertions:
small/array implicit branches0 and2 print0,0,0 instead of11,22,33. The executed
early-return branch1 and all plain/explicit controls pass. Job19720 completed0
as a diagnostic collector, not as a successful seed correctness gate.

This isolates the defect to the legacy producer used to build the compiler
artifact, rather than demonstrating a matching bug in current modern IR lowering.

## Source decision and scope of attribution

The consulted lean_single.sio SHA is
e5c82b67fd14cc43ff95e0f3410970ba6c70f2cc1f0cfb290f8266ed5497704a,
identical in raw03's source snapshot and the seed capsule's source tree.
The code path is a token/direct-native producer; no producer IR artifact has
been obtained. Machine/ABI evidence is kept separate from modern IR traces.

compile_stmt resets LAST_STMT_HAS_VALUE at entry. Its if join (~32486) retains
a value only if BOTH branches report HAS_VALUE1 and nonUnit types. Match joins
similarly require all arms to report values. Explicit return does not provide a
separate no-fallthrough fact. Thus a branch that returns from the function is
merged as if it were a reachable Unit branch.

compile_all implicit epilogue (~38462) emits xor eax,eax when HAS_VALUE0 and
skips stabilize_return_agg_x86 even though CURRENT_SRET_SLOT is positive. That
matches the observed common machine epilogue and the failing seed fixture.
The current modern lowerer fn_ends_with_return/body result path is not blamed
without further evidence; the modern compiler executes the minimal cases correctly.

## Proposed shared fix, not implemented

No existing block/statement divergence flag was found alongside HAS_VALUE/TY/HASH.
Do not replace AND with OR: a reachable Unit branch plus a reachable value branch
must remain a type error.

Add explicit control-flow outcome metadata, independent of value/type:
FALLTHROUGH, RETURN, BREAK, CONTINUE and ABORT (a small outcome bitset).
Sequential composition adds the next statement's outcomes only on prior FALL
paths; unreachable later statements must not erase an earlier transfer.
Snapshot/reset/restore this state across nested blocks, function bodies and
closures, including expression-bodied paths.

At if/if-let/match joins, type/value/aggregate and f64/FO shadow metadata come
only from branches reaching that join. A returning arm contributes no Unit.
Both returning arms have no fallthrough value. A reachable Unit arm is still
Unit and cannot borrow a value from the other arm. Match exhaustiveness must be
established before eliminating unmatched fallthrough. Unknown calls must remain
fallthrough; panic/abort may be marked only at their resolved intrinsic emitter,
never by bare name. Loops consume break/continue conservatively and may execute
zero times; no optimistic divergence inference from an arbitrary loop.

Preserve variance/gradient/f128 metadata from the live arm. Do not infer control
flow from the last emitted byte: existing nop-at-merge defenses already document
why that heuristic is insufficient. Do not repair each lowerer method by rewriting
its return as explicit, and do not patch the cached9cf ELF in place.

Required before/after matrix for a new producer: current Packet cases, f64 bitwise
returns with one returning arm, nested ifs and exhaustive matches, both-returning
arms, reachable Unit/value rejection, Unit/Unit acceptance, and return/break/
continue/resolved-panic boundaries. Include small and280-word aggregates,
closure/nesting restoration, source/type mismatch controls and actual original
chemical/graph/NN/tensor compilation after a qualified producer rebuild.
No implementation or rebuilt producer is claimed by this diagnosis.

## Independent oracle and custody

C++23 job19723 completed0 in1second. Its exact field oracle accepts all modern
Packet outputs and all14 correct seed outputs; it independently refuses the
four wrong seed outputs with4. Wrong-marker, duplicate-stream and wrong-value
controls are refused with3/5/4. This is a small ABI/return diagnostic, not a
scientific benchmark or a replacement mathematical model.

Custody job19722 completed0 in16seconds, with archive readback and tar validation:
 /orangefs/training/sounio-foundry/runs/stdlib-wave3-20261027/core/sig139-census01/source-raw-seed-tracers-matrices-receipts.tar.gz
 SHA25620c5dfc00234a5fcac5f6ea65df8994fb290b622dd1ebdcf7df299a171bf8552.
It contains the original SDK once, raw03, immutable seed9cf, all tracer sources
and binaries, all matrix sources/native outputs and all complete manifests.
Compact Git receipts summarize the12587-entry source guards; the full guard
files remain in that archive and the original job directories.

The reachability design is refined in FLOW_DESIGN.md. It is not an implemented
compiler patch. Scientific-loop must-def19724 is separately documented in
WIDE_MUSTDEF.md and is not attributed to this producer bug.
