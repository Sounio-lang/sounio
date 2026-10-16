<!-- docs:meta
topic_id: repo.docs.compiler.madaros-macos-corpus
authority: repo_only
audience: contributors
last_validated: unvalidated
validated_by: none
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.compiler.madaros-macos-corpus
-->

# Native macOS Madaros: pinned 200-program corpus

The immutable selection is `tests/manifests/madaros_macos_200.txt`, recovered
from the original `scratchpad/list200.txt`. It is not a new sample or a filtered
subset. The manifest pins membership; source snapshots record the exact fixture
revision. One explicitly approved semantic correction is disclosed below.
Manifest SHA-256:
`403761e4e3581328c8ea8193aad0ebfea38953f04d015269fa11ece21fb269d6`.

Build the C++23 execution/counting harness with a C++23-capable compiler:

```sh
/usr/bin/clang++ -std=c++23 -O2 -pthread scripts/ci/madaros_macos_corpus.cpp -o /tmp/madaros-corpus
SOUNIO_STDLIB_PATH="$PWD/stdlib" /tmp/madaros-corpus \
  artifacts/self-hosted/madaros-arm64-macos \
  tests/manifests/madaros_macos_200.txt /tmp/madaros-corpus-results 4 180
```

The harness calls the specified native Madaros directly, passes the explicit
`aarch64-macos` target, requires a Mach-O arm64 output, signs and verifies it,
then executes with a 30-second limit. Build limit is the final argument in
seconds. The oracle follows the repository's canonical test harness:
each `expect-stdout` or `expect-stdout-contains` must occur in stdout, and the
program must exit successfully (or with its declared expected exit code).
Unrecognized `expect-*` annotations fail closed. No engine fallback or test
skipping is used. Compiler/manifest hashes, per-test logs, exit statuses,
expected stdout, result TSV, summary and initial failure causes are retained.

`CORPUS_REPLAY=1` re-evaluates existing complete execution receipts without
claiming a new execution. It refuses compiler/manifest identity changes. This
is useful when correcting harness interpretation, not to validate a new compiler.
Replay also refuses changed fixture source, before opening any result receipt.
Fresh runs refuse to overwrite an existing result directory.

## Main-integration regression, 2026-10-15

### Post-merge self-check capacity and scalar initialization follow-up

The port merged as `f5b2c5850be9f3d3995719ffcade44a6d851658a`. Its post-merge
CI `38796025054` independently passed the native macOS/Metal 200-member gate,
but failed the Linux self-compile ladder at `check`: `E-USE-MOD-CAP`. The
module-scoped import registry exceeded 1024 entries. A follow-up aligns the
checker, specializer and lowerer at 4096 entries, retaining module identity,
full-name collision checks and hard overflow refusal.

`scripts/ci/module_registry_capacity_gate.sh COMPILER` generates a fixture
from the production registry functions and executes it. The C++23 generator
is not an alternate registry implementation. Only the extracted hash's effect
annotation is widened for the bootstrap seed's local-mutation rules; its body
is unchanged. The probe exercises hash collisions, module isolation,
duplicates at capacity, the 4096/4097 refusal edge, and scalar boolean globals.
The previous source fails the capacity probe with rc 5. The candidate passes.

The same probe exposed an ARM64 startup defect: a scalar boolean's one-byte
logical size divided by a word stride produced zero iterations in a
post-tested BSS fill loop, which then wrote beyond `__DATA` (Bus error).
The follow-up emits one word store for the eight-byte-padded scalar slot,
as on x86, and the candidate executes the previously crashing probe.

Slurm job 14482 on `cpuops-t560-proxmox` rebuilt the Linux candidate, executed
the registry probe and checked the full compiler entry point, all with rc 0.
Linux candidate SHA-256:
`6831d6b09b3f99716ec21854dbabc1b44580bd74e2390ea43166d7ad741686f4`.
The independently rebuilt and signed native macOS candidate passed ABI
acceptance, extended positive/negative regressions and a fresh **200/200**
with mandatory real Metal. Its C++23 binary64 oracle checked 40960 comparisons
with zero failures. Signed host SHA-256:
`58c1cd35986f149e5abc45958dffb939e1779e02860e47c272729a5b48a16b10`.
All binaries and generated receipts remain excluded.

`fixed_point_named_diagnostic_gate.sh` also proves that a named checker error
is counted rather than reported as zero: the injected failure remains rc 1
and rung `none` both before and after the reporting fix. These focused and
corpus passes do **not** establish recovery of gen2/run, a modular Madaros
fixed point, or green remote CI for the follow-up.

### Combined FO-inline dependency validation, 2026-10-16

The locally integrated sources from PRs #2980 and #2981 were rebuilt into a
fresh signed Mach-O arm64 host, SHA-256
`93e611061c3f81fff935b7e3f00aae6108823d19fd6de443ab78b07efd5403cc`.
This candidate passed native ABI acceptance, the production registry probe,
extended port regressions, all three FO-inline/loop fixtures from #2980, and
the unresolved-identifier negative control (E137, nonzero build status and no
published executable). A fresh corrected pinned corpus passed **200/200**
with required real Metal launches. The binary64 C++23 oracle recorded
`checked=40960 failures=0` on Apple M5 Max.

This is a new combined-source execution, separate from the original 199/200,
the historical corrected 200/200, the PBPK queue failure at 199/200, and the
registry-only follow-up above. Local logs and generated receipts are retained
outside Git under `/private/tmp/pr2981-combined-*`. Slurm job 14547 is the
independent Linux rebuild and ratcheted ladder for the combination. Its
ladder failed at `check` with rc 1 and 292 E137 diagnostics, before gen2.
Declared locals such as `p42` in `native_v2_write_contract` were reported as
undeclared. A separate generated pressure fixture also reproduces lost local
bindings when many globals fill the shared 4096-entry type environment; this
capacity path silently drops insertions and requires a fail-closed repair.
The combined native self-check reproduces missing-local diagnostics too;
the 200-member corpus does not cover this self-check capacity boundary.
Its source-built Linux host (SHA-256
`c1a652b67f2746b91761ef191421577e2406e5a03d827f1dd28083a534bb0d31`)
was built with rc 0, executed the three dependency fixtures with rc 0, and
refused the unresolved identifier with E137 and rc 1. Neither the native corpus nor PR-only
CI establishes a modular fixed point or a green post-merge main revision.

The subsequent type-environment candidate uses 8192 binding slots consistently
and carries sticky overflow state through scope exits. Collection and both
checker entry paths convert it to `E-TYPE-ENV-CAP`, nonzero status and refusal
to publish. The old host accepted 8300 globals and an empty `main` with check
rc 0 despite dropping bindings; the repaired native host refuses that input.
The old 4090-global/local pressure case fails with E137, while the repaired
host checks it successfully. The new production-extracted capacity probe
executes the actual `TypeEnv.bind()` consumer path, including exact capacity,
scope shadowing/restoration, mutability, sticky overflow and fresh reset.
The private `type_env_bind_mut` helper has no production callers and is not
claimed as covered by that executable probe.

This native candidate is signed SHA-256
`167ea6daca04a0ded6a966d63a097c9697ca073cf9fd8b1b8ecf3a298aefa3d0`.
Its capacity/pressure/refusal gate and full compiler self-check passed
(`verdict=0`, `check: OK`). Extended port regressions and all three dependency
fixtures also passed, followed by a new **200/200** corpus with mandatory real
Metal and C++23 binary64 conformance (`checked=40960 failures=0`). This is not
a replay of the earlier combined-source run.

Slurm job 14566's rebuilt Linux host is SHA-256
`bb1711748557bb2013827c9504a221e223b79e644cb1f4041d2ffca3049d4ce5`.
Its compiler self-check passed with rc 0 and zero errors, recovering the
292-E137 failure above. Its gen2 ladder subsequently stopped after 14 modules
with rc 1: `spec_literal_scalar_type` referenced the nonexistent variant
`ExprKind::ExprBoolLit`. No gen2 executable was published.
An independent probe step initially used Linux's 8192 KiB shell stack and
returned 139. Its repeat uses the existing `souc_invoke.sh` raw-compiler
limits contract; both traces are retained separately. Capacity gates use
that same helper, consistent with the public wrapper and self-compile ladder.
With that contract, the updated Linux production probe checks and executes
with rc 0; the pressure case checks with rc 0; overflow check and build both
return rc 1 with `E-TYPE-ENV-CAP`, and no executable is published. Its host
and both repaired source-file hashes match the ladder candidate exactly.

The AST defines `ExprBoolTrue` and `ExprBoolFalse`. The specializer and closure
peer-inference checker each contained the stale `ExprBoolLit` spelling; the
next candidate handles both real variants. The new generic boolean-literal
fixture fails against the preceding host (E009/E005) and is included in native
port regressions. A C++23 surface preflight rejects both old references and
passes after the repair. It audits explicit consumers importing `parser::ast`,
not unrelated prototype enums also named `ExprKind`.

The rebuilt, signed macOS host is SHA-256
`c31722bcca7b9850712a1f63328bdf6ba677684c86ad24ae9f6001ab90ad74e5`.
It passes native ABI acceptance, the capacity/refusal gates, full compiler
self-check (`verdict=0`, `check: OK`), extended regressions and the three
dependency fixtures. The new boolean fixture compiles, is signed, and executes
with exit 0 and `GENERIC_BOOL_LITERAL_OK`, discriminating the preceding
E009/E005 failure. A fresh corpus for this host passes **200/200**, including
mandatory real Metal launches; binary64 conformance on Apple M5 Max reports
`checked=40960 failures=0` with the C++23 oracle.

Slurm job 14578 rebuilt the repaired Linux host, SHA-256
`34aff9e546860749c7256b02fbe5e58d9b083462a133c228d32d79826ee09c1b`.
The boolean and three dependency fixtures, production capacity and registry
probes, pressure check and absence-of-overflow-artifact test all return 0;
overflow check/build refuse with rc 1 as required. Compiler self-check passes
(rc 0, zero errors). Gen2 completes 23 modules before rc 137; Slurm explicitly
records an OOM kill in step 14578.0, with peak RSS 33471672 KiB under the 32 GiB
allocation. This is separate from the earlier semantic failures. A 64 GiB
ladder repeat reuses the identical verified host and source without rebuilding
or changing the required rung, minimum module count or IR limits. There is
still no proof of rung `run`, a modular fixed point, or green post-merge main.

The 64 GiB repeat (Slurm 14585) completes lowering of 129 modules and reports
14733 merged functions, but refuses emission with rc 1: the instruction arena
holds 4194292 of 4194304 slots, with live capacity 2215828 and 1978464 slots
abandoned by growth. Peak RSS is 62642236 KiB; this run has no OOM report.
The fixed arena limit and fail-closed emission check are retained. The next
candidate extends a region in place only when its end equals the arena top;
non-tail growth still copies, and the checked reserve API still advances its
generation so stale handles remain refused.

A C++23 generator extracts the production growth, reserve, handle-validation
and scalar-copy functions unchanged. Its native-seed-compiled discriminator
returns 2 on the preceding allocator and prints `IR_TAIL_GROWTH_OK` with rc 0
on the candidate. It exercises tail growth, non-tail relocation with neighbor
and payload preservation, stale/sealed refusal, exact-capacity extension and
overflow refusal without mutation. The candidate module typechecks. This is
not yet a new-host result: the preceding modular ARM64 host refuses the probe
at `ir_region_status_v`, instruction 4, and that trace remains an open
validation gap. New modular hosts, the Linux ladder and a fresh Metal corpus
are being rebuilt/validated before publication.

Tracing the refusal identifies `IrLoadGlobal` at offset 571090960: ARM64 data
address emission and global initialization each imposed a 128 MiB ceiling,
despite the larger mapped Mach-O data window. The next backend candidate
derives its capacity from the existing data VM address and dyld shared-region
boundary, reserving import/linkedit pages, and explicitly checks ADRP's signed
page reach. The final Mach-O overlap refusal remains unchanged. No IR arena
capacity is increased. A production-function address probe compares emitted
ADRP/ADD bytes with C++23 expected encodings, including offsets above 128 MiB,
the largest permitted access, and refusal immediately outside the window.
The preceding source returns 57; the repaired functions execute with rc 0 and
`ARM64_DATA_ADDRESS_OK`, compiled by the preceding modular host. The new
`ir_tail_growth_gate.sh` requires both production probes to compile and execute
through the selected modular host, and is wired into Linux and macOS CI.
Backend module check and workflow-reference checks pass. New-host execution
of the combined allocator/address fix and another fresh Metal corpus are
still pending; the earlier tail-only host's probe refusal is not a success.

The rebuilt allocator/address candidate is a signed native host with SHA-256
`c4151f7eadd98328135c15ce42ab48f9a3d2a15482cc289dc195ad05b0930515`.
It passes ABI acceptance and the new gate: the selected Madaros compiles both
production probes to Mach-O, signs and executes them, yielding
`IR_TAIL_GROWTH_OK` and `ARM64_DATA_ADDRESS_OK`, each with exit 0. TypeEnv
capacity/refusal and full compiler self-check also pass (`verdict=0`,
`check: OK`). Large-frame warnings, including the existing enormous nested
HLIR layout estimates, are also present in the preceding boolean-host log;
they are not claimed fixed here. Fresh regressions/corpus and the Linux
ladder remained pending at that observation. The same candidate subsequently
passes the full extended regressions and a fresh **200/200** corpus with
required real Metal launches. Binary64 conformance on Apple M5 Max reports
`checked=40960 failures=0`, using the C++23 oracle. The native validation
finishes with `PASS: production IR growth/address gate, native self-check,
regressions and fresh real Metal`. This fresh corpus belongs to the
allocator/address candidate, not to either preceding host.
Slurm 14593 is now running, not
merely queued; its snapshot predates the ARM64 address-window edit and must
not be described as identical to this macOS source snapshot.

Slurm 14593 subsequently stops with rc 137 after lowering all 129 modules:
Slurm explicitly reports an OOM kill in step 14593.0, with peak RSS 66962948
KiB under the 64 GiB allocation. It emits no ELF. This does not prove the
tail allocator fix alone closes the full-build memory wall.

The dependency-reclaim path still used pre-scalar-pool reboxing: it copied
argument registers to a 32768-site, eight-argument scratch, then rebuilt Box
lists and appended them to the already persistent argument pool. Every
module in the failed run skipped reclaim. The next candidate removes this
obsolete copying, validates flat argument intervals, and resets only when
the validation and arena contract are clean. Sret spans are not flat-pool
lists. No instruction, pool, function or rung limit is increased. The lifetime
audit distinguishes BSS argument/name lanes and independent heap-allocated
function-table chunks from temporary Box lists; module metadata is by value.

The production-composed regression creates 33000 call sites (including nine
arguments), resets and churns the Box arena, then verifies order, pool length,
function-table survival and invalid-range refusal. It preserves production
bodies/layouts; only test-copy effect annotations are widened for the stricter
bootstrap checker. Compiled and executed with the native seed, the previous
scanner returns 2 and the candidate prints `FLAT_ARG_RECLAIM_OK` with rc 0.
The candidate module typechecks. Full new-host and measured ladder results
remain pending; no reduced peak RSS is claimed yet.

The preceding modular ARM64 backend refuses this larger runtime probe at
`heap_alloc` rather than emitting a stub (`ARM64_REFUSAL unresolved_call`).
That explicit backend capability gap is not a passing native runtime proof.
The new Linux CI gate exercises the production reset/heap builtins through
the modular Linux emitter; native macOS validation still requires fresh
compiler/corpus execution, and does not claim arbitrary heap/arena builtins
are complete in generated ARM64 programs.

The extra imported nine-argument fixture, `madaros_flat_arg_dep_reset`, pins
actual compiler reclaim independently of that generated-runtime limitation.
Its preceding-host build reports `arena_reset_skipped (call-arg scratch
overflow) module 1 sites 5`. The native regression gate now requires an
`arena_reset_ok` record, no skipped-reset record, and correct program output;
the new-host result is pending. This is outside the pinned 200 denominator.
The first new Linux launcher stopped before Slurm submission because GCC's
`int64_t` type differs from Darwin's in mixed initializer lists. Explicit
`std::array<std::int64_t, N>` fixes the generator's portable C++23 typing;
the original failed launcher log is retained separately from the retry.

The reclaim candidate's signed macOS host is SHA-256
`98cf37a9ea92e8f9617b64b501067cac4b6050d4dd8c9460a429e2b64039bae6`.
Its dependency fixture now records `arena_reset_ok module 1 flat_arg_sites 5`
and totals `ok=1 skip=0`, then the signed program prints
`FLAT_ARG_DEP_RESET_OK` with rc 0. This discriminates the preceding skipped
reset without relying on the unsupported generated-runtime heap builtin.
ABI acceptance, IR growth/address and capacity gates, compiler self-check
(`verdict=0`, `check: OK`) and extended regressions pass. A new **200/200**
corpus with mandatory real Metal launches also passes; Apple M5 Max binary64
conformance reports `checked=40960 failures=0`, with the C++23 oracle.
Slurm 14605 started at 03:38:56 UTC, rebuilding the combined current source;
its measured ladder remains pending. No lower peak memory or rung `run`
is claimed from the native selected-corpus result alone.

**Blocking regression in the modular Linux runtime:** job 14605 builds its
host successfully (SHA-256
`64a3d1d99605f034498304c4c1295e84ecea4216e7227c676b61645b54774998`),
but the production-composed flat reclaim probe returns **5** after reset and
churn. Tail/address, imported-reset, boolean, dependency, TypeEnv and registry
fixtures return 0; these do not override the failing lifetime discriminator.
The ladder's check passes and gen2 is still running at this observation.

A separate diagnostic step on the same Slurm allocation retains pool length
33008 but shows `arena_violation=1`, zero scanned sites and corrupted function
metadata: function 0 has instruction count 16 and region slot/generation 0/0.
Reading the next corrupted function during diagnostics terminates with 139;
that secondary diagnostic crash is separate from the original probe's rc 5.
Its logs are retained under `flat-diagnostic-result` beside the job results.
The lifetime argument based only on the independently allocated chunk is
therefore insufficient: values referenced by that chunk can still live in
the modular runtime's shared allocation arena. The native-seed allocator's
separate aggregate pool does not establish the modular emitter's contract.
**Do not publish this reclaim candidate or accept its new gate until that
regression passes.** No reset/churn assertion is removed, no zero-result
runtime stub is substituted, and no ladder progress is counted as its proof.

The merge-queue run `38776296357`, integration commit
`9d99a9a817da9c353a2879da3dcb2ef3918475f8`, must not be conflated with the
earlier successful corrected-corpus run: it reported 199 passes and one compile
failure, `rapamycin_epistemic_pbpk.sio`. All selected GPU members passed.
This is a different failure from the original, uncorrected
`epistemic_subsumption.sio` baseline.

A native rebuild integrating main `9b5dba8b3f09a952802cebf68f7ab700f3ab88b4`
reproduced the PBPK refusal: `simulate_epistemic_rp3` requested 33981 IR
instructions, exceeding `IR_MAX_INSTRS=16384`. The struct-field variance
projection copied a short helper's transitive call tree repeatedly. The
existing 24-statement guard now includes transitive helper costs for these
projections, without raising IR capacity or changing the scalar-call policy.

The rebuilt native candidate, signed SHA-256
`040e5358124fd410728c226a79022c1581f1454c899e04d7e9d8ccb409321378`,
passed native acceptance and the extended port regression gate. That gate
fails on PBPK before the fix and passes afterward, and also executes main's
five new first-order call/closure/loop/reference fixtures. This focused result
does not replace a fresh complete corpus or post-merge CI result.

Failed corpus members now print their build/run return codes and logs into
the CI console. A deliberate invalid-compiler control fails the diagnostics
gate with the previous harness and passes with the new harness; it cannot
turn an actual corpus failure into a pass. Generated binaries and receipts
remain outside version control.

## Baseline, 2026-10-15

Signed native Madaros SHA-256:
`f73d21a578b74f1cb01afb9aa38ef5ffdecae1815ed922d0b2cd99bddc5f40f2`.

- 200 selected, 200 attempted; 41 pass, 159 compile failures.
- The original exact-output interpretation reported 37 pass plus four stdout
  mismatches. Re-evaluating those execution receipts with the canonical
  substring oracle produced the 41/159 baseline above; no compiler change is
  credited for the four harness corrections.
- The first failing opcode per fixture included array indexing/stores,
  global load/store, pointer stores and indirect calls. Floating operations
  and unresolved runtime calls also failed. These categories overlap within
  a program; fixing the first refusal is not proof the whole program passes.
- Frontend failures are retained in the denominator and reported separately
  from backend failures. A passing 200-program run would still not establish
  every feature in the language or a Madaros self-hosting fixed point.

Results after implementation changes must come from a newly built compiler and
a fresh result directory, not from replaying the baseline receipts.

## Measured implementation rounds

| Native host | Pass | Compile failure | Exit mismatch | Stdout mismatch | New passes / regressions |
|---|---:|---:|---:|---:|---:|
| Baseline | 41 | 159 | 0 | 0 | — |
| Round 1: arrays, pointer/global ops, indirect/stack calls, floating compares/output | 75 | 99 | 23 | 3 | 34 / 0 |
| Round 2: BSS provenance/initialization, reachability, sret arena/callee contract | 98 | 87 | 15 | 0 | 23 / 0 |
| Round 3: host rebuilt with tuple-value-copy bootstrap | 149 | 46 | 5 | 0 | 51 / 0 |
| Round 4: mixed float/int kinds, file runtime, floor/ceil | 175 | 19 | 5 | 1 | 26 / 0 |
| Round 5: host rebuilt with property-rooted array stores | 180 | 19 | 0 | 1 | 5 / 0 |
| Round 6: dyld/libSystem math imports | 182 | 17 | 0 | 1 | 2 / 0 |
| Round 7: nested float-array rank propagation | 183 | 17 | 0 | 0 | 1 / 0 |
| Round 8: omitted closure parameter is infer, not unit | 185 | 15 | 0 | 0 | 2 / 0 |
| Round 9: heap environments for escaping ARM64 closures | 187 | 13 | 0 | 0 | 2 / 0 |
| Round 12: Metal launches and target-scoped libSystem preflight | 198 | 2 | 0 | 0 | 11 / 0 |
| Round 14: numeric closure constraints and negative type guards | 198 | 2 | 0 | 0 | 0 / 0 |
| Round 18: generic HOF specialization handoff, original fixture sources | 199 | 1 | 0 | 0 | 1 / 0 |
| Same round-18 host, approved corrected epistemic fixture | 200 | 0 | 0 | 0 | fixture correction / 0 |
| Round 20: wide arithmetic, aggregate-repeat copies, capture-capacity guard; corrected corpus | 200 | 0 | 0 | 0 | 0 / 0 against corrected round 18 |

Every row attempted the same 200 members with fresh execution receipts; the
explicitly labelled fixture-correction row reuses the same host but rebuilds and
executes every program from the corrected source corpus.
Round-2 signed host SHA-256:
`ab427607c3ff571c87cb18db2b770e71ab7c83bcbbd2819e4d2af68793aa6861`.
Both candidates also passed the bounded macOS acceptance gate. These are
intermediate measurements, not a completed-backend claim.

The isolated `scripts/ci/madaros_macos_corpus_ir_gate.sh SEED` regression executes
arrays/references, positive/negative float formatting, ordered float comparison,
an indirect call, a 12-argument call, sret, and global init/store/relocation.
It caught a real shared-arena bug: `IrCallSret`'s argument count was being cleared
because its contiguous-register convention does not supply an ordinary argument
list. Correct storage retains that count without fabricating a pooled list.

The round-2 compiler still used the prior bootstrap. A separate bootstrap
regression, `a64_tuple_call_snapshot.sio`, exposed tuple-result aliasing: two
method calls printed `1 1` instead of `0 1`. The ARM64 local binding retained a
callee-owned return buffer for a synthetic tuple with no nominal struct row.
Copying its cache-known layout fixes the fixture to `0 1`; the patched seed raw
generations are byte-identical with SHA-256
`475d3fb8fbe3d4c80a331ee8c21ef0f2cfa33b93225fd03d96e2a43518f6682f`.
A full host/corpus rebuild is required before crediting that fix to the corpus.

Round 3 is that rebuild: signed host SHA-256
`532e1f934231c01bcfcc18dbedbcdab51ab483e68f9159e1ca3303a862e634d5`.
The correlated GUM fixture now prints its own expected variance values and PASS.
Five field/nesting-related GUM/Hessian fixtures still exited 1. Ten compile
failures were external GPU calls, retained in the denominator.

A second bootstrap discriminator, `a64_box_large_field_table.sio`, initially
printed -1 instead of 77. The lvalue detector accepted only an inner base
beginning with `(`, excluding `(*lo.big).field[i] = value`. Such stores fell
through and were discarded. Allowing property-rooted dereference expressions
fixes the fixture to 77. Its bootstrap raw generations are byte-identical:
`6d758e7cbf21374a41e87b6f9dacc4ff71b57ad0d747752989249f7df404427d`.
Again, corpus credit requires a freshly rebuilt native host and a fresh run.

Round 5 is that rebuild. Round 7 also executes the dedicated
`madaros_nested_float_array_rank.sio` discriminator: returned matrices, an
inferred row local, rank-three literals and integer nested arrays.

The ARM64 closure environment is a process-lifetime object holding an aligned
code pointer and captured words. A tagged descriptor is distinct from a plain
aligned function pointer; indirect dispatch inserts the environment as a hidden
leading argument and supports register/stack argument transfer. Other targets
retain the existing fail-closed escape guard. The isolated real-IR gate checks
both plain indirect calls and a tagged captured-value call.

## Metal integration verification

The Metal bridge is C++23/Objective-C++, compiled by
`scripts/dev/build_native_metal_runtime.sh`. It compiles MSL, creates shared
buffers with alias preservation, submits a compute command, waits for completion
and copies mutable results back only after successful GPU completion. Missing
devices, shader/ABI errors and command failures are errors, not CPU fallback.

Source f64 uses a pinned, licensed MSL software-binary64 header; no Rust code or
oracle is used. `scripts/ci/metal_binary64_conformance.cpp` compares actual GPU
add/subtract/multiply/divide/sqrt with C++23, including signed zero, subnormal,
overflow, infinity/NaN and deterministic random inputs. Local Apple M5 Max
receipt: 40,960 comparisons, zero mismatches (NaN payloads are not compared).
This is bounded conformance evidence, not an exhaustive IEEE-754 proof.

`tests/native/arm64_macos_metal_ir.sio` additionally parses an existing Sounio
kernel, emits a native Mach-O launch, loads the bridge and executes that kernel
on the GPU. Local stdout was `3.750000`, with a hardware execution receipt for
`sounio_kernel_vec_add_in_place`, four lanes, software binary64.

The full gate is `scripts/ci/madaros_macos_corpus_gate.sh COMPILER OUTPUT_DIR`.
It builds the runtime, runs binary64 conformance and attempts all 200. Its strict
mode additionally requires Metal execution receipts for the ten previously
refused GPU programs. Round 12 completed a full native CLI rebuild and fresh
strict-mode corpus: 198 pass, two compile failures, no regressions. All ten
required GPU fixtures executed successfully with hardware receipts; `atan` also
passed. Signed host SHA-256:
`638339e3ce2209786e6956bd785bcb6953f0de950060f4a4a82cf27181972c23`.
Round 14 repeated 198/200 with no regressions after adding concrete numeric
constraints for unannotated closures. Its negative discriminator rejects a
boolean call to `|x| x + 1` with E009 (expected i64, found bool), rather than
publishing an unconstrained wildcard signature.

The bootstrap element-borrow regression `a64_array_element_borrow.sio` now prints
77 and 78. Two bugs were required to explain it: an ordinary `&array[index]`
was unconditionally routed to the range-slice parser, and ARM64's aggregate
element borrow returned the pointer-cell address instead of loading the object
pointer (the x86 #1678 path already had the latter distinction). The corrected
bootstrap raw generations are byte-identical with SHA-256
`59c10e6b4a8d89551d3e2efcd6d370a80bd12d352c4629475aeedcfedb131853`.

One original fixture is independently ill-typed under the committed semantics:
`epistemic_subsumption.sio` assigns `measure(500.0, uncertainty: 5.0)` to a
variance-bound epsilon below 0.01. Its actual variance is 25, not a relative error
of 0.01. S08 explicitly defines epsilon as a predicate over runtime variance.
The original round-18 receipts remain separate: 199 pass, one compile refusal.
Signed compiler SHA-256:
`641e2ab0266be7dacbc867e37e0937175b558a82c347e37820fea494730b1095`.

Demetrios explicitly approved changing this fixture to `uncertainty: 0.05` on
2026-10-15. That value has variance 0.0025 and preserves the intended tighter-to-
looser subsumption test. The original uncertainty-5.0 input is retained as
`tests/compile-fail/madaros_measure_excess_variance_not_attested.sio`; the native
round-18 compiler rejects it with E001 and accepts the corrected positive input.
This is a fixture correction, not a compiler improvement credited to the original
200-source result. Never replay the old receipts against the corrected source.

The generic HOF fix infers proven scalar literal arguments across all formal
parameters, substitutes nested function signatures, and passes the actual
specialized item list into CLI preflight rather than a discarded field update.
Round 18 executes the original generic and named callback cases. It also turns
the ontology closure downgrade negative test from SIGSEGV into E008; named-type
optional-inner equality now uses nested matches rather than an aggregate tuple
match that corrupted the ARM64 bootstrap.

## Box-domain repair follow-up (2026-10-16, validation pending)

Slurm job 14605 did not close the ladder: gen2 exited 137 after 129 modules,
with an explicit Slurm OOM report and maximum RSS 66,962,128 KiB. Its separate
flat-argument lifetime probe exited 5 after reset/churn; the diagnostic variant
subsequently exited 139 while inspecting corrupted function metadata. These are
distinct failures. Neither the native Mac corpus success nor reset log markers
override that Linux lifetime failure.

The next candidate distinguishes Box allocation in IR, appends a separate Box
base/cursor/limit to the runtime context (old offsets preserved), and makes the
x86-64 arena mark/reset operate only on that domain. Aggregate allocations keep
their original cursor. Invalid reset marks must refuse with exit 183. This is
an implementation candidate, not yet executable proof of repaired lifetime or
reduced peak memory. Runtime-context, IR, lowering and x86 backend module checks
passed through the preceding native host; the separate main check timed out
(124), so it is not recorded as a pass.

The strengthened production probe requires actual Box cursor advancement,
rewind to the saved mark, and survival of an object allocated before the mark,
in addition to the existing 33,000-site/reset/churn/metadata checks. On the
ARM64 bootstrap seed this stronger test exits 13: its Box lowering uses separate
`mmap` allocations while arena marks track the AST arena cursor. Consequently,
the earlier seed probe success did not prove actual Box memory reclamation.
It remains useful only for the assertions it actually exercised. No-op reset,
skipped reclaim, or seed fallback is accepted as a Linux gate success.

Job 14617 (`pr2981-boxdom`) started on Slurm with the new Linux candidate and
unchanged ladder requirements. Results are separate under
`/tmp/pr2981-bool.QVgKpb/boxdom-result/out`; the prior failures remain preserved.
The invalid-mark test is included in that snapshot. The additional empty-arena
mark/lazy-initialization regression was added afterward and still requires a
separate Linux execution. Job 14617's rebuilt modular Linux host passed the
strengthened lifetime probe (exit 0), the invalid-mark refusal (183), IR/address,
boolean, FO/lopt, TypeEnv and registry fixtures. Its main check passed with zero
errors and gen2 is still running; this is not yet a rung `run` result.

The new native Mac build completed bounded acceptance, IR/TypeEnv/registry
gates, full main self-check, regressions and a fresh required-real-Metal corpus:
200/200. Its SHA-256 is
`e8a18cab69f5f5238826a98bddff8be7e7649ee417a9c50d1fbc313d3a6fc48e`.
Metal binary64 conformance on Apple M5 Max reports 40,960 checked, zero failures,
with the C++23 oracle. Logs are under `/private/tmp/pr2981-boxdom-native-*`.
The additional empty-mark fixture subsequently compiled and executed through
the same Linux host (SHA-256
`60101c259fd9523fb560547f19d158b18c74a0047c278126035a7b687904e898`),
with build/run statuses 0/0 and `BOX_ARENA_EMPTY_MARK_OK`. Its node-local logs
are in `out/empty-mark` alongside the ladder evidence. The first attempted
invocation used a workspace-local input path unavailable on the compute node;
it failed before compiling the fixture and is not a compiler regression. The
successful run transferred the fixture to the allocated node explicitly.
Job 14617 subsequently failed the ladder: gen2 exited 137 after all 129
dependencies and `Merged IR: 14729 functions`, with an explicit Slurm OOM event
at 05:10:10 UTC and maximum RSS 66,961,892 KiB. The reached rung remains
`check`, expected `run`; no gen2 ELF was produced. This does not invalidate the
focused lifetime repair, but it does disprove that this repair alone closes
the self-compilation memory wall. All output remains in `boxdom-result/out`.
The merged route next enters cleanup (when enabled), sealing and audit before
native emission; the last log alone does not identify which of those stages
caused the peak. Stage-specific measurement is required before another fix.
The diagnostic candidate now emits `compile_stage:` markers at merged cleanup,
sealing, ordered audit, float probe and native emission boundaries. A separate
Slurm wrapper samples process memory from `/proc` every ten seconds and keeps
`stageprobe-result` separate from the failed `boxdom-result`. These markers do
not alter the compilation passes or ratchet. The frontend source SHA-256 is
`0b7e00de2666437ad7eb3b43c4d652697f5c214078b4499c1305e1d7488a7434`,
verified against the transferred snapshot. Launch preparation is not a completed
measurement; no stage attribution or fresh execution success is claimed yet.
The first instrumented allocation, job 14633, was cancelled after 57 seconds
with signal 15 (wrapper status 143), before any result archive was returned.
Slurm reported `CANCELLED`, not OOM; the cause of cancellation is not established.
Its stderr, launcher log and empty result archive are retained. A separate
`stageprobe2` retry uses the same diagnostics and criteria without overwriting
that evidence.
The instrumented retry 14636 also suffered an explicit Slurm OOM. Its last
stage marker was `merged_native_emit_begin`, after cleanup, sealing, ordered
audit and float-probe markers. The sampler recorded RSS 66,779,232 KiB at
06:14:50 UTC, mostly anonymous memory; Slurm reported OOM at 06:15:07 UTC.
This localizes the final failure to the native-emission call, not to cleanup.
It does not yet distinguish backend initialization from the function loop.
The existing `SOUNIO_NV2_ARENA_TRACE` diagnostic is selected for a separate
same-host measurement; no optimization, limit or refusal is disabled.
Job 14668 completed with an explicit Slurm OOM at 06:54:22 UTC: check passed,
gen2 lowered 129 dependencies / 14,729 merged functions and exited 137. Native
trace samples showed the Box arena cursor stable while the aggregate pool
cursor advanced across emitted-function samples; no generic pool rewind is
safe because its blocks may escape. Logs remain under `nvtrace-result/out`.

A candidate now memoizes the region/dynlink, wide-multiply and IEEE diagnostic
configuration, following the lowering's existing process-lifetime flag policy,
and reads the arena-trace option before the function loop. The backend module
check passed. The production-extract C++23 fixture instruments real `read_env`
calls and checks all four flag values plus the lower-IR IEEE setter. The strict
seed test requires an additional Mut annotation on the copied setter, without
changing its body. Before: 1,024 reads / exit 5; after: four reads / exit 0 in
both off and on environments. This establishes memoization and flag behavior,
not that the complete OOM is fixed. The ARM64 seed's pool-mark values were zero
and therefore provide no allocation measurement. A raw Madaros ARM64 build of
this standalone reader fixture refused `str_from_bytes`; it is explicitly not
a passing native gate. The new CI fixture gate is connected on Linux only.

A separate `envcache` Slurm rebuild and unchanged full ladder are being
launched, alongside a fresh native Mac build/corpus watcher. Backend source
SHA-256 is `0439eefbf0cec0c38d9683f653531e40c13c51be7e1e8aa462d9c2cf1bb45fe5`.
Neither launch preparation nor the extracted seed fixture closes the ladder.
Job 14701's newly built Linux Madaros subsequently compiled and executed both
fixture revisions: before status 5, after status 0, after with enabled flags
status 0. IR/address, lifetime, bool, FO/lopt, TypeEnv and registry fixtures also
passed; invalid reset remained 183. Main check passed with zero errors and
gen2 is in progress, so the whole-program OOM repair is still unproven.
The new Mac host SHA-256 is
`54843092b8efbc43fddc61fa46ecf8e127c923cc9089d146a69c4e11b3eaa922`.
It passed bounded acceptance, main self-check and regressions, and subsequently
completed the fresh required-real-Metal corpus: 200/200. The Metal binary64 log
records 40,960 checked and zero failures. Final marker is in
`/private/tmp/pr2981-envcache-native-validation.log`. This is a Mac result, not
a completed Linux self-compilation ladder.

### Final combined local gate: rung run, not fixed point

Slurm job 14701 completed the unchanged fixed-point ladder gate with status 0:
check rc=0/errors=0, gen2 rc=0, 129 dependencies against minimum 122,
14,732 merged IR functions, and a gen2 `Madaros` version banner. The reached
and expected rung are both `run`. Maximum resident set size was 63,570,060 KiB
under the same 64 GiB allocation; preceding same-source-family runs were killed
at the allocation limit. This is executable evidence that the candidate closes
the previously measured OOM wall for this local rung, not a general memory bound.

Linux gen1 SHA-256:
`97885b74deee41a3b1bd9121e97ea582ec406f4751735286013200e410e43651`.
Produced gen2 SHA-256:
`29fbfa0168b57c8ec9a5aabc5a3410aed16311df7d01ad4d0c07724107531f20`.
Evidence is retained separately in the workspace
`/tmp/pr2981-bool.QVgKpb/envcache-result/out`, including source identities,
fixtures, ladder logs and resource samples. No generated binary or receipt is
included in the source change.

The optional next rung, gen3, failed with SIGSEGV (139) before compiling itself.
No modular fixed point, gen3 reproducibility or determinism is claimed. The
gate was already configured to require `run`; it was not lowered to accept
this result. The native reader fixture also reports pool-mark sentinel -1
through this Linux emitter, so only the complete process RSS measurement, not
that fixture's cursor values, supports the memory conclusion. Fresh PR/queue
and exact post-merge main CI remain required before final delivery.
No combined commit, push or merge is justified before the ladder completes
successfully.

## Additional wide-integer backend gate

`scripts/ci/madaros_macos_wide_ir_gate.sh SEED OUTPUT_DIR` uses a C++23
Boost.Multiprecision exact oracle to generate real IR, compiles a driver from the
current backend source, emits and signs a fresh ARM64 Mach-O, and executes it.
The local isolated gate passed 672 cases / 1,860 limb comparisons, covering ten
wide opcodes at 64, 128, 256 and 512 bits: modular add/subtract/multiply, logical
bit/limb shifts, unsigned single/full-divisor division and remainder, and signed/
unsigned comparisons. Alternating disjoint/in-place destinations exercise alias
safety. This is separate from the 200-program source corpus and does not claim
that every wide width or all target-specific vector opcodes are validated.

## Final installed local candidate

Round 20 rebuilt the complete modular host from the recorded current-source
bootstrap, signed it, passed bounded native acceptance, the extra positive/
negative port regressions, and a fresh strict full corpus: **200/200**. The
corrected round-18 to round-20 comparison has zero fixture changes and zero
regressions. The approved fixture correction remains the only source difference
between the original and corrected corpus snapshots.

Installed host: `artifacts/self-hosted/madaros-arm64-macos`, Mach-O arm64,
ad-hoc signature verified, SHA-256:
`8277c7a27f4c84a273a4a1d84d59ea6692d8575e2f7236da9f845c38205c203f`.
Runtime SHA-256:
`3bb8e18b087c178886a1497fb2eccf08d226c1a68d9fd1bc7863c2406a631bfb`.

Public `bin/souc` now resolves that host. Fresh public probes returned check=0,
program exit=7 (propagated unchanged), and GPU run=0 with `PASS: GPU vec_add`
and an Apple M5 Max 1,024-lane execution receipt. These probes used the default
automatic native target. `--science-boundary off` excluded the separate legacy
Python advisory attestor, not compiler checking or code generation; no science-
governance attestation is claimed.

The new repeated-aggregate identifier copy discriminator now prints 77 and 78,
not 78 and 78. Copying each value inside the emitted fill loop preserves
independent objects. Capture-table overflow now has a dedicated hard refusal
instead of dropping a capture and reaching a generic unresolved-name lowering
error. Its regression requires nonzero build status and no emitted artifact.

Local durable evidence is in
`artifacts/self-hosted/macos-native-validation-20261015-round20/`: original and
corrected per-test source snapshots, build/sign/stdout/stderr receipts, result
tables, source identities, candidate comparisons, and extra gate logs. Fresh
round-20 programs also have individual signed-artifact SHA-256 receipts. The
previous installed host is backed up there. Generated evidence is Git-ignored;
binaries and generated receipts are excluded from the source commit. This closes
the local selected-corpus gate, not full language/IR coverage, remote CI, Linux
parity, region reclamation or a Madaros self-hosting fixed point.

Commit preparation removed one trailing blank line from the backend source after
the recorded build snapshot and reran its module check successfully. No semantic
compiler changes were made after the full execution gate; the recorded binary
and source-snapshot identities continue to identify the tested build itself.
