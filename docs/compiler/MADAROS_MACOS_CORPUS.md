<!-- docs:meta
topic_id: repo.docs.compiler.madaros-macos-corpus
authority: repo_only
audience: contributors
last_validated: unvalidated
validated_by: none
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.compiler.madaros-macos-corpus
-->

# Native macOS Madaros: pinned 200-program corpus

## Single captured diagnostic-flag read (candidate after `b8ec6be2f9`)

Remote CI 39023869483 passed native macOS 200/200 with real Metal and the
dedicated Gen2 run 39023869475 passed, but the environment refresh witness
failed with exit 5. The exact CI host (SHA-256
`c5d083408877220b44c96d36cf38a0f3b430ddfe3da5508b001e09bb19cc940e`)
reproduced five reads in both off/on configurations on Slurm, whereas the
direct-bootstrap host made four. Diagnostic instrumentation located the extra
read at `SOUNIO_SABOTAGE_DYNLINK_THUNK_CALLRET` in the assigned if-expression;
the counter started at zero and reading `PROBE_ON` did not increment it.

The candidate captures that allocating read once and uses statement-form
branches to assign the flag. The unchanged production-extraction witness,
compiled by the same CI host on Slurm, then made exactly four reads and
returned zero with correct flags in both configurations. This is a scoped
refresh-path correction, not a claim that general effectful if-expression
re-lowering or all differences between bootstrap chains are resolved.
The gate now exposes the captured output on failure while preserving its
original exit status and exact four-read/256-cycle assertions.

Evidence remains outside Git in the remote scratch
`/tmp/pr2981-bool.QVgKpb/envgate-ci-result/out` (before),
`envgate-diagnostic-result/out` (key trace) and `envgate-once-result/out`
(after). A full rebuild through the source-tracking Linux bootstrap chain
and a fresh Mac build/corpus are pending; no earlier 200/200 receipts are
substituted for this candidate.

## Region witness cursor-domain correction (PR #2981 follow-up)

The published `7f13ba664a` CI failed the syscall escape witness with exit 2:
the witness observed `__arena_mark`, now the independent Box arena cursor,
while its allocations live in the aggregate/region heap. The candidate binds
the already implemented `__rgn_cursor` diagnostic in the checker and changes
only the witness's three cursor observations, preserving its assertions and
iteration counts. A new allocation-domain witness checks independent cursor
advancement and preservation of aggregates across an actual Box reset.

Slurm job 14729 rebuilt the compiler from the canonical seed and passed the
complete region gate, including all nine barrier sabotage discriminators,
streaming coverage, refusal in both lanes and the opt-out comparison. Removing
the syscall barrier still yields exit 2. The full current-source f64 gate also
passed. The modular ladder returned check=0, gen2=0, 129 modules (minimum 122),
14,732 merged functions, and executed the gen2 Madaros banner. The optional
gen3 remains SIGSEGV 139: this is rung `run`, not a fixed-point claim.
Remote evidence is retained separately in
`/tmp/pr2981-bool.QVgKpb/domain-result/out`. The rebuilt Linux host SHA-256 is
`21d96e479f262c88f1550c0308fb34afa683969dd5345db704dc5bad2e2ab65b`;
the ladder's recorded peak RSS is 64,581,320 KiB and its exit status is 0.
The fresh signed native macOS host SHA-256 is
`a63c96ca67e2ba775b7aafc42d836559fb733e64ee0224ff081709f887c20b1a`.
It passed bounded native acceptance, the production IR/address and TypeEnv
probes, full main self-check (`verdict=0`, `check: OK`), module registry and
extended positive/negative port regressions. The fresh pinned corpus completed
200/200 with required real Metal launches. The Apple M5 Max Metal conformance
run checked 40,960 results with zero failures against the C++23 oracle.
Local logs and generated receipts remain outside Git in
`/private/tmp/pr2981-domain-native-validation.log` and
`/private/tmp/pr2981-domain-native-corpus/`; no prior receipts substitute for
this fresh execution. This is local validation, not a claim that the new head
or the eventual main merge revision has passed remote CI.
The original 199/200, historical corrected 200/200, PBPK queue 199/200 and fresh
candidate runs remain distinct; the pinned 200-member manifest is unchanged.

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

### Published-head CI boundary follow-up (not yet accepted)

Head `10a03ee94b` reached remote f64 validation, but the existing 8,191-global
boundary gate failed with `E-TYPE-ENV-CAP` at 8,192 total bindings. The checker
also needs other bindings and lexical locals; a total table only one slot above
the advertised BSS-global limit is insufficient. This is a real regression in
the earlier fail-closed checker repair, not an OOM or a reason to weaken the
global gate. The dedicated gen2 workflow stopped at the same pre-ladder gate.
Logs: `/private/tmp/pr2981-head10a-f64-job-raw.log`.

The unpublished candidate uses a 12,288-entry TypeEnv, preserving sticky
overflow and all BSS/IR limits. Its production-container generator derives
overflow from `TYPE_ENV_CAPACITY + 1` and adds a mixed fixture at the actual
`BSS_MAX_GLOBALS` with 16 locals. The previous host refuses that mixed fixture
with `E-TYPE-ENV-CAP`; candidate module checking passed. No candidate runtime
success is claimed until rebuilding. A separate `envreserve` Linux snapshot
includes the entire existing f64 gate as well as the focused gates and ladder;
a fresh Mac build and strict corpus watcher are also pending. Changed TypeEnv
source SHA-256:
`e0c2f00b61c6e7f7cf0bd49c66ef2f9e4ee88065407a5801b92bc47f9eb90f80`.
Slurm job 14715 subsequently passed the complete existing f64 gate, including
8,191 globals compiling/executing and 8,192 cleanly refusing; imported capacity
16,383/16,384 and ordered 7/8/9/13-argument calls also passed. The mixed
global/local check passed, exact production-container tests passed, and the
larger TypeEnv's overflow check refused with status 1. Focused lifetime,
IR/address, boolean, FO/lopt and registry fixtures passed. Main check returned
zero errors; gen2 is still running, so no new rung result is claimed yet.
Fresh native Mac validation completed 200/200 with required real Metal,
self-check and regressions. Host SHA-256:
`4b92b489d4699871954ea8168190cb8af86df547c6488d872fb35b31cd3c0e5c`.
Its Metal log reports 40,960 checked and zero failures with the C++23 oracle;
the final validation marker is `/private/tmp/pr2981-envreserve-native-validation.log`.
The same candidate subsequently completed Slurm 14715's unchanged ladder with
status 0: check zero errors, gen2 rc=0, 129 dependencies, 14,732 merged functions,
and a running Madaros gen2 banner. Reached/expected rung remain `run`; gen3 still
exits 139, so there is no fixed-point claim. Peak RSS was 64,603,280 KiB under
the same 64 GiB allocation. Linux gen1 SHA-256:
`3170660e1b74b5539df798a3d45a2678fa15b4bbb166750d56560720222f9b36`;
gen2 SHA-256:
`ab77c3f937148b11180202ea88fba50c603c1630277dc074ac7c82a3d453ca14`.
Full f64 gate and ladder statuses are both 0 in
`/tmp/pr2981-bool.QVgKpb/envreserve-result/out`. This closes local verification
of the boundary follow-up, not fresh CI or the exact post-merge main result.
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

## Linux startup-memory experiment — 2026-10-17, validation pending

Slurm 14817 rebuilt the Linux host with the anonymous-BSS zero initializer
candidate. The selected host SHA-256 is
`3092a2b2f2d89360361e622028fec40c8598c12d63f67dd8d9d7dc6ae331d053`.
The frozen input archive is
`ea54d0a589738606352d6898921b842f6b46577210524684f9c417b63588d7e3`.
Three successful compile/run trials per small case gave these preliminary
comparisons against the earlier CI host (peak MiB; median wall seconds):

| Case | Before RAM | Candidate RAM | Before time | Candidate time |
|---|---:|---:|---:|---:|
| Minimal | 1306.023438 | 254.023438 | 0.543709 | 0.105984 |
| Imports | 1359.023438 | 307.023438 | 0.607614 | 0.147256 |
| PBPK | 1361.023438 | 308.023438 | 1.386511 | 0.934198 |

Conversions and medians were computed in C++23. This comparison also includes
the env-once repair; it does not isolate the BSS change. Slurm 14846 is the
paired control against the env-once host, with before/after executed
serially in one allocation on identical frozen sources and resource limits.
At 15:26 UTC, its small-case measurements and comparison completed successfully
on `/tmp/pr2981-paired-node.2oJvLp`; both small harness statuses and the C++23
comparison status were zero. The matched env-once versus seed-only BSS small
results are below (three trials per case); the subsequently completed paired
self-compilation results are recorded separately below.

| Paired case | Before peak MiB | After peak MiB | Before median seconds | After median seconds |
|---|---:|---:|---:|---:|
| Minimal | 1306.007812 | 225.023438 | 0.884308 | 0.106098 |
| Imports | 1359.007812 | 277.523438 | 0.715732 | 0.141589 |
| PBPK | 1360.507812 | 280.523438 | 1.406794 | 0.948180 |

Raw evidence is in job 14846's `out/{before-small,after-small}/measurements.tsv`
and `out/small-comparison.tsv`. This control isolates the seed-only change;
it does not validate the later modular-emitter candidate in job 14859.
At 15:56 UTC, the paired control's before-self harness completed with status 0:
`self.compile` took 1,656.150854 seconds with 62,920.152344 MiB peak RSS;
`self.execute` (`--version`) took 0.522682 seconds with 1,122.578125 MiB peak RSS.
Both command exits were zero. These conversions were produced by the C++23
reporter from `before-self/measurements.tsv`, also preserved separately as
`/tmp/pr2981-paired.367Vo0/before-self.tsv`.

The paired experiment subsequently completed with all six harness/comparison
statuses zero. C++23 comparison of the same frozen source in the same allocation
gave the following final self results (one trial per host):

| Paired self case | Before peak MiB | After peak MiB | Before seconds | After seconds |
|---|---:|---:|---:|---:|
| Compile | 62920.152344 | 62361.875000 | 1656.150854 | 1731.231835 |
| Generated `--version` | 1122.578125 | 1122.515625 | 0.522682 | 0.504429 |

The compile peak decreased 0.887279%, while observed compile time increased
4.533463%. This is not evidence of a self-compilation speedup, and one trial per
host does not establish statistical significance. Generated startup RAM was
essentially unchanged. Both `/proc/swaps` receipts contain only the header.
The seed-only change therefore does not resolve the dominant self-compilation
memory cost or propagate its small-case startup saving to the generated host.

A read-only C++23 reduction of the paired 500 ms memory samples locates most of
the retained memory before native emission: at the first observed
`merged_native_emit_begin` label, before-self already had 61,547.503906 MiB RSS
and after-self had 60,773.539062 MiB RSS. These are observer samples, not exact
instruction-boundary readings. The earlier interval remains labelled `unmarked`,
so this does not yet distinguish loading, checking and lowering or identify an
allocation owner. It does rule out attributing the entire peak to allocation
inside the final native-emission call. The phase reports are preserved as
`/tmp/pr2981-paired.367Vo0/{before,after}-phases.tsv`; the reporting source is
`/tmp/pr2981-memory-phase-report.cpp` outside Git. No measured compiler was
rerun or modified to produce these reports.

The launcher had written `before: OK` and `after: OK` checksum diagnostics into
stdout before the gzip archive. The raw `result.tar.gz` was preserved; a C++23
recovery tool verified the exact 21-byte diagnostic prefix and gzip signature,
then wrote `result.recovered.tar.gz`. `gzip -t` and extraction succeeded. Final
TSVs and status receipts are under
`/tmp/pr2981-paired.367Vo0/recovered/out/`. The launcher was corrected to send
checksum diagnostics to stderr; no measured compiler execution was repeated.
The seed-only pinned-source self-compilation measurement completed: exit 0,
1,678,305,975 microseconds and 63,863,584 KiB peak RSS. The generated compiler's
`--version` still required 1,149,520 KiB, so the seed-only repair does not propagate
the startup gain to gen2. Strong lifetime and region checks passed; its ladder
reached the required `run` rung in 30:58.08 with 64,029,180 KiB peak RSS and zero
reported swaps. Optional gen3 still exited 139. None of these results validates
the subsequent modular-emitter change, whose Linux validation is Slurm 14859.
The fresh macOS regression
run completed all 200 members with required real Metal execution, including
40,960 binary64 checks with zero failures. It used the existing signed host
`fb2a3c23dc1c6e835a6cb13df0b143e99ffd252c1eaece54b6e5dc815573514e`;
the change under test is in the Linux-only seed emitter, not its modular Mac
sources. This is a fresh execution receipt, not a newly rebuilt Mac host.

A subsequent rebuild including the changed modular Linux emitter produced the
signed ARM64 host
`e616d77e72d357f2f6a9ee7786551b57ad3cc05343e99f19ad5056d992b3df17`.
It passed native acceptance, self-check, IR/address, TypeEnv, registry, extended
positive/negative regressions and a fresh 200/200 corpus with required real Metal.
Its binary64 Metal receipt again records 40,960 checks and zero failures. This
Mac success does not substitute for the Linux receipts subsequently recorded
below or turn the seed-only paired result into a modular-backend result.

Slurm 14859 subsequently built the modular candidate successfully. Its native
host SHA-256 is
`8f63b9a1d6219a5c410fc2a6da50406d6897740e40e7346c7e14db54fabe3dd7`.
Both direct-seed order tests passed. The modular BSS-pages test passed normally
and under syscall denial, printing `BSS_ZERO_PAGES_OK` and
`MADVISE_WITNESS calls=2 enosys=2 child_exit=0`; build, normal and fallback
statuses were zero. This closes the emitted modular-path discriminator whose
old-backend witness saw zero calls. It does not yet certify the entire permanent
gate script; the final self, lifetime, region and ladder receipts follow below.
Its three-trial small-case harness passed compile and execution for every case;
C++23 summaries were minimum 254.039062 MiB / 0.110815 seconds, imports
307.039062 MiB / 0.158350 seconds, and PBPK 308.039062 MiB / 1.040846 seconds.
These are not the serial paired-control results above. Evidence is on node
`/tmp/pr2981-discard-node.YUJnGW/out/`, with the small TSV additionally retained
as `/tmp/pr2981-modular-discard.UnY6In/after-small.tsv`.

Slurm 14859 then completed with every recorded stage status zero: seed/build,
both seed-order cases, BSS normal/denied, small/self harnesses, strong lifetime,
region gate and ladder. The ladder checked successfully, emitted 14,734 merged
functions from 129 dependencies (minimum 122), and executed gen2's identification
banner. Its wall receipt is 29:43.28 with 64,139,976 KiB peak RSS and zero swaps.
The optional gen3 compile still exited 139; this is not a fixed point.

The frozen-source self harness recorded `self.compile` exit 0 in 1,677.983772
seconds with 62,469.800781 MiB peak RSS, and generated `--version` exit 0 in
0.001293 seconds with 7.593750 MiB peak RSS. Thus the modular repair propagates
the startup saving to the generated compiler, but does not solve the dominant
self-compilation memory cost. Compared descriptively with the env-once control's
before-self measurement on the same frozen source, compile RAM decreased
0.715751% and observed time increased 1.318293%. These are separate one-trial
runs, not a new serial paired or statistical performance claim. The C++23
comparison is `self-v-envonce.tsv` in the candidate snapshot.

All final Linux artifacts are under
`/tmp/pr2981-modular-discard.UnY6In/result/out/`, with the original archive
preserved. The supplemental production-container/f64 checks and finer frontend
allocation diagnosis still need separate execution. A read-only C++23 external
observer was prepared to correlate existing load/check/lower log milestones
with `/proc` samples, avoiding the internal allocating RSS probes. It passed a
short live-process control (exit 0) and missing-PID refusal (exit 3); this is
observer validation, not evidence identifying the dominant allocation owner.

The separate diagnostic allocation, Slurm 14926, started with the checked host
and seed identities. Its first supplemental attempt stopped at the empty-mark
fixture because that file was missing from the older source snapshot; the log
reports an unreadable input, not a compiler semantic failure. Earlier production
IR/address, registry, TypeEnv boundary/refusal, environment and invalid-mark
receipts were retained, but the supplemental status remains 1 and is not counted
as a full pass. The existing unchanged empty-mark fixture was transferred with
SHA-256 `a5a7cfd4c712fd6a370cbf14e8ba82ebda5683984cb0542a7262d62e94377575`.
The corrected wrapper accepts that explicit fixture path and checks its presence
before compiling anything. Slurm 14935 repeats only the supplemental checks in a
separate allocation; it does not rerun the measured 14859 pipeline.
Slurm 14935 subsequently completed with supplemental status 0, including the
empty-mark case, observed seed/modular fallback and full current-source f64
gate. Its receipts are in
`/tmp/pr2981-supplemental-retry.2QOsJR/result/out/`; the failed first attempt
remains separate. This execution used the real precompiled production generators
and permanent helper bodies; it does not claim that the entire permanent shell
gate was executed with a node-local C++ toolchain.

Slurm 14926's external frontend observer runs independently of those checks.
Its partial report showed 11,037.082031 MiB at the first observed imported
typecheck-begin label, compared with the much larger pre-emission RSS observed
in the completed paired experiment. The current report is incomplete: neither
the final largest-growth phase nor an allocation owner has yet been established.
The initial `unmarked` interval contains the visibility preflight before the
imported compile route, as corroborated by its `run_check_mode` log and caller;
it must not be relabelled as native emission. Diagnostic elapsed times are not
substitutes for the isolated benchmark receipts.
The later partial profile localizes the dominant growth to dependency lowering:
the first observed lower-begin sample had 14,266.582031 MiB RSS, the seed-lowering
interval peaked at 17,644.984375 MiB, and the first sample for dependency 58 had
51,316.238281 MiB. These values were computed by the C++23 frontend-stage
reporter from external samples. The profile is still running, but this already
distinguishes the large lowering accumulation from the much smaller earlier
load/check footprint. The precise allocating subpass and object owner remain
unproven; successful Box resets do not authorize resetting persistent aggregate
storage.

Slurm 14926 subsequently completed its diagnostic compile and generated-version
execution with status 0; its failed first supplemental attempt remains status 1,
and is closed separately by the successful 14935 receipts above. The final
external profile reaches 60,819.218750 MiB at the first observed lower-done label
and 62,320.875000 MiB peak during native emission. Its final phase report is
`/tmp/pr2981-frontdiag.WGDaEf/frontstage-final.tsv`; raw evidence is under
`result/out/frontend-diagnostic.uu1FWt/` in that snapshot.

The next diagnostic adds opt-in `SOUNIO_INTO_ACC_ALLOC_TRACE` receipts between
lowerer construction, external registration, own preseed, body lowering,
implementation fill and validated-call patching. The markers capture existing
pool-refill/mmap counters before formatting and never reclaim either domain.
The changed lowering module (SHA-256
`2590ae53f1e471ef30d116f66fdf2d4a8f0a94367463c69353fb1f6e0abb366d`)
passed module checking; this is instrumentation, not a memory repair. Slurm
14953 builds it, requires successful imports-fixture execution and markers with
the flag enabled plus silence with it disabled, and only then runs the same
frozen-source diagnostic. Its snapshot is `/tmp/pr2981-subpass.hDJiGR`, node
`/tmp/pr2981-subpass-node.wm9EZ4/`. These runtime controls and attribution were
still pending when recorded. The earlier Mac corpus and full Linux gates do not
certify this later instrumented source revision.

Slurm 14953 subsequently failed its build with E200 for `__rgn_cursor`: the
bootstrap seed does not implement that intrinsic. The build gate refused the
artifact, and the failure remains preserved in its snapshot. The diagnostic
was corrected to report that cursor as unavailable (`-1`), not as zero or as a
different allocation domain. Cumulative refill/mmap counters remain the basis
for attribution. The revised lower module, SHA-256
`8c62899e76e9b7d67a452493d9a41b0381a65cedc76061f94827b1c2ae192a0c`,
passed module checking. Slurm 14966, snapshot `/tmp/pr2981-subpass.4xarbB`, node
`/tmp/pr2981-subpass-node.RIxVX3/`, passed build and both imports compile/run
controls, trace-off silence and trace-on markers (all status 0). Its frozen
self-compilation diagnostic entered lowering; final counter attribution and
the dominant-memory repair are still pending. These are diagnostic controls,
not new before/after performance or full-corpus acceptance receipts.

The running 14966 diagnostic subsequently yielded 125 complete dependency
marker blocks. The C++23 subpass reporter attributes cumulative pool-refill
increments of 32,000 MiB to external registration, 9,216 MiB to body lowering,
1,024 MiB to lowerer construction and 512 MiB to own preseed; implementation
fill and validated-call patching crossed no refill boundary in those blocks.
Runtime-mmap counter increments were zero within these marked transitions.
These are allocation-counter deltas, not net resident-memory measurements:
refill granularity can charge an earlier allocation to a later boundary. The
partial raw log and complete blocks are preserved as `compile.partial.log` and
`complete-blocks.partial.log` in `/tmp/pr2981-subpass.4xarbB/`. The largest
measured subpass is now external registration, but its precise allocating
operation and a safe repair still require discriminating evidence. The job was
still active, and no final benchmark or acceptance claim follows from this
partial diagnostic.

Slurm 14966 subsequently completed: build, both flag controls, diagnostic
compile, observer and diagnostic wrapper all returned 0, and the generated
compiler printed the Madaros version banner. The final 129-block C++ report
retains 32,000 MiB for external registration and reports 9,472 MiB for bodies,
1,024 MiB for construction and 512 MiB for own preseed. The final raw log is
`result/out/frontend-diagnostic.54ivVz/compile.log`, and the report is
`subpass-final.tsv` in its snapshot. This diagnostic does not establish a
memory repair or fixed point.

A further diagnostic revision counts real `SOUNIO_SCO_ABLATION` reads in
strategy selection and captures refill increments immediately around each
read. It neither memoizes the ablation value nor changes the strategy rule.
Its lower-module SHA-256 is
`4da36bf6b0ea422e428b56620df50708f6377fbe673355e056c7eadc4ed2cd2f`,
and module checking passed. Slurm 14997 is running this revision from the
separate snapshot `/tmp/pr2981-subpass.bONRku`; runtime controls and allocation
attribution remain pending. The local-model suggestion used placeholder
environment/counter functions and was rejected, not used as executed evidence.

The 14997 snapshot was then found to have been copied before the source upload
completed: its source hash differed from the checked revision. That exact job
was cancelled and its snapshot retained as failed preparation, not compiler
evidence. After the complete remote source was verified, replacement Slurm
14998 started from `/tmp/pr2981-subpass.ivP42g`. The separate C++23 environment
counter reporter compiled with warnings-as-errors and passed a synthetic
8-read/1-MiB control; it refuses older logs missing those counters. Production
results remain pending and must also pass the seven-stage completeness report.

Slurm 14998 subsequently passed build, both imports-fixture compile/run
controls, trace-off silence and trace-on markers, all status 0. Its exact node
is `/tmp/pr2981-subpass-node.vO4FDh/`; the live frozen-source diagnostic entered
seed lowering. The controls show the new real-read counters present, not a
production attribution result. A guarded future launcher now checks an
explicit `PR2981_EXPECTED_LOWER_SHA` both before and after copying the source,
without modifying this running job or its snapshot.

The later 14998 partial report covers 128 complete blocks: 525,342 actual
`SOUNIO_SCO_ABLATION` reads during external registration account for 28,672 MiB
of refill increments, within that subpass's 32,000 MiB total. Own preseed and
bodies report 10,932 and 10,929 reads respectively, each with 512 MiB of refill
increments. This directly localizes a major allocating operation; it is still
not a resident-memory before/after benchmark.

The candidate repair caches only the process-scoped ablation boolean. It
retains the first real read and both flag meanings, without retaining a string
pointer or reclaiming persistent storage. Lower-module SHA-256
`81b24cf25319869525d6c9fcb6d53d80702d3be6c69887324e044ed66d7e85a0`
passed module checking. A new `sco_env_once_gate.sh` requires one real read in
the imports fixture and checks the named NonAssoc strategy under ablation 0/1.
Its actual before/after execution remains pending. Slurm 15000, snapshot
`/tmp/pr2981-sco-once.yBws1r`, was submitted and is pending resources: it rebuilds
the candidate, runs that gate, then benchmarks baseline and candidate serially
on the identical frozen inputs before lifetime, region and rung-run checks.
No benchmark improvement is claimed yet. A fresh Mac build and subsequent
full Metal validation were also started, with logs under
`/private/tmp/pr2981-sco-once-native-*`; these are not yet passing receipts.

Slurm 14998 subsequently completed all controls, diagnostic compile and
observer with status 0. Its complete environment-attribution report retains
525,342 reads / 28,672 MiB in external registration; own preseed and bodies
report 10,949 and 10,946 reads respectively, each crossing 512 MiB of refills.

Slurm 15000 is now running at `/tmp/pr2981-sco-once-node.T60vXJ/`; build and the
entire new SCO gate passed (`SCO_ENV_ONCE_OK`). The candidate host SHA-256 is
`20604ca5110c427d3f77be083c9b3a6e5e5987d0b716aa1fc0cd2177ac5fc5a1`.
The serial small-case comparison is not uniformly better: minimum compile
225.039062 -> 254.046875 MiB / 0.106770 -> 0.130607 s; imports
278.039062 -> 306.546875 MiB / 0.176322 -> 0.153128 s; PBPK
280.539062 -> 306.546875 MiB / 1.182280 -> 0.945409 s. Each side has three
trials; the increased small-case RSS remains an open regression to investigate,
not a claimed improvement. Self-compilation was still measuring the before
host. The SCO gate was added to the Linux workflow after its executed pass;
workflow references validate at 342. A separate before-gate negative control
is scheduled only after job 15000 so it cannot contaminate that benchmark.

Slurm 15000 subsequently completed every recorded gate/harness/comparison
status with 0. Its serial same-allocation, same-input self-compilation comparison
is **62,426.964844 -> 31,390.320312 MiB**, and
**1,634.053042 -> 1,371.061383 s** (C++23 report: 49.716728% less peak RAM,
16.094438% less elapsed time). Raw wait4 receipts are 63,925,212 -> 32,143,688
KiB and 1,634,053,042 -> 1,371,061,383 microseconds. Generated `--version`
remains 7.593750 MiB on both sides. This is one self-compilation trial per host,
not a statistical significance claim. Both swap files contain only the header.
Strong lifetime, all region sabotage checks and rung `run` passed; the latter
lowered 129 modules / 14,743 merged functions. Optional gen3 remains exit 139,
so this is not a fixed point.

Slurm 15002 also completed: the real pre-fix compiler fails the whole new gate
with status 1 and the specific repeated-environment-read diagnostic; the
negative control status is 0. The candidate's whole gate passed in 15000.
Artifacts remain at `/tmp/pr2981-sco-before.3CUxWG/result/out/` and
`/tmp/pr2981-sco-once.yBws1r/result/out/` respectively.

The small-case RSS increase remains under investigation. A separate C++23
external smaps diagnostic on the same frozen minimum input completed three
runs per host with nearly equal wait4 peaks (230,440 KiB before / 230,448 KiB
after), so it did not reproduce the original increase. Its one-millisecond
observer changes the diagnostic conditions and is not a replacement benchmark.
Raw mappings and measurements are preserved at
`/tmp/pr2981-vma.yP7b9k/result/out/`. A follow-up uses the original frozen
harness with reversed host order before any map sampling. Full candidate
supplemental validation and the fresh Mac Metal result are still pending;
none of these partial results authorize a completion or main-green claim.

The reversed-order repetition with the original frozen harness also completed
all six compile/run sequences successfully. C++23 comparison in
`/tmp/pr2981-vma.SJNn40/result/out/` reports minimum 225.039062 -> 225.046875
MiB, imports 278.039062 -> 277.546875 MiB and PBPK 280.539062 -> 279.046875
MiB. Thus the original approximately 29-MiB increase was not reproduced by
either this harness repetition or the map observer. Both original and repeated
receipts remain evidence; neither is discarded. Repeated minimum latency
0.115805 -> 0.219778 s is worse, while imports 0.219019 -> 0.168848 s and PBPK
0.934956 -> 0.924175 s are better. These short-run results do not support a
uniform performance-improvement claim or establish the cause of their variance.

Slurm 15012 then passed the full supplemental suite on the exact candidate
without rebuilding: f64, production-container/refusal, TypeEnv, registry,
environment flags, invalid/empty arena marks, initializer order and observed
syscall fallback. Evidence is
`/tmp/pr2981-sco-supplemental.smbNh1/result/out/extra.yY07f4/`, with
`supplemental.status=0`. These checks use the production generators and real
precompiled helper bodies. The entire standalone BSS shell gate still needs
its own C++-toolchain execution: the node reports gcc/cc but not g++/c++.
The new Mac build emitted its executable and passed IR/address and TypeEnv;
selfcheck and the fresh Metal corpus were still running when recorded.

The fresh Mac rebuild subsequently completed all acceptance, signature,
selfcheck, registry, port regressions and the required real-Metal corpus:
200/200 manifest members compiled and ran. The new signed ARM64 host is
`/private/tmp/pr2981-sco-once-madaros`, SHA-256
`eb350428b49284c3b87c387d550fb7576b60d18d4b406991a732f25e4874cad5`.
`/private/tmp/pr2981-sco-once-native-corpus/metal-conformance.log` records
Apple M5 Max execution with 40,960 checks, zero failures and the C++23 oracle.
This is a new build and run of the candidate source, not reuse of the older
Mac receipt. Validation log: `/private/tmp/pr2981-sco-once-native-validation.log`.

The standalone BSS script's first whole-script allocation stopped before
execution because the node's package indexes could not locate the matching
C++ development packages. Its evidence is retained at
`/tmp/pr2981-whole-bss.zdUPmp/result/out/`; it is a preparation failure, not a
gate pass or compiler failure. A replacement packages the workspace's existing
GCC 13 C++ components privately (no system installation) so the unchanged gate
can compile its own helpers in Slurm. Archive SHA-256:
`5ef82ee8bdc569e96be1a044631e2e6c046ff2669e377ba66373dfe0cab7be15`.

The initial private-toolchain archive also lacked the linker plugin; that
preparation failure is preserved at `/tmp/pr2981-whole-bss.dMgITO/`. Including
the complete GCC libexec directory fixed it without changing any gate or
compiler source. The corrected archive SHA-256 is
`4bf53bb15825a4e474c4004075a49a946d98251e87409c3ef128cf9d7bbb0409`;
both real helpers compiled successfully in a private-toolchain preflight.
The subsequent complete standalone `bss_zero_pages_gate.sh` execution in Slurm
passed with `gate.status=0` and `slurm.status=0`, evidence
`/tmp/pr2981-whole-bss.QCC3KZ/result/out/`. Both seed initializer order and
modular BSS page fixtures passed normally and under actual seccomp denial,
each with `MADVISE_WITNESS calls=2 enosys=2 child_exit=0`. This closes the
whole-script execution gap; it is no longer only manually equivalent checks.
These local/Slurm/Mac proofs still do not establish remote CI success, merge,
or exact-main acceptance: those require fresh receipts after publication.

The initializer-order fixture passed through both direct seed entrypoints.
The candidate-emitted executable also passed with seccomp denying `madvise`:
`GLOBAL_ZERO_ORDER_OK`, `MADVISE_WITNESS calls=2 enosys=2 child_exit=0`.
This is executed fallback evidence, not merely the independent C++ microprobe.
The earlier prefix-elision attempt (Slurm 14797) passed rung `run` but did not
reduce small-case RAM and is not the selected memory repair. Neither attempt
establishes a gen3 fixed point. No swap capacity or gate threshold was changed.

### Canonical Linux seed refresh after published CI

Fresh PR CI for `0dc219a26cdc34a451241b1eb99e4d2f13fa7d42` exposed
the expected seed-surface coupling: `Contracts Core` rejected the old committed
Lean seed because its self-compile differed from the changed `lean_single.sio`.
The exact failing gate was `canonical_compiler_gate.sh`, not the memory,
f64, Metal or rung checks. No gate criterion was changed.

Slurm job 15059 derived the replacement from the committed seed and exact source
SHA-256 `cdd399c7e9db6ccf0faa7e52a1b2fb87348aef3f8741d06e1310bc6125c91501`.
Generation 1 differed; generations 2 and 3 were byte-identical. The settled
candidate also reproduced itself in two additional compilations with identical
bytes. Their MD5 is `d83da2359f74ea04317991aebc0b21d9`, SHA-256
`2ef3cb5a84c0b899e02a6ccf97dfa196cfdb27609f189ab73372c370c3610425`.
Evidence is preserved at `/tmp/pr2981-seed-settle.GU17Cl/result/out/` on the
workspace. An independent C++23/OpenSSL receipt emitter verified the generation,
source and output hashes and refused an incorrect source-SHA control. This
establishes the Lean seed's M1/M2/M3 properties only; it does not establish a
modular Madaros gen3 fixed point or diverse double-compilation.

The follow-up Slurm snapshot `/tmp/pr2981-seed-check.Tvdv1m/result/out/`
passed the original `verify_lean_seed.sh` (fixed point and determinism), and
the entire standalone BSS gate with the refreshed seed. Both initializer-order
and modular-page tests passed normally and under observed denial, with
`MADVISE_WITNESS calls=2 enosys=2 child_exit=0`. The receipt records the named
canonical gate as pending fresh CI rather than claiming its Python-based
provenance wrapper was executed locally. Only the canonical Linux seed and
its matching receipt are refreshed; no `.sio` body or gate criterion changed.

### Residual lowering allocation: process-scoped BOX trace option (2026-10-19)

The follow-up to PR #2981 measures a separate residual allocation owner, not a
second claim for the earlier SCO improvement. An immutable current-source probe
counted 66,042 actual `read_env("SOUNIO_BOX_TRACE")` calls at the field-read site,
with 3,840 MiB of pool refill directly between the before/after counters. The
bind site counted 1,317 reads with no refill at that boundary. These are pool
refill counters, not net RSS; overlapping body/expression counters must not be
added to them. Evidence: Slurm 15252,
`/tmp/madaros-residual-box.RapscT/result/out/frontend-diagnostic.OIpuE4/`.

`lower_box_trace_enabled` now retains only the parsed process-scoped boolean,
including false/unset, after the first actual read. It does not retain an
environment-string pointer or reset an arena. Field lowering, Box parameter
binding, diagnostic contents, initializer order and allocation lifetimes remain
unchanged. The opt-in allocation receipt counts real reads at the three sites.
The source SHA-256 of `self-hosted/ir/lower.sio` used for the following proofs is
`123099c45918c1c35de8a852d7c6df604407f6ad8d331118b29b6bb0a730dc4d`.

The paired measurement (Slurm 15368) ran before then after serially in one
64 GiB / one-CPU allocation, using the identical frozen input archive
`ea54d0a589738606352d6898921b842f6b46577210524684f9c417b63588d7e3`
and C++23 harness
`9dc2dc3c45a2a25d5582ee1fc9f406e17e9661426326a28f3e2682fdeb708ac2`.
All diagnostic flags were unset. Both compilations and both generated-compiler
executions exited zero. `/proc/swaps` contained only its header before and after.

| Self compilation | Before | After |
| --- | ---: | ---: |
| Peak RSS (KiB) | 32,125,608 | 28,313,996 |
| Wall time (microseconds) | 1,374,707,649 | 1,375,939,696 |
| Peak RSS (MiB, C++23 report) | 31,372.664062 | 27,650.386719 |

The C++23 comparison reports 11.864716% less RAM and 0.089622% more wall time.
This is one self-compilation trial, not statistical significance, and substantial
residual memory remains. The generated version command used 7.601562 MiB in both
cases. Before compiler SHA-256:
`b4f46bd7cae939da10fc7284e837c8cbee7894e346cd62750cb3d4a990d73c37`;
after: `ca2fb9ff55496af0d5485cccb192a96b1be0d3744e314c33bb518ff976123fad`.
Receipts: `/tmp/madaros-box-pair.F1O5n6/result/out/`.

Small-input results are not uniformly better. The initial three-trial run
showed higher RSS (minimum 225.027344 to 252.031250 MiB; imports 276.777344 to
304.281250; PBPK 279.777344 to 295.281250). In a fresh original-harness run in
reversed host order, before any intrusive map sampling, the corresponding
medians were 225.027344 to 225.031250 MiB, 277.527344 to 277.531250, and
280.027344 to 269.531250. Three subsequent C++23 external map profiles of the
minimum each reported wait4 RSS 230,428 KiB before and 230,432 KiB after. The
initial RSS increase was therefore not reproduced; its exact cause is not
established. Minimum-input latency was slightly worse in both runs. The map
observer perturbs execution and is not a replacement benchmark. All results
are retained in `/tmp/madaros-box-vma.SmSk6F/result/out/` (Slurm 15387).

The durable `box_env_once_gate.sh` compiles its C++23 verifier, executes the
import/Box/decoy-field witness with trace 0, 1 and unset, checks runtime values
and trace behavior, then requires one real environment read. The whole script
failed on the instrumented prior compiler with `BOX_ENV_NOT_ONCE` only after
semantic controls passed, and passed on the candidate (Slurm 15427,
`/tmp/madaros-box-durable.ZfD2d0/result/out/`). The separate original regression
also established six reads before versus one after and exact trace parity.

Additional candidate acceptance:

- Slurm 15397: original fixed-point gate check and gen2 compilation exit zero,
  14,825 merged IR functions, 129 modules against minimum 122, and generated
  Madaros execution reaching required rung `run`. Optional gen3 still exits 139:
  this is not a full fixed point. Historical five empty stubs are not resolved.
- Supplemental snapshot `/tmp/madaros-box-supplemental.ByJqYA/result/out/`:
  full f64, IR/address, TypeEnv pressure/boundary/refusal, registry, environment,
  invalid/empty mark, initializer-order and observed BSS denial checks passed.
- Slurm 15412 (`/tmp/madaros-box-final.FjxwNE/result/out/`): the entire permanent
  BSS script compiled its C++ helpers and passed normal/denied paths, each with
  `MADVISE_WITNESS calls=2 enosys=2 child_exit=0`; flat33000 and region reclaim
  including negative barrier controls passed.
- Fresh signed native macOS candidate SHA-256
  `514502c9dca7552d7c5fc39f421140561adba87cb9913bd4d1c0e652960c65fe`:
  ABI/semantics/trace parity/selfcheck/IR/TypeEnv/registry/port regressions and
  required real Metal 200/200 passed. Apple M5 Max binary64 conformance checked
  40,960 results with zero failures against C++23. Evidence:
  `/private/tmp/madaros-box-once-native-resume-validation.log` and
  `/private/tmp/madaros-box-once-native-resume-corpus/`.

Failed preparation attempts remain distinct: Slurm 15289 ran an x86 seed on
ARM and failed before compilation; 15298 timed out with an incomplete pair;
15395 lacked the external time utility; the first supplementary snapshot lacked
the canonical wrapper. None is counted as a passing gate. Canonical wrappers
and the original gate thresholds were preserved in the successful replacements.
These are pre-publication proofs of this source; a subsequent PR merge and
fresh exact-main CI are separate acceptance steps, not implied by this audit.
