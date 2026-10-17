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
