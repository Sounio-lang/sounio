<!-- docs:meta
topic_id: repo.docs.llm-guide.reviews.example-execution
authority: repo_only
audience: users
last_validated: unvalidated
validated_by: none
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.llm-guide.reviews.example-execution
-->

# Example Execution Receipt

## Observed Result

On 2026-10-16, Slurm job `14651` executed **50 actual public-wrapper
invocations** over **47 distinct, unchanged source files** from
`f5b2c5850be9f3d3995719ffcade44a6d851658a`. The terminal ledger records:

| Verification state | Cases |
|---|---:|
| `run-passed` | 30 |
| `check-passed` | 4 |
| `expected-refusal` | 8 |
| `unexpected-failure` | 8 |

There is **no all-pass result**. The eight failures remain in the manifest
with their original sources, kinds, expected exits, and asserted diagnostic
substrings. Six are intended positive programs; two are intended negative
programs that exited nonzero without their required diagnostic. They must
remain quarantined from claims of successful use on this compiler profile.

The scheduler's `COMPLETED / 0:0` and the case runner's exit `0` mean that
collection completed. They do not override any individual failed assertion.
Every case has its own captured exit, command, source hash, stdout, stderr,
verification state, and classification reason in
[the complete execution records](../example-results.json). These are observed
case records, not reconstructed output. [The manifest](../example-manifest.json)
contains the original expectations and per-example verification objects.

The required `hello`, `struct`, `generics`, and `effects` cases passed.
All 21 requested `cook-*` bindings were retained. `cook-hello`,
`cook-struct`, and `cook-while` each repeat a source used under another ID;
each was actually invoked again. These aliases are not additional independent
programs. `cook-while` has its own final command/stdout/stderr and exit `0`.
No second job was submitted.

## Profile, Isolation, and Custody

| Field | Observed value |
|---|---|
| Source revision | `f5b2c5850be9f3d3995719ffcade44a6d851658a` |
| Compiler profile | Baseline committed Madaros Linux x86-64 prebuilt; not rebuilt from compiler source |
| Raw compiler SHA-256 | `125c25703ac4062fc6befb7acf77e945abb13405f0adb5ab09fbc5a8d4b5b29f` |
| Actual raw compiler path | `/tmp/llm-guide-14651.qlVJ68/source/bin/madaros-linux-x86_64` |
| Compiler version stdout | `Madaros v0.80.0 -- the Sounio self-hosted compiler` |
| Public entrypoint | `bin/souc`, through the existing Madaros launcher/resolver |
| Control host / object repository | `t560-proxmox` / `/home/devsounio/sounio` |
| Control repository live HEAD, read only | `9feae917a090c0236bf2d171a722265259fe637d`; not used as the archive revision |
| Scheduling route | Existing `ops/lab-ops.sh` → `lab_slurm_exec` → baseline `scripts/dev/slurm_srun_minimal.sh` |
| Job / account identity | `14651` / UID:GID `1000:1000`, `sounio:sounio` |
| Partition / QoS / actual node | `cpu-ops` / `normal` / `cpuops-t560-proxmox` |
| Requested resources | One node, one task, two CPUs, `8G`, no GPU |
| Scheduler start / end | `2026-10-16T05:47:36` / `2026-10-16T05:53:02` |
| Scheduler elapsed / walltime limit | `00:05:26` / `00:12:00` |
| Worker identity UTC | `2026-10-16T05:47:36Z` |
| Worker kernel / architecture | Linux `7.0.2-5-pve` / `x86_64` |
| Compute-local source / scratch | `/tmp/llm-guide-14651.qlVJ68/source` / `/tmp/llm-guide-14651.qlVJ68` |
| Per-case / whole case-loop timeout | `30s` with `5s` kill grace / `600s` with `10s` kill grace |
| Source archive SHA-256 | `a1cfa4a92f7066a0ff2b281b1a62e027f4ed3ddce29140c65f7f814ef01c97fc` |
| Receipt archive SHA-256 | `8484714c86d852a7cf4f95b5e8c9216c6d8b033eb2d9ba3175dd90e06d24d55e` |
| Host-visible run root | `/var/lib/cephfs-slurm/training/sounio-foundry/runs/llm-guide-f5b2-20261016.yIbh1N` |
| Login/compute-visible run root | `/orangefs/training/sounio-foundry/runs/llm-guide-f5b2-20261016.yIbh1N` |

The raw compiler digest above identifies the executable, not the shell
wrapper. The separately observed wrapper SHA-256 is
`af45052868e2e9a06b51b2736c665bee49b2a7a6a81372593b5ba450892656e0`
for `bin/souc`; the two identities must not be substituted for each other.

Preparation checked every selected local file against the baseline Git blob,
then hashed the exact bytes. Remote inputs came from `git archive` of the
explicit baseline, with archive PAX commit identity checked before dispatch.
The source archive includes the selected programs, their conformance import
provider, baseline wrappers/resolver helpers, the full baseline `stdlib`,
and the existing science-boundary metadata tooling. Standard-library paths
were made read-only before the cases ran. No source or library edit occurred.

Each case rehashed its source before invocation. All 50 matched their
manifest digest. After the loop, SHA-256 verification of all **2016 archived
source files** succeeded. The source archive and receipt bundle were retained
separately in the shared run root. No compiler rebuild, full stress run,
branch operation, commit, push, infrastructure repair, or queue bypass ran.

Preflight read the actual cluster instructions, Slurm queue/controller,
worker readiness/events/node-pressure conditions, and shared/worker-local
storage. CPUops was `2/2 Running`, its node Ready without memory, disk, or
PID pressure, with no worker events. Worker-local scratch reported `84G`
available. The job initially waited for priority, then received its normal
allocation. Other jobs were left intact.

## Invocation and Assertion Rules

Execution reused the same established Slurm transport as
[the bounded conformance receipt](../../spec/2026-10/15-bounded-conformance-receipt.md).
The example case loop was a bounded shell loop specific to this documentation
manifest. It did **not** execute or replace the canonical Sounio corpus
harness or claim that `serious_language_conformance_gate.sh` ran for these
50 cases.

The raw binary was materialized from the baseline compressed artifact using
the existing `scripts/lib/materialize_madaros_prebuilt.sh --verify`, checked
against the committed checksum document, and independently matched to the
pinned digest. The public wrappers then received these exact environment
selections:

```sh
MADAROS_RAW_BIN=/tmp/llm-guide-14651.qlVJ68/source/bin/madaros-linux-x86_64
SOUNIO_SOUC_ENGINE=madaros
SOUNIO_REQUIRE_COMMITTED_MADAROS=1
SOUNIO_STDLIB_PATH=/tmp/llm-guide-14651.qlVJ68/source/stdlib
TMPDIR=/tmp/llm-guide-14651.qlVJ68/tmp
```

For each manifest `run` entry, the recorded command is
`timeout --signal=TERM --kill-after=5s 30 bin/souc run <source>`.
Both `check` and `check-fail` invoke
`timeout --signal=TERM --kill-after=5s 30 bin/souc check <source>`.
The complete per-ID command strings are retained in the JSON and raw bundle.

The shell loop compares the declared exit and fixed stdout/stderr substrings;
it does not calculate a new scientific oracle. Trailing spaces in the
while-loop marker were retained. Timeout/killed, missing-command/launcher,
signal/crash, unexpected exit/acceptance, and missing-marker reasons were
classified separately. A negative result needs a nonzero checker exit
**and its intended diagnostic on the declared stream**; a crash or unrelated
parse refusal does not qualify. Every negative marker in this manifest is
asserted against stdout, where the selected compiler emitted diagnostics.

The positive enum case deliberately returns `42` and passed that exact exit.
Both bare-angle generic cases also expected `42`, but failed with `1`;
their expectations were not changed. Import-provider/import-main,
Knowledge-value, and Option-shorthand cases were check-only; successful checks
do not establish that those programs were executed. In particular, the
import-main source has a nonzero-returning entry point and was never run.

The stdlib Result, Vec, epistemic, EISA, and math-import programs exercised
their **existing embedded assertions and markers**. Their success is not a
new independent numerical verification or certification of the scientific
formulas. No new Sounio code, scientific computation, Python/Rust/Julia
numerical oracle, or Lean proof was introduced. Existing launcher metadata
machinery may invoke Python for provenance/policy metadata; it is not an
independent numerical reference.

## All Attempted Cases

Each ID below has its complete raw stdout/stderr in the `cases` array of
[example-results.json](../example-results.json), selected by the same `id`.
All observed source digests equal the corresponding manifest digest.

| ID | Requested mode | Expected exit | Actual exit | Verification state | Reason |
|---|---|---|---|---|---|
| `hello` | `run` | `0` | `0` | `run-passed` | ok |
| `struct` | `run` | `0` | `0` | `run-passed` | ok |
| `generics` | `run` | `0` | `0` | `run-passed` | ok |
| `effects` | `run` | `0` | `0` | `run-passed` | ok |
| `cook-hello` | `run` | `0` | `0` | `run-passed` | ok |
| `cook-struct` | `run` | `0` | `0` | `run-passed` | ok |
| `cook-for-array` | `run` | `0` | `0` | `run-passed` | ok |
| `cook-tuple` | `run` | `0` | `0` | `run-passed` | ok |
| `cook-borrow-release` | `run` | `0` | `0` | `run-passed` | ok |
| `cook-function-ref` | `run` | `0` | `0` | `run-passed` | ok |
| `cook-closure` | `run` | `0` | `1` | `unexpected-failure` | unexpected-exit |
| `cook-result` | `run` | `0` | `0` | `run-passed` | ok |
| `cook-range-option` | `run` | `0` | `0` | `run-passed` | ok |
| `cook-tuple-signature` | `run` | `0` | `0` | `run-passed` | ok |
| `cook-array-ref` | `run` | `0` | `0` | `run-passed` | ok |
| `cook-import-provider` | `check` | `0` | `0` | `check-passed` | ok |
| `cook-import-main` | `check` | `0` | `0` | `check-passed` | ok |
| `cook-vec` | `run` | `0` | `0` | `run-passed` | ok |
| `cook-enum` | `run` | `42` | `42` | `run-passed` | ok |
| `cook-generic-fn` | `run` | `42` | `1` | `unexpected-failure` | unexpected-exit |
| `cook-generic-pair` | `run` | `42` | `1` | `unexpected-failure` | unexpected-exit |
| `cook-epistemic-library` | `run` | `0` | `0` | `run-passed` | ok |
| `cook-eisa` | `run` | `0` | `0` | `run-passed` | ok |
| `cook-math-import` | `run` | `0` | `0` | `run-passed` | ok |
| `native-hello` | `run` | `0` | `0` | `run-passed` | ok |
| `while-loop` | `run` | `0` | `0` | `run-passed` | ok |
| `functions` | `run` | `0` | `0` | `run-passed` | ok |
| `tuple-destructure` | `run` | `0` | `1` | `unexpected-failure` | unexpected-exit |
| `shared-borrow` | `run` | `0` | `0` | `run-passed` | ok |
| `knowledge-input` | `run` | `0` | `0` | `run-passed` | ok |
| `knowledge-measured` | `run` | `0` | `0` | `run-passed` | ok |
| `knowledge-value` | `check` | `0` | `0` | `check-passed` | ok |
| `unit-add` | `run` | `0` | `1` | `unexpected-failure` | unexpected-exit |
| `unit-scale` | `run` | `0` | `1` | `unexpected-failure` | unexpected-exit |
| `struct-parameter` | `run` | `0` | `0` | `run-passed` | ok |
| `struct-return` | `run` | `0` | `0` | `run-passed` | ok |
| `nested-struct` | `run` | `0` | `0` | `run-passed` | ok |
| `strings` | `run` | `0` | `0` | `run-passed` | ok |
| `option-type` | `check` | `0` | `0` | `check-passed` | ok |
| `negative-effect-missing` | `check-fail` | `nonzero` | `1` | `expected-refusal` | ok |
| `negative-observe-io` | `check-fail` | `nonzero` | `1` | `expected-refusal` | ok |
| `negative-borrow-conflict` | `check-fail` | `nonzero` | `1` | `expected-refusal` | ok |
| `negative-knowledge-unwrap` | `check-fail` | `nonzero` | `1` | `expected-refusal` | ok |
| `negative-unknown-effect` | `check-fail` | `nonzero` | `1` | `expected-refusal` | ok |
| `negative-unit-add` | `check-fail` | `nonzero` | `1` | `expected-refusal` | ok |
| `negative-unit-return` | `check-fail` | `nonzero` | `1` | `unexpected-failure` | missing-stdout-marker |
| `negative-unit-argument` | `check-fail` | `nonzero` | `1` | `unexpected-failure` | missing-stdout-marker |
| `negative-mut-effect` | `check-fail` | `nonzero` | `1` | `expected-refusal` | ok |
| `negative-builtin-arity` | `check-fail` | `nonzero` | `1` | `expected-refusal` | ok |
| `cook-while` | `run` | `0` | `0` | `run-passed` | ok |

## Quarantined Failures

These are executable observations about the pinned committed prebuilt.
They do not redefine language syntax, validity, effects, units, or library
design. No failed positive was relabeled as an intentional negative.

| ID / source | Actual refusal | Unmet expectation and practical consequence |
|---|---|---|
| `cook-closure` / `tests/run-pass/closure_hof.sio` | `E004` expects `()`, finds `i64` at `178..184`; `E009` expects `fn#6`, finds `fn#11` at `168..195` | Exit `1` instead of `0`; untyped passed closure is not a verified recipe on this prebuilt. The separate named-function-reference recipe passed. |
| `cook-generic-fn` / `tests/selfhost/native_runtime/generic_explicit_fn_id_42.sio` | `E137` reports undeclared `id` and `i64` | Exit `1` instead of `42`; bare-angle spelling is not demonstrated by this run. The required turbofish `generics` case passed. |
| `cook-generic-pair` / `tests/selfhost/native_runtime/generic_explicit_struct_pair_42.sio` | Module failed to parse; AST closure `parse_failed=true` | Exit `1` instead of `42`; no generic-struct construction success is claimed. |
| `tuple-destructure` / `tests/run-pass/tuple_destructure_let.sio` | `E137` reports undeclared nested bindings `y`, `z` | Exit `1` instead of `0`; the full nested-destructuring program is quarantined. Both mixed tuple access and tuple signatures passed independently. |
| `unit-add` / `tests/run-pass/unit_same_add.sio` | `E041`: `mass^1` versus `(dimensionless)`; then `E000` | Exit `1` instead of `0`; this complete file includes comparison to a bare numeric literal. It does not isolate addition alone, so the result must not be described as disproving same-unit addition. |
| `unit-scale` / `tests/run-pass/unit_scalar_mul.sio` | `E041`: `mass^1` versus `(dimensionless)`; then `E000` | Exit `1` instead of `0`; this complete file also compares to a bare literal. No inference about which arithmetic operation caused the mismatch is promoted into a language rule. |
| `negative-unit-return` / `tests/compile-fail/unit_return_wrong_unit.sio` | `E008`: return value does not match function's declared return type; printed expected/found types both `f64` | Exit `1` was expected, but required `unit mismatch in return value` is absent. It remains a diagnostic-assertion failure, not a passed negative. |
| `negative-unit-argument` / `tests/compile-fail/unit_literal_suffix_reject_length_as_mass.sio` | Module failed to parse; AST closure `parse_failed=true` | Exit `1` was expected, but required `unit mismatch in call argument` is absent. The run does not establish the intended dimensional call-boundary refusal. |

The passed `negative-unit-add` has the intended `E041` dimension mismatch,
with `mass^1` versus `length^1`, followed by the aggregate `E000`.
It was not counted solely because some unrelated error occurred.
The other intended-refusal witnesses identify missing effects (`E035`),
Observe/IO (`E036`), borrow conflicts (`E038`), Knowledge mismatch
(`E001` with `found Knowledge<f64>`), unknown effects (`E246`),
and builtin arity (`E010`).

## Promotion Investigations and Limits

The baseline checkout has no `.claude/PARALLEL_BLOCKER_CONTRACT.md` or
`.claude/AGENT_OFFLOAD_POLICY.md` file. The session's operational instructions
still apply. The following are promotion investigations, not formal blocker
records or claims that every failure is a compiler defect. A future repair
lane must classify its cause and establish the applicable ownership/gate record.

| Investigation-ID | Priority / observed boundary / evidence | Tracking owner | Promotion gate | Next action |
|---|---|---|---|---|
| `LLM-EX-20261016-CLOSURE` | P2 / implementation-profile / executable pinned-prebuilt receipt | Parent guide coordinator; compiler repair lane unassigned | Same source passes intended run on an independently identified compiler, or a separately validated existing positive is explicitly selected | Keep `cook-closure` quarantined; source-based design discussion may remain with an honest execution boundary |
| `LLM-EX-20261016-GENERICS` | P2 / implementation-profile / executable pinned-prebuilt receipt | Parent guide coordinator; compiler repair lane unassigned | Both exact exit-`42` cases pass on an identified profile | Keep both bare-angle examples quarantined; the already-passed turbofish case remains available |
| `LLM-EX-20261016-TUPLE` | P2 / implementation-profile / executable pinned-prebuilt receipt | Parent guide coordinator; compiler repair lane unassigned | Existing complete nested-destructuring source passes intended run | Keep it quarantined; use verified tuple access/signature examples for executable claims |
| `LLM-EX-20261016-UNITS-POSITIVE` | P2 / complete-fixture arithmetic/comparison boundary / executable pinned-prebuilt receipt | Parent guide coordinator; compiler repair lane unassigned | Both unchanged positive sources pass, or a separately reviewed existing fixture isolates the intended teaching point | Preserve full diagnostics; do not attribute the failure to one operator without isolation |
| `LLM-EX-20261016-UNITS-DIAGNOSTIC` | P2 / checker diagnostic or parse boundary / executable pinned-prebuilt receipt | Parent guide coordinator; compiler repair lane unassigned | Both original asserted diagnostic substrings appear with nonzero checker exit | Keep the two failing negatives distinct from the eight demonstrated refusals |

No execution/queue blocker remains for this bounded run. These are outstanding
example-promotion/implementation-profile investigations, not permission to repair
compiler or library sources in this documentation lane. Legacy paths were
preserved; no legacy engine or fallback profile was exercised.

The immutable archive omits the compiler source tree and several package
policy directories. Launchers therefore emit advisory missing-policy-path
warnings and `science-boundary: mode=advisory verdict=UNKNOWN`. These stderr
messages are retained verbatim. No science-boundary certification is claimed.
Version provenance also says `tree=no-git` and warns that the ELF is the
committed binary, not built from the displayed source. Baseline identity rests
on the external archive commit/digest and per-file hashes.

Local manifest/result consistency checks passed for all 50 IDs: observed
exits/states match, recorded sources retain their hashes, every passed stream
assertion is present, and every passed negative contains its intended
diagnostic. Receipt metadata and whitespace checks passed. A structural
validator snapshot independently matched all 50 examples to baseline blobs;
its overall result still reported unfinished EN/PT guide marker/section/link
work outside this lane. Final whole-guide acceptance belongs to the parent
coordinator after the parallel writers complete.

The raw receipt bundle retains `identity.txt`, `worker-command.txt`,
`run-cases.sh`, manifest/TSV inputs, archive/compiler identity checks,
`source-files.sha256`, `source-postverify.txt`, `results.tsv`,
`case-runner-result.txt`, `job-result.txt`, compiler version/header logs,
and one command/stdout/stderr triplet per attempted case. The local JSON
preserves all case triplets and key identity records; the shared source
archive preserves exact input bytes. The receipt archive was independently
read back, SHA-256 checked, decompressed, and its 50 case records inspected.

No fresh source build, fixed point, full corpus, sanitizer/stress gate,
proof build, cross-target run, or independent scientific audit is asserted.
Markdown metadata intentionally remains `unvalidated`: this bounded execution
does not validate every guide statement. No separate LLM-offload review was
invoked by this example-execution lane; parent guide reviews are separate
evidence and do not substitute for the captured executions.
