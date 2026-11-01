<!-- docs:meta
topic_id: repo.docs.evidence.call-arg-release-stack.readme
authority: repo_only
audience: users
last_validated: 2026-11-01
validated_by: Slurm 20996.4 targeted stack/borrow gate; see recorded per-test results
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.evidence.call-arg-release-stack.readme
-->

# Constant-stack call-argument borrow release

Status: the requested targeted stack/borrow acceptance gates PASS on the pinned
source-built candidate. The supplementary full checker-source check is still
running, and the full stdlib sprint remains OPEN.

## Scope and source correspondence

The patch replaces only the recursive traversal in
`checker_release_call_arg_borrows_inplace` in `self-hosted/check/check.sio`.
Argument and parameter cursors advance in the same order. An absent parameter
list stays absent; an exhausted argument list ends traversal. Implicit
identifier references and explicit OpRef/OpRefMut releases are unchanged.

The four-argument signature, including `call_start_borrows: BorrowEnv`, remains
intact. No caller snapshot or BorrowEnv representation changes are included.
The persistent paged representation from `05359adfc6` is retained. Its precise
contribution to the historical/current stack difference has not been measured.

Patch branch: `cursor/call-arg-release-stack`, based on
`ed9de5bc0775a55237bc7ac8ac2d925b754e87e2`. The local origin/main at inspection
(`137bebc92647`) already contains a separate removal of the fourth argument.
Integration must reconcile that independently reviewed change; this patch
intentionally follows the requested four-argument contract.

## Historical proof

`historical-artifacts.sha256` pins all recovered executables and the compressed
core. The distributed executable is
`e33dcf87f3a7cbca50b2c5c008da2a06153bfa552c73475b01239aff87415514`.
It has no symbol table. Source attribution uses the demonstrated body: four
arguments, argument/parameter list traversal, both borrow-release paths, and a
final self-call.

`historical-disassembly.txt` was regenerated from that exact executable using:

```sh
objdump -D -b binary -m i386:x86-64 --adjust-vma=0x400000 \
  --start-address=0x51b036d --stop-address=0x51b11d8 shipped
```

The frame reservation at 0x51b0371 is 0x2af00. The call at 0x51b11b9 targets
0x51b036d; its return address is 0x51b11be. The instruction at 0x51b0386 probes
the stack page addressed by RAX.

`historical-core.txt` was regenerated from the exact core with
`core_inspect.cpp` (C++23; ELF64 Linux x86-64 notes). It records RIP=0x51b0386,
RAX=fault address=0x7ffcbba72398, stack mapping
0x7ffcbba73000..0x7ffcbc273000, and ten consecutive frames returning to
0x51b11be. This distinguishes stack exhaustion from a heap-field fault.

Original artifacts are on gpuorangefs-multi-r740-proxmox:
`/scratch/param-stack-repair-20261101`. Historical jobs: 20994 (matrix), 20995
(core). `historical-matrix.tsv` preserves the original results: declaration-only
24/25/26 pass at 8 MiB; the distributed binary's calls pass at 24 and SIGSEGV at
25/26; the older workspace binary's calls fail at all three. Both pass those
calls at 16 MiB. These are historical results, separate from the new candidate.

## Candidate provenance and validation

Private worker directory:
`/scratch/call-arg-release-stack-20261101`.

The verification source is a private copy of the previously source-built
raw-cast repair snapshot at `/scratch/raw-cast-fix-20261101/source`, with only
this function replaced. It includes pre-existing sprint changes and the
raw-cast lower.sio repair; those are validation dependencies, not changes
introduced by this patch. The before checker is preserved in
`results/check.before.sio`; the complete self-hosted source manifest and seed
hashes are captured before compilation. No shared worktree or compiler binary
is replaced.

Slurm job 20996 runs the checker source check and builds via the source-derived
lean_single seed `./-o` using its positional output protocol. Step 20996.0
builds the same source with `--debug-fn-map` to identify the emitted function
and then runs `scripts/ci/call_arg_release_stack_gate.sh`.
The redundant queued job 20997 was canceled before starting and replaced by
step 20996.0. Build stack allowance is independent from test stack acceptance.

The gate invokes the raw candidate directly. Every compiler and generated
program invocation records effective soft/hard stack limits of 8192 KiB and
its own exit code, stdout and stderr. For arities 24,25,26,32,48,64 it checks
declaration-only and flat-call programs, builds and executes both, with an
assertion for every argument position. Additional tests cover 64 shared and
64 exclusive references with consecutive implicit/explicit calls, mixed
parameter kinds, nested calls, raw-cast field access, and negative borrow
conflicts. Negative diagnostics are compared byte-for-byte against the
previous raw-cast source-built compiler. A 65-argument call must still fail
build with exit 1; improving its generic diagnostic is a separate follow-up.

### Recorded targeted acceptance

Candidate SHA-256:
`a08c5660182d10d9d91ff6cf3551e70362437046d480b05a260dca20494c561f`.
`results/FUNCTION_MATCH.txt` verifies that the verification snapshot changes no
checker bytes outside the target function and that its function matches the
branch function after whitespace removal.

The source-derived seed emitted a full function map. `map_bounds.cpp` extracts
the candidate bounds from that map: 0x5305a5c..0x5306863. The prologue reserves
0x12f0 bytes. Its four direct calls target 0x7be049e (twice), 0x5217a34, and
0x1bc3b10; none targets the function itself. The loop back-edge at 0x5306853
targets 0x5305b05 inside the existing frame. See `candidate-disassembly.txt`.

The recent recursive baseline at the corresponding function entry reserves
0x1380 and calls itself at 0x5306d91. See `baseline-disassembly.txt`. This
comparison separates removal of arity-dependent recursive frames from the
much larger historical BorrowEnv representation change. The historical
0x2af00-to-current difference is not attributed solely to this patch.

`verify_no_self_call.sh` is a retained machine-code guard. It FAILS on the
recent baseline range 0x5305a5c..0x5306dab and PASSES on the candidate range
0x5305a5c..0x5306863. Thus the structural regression test is red on the direct
recursive baseline even though its paged BorrowEnv already passes the small
8 MiB scalar probes.

`results/gate-v3/matrix.tsv` and `RESULT` record PASS:
- 24/25/26/32/48/64 declaration-only and flat-call check/build/run, every
  parameter position asserted;
- 64 shared and 64 exclusive explicit references, consecutive native calls;
- shared/exclusive implicit references, including consecutive calls at arity
  64, in checker mode;
- mixed parameter types, nested calls, consecutive explicit calls;
- raw-cast field check/build/run, output `raw cast field PASS`;
- four incompatible-borrow refusals, exit 1, with diagnostics byte-for-byte
  identical to the baseline;
- 65-argument call: check 0, build 1; generic lowering refusal retained.

Every `*.limits` in the passing run is exactly
`soft=8192 hard=8192`. The raw-cast regression also passes independently at
16 MiB as a comparison only; 8 MiB remains the acceptance result.

The initial extra test that executed scalar implicit references failed with
139 in BOTH baseline and candidate. The minimal one-argument discriminator
and source localisation are in `IMPLICIT_NATIVE_FOLLOWUP.md`. Native implicit
reference execution is not qualified. The gate explicitly separates checker
coverage of implicit loans from native explicit-reference execution.
Attempt v2 then stopped because the private source snapshot lacked an existing
run-pass fixture; the fixture was copied unchanged and v3 passed. All attempts
remain under the private worker results directory.

The supplementary full `madaros check self-hosted/check/check.sio` in the
20996 batch is still running at this receipt snapshot. Its following ordinary
build therefore has no result yet. The mapped candidate build in step 20996.0
has completed and is the compiler exercised above. The step's original
`MAP_BUILD_STATUS=139` covers the first combined build-plus-gate script, whose
runtime probe failed; it is not a failure to produce the candidate ELF.
Use the per-attempt matrix and v3 RESULT rather than that combined status.

## Full sprint continuity

The core collections/ownership, Unicode/text, I/O/concurrency, and scientific
stdlib objective remains open. The modular Gen2 lane has its own acceptance
record. After this compiler repair passes, resume the complete stdlib gates
against the pinned candidate. The full massive-sprint gate has actually resumed under 20996 in a separate
step using the pinned candidate. See `STDLIB_CONTINUITY.md` for observed
failures and native permission-replay results. This does not establish
whole-sprint completion. No merge to main is authorized.
