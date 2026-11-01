<!-- docs:meta
topic_id: repo.docs.evidence.call-arg-release-stack.stdlib-continuity
authority: repo_only
audience: users
last_validated: 2026-11-01
validated_by: Slurm 20996 live receipt snapshot; whole-sprint qualification remains open
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.evidence.call-arg-release-stack.stdlib-continuity
-->

# Full stdlib sprint continuity

Status: OPEN. The targeted borrow-release stack gate passed in Slurm step
20996.4. The full stdlib qualification was then resumed with the same pinned
candidate a08c5660...; its results are not interchangeable with that gate.

## Execution and provenance

Private root: `/scratch/call-arg-release-stack-20261101` on the r740 Slurm worker.
Step 20996.6 runs `stdlib-full-v2.sh`, invoking
`scripts/stdlib/massive_sprint_gate.sh`, which includes the slow acceptance
suite with compound gates and two workers. Results:
`results/stdlib-full-v2`. The outer batch is a two-hour allocation; a terminal
Slurm receipt is required before treating a long-running phase as finished.

This source keeps the pinned compiler source intact. Stdlib, test, harness,
runtime, example, documentation and tool inputs were copied from the existing
sprint worktree, including its pre-existing dirty files. The suite captures its
input hashes. Full tools were added during staging after directory tests had
started; that addition must remain explicit when auditing suite-wide input
coverage. A clean, fully staged rerun is needed for final whole-sprint closure.

## Observed results

- The first attempt (20996.5) exited 127 before tests: the SDK's `g++` wrapper
  pointed to a removed historical GCC path.
- The rerun uses a private `tools/g++` symlink to the existing, correctly
  relocated `g++-portable`. Shared SDK files were not edited.
- Platform, callback, channel, file, socket-ready, VCF and related C++ reference
  phases reached successful results in the recorded prefix of the matrix.
- Several Sounio checks pass, but native lowering still refuses programs such
  as the byte-buffer UTF-8 path. The recorded error localises the first failed
  function to `byte_buffer_utf8_count_range_code`; the generic raise site is
  lower.sio:1876.
- The existing `tests/reference/bigrat_column_reference.cpp` does not compile:
  `natural_from_i64_reference` appears inside `add_reference` at source line
  40. This is an existing sprint source defect, outside the traversal patch.
- The fMRI gate returned 1. Its diagnostic log is preserved; it is not qualified.
- The whole slow acceptance suite is RUNNING at this snapshot. No final pass,
  failure total, or skip qualification is claimed.

The initial harness invoked generated ELF files without setting execute mode.
Ten native phases therefore returned 126 (permission denied). A separate Slurm
step (20996.7) applied `chmod +x` to those already-built files and reran them
under the same full-sprint SDK/library environment. All ten returned 0:
dynamic collections, binary, channel, native file, time, PNG checked, atomic,
mutex, atomic demo, and profiler.

`results/stdlib-full-v2/native-replay/RESULTS.tsv` is the runtime receipt.
This replay does not rewrite the original matrix or automatically pass its
downstream concordance stages. The full harness still needs an execution-mode
repair and a clean rerun before closure. Its larger stack allowance is not
the targeted 8 MiB acceptance result.

## Remaining gates

1. Obtain the terminal result of the full checker-source check in the 20996
   batch and its following ordinary build. They are still pending here; the
   mapped candidate used in the targeted gate is already built and pinned.
2. Finish/audit the slow suite's terminal receipt, preserving failures/skips.
3. Repair the harness's generated-ELF execute mode in its own claimed change
   and rerun affected runtime/concordance phases.
4. Fix the existing bigrat reference syntax and remaining lowering/negative
   diagnostic mismatches in the continuing stdlib lane.
5. Keep the scalar implicit-reference native lowering defect separately
   visible (see IMPLICIT_NATIVE_FOLLOWUP.md), and improve the generic 65-argument
   refusal separately.
6. Preserve the full collections/ownership, Unicode/text, I/O/concurrency and
   science objective. Modular Gen2 has a separate lane and acceptance record.

The release-loop patch preserves four arguments on its requested base. The
current main branch's separate three-argument change needs integration review.
No merge to main or compiler installation has been performed.
