# First-request comparison:11939 and11956

The paired logs reproduce the first-request boundary comparison. Their SHA256
digests match the previously committed canary and failed-attempt receipts.
The successful canary used344 input tokens; the failed no-ontology pilot used303.
Seed0, temperature0.7, stop tokens and4096-token output ceiling match. The reported
execution profiles differ only in their canary/pilot scope label.

At request start, both runs report81,697,696,768 CUDA allocated bytes and
81,801,510,912 reserved bytes. At prefill end the failed pilot has12MiB less
CUDA reservation,1536 fewer allocated bytes, and more host MemAvailable:
533.254MiB more on rank0 and37.414MiB more on rank1. Its process RSS is
81.805/78.824MiB higher, mostly file-backed PSS (~70MiB per rank). These are
observed differences, not proof of a cause.

Both runs emit token200004 first.11956 then stops under the unchanged33GiB guard
before saving any complete proposal. Existing logs contain no per-decode-step
memory observations, so they cannot assign the stop to KV growth, allocations,
file-backed LM-head activity or other host activity. Different prompts and run
times prevent treating this as a controlled causal comparison. Zero saved
proposals does not mean only one token was computed before interruption.

The runtime Git diff from canary source cd72baf to frozen pilot74789 changes
only accepted bundle counts/index validation and the profile scope label.
The first-request inference loop is unchanged. The pilot's32-request count is
not evidence of accumulated inter-request state: only request0 began.

## Reproduce the comparison

Run ops/compare_first_request.py with --old-log, --new-log, --old-bundle and
--new-bundle. Exact paths and SHA256 digests are recorded in
validation/first-request-comparison-11939-11956.json. Missing or duplicated
paired stage evidence and mixed job identities refuse comparison.

## Prepared diagnostic, not hardware acceptance

ops/prepare_decode_probe.py accepts only the recorded runtime SHA256 and writes
a new file exclusively. It adds before/after decode observations for request0:
rank, request index, step, monotonic time, MemAvailable, process RSS/PSS and CUDA
allocated/reserved bytes. It adds no device synchronization or tensor retention.
Reversing the two observation blocks restores the input source byte-for-byte.
The existing frozen runtime and manifests remain unchanged.

The generated artifact and hash are in validation/decode-probe-build-11956.json.
Three local control tests pass: missing/duplicate/wrong-job evidence refusal,
exact probe-only transformation with changed-source refusal, and refusal to
overwrite an existing artifact. This does not qualify execution on hardware.

The next hardware diagnostic must bind this distinct runtime hash in its launch
and receipt inventory, replay the frozen pilot inputs and generation parameters,
and retain33GiB/32GiB limits. It must use a new immutable attempt, preserving11956.
Additional procfs reads/log writes can affect timing; this diagnostic cannot
supply performance evidence. If it completes, compare token responses across
both ranks and qualify an uninstrumented follow-up before resuming the pilot.
A diagnostic success never counts as a completed32-request pilot cell.
