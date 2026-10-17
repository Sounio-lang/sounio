# Loaded-model observer screening: negative closure, jobs 11976 / 11977

The frozen v2 screen failed. Baseline 11976 completed 8/8 requests per rank;
observed 11977 saved 6/8 per rank and stopped during request index 6. Slurm
records FAILED / 75:0, 2026-09-09T22:24:15 through 22:41:23 (scheduler timestamps).
No retry was performed. The profile, 33 GiB guardian and 32 GiB floor were unchanged.

## Evidence and acceptance

| Property | Baseline 11976 | Observed 11977 |
| --- | --- | --- |
| Saved requests per rank | 8/8 | 6/8 |
| Saved output tokens per rank | 943 | 701 |
| Rank byte parity on saved outputs | PASS | PASS on indices 0..5 |
| Minimum host available, rank 0 | 35,466,088,448 bytes | 35,886,645,248 bytes |
| Minimum host available, rank 1 | 35,747,856,384 bytes | 35,426,271,232 bytes |
| Full paired acceptance | Baseline custody PASS | FAIL, incomplete collection |

Rank 1 stopped at 32.9932861328125 GiB, 7,208,960 bytes below the 33 GiB
early-stop threshold and 1,066,532,864 bytes above the protected floor at that
sample. Its minimum is 321,585,152 bytes (306.6875 MiB) lower than baseline,
exceeding the frozen 256 MiB loss bound. These are sampled minima from unequal
completed workloads; they do not measure causal observer overhead.

The external journals contain 3,036 / 3,001 samples. Rank 0's maximum interval is
610,521,254 ns, exceeding the frozen 500,000,000 ns ceiling; rank 1's maximum is
353,171,531 ns. All required metrics were readable in the collected samples;
the sampled task-cgroup OOM counters did not change. Rank 1 recorded target
invalidation. Rank 0's observer was interrupted by paired-job cleanup and its
journal ends at SAMPLE. Neither lifecycle has its required completion tail.

Cross-arm response equality also fails on the available prefix: request index 3
has 121 output tokens in both arms, but token offset 76 (zero-based) is 19 in
baseline and 17 in observed. All other response fields match after excluding
only job; indices 0, 1, 2, 4 and 5 match in full. This is a token-level difference,
not an established cause or a decoded semantic diagnosis.

The unchanged qualifier rejects the pair with ValueError: incomplete collection.
Both completion receipts and requests 006 / 007 are absent, explicitly listed
in collection.json. No eight-request timing ratio or full response-parity result
is reported. loaded_model_overhead_qualified=false and pilot_acceptance=false.

## Custody and reproduction

The observed archive preserves 246 hashed artifacts plus collection.json:
worker identities before/after collection, boot IDs, both 104-file runtime
snapshots, pre-entry hash barriers, logs, partial responses and external journals.
The recorded worker identities, input hashes, runtime bytes, attachment
acknowledgements and journal bindings were checked. This is integrity of the
available negative evidence, not successful pair custody.

- Baseline collection SHA256: 510f98de0c30d183fa71caf21b6955ad86a59bbcdcd36de636a9c0f333da738f
- Observed collection SHA256: 6703d22ee93fa2e71068dedc44df4c5af3e7b91a5a987601960ec97a3d43b9a7
- Frozen specification SHA256: 5e641d4c1ba31a8a804f72d6b1e6829f35074db5e8e13de52cec1fe359ae9a36
- Runtime source: 054b7d672d3e5f6546c1e807b695f4ca78690ede; its CI run 34402455439 passed before launch.

From the repository root:

    python3 tools/pireus/continuity/validation/external-overhead-inference-20260909/replay_negative.py

This verifies both pinned collections and reproduces negative-result.json without
access to Spark or Slurm. Exit zero means the negative-evidence replay succeeded;
it does not mean that the experiment passed. qualification-refusal.json records
the unmodified runtime qualifier's refusal, which also used the private frozen
bundle to revalidate baseline admission.

## Continuity

The next bounded step is offline attribution using these immutable journals:
align lifecycle hooks with sample durations and metric read durations around the
610.5 ms gap; decompose rank-1 host pressure into process PSS, cgroup anon/file,
page-cache/reclaim and phase-local changes; inspect the single differing token
against the frozen prompt and generation settings. Keep these as separate
investigations. A single fixed-order pair cannot isolate observer cost from
cache/order/host-state effects or explain token nondeterminism.

Any proposed observer or runtime change needs a new versioned profile and a
separate predeclared qualification. This failed screen is closed and cannot be
re-entered. CPU-control qualification is preserved; pilot resumption, HTTP,
general 16K serving, formal V13/V14 and gain eligibility are not promoted.

Blocker-ID: BLK-20260909-pireus-observer-loaded-screen
Status: classified
Severity: B1
Class: platform-resource
Owner: codex-pireus
Lane: continuity-20260906 / loaded-model observer qualification
Worktree: /workspace/.wt/pireus-integration-20260906
Branch: codex/pireus-inkling-cycle-20260906
Files-Owned: tools/pireus/continuity/validation/external-overhead-inference-20260909; tools/pireus/continuity/status.json
Repro: replay_negative.py above; frozen job is not resubmittable
Observed: 6/8, guardian stop, sampling gap and prefix token divergence
Expected: 8/8 both arms with all frozen acceptance conditions
Acceptance-Gate: versioned paired protocol plus external_overhead_custody.qualify
Evidence-Level: E4
Evidence: observed-11977/accounting.txt; negative-result.json; qualification-refusal.json
Fallback-Path: none
Legacy-Kept: yes, prior CPU and loaded-model negative evidence
LLM-Offload: not-required (operational evidence report; no math, clinical or publication claim)
Next-Action: offline phase-aligned analysis of archived sample/read-duration and memory decomposition; no new inference job
